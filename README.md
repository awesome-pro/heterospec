# HeteroSpec

Request-aware adaptive speculative decoding for heterogeneous SGLang workloads.

> **Status: Session 1 ready, awaiting GPU.** The harness, analysis and oracle are
> built and tested on the Mac (387 tests, 0 GPU hours). No performance number
> exists yet. Nothing in this README is a result.

## 1. Problem

SGLang's merged adaptive speculative decoding adjusts `speculative_num_steps`
(**K**) at runtime. Its controller reduces the accepted draft length per request
to a **single batch mean**, smooths it into one EMA, and maps that scalar to a
tier (`adaptive_spec_params.py:186-250`):

```python
batch_avg = sum(num_correct_drafts_per_req) / len(num_correct_drafts_per_req)
self.ema_accept_len = (1 - alpha) * self.ema_accept_len + alpha * batch_avg
```

A batch is therefore treated as one homogeneous acceptance profile, even when the
requests in it disagree sharply.

**Hypothesis.** In heterogeneous traffic, a batch contains requests with
materially different acceptance behaviour. Keeping **request-local** acceptance
state lets the controller reconstruct the batch's pooled acceptance curve — which
a single scalar cannot — and choose a better `K`, **without changing SGLang's
execution shape**: still one rectangular `K` per batch, reusing the existing
`SpecRuntimeState`, CUDA graphs and verify layout.

This is only interesting if it is true, so the project is built around finding out
cheaply and reporting honestly if it is not.

### A precision that matters

`docs/03-oracle-design.md` proves that `sum_i E[acc_i|K] = n · sum_{k<=K} S̄_k`,
so for a fixed batch the optimal `K` depends **only on the pooled curve**. A
"per-request vs batch-average" comparison is therefore **zero by construction**,
and must never be reported. The measurable gap is **curve vs scalar**: the shipped
controller has one number, and a scalar cannot identify a curve.

Request-level state remains the right mechanism, because composition changes every
step and a passive batch histogram only describes the past. Per-request curves let
the controller **re-pool for the composition actually in front of it**.

## 2. Existing SGLang adaptive stack

Already merged upstream; we do not build it. Full audit with `file:line`
references: [`docs/01-adaptive-stack.md`](docs/01-adaptive-stack.md).

| Component | Role |
| --- | --- |
| `AdaptiveController` | Decision-making and atomic runtime-state switching |
| `AdaptiveSpecPolicy` | Pluggable interface — the extension point we target |
| `AdaptiveSpeculativeParams` | Default EMA policy, routed per batch-size slot |
| `AdaptiveStepSlot` | Per-BS-slot EMA, warmup, interval, hysteresis, ceiling |
| `SpecRuntimeState` | Prebuilt draft/verify/extend backends + CUDA graphs per tier |

Default ladders include a **step-0 tier**, so speculation can be disabled outright
at high batch size (`[1,3,5,7]` at BS≥1; `[0,1,3]` at BS≥8; `[0,1]` at BS≥32;
`[0]` at BS≥64).

## 3. Hypothesis

To be stated as a falsifiable prediction once Session 1 lands. Expected shape:

```text
homogeneous traffic:  HeteroSpec ~= existing controller   (built-in control)
heterogeneous traffic: HeteroSpec > existing controller
```

**Load-bearing assumption.** All oracle analysis assumes
`P(position k accepted)` does not depend on `K`. Approximately true for EAGLE
topk=1 chain drafting, but an assumption. It is tested explicitly, not assumed —
see §6.

## 4. Request-level telemetry

Full detail: [`docs/01-adaptive-stack.md`](docs/01-adaptive-stack.md) §7.

**Already in SGLang**, exposed in `meta_info` with no source change:

```text
spec_verify_ct, spec_num_correct_drafts, spec_correct_drafts_histogram
```

Because topk=1 chain acceptance is prefix-contiguous,
`P(position k accepted) = sum(histogram[k:]) / sum(histogram)`.

**The hazard.** That derivation is only unbiased when `K` is constant. Under
adaptive decoding a round with active K=2 can never report `accept >= 5`, so deep
positions are scored as rejected although never proposed — and the histogram does
not record `K`, so the bias cannot be corrected post-hoc. It is therefore enforced
in code (`heterospec/telemetry.py` returns an explicit validity bound, and
`heterospec/analysis/traces.py` refuses an adaptive capture without a trace)
rather than left to discipline.

**What was missing**, and is now implemented in the fork on branch
`heterospec/iter-telemetry`:

```text
{"iteration": 182, "batch_size": 8, "active_k": 5,
 "requests": [{"rid": "r1", "accepted": 5}, {"rid": "r2", "accepted": 1}]}
```

A 12-line hook in `batch_result_processor._resolve_spec_v2_tokens`, plus a
stdlib-only, env-gated, bounded tracer module
(`SGLANG_HETEROSPEC_TRACE=/path/to/trace.jsonl`). It follows the existing DSpark
dump convention rather than inventing a format, costs one global lookup when
disabled, and is **not** proposed upstream as-is — see §12.

## 5. Workload methodology

`heterospec/workloads.py`. Families: `high`, `low`, `mixed_75_25`, `mixed_50_50`,
`mixed_25_75`, `phase_shift`, `real_mixed`.

Two deliberate choices:

* **Prompt classes are named neutrally** (`repetitive`, `open_ended`, `code`,
  `reasoning`, `chat`) and carry an explicit hypothesis string. Code, reasoning and
  chat are marked *unmeasured*. Classes are characterised from observed traces,
  never assumed.
* **Composition is interleaved across arrival order**, not blocked. Wave dispatch
  submits `concurrency` requests at a time, so a batch is a sliding window of that
  sequence. Blocking by class would mean no batch ever mixes, silently turning a
  mixed workload into two homogeneous phases and producing a null result that
  looks like evidence against the hypothesis. Tests assert every 8-request window
  at 50/50 contains both classes.

## 6. K-invariance check

`heterospec/analysis/invariance.py`. The calibration grid already runs identical
prompts at static `K ∈ {1,3,5,7}`, so the check is free. Tests assert it **passes**
on genuinely invariant data and **fails** on depth-dependent data; a check that
cannot fail would leave the central assumption untested. Single-depth comparisons
are reported `INCONCLUSIVE`, not as agreement.

**If this fails, every oracle number is an uninterpretable upper bound and the
project stops.**

## 7. Oracle-gap analysis

`heterospec/analysis/oracle.py`, design and rationale in
[`docs/03-oracle-design.md`](docs/03-oracle-design.md).

| Level | Information | Emits |
| --- | --- | --- |
| **L0** | none | one `K` for the run |
| **L1** | a scalar per batch | one `K` per batch |
| **L2** | the batch's pooled survival curve | one `K` per batch |
| **L3** | per-request curves | `K` per request (upper bound only) |

```text
recoverable rectangular gap = L2 - L1
```

Three methodological requirements, each of which cost a bug to discover:

1. **A train/test split is mandatory.** Fitting and evaluating on the same batches
   lets a fine-grained scalar policy place every batch in its own bin and match L2
   exactly, reporting a **false 0.00% gap**.
2. **The scalar baseline is `max(L1 fitted, shipped)`.** A fitted map can overfit
   and land below the fixed shipped rule on held-out data (observed 4.8730 vs
   4.8894). Reporting against the overfit policy would overstate the result.
3. **Throughput aggregates as `sum(tokens)/sum(time)`**, a token-weighted harmonic
   mean, not a sum of per-batch rates. Getting this wrong made L3 report `982`
   against `~5`.

Even once calibrated, `Cost(K, n)` is an **effective serving cost proxy at
concurrency `n`**, not a model-step cost. That is enough for a go/no-go screen; it
is not enough to defend a headline number in the 3–5% band, where cost must be
measured server-side instead. See `docs/06-gpu-runbook.md` §8.

Synthetic adversarial upper bound so far: **~1.03%**, with a **placeholder** cost
model. Why that is a ceiling, and every assumption that could move it:
[`docs/05-upper-bound-and-assumptions.md`](docs/05-upper-bound-and-assumptions.md).

**The gate is the K-invariance check, and it runs first.** If per-position
acceptance depends on `K`, the gap is uninterpretable whatever its size. Session 1
then screens on `mixed_50_50`; a positive result must be confirmed on `real_mixed`
and `phase_shift` in a small Session 2 before any policy work begins.

## 8. HeteroSpec policy

Not built. Deliberately gated behind the Session 1 result.

Target shape, reusing everything:

```text
R1..Rn request-local acceptance state
        |
        v
request-aware policy  ->  ONE K per batch  ->  existing SpecRuntimeState
```

### Prior art, handled honestly

* **#28045 (cost-aware) is not merged.** There is no cost-aware code in the tree.
  If compared, it must be labelled *"the open cost-aware policy from SGLang PR
  #28045 at commit X"* — never "the SGLang baseline".
* **DSpark's block accept estimator** already keeps rid-keyed per-request state and
  cleans it up via `note_request_finished(rid=...)`, which
  `batch_result_processor.py:1315` already calls. It is **not** the same mechanism:
  it estimates per-request *block* acceptance for DSpark's own planner, and
  `adaptive_unsupported_reason` confirms adaptive mode is EAGLE/EAGLE3-only. Our
  claim stays narrow: request-local curves feeding batch-wide single-`K` selection
  in the EAGLE adaptive path. Cite it as related work in the same tree.

## 9. Results

None yet. Intended matrix:

| Workload | No spec | Static K | SGLang adaptive | HeteroSpec |
| --- | --- | --- | --- | --- |
| High homogeneous | compare | compare | compare | compare |
| Low homogeneous | compare | compare | compare | compare |
| 75/25, 50/50, 25/75 mixed | compare | compare | compare | compare |
| Phase-shift | compare | compare | compare | compare |
| Real mixed | compare | compare | compare | compare |

Metrics: output tokens/s, TPOT/ITL, request latency, accepted length, draft tokens
generated/rejected, active K over time, K switches, and

```text
DraftWaste = (drafted - accepted) / drafted
```

`DraftWaste` is reported only when the depth each request actually ran at is
known; for an adaptive run without the trace it is reported as `None` rather than
computed from a fabricated denominator.

## 10. Failure cases / limitations

To be populated from Session 1. Candidates: requests too short to build local
history; batch composition already homogeneous; acceptance drifting faster than
the estimator; stale history; switching overhead exceeding benefit; small batches
making differences negligible. Mitigations: population prior cold-start, minimum
observations, EMA decay, fallback to the stock controller, state cleanup.

## 11. Reproduce

Everything below runs on the Mac with no GPU:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/python -m pytest                      # 387 tests

# full pipeline against a mock SGLang, no GPU
.venv/bin/python -m heterospec.benchmark \
    --workload mixed_50_50 --policy static_k3 \
    --num-requests 120 --concurrency 16 --mock

# the exact launch commands for the GPU host
.venv/bin/python -m heterospec.benchmark --print-commands

# Session 1 plan, and its GPU-time estimate
.venv/bin/python -m heterospec.session \
    --launch-config configs/models/llama31_8b_eagle3.json --plan
```

The GPU session itself: [`docs/06-gpu-runbook.md`](docs/06-gpu-runbook.md).

## 12. Upstream SGLang contributions

Status: **none opened yet.** Target shape, smallest first.

| PR | Purpose | Status |
| --- | --- | --- |
| PR 1 | Pass request identity through the existing CPU verify-feedback path, so a policy can keep request-local state | **implemented** on `heterospec/policy-feedback-identity` (88 lines of production code, 226 of tests); not yet opened |
| PR 2 | Policy-interface cleanup | **only if the code genuinely needs it** |
| PR 3 | Request-aware mixed-workload policy, backed by evidence | not started |

PR 1 is independent of the go/no-go: a policy interface that cannot maintain
request-local state is a real gap regardless of whether HeteroSpec wins. It is
carved off the pinned base rather than stacked on the trace branch, so its diff
stays reviewable. See `docs/02-reproducibility.md` for the branch layout.

PR 1's justification, as it stands today:

> The `AdaptiveSpecPolicy` interface receives per-request acceptance counts but not
> the identities those counts correspond to, so a policy cannot maintain
> request-local state across decode iterations — even though the worker already
> receives `rid` on the request-finish path, and DSpark already keeps rid-keyed
> per-request estimator state.

The iteration trace is **not** part of PR 1. It is a research facility; the
upstream-worthy change is the identity on the feedback path.

## Repository layout

PROJECT.md's intended layout maps onto this repo as:

| PROJECT.md | Here |
| --- | --- |
| `benchmarks/` | `heterospec/benchmark.py`, `heterospec/session.py`, `scripts/` |
| `workloads/` | `heterospec/workloads.py` |
| `analysis/` | `heterospec/analysis/` |
| `configs/` | `configs/` |
| `results/` | `results/` (payloads gitignored) |
| `docs/` | `docs/` |

## Docs

| Doc | Contents |
| --- | --- |
| [`01-adaptive-stack.md`](docs/01-adaptive-stack.md) | The merged adaptive stack, audited with `file:line` refs, and the precise gap list |
| [`02-reproducibility.md`](docs/02-reproducibility.md) | Pinning rules, the two independent fork branches, and which tests need which |
| [`03-oracle-design.md`](docs/03-oracle-design.md) | The linearity correction, the four levels, and three measurement traps |
| [`05-upper-bound-and-assumptions.md`](docs/05-upper-bound-and-assumptions.md) | Why the synthetic number is a ceiling, 11 assumptions with bias direction, the pre-registered decision rule |
| [`06-gpu-runbook.md`](docs/06-gpu-runbook.md) | GPU choice with VRAM math, RunPod setup, the exact commands, failure table |

SGLang source is **never copied here**. This repo links to fork commits and PRs.
