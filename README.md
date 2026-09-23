# HeteroSpec

Request-aware adaptive speculative decoding for heterogeneous SGLang workloads.

> **Status: Phase 0 (setup).** No measurements yet. Results sections below are
> placeholders until the go/no-go experiment completes. Do not cite anything here.

## 1. Problem

SGLang's merged adaptive speculative decoding adjusts `speculative_num_steps`
(hereafter **K**) at runtime. Its controller observes the accepted draft length
per request and smooths it into an **EMA over the whole batch average**
(`adaptive_spec_params.py`, `AdaptiveStepSlot.update`):

```python
batch_avg = sum(num_correct_drafts_per_req) / len(num_correct_drafts_per_req)
self.ema_accept_len = (1 - alpha) * self.ema_accept_len + alpha * batch_avg
```

The policy is only ever fed per-batch-size *aggregates*. A batch is therefore
treated as one homogeneous acceptance profile, even when the requests in it
disagree sharply.

**Hypothesis.** In heterogeneous traffic, a batch contains requests with
materially different acceptance behaviour. A single batch-wide K chosen from
request-local acceptance estimates beats a K chosen from the batch mean,
**without changing SGLang's execution shape** (still one rectangular K per
batch, reusing existing `SpecRuntimeState` / CUDA graphs / verify layout).

This is only interesting if it is true. The project is structured to find out
cheaply, and to report honestly if it is not.

## 2. Existing SGLang adaptive stack

Already merged in upstream `main`; we do not build this. See
[`docs/01-adaptive-stack.md`](docs/01-adaptive-stack.md) for the full call graph.

| Component | Role |
| --- | --- |
| `AdaptiveController` | Owns decision-making and atomic runtime-state switching |
| `AdaptiveSpecPolicy` (Protocol) | Pluggable interface — the extension point we target |
| `AdaptiveSpeculativeParams` | Default EMA policy, routed per batch-size slot |
| `AdaptiveStepSlot` | Per-BS-slot EMA + hysteresis + ceiling |
| `SpecRuntimeState` | Bundle of draft/verify/extend backends + CUDA graphs per tier |

Default candidate ladders (current `main`) include a **step-0 tier**, so the
controller can disable speculation entirely at high batch size:

```text
BS >= 1  -> [1, 3, 5, 7]
BS >= 8  -> [0, 1, 3]
BS >= 32 -> [0, 1]
BS >= 64 -> [0]
```

## 3. Hypothesis

TODO — stated as a falsifiable prediction once Phase 6 completes.

**Load-bearing assumption.** All oracle analysis assumes
**P(draft position k accepted) is invariant to K** — the k-th draft token is
generated from the same prefix regardless of how many total steps run. This is
approximately true for EAGLE topk=1 chain drafting but is *not* free. It is
tested explicitly (§6), not assumed. If it fails, every oracle number below is
an uninterpretable upper bound and the project pivots.

## 4. Request-level telemetry

`docs/01-adaptive-stack.md` §Telemetry records what already exists versus what
does not.

**Already in SGLang** (exposed in response `meta_info`, no source change):

```text
spec_verify_ct                  verify rounds for this request
spec_num_correct_drafts         cumulative accepted drafts (excludes bonus)
spec_correct_drafts_histogram   histogram of accepted drafts per round
```

Since `accept_len >= k` iff draft position `k` survived verification:

```text
P(draft position k accepted) = sum(histogram[k:]) / sum(histogram)
```

**Caveat that drives the design.** This derivation is only unbiased when **K is
constant** over the request's lifetime. Under adaptive decoding K varies, and a
round with active K=2 can never produce `accept >= 5`; those rounds are silently
counted as "position 5 rejected" when position 5 was never proposed. The
histogram does not record K per round, so the bias cannot be corrected
post-hoc.

Therefore:

- the step-1 per-request study runs against **static-K** servers;
- iteration-level tracing exists to make the adaptive case *correct*, not just
  richer.

**Missing** — iteration-level records preserving
`(request identity, accepted drafts, batch composition, active K)`.

## 5. Workload methodology

TODO — Phase 4. Families: `high`, `low`, `mixed_75_25`, `mixed_50_50`,
`mixed_25_75`, `phase_shift`, `real_mixed`. Prompt classes are **characterized by
measurement, never assumed** high/low.

## 6. K-invariance check

Deliverable, not an afterthought. Same prompt set at static `K ∈ {1,3,5,7}`;
compare per-position acceptance curves on overlapping positions. Runs on the
same grid as cost calibration, so it is nearly free.

## 7. Oracle-gap analysis

TODO — Phase 7. Compares, offline and per capture batch:

```text
global static K
  -> best batch-wide K in hindsight
  -> best batch-wide K using request-specific acceptance info   <- recoverable rectangular gap
  -> hypothetical independent K_i per request                    <- upper bound only
```

## 8. HeteroSpec policy

TODO — Phase 8, and only if the gap justifies it. Target shape:

```text
R1..Rn request-local acceptance state
        |
        v
request-aware policy
        |
        v
    ONE K per batch  ->  existing SpecRuntimeState
```

### Relationship to prior art (must be handled honestly)

- **#28045 (cost-aware)** is **not merged**. We do not have its machinery and
  must not describe it as "the current SGLang baseline". If compared, it is
  labelled as *"the open cost-aware policy from SGLang PR #28045 at commit X."*
- **DSpark block accept estimator** (already in `main`) keeps rid-keyed
  per-request acceptance state in the worker and cleans it up through
  `note_request_finished(rid=...)`. It is **not** the same mechanism: it
  estimates per-request *block* acceptance for DSpark's own planner, and
  `adaptive_unsupported_reason` confirms adaptive mode is EAGLE/EAGLE3-only, so
  the two paths do not overlap. Our claim is narrower and stays narrow:
  request-local acceptance feeding **batch-wide single-K selection in the EAGLE
  adaptive path**, which today uses only a batch mean. Cite DSpark as related
  work in the same tree.

## 9. Results

TODO — Phase 9/10.

Metrics: output tokens/s, TPOT/ITL, request latency, accepted length, draft
tokens generated/rejected, active K over time, K switches, and

```text
DraftWaste = (drafted - accepted) / drafted
```

Intended matrix (workload x system):

| Workload | No spec | Static K | SGLang adaptive | HeteroSpec |
| --- | --- | --- | --- | --- |
| High homogeneous | compare | compare | compare | compare |
| Low homogeneous | compare | compare | compare | compare |
| 75/25, 50/50, 25/75 mixed | compare | compare | compare | compare |
| Phase-shift | compare | compare | compare | compare |
| Real mixed | compare | compare | compare | compare |

Expected ideal outcome: `HeteroSpec ≈ adaptive` on homogeneous traffic and
`HeteroSpec > adaptive` on heterogeneous traffic.

## 10. Failure cases / limitations

TODO — Phase 10. Candidates: requests too short to build local history; batch
composition already homogeneous; acceptance drifts faster than the estimator;
stale history; switching overhead exceeding benefit; small batches making
differences negligible. Mitigations: global prior cold-start, minimum
observations, EMA decay, fallback to the stock adaptive controller, state
cleanup.

## 11. Reproduce

TODO — filled in as phases land.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/python -m pytest
```

## 12. Upstream SGLang contributions

TODO — Phase 10. Target shape, smallest first:

| PR | Purpose | Status |
| --- | --- | --- |
| PR 1 | Pass request identity through the existing CPU verify-feedback path so policies can keep request-local state | not started |
| PR 2 | Policy-interface cleanup | **only if the code genuinely needs it** |
| PR 3 | Request-aware mixed-workload policy, backed by benchmark evidence | not started |

The strongest justification for PR 1, as it stands today:

> The custom `AdaptiveSpecPolicy` interface receives per-request acceptance
> counts but not the identities those counts correspond to, so a policy cannot
> maintain request-local state across decode iterations — even though the worker
> already receives `rid` on the request-finish path, and DSpark already keeps
> rid-keyed per-request estimator state.

## Repository layout

PROJECT.md's intended layout maps onto this repo as:

| PROJECT.md | Here |
| --- | --- |
| `benchmarks/` | `heterospec/benchmark.py` (CLI) + `scripts/` |
| `workloads/` | `heterospec/workloads.py` + `workloads/` (prompt data) |
| `analysis/` | `heterospec/analysis/` |
| `configs/` | `configs/` |
| `results/` | `results/` (payloads gitignored) |
| `docs/` | `docs/` |

SGLang source is **never copied here**. This repo links to fork commits/PRs.
