# 07 — Session 1 results (real GPU, first run)

**Status: the pre-registered gate fired. The oracle gap is NOT reportable.**

This is the first document in the repo containing actual measurements. Everything
before it was method, synthetic study and assumption bookkeeping.

## Provenance

```text
date            2026-09-23
host            RunPod 1x RTX A6000 (46068 MiB usable), driver 580.159.03
python          3.12.3
SGLang          fork awesome-pro/sglang @ f9281ec128 (heterospec/iter-telemetry)
pinned base     66ce8c55cc
working tree    clean  -> citable (see results/README.md)
harness         235d527 ../ heterospec main
model           meta-llama/Llama-3.1-8B-Instruct (fp16)
draft           lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B
speculation     EAGLE3, eagle_topk=1, num_draft_tokens = num_steps + 1
workload        mixed_50_50 (repetitive 0.5 / open_ended 0.5), max_new_tokens 256
dispatch        waves, greedy (temperature 0)
grid            K in {1,3,5,7} x concurrency in {1,8,32}, plus no-spec at c32
```

Raw payloads are **not** in git by policy (`results/README.md`). The run tree is
21 MB: 16 run directories, `session1_report.json`, 6 server logs, and a 15 MB
iteration trace. A 56-file manifest of the original tree hashes to
`750ec6ae90c43427da2c1590b8ca1324`.

## Execution was clean

```text
16/16 runs ok, 0 failed requests
cost grid: 12/12 cells measured, complete, 0 clamped cost queries
iteration trace: 31,403 records (4729 / 10168 / 16506 at c1 / c8 / c32)
sum(histogram) == spec_verify_ct for all 12 calibration cells
```

No cell was lost, so nothing below is an artefact of a partial grid.

## Result 1 — the gate fired: A2 is false

The K-invariance check compares `P(A_k)` measured at several static depths, at
one concurrency, with the same seed and therefore identical prompts and batching.

```text
K-invariance: FAIL — max deviation 0.0802 at position 1 (tolerance 0.05)

pos | K=1     K=3     K=5     K=7   | spread
  1 | 0.4400  0.3762  0.3598  0.3617 | 0.0802
  2 |    -    0.2162  0.2001  0.2044 | 0.0160
  3 |    -    0.1335  0.1167  0.1168 | 0.0168
  4 |    -       -    0.0866  0.0866 | 0.0000
  5 |    -       -    0.0692  0.0689 | 0.0003
```

Three properties make this credible rather than sampling noise:

* **It replicates at every concurrency** — spread 0.0881 (c1), 0.0818 (c8),
  0.0802 (c32).
* **It is monotone in K** at c1 and c8: 0.42 -> 0.36 -> 0.34 -> 0.33.
* **K >= 3 agree with each other** within tolerance at all three concurrencies
  (<= 0.036), so the dependence is concentrated in the shallowest configuration.

Ruled out as artefacts: our `num_draft_tokens = num_steps + 1` convention matches
SGLang's own fixture (`spec_steps=5, spec_tokens=6`); the histogram/round-count
identity holds in every cell; and sample sizes are 4324–136238 rounds per cell.

**Consequence.** The oracle evaluates `E[acc | K]` at depths a request never ran,
using one capture's curve. That curve is wrong for K=1 by 5.4% in tokens per round
— larger than the effect being measured. Per the pre-registered rule
(`docs/05` §4) this is a **no-go for the gap**, whatever its size.

## Result 2 — on this configuration, speculation loses

```text
cell        tok/s    tok/round   tokens per position
K=0 c32    1049.7       1.0000          1.000
K=1 c32     745.5       1.4431          0.722
K=3 c32     753.1       1.7251          0.431
K=5 c32     655.5       1.8318          0.305
K=7 c32     586.3       1.9137          0.239
```

`tokens per position` is `tok/round / num_draft_tokens`. Deeper `K` raises tokens
per round while lowering tokens per position, and the position cost is not
amortised enough to compensate: **no-spec is 1.39x faster than the best
speculative cell at c32**, and deeper `K` is monotonically worse at c8 and c32.
The same ordering holds at c8 and, apart from a noisy c1 (24 requests), at c1.

**This is a statement about our configuration, not about EAGLE3 in general.** See
the confounds section — we do not run SGLang's default backend.

## Result 3 — the workload's heterogeneity is real and large

Per-class mean accepted drafts per round, c32, 384 prompts each:

```text
class         K=1     K=3     K=5     K=7
repetitive   0.606   1.077   1.391   1.660
open_ended   0.305   0.476   0.485   0.499
```

A 3.3x spread at K=7. The draft model is doing exactly its job: it predicts
predictable text much better than open-ended text. The pooled acceptance is low
because `mixed_50_50` is half open-ended **by design** — this workload was built
to span easy and hard, and it does.

That is the project's premise, and it survives. What does not survive is the
*oracle* used to quantify it.

## Result 4 — the shipped controller already drives shallow

Active `K` per decode iteration, pooled over the three adaptive captures:

```text
K=0: 17.4%   K=1: 48.7%   K=3: 31.8%   K=5: 1.2%   K=7: 0.9%
```

An independent signal that agrees with Result 2: SGLang's merged adaptive
controller almost never selects `K >= 5` here.

## Correctness control — speculation is lossless

At temperature 0, speculative decoding must reproduce the target's greedy output
exactly. Comparing `completion_tokens` per prompt index across depths:

```text
c8 : K=1 vs K=3, K=5, K=7 -> 0/192 prompts differ
c32: K=1 vs K=3, K=5, K=7 -> 0/768 prompts differ
```

Identical token counts for every prompt at every depth. Combined with Result 3
(the draft is better on predictable text, as it should be), this is evidence that
the EAGLE3 integration is working, and that the throughput result is a genuine
cost/benefit outcome rather than a broken pipeline.

## Confounds — read Result 2 with these attached

SGLang's own EAGLE3 fixture and our launch config differ on three axes, and the
first is a plausible large contributor to Result 2:

| Setting | SGLang fixture | ours | note |
| --- | --- | --- | --- |
| `attention_backend` | **flashinfer** | **triton** | deviation; tradeoff never weighed |
| `dtype` | bfloat16 | float16 | |
| `mem_fraction_static` | 0.80 | 0.70 | sized for 48 GB |

The backend choice is *recorded* — the lockfile header notes that running triton
keeps flashinfer off the attention path, which avoids pulling flashinfer's separate
cubin/jit-cache wheels — but the tradeoff against SGLang's own default is nowhere
weighed, and this is the setting most likely to affect the *verify* pass
specifically: that pass attends over `K+1` positions, exactly the shape a backend
can handle well or badly. **Result 2 therefore describes EAGLE3-on-triton, not
EAGLE3.**

The *relative* ordering (shallow >= deep) is internally valid because every arm
used the same backend. Its *portability* is not established.

## Not established

* **No gap number.** Suppressed by the gate; not computed in any reportable form.
* **No claim about EAGLE3 or SGLang in general.** Only about this configuration.
* **Not a bug report.** Result 4 of `docs/05` (the `K=0` step tier) is not
  measured: `K=0` cells exist only incidentally as a reference at c32, and the
  oracle's action set excludes them by design.
* **No Session 2 justification yet.** Nothing here meets the bar.

## Open questions, cheapest first

1. **Why is `P(A_1)` higher at K=1 than at K>=3?** CPU-only: read
   `eagle_worker_v2.py` at the pinned commit for a single-step code path. If K=1
   executes differently, the invariance failure may be a `K=1` implementation
   artefact rather than a property of the acceptance distribution.
2. **Does the attention backend change Result 2?** One short A6000 run of the
   `K=0/1/3` cells with `flashinfer` would separate "speculation does not pay"
   from "triton does not pay for speculation".
3. **Is the invariance failure backend-dependent?** Same run answers this.

---

## Appendix — fork unit tests verified on this host

These import torch, so the GPU host is the first place they can run at all
(runbook §4b). Both branches were exercised during the session:

```text
PR 1 — heterospec/policy-feedback-identity @ 65aaba0418
$ python3 -m pytest -q \
    test/registered/unit/spec/test_adaptive_runtime_state.py \
    test/registered/unit/spec/test_adaptive_spec_params.py \
    test/registered/unit/managers/test_batch_result_processor_spec_grammar.py
52 passed, 15 warnings

Trace patch — heterospec/iter-telemetry @ f9281ec128
$ python3 -m pytest -q test/registered/unit/spec/test_heterospec_trace.py
8 passed, 1 warning
```

Warnings are unrelated: an unknown `asyncio_mode` option in SGLang's own
`pytest.ini`, and a `torch.jit` deprecation notice. No failures.
