# 05 — Why the synthetic gap is an "upper bound", and what could move it

`docs/03-oracle-design.md` reports a recoverable rectangular gap of **~1.03%** on a
synthetic adversarial population. This document states precisely what that number
is, and every assumption standing between it and a real measurement.

The short version: **it is a ceiling, and most of the identified biases push it
down rather than up.** The reasons are listed with their direction so the
Session 1 result can be interpreted rather than merely read.

---

## 1. What "upper bound" means here — three separate senses

The phrase is doing three jobs and they should not be conflated.

| Sense | Claim | Strength |
| --- | --- | --- |
| **Information** | The gap measures what a *pooled curve* buys over a *scalar*, i.e. an upper bound on what better information can recover | solid — follows from the linearity identity |
| **Policy** | L2 has the realised curve of the very batch it is deciding for, so no deployable policy can do better | optimistic — a real policy must *predict* |
| **Population** | The batch types were constructed to maximise scalar-confounding disagreement | optimistic — real traffic may be far milder |

Only the first is a theorem. The other two are the reasons the number is not a
result.

## 2. Assumptions, with direction of bias

"↑ overstates the gap" means the assumption makes the synthetic number look
*better* than reality; "↓ understates" means reality could be better.

### A1. The cost model — placeholder now, and a proxy even when measured

`LinearCostModel(alpha=1.0, beta=0.35, gamma=0.0)`: cost depends linearly on `K`
and **not at all on batch size**. → **direction unknown.** This alone is why the
number is not yet a result. *Tested by:* the calibration grid (Session 1).

That grid replaces the placeholder with a *measured* surface, but the measured one
is still an **effective serving cost proxy at concurrency `n`**, not a model-step
cost, for two reasons:

* `n` is *client concurrency*. The real decode batch size decays within a wave as
  requests finish (32 → 27 → 23 → …), so `n` overstates the depth of most rounds.
* Wall time includes prefill, HTTP overhead, scheduler gaps, EOS variation and
  queue transitions — not just the target/draft decode steps.

That is sufficient for a go/no-go screen, which is all Session 1 is. It is **not**
sufficient to defend a headline number in the 3–5% band; there the result lives or
dies on a few percentage points, and cost must be measured directly (SGLang exposes
serving metrics and profiling) rather than inferred from client throughput. See
the conditional rule in §4.

### A2. `P(position k accepted)` is invariant to `K` — **the load-bearing one**

Every level evaluates `E[acc | K]` at depths a request may never have run. If the
per-position acceptance curve shifts with the total number of draft steps — draft
drift, KV-cache state, correlated errors over longer chains — then curves
measured at `K=3` do not predict `K=7`.

* If `S_k` **decays faster** at larger `K`, curves measured shallow overstate what
  deep tiers deliver, and the oracle overstates the benefit of depth. → **↑**
* If `S_k` is **stable**, the oracle is sound.
* If `S_k` **improves** with depth (unlikely), → **↓**.

*Tested by:* the same prompts at static `K ∈ {1,3,5,7}`, comparing survival curves
on overlapping positions. Free, because the calibration grid already produces
these runs. **If this fails, every oracle number becomes an uninterpretable upper
bound and the project stops.**

### A3. Wave dispatch batches ≈ real batches

Wave dispatch submits exactly `concurrency` requests and waits, so a wave is a
controlled mixture. Real continuous batching is messier: retractions, early
finishes, unequal progress. → **↑, probably.** Our workload generator *guarantees*
each wave mixes classes (asserted in tests), which maximises within-batch
heterogeneity by construction. Natural traffic may be more clustered.

*Tested by:* the adaptive run with the iteration trace gives true batch
composition. Comparing the gap from waves against the gap from real iterations
measures this directly.

### A4. Per-request curves use the request's whole lifetime

`batches_from_waves` builds each request's survival from its cumulative
histogram. A real online policy starts each request with **no** local history and
must blend from a population prior. → **↑.** The oracle gives every batch perfect
hindsight over the request's entire life.

*Mitigation:* Session 1 can re-run the gap with curves estimated causally (only
observations before the batch). If the gap collapses under that restriction, the
mechanism is not deployable even if the oracle is large. Worth doing if the
uncorrected gap is anywhere near the threshold.

### A5. The synthetic population was built to be adversarial

`X` (homogeneous medium, optimal `K=1`) and `Y` (bimodal, optimal `K=3`) were
*constructed*, with the low-acceptance profile tuned so both have the same batch
mean accepted at `K=3`. That is the best case for a curve-aware policy. → **↑,
strongly.** Real traffic is very unlikely to be this well-posed.

*Tested by:* every real workload family — `high`, `low`, the 75/25, 50/50, 25/75
mixtures, `phase_shift`, and `real_mixed` (code/reasoning/chat).

### A6. L2 is an oracle, not a policy

L2 chooses `K` using the *realised* pooled curve of the batch it is deciding
about. A deployable policy must predict that curve from history, and will sit
between L1 and L2. → **↑.** The gap is an upper bound on achievable policy gain,
not an estimate of it.

### A7. Cost is assumed independent of batch size

`gamma = 0` in the placeholder. If a round's fixed cost is amortised over `n`,
`Cost(K, n)` grows sublinearly in `n`, which changes which `K` is optimal at high
batch size. → **direction unknown.** *Tested by:* the `concurrency` axis of the
calibration grid.

### A8. Clean, temperature-0, 256-token generations

No preemption, no queue churn, deterministic sampling, one generation length.
Real serving has all of these. → **↑, probably**, since controlled conditions make
batches more predictable and the curve more informative than it is in production.

### A9. The scalar baseline may still be beatable

L1 is fitted on the training split and can overfit (observed: 4.8730 vs 4.8894 for
the fixed shipped rule), so we report against `max(L1, shipped)`. If a *better*
scalar policy exists outside our 20-bin grid, the true scalar baseline is
stronger and the gap smaller. → **↑.** Partly mitigated by using fine bins, and
by the fact that bins are fitted rather than evaluated in-sample.

### A10. Baseline is the shipped *mechanism's* class, not its tuning

`adaptive_spec_params.py` maps `round(ema)+1` through hysteresis, warmup and a
per-BS ladder. Our L1 is the best policy *of that information class*, which is a
stronger baseline than the shipped configuration. → **↓** relative to comparing
against the shipped controller as-is. This is deliberate.

### A11. Direction of the net effect

Assumptions that **overstate**: A3, A4, A5, A6, A8, A9.
Assumptions with **unknown** direction, each capable of moving the answer a lot:
A1, A2, A7.
Assumption that **understates**: A10.

Since the oracle optimisms all point the same way, the synthetic ~1% should be
treated as a **ceiling that is more likely to fall than to rise.** The honest
expectation going into Session 1 is that the real gap is *smaller* than 1%.

### Controls applied in Session 1

Distinct from the assumptions above, these are measurement confounders that are
now **controlled by construction** rather than left to hope:

| Confounder | Control | Why it matters |
| --- | --- | --- |
| Prefix/radix cache state | every server launches with `--disable-radix-cache` | The grid reuses one seeded prompt pool across concurrencies on one server. A warming cache would make later runs look cheaper and contaminate the batch-size axis — the same direction a real `gamma > 0` would produce, so the two would be indistinguishable. |
| Cold-start runtime paths | untimed warm-up at each step's own concurrency, discarded | `/health`-ready does not mean Triton kernels, allocator growth and lazy runtime paths for the measured shapes are warm. At a 1–3% effect size this is not a rounding error. |
| Tracer overhead on timed runs | tracing enabled **only** for the adaptive capture | The tracer does synchronous JSON work once per decode iteration. Calibration produces the cost model, i.e. the oracle's denominator, and its oracle input comes from `meta_info`, not the trace. |
| Traced-vs-untraced throughput comparison | adaptive capture is a *diagnostic*, not the adaptive-throughput baseline | Comparing a traced adaptive run against untraced static runs would not be a fair performance comparison. If HeteroSpec survives Session 2, the performance baselines are rerun with tracing off. |

Note that `--disable-radix-cache` lowers *absolute* throughput, so absolute numbers
from this session are not comparable with a cache-enabled run. Only the ratios this
project uses are — which is also why `Cost(K, n)` must be read as a relative proxy.

---

## 3. What could make the real gap *larger* than 1%

For completeness, the mechanisms that would work in the project's favour:

1. **A steeper real cost curve in `K`.** If `Cost` grows faster than `0.35K` at
   large batch size, choosing the wrong `K` is more expensive, so the same curve
   advantage converts into more throughput. The placeholder may understate this.
2. **Stronger real heterogeneity than the constructed pair.** Real code vs
   creative-writing traffic might separate *shapes* more than my `X`/`Y` pair does
   at equal mean.
3. **Larger batches.** A shared `K` is a bigger compromise across 32 requests than
   across 4; the placeholder's `gamma = 0` hides this entirely.
4. **Phase-shift traffic**, where composition changes faster than the EMA's
   `update_interval = 5` + `warmup_batches = 10` can track. The shipped policy is
   deliberately slow to switch; a curve-aware one need not be.
5. **The step-0 tier** in the default ladder means the controller can disable
   speculation entirely. If it does so at the wrong time, the cost of the mistake
   is large — larger than a wrong choice among `{1,3,5,7}`.

Mechanism 5 is the most promising and is not represented in the synthetic study
at all: the synthetic candidate set was `0..7` with a smooth cost, whereas the
real ladder is `[1,3,5,7]` at BS≥1 and includes `0` only at BS≥8.

---

## 4. Pre-registered decision rule

Fixed **before** seeing Session 1 data, so the result cannot be rationalised after
the fact. "Repeatable" means consistent in sign across the static-K captures and
the adaptive capture, and across concurrencies.

### Step 0 — the K-invariance gate, before reading the gap at all

Every level evaluates `E[acc | K]` at depths a request may never have run, which is
only valid if assumption A2 holds. So the check runs **first**, and:

```text
k_invariance FAIL          ->  gap uninterpretable. No-go, whatever its size.
k_invariance NOT ASSESSED  ->  fewer than two static depths usable. Fix and rerun.
k_invariance PASS          ->  proceed to the gap below.
```

Reporting a gap without this check would be reporting a number that may not mean
anything.

### Then the gap

| Measured real recoverable gap | Decision |
| --- | --- |
| `< 2%` | **No-go.** Do not build the policy. Publish the negative result. |
| `2–3%` | **Marginal.** Report honestly; pursue only if the mechanism is unusually clean. |
| `3–5%` | **Validate the cost model first.** The proxy is not precise enough to defend a number in this band; measure server-side step timing before claiming anything. |
| `> 5%` | **Pursue**, but only after the Session 2 confirmation below. |

### Session 2 — confirm before implementing

Session 1 screens on `mixed_50_50`, a *deliberately heterogeneous synthetic*
mixture. That is the right first target: if even a purpose-built mixture yields
`<2%`, the project is dead and nothing else needs measuring.

But a positive result could be an artifact of that mixture. So before implementing
HeteroSpec, run a smaller Session 2 on `real_mixed` and `phase_shift` and confirm
the phenomenon survives outside the constructed population. This keeps the
expensive part of the project gated on evidence rather than momentum.

Regardless of outcome, these are unaffected and remain worth doing:

* the **iteration-level trace** (already implemented, unit-tested, useful for any
  speculative-diagnostics work);
* the **request-identity feedback change** upstream, since a policy interface that
  cannot maintain request-local state is a real gap independent of whether
  HeteroSpec wins;
* the **linearity correction** in `docs/03`, which is a reusable methodological
  result about this class of controller.

## 5. What Session 1 must therefore produce

One session, six server launches, ~16 runs (`python -m heterospec.session --plan`):

1. the **measured** `Cost(K, batch_size)` grid — A1, A7;
2. the **K-invariance check** on identical prompts across `K` — A2;
3. **per-request acceptance profiles** per workload class — A5;
4. the gap from **wave batches** and from **true iteration batches** — A3;
5. the gap from every real workload family, not just the constructed one — A5;
6. a report that states which of the above passed, failed, or was inconclusive.

If A2 fails, or the real gap is below 2%, the answer is no-go and we say so.
