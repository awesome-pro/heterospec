# 03 — What the oracle can and cannot measure

This document corrects the oracle framing in PROJECT.md, and records two
measurement traps found by testing the implementation. Both would have produced
confidently wrong numbers after a GPU spend.

Status: **design settled, cost model still a placeholder.** No number in this
document may be quoted as a result until `Cost(K, n)` is GPU-calibrated
(`docs/04-cost-model.md`).

---

## 1. The objective

For a batch of `n` requests decoded with one rectangular depth `K`:

```text
tokens this round = n + sum_i acc_i        (one bonus token per request)
goodput(K)        = tokens / Cost(K, n)
```

**Aggregation trap.** Throughput over a run is *not* the sum of per-batch
goodputs. A batch emitting `T` tokens at depth `K` needs about
`T / tokens_per_round(K)` rounds at `Cost(K, n)` each, so its time is
`T · Cost(K, n) / tokens_per_round(K)`. Total throughput is `sum(T) / sum(time)` —
a token-weighted *harmonic* aggregate. A fast batch cannot simply be added to a
slow one. So every level is compared as:

```text
throughput = total_weight / sum_b weight_b · Cost(K_b, n_b) / tokens_per_round_b(K_b)
```

Maximising throughput is equivalent to minimising summed time per token, and that
is the criterion used to fit L0 and L1. An early implementation summed per-batch
rates for L3 and produced `982` against `~5` for the other levels — a
meaningless ratio that would have looked like a spectacular upper bound.

## 2. The linearity identity — and what it rules out

With per-request survival `S^i_k = P(request i accepts draft position k)`:

```text
sum_i E[acc_i | K] = sum_i sum_{k<=K} S^i_k
                   = sum_{k<=K} sum_i S^i_k
                   = n · sum_{k<=K} Sbar_k
```

where `Sbar_k = (1/n) sum_i S^i_k` is the **pooled** survival curve. Verified to
machine precision: the maximum discrepancy was `1.8e-15` over 2000 random batches
of random size and shape.

**Consequence.** For a fixed batch, `goodput(K)` is a function of `Sbar` alone.
Per-request structure — however heterogeneous — cannot change the optimal `K` or
the achievable goodput.

Therefore the comparison PROJECT.md originally implies:

```text
"best batch-wide K using request-specific info"   vs   "batch-average K"
```

is **zero by construction**, because both have `Sbar`. Measuring it would have
returned a null result that reads as evidence against the hypothesis. This is the
single most important correction in this document.

## 3. Where the real, recoverable gap is

The shipped controller does not have `Sbar`. It has a **scalar**:
`AdaptiveStepSlot` reduces the per-request list to `batch_avg`, smooths it into
one EMA, and maps that single number to a tier
(`adaptive_spec_params.py:186-250`).

A scalar cannot identify a curve. Two batches can share a mean and differ
everywhere else, and therefore want different `K`.

| Level | Policy class | Information | Emits |
| --- | --- | --- | --- |
| **L0** | Global static K | none | one K for the run |
| **L1** | Strongest scalar-feedback policy | a scalar summary per batch | one K per batch |
| **L2** | Curve-aware policy | pooled curve `Sbar` of the current batch | one K per batch |
| **L3** | Ragged per-request K | per-request curves | K per request (upper bound) |

```text
recoverable rectangular gap = L2 - L1     <- the number that decides the project
total per-batch adaptation  = L2 - L0
needs ragged execution      = L3 - L2     <- upper bound only, out of scope
```

**The claim must be stated as "a curve beats a scalar", not "per-request beats
batch-average".** The latter is provably a no-op and is an easy review rejection.

## 4. Why request-level telemetry is still the right mechanism

If `Sbar` is all that matters, why keep per-request state instead of a batch
histogram? Because of *when* the decision is made and *what changes*:

* The controller picks `K` **before** the round runs, for the composition about to
  be scheduled.
* Composition changes every step: requests finish, new ones are admitted,
  retractions remove others.
* The pooled curve of the **previous** batch is not the pooled curve of the
  **next** one. A passive batch-level histogram only reports the past.

Per-request curves let the controller **re-pool for the composition actually in
front of it**. That re-pooling is the contribution; the per-request decomposition
is the enough-statistic for it.

Useful corollary, and a built-in control: for a *homogeneous* batch, re-pooling
reduces to the scalar path, so the expected result is
`HeteroSpec ≈ existing controller` on homogeneous traffic.

## 5. Measurement trap 1: fitting and evaluating on the same batches

L0 and L1 are **fitted** policies. Evaluating them on the batches used for fitting
lets a finely binned scalar policy place each batch in its own bin, memorise it,
and match L2 exactly — reporting a gap of **zero** regardless of the truth.

An early implementation binned the scalar relative to the observed range, which
with few batches makes bins arbitrarily fine. Two batches whose scalars differed
by `0.0004` landed in different bins and the measured gap was `0.00%`.

**Fix:** a mandatory train/test split. Bin edges and L0's global `K` are fitted on
the training batches; all levels are reported on held-out batches. The scalar
policy is then the best *generalising* scalar policy, which is what a real
controller is.

Also required: every bin must have an action. An evaluation batch can land in a
bin no training batch occupied, which is the common case with many bins and a
finite split. A sparse lookup raised `KeyError: 3` before this was fixed; empty
bins now inherit the nearest populated bin's action.

## 6. Measurement trap 2: a bound that is not a bound

PROJECT.md asks for L3 only as an upper bound. Costing the ragged round at the
**deepest** `K` present ("one fused pass") is a *pessimistic* model of ragged
execution and can land **below** L2 — observed at `4.886` against `4.940`. That
misrepresents a bound as an achievable target.

**Fix:** L3 is deliberately loose, as a bound should be. For each batch it grants
the per-request-optimal token count while charging the **cheapest** rectangular
round cost the batch could have paid. Both concessions are favourable, so
`L3 >= L2` holds by construction — now asserted in the tests.

## 7. The demonstration, at population level

Two batch types of size 4:

```text
X = 4 homogeneous "medium" requests                  -> optimal K = 1
Y = 2 deep acceptors + 2 near-dead requests          -> optimal K = 3
```

`Y`'s low-acceptance profile is chosen so that the batch mean accepted at `K=3`
**equals** `X`'s (`1.3806`). The scalar therefore carries no information about
which type a batch is.

Synthetic populations of 400 batches, placeholder cost model, 50/50 train/test
split, 20 bins:

| Population | optimal K X/Y | L0 | L1 fitted | shipped | **L1 baseline** | L2 | L3 | gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Means matched, tiny noise | 1 / 3 | 4.8894 | 4.8730 | 4.8894 | 4.8894 | 4.9396 | 8.6205 | **+1.03%** |
| Means matched, realistic noise | 1 / 3 | 4.8894 | 4.8757 | 4.8894 | 4.8894 | 4.9396 | 8.6205 | **+1.03%** |
| Y shifted +0.06 (scalar informative) | 1 / 3 | 4.8894 | 4.9191 | 4.8894 | 4.9191 | 4.9396 | 8.6205 | **+0.42%** |
| Homogeneous control | 1 / 1 | 5.2741 | 5.2741 | 5.2741 | 5.2741 | 5.2741 | — | **0.000000%** |

Note the fitted L1 column sits *below* the fixed shipped rule in rows 1–2
(4.8730 vs 4.8894). That is ordinary overfitting on held-out data, and it is why
the scalar baseline is `max(L1, shipped)` rather than the fitted map alone.
Reporting the gap against a policy that happened to overfit would overstate the
curve's advantage.

Four readings:

1. **The gap measures how much the scalar reveals about the optimal K.** When the
   scalar separates the types (row 3) the gap collapses by ~60%. When it cannot
   (rows 1–2) the curve is worth ~1%.
2. **The homogeneous control is exactly zero** — `0.000000%`, not merely small —
   as the linearity identity requires. This is the built-in falsification check.
3. **~1% is an adversarial *upper* bound, and it is comfortably below
   PROJECT.md's `<2–3% ⇒ do not build` threshold.** The cost model is a
   placeholder and the population is synthetic, so this is not a result. But it
   means the go/no-go genuinely hinges on whether real workloads produce *more*
   scalar-confounding heterogeneity than a purpose-built adversarial family could.
   That question is now the single most important thing GPU Session 1 must answer.
4. **The fitted L1 can be worse than no fitting at all**, so L1 is only useful as
   a *bound* on the scalar class, not as a deployable policy.

An earlier pair-level version of this demonstration reported `+1.73%` by
hand-matching two individual batches. The population figure is the honest one;
the pair number was cherry-picked and has been dropped.

## 8. How L1 is operationalised

L1 must be the **strongest** scalar policy, not the shipped heuristic. Beating a
weak baseline would overstate the result. Procedure:

1. Compute the scalar each batch would expose (batch mean accepted at the active
   `K`).
2. Fit equal-width bin edges over the **training** scalars.
3. Within each bin choose the `K` minimising summed time per token of the training
   batches in it; empty bins inherit the nearest populated bin.

This is the Bayes-optimal action given the binned scalar, so `L2 − L1` is a
*lower* bound on the curve advantage. Finer bins make L1 stronger and the gap
smaller — the conservative direction, deliberately.

**The headline baseline is `max(L1 fitted, shipped)`.** The shipped
`round(ema)+1` rule *is* a scalar policy, so the scalar class must be credited
with it; and a map fitted on the training split can generalise worse than the
fixed rule (observed: 4.8730 vs 4.8894), so the maximum is both correct and
conservative. `recoverable_rectangular_gap` is measured against that maximum.

The fitted-vs-shipped difference is reported separately as `scalar_fit_overfit`.
It is a diagnostic, not a result: it measures the tuning behaviour of one
heuristic, not the value of better information.

## 9. The counterfactual, restated

Every level evaluates `E[acc_i | K]` at depths a request may never have run.
That is valid only if **`S^i_k` does not depend on `K`** — the k-th draft token is
generated from the same prefix regardless of total steps. Approximately true for
EAGLE topk=1 chain drafting, but an assumption, and load-bearing for every number
above.

Tested explicitly (`docs/05-k-invariance.md`) on the same static-K grid that
calibrates the cost model. If it fails, all oracle numbers become
uninterpretable upper bounds and the project stops.

## 10. Consequences to carry forward

* **Framing in README and PRs must be "curve vs scalar".** "Per-request vs batch"
  is provably a no-op.
* **The comparison to beat is the best generalising scalar policy**, fitted on
  held-out data — not the EMA heuristic.
* **The telemetry requirement is weaker than first thought.** The policy needs
  enough information to reconstruct the current batch's pooled curve; per-request
  histograms supply it by re-pooling.
* **Homogeneous batches must show no gain.** Built-in control, currently exact.
* **Synthetic adversarial upper bound is ~1.3%, below the 2–3% threshold.** The
  go/no-go rests on real traces. If real workloads cannot produce more
  scalar-confounding heterogeneity than a crafted adversarial family, the honest
  answer is *do not build the controller* — and that is a publishable, useful
  result, not a failure.
