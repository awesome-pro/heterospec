"""Oracle analysis: how much throughput is recoverable from better information.

The levels, and why they are defined this way, are derived in
`docs/03-oracle-design.md`. In short:

* ``sum_i E[acc_i | K] = n * sum_{k<=K} Sbar_k`` exactly, so for a **fixed batch**
  the optimal K depends only on the *pooled* survival curve ``Sbar``. Per-request
  structure cannot change it. A "per-request vs batch-average" comparison is
  therefore zero by construction and must not be reported.
* The real, recoverable gap is **curve vs scalar**: the shipped controller has a
  single EMA number, and a scalar cannot identify a curve. Two batches can share
  a mean and want different K.

Levels:

| Level | Information | Emits |
| --- | --- | --- |
| L0 | none (one K for the run) | one K |
| L1 | a scalar summary of the batch | one K per batch |
| L2 | pooled survival curve of the batch | one K per batch |
| L3 | per-request curves | K per request (upper bound, not implementable) |

``recoverable_rectangular_gap = L2 - L1``.

Aggregation
-----------
Throughput is **not** the sum of per-batch goodputs. A batch that emits `T`
tokens at depth `K` needs about `T / tokens_per_round(K)` rounds, each costing
`Cost(K, n)`, so its time is `T * Cost(K, n) / tokens_per_round(K)`. Total
throughput is `sum(T) / sum(time)` -- a token-weighted harmonic aggregate, in
which a fast batch cannot compensate for a slow one by being added separately.
Maximising throughput is therefore equivalent to **minimising total time per
token**, and the shared-K levels (L0, L1) must be optimised on that criterion,
not on a sum of rates.

Because the per-batch cost per token is `Cost(K, n) / tokens_per_round(K)`, the
per-batch argmax is the same either way -- so L2's per-batch choice is
unaffected. It is L0 and L1, which pick one K for many batches, where the
distinction bites.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass, field

__all__ = [
    "BatchDecision",
    "BatchTrace",
    "CostModel",
    "LinearCostModel",
    "OracleResult",
    "aggregate_cost_per_token",
    "analyse_batches",
    "batch_cost_per_token",
    "batch_goodput",
    "best_k_curve_aware",
    "curve_aware_oracle",
    "expected_accepted",
    "per_request_best_k",
    "ScalarPolicy",
    "bin_index",
    "fit_bin_edges",
    "fit_scalar_policy",
    "pooled_survival",
    "scalar_oracle",
    "throughput",
    "shipped_heuristic_k",
    "tokens_per_round",
]


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------


class CostModel:
    """Per-round cost of one decode round at depth ``k`` for batch ``n``.

    Subclass this, or pass any object with a matching ``cost`` method. Costs are
    *relative*; only ratios across `(k, n)` matter, so any consistent unit
    (milliseconds per round) works.
    """

    def cost(self, k: int, n: int) -> float:  # pragma: no cover - interface
        raise NotImplementedError


@dataclass(frozen=True)
class LinearCostModel(CostModel):
    """``cost(k, n) = alpha + beta * k + gamma * k * n``.

    A placeholder until the GPU calibration grid provides a fitted model. The
    defaults encode only the qualitative facts that draft steps are cheaper than
    a full target verify and that a round's fixed cost is amortised over the
    batch. They are **not** measurements, so any gap computed with them is a
    sensitivity analysis, not an oracle result.
    """

    alpha: float = 1.0
    beta: float = 0.35
    gamma: float = 0.0

    def cost(self, k: int, n: int) -> float:
        if n <= 0:
            raise ValueError("batch size must be > 0")
        c = self.alpha + self.beta * k + self.gamma * k * n
        if c <= 0:
            raise ValueError(f"non-positive cost for k={k}, n={n}")
        return c


# ---------------------------------------------------------------------------
# Batch input
# ---------------------------------------------------------------------------


@dataclass
class BatchTrace:
    """One captured batch: per-request survival curves plus what was observed.

    ``survivals[rid][k-1] = S_k = P(accept >= k)``. Pass only clean
    (unconfounded) curves -- see `heterospec.telemetry`.
    """

    batch_id: int
    survivals: dict[str, tuple[float, ...]]
    active_k: int | None = None
    """Depth actually used when this batch was observed, if known."""

    observed_mean_accepted: float | None = None
    """Scalar the shipped controller would have seen for this batch."""

    weight: float = 1.0
    """Relative token volume of this batch. Equal weights by default."""

    def __post_init__(self) -> None:
        if not self.survivals:
            raise ValueError(f"batch {self.batch_id} has no requests")
        if self.weight <= 0:
            raise ValueError(f"batch {self.batch_id}: weight must be > 0")
        for rid, s in self.survivals.items():
            if not s:
                raise ValueError(f"batch {self.batch_id}: empty survival for {rid}")
            if any(b > a + 1e-12 for a, b in zip(s, s[1:], strict=False)):
                raise ValueError(
                    f"batch {self.batch_id}: survival for {rid} is not monotone "
                    f"non-increasing: {s}"
                )

    @property
    def n(self) -> int:
        return len(self.survivals)

    @property
    def max_depth(self) -> int:
        """Deepest K at which every request's curve is defined."""
        return min(len(s) for s in self.survivals.values())

    def scalar(self) -> float | None:
        """Scalar summary the current controller could use.

        Prefers ``observed_mean_accepted`` -- the honest case, since the
        controller only ever sees acceptance at the depth it actually ran. Falls
        back to the pooled mean at the deepest common depth, which is a
        *stronger* signal than the controller really has; that makes L1 stronger
        and the measured gap smaller, i.e. conservative.
        """
        if self.observed_mean_accepted is not None:
            return self.observed_mean_accepted
        if self.active_k is not None:
            if self.active_k == 0:
                return 0.0
            return (
                sum(sum(s[: self.active_k]) for s in self.survivals.values()) / self.n
            )
        d = self.max_depth
        if d == 0:
            return 0.0
        return sum(sum(s[:d]) for s in self.survivals.values()) / self.n


# ---------------------------------------------------------------------------
# Core quantities
# ---------------------------------------------------------------------------


def pooled_survival(batch: BatchTrace, depth: int | None = None) -> tuple[float, ...]:
    """``Sbar_k = (1/n) sum_i S^i_k`` out to ``depth`` (default: common depth)."""
    depth = batch.max_depth if depth is None else min(depth, batch.max_depth)
    if depth <= 0:
        return ()
    n = batch.n
    return tuple(sum(s[k] for s in batch.survivals.values()) / n for k in range(depth))


def expected_accepted(batch: BatchTrace, k: int) -> float:
    """``sum_i E[acc_i | k]`` via the pooled curve.

    Uses the linearity identity rather than summing per request: cheaper, and it
    is the point -- per-request detail cannot change this number.
    """
    if k <= 0:
        return 0.0
    return batch.n * sum(pooled_survival(batch, k))


def tokens_per_round(batch: BatchTrace, k: int) -> float:
    """Tokens emitted by one decode round: accepted drafts plus a bonus each."""
    return batch.n + expected_accepted(batch, k)


def batch_goodput(batch: BatchTrace, k: int, cost: CostModel) -> float:
    """``tokens_per_round / Cost(k, n)`` -- this batch's own throughput rate."""
    return tokens_per_round(batch, k) / cost.cost(k, batch.n)


def batch_cost_per_token(batch: BatchTrace, k: int, cost: CostModel) -> float:
    """``Cost(k, n) / tokens_per_round`` -- time per token. Lower is better.

    This is the quantity that aggregates correctly: total time is
    ``sum_b weight_b * cost_per_token_b``, and throughput is its reciprocal.
    """
    return cost.cost(k, batch.n) / tokens_per_round(batch, k)


def aggregate_cost_per_token(
    batches: Sequence[BatchTrace], ks: Sequence[int], cost: CostModel
) -> float:
    """Token-weighted total time per token, in relative units. Lower is better."""
    return sum(
        b.weight * batch_cost_per_token(b, k, cost)
        for b, k in zip(batches, ks, strict=True)
    )


def throughput(
    batches: Sequence[BatchTrace], ks: Sequence[int], cost: CostModel
) -> float:
    """Relative throughput = total tokens / total time.

    Normalised so that identical batches with a constant cost model give a
    value equal to the number of batches, making levels comparable.
    """
    total_weight = sum(b.weight for b in batches)
    return total_weight / aggregate_cost_per_token(batches, ks, cost)


def _per_request_goodput(
    survival: tuple[float, ...], k: int, cost: CostModel, n: int
) -> float:
    """Ranking criterion for one request's own best depth.

    Costed at the batch's measured size `n`, **not** at a batch of one. A batch of
    one is a shape no session measures -- calibration captures run at real
    concurrencies -- so `cost.cost(k, 1)` falls outside the grid and the cost model
    answers it by *clamping* to the smallest measured batch size. That would make
    L3's per-request choice rest on an unmeasured cost, which is precisely what the
    oracle is not allowed to do.

    Since `n` is constant across `k` for a given batch, this is an ordering on
    `tokens / round_cost` and it is used only to pick an argmax; it is not a rate
    and is never summed or averaged.
    """
    return (1.0 + sum(survival[:k])) / cost.cost(k, n)


def _candidates(batch: BatchTrace, candidates: Sequence[int] | None) -> list[int]:
    depth = batch.max_depth
    if candidates is None:
        return list(range(0, depth + 1))
    ks = [k for k in candidates if 0 <= k <= depth]
    if not ks:
        raise ValueError(
            f"no candidates within batch {batch.batch_id}'s common depth "
            f"{depth}: {list(candidates)}"
        )
    return sorted(set(ks))


def best_k_curve_aware(
    batch: BatchTrace, cost: CostModel, candidates: Sequence[int] | None = None
) -> int:
    """L2: best single K for this batch given its pooled curve."""
    ks = _candidates(batch, candidates)
    return max(ks, key=lambda k: batch_goodput(batch, k, cost))


def per_request_best_k(
    batch: BatchTrace, cost: CostModel, candidates: Sequence[int] | None = None
) -> tuple[int, ...]:
    """L3: each request independently at its own best K (upper bound only)."""
    ks = _candidates(batch, candidates)
    return tuple(
        max(ks, key=lambda k: _per_request_goodput(s, k, cost, batch.n))
        for s in batch.survivals.values()
    )


def shipped_heuristic_k(
    scalar: float,
    candidates: Sequence[int],
    *,
    ceiling: int | None = None,
) -> int:
    """Approximate SGLang's mapping ``target ~= round(ema_accept_len) + 1``.

    A *reference point*, not L1. The shipped rule is one particular scalar-to-K
    mapping; L1 maximises over all of them, so ``L1 >= shipped`` by construction.
    Reporting the gap against the shipped heuristic instead of against L1 would
    overstate the result.
    """
    target = int(round(scalar)) + 1
    if ceiling is not None:
        target = min(target, ceiling)
    ks = sorted(set(candidates))
    return min(ks, key=lambda k: (abs(k - target), k))


# ---------------------------------------------------------------------------
# Levels
# ---------------------------------------------------------------------------


@dataclass
class BatchDecision:
    batch_id: int
    n: int
    scalar: float | None
    k_scalar: int | None
    k_curve: int
    k_shipped: int | None
    k_per_request: tuple[int, ...]
    goodput_curve: float
    goodput_scalar: float | None
    goodput_shipped: float | None

    @property
    def k_disagrees(self) -> bool:
        """Whether scalar feedback and the pooled curve pick different K."""
        return self.k_scalar is not None and self.k_scalar != self.k_curve


@dataclass
class OracleResult:
    """Relative throughput at each level, plus per-batch decisions."""

    n_batches: int
    candidates: tuple[int, ...]
    cost_model_repr: str

    l0_k: int
    l0_throughput: float
    l1_throughput: float
    l1_shipped_throughput: float
    l2_throughput: float
    l3_throughput: float

    decisions: list[BatchDecision] = field(default_factory=list)
    scalar_bins: int = 0
    notes: list[str] = field(default_factory=list)

    n_train: int = 0
    n_test: int = 0
    train_fraction: float = 0.5
    seed: int = 0
    scalar_policy_repr: str = ""

    def _rel(self, a: float, b: float) -> float:
        return (a - b) / b if b else float("nan")

    @property
    def l1_baseline_throughput(self) -> float:
        """The strongest scalar policy actually available: `max(L1, shipped)`.

        The shipped ``round(ema)+1`` rule *is* a scalar policy, so the scalar
        baseline must be at least as strong as it. A scalar map fitted on the
        training split can generalise *worse* than this fixed rule -- ordinary
        overfitting -- so taking the maximum is both correct and conservative:
        it can only shrink the measured gap.
        """
        return max(self.l1_throughput, self.l1_shipped_throughput)

    @property
    def recoverable_rectangular_gap(self) -> float:
        """**L2 vs the stronger scalar baseline**, relative.

        The number that decides whether the project is worth building. Measured
        against `l1_baseline_throughput`, not against a weak heuristic.
        """
        return self._rel(self.l2_throughput, self.l1_baseline_throughput)

    @property
    def total_adaptation_gain(self) -> float:
        """L2 - L0: everything per-batch adaptation buys."""
        return self._rel(self.l2_throughput, self.l0_throughput)

    @property
    def ragged_upper_bound(self) -> float:
        """L3 - L2: what would need heterogeneous execution (out of scope)."""
        return self._rel(self.l3_throughput, self.l2_throughput)

    @property
    def scalar_fit_overfit(self) -> float:
        """Fitted scalar map vs the fixed shipped rule, on held-out batches.

        Negative means the fitted map generalised worse than the fixed rule --
        overfitting, not a code fault. Reported as a diagnostic rather than a
        result.
        """
        return self._rel(self.l1_throughput, self.l1_shipped_throughput)

    @property
    def fraction_batches_disagreeing(self) -> float:
        if not self.decisions:
            return 0.0
        return sum(d.k_disagrees for d in self.decisions) / len(self.decisions)

    def summary(self) -> dict[str, object]:
        return {
            "n_batches": self.n_batches,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "train_fraction": self.train_fraction,
            "seed": self.seed,
            "scalar_policy": self.scalar_policy_repr,
            "candidates": list(self.candidates),
            "cost_model": self.cost_model_repr,
            "l0_k": self.l0_k,
            "l0_throughput": self.l0_throughput,
            "l1_throughput": self.l1_throughput,
            "l1_shipped_throughput": self.l1_shipped_throughput,
            "l1_baseline_throughput": self.l1_baseline_throughput,
            "l2_throughput": self.l2_throughput,
            "l3_throughput": self.l3_throughput,
            "recoverable_rectangular_gap": self.recoverable_rectangular_gap,
            "total_adaptation_gain": self.total_adaptation_gain,
            "ragged_upper_bound": self.ragged_upper_bound,
            "scalar_fit_overfit": self.scalar_fit_overfit,
            "fraction_batches_disagreeing": self.fraction_batches_disagreeing,
            "notes": self.notes,
        }


def fit_bin_edges(scalars: Sequence[float], n_bins: int) -> list[float]:
    """Equal-width bin edges spanning the scalars seen during *fitting*.

    Edges come from the training split only. That matters: binning relative to
    the *evaluation* split's range would let the scalar policy place every
    evaluation batch in its own bin, memorising them and collapsing the measured
    gap to zero regardless of the truth.
    """
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    lo, hi = min(scalars), max(scalars)
    if hi <= lo:
        # Degenerate range: one bin, so the policy has a single action.
        return [lo, lo + 1e-9]
    width = (hi - lo) / n_bins
    return [lo + i * width for i in range(n_bins + 1)]


def bin_index(scalar: float, edges: Sequence[float]) -> int:
    """Index of the bin containing ``scalar``, clamped to the fitted range.

    Clamping is deliberate: an evaluation batch outside the training range must
    fall back to the nearest fitted action, not invent a new one.
    """
    if len(edges) < 2:
        raise ValueError("need at least two edges")
    n_bins = len(edges) - 1
    width = edges[1] - edges[0]
    if width <= 0:
        return 0
    idx = int((scalar - edges[0]) / width)
    return max(0, min(n_bins - 1, idx))


@dataclass
class ScalarPolicy:
    """A fitted scalar-to-K lookup: the strongest policy of its information class."""

    edges: list[float]
    bin_to_k: dict[int, int]
    n_bins: int

    def k_for(self, scalar: float) -> int:
        return self.bin_to_k[bin_index(scalar, self.edges)]

    def describe(self) -> str:
        return (
            f"ScalarPolicy({self.n_bins} bins over "
            f"[{self.edges[0]:.3f}, {self.edges[-1]:.3f}], "
            f"K values {sorted(set(self.bin_to_k.values()))})"
        )


def fit_scalar_policy(
    batches: Sequence[BatchTrace],
    cost: CostModel,
    candidates: Sequence[int],
    *,
    n_bins: int = 20,
) -> ScalarPolicy:
    """Fit the best scalar-to-K map on ``batches`` (the training split).

    Within each bin pick the single K minimising the summed time per token of the
    batches in that bin. That is the Bayes-optimal action given the binned
    scalar, so the resulting policy is a genuinely strong baseline: the shipped
    ``round(ema)+1`` rule is one particular map, and this maximises over maps at
    the same information level.
    """
    scalars = [b.scalar() for b in batches]
    if any(s is None for s in scalars):
        raise ValueError("every batch needs a scalar to fit the scalar policy")
    vals = [s for s in scalars if s is not None]

    edges = fit_bin_edges(vals, n_bins)
    bins: dict[int, list[int]] = {}
    for i, s in enumerate(vals):
        bins.setdefault(bin_index(s, edges), []).append(i)

    bin_to_k: dict[int, int] = {}
    for b, idxs in bins.items():
        bin_to_k[b] = min(
            candidates,
            key=lambda k: sum(
                batches[i].weight * batch_cost_per_token(batches[i], k, cost)
                for i in idxs
            ),
        )

    # Every bin must have an action, not just the populated ones. An evaluation
    # batch can easily land in a bin no training batch occupied -- with many bins
    # and a finite training split that is the common case, not an edge case -- and
    # a lookup policy must still return an action for it.
    populated = sorted(bin_to_k)
    if not populated:  # pragma: no cover - defensive; fit_bin_edges needs data
        raise ValueError("no training batches populated any bin")
    for b in range(n_bins):
        if b in bin_to_k:
            continue
        # Nearest populated bin by index distance; ties go to the lower bin.
        nearest = min(populated, key=lambda p: (abs(p - b), p))
        bin_to_k[b] = bin_to_k[nearest]

    return ScalarPolicy(edges=edges, bin_to_k=bin_to_k, n_bins=n_bins)


def scalar_oracle(
    train: Sequence[BatchTrace],
    test: Sequence[BatchTrace],
    cost: CostModel,
    candidates: Sequence[int],
    *,
    n_bins: int = 20,
) -> tuple[float, list[int], ScalarPolicy]:
    """L1: fit the scalar policy on ``train``, evaluate it on ``test``.

    Returns ``(throughput_on_test, per_batch_k_on_test, policy)``.
    """
    policy = fit_scalar_policy(train, cost, candidates, n_bins=n_bins)
    per_batch_k = [policy.k_for(b.scalar()) for b in test]  # type: ignore[arg-type]
    return throughput(test, per_batch_k, cost), per_batch_k, policy


def curve_aware_oracle(
    batches: Sequence[BatchTrace],
    cost: CostModel,
    candidates: Sequence[int] | None = None,
) -> tuple[float, list[int]]:
    """L2: per-batch optimum from the pooled curve. One K per batch."""
    ks = [best_k_curve_aware(b, cost, _candidates(b, candidates)) for b in batches]
    return throughput(batches, ks, cost), ks


def analyse_batches(
    batches: Sequence[BatchTrace],
    cost: CostModel | None = None,
    *,
    candidates: Sequence[int] | None = None,
    n_bins: int = 20,
    train_fraction: float = 0.5,
    seed: int = 0,
) -> OracleResult:
    """Compute L0..L3 and the recoverable rectangular gap.

    A **train/test split is required**, and it is not a formality. L0 (one global
    K) and L1 (a scalar-to-K map) are *fitted* policies: they must generalise
    from batches they were tuned on to batches they have not seen. Evaluating
    them on the same batches used for fitting would let a finely binned scalar
    policy place every batch in its own bin, memorise it, and match L2 exactly --
    reporting a gap of zero no matter how much information the curve really
    carries.

    L2 and L3 need no fitting: they act per batch directly on that batch's own
    curve, which is precisely the information a request-aware policy would have.

    All levels are evaluated on the **test** split; fitting uses the train split.
    """
    if not batches:
        raise ValueError("no batches to analyse")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1)")
    cost = cost or LinearCostModel()

    # Deterministic split.
    order = list(range(len(batches)))
    random.Random(seed).shuffle(order)
    n_train = min(len(batches) - 1, max(1, round(len(batches) * train_fraction)))
    train = [batches[i] for i in order[:n_train]]
    test = [batches[i] for i in order[n_train:]]
    if not test:  # only possible with a single batch
        train, test = batches, batches

    # L0 and L1 apply one K across many batches, so candidates must exist within
    # every batch's common depth.
    common = min(b.max_depth for b in batches)
    if candidates is None:
        ks = list(range(0, common + 1))
    else:
        ks = sorted({k for k in candidates if 0 <= k <= common})
    if not ks:
        raise ValueError(
            f"no candidate depths within the common depth {common} of all batches"
        )

    # L0: one global K, fitted on train, evaluated on test.
    l0_k = min(
        ks,
        key=lambda k: aggregate_cost_per_token(train, [k] * len(train), cost),
    )
    l0 = throughput(test, [l0_k] * len(test), cost)

    # L1: best scalar-to-K map fitted on train, evaluated on test.
    l1, l1_ks, policy = scalar_oracle(train, test, cost, ks, n_bins=n_bins)

    # The shipped rule has no fitted parameters, so it is applied directly.
    shipped_ks: list[int | None] = []
    for b in test:
        s = b.scalar()
        shipped_ks.append(None if s is None else shipped_heuristic_k(s, ks))
    if any(k is None for k in shipped_ks):
        raise ValueError("every evaluated batch needs a scalar for the shipped rule")
    shipped_ks_int = [k for k in shipped_ks if k is not None]
    l1_shipped = throughput(test, shipped_ks_int, cost)

    # L2: per-batch optimum from that batch's own pooled curve, no fitting.
    l2, l2_ks = curve_aware_oracle(test, cost, ks)

    # L3: ragged per-request execution, as an upper bound.
    #
    # Deliberately loose, because PROJECT.md only ever wants an upper bound here.
    # For each batch we grant the *per-request-optimal* token count (every request
    # at its own best K) while charging only the *cheapest* rectangular round cost
    # the batch could have paid. Both concessions are favourable, so the result is
    # guaranteed >= L2.
    #
    # Costing the ragged round at the deepest K present instead would be
    # *pessimistic* and can land below L2, which would misrepresent a bound as an
    # achievable target.
    #
    # Uses the same harmonic aggregation as L0/L1/L2: summing per-batch rates here
    # would mix scales and make the ratio meaningless.
    l3_terms: list[float] = []
    decisions: list[BatchDecision] = []
    for i, b in enumerate(test):
        kpr = per_request_best_k(b, cost, _candidates(b, ks))
        per_req_tokens = sum(
            1.0 + sum(s[:k]) for s, k in zip(b.survivals.values(), kpr, strict=True)
        )
        cheapest_round = min(cost.cost(k, b.n) for k in ks)
        l3_terms.append(b.weight * cheapest_round / per_req_tokens)
        decisions.append(
            BatchDecision(
                batch_id=b.batch_id,
                n=b.n,
                scalar=b.scalar(),
                k_scalar=l1_ks[i],
                k_curve=l2_ks[i],
                k_shipped=shipped_ks_int[i],
                k_per_request=kpr,
                goodput_curve=batch_goodput(b, l2_ks[i], cost),
                goodput_scalar=batch_goodput(b, l1_ks[i], cost),
                goodput_shipped=batch_goodput(b, shipped_ks_int[i], cost),
            )
        )

    l3 = sum(b.weight for b in test) / sum(l3_terms)

    notes = [
        "L0 and L1 are fitted on the train split and evaluated on the test "
        "split; L2 and L3 act per batch with no fitting. Evaluating fitted "
        "policies on their own training batches would let a fine-grained scalar "
        "policy memorise each batch and report a false zero gap.",
        "L3 is a deliberately loose upper bound: it grants per-request-optimal "
        "token counts while charging the cheapest rectangular round cost. L3 >= "
        "L2 holds by construction. It is not a target and is not implementable "
        "within one rectangular K per batch.",
        "All levels aggregate as total tokens / total time (token-weighted time "
        "per token), not as a sum of per-batch rates.",
    ]
    if isinstance(cost, LinearCostModel):
        notes.append(
            "Cost model is a placeholder (LinearCostModel), not a GPU "
            "measurement. Treat any gap as a sensitivity analysis."
        )
    if n_train < 4:
        notes.append(
            f"Only {n_train} training batches; the scalar policy is barely "
            f"constrained and L1 is likely optimistic."
        )

    return OracleResult(
        n_batches=len(batches),
        candidates=tuple(ks),
        cost_model_repr=repr(cost),
        l0_k=l0_k,
        l0_throughput=l0,
        l1_throughput=l1,
        l1_shipped_throughput=l1_shipped,
        l2_throughput=l2,
        l3_throughput=l3,
        decisions=decisions,
        scalar_bins=n_bins,
        notes=notes,
        n_train=len(train),
        n_test=len(test),
        train_fraction=train_fraction,
        seed=seed,
        scalar_policy_repr=policy.describe(),
    )
