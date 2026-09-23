"""Tests for the oracle analysis.

Two of these are regression tests for bugs that would each have produced
confidently wrong numbers after a GPU spend:

* ``test_empty_bins_do_not_raise`` — an evaluation batch landing in an unpopulated
  bin raised ``KeyError`` (a sparse lookup policy).
* ``test_l3_is_a_true_upper_bound`` — costing ragged execution at the deepest K
  present made "the upper bound" *lower* than L2, misrepresenting a bound as a
  target.

The load-bearing scientific test is ``test_linearity_identity``: the identity it
checks is the reason a "per-request vs batch-average" comparison is a no-op.
"""

import random

import pytest

from heterospec.analysis.oracle import (
    BatchTrace,
    LinearCostModel,
    ScalarPolicy,
    aggregate_cost_per_token,
    analyse_batches,
    batch_cost_per_token,
    batch_goodput,
    best_k_curve_aware,
    bin_index,
    expected_accepted,
    fit_bin_edges,
    fit_scalar_policy,
    per_request_best_k,
    pooled_survival,
    scalar_oracle,
    shipped_heuristic_k,
    throughput,
    tokens_per_round,
)


def surv(*p: float) -> tuple[float, ...]:
    """Survival curve from conditional per-position acceptance probabilities."""
    out, prod = [], 1.0
    for x in p:
        prod *= x
        out.append(prod)
    return tuple(out)


X_P = (0.78, 0.55, 0.40, 0.26, 0.16, 0.10, 0.06)
HI_P = (0.98, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70)
# Tuned so 2*HI + 2*LO has the same mean accepted at K=3 as 4*X (1.3806),
# while its optimal K is 3 rather than 1. This makes the scalar uninformative
# about which type a batch is -- the adversarial case for a scalar policy.
LO_P = (0.0123, 0.001, 0.0, 0.0, 0.0, 0.0, 0.0)

COST = LinearCostModel(alpha=1.0, beta=0.35)


def _homogeneous(
    batch_id: int, n: int = 4, p=X_P, active_k: int | None = 3
) -> BatchTrace:
    return BatchTrace(
        batch_id=batch_id,
        survivals={f"x{i}": surv(*p) for i in range(n)},
        active_k=active_k,
    )


def _bimodal(batch_id: int, active_k: int | None = 3) -> BatchTrace:
    return BatchTrace(
        batch_id=batch_id,
        survivals={
            "h0": surv(*HI_P),
            "h1": surv(*HI_P),
            "l0": surv(*LO_P),
            "l1": surv(*LO_P),
        },
        active_k=active_k,
    )


# ---------------------------------------------------------------------------
# The identity that shapes the whole design
# ---------------------------------------------------------------------------


def test_linearity_identity():
    """sum_i E[acc_i|K] must equal n * sum_{k<=K} Sbar_k exactly.

    This is why per-request structure cannot change the optimal K for a fixed
    batch, and therefore why "per-request vs batch-average" is a no-op.
    """
    rng = random.Random(0)
    worst = 0.0
    for _ in range(500):
        n = rng.randint(1, 8)
        curves = {
            f"r{i}": surv(*(rng.uniform(0.01, 1.0) for _ in range(7))) for i in range(n)
        }
        b = BatchTrace(batch_id=0, survivals=curves)
        for k in range(1, 8):
            lhs = sum(sum(s[:k]) for s in curves.values())
            rhs = n * sum(pooled_survival(b, k))
            worst = max(worst, abs(lhs - rhs))
    assert worst < 1e-12, f"linearity identity violated by {worst}"


def test_two_batches_with_same_pooled_curve_score_identically():
    """Different composition, identical pooled curve -> identical goodput."""
    a = BatchTrace(
        batch_id=0,
        survivals={"p": surv(0.9, 0.8), "q": surv(0.5, 0.2)},
    )
    # Swap which request has which curve: composition differs, pooled doesn't.
    b = BatchTrace(
        batch_id=1,
        survivals={"p": surv(0.5, 0.2), "q": surv(0.9, 0.8)},
    )
    assert pooled_survival(a) == pytest.approx(pooled_survival(b))
    for k in range(0, 3):
        assert batch_goodput(a, k, COST) == pytest.approx(batch_goodput(b, k, COST))
    assert best_k_curve_aware(a, COST) == best_k_curve_aware(b, COST)


# ---------------------------------------------------------------------------
# BatchTrace validation
# ---------------------------------------------------------------------------


def test_batch_rejects_empty_survivals():
    with pytest.raises(ValueError, match="no requests"):
        BatchTrace(batch_id=0, survivals={})


def test_batch_rejects_empty_curve():
    with pytest.raises(ValueError, match="empty survival"):
        BatchTrace(batch_id=0, survivals={"a": ()})


def test_batch_rejects_non_monotone_curve():
    """A survival curve must be non-increasing; a rising one is a data bug."""
    with pytest.raises(ValueError, match="not monotone"):
        BatchTrace(batch_id=0, survivals={"a": (0.5, 0.8)})


def test_batch_rejects_non_positive_weight():
    with pytest.raises(ValueError, match="weight"):
        BatchTrace(batch_id=0, survivals={"a": (0.5,)}, weight=0.0)


def test_batch_max_depth_is_the_minimum_common_depth():
    b = BatchTrace(batch_id=0, survivals={"a": (0.9, 0.5, 0.2), "b": (0.8, 0.4)})
    assert b.max_depth == 2
    assert b.n == 2


# ---------------------------------------------------------------------------
# Core quantities
# ---------------------------------------------------------------------------


def test_pooled_survival_hand_computed():
    b = BatchTrace(batch_id=0, survivals={"a": (0.8, 0.4), "b": (0.6, 0.2)})
    assert pooled_survival(b) == pytest.approx((0.7, 0.3))


def test_tokens_per_round_includes_bonus_per_request():
    b = _homogeneous(0, n=4, p=(1.0, 1.0, 1.0))
    # 4 requests x (1 bonus + 3 accepted) = 16
    assert tokens_per_round(b, 3) == pytest.approx(16.0)
    assert tokens_per_round(b, 0) == pytest.approx(4.0)


def test_goodput_and_cost_per_token_are_reciprocal():
    b = _homogeneous(0)
    for k in range(0, 5):
        assert batch_goodput(b, k, COST) == pytest.approx(
            1.0 / batch_cost_per_token(b, k, COST)
        )


def test_expected_accepted_matches_manual_sum():
    b = BatchTrace(batch_id=0, survivals={"a": (0.9, 0.5, 0.1)})
    # S = (0.9, 0.5, 0.1); at K=2 -> 1*(0.9+0.5) = 1.4
    assert expected_accepted(b, 2) == pytest.approx(1.4)
    assert expected_accepted(b, 0) == 0.0


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------


def test_linear_cost_model_values():
    c = LinearCostModel(alpha=1.0, beta=0.5, gamma=0.0)
    assert c.cost(0, 4) == pytest.approx(1.0)
    assert c.cost(4, 4) == pytest.approx(3.0)


def test_linear_cost_model_rejects_bad_batch_size():
    with pytest.raises(ValueError, match="batch size"):
        LinearCostModel().cost(1, 0)


def test_linear_cost_model_rejects_non_positive_cost():
    with pytest.raises(ValueError, match="non-positive cost"):
        LinearCostModel(alpha=-5.0, beta=1.0).cost(1, 1)


# ---------------------------------------------------------------------------
# Scalar policy fitting -- regression: sparse lookup raised KeyError
# ---------------------------------------------------------------------------


def test_fit_bin_edges_spans_training_range():
    edges = fit_bin_edges([0.0, 1.0], 4)
    assert len(edges) == 5
    assert edges[0] == 0.0 and edges[-1] == 1.0


def test_fit_bin_edges_degenerate_range():
    edges = fit_bin_edges([2.0, 2.0], 4)
    assert len(edges) == 2
    assert edges[1] > edges[0]


def test_bin_index_clamps_outside_range():
    edges = fit_bin_edges([0.0, 1.0], 4)
    assert bin_index(-5.0, edges) == 0
    assert bin_index(99.0, edges) == 3
    assert bin_index(0.5, edges) == 2


def test_empty_bins_do_not_raise():
    """Regression: an evaluation batch in an unpopulated bin raised KeyError.

    With many bins and a finite training split this is the common case, not an
    edge case.
    """
    train = [_homogeneous(i) for i in range(3)]
    policy = fit_scalar_policy(train, COST, list(range(0, 8)), n_bins=50)
    assert len(policy.bin_to_k) == 50, "every bin must have an action"
    # A scalar far outside the training range must still return an action.
    assert policy.k_for(1e6) in range(0, 8)
    assert policy.k_for(-1e6) in range(0, 8)


def test_scalar_policy_returns_actions_for_all_test_batches():
    train = [_homogeneous(i) for i in range(10)]
    test = [_bimodal(i) for i in range(10, 15)]
    total, ks, policy = scalar_oracle(train, test, COST, list(range(0, 8)), n_bins=32)
    assert len(ks) == len(test)
    assert all(isinstance(k, int) for k in ks)
    assert isinstance(policy, ScalarPolicy)
    assert total > 0


# ---------------------------------------------------------------------------
# Train/test split -- regression: fitting and evaluating on the same batches
# ---------------------------------------------------------------------------


def test_train_test_split_sizes():
    batches = [_homogeneous(i) for i in range(10)]
    r = analyse_batches(batches, COST, train_fraction=0.5, seed=0)
    assert r.n_train == 5
    assert r.n_test == 5
    assert len(r.decisions) == 5


def test_analyse_requires_a_valid_train_fraction():
    with pytest.raises(ValueError, match="train_fraction"):
        analyse_batches([_homogeneous(0), _homogeneous(1)], COST, train_fraction=1.0)


def test_split_is_deterministic_for_a_seed():
    batches = [_homogeneous(i) for i in range(20)]
    a = analyse_batches(batches, COST, seed=3).summary()
    b = analyse_batches(batches, COST, seed=3).summary()
    assert a == b


def test_analysis_is_a_noop_for_a_homogeneous_population():
    """Built-in control: with no heterogeneity the curve cannot help.

    The linearity identity makes this exact, so a nonzero gap here would mean the
    oracle itself is broken.
    """
    batches = [_homogeneous(i) for i in range(200)]
    r = analyse_batches(batches, COST, n_bins=8, seed=0)
    assert r.recoverable_rectangular_gap == pytest.approx(0.0, abs=1e-12)
    assert r.fraction_batches_disagreeing == 0.0
    assert r.l2_throughput == pytest.approx(r.l1_throughput)


def _adversarial_population(
    n: int = 400, shift: float = 0.0, noise: float = 0.02, seed: int = 5
) -> list[BatchTrace]:
    rng = random.Random(seed)
    out = []
    for b in range(n):
        base = _homogeneous(b) if b % 2 == 0 else _bimodal(b)
        true_mean = sum(sum(v[:3]) for v in base.survivals.values()) / base.n
        if b % 2:
            true_mean += shift
        out.append(
            BatchTrace(
                batch_id=b,
                survivals=base.survivals,
                active_k=3,
                observed_mean_accepted=true_mean + rng.gauss(0, noise),
            )
        )
    return out


def test_gap_is_positive_when_scalar_cannot_identify_the_optimal_k():
    """Matched means, different optimal K: a scalar must pick one K for both."""
    pop = _adversarial_population()
    r = analyse_batches(pop, COST, n_bins=20, train_fraction=0.5, seed=0)
    assert r.recoverable_rectangular_gap > 0
    # Sanity on what the population actually contains:
    x = [b for b in pop if b.batch_id % 2 == 0]
    y = [b for b in pop if b.batch_id % 2 == 1]
    assert best_k_curve_aware(x[0], COST) == 1
    assert best_k_curve_aware(y[0], COST) == 3
    # ...and their scalars are indistinguishable to within the noise.
    mx = sum(b.scalar() for b in x) / len(x)
    my = sum(b.scalar() for b in y) / len(y)
    assert abs(mx - my) < 0.01


def test_gap_shrinks_when_the_scalar_becomes_informative():
    """If the scalar reveals the type, the curve has less to add."""
    uninformative = analyse_batches(
        _adversarial_population(shift=0.0), COST, n_bins=20, seed=0
    )
    informative = analyse_batches(
        _adversarial_population(shift=0.06), COST, n_bins=20, seed=0
    )
    assert informative.recoverable_rectangular_gap < (
        uninformative.recoverable_rectangular_gap
    )


# ---------------------------------------------------------------------------
# L3 as a genuine upper bound -- regression
# ---------------------------------------------------------------------------


def test_l3_is_a_true_upper_bound():
    """Regression: L3 fell below L2 when ragged cost was priced at the deepest K.

    PROJECT.md asks for L3 only as an upper bound; a "bound" below the achievable
    value misrepresents it as a target.
    """
    pops = [
        _adversarial_population(shift=0.0),
        _adversarial_population(shift=0.06),
        [_homogeneous(i) for i in range(50)],
        [_homogeneous(i) for i in range(25)] + [_bimodal(i) for i in range(25, 50)],
    ]
    for pop in pops:
        r = analyse_batches(pop, COST, n_bins=20, seed=0)
        assert r.l3_throughput >= r.l2_throughput - 1e-9, (
            f"L3 ({r.l3_throughput}) below L2 ({r.l2_throughput}); "
            f"the ragged bound is pessimistic, not a bound"
        )
        assert r.ragged_upper_bound >= -1e-12


def test_l3_uses_harmonic_aggregation_like_the_other_levels():
    """Regression: summing per-batch rates produced ~982 against ~5.

    L3 must be the same order of magnitude as L1/L2, not a sum of rates.
    """
    pop = _adversarial_population(n=200)
    r = analyse_batches(pop, COST, n_bins=20, seed=0)
    assert 0.5 < (r.l3_throughput / r.l2_throughput) < 5.0


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def test_throughput_is_harmonic_not_additive():
    """A slow batch must drag the total down, not be averaged away."""
    fast = _homogeneous(0)
    slow = BatchTrace(
        batch_id=1,
        survivals={
            f"s{i}": surv(0.2, 0.05, 0.01, 0.0, 0.0, 0.0, 0.0) for i in range(4)
        },
    )
    both = [fast, slow]
    t = throughput(both, [1, 1], COST)
    # Strictly less than the mean of the two individual rates.
    m = (batch_goodput(fast, 1, COST) + batch_goodput(slow, 1, COST)) / 2
    assert t < m


def test_aggregate_cost_per_token_is_weighted():
    a = _homogeneous(0)
    b = _homogeneous(1)
    c1 = aggregate_cost_per_token([a, b], [1, 1], COST)
    c2 = aggregate_cost_per_token([a, b], [1, 3], COST)
    assert c2 > c1  # deeper costs more


def test_throughput_normalisation_for_identical_batches():
    batches = [_homogeneous(i) for i in range(4)]
    t = throughput(batches, [2] * 4, COST)
    assert t == pytest.approx(1.0 / batch_cost_per_token(batches[0], 2, COST))


# ---------------------------------------------------------------------------
# Shipped heuristic reference
# ---------------------------------------------------------------------------


def test_shipped_heuristic_is_round_plus_one():
    ks = [0, 1, 3, 5, 7]
    assert shipped_heuristic_k(1.0, ks) == 1
    assert shipped_heuristic_k(0.0, ks) == 1
    assert shipped_heuristic_k(2.4, ks) == 3
    assert shipped_heuristic_k(6.9, ks) == 7


def test_shipped_heuristic_respects_ceiling():
    assert shipped_heuristic_k(6.0, [0, 1, 3, 5, 7], ceiling=3) == 3


def test_shipped_heuristic_never_exceeds_available_candidates():
    assert shipped_heuristic_k(99.0, [0, 1, 2]) == 2


def test_scalar_baseline_is_never_weaker_than_either_scalar_policy():
    """The headline baseline is max(L1, shipped), so it cannot be the weaker one."""
    pop = _adversarial_population(n=200)
    r = analyse_batches(pop, COST, n_bins=20, seed=0)
    assert r.l1_baseline_throughput >= r.l1_throughput - 1e-12
    assert r.l1_baseline_throughput >= r.l1_shipped_throughput - 1e-12
    assert r.recoverable_rectangular_gap <= (
        (r.l2_throughput - r.l1_throughput) / r.l1_throughput + 1e-12
    )


def test_fitted_scalar_map_may_overfit_the_fixed_rule():
    """A fitted map is not guaranteed to beat the fixed rule on held-out data.

    This is why the scalar baseline is the *maximum* of the two rather than the
    fitted map alone: reporting the gap against a policy that happened to
    overfit would overstate the curve's advantage. Here the diagnostic is simply
    allowed to be either sign.
    """
    r = analyse_batches(_adversarial_population(n=200), COST, n_bins=20, seed=0)
    assert isinstance(r.scalar_fit_overfit, float)
    assert r.summary()["scalar_fit_overfit"] == r.scalar_fit_overfit


# ---------------------------------------------------------------------------
# Result shape
# ---------------------------------------------------------------------------


def test_summary_is_json_serialisable():
    import json

    r = analyse_batches(_adversarial_population(n=40), COST, n_bins=8, seed=0)
    s = r.summary()
    assert json.loads(json.dumps(s))["n_batches"] == 40
    assert "candidates" in s and "notes" in s


def test_notes_flag_the_placeholder_cost_model():
    """A synthetic-cost result must never be presentable as a measurement."""
    r = analyse_batches(_adversarial_population(n=20), COST, n_bins=8, seed=0)
    assert any("placeholder" in n for n in r.notes)
    assert any("upper bound" in n for n in r.notes)


def test_analyse_rejects_empty_input():
    with pytest.raises(ValueError, match="no batches"):
        analyse_batches([], COST)


def test_per_request_best_k_returns_one_per_request():
    b = _bimodal(0)
    kpr = per_request_best_k(b, COST)
    assert len(kpr) == b.n


def test_candidates_outside_common_depth_are_dropped():
    b = BatchTrace(batch_id=0, survivals={"a": (0.9, 0.5)})
    r = analyse_batches(
        [b, BatchTrace(batch_id=1, survivals={"a": (0.9, 0.5)})],
        COST,
        candidates=[1, 3, 7],
    )
    assert r.candidates == (1,)
