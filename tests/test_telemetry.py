"""Tests for the telemetry derivations.

Values are hand-computed so a regression in the maths fails loudly rather than
producing slightly-wrong plots.
"""

import math

import pytest

from heterospec.telemetry import (
    AcceptStats,
    expected_accepted_drafts_at_depth,
    histogram_mean,
    histogram_std,
    histogram_variance,
    position_acceptance,
    preferred_depth,
    rounds_consistency,
    survival_curve,
)

# histogram[j] = rounds in which exactly j drafts were accepted.
# n = 20; mean = (0*2 + 1*4 + 2*8 + 3*5 + 4*1)/20 = 39/20 = 1.95
H = [2, 4, 8, 5, 1]


# ---------------------------------------------------------------------------
# Raw statistics
# ---------------------------------------------------------------------------


def test_histogram_mean():
    assert histogram_mean(H) == pytest.approx(1.95)


def test_histogram_variance():
    # E[X^2] = (0*2 + 1*4 + 4*8 + 9*5 + 16*1)/20 = 97/20 = 4.85
    assert histogram_variance(H) == pytest.approx(4.85 - 1.95**2)


def test_histogram_std():
    assert histogram_std(H) == pytest.approx(math.sqrt(4.85 - 1.95**2))


def test_empty_histogram_is_nan_not_crash():
    assert math.isnan(histogram_mean([]))
    assert math.isnan(histogram_variance([]))
    assert math.isnan(histogram_std([]))
    assert survival_curve([]) == ()


def test_single_round_histogram():
    assert histogram_mean([0, 1]) == 1.0
    assert histogram_variance([0, 1]) == 0.0


# ---------------------------------------------------------------------------
# Survival curve
# ---------------------------------------------------------------------------


def test_survival_curve_hand_computed():
    # S_k = sum(histogram[k:]) / 20
    assert survival_curve(H) == pytest.approx((0.9, 0.7, 0.3, 0.05))


def test_survival_is_monotone_nonincreasing():
    s = survival_curve(H)
    assert all(a >= b for a, b in zip(s, s[1:], strict=False))


def test_survival_k_max_extends_with_zeros():
    assert survival_curve(H, k_max=6) == pytest.approx((0.9, 0.7, 0.3, 0.05, 0.0, 0.0))


def test_survival_k_max_zero_returns_empty():
    assert survival_curve(H, k_max=0) == ()


# ---------------------------------------------------------------------------
# The identity that makes this approach work
# ---------------------------------------------------------------------------


def test_expected_accepted_at_observed_depth_equals_histogram_mean():
    """E[accepted] = sum_k S_k must reproduce the histogram mean exactly.

    This is the core consistency identity. If it ever fails, either the survival
    curve or the expectation is wrong -- and the whole oracle rests on it.
    """
    for hist in ([0, 1], [1, 0], [2, 4, 8, 5, 1], [5, 0, 0, 5], [0, 0, 0, 0, 1]):
        s = survival_curve(hist)
        depth = len(hist) - 1
        assert expected_accepted_drafts_at_depth(s, depth) == pytest.approx(
            histogram_mean(hist)
        ), f"identity failed for {hist}"


def test_expected_accepted_is_monotone_in_depth():
    s = survival_curve(H, k_max=7)
    vals = [expected_accepted_drafts_at_depth(s, k) for k in range(0, 8)]
    assert vals[0] == 0.0
    assert all(a <= b for a, b in zip(vals, vals[1:], strict=False))


def test_expected_accepted_negative_depth_is_zero():
    assert expected_accepted_drafts_at_depth(survival_curve(H), -3) == 0.0


# ---------------------------------------------------------------------------
# PositionAcceptance: validity bounds are the point
# ---------------------------------------------------------------------------


def test_static_k_marks_capture_clean():
    pa = position_acceptance(H, k_static=4)
    assert not pa.k_confounded
    assert pa.safe_up_to == 4
    assert pa.achievable_depth == 4
    assert pa.require_clean() is pa
    assert pa.position_acceptance_probs() == pytest.approx((0.9, 0.7, 0.3, 0.05))


def test_static_k_zero_carries_no_information_and_is_marked_confounded():
    """K=0 proposes no drafts at all, so it cannot inform position acceptance."""
    pa = position_acceptance([20], k_static=0)
    assert pa.safe_up_to == 0
    assert pa.k_confounded


def test_unknown_k_is_confounded_not_silently_biased():
    """The default must refuse to hand back plausible-looking biased numbers."""
    pa = position_acceptance(H)
    assert pa.k_confounded
    assert pa.safe_up_to == 0
    assert pa.valid_survival() == ()
    with pytest.raises(ValueError, match="K-confounded"):
        pa.require_clean()


def test_min_k_proposed_from_trace_unlocks_up_to_that_depth():
    pa = position_acceptance(H, min_k_proposed=3)
    assert not pa.k_confounded
    assert pa.safe_up_to == 3
    assert pa.valid_survival() == pytest.approx((0.9, 0.7, 0.3))
    # depth 4 is observed but not trustworthy, because some round proposed < 4
    with pytest.raises(ValueError, match="exceeds trustworthy range"):
        pa.expected_accepted(4)
    assert pa.expected_accepted(3) == pytest.approx(0.9 + 0.7 + 0.3)


def test_static_k_larger_than_observed_gives_zero_survival_beyond_data():
    pa = position_acceptance([2, 4, 8, 5, 1], k_static=7)
    assert pa.safe_up_to == 7
    assert pa.valid_survival() == pytest.approx((0.9, 0.7, 0.3, 0.05, 0.0, 0.0, 0.0))


def test_expected_accepted_refuses_extrapolation_beyond_observed():
    """Requesting a deeper tier than was ever run must raise, not invent data."""
    pa = position_acceptance(H, k_static=4)
    assert pa.achievable_depth == 4
    with pytest.raises(ValueError, match="exceeds trustworthy range"):
        pa.expected_accepted(5)


def test_describe_reports_provenance():
    assert "static K=4" in position_acceptance(H, k_static=4).describe()
    assert "min K proposed=2" in position_acceptance(H, min_k_proposed=2).describe()
    assert "K unknown" in position_acceptance(H).describe()


def test_negative_k_rejected():
    with pytest.raises(ValueError, match="k_static must be >= 0"):
        position_acceptance(H, k_static=-1)
    with pytest.raises(ValueError, match="min_k_proposed must be >= 0"):
        position_acceptance(H, min_k_proposed=-1)


# ---------------------------------------------------------------------------
# Demonstrating the confounding hazard the design guards against
# ---------------------------------------------------------------------------


def test_step_zero_rounds_halve_the_naive_position_estimate():
    """A concrete demonstration of why adaptive captures cannot be pooled naively.

    A request experiences 10 rounds at K=1 and 10 rounds at the step-0 tier.
    Among the K=1 rounds, position 1 is genuinely accepted 6/10 = 0.60 of the
    time. The step-0 rounds propose nothing but still record 0 accepted, so the
    pooled histogram yields 0.30 -- exactly half the truth, and nothing in the
    histogram reveals it.
    """
    k1_rounds = [4, 6]  # 4 rounds accepted 0, 6 accepted 1  -> true P(A1) = 0.60
    step0_rounds = [10]  # 10 rounds proposed nothing, recorded 0
    pooled = [k1_rounds[0] + step0_rounds[0], k1_rounds[1]]  # [14, 6]

    assert survival_curve(pooled)[0] == pytest.approx(0.30)
    assert survival_curve(k1_rounds)[0] == pytest.approx(0.60)

    # Same histogram, two very different provenance stories:
    assert position_acceptance(pooled, k_static=1).valid_survival() == pytest.approx(
        (0.30,)
    )
    assert position_acceptance(pooled).k_confounded
    assert position_acceptance(k1_rounds, k_static=1).valid_survival() == (
        pytest.approx((0.60,))
    )


# ---------------------------------------------------------------------------
# Cost-aware preferred depth
# ---------------------------------------------------------------------------


def test_preferred_depth_free_lunch_prefers_deepest():
    """Perfect acceptance S_k=1: E[accepted]=k, so with flat cost the deepest wins."""
    s = (1.0,) * 7
    assert preferred_depth(s, candidates=[1, 3, 5, 7], cost_fn=lambda k: 1.0) == 7


def test_preferred_depth_breaks_ties_toward_shallower_tier():
    """Once the survival curve flattens to zero, deeper tiers add nothing.

    H has S_5 = S_6 = S_7 = 0, so depths 5 and 7 yield identical expected
    accepted drafts. Ties must go to the shallower tier: same tokens, fewer
    draft steps. This also protects against needlessly deep speculation when the
    cost model is flat.
    """
    s = survival_curve(H, k_max=7)
    assert (s[4], s[5], s[6]) == (0.0, 0.0, 0.0)
    assert preferred_depth(s, candidates=[1, 3, 5, 7], cost_fn=lambda k: 1.0) == 5


def test_preferred_depth_bounded_by_cost_growth():
    """If cost grows quadratically, an early tier wins."""
    s = survival_curve(H, k_max=7)
    chosen = preferred_depth(s, candidates=[1, 3, 5, 7], cost_fn=lambda k: float(k**2))
    assert chosen == 1


def test_preferred_depth_respects_flat_survival_and_linear_cost():
    # S = (1,1,1,1): E[accepted]=k; score = k/k = 1 for all -> first wins ties
    s = (1.0, 1.0, 1.0, 1.0)
    assert preferred_depth(s, candidates=[1, 2, 3, 4], cost_fn=lambda k: float(k)) == 1


def test_preferred_depth_rejects_nonpositive_cost():
    with pytest.raises(ValueError, match="must be positive"):
        preferred_depth([1.0], candidates=[1], cost_fn=lambda k: 0.0)


# ---------------------------------------------------------------------------
# Integrity check
# ---------------------------------------------------------------------------


def test_rounds_consistency_ok():
    ok, msg = rounds_consistency(H, 20)
    assert ok and "consistent" in msg


def test_rounds_consistency_flags_mismatch():
    ok, msg = rounds_consistency(H, 17)
    assert not ok and "INCONSISTENT" in msg and "delta 3" in msg


def test_rounds_consistency_unchecked_when_absent():
    ok, msg = rounds_consistency(H, None)
    assert ok and "unchecked" in msg


# ---------------------------------------------------------------------------
# AcceptStats
# ---------------------------------------------------------------------------


def test_accept_stats_from_histogram():
    st = AcceptStats.from_histogram(H)
    assert st.n_rounds == 20
    assert st.mean == pytest.approx(1.95)
    assert st.std == pytest.approx(math.sqrt(4.85 - 1.95**2))
    assert st.max_observed == 4
    assert st.to_dict()["n_rounds"] == 20
