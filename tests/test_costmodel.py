"""Tests for the measured cost model.

The identity `cost_per_round(K, n) / (n * tokens_per_request_round) == 1/throughput`
is checked explicitly: the whole calibration rests on it, and if it drifts the
oracle silently compares levels under inconsistent units.
"""

import pytest

from heterospec.analysis.oracle import (
    BatchTrace,
    analyse_batches,
    best_k_curve_aware,
)
from heterospec.costmodel import CalibrationPoint, MeasuredCostModel, cost_from_run


def _pt(
    k: int, n: int, *, throughput: float = 1000.0, tpr: float = 2.0
) -> CalibrationPoint:
    """A calibration point with a caller-chosen throughput and tokens/round."""
    tokens = int(throughput * 10)  # wall_time = 10s
    return CalibrationPoint(
        k=k,
        batch_size=n,
        wall_time_s=10.0,
        total_completion_tokens=tokens,
        # total tokens = request_rounds * tpr  ->  rounds = tokens / tpr
        total_request_rounds=int(tokens / tpr),
    )


# ---------------------------------------------------------------------------
# CalibrationPoint
# ---------------------------------------------------------------------------


def test_point_rejects_bad_inputs():
    with pytest.raises(ValueError, match="k must be"):
        _pt(-1, 4)
    with pytest.raises(ValueError, match="batch_size"):
        _pt(1, 0)
    with pytest.raises(ValueError, match="wall_time_s"):
        CalibrationPoint(
            k=1,
            batch_size=1,
            wall_time_s=0.0,
            total_completion_tokens=10,
            total_request_rounds=5,
        )
    with pytest.raises(ValueError, match="total_completion_tokens"):
        CalibrationPoint(
            k=1,
            batch_size=1,
            wall_time_s=1.0,
            total_completion_tokens=0,
            total_request_rounds=0,
        )


def test_cost_per_token_equals_inverse_throughput():
    """The identity that grounds the calibration in measured tokens/second."""
    p = _pt(k=3, n=8, throughput=2500.0, tpr=2.5)
    # _pt uses a 10 s wall time and tokens = throughput * 10.
    assert p.throughput == pytest.approx(2500.0)
    assert p.cost_per_token == pytest.approx(1.0 / 2500.0)
    # cost_per_round / (n * tpr) must equal cost_per_token
    assert p.cost_per_round / (p.batch_size * p.tokens_per_request_round) == (
        pytest.approx(p.cost_per_token)
    )


def test_tokens_per_request_round_defaults_to_one_without_rounds():
    """A no-spec run has no verify rounds; each round emits one token per request."""
    p = CalibrationPoint(
        k=0,
        batch_size=4,
        wall_time_s=5.0,
        total_completion_tokens=100,
        total_request_rounds=0,
    )
    assert p.tokens_per_request_round == 1.0


def test_cost_falls_as_throughput_rises():
    fast = _pt(3, 8, throughput=2000.0)
    slow = _pt(3, 8, throughput=1000.0)
    assert fast.cost_per_round < slow.cost_per_round


def test_cost_rises_with_depth_at_fixed_throughput():
    shallow = _pt(1, 8, throughput=1000.0, tpr=1.5)
    deep = _pt(7, 8, throughput=1000.0, tpr=4.0)
    assert deep.cost_per_round > shallow.cost_per_round


# ---------------------------------------------------------------------------
# cost_from_run
# ---------------------------------------------------------------------------


class _Rec:
    def __init__(self, ok, completion_tokens, spec_verify_ct):
        self.ok = ok
        self.completion_tokens = completion_tokens
        self.spec_verify_ct = spec_verify_ct


def test_cost_from_run_sums_successful_records():
    recs = [_Rec(True, 100, 40), _Rec(True, 200, 80)]
    p = cost_from_run(k=3, batch_size=8, wall_time_s=2.0, records=recs)
    assert p.total_completion_tokens == 300
    assert p.total_request_rounds == 120
    assert p.n_requests == 2
    assert p.throughput == pytest.approx(150.0)


def test_cost_from_run_ignores_failures():
    """A failed request would deflate throughput and make the GPU look slow."""
    recs = [_Rec(True, 100, 40), _Rec(False, 9999, 9999)]
    p = cost_from_run(k=3, batch_size=8, wall_time_s=2.0, records=recs)
    assert p.total_completion_tokens == 100
    assert p.n_requests == 1


def test_cost_from_run_rejects_all_failed():
    with pytest.raises(ValueError, match="no successful records"):
        cost_from_run(k=3, batch_size=8, wall_time_s=1.0, records=[_Rec(False, 1, 1)])


# ---------------------------------------------------------------------------
# Interpolation
# ---------------------------------------------------------------------------


def _grid() -> MeasuredCostModel:
    pts = []
    for k in (1, 3, 7):
        for n in (1, 8, 32):
            # Throughput degrades with both depth and batch size.
            thr = 3000.0 / (1.0 + 0.35 * k) / (1.0 + 0.01 * n)
            pts.append(_pt(k, n, throughput=thr, tpr=1.0 + 0.5 * k))
    return MeasuredCostModel.from_points(pts)


def test_exact_grid_nodes_are_returned_verbatim():
    m = _grid()
    for pt in m.points:
        assert m.cost(pt.k, pt.batch_size) == pytest.approx(pt.cost_per_round)


def test_interpolation_is_between_neighbours():
    m = _grid()
    lo = m.cost(1, 8)
    hi = m.cost(3, 8)
    mid = m.cost(2, 8)
    assert min(lo, hi) <= mid <= max(lo, hi)


def test_interpolation_reproduces_a_linear_surface_exactly():
    """On a linear cost surface, bilinear interpolation must be exact."""
    pts = [
        CalibrationPoint(
            k=k,
            batch_size=n,
            wall_time_s=1.0,
            total_completion_tokens=int(1000 * (2.0 + k + 0.5 * n)),
            total_request_rounds=int(1000 * (2.0 + k + 0.5 * n)),
        )
        for k in (1, 5)
        for n in (1, 9)
    ]
    m = MeasuredCostModel.from_points(pts)
    # cost_per_round = n * 1.0 / throughput; check a midpoint against the
    # surface implied by the four corners via explicit interpolation.
    c_11, c_19 = m.cost(1, 1), m.cost(1, 9)
    c_51, c_59 = m.cost(5, 1), m.cost(5, 9)
    tk, tn = 0.5, 0.5
    expect = (c_11 + tk * (c_51 - c_11)) * (1 - tn) + (c_19 + tk * (c_59 - c_19)) * tn
    assert m.cost(3, 5) == pytest.approx(expect)


def test_clamping_outside_grid_is_counted():
    m = _grid()
    before = m.clamp_count
    m.cost(99, 4)  # k above range
    m.cost(1, 1000)  # n above range
    assert m.clamp_count == before + 2
    assert m.coverage()["clamped_queries"] == m.clamp_count


def test_clamped_query_returns_edge_value():
    m = _grid()
    assert m.cost(99, 8) == pytest.approx(m.cost(7, 8))
    assert m.cost(1, 0 + 1) == pytest.approx(m.cost(1, 1))


def test_incomplete_grid_raises_rather_than_inventing_a_cell():
    """A missing grid cell must fail loudly, not be silently fabricated."""
    pts = [_pt(1, 1), _pt(1, 8), _pt(3, 8)]  # (3, 1) missing
    m = MeasuredCostModel.from_points(pts)
    assert not m.coverage()["complete"]
    with pytest.raises(ValueError, match="no measurement for"):
        m.cost(3, 1)


def test_cost_rejects_bad_batch_size():
    with pytest.raises(ValueError, match="batch size"):
        _grid().cost(1, 0)


# ---------------------------------------------------------------------------
# Serialisation and coverage
# ---------------------------------------------------------------------------


def test_jsonl_round_trip(tmp_path):
    m = _grid()
    path = tmp_path / "cost_model.jsonl"
    assert m.to_jsonl(path) == len(m.points)
    m2 = MeasuredCostModel.from_jsonl(path)
    assert m2.k_values == m.k_values
    assert m2.batch_sizes == m.batch_sizes
    for k in m.k_values:
        for n in m.batch_sizes:
            assert m2.cost(k, n) == pytest.approx(m.cost(k, n))


def test_coverage_reports_shape_and_completeness():
    cov = _grid().coverage()
    assert cov["k_values"] == [1, 3, 7]
    assert cov["batch_sizes"] == [1, 8, 32]
    assert cov["n_cells_measured"] == 9
    assert cov["complete"] is True


def test_empty_model_rejected():
    with pytest.raises(ValueError, match="no calibration points"):
        MeasuredCostModel(points=[])


# ---------------------------------------------------------------------------
# Integration with the oracle
# ---------------------------------------------------------------------------


def _batch(bid: int, p, n: int = 4) -> BatchTrace:
    def surv(*q):
        o, pr = [], 1.0
        for x in q:
            pr *= x
            o.append(pr)
        return tuple(o)

    return BatchTrace(
        batch_id=bid, survivals={f"r{i}": surv(*p) for i in range(n)}, active_k=3
    )


def test_measured_model_drops_into_the_oracle():
    """A measured cost surface must replace the placeholder cleanly."""
    m = _grid()
    pop = [_batch(i, (0.8, 0.6, 0.4, 0.25, 0.15, 0.1, 0.05)) for i in range(10)]
    pop += [_batch(100 + i, (0.3, 0.1, 0.02, 0.0, 0.0, 0.0, 0.0)) for i in range(10)]
    r = analyse_batches(pop, m, n_bins=8, seed=0)
    assert r.l0_throughput > 0
    assert r.l2_throughput > 0
    # The placeholder-cost warning must NOT appear for a measured model.
    assert not any("placeholder" in n for n in r.notes)


def test_oracle_notes_absent_for_measured_cost():
    m = _grid()
    r = analyse_batches(
        [_batch(i, (0.9, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05)) for i in range(12)],
        m,
        n_bins=4,
        seed=0,
    )
    assert "LinearCostModel" not in r.cost_model_repr
    assert any("train split" in n for n in r.notes)


def test_best_k_changes_with_the_cost_model():
    """A depth that is optimal under one cost surface need not be under another."""
    shallow_batch = _batch(0, (0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65))
    cheap_deep = MeasuredCostModel.from_points(
        [
            CalibrationPoint(
                k=k,
                batch_size=4,
                wall_time_s=1.0,
                total_completion_tokens=1000,
                total_request_rounds=500,
            )
            for k in (1, 7)
        ]
    )
    expensive_deep = MeasuredCostModel.from_points(
        [
            CalibrationPoint(
                k=1,
                batch_size=4,
                wall_time_s=1.0,
                total_completion_tokens=1000,
                total_request_rounds=500,
            ),
            CalibrationPoint(
                k=7,
                batch_size=4,
                wall_time_s=1.0,
                total_completion_tokens=100,
                total_request_rounds=50,
            ),
        ]
    )
    k_cheap = best_k_curve_aware(shallow_batch, cheap_deep)
    k_expensive = best_k_curve_aware(shallow_batch, expensive_deep)
    assert k_cheap > k_expensive, (
        f"with equal cost per round, deeper should win; got {k_cheap} vs {k_expensive}"
    )
