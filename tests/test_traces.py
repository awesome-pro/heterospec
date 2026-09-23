"""Tests for turning a real capture into oracle input.

These use the mock server so the whole path runs on the Mac: workloads ->
requests -> wave reconstruction -> survival curves -> oracle. That is the path
the GPU session will exercise, so any bug here would otherwise be found at $1+ an
hour.
"""

import pytest

from heterospec.analysis.oracle import LinearCostModel, analyse_batches
from heterospec.analysis.traces import (
    batches_from_iterations,
    batches_from_waves,
    describe_skips,
)
from heterospec.mockserver import MockSGLangServer
from heterospec.records import IterationRecord, RequestRecord
from heterospec.runner import RunConfig, run_benchmark


@pytest.fixture
def captured(tmp_path):
    """A real wave-dispatch static-K capture against the mock."""
    with MockSGLangServer(k=3, seed=0) as srv:
        cfg = RunConfig(
            workload="mixed_50_50",
            policy_id="static_k3",
            base_url=srv.base_url,
            num_requests=64,
            concurrency=8,
            dispatch="waves",
            seed=0,
            results_root=tmp_path,
            sglang_path=None,
            static_k=3,
            progress=None,
        )
        yield run_benchmark(cfg)


# ---------------------------------------------------------------------------
# Waves
# ---------------------------------------------------------------------------


def test_waves_reconstruct_one_batch_per_wave(captured):
    batches, skips = batches_from_waves(captured.records, concurrency=8, k_static=3)
    assert len(batches) == 8  # 64 requests / 8 per wave
    assert all(b.n == 8 for b in batches)
    assert not skips


def test_wave_batches_carry_composition_and_scalar(captured):
    batches, _ = batches_from_waves(captured.records, concurrency=8, k_static=3)
    b = batches[0]
    assert b.active_k == 3
    assert b.scalar() is not None
    assert 0.0 <= b.scalar() <= 3.0  # accepted drafts cannot exceed K


def test_wave_batches_preserve_mixed_composition(captured):
    """Each controlled wave must still contain both prompt classes."""
    by_rid = {r.rid: r for r in captured.records}
    batches, _ = batches_from_waves(captured.records, concurrency=8, k_static=3)
    for b in batches:
        classes = {by_rid[rid].prompt_class for rid in b.survivals}
        assert classes == {"repetitive", "open_ended"}, classes


def test_survival_curves_are_monotone_and_clean(captured):
    """BatchTrace validates monotonicity; confounded curves must be dropped."""
    batches, _ = batches_from_waves(captured.records, concurrency=8, k_static=3)
    for b in batches:
        for rid, curve in b.survivals.items():
            assert len(curve) == 3, f"{rid}: expected depth 3, got {len(curve)}"
            assert all(a >= c for a, c in zip(curve, curve[1:], strict=False))


def test_waves_refuses_adaptive_capture_without_a_trace(captured):
    """Without the trace, an adaptive capture would yield K-confounded curves."""
    with pytest.raises(ValueError, match="K-confounded"):
        batches_from_waves(captured.records, concurrency=8, k_static=0)
    with pytest.raises(ValueError, match="static K"):
        batches_from_waves(captured.records, concurrency=8, k_static=None)


def test_waves_rejects_bad_concurrency(captured):
    with pytest.raises(ValueError, match="concurrency"):
        batches_from_waves(captured.records, concurrency=0, k_static=3)


def test_class_separation_survives_into_the_batches(captured):
    """The mock's signal must still be visible after curve reconstruction.

    Compare expected accepted drafts (the sum of the survival curve), not
    `S_1`: position-1 acceptance is high for both classes; it is the tail that
    separates them.
    """
    by_rid = {r.rid: r for r in captured.records}
    batches, _ = batches_from_waves(captured.records, concurrency=8, k_static=3)
    rep, low = [], []
    for b in batches:
        for rid, curve in b.survivals.items():
            e_acc = sum(curve)
            (rep if by_rid[rid].prompt_class == "repetitive" else low).append(e_acc)
    assert sum(rep) / len(rep) > 2 * (sum(low) / len(low))


def test_end_to_end_mock_capture_feeds_the_oracle(captured):
    """The full Mac-only path: real requests -> batches -> oracle."""
    batches, skips = batches_from_waves(captured.records, concurrency=8, k_static=3)
    r = analyse_batches(batches, LinearCostModel(), n_bins=4, seed=0)
    assert r.n_batches == len(batches)
    assert r.l2_throughput > 0
    assert r.l0_throughput > 0
    assert describe_skips(skips)["n_skipped"] == 0


# ---------------------------------------------------------------------------
# Iteration trace
# ---------------------------------------------------------------------------


def _iter(rids, active_k, iteration):
    return IterationRecord(
        iteration=iteration,
        batch_size=len(rids),
        active_k=active_k,
        requests=[{"rid": r, "accepted": 0} for r in rids],
    )


def test_iteration_batches_bound_k_by_the_minimum_proposed(captured):
    """A request present in a K=1 iteration is only trustworthy to depth 1."""
    rids = [r.rid for r in captured.records[:8]]
    iterations = [
        _iter(rids, 3, 0),
        _iter(rids, 1, 1),
        _iter(rids, 3, 2),
    ]
    batches, skips = batches_from_iterations(captured.records, iterations)
    # Only the 8 traced requests can be used; the other 56 are correctly skipped.
    assert batches, "traced requests must produce batches"
    assert all(set(b.survivals) <= set(rids) for b in batches)
    for b in batches:
        for curve in b.survivals.values():
            assert len(curve) == 1, "min proposed K=1 must cap every curve at depth 1"


def test_iteration_batches_use_true_composition(captured):
    rids = [r.rid for r in captured.records[:4]]
    iterations = [_iter(rids, 3, 0)]
    batches, _ = batches_from_iterations(captured.records, iterations)
    assert len(batches) == 1
    assert batches[0].n == 4
    assert set(batches[0].survivals) == set(rids)


def test_iteration_batches_skip_requests_absent_from_the_trace(captured):
    """A request with no trace entry has no K bound, so it cannot be used."""
    traced = [r.rid for r in captured.records[:8]]
    iterations = [_iter(traced, 3, 0)]
    batches, skips = batches_from_iterations(captured.records, iterations)
    reasons = describe_skips(skips)["by_reason"]
    assert any("absent from iteration trace" in k for k in reasons)
    assert all(set(b.survivals) <= set(traced) for b in batches)


def test_iteration_batches_drop_singleton_batches(captured):
    """A batch of one has nothing to be heterogeneous about."""
    iterations = [
        _iter([captured.records[0].rid], 3, 0),
        _iter([r.rid for r in captured.records[:4]], 3, 1),
    ]
    batches, _ = batches_from_iterations(captured.records, iterations)
    assert len(batches) == 1
    assert batches[0].n == 4


def test_iteration_max_iterations_is_respected(captured):
    rids = [r.rid for r in captured.records[:8]]
    iterations = [_iter(rids, 3, i) for i in range(5)]
    batches, _ = batches_from_iterations(captured.records, iterations, max_iterations=2)
    assert len(batches) == 2


def test_iteration_active_k_is_recorded(captured):
    rids = [r.rid for r in captured.records[:8]]
    iterations = [_iter(rids, 5, 0)]
    batches, _ = batches_from_iterations(captured.records, iterations)
    assert batches[0].active_k == 5


# ---------------------------------------------------------------------------
# Skips
# ---------------------------------------------------------------------------


def test_non_speculative_records_are_skipped_with_a_reason():
    recs = [
        RequestRecord(rid="a", ok=True, completion_tokens=10, prompt_class="x"),
        RequestRecord(rid="b", ok=True, completion_tokens=10, prompt_class="x"),
    ]
    batches, skips = batches_from_waves(recs, concurrency=2, k_static=3)
    assert batches == []
    assert all("no spec histogram" in s.reason for s in skips)


def test_failed_requests_are_skipped(captured):
    recs = list(captured.records)
    recs[0] = RequestRecord(
        rid=recs[0].rid,
        index=recs[0].index,
        ok=False,
        error="boom",
        prompt_class="open_ended",
    )
    _, skips = batches_from_waves(recs, concurrency=8, k_static=3)
    assert any("failed" in s.reason for s in skips)


def test_describe_skips_groups_by_reason():
    recs = [
        RequestRecord(rid="a", ok=False, error="x", index=0),
        RequestRecord(rid="b", ok=False, error="x", index=1),
        RequestRecord(rid="c", ok=True, completion_tokens=5, index=2),
    ]
    _, skips = batches_from_waves(recs, concurrency=2, k_static=3)
    d = describe_skips(skips)
    assert d["n_skipped"] == 3
    assert sum(d["by_reason"].values()) == 3
