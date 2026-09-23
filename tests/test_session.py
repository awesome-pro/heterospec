"""Tests for GPU-session orchestration.

The integration test runs the *entire* Session-1 pipeline against the mock —
plan -> per-policy runs -> cost grid -> oracle gap report -> session report — so
the only untested surface left on the GPU host is the hardware itself. A failure
in the analysis path discovered at $1+/hour is the expensive mistake this
prevents.
"""

import json
from pathlib import Path

import pytest

from heterospec.config import load_launch_configs
from heterospec.mockserver import MockSGLangServer
from heterospec.runner import RunConfig, run_benchmark
from heterospec.session import (
    SessionStep,
    StepResult,
    build_session_plan,
    cost_model_from_results,
    oracle_gap_report,
    write_session_report,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCH_CONFIG = REPO_ROOT / "configs" / "models" / "llama31_8b_eagle3.json"
LAUNCHES = {c.id: c for c in load_launch_configs(LAUNCH_CONFIG)}


# ---------------------------------------------------------------------------
# Planning -- inspectable for free, before any GPU is rented
# ---------------------------------------------------------------------------


def test_plan_covers_the_calibration_grid():
    steps = build_session_plan(
        LAUNCHES, ks=[1, 3], concurrencies=[4, 8], waves_per_step=8
    )
    cal = [s for s in steps if s.purpose == "calibration"]
    assert len(cal) == 4
    assert {(s.static_k, s.concurrency) for s in cal} == {
        (1, 4),
        (1, 8),
        (3, 4),
        (3, 8),
    }


def test_plan_uses_one_server_per_policy():
    steps = build_session_plan(LAUNCHES, ks=[1, 3], concurrencies=[4, 8])
    assert {s.policy_id for s in steps} == {
        "static_k1",
        "static_k3",
        "sglang_adaptive",
        "no_spec",
    }


def test_plan_scales_requests_with_concurrency():
    """A flat request count would make the concurrency-1 step dominate."""
    steps = build_session_plan(
        LAUNCHES, ks=[1], concurrencies=[1, 8, 32], waves_per_step=10
    )
    by_conc = {s.concurrency: s.num_requests for s in steps}
    assert by_conc[1] == 10
    assert by_conc[8] == 80
    assert by_conc[32] == 320


def test_flat_num_requests_override():
    steps = build_session_plan(LAUNCHES, ks=[1], concurrencies=[1, 8], num_requests=50)
    assert {s.num_requests for s in steps} == {50}


def test_adaptive_steps_have_no_static_k():
    steps = build_session_plan(LAUNCHES, ks=[1], concurrencies=[4])
    for s in steps:
        if s.purpose == "adaptive":
            assert s.static_k is None
        if s.purpose == "calibration":
            assert s.static_k == 1


def test_plan_can_exclude_baselines():
    steps = build_session_plan(
        LAUNCHES,
        ks=[1],
        concurrencies=[4],
        include_adaptive=False,
        include_nospec=False,
    )
    assert {s.purpose for s in steps} == {"calibration"}


def test_plan_rejects_empty_axes():
    with pytest.raises(ValueError, match="ks must be non-empty"):
        build_session_plan(LAUNCHES, ks=[], concurrencies=[4])
    with pytest.raises(ValueError, match="concurrencies must be non-empty"):
        build_session_plan(LAUNCHES, ks=[1], concurrencies=[])
    with pytest.raises(ValueError, match="waves_per_step"):
        build_session_plan(LAUNCHES, ks=[1], concurrencies=[4], waves_per_step=0)


def test_plan_raises_for_a_missing_static_k():
    with pytest.raises(KeyError, match="no static launch config with num_steps=2"):
        build_session_plan(LAUNCHES, ks=[2], concurrencies=[4])


def test_step_describe_is_readable():
    s = SessionStep(
        policy_id="static_k3",
        concurrency=8,
        workload="mixed_50_50",
        num_requests=192,
        seed=0,
        purpose="calibration",
        static_k=3,
    )
    assert "static_k3" in s.describe() and "K=3" in s.describe()
    assert "concurrency=8" in s.describe()


# ---------------------------------------------------------------------------
# Cost-grid assembly and reporting
# ---------------------------------------------------------------------------


def _fake_ok_result(
    run_dir: Path, k: int, conc: int, purpose="calibration"
) -> StepResult:
    return StepResult(
        step=SessionStep(
            policy_id=f"static_k{k}" if purpose == "calibration" else "sglang_adaptive",
            concurrency=conc,
            workload="mixed_50_50",
            num_requests=0,
            seed=0,
            purpose=purpose,
            static_k=k if purpose == "calibration" else None,
        ),
        ok=True,
        run_dir=str(run_dir),
    )


def test_cost_model_reports_missing_cells_rather_than_interpolating(tmp_path):
    """An incomplete grid means untested regions of the cost surface."""
    good = _capture(tmp_path, k=1, conc=4, waves=4, purpose="calibration")
    # A failed sibling must be reported, not silently dropped.
    failed = StepResult(
        step=SessionStep("static_k3", 4, "mixed_50_50", 0, 0, "calibration", 3),
        ok=False,
        error="boom",
    )
    model, missing = cost_model_from_results([good, failed], tmp_path)
    assert any("K=3,n=4" in m for m in missing), missing
    # The usable cell is still present.
    assert model.coverage()["n_cells_measured"] == 1


def test_cost_model_raises_with_reasons_when_nothing_is_usable(tmp_path):
    """Zero usable cells must fail with something an operator can act on."""
    d = tmp_path / "empty"
    d.mkdir()
    (d / "aggregate.json").write_text(json.dumps({"dispatch_wall_time_s": 1.0}))
    (d / "requests.jsonl").write_text("")
    with pytest.raises(ValueError, match="no usable calibration cells"):
        cost_model_from_results([_fake_ok_result(d, 1, 4)], tmp_path)


# ---------------------------------------------------------------------------
# Full pipeline against the mock
# ---------------------------------------------------------------------------


def _capture(
    tmp_path, *, k: int, conc: int, waves: int, purpose: str, trace: bool = False
) -> StepResult:
    """Create a real run directory using the mock server, and a StepResult."""
    with MockSGLangServer(k=k, seed=0) as srv:
        cfg = RunConfig(
            workload="mixed_50_50",
            policy_id=f"k{k}_c{conc}",
            base_url=srv.base_url,
            num_requests=waves * conc,
            concurrency=conc,
            dispatch="waves",
            seed=0,
            results_root=tmp_path,
            sglang_path=None,
            static_k=k if purpose == "calibration" else None,
            progress=None,
        )
        r = run_benchmark(cfg)
    step = SessionStep(
        policy_id=f"static_k{k}" if purpose == "calibration" else "sglang_adaptive",
        concurrency=conc,
        workload="mixed_50_50",
        num_requests=waves * conc,
        seed=0,
        purpose=purpose,
        static_k=k if purpose == "calibration" else None,
    )
    return StepResult(
        step=step,
        ok=True,
        run_dir=str(r.run_dir),
        dispatch_wall_time_s=r.dispatch_wall_time_s,
        n_ok=r.aggregate.get("n_ok", 0),
    )


def test_end_to_end_session_pipeline_on_the_mock(tmp_path):
    """plan -> captures -> measured cost grid -> oracle gap -> session report."""
    steps = build_session_plan(
        LAUNCHES,
        ks=[1, 3],
        concurrencies=[4, 8],
        waves_per_step=6,
        include_adaptive=False,
        include_nospec=False,
    )
    results = [
        _capture(
            tmp_path, k=s.static_k, conc=s.concurrency, waves=6, purpose="calibration"
        )
        for s in steps
    ]

    model, missing = cost_model_from_results(results, tmp_path)
    assert not missing, f"unexpected missing cells: {missing}"
    assert model.coverage()["complete"] is True
    assert model.k_values == [1, 3]
    assert model.batch_sizes == [4, 8]

    gap = oracle_gap_report(results, model, n_bins=4, seed=0)
    assert gap["captures"], "no capture produced a result"
    # Every capture should have produced a gap or an explicit error.
    for c in gap["captures"]:
        assert "recoverable_rectangular_gap" in c or "error" in c, c
    assert "gap_summary" in gap

    report_path = write_session_report(
        results,
        results_root=tmp_path,
        cost=model,
        missing_cells=missing,
        gap_report=gap,
    )
    payload = json.loads(report_path.read_text())
    assert payload["n_ok"] == 4
    assert payload["n_failed"] == 0
    assert payload["cost_model"]["complete"] is True
    assert "gap" in payload


def test_adaptive_capture_without_a_trace_is_reported_not_crashed(tmp_path):
    """The adaptive path needs the trace; its absence must be diagnosed."""
    results = [_capture(tmp_path, k=3, conc=8, waves=6, purpose="adaptive")]
    model, _ = cost_model_from_results(
        [_capture(tmp_path, k=3, conc=8, waves=6, purpose="calibration")], tmp_path
    )
    gap = oracle_gap_report(results, model, n_bins=4, seed=0)
    entry = gap["captures"][0]
    assert "error" in entry
    assert "trace" in entry["error"]


def test_gap_report_survives_a_failed_capture(tmp_path):
    model, _ = cost_model_from_results(
        [_capture(tmp_path, k=3, conc=8, waves=6, purpose="calibration")], tmp_path
    )
    bad = StepResult(
        step=SessionStep("static_k3", 8, "mixed_50_50", 0, 0, "calibration", 3),
        ok=False,
        error="server died",
    )
    gap = oracle_gap_report([bad], model, n_bins=4, seed=0)
    assert gap["captures"] == []
    assert gap["warnings"]


def test_too_few_batches_is_reported_not_analysed(tmp_path):
    results = [_capture(tmp_path, k=3, conc=8, waves=2, purpose="calibration")]
    model, _ = cost_model_from_results(results, tmp_path)
    gap = oracle_gap_report(results, model, n_bins=4, seed=0)
    entry = gap["captures"][0]
    assert "error" in entry and "train/test split" in entry["error"]


def test_step_result_serialises(tmp_path):
    r = StepResult(
        step=SessionStep("static_k1", 4, "high", 40, 0, "calibration", 1),
        ok=True,
        run_dir=str(tmp_path),
        n_ok=40,
    )
    d = r.to_dict()
    assert json.loads(json.dumps(d))["step"]["static_k"] == 1
    assert d["step"]["purpose"] == "calibration"


# ---------------------------------------------------------------------------
# K-invariance wiring
# ---------------------------------------------------------------------------


def test_k_invariance_from_results_uses_one_concurrency(tmp_path):
    """Batch size itself affects acceptance, so depths must be compared at one."""
    from heterospec.session import k_invariance_from_results

    steps = build_session_plan(
        LAUNCHES,
        ks=[1, 3],
        concurrencies=[4, 8],
        waves_per_step=6,
        include_adaptive=False,
        include_nospec=False,
    )
    results = [
        _capture(
            tmp_path, k=s.static_k, conc=s.concurrency, waves=6, purpose="calibration"
        )
        for s in steps
    ]
    rep = k_invariance_from_results(results, concurrency=8)
    assert rep is not None
    assert rep.ks == (1, 3)
    assert rep.n_requests_by_k[1] > 0 and rep.n_requests_by_k[3] > 0
    assert isinstance(rep.verdict(), str)


def test_k_invariance_needs_two_depths(tmp_path):
    from heterospec.session import k_invariance_from_results

    steps = build_session_plan(
        LAUNCHES,
        ks=[3],
        concurrencies=[8],
        waves_per_step=6,
        include_adaptive=False,
        include_nospec=False,
    )
    results = [
        _capture(
            tmp_path, k=s.static_k, conc=s.concurrency, waves=6, purpose="calibration"
        )
        for s in steps
    ]
    assert k_invariance_from_results(results) is None


def test_session_report_includes_k_invariance(tmp_path):
    from heterospec.session import k_invariance_from_results

    steps = build_session_plan(
        LAUNCHES,
        ks=[1, 3],
        concurrencies=[8],
        waves_per_step=6,
        include_adaptive=False,
        include_nospec=False,
    )
    results = [
        _capture(
            tmp_path, k=s.static_k, conc=s.concurrency, waves=6, purpose="calibration"
        )
        for s in steps
    ]
    model, missing = cost_model_from_results(results, tmp_path)
    gap = oracle_gap_report(results, model, n_bins=4, seed=0)
    inv = k_invariance_from_results(results)
    path = write_session_report(
        results,
        results_root=tmp_path,
        cost=model,
        missing_cells=missing,
        gap_report=gap,
        invariance=inv.summary(),
    )
    payload = json.loads(path.read_text())
    assert "k_invariance" in payload
    assert "verdict" in payload["k_invariance"]
    assert "passed" in payload["k_invariance"]


# # ---------------------------------------------------------------------------
# Tier-aware time estimate
# ---------------------------------------------------------------------------


def test_estimate_reports_a_range_not_a_point():
    from heterospec.session import estimate_session_minutes

    steps = build_session_plan(LAUNCHES, ks=[1, 3], concurrencies=[8], waves_per_step=4)
    est = estimate_session_minutes(steps, LAUNCHES)
    assert est["nominal_minutes"] > 0
    assert est["pessimistic_minutes"] > est["nominal_minutes"]


def test_adaptive_policy_needs_more_startup_than_static():
    """Each adaptive tier owns its own CUDA graphs, so startup scales with tiers."""
    from heterospec.session import estimate_session_minutes

    steps = build_session_plan(
        LAUNCHES, ks=[1], concurrencies=[8], waves_per_step=4, include_nospec=False
    )
    est = estimate_session_minutes(steps, LAUNCHES)
    by_policy = {row["policy"]: row for row in est["per_policy"]}
    assert by_policy["sglang_adaptive"]["tiers"] > by_policy["static_k1"]["tiers"]
    assert (
        by_policy["sglang_adaptive"]["startup_s"] > by_policy["static_k1"]["startup_s"]
    )


def test_estimate_breakdown_covers_every_policy():
    from heterospec.session import estimate_session_minutes

    steps = build_session_plan(LAUNCHES, ks=[1], concurrencies=[8], waves_per_step=4)
    est = estimate_session_minutes(steps, LAUNCHES)
    assert est["n_servers"] == len(est["per_policy"])
    assert est["n_runs"] == len(steps)
    assert est["assumptions"]["pessimistic_multiplier"] == 1.6


def test_estimate_is_json_serialisable():
    import json

    from heterospec.session import estimate_session_minutes

    steps = build_session_plan(LAUNCHES, ks=[1], concurrencies=[8], waves_per_step=4)
    assert json.loads(json.dumps(estimate_session_minutes(steps, LAUNCHES)))
