"""Tests for benchmark orchestration and the CLI.

The runner is where a subtle mistake becomes expensive: a mis-stamped K would
produce biased position-acceptance curves that look perfectly valid, and it
would only be discovered after paying for the GPU time. These tests pin the
static-K/adaptive-K distinction and the dispatch modes.
"""

import json
from pathlib import Path

import pytest

from heterospec.benchmark import main
from heterospec.mockserver import MockSGLangServer
from heterospec.records import load_request_records, read_json
from heterospec.runner import RunConfig, run_benchmark

REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCH_CONFIG = REPO_ROOT / "configs" / "models" / "llama31_8b_eagle3.json"


@pytest.fixture
def server():
    with MockSGLangServer(k=3, seed=0) as srv:
        yield srv


def _cfg(server, tmp_path, **kw) -> RunConfig:
    base = dict(
        workload="mixed_50_50",
        policy_id="static_k3",
        base_url=server.base_url,
        num_requests=24,
        concurrency=8,
        seed=0,
        results_root=tmp_path,
        sglang_path=None,
        static_k=3,
        progress=None,
    )
    base.update(kw)
    return RunConfig(**base)


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def test_config_rejects_bad_dispatch(server, tmp_path):
    with pytest.raises(ValueError, match="dispatch must be"):
        _cfg(server, tmp_path, dispatch="nonsense")


def test_config_rejects_zero_requests(server, tmp_path):
    with pytest.raises(ValueError, match="num_requests"):
        _cfg(server, tmp_path, num_requests=0)


def test_config_rejects_zero_concurrency(server, tmp_path):
    with pytest.raises(ValueError, match="concurrency"):
        _cfg(server, tmp_path, concurrency=0)


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------


def test_dry_run_sends_nothing_and_writes_nothing(server, tmp_path):
    result = run_benchmark(_cfg(server, tmp_path, dry_run=True))
    assert result.records == []
    assert result.run_dir is None
    assert result.plan_summary["total"] == 24
    assert not list(tmp_path.glob("*"))


def test_dry_run_still_reports_the_plan(server, tmp_path):
    r = run_benchmark(_cfg(server, tmp_path, dry_run=True, num_requests=40))
    assert r.plan_summary["by_class"] == {"open_ended": 20, "repetitive": 20}


# ---------------------------------------------------------------------------
# Dispatch modes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dispatch", ["waves", "continuous"])
def test_both_dispatch_modes_complete_all_requests(server, tmp_path, dispatch):
    r = run_benchmark(_cfg(server, tmp_path, dispatch=dispatch))
    assert len(r.records) == 24
    assert all(rec.ok for rec in r.records)
    assert r.aggregate["n_ok"] == 24


def test_records_are_sorted_by_plan_index(server, tmp_path):
    r = run_benchmark(_cfg(server, tmp_path, concurrency=8))
    assert [rec.index for rec in r.records] == list(range(24))


def test_wave_dispatch_batches_mixed_classes(tmp_path):
    """A wave is a controlled mixture, which is what the oracle needs.

    With concurrency 8 on a 50/50 plan, every wave must contain both classes --
    otherwise 'the best K for this batch' is undefined.
    """
    with MockSGLangServer(k=3, seed=0) as srv:
        r = run_benchmark(_cfg(srv, tmp_path, num_requests=32, concurrency=8))
    for start in range(0, 32, 8):
        wave = r.records[start : start + 8]
        assert {rec.prompt_class for rec in wave} == {"repetitive", "open_ended"}


# ---------------------------------------------------------------------------
# The static-K / adaptive-K distinction -- the expensive mistake
# ---------------------------------------------------------------------------


def test_static_run_stamps_known_k_on_every_record(server, tmp_path):
    r = run_benchmark(_cfg(server, tmp_path, static_k=3))
    assert {rec.inferred_k_static for rec in r.records} == {3}
    assert all(rec.effective_k() == 3.0 for rec in r.records)


def test_static_run_yields_clean_position_acceptance(server, tmp_path):
    """Static K is what makes P(A_k) unbiased -- the whole point of the mode."""
    r = run_benchmark(_cfg(server, tmp_path, static_k=3))
    for rec in r.records:
        pa = rec.position_acceptance()
        assert pa is not None
        assert not pa.k_confounded, "a static-K capture must not be confounded"
        assert pa.safe_up_to == 3


def test_adaptive_run_leaves_k_unknown_and_refuses_position_acceptance(
    server, tmp_path
):
    """An adaptive run must not fabricate P(A_k) from a K-varying histogram."""
    r = run_benchmark(
        _cfg(server, tmp_path, static_k=None, policy_id="sglang_adaptive")
    )
    assert {rec.inferred_k_static for rec in r.records} == {None}
    assert all(rec.effective_k() is None for rec in r.records)
    for rec in r.records:
        pa = rec.position_acceptance()
        assert pa is not None and pa.k_confounded
    # ...and DraftWaste is undefined rather than guessed:
    assert r.aggregate["draft_waste"] is None
    assert r.aggregate["n_with_known_k"] == 0
    assert r.aggregate["draft_waste_covers_all_speculative"] is False


def test_static_run_reports_full_coverage_draft_waste(server, tmp_path):
    r = run_benchmark(_cfg(server, tmp_path, static_k=3))
    assert r.aggregate["draft_waste"] is not None
    assert r.aggregate["draft_waste_covers_all_speculative"] is True
    assert r.aggregate["n_with_known_k"] == r.aggregate["n_speculative"]


# ---------------------------------------------------------------------------
# Failures are recorded, not raised
# ---------------------------------------------------------------------------


def test_unreachable_server_records_failures_without_raising(tmp_path):
    """A crashed sweep wastes the GPU hour it was paying for."""
    cfg = RunConfig(
        workload="high",
        policy_id="static_k3",
        base_url="http://127.0.0.1:1",
        num_requests=6,
        concurrency=3,
        timeout_s=2.0,
        results_root=tmp_path,
        sglang_path=None,
        static_k=3,
    )
    r = run_benchmark(cfg)
    assert len(r.records) == 6
    assert all(not rec.ok for rec in r.records)
    assert all(rec.error for rec in r.records)
    assert r.aggregate["n_failed"] == 6
    assert r.aggregate["n_ok"] == 0


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def test_run_writes_metadata_requests_and_aggregate(server, tmp_path):
    r = run_benchmark(_cfg(server, tmp_path))
    assert r.run_dir is not None
    assert (r.run_dir / "metadata.json").is_file()
    assert (r.run_dir / "requests.jsonl").is_file()
    assert (r.run_dir / "aggregate.json").is_file()

    meta = read_json(r.run_dir / "metadata.json")
    assert meta["policy_id"] == "static_k3"
    assert meta["workload_name"] == "mixed_50_50"
    assert meta["is_mock"] is True
    assert meta["citable"] is False
    assert "mock" in meta["citable_reason"]


def test_written_records_round_trip(server, tmp_path):
    r = run_benchmark(_cfg(server, tmp_path))
    loaded = load_request_records(r.run_dir / "requests.jsonl")
    assert len(loaded) == len(r.records)
    assert loaded[0].rid == r.records[0].rid
    assert loaded[0].spec_correct_drafts_histogram == (
        r.records[0].spec_correct_drafts_histogram
    )


def test_metadata_records_launch_and_workload_provenance(server, tmp_path):
    launch = {"id": "static_k3", "spec": {"num_steps": 3, "adaptive": False}}
    r = run_benchmark(_cfg(server, tmp_path, launch=launch, concurrency=4, seed=9))
    meta = read_json(r.run_dir / "metadata.json")
    assert meta["launch"]["spec"]["num_steps"] == 3
    assert meta["workload"]["seed"] == 9
    assert meta["workload"]["concurrency"] == 4
    assert meta["workload"]["dispatch"] == "waves"


def test_no_write_suppresses_output(server, tmp_path):
    r = run_benchmark(_cfg(server, tmp_path, write_results=False))
    assert r.run_dir is None
    assert not list(tmp_path.glob("*"))


def test_run_dir_name_encodes_workload_and_policy(server, tmp_path):
    r = run_benchmark(_cfg(server, tmp_path))
    assert "mixed_50_50" in r.run_dir.name
    assert "static_k3" in r.run_dir.name


def test_progress_callback_receives_updates(server, tmp_path):
    seen: list[str] = []
    run_benchmark(
        _cfg(server, tmp_path, progress=seen.append, progress_every=8, num_requests=16)
    )
    assert any("plan:" in s for s in seen)
    assert any("/16 sent" in s for s in seen)


def test_aggregate_reports_class_separation(server, tmp_path):
    """End-to-end sanity: the mock's signal must survive the whole pipeline."""
    r = run_benchmark(_cfg(server, tmp_path, num_requests=60, static_k=3))
    rep = r.aggregate["by_class"]["repetitive"]["mean_accepted_drafts"]
    low = r.aggregate["by_class"]["open_ended"]["mean_accepted_drafts"]
    assert rep > low * 2


def test_histograms_are_self_consistent_through_the_pipeline(server, tmp_path):
    """sum(histogram) == spec_verify_ct must hold for every record."""
    r = run_benchmark(_cfg(server, tmp_path))
    for rec in r.records:
        ok, msg = rec.histogram_consistent()
        assert ok, f"{rec.rid}: {msg}"


def test_completion_token_accounting_invariant(server, tmp_path):
    """completion_tokens == accepted drafts + one bonus token per round."""
    r = run_benchmark(_cfg(server, tmp_path))
    for rec in r.records:
        assert rec.completion_tokens == (
            rec.spec_num_correct_drafts + rec.spec_verify_ct
        ), rec.rid


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_list_workloads(capsys):
    assert main(["--list-workloads"]) == 0
    out = capsys.readouterr().out
    for name in ("high", "low", "mixed_50_50", "phase_shift", "real_mixed"):
        assert name in out


def test_cli_list_policies(capsys):
    assert main(["--list-policies", "--launch-config", str(LAUNCH_CONFIG)]) == 0
    out = capsys.readouterr().out
    assert "sglang_adaptive" in out and "adaptive" in out
    assert "static_k3" in out and "static" in out


def test_cli_print_commands_emits_full_launch(capsys):
    assert main(["--print-commands", "--launch-config", str(LAUNCH_CONFIG)]) == 0
    out = capsys.readouterr().out
    assert "sglang.launch_server" in out
    assert "--speculative-adaptive" in out
    assert "--speculative-draft-model-path" in out


def test_cli_requires_workload_and_policy(capsys):
    assert main([]) == 2
    assert main(["--workload", "high"]) == 2


def test_cli_rejects_unknown_workload(capsys):
    assert main(["--workload", "nope", "--policy", "static_k3"]) == 2
    assert "unknown workload" in capsys.readouterr().err


def test_cli_rejects_unknown_policy(capsys):
    assert main(["--workload", "high", "--policy", "nope"]) == 2
    assert "unknown policy" in capsys.readouterr().err


def test_cli_end_to_end_with_mock(tmp_path, capsys):
    rc = main(
        [
            "--workload",
            "mixed_50_50",
            "--policy",
            "static_k3",
            "--num-requests",
            "16",
            "--concurrency",
            "8",
            "--mock",
            "--seed",
            "0",
            "--results-root",
            str(tmp_path),
            "--sglang-path",
            str(tmp_path),  # not a checkout: git info degrades
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "citable" in out
    assert "mock" in out
    run_dirs = list(tmp_path.glob("*/metadata.json"))
    assert len(run_dirs) == 1
    meta = json.loads(run_dirs[0].read_text())
    assert meta["is_mock"] is True
    assert meta["citable"] is False


def test_cli_quiet_suppresses_output(tmp_path, capsys):
    rc = main(
        [
            "--workload",
            "high",
            "--policy",
            "static_k1",
            "--num-requests",
            "8",
            "--mock",
            "--quiet",
            "--results-root",
            str(tmp_path),
            "--sglang-path",
            str(tmp_path),
        ]
    )
    assert rc == 0
    assert capsys.readouterr().out.strip() == ""


def test_cli_dry_run(tmp_path, capsys):
    rc = main(
        [
            "--workload",
            "phase_shift",
            "--policy",
            "static_k1",
            "--num-requests",
            "40",
            "--dry-run",
            "--results-root",
            str(tmp_path),
            "--sglang-path",
            str(tmp_path),
        ]
    )
    assert rc == 0
    assert not list(tmp_path.glob("*"))
