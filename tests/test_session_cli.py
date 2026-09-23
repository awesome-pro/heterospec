"""Regression tests that exercise `python -m heterospec.session` as a *process*.

The bug these exist for is invisible to every import-based test. `main()` called
`k_invariance_from_results()`, which was defined **below** the
``if __name__ == "__main__":`` guard. Importing the module never runs the guard, so
`main` was bound and all unit tests passed; running the module executed `main()`
before the function existed, raising `NameError`.

What made it dangerous rather than merely broken is that `main()` wraps the
invariance check in a broad `except`. The NameError was swallowed, the K-invariance
gate was treated as "not assessed", and the report came back with

    status = "suppressed"

and a **zero exit code**. On the GPU host that reads as a legitimate scientific
no-go -- "the gap could not be interpreted" -- rather than as a defect in the code,
*after* the session had been paid for. Only a real subprocess can catch that.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from heterospec.mockserver import MockSGLangServer
from heterospec.runner import RunConfig, run_benchmark
from heterospec.session import SessionStep, StepResult, write_session_report

REPO_ROOT = Path(__file__).resolve().parents[1]
SESSION_SRC = REPO_ROOT / "heterospec" / "session.py"
SESSION_MODULE = REPO_ROOT / "heterospec" / "session.py"
LAUNCH_CONFIG = REPO_ROOT / "configs" / "models" / "llama31_8b_eagle3.json"


def _make_capture(tmp_path: Path, *, k: int, conc: int, waves: int) -> StepResult:
    """One real calibration capture in a real run directory, via the mock server."""
    with MockSGLangServer(k=k, seed=0) as srv:
        cfg = RunConfig(
            workload="mixed_50_50",
            policy_id=f"static_k{k}_c{conc}",
            base_url=srv.base_url,
            num_requests=waves * conc,
            concurrency=conc,
            dispatch="waves",
            seed=0,
            results_root=tmp_path,
            sglang_path=None,
            static_k=k,
            progress=None,
        )
        r = run_benchmark(cfg)
    return StepResult(
        step=SessionStep(
            policy_id=f"static_k{k}",
            concurrency=conc,
            workload="mixed_50_50",
            num_requests=waves * conc,
            seed=0,
            purpose="calibration",
            static_k=k,
        ),
        ok=True,
        run_dir=str(r.run_dir),
        wall_time_s=r.wall_time_s,
        dispatch_wall_time_s=r.dispatch_wall_time_s,
        n_ok=r.aggregate.get("n_ok", 0),
        n_failed=r.aggregate.get("n_failed", 0),
    )


@pytest.fixture
def results_root(tmp_path: Path) -> Path:
    """A results tree with enough static captures for the whole analysis chain.

    Two depths at one concurrency gives the invariance check something to compare,
    and enough waves per depth gives the oracle its required train/test split.
    """
    root = tmp_path / "session1"
    root.mkdir()
    results = [
        _make_capture(root, k=1, conc=4, waves=8),
        _make_capture(root, k=3, conc=4, waves=8),
    ]
    write_session_report(results, results_root=root)
    return root


def test_the_main_guard_is_the_last_statement_in_the_module():
    """Static half of the guard: nothing may be defined after `__main__`.

    A cheap structural check that fails the moment someone appends a function
    below the guard, which is how the original bug was introduced.
    """
    src = SESSION_MODULE.read_text()
    idx = src.rindex('if __name__ == "__main__":')
    tail = src[idx:]
    assert "\ndef " not in tail, (
        "a function is defined after the __main__ guard, so running "
        "`python -m heterospec.session` can call main() before it exists:\n"
        + tail[:400]
    )
    assert "\nclass " not in tail, "a class is defined after the __main__ guard"


def test_module_runs_and_actually_assesses_k_invariance(results_root: Path):
    """The behavioural half: run the module and require the gate to be *assessed*.

    `K-invariance: PASS` is the assertion that matters. It can only be printed if
    `k_invariance_from_results` was bound when `main()` ran. The old ordering could
    not produce it at any price.
    """
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "heterospec.session",
            "--analyze-only",
            "--launch-config",
            str(LAUNCH_CONFIG),
            "--results-root",
            str(results_root),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    combined = proc.stdout + proc.stderr

    # The precise failure mode: a swallowed NameError downgrading the gate to
    # "not assessed" while still exiting successfully.
    assert "NameError" not in combined, combined
    assert "failed to run" not in combined, combined
    assert "K-invariance: NOT ASSESSED" not in combined, combined
    assert "K-invariance: PASS" in combined, combined
    assert "DECISION GAP SUPPRESSED" not in combined, (
        "the gap was suppressed even though two static depths were captured, so "
        "the invariance check cannot have run:\n" + combined
    )


def test_analyze_only_works_after_the_results_root_is_moved(results_root: Path):
    """Session 1 runs on a rented host; its results are then copied elsewhere.

    Every recorded path must therefore survive the move. With absolute paths the
    re-analysis would find no `requests.jsonl` and report an analysis failure on
    data that is perfectly intact.
    """
    report = json.loads((results_root / "session1_report.json").read_text())
    # The stored paths must be relative to the results root, or the move cannot work.
    assert all(not Path(s["run_dir"]).is_absolute() for s in report["steps"]), (
        f"run_dir stored as an absolute path: {[s['run_dir'] for s in report['steps']]}"
    )

    moved = results_root.parent / "pulled-back" / "session1"
    moved.parent.mkdir()
    results_root.rename(moved)

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "heterospec.session",
            "--analyze-only",
            "--launch-config",
            str(LAUNCH_CONFIG),
            "--results-root",
            str(moved),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=600,
    )
    combined = proc.stdout + proc.stderr
    assert proc.returncode == 0, combined
    assert "K-invariance: PASS" in combined, combined

    after = json.loads((moved / "session1_report.json").read_text())
    gap = after["gap"]
    assert gap["status"] == "ok", gap
    # Every capture must still resolve, i.e. no "no such file" / "unusable" error.
    for cap in gap["captures"]:
        assert "error" not in cap, cap
        assert cap["n_batches"] >= 4, cap
        assert Path(moved, cap["run_dir"]).is_dir() or Path(cap["run_dir"]).is_dir(), (
            cap
        )

    # The action set is the measured grid: never an interpolated K, never a
    # clamped K=0.
    assert gap["candidate_ks"] == sorted(gap["cost_grid_ks"])
    assert 0 not in gap["candidate_ks"]
    assert gap["no_spec_reference"]["is_oracle_action"] is False
