"""Tests for the session orchestrator itself: subprocess launch, readiness
polling, teardown, and failure handling.

This is the code that runs on the rented host, and it is the last thing that
should ever be debugged at $1+/hour. `tests/stub_server/launch_server.py` stands
in for SGLang, so the *real* orchestration path is exercised on the Mac: a genuine
subprocess, a genuine socket, genuine health polling, genuine SIGTERM teardown.
"""

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from heterospec.config import load_launch_configs
from heterospec.session import (
    ServerProcess,
    SessionStep,
    build_session_plan,
    cost_model_from_results,
    k_invariance_from_results,
    oracle_gap_report,
    run_session,
    write_session_report,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
STUB_DIR = REPO_ROOT / "tests" / "stub_server"
LAUNCH_CONFIG = REPO_ROOT / "configs" / "models" / "llama31_8b_eagle3.json"
LAUNCHES = {c.id: c for c in load_launch_configs(LAUNCH_CONFIG)}


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _stub_env(**extra) -> dict[str, str]:
    """PYTHONPATH so the child can import the stub and heterospec."""
    existing = os.environ.get("PYTHONPATH", "")
    parts = [str(STUB_DIR), str(REPO_ROOT)] + ([existing] if existing else [])
    return {"PYTHONPATH": os.pathsep.join(parts), **extra}


@pytest.fixture
def port():
    return _free_port()


# ---------------------------------------------------------------------------
# ServerProcess in isolation
# ---------------------------------------------------------------------------


def test_server_process_starts_and_stops(tmp_path, port):
    sp = ServerProcess(
        LAUNCHES["static_k3"],
        sglang_path=tmp_path,
        port=port,
        log_path=tmp_path / "server.log",
        python_exe=sys.executable,
        env_extra=_stub_env(),
        launcher_module="launch_server",
    )
    try:
        sp.start()
        assert sp.wait_ready(timeout_s=60, poll_s=0.2) is True
    finally:
        sp.stop()
    # The port must be free again: the child is really gone.
    assert _port_is_free(port), "server process was not reaped"


def _port_is_free(port: int) -> bool:
    with socket.socket() as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def test_argv_uses_the_configured_launcher(tmp_path, port):
    sp = ServerProcess(
        LAUNCHES["static_k3"],
        sglang_path=tmp_path,
        port=port,
        log_path=tmp_path / "l.log",
        python_exe="/usr/bin/python3",
        launcher_module="some.other.launcher",
    )
    argv = sp._argv()
    assert argv[:3] == ["/usr/bin/python3", "-m", "some.other.launcher"]
    assert "--port" in argv and str(port) in argv


def test_argv_carries_the_full_launch_config(tmp_path, port):
    sp = ServerProcess(
        LAUNCHES["sglang_adaptive"],
        sglang_path=tmp_path,
        port=port,
        log_path=tmp_path / "l.log",
    )
    argv = sp._argv()
    assert "--speculative-adaptive" in argv
    assert "--speculative-draft-model-path" in argv


def test_wait_ready_raises_with_log_tail_when_server_exits(tmp_path, port):
    """A crashed server must fail fast with its log, not hang until timeout.

    On a rented host, hanging for 30 minutes before reporting nothing is nearly
    as bad as crashing.
    """
    bad = tmp_path / "bad_launcher.py"
    bad.write_text("import sys; print('boom: no CUDA'); sys.exit(3)\n")
    sp = ServerProcess(
        LAUNCHES["static_k1"],
        sglang_path=tmp_path,
        port=port,
        log_path=tmp_path / "bad.log",
        python_exe=sys.executable,
        env_extra={"PYTHONPATH": str(tmp_path)},
        launcher_module="bad_launcher",
    )
    sp.start()
    with pytest.raises(RuntimeError, match="exited early with code 3"):
        sp.wait_ready(timeout_s=30, poll_s=0.2)
    sp.stop()
    assert "boom: no CUDA" in sp.tail_log()


def test_wait_ready_times_out_returns_false(tmp_path, port):
    """A launcher that never listens must return False, not hang."""
    sleeper = tmp_path / "sleeper.py"
    sleeper.write_text("import time\ntime.sleep(300)\n")
    sp = ServerProcess(
        LAUNCHES["static_k1"],
        sglang_path=tmp_path,
        port=port,
        log_path=tmp_path / "s.log",
        python_exe=sys.executable,
        env_extra={"PYTHONPATH": str(tmp_path)},
        launcher_module="sleeper",
    )
    sp.start()
    try:
        t0 = time.perf_counter()
        assert sp.wait_ready(timeout_s=2, poll_s=0.2) is False
        assert time.perf_counter() - t0 < 15
    finally:
        sp.stop()


def test_stop_is_idempotent(tmp_path, port):
    sp = ServerProcess(
        LAUNCHES["static_k1"],
        sglang_path=tmp_path,
        port=port,
        log_path=tmp_path / "l.log",
        python_exe=sys.executable,
        env_extra=_stub_env(),
        launcher_module="launch_server",
    )
    sp.stop()  # never started
    sp.start()
    sp.stop()
    sp.stop()  # again


def test_env_extra_reaches_the_child(tmp_path, port):
    """The mechanism that carries HF_HOME and HF_TOKEN on the GPU host."""
    sp = ServerProcess(
        LAUNCHES["static_k1"],
        sglang_path=tmp_path,
        port=port,
        log_path=tmp_path / "l.log",
        python_exe=sys.executable,
        env_extra=_stub_env(
            HF_HOME="/workspace/hf",
            HF_TOKEN="secret-token",
            STUB_ENV_DUMP=str(tmp_path / "env.json"),
        ),
        launcher_module="launch_server",
    )
    try:
        sp.start()
        assert sp.wait_ready(timeout_s=60, poll_s=0.2)
        dump = json.loads((tmp_path / "env.json").read_text())
        assert dump["HF_HOME"] == "/workspace/hf"
        assert dump["HF_TOKEN"] == "secret-token"
    finally:
        sp.stop()


def test_trace_env_var_is_set_only_when_tracing(tmp_path, port):
    trace = tmp_path / "trace.jsonl"
    sp = ServerProcess(
        LAUNCHES["sglang_adaptive"],
        sglang_path=tmp_path,
        port=port,
        log_path=tmp_path / "l.log",
        python_exe=sys.executable,
        trace_path=trace,
        env_extra=_stub_env(STUB_ENV_DUMP=str(tmp_path / "env.json")),
        launcher_module="launch_server",
    )
    try:
        sp.start()
        assert sp.wait_ready(timeout_s=60, poll_s=0.2)
        dump = json.loads((tmp_path / "env.json").read_text())
        assert dump["SGLANG_HETEROSPEC_TRACE"] == str(trace)
    finally:
        sp.stop()


def test_server_log_records_the_command(tmp_path, port):
    sp = ServerProcess(
        LAUNCHES["static_k1"],
        sglang_path=tmp_path,
        port=port,
        log_path=tmp_path / "cmd.log",
        python_exe=sys.executable,
        env_extra=_stub_env(),
        launcher_module="launch_server",
    )
    try:
        sp.start()
        sp.wait_ready(timeout_s=60, poll_s=0.2)
    finally:
        sp.stop()
    text = (tmp_path / "cmd.log").read_text()
    assert "-m launch_server" in text
    assert "--speculative-num-steps 1" in text


# ---------------------------------------------------------------------------
# run_session end to end
# ---------------------------------------------------------------------------


def _small_steps() -> list[SessionStep]:
    return build_session_plan(
        LAUNCHES,
        ks=[1, 3],
        concurrencies=[8],
        waves_per_step=6,
        include_adaptive=True,
        include_nospec=True,
    )


def test_run_session_end_to_end(tmp_path, port):
    """The full GPU-host path, on the Mac: launch -> poll -> benchmark -> teardown."""
    events: list[str] = []
    results = run_session(
        _small_steps(),
        launches=LAUNCHES,
        sglang_path=tmp_path,
        results_root=tmp_path / "results",
        logs_dir=tmp_path / "logs",
        port=port,
        python_exe=sys.executable,
        server_timeout_s=90,
        env_extra=_stub_env(),
        launcher_module="launch_server",
        on_event=events.append,
    )

    assert len(results) == 4, [r.error for r in results]
    assert all(r.ok for r in results), [r.error for r in results]
    assert all(r.run_dir for r in results)
    assert all(Path(r.run_dir).is_dir() for r in results)
    assert any("-> ok" in e for e in events)
    # Server logs were written.
    assert list((tmp_path / "logs").glob("server_*.log"))
    # The port is released at the end.
    assert _port_is_free(port)


def test_run_session_groups_steps_by_policy(tmp_path, port):
    """One server launch per policy, not per step, or the hour is wasted."""
    events: list[str] = []
    steps = build_session_plan(
        LAUNCHES,
        ks=[1],
        concurrencies=[4, 8],
        waves_per_step=4,
        include_adaptive=False,
        include_nospec=False,
    )
    results = run_session(
        steps,
        launches=LAUNCHES,
        sglang_path=tmp_path,
        results_root=tmp_path / "results",
        logs_dir=tmp_path / "logs",
        port=port,
        python_exe=sys.executable,
        server_timeout_s=90,
        env_extra=_stub_env(),
        launcher_module="launch_server",
        on_event=events.append,
    )
    assert len(results) == 2
    assert all(r.ok for r in results)
    # Two steps, one policy -> exactly one server, hence one ready line.
    assert sum(1 for e in events if "server ready:" in e) == 1
    assert sum(1 for e in events if "=== server:" in e) == 1
    assert sum(1 for e in events if "-> running" in e) == 2


def test_run_session_continues_after_a_failed_step(tmp_path, port):
    """Losing a paid session to one flaky run is the expensive failure mode."""
    steps = [
        SessionStep("static_k1", 8, "mixed_50_50", 48, 0, "calibration", 1),
        # K=99 has no launch config, so this step cannot even build its server.
        SessionStep("nonexistent_policy", 8, "mixed_50_50", 48, 0, "calibration", 99),
        SessionStep("static_k3", 8, "mixed_50_50", 48, 0, "calibration", 3),
    ]
    results = run_session(
        steps,
        launches=LAUNCHES,
        sglang_path=tmp_path,
        results_root=tmp_path / "results",
        logs_dir=tmp_path / "logs",
        port=port,
        python_exe=sys.executable,
        server_timeout_s=90,
        env_extra=_stub_env(),
        launcher_module="launch_server",
    )
    assert len(results) == 3
    ok = [r for r in results if r.ok]
    failed = [r for r in results if not r.ok]
    assert len(ok) == 2, "the good steps must still have run"
    assert len(failed) == 1
    assert "nonexistent_policy" in failed[0].error
    assert failed[0].step.policy_id == "nonexistent_policy"


def test_run_session_produces_analysable_output(tmp_path, port):
    """Session output must flow straight into the cost model, oracle and check."""
    steps = build_session_plan(
        LAUNCHES,
        ks=[1, 3],
        concurrencies=[8],
        waves_per_step=8,
        include_adaptive=False,
        include_nospec=False,
    )
    results = run_session(
        steps,
        launches=LAUNCHES,
        sglang_path=tmp_path,
        results_root=tmp_path / "results",
        logs_dir=tmp_path / "logs",
        port=port,
        python_exe=sys.executable,
        server_timeout_s=90,
        env_extra=_stub_env(),
        launcher_module="launch_server",
    )
    assert all(r.ok for r in results)

    model, missing = cost_model_from_results(results, tmp_path / "results")
    assert not missing, missing
    assert model.k_values == [1, 3]

    inv = k_invariance_from_results(results)
    assert inv is not None
    assert isinstance(inv.verdict(), str)

    gap = oracle_gap_report(results, model, invariance=inv, n_bins=4, seed=0)
    assert gap["captures"], gap
    assert "gap_summary" in gap

    report = write_session_report(
        results,
        results_root=tmp_path / "results",
        cost=model,
        missing_cells=missing,
        gap_report=gap,
        invariance=inv.summary() if inv else None,
    )
    payload = json.loads(report.read_text())
    assert payload["n_ok"] == 2
    assert "k_invariance" in payload
    assert "gap" in payload


def test_run_session_records_trace_when_adaptive(tmp_path, port):
    """The adaptive step must leave an iteration trace behind."""
    steps = [SessionStep("sglang_adaptive", 8, "mixed_50_50", 48, 0, "adaptive", None)]
    results = run_session(
        steps,
        launches=LAUNCHES,
        sglang_path=tmp_path,
        results_root=tmp_path / "results",
        logs_dir=tmp_path / "logs",
        port=port,
        python_exe=sys.executable,
        server_timeout_s=90,
        env_extra=_stub_env(),
        launcher_module="launch_server",
    )
    assert all(r.ok for r in results)
    # The stub does not implement the trace, so the file is absent; what matters
    # is that the orchestrator looked for it and did not crash.
    assert results[0].trace_records == 0


def test_run_session_no_orphan_processes(tmp_path, port):
    """Every launched server must be reaped, including across multiple policies."""
    steps = build_session_plan(
        LAUNCHES,
        ks=[1, 3],
        concurrencies=[4],
        waves_per_step=4,
        include_adaptive=True,
        include_nospec=False,
    )
    run_session(
        steps,
        launches=LAUNCHES,
        sglang_path=tmp_path,
        results_root=tmp_path / "results",
        logs_dir=tmp_path / "logs",
        port=port,
        python_exe=sys.executable,
        server_timeout_s=90,
        env_extra=_stub_env(),
        launcher_module="launch_server",
    )
    time.sleep(0.5)
    out = subprocess.run(
        ["pgrep", "-f", "launch_server"], capture_output=True, text=True, check=False
    )
    leftover = [p for p in out.stdout.split() if p.isdigit()]
    assert not leftover, f"orphaned server processes: {leftover}"
    assert _port_is_free(port)


# ---------------------------------------------------------------------------
# Warm-up in the orchestrator
# ---------------------------------------------------------------------------


def test_run_session_performs_warmup_before_timing(tmp_path, port):
    events: list[str] = []
    steps = [SessionStep("static_k1", 8, "mixed_50_50", 48, 0, "calibration", 1)]
    results = run_session(
        steps,
        launches=LAUNCHES,
        sglang_path=tmp_path,
        results_root=tmp_path / "results",
        logs_dir=tmp_path / "logs",
        port=port,
        python_exe=sys.executable,
        server_timeout_s=90,
        env_extra=_stub_env(),
        launcher_module="launch_server",
        warmup_requests=4,
        on_event=events.append,
    )
    assert all(r.ok for r in results)
    warm = [e for e in events if "warm-up:" in e]
    assert warm, events
    assert "0 failed" in warm[0]


def test_warmup_can_be_disabled(tmp_path, port):
    events: list[str] = []
    steps = [SessionStep("static_k1", 8, "mixed_50_50", 48, 0, "calibration", 1)]
    run_session(
        steps,
        launches=LAUNCHES,
        sglang_path=tmp_path,
        results_root=tmp_path / "results",
        logs_dir=tmp_path / "logs",
        port=port,
        python_exe=sys.executable,
        server_timeout_s=90,
        env_extra=_stub_env(),
        launcher_module="launch_server",
        warmup_requests=0,
        on_event=events.append,
    )
    assert not [e for e in events if "warm-up:" in e]


def test_all_warmup_failing_fails_the_step_with_a_clear_reason(tmp_path, port):
    """A server that is up but not serving must not produce a timed run.

    Timing a server that cannot answer would produce a cost model built on
    failures -- worse than no result, because it looks like data.
    """
    steps = [SessionStep("static_k1", 8, "mixed_50_50", 48, 0, "calibration", 1)]
    results = run_session(
        steps,
        launches=LAUNCHES,
        sglang_path=tmp_path,
        results_root=tmp_path / "results",
        logs_dir=tmp_path / "logs",
        port=port,
        python_exe=sys.executable,
        server_timeout_s=90,
        env_extra=_stub_env(STUB_FAIL_GENERATE="1"),
        launcher_module="launch_server",
        warmup_requests=2,
    )
    assert len(results) == 1
    assert not results[0].ok
    assert "every warm-up request failed" in (results[0].error or "")


# ---------------------------------------------------------------------------
# Tracing split
# ---------------------------------------------------------------------------


def _child_env_for(tmp_path, port, step) -> dict:
    """Launch one step and return the server child's environment."""
    dump = tmp_path / f"env_{step.purpose}.json"
    run_session(
        [step],
        launches=LAUNCHES,
        sglang_path=tmp_path,
        results_root=tmp_path / "results",
        logs_dir=tmp_path / "logs",
        port=port,
        python_exe=sys.executable,
        server_timeout_s=90,
        env_extra=_stub_env(STUB_ENV_DUMP=str(dump)),
        launcher_module="launch_server",
        warmup_requests=0,
    )
    return json.loads(dump.read_text())


def test_the_calibration_server_never_gets_the_trace_env_var(tmp_path, port):
    """The tracer does synchronous JSON work once per decode iteration.

    Paying that on the runs that produce the cost model -- the oracle's
    denominator -- would be careless when the effect under study is a few percent.
    Calibration's oracle input comes from meta_info, not from the trace.

    Asserting on the child's environment is stronger than checking for a trace
    file, which a stub server would never write anyway.
    """
    env = _child_env_for(
        tmp_path,
        port,
        SessionStep("static_k1", 8, "mixed_50_50", 48, 0, "calibration", 1),
    )
    assert "SGLANG_HETEROSPEC_TRACE" not in env, (
        "calibration server was launched with tracing enabled"
    )


def test_the_adaptive_server_does_get_the_trace_env_var(tmp_path, port):
    """The other half of the split: the adaptive capture must still be traced."""
    env = _child_env_for(
        tmp_path,
        port,
        SessionStep("sglang_adaptive", 8, "mixed_50_50", 48, 0, "adaptive", None),
    )
    assert "SGLANG_HETEROSPEC_TRACE" in env
    assert env["SGLANG_HETEROSPEC_TRACE"].endswith("trace_sglang_adaptive.jsonl")


# ---------------------------------------------------------------------------
# Config-recorded environment
# ---------------------------------------------------------------------------


def test_the_config_recorded_env_reaches_the_server_process(tmp_path, port):
    """The EAGLE3 pair cannot start at all without this variable.

    SGLang's `ModelConfig._derive_context_length` raises when the draft head's
    `max_position_embeddings` (2048) is below the target's context length
    (131072). That kills the scheduler's draft worker; SGLang then tears down its
    own process tree and the launcher dies with SIGKILL, which on a rented host
    looks exactly like an OOM kill -- it cost a session to diagnose, because the
    real error is buried under "server exited early with code -9".

    The setting therefore lives in the launch config, so a re-run from the config
    alone reproduces the launch instead of silently depending on an exported
    shell variable.
    """
    env = _child_env_for(
        tmp_path,
        port,
        SessionStep("static_k1", 8, "mixed_50_50", 48, 0, "calibration", 1),
    )
    assert env.get("SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN") == "1", (
        "the server would refuse to build the EAGLE3 draft worker without this"
    )


def test_the_required_env_is_recorded_in_the_result_metadata():
    """It must be in `metadata.json`, not only in the process environment.

    A reader has to be able to see that the run needed it; the config's `to_dict`
    is what lands in the per-run metadata.
    """
    for policy_id, cfg in LAUNCHES.items():
        recorded = cfg.to_dict()["model"]["env"]
        assert recorded.get("SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN") == "1", (
            f"{policy_id} would not record the required env var"
        )
