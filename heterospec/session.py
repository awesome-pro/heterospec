"""One-shot GPU session orchestration: calibrate, capture, analyse.

The point of this module is to spend as few GPU hours as possible. It encodes the
whole of Session 1 as a plan that can be *built and inspected on the Mac*, then
executed once on the rented host:

1. **Calibration grid** — static `K` x concurrency, which simultaneously yields
   the cost model, the per-request acceptance profiles, and the data for the
   K-invariance check. One grid, four deliverables.
2. **Adaptive baseline** — the merged controller, with the iteration trace on, to
   validate the trace patch against real hardware.
3. **No-spec baseline** — the absolute target-only reference.

Then it assembles the measured cost model and computes the recoverable
rectangular gap from real traces.

Robustness is deliberate: a failed step records its error and the session
continues, because losing a whole paid session to one flaky run is the expensive
failure mode. The server process is always torn down in a `finally`.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from heterospec.client import SGLangClient
from heterospec.config import LaunchConfig
from heterospec.costmodel import CalibrationPoint, MeasuredCostModel
from heterospec.records import read_json
from heterospec.runner import RunConfig, run_benchmark, warmup_server
from heterospec.workloads import build_plan

__all__ = [
    "SessionStep",
    "StepResult",
    "build_session_plan",
    "ServerProcess",
    "run_session",
    "cost_model_from_results",
    "estimate_session_minutes",
    "k_invariance_from_results",
    "load_session_results",
    "oracle_gap_report",
    "portable_path",
    "resolve_stored_path",
    "write_session_report",
]


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SessionStep:
    """One benchmark run within a session."""

    policy_id: str
    concurrency: int
    workload: str
    num_requests: int
    seed: int
    purpose: str  # "calibration" | "adaptive" | "nospec"
    static_k: int | None = None

    @property
    def key(self) -> str:
        return f"{self.purpose}:{self.policy_id}:c{self.concurrency}"

    def describe(self) -> str:
        k = "-" if self.static_k is None else self.static_k
        return (
            f"{self.purpose:<11} policy={self.policy_id:<9} K={k:<4} "
            f"concurrency={self.concurrency:<3} n={self.num_requests}"
        )


def _policy_for_k(launches: dict[str, LaunchConfig], k: int) -> str:
    """The launch-config id implementing static depth `k`."""
    for cid, cfg in launches.items():
        if cfg.spec is not None and not cfg.spec.adaptive and cfg.spec.num_steps == k:
            return cid
    raise KeyError(
        f"no static launch config with num_steps={k}; have {sorted(launches)}"
    )


def _adaptive_policy(launches: dict[str, LaunchConfig]) -> str:
    for cid, cfg in launches.items():
        if cfg.spec is not None and cfg.spec.adaptive:
            return cid
    raise KeyError(f"no adaptive launch config; have {sorted(launches)}")


def _nospec_policy(launches: dict[str, LaunchConfig]) -> str:
    for cid, cfg in launches.items():
        if cfg.spec is None:
            return cid
    raise KeyError(f"no no-spec launch config; have {sorted(launches)}")


def build_session_plan(
    launches: dict[str, LaunchConfig],
    *,
    ks: Sequence[int] = (1, 3, 5, 7),
    concurrencies: Sequence[int] = (1, 8, 32),
    workload: str = "mixed_50_50",
    waves_per_step: int = 24,
    num_requests: int | None = None,
    seed: int = 0,
    include_nospec: bool = True,
    include_adaptive: bool = True,
) -> list[SessionStep]:
    """Build the Session 1 plan. Pure function: inspect it before paying for it.

    The calibration grid is `static K x concurrency`. Using the same workload and
    seed at every static `K` is what makes the K-invariance check possible at no
    extra cost: identical prompts at four depths.

    Request counts are derived from a fixed **number of waves**, not a fixed
    request count. With wave dispatch a step costs about
    `(n / concurrency) * per_request_time`, so holding `n / concurrency` constant
    keeps every step's wall time comparable. A flat `num_requests` would make the
    concurrency-1 step dominate the whole session, which is exactly the kind of
    imbalance that wastes a rented hour.
    """
    if not ks:
        raise ValueError("ks must be non-empty")
    if not concurrencies:
        raise ValueError("concurrencies must be non-empty")
    if num_requests is None and waves_per_step <= 0:
        raise ValueError("waves_per_step must be > 0")

    def n_for(conc: int) -> int:
        return int(num_requests) if num_requests is not None else waves_per_step * conc

    steps: list[SessionStep] = []
    for k in ks:
        cid = _policy_for_k(launches, k)
        for conc in concurrencies:
            steps.append(
                SessionStep(
                    policy_id=cid,
                    concurrency=conc,
                    workload=workload,
                    num_requests=n_for(conc),
                    seed=seed,
                    purpose="calibration",
                    static_k=k,
                )
            )

    if include_adaptive:
        cid = _adaptive_policy(launches)
        for conc in concurrencies:
            steps.append(
                SessionStep(
                    policy_id=cid,
                    concurrency=conc,
                    workload=workload,
                    num_requests=n_for(conc),
                    seed=seed,
                    purpose="adaptive",
                    static_k=None,
                )
            )

    if include_nospec:
        cid = _nospec_policy(launches)
        # One concurrency is enough: no-spec is a reference point, and its cost
        # does not enter the cost model's K axis.
        steps.append(
            SessionStep(
                policy_id=cid,
                concurrency=max(concurrencies),
                workload=workload,
                num_requests=n_for(max(concurrencies)),
                seed=seed,
                purpose="nospec",
                static_k=0,
            )
        )
    return steps


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------


class ServerProcess:
    """Launch and tear down one SGLang server. Always stops, even on error."""

    def __init__(
        self,
        launch: LaunchConfig,
        *,
        sglang_path: Path,
        port: int,
        log_path: Path,
        python_exe: str = sys.executable,
        trace_path: Path | None = None,
        env_extra: dict[str, str] | None = None,
        launcher_module: str = "sglang.launch_server",
    ) -> None:
        self.launch = launch
        self.sglang_path = Path(sglang_path)
        self.port = port
        self.log_path = Path(log_path)
        self.python_exe = python_exe
        self.trace_path = trace_path
        self.env_extra = dict(env_extra or {})
        self.launcher_module = launcher_module
        self._proc: subprocess.Popen | None = None
        self._log = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def _argv(self) -> list[str]:
        launch = LaunchConfig(
            id=self.launch.id,
            description=self.launch.description,
            model=self.launch.model,
            spec=self.launch.spec,
            port=self.port,
            host="127.0.0.1",
            enable_metrics=self.launch.enable_metrics,
        )
        return [self.python_exe, "-m", self.launcher_module, *launch.to_cli_args()]

    def start(self) -> ServerProcess:
        import os

        env = dict(os.environ)
        env.update(self.env_extra)
        if self.trace_path is not None:
            # Truncate so each step gets its own trace.
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)
            if self.trace_path.exists():
                self.trace_path.unlink()
            env["SGLANG_HETEROSPEC_TRACE"] = str(self.trace_path)

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = self.log_path.open("w")
        self._log.write("# " + " ".join(self._argv()) + "\n\n")
        self._log.flush()
        self._proc = subprocess.Popen(
            self._argv(),
            cwd=str(self.sglang_path),
            stdout=self._log,
            stderr=subprocess.STDOUT,
            env=env,
        )
        return self

    def wait_ready(self, *, timeout_s: float = 1800.0, poll_s: float = 5.0) -> bool:
        """Poll `/health` until the server responds or the timeout expires."""
        deadline = time.monotonic() + timeout_s
        with SGLangClient(self.base_url, timeout_s=10.0) as client:
            while time.monotonic() < deadline:
                if self._proc is not None and self._proc.poll() is not None:
                    raise RuntimeError(
                        f"server exited early with code {self._proc.returncode}; "
                        f"see {self.log_path}"
                    )
                if client.health():
                    return True
                time.sleep(poll_s)
        return False

    def stop(self, *, grace_s: float = 20.0) -> None:
        proc = self._proc
        if proc is None:
            return
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=grace_s)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=grace_s)
        finally:
            self._proc = None
            if self._log is not None:
                self._log.close()
                self._log = None

    def __enter__(self) -> ServerProcess:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()

    def tail_log(self, n: int = 40) -> str:
        try:
            lines = self.log_path.read_text(errors="replace").splitlines()
        except OSError:
            return ""
        return "\n".join(lines[-n:])


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


@dataclass
class StepResult:
    step: SessionStep
    ok: bool
    run_dir: str | None = None
    error: str | None = None
    wall_time_s: float = 0.0
    dispatch_wall_time_s: float = 0.0
    n_ok: int = 0
    n_failed: int = 0
    trace_records: int = 0
    trace_path: str | None = None
    """Recorded explicitly rather than globbed, so offline analysis cannot pick
    up the wrong trace after the trace file naming changed."""
    mean_accepted: float | None = None

    def to_dict(self, *, root: Path | None = None) -> dict[str, Any]:
        """Serialize, storing paths relative to `root` when they live under it.

        Relative paths are what make `--analyze-only` usable on a *copied* results
        directory. Session 1 runs on a rented host and its results are then pulled
        back to a laptop, so every absolute `/workspace/...` path in the report is
        dead on arrival. Relative paths rebase onto wherever the directory lands.
        Paths outside the root are kept absolute and are simply not portable.
        """
        d = asdict(self)
        d["step"] = asdict(self.step)
        if root is not None:
            d["run_dir"] = portable_path(self.run_dir, root)
            d["trace_path"] = portable_path(self.trace_path, root)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> StepResult:
        step = SessionStep(**d["step"])
        known = {f for f in cls.__dataclass_fields__ if f != "step"}
        return cls(step=step, **{k: v for k, v in d.items() if k in known})


def run_session(
    steps: Sequence[SessionStep],
    *,
    launches: dict[str, LaunchConfig],
    sglang_path: Path,
    results_root: Path,
    logs_dir: Path,
    port: int = 30000,
    python_exe: str = sys.executable,
    server_timeout_s: float = 1800.0,
    max_new_tokens: int | None = None,
    dispatch: str = "waves",
    on_event: Callable[[str], None] | None = None,
    collect_trace: bool = True,
    warmup_requests: int = 8,
    env_extra: dict[str, str] | None = None,
    launcher_module: str = "sglang.launch_server",
) -> list[StepResult]:
    """Execute a session plan, one server launch per policy.

    Steps sharing a policy share one server: launching an 8B model and capturing
    per-tier CUDA graphs takes minutes, so restarting per concurrency would waste
    most of the session. Steps are grouped by policy and run in plan order within
    each group.

    `env_extra` is forwarded to every server process. On a rented host this is
    needed for `HF_HOME` (so weights come off the persistent volume instead of
    being re-downloaded) and for `HF_TOKEN` on gated models.

    `launcher_module` exists so this orchestration can be exercised against a stub
    server in tests. On real hardware it is always SGLang's launcher.

    `warmup_requests` untimed requests are sent at each step's own concurrency
    before timing begins. Set it to 0 only if you have some other reason to
    believe the server is already at steady state.
    """
    results: list[StepResult] = []

    by_policy: dict[str, list[SessionStep]] = {}
    for step in steps:
        by_policy.setdefault(step.policy_id, []).append(step)

    def log(msg: str) -> None:
        if on_event:
            on_event(msg)

    for policy_id, group in by_policy.items():
        launch = launches.get(policy_id)
        if launch is None:
            # An unresolvable policy must fail its own steps, not abort the
            # session. Losing the whole paid run to one bad plan entry is the
            # expensive failure mode. (Found by a test: this lookup used to sit
            # outside any error handling.)
            available = sorted(launches)
            for step in group:
                results.append(
                    StepResult(
                        step=step,
                        ok=False,
                        error=(
                            f"no launch config for policy {policy_id!r}; "
                            f"available: {available}"
                        ),
                    )
                )
            log(f"\n=== server: {policy_id} -> SKIPPED (no launch config) ===")
            continue

        # ONE server per policy group. Steps in a group share a policy, and
        # relaunching an 8B model plus its per-tier CUDA graphs costs minutes each
        # time. Starting a server per STEP silently multiplied the session's
        # startup cost by the number of concurrencies -- 16 launches instead of 6
        # -- so the plan's time estimate was roughly half the truth. The slot for
        # the server must therefore be outside the step loop, not inside it.
        log(f"\n=== server: {policy_id} ({len(group)} runs) ===")

        # A policy has exactly one purpose, so tracing is a group-level decision.
        # Only the adaptive capture is traced; see the note in the README.
        traced = collect_trace and any(s.purpose == "adaptive" for s in group)
        trace_path = (results_root / f"trace_{policy_id}.jsonl") if traced else None

        server = ServerProcess(
            launch,
            sglang_path=sglang_path,
            port=port,
            log_path=logs_dir / f"server_{policy_id}.log",
            python_exe=python_exe,
            trace_path=trace_path,
            env_extra=env_extra,
            launcher_module=launcher_module,
        )

        try:
            server.start()
            if not server.wait_ready(timeout_s=server_timeout_s):
                raise TimeoutError(
                    f"server not ready within {server_timeout_s}s\n" + server.tail_log()
                )
        except Exception as e:  # noqa: BLE001
            # The server never came up: fail every step in this group, then let
            # the rest of the session proceed. Losing the whole paid session to
            # one bad group is the expensive failure mode.
            server.stop()
            for step in group:
                results.append(
                    StepResult(
                        step=step,
                        ok=False,
                        error=f"server launch failed: {type(e).__name__}: {e}",
                    )
                )
            log(f"  SERVER FAILED: {type(e).__name__}: {e}")
            continue

        log(f"  server ready: {policy_id} @ {server.base_url}")
        try:
            for step in group:
                log(f"  {step.describe()} -> running")
                t0 = time.perf_counter()
                try:
                    # Untimed warm-up at this step's own concurrency: /health-ready
                    # does not mean the kernels and allocator for these shapes are
                    # warm, and the first requests would otherwise be slower.
                    if warmup_requests > 0:
                        # Only the prompts and token budgets of this plan are used;
                        # `warmup_server` assigns each request its own rid. The plan
                        # exists so warm-up traffic has the same shape as the run.
                        specs = build_plan(
                            step.workload,
                            max(warmup_requests, step.concurrency),
                            seed=step.seed,
                            max_new_tokens=max_new_tokens,
                        )
                        w_ok, w_fail = warmup_server(
                            server.base_url,
                            specs,
                            concurrency=step.concurrency,
                            num_requests=max(warmup_requests, step.concurrency),
                            max_new_tokens=max_new_tokens,
                            timeout_s=min(server_timeout_s, 900.0),
                            # Steps in a group share a server, so warm-up rids must
                            # differ per step or one trace file holds several
                            # different requests under one name.
                            rid_prefix=f"warmup-{step.policy_id}-c{step.concurrency}",
                        )
                        log(f"  warm-up: {w_ok} ok, {w_fail} failed (discarded)")
                        if w_ok == 0:
                            raise RuntimeError(
                                "every warm-up request failed; the server is not "
                                "serving, so the timed run would be meaningless.\n"
                                + server.tail_log()
                            )
                    cfg = RunConfig(
                        workload=step.workload,
                        policy_id=f"{step.policy_id}_c{step.concurrency}",
                        base_url=server.base_url,
                        num_requests=step.num_requests,
                        concurrency=step.concurrency,
                        dispatch=dispatch,
                        seed=step.seed,
                        max_new_tokens=max_new_tokens,
                        results_root=results_root,
                        sglang_path=sglang_path,
                        launch=launch.to_dict(),
                        static_k=step.static_k,
                        progress=None,
                        notes=f"session1 {step.purpose}",
                    )
                    r = run_benchmark(cfg)
                    trace_records = 0
                    if trace_path is not None and trace_path.exists():
                        trace_records = sum(1 for _ in trace_path.open())
                    results.append(
                        StepResult(
                            step=step,
                            ok=True,
                            run_dir=str(r.run_dir) if r.run_dir else None,
                            wall_time_s=r.wall_time_s,
                            dispatch_wall_time_s=r.dispatch_wall_time_s,
                            n_ok=r.aggregate.get("n_ok", 0),
                            n_failed=r.aggregate.get("n_failed", 0),
                            trace_records=trace_records,
                            trace_path=str(trace_path) if trace_path else None,
                            mean_accepted=r.aggregate.get("mean_accepted_drafts"),
                        )
                    )
                    log(
                        f"  {step.describe()} -> ok "
                        f"({r.aggregate.get('n_ok', 0)} ok, "
                        f"{r.aggregate.get('n_failed', 0)} failed, "
                        f"{r.dispatch_wall_time_s:.1f}s)"
                    )
                except Exception as e:  # noqa: BLE001 - one bad step must not kill the group
                    results.append(
                        StepResult(
                            step=step,
                            ok=False,
                            error=f"{type(e).__name__}: {e}",
                            wall_time_s=time.perf_counter() - t0,
                        )
                    )
                    log(f"  {step.describe()} -> FAILED: {type(e).__name__}: {e}")
        finally:
            server.stop()

    return results


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def _run_records(run_dir: Path):
    from heterospec.records import load_request_records

    return load_request_records(run_dir / "requests.jsonl")


def cost_model_from_results(
    results: Sequence[StepResult], results_root: Path
) -> tuple[MeasuredCostModel, list[str]]:
    """Assemble the measured cost grid from successful calibration runs.

    Returns `(model, missing_cells)`. Missing cells are reported rather than
    silently interpolated: an incomplete grid means the cost surface has untested
    regions, and the operator can re-run just those steps.
    """
    points: list[CalibrationPoint] = []
    seen: set[tuple[int, int]] = set()
    missing: list[str] = []

    for r in results:
        if r.step.purpose != "calibration" or not r.ok or not r.run_dir:
            if r.step.purpose == "calibration" and not r.ok:
                missing.append(
                    f"K={r.step.static_k},n={r.step.concurrency} (run failed)"
                )
            continue
        run_dir = Path(r.run_dir)
        agg = read_json(run_dir / "aggregate.json")
        wall = agg.get("dispatch_wall_time_s") or agg.get("wall_time_s")
        records = _run_records(run_dir)
        failed = [x for x in records if not x.ok]
        if failed:
            # A cell with ANY failed request must not enter the cost surface.
            # Throughput is total_tokens / wall_time, and a failed request
            # contributes its elapsed time to the denominator while contributing
            # no tokens to the numerator -- so the cell would look slower than the
            # hardware is, and the bias would land arbitrarily across the grid.
            # Rejecting the cell is the only honest option: partially-correct
            # timing cannot be repaired after the fact.
            missing.append(
                f"K={r.step.static_k},n={r.step.concurrency} "
                f"(excluded: {len(failed)}/{len(records)} requests failed; their "
                f"time is in the denominator but their tokens are not)"
            )
            continue
        ok = records
        if not ok or not wall:
            missing.append(
                f"K={r.step.static_k},n={r.step.concurrency} (no usable data)"
            )
            continue
        points.append(
            CalibrationPoint(
                k=int(r.step.static_k or 0),
                batch_size=int(r.step.concurrency),
                wall_time_s=float(wall),
                total_completion_tokens=sum(int(x.completion_tokens or 0) for x in ok),
                total_request_rounds=sum(int(x.spec_verify_ct or 0) for x in ok),
                n_requests=len(ok),
                run_dir=str(run_dir),
                notes=f"{r.step.workload} seed={r.step.seed}",
            )
        )
        seen.add((int(r.step.static_k or 0), int(r.step.concurrency)))

    for r in results:
        if r.step.purpose == "calibration" and r.ok:
            key = (int(r.step.static_k or 0), int(r.step.concurrency))
            if key not in seen:
                missing.append(f"K={key[0]},n={key[1]} (unusable)")

    if not points:
        # Fail with the reasons attached: "no calibration points" alone gives an
        # operator nothing to act on when they are paying by the hour.
        detail = "; ".join(missing) if missing else "no calibration steps were run"
        raise ValueError(
            f"no usable calibration cells, so no cost model can be built. "
            f"Problems: {detail}"
        )

    return MeasuredCostModel.from_points(points), missing


def portable_path(path: str | None, root: Path) -> str | None:
    """Store `path` relative to `root` when it lives under it.

    A path outside the root is returned unchanged: it cannot be made portable, and
    inventing a relative path with `..` would just be a fragile absolute path in
    disguise.
    """
    if not path:
        return path
    try:
        return str(Path(path).resolve().relative_to(Path(root).resolve()))
    except ValueError:
        return path


def resolve_stored_path(
    path: str | None,
    root: Path,
    *,
    recorded_root: str | None = None,
) -> str | None:
    """Turn a path from a report into one valid under `root`.

    Three cases, in order:

    1. Relative path -- the portable format. Joined onto `root`.
    2. Absolute path that still exists -- left alone.
    3. Absolute path that is gone, but the report recorded the root it was
       written under -- rebased by its path relative to that old root. This is
       what rescues a report copied from a rented host before relative paths were
       in use, where the prefix (`/workspace/session1`) differs but the tail
       (`run_xyz`) is intact.
    """
    if not path:
        return path
    stored = Path(path)
    if not stored.is_absolute():
        return str(Path(root) / stored)
    if stored.exists() or not recorded_root:
        return str(stored)
    try:
        tail = stored.resolve().relative_to(Path(recorded_root).resolve())
    except ValueError:
        return str(stored)
    candidate = Path(root) / tail
    return str(candidate) if candidate.exists() else str(stored)


def load_session_results(results_root: Path) -> list[StepResult]:
    """Reconstruct a session's steps from `session1_report.json`.

    Enables `--analyze-only`: the report already records every step and its run
    directory, so offline re-analysis needs no GPU and no re-run.

    Paths are rebased onto `results_root`, so the directory can be copied off the
    rented host and analysed anywhere.
    """
    report_path = Path(results_root) / "session1_report.json"
    if not report_path.is_file():
        raise FileNotFoundError(
            f"no session report at {report_path}; run the session first, or point "
            f"--results-root at a directory that has one"
        )
    payload = read_json(report_path)
    raw_steps = payload.get("steps") or []
    if not raw_steps:
        raise FileNotFoundError(f"{report_path} records no steps")
    recorded_root = payload.get("results_root")
    out: list[StepResult] = []
    for d in raw_steps:
        r = StepResult.from_dict(d)
        r.run_dir = resolve_stored_path(
            r.run_dir, results_root, recorded_root=recorded_root
        )
        r.trace_path = resolve_stored_path(
            r.trace_path, results_root, recorded_root=recorded_root
        )
        out.append(r)
    return out


def oracle_gap_report(
    results: Sequence[StepResult],
    cost: MeasuredCostModel,
    *,
    invariance: Any | None = None,
    primary_k: int | None = None,
    candidates: Sequence[int] | None = None,
    n_bins: int = 20,
    train_fraction: float = 0.5,
    seed: int = 0,
) -> dict[str, Any]:
    """Compute the recoverable rectangular gap, gated on K-invariance.

    Two structural rules, both of which were wrong before and both of which would
    have produced a misleading headline:

    **1. The K-invariance check is a gate, not a diagnostic.** Every level here
    evaluates `E[acc | K]` at depths a request may never have run, which is only
    valid if per-position acceptance does not depend on `K`. If the check failed,
    or could not be assessed, no decision gap is computed or reported: the report
    comes back with ``status = "suppressed"`` and the reason. Returning a number
    in that state invites quoting it.

    **2. The headline uses only the deepest static capture per concurrency.**
    Averaging gaps across `K = 1, 3, 5, 7` captures biases the result downward,
    because a capture taken at `K` can only ever choose from `0..K` -- a `K=1`
    capture is scored on a two-point candidate set and cannot express the gap the
    project is looking for. The deepest static capture at each concurrency has the
    full ladder available and is the controlled primary measurement. Adaptive
    captures are reported separately as a diagnostic, never averaged into the
    headline, because they are traced and their throughput is not comparable.
    """
    from heterospec.analysis.oracle import analyse_batches
    from heterospec.analysis.traces import (
        batches_from_iterations,
        batches_from_waves,
        describe_skips,
    )

    report: dict[str, Any] = {
        "status": "ok",
        "cost_model": cost.coverage(),
        "captures": [],
        "diagnostic": [],
        "excluded": [],
        "warnings": [],
    }

    # ---- gate ---------------------------------------------------------------
    if invariance is None:
        report["status"] = "suppressed"
        report["suppressed_reason"] = (
            "K-invariance was not assessed, so the oracle's central assumption is "
            "untested and any gap would be uninterpretable."
        )
        return report
    if not getattr(invariance, "passed", False):
        report["status"] = "suppressed"
        report["suppressed_reason"] = "K-invariance did not pass: " + str(
            invariance.verdict()
        )
        report["k_invariance"] = (
            invariance.summary() if hasattr(invariance, "summary") else None
        )
        return report
    report["k_invariance"] = (
        invariance.summary() if hasattr(invariance, "summary") else None
    )

    # ---- which static K is the primary? ------------------------------------
    static_ks = sorted(
        {
            int(r.step.static_k)
            for r in results
            if r.ok and r.step.purpose == "calibration" and r.step.static_k is not None
        }
    )
    if primary_k is None:
        if not static_ks:
            report["status"] = "suppressed"
            report["suppressed_reason"] = "no usable static calibration captures"
            return report
        primary_k = static_ks[-1]
    report["primary_k"] = primary_k
    report["available_static_ks"] = static_ks

    # ---- candidate depths: the measured grid only ---------------------------
    #
    # An oracle that is allowed to pick any integer depth will happily report a
    # gap at K=2, 4 or 6 -- depths whose cost was never measured and which the
    # cost surface can only reach by interpolation -- or at K=0, which the surface
    # reaches by *clamping* to K=1 (no-spec has no verify round at all, so a
    # clamped query is not even the right shape, let alone the right number). A
    # go/no-go decision must not rest on a number the session never measured, so
    # the action set is exactly the grid the calibration cells define.
    #
    # No-spec (K=0) is deliberately NOT an oracle action: it is a reference point.
    # It is reported as such in `no_spec_reference`, and it never enters the cost
    # model's K axis (`cost_model_from_results` only accepts `purpose ==
    # "calibration"`, and the plan files no-spec under its own purpose).
    grid = sorted({int(k) for k in cost.k_values})
    if not candidates:
        candidates = [k for k in grid if k > 0]
    else:
        candidates = sorted({int(k) for k in candidates})
        # An explicit action set is still checked against the grid: silently
        # accepting an unmeasured K here would reintroduce exactly the bug.
        off_grid = [k for k in candidates if k not in grid]
        if off_grid:
            raise ValueError(
                f"candidates {off_grid} are not in the measured cost grid {grid}; "
                f"their cost would be interpolated or clamped, and a go/no-go "
                f"decision must rest only on measured cells"
            )
    if not candidates:
        report["status"] = "suppressed"
        report["suppressed_reason"] = (
            f"the measured cost grid {grid} contains no positive K, so there is no "
            f"measured action for the oracle to choose"
        )
        return report
    report["candidate_ks"] = list(candidates)
    report["candidate_source"] = "measured cost grid"
    report["cost_grid_ks"] = grid
    report["no_spec_reference"] = {
        "is_oracle_action": False,
        "in_cost_grid": 0 in grid,
        "note": (
            "no-spec (K=0) is a reference point, not an oracle action: it has no "
            "verify round, so its cost is not on the K axis and was not measured. "
            "Interpolating or clamping it would fabricate a number."
        ),
    }

    def _analyse(entry: dict[str, Any], r: StepResult) -> None:
        run_dir = Path(r.run_dir)
        records = _run_records(run_dir)
        try:
            if r.step.purpose == "calibration":
                batches, skips = batches_from_waves(
                    records,
                    concurrency=r.step.concurrency,
                    k_static=int(r.step.static_k),
                )
            else:
                trace = Path(r.trace_path) if r.trace_path else None
                if trace is None or not trace.exists():
                    entry["error"] = (
                        "no iteration trace recorded for this capture "
                        "(trace_path absent or missing on disk)"
                    )
                    return
                from heterospec.records import load_iteration_records

                batches, skips = batches_from_iterations(
                    records, load_iteration_records(trace)
                )

            entry["n_batches"] = len(batches)
            entry["skips"] = describe_skips(skips)
            if len(batches) < 4:
                entry["error"] = (
                    f"only {len(batches)} usable batches; need at least 4 for a "
                    f"train/test split"
                )
                return
            # Clamp accounting is per capture: a clamped cost query means the
            # capture asked about a cell the session never measured, which would
            # put an unmeasured number into a go/no-go decision.
            before_clamp = getattr(cost, "clamp_count", 0)
            res = analyse_batches(
                batches,
                cost,
                candidates=candidates,
                n_bins=n_bins,
                train_fraction=train_fraction,
                seed=seed,
            )
            clamped = getattr(cost, "clamp_count", 0) - before_clamp
            entry["clamped_cost_queries"] = clamped
            if clamped:
                report["warnings"].append(
                    f"{entry['policy_id']} c{entry['concurrency']}: {clamped} cost "
                    f"queries fell outside the measured grid and were clamped; the "
                    f"gap for this capture refers to an unmeasured cost"
                )
            entry["result"] = res.summary()
            entry["recoverable_rectangular_gap"] = res.recoverable_rectangular_gap
        except Exception as e:  # noqa: BLE001 - report, don't abort the analysis
            entry["error"] = f"{type(e).__name__}: {e}"

    for r in results:
        if not r.ok or not r.run_dir:
            continue
        entry: dict[str, Any] = {
            "policy_id": r.step.policy_id,
            "purpose": r.step.purpose,
            "concurrency": r.step.concurrency,
            "static_k": r.step.static_k,
            "run_dir": str(r.run_dir),
        }
        if r.step.purpose not in ("calibration", "adaptive"):
            continue

        if r.step.purpose == "adaptive":
            # Diagnostic only. Never averaged into the headline: the capture is
            # traced, so its throughput is not comparable with the untraced grid.
            _analyse(entry, r)
            report["diagnostic"].append(entry)
            continue

        if int(r.step.static_k or -1) != primary_k:
            entry["excluded_reason"] = (
                f"capture taken at K={r.step.static_k}; only the deepest static "
                f"capture (K={primary_k}) has the full candidate ladder, so a "
                f"shallower one cannot express the gap and would bias the mean "
                f"downward"
            )
            report["excluded"].append(entry)
            continue

        _analyse(entry, r)
        report["captures"].append(entry)

    gaps = [
        c["recoverable_rectangular_gap"]
        for c in report["captures"]
        if "recoverable_rectangular_gap" in c
    ]
    if gaps:
        report["gap_summary"] = {
            "primary_k": primary_k,
            "n_captures": len(gaps),
            "concurrencies": [
                c["concurrency"]
                for c in report["captures"]
                if "recoverable_rectangular_gap" in c
            ],
            "min": min(gaps),
            "max": max(gaps),
            "mean": sum(gaps) / len(gaps),
            "note": (
                "primary controlled measurement: static captures at the deepest K, "
                "one per concurrency. Not averaged across K, and excludes the "
                "traced adaptive diagnostic."
            ),
        }
    else:
        report["warnings"].append(
            f"no primary (K={primary_k}) capture produced a gap; see per-capture errors"
        )
    return report


def write_session_report(
    results: Sequence[StepResult],
    *,
    results_root: Path,
    cost: MeasuredCostModel | None = None,
    missing_cells: Sequence[str] = (),
    gap_report: dict[str, Any] | None = None,
    invariance: dict[str, Any] | None = None,
) -> Path:
    payload = {
        # Recorded so `--analyze-only` can rebase paths that were written before
        # relative serialization, when the report is copied to another machine.
        "results_root": str(Path(results_root).resolve()),
        "steps": [r.to_dict(root=Path(results_root)) for r in results],
        "n_ok": sum(r.ok for r in results),
        "n_failed": sum(not r.ok for r in results),
        "total_dispatch_wall_time_s": sum(r.dispatch_wall_time_s for r in results),
        "missing_cost_cells": list(missing_cells),
    }
    if cost is not None:
        payload["cost_model"] = cost.coverage()
    if gap_report is not None:
        payload["gap"] = gap_report
    if invariance is not None:
        payload["k_invariance"] = invariance
    path = Path(results_root) / "session1_report.json"
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


#: Union of the built-in adaptive ladder ``[0,1,3,5,7]``. A tier is one
#: ``SpecRuntimeState``: its own attention backends and its own CUDA graphs, so
#: startup cost scales with the number of tiers, not with a flat constant.
DEFAULT_ALLOCATION_TIERS = 5


def _tier_count(launch: LaunchConfig) -> int:
    """How many runtime states a policy must build at startup."""
    spec = launch.spec
    if spec is None or not spec.adaptive:
        return 1
    if spec.adaptive_config:
        try:
            from heterospec.config import load_adaptive_config, resolve_candidate_steps

            return max(
                1,
                len(
                    resolve_candidate_steps(load_adaptive_config(spec.adaptive_config))
                ),
            )
        except (OSError, ValueError, KeyError):
            return DEFAULT_ALLOCATION_TIERS
    return DEFAULT_ALLOCATION_TIERS


def estimate_session_minutes(
    steps: Sequence[SessionStep],
    launches: dict[str, LaunchConfig] | None = None,
    *,
    per_run_s: float = 150.0,
    model_load_s: float = 120.0,
    per_tier_s: float = 90.0,
    warmup_s: float = 20.0,
) -> dict[str, Any]:
    """Nominal and pessimistic GPU-minute estimates for a plan.

    Startup is modelled as **model load plus a per-tier cost**, because each
    adaptive tier carries its own CUDA graphs and attention backends: the
    adaptive server has to build several, so it starts markedly slower than a
    static-K server. A flat startup constant hides that and produced an
    over-optimistic total.

    All coefficients are order-of-magnitude placeholders. The value of this
    function is the *range*, not the point estimate: budget against the
    pessimistic figure and replace the coefficients with observed timings after
    the first session.
    """
    by_policy: dict[str, list[SessionStep]] = {}
    for s in steps:
        by_policy.setdefault(s.policy_id, []).append(s)

    nominal = 0.0
    pessimistic = 0.0
    breaker: list[dict[str, Any]] = []
    for pid, group in by_policy.items():
        tiers = _tier_count(launches[pid]) if launches and pid in launches else 1
        startup = model_load_s + per_tier_s * tiers
        runs = len(group) * (per_run_s + warmup_s)
        nominal += startup + runs
        # 1.6x covers slower-than-expected graph capture, downloads on a cold
        # volume, retries and wave-barrier tail dead time.
        pessimistic += startup * 1.6 + runs * 1.6
        breaker.append(
            {
                "policy": pid,
                "tiers": tiers,
                "runs": len(group),
                "startup_s": round(startup, 1),
            }
        )

    return {
        "n_runs": len(steps),
        "n_servers": len(by_policy),
        "nominal_minutes": nominal / 60.0,
        "pessimistic_minutes": pessimistic / 60.0,
        "assumptions": {
            "per_run_s": per_run_s,
            "model_load_s": model_load_s,
            "per_tier_s": per_tier_s,
            "warmup_s": warmup_s,
            "pessimistic_multiplier": 1.6,
        },
        "per_policy": breaker,
    }


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    from heterospec.config import load_launch_configs as _load

    p = argparse.ArgumentParser(
        prog="python -m heterospec.session",
        description="Plan, run, or analyse GPU Session 1.",
    )
    p.add_argument("--launch-config", type=Path, required=True)
    p.add_argument("--results-root", type=Path, default=Path("results/session1"))
    p.add_argument("--logs-dir", type=Path, default=Path("results/session1/logs"))
    p.add_argument("--sglang-path", type=Path, default=Path("../sglang"))
    p.add_argument("--workload", default="mixed_50_50")
    p.add_argument(
        "--waves-per-step",
        type=int,
        default=24,
        help="waves per calibration step; requests = waves * concurrency, which "
        "keeps every step's wall time comparable",
    )
    p.add_argument(
        "--num-requests",
        type=int,
        default=None,
        help="override: a flat request count for every step (usually worse)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--concurrencies", type=int, nargs="+", default=[1, 8, 32])
    p.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5, 7])
    p.add_argument("--port", type=int, default=30000)
    p.add_argument("--max-new-tokens", type=int, default=None)
    p.add_argument("--dispatch", choices=("waves", "continuous"), default="waves")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--no-nospec", action="store_true")
    p.add_argument("--no-adaptive", action="store_true")
    p.add_argument("--plan", action="store_true", help="print the plan and exit")
    p.add_argument("--analyze-only", action="store_true")
    args = p.parse_args(argv)

    launches = {c.id: c for c in _load(args.launch_config)}
    for c in launches.values():
        c.validate()

    steps = build_session_plan(
        launches,
        ks=args.ks,
        concurrencies=args.concurrencies,
        workload=args.workload,
        waves_per_step=args.waves_per_step,
        num_requests=args.num_requests,
        seed=args.seed,
        include_nospec=not args.no_nospec,
        include_adaptive=not args.no_adaptive,
    )

    if args.plan or args.analyze_only:
        print(
            f"Session 1 plan: {len(steps)} runs, "
            f"{len({s.policy_id for s in steps})} server launches"
        )
        for s in steps:
            print(f"  {s.describe()}")
        est = estimate_session_minutes(steps, launches)
        print(
            f"\nGPU-time estimate: ~{est['nominal_minutes']:.0f} min nominal, "
            f"~{est['pessimistic_minutes']:.0f} min pessimistic "
            f"(placeholder coefficients; budget against the pessimistic figure)"
        )
        for row in est["per_policy"]:
            print(
                f"    {row['policy']:<18} tiers={row['tiers']:<2} "
                f"runs={row['runs']:<2} startup~{row['startup_s']:.0f}s"
            )
        if args.plan:
            return 0

    results: list[StepResult] = []
    if args.analyze_only:
        # Reconstruct the run from disk. Without this, --analyze-only passed an
        # empty list into the analysis, which raised and printed "analysis
        # failed" -- so the runbook's offline re-analysis instructions did not
        # work, which is exactly when you least want a broken command.
        try:
            results = load_session_results(args.results_root)
        except FileNotFoundError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        print(f"re-analysing {len(results)} steps from {args.results_root}")
    else:
        results = run_session(
            steps,
            launches=launches,
            sglang_path=args.sglang_path,
            results_root=args.results_root,
            logs_dir=args.logs_dir,
            port=args.port,
            python_exe=args.python,
            max_new_tokens=args.max_new_tokens,
            dispatch=args.dispatch,
            on_event=print,
        )

    cost = None
    missing: list[str] = []
    gap = None
    invariance = None

    # ORDER MATTERS. The invariance check runs first because it gates the gap:
    # every oracle level evaluates E[acc | K] at depths a request may never have
    # run, so a gap computed under a failed invariance assumption is a number
    # nobody should quote.
    try:
        invariance = k_invariance_from_results(results)
    except Exception as e:  # noqa: BLE001
        print(f"K-invariance check failed to run: {type(e).__name__}: {e}")

    try:
        cost, missing = cost_model_from_results(results, args.results_root)
    except Exception as e:  # noqa: BLE001
        print(f"cost model failed: {type(e).__name__}: {e}")

    if cost is not None:
        try:
            gap = oracle_gap_report(results, cost, invariance=invariance)
        except Exception as e:  # noqa: BLE001
            print(f"oracle gap failed: {type(e).__name__}: {e}")

    report = write_session_report(
        results,
        results_root=args.results_root,
        cost=cost,
        missing_cells=missing,
        gap_report=gap,
        invariance=(invariance.summary() if invariance is not None else None),
    )
    print(f"\nreport: {report}")
    if missing:
        print(f"WARNING: {len(missing)} cost-grid cells missing: {missing}")
    if invariance is not None:
        print(f"K-invariance: {invariance.verdict()}")
    elif not args.analyze_only:
        print(
            "K-invariance: NOT ASSESSED (fewer than two static depths usable). "
            "The oracle's central assumption is therefore untested."
        )
    if gap and gap.get("status") == "suppressed":
        print(f"DECISION GAP SUPPRESSED: {gap.get('suppressed_reason')}")
    elif gap and "gap_summary" in gap:
        g = gap["gap_summary"]
        print(
            f"PRIMARY gap (static K={g['primary_k']}, "
            f"concurrencies {g['concurrencies']}): "
            f"mean {g['mean'] * 100:+.2f}% "
            f"(min {g['min'] * 100:+.2f}%, max {g['max'] * 100:+.2f}%)"
        )
        if gap.get("excluded"):
            print(
                f"  ({len(gap['excluded'])} shallower-K captures excluded from the "
                f"primary; see report)"
            )
        if gap.get("diagnostic"):
            got = [
                d["recoverable_rectangular_gap"]
                for d in gap["diagnostic"]
                if "recoverable_rectangular_gap" in d
            ]
            if got:
                print(
                    f"  diagnostic (traced adaptive, NOT the headline): "
                    f"{[f'{v * 100:+.2f}%' for v in got]}"
                )
    return 0


def k_invariance_from_results(
    results: Sequence[StepResult],
    *,
    concurrency: int | None = None,
    tolerance: float = 0.05,
) -> Any | None:
    """Run the K-invariance check on the calibration captures.

    Only captures at **one concurrency** are compared, because batch size itself
    affects acceptance behaviour; mixing batch sizes would confound depth with
    batching. Defaults to the largest concurrency available, which has the most
    requests and therefore the least sampling noise.

    Returns `None` when fewer than two static depths are available, which is
    itself reported rather than treated as a pass.
    """
    from heterospec.analysis.invariance import k_invariance_check
    from heterospec.records import load_request_records

    available = sorted(
        {
            r.step.concurrency
            for r in results
            if r.step.purpose == "calibration" and r.ok and r.run_dir
        }
    )
    if not available:
        return None
    conc = concurrency if concurrency is not None else available[-1]

    by_k: dict[int, list] = {}
    for r in results:
        if (
            r.step.purpose == "calibration"
            and r.ok
            and r.run_dir
            and r.step.concurrency == conc
            and r.step.static_k is not None
        ):
            by_k[int(r.step.static_k)] = load_request_records(
                Path(r.run_dir) / "requests.jsonl"
            )
    if len(by_k) < 2:
        return None
    return k_invariance_check(by_k, tolerance=tolerance)


if __name__ == "__main__":
    # MUST stay last: main() calls k_invariance_from_results, which is defined
    # above this line. With the guard in the middle of the module, `python -m
    # heterospec.session` raised NameError before the function was ever bound --
    # and import-based tests could not see it, because an import never executes
    # the guard.
    raise SystemExit(main())
