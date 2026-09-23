"""Benchmark orchestration: plan -> concurrent requests -> structured results.

Two dispatch modes, because they answer different questions:

``waves``
    Submit exactly ``concurrency`` requests, wait for all to finish, repeat.
    Every wave is a *controlled* mixture, so a batch's composition is known
    exactly. This is the mode the oracle study wants, because "the best K for
    this batch" is only well-defined if the batch is well-defined.

``continuous``
    Keep a sliding window of ``concurrency`` requests in flight. Realistic
    serving behaviour, and the mode that tests whether an effect survives
    real arrival dynamics -- but batch composition is emergent rather than
    designed.

Correctness detail that matters for the whole project: for a **static-K** run the
runner stamps ``inferred_k_static`` on every record, which is what makes
``position_acceptance`` unbiased. For an **adaptive** run it deliberately leaves
K unknown, so the analysis refuses to report position acceptance rather than
reporting a biased curve. The iteration-level trace is what unlocks the adaptive
case.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from heterospec.client import SGLangClient
from heterospec.envinfo import collect_environment
from heterospec.records import (
    RequestRecord,
    RunMetadata,
    make_run_dir,
    summarise_records,
    write_json,
    write_jsonl,
)
from heterospec.workloads import (
    RequestSpec,
    WorkloadSpec,
    build_plan,
    get_workload,
    summarise_plan,
)

__all__ = ["RunConfig", "RunResult", "run_benchmark"]


@dataclass
class RunConfig:
    """Everything needed to execute one benchmark run."""

    workload: str | WorkloadSpec
    policy_id: str
    base_url: str = "http://127.0.0.1:30000"

    num_requests: int = 200
    concurrency: int = 8
    dispatch: str = "waves"  # "waves" | "continuous"
    seed: int = 0

    max_new_tokens: int | None = None
    temperature: float = 0.0
    timeout_s: float = 900.0

    results_root: Path = Path("results")
    sglang_path: Path | None = None
    launch: dict[str, Any] = field(default_factory=dict)
    static_k: int | None = None
    """K for a static run; stamped onto records. None for adaptive/no-spec."""

    notes: str = ""
    dry_run: bool = False
    write_results: bool = True
    progress_every: int = 25
    progress: Callable[[str], None] | None = None

    def __post_init__(self) -> None:
        if self.num_requests <= 0:
            raise ValueError("num_requests must be > 0")
        if self.concurrency <= 0:
            raise ValueError("concurrency must be > 0")
        if self.dispatch not in ("waves", "continuous"):
            raise ValueError(
                f"dispatch must be 'waves' or 'continuous', got {self.dispatch!r}"
            )


@dataclass
class RunResult:
    """Outcome of a run: records, aggregate, and where they were written."""

    records: list[RequestRecord]
    aggregate: dict[str, Any]
    metadata: RunMetadata
    run_dir: Path | None = None
    wall_time_s: float = 0.0
    plan_summary: dict[str, Any] = field(default_factory=dict)

    @property
    def citable(self) -> tuple[bool, str]:
        return self.metadata.citable()


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


class _Dispatcher:
    """Thread-local clients so each worker has its own connection pool."""

    def __init__(self, cfg: RunConfig) -> None:
        self.cfg = cfg
        self._local = threading.local()
        self._clients: list[SGLangClient] = []
        self._lock = threading.Lock()
        self._done = 0
        self._ok = 0

    def _client(self) -> SGLangClient:
        c = getattr(self._local, "client", None)
        if c is None:
            c = SGLangClient(self.cfg.base_url, timeout_s=self.cfg.timeout_s)
            self._local.client = c
            with self._lock:
                self._clients.append(c)
        return c

    def close(self) -> None:
        with self._lock:
            for c in self._clients:
                c.close()
            self._clients.clear()

    def send(self, spec: RequestSpec) -> RequestRecord:
        c = self._client()
        res = c.generate(
            spec.prompt,
            max_new_tokens=spec.max_new_tokens,
            temperature=self.cfg.temperature,
            seed=spec.seed,
            rid=spec.rid,
        )
        rec = RequestRecord(
            rid=spec.rid,
            index=spec.index,
            workload=spec.workload,
            prompt_class=spec.prompt_class,
            phase_label=spec.phase_label,
            prompt=spec.prompt,
            max_new_tokens=spec.max_new_tokens,
            seed=spec.seed,
            ok=res.ok,
            error=res.error,
            latency_s=res.latency_s,
            started_at=res.started_at,
            finished_at=res.finished_at,
            completion_tokens=res.completion_tokens,
            spec_verify_ct=res.telemetry.spec_verify_ct,
            spec_num_correct_drafts=res.telemetry.spec_num_correct_drafts,
            spec_correct_drafts_histogram=res.telemetry.spec_correct_drafts_histogram,
            # Static runs know their K exactly; adaptive runs do not, and must
            # not pretend to. See module docstring.
            inferred_k_static=self.cfg.static_k,
        )
        with self._lock:
            self._done += 1
            self._ok += int(rec.ok)
            n, ok = self._done, self._ok
        if self.cfg.progress and (
            n % self.cfg.progress_every == 0 or n == self.cfg.num_requests
        ):
            self.cfg.progress(f"  {n}/{self.cfg.num_requests} sent ({ok} ok)")
        return rec


def _run_waves(
    specs: list[RequestSpec], cfg: RunConfig, d: _Dispatcher
) -> list[RequestRecord]:
    out: list[RequestRecord] = []
    for start in range(0, len(specs), cfg.concurrency):
        wave = specs[start : start + cfg.concurrency]
        with ThreadPoolExecutor(max_workers=len(wave)) as pool:
            futures = [pool.submit(d.send, s) for s in wave]
            for f in as_completed(futures):
                out.append(f.result())
    return out


def _run_continuous(
    specs: list[RequestSpec], cfg: RunConfig, d: _Dispatcher
) -> list[RequestRecord]:
    out: list[RequestRecord] = []
    with ThreadPoolExecutor(max_workers=cfg.concurrency) as pool:
        futures = [pool.submit(d.send, s) for s in specs]
        for f in as_completed(futures):
            out.append(f.result())
    return out


# ---------------------------------------------------------------------------
# Server introspection
# ---------------------------------------------------------------------------


def _probe_server(cfg: RunConfig) -> dict[str, Any]:
    """Best-effort server description, including mock detection."""
    info: dict[str, Any] = {"base_url": cfg.base_url, "reachable": False}
    with SGLangClient(cfg.base_url, timeout_s=min(cfg.timeout_s, 30.0)) as c:
        if not c.health():
            return info
        info["reachable"] = True
        try:
            si = c.server_info()
            info["server_info"] = si
            if si.get("mock"):
                info["mode"] = "mock"
            state = (si.get("internal_states") or [{}])[0]
            for key in ("speculative_num_steps", "avg_spec_accept_length"):
                if key in state:
                    info[key] = state[key]
        except Exception as e:  # noqa: BLE001 - introspection is best-effort
            info["server_info_error"] = f"{type(e).__name__}: {e}"
    return info


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_benchmark(cfg: RunConfig) -> RunResult:
    """Execute one run and persist structured results."""
    started = time.perf_counter()
    spec = get_workload(cfg.workload) if isinstance(cfg.workload, str) else cfg.workload

    plan = build_plan(
        spec,
        cfg.num_requests,
        seed=cfg.seed,
        max_new_tokens=cfg.max_new_tokens,
        # Prefix rids with the policy so a trace file's rids identify their run.
        rid_prefix=f"{cfg.policy_id[:6]}",
    )
    plan_sum = summarise_plan(plan).to_dict()

    if cfg.progress:
        cfg.progress(
            f"plan: {plan_sum['total']} requests, classes={plan_sum['by_class']}, "
            f"phases={plan_sum['by_phase']}, "
            f"max same-class run={plan_sum['max_run_of_single_class']}"
        )

    server = _probe_server(cfg)
    env = collect_environment(cfg.sglang_path)

    metadata = RunMetadata.new(
        run_id="",
        policy_id=cfg.policy_id,
        workload_name=spec.name,
        launch=cfg.launch,
        workload={
            "name": spec.name,
            "description": spec.description,
            "num_requests": len(plan),
            "concurrency": cfg.concurrency,
            "dispatch": cfg.dispatch,
            "seed": cfg.seed,
            "temperature": cfg.temperature,
            "max_new_tokens": cfg.max_new_tokens or spec.max_new_tokens,
            "plan_summary": plan_sum,
            "notes": cfg.notes or spec.notes,
        },
        environment=env,
        server=server,
        notes=cfg.notes,
    )

    if cfg.dry_run:
        if cfg.progress:
            cfg.progress("dry run: no requests sent")
        return RunResult(
            records=[],
            aggregate={},
            metadata=metadata,
            wall_time_s=time.perf_counter() - started,
            plan_summary=plan_sum,
        )

    dispatcher = _Dispatcher(cfg)
    try:
        if cfg.dispatch == "waves":
            records = _run_waves(plan, cfg, dispatcher)
        else:
            records = _run_continuous(plan, cfg, dispatcher)
    finally:
        dispatcher.close()

    records.sort(key=lambda r: r.index)
    aggregate = summarise_records(records)
    aggregate["workload"] = spec.name
    aggregate["policy_id"] = cfg.policy_id
    aggregate["dispatch"] = cfg.dispatch
    aggregate["concurrency"] = cfg.concurrency
    aggregate["static_k"] = cfg.static_k
    aggregate["wall_time_s"] = time.perf_counter() - started

    run_dir: Path | None = None
    if cfg.write_results:
        run_dir = make_run_dir(
            cfg.results_root,
            workload=spec.name,
            policy=cfg.policy_id,
            commit=metadata.sglang_commit,
        )
        metadata.run_id = run_dir.name
        write_json(run_dir / "metadata.json", metadata.to_dict())
        write_jsonl(run_dir / "requests.jsonl", records)
        write_json(run_dir / "aggregate.json", aggregate)

    return RunResult(
        records=records,
        aggregate=aggregate,
        metadata=metadata,
        run_dir=run_dir,
        wall_time_s=time.perf_counter() - started,
        plan_summary=plan_sum,
    )
