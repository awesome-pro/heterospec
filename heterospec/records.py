"""The result schema shared by the harness, the analysis, and the simulator.

One dated directory per serious run (see results/README.md):

    results/<date>_<workload>_<policy>_<shortsha>/
        metadata.json
        requests.jsonl
        speculative_steps.jsonl   (only with the iteration-level trace patch)
        aggregate.json

`RequestRecord` is deliberately permissive about missing telemetry: on a no-spec
baseline there is no `spec_correct_drafts_histogram` at all, and a run should
record that fact rather than crash or invent zeros. Every acceptance accessor
returns `None`/empty when the data is absent, and `histogram_consistent()`
reports the free integrity check from `telemetry.rounds_consistency`.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass, field, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from heterospec import SGLANG_BASE_COMMIT, __version__
from heterospec.telemetry import (
    AcceptStats,
    PositionAcceptance,
    position_acceptance,
    rounds_consistency,
)

__all__ = [
    "RequestRecord",
    "RunMetadata",
    "IterationRecord",
    "make_run_dir",
    "read_json",
    "read_jsonl",
    "write_json",
    "write_jsonl",
]


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Per-request record
# ---------------------------------------------------------------------------


@dataclass
class RequestRecord:
    """One completed (or failed) request."""

    # -- identity / provenance --
    rid: str
    index: int = -1
    workload: str = ""
    prompt_class: str = ""
    phase_label: str = ""
    prompt: str = ""
    prompt_sha256: str = ""
    max_new_tokens: int = 0
    seed: int = 0

    # -- outcome --
    ok: bool = True
    error: str | None = None
    latency_s: float | None = None
    started_at: float | None = None
    finished_at: float | None = None

    # -- SGLang meta_info (absent on non-speculative runs) --
    completion_tokens: int | None = None
    spec_verify_ct: int | None = None
    spec_num_correct_drafts: int | None = None
    spec_correct_drafts_histogram: list[int] | None = None

    # -- active-K observation, if the iteration trace was enabled --
    mean_active_k: float | None = None
    inferred_k_static: int | None = None

    def __post_init__(self) -> None:
        if self.prompt and not self.prompt_sha256:
            self.prompt_sha256 = sha256_text(self.prompt)

    # -- acceptance accessors ------------------------------------------------

    @property
    def has_histogram(self) -> bool:
        return bool(self.spec_correct_drafts_histogram)

    @property
    def is_speculative(self) -> bool:
        return self.spec_verify_ct is not None and self.spec_verify_ct > 0

    def accept_stats(self) -> AcceptStats | None:
        if not self.has_histogram:
            return None
        return AcceptStats.from_histogram(self.spec_correct_drafts_histogram or [])

    def mean_accepted_drafts(self) -> float | None:
        st = self.accept_stats()
        return None if st is None else st.mean

    def position_acceptance(
        self, *, k_static: int | None = None, min_k_proposed: int | None = None
    ) -> PositionAcceptance | None:
        """Position acceptance, honouring the K-confounding rule.

        `k_static` takes precedence; when neither argument is supplied the result
        is marked confounded rather than silently returning biased numbers.
        """
        if not self.has_histogram:
            return None
        if k_static is None:
            k_static = self.inferred_k_static
        return position_acceptance(
            self.spec_correct_drafts_histogram or [],
            k_static=k_static,
            min_k_proposed=min_k_proposed,
        )

    def histogram_consistent(self) -> tuple[bool, str]:
        """Free integrity check: `sum(histogram) == spec_verify_ct`."""
        if not self.has_histogram:
            return True, "no histogram (non-speculative or missing telemetry)"
        return rounds_consistency(
            self.spec_correct_drafts_histogram or [], self.spec_verify_ct
        )

    def acceptance_length(self) -> float | None:
        """Completion tokens per verify round -- includes the bonus token.

        Distinct from `mean_accepted_drafts`, which excludes it. Recorded
        separately because the two are easy to confuse when comparing against
        SGLang's own `avg_spec_accept_length` metric.
        """
        if not self.is_speculative or self.completion_tokens is None:
            return None
        assert self.spec_verify_ct  # narrowed by is_speculative
        return self.completion_tokens / self.spec_verify_ct

    def effective_k(self) -> float | None:
        """Draft depth this request actually ran at, when known.

        Prefers the exact value from a static-K capture, then the iteration
        trace's mean. `None` when unknown, which is the honest answer for an
        adaptive run without the trace patch -- DraftWaste is undefined then,
        and callers must not substitute a guess.
        """
        if self.inferred_k_static is not None:
            return float(self.inferred_k_static)
        return self.mean_active_k

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RequestRecord:
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


# ---------------------------------------------------------------------------
# Iteration-level record (requires the fork trace patch)
# ---------------------------------------------------------------------------


@dataclass
class IterationRecord:
    """One decode iteration's speculative state.

    This is the record UPDATE.md identifies as the missing piece: it preserves
    `(request identity, accepted drafts, batch composition, active K)`, which the
    per-request histogram alone cannot reconstruct.
    """

    iteration: int
    batch_size: int
    active_k: int
    requests: list[dict[str, Any]] = field(default_factory=list)
    # Server-side wall clock, if the patch supplies one.
    ts: float | None = None

    @property
    def rids(self) -> list[str]:
        return [r.get("rid") for r in self.requests]

    @property
    def accepted(self) -> list[int]:
        return [r.get("accepted", 0) for r in self.requests]

    def accepted_for(self, rid: str) -> int | None:
        for r in self.requests:
            if r.get("rid") == rid:
                return r.get("accepted")
        return None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> IterationRecord:
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


def min_k_proposed_per_request(
    iterations: Iterable[IterationRecord],
) -> dict[str, int]:
    """Smallest number of drafts actually proposed to each request.

    Only the *proposed* count matters for the confounding rule: a request that
    was present in a batch whose active K was 2 had at most 2 drafts proposed,
    regardless of how many it accepted. Feeding this into
    `position_acceptance(min_k_proposed=...)` is what makes an adaptive capture
    interpretable.
    """
    out: dict[str, int] = {}
    for it in iterations:
        for rid in it.rids:
            if rid is None:
                continue
            prev = out.get(rid)
            k = int(it.active_k)
            out[rid] = k if prev is None else min(prev, k)
    return out


# ---------------------------------------------------------------------------
# Run metadata
# ---------------------------------------------------------------------------


@dataclass
class RunMetadata:
    """Everything needed to interpret a run without asking anyone."""

    run_id: str
    created_at: str
    harness_version: str = __version__
    sglang_base_commit_expected: str = SGLANG_BASE_COMMIT

    launch: dict[str, Any] = field(default_factory=dict)
    workload: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, Any] = field(default_factory=dict)
    server: dict[str, Any] = field(default_factory=dict)

    policy_id: str = ""
    workload_name: str = ""
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    # -- reproducibility gates ----------------------------------------------

    @property
    def sglang_commit(self) -> str | None:
        return (self.environment.get("sglang_git") or {}).get("commit")

    @property
    def sglang_dirty(self) -> bool | None:
        return (self.environment.get("sglang_git") or {}).get("dirty")

    def commit_mismatch(self) -> bool:
        """Whether the checkout differs from the documented pinned base.

        A mismatch is not automatically fatal -- a deliberate policy branch is
        expected to differ -- but it must be visible, because results/README.md
        forbids comparing across commits silently.
        """
        c = self.sglang_commit
        if c is None:
            return True
        return not c.startswith(self.sglang_base_commit_expected[:12])

    def citable(self) -> tuple[bool, str]:
        """Whether this run may appear in README results tables.

        Encodes results/README.md rule 1: no dirty runs, and the commit must be
        known.
        """
        if self.sglang_dirty is None:
            return False, "SGLang git state unknown (not a checkout?)"
        if self.sglang_dirty:
            return False, "SGLang tree was dirty; run is experimental, not a result"
        if self.sglang_commit is None:
            return False, "SGLang commit unknown"
        if self.commit_mismatch():
            return False, (
                f"commit {self.sglang_commit[:12]} differs from pinned base "
                f"{self.sglang_base_commit_expected[:12]}; re-baseline or label "
                f"explicitly"
            )
        return True, "clean tree at pinned commit"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        ok, reason = self.citable()
        d["citable"] = ok
        d["citable_reason"] = reason
        return d

    @classmethod
    def new(
        cls,
        *,
        run_id: str,
        policy_id: str,
        workload_name: str,
        launch: dict[str, Any] | None = None,
        workload: dict[str, Any] | None = None,
        environment: dict[str, Any] | None = None,
        server: dict[str, Any] | None = None,
        notes: str = "",
    ) -> RunMetadata:
        return cls(
            run_id=run_id,
            created_at=_now_iso(),
            policy_id=policy_id,
            workload_name=workload_name,
            launch=launch or {},
            workload=workload or {},
            environment=environment or {},
            server=server or {},
            notes=notes,
        )

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RunMetadata:
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _slug(text: str) -> str:
    return _SAFE.sub("-", text).strip("-") or "run"


def make_run_dir(
    root: str | Path,
    *,
    workload: str,
    policy: str,
    commit: str | None,
    when: datetime | None = None,
) -> Path:
    """Create `results/<date>_<workload>_<policy>_<shortsha>/` and return it."""
    root = Path(root)
    when = when or datetime.now()
    short = (commit or "unknown")[:9]
    name = "_".join(
        _slug(p) for p in (when.strftime("%Y-%m-%d"), workload, policy, short)
    )
    path = root / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def write_jsonl(path: str | Path, rows: Iterable[Any]) -> int:
    """Write one JSON object per line. Returns the number of rows written."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with p.open("w") as f:
        for row in rows:
            obj = row.to_dict() if hasattr(row, "to_dict") else row
            f.write(json.dumps(obj, default=str) + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return
    with p.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_request_records(path: str | Path) -> list[RequestRecord]:
    return [RequestRecord.from_dict(d) for d in read_jsonl(path)]


def load_iteration_records(path: str | Path) -> list[IterationRecord]:
    return [IterationRecord.from_dict(d) for d in read_jsonl(path)]


def summarise_records(records: Sequence[RequestRecord]) -> dict[str, Any]:
    """Run-level aggregate used for aggregate.json and quick inspection."""
    n = len(records)
    ok = [r for r in records if r.ok]
    spec = [r for r in ok if r.is_speculative]
    lat = [r.latency_s for r in ok if r.latency_s is not None]
    means = [m for m in (r.mean_accepted_drafts() for r in spec) if m is not None]
    inconsistent = [r.rid for r in records if not r.histogram_consistent()[0]]

    def _mean(xs: list[float]) -> float | None:
        return sum(xs) / len(xs) if xs else None

    verified = sum(r.spec_verify_ct or 0 for r in spec)

    # DraftWaste = (drafted - accepted) / drafted needs the depth each request
    # actually ran at. That is unknown for an adaptive run without the iteration
    # trace, so we compute it over the subset that has a known K and report how
    # much of the run that covers, rather than quietly assuming a depth.
    k_known = [r for r in spec if r.effective_k() is not None]
    drafted = sum(
        (r.spec_verify_ct or 0) * float(r.effective_k() or 0.0) for r in k_known
    )
    accepted = sum(r.spec_num_correct_drafts or 0 for r in k_known)

    return {
        "n_requests": n,
        "n_ok": len(ok),
        "n_failed": n - len(ok),
        "n_speculative": len(spec),
        "n_with_known_k": len(k_known),
        "mean_latency_s": _mean([float(x) for x in lat]),
        "max_latency_s": max(lat) if lat else None,
        "mean_completion_tokens": _mean(
            [float(r.completion_tokens) for r in ok if r.completion_tokens is not None]
        ),
        "mean_accepted_drafts": _mean([float(m) for m in means]),
        "total_verify_rounds": verified,
        "total_draft_tokens": drafted if k_known else None,
        "total_accepted_drafts": accepted if k_known else None,
        "draft_waste": ((drafted - accepted) / drafted if drafted > 0 else None),
        "draft_waste_covers_all_speculative": bool(k_known)
        and len(k_known) == len(spec),
        "histogram_inconsistencies": inconsistent,
        "by_class": _group_summary(spec),
    }


def _group_summary(records: Sequence[RequestRecord]) -> dict[str, Any]:
    groups: dict[str, list[RequestRecord]] = {}
    for r in records:
        groups.setdefault(r.prompt_class, []).append(r)
    out: dict[str, Any] = {}
    for cls, rs in sorted(groups.items()):
        means = [m for m in (r.mean_accepted_drafts() for r in rs) if m is not None]
        out[cls] = {
            "n": len(rs),
            "mean_accepted_drafts": (sum(means) / len(means)) if means else None,
            "mean_completion_tokens": (
                sum(float(r.completion_tokens or 0) for r in rs) / len(rs)
            ),
        }
    return out
