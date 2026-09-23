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
    #
    # A run is reproducible when *which code produced it* is recorded exactly and
    # nothing was uncommitted at the time. That is a three-part statement, and
    # conflating any two of them is how a reproducibility story goes wrong:
    #
    #   base_sha               the pinned upstream commit this project baselines on
    #   experiment_patch_sha   the actual SGLang HEAD during the run
    #   working_tree_dirty     whether uncommitted edits were present
    #
    # A clean checkout of a known research patch is a perfectly citable artifact:
    # `base_sha + experiment_patch_sha` names it precisely. An earlier version of
    # this class treated "HEAD != pinned base" as non-citable, which would have
    # marked every run on the telemetry branch unusable -- i.e. all of Session 1.

    @property
    def base_sha(self) -> str:
        """The pinned upstream commit this project baselines against."""
        return self.sglang_base_commit_expected

    @property
    def sglang_commit(self) -> str | None:
        """Deprecated alias for `experiment_patch_sha`."""
        return self.experiment_patch_sha

    @property
    def experiment_patch_sha(self) -> str | None:
        """Actual SGLang HEAD during the run."""
        return (self.environment.get("sglang_git") or {}).get("commit")

    @property
    def sglang_dirty(self) -> bool | None:
        """Deprecated alias for `working_tree_dirty`."""
        return self.working_tree_dirty

    @property
    def working_tree_dirty(self) -> bool | None:
        """Whether the SGLang checkout had uncommitted changes."""
        return (self.environment.get("sglang_git") or {}).get("dirty")

    @property
    def sglang_branch(self) -> str | None:
        return (self.environment.get("sglang_git") or {}).get("branch")

    @property
    def runs_on_pinned_base(self) -> bool:
        """Whether HEAD *is* the pinned base, i.e. no research patch was applied.

        Informational. False is normal and says nothing about citability.
        """
        c = self.experiment_patch_sha
        if c is None:
            return False
        return c.startswith(self.base_sha[:12])

    def commit_mismatch(self) -> bool:
        """Inverse of `runs_on_pinned_base`, kept for compatibility."""
        return not self.runs_on_pinned_base

    def citable(self) -> tuple[bool, str]:
        """Whether this run may appear in README results tables.

        Citable requires: not a mock, a *clean* working tree, and a recorded
        `experiment_patch_sha`. A run on a research patch qualifies, provided the
        patch commit is named -- that is what makes it reproducible.

        Deliberately does **not** require `experiment_patch_sha == base_sha`. The
        session runs on the telemetry branch, off-base by design.
        """
        if self.is_mock:
            return False, "server was a mock; synthetic data is not a result"
        if self.working_tree_dirty is None:
            return False, "SGLang git state unknown (not a checkout?)"
        if self.working_tree_dirty:
            return False, (
                "SGLang tree had uncommitted changes; the exact code cannot be "
                "reconstructed, so this run is not citable"
            )
        sha = self.experiment_patch_sha
        if sha is None:
            return False, "SGLang commit unknown; cannot name the code that ran"
        if self.runs_on_pinned_base:
            return True, f"clean tree at pinned base {sha[:12]}"
        return True, (
            f"clean tree at experiment patch {sha[:12]} "
            f"(base {self.base_sha[:12]}, branch {self.sglang_branch})"
        )

    @property
    def is_mock(self) -> bool:
        """Whether the results came from `heterospec.mockserver`."""
        if self.server.get("mode") == "mock":
            return True
        # Belt and braces: also trust the server's own /server_info flag.
        return bool((self.server.get("server_info") or {}).get("mock"))

    def to_dict(self) -> dict[str, Any]:
        """Serializable metadata, with the derived verdicts surfaced.

        `sglang_commit`, `sglang_dirty` and `is_mock` are properties, not
        fields, so `asdict` would omit them. They are copied in explicitly:
        reading metadata.json should not require unpacking `environment`.
        """
        d = asdict(self)
        ok, reason = self.citable()
        d["citable"] = ok
        d["citable_reason"] = reason
        # Three-part provenance, surfaced at the top level so reading
        # metadata.json does not require unpacking `environment`.
        d["base_sha"] = self.base_sha
        d["experiment_patch_sha"] = self.experiment_patch_sha
        d["working_tree_dirty"] = self.working_tree_dirty
        d["sglang_branch"] = self.sglang_branch
        d["runs_on_pinned_base"] = self.runs_on_pinned_base
        d["is_mock"] = self.is_mock
        # Deprecated aliases, kept so older result directories still read.
        d["sglang_commit"] = self.experiment_patch_sha
        d["sglang_dirty"] = self.working_tree_dirty
        d["commit_mismatch"] = self.commit_mismatch()
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
