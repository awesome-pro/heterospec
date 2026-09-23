"""The K-invariance check: does `P(position k accepted)` depend on `K`?

This tests the assumption every oracle number rests on
(`docs/05-upper-bound-and-assumptions.md`, A2). The oracle evaluates
`E[acc | K]` at depths a request may never have run, which is only valid if the
per-position acceptance curve is a property of the *request* rather than of the
*depth*.

Method. The Session 1 calibration grid runs the **same workload with the same
seed** at static `K ∈ {1,3,5,7}`. Identical prompts, different depths. For each
position `k`, pool the survival across all requests and compare the value
measured at every `K >= k`.

A pass means the shallow measurements predict the deep ones. A failure means the
oracle is comparing quantities that do not exist, and the project stops rather
than reporting an uninterpretable number.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from heterospec.records import RequestRecord

__all__ = [
    "PositionComparison",
    "InvarianceReport",
    "k_invariance_check",
]


@dataclass
class PositionComparison:
    """One draft position, as measured at several depths."""

    k: int
    survival_by_k: dict[int, float]
    max_deviation: float
    spread: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "k": self.k,
            "survival_by_k": dict(sorted(self.survival_by_k.items())),
            "max_deviation": self.max_deviation,
            "spread": self.spread,
        }


@dataclass
class InvarianceReport:
    """Per-position agreement across depths, plus a verdict."""

    ks: tuple[int, ...]
    n_requests_by_k: dict[int, int]
    comparisons: list[PositionComparison] = field(default_factory=list)
    tolerance: float = 0.05
    per_request_mad: dict[int, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def max_deviation(self) -> float:
        return max((c.max_deviation for c in self.comparisons), default=0.0)

    @property
    def worst_position(self) -> int | None:
        if not self.comparisons:
            return None
        return max(self.comparisons, key=lambda c: c.max_deviation).k

    @property
    def passed(self) -> bool:
        """Whether every position agrees across depths within `tolerance`.

        Requires at least two depths observing a position; a single-depth
        comparison is vacuous, not a pass.
        """
        return bool(self.comparisons) and self.max_deviation <= self.tolerance

    def verdict(self) -> str:
        if not self.comparisons:
            return (
                "INCONCLUSIVE: no position was observed at two or more depths, "
                "so K-invariance cannot be assessed"
            )
        if self.passed:
            return (
                f"PASS: per-position acceptance agrees across K "
                f"(max deviation {self.max_deviation:.4f} <= {self.tolerance})"
            )
        return (
            f"FAIL: per-position acceptance depends on K "
            f"(max deviation {self.max_deviation:.4f} at position "
            f"{self.worst_position} > {self.tolerance}). Oracle numbers are "
            f"uninterpretable upper bounds; stop and reassess."
        )

    def summary(self) -> dict[str, Any]:
        return {
            "ks": list(self.ks),
            "n_requests_by_k": dict(sorted(self.n_requests_by_k.items())),
            "tolerance": self.tolerance,
            "max_deviation": self.max_deviation,
            "worst_position": self.worst_position,
            "passed": self.passed,
            "verdict": self.verdict(),
            "per_position": [c.to_dict() for c in self.comparisons],
            "per_request_mad": dict(sorted(self.per_request_mad.items())),
            "notes": self.notes,
        }


def _pooled_survival(records: Sequence[RequestRecord], k: int) -> float | None:
    """Round-weighted pooled `S_k` across requests.

    Pooled rather than averaged per request, so requests with more decode rounds
    (longer generations) carry their proper weight -- the same weighting the
    batch-level quantity has. Uses `sum(hist[k:]) / sum(hist)` directly rather
    than a precomputed curve, so a request that never reached position `k`
    contributes rounds to the denominator with zero survival, which is the
    correct treatment.
    """
    numer = 0
    denom = 0
    for r in records:
        if not r.has_histogram or not r.spec_verify_ct:
            continue
        hist = r.spec_correct_drafts_histogram or []
        numer += sum(hist[k:])
        denom += sum(hist)
    return numer / denom if denom else None


def _per_request_survival(records: Sequence[RequestRecord], k: int) -> dict[str, float]:
    """`S_k` for each request, keyed by prompt hash.

    No filtering on observed curve length: excluding requests that never reached
    position `k` would keep only the high-acceptance ones and make the
    per-request comparison look far more stable than it is.
    """
    out: dict[str, float] = {}
    for r in records:
        if not r.ok or not r.has_histogram or not r.spec_verify_ct:
            continue
        hist = r.spec_correct_drafts_histogram or []
        denom = sum(hist)
        if denom:
            out[r.prompt_sha256] = sum(hist[k:]) / denom
    return out


def k_invariance_check(
    runs: dict[int, Sequence[RequestRecord]],
    *,
    tolerance: float = 0.05,
    require_matched_prompts: bool = True,
) -> InvarianceReport:
    """Compare per-position acceptance across static-K captures.

    Parameters
    ----------
    runs:
        `{K: records}` from static captures of the **same workload and seed**.
        Identical prompts at different depths is what makes the comparison
        meaningful.
    tolerance:
        Maximum acceptable absolute deviation in `S_k` across depths. 0.05 is a
        deliberately generous default: the check is looking for a real trend, not
        sampling noise.
    require_matched_prompts:
        Refuse to compare captures whose prompt sets differ. Comparing two
        different prompt mixtures would confound depth with content and could
        pass or fail for the wrong reason.
    """
    if len(runs) < 2:
        raise ValueError(f"need captures at two or more K values, got {sorted(runs)}")
    for k in runs:
        if k <= 0:
            raise ValueError(f"K must be positive for this check, got {k}")

    ks = tuple(sorted(runs))
    n_by_k = {k: sum(1 for r in runs[k] if r.ok) for k in ks}

    notes: list[str] = []
    if require_matched_prompts:
        prompt_sets = {k: {r.prompt_sha256 for r in runs[k] if r.ok} for k in ks}
        common = set.intersection(*prompt_sets.values())
        if not common:
            raise ValueError(
                "the K captures share no prompts; they cannot be compared. "
                "Use the same workload and seed for every static K."
            )
        if any(len(s) != len(common) for s in prompt_sets.values()):
            notes.append(
                "prompt sets differ across K; restricted to the "
                f"{len(common)} shared prompts"
            )
        restricted = {
            k: [r for r in runs[k] if r.ok and r.prompt_sha256 in common] for k in ks
        }
    else:
        restricted = {k: [r for r in runs[k] if r.ok] for k in ks}

    # -- pooled comparison ---------------------------------------------------
    max_depth = max(ks)
    comparisons: list[PositionComparison] = []
    for k in range(1, max_depth + 1):
        by_k: dict[int, float] = {}
        for run_k in ks:
            if run_k < k:
                continue  # this capture never proposed position k
            val = _pooled_survival(restricted[run_k], k)
            if val is not None:
                by_k[run_k] = val
        if len(by_k) < 2:
            continue
        lo, hi = min(by_k.values()), max(by_k.values())
        comparisons.append(
            PositionComparison(
                k=k,
                survival_by_k=by_k,
                max_deviation=hi - lo,
                spread=hi - lo,
            )
        )

    # -- per-request comparison ---------------------------------------------
    mad: dict[int, float] = {}
    for k in range(1, max_depth + 1):
        observers = [rk for rk in ks if rk >= k]
        if len(observers) < 2:
            continue
        per_obs = {rk: _per_request_survival(restricted[rk], k) for rk in observers}
        shared = set.intersection(*[set(d) for d in per_obs.values()])
        if not shared:
            continue
        devs = [
            max(per_obs[rk][p] for rk in observers)
            - min(per_obs[rk][p] for rk in observers)
            for p in shared
        ]
        mad[k] = sum(devs) / len(devs)

    if not comparisons:
        notes.append(
            "no position was observed at two or more depths. The captures may "
            "have been run at a single K, or every request finished before "
            "position 1."
        )

    return InvarianceReport(
        ks=ks,
        n_requests_by_k=n_by_k,
        comparisons=comparisons,
        tolerance=tolerance,
        per_request_mad=mad,
        notes=notes,
    )
