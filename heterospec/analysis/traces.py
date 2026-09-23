"""Turn a real capture into oracle input.

The oracle needs `BatchTrace` objects: per-request survival curves plus the batch
composition. There are two ways to build them, and the choice has a direct
methodological consequence.

**From waves** (`batches_from_waves`) — no source change needed.
With ``--dispatch waves`` the harness submits exactly `concurrency` requests and
waits for all of them, so each wave *is* a batch with known composition. This is
available from the `meta_info`-only study on unmodified SGLang, and it is the
reason `waves` is the default dispatch mode.

**From the iteration trace** (`batches_from_iterations`) — exact.
Requires the fork trace patch. Batches are the real decode iterations, so
composition reflects retractions, early finishes and continuous batching rather
than a controlled submission pattern.

The K-confounding rule still applies and is enforced here, not left to callers:
a request's survival curve is only trusted up to the smallest number of drafts it
was ever *proposed*. Without the trace, that is only known for a static-K
capture; with the trace it comes from `min_k_proposed_per_request`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from heterospec.analysis.oracle import BatchTrace
from heterospec.records import (
    IterationRecord,
    RequestRecord,
    min_k_proposed_per_request,
)
from heterospec.telemetry import position_acceptance

__all__ = [
    "CaptureSkip",
    "batches_from_waves",
    "batches_from_iterations",
    "describe_skips",
]


@dataclass
class CaptureSkip:
    """A request that could not contribute a trustworthy survival curve."""

    rid: str
    reason: str


def _scalar_for(records: Sequence[RequestRecord]) -> float:
    """The scalar the shipped controller sees for this batch.

    The controller smooths the per-round accepted-draft counts into an EMA, so
    its steady-state value is the mean over requests of
    ``spec_num_correct_drafts / spec_verify_ct`` — accepted drafts per round.
    Using the lifetime mean is deliberate: it is what the EMA converges to, and
    it is the information the scalar policy actually has.
    """
    vals = []
    for r in records:
        if r.spec_verify_ct and r.spec_num_correct_drafts is not None:
            vals.append(r.spec_num_correct_drafts / r.spec_verify_ct)
    return sum(vals) / len(vals) if vals else 0.0


def _curves(
    records: Sequence[RequestRecord],
    *,
    k_static: int | None,
    min_k_proposed: dict[str, int] | None,
) -> tuple[dict[str, tuple[float, ...]], list[CaptureSkip]]:
    """Per-request survival curves, dropping any that are not trustworthy."""
    curves: dict[str, tuple[float, ...]] = {}
    skips: list[CaptureSkip] = []

    for r in records:
        if not r.ok:
            skips.append(CaptureSkip(r.rid, "request failed"))
            continue
        if not r.has_histogram:
            skips.append(CaptureSkip(r.rid, "no spec histogram (non-speculative?)"))
            continue

        mkp = None
        if min_k_proposed is not None:
            mkp = min_k_proposed.get(r.rid)
            if mkp is None:
                # The request was never seen in the trace: we cannot bound its K.
                skips.append(
                    CaptureSkip(r.rid, "absent from iteration trace; K unbounded")
                )
                continue
        elif k_static is None:
            skips.append(
                CaptureSkip(
                    r.rid,
                    "no k_static and no iteration trace; P(A_k) would be K-confounded",
                )
            )
            continue

        pa = position_acceptance(
            r.spec_correct_drafts_histogram or [],
            k_static=k_static if mkp is None else None,
            min_k_proposed=mkp,
        )
        curve = pa.valid_survival()
        if not curve or pa.safe_up_to < 1:
            skips.append(
                CaptureSkip(
                    r.rid,
                    f"no trustworthy depth (safe_up_to={pa.safe_up_to}, "
                    f"rounds={pa.n_rounds})",
                )
            )
            continue
        curves[r.rid] = curve

    return curves, skips


def batches_from_waves(
    records: Sequence[RequestRecord],
    *,
    concurrency: int,
    k_static: int,
) -> tuple[list[BatchTrace], list[CaptureSkip]]:
    """Build batches from wave dispatch: each group of `concurrency` requests.

    Reconstructs batch membership from the plan indices, which works because wave
    dispatch submits exactly `concurrency` requests at a time and waits. Requires
    a static-K capture: without the trace there is no other way to bound K.

    Raises if the capture is not static-K, because the resulting curves would be
    biased (see `heterospec.telemetry`).
    """
    if concurrency <= 0:
        raise ValueError("concurrency must be > 0")
    if k_static is None or k_static <= 0:
        raise ValueError(
            "batches_from_waves needs a positive static K. An adaptive capture "
            "must use the iteration trace, because P(A_k) is otherwise "
            "K-confounded and cannot be corrected post-hoc."
        )

    ordered = sorted(records, key=lambda r: r.index)
    batches: list[BatchTrace] = []
    skips: list[CaptureSkip] = []

    for start in range(0, len(ordered), concurrency):
        wave = ordered[start : start + concurrency]
        curves, wave_skips = _curves(wave, k_static=k_static, min_k_proposed=None)
        skips.extend(wave_skips)
        if len(curves) < 2:
            # A batch of one has nothing to be heterogeneous about, and the
            # oracle's per-batch comparison is meaningless there.
            continue
        batches.append(
            BatchTrace(
                batch_id=len(batches),
                survivals=curves,
                active_k=k_static,
                observed_mean_accepted=_scalar_for(
                    [r for r in wave if r.rid in curves]
                ),
            )
        )
    return batches, skips


def batches_from_iterations(
    records: Sequence[RequestRecord],
    iterations: Sequence[IterationRecord],
    *,
    max_iterations: int | None = None,
    min_batch_size: int = 2,
) -> tuple[list[BatchTrace], list[CaptureSkip]]:
    """Build batches from real decode iterations (requires the trace patch).

    Composition is the true batch, so this captures retractions, early finishes
    and continuous batching. K is bounded per request by
    `min_k_proposed_per_request`, which is what makes an adaptive capture
    interpretable.
    """
    by_rid = {r.rid: r for r in records}
    mkp = min_k_proposed_per_request(iterations)

    # Curves use the request's whole lifetime history, which is exactly what a
    # request-aware policy would hold. The bound comes from the trace.
    curves, skips = _curves(records, k_static=None, min_k_proposed=mkp)

    batches: list[BatchTrace] = []
    iters = list(iterations)
    if max_iterations is not None:
        iters = iters[:max_iterations]

    for it in iters:
        present = [rid for rid in it.rids if rid in curves]
        if len(present) < max(2, min_batch_size):
            continue
        sub = {rid: curves[rid] for rid in present}
        it_records = [by_rid[rid] for rid in present if rid in by_rid]
        batches.append(
            BatchTrace(
                batch_id=len(batches),
                survivals=sub,
                active_k=int(it.active_k),
                observed_mean_accepted=_scalar_for(it_records),
            )
        )
    return batches, skips


def describe_skips(skips: Sequence[CaptureSkip]) -> dict[str, Any]:
    """Group skip reasons for reporting.

    Skips are a finding, not noise: if most requests are dropped, the capture
    cannot support the analysis and that must be visible rather than silently
    producing a small sample.
    """
    by_reason: dict[str, int] = {}
    for s in skips:
        by_reason[s.reason] = by_reason.get(s.reason, 0) + 1
    return {
        "n_skipped": len(skips),
        "by_reason": dict(sorted(by_reason.items(), key=lambda kv: -kv[1])),
        "example_rids": [s.rid for s in skips[:5]],
    }
