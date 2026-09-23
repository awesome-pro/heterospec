"""Measured cost model `Cost(K, batch_size)` from a GPU calibration grid.

The oracle's placeholder `LinearCostModel` cannot produce a real gap number. This
module turns calibration runs into a fitted cost surface.

How the cost is measured (client-side only)
-------------------------------------------
No server-side timing instrumentation is needed. A calibration run at a fixed
static `K` and fixed concurrency `n` yields, from `meta_info` plus the harness
wall clock:

```text
throughput        = total_completion_tokens / wall_time_s          [tokens/s]
tokens_per_rr     = total_completion_tokens / total_request_rounds
                    [tokens per request-round]
```

so the per-batch-round cost is

```text
Cost(K, n) = n * tokens_per_rr(K, n) / throughput(K, n)
```

with units of "cost per batch decode round", relative across the grid. This is
exactly the quantity the oracle needs: `Cost(k, n) / tokens_per_round(k)` is the
time per token, and it reduces to `1 / throughput` -- so the calibration is
ultimately grounded in measured tokens per second.

Note the deliberate separation: **cost comes from measurement, token counts come
from per-batch prediction.** The oracle never mixes the two.

Generalisations and their honesty
---------------------------------
The grid cannot cover every `(K, n)`. Interpolation is bilinear on the grid.
Points outside it are clamped to the nearest grid edge, and the clamp is counted
and reported rather than hidden, because a clamped query is an untested
extrapolation and a gap that depends on one deserves to be flagged.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "CalibrationPoint",
    "MeasuredCostModel",
    "cost_from_run",
]


@dataclass(frozen=True)
class CalibrationPoint:
    """One measured `(K, batch_size)` cell of the calibration grid."""

    k: int
    batch_size: int
    wall_time_s: float
    total_completion_tokens: int
    total_request_rounds: int
    n_requests: int = 0
    run_dir: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        if self.k < 0:
            raise ValueError(f"k must be >= 0, got {self.k}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {self.batch_size}")
        if self.wall_time_s <= 0:
            raise ValueError(f"wall_time_s must be > 0, got {self.wall_time_s}")
        if self.total_completion_tokens <= 0:
            raise ValueError("total_completion_tokens must be > 0")

    @property
    def throughput(self) -> float:
        """Measured output tokens per second."""
        return self.total_completion_tokens / self.wall_time_s

    @property
    def tokens_per_request_round(self) -> float:
        """Measured tokens emitted per request per decode round.

        Equals `1 + mean_accepted_drafts`: one bonus token plus accepted drafts.
        """
        if self.total_request_rounds <= 0:
            # A no-spec run has no verify rounds; every round emits exactly one
            # token per request.
            return 1.0
        return self.total_completion_tokens / self.total_request_rounds

    @property
    def cost_per_round(self) -> float:
        """`Cost(K, n)`, relative across the grid."""
        return self.batch_size * self.tokens_per_request_round / self.throughput

    @property
    def cost_per_token(self) -> float:
        """`1 / throughput` -- the sanity identity this measurement relies on."""
        return 1.0 / self.throughput

    def to_dict(self) -> dict[str, Any]:
        d = {
            "k": self.k,
            "batch_size": self.batch_size,
            "wall_time_s": self.wall_time_s,
            "total_completion_tokens": self.total_completion_tokens,
            "total_request_rounds": self.total_request_rounds,
            "n_requests": self.n_requests,
            "run_dir": self.run_dir,
            "notes": self.notes,
        }
        d["throughput"] = self.throughput
        d["cost_per_round"] = self.cost_per_round
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CalibrationPoint:
        return cls(
            k=int(d["k"]),
            batch_size=int(d["batch_size"]),
            wall_time_s=float(d["wall_time_s"]),
            total_completion_tokens=int(d["total_completion_tokens"]),
            total_request_rounds=int(d.get("total_request_rounds") or 0),
            n_requests=int(d.get("n_requests") or 0),
            run_dir=d.get("run_dir", ""),
            notes=d.get("notes", ""),
        )


def cost_from_run(
    *,
    k: int,
    batch_size: int,
    wall_time_s: float,
    records: Sequence[Any],
    run_dir: str = "",
    notes: str = "",
) -> CalibrationPoint:
    """Build a calibration point from a completed run's records.

    Only successful requests contribute. Failed requests would deflate the
    throughput and make the cost look worse than the hardware really is.
    """
    ok = [r for r in records if getattr(r, "ok", False)]
    if not ok:
        raise ValueError(f"no successful records for k={k}, n={batch_size}")
    return CalibrationPoint(
        k=k,
        batch_size=batch_size,
        wall_time_s=wall_time_s,
        total_completion_tokens=sum(int(r.completion_tokens or 0) for r in ok),
        total_request_rounds=sum(int(r.spec_verify_ct or 0) for r in ok),
        n_requests=len(ok),
        run_dir=run_dir,
        notes=notes,
    )


@dataclass
class MeasuredCostModel:
    """Bilinear-interpolated `Cost(K, batch_size)` over a measured grid.

    Satisfies the `heterospec.analysis.oracle.CostModel` interface, so it drops
    straight into `analyse_batches`.
    """

    points: list[CalibrationPoint]
    _grid: dict[tuple[int, int], float] = field(
        init=False, repr=False, default_factory=dict
    )
    _ks: list[int] = field(init=False, repr=False, default_factory=list)
    _ns: list[int] = field(init=False, repr=False, default_factory=list)
    clamp_count: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        if not self.points:
            raise ValueError("no calibration points")
        self._grid = {(p.k, p.batch_size): p.cost_per_round for p in self.points}
        self._ks = sorted({p.k for p in self.points})
        self._ns = sorted({p.batch_size for p in self.points})
        # Reject a grid that cannot support interpolation in both axes without
        # silently extrapolating everywhere.
        if len(self._ks) < 1 or len(self._ns) < 1:  # pragma: no cover
            raise ValueError("degenerate calibration grid")

    # -- construction --------------------------------------------------------

    @classmethod
    def from_points(cls, points: Sequence[CalibrationPoint]) -> MeasuredCostModel:
        return cls(points=list(points))

    @classmethod
    def from_jsonl(cls, path: str | Path) -> MeasuredCostModel:
        rows = []
        with Path(path).open() as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(CalibrationPoint.from_dict(json.loads(line)))
        return cls(points=rows)

    def to_jsonl(self, path: str | Path) -> int:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w") as f:
            for pt in self.points:
                f.write(json.dumps(pt.to_dict()) + "\n")
        return len(self.points)

    # -- description ---------------------------------------------------------

    @property
    def k_values(self) -> list[int]:
        return list(self._ks)

    @property
    def batch_sizes(self) -> list[int]:
        return list(self._ns)

    @property
    def n_cells(self) -> int:
        return len(self._grid)

    def coverage(self) -> dict[str, Any]:
        """Grid shape plus which requested cells are actually measured."""
        expected = len(self._ks) * len(self._ns)
        return {
            "k_values": self._ks,
            "batch_sizes": self._ns,
            "n_cells_measured": len(self._grid),
            "n_cells_expected_on_full_grid": expected,
            "complete": len(self._grid) == expected,
            "clamped_queries": self.clamp_count,
        }

    # -- the CostModel interface --------------------------------------------

    def cost(self, k: int, n: int) -> float:
        """Cost of one decode round at depth `k` for batch size `n`.

        Bilinear on the grid; clamped to the nearest measured edge outside it.
        Clamps are counted (see `clamp_count`) so a gap that leans on an
        extrapolation can be identified rather than quietly trusted.
        """
        if n <= 0:
            raise ValueError("batch size must be > 0")
        kq = _clamp_to(k, self._ks, self)
        nq = _clamp_to(n, self._ns, self)

        k_lo, k_hi = _bracket(kq, self._ks)
        n_lo, n_hi = _bracket(nq, self._ns)

        def cell(kk: int, nn: int) -> float:
            v = self._grid.get((kk, nn))
            if v is None:
                raise ValueError(
                    f"calibration grid has no measurement for K={kk}, n={nn}; "
                    f"measured cells: {sorted(self._grid)}"
                )
            return v

        c_ll, c_lh = cell(k_lo, n_lo), cell(k_lo, n_hi)
        c_hl, c_hh = cell(k_hi, n_lo), cell(k_hi, n_hi)

        tk = 0.0 if k_hi == k_lo else (kq - k_lo) / (k_hi - k_lo)
        tn = 0.0 if n_hi == n_lo else (nq - n_lo) / (n_hi - n_lo)

        c_lo = c_ll + tk * (c_hl - c_ll)
        c_hi = c_lh + tk * (c_hh - c_lh)
        c = c_lo + tn * (c_hi - c_lo)
        if c <= 0:
            raise ValueError(f"non-positive interpolated cost for k={k}, n={n}")
        return c

    def __repr__(self) -> str:
        return (
            f"MeasuredCostModel(grid={len(self._grid)} cells, "
            f"K={self._ks}, n={self._ns})"
        )


def _clamp_to(value: int, values: Sequence[int], model: MeasuredCostModel) -> int:
    if not values:  # pragma: no cover - guarded by __post_init__
        raise ValueError("empty axis")
    if value < values[0]:
        model.clamp_count += 1
        return values[0]
    if value > values[-1]:
        model.clamp_count += 1
        return values[-1]
    return value


def _bracket(value: int, values: Sequence[int]) -> tuple[int, int]:
    """Nearest measured values bracketing `value` (equal when exactly on a node)."""
    if value <= values[0]:
        return values[0], values[0]
    if value >= values[-1]:
        return values[-1], values[-1]
    for lo, hi in zip(values, values[1:], strict=False):
        if lo <= value <= hi:
            return lo, hi
    return values[-1], values[-1]  # pragma: no cover - unreachable
