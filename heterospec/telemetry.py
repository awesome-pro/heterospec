"""Deriving request-level acceptance behaviour from SGLang telemetry.

Source data is SGLang's per-request `spec_correct_drafts_histogram`, exposed in
response `meta_info` with no source modification. `histogram[j]` is the number of
verify rounds in which exactly `j` draft tokens were accepted, **excluding the
bonus token** (`batch_result_processor.py:757-760` subtracts
`num_non_draft_tokens_per_req`, which defaults to 1).

Why the survival curve is the right primitive
---------------------------------------------
For topk=1 chain drafting, acceptance is prefix-contiguous: accepted tokens form
a prefix, so "draft position k was accepted" is exactly "accept_len >= k".
Therefore

    S_k := P(accept_len >= k) = sum(histogram[k:]) / sum(histogram)

is read straight off the histogram, and the expected number of accepted drafts
if the request were run to depth K is

    E[accepted | K] = sum_{k=1..K} S_k

**No independence assumption between positions is needed for this** -- the
histogram already contains the joint prefix probabilities. (Independence would
only be needed to extrapolate beyond observed depths, which we do not do.)

The K-confounding hazard
------------------------
`S_k` is only an unbiased estimate if **every** round the request experienced
proposed at least `k` drafts. Under adaptive decoding `K` varies, and the
histogram does not record `K` per round, so:

* a round with active K=2 can never produce `accept_len >= 5`; it lands in a low
  bucket and is silently scored as "position 5 rejected";
* with a step-0 tier (which SGLang's default ladder enables at BS>=8), a round
  can propose *nothing* and still records 0 accepted -- biasing even `S_1`.

So the bias cannot be corrected post-hoc, and it is not limited to deep
positions. :func:`position_acceptance` therefore returns an explicit validity
bound rather than a bare list of floats, and callers are expected to use
:meth:`PositionAcceptance.valid_survival`. Clean numbers require either a
static-K capture or the iteration-level trace.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

__all__ = [
    "AcceptStats",
    "PositionAcceptance",
    "expected_accepted_drafts_at_depth",
    "histogram_mean",
    "histogram_std",
    "histogram_variance",
    "position_acceptance",
    "preferred_depth",
    "rounds_consistency",
    "survival_curve",
]


# ---------------------------------------------------------------------------
# Raw histogram statistics
# ---------------------------------------------------------------------------


def _n_rounds(histogram: Sequence[int]) -> int:
    return int(sum(histogram))


def histogram_mean(histogram: Sequence[int]) -> float:
    """Mean accepted drafts per round. NaN when there are no rounds."""
    n = _n_rounds(histogram)
    if n == 0:
        return float("nan")
    return sum(k * c for k, c in enumerate(histogram)) / n


def histogram_variance(histogram: Sequence[int]) -> float:
    """Population variance of accepted drafts per round. NaN when no rounds."""
    n = _n_rounds(histogram)
    if n == 0:
        return float("nan")
    mean = histogram_mean(histogram)
    return sum(c * (k - mean) ** 2 for k, c in enumerate(histogram)) / n


def histogram_std(histogram: Sequence[int]) -> float:
    var = histogram_variance(histogram)
    return float("nan") if var != var else var**0.5  # NaN-safe


def survival_curve(
    histogram: Sequence[int], k_max: int | None = None
) -> tuple[float, ...]:
    """`S_k = P(accept_len >= k)` for `k = 1..k_max`.

    `k_max` defaults to the largest observed accepted-draft count. Entries are
    read as `result[k-1]`. Returns an empty tuple when there are no rounds.

    The values are *arithmetically* `P(accept_len >= k)`; whether they are
    *unbiased estimates of position acceptance* is the confounding question,
    handled by :func:`position_acceptance`.
    """
    n = _n_rounds(histogram)
    if n == 0:
        return ()
    if k_max is None:
        k_max = len(histogram) - 1
    if k_max <= 0:
        return ()
    return tuple(sum(histogram[k:]) / n for k in range(1, k_max + 1))


def expected_accepted_drafts_at_depth(survival: Sequence[float], depth: int) -> float:
    """`E[accepted drafts] = sum_{k=1..depth} S_k`.

    Truncated at the length of `survival`: we never extrapolate beyond observed
    depths, because that would require an unvalidated independence assumption.
    """
    if depth <= 0:
        return 0.0
    return float(sum(survival[:depth]))


# ---------------------------------------------------------------------------
# Position acceptance with an explicit validity bound
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PositionAcceptance:
    """Per-request position acceptance plus how far it can be trusted.

    Attributes
    ----------
    survival:
        `survival[k-1] = S_k = P(accept_len >= k)`, arithmetic values.
    n_rounds:
        Verify rounds summarised, i.e. `sum(histogram)`.
    max_observed:
        Largest accepted-draft count seen (`len(histogram) - 1`).
    safe_up_to:
        Largest `k` for which `S_k` is unbiased, or 0 when unknown/confounded.
    k_static:
        The fixed K this capture ran at, when known.
    min_k_proposed:
        Smallest number of drafts proposed in any round, when known.
    """

    survival: tuple[float, ...]
    n_rounds: int
    max_observed: int
    safe_up_to: int
    k_static: int | None = None
    min_k_proposed: int | None = None

    @property
    def k_confounded(self) -> bool:
        """True when no position probability from this capture is trustworthy."""
        return self.safe_up_to < 1

    @property
    def achievable_depth(self) -> int:
        """Deepest `K` we may evaluate without extrapolating."""
        return min(len(self.survival), self.safe_up_to)

    def valid_survival(self) -> tuple[float, ...]:
        """Only the trustworthy prefix, `S_1..S_{safe_up_to}`."""
        return self.survival[: self.safe_up_to]

    def position_acceptance_probs(self) -> tuple[float, ...]:
        """`P(position k accepted)` for trustworthy k only."""
        return self.valid_survival()

    def expected_accepted(self, depth: int) -> float:
        """Expected accepted drafts at `depth`, refusing to extrapolate.

        Raises
        ------
        ValueError
            If `depth` exceeds the trustworthy range, which would silently
            produce a number the data cannot support.
        """
        if depth > self.achievable_depth:
            raise ValueError(
                f"depth {depth} exceeds trustworthy range "
                f"{self.achievable_depth} (safe_up_to={self.safe_up_to}, "
                f"observed={self.max_observed}, k_static={self.k_static}, "
                f"min_k_proposed={self.min_k_proposed}). Extrapolating past the "
                f"observed/proposed depth requires a validated K-invariance "
                f"assumption."
            )
        return expected_accepted_drafts_at_depth(self.survival, depth)

    def require_clean(self) -> PositionAcceptance:
        """Raise unless position acceptance is trustworthy, else return self."""
        if self.k_confounded:
            raise ValueError(
                "position acceptance is K-confounded: this capture ran under a "
                "varying or unknown K, so P(A_k) cannot be recovered from the "
                "histogram alone (see heterospec/telemetry.py). Capture at "
                "static K, or supply min_k_proposed from the iteration-level "
                "trace."
            )
        return self

    def describe(self) -> str:
        src = (
            f"static K={self.k_static}"
            if self.k_static is not None
            else (
                f"min K proposed={self.min_k_proposed}"
                if self.min_k_proposed is not None
                else "K unknown"
            )
        )
        return (
            f"{self.n_rounds} rounds, {src}, max observed={self.max_observed}, "
            f"trustworthy up to k={self.safe_up_to}"
        )


def position_acceptance(
    histogram: Sequence[int],
    *,
    k_static: int | None = None,
    min_k_proposed: int | None = None,
) -> PositionAcceptance:
    """Build a :class:`PositionAcceptance` from a request's histogram.

    Parameters
    ----------
    histogram:
        `spec_correct_drafts_histogram` for one request.
    k_static:
        The fixed K this capture ran at. When given, the result is clean and
        `safe_up_to == k_static`. Use this for the primary per-request study.
    min_k_proposed:
        Smallest number of drafts proposed in any round, from the
        iteration-level trace. Clean up to that value.

    When neither is supplied the result is explicitly marked confounded with
    `safe_up_to == 0`, rather than returning plausible-looking biased numbers.

    Raises
    ------
    ValueError
        If `k_static` is negative, or `k_static`/`min_k_proposed` exceed the
        largest depth the histogram could possibly speak to.
    """
    if k_static is not None and k_static < 0:
        raise ValueError(f"k_static must be >= 0, got {k_static}")
    if min_k_proposed is not None and min_k_proposed < 0:
        raise ValueError(f"min_k_proposed must be >= 0, got {min_k_proposed}")

    n = _n_rounds(histogram)
    max_observed = len(histogram) - 1 if histogram else 0

    if k_static is not None:
        safe_up_to = k_static
    elif min_k_proposed is not None:
        safe_up_to = min_k_proposed
    else:
        safe_up_to = 0

    # The curve is computed out to the deepest k we could be asked about.
    k_max = max(max_observed, safe_up_to)
    curve = survival_curve(histogram, k_max=k_max)

    return PositionAcceptance(
        survival=curve,
        n_rounds=n,
        max_observed=max_observed,
        safe_up_to=safe_up_to,
        k_static=k_static,
        min_k_proposed=min_k_proposed,
    )


# ---------------------------------------------------------------------------
# Helpers used by the oracle study
# ---------------------------------------------------------------------------


def preferred_depth(
    survival: Sequence[float],
    *,
    candidates: Iterable[int],
    cost_fn,
) -> int:
    """Depth maximising `E[accepted]/cost`, for one request's survival curve.

    `cost_fn(k)` is the per-round cost of running depth `k`; it should come from
    a GPU-calibrated model (`Cost(k, batch_size)`), never from intuition.
    Depths beyond `len(survival)` get `E[accepted]` from the observed prefix,
    which understates deeper tiers -- callers must ensure `candidates` is within
    the validated range.
    """
    best_k, best_score = 0, float("-inf")
    for k in sorted(set(candidates)):
        c = cost_fn(k)
        if c <= 0:
            raise ValueError(f"cost_fn({k}) must be positive, got {c}")
        score = expected_accepted_drafts_at_depth(survival, k) / c
        if score > best_score:
            best_k, best_score = k, score
    return best_k


def rounds_consistency(
    histogram: Sequence[int], spec_verify_ct: int | None
) -> tuple[bool, str]:
    """Check `sum(histogram) == spec_verify_ct`.

    Free integrity check on a capture. Both counters are incremented in the same
    branch of `_resolve_spec_v2_tokens` (which excludes retracted/finished
    requests), so they should agree. A mismatch means the telemetry path did
    something we do not understand yet -- worth knowing before analysing.

    Returns `(ok, message)` rather than raising, because a mismatch is a finding,
    not a crash.
    """
    if spec_verify_ct is None:
        return True, "spec_verify_ct not provided; consistency unchecked"
    n = _n_rounds(histogram)
    if n == spec_verify_ct:
        return True, f"consistent: {n} rounds"
    return (
        False,
        f"INCONSISTENT: sum(histogram)={n} but spec_verify_ct={spec_verify_ct} "
        f"(delta {n - spec_verify_ct})",
    )


@dataclass(frozen=True)
class AcceptStats:
    """Descriptive statistics for one request's or one class's acceptance."""

    n_rounds: int
    mean: float
    variance: float
    std: float
    max_observed: int

    @classmethod
    def from_histogram(cls, histogram: Sequence[int]) -> AcceptStats:
        return cls(
            n_rounds=_n_rounds(histogram),
            mean=histogram_mean(histogram),
            variance=histogram_variance(histogram),
            std=histogram_std(histogram),
            max_observed=len(histogram) - 1 if histogram else 0,
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "n_rounds": self.n_rounds,
            "mean_accepted": self.mean,
            "variance_accepted": self.variance,
            "std_accepted": self.std,
            "max_observed": self.max_observed,
        }
