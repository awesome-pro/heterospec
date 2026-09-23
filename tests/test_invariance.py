"""Tests for the K-invariance check.

Two kinds of test matter here:

* **It must pass on genuinely K-invariant data** — otherwise the project stops for
  the wrong reason.
* **It must fail on K-dependent data** — otherwise the load-bearing assumption is
  never actually tested and the oracle reports uninterpretable numbers.

Both are constructed by synthesising histograms with known per-position behaviour.
"""

import pytest

from heterospec.analysis.invariance import k_invariance_check
from heterospec.records import RequestRecord


def _hist_from_accept_probs(p, k, n_rounds, rng_seed=0):
    """Build a histogram by sampling a chain with conditional acceptance `p`."""
    import random

    rng = random.Random(rng_seed)
    hist = [0] * (k + 1)
    for _ in range(n_rounds):
        acc = 0
        for i in range(k):
            if rng.random() < (p[i] if i < len(p) else p[-1]):
                acc += 1
            else:
                break
        hist[acc] += 1
    return hist


def _records(k, p, *, n=40, rounds=60, prompt_prefix="prompt", seed=0):
    out = []
    for i in range(n):
        hist = _hist_from_accept_probs(p, k, rounds, rng_seed=seed * 1000 + i)
        out.append(
            RequestRecord(
                rid=f"{prompt_prefix}{i}",
                index=i,
                prompt=f"{prompt_prefix}{i}",
                ok=True,
                spec_verify_ct=sum(hist),
                spec_num_correct_drafts=sum(j * c for j, c in enumerate(hist)),
                spec_correct_drafts_histogram=hist,
                completion_tokens=sum(hist) + sum(j * c for j, c in enumerate(hist)),
            )
        )
    return out


# ---------------------------------------------------------------------------
# Invariant data must pass
# ---------------------------------------------------------------------------


def test_passes_on_k_invariant_acceptance():
    """Same request-level behaviour measured at four depths must agree."""
    p = (0.85, 0.7, 0.55, 0.4, 0.3, 0.2, 0.12)
    runs = {k: _records(k, p, seed=k) for k in (1, 3, 5, 7)}
    rep = k_invariance_check(runs, tolerance=0.05)
    assert rep.passed, rep.verdict()
    assert rep.max_deviation <= 0.05
    assert "PASS" in rep.verdict()


def test_perfectly_identical_runs_show_near_zero_deviation():
    """The same histogram at every K is the degenerate invariant case."""
    p = (0.9, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05)
    base = _records(7, p, seed=0)
    runs = {k: base for k in (3, 5, 7)}
    rep = k_invariance_check(runs)
    assert rep.max_deviation == pytest.approx(0.0, abs=1e-12)
    assert rep.passed


# ---------------------------------------------------------------------------
# K-dependent data must fail -- otherwise the check is vacuous
# ---------------------------------------------------------------------------


def test_fails_when_deeper_K_lowers_position_acceptance():
    """Draft drift: deeper chains accept less at each position. Must FAIL.

    This is the failure mode the check exists to detect. If it did not fail here,
    the check would be useless.
    """
    p_weak = (0.6, 0.35, 0.18, 0.08, 0.03, 0.01, 0.005)
    p_strong = (0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65)
    runs = {
        1: _records(1, p_strong, seed=1),
        3: _records(3, p_strong, seed=3),
        7: _records(7, p_weak, seed=7),  # deep chains degrade
    }
    rep = k_invariance_check(runs, tolerance=0.05)
    assert not rep.passed, rep.verdict()
    assert "FAIL" in rep.verdict()
    assert rep.max_deviation > 0.05


def test_worst_position_is_reported_on_failure():
    p_weak = (0.6, 0.35, 0.18, 0.08, 0.03, 0.01, 0.005)
    p_strong = (0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65)
    runs = {1: _records(1, p_strong, seed=1), 7: _records(7, p_weak, seed=7)}
    rep = k_invariance_check(runs, tolerance=0.05)
    assert rep.worst_position is not None
    assert rep.worst_position in {1, 2, 3, 4, 5, 6, 7}


def test_tolerance_is_respected():
    """A borderline deviation flips the verdict as tolerance changes."""
    p_a = (0.85, 0.7, 0.55, 0.4, 0.3, 0.2, 0.12)
    p_b = (0.80, 0.66, 0.52, 0.38, 0.28, 0.19, 0.11)
    runs = {3: _records(3, p_a, seed=1), 7: _records(7, p_b, seed=7)}
    strict = k_invariance_check(runs, tolerance=0.001)
    loose = k_invariance_check(runs, tolerance=0.5)
    assert not strict.passed
    assert loose.passed


# ---------------------------------------------------------------------------
# Structure of the report
# ---------------------------------------------------------------------------


def test_comparisons_only_cover_positions_with_two_observers():
    """A position is only comparable where two depths actually proposed it.

    K=1 observes position 1 only; K=3 observes 1, 2 and 3. So position 1 is the
    only comparable one, even though the deeper capture reports more positions.
    """
    p = (0.9, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05)
    runs = {1: _records(1, p, seed=1), 3: _records(3, p, seed=3)}
    rep = k_invariance_check(runs)
    assert {c.k for c in rep.comparisons} == {1}
    assert all(len(c.survival_by_k) >= 2 for c in rep.comparisons)


def test_three_depths_give_more_comparable_positions():
    p = (0.9, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05)
    runs = {k: _records(k, p, seed=k) for k in (1, 3, 7)}
    rep = k_invariance_check(runs)
    # Position 1 has three observers; positions 2 and 3 have two (K=3, K=7).
    assert {c.k for c in rep.comparisons} == {1, 2, 3}
    assert len(rep.comparisons[0].survival_by_k) == 3


def test_two_depths_are_enough_to_compare():
    p = (0.9, 0.8, 0.6)
    rep = k_invariance_check({4: _records(4, p, seed=1), 5: _records(5, p, seed=2)})
    assert {c.k for c in rep.comparisons} == {1, 2, 3, 4}


def test_requires_two_depths():
    with pytest.raises(ValueError, match="two or more K values"):
        k_invariance_check({3: _records(3, (0.9, 0.8), seed=1)})


def test_rejects_non_positive_k():
    with pytest.raises(ValueError, match="K must be positive"):
        k_invariance_check(
            {0: _records(3, (0.9, 0.8), seed=1), 3: _records(3, (0.9, 0.8), seed=2)}
        )


def test_refuses_captures_with_no_shared_prompts():
    """Comparing different prompt sets would confound depth with content."""
    runs = {
        3: _records(3, (0.9, 0.8), seed=1, prompt_prefix="alpha"),
        7: _records(7, (0.9, 0.8), seed=2, prompt_prefix="beta"),
    }
    with pytest.raises(ValueError, match="share no prompts"):
        k_invariance_check(runs)


def test_restricts_to_shared_prompts_and_notes_it():
    common = _records(3, (0.9, 0.8), seed=1, prompt_prefix="shared")
    extra = _records(3, (0.9, 0.8), seed=2, prompt_prefix="extra")
    runs = {
        3: common + extra,
        7: common + _records(7, (0.9, 0.85), seed=3, prompt_prefix="other"),
    }
    rep = k_invariance_check(runs)
    assert any("shared prompts" in n for n in rep.notes)


def test_report_is_json_serialisable():
    import json

    p = (0.9, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05)
    rep = k_invariance_check({3: _records(3, p, seed=1), 7: _records(7, p, seed=2)})
    payload = json.loads(json.dumps(rep.summary()))
    assert payload["passed"] is True
    assert "per_position" in payload
    assert payload["ks"] == [3, 7]


def test_requests_without_histograms_are_ignored():
    p = (0.9, 0.8, 0.6)
    good = _records(3, p, seed=1)
    blank = [RequestRecord(rid="x", prompt="x", ok=True, completion_tokens=5)]
    rep = k_invariance_check({3: good + blank, 5: _records(5, p, seed=2) + blank})
    assert rep.comparisons
    assert rep.n_requests_by_k[3] == len(good) + 1  # ok count, histogram not required


def test_per_request_mad_is_reported():
    p = (0.9, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05)
    rep = k_invariance_check({3: _records(3, p, seed=1), 7: _records(7, p, seed=2)})
    assert rep.per_request_mad
    assert all(v >= 0 for v in rep.per_request_mad.values())
