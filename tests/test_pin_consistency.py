"""Guards against the pinned SGLang revision drifting between sources.

The pin lives in three places: `configs/sglang_base.json` (source of truth for
humans), `heterospec.SGLANG_BASE_COMMIT` (used by the harness at runtime), and
this test. If they disagree, metadata.json would silently record the wrong
commit, which is the exact failure mode docs/02-reproducibility.md exists to
prevent.
"""

import json
from pathlib import Path

import heterospec

REPO_ROOT = Path(__file__).resolve().parents[1]
PIN_PATH = REPO_ROOT / "configs" / "sglang_base.json"


def _pin() -> dict:
    with PIN_PATH.open() as f:
        return json.load(f)


def test_pin_file_exists():
    assert PIN_PATH.is_file(), f"missing pin file: {PIN_PATH}"


def test_package_commit_matches_pin_file():
    pin = _pin()
    assert heterospec.SGLANG_BASE_COMMIT == pin["base_commit"], (
        "heterospec.SGLANG_BASE_COMMIT disagrees with configs/sglang_base.json; "
        "update both together"
    )


def test_short_commit_is_prefix_of_full():
    pin = _pin()
    assert pin["base_commit"].startswith(pin["base_commit_short"])


def test_commit_is_a_full_sha():
    commit = _pin()["base_commit"]
    assert len(commit) == 40, f"expected a full 40-char sha, got {commit!r}"
    assert all(c in "0123456789abcdef" for c in commit)


def test_unmerged_cost_aware_pr_is_marked_not_merged():
    """PR #28045 must never be described as part of the SGLang baseline.

    README section 8 depends on this. If it ever merges upstream, this test
    fails on purpose so the docs get updated rather than silently going stale.
    """
    prs = _pin()["relevant_upstream_prs"]
    assert "NOT MERGED" in prs["28045"]
    assert "28045" not in prs["37274"]
