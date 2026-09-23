"""Tests for the result schema, reproducibility gates and JSONL IO."""

import json
from datetime import datetime

import pytest

from heterospec import SGLANG_BASE_COMMIT
from heterospec.records import (
    IterationRecord,
    RequestRecord,
    RunMetadata,
    load_iteration_records,
    load_request_records,
    make_run_dir,
    min_k_proposed_per_request,
    read_json,
    read_jsonl,
    summarise_records,
    write_json,
    write_jsonl,
)

# histogram[J] = rounds accepting exactly J drafts (bonus token excluded).
H = [2, 4, 8, 5, 1]  # n=20, mean=1.95


def _spec_record(rid="r1", **kw) -> RequestRecord:
    base = dict(
        rid=rid,
        prompt_class="repetitive",
        spec_verify_ct=20,
        spec_num_correct_drafts=39,
        spec_correct_drafts_histogram=list(H),
        completion_tokens=59,  # 39 accepted drafts + 20 bonus tokens
        inferred_k_static=4,
        latency_s=1.0,
        ok=True,
    )
    base.update(kw)
    return RequestRecord(**base)


# ---------------------------------------------------------------------------
# RequestRecord accessors
# ---------------------------------------------------------------------------


def test_prompt_hash_is_computed_automatically():
    r = RequestRecord(rid="r", prompt="hello")
    assert r.prompt_sha256 and len(r.prompt_sha256) == 64
    assert r.prompt_sha256 == RequestRecord(rid="r", prompt="hello").prompt_sha256


def test_explicit_prompt_hash_is_preserved():
    r = RequestRecord(rid="r", prompt="hello", prompt_sha256="deadbeef")
    assert r.prompt_sha256 == "deadbeef"


def test_non_speculative_record_has_no_histogram():
    r = RequestRecord(rid="r", prompt="x", completion_tokens=10)
    assert not r.has_histogram
    assert not r.is_speculative
    assert r.accept_stats() is None
    assert r.mean_accepted_drafts() is None
    assert r.position_acceptance() is None
    assert r.histogram_consistent()[0] is True


def test_accept_stats_matches_hand_computed_values():
    st = _spec_record().accept_stats()
    assert st is not None
    assert st.n_rounds == 20
    assert st.mean == pytest.approx(1.95)
    assert st.max_observed == 4


def test_position_acceptance_uses_inferred_static_k():
    """A static-K run needs no extra argument; the record knows its own K."""
    pa = _spec_record().position_acceptance()
    assert pa is not None
    assert not pa.k_confounded
    assert pa.safe_up_to == 4
    assert pa.valid_survival() == pytest.approx((0.9, 0.7, 0.3, 0.05))


def test_position_acceptance_confounded_without_k():
    """Adaptive run, no trace: must be marked confounded, not guessed."""
    r = _spec_record(inferred_k_static=None)
    pa = r.position_acceptance()
    assert pa is not None
    assert pa.k_confounded


def test_position_acceptance_explicit_min_k_from_trace():
    r = _spec_record(inferred_k_static=None)
    pa = r.position_acceptance(min_k_proposed=3)
    assert pa is not None and not pa.k_confounded and pa.safe_up_to == 3


def test_acceptance_length_includes_bonus_token():
    """completion_tokens/spec_verify_ct = (39 drafts + 20 bonus)/20 = 2.95."""
    r = _spec_record()
    assert r.acceptance_length() == pytest.approx(2.95)
    # ...and is deliberately NOT the same as mean accepted drafts (1.95)
    assert r.mean_accepted_drafts() == pytest.approx(1.95)


def test_acceptance_length_none_without_spec_or_tokens():
    assert RequestRecord(rid="r", completion_tokens=10).acceptance_length() is None
    assert _spec_record(completion_tokens=None).acceptance_length() is None


def test_histogram_consistency_flags_mismatch():
    ok, msg = _spec_record().histogram_consistent()
    assert ok, msg
    bad = _spec_record(spec_verify_ct=17)
    ok, msg = bad.histogram_consistent()
    assert not ok and "INCONSISTENT" in msg


def test_effective_k_prefers_static_then_mean():
    assert _spec_record().effective_k() == 4.0
    assert _spec_record(inferred_k_static=None, mean_active_k=2.5).effective_k() == 2.5
    assert (
        _spec_record(inferred_k_static=None, mean_active_k=None).effective_k() is None
    )


def test_record_round_trip_is_lossless():
    r = _spec_record()
    assert RequestRecord.from_dict(r.to_dict()) == r


def test_from_dict_ignores_unknown_future_fields():
    d = _spec_record().to_dict()
    d["some_field_added_later"] = 123
    assert RequestRecord.from_dict(d).rid == "r1"


# ---------------------------------------------------------------------------
# Iteration records
# ---------------------------------------------------------------------------


def test_iteration_record_accessors():
    it = IterationRecord(
        iteration=182,
        batch_size=4,
        active_k=5,
        requests=[
            {"rid": "r1", "accepted": 5},
            {"rid": "r2", "accepted": 1},
            {"rid": "r7", "accepted": 0},
            {"rid": "r12", "accepted": 4},
        ],
    )
    assert it.rids == ["r1", "r2", "r7", "r12"]
    assert it.accepted == [5, 1, 0, 4]
    assert it.accepted_for("r7") == 0
    assert it.accepted_for("nope") is None
    assert IterationRecord.from_dict(it.to_dict()) == it


def test_min_k_proposed_takes_the_minimum_across_iterations():
    """Confounding depends on the smallest K a request was ever subject to."""
    iterations = [
        IterationRecord(
            iteration=0,
            batch_size=2,
            active_k=7,
            requests=[{"rid": "a", "accepted": 7}, {"rid": "b", "accepted": 1}],
        ),
        IterationRecord(
            iteration=1,
            batch_size=2,
            active_k=2,
            requests=[{"rid": "a", "accepted": 2}, {"rid": "b", "accepted": 0}],
        ),
        IterationRecord(
            iteration=2,
            batch_size=2,
            active_k=5,
            requests=[{"rid": "a", "accepted": 5}, {"rid": "b", "accepted": 0}],
        ),
    ]
    assert min_k_proposed_per_request(iterations) == {"a": 2, "b": 2}


def test_min_k_proposed_single_iteration():
    iterations = [
        IterationRecord(
            iteration=0,
            batch_size=1,
            active_k=3,
            requests=[{"rid": "a", "accepted": 3}],
        )
    ]
    assert min_k_proposed_per_request(iterations) == {"a": 3}


def test_min_k_proposed_empty():
    assert min_k_proposed_per_request([]) == {}


# ---------------------------------------------------------------------------
# Reproducibility gates -- the citable() logic
# ---------------------------------------------------------------------------


#: A clean research-patch commit, i.e. NOT the pinned base. Session 1 runs on a
#: branch like this, so it must be citable.
_PATCH_SHA = "f9281ec128" + "a" * 30


def _meta(dirty, commit=SGLANG_BASE_COMMIT, branch="heterospec/base") -> RunMetadata:
    return RunMetadata.new(
        run_id="t",
        policy_id="static_k3",
        workload_name="mixed_50_50",
        environment={
            "sglang_git": {
                "commit": commit,
                "dirty": dirty,
                "branch": branch,
            }
        },
    )


def test_clean_pinned_commit_is_citable():
    ok, reason = _meta(False).citable()
    assert ok, reason
    assert "pinned base" in reason


def test_dirty_tree_is_not_citable():
    ok, reason = _meta(True).citable()
    assert not ok and "uncommitted" in reason


def test_unknown_dirty_state_is_not_citable():
    ok, reason = _meta(None).citable()
    assert not ok and "unknown" in reason


def test_unknown_commit_is_not_citable():
    ok, reason = _meta(False, commit=None).citable()
    assert not ok and "unknown" in reason


def test_clean_research_patch_is_citable_off_base():
    """The whole point of the three-part model.

    Session 1 runs the telemetry branch, which is deliberately NOT the pinned
    base. Treating "off base" as non-citable would have marked every Session 1
    result unusable -- by the project's own tooling.
    """
    ok, reason = _meta(False, commit=_PATCH_SHA).citable()
    assert ok, reason
    assert "experiment patch" in reason
    assert _PATCH_SHA[:12] in reason


def test_off_base_run_reports_base_and_patch_separately():
    m = _meta(False, commit=_PATCH_SHA)
    assert m.base_sha == SGLANG_BASE_COMMIT
    assert m.experiment_patch_sha == _PATCH_SHA
    assert m.working_tree_dirty is False
    assert m.runs_on_pinned_base is False
    assert m.sglang_branch == "heterospec/base"


def test_runs_on_pinned_base_is_true_for_the_base_commit():
    m = _meta(False)
    assert m.runs_on_pinned_base is True
    assert m.commit_mismatch() is False


def test_provenance_is_surfaced_at_top_level():
    """metadata.json must be readable without unpacking `environment`."""
    d = _meta(False, commit=_PATCH_SHA).to_dict()
    assert d["base_sha"] == SGLANG_BASE_COMMIT
    assert d["experiment_patch_sha"] == _PATCH_SHA
    assert d["working_tree_dirty"] is False
    assert d["runs_on_pinned_base"] is False
    assert d["citable"] is True


def test_to_dict_includes_citability_verdict():
    d = _meta(False).to_dict()
    assert d["citable"] is True and "citable_reason" in d
    d2 = _meta(True).to_dict()
    assert d2["citable"] is False


def test_commit_mismatch_tolerates_short_prefix_match():
    m = _meta(False, commit=SGLANG_BASE_COMMIT + "extra")
    assert not m.commit_mismatch()


def test_metadata_round_trip():
    m = _meta(False)
    m.notes = "hello"
    assert RunMetadata.from_dict(m.to_dict()).notes == "hello"


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------


def test_make_run_dir_naming(tmp_path):
    d = make_run_dir(
        tmp_path,
        workload="mixed_50_50",
        policy="static_k3",
        commit="66ce8c55cc6c656225d33f26bbcaeeac8ba92e93",
        when=datetime(2026, 9, 23),
    )
    assert d.name == "2026-09-23_mixed_50_50_static_k3_66ce8c55c"
    assert d.is_dir()


def test_make_run_dir_sanitises_path_characters(tmp_path):
    d = make_run_dir(
        tmp_path,
        workload="a/b c",
        policy="x:y",
        commit="abc",
        when=datetime(2026, 1, 2),
    )
    assert "/" not in d.name and " " not in d.name and ":" not in d.name


def test_write_and_read_json_round_trip(tmp_path):
    p = tmp_path / "nested" / "metadata.json"
    write_json(p, {"b": 1, "a": [1, 2]})
    assert read_json(p) == {"b": 1, "a": [1, 2]}


def test_jsonl_round_trip_preserves_records(tmp_path):
    p = tmp_path / "requests.jsonl"
    rows = [_spec_record("r1"), _spec_record("r2")]
    assert write_jsonl(p, rows) == 2
    loaded = load_request_records(p)
    assert [r.rid for r in loaded] == ["r1", "r2"]
    assert loaded[0] == rows[0]


def test_jsonl_round_trip_for_iteration_records(tmp_path):
    p = tmp_path / "speculative_steps.jsonl"
    rows = [
        IterationRecord(
            iteration=0,
            batch_size=1,
            active_k=3,
            requests=[{"rid": "a", "accepted": 2}],
        )
    ]
    write_jsonl(p, rows)
    loaded = load_iteration_records(p)
    assert loaded == rows


def test_read_jsonl_of_missing_file_is_empty():
    assert list(read_jsonl("/nonexistent/path.jsonl")) == []


def test_jsonl_skips_blank_lines(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text('{"a": 1}\n\n{"a": 2}\n')
    assert [r["a"] for r in read_jsonl(p)] == [1, 2]


def test_write_jsonl_accepts_plain_dicts(tmp_path):
    p = tmp_path / "x.jsonl"
    write_jsonl(p, [{"a": 1}])
    assert json.loads(p.read_text().strip()) == {"a": 1}


# ---------------------------------------------------------------------------
# Aggregate summary
# ---------------------------------------------------------------------------


def test_summarise_records_counts_ok_and_failed():
    recs = [
        _spec_record("r1"),
        _spec_record("r2"),
        RequestRecord(rid="r3", ok=False, error="timeout"),
    ]
    s = summarise_records(recs)
    assert s["n_requests"] == 3
    assert s["n_ok"] == 2
    assert s["n_failed"] == 1
    assert s["n_speculative"] == 2


def test_summarise_records_computes_draft_waste_from_known_k():
    # 20 rounds at K=4 -> 80 drafts proposed, 39 accepted
    s = summarise_records([_spec_record("r1")])
    assert s["n_with_known_k"] == 1
    assert s["total_draft_tokens"] == 80
    assert s["total_accepted_drafts"] == 39
    assert s["draft_waste"] == pytest.approx((80 - 39) / 80)
    assert s["draft_waste_covers_all_speculative"] is True


def test_draft_waste_is_none_when_k_unknown_not_guessed():
    """An adaptive run without the trace must not report a fabricated waste."""
    recs = [_spec_record("r1", inferred_k_static=None, mean_active_k=None)]
    s = summarise_records(recs)
    assert s["n_with_known_k"] == 0
    assert s["draft_waste"] is None
    assert s["total_draft_tokens"] is None
    assert s["draft_waste_covers_all_speculative"] is False


def test_draft_waste_marks_partial_coverage():
    recs = [
        _spec_record("r1"),
        _spec_record("r2", inferred_k_static=None, mean_active_k=None),
    ]
    s = summarise_records(recs)
    assert s["n_with_known_k"] == 1
    assert s["draft_waste_covers_all_speculative"] is False


def test_summarise_records_groups_by_class():
    recs = [
        _spec_record("r1", prompt_class="repetitive"),
        _spec_record("r2", prompt_class="open_ended"),
        _spec_record("r3", prompt_class="repetitive"),
    ]
    s = summarise_records(recs)
    assert set(s["by_class"]) == {"repetitive", "open_ended"}
    assert s["by_class"]["repetitive"]["n"] == 2


def test_summarise_records_reports_histogram_inconsistencies():
    recs = [_spec_record("good"), _spec_record("bad", spec_verify_ct=99)]
    s = summarise_records(recs)
    assert s["histogram_inconsistencies"] == ["bad"]


def test_summarise_records_handles_empty():
    s = summarise_records([])
    assert s["n_requests"] == 0
    assert s["draft_waste"] is None
    assert s["mean_latency_s"] is None


def test_summarise_records_is_json_serialisable():
    s = summarise_records([_spec_record("r1")])
    assert json.loads(json.dumps(s))["n_requests"] == 1
