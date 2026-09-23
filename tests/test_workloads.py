"""Tests for workload construction.

The load-bearing property is that mixed workloads actually interleave. If class
composition were emitted in blocks, every batch would be homogeneous and the
"mixed" families would silently measure two separate homogeneous phases --
producing a null result that looks like evidence against the hypothesis.
"""

import pytest

from heterospec.workloads import (
    PROMPT_CLASSES,
    Phase,
    WorkloadSpec,
    build_plan,
    get_workload,
    summarise_plan,
    workload_names,
)

# ---------------------------------------------------------------------------
# Prompt classes
# ---------------------------------------------------------------------------


def test_every_class_has_multiple_prompts():
    """Multiple prompts per class avoid a single prompt driving the result."""
    for name, cls in PROMPT_CLASSES.items():
        assert len(cls.prompts) >= 4, f"{name} has only {len(cls.prompts)} prompts"


def test_classes_are_named_neutrally_not_by_expected_acceptance():
    """Code/reasoning/chat must not be pre-labelled high or low."""
    for name in ("code", "reasoning", "chat"):
        assert "unmeasured" in PROMPT_CLASSES[name].hypothesis
    assert PROMPT_CLASSES["repetitive"].hypothesis == "expected high acceptance"
    assert PROMPT_CLASSES["open_ended"].hypothesis == "expected low acceptance"


def test_prompts_are_nonempty_strings():
    for cls in PROMPT_CLASSES.values():
        for p in cls.prompts:
            assert isinstance(p, str) and p.strip()


# ---------------------------------------------------------------------------
# Workload definitions
# ---------------------------------------------------------------------------


def test_expected_workload_families_exist():
    assert set(workload_names()) == {
        "high",
        "low",
        "mixed_75_25",
        "mixed_50_50",
        "mixed_25_75",
        "phase_shift",
        "real_mixed",
    }


def test_all_workloads_construct_and_reference_known_classes():
    for name in workload_names():
        get_workload(name)  # must not raise


def test_unknown_workload_raises_with_helpful_message():
    with pytest.raises(KeyError, match="available"):
        get_workload("nope")


def test_unknown_prompt_class_rejected():
    with pytest.raises(ValueError, match="unknown prompt classes"):
        WorkloadSpec(
            name="bad",
            description="",
            phases=(Phase({"not_a_class": 1.0}, 5),),
        )


def test_zero_or_negative_requests_rejected():
    with pytest.raises(ValueError, match="n_requests must be > 0"):
        WorkloadSpec(name="bad", description="", phases=(Phase({"code": 1.0}, 0),))


def test_zero_weight_rejected():
    with pytest.raises(ValueError, match="sum to <= 0"):
        WorkloadSpec(name="bad", description="", phases=(Phase({"code": 0.0}, 5),))


# ---------------------------------------------------------------------------
# Scaling
# ---------------------------------------------------------------------------


def test_phase_shift_has_four_phases():
    w = get_workload("phase_shift")
    assert len(w.phases) == 4
    assert w.is_phase_shifting
    assert [p.label for p in w.phases] == ["low_1", "high_1", "low_2", "high_2"]


def test_scaled_sums_to_requested_total():
    for name in workload_names():
        for n in (40, 137, 500):
            scaled = get_workload(name).scaled(n)
            assert scaled.total_requests == n, f"{name} at n={n}"


def test_scaled_preserves_two_class_proportions():
    scaled = get_workload("mixed_75_25").scaled(400)
    composition = scaled.phases[0].composition
    assert composition["repetitive"] == 0.75
    assert composition["open_ended"] == 0.25


def test_scaling_below_phase_count_is_rejected():
    """Four phases cannot be filled by three requests; dropping one silently
    would remove a composition regime from the experiment."""
    with pytest.raises(ValueError, match="too few to fill each phase"):
        get_workload("phase_shift").scaled(3)


def test_scaled_phase_shift_keeps_every_phase_nonempty():
    scaled = get_workload("phase_shift").scaled(9)
    assert all(p.n_requests >= 1 for p in scaled.phases)
    assert scaled.total_requests == 9


# ---------------------------------------------------------------------------
# Plan construction -- determinism and counts
# ---------------------------------------------------------------------------


def test_build_plan_is_deterministic_for_a_seed():
    a = build_plan("mixed_50_50", 100, seed=7)
    b = build_plan("mixed_50_50", 100, seed=7)
    assert [(r.rid, r.prompt, r.seed) for r in a] == [
        (r.rid, r.prompt, r.seed) for r in b
    ]


def test_different_seeds_give_different_prompt_draws():
    a = build_plan("high", 60, seed=1)
    b = build_plan("high", 60, seed=2)
    assert [r.prompt for r in a] != [r.prompt for r in b]


def test_plan_length_and_rid_uniqueness():
    plan = build_plan("mixed_50_50", 250, seed=0)
    assert len(plan) == 250
    assert len({r.rid for r in plan}) == 250
    assert [r.index for r in plan] == list(range(250))


def test_plan_respects_exact_class_counts_at_75_25():
    plan = build_plan("mixed_75_25", 200, seed=0)
    s = summarise_plan(plan)
    assert s.by_class["repetitive"] == 150
    assert s.by_class["open_ended"] == 50


def test_plan_respects_exact_class_counts_at_25_75():
    plan = build_plan("mixed_25_75", 200, seed=0)
    s = summarise_plan(plan)
    assert s.by_class["repetitive"] == 50
    assert s.by_class["open_ended"] == 150


def test_homogeneous_workloads_contain_one_class():
    assert set(summarise_plan(build_plan("high", 50)).by_class) == {"repetitive"}
    assert set(summarise_plan(build_plan("low", 50)).by_class) == {"open_ended"}


def test_real_mixed_covers_three_classes():
    s = summarise_plan(build_plan("real_mixed", 99, seed=3))
    assert set(s.by_class) == {"code", "reasoning", "chat"}


# ---------------------------------------------------------------------------
# Interleaving -- the property that makes "mixed" mean anything
# ---------------------------------------------------------------------------


def test_mixed_plan_does_not_block_by_class():
    """No long same-class run, which would make batch windows homogeneous."""
    for name in ("mixed_50_50", "mixed_75_25", "mixed_25_75", "real_mixed"):
        s = summarise_plan(build_plan(name, 300, seed=0))
        assert s.max_run_of_single_class <= 4, (
            f"{name} has a run of {s.max_run_of_single_class} same-class "
            f"requests; arrivals are clustered, so batches would not mix"
        )


def test_every_window_of_eight_contains_both_classes_at_50_50():
    """With realistic batch sizes, both classes must be present throughout."""
    plan = build_plan("mixed_50_50", 200, seed=0)
    classes = [r.prompt_class for r in plan]
    for start in range(0, len(classes) - 8, 8):
        window = set(classes[start : start + 8])
        assert window == {"repetitive", "open_ended"}, f"window at {start}: {window}"


def test_classes_appear_in_first_small_window_for_all_mixed_families():
    """A concurrency-8 run must see the mixture from the very first batch."""
    for name in ("mixed_50_50", "mixed_75_25", "mixed_25_75", "real_mixed"):
        plan = build_plan(name, 120, seed=11)
        first = {r.prompt_class for r in plan[:8]}
        assert len(first) >= 2, f"{name} first window single-class: {first}"
        assert first == {r.prompt_class for r in plan[:8] if r.prompt_class in first}


def test_real_mixed_first_window_has_all_three_classes():
    plan = build_plan("real_mixed", 120, seed=11)
    assert {r.prompt_class for r in plan[:8]} == {"code", "reasoning", "chat"}


# ---------------------------------------------------------------------------
# Phase shift
# ---------------------------------------------------------------------------


def test_phase_shift_phases_are_contiguous_and_labelled():
    plan = build_plan("phase_shift", 120, seed=0)
    s = summarise_plan(plan)
    assert s.by_phase == {"low_1": 30, "high_1": 30, "low_2": 30, "high_2": 30}
    labels = [r.phase_label for r in plan]
    assert labels == sorted(labels, key=lambda x: list(s.by_phase).index(x))


def test_phase_shift_alternates_classes_no_self_run_across_phases():
    plan = build_plan("phase_shift", 120, seed=0)
    classes = [r.prompt_class for r in plan]
    # Within a phase it is homogeneous by design, so runs are phase-sized.
    assert classes[:30] == ["open_ended"] * 30
    assert classes[30:60] == ["repetitive"] * 30


# ---------------------------------------------------------------------------
# Metadata carried on each request
# ---------------------------------------------------------------------------


def test_requests_carry_reproducibility_fields():
    r = build_plan("mixed_50_50", 10, seed=5)[0]
    assert r.workload == "mixed_50_50"
    assert r.prompt_class in PROMPT_CLASSES
    assert r.max_new_tokens > 0
    assert 0 <= r.seed < 2**31


def test_max_new_tokens_override_applies_to_all_requests():
    plan = build_plan("high", 20, max_new_tokens=64)
    assert {r.max_new_tokens for r in plan} == {64}


def test_rid_prefix_is_configurable():
    plan = build_plan("high", 5, rid_prefix="het")
    assert plan[0].rid == "het00000"


def test_summarise_plan_to_dict_is_json_serialisable():
    import json

    d = summarise_plan(build_plan("mixed_50_50", 40, seed=0)).to_dict()
    assert json.loads(json.dumps(d))["total"] == 40
