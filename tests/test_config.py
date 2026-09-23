"""Tests for launch-config construction and adaptive-eligibility guarding.

The highest-value test here is the one proving that a flag hidden in
``extra_args`` cannot silently disable adaptive mode: that failure would produce
a plausible-looking but worthless "SGLang adaptive" result on a rented GPU.
"""

import json
from pathlib import Path

import pytest

from heterospec.config import (
    LaunchConfig,
    ModelConfig,
    SpecConfig,
    adaptive_unsupported_reasons,
    flag_value,
    has_flag,
    load_adaptive_config,
    load_launch_configs,
    resolve_candidate_steps,
    validate_adaptive_config,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_CONFIG = REPO_ROOT / "configs" / "models" / "llama31_8b_eagle3.json"
MODEL_CONFIG_DIR = REPO_ROOT / "configs" / "models"
ADAPTIVE_DIR = REPO_ROOT / "configs" / "adaptive"
ALL_MODEL_CONFIGS = sorted(MODEL_CONFIG_DIR.glob("*.json"))


# ---------------------------------------------------------------------------
# Shipped configs must all be valid
# ---------------------------------------------------------------------------


def test_model_config_file_exists():
    assert MODEL_CONFIG.is_file()


@pytest.mark.parametrize("path", ALL_MODEL_CONFIGS, ids=lambda p: p.name)
def test_every_shipped_model_config_is_valid(path):
    """Every configuration shipped must load and pass adaptive-eligibility.

    Hardware-specific variants (e.g. a 40GB card) drift out of sync easily; this
    keeps them honest.
    """
    launches = load_launch_configs(path)
    assert launches, f"{path.name} defines no launches"
    for launch in launches:
        launch.validate()
        args = launch.to_cli_args()
        assert "--model-path" in args


@pytest.mark.parametrize("path", ALL_MODEL_CONFIGS, ids=lambda p: p.name)
def test_every_shipped_model_config_pins_the_same_model_pair(path):
    """Hardware variants may differ in memory settings, never in the model."""
    for launch in load_launch_configs(path):
        assert launch.model.target == "meta-llama/Llama-3.1-8B-Instruct"
        assert launch.model.draft == "lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B"
        assert launch.model.dtype == "float16"
        assert launch.model.attention_backend == "triton"


@pytest.mark.parametrize("path", ALL_MODEL_CONFIGS, ids=lambda p: p.name)
def test_every_shipped_model_config_offers_the_same_policies(path):
    ids = {c.id for c in load_launch_configs(path)}
    assert ids == {
        "no_spec",
        "static_k1",
        "static_k3",
        "static_k5",
        "static_k7",
        "sglang_adaptive",
    }


def test_all_shipped_launches_validate():
    """Every baseline we intend to run must pass validation."""
    launches = load_launch_configs(MODEL_CONFIG)
    assert launches, "no launches found"
    for launch in launches:
        launch.validate()


def test_expected_baseline_ids_present():
    ids = {c.id for c in load_launch_configs(MODEL_CONFIG)}
    assert ids == {
        "no_spec",
        "static_k1",
        "static_k3",
        "static_k5",
        "static_k7",
        "sglang_adaptive",
    }


def test_pinned_model_pair_matches_sglang_defaults():
    """These are SGLang's canonical EAGLE3 test defaults; changing them changes
    the experiment, so pin the test to the exact strings."""
    cfg = load_launch_configs(MODEL_CONFIG)[0]
    assert cfg.model.target == "meta-llama/Llama-3.1-8B-Instruct"
    assert cfg.model.draft == "lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B"
    assert cfg.model.dtype == "float16"
    assert cfg.model.attention_backend == "triton"


def test_shipped_adaptive_configs_are_valid():
    files = sorted(ADAPTIVE_DIR.glob("*.json"))
    assert files, "no adaptive configs shipped"
    for path in files:
        cfg = load_adaptive_config(path)
        assert resolve_candidate_steps(cfg), f"{path.name} has no candidate steps"


def test_heterospec_probe_covers_zero_and_seven():
    """The probe ladder exists to make the controller switch; it needs range."""
    cfg = load_adaptive_config(ADAPTIVE_DIR / "heterospec_probe.json")
    steps = resolve_candidate_steps(cfg)
    assert 0 in steps, "probe should be able to disable speculation"
    assert 7 in steps, "probe should reach the deepest tier"


def test_shipped_default_matches_documented_sglang_ladder():
    cfg = load_adaptive_config(ADAPTIVE_DIR / "default.json")
    assert cfg["1"]["candidate_steps"] == [1, 3, 5, 7]
    assert cfg["8"]["candidate_steps"] == [0, 1, 3]
    assert cfg["32"]["candidate_steps"] == [0, 1]
    assert cfg["64"]["candidate_steps"] == [0]
    assert cfg["ema_alpha"] == 0.2
    assert cfg["warmup_batches"] == 10
    assert cfg["update_interval"] == 5


# ---------------------------------------------------------------------------
# CLI construction
# ---------------------------------------------------------------------------


def _launch(launch_id: str) -> LaunchConfig:
    return {c.id: c for c in load_launch_configs(MODEL_CONFIG)}[launch_id]


def test_static_k3_cli_args():
    args = _launch("static_k3").to_cli_args()
    assert flag_value(args, "--speculative-algorithm") == "EAGLE3"
    assert flag_value(args, "--speculative-num-steps") == "3"
    assert flag_value(args, "--speculative-num-draft-tokens") == "4"
    assert flag_value(args, "--speculative-eagle-topk") == "1"
    assert "--speculative-adaptive" not in args
    assert flag_value(args, "--model-path") == "meta-llama/Llama-3.1-8B-Instruct"


def test_no_spec_cli_has_no_speculative_flags():
    args = _launch("no_spec").to_cli_args()
    assert not any(a.startswith("--speculative") for a in args)


def test_adaptive_cli_includes_flag_and_draft_path():
    args = _launch("sglang_adaptive").to_cli_args()
    assert "--speculative-adaptive" in args
    assert (
        flag_value(args, "--speculative-draft-model-path")
        == "lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B"
    )


def test_command_is_shell_joinable():
    cmd = _launch("static_k1").command()
    assert cmd.startswith("python3 -m sglang.launch_server ")
    assert "--speculative-num-steps 1" in cmd


def test_metadata_dict_records_cli_args():
    d = _launch("static_k3").to_dict()
    assert d["cli_args"] == _launch("static_k3").to_cli_args()
    assert d["spec"]["num_steps"] == 3
    assert d["model"]["target"] == "meta-llama/Llama-3.1-8B-Instruct"


# ---------------------------------------------------------------------------
# SpecConfig invariants
# ---------------------------------------------------------------------------


def test_num_draft_tokens_must_be_steps_plus_one():
    spec = SpecConfig(algorithm="EAGLE3", num_steps=3, num_draft_tokens=3)
    with pytest.raises(ValueError, match="num_draft_tokens"):
        spec.validate()


def test_topk_must_be_one():
    spec = SpecConfig(algorithm="EAGLE3", num_steps=3, num_draft_tokens=4, eagle_topk=4)
    with pytest.raises(ValueError, match="eagle_topk"):
        spec.validate()


def test_adaptive_cannot_start_at_zero_steps():
    spec = SpecConfig(
        algorithm="EAGLE3", num_steps=0, num_draft_tokens=1, adaptive=True
    )
    with pytest.raises(ValueError, match="num_steps >= 1"):
        spec.validate()


# ---------------------------------------------------------------------------
# The silent-disable guard -- the point of this module
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "flag,reason_fragment",
    [
        ("--enable-dp-attention", "enable_dp_attention"),
        ("--enable-two-batch-overlap", "two_batch_overlap"),
        ("--enable-pdmux", "enable_pdmux"),
        ("--enable-multi-layer-eagle", "multi_layer_eagle"),
    ],
)
def test_adaptive_rejected_when_unsupported_flag_hidden_in_extra_args(
    flag, reason_fragment
):
    base = _launch("sglang_adaptive")
    launch = LaunchConfig(
        id="bad",
        description="",
        model=ModelConfig(
            target=base.model.target,
            draft=base.model.draft,
            extra_args=[flag],
        ),
        spec=base.spec,
    )
    with pytest.raises(ValueError, match="silently disable"):
        launch.validate()
    reasons = launch.adaptive_unsupported_reasons()
    assert any(reason_fragment in r for r in reasons)


def test_non_adaptive_config_is_unaffected_by_unsupported_flags():
    """A static-K run with two-batch-overlap is legitimate; only adaptive cares."""
    base = _launch("static_k3")
    launch = LaunchConfig(
        id="static_with_tbo",
        description="",
        model=ModelConfig(
            target=base.model.target, extra_args=["--enable-two-batch-overlap"]
        ),
        spec=base.spec,
    )
    launch.validate()  # must not raise


def test_adaptive_unsupported_reasons_reports_all_problems_at_once():
    reasons = adaptive_unsupported_reasons(
        speculative_algorithm="NGRAM",
        speculative_eagle_topk=4,
        enable_dp_attention=True,
    )
    assert len(reasons) == 3


def test_adaptive_unsupported_reasons_empty_for_valid_setup():
    assert (
        adaptive_unsupported_reasons(
            speculative_algorithm="EAGLE3", speculative_eagle_topk=1
        )
        == []
    )


# ---------------------------------------------------------------------------
# Adaptive-config validation mirrors SGLang's loader
# ---------------------------------------------------------------------------


def test_adaptive_config_requires_integer_key():
    with pytest.raises(ValueError, match="at least one integer-string BS key"):
        validate_adaptive_config({"ema_alpha": 0.2})


def test_adaptive_config_requires_non_empty_candidate_steps():
    with pytest.raises(ValueError, match="non-empty list"):
        validate_adaptive_config({"1": {"candidate_steps": []}})


def test_adaptive_config_rejects_negative_steps():
    with pytest.raises(ValueError, match="non-negative ints"):
        validate_adaptive_config({"1": {"candidate_steps": [1, -3]}})


def test_adaptive_config_rejects_bad_ema_alpha():
    with pytest.raises(ValueError, match="ema_alpha"):
        validate_adaptive_config({"ema_alpha": 1.5, "1": {"candidate_steps": [1]}})


def test_load_adaptive_config_strips_underscore_keys():
    cfg = load_adaptive_config(ADAPTIVE_DIR / "default.json")
    assert not any(k.startswith("_") for k in cfg)


# ---------------------------------------------------------------------------
# Small argv helpers
# ---------------------------------------------------------------------------


def test_has_flag_handles_both_forms():
    args = ["--a", "1", "--b=2"]
    assert has_flag(args, "--a")
    assert has_flag(args, "--b")
    assert not has_flag(args, "--c")


def test_flag_value_handles_both_forms():
    args = ["--a", "1", "--b=2"]
    assert flag_value(args, "--a") == "1"
    assert flag_value(args, "--b") == "2"
    assert flag_value(args, "--c") is None


def test_flag_value_does_not_confuse_similar_prefixes():
    args = ["--speculative-num-steps", "3", "--speculative-num-draft-tokens", "4"]
    assert flag_value(args, "--speculative-num-steps") == "3"
    assert flag_value(args, "--speculative-num-draft-tokens") == "4"


def test_load_launch_configs_ignores_top_level_metadata_keys():
    with MODEL_CONFIG.open() as f:
        raw = json.load(f)
    assert "_hardware_note" in raw  # present in the file...
    launches = load_launch_configs(MODEL_CONFIG)  # ...but must not break loading
    assert len(launches) == 6
