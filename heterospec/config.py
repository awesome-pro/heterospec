"""Launch configuration for SGLang servers, and validation of adaptive eligibility.

Two jobs:

1. Turn a declarative config into the exact CLI invocation, so a GPU session is
   execution rather than authoring. Every command that produced a result is
   recoverable from `results/<run>/metadata.json`.
2. **Refuse to silently measure the wrong thing.** SGLang disables adaptive mode
   without raising when a config is ineligible (``adaptive_unsupported_reason``
   in ``speculative/adaptive_spec_params.py``, logged and then
   ``speculative_adaptive=False``). A run intended as "SGLang adaptive" that
   actually ran static speculation would look plausible and be worthless. We
   mirror those conditions and fail loudly instead.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Adaptive eligibility, mirrored from SGLang
# ---------------------------------------------------------------------------

ADAPTIVE_SUPPORTED_ALGORITHMS = ("EAGLE", "EAGLE3")


def has_flag(args: list[str], flag: str) -> bool:
    """Whether ``flag`` appears in an argv list, in either ``--x`` or ``--x=v`` form."""
    return any(a == flag or a.startswith(f"{flag}=") for a in args)


def flag_value(args: list[str], flag: str) -> str | None:
    """Value of ``flag`` in an argv list, or ``None`` if absent.

    Handles both ``--flag value`` and ``--flag=value``.
    """
    for i, a in enumerate(args):
        if a == flag:
            return args[i + 1] if i + 1 < len(args) else None
        if a.startswith(f"{flag}="):
            return a.split("=", 1)[1]
    return None


def adaptive_unsupported_reasons(
    *,
    speculative_algorithm: str | None,
    speculative_eagle_topk: int | None,
    enable_dp_attention: bool = False,
    enable_multi_layer_eagle: bool = False,
    enable_two_batch_overlap: bool = False,
    enable_pdmux: bool = False,
) -> list[str]:
    """Return every reason SGLang would disable adaptive mode. Empty == eligible.

    Mirrors ``adaptive_unsupported_reason`` at SGLang 66ce8c55cc. Returns *all*
    reasons rather than the first, so a misconfigured launch reports everything
    wrong with it at once.
    """
    reasons: list[str] = []

    if speculative_algorithm not in ADAPTIVE_SUPPORTED_ALGORITHMS:
        reasons.append(
            f"speculative_algorithm={speculative_algorithm!r} "
            f"(only {ADAPTIVE_SUPPORTED_ALGORITHMS} are supported)"
        )
    if speculative_eagle_topk is not None and speculative_eagle_topk != 1:
        reasons.append(f"speculative_eagle_topk={speculative_eagle_topk} (only topk=1)")
    if enable_dp_attention:
        reasons.append(
            "enable_dp_attention=True (tier decisions are not synchronized "
            "across DP ranks)"
        )
    if enable_multi_layer_eagle:
        reasons.append(
            "enable_multi_layer_eagle=True (MultiLayerEagleWorkerV2 does not "
            "implement adaptive)"
        )
    if enable_two_batch_overlap:
        reasons.append(
            "enable_two_batch_overlap=True (adaptive state swap would discard "
            "the TboAttnBackend wrapper)"
        )
    if enable_pdmux:
        reasons.append(
            "enable_pdmux=True (adaptive state swap does not update "
            "decode_attn_backend_group)"
        )
    return reasons


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelConfig:
    """Target + draft pair. Pinned; not a free experimental variable."""

    target: str
    draft: str | None = None
    dtype: str = "float16"
    attention_backend: str = "triton"
    mem_fraction_static: float = 0.7
    cuda_graph_max_bs_decode: int = 64
    cuda_graph_backend_prefill: str = "disabled"
    extra_args: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ModelConfig:
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass(frozen=True)
class SpecConfig:
    """Speculative settings. ``None`` on a LaunchConfig means no speculation."""

    algorithm: str
    num_steps: int
    num_draft_tokens: int
    eagle_topk: int = 1
    draft_model: str | None = None
    adaptive: bool = False
    adaptive_config: str | None = None
    adaptive_config_inline: dict[str, Any] | None = None

    def validate(self) -> None:
        """Catch internally-inconsistent configs before they reach a GPU."""
        if self.algorithm not in ADAPTIVE_SUPPORTED_ALGORITHMS:
            raise ValueError(f"unsupported algorithm: {self.algorithm!r}")
        if self.num_steps < 0:
            raise ValueError(f"num_steps must be >= 0, got {self.num_steps}")
        if self.eagle_topk != 1:
            raise ValueError(
                f"eagle_topk must be 1 for this project, got {self.eagle_topk}"
            )
        # EAGLE topk=1 chain drafting proposes one token beyond the step count.
        if self.num_draft_tokens != self.num_steps + 1:
            raise ValueError(
                f"num_draft_tokens must be num_steps + 1 "
                f"({self.num_steps + 1}), got {self.num_draft_tokens}"
            )
        if self.adaptive and self.num_steps == 0:
            # SGLang permits a step-0 *tier*, but the launch config's initial
            # steps must be positive for the ladder to be meaningful.
            raise ValueError("adaptive config must start at num_steps >= 1")


@dataclass(frozen=True)
class LaunchConfig:
    """A complete, reproducible server launch."""

    id: str
    description: str
    model: ModelConfig
    spec: SpecConfig | None = None
    port: int = 30000
    host: str = "127.0.0.1"
    enable_metrics: bool = True

    def adaptive_unsupported_reasons(self) -> list[str]:
        """Every reason adaptive mode would be disabled, including via extra_args.

        Flags hidden in ``extra_args`` are just as capable of silently turning
        adaptive mode off as the structured fields are, so they are scanned for
        too.
        """
        if self.spec is None or not self.spec.adaptive:
            return []
        return adaptive_unsupported_reasons(
            speculative_algorithm=self.spec.algorithm,
            speculative_eagle_topk=self.spec.eagle_topk,
            enable_dp_attention=has_flag(
                self.model.extra_args, "--enable-dp-attention"
            ),
            enable_multi_layer_eagle=has_flag(
                self.model.extra_args, "--enable-multi-layer-eagle"
            ),
            enable_two_batch_overlap=has_flag(
                self.model.extra_args, "--enable-two-batch-overlap"
            ),
            enable_pdmux=has_flag(self.model.extra_args, "--enable-pdmux"),
        )

    def validate(self) -> None:
        if self.spec is not None:
            self.spec.validate()
        reasons = self.adaptive_unsupported_reasons()
        if reasons:
            raise ValueError(
                f"launch config {self.id!r} requests adaptive mode but SGLang "
                f"would silently disable it: " + "; ".join(reasons)
            )

    def to_cli_args(self) -> list[str]:
        """The exact ``python -m sglang.launch_server`` argv for this config."""
        self.validate()
        m = self.model
        args = [
            "--model-path",
            m.target,
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--dtype",
            m.dtype,
            "--attention-backend",
            m.attention_backend,
            "--mem-fraction-static",
            str(m.mem_fraction_static),
            "--cuda-graph-max-bs-decode",
            str(m.cuda_graph_max_bs_decode),
            f"--cuda-graph-backend-prefill={m.cuda_graph_backend_prefill}",
        ]
        if self.enable_metrics:
            args.append("--enable-metrics")
        args.extend(m.extra_args)

        if self.spec is not None:
            s = self.spec
            draft = s.draft_model or m.draft
            if draft is None:
                raise ValueError(
                    f"launch config {self.id!r} has speculation but no draft model"
                )
            args += [
                "--speculative-algorithm",
                s.algorithm,
                "--speculative-draft-model-path",
                draft,
                "--speculative-num-steps",
                str(s.num_steps),
                "--speculative-eagle-topk",
                str(s.eagle_topk),
                "--speculative-num-draft-tokens",
                str(s.num_draft_tokens),
            ]
            if s.adaptive:
                args.append("--speculative-adaptive")
                if s.adaptive_config is not None:
                    args += ["--speculative-adaptive-config", s.adaptive_config]
        return args

    def command(self, python: str = "python3") -> str:
        return " ".join([python, "-m", "sglang.launch_server", *self.to_cli_args()])

    def to_dict(self) -> dict[str, Any]:
        """Serializable form recorded in results metadata."""
        return {
            "id": self.id,
            "description": self.description,
            "model": {
                "target": self.model.target,
                "draft": self.model.draft,
                "dtype": self.model.dtype,
                "attention_backend": self.model.attention_backend,
                "mem_fraction_static": self.model.mem_fraction_static,
                "cuda_graph_max_bs_decode": self.model.cuda_graph_max_bs_decode,
                "cuda_graph_backend_prefill": (self.model.cuda_graph_backend_prefill),
                "extra_args": list(self.model.extra_args),
            },
            "spec": None
            if self.spec is None
            else {
                "algorithm": self.spec.algorithm,
                "num_steps": self.spec.num_steps,
                "num_draft_tokens": self.spec.num_draft_tokens,
                "eagle_topk": self.spec.eagle_topk,
                "draft_model": self.spec.draft_model or self.model.draft,
                "adaptive": self.spec.adaptive,
                "adaptive_config": self.spec.adaptive_config,
                "adaptive_config_inline": self.spec.adaptive_config_inline,
            },
            "cli_args": self.to_cli_args(),
        }


def validate_adaptive_config(cfg: dict[str, Any]) -> None:
    """Validate an adaptive config against SGLang's own rules.

    Mirrors ``_load_adaptive_config`` at SGLang 66ce8c55cc, so a config that
    passes here will not be rejected at server startup on a rented GPU. Integer
    string keys define batch-size slots and are the only keys that may carry
    ``candidate_steps``; other keys are global overrides.
    """
    bs_entries = {k: v for k, v in cfg.items() if k.isdigit()}
    if not bs_entries:
        raise ValueError(
            "adaptive config must contain at least one integer-string BS key, "
            f'e.g. {{"1": {{"candidate_steps": [1,3,7]}}}}. Got keys: {list(cfg)}'
        )

    for key, entry in bs_entries.items():
        if not isinstance(entry, dict):
            raise ValueError(f"BS {key}: slot must be an object, got {entry!r}")
        steps = entry.get("candidate_steps")
        if not isinstance(steps, list) or not steps:
            raise ValueError(
                f"BS {key}: candidate_steps must be a non-empty list, got {steps!r}"
            )
        if not all(
            isinstance(s, int) and not isinstance(s, bool) and s >= 0 for s in steps
        ):
            raise ValueError(
                f"BS {key}: candidate_steps must be non-negative ints, got {steps!r}"
            )

    for gkey in ("ema_alpha", "warmup_batches", "update_interval"):
        if gkey in cfg and not isinstance(cfg[gkey], (int, float)):
            raise ValueError(f"{gkey} must be numeric, got {cfg[gkey]!r}")
    if "ema_alpha" in cfg and not 0.0 < float(cfg["ema_alpha"]) <= 1.0:
        raise ValueError(f"ema_alpha must be in (0, 1], got {cfg['ema_alpha']!r}")


def load_adaptive_config(path: str | Path) -> dict[str, Any]:
    """Load and validate an adaptive config JSON file."""
    with Path(path).open() as f:
        cfg = json.load(f)
    cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}
    validate_adaptive_config(cfg)
    return cfg


def resolve_candidate_steps(cfg: dict[str, Any]) -> list[int]:
    """Union of every BS slot's candidate steps -- the tiers to build states for."""
    steps: set[int] = set()
    for key, entry in cfg.items():
        if key.isdigit():
            steps.update(entry["candidate_steps"])
    return sorted(steps)


def load_launch_config(path: str | Path) -> LaunchConfig:
    """Load a launch config JSON file.

    Accepts either a single object or ``{"model": ..., "launches": [...]}``, in
    which case the first launch is returned. Use :func:`load_launch_configs` to
    get them all.
    """
    configs = load_launch_configs(path)
    if not configs:
        raise ValueError(f"no launch configs found in {path}")
    return configs[0]


def load_launch_configs(path: str | Path) -> list[LaunchConfig]:
    """Load every launch config in a file."""
    with Path(path).open() as f:
        raw = json.load(f)

    model = ModelConfig.from_dict(raw["model"])
    launches = raw.get("launches")
    if launches is None:
        launches = [raw["launch"]] if "launch" in raw else [raw]

    out: list[LaunchConfig] = []
    for entry in launches:
        spec_raw = entry.get("spec")
        spec = SpecConfig(**spec_raw) if spec_raw is not None else None
        out.append(
            LaunchConfig(
                id=entry["id"],
                description=entry.get("description", ""),
                model=model,
                spec=spec,
                port=entry.get("port", 30000),
                host=entry.get("host", "127.0.0.1"),
                enable_metrics=entry.get("enable_metrics", True),
            )
        )
    return out
