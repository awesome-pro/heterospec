"""Capture the environment a run happened in.

Reproducibility (docs/02-reproducibility.md) requires that every results
directory be interpretable without asking anyone what machine it ran on. This
module collects that, degrading gracefully so the same code runs on the macOS
harness (no NVIDIA, possibly no torch) and on a rented GPU host.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

__all__ = [
    "collect_environment",
    "git_info",
    "gpu_info",
    "python_env_info",
]


def _run(
    cmd: list[str], cwd: str | Path | None = None, timeout: int = 20
) -> str | None:
    """Run a command, returning stdout or None. Never raises."""
    try:
        p = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if p.returncode != 0:
        return None
    return p.stdout.strip()


def git_info(repo_path: str | Path) -> dict[str, Any]:
    """Commit, branch and **dirtiness** of a git checkout.

    `dirty` is the field that matters: results/README.md says a run from a
    modified tree is not citable, so this must be recorded mechanically rather
    than remembered. `status_porcelain` is truncated to keep metadata small but
    keeps enough to identify what changed.
    """
    repo = Path(repo_path)
    out: dict[str, Any] = {
        "path": str(repo),
        "commit": None,
        "commit_short": None,
        "commit_date": None,
        "subject": None,
        "branch": None,
        "describe": None,
        "dirty": None,
        "status_porcelain": None,
        "available": False,
    }
    if not (repo / ".git").exists():
        out["error"] = f"not a git checkout: {repo}"
        return out

    out["available"] = True
    out["commit"] = _run(["git", "rev-parse", "HEAD"], cwd=repo)
    out["commit_short"] = _run(["git", "rev-parse", "--short", "HEAD"], cwd=repo)
    out["commit_date"] = _run(["git", "log", "-1", "--format=%cI"], cwd=repo)
    out["subject"] = _run(["git", "log", "-1", "--format=%s"], cwd=repo)
    out["branch"] = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo)
    out["describe"] = _run(["git", "describe", "--tags", "--always"], cwd=repo)

    status = _run(["git", "status", "--porcelain"], cwd=repo)
    if status is None:
        out["dirty"] = None
    else:
        out["dirty"] = bool(status)
        if status:
            lines = status.splitlines()
            out["status_porcelain"] = lines[:50]
            out["status_porcelain_truncated"] = len(lines) > 50
    return out


def gpu_info() -> dict[str, Any]:
    """NVIDIA device info via nvidia-smi. Absent on the macOS harness."""
    out: dict[str, Any] = {"available": False, "devices": []}
    if shutil.which("nvidia-smi") is None:
        out["error"] = "nvidia-smi not found (expected on the macOS harness)"
        return out

    out["available"] = True
    out["driver_version"] = _run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
    )
    listing = _run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,compute_cap",
            "--format=csv,noheader,nounits",
        ]
    )
    if listing:
        for line in listing.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 5:
                out["devices"].append(
                    {
                        "index": int(parts[0]),
                        "name": parts[1],
                        "memory_total_mib": int(parts[2]),
                        "memory_used_mib": int(parts[3]),
                        "compute_capability": parts[4],
                    }
                )
    # CUDA runtime version is best taken from torch; see python_env_info().
    return out


def python_env_info() -> dict[str, Any]:
    """Python, torch, CUDA-runtime and key library versions.

    Imports are attempted defensively: the harness env deliberately does not
    depend on torch, so `torch` is often absent where this is called.
    """
    out: dict[str, Any] = {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }

    for mod in ("torch", "transformers", "sglang", "triton", "flashinfer"):
        try:
            m = __import__(mod)
            out[f"{mod}_version"] = getattr(m, "__version__", "unknown")
        except Exception as e:  # noqa: BLE001 - any import failure is just "absent"
            out[f"{mod}_version"] = None
            out[f"{mod}_import_error"] = type(e).__name__

    try:
        import torch  # noqa: PLC0415

        out["cuda_available"] = bool(torch.cuda.is_available())
        out["cuda_version"] = getattr(torch.version, "cuda", None)
        out["cudnn_version"] = (
            torch.backends.cudnn.version() if torch.cuda.is_available() else None
        )
        if torch.cuda.is_available():
            out["device_name"] = torch.cuda.get_device_name(0)
            out["device_count"] = torch.cuda.device_count()
            out["device_capability"] = ".".join(
                str(x) for x in torch.cuda.get_device_capability(0)
            )
    except Exception:  # noqa: BLE001
        out.setdefault("cuda_available", None)

    return out


def collect_environment(
    sglang_path: str | Path | None = None,
    *,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Everything needed to interpret and reproduce a run."""
    env: dict[str, Any] = {
        "python_env": python_env_info(),
        "gpu": gpu_info(),
        "hostname": platform.node(),
    }
    if sglang_path is not None:
        env["sglang_git"] = git_info(sglang_path)
    if extra:
        env["extra"] = extra
    return env
