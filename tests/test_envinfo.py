"""Tests for environment capture.

The `git_info` test runs against the real SGLang checkout when it is present,
because recording the commit and dirtiness is what makes a run citable
(results/README.md rule 1) and a silently-broken capture would undermine every
result downstream.
"""

from pathlib import Path

import pytest

from heterospec.envinfo import (
    collect_environment,
    git_info,
    gpu_info,
    python_env_info,
)

HETEROSPEC_ROOT = Path(__file__).resolve().parents[1]
SGLANG_ROOT = HETEROSPEC_ROOT.parent / "sglang"


# ---------------------------------------------------------------------------
# git_info
# ---------------------------------------------------------------------------


def test_git_info_on_non_repo_reports_unavailable(tmp_path):
    info = git_info(tmp_path)
    assert info["available"] is False
    assert "error" in info


@pytest.mark.skipif(
    not (SGLANG_ROOT / ".git").exists(), reason="sibling sglang checkout not present"
)
def test_git_info_on_real_sglang_checkout():
    info = git_info(SGLANG_ROOT)
    assert info["available"] is True
    assert info["commit"] and len(info["commit"]) == 40
    assert info["commit_short"] and len(info["commit_short"]) <= 12
    assert isinstance(info["dirty"], bool)
    assert info["branch"]


@pytest.mark.skipif(
    not (SGLANG_ROOT / ".git").exists(), reason="sibling sglang checkout not present"
)
def test_git_info_records_status_when_dirty(tmp_path):
    """A dirty tree must be detected, since results from one are not citable."""
    from heterospec.envinfo import _run

    repo = tmp_path / "repo"
    repo.mkdir()
    assert _run(["git", "init"], cwd=repo) is not None
    (repo / "untracked.txt").write_text("hello")
    info = git_info(repo)
    assert info["available"] is True
    assert info["dirty"] is True
    assert info["status_porcelain"]


def test_git_info_dirty_is_false_for_clean_repo(tmp_path):
    from heterospec.envinfo import _run

    repo = tmp_path / "clean"
    repo.mkdir()
    _run(["git", "init"], cwd=repo)
    _run(["git", "config", "user.email", "t@example.com"], cwd=repo)
    _run(["git", "config", "user.name", "t"], cwd=repo)
    (repo / "a.txt").write_text("x")
    _run(["git", "add", "-A"], cwd=repo)
    _run(["git", "commit", "-m", "init"], cwd=repo)
    info = git_info(repo)
    assert info["dirty"] is False
    assert info["status_porcelain"] is None


# ---------------------------------------------------------------------------
# gpu / python env
# ---------------------------------------------------------------------------


def test_gpu_info_is_graceful_off_nvidia():
    """Must not raise on the macOS harness, where nvidia-smi is absent."""
    info = gpu_info()
    assert isinstance(info, dict)
    assert "available" in info
    assert isinstance(info["devices"], list)


def test_python_env_info_reports_interpreter():
    info = python_env_info()
    assert info["python"]
    assert info["python_executable"]
    assert "torch_version" in info  # present, possibly None


def test_python_env_info_survives_missing_torch():
    info = python_env_info()
    # The harness env has no torch; that must be recorded, not crash.
    assert info["torch_version"] is None
    assert info["torch_import_error"] == "ModuleNotFoundError"


def test_collect_environment_without_sglang_path():
    env = collect_environment()
    assert "python_env" in env and "gpu" in env
    assert "sglang_git" not in env


@pytest.mark.skipif(
    not (SGLANG_ROOT / ".git").exists(), reason="sibling sglang checkout not present"
)
def test_collect_environment_includes_sglang_git():
    env = collect_environment(SGLANG_ROOT, extra={"note": "test"})
    assert env["sglang_git"]["commit"]
    assert env["extra"] == {"note": "test"}


def test_collect_environment_is_json_serialisable():
    import json

    env = collect_environment(extra={"nested": {"a": [1, 2]}})
    assert json.loads(json.dumps(env, default=str))["extra"]["nested"]["a"] == [1, 2]
