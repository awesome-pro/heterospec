"""Tests for the fork's iteration-trace patch.

The tracer module is stdlib-only by design precisely so it can be exercised here,
on the Mac, without torch or a GPU. It is loaded directly from the SGLang
checkout by path.

The most valuable test is
``test_trace_output_loads_as_iteration_records``: it proves the fork's output
feeds `heterospec.records.load_iteration_records`, i.e. that the trace patch and
the analysis pipeline actually agree on a format. Discovering a mismatch on the
GPU host would cost real money.
"""

import importlib.util
import json
import sys
import uuid
from pathlib import Path

import pytest

from heterospec.records import load_iteration_records, min_k_proposed_per_request

REPO_ROOT = Path(__file__).resolve().parents[1]
TRACE_PATH = (
    REPO_ROOT.parent
    / "sglang"
    / "python"
    / "sglang"
    / "srt"
    / "speculative"
    / "heterospec_trace.py"
)

pytestmark = pytest.mark.skipif(
    not TRACE_PATH.is_file(),
    reason="sibling sglang checkout with the patch not present",
)

_ENV_KEYS = (
    "SGLANG_HETEROSPEC_TRACE",
    "SGLANG_HETEROSPEC_TRACE_MAX_RECORDS",
    "SGLANG_HETEROSPEC_TRACE_FLUSH_EVERY",
)


def _load(monkeypatch, **env):
    """Load the tracer fresh, so its module-level env read is re-evaluated."""
    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, str(value))

    name = f"heterospec_trace_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(name, TRACE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# Disabled path
# ---------------------------------------------------------------------------


def test_disabled_without_env_var(monkeypatch):
    mod = _load(monkeypatch)
    assert mod.enabled() is False
    assert mod._TRACER is None


def test_record_is_a_noop_when_disabled(monkeypatch):
    mod = _load(monkeypatch)
    # Must not raise, and must not create anything.
    mod.record_iteration(active_k=3, rids=["a", "b"], accepted=[2, 1])
    mod.flush()
    mod.close()


# ---------------------------------------------------------------------------
# Enabled path
# ---------------------------------------------------------------------------


def test_enabled_with_env_var(monkeypatch, tmp_path):
    out = tmp_path / "trace.jsonl"
    mod = _load(monkeypatch, SGLANG_HETEROSPEC_TRACE=out)
    assert mod.enabled() is True
    assert mod._TRACER is not None
    mod.close()


def test_record_writes_expected_shape(monkeypatch, tmp_path):
    out = tmp_path / "trace.jsonl"
    mod = _load(monkeypatch, SGLANG_HETEROSPEC_TRACE=out)
    mod.record_iteration(active_k=5, rids=["r1", "r2", "r3"], accepted=[5, 1, 0])
    mod.close()

    rows = _read(out)
    assert len(rows) == 1
    row = rows[0]
    assert row["active_k"] == 5
    assert row["batch_size"] == 3
    assert row["iteration"] == 0
    assert row["requests"] == [
        {"rid": "r1", "accepted": 5},
        {"rid": "r2", "accepted": 1},
        {"rid": "r3", "accepted": 0},
    ]
    assert isinstance(row["ts"], float)


def test_iteration_counter_increments(monkeypatch, tmp_path):
    out = tmp_path / "trace.jsonl"
    mod = _load(monkeypatch, SGLANG_HETEROSPEC_TRACE=out)
    for i in range(5):
        mod.record_iteration(active_k=3, rids=[f"r{i}"], accepted=[1])
    mod.close()
    assert [r["iteration"] for r in _read(out)] == [0, 1, 2, 3, 4]


def test_active_k_may_be_none(monkeypatch, tmp_path):
    """A worker without `speculative_num_steps` must not crash the trace."""
    out = tmp_path / "trace.jsonl"
    mod = _load(monkeypatch, SGLANG_HETEROSPEC_TRACE=out)
    mod.record_iteration(active_k=None, rids=["a", "b"], accepted=[1, 0])
    mod.close()
    assert _read(out)[0]["active_k"] is None


def test_mismatched_lengths_use_the_shorter(monkeypatch, tmp_path):
    """Defensive: never write a record that pairs a rid with the wrong count."""
    out = tmp_path / "trace.jsonl"
    mod = _load(monkeypatch, SGLANG_HETEROSPEC_TRACE=out)
    mod.record_iteration(active_k=3, rids=["a", "b", "c"], accepted=[1, 2])
    mod.close()
    row = _read(out)[0]
    assert row["batch_size"] == 2
    assert [r["rid"] for r in row["requests"]] == ["a", "b"]


def test_empty_batch_is_recorded_without_error(monkeypatch, tmp_path):
    out = tmp_path / "trace.jsonl"
    mod = _load(monkeypatch, SGLANG_HETEROSPEC_TRACE=out)
    mod.record_iteration(active_k=3, rids=[], accepted=[])
    mod.close()
    assert _read(out)[0]["batch_size"] == 0


# ---------------------------------------------------------------------------
# Bounded output
# ---------------------------------------------------------------------------


def test_record_cap_is_enforced(monkeypatch, tmp_path):
    """A long run must not be able to fill the disk."""
    out = tmp_path / "trace.jsonl"
    mod = _load(
        monkeypatch, SGLANG_HETEROSPEC_TRACE=out, SGLANG_HETEROSPEC_TRACE_MAX_RECORDS=3
    )
    for i in range(10):
        mod.record_iteration(active_k=3, rids=[f"r{i}"], accepted=[1])
    mod.close()
    assert len(_read(out)) == 3
    assert mod._TRACER.count == 3


def test_invalid_max_records_falls_back_to_default(monkeypatch, tmp_path):
    out = tmp_path / "trace.jsonl"
    mod = _load(
        monkeypatch,
        SGLANG_HETEROSPEC_TRACE=out,
        SGLANG_HETEROSPEC_TRACE_MAX_RECORDS="not-a-number",
    )
    assert mod._MAX_RECORDS == 200_000
    mod.close()


def test_negative_max_records_falls_back_to_default(monkeypatch, tmp_path):
    out = tmp_path / "trace.jsonl"
    mod = _load(
        monkeypatch, SGLANG_HETEROSPEC_TRACE=out, SGLANG_HETEROSPEC_TRACE_MAX_RECORDS=-5
    )
    assert mod._MAX_RECORDS == 200_000
    mod.close()


# ---------------------------------------------------------------------------
# The integration that matters
# ---------------------------------------------------------------------------


def test_trace_output_loads_as_iteration_records(monkeypatch, tmp_path):
    """The fork's output must feed the analysis pipeline unchanged.

    This is the contract between the trace patch and `heterospec.records`. A
    mismatch found on the GPU host would cost real money.
    """
    out = tmp_path / "trace.jsonl"
    mod = _load(monkeypatch, SGLANG_HETEROSPEC_TRACE=out)

    mod.record_iteration(active_k=7, rids=["a", "b"], accepted=[7, 1])
    mod.record_iteration(active_k=2, rids=["a", "b"], accepted=[2, 0])
    mod.record_iteration(active_k=5, rids=["a", "b"], accepted=[5, 2])
    mod.close()

    records = load_iteration_records(out)
    assert len(records) == 3
    assert records[0].active_k == 7
    assert records[0].rids == ["a", "b"]
    assert records[0].accepted == [7, 1]
    assert records[0].accepted_for("b") == 1

    # ...and the K-confounding bound the oracle needs comes out right.
    assert min_k_proposed_per_request(records) == {"a": 2, "b": 2}


def test_trace_file_is_valid_jsonl_line_by_line(monkeypatch, tmp_path):
    """A crash mid-run must leave a usable prefix, not a corrupt file."""
    out = tmp_path / "trace.jsonl"
    mod = _load(monkeypatch, SGLANG_HETEROSPEC_TRACE=out)
    for i in range(20):
        mod.record_iteration(active_k=3, rids=[f"r{i}"], accepted=[i % 4])
    mod.flush()
    # Read without closing, mimicking a snapshot taken mid-run.
    rows = _read(out)
    assert len(rows) >= 1
    assert all(
        set(r) == {"iteration", "batch_size", "active_k", "ts", "requests"}
        for r in rows
    )
    mod.close()


def test_flush_every_defaults_and_is_configurable(monkeypatch, tmp_path):
    out = tmp_path / "trace.jsonl"
    mod = _load(
        monkeypatch, SGLANG_HETEROSPEC_TRACE=out, SGLANG_HETEROSPEC_TRACE_FLUSH_EVERY=2
    )
    assert mod._FLUSH_EVERY == 2
    mod.record_iteration(active_k=3, rids=["a"], accepted=[1])
    mod.record_iteration(active_k=3, rids=["b"], accepted=[1])
    mod.close()
    assert len(_read(out)) == 2


# ---------------------------------------------------------------------------
# The patch is present and minimal
# ---------------------------------------------------------------------------


def test_patch_is_wired_into_the_result_processor():
    """The hook must exist at the call site the design doc identifies."""
    src = (
        REPO_ROOT.parent
        / "sglang"
        / "python"
        / "sglang"
        / "srt"
        / "managers"
        / "scheduler_components"
        / "batch_result_processor.py"
    ).read_text()
    assert "heterospec_trace" in src
    assert "_heterospec_trace.enabled()" in src
    assert "speculative_num_steps" in src
    # Guarded, so a disabled server does not build the rid list.
    guard = src.index("_heterospec_trace.enabled()")
    call = src.index("_heterospec_trace.record_iteration")
    assert guard < call < guard + 400


def test_tracer_module_imports_only_stdlib():
    """It sits on the scheduler hot path and must not pull in torch."""
    src = TRACE_PATH.read_text()
    import_lines = [
        line
        for line in src.splitlines()
        if line.startswith(("import ", "from ")) and "__future__" not in line
    ]
    allowed = ("atexit", "json", "logging", "os", "threading", "time", "typing")
    for line in import_lines:
        assert any(mod in line for mod in allowed), f"unexpected import: {line}"
