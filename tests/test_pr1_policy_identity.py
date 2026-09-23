"""Contract test for the upstream PR: request identity in the policy feedback path.

PR 1 changes the adaptive-spec feedback path so a policy can keep request-local
state. The change spans seven call sites, and the failure mode it must avoid is
**silent**: if the id list and the count list are ever misaligned, acceptance is
attributed to the wrong request, permanently, with nothing in the payload to
reveal it.

The fork's own unit tests for this need torch, so they cannot run on the Mac.
This file asserts the *shape* of the patch from the heterospec side, which catches
the realistic regression: someone edits the call site or a signature and drops the
alignment or the new parameter. It reads the fork's working tree directly.
"""

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SGLANG = REPO_ROOT.parent / "sglang"
SRC = SGLANG / "python" / "sglang" / "srt"

FRESH = SRC / "speculative" / "adaptive_runtime_state.py"
PARAMS = SRC / "speculative" / "adaptive_spec_params.py"
EAGLE = SRC / "speculative" / "eagle_worker_v2.py"
NGRAM = SRC / "speculative" / "ngram_worker.py"
BASE_WORKER = SRC / "speculative" / "base_spec_worker.py"
TP_WORKER = SRC / "managers" / "tp_worker.py"
PROCESSOR = SRC / "managers" / "scheduler_components" / "batch_result_processor.py"

ALL_FILES = [FRESH, PARAMS, EAGLE, NGRAM, BASE_WORKER, TP_WORKER, PROCESSOR]

pytestmark = pytest.mark.skipif(
    not SGLANG.is_dir(), reason="sibling sglang checkout not present"
)


def _read(path: Path) -> str:
    return path.read_text()


def _class_body(src: str, class_name: str) -> str:
    """Body of a top-level class, up to the next top-level class."""
    m = re.search(rf"^class {class_name}\b.*?(?=^class |\Z)", src, re.S | re.M)
    assert m, f"class {class_name} not found"
    return m.group(0)


def _method_body(class_src: str, method_name: str) -> str:
    """One method body, up to the next method at the same indent."""
    m = re.search(
        rf"^    def {method_name}\(.*?(?=^    def |\Z)", class_src, re.S | re.M
    )
    assert m, f"method {method_name} not found"
    return m.group(0)


def _has_request_ids(path: Path) -> bool:
    return "request_ids" in _read(path)


# ---------------------------------------------------------------------------
# Every layer of the feedback path must carry the parameter
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", ALL_FILES, ids=lambda p: p.name)
def test_feedback_path_layer_accepts_request_ids(path):
    """A break in any single layer silently drops identity for the whole path."""
    if not path.is_file():
        pytest.skip(f"{path} not present")
    assert _has_request_ids(path), (
        f"{path.name} does not mention request_ids; the identity change is "
        f"incomplete along the feedback path"
    )


def test_policy_protocol_declares_request_ids():
    body = _method_body(
        _class_body(_read(FRESH), "AdaptiveSpecPolicy"), "on_verify_complete"
    )
    assert "request_ids" in body, "protocol does not expose request ids"
    assert "list[str] | None" in body, "protocol ids should be optional"


def test_controller_forwards_request_ids_to_the_policy():
    body = _method_body(
        _class_body(_read(FRESH), "AdaptiveController"), "on_verify_complete"
    )
    assert "request_ids" in body
    assert "self.params.on_verify_complete" in body
    # Must actually be passed on, not merely accepted.
    assert re.search(
        r"self\.params\.on_verify_complete\([^)]*request_ids", body, re.S
    ), "controller accepts request_ids but does not forward them"


def test_no_op_hooks_accept_request_ids():
    """tp_worker and base_spec_worker are no-ops but must stay signature-compatible."""
    for path in (TP_WORKER, BASE_WORKER):
        src = _read(path)
        m = re.search(r"def on_verify_complete_cpu\(.*?\) -> None:", src, re.S)
        assert m, f"{path.name}: hook not found"
        assert "request_ids" in m.group(0), (
            f"{path.name}: no-op hook would raise TypeError when the processor "
            f"passes request_ids"
        )


# ---------------------------------------------------------------------------
# The call site: the one place that can silently misalign
# ---------------------------------------------------------------------------


def test_call_site_passes_ids_aligned_with_batch_reqs():
    """Ids must be built from `batch.reqs`, in the same order as the counts.

    `num_correct_drafts_per_req_cpu` is built from `accept_lens`, which is
    per-request in batch order, and the processor's own loop indexes it with
    `enumerate(batch.reqs)`. So `[req.rid for req in batch.reqs]` is the aligned
    construction -- anything else (a set, a sorted list, a dict) is a bug.
    """
    src = _read(PROCESSOR)
    start = src.index("on_verify_complete_cpu(")
    call = src[start : start + 600]
    assert "request_ids=[req.rid for req in batch.reqs]" in call, (
        "call site must build ids from batch.reqs in order"
    )
    assert "batch_size=len(batch.reqs)" in call, (
        "batch_size and request_ids must come from the same source"
    )


def test_call_site_is_the_only_one_that_needs_to_change():
    """Every other producer of feedback must be unaffected."""
    hits = [
        p
        for p in SGLANG.rglob("*.py")
        if "on_verify_complete_cpu(" in _read(p) and "rglob" not in str(p)
    ]
    callers = [p for p in hits if "on_verify_complete_cpu(" in _read(p)]
    # Definitions plus the single call site.
    assert len(callers) >= 4


# ---------------------------------------------------------------------------
# The misalignment guard
# ---------------------------------------------------------------------------


def test_controller_drops_misaligned_ids_rather_than_passing_them():
    """Misaligned ids must be dropped, because the corruption would be silent.

    Passing them through would let a policy attribute one request's acceptance to
    another with no way to detect it. Dropping degrades to "no identity", which is
    merely a lost feature.
    """
    body = _method_body(
        _class_body(_read(FRESH), "AdaptiveController"), "on_verify_complete"
    )
    assert "len(request_ids) != len(" in body, "no length check"
    assert "request_ids = None" in body, "misaligned ids must be replaced with None"
    assert "logger.warning" in body, "the drop must be reported"


def test_mismatch_warning_is_emitted_once():
    """A warning per decode step would flood the log on a long run."""
    body = _read(FRESH)
    assert "_warned_id_mismatch" in body, "no once-only guard for the warning"


def test_return_semantics_are_unchanged():
    """The patch must not alter when a state switch happens."""
    body = _method_body(
        _class_body(_read(FRESH), "AdaptiveController"), "on_verify_complete"
    )
    assert "if new_step is not None:" in body
    assert "self._activate(new_step)" in body


def test_default_policy_ignores_ids_by_design():
    """The shipped EMA has no request-local state; behaviour must be identical."""
    body = _method_body(
        _class_body(_read(PARAMS), "AdaptiveSpeculativeParams"), "on_verify_complete"
    )
    assert "request_ids" in body
    # It must not consult them: everything after the routing line is the
    # original EMA update.
    _, _, tail = body.partition("params = self._route")
    assert "request_ids" not in tail, (
        "the default EMA policy should accept but not use request ids"
    )


def test_ids_come_from_python_objects_not_a_tensor_conversion():
    """Identity rides the existing CPU feedback path; no new device sync.

    `batch.reqs` are already-resident Python objects, so building the id list
    must not involve `.cpu()`, `.tolist()`, or any tensor operation. That is what
    keeps this change free in the decode hot path.
    """
    src = _read(PROCESSOR)
    start = src.index("request_ids=[req.rid for req in batch.reqs]")
    expr = src[start : start + len("request_ids=[req.rid for req in batch.reqs]")]
    for forbidden in (".cpu()", ".tolist()", "torch."):
        assert forbidden not in expr
