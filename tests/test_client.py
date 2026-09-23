"""Tests for the HTTP client and the mock SGLang server.

Two things matter here:

1. The **client must never raise** for a request-level failure. A crashed sweep
   wastes the GPU hour it was paying for, so timeouts, HTTP errors and malformed
   bodies all come back as a failed `GenerationResult`.
2. The **mock must be a faithful double** for the real capture contract. The
   strongest check is the accounting invariant
   `completion_tokens == spec_num_correct_drafts + spec_verify_ct`, which is
   exactly what a real SGLang capture satisfies, plus
   `sum(histogram) == spec_verify_ct`. If the harness works against the mock, the
   only unknown left on the GPU is the hardware.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from heterospec.client import SGLangClient, extract_spec_telemetry
from heterospec.mockserver import (
    ACCEPTANCE_PROFILES,
    MockSGLangServer,
    _classify,
    expected_accept_length,
    simulate_rounds,
)
from heterospec.records import RunMetadata
from heterospec.workloads import PROMPT_CLASSES

# ---------------------------------------------------------------------------
# Mock as a faithful double for our actual prompts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("class_name", sorted(PROMPT_CLASSES))
def test_mock_classifies_every_workload_prompt_as_declared(class_name):
    """The mock infers class from prompt text; it must agree with the workload.

    If this fails for a prompt, the mock is not a faithful double for that
    workload and any pipeline test using it would be misleading.
    """
    cls = PROMPT_CLASSES[class_name]
    misclassified = [p for p in cls.prompts if _classify(p) != class_name]
    assert not misclassified, (
        f"{class_name} prompts misclassified by the mock: {misclassified}"
    )


# ---------------------------------------------------------------------------
# Simulation maths (no HTTP)
# ---------------------------------------------------------------------------


def test_expected_accept_length_hand_computed():
    # p = (1.0, 0.5): S_1 = 1.0, S_2 = 0.5 -> E = 1.5
    assert expected_accept_length((1.0, 0.5), 2) == pytest.approx(1.5)


def test_expected_accept_length_zero_for_zero_depth():
    assert expected_accept_length((0.9, 0.8), 0) == 0.0


def test_expected_accept_length_is_monotone_in_depth():
    p = (0.9, 0.7, 0.5, 0.3)
    vals = [expected_accept_length(p, k) for k in range(1, 5)]
    assert all(a < b for a, b in zip(vals, vals[1:], strict=False))


def test_repetitive_profile_accepts_more_than_open_ended():
    """The mock must actually contain the signal the project is looking for."""
    rep = expected_accept_length(ACCEPTANCE_PROFILES["repetitive"], 7)
    low = expected_accept_length(ACCEPTANCE_PROFILES["open_ended"], 7)
    assert rep > low * 2, f"insufficient separation: {rep} vs {low}"


def test_simulate_rounds_accounting_invariant():
    """completion_tokens == accepted drafts + one bonus token per round."""
    import random

    rng = random.Random(0)
    accepted, tokens = simulate_rounds(
        p=(0.9, 0.7, 0.5, 0.3), k=4, max_new_tokens=256, rng=rng
    )
    assert tokens == sum(accepted) + len(accepted)
    assert tokens >= 256


def test_simulate_rounds_respects_zero_acceptance_profile():
    import random

    rng = random.Random(0)
    accepted, tokens = simulate_rounds(
        p=(0.0, 0.0, 0.0), k=3, max_new_tokens=10, rng=rng
    )
    assert set(accepted) == {0}
    assert tokens == len(accepted)  # 1 bonus token per round


def test_simulate_rounds_is_deterministic_for_a_seed():
    import random

    a, _ = simulate_rounds(p=(0.8, 0.6), k=2, max_new_tokens=50, rng=random.Random(5))
    b, _ = simulate_rounds(p=(0.8, 0.6), k=2, max_new_tokens=50, rng=random.Random(5))
    assert a == b


def test_simulate_rounds_never_exceeds_max_rounds():
    import random

    accepted, _ = simulate_rounds(
        p=(0.0,), k=1, max_new_tokens=10**9, rng=random.Random(0), max_rounds=17
    )
    assert len(accepted) == 17


# ---------------------------------------------------------------------------
# Extract telemetry from meta_info
# ---------------------------------------------------------------------------


def test_extract_telemetry_full():
    t = extract_spec_telemetry(
        {
            "spec_verify_ct": 20,
            "spec_num_correct_drafts": 39,
            "spec_correct_drafts_histogram": [2, 4, 8, 5, 1],
        }
    )
    assert t.present
    assert t.spec_verify_ct == 20
    assert t.spec_correct_drafts_histogram == [2, 4, 8, 5, 1]
    assert t.missing_fields == []


def test_extract_telemetry_absent_stays_none_not_zero():
    """A no-spec run must record absence, not fabricate zeros."""
    t = extract_spec_telemetry({"completion_tokens": 10})
    assert not t.present
    assert t.spec_verify_ct is None
    assert t.spec_num_correct_drafts is None
    assert t.spec_correct_drafts_histogram is None
    assert set(t.missing_fields) == {
        "spec_verify_ct",
        "spec_num_correct_drafts",
        "spec_correct_drafts_histogram",
    }


def test_extract_telemetry_handles_none_and_empty():
    assert not extract_spec_telemetry(None).present
    assert not extract_spec_telemetry({}).present


def test_extract_telemetry_tolerates_garbage_values():
    t = extract_spec_telemetry(
        {
            "spec_verify_ct": "12",
            "spec_num_correct_drafts": "bad",
            "spec_correct_drafts_histogram": "nope",
        }
    )
    assert t.spec_verify_ct == 12
    assert t.spec_num_correct_drafts is None
    assert t.spec_correct_drafts_histogram is None


def test_extract_telemetry_missing_histogram_but_other_fields_present():
    """SGLang omits the histogram when it is empty; that is not an error."""
    t = extract_spec_telemetry({"spec_verify_ct": 0, "spec_num_correct_drafts": 0})
    assert t.spec_verify_ct == 0
    assert t.spec_correct_drafts_histogram is None
    assert t.missing_fields == ["spec_correct_drafts_histogram"]


# ---------------------------------------------------------------------------
# Mock server over HTTP
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_server():
    with MockSGLangServer(k=4, seed=0) as srv:
        yield srv


def test_mock_health(mock_server):
    with SGLangClient(mock_server.base_url) as c:
        assert c.health() is True


def test_mock_server_info_flags_itself_as_mock(mock_server):
    with SGLangClient(mock_server.base_url) as c:
        info = c.server_info()
    assert info["mock"] is True
    assert info["internal_states"][0]["speculative_num_steps"] == 4


def test_mock_generate_satisfies_capture_invariants(mock_server):
    with SGLangClient(mock_server.base_url) as c:
        r = c.generate("Output exactly 100 lines.", max_new_tokens=64, rid="r1")

    assert r.ok, r.error
    t = r.telemetry
    assert t.spec_verify_ct and t.spec_verify_ct > 0
    # The invariant a real capture must satisfy:
    assert r.completion_tokens == t.spec_num_correct_drafts + t.spec_verify_ct
    # ...and the histogram/trace agreement the harness checks:
    assert sum(t.spec_correct_drafts_histogram) == t.spec_verify_ct


def test_mock_is_deterministic_per_rid(mock_server):
    with SGLangClient(mock_server.base_url) as c:
        a = c.generate("Output exactly 100 lines.", max_new_tokens=64, rid="same")
        b = c.generate("Output exactly 100 lines.", max_new_tokens=64, rid="same")
    assert a.telemetry.spec_correct_drafts_histogram == (
        b.telemetry.spec_correct_drafts_histogram
    )


def test_mock_repetitive_beats_open_ended_end_to_end(mock_server):
    """The class separation must survive the whole HTTP + parsing path."""
    rep_prompt = "Output exactly 200 lines, one per line."
    low_prompt = "Compose a poem about quantum entanglement."

    with SGLangClient(mock_server.base_url) as c:
        rep = [
            c.generate(rep_prompt, max_new_tokens=128, rid=f"rep{i}").telemetry
            for i in range(20)
        ]
        low = [
            c.generate(low_prompt, max_new_tokens=128, rid=f"low{i}").telemetry
            for i in range(20)
        ]

    rep_mean = sum(t.spec_num_correct_drafts for t in rep) / sum(
        t.spec_verify_ct for t in rep
    )
    low_mean = sum(t.spec_num_correct_drafts for t in low) / sum(
        t.spec_verify_ct for t in low
    )
    assert rep_mean > low_mean * 2, f"no separation: {rep_mean} vs {low_mean}"


def test_mock_k_schedule_varies_active_k():
    """A scheduled K is how we exercise the K-confounded analysis path."""
    with MockSGLangServer(k_schedule=[1, 3, 7], seed=0) as srv:
        with SGLangClient(srv.base_url) as c:
            ks = set()
            for i in range(6):
                c.generate("Output exactly 50 lines.", max_new_tokens=32, rid=f"r{i}")
            info = c.server_info()
            ks.add(info["internal_states"][0]["speculative_num_steps"])
    assert ks  # active K tracked; last value is a real tier


def test_mock_unknown_path_returns_404(mock_server):
    import requests

    r = requests.get(f"{mock_server.base_url}/nope", timeout=5)
    assert r.status_code == 404


def test_mock_generate_rejects_invalid_json(mock_server):
    import requests

    r = requests.post(f"{mock_server.base_url}/generate", data=b"{not json", timeout=5)
    assert r.status_code == 400


# ---------------------------------------------------------------------------
# Client failure modes -- must never raise
# ---------------------------------------------------------------------------


def test_client_reports_connection_refused():
    # Port 1 is reserved and nothing listens there.
    with SGLangClient("http://127.0.0.1:1", timeout_s=2.0) as c:
        r = c.generate("hi", max_new_tokens=4)
    assert not r.ok
    assert r.error, "a failed request must carry a reason"
    assert "Error" in r.error or "error" in r.error
    assert r.latency_s >= 0


def test_client_reports_http_error_status():
    with _one_shot_server(500, b'{"error": "boom"}') as url:
        with SGLangClient(url, timeout_s=5.0) as c:
            r = c.generate("hi", max_new_tokens=4)
    assert not r.ok
    assert r.http_status == 500
    assert "500" in (r.error or "")


def test_client_reports_invalid_json_body():
    with _one_shot_server(200, b"<html>not json</html>") as url:
        with SGLangClient(url, timeout_s=5.0) as c:
            r = c.generate("hi", max_new_tokens=4)
    assert not r.ok
    assert "invalid JSON" in (r.error or "")


def test_client_reports_error_field_with_http_200():
    with _one_shot_server(200, b'{"error": "bad sampling params"}') as url:
        with SGLangClient(url, timeout_s=5.0) as c:
            r = c.generate("hi", max_new_tokens=4)
    assert not r.ok
    assert "bad sampling params" in (r.error or "")


def test_client_health_false_when_unreachable():
    with SGLangClient("http://127.0.0.1:1", timeout_s=2.0) as c:
        assert c.health() is False


def test_client_wait_until_ready_times_out_without_raising():
    with SGLangClient("http://127.0.0.1:1", timeout_s=2.0) as c:
        assert c.wait_until_ready(timeout_s=1.0, poll_s=0.2) is False


def test_client_base_url_normalised():
    c = SGLangClient("http://127.0.0.1:30000/")
    assert c.base_url == "http://127.0.0.1:30000"
    c.close()


def test_client_rid_is_sent_and_echoed_in_trace_keys(mock_server):
    """The rid we send is the identity the server records -- exact join, no guessing."""
    with SGLangClient(mock_server.base_url) as c:
        r = c.generate("Output exactly 20 lines.", max_new_tokens=16, rid="het00042")
    assert r.ok
    assert r.telemetry.spec_verify_ct is not None


def test_active_num_steps_from_server_info(mock_server):
    with SGLangClient(mock_server.base_url) as c:
        assert c.active_num_steps() == 4


def test_active_num_steps_none_when_unreachable():
    with SGLangClient("http://127.0.0.1:1", timeout_s=2.0) as c:
        assert c.active_num_steps() is None


# ---------------------------------------------------------------------------
# Mock runs must never be citable
# ---------------------------------------------------------------------------


def test_mock_run_is_not_citable():
    from heterospec import SGLANG_BASE_COMMIT
    from heterospec.envinfo import collect_environment

    env = collect_environment()
    env["sglang_git"] = {"commit": SGLANG_BASE_COMMIT, "dirty": False}
    meta = RunMetadata.new(
        run_id="t",
        policy_id="static_k3",
        workload_name="high",
        environment=env,
        server={"mode": "mock", "base_url": "http://x"},
    )
    assert meta.is_mock
    ok, reason = meta.citable()
    assert not ok and "mock" in reason


def test_mock_detected_from_server_info_flag_alone():
    meta = RunMetadata.new(
        run_id="t",
        policy_id="p",
        workload_name="w",
        server={"server_info": {"mock": True}},
    )
    assert meta.is_mock


# ---------------------------------------------------------------------------
# helper: a server that returns one canned response
# ---------------------------------------------------------------------------


class _CannedHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):  # noqa: D102, ANN002
        pass

    def do_POST(self):  # noqa: N802
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        body = self.server.canned_body  # type: ignore[attr-defined]
        self.send_response(self.server.canned_status)  # type: ignore[attr-defined]
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class _one_shot_server:
    """Context manager yielding a base URL that returns a canned response."""

    def __init__(self, status: int, body: bytes) -> None:
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), _CannedHandler)
        self.httpd.daemon_threads = True
        self.httpd.canned_status = status  # type: ignore[attr-defined]
        self.httpd.canned_body = body  # type: ignore[attr-defined]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self) -> str:
        self.thread.start()
        host, port = self.httpd.server_address[0], self.httpd.server_address[1]
        return f"http://{host}:{port}"

    def __exit__(self, *exc: object) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)


# ---------------------------------------------------------------------------
# Request schema: the real server does not ignore unknown sampling keys
# ---------------------------------------------------------------------------


def test_sampling_params_use_only_field_names_the_server_accepts():
    """The bug this pins cost a GPU session.

    SGLang builds ``SamplingParams(**sampling_kwargs)``, so a key that is not a
    field name raises ``TypeError`` -- which reaches the client as a bare
    ``HTTP 500: Internal Server Error`` on every request. The harness sent
    ``seed``; the field is ``sampling_seed``. Warm-up sends no seed, so it kept
    succeeding while every measured request failed, which made it look like a
    load or resource problem rather than a one-word payload bug.

    Asserting the exact key set is what turns this from a $1 discovery into a
    unit test.
    """
    from heterospec.mockserver import VALID_SAMPLING_KEYS

    captured: dict = {}

    class _FakeSession:
        def post(self, url, json=None, timeout=None):  # noqa: A002
            captured.update(json)
            raise AssertionError("payload captured; no real request needed")

    client = SGLangClient("http://127.0.0.1:1")
    client._session = _FakeSession()  # type: ignore[assignment]
    try:
        client.generate(
            "hello",
            max_new_tokens=16,
            temperature=0.0,
            top_p=0.9,
            seed=12345,
            rid="r-00001",
        )
    except AssertionError:
        pass

    sampling = captured["sampling_params"]
    unknown = sorted(set(sampling) - VALID_SAMPLING_KEYS)
    assert not unknown, (
        f"{unknown} are not SGLang SamplingParams fields; the server answers 500 "
        f"for each request. `seed` must be sent as `sampling_seed`."
    )
    assert sampling["sampling_seed"] == 12345
    assert "seed" not in sampling, "the field name is sampling_seed, not seed"


def test_the_mock_rejects_an_unknown_sampling_key_like_the_real_server():
    """Otherwise the mock silently accepts payloads the GPU host will reject.

    This is the guard for the guard: if the mock goes back to accepting anything,
    the schema test above can pass while the real server still 500s.
    """
    import urllib.error
    import urllib.request

    with MockSGLangServer(k=3, seed=0) as srv:
        body = json.dumps(
            {
                "text": "hello",
                "sampling_params": {"temperature": 0.0, "max_new_tokens": 4, "seed": 1},
                "return_logprob": False,
            }
        ).encode()
        req = urllib.request.Request(
            f"{srv.base_url}/generate",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(req, timeout=10)
        assert exc.value.code == 500
        detail = json.loads(exc.value.read())
        assert "Unexpected keyword argument" in detail["error"]
        assert "sampling_seed" in detail["error"]
