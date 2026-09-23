"""HTTP client for SGLang's native `/generate` API.

Design notes
------------
* **We supply our own `rid`.** `GenerateReqInput.rid` is a plain identity field
  with no side effects (session identity is the separate `session_id`), so the
  rid we send is the same rid the server uses in `note_request_finished(rid=...)`
  and will use in the iteration-level trace. That makes the join between
  per-request records and iteration records exact rather than heuristic.

* **Non-streaming.** The full `meta_info`, including
  `spec_correct_drafts_histogram`, is only assembled at completion.

* **One request per HTTP call**, so the response *is* the correlation. The
  client never has to guess which response belongs to which request.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import requests

__all__ = [
    "GenerationResult",
    "SGLangClient",
    "SpecTelemetry",
    "extract_spec_telemetry",
]


# ---------------------------------------------------------------------------
# meta_info extraction
# ---------------------------------------------------------------------------

#: The three per-request speculative fields SGLang exposes with no source change
#: (managers/tokenizer_manager.py:2944-2992).
_SPEC_FIELDS = (
    "spec_verify_ct",
    "spec_num_correct_drafts",
    "spec_correct_drafts_histogram",
)


@dataclass
class SpecTelemetry:
    """Per-request speculative telemetry, or explicit absence of it."""

    spec_verify_ct: int | None = None
    spec_num_correct_drafts: int | None = None
    spec_correct_drafts_histogram: list[int] | None = None

    @property
    def present(self) -> bool:
        return any(
            getattr(self, f) is not None
            for f in (
                "spec_verify_ct",
                "spec_num_correct_drafts",
                "spec_correct_drafts_histogram",
            )
        )

    @property
    def missing_fields(self) -> list[str]:
        return [f for f in _SPEC_FIELDS if getattr(self, f) is None]

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec_verify_ct": self.spec_verify_ct,
            "spec_num_correct_drafts": self.spec_num_correct_drafts,
            "spec_correct_drafts_histogram": self.spec_correct_drafts_histogram,
        }


def extract_spec_telemetry(meta_info: dict[str, Any] | None) -> SpecTelemetry:
    """Pull speculative telemetry out of a `meta_info` mapping.

    Absent fields stay `None`. Note that SGLang only emits
    `spec_correct_drafts_histogram` when it is non-empty
    (`tokenizer_manager.py:2989`), so a non-speculative or very short request
    legitimately has no histogram -- that is not an error, and must not be
    silently turned into zeros.
    """
    if not meta_info:
        return SpecTelemetry()

    def _int(key: str) -> int | None:
        v = meta_info.get(key)
        if v is None:
            return None
        try:
            return int(v)
        except (TypeError, ValueError):
            return None

    hist = meta_info.get("spec_correct_drafts_histogram")
    hist_list = [int(x) for x in hist] if isinstance(hist, (list, tuple)) else None

    return SpecTelemetry(
        spec_verify_ct=_int("spec_verify_ct"),
        spec_num_correct_drafts=_int("spec_num_correct_drafts"),
        spec_correct_drafts_histogram=hist_list,
    )


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class GenerationResult:
    """Outcome of one `/generate` call."""

    ok: bool
    latency_s: float
    started_at: float
    finished_at: float
    text: str | None = None
    meta_info: dict[str, Any] = field(default_factory=dict)
    telemetry: SpecTelemetry = field(default_factory=SpecTelemetry)
    error: str | None = None
    http_status: int | None = None

    @property
    def completion_tokens(self) -> int | None:
        v = self.meta_info.get("completion_tokens")
        try:
            return int(v) if v is not None else None
        except (TypeError, ValueError):
            return None


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class SGLangClient:
    """Thin, synchronous client. Concurrency is the caller's business."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:30000",
        *,
        timeout_s: float = 900.0,
        session: requests.Session | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self._session = session or requests.Session()
        self._owns_session = session is None

    # -- lifecycle -----------------------------------------------------------

    def close(self) -> None:
        if self._owns_session:
            self._session.close()

    def __enter__(self) -> SGLangClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- introspection -------------------------------------------------------

    def health(self, *, timeout_s: float = 5.0) -> bool:
        """Whether the server answers `/health`. Never raises."""
        try:
            r = self._session.get(f"{self.base_url}/health", timeout=timeout_s)
        except requests.RequestException:
            return False
        return r.status_code == 200

    def server_info(self, *, timeout_s: float = 30.0) -> dict[str, Any]:
        """`/server_info`, which carries the active tier and avg accept length."""
        r = self._session.get(f"{self.base_url}/server_info", timeout=timeout_s)
        r.raise_for_status()
        return r.json()

    def wait_until_ready(
        self, *, timeout_s: float = 900.0, poll_s: float = 3.0
    ) -> bool:
        """Poll `/health` until ready. Returns False on timeout rather than raising."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.health():
                return True
            time.sleep(poll_s)
        return False

    def active_num_steps(self) -> int | None:
        """Current adaptive tier, from `/server_info` (best-effort)."""
        try:
            info = self.server_info()
        except (requests.RequestException, ValueError):
            return None
        try:
            state = info["internal_states"][0]
        except (KeyError, IndexError, TypeError):
            return None
        for key in ("speculative_num_steps", "num_steps"):
            if key in state:
                try:
                    return int(state[key])
                except (TypeError, ValueError):
                    return None
        return None

    # -- generation ----------------------------------------------------------

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float | None = None,
        seed: int | None = None,
        rid: str | None = None,
        timeout_s: float | None = None,
    ) -> GenerationResult:
        """Issue one request. Never raises for server-side errors; reports them.

        Returning a failed `GenerationResult` rather than raising keeps a
        long benchmark running when individual requests time out or are
        rejected -- a crashed sweep wastes the GPU hour it was paying for.
        """
        sampling: dict[str, Any] = {
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
        }
        if top_p is not None:
            sampling["top_p"] = top_p
        if seed is not None:
            # SGLang's field is `sampling_seed`, NOT `seed`. The server builds
            # SamplingParams(**sampling_kwargs) and raises
            # "TypeError: Unexpected keyword argument 'seed'" for anything else,
            # which surfaces to the client as a bare HTTP 500 -- one per request,
            # while warm-up (which sends no seed) succeeds. See
            # sglang/srt/sampling/sampling_params.py.
            sampling["sampling_seed"] = seed

        payload: dict[str, Any] = {
            "text": prompt,
            "sampling_params": sampling,
            "return_logprob": False,
        }
        if rid is not None:
            payload["rid"] = rid

        started = time.perf_counter()
        try:
            r = self._session.post(
                f"{self.base_url}/generate",
                json=payload,
                timeout=timeout_s if timeout_s is not None else self.timeout_s,
            )
        except requests.Timeout:
            return GenerationResult(
                ok=False,
                latency_s=time.perf_counter() - started,
                started_at=started,
                finished_at=time.perf_counter(),
                error=f"timeout after {timeout_s or self.timeout_s}s",
            )
        except requests.RequestException as e:
            return GenerationResult(
                ok=False,
                latency_s=time.perf_counter() - started,
                started_at=started,
                finished_at=time.perf_counter(),
                error=f"{type(e).__name__}: {e}",
            )

        finished = time.perf_counter()
        if r.status_code != 200:
            return GenerationResult(
                ok=False,
                latency_s=finished - started,
                started_at=started,
                finished_at=finished,
                http_status=r.status_code,
                error=f"HTTP {r.status_code}: {r.text[:400]}",
            )

        try:
            data = r.json()
        except ValueError as e:
            return GenerationResult(
                ok=False,
                latency_s=finished - started,
                started_at=started,
                finished_at=finished,
                http_status=r.status_code,
                error=f"invalid JSON response: {e}",
            )

        # An error body can arrive with HTTP 200 in some paths.
        if isinstance(data, dict) and data.get("error"):
            return GenerationResult(
                ok=False,
                latency_s=finished - started,
                started_at=started,
                finished_at=finished,
                http_status=r.status_code,
                error=f"server error: {data['error']}",
            )

        meta = (data or {}).get("meta_info") or {}
        return GenerationResult(
            ok=True,
            latency_s=finished - started,
            started_at=started,
            finished_at=finished,
            text=(data or {}).get("text"),
            meta_info=meta,
            telemetry=extract_spec_telemetry(meta),
            http_status=r.status_code,
        )
