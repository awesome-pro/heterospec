"""A fake SGLang server for exercising the harness without a GPU.

Why this exists
---------------
Every line of harness code -- request dispatch, concurrency, `meta_info`
parsing, the histogram -> survival derivation, aggregate statistics, the oracle
study, the plots -- can be written, debugged and tested on the Mac for $0. Only
the *numbers* need a GPU. Without this, the first GPU hour would be spent
debugging `KeyError` and off-by-one bugs at $1/hour.

It is a mock, so it is dangerous if mistaken for real output. Two guards:

* `/server_info` reports ``"mock": true``;
* :meth:`RunMetadata.citable` refuses any run whose server mode is ``mock``.

Simulation model
----------------
Per prompt class, a vector of conditional per-position acceptance probabilities
``p = (p_1..p_K)``. A round walks the chain, stopping at the first rejection, so
``P(accept >= k) = prod_{i<=k} p_i`` -- the same prefix-contiguous structure real
topk=1 chain drafting has. Rounds are drawn until the token budget is met, and
``completion_tokens`` is derived from the accepted drafts plus one bonus token
per round. That makes the mock self-consistent:

    completion_tokens == spec_num_correct_drafts + spec_verify_ct

which is exactly the invariant a real capture must satisfy, so the harness is
tested against the relationship it will rely on.
"""

from __future__ import annotations

import itertools
import json
import random
import re
import threading
import zlib
from collections.abc import Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

__all__ = [
    "ACCEPTANCE_PROFILES",
    "CLASS_KEYWORDS",
    "CLASS_ORDER",
    "MockSGLangServer",
    "expected_accept_length",
    "simulate_rounds",
]

#: Conditional per-position acceptance, by prompt class. Deliberately spread
#: wide: `repetitive` accepts deeply, `open_ended` collapses almost at once.
#: The point is to give the analysis pipeline a real signal to find.
ACCEPTANCE_PROFILES: dict[str, tuple[float, ...]] = {
    "repetitive": (0.98, 0.96, 0.94, 0.92, 0.90, 0.88, 0.86),
    "open_ended": (0.62, 0.30, 0.13, 0.05, 0.02, 0.01, 0.01),
    "code": (0.90, 0.82, 0.72, 0.60, 0.50, 0.40, 0.32),
    "reasoning": (0.80, 0.64, 0.48, 0.33, 0.22, 0.14, 0.09),
    "chat": (0.70, 0.44, 0.24, 0.11, 0.05, 0.03, 0.02),
    # Unknown class: middling, so a mislabelled class shows up as an outlier
    # rather than masquerading as a strong result.
    "_default": (0.75, 0.55, 0.35, 0.20, 0.12, 0.07, 0.04),
}


def expected_accept_length(p: tuple[float, ...], k: int) -> float:
    """`E[accepted drafts] = sum_{k} P(accept >= k)` (bonus token excluded)."""
    total, prod = 0.0, 1.0
    for i in range(min(k, len(p))):
        prod *= p[i]
        total += prod
    return total


# ---------------------------------------------------------------------------
# Prompt classification for the mock
# ---------------------------------------------------------------------------
#
# Distinctive phrases per class, matched on **word boundaries**. Naive substring
# matching is not good enough: `"should i"` matches inside `"should introduce"`,
# which silently collapsed an open-ended prompt into `chat`. Scoring also beats
# first-match-wins here, because words like "exactly" appear in more than one
# class and only a distinctive phrase identifies the intent.
#
# A test asserts that every prompt in `heterospec.workloads.PROMPT_CLASSES`
# classifies as its declared class, so this table cannot silently drift from the
# workloads it is meant to double.

CLASS_ORDER: tuple[str, ...] = (
    "repetitive",
    "open_ended",
    "code",
    "reasoning",
    "chat",
)

CLASS_KEYWORDS: dict[str, tuple[str, ...]] = {
    "repetitive": (
        "one per line",
        "own line",
        "no other text",
        "no commentary",
        "no numbering",
        "nothing else",
        "each identical",
        "consisting only",
        "output exactly",
        "repeat the following",
        "write the integers",
        "print the string",
    ),
    "open_ended": (
        "poem",
        "biographies",
        "inventors",
        "diary",
        "headlines",
        "dream sequences",
        "colours",
        "colors",
        "impossible machines",
        "dialogue",
        "flora",
    ),
    "code": (
        "function",
        "implement",
        "query",
        "script",
        "decorator",
        "quicksort",
        "trie",
        "duplicate files",
        "parses",
        "sql",
    ),
    "reasoning": (
        "how much",
        "how many",
        "prove",
        "derive",
        "weighings",
        "reasoning",
        "puzzle",
        "induction",
        "probability",
        "each step",
    ),
    "chat": (
        "hey",
        "how's it",
        "tips",
        "suggestions",
        "motivated",
        "ocean",
        "cook",
        "cooking",
        "help with",
        "what you are",
        "rough day",
        "getting a pet",
    ),
}

# A few dozen words is not a measurable cost, and \b prevents the substring
# collisions above.
_EXTRA_PATTERNS: dict[str, tuple[str, ...]] = {
    "repetitive": (r"\b\d{2,} lines\b",),
}

_COMPILED: dict[str, list[re.Pattern[str]]] = {
    cls: [re.compile(rf"\b{re.escape(kw)}\b", re.I) for kw in kws]
    + [re.compile(p, re.I) for p in _EXTRA_PATTERNS.get(cls, ())]
    for cls, kws in CLASS_KEYWORDS.items()
}


#: Sampling keys the pinned SGLang's ``SamplingParams`` accepts.
#:
#: The mock validates against this because the real server builds
#: ``SamplingParams(**sampling_kwargs)`` and raises ``TypeError: Unexpected
#: keyword argument 'X'`` for anything else -- surfacing as HTTP 500 on every
#: request. Nothing caught that locally for 400+ tests, because the mock accepted
#: whatever it was sent, and the failure only appeared on a rented GPU. Keep this
#: list in step with sglang/srt/sampling/sampling_params.py.
VALID_SAMPLING_KEYS = frozenset(
    {
        "max_new_tokens",
        "stop",
        "stop_token_ids",
        "stop_regex",
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "frequency_penalty",
        "presence_penalty",
        "repetition_penalty",
        "min_new_tokens",
        "n",
        "ignore_eos",
        "skip_special_tokens",
        "spaces_between_special_tokens",
        "no_stop_trim",
        "stream_interval",
        "logit_bias",
        "sampling_seed",
        "custom_params",
    }
)


def _classify(prompt: str) -> str:
    """Infer a prompt class from its text by weighted keyword scoring.

    The mock is deliberately given the *prompt*, not the harness's class label,
    so a mismatch between what the harness thinks it sent and what arrived still
    shows up as unexpected acceptance behaviour.
    """
    text = prompt.lower()
    best, best_score = "_default", 0
    for cls in CLASS_ORDER:
        score = sum(1 for pat in _COMPILED[cls] if pat.search(text))
        if score > best_score:
            best, best_score = cls, score
    return best


def simulate_rounds(
    *,
    p: tuple[float, ...],
    k: int,
    max_new_tokens: int,
    rng: random.Random,
    max_rounds: int = 4096,
) -> tuple[list[int], int]:
    """Draw acceptance rounds until the token budget is met.

    Returns `(accepted_per_round, completion_tokens)` where completion tokens
    counts accepted drafts plus one bonus token per round -- the same accounting
    SGLang uses.
    """
    accepted: list[int] = []
    tokens = 0
    while tokens < max_new_tokens and len(accepted) < max_rounds:
        acc = 0
        for i in range(k):
            if rng.random() < (p[i] if i < len(p) else p[-1]):
                acc += 1
            else:
                break
        accepted.append(acc)
        tokens += acc + 1  # accepted drafts + one bonus token
    return accepted, tokens


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    # Silence per-request logging; the tests would drown.
    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A002
        if self.server.verbose:  # type: ignore[attr-defined]
            super().log_message(fmt, *args)

    # -- helpers -------------------------------------------------------------

    def _send(self, code: int, payload: Any) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # -- routes --------------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._send(200, {"status": "ok"})
        elif self.path == "/server_info":
            self._send(
                200,
                {
                    "mock": True,
                    "internal_states": [
                        {
                            "speculative_num_steps": self.server.active_k,  # type: ignore[attr-defined]
                            "avg_spec_accept_length": self.server.last_accept_length,  # type: ignore[attr-defined]
                            "mock": True,
                        }
                    ],
                },
            )
        else:
            self._send(404, {"error": f"unknown path {self.path}"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/generate":
            self._send(404, {"error": f"unknown path {self.path}"})
            return

        length = int(self.headers.get("Content-Length") or 0)
        try:
            req = json.loads(self.rfile.read(length) or b"{}")
        except ValueError:
            self._send(400, {"error": "invalid JSON body"})
            return

        srv = self.server
        if getattr(srv, "fail_generate", False):
            self._send(500, {"error": "stub configured to fail /generate"})
            return
        prompt = req.get("text") or ""
        rid = req.get("rid") or f"mock-{next(srv.counter)}"  # type: ignore[attr-defined]
        sampling = req.get("sampling_params") or {}
        unknown = sorted(set(sampling) - VALID_SAMPLING_KEYS)
        if unknown:
            # Mirror the real server: it does not ignore unknown sampling keys, it
            # raises TypeError, which the client sees as an opaque HTTP 500.
            self._send(
                500,
                {
                    "error": (
                        "TypeError: Unexpected keyword argument "
                        f"'{unknown[0]}' (invalid sampling_params keys: {unknown}; "
                        "seed is spelled 'sampling_seed')"
                    )
                },
            )
            return
        max_new_tokens = int(sampling.get("max_new_tokens") or 128)

        cls = _classify(prompt)
        p = srv.profiles.get(cls, srv.profiles["_default"])  # type: ignore[attr-defined]

        # Active K: fixed, or cycled from a schedule when simulating adaptive mode.
        with srv.lock:  # type: ignore[attr-defined]
            k = srv.next_k()  # type: ignore[attr-defined]

        # Deterministic per (rid, k) so a rerun reproduces exactly.
        # zlib.crc32, not hash(): builtin hash() is salted per process, so a
        # rerun in a new interpreter would not reproduce.
        stable_seed = zlib.crc32(f"{rid}|{k}|{srv.seed}".encode())  # type: ignore[attr-defined]
        rng = random.Random(stable_seed)
        accepted, completion_tokens = simulate_rounds(
            p=p, k=k, max_new_tokens=max_new_tokens, rng=rng
        )

        histogram = [0] * (max(accepted) + 1 if accepted else 1)
        for a in accepted:
            histogram[a] += 1

        meta = {
            "completion_tokens": completion_tokens,
            "spec_verify_ct": len(accepted),
            "spec_num_correct_drafts": sum(accepted),
            "spec_correct_drafts_histogram": histogram,
            "prompt_tokens": max(1, len(prompt) // 4),
            "finish_reason": {"type": "length", "length": completion_tokens},
        }

        with srv.lock:  # type: ignore[attr-defined]
            srv.last_accept_length = (
                completion_tokens / len(accepted) if accepted else 0.0
            )

        self._send(
            200,
            {
                "text": "x" * completion_tokens,
                "meta_info": meta,
                "output_ids": list(range(completion_tokens)),
            },
        )


class MockSGLangServer:
    """Threaded mock SGLang server. Use as a context manager or start/stop."""

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        k: int | None = 4,
        k_schedule: list[int] | None = None,
        seed: int = 0,
        profiles: Mapping[str, tuple[float, ...]] | None = None,
        verbose: bool = False,
    ) -> None:
        if k is None and not k_schedule:
            raise ValueError("provide either k or k_schedule")
        if k_schedule is not None and not k_schedule:
            raise ValueError("k_schedule must be non-empty when given")

        self._k = k
        self._k_schedule = k_schedule
        self._k_index = 0
        self.seed = seed
        self.profiles = dict(profiles or ACCEPTANCE_PROFILES)
        self.verbose = verbose
        self.lock = threading.Lock()
        self.counter = itertools.count()
        self.last_accept_length = 0.0

        self._httpd = ThreadingHTTPServer((host, port), _Handler)
        self._httpd.daemon_threads = True
        # Attributes read by the handler.
        self._httpd.verbose = verbose  # type: ignore[attr-defined]
        self._httpd.profiles = self.profiles  # type: ignore[attr-defined]
        self._httpd.seed = seed  # type: ignore[attr-defined]
        self._httpd.lock = self.lock  # type: ignore[attr-defined]
        self._httpd.counter = self.counter  # type: ignore[attr-defined]
        self._httpd.next_k = self._next_k  # type: ignore[attr-defined]
        self._httpd.active_k = k if k is not None else k_schedule[0]  # type: ignore[attr-defined]
        self._httpd.last_accept_length = 0.0  # type: ignore[attr-defined]
        # Read by the handler; exposed via the `fail_generate` property.
        self._httpd.fail_generate = False  # type: ignore[attr-defined]
        self._thread: threading.Thread | None = None

    # -- fault injection -----------------------------------------------------

    @property
    def fail_generate(self) -> bool:
        """When true, `/generate` returns 500 while `/health` stays OK.

        Lets tests exercise the "server is up but not serving" path, which is the
        one that would otherwise silently produce a cost model built on failures.
        """
        return bool(self._httpd.fail_generate)  # type: ignore[attr-defined]

    @fail_generate.setter
    def fail_generate(self, value: bool) -> None:
        self._httpd.fail_generate = bool(value)  # type: ignore[attr-defined]

    # -- active K ------------------------------------------------------------

    def _next_k(self) -> int:
        if self._k_schedule:
            k = self._k_schedule[self._k_index % len(self._k_schedule)]
            self._k_index += 1
        else:
            assert self._k is not None
            k = self._k
        self._httpd.active_k = k  # type: ignore[attr-defined]
        return k

    # -- lifecycle -----------------------------------------------------------

    @property
    def port(self) -> int:
        return int(self._httpd.server_address[1])

    @property
    def base_url(self) -> str:
        host, port = self._httpd.server_address[0], self._httpd.server_address[1]
        return f"http://{host}:{port}"

    def start(self) -> MockSGLangServer:
        self._thread = threading.Thread(
            target=self._httpd.serve_forever, name="mock-sglang", daemon=True
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def __enter__(self) -> MockSGLangServer:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()
