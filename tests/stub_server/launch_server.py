"""A stub ``python -m launch_server`` for exercising the session orchestrator.

`heterospec.session.run_session` launches SGLang as a subprocess and polls its
health endpoint. That orchestration -- argument construction, readiness polling,
teardown, failure handling -- is the code that runs on the rented host, and it
cannot be tested against a real server on a Mac.

This module stands in for it. It is pointed at by
``ServerProcess(launcher_module="launch_server")`` with this directory on
``PYTHONPATH``, so the *real* orchestration path is exercised: a genuine
subprocess, a genuine socket, genuine readiness polling, genuine SIGTERM teardown.

It parses the depth from ``--speculative-num-steps`` so a static-K config behaves
like a static-K server, which keeps the resulting captures realistic enough to
flow through the cost model and the oracle.

It also dumps its environment to ``<port>.env.json`` so tests can assert that
`env_extra` really reaches the child process -- the mechanism that carries
`HF_HOME` and `HF_TOKEN` on the GPU host.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import threading
import time
from pathlib import Path

# The heterospec package is importable from the repo root during tests.
from heterospec.mockserver import MockSGLangServer


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="launch_server", add_help=False)
    p.add_argument("--port", type=int, default=30000)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--speculative-num-steps", type=int, default=0)
    p.add_argument("--model-path", default="")
    # Swallow everything else; the real launcher has dozens of flags.
    args, _unknown = p.parse_known_args(argv)
    return args


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = _parse_args(argv)

    # Absolute path via env var so tests can find it regardless of the child's
    # working directory (which is the sglang checkout, not the test's cwd).
    env_dump = Path(os.environ.get("STUB_ENV_DUMP", f"{args.port}.env.json"))
    env_dump.write_text(json.dumps(dict(os.environ), indent=2))

    print(
        f"[stub launch_server] port={args.port} k={args.speculative_num_steps}",
        flush=True,
    )

    server = MockSGLangServer(port=args.port, k=max(1, args.speculative_num_steps))
    # Test hook: answer /health but fail /generate, to exercise the orchestrator's
    # "server is up but not serving" path.
    if os.environ.get("STUB_FAIL_GENERATE") == "1":
        server.fail_generate = True
    server.start()

    stop = threading.Event()

    def _handle(signum, _frame):  # noqa: ANN001
        print(f"[stub launch_server] signal {signum}, shutting down", flush=True)
        stop.set()

    signal.signal(signal.SIGTERM, _handle)
    signal.signal(signal.SIGINT, _handle)

    try:
        while not stop.is_set():
            time.sleep(0.1)
    finally:
        server.stop()
    print("[stub launch_server] stopped", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
