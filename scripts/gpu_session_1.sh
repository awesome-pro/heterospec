#!/usr/bin/env bash
# GPU Session 1 — the single decisive real-GPU experiment.
#
# Everything that can be done on the Mac is done on the Mac. This script is the
# only thing that runs on the rented host, and it is designed to complete in
# about an hour so the bill stays small.
#
# What it produces
#   1. a measured Cost(K, batch_size) grid         (static K x concurrency)
#   2. per-request acceptance profiles              (same runs)
#   3. the K-invariance check                       (same runs, different K)
#   4. an adaptive baseline with the iteration trace (validates the trace patch)
#   5. a recoverable rectangular gap from REAL traces + REAL costs
#
# Usage, from the heterospec repo root on the GPU host:
#
#   bash scripts/gpu_session_1.sh --sglang-path /workspace/sglang
#
# Add --plan to print the run plan and exit without touching the GPU.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SGLANG_PATH="${SGLANG_PATH:-$REPO_ROOT/../sglang}"
LAUNCH_CONFIG="${LAUNCH_CONFIG:-configs/models/llama31_8b_eagle3.json}"
RESULTS_ROOT="${RESULTS_ROOT:-$REPO_ROOT/results/session1}"
PYTHON="${PYTHON:-python3}"

# Verify the fork is at the pinned commit before spending money.
PINNED="66ce8c55cc"
echo "== preflight =================================================="
if [ -d "$SGLANG_PATH/.git" ]; then
  ACTUAL="$(git -C "$SGLANG_PATH" rev-parse --short=10 HEAD)"
  echo "sglang HEAD      : $ACTUAL"
  if [ "$ACTUAL" != "$PINNED" ] && [ "${ALLOW_OTHER_COMMIT:-0}" != "1" ]; then
    echo "WARNING: HEAD ($ACTUAL) != pinned base ($PINNED)."
    echo "         Results will be tagged non-citable. Set ALLOW_OTHER_COMMIT=1"
    echo "         if this is deliberate (e.g. the telemetry branch)."
  fi
  if [ -n "$(git -C "$SGLANG_PATH" status --porcelain)" ]; then
    echo "note: sglang tree is dirty (expected on the telemetry branch)."
    echo "      metadata.json will record dirty=true and results will not be citable."
  fi
else
  echo "ERROR: no git checkout at $SGLANG_PATH" >&2
  exit 1
fi

echo "python           : $($PYTHON --version 2>&1)"
echo "launch config    : $LAUNCH_CONFIG"
echo "results root     : $RESULTS_ROOT"
echo "gpu              :"
nvidia-smi --query-gpu=name,memory.total,driver_version \
  --format=csv,noheader 2>/dev/null || echo "  (nvidia-smi unavailable)"
echo

mkdir -p "$RESULTS_ROOT/logs"

# Plan first: cheap, and shows the operator exactly what will be run.
exec "$PYTHON" -m heterospec.session \
  --launch-config "$LAUNCH_CONFIG" \
  --sglang-path "$SGLANG_PATH" \
  --results-root "$RESULTS_ROOT" \
  --logs-dir "$RESULTS_ROOT/logs" \
  --python "$PYTHON" \
  "$@"
