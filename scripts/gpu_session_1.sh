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

# The exact SGLang commit this session expects. Session 1 runs the iteration-trace
# branch, so the expected commit is the telemetry patch, NOT the pinned upstream
# base. Override EXPECTED_SGLANG_SHA deliberately if you are running something else.
EXPECTED_SGLANG_SHA="${EXPECTED_SGLANG_SHA:-f9281ec128}"
BASE_SHA="66ce8c55cc"

echo "== preflight =================================================="
if [ -d "$SGLANG_PATH/.git" ]; then
  ACTUAL="$(git -C "$SGLANG_PATH" rev-parse HEAD)"
  BRANCH="$(git -C "$SGLANG_PATH" rev-parse --abbrev-ref HEAD)"
  echo "sglang branch    : $BRANCH"
  echo "sglang HEAD      : ${ACTUAL:0:10}"
  echo "expected         : ${EXPECTED_SGLANG_SHA}"
  echo "pinned base      : ${BASE_SHA}"

  # FAIL, do not warn. A run at an unexpected commit is not reproducible, and the
  # whole point of the preflight is to catch that before the meter starts.
  if [ "${ALLOW_UNPINNED:-0}" != "1" ]; then
    case "$ACTUAL" in
      "$EXPECTED_SGLANG_SHA"*)
        echo "commit check     : OK"
        ;;
      *)
        echo >&2
        echo "ERROR: sglang HEAD (${ACTUAL:0:10}) is not the expected commit" >&2
        echo "       (${EXPECTED_SGLANG_SHA}). Refusing to spend GPU time on an" >&2
        echo "       unrecorded revision." >&2
        echo >&2
        echo "       Fix:  git -C $SGLANG_PATH checkout heterospec/iter-telemetry" >&2
        echo "       Or, if this is deliberate:" >&2
        echo "             EXPECTED_SGLANG_SHA=<sha> bash $0 ..." >&2
        echo "       Or, to run at whatever is checked out (results will record" >&2
        echo "       the actual SHA but will not be comparable to Session 1):" >&2
        echo "             ALLOW_UNPINNED=1 bash $0 ..." >&2
        exit 1
        ;;
    esac
  else
    echo "commit check     : SKIPPED (ALLOW_UNPINNED=1)"
  fi

  # A clean checkout of a known patch commit IS reproducible and citable; only
  # uncommitted edits are not. This is a warning, not a refusal.
  if [ -n "$(git -C "$SGLANG_PATH" status --porcelain)" ]; then
    echo "working tree     : DIRTY -- uncommitted changes present."
    echo "                   metadata.json will record working_tree_dirty=true and"
    echo "                   the run will NOT be citable. Commit or stash first if"
    echo "                   you want these results to count."
  else
    echo "working tree     : clean"
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
