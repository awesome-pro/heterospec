#!/usr/bin/env bash
# Bootstrap a fresh GPU host for HeteroSpec Session 1.
#
# This is the *setup* half of the session, split out so it can run once on the
# first boot and be re-run safely later. It deliberately does NOT run the paid
# experiment: it ends by printing the exact command to start it, so the meter
# never starts as a side effect of a setup script.
#
#   bash scripts/gpu_bootstrap.sh                 # full setup, then the free dry run
#   bash scripts/gpu_bootstrap.sh --check          # verify only, change nothing
#   bash scripts/gpu_bootstrap.sh --a100-40gb      # use the 40 GB launch config
#   bash scripts/gpu_bootstrap.sh --skip-models    # weights already on the volume
#   bash scripts/gpu_bootstrap.sh --skip-tests     # skip the CPU unit tests
#
# Everything it does is also written out by hand in docs/06-gpu-runbook.md; this
# script only sequences those steps and fails early. It touches no harness logic.

set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
HF_HOME="${HF_HOME:-$WORKSPACE/hf}"
SGLANG_DIR="$WORKSPACE/sglang"
HETEROSPEC_DIR="$WORKSPACE/heterospec"

# Pinned revisions. These must match scripts/gpu_session_1.sh, which hard-fails the
# paid run if HEAD is not TELEMETRY_SHA.
TELEMETRY_BRANCH="heterospec/iter-telemetry"
TELEMETRY_SHA="f9281ec128"
BASE_SHA="66ce8c55cc"
PR1_BRANCH="heterospec/policy-feedback-identity"
PR1_SHA="65aaba0418"

SGLANG_URL="${SGLANG_URL:-https://github.com/awesome-pro/sglang.git}"
HETEROSPEC_URL="${HETEROSPEC_URL:-https://github.com/awesome-pro/heterospec.git}"

TARGET_MODEL="meta-llama/Llama-3.1-8B-Instruct"
DRAFT_MODEL="lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B"

CONFIG_48GB="configs/models/llama31_8b_eagle3.json"
CONFIG_40GB="configs/models/llama31_8b_eagle3_a100_40gb.json"

LAUNCH_CONFIG="$CONFIG_48GB"
CHECK_ONLY=0
DO_MODELS=1
DO_TESTS=1

WARNINGS=()
FAILURES=()

# ---------------------------------------------------------------------------
# output helpers
# ---------------------------------------------------------------------------

if [ -t 1 ]; then
  BOLD=$'\033[1m'; RED=$'\033[31m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'; OFF=$'\033[0m'
else
  BOLD=""; RED=""; GREEN=""; YELLOW=""; OFF=""
fi

step()  { printf '\n%s== %s %s%s\n' "$BOLD" "$1" "$(printf '=%.0s' $(seq 1 $((60 - ${#1}))))" "$OFF"; }
ok()    { printf '  %sok%s   %s\n' "$GREEN" "$OFF" "$1"; }
warn()  { printf '  %swarn%s %s\n' "$YELLOW" "$OFF" "$1"; WARNINGS+=("$1"); }
fail()  { printf '  %sFAIL%s %s\n' "$RED" "$OFF" "$1"; FAILURES+=("$1"); }
info()  { printf '       %s\n' "$1"; }

die() { printf '\n%serror:%s %s\n\n' "$RED" "$OFF" "$1" >&2; exit 1; }

usage() { sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 0; }

for arg in "$@"; do
  case "$arg" in
    --check)       CHECK_ONLY=1 ;;
    --a100-40gb)   LAUNCH_CONFIG="$CONFIG_40GB" ;;
    --skip-models) DO_MODELS=0 ;;
    --skip-tests)  DO_TESTS=0 ;;
    -h|--help)     usage ;;
    *) die "unknown argument: $arg (try --help)" ;;
  esac
done

printf '%sHeteroSpec Session 1 — host bootstrap%s\n' "$BOLD" "$OFF"
[ "$CHECK_ONLY" = "1" ] && info "mode: --check (will not modify anything)"

# ---------------------------------------------------------------------------
# 1. host prerequisites
# ---------------------------------------------------------------------------

step "1. host prerequisites"

command -v git >/dev/null 2>&1 && ok "git $(git --version | awk '{print $3}')" \
  || fail "git not found"

PYTHON="${PYTHON:-python3}"
if command -v "$PYTHON" >/dev/null 2>&1; then
  PYVER="$("$PYTHON" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || echo "?")"
  # heterospec requires >=3.11,<3.13. A rented image at 3.10 or 3.13 fails the
  # editable install, and discovering that after the model download wastes money.
  case "$PYVER" in
    3.11|3.12) ok "python $PYVER ($("$PYTHON" -c 'import sys; print(sys.executable)'))" ;;
    *)         fail "python $PYVER is outside heterospec's requires-python >=3.11,<3.13" ;;
  esac
else
  fail "$PYTHON not found"
fi

# `python -m pip` is what the install step uses. Some minimal images ship a python
# with no pip module at all, and finding that out after the model download is an
# expensive way to learn it.
if command -v "$PYTHON" >/dev/null 2>&1; then
  if "$PYTHON" -m pip --version >/dev/null 2>&1; then
    ok "$("$PYTHON" -m pip --version | awk '{print $1, $2}')"
  else
    fail "python -m pip is unavailable"
    info "install pip first, e.g. python -m ensurepip --upgrade"
  fi
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_LINE="$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | head -1)"
  ok "gpu: $GPU_LINE"
  MEM_MB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)"
  if [ "${MEM_MB:-0}" -lt 40000 ]; then
    fail "only ${MEM_MB} MiB VRAM: Llama-3.1-8B fp16 + EAGLE3 draft need ~17.6 GB before any KV cache"
    info "use a 48 GB (L40S/A6000) or 40 GB (A100) card; 24 GB cannot work"
  elif [ "${MEM_MB:-0}" -lt 44000 ] && [ "$LAUNCH_CONFIG" = "$CONFIG_48GB" ]; then
    warn "${MEM_MB} MiB is under 48 GB: re-run with --a100-40gb"
  fi
else
  warn "nvidia-smi unavailable — cannot verify the GPU from here; the session will fail at launch"
fi

mkdir -p "$WORKSPACE" 2>/dev/null || true
if [ -d "$WORKSPACE" ]; then
  AVAIL_GB="$(df -Pk "$WORKSPACE" | awk 'NR==2 {printf "%d", $4/1024/1024}')"
  if [ "${AVAIL_GB:-0}" -lt 60 ]; then
    warn "only ${AVAIL_GB} GB free at $WORKSPACE: weights (~17 GB) + build + traces want 60 GB+"
  else
    ok "${AVAIL_GB} GB free at $WORKSPACE"
  fi
fi

[ "$CHECK_ONLY" = "1" ] || mkdir -p "$HF_HOME"
ok "HF_HOME=$HF_HOME"
if [ -n "${HF_TOKEN:-}" ]; then
  ok "HF_TOKEN is set (needed for the gated Llama-3.1 licence)"
else
  warn "HF_TOKEN is not set: the gated target model cannot be downloaded"
  info "accept the licence at https://huggingface.co/$TARGET_MODEL then export HF_TOKEN=hf_..."
fi

if [ "${#FAILURES[@]}" -gt 0 ]; then
  printf '\n%s%d prerequisite(s) failed — fix these before spending GPU time.%s\n' \
    "$RED" "${#FAILURES[@]}" "$OFF"
  exit 1
fi

# ---------------------------------------------------------------------------
# 2. code at the pinned revisions
# ---------------------------------------------------------------------------

step "2. code onto the host"

clone_or_update() {
  local dir="$1" url="$2" branch="$3" sha="$4" label="$5"
  if [ -d "$dir/.git" ]; then
    if [ "$CHECK_ONLY" = "1" ]; then
      ok "$label present at $dir"
    else
      git -C "$dir" fetch --quiet origin "$branch"
      ok "$label fetched"
    fi
  elif [ "$CHECK_ONLY" = "1" ]; then
    fail "$label missing at $dir"
    return 0
  else
    git clone --quiet --branch "$branch" "$url" "$dir" || { fail "clone $label failed"; return 0; }
    ok "$label cloned"
  fi
  local head
  head="$(git -C "$dir" rev-parse HEAD 2>/dev/null || echo "?")"
  if [ -z "$sha" ]; then
    ok "$label at ${head:0:10} (branch $branch, no pin)"
  elif [ "${head:0:10}" = "$sha" ]; then
    ok "$label HEAD ${head:0:10} matches the pin"
  else
    warn "$label HEAD ${head:0:10} != pinned $sha (branch $branch)"
  fi
}

clone_or_update "$SGLANG_DIR" "$SGLANG_URL" "$TELEMETRY_BRANCH" "$TELEMETRY_SHA" "sglang"
clone_or_update "$HETEROSPEC_DIR" "$HETEROSPEC_URL" "main" "" "heterospec"

# The session's preflight hard-fails on any other commit, so check it here too:
# it is the single most likely reason a paid run refuses to start.
if [ -d "$SGLANG_DIR/.git" ]; then
  if [ "$CHECK_ONLY" != "1" ]; then
    git -C "$SGLANG_DIR" checkout --quiet "$TELEMETRY_BRANCH" 2>/dev/null || true
  fi
  HEAD_NOW="$(git -C "$SGLANG_DIR" rev-parse HEAD 2>/dev/null || echo "?")"
  if [ "${HEAD_NOW:0:10}" = "$TELEMETRY_SHA" ]; then
    ok "sglang is on $TELEMETRY_BRANCH at the expected commit"
  else
    fail "sglang HEAD ${HEAD_NOW:0:10} is not $TELEMETRY_SHA; the session preflight will refuse to run"
    info "git -C $SGLANG_DIR checkout $TELEMETRY_BRANCH"
  fi
  if [ -n "$(git -C "$SGLANG_DIR" status --porcelain)" ]; then
    warn "sglang working tree is DIRTY — metadata.json will record it and the run will not be citable"
  else
    ok "sglang working tree clean"
  fi
  if [ -f "$SGLANG_DIR/python/sglang/srt/speculative/heterospec_trace.py" ]; then
    N="$(grep -c record_iteration "$SGLANG_DIR/python/sglang/srt/speculative/heterospec_trace.py" || true)"
    [ "${N:-0}" -gt 0 ] && ok "iteration-trace patch present ($N record_iteration call sites)" \
      || fail "trace module exists but looks empty — wrong revision?"
  else
    fail "trace module missing: this checkout does not have the iteration trace"
  fi
fi

# ---------------------------------------------------------------------------
# 3. install
# ---------------------------------------------------------------------------

step "3. install"

if [ "$CHECK_ONLY" = "1" ]; then
  if "$PYTHON" -c "import sglang, heterospec" >/dev/null 2>&1; then
    ok "sglang and heterospec import"
  else
    fail "sglang/heterospec do not import (run without --check to install)"
  fi
else
  info "installing SGLang (editable) — this is the slow step, several minutes"
  if "$PYTHON" -m pip install --quiet -e "$SGLANG_DIR/python"; then
    ok "sglang installed"
  else
    fail "pip install -e $SGLANG_DIR/python failed"
  fi
  if "$PYTHON" -m pip install --quiet -e "$HETEROSPEC_DIR"; then
    ok "heterospec installed (with requests/numpy/pandas/matplotlib/scipy)"
  else
    fail "pip install -e $HETEROSPEC_DIR failed"
  fi
  if "$PYTHON" -c "import sglang, heterospec" >/dev/null 2>&1; then
    ok "both import cleanly"
  else
    fail "import check failed after install"
  fi
fi

# ---------------------------------------------------------------------------
# 4. models (before the clock matters)
# ---------------------------------------------------------------------------

step "4. model weights"

if [ "$DO_MODELS" = "0" ]; then
  info "skipped (--skip-models)"
elif [ "$CHECK_ONLY" = "1" ]; then
  for m in "$TARGET_MODEL" "$DRAFT_MODEL"; do
    d="$HF_HOME/hub/models--$(echo "$m" | tr '/' '-')"
    [ -d "$d" ] && ok "cached: $m" || warn "not in cache: $m"
  done
else
  "$PYTHON" -m pip install --quiet -U "huggingface_hub[cli]" || warn "could not update huggingface_hub"
  if command -v hf >/dev/null 2>&1; then DL=(hf download); else DL=(huggingface-cli download); fi
  for m in "$TARGET_MODEL" "$DRAFT_MODEL"; do
    info "downloading $m (~17 GB total for both)"
    if "${DL[@]}" "$m" >/dev/null 2>&1; then
      ok "downloaded $m"
    else
      fail "download failed for $m"
      case "$m" in
        "$TARGET_MODEL") info "this model is gated: accept the licence and export HF_TOKEN" ;;
      esac
    fi
  done
fi

# ---------------------------------------------------------------------------
# 5. the fork's CPU unit tests (first place they can run at all)
# ---------------------------------------------------------------------------

step "5. fork unit tests"

if [ "$DO_TESTS" = "0" ]; then
  info "skipped (--skip-tests)"
elif [ "$CHECK_ONLY" = "1" ]; then
  info "skipped in --check mode (these run against the checkout)"
elif [ -d "$SGLANG_DIR/.git" ]; then
  cd "$SGLANG_DIR"
  run_test() {
    local f="$1"
    if [ ! -f "$f" ]; then fail "missing test file $f"; return 0; fi
    if "$PYTHON" -m pytest -q "$f" >/tmp/ht_$(basename "$f").log 2>&1; then
      ok "$(basename "$f")"
    else
      fail "$(basename "$f") — see /tmp/ht_$(basename "$f").log"
    fi
  }
  # PR 1 lives on a different, independent branch, so it needs its own checkout.
  info "checking out $PR1_BRANCH for the three PR 1 files"
  git checkout --quiet "$PR1_BRANCH" 2>/dev/null || warn "could not check out $PR1_BRANCH"
  P1="$(git rev-parse HEAD 2>/dev/null || echo '?')"
  [ "${P1:0:10}" = "$PR1_SHA" ] && ok "PR 1 branch at $PR1_SHA" \
    || warn "PR 1 branch at ${P1:0:10}, expected $PR1_SHA"
  run_test test/registered/unit/spec/test_adaptive_runtime_state.py
  run_test test/registered/unit/spec/test_adaptive_spec_params.py
  run_test test/registered/unit/managers/test_batch_result_processor_spec_grammar.py

  info "checking out $TELEMETRY_BRANCH for the trace patch tests"
  git checkout --quiet "$TELEMETRY_BRANCH" 2>/dev/null || warn "could not check out $TELEMETRY_BRANCH"
  run_test test/registered/unit/spec/test_heterospec_trace.py

  # The session runs the telemetry branch, so leave the tree there either way.
  git checkout --quiet "$TELEMETRY_BRANCH" 2>/dev/null || true
  cd - >/dev/null
  ok "left sglang on $TELEMETRY_BRANCH (the session branch)"
  if [ "${#FAILURES[@]}" -gt 0 ]; then
    info "PR 1 must not be opened until its three files pass. The session itself is unaffected."
  fi
fi

# ---------------------------------------------------------------------------
# 6. the free dry run
# ---------------------------------------------------------------------------

step "6. dry run (free — no GPU work)"

if [ "$CHECK_ONLY" = "1" ]; then
  info "skipped in --check mode"
elif [ -d "$HETEROSPEC_DIR" ]; then
  cd "$HETEROSPEC_DIR"
  if "$PYTHON" -m heterospec.session --launch-config "$LAUNCH_CONFIG" --plan 2>&1 | sed 's/^/       /'; then
    ok "plan generated"
  else
    fail "could not generate the plan"
  fi
  cd - >/dev/null
fi

# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------

step "summary"

printf '  launch config : %s\n' "$LAUNCH_CONFIG"
printf '  sglang        : %s\n' "$SGLANG_DIR"
printf '  heterospec    : %s\n' "$HETEROSPEC_DIR"
printf '  HF_HOME       : %s\n' "$HF_HOME"

if [ "${#WARNINGS[@]}" -gt 0 ]; then
  printf '\n  %d warning(s):\n' "${#WARNINGS[@]}"
  for w in "${WARNINGS[@]}"; do printf '    - %s\n' "$w"; done
fi

if [ "${#FAILURES[@]}" -gt 0 ]; then
  printf '\n  %s%d problem(s):%s\n' "$RED" "${#FAILURES[@]}" "$OFF"
  for f in "${FAILURES[@]}"; do printf '    - %s\n' "$f"; done
  printf '\n  Fix these before starting the paid run.\n\n'
  exit 1
fi

if [ "$CHECK_ONLY" = "1" ]; then
  printf '\n  %sHost looks fit for Session 1.%s\n\n' "$GREEN" "$OFF"
  exit 0
fi

cat <<EOF

  Setup complete. Nothing has been run on the GPU yet.

  Now start the session — this is the only step that costs money:

    cd $HETEROSPEC_DIR
    bash scripts/gpu_session_1.sh --sglang-path $SGLANG_DIR \\
        --launch-config $LAUNCH_CONFIG

  Expect 16 runs across 6 server launches, ~72 min nominal / ~116 min pessimistic.
  When it finishes, copy the whole results directory off the host before stopping
  the machine:

    $HETEROSPEC_DIR/results/session1/

EOF
