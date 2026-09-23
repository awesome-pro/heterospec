# 06 — GPU runbook: Session 1

The only work that must happen on rented hardware, written so nothing on the GPU
side needs authoring or debugging. Everything it depends on is built and tested on
the Mac (420 tests, no GPU).

Target: **roughly 16 benchmark runs across 6 server launches**, estimated at
~72 min nominal / ~116 min pessimistic. Budget against the pessimistic figure:
the dominant uncertainty is per-tier CUDA graph capture, and the adaptive server
builds five tiers (one per candidate step) against one for each static arm, so it
starts far slower than a static server. Both figures are order-of-magnitude
placeholders until the first real session replaces the coefficients.

---

## 1. Which GPU

**Recommendation: a 48 GB card — L40S or A6000.** On a tighter budget, A100 40 GB
works with the 40 GB config variant.

The server OOMs **before allocating any KV cache**. Lowering
`mem_fraction_static` does not help — the weights are the weights. The canonical
SGLang EAGLE3 test config (`test_basic_sanity_eagle3.py`) uses `mem_fraction_static
0.7`, which is safe on 40 GB+ and not on 24 GB.

| Card | Config | `mem_fraction_static` | `cuda_graph_max_bs_decode` | Notes |
| --- | --- | --- | --- | --- |
| 48 GB (L40S / A6000) | `configs/models/llama31_8b_eagle3.json` | 0.7 | 64 | preferred |
| 40 GB (A100 40GB) | `configs/models/llama31_8b_eagle3_a100_40gb.json` | 0.85 | 32 | works; ~16 GB left for KV + graphs |

Do **not** reach for a quantized target to fit a smaller card. AWQ-INT4 would fit
24 GB but changes acceptance behaviour, which is precisely the quantity being
measured. It would be a confounder, not a saving.

## 2. Persist the machine

Set up once, then reuse for the whole project:

* **Network volume** mounted at `/workspace` (100 GB is ample). Model weights are
  ~17 GB and would otherwise be re-downloaded on every start.
* **Container disk** ≥ 60 GB, ideally 100 GB. 30 GB is **not** enough: the weights
  are ~17.6 GB, and the pinned SGLang pulls `torch==2.13.0` plus cu13
  `flashinfer`/`sgl-kernel` wheels on top of the base image. Container disk is
  ~$0.004/hr per 30 GB, so the extra space costs about a cent per hour.
* Set `HF_HOME=/workspace/hf` as an environment variable so weights land on the
  volume.

### The image must match the pinned SGLang

The fork's `docker/Dockerfile` at the session commit is the authority here, not
"whatever the newest template is". It builds from
`nvidia/cuda:13.0.3-cudnn-devel-ubuntu24.04` with Python 3.12, and
`python/pyproject.toml` pins:

```text
torch==2.13.0
cuda-python>=13.0
flashinfer_python[cu13]==0.6.18
```

So pick **CUDA 13.0** and a **PyTorch 2.13.0** image. Two failure modes to avoid:

* A **cu12** image (e.g. a "PyTorch 2.8.0 / CUDA 12.8" template) is wrong on both
  axes. `pip install -e` will then drag in torch 2.13.0 and cu13 kernels — several
  GB — and the preinstalled torchvision/flashinfer built against the old torch can
  be left ABI-mismatched. This is the most likely way to lose 20 minutes.
* Python **3.13+** or **3.10**: `heterospec` requires `>=3.11,<3.13`. The official
  image uses 3.12, which is the target.

`scripts/gpu_bootstrap.sh --check` verifies all of this before anything is spent,
including a `torch`/CUDA version check against the pin.

## 3. Pre-download the models (do this before the clock matters)

Either during setup or as the first step of the session:

```bash
export HF_HOME=/workspace/hf
pip install -U "huggingface_hub[cli]"
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct
huggingface-cli download lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B
```

`meta-llama/Llama-3.1-8B-Instruct` is gated: accept the licence on Hugging Face and
set `HF_TOKEN` first, or the session dies at the first server launch.

## 4. Get the code onto the host

Sections 4–5 are scripted, because they are the part that costs billable time if
done by hand and wrong. The script sequences exactly the steps below, verifies each
one, and **never starts the paid run** — it ends by printing the command for
section 6:

```bash
# verify the host without changing anything (safe to run first)
bash scripts/gpu_bootstrap.sh --check

# clone both repos at the pinned revisions, install, pre-download, dry-run
bash scripts/gpu_bootstrap.sh                 # add --a100-40gb on a 40 GB card
```

It fails fast on the things that otherwise surface mid-session: Python outside
`>=3.11,<3.13`, a missing `pip`, too little VRAM or disk, a missing `HF_TOKEN`, the
wrong SGLang commit, or a dirty fork checkout. The manual equivalent follows.

```bash
export HF_HOME=/workspace/hf      # weights live on the volume, not the container

cd /workspace
# the fork, on the telemetry branch (the trace patch lives there)
git clone --branch heterospec/iter-telemetry \
    https://github.com/awesome-pro/sglang.git sglang
# the harness
git clone https://github.com/awesome-pro/heterospec.git heterospec

# SGLang itself
pip install -e /workspace/sglang/python
# ...and the harness, which brings requests/numpy/pandas/matplotlib/scipy.
# Without this, `python -m heterospec.session` only works from the repo root,
# because `python -m` happens to put the cwd on sys.path.
pip install -e /workspace/heterospec
```

Verify both, before spending anything:

```bash
python -c "import sglang, heterospec; print('imports ok')"
grep -c record_iteration \
    /workspace/sglang/python/sglang/srt/speculative/heterospec_trace.py
```

`HF_HOME` is inherited by every server the session launches, so weights are read
from the volume rather than re-downloaded per launch.

### 4b. Run the fork's unit tests (five minutes, and it verifies PR 1)

SGLang's spec unit tests import torch, so they **cannot run on the Mac**. The host
is the first place they can be executed, and doing it here means the upstream PR
can be opened with verified tests rather than hopeful ones:

```bash
cd /workspace/sglang
git checkout heterospec/policy-feedback-identity

# PR 1: request identity on the policy feedback path (three files)
python -m pytest -q test/registered/unit/spec/test_adaptive_runtime_state.py
python -m pytest -q test/registered/unit/spec/test_adaptive_spec_params.py
python -m pytest -q test/registered/unit/managers/test_batch_result_processor_spec_grammar.py

# the trace patch -- a DIFFERENT branch, so a separate checkout
git checkout heterospec/iter-telemetry
python -m pytest -q test/registered/unit/spec/test_heterospec_trace.py
```

All four are registered in CPU CI, so they need no GPU. If any PR 1 test fails,
report it before running the session — the session's results are unaffected, but
PR 1 must not be opened until its three files pass.

Then switch back for the session:

```bash
git checkout heterospec/iter-telemetry   # the session runs this branch
```

The two branches are independent (`docs/02-reproducibility.md`), so only one can
be checked out at a time — which is why the tests are listed in two groups.

## 5. Dry-run the plan (free, and worth doing)

```bash
cd /workspace/heterospec
python -m heterospec.session \
    --launch-config configs/models/llama31_8b_eagle3.json \
    --plan
```

Expect 16 runs across 6 server launches. Fix anything wrong **here**, not while
paying.

## 6. Run the session

```bash
cd /workspace/heterospec
bash scripts/gpu_session_1.sh --sglang-path /workspace/sglang \
    --launch-config configs/models/llama31_8b_eagle3.json
```

No override is needed. The preflight **exits non-zero** unless HEAD is the
expected telemetry commit, because a run at an unrecorded revision is not
reproducible and the point of the preflight is to catch that before the meter
starts. If you deliberately want another revision:

```bash
EXPECTED_SGLANG_SHA=<sha> bash scripts/gpu_session_1.sh ...   # named replacement
ALLOW_UNPINNED=1         bash scripts/gpu_session_1.sh ...    # whatever is checked out
```

A **clean** checkout of the telemetry patch is citable: `base_sha +
experiment_patch_sha` names it exactly. Only uncommitted changes disqualify a
run, and the script warns if any are present.

Two things the orchestrator now does that matter:

* **Untimed warm-up** at each step's own concurrency, before timing starts.
  `/health` returning 200 does not mean Triton kernels and allocator paths for
  the shapes you are about to measure are warm, and the first requests would
  otherwise be slower. Warm-up uses `warmup-*` rids and is discarded. If *every*
  warm-up request fails, the step is failed rather than timed — a cost model
  built on failures looks like data, which is worse than no result.
* **Tracing only the adaptive capture.** The tracer does synchronous JSON work
  once per decode iteration, so paying it on the runs that produce the cost
  model would be careless when the effect under study is a few percent.
  Calibration's oracle input comes from response `meta_info`, not the trace.

Every server is launched with `--disable-radix-cache`, set in the model config.
Prefix caching is not a variable under study, and the grid reuses the same seeded
prompt pool across concurrencies on one server, so a warm cache would make later
runs look cheaper and contaminate the batch-size axis. SGLang's own benchmarks
pass this flag for the same reason. Note it also lowers *absolute* throughput, so
absolute numbers are not comparable with a cache-enabled run — only the ratios
this project uses are.

The script prints progress and, at the end:

```text
report:             results/session1/session1_report.json
K-invariance:       PASS / FAIL / NOT ASSESSED
recoverable rectangular gap over N captures: mean +X.XX% (min, max)
```

Use `--analyze-only` to re-analyse without re-running anything. It reads the
report back and rebuilds the steps from it, so it needs no server and no GPU. Run
it against the **copied** directory: paths in the report are stored relative to
`results_root`, so the re-analysis rebases onto wherever you put it.

The number it prints is the gap **at the deepest static K**, over the confluence
of measured cells — not an average across `K`. Shallower `K` captures are listed
under `excluded` with the reason, and the traced adaptive capture is reported
separately as a diagnostic. If the report shows `DECISION GAP SUPPRESSED`, the
K-invariance gate did not pass and no gap should be quoted from that run.

## 7. What to bring back

Copy the whole `results/session1/` directory off the host (or `git add` it into a
branch). It contains:

```text
session1_report.json        plan outcome, cost grid, gap report, invariance
cost_model.jsonl            the measured Cost(K, batch_size) grid
trace_*.jsonl               iteration-level traces
*/metadata.json             per-run provenance (commit, GPU, dirty flag)
*/requests.jsonl            per-request telemetry
*/aggregate.json            per-run summary
logs/server_*.log           server logs, for diagnosing failures
```

Then stop the pod. Analysis is offline and free.

## 8. Reading the result

### Step 0 — check `k_invariance` first, before anything else

The report's `k_invariance` field is the **gate on the entire analysis**, not a
diagnostic. Every oracle number evaluates `E[acc | K]` at depths a request may
never have run, which is only valid if per-position acceptance does not depend on
`K`.

```text
FAIL  ->  the gap is uninterpretable. No-go, regardless of its size.
PASS  ->  proceed to the gap.
NOT ASSESSED -> fewer than two static depths produced usable data; fix and rerun.
```

Checking the gap before this would be reading a number that may not mean
anything.

### Then the gap, against the pre-registered thresholds

Fixed in `docs/05-upper-bound-and-assumptions.md` §4 **before** seeing data:

| Real recoverable gap | Decision |
| --- | --- |
| `< 2%` | **No-go.** Publish the negative result. |
| `2–3%` | **Marginal.** Report honestly; pursue only if the mechanism is clean. |
| `3–5%` | **Too close to call on a proxy cost model.** Validate the cost model before claiming anything — see below. |
| `> 5%` | **Pursue**, after the Session 2 confirmation below. |

### The cost model is a proxy, and the wording matters

`Cost(K, n)` derived from client-side timing is an **effective serving cost proxy
at concurrency `n`**, not a measured model-step cost:

* `n` is *client concurrency*, and the real decode batch size decays within a wave
  as requests finish (32 → 27 → 23 → …).
* Wall time includes prefill, HTTP overhead, scheduler gaps, EOS variation and
  queue transitions — not just the target/draft decode step.

That is good enough for a go/no-go screen, which is all Session 1 is. It is **not**
good enough to defend a headline number in the 3–5% band. If the gap lands there,
measure server-side step timing (SGLang exposes serving metrics and profiling)
before claiming a result, rather than inferring cost from client throughput.

### Session 2 is a separate, cheaper decision

Session 1 screens on `mixed_50_50` — a *deliberately heterogeneous* synthetic
mixture. That is the right first target: if even a purpose-built mixture produces
`<2%`, the project is dead and nothing else needs measuring.

But a positive Session 1 result could be an artifact of that mixture. So:

| Session 1 result | Next step |
| --- | --- |
| `< 2%` | Stop. Write up the negative result. |
| `2–5%` | Validate the cost model, then rerun the affected captures. |
| `> 3–5%` | **Do not implement HeteroSpec yet.** Run a small Session 2 on `real_mixed` and `phase_shift` to confirm the phenomenon survives outside the synthetic mixture. |

Only after the phenomenon is confirmed on real traffic does implementing the
policy make sense. This keeps the expensive part of the project gated on evidence
rather than on momentum.

## 9. Cost control

* The session is one script; do not leave the pod idle between steps.
* `results/session1/session1_report.json` and `missing_cost_cells` tell you if any
  cell failed. Re-run **only** those with `--analyze-only` after a targeted rerun,
  rather than repeating the whole grid.
* Everything after the copy step is offline and free.

## 10. If something fails

| Symptom | Likely cause |
| --- | --- |
| Preflight exits 1 | HEAD is not the expected commit. It says which branch to check out, or pass `EXPECTED_SGLANG_SHA=<sha>` / `ALLOW_UNPINNED=1`. |
| `working tree : DIRTY` | Uncommitted edits in the SGLang checkout. Commit or stash, or accept that results will not be citable. |
| OOM at server start | 24 GB card, or `mem_fraction_static` too high for the card |
| Server not ready in 30 min | gated model without `HF_TOKEN`, or weights not pre-downloaded |
| `every warm-up request failed` | Server answers `/health` but cannot serve. Read `logs/server_*.log`; the step was skipped rather than timed. |
| `no usable calibration cells` | every calibration step failed; read `logs/server_*.log` |
| `no iteration trace found` | trace patch absent from the clone (wrong branch) |
| `K-invariance: NOT ASSESSED` | fewer than two static depths produced usable data |
| `no candidate depths within the common depth` | requests finished before any draft position was observed; raise `--max-new-tokens` |
| Adaptive capture much slower than static at the same concurrency | expected: the trace is enabled only for the adaptive capture, so its throughput is **not** comparable. It is a diagnostic capture, not the adaptive-throughput baseline. |
