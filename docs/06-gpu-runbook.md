# 06 — GPU runbook: Session 1

The only work that must happen on rented hardware, written so nothing on the GPU
side needs authoring or debugging. Everything it depends on is built and tested on
the Mac (346 tests, no GPU).

Target: **under an hour of GPU time.** Roughly 16 benchmark runs across 6 server
launches.

---

## 1. Which GPU

**Recommendation: a 48 GB card — L40S or A6000.** On a tighter budget, A100 40 GB
works with the 40 GB config variant. Avoid 24 GB.

Why 24 GB fails, concretely:

```text
Llama-3.1-8B in fp16                    ~16.1 GB
EAGLE3 draft model                       ~1.5 GB
static weights total                    ~17.6 GB

mem_fraction_static 0.7 on a 24 GB card = 16.8 GB  <  17.6 GB
```

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
* **Template**: any PyTorch image with CUDA 12.x. Verify in the console — image
  tags change, and a wrong tag wastes the first 20 minutes.
* **Container disk** ≥ 40 GB for the environment.
* Set `HF_HOME=/workspace/hf` as an environment variable so weights land on the
  volume.

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

# PR 1: request identity on the policy feedback path
python -m pytest -q test/registered/unit/spec/test_adaptive_runtime_state.py
python -m pytest -q test/registered/unit/spec/test_adaptive_spec_params.py
python -m pytest -q test/registered/unit/managers/test_batch_result_processor_spec_grammar.py

# the trace patch (only present on heterospec/iter-telemetry)
python -m pytest -q test/registered/unit/spec/test_heterospec_trace.py
```

All four are registered in CPU CI, so they need no GPU. If any fails, report the
failure before running the session — the session's results are unaffected, but
PR 1 must not be opened until its tests pass.

Note that `heterospec/iter-telemetry` and `heterospec/policy-feedback-identity`
are separate branches, so the two groups of tests need different checkouts:

```bash
git checkout heterospec/policy-feedback-identity   # PR 1 tests
git checkout heterospec/iter-telemetry             # trace patch tests + the session
```

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
ALLOW_OTHER_COMMIT=1 bash scripts/gpu_session_1.sh \
    --sglang-path /workspace/sglang \
    --launch-config configs/models/llama31_8b_eagle3.json
```

`ALLOW_OTHER_COMMIT=1` is expected: the telemetry branch is deliberately not the
pinned base commit. Runs are tagged `dirty: true` / off-base and are **not
citable as results** — they are the measurement that decides whether to continue.

The script prints progress and, at the end:

```text
report:             results/session1/session1_report.json
K-invariance:       PASS / FAIL / NOT ASSESSED
recoverable rectangular gap over N captures: mean +X.XX% (min, max)
```

Use `--analyze-only` to re-analyse without re-running anything.

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

Pre-registered in `docs/05-upper-bound-and-assumptions.md` §4, fixed before seeing
data:

| Real recoverable gap | Decision |
| --- | --- |
| `< 2%` | **No-go.** Publish the negative result. |
| `2–3%` | **Marginal.** Report honestly; pursue only if the mechanism is clean. |
| `> 3–5%` | **Pursue.** Implement HeteroSpec as an `AdaptiveSpecPolicy`. |

Check `k_invariance` **first**. If it failed, the gap number is not interpretable
at all and the answer is no-go regardless of its size.

## 9. Cost control

* The session is one script; do not leave the pod idle between steps.
* `results/session1/session1_report.json` and `missing_cost_cells` tell you if any
  cell failed. Re-run **only** those with `--analyze-only` after a targeted rerun,
  rather than repeating the whole grid.
* Everything after the copy step is offline and free.

## 10. If something fails

| Symptom | Likely cause |
| --- | --- |
| OOM at server start | 24 GB card, or `mem_fraction_static` too high for the card |
| Server not ready in 30 min | gated model without `HF_TOKEN`, or weights not pre-downloaded |
| `no usable calibration cells` | every calibration step failed; read `logs/server_*.log` |
| `no iteration trace found` | trace patch absent from the clone (wrong branch) |
| `K-invariance: NOT ASSESSED` | fewer than two static depths produced usable data |
| `no candidate depths within the common depth` | requests finished before any draft position was observed; raise `--max-new-tokens` |
