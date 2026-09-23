# SESSION 1 — the only guide you need

**Read nothing else.** This file goes from "the pod is deployed" to "the decision
is made". The five documents in `docs/` are reference material for when something
goes wrong or when you want to know *why*; you do not need them to run this.

Budget: **~75–120 minutes of GPU time, about $0.65–$1.05** on the A6000 you picked.

There are 8 phases. Each one ends with a **✅ check** — do not move on until it
passes. That is the entire discipline: catch problems while they are free.

---

## Phase 1 — Deploy the pod (before the meter starts)

Fix these settings, then click **Deploy Pod**. Getting this wrong is the most
expensive mistake available, because it wastes time *and* a re-deploy.

| Setting | Value | Why |
| --- | --- | --- |
| GPU | 1× RTX A6000 (48 GB) | The recommended card |
| **CUDA version** | **13.0** | The pinned SGLang builds from `cuda:13.0.3` |
| **Container disk** | **100 GB** | Default 30 GB cannot hold weights + wheels + results |
| Volume | **none — skip it** | See below |
| PyTorch | leave the template's version | The bootstrap installs `torch==2.13.0` itself |

### 1a. Do NOT add a volume

The console will push you toward a persistent volume ("Nothing mounted at the
template's path"). **Ignore that warning and deploy without a volume.**

The volume exists to avoid re-downloading the ~17.6 GB of weights. That download
takes 3–5 minutes, which at $0.53/hr is about **$0.03 of GPU time**. The volume
costs **$3.50–7.00 per month**, billed continuously while it exists — including
while no pod is running. Break-even is around 100 sessions in a month; this
project runs one or two.

The volume also has to live in the same datacenter as the pod, which is an extra
way to get stuck. Skipping it removes the problem rather than managing it.

Set the container disk to **100 GB** instead (~$0.013/hr — about 3 cents for the
whole session) and keep everything on the container. Container disk is erased when
the pod is terminated, which is exactly what you want here: copy the results off
(Phase 6) and let the rest go.

### 1b. Environment variables

You do **not** need to set `HF_HOME` in the console. Export it in the shell in
Phase 3; it is inherited by every server the session launches. Without a volume
the default cache location is fine.

> **The PyTorch version on the template does not matter.** If it says 2.8.0, the
> bootstrap's `pip install -e` replaces it with the pinned 2.13.0. That is another
> reason the container disk must be 100 GB. Phase 4 prints
> `ok torch 2.13.0 matches the pin` to confirm.

> **When a volume does pay off:** only if you run many sessions over a long period,
> or if RunPod's bandwidth from your chosen datacenter turns out to be slow. If
> Session 1 says "pursue" and you find yourself running five more sessions, revisit
> this — at 50 GB it is $3.50/month.
### 1c. Where these settings actually live: "Edit template overrides"

The deploy screen shows a summary, but the values are edited in the **Edit template
overrides** dialog. Open it and change exactly one thing:

| Field in the dialog | Action |
| --- | --- |
| **Container disk** | **31 → 100.** This is the one that must change |
| Container image | leave it (`...cu1281-torch280-ubuntu2404` is fine — see below) |
| SSH terminal access | **keep checked** — this is how you connect |
| Start Jupyter notebook | optional; unchecking frees a little RAM |
| Volume mount path (`/workspace`) | leave; irrelevant with no volume attached |
| Exposed ports 8888 / 22 | leave |
| Environment variables | leave empty — export `HF_HOME`/`HF_TOKEN` in the shell instead |

Then click **Set overrides** and confirm the summary reads **Total disk 100 GB**
before deploying.

> **The container image's CUDA version does not need to be 13.** This corrects an
> earlier over-strict claim of mine. I checked SGLang's dependency tree at the
> session commit, and the CUDA 13 runtime arrives through **pip**, not the base
> image: `torch==2.13.0`, `flashinfer_python[cu13]==0.6.18`,
> `humming-kernels[cu13]==0.1.12`, `sglang-kernel==0.4.7`, plus
> `nvidia-cudnn-cu13`, `nvidia-nccl-cu13`, `nvidia-nvshmem-cu13`. Torch links
> against those bundled pip libraries, not the system toolkit, so a cu128 base
> image is fine. What must support CUDA 13 is the **host driver**, and the deploy
> screen offering CUDA 13.0/13.2 means it does.
>
> The pinned lockfile at `configs/env/sglang-py312-linux.lock` records this full
> resolution. Phase 4 still prints the actual `torch` and CUDA versions it landed
> on, so this is verified rather than assumed.

**✅ Check:** the pod shows `Running` and "Total disk" reads **100 GB**. There is
deliberately **no volume**, so the "Nothing mounted at the template's path"
warning will still be visible — that is expected, ignore it. Once the pod is up,
`nvidia-smi` should report an RTX A6000 with ~48 GB.

---

## Phase 2 — Connect

In the RunPod console: **Connect → Start Web Terminal** (or SSH, if you have a key
configured). Everything below is typed in that terminal, on the pod.

**✅ Check:** you have a shell prompt that is *not* your Mac.

---

## Phase 3 — Pre-download the models (~10–15 min, mostly waiting)

The target model is **gated**: it will fail unless you have accepted the licence.

1. Go to <https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct> and accept it
   (do this in your browser, on your Mac — it is a one-time click).
2. Create a token at <https://huggingface.co/settings/tokens>.
3. On the pod:

```bash
export HF_HOME=/root/hf
export HF_TOKEN=hf_paste_your_token_here
```

`HF_HOME` is inherited by every server the session launches, so the weights are
read from one cache instead of being re-fetched per launch.

**✅ Check:** `echo $HF_TOKEN` prints your token.

---

## Phase 4 — Bootstrap (the big step, ~15–25 min)

One command does the whole setup: clones both repos at the exact pinned commits,
installs SGLang and the harness, checks the runtime matches, runs the fork's unit
tests, downloads the weights, and prints the plan.

```bash
mkdir -p /workspace && cd /workspace
git clone https://github.com/awesome-pro/heterospec.git
cd heterospec
bash scripts/gpu_bootstrap.sh
```

It takes a while — `pip install -e` for SGLang is the slow part. Do not interrupt
it. Watch for three lines near the end of the install section:

```text
  ok   torch 2.13.0 matches the pin
  ok   CUDA 13.x
  ok   both import cleanly
```

**✅ Check:** the script ends with `Setup complete` and prints the session command.
If it exits with **problems**, stop and send me the output — that is exactly what
it is for, and every check it makes would otherwise have failed *after* you
started paying.

> **Now go straight to Phase 5. Do not stop the pod to "save money" in between.**
> With no volume, stopping erases the container disk — the ~17.6 GB of weights, the
> SGLang install, and the clones — so stopping means paying for the whole bootstrap
> again. Phases 4 and 5 must run in one sitting. The session itself is only ~$1, so
> there is nothing to save by pausing.


> If it warns `torch ... != pinned 2.13.0` or `CUDA 12.x`, the image is wrong.
> Tell me and I'll tell you whether to let pip fix it or re-deploy.

---

## Phase 5 — Run the session (PAID — the only step that costs money)

The bootstrap printed this command. It looks like:

```bash
cd /workspace/heterospec
bash scripts/gpu_session_1.sh --sglang-path /workspace/sglang \
  --launch-config configs/models/llama31_8b_eagle3.json
```

It first runs a **preflight** that refuses to start unless SGLang is at the exact
expected commit. Then it runs **16 benchmark runs across 6 server launches**.

Expected final output:

```text
report:             results/session1/session1_report.json
K-invariance:       PASS / FAIL / NOT ASSESSED
recoverable rectangular gap over N captures: mean +X.XX% (min, max)
```

**✅ Check:** the command finished and the `report:` line printed a path.

**Leave the pod running** until Phase 6 is done.

---

## Phase 6 — Take the results off the pod, THEN stop it

The results are the only thing of value on that machine. Copy them first.

**From your Mac, in a new terminal** (not on the pod):

```bash
mkdir -p ~/Desktop/heterospec/results
scp -r <pod-ssh-host>:/workspace/heterospec/results/session1 \
    ~/Desktop/heterospec/results/
```

> **Use the "SSH over exposed TCP" form, not the plain SSH one.** RunPod lists two
> connections and they are not interchangeable: the proxy (`ssh.runpod.io`) is
> explicitly marked *"No support for SCP or SFTP"*, so `scp` over it fails. Use the
> direct TCP entry, which is marked *"Supports SCP & SFTP"*:

```bash
scp -P <port> -i ~/.ssh/id_ed25519 -r \
    root@<public-ip>:/workspace/heterospec/results/session1 \
    ~/Desktop/heterospec/results/
```

> Copy the exact host and port from **Connect → SSH over exposed TCP**. If you have
> no SSH key registered, copy the results from inside the pod instead — e.g. tar
> them and base64 them out through the web terminal, or push them to cloud storage
> from the pod.

**✅ Check:** on your Mac, this prints the files without error:

```bash
ls ~/Desktop/heterospec/results/session1/
```

You should see `session1_report.json`, `cost_model.jsonl`, some
`trace_*.jsonl`, and per-run directories.

**Now stop the pod** in the RunPod console. There is nothing else to preserve:
there is no volume, and the container disk is temporary by design.

> **Do this in the right order.** Copy the results off *first*, then stop. The
> overrides dialog states it plainly: *"Temporary storage that will be erased when
> the Pod is stopped."* Stopping before the copy destroys the only copy of the run.

---

## Phase 7 — Analyse on the Mac (free, no GPU)

```bash
cd /Users/abhinandan/Desktop/heterospec/heterospec
.venv/bin/python -m heterospec.session --analyze-only \
  --launch-config configs/models/llama31_8b_eagle3.json \
  --results-root ~/Desktop/heterospec/results/session1
```

This costs nothing and works on the copied directory.

**✅ Check:** it prints `K-invariance: PASS` (or `FAIL`/`NOT ASSESSED`).

---

## Phase 8 — Read the decision

### Step 0 — the gate, first, before anything else

| `K-invariance` | Meaning | Action |
| --- | --- | --- |
| **FAIL** | The oracle's core assumption is broken | **No-go, whatever the gap says.** Send it to me |
| **NOT ASSESSED** | Fewer than two usable static depths | Tell me — something failed in capture |
| **PASS** | Proceed | Read the gap below |

If the report says `DECISION GAP SUPPRESSED`, the gate did its job. That is not a
crash — it means no gap should be quoted.

### Then the gap, against the thresholds fixed *before* the run

| Measured gap | Decision |
| --- | --- |
| **< 2%** | **No-go.** The scalar policy is good enough |
| **2–3%** | Marginal — needs a judged call |
| **3–5%** | Validate the cost model against server-side numbers first |
| **> 5%** | **Pursue** — confirm with Session 2 |

---

## What to send me

**One file:** `session1_report.json` from Phase 6.

I will read the invariance verdict, the primary-K gap, the per-capture breakdown
and the warnings, and tell you which branch of the rule you landed in — including
whether anything in the `excluded` / `diagnostic` sections changes the reading.

---

## Phase 9 (later) — the SGLang contribution

You chose to open the upstream PR **after** Session 1. The bootstrap already ran
that verification for you in Phase 4 — look for these lines in its output:

```text
  ok   test_adaptive_runtime_state.py
  ok   test_adaptive_spec_params.py
  ok   test_batch_result_processor_spec_grammar.py
```

Those are the three PR 1 files. If all three passed, send me that part of the
output and I will prepare the pull request. If any failed, send me the failure —
PR 1 must not be opened until they pass.

---

## If something goes wrong

| Symptom | What it means | Do this |
| --- | --- | --- |
| `ERROR: sglang HEAD ... is not the expected commit` | Wrong branch checked out | `git -C /workspace/sglang checkout heterospec/iter-telemetry` |
| Bootstrap says `working tree is DIRTY` | Uncommitted edits in the fork | Results would not be citable — tell me |
| `python ... outside requires-python` | Image has Python 3.10 or 3.13 | Re-deploy with a 3.12 image |
| Download fails on the Llama model | Licence not accepted or no token | Redo Phase 3 |
| OOM during server launch | Disk/GPU too small, or wrong config | Check 48 GB GPU and `mem_fraction_static` |
| `no session report at ...` | Session did not finish | Send me the pod's terminal output |
| Bootstrap exits with "problems" | A preflight check failed | **Send me the output** — do not push on |

**Do not** debug by changing the harness. It is frozen, it has 420 passing tests,
and any edit invalidates the run's citability. If something fails, the correct move
is always: stop, send me the output, wait.

---

## The short version

```bash
# 1. Deploy: A6000 48GB, CUDA 13.0, 100GB container disk, NO volume
# 2. Connect to the pod terminal

# 3. On the pod:
export HF_HOME=/root/hf
export HF_TOKEN=hf_...              # after accepting the Llama-3.1 licence

# 4. On the pod:
mkdir -p /workspace && cd /workspace
git clone https://github.com/awesome-pro/heterospec.git
cd heterospec && bash scripts/gpu_bootstrap.sh

# 5. On the pod (PAID — the only paid step):
bash scripts/gpu_session_1.sh --sglang-path /workspace/sglang \
  --launch-config configs/models/llama31_8b_eagle3.json

# 6. On your Mac — copy results off, then STOP THE POD:
scp -P <port> -r root@<host>:/workspace/heterospec/results/session1 \
    ~/Desktop/heterospec/results/

# 7. On your Mac (free):
cd /Users/abhinandan/Desktop/heterospec/heterospec
.venv/bin/python -m heterospec.session --analyze-only \
  --launch-config configs/models/llama31_8b_eagle3.json \
  --results-root ~/Desktop/heterospec/results/session1

# 8. Send me session1_report.json
```
