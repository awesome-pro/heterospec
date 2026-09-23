# 02 — Reproducibility and pinning

SGLang upstream moves roughly every 30 minutes. Anything we measure without a
pinned revision is worthless. This document defines the rules.

## The pinned base

`configs/sglang_base.json` is the machine-readable pin. The Python package also
carries it (`heterospec.SGLANG_BASE_COMMIT`) so the harness can assert it at
runtime.

```text
base_commit  66ce8c55cc6c656225d33f26bbcaeeac8ba92e93
date         2026-09-23T13:56:24+08:00
subject      [HiCache] ci: add HiCache and unified radix rerun group (#40831)
fork branch  heterospec/base
```

## Remotes

```text
origin    git@github.com:awesome-pro/sglang.git          (our fork, push here)
upstream  https://github.com/sgl-project/sglang.git      (read-only, fetch here)
```

## Rules

1. **One frozen commit per experiment campaign.** A GPU session runs against one
   commit. Branch `heterospec/base` marks it. Never re-sync mid-session.
2. **Re-sync deliberately, between sessions only**, and re-run the affected
   baselines if the speculative/managers paths changed. Check with:

   ```bash
   git log --oneline <old>..<new> -- python/sglang/srt/speculative/ \
       python/sglang/srt/managers/scheduler_components/
   ```

3. **No dirty runs are citable.** The harness records
   `git status --porcelain` output; if non-empty, `metadata.json` gets
   `"dirty": true` and the run is excluded from README results tables. During
   policy development we *will* have a dirty tree — those runs are labelled
   experimental, not results.
4. **Every results directory records the commit** (see `results/README.md`).
5. **Never compare across commits** without saying so explicitly and recording
   both.

## Branch naming in the fork

```text
heterospec/base                     frozen experiment base (== pinned upstream commit)
heterospec/iter-telemetry           iteration-level trace facility (research)
heterospec/policy-feedback-identity PR 1: request identity on the policy feedback path
heterospec/request-aware-policy     HeteroSpec policy (only if the gap justifies it)
```

Work branches are kept narrow so upstream PRs can be carved out cleanly.

### Why the two work branches are separate, and what that means for tests

`heterospec/iter-telemetry` and `heterospec/policy-feedback-identity` are
**independent branches off `heterospec/base`**, not a stack. That is deliberate:

* PR 1 is intended for upstream and must be a clean, minimal diff. Bundling a
  research trace facility into it would make it much harder to review.
* The trace patch is explicitly **not** proposed upstream as-is (see
  `docs/01-adaptive-stack.md` §6 and the README).

The cost is that only one can be checked out at a time, so branch-sensitive tests
skip when their patch is absent:

| Test file | Needs |
| --- | --- |
| `tests/test_trace_patch.py` | `heterospec/iter-telemetry` checked out |
| `tests/test_pr1_policy_identity.py` | `heterospec/policy-feedback-identity` checked out |

Both skip with an actionable message rather than failing, and both assert against
the fork's working tree. To run everything:

```bash
git -C ../sglang checkout heterospec/iter-telemetry
.venv/bin/python -m pytest -q          # trace-patch tests run, PR 1 tests skip

git -C ../sglang checkout heterospec/policy-feedback-identity
.venv/bin/python -m pytest -q          # PR 1 tests run, trace-patch tests skip
```

An alternative is a second `git worktree` of the fork, but the harness reads a
single sibling path, so the branch switch is simpler.

## Local environment

The system Python (3.14) has no wheels for the GPU stack; the harness uses its
own 3.12 env.

```bash
cd heterospec
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/python -m pytest
```

### uv locally, pip on the GPU host

The split is deliberate:

* **Local analysis env: `uv`.** Nothing about the analysis needs SGLang, so the
  fast resolver is free to use.
* **SGLang on the GPU host: `pip`.** SGLang's own `docker/Dockerfile` installs
  everything with `python3 -m pip install`, and that is the path its maintainers
  test. `uv` *can* resolve the tree — it does all 204 packages in about six
  seconds and picks the same critical pins (`torch==2.13.0`,
  `flashinfer-python==0.6.18`, `cuda-tile==1.6.0rc5`) — but the install is a small
  part of a ~$1 session, so trading a tested installer for an untested one buys
  cents and risks the one run that has to be defensible.

What `uv` is genuinely useful for here is **freezing** the environment, which pip
does not do on its own:
[`configs/env/sglang-py312-linux.lock`](../configs/env/sglang-py312-linux.lock)
records the full resolution for the host platform, with instructions for diffing
it against `pip freeze` on the host. SGLang ships no lockfile, so without that
record nothing would prove Session 2 ran the same environment as Session 1.

The harness env is **CPU-only and independent of SGLang**. SGLang itself is only
installed where it is actually executed (a GPU host, or a separate local env for
the CPU feasibility probe in Phase 2). This keeps analysis reproducible without
a 3 GB dependency tree.
