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
heterospec/base                 frozen experiment base
heterospec/iter-telemetry       iteration-level trace facility (experimental)
heterospec/request-aware-policy HeteroSpec policy
```

Work branches are kept narrow so upstream PRs can be carved out cleanly.

## Local environment

The system Python (3.14) has no wheels for the GPU stack; the harness uses its
own 3.12 env.

```bash
cd heterospec
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/python -m pytest
```

The harness env is **CPU-only and independent of SGLang**. SGLang itself is only
installed where it is actually executed (a GPU host, or a separate local env for
the CPU feasibility probe in Phase 2). This keeps analysis reproducible without
a 3 GB dependency tree.
