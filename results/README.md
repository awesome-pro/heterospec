# results/

Structured experiment output. **Payloads are gitignored** — this directory
tracks only this README. What matters for reproducibility is the *method* and
the *metadata*, which live in `configs/` and `docs/`.

Each serious run produces one dated directory:

```text
results/
  <YYYY-MM-DD>_<workload>_<policy>_<shortsha>/
    metadata.json           # everything needed to reproduce/interpret
    requests.jsonl          # one line per completed request
    speculative_steps.jsonl # iteration-level records (requires trace patch)
    aggregate.json          # run-level summary metrics
    cost_model.json         # Cost(K, batch_size) fit (calibration runs only)
```

`metadata.json` must record at minimum:

```text
base_sha, experiment_patch_sha, working_tree_dirty, sglang_branch
gpu, cuda_version, torch_version
model, draft_model
speculation: algorithm, eagle_topk, num_steps, num_draft_tokens, adaptive (bool)
adaptive_config_path + adaptive_config (inline)
batch_size / concurrency, arrival process
generation lengths, temperature, seed
workload family + prompt class composition
harness_version
```

## Provenance is three separate facts

A run is reproducible when **which code produced it** is recorded exactly and
nothing was uncommitted. That is three statements, and conflating them is how a
reproducibility story goes wrong:

| Field | Meaning |
| --- | --- |
| `base_sha` | the pinned upstream commit this project baselines on |
| `experiment_patch_sha` | the actual SGLang HEAD during the run |
| `working_tree_dirty` | whether uncommitted edits were present |

A **clean checkout of a known research patch is citable**: `base_sha +
experiment_patch_sha` names it precisely. Session 1 runs the
`heterospec/iter-telemetry` branch, which is deliberately *not* the base, and
those runs are legitimate results.

An earlier version of this rule treated "HEAD != pinned base" as non-citable,
which would have marked every Session 1 result unusable — by the project's own
tooling. Only uncommitted changes are disqualifying.

## Session 1 (2026-09-23) — where the payload lives

The first real-GPU session ran 16 captures (K in {1,3,5,7} x concurrency in
{1,8,32}, plus no-spec at c32) on a RunPod RTX A6000. The **analysis** is in
`docs/07-session1-results.md`; the payload is deliberately not committed here.

```text
tree        results/session1/  (21 MB, 56 files)
manifest    750ec6ae90c43427da2c1590b8ca1324  (md5 of the sorted file hashes)
contents    16 run dirs, session1_report.json, logs/server_*.log,
            trace_sglang_adaptive.jsonl (15 MB, 31,403 records)
```

It is reproduced from the provenance block in `docs/07`: the pinned base, the
experiment patch SHA, the launch config and the workload definition are all that
is needed. If a claim in `docs/07` ever needs the raw rows, treat this tree as the
source of truth and verify it against the manifest hash before use.

## Hard rules

1. **No run with `working_tree_dirty: true` is citable.** The exact code cannot be
   reconstructed. `RunMetadata.citable()` enforces this and records the reason.
2. **Never compare runs from different `experiment_patch_sha` values** unless the
   README says so explicitly and records both.
3. **Never report `P(A_k)` derived from a histogram collected under a varying
   K.** See README §4. Static-K captures only, or use the iteration trace.
4. **No run from a mock server is citable.** `citable()` refuses it outright.
5. Negative and control results are kept, not deleted. A null result is a
   deliverable (PROJECT.md "Definition of done" item 12).
6. **Every calibration server runs with `--disable-radix-cache`.** Prefix caching
   is not a variable under study, and the grid reuses the same seeded prompt pool
   across concurrencies on one server, so a warm cache would make later runs look
   cheaper and contaminate the batch-size axis of the cost model.
