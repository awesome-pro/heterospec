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
sglang_commit, sglang_dirty (bool)
gpu, cuda_version, torch_version
model, draft_model
speculation: algorithm, eagle_topk, num_steps, num_draft_tokens, adaptive (bool)
adaptive_config_path + adaptive_config (inline)
batch_size / concurrency, arrival process
generation lengths, temperature, seed
workload family + prompt class composition
harness_version
```

## Hard rules

1. **No run is valid without a clean tree.** If `git status --porcelain` is
   non-empty in the SGLang checkout, the run is tagged `dirty: true` and is not
   citable in the README results table.
2. **Never compare runs from different commits** unless the README says so
   explicitly and records both.
3. **Never report `P(A_k)` derived from a histogram collected under a varying
   K.** See README §4. Static-K captures only, or use the iteration trace.
4. Negative and control results are kept, not deleted. A null result is a
   deliverable (PROJECT.md "Definition of done" item 12).
