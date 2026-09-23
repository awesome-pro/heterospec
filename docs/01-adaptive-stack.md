# 01 — How SGLang's adaptive speculative decoding actually works

Phase 1 deliverable (PROJECT.md). Audited at SGLang commit
`66ce8c55cc6c656225d33f26bbcaeeac8ba92e93` (2026-09-23). All `file:line` refs are
to that commit. Adaptive speculative decoding is **already merged upstream**; we
read current `main`, not PR branches, because the behaviour has drifted from the
original PR descriptions.

---

## The short answer

> **At what point does SGLang observe accepted tokens, where is that
> aggregated, how does it choose a new tier, and what actually changes when K
> changes?**

Accepted tokens are observed **on the CPU, in the scheduler process**, in
`batch_result_processor._resolve_spec_v2_tokens()` — deliberately *not* in the
worker, to avoid a synchronous GPU→CPU copy in the hot path. The scheduler turns
`accept_lens` into a per-request accepted-draft count, **subtracting the bonus
token**, then immediately (a) accumulates per-request aggregates onto each `Req`,
and (b) hands the *whole per-request list* to the worker's
`on_verify_complete_cpu`. The adaptive policy then **throws away the per-request
structure**: it takes the plain mean of that list, blends it into one EMA per
batch-size slot, and only every `update_interval` batches — after a warmup and
through hysteresis gates — moves one step up or down the slot's ladder. The new
step is applied by an atomic swap of a prebuilt `SpecRuntimeState`: attention
backends, CUDA graph runners, and draft-chain buffers are *reference-swapped*,
never recaptured. So the only thing that changes is the shape/config bundle;
the execution graph is untouched and intact.

Everything interesting for HeteroSpec is in that one sentence: **the
per-request list exists and is complete at the moment the policy is called, and
the policy reduces it to a scalar mean.**

---

## 1. Where accepted tokens are observed

`python/sglang/srt/managers/scheduler_components/batch_result_processor.py`

```python
# _resolve_spec_v2_tokens(), ~line 745
assert result.next_token_ids.is_cpu
assert result.accept_lens.is_cpu

next_token_ids = result.next_token_ids.tolist()
accept_lens = result.accept_lens.tolist()
stride = _get_speculative_output_stride(result)
num_non_draft = result.num_non_draft_tokens_per_req  # 757

result.num_correct_drafts_per_req_cpu = [  # 758
    length - num_non_draft for length in accept_lens
]
result.num_correct_drafts = sum(result.num_correct_drafts_per_req_cpu)
```

Key points:

- `accept_lens` is already on CPU (asserted) — the D2H copy happened upstream of
  this function.
- `num_non_draft_tokens_per_req` defaults to `1` (`managers/utils.py:84`). So
  **`num_correct_drafts` excludes the bonus token**: accepted *drafts* only.
  With K steps, `accept_lens ∈ [1, K+1]` and `num_correct_drafts ∈ [0, K]`.
- The comment at lines 774–776 states the design intent explicitly:

  > Feed the adaptive controller now that accept_lens is on CPU, instead of
  > doing a synchronous GPU→CPU copy in the worker hot path.

Immediately after, the feedback call:

```python
self.model_worker.on_verify_complete_cpu(  # 777
    result.num_correct_drafts_per_req_cpu, batch_size=len(batch.reqs)
)
```

**`batch.reqs` is in scope one line earlier.** This is the entire reason the
HeteroSpec policy interface change is small.

## 2. Where it is aggregated

Two independent aggregations happen, at different granularities.

### Per-request (cumulative, on `Req`)

```python
# same function, ~805
req.spec_verify_ct += 1
num_correct_drafts = result.num_correct_drafts_per_req_cpu[i]  # 807
req.spec_num_correct_drafts += num_correct_drafts  # 808
req.update_spec_correct_drafts_histogram(num_correct_drafts)  # 809
```

`update_spec_correct_drafts_histogram` (`managers/schedule_batch.py:1455`) grows a
list on demand and increments bucket `num_correct_drafts`:

```python
if len(self.spec_correct_drafts_histogram) <= num_correct_drafts:
    self.spec_correct_drafts_histogram.extend(
        [0] * (num_correct_drafts - len(self.spec_correct_drafts_histogram) + 1)
    )
self.spec_correct_drafts_histogram[num_correct_drafts] += 1
```

So a histogram `[2, 4, 8, 5, 1]` means: 2 rounds accepted 0 drafts, 4 accepted 1,
8 accepted 2, 5 accepted 3, 1 accepted 4.

Because `accept_len >= k` iff draft position `k` survived verification:

```text
P(draft position k accepted) = sum(histogram[k:]) / sum(histogram)
```

**This formula depends on acceptance being prefix-contiguous** — accepted tokens
form a prefix, so "position `k` accepted" ⟺ "accept_len ≥ k". That holds for
topk=1 chain drafting, which adaptive mode already requires (§9). It would *not*
hold for a tree/topk>1 verify, where a later branch token can survive while an
earlier one does not. Do not reuse this derivation outside topk=1.

`sum(histogram) == spec_verify_ct` should hold — a useful harness
self-consistency assertion. Both counters are incremented in the same branch
that excludes retracted/finished requests.

### Batch-level (in the policy) — this is where request structure is lost

`python/sglang/srt/speculative/adaptive_spec_params.py:176`

```python
def update(self, num_correct_drafts_per_req: list[int]) -> bool:
    if not num_correct_drafts_per_req:
        return False

    if self.current_steps > 0:
        batch_avg = sum(num_correct_drafts_per_req) / len(num_correct_drafts_per_req)
        self.ema_accept_len = (
            1 - self.ema_alpha
        ) * self.ema_accept_len + self.ema_alpha * batch_avg
    ...
```

The per-request list arrives intact and is **immediately reduced to
`batch_avg`**. Note also `if self.current_steps > 0`: at the step-0 tier the EMA
is frozen.

## 3. How a new tier is chosen

**Routing.** `AdaptiveSpeculativeParams._route(batch_size)` = pad the batch size
*up* to the nearest captured CUDA-graph batch size, then take the closest slot
key `<=` that value:

```python
def _pad_to_cuda_graph_bs(self, batch_size: int) -> int:  # 337
    idx = bisect.bisect_left(self._cuda_graph_bs, batch_size)
    return self._cuda_graph_bs[idx] if idx < len(...) else batch_size


def _find_closest_bs(self, target: int) -> int:  # 345
    idx = bisect.bisect_right(self._bs_list, target) - 1
    return self._bs_list[max(0, idx)]
```

**Gates.** `AdaptiveStepSlot.update` applies, in order:

1. `warmup_batches` (default 10) — no switching at all while
   `_batch_count <= warmup_batches`.
2. `update_interval` (default 5) — recompute only when
   `(_batch_count - warmup_batches) % update_interval == 0`.

**Recompute.** `_recompute_params()`:

```text
move DOWN while ema <= prev_step - 0.5 + down_hysteresis   (0.5 if prev_step == 0)
if not moved down:
    move UP while ema > current_step - 0.5 + up_hysteresis
optional ceiling_coeff: caps downward only (never blocks step-ups)
```

Special case at the step-0 tier: probe the next positive step and reset the EMA
to that step's neutral value `target - 1`, because a slot initialised at 0 has no
acceptance history.

Conceptually `target_steps ≈ clamp(round(ema_accept_len) + 1, ...)` — the policy
**probes one step beyond observed acceptance**.

Hysteresis defaults differ by ladder: BS≥1 uses `down_hysteresis = -0.25`, while
BS≥8/32/64 use `0.0`. The current default ladders include a step-0 tier, so
SGLang can **disable speculation entirely** at high batch size:

```text
BS >= 1  -> [1, 3, 5, 7]   up 0.0   down -0.25  ceiling 0
BS >= 8  -> [0, 1, 3]      up 0.0   down  0.0   ceiling 0
BS >= 32 -> [0, 1]         up 0.0   down  0.0   ceiling 0
BS >= 64 -> [0]            up 0.0   down  0.0   ceiling 0
```

**Two entry points, called from different places:**

| When | Call | Batch size used | Code |
| --- | --- | --- | --- |
| Pre-draft (per decode step) | `activate_step_by_batch(bs)` | `batch.seq_lens.shape[0]` | `eagle_worker_v2.py:1450` |
| Post-verify | `on_verify_complete_cpu(counts, bs)` | `len(batch.reqs)` | `batch_result_processor.py:777` |

> **Open question to verify empirically.** These two batch-size definitions can
> differ (padded `seq_lens` shape vs actual request count). If they do, the
> pre-draft lookup and the post-verify update can route to *different* BS slots,
> so the policy would be updated on one slot and read on another. Worth
> instrumenting before we build anything on top. Not confirmed as a bug — noted
> as a measurement.

## 4. What actually changes when K changes

`SpecRuntimeState` (`speculative/adaptive_runtime_state.py:10`) is the unit of
change — a complete, prebuilt resource bundle per tier:

```text
speculative_num_steps, speculative_num_draft_tokens      # config / shapes
draft_attn_backend,               cuda_graph_runner
target_attn_backend,              target_graph_runner
draft_extend_attn_backend,        cuda_graph_runner_for_draft_extend
```

Because `CudaGraphRunner` is shape-dependent, **each tier owns its own graphs and
backend state**. They are built once at startup, never on the hot path:

```text
EagleWorkerV2.__init__          -> AdaptiveController(worker, AdaptiveSpeculativeParams(...))
EagleWorkerV2.init_cuda_graphs  -> controller.register(current SpecRuntimeState)   # 1378
                                -> controller.init_states(cuda_graph_bs)
                                     for steps in candidate_steps:
                                         pruned = params.cuda_graph_bs_for_step(steps)
                                         state  = worker.build_adaptive_runtime_state(
                                                      steps, steps + 1, pruned)
                                -> controller._activate(worker.speculative_num_steps)
```

Note `speculative_num_draft_tokens = steps + 1` — EAGLE topk=1 chain drafting
proposes one extra token beyond the step count.

Switching is `EagleWorkerV2.apply_runtime_state` (`eagle_worker_v2.py:1737`), and
it is an **early-returning reference swap**:

```python
if self.speculative_num_steps == state.speculative_num_steps:
    return
```

then it reassigns, on the draft side: `speculative_num_steps`,
`speculative_num_draft_tokens`, `draft_attn_backend`,
`draft_runner.draft_attn_backend`, `cuda_graph_runner`,
`draft_extend_attn_backend`, `cuda_graph_runner_for_draft_extend`, and calls
`dw._rebuild_topk1_chain_buffers()`; on the target side:
`model_runner.attn_backend` and `model_runner.decode_cuda_graph_runner`; finally
it syncs `server_args` through
`get_context().override("adaptive_spec.restore", ...)`.

**No kernel recompilation, no graph recapture, no KV reshape.** Switches happen
only *after* a round completes — backends and graphs are never swapped mid-round.

## 5. The policy interface (our extension point)

`adaptive_runtime_state.py:52` — this is what HeteroSpec must implement:

```python
class AdaptiveSpecPolicy(Protocol):
    @property
    def candidate_steps(self) -> list[int]: ...
    def set_cuda_graph_bs(self, cuda_graph_bs: list[int] | None) -> None: ...
    def get_steps_for_batch(self, batch_size: int) -> int: ...
    def on_verify_complete(
        self, num_correct_drafts_per_req: list[int], batch_size: int
    ) -> int | None: ...
    def cuda_graph_bs_for_step(self, step: int) -> list[int] | None: ...
```

**Caveat: the extension point is code-level only.** The policy is hardcoded at
exactly one site (`eagle_worker_v2.py:1328`):

```python
if get_spec().speculative_adaptive and self._hosts_draft:
    self.adaptive_controller = AdaptiveController(
        self,
        AdaptiveSpeculativeParams(
            initial_steps=self.speculative_num_steps,
            cfg_path=get_spec().speculative_adaptive_config,
        ),
    )
```

There is no registry, no config-selected dotted path. PR #37274 made
`AdaptiveController` *accept* any `AdaptiveSpecPolicy`; selecting one still means
editing this line. (Also present: `standalone_worker_v2.py:201`.)

## 6. Request identity: present on the finish path, absent on the feedback path

This is the sharpest finding in the audit, and it substantially de-risks the
upstream contribution.

`BaseSpecWorker` (`speculative/base_spec_worker.py:346`) already defines a
**rid-carrying** lifecycle hook:

```python
def note_request_finished(self, *, rid: str, natural_stop: bool) -> None:
    """Hook called by the batch-result processor when a request finishes.

    Default no-op. DSpark overrides this to settle / censor its
    block-accept estimator state for the finished request.
    """
```

and it **is already called**, from the same file as our target call site:

```python
# batch_result_processor.py:1313
if req.finished():
    if isinstance(self.draft_worker, BaseSpecWorker):
        self.draft_worker.note_request_finished(
            rid=req.rid,
            natural_stop=isinstance(req.finished_reason, FINISH_MATCHED_TOKEN),
        )
```

Meanwhile the *feedback* path delivers per-request counts with no identity:

```python
on_verify_complete_cpu(num_correct_drafts_per_req: list[int], batch_size: int = 0)
```

So: **request-local state, keyed by `rid`, with correct cleanup on request
completion, is already an accepted pattern in this codebase — the verify
feedback path is simply the one per-request signal that arrives anonymous.**

### DSpark is the precedent — and it bounds our novelty claim

`speculative/dspark_components/dspark_block_accept_estimator.py` is a
fully-developed instance of the pattern:

- rid-keyed per-request state: `self._states.pop(rid, None)`
- lifecycle cleanup via `note_request_finished` (`:352`)
- state expiry/sweep: `_STATE_SWEEP_INTERVAL = 1024`, `_STATE_EXPIRE_STEPS = 4096`
- online windowed estimate: `_DEFAULT_ONLINE_WINDOW_STEPS = 256`
- **env-gated, bounded, JSONL iteration dumps**:
  `SGLANG_DSPARK_DEBUG_DUMP` (`environ.py:536`), `INFO_DUMP_MAX_RECORDS = 200_000`,
  `INFO_DUMP_MAX_STEP_CPU_SECONDS = 1.0`, `_FLUSH_EVERY_STEPS = 16`, with
  per-step records carrying `rids` (`dspark_observability.py:255`).

Two consequences:

1. **Imitation beats invention.** The eventual trace facility should follow this
   dump convention (env-gated component tokens, bounded record count, flush
   interval, CPU-time guard, fast disable) rather than defining a new format.
   This is the single highest-leverage choice for upstream acceptance.
2. **Scope our claim precisely.** DSpark estimates per-request *block* acceptance
   for DSpark's own planner. It does not feed a batch-wide K decision for EAGLE.
   `adaptive_unsupported_reason` confirms adaptive mode is EAGLE/EAGLE3-only, so
   the paths do not overlap. Our novelty is narrow and should stay narrow:
   request-local acceptance feeding **batch-wide single-K selection in the EAGLE
   adaptive path**, which today uses only a batch mean. DSpark belongs in
   *related work in the same tree*, cited, not ignored.

## 7. Telemetry: what already reaches the client

Exposed in response `meta_info` with **no source modification**
(`managers/tokenizer_manager.py:2944–2992`; OpenAI mapping in
`entrypoints/openai/protocol.py:430` and `utils.py:321–338`):

```text
spec_verify_ct                  verify rounds for this request
spec_num_correct_drafts         cumulative accepted drafts (bonus excluded)
spec_correct_drafts_histogram   accepted-drafts-per-round histogram
```

Server-side monitoring (`scheduler_components/metrics_reporter.py:898–940`):
`spec_accept_length = spec_num_accept_tokens / spec_num_forward_ct`,
`spec_accept_rate`, plus `avg_spec_accept_length` and the active
`speculative_num_steps` on `/server_info`.

### The bias that constrains Phase 4

`P(A_k) = sum(histogram[k:]) / sum(histogram)` is **only unbiased when K is
constant** across the request's rounds. Under adaptive decoding K varies, and a
round with active K=2 can never yield `accept >= 5`; the histogram records those
as bucket ≤2, so position 5 is scored "rejected" although it was **never
proposed**. `spec_correct_drafts_histogram` does not record K per round, so the
bias **cannot be corrected post-hoc**.

Therefore the first per-request study runs against **static-K** servers (free —
a launch flag), and the iteration-level trace is required for the adaptive case
to be *correct*, not merely richer.

## 8. What is missing, precisely

| # | Gap | Where | Size |
| --- | --- | --- | --- |
| G1 | Policy receives per-request counts with no identity → cannot keep request-local state across iterations | `AdaptiveSpecPolicy.on_verify_complete` | small |
| G2 | Per-request structure is reduced to `batch_avg` before the policy sees it | `adaptive_spec_params.py:186` | by design (default policy) |
| G3 | No iteration-level record joining active K ↔ per-request accepted ↔ batch composition | absent | small, opt-in |
| G4 | No cost model `Cost(K, batch_size)` anywhere in tree | absent | GPU calibration |
| G5 | `P(A_k)` from histograms is K-confounded | inherent to the histogram | mitigated by G3 |

Not gaps, contrary to PROJECT.md's original assumption: per-request acceptance
aggregation (exists), policy pluggability (exists, #37274), request-finish
lifecycle hook with `rid` (exists), and an in-repo convention for bounded opt-in
tracing (exists, DSpark).

## 9. Unsupported configurations

`adaptive_unsupported_reason` (`adaptive_spec_params.py:54`) forces
`speculative_adaptive=False` (logged at `speculative_hook.py:1222`) when any of:

| Condition | Why |
| --- | --- |
| algorithm not EAGLE/EAGLE3 | no adaptive support elsewhere |
| `speculative_eagle_topk != 1` | only topk=1 chain |
| `enable_dp_attention` | tier decisions are not synchronised across DP ranks |
| `enable_multi_layer_eagle` | `MultiLayerEagleWorkerV2` does not implement adaptive |
| `enable_two_batch_overlap` | state swap would discard the `TboAttnBackend` wrapper |
| `enable_pdmux` | state swap does not update `decode_attn_backend_group` |

**Our experiment launches must avoid all six**, or adaptive mode silently
disables and we would be measuring static speculation.

## 10. Load-bearing assumptions to test, not assume

1. **K-invariance.** All oracle analysis assumes `P(draft position k accepted)`
   does not depend on K — i.e. the k-th draft token is generated from the same
   prefix regardless of total steps. Approximately true for EAGLE topk=1 chain
   drafting, but it is the foundation of the oracle. **Test:** same prompts at
   static `K ∈ {1,3,5,7}`, compare per-position curves on overlapping positions.
   Rides along with the cost-calibration grid, so it is nearly free.
2. **BS routing consistency.** Pre-draft and post-verify may use different
   batch-size definitions (§3). Verify before building on the routing.
3. **Histogram/verify-count consistency.** `sum(histogram) == spec_verify_ct`.
   Assert this in the harness as a free integrity check.

## 11. Refs

Audited code:

```text
python/sglang/srt/speculative/adaptive_runtime_state.py
python/sglang/srt/speculative/adaptive_spec_params.py
python/sglang/srt/speculative/eagle_worker_v2.py
python/sglang/srt/speculative/base_spec_worker.py
python/sglang/srt/managers/scheduler_components/batch_result_processor.py
python/sglang/srt/managers/scheduler_components/metrics_reporter.py
python/sglang/srt/managers/scheduler_components/output_streamer.py
python/sglang/srt/managers/schedule_batch.py
python/sglang/srt/managers/tokenizer_manager.py
python/sglang/srt/speculative/dspark_components/dspark_block_accept_estimator.py
python/sglang/srt/speculative/dspark_components/dspark_observability.py
```

Tests and benchmarks to reuse:

```text
test/registered/unit/spec/test_adaptive_runtime_state.py     (132 lines)
test/registered/unit/spec/test_adaptive_spec_params.py       (425 lines)
test/registered/spec/eagle/test_adaptive_speculative.py      (316 lines)
benchmark/bench_adaptive_speculative.py                      (263 lines, high/low/transition)
```

Docs: `docs/docs/advanced_features/adaptive_speculative_decoding.mdx`
