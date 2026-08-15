# Wave 27 — AY: why decode aggregate throughput FALLS as batch rises

**Scope.** Source investigation only. No GPU rented. Every claim below is
either CONFIRMED (code read, path traced, `file:line` cited) or SUSPECTED
(inferred, labelled). Base: `origin/master` @ `d437f72fb`, plus PR #51's
`memory/mission/wave26-AX-h200-measurement.md` read from
`origin/perf/wave26-h200-batched-measurement` (`5b1394849`).

---

## 0. The one-paragraph answer

The prime suspect is **not** the cause. The GPU radix top-k sampler *is* 100%
dead — for a reason I confirmed exactly — but the measured run used
`temperature=0.0`, which takes the greedy branch and never reaches that code.
The real cause is upstream of sampling: **Arc's fused 2-bit MoE gather kernel
is hard-capped at 8 tokens per step**, and above that cap decode leaves the
fused path entirely. The engine's own scheduler log in AX's report shows the
step token count crossing that cap between B=16 (8 running) and B=32 (13
running) — exactly where aggregate throughput halves. Compounding it, V4
disables PagedAttention, which forces the length-bucketing `DefaultScheduler`
and waitlists half the batch every step.

---

## 1. The prime suspect: GPU radix top-k — CONFIRMED DEAD, but explains ~0%

### 1a. Why it fails — the exact condition

`arc-cuda-graph/src/flashmlasparse.rs:188` (pre-fix):

```rust
fn cuda_tensor_ptr(t: &Tensor) -> Result<usize> {
    let t = t.contiguous()?;
    let (storage, layout) = t.storage_and_layout();
    match &*storage {
        candle_core::Storage::Cuda(cuda_storage) => {
            let slice = cuda_storage.as_cuda_slice::<u8>()?;   // <-- always u8
```

`CudaStorage::as_cuda_slice::<T>()` type-checks `T` against the storage's
dtype and returns `Error::UnexpectedDType` otherwise — candle
`candle-core/src/cuda_backend/mod.rs:1295-1304` (checkout
`~/.cargo/git/checkouts/candle-601b509b8bac78d2/d2d1d07`), whose `Display`
renders precisely:

```
unexpected dtype, expected: U8, got: F32
```

That is, verbatim, the string AX recorded from the H200 server
(`wave26-AX-h200-measurement.md:102-105`). So the failure is **unconditional
for every non-U8 tensor**, not a shape/vocab/hardware edge case.

Path: `mistralrs-core/src/sampler.rs:812` calls
`arc_cuda_graph::flashmlasparse::radix_topk_rows_f32`, which at
`arc-cuda-graph/src/flashmlasparse.rs:451` takes `cuda_tensor_ptr(&scores_c)`
on an **F32** tensor. It can never succeed. The `Err(e)` arm at
`mistralrs-core/src/sampler.rs:1279-1282` logs the warning and falls through.

It is NOT the documented "exactness cannot be guaranteed" `Ok(None)` bail
(`sampler.rs:791-800`): `GPU_RADIX_TOPK_SIZES` (`sampler.rs:295`) and
`SUPPORTED_TOPK` (`flashmlasparse.rs:64`) are the identical
`[64,128,256,512,1024]`, so the size guard is satisfiable.

### 1b. Is the CPU fallback per-sequence serialized? — **No.**

`mistralrs-core/src/pipeline/sampling.rs:437-482` builds one future per
sequence and `futures::future::join_all`s them; with `use_async_pool =
seqs_len > 1` (`sampling.rs:448`) each one runs on the rayon pool via
`tokio_rayon::spawn` (`sampling.rs:503`). Sampling is concurrent across the
batch, not serialized. (It is still *per-sequence work* — a full-vocab D2H at
`mistralrs-core/src/sampler.rs:1288` plus a host sort — but it is not a
serial loop.)

### 1c. How much of the gap does it explain? — **None of it.**

`arc-tools/quality/batch_load_probe.py:147` and `:160` send `top_p: 1.0`, and
`:629` defaults `--temperature 0.0`; AX confirms "temperature 0"
(`wave26-AX-h200-measurement.md:48`). `mistralrs-core/src/sampler.rs:324`
maps any temperature `< 1e-7` to `None`, and `sampler.rs:1224-1240` then
takes the greedy branch: a single GPU `argmax` plus a 4-byte D2H. The radix
top-k branch at `sampler.rs:1276` sits **below** that early `return` and is
unreachable for the measured requests. The startup warning came from the
warmup/dummy run, not from the sweep.

**This is the project's dominant failure mode showing up again: the most
plausible-looking culprit, with a real confirmed defect behind it, on a code
path the measurement never executed.** Fixed anyway (§5) — but it is not the
answer.

---

## 2. The actual cause, ranked

### #1 — CONFIRMED — the fused 2-bit MoE gather is capped at 8 tokens/step

Both QTIP rungs make the same decision:

- `mistralrs-quant/src/qtip/mod.rs:3189-3202` (LUT rung)
- `mistralrs-quant/src/qtip/bitshift.rs:1495-1500` (2-bit rung)

```rust
let ondevice_max_tokens = std::env::var("ARC_QTIP_ONDEVICE_MOE_MAX_TOKENS")
    .ok().and_then(|v| v.parse::<usize>().ok())
    .unwrap_or(8);
if !ondevice_disabled && n_tokens <= ondevice_max_tokens {
    return self.gather_forward_cuda_ondevice(a, indices);
}
```

Above 8 tokens per forward:

- **LUT rung** falls to `gather_forward_cuda`, which the code's own comment
  at `mistralrs-quant/src/qtip/mod.rs:3205-3209` describes as
  "the per-expert dequantize below **materializes weights to HBM**", guarded
  by `warn_dequant_materialize_at_decode(...)`. A 2-bit expert stack gets
  expanded to BF16 in HBM — an 8× write plus an 8× read against the
  quantized bytes the whole artifact exists to avoid.
- **2-bit rung** falls to `gather_forward_batched` →
  `grouped_gemm_2b_cuda` (`bitshift.rs:1503-1526`), and only when
  `in_features % GROUPED_TILE_K == 0` (`GROUPED_TILE_K = 64`,
  `mistralrs-quant/src/qtip/grouped.rs:33`) and the dtype is BF16/F16;
  otherwise `gather_forward_cpu` (`bitshift.rs:1529-1530`).

`n_tokens` is the batch token count for the step:
`mistralrs-core/src/moe/experts.rs:576` (`num_tokens = b_size * seq_len`),
reaching the layer via `gather_forward_autocast` at `experts.rs:583`, `:586`,
`:614`, dispatched from `mistralrs-core/src/models/deepseek4.rs:2019`.

**AX's engine log is the confirmation.** `wave26-AX-h200-measurement.md:83-88`
records the scheduler's own running-set size:

| B | running | tokens/step | MoE path | agg tok/s | step ms | **ms per seq in step** |
|---|---|---|---|---|---|---|
| 1 | 1 | 1 | fused on-device | 15.35 | 65.1 | **65.1** |
| 8 | 5 | 5 | fused on-device | 14.83 | 337 | **67.4** |
| 16 | 8 | 8 | fused (at the cap) | 10.31 | 776 | **97.0** |
| 32 | 13 | 13 | **over the cap** | 5.07 | 2564 | **197.2** |
| 64 | 32 | 32 | over the cap | 8.14 | 3931 | **122.8** |

(step ms = running / aggregate; both columns are AX's measured values.)

Two things fall straight out:

1. **Below the cap, per-sequence cost is flat**: 65.1 ms at 1 token, 67.4 ms
   at 5. Batching buys *nothing* — see #2.
2. **The cost per sequence doubles exactly where the batch crosses 8**:
   97.0 ms at 8 tokens → 197.2 ms at 13. That is the path switch, not a
   gradual roofline effect. Aggregate halves (10.31 → 5.07) for a 1.6×
   larger batch.

The fleet thesis is batches ≫ 8. **Arc's fleet regime never runs Arc's fast
MoE kernel.** This is the same finding the goal audit already named as the
keystone ("trellis grouped-GEMM"), now with an end-to-end number attached and
a specific numeric threshold to move.

### #2 — CONFIRMED — even under the cap, the fused path has zero batch amortization

`mistralrs-quant/src/qtip/bitshift.rs:1311-1348`
(`gather_forward_cuda_ondevice`) flattens to
`n_tokens * n_experts_per_tok` pairs and calls `gather_gemv_2b_cuda`
(`mistralrs-quant/src/qtip/cuda_ops.rs:1333`), which asserts
`indices.elem_count() == n_pairs` (`cuda_ops.rs:1365-1368`). It is a **GEMV
per (token, expert) pair** with no dedup: 32 tokens routed to the same expert
decode that expert's trellis 32 times.

Cost is therefore exactly linear in tokens — which is what the 65.1 → 67.4
ms/seq numbers say. A batched decode forward should cost roughly the *same*
wall-time for 1 token as for 8, because weights dominate.

### #3 — CONFIRMED — V4 disables PagedAttention, which waitlists half the batch

Chain, all four links read:

1. `mistralrs-core/src/pipeline/loaders/normal_loaders.rs:3231-3237` —
   `DeepSeekV4Loader::supports_paged_attention` returns `Ok(false)`
   (MLA `head_dim=512` exceeds the paged kernel's supported head sizes).
2. `mistralrs-core/src/pipeline/normal.rs:345-347` — nulls
   `paged_attn_config`, so `cache_config` stays `None`
   (`normal.rs:1155`, `:1206`).
3. `mistralrs-server-core/src/mistralrs_for_server_builder.rs:1263-1285` —
   `init_scheduler_config` with no `cache_config` selects
   `SchedulerConfig::DefaultScheduler`.
4. `mistralrs-core/src/scheduler/default_scheduler.rs:76-171` —
   `FixedBucketingManager` buckets running sequences by
   `(seq.len(), has_imgs && is_prompt, token_offset)` and, when there is more
   than one bucket, runs **only the winning bucket** and waitlists the rest
   (`default_scheduler.rs:128-168`). Winner is max summed
   `compute_priority()` = `scheduling_urgency + log2(len)`
   (`mistralrs-core/src/sequence.rs:625-628`).

Because sequences only share a bucket when their token counts are *exactly*
equal, a continuously-batched workload with varied prompt lengths never gets
the full batch into one forward. AX's log
(`wave26-AX-h200-measurement.md:83-88`) is the direct measurement of this:
`32 running, 32 waiting` at B=64; `13 running, 19 waiting` at B=32;
`8 running, 8 waiting` at B=16. A flat **2×–2.5× loss** on top of #1/#2.

This is not a bug in the bucketing — the non-paged batched KV cache is a
dense `[B, H, L, D]` tensor and genuinely cannot mix lengths
(`mistralrs-core/src/kv_cache/mod.rs:353-380`, `:840-865`). The defect is
that V4 has no paged/ragged attention path at all, so it inherits the
scheduler that cannot do continuous batching.

### #4 — CONFIRMED — V4 also loses the CUDA-graph autonomous decode path

Same root as #3. `mistralrs-core/src/pipeline/normal.rs:1841-1847`:

```rust
let block_size = match self.metadata.cache_config.as_ref() {
    Some(c) => c.block_size,
    None => { tracing::debug!("autonomous_decode: no cache_config, falling back"); return Ok(None); }
};
```

`cache_config` is `None` on V4, so the entire 3-tier GPU-autonomous decode
plan is unreachable on the one model Arc actually serves. It is `debug!`, so
nothing in the session log says so.

### #5 — CONFIRMED — `cuda_tensor_ptr` also kills two other GPU paths

The same one-line dtype bug (§1a) exists twice:
`arc-cuda-graph/src/flashmlasparse.rs:188` and
`arc-cuda-graph/src/sampling_cuda.rs:218`. Beyond the radix sampler it takes
out:

- **The V4 Lightning Indexer's fused CUDA kernel** —
  `mistralrs-core/src/models/dsv4_indexer.rs:419` calls `score_and_topk_bf16`
  with `?` and no fallback; that function takes `cuda_tensor_ptr` on BF16
  tensors at `flashmlasparse.rs:376-381`. It does not crash generation today
  only because its guard almost never passes: `dsv4_indexer.rs:408` requires
  `SUPPORTED_TOPK.contains(&self.topk.min(t_c))` where `t_c = t_full / ratio`
  (`dsv4_indexer.rs:317`) — the *compressed* context length, ~33 at the
  session's ~132-token contexts. So the kernel is doubly dead: unreachable by
  guard, and broken if reached. Meanwhile the pure-Rust reference path
  (`dsv4_indexer.rs:344-380`) recomputes the inner compressor over the
  **entire K history** every layer every step, with no caching of
  `indexer_k`.
- **The fused GPU sampler** `CudaSampler`
  (`arc-cuda-graph/src/sampling_cuda.rs:361-368`), which is only reachable
  through the autonomous-decode runner — itself dead on V4 per #4.

Three "wired but never invoked" paths from one hardcoded type parameter.
BACKLOG cases 6, 7, 8.

---

## 3. The cost model — how much of the ~500× is accounted for

AX's roofline (`wave26-AX-h200-measurement.md:95-97`) is 74.2 GB ÷ 4.8 TB/s ⇒
15.46 ms/step ⇒ ~4,141 tok/s at B=64, against a measured 8.14 — **509×**.

Two corrections to that denominator first, then the decomposition.

**Correction A — the 74.2 GB roofline is only valid asymptotically.** For a
256-expert top-8 MoE, a step reads only the *activated* expert set. Expected
distinct experts for `n` tokens is `256·(1 − (1 − 8/256)^n)`:

| tokens/step | 1 | 5 | 8 | 13 | 32 |
|---|---|---|---|---|---|
| distinct experts | 8 | 38 | 57 | 86 | 163 |

So at 32 tokens/step a *perfect* implementation reads ~163/256 of the expert
bytes, not all of them — and, crucially, **a correct grouped kernel only gains
1.57× going from 1 token to 32**, because the activated set grows almost
linearly in this range. Batching a 256-expert top-8 MoE is close to free of
headroom below ~64 tokens/step. That is a property of the architecture, not a
defect, and it should be stated next to any "×4–8 per node" claim.

**Correction B — the honest roofline at the measured shapes.** Taking dense ≈
7 GB and experts ≈ 67 GB of the 74.2 GB artifact:

- 1 token/step: 7 + 67·(8/256) = 9.1 GB ⇒ 1.9 ms ⇒ ~527 tok/s
- 32 tokens/step: 7 + 67·(163/256) = 49.6 GB ⇒ 10.3 ms ⇒ ~3,100 tok/s

**Decomposition of the measured gap at B=64:**

| term | factor | evidence |
|---|---|---|
| single-stream is already far off roofline | **34×** | 65.1 ms measured vs 1.9 ms roofline at 1 token/step. Pre-existing, not new. |
| scheduler waitlists half the batch (#3) | **2.0×** | engine log `32 running, 32 waiting`; `default_scheduler.rs:128-168` |
| batching costs *more* per token instead of less (#1+#2) | **5.6×** | measured 122.8 ms/seq at 32 tokens vs 65.1 ms/seq at 1, where a correct grouped kernel predicts 65.1/1.57 = 41.5 ms/seq. 122.8/41.5 = 2.96×, times the 1.9× that a correct kernel would have *gained* |
| product | **≈381×** | vs the 3,100 tok/s corrected roofline at 32 running |
| ×2.0 for the 64→32 waitlisting against the B=64 roofline | **≈509×** | matches the observed 4,141 / 8.14 |

The single-stream 34× is the largest single term and was *masked* by AX's
roofline, which used the full 74.2 GB — a denominator only correct at large
batch. Under the corrected b=1 roofline of ~527 tok/s, the measured 15.35 is
2.9%, not "at full expected speed"; it is only "at full expected speed"
relative to Arc's own history (14.58).

**The specific batching defect — the part that inverts the fleet thesis — is
the 5.6× term, and it is entirely #1 and #2.** Below 8 tokens/step decode
cost is exactly linear in tokens (65.1 → 67.4 ms/seq), so aggregate is flat;
above 8 tokens/step the code leaves the fused kernel and cost per sequence
roughly doubles, so aggregate *falls*. That is the measured shape, precisely.

---

## 4. Other per-sequence work on the decode path (ranked, all CONFIRMED)

Each of these is real but an order of magnitude below §2. Listed so the next
GPU session can attribute residuals rather than re-derive them.

1. **Per-sequence GPU `Tensor::new` in the decode input build** —
   `mistralrs-core/src/pipeline/inputs_processor.rs:526` opens
   `for (seq, ctxt) in input_seqs.iter().zip(toks)` and line `:538` does
   `Tensor::new(ctxt, device)` on the **model device**, one 1-element H2D per
   sequence per step, then `Tensor::cat` at `:815`. This is exactly the
   CLAUDE.md pitfall #5. Every *other* per-step tensor in that file is
   correctly built on `&Device::Cpu` (`:628-731`) and uploaded once.
2. **The global KV-manager mutex is taken inside that same loop** —
   `inputs_processor.rs:541`, released `:546`. `get_mut_arcmutex!` is a
   spin-yield loop, not a parking lock
   (`mistralrs-core/src/utils/mod.rs:15-27`).
3. **RoPE takes a per-sequence loop whenever batch > 1** — every impl
   branches on `seqlen_offsets.len() == 1`; V4 goes through
   `mistralrs-core/src/layers.rs:1627` (`DeepSeekV2RotaryEmbedding::forward`),
   per-seq `.i(i)` + `.contiguous()` + separate kernel + final `cat`, per
   layer per step. Note the paged scheduler already forces equal lengths
   (`mistralrs-core/src/paged_attention/scheduler.rs:139-143`, `:378`), so
   the fast branch would be numerically identical — but `len() == 1` is false
   and the slow loop runs anyway.
4. **Per-sequence full-vocab D2H when batch > 1** —
   `mistralrs-core/src/pipeline/mod.rs:932-943` (default path) and
   `:1155-1166` (paged): `logits_on_cpu = logits.len() > 1`, then N separate
   blocking GPU→CPU vocab-sized copies. The single-sequence case keeps logits
   on GPU, so the batched case is strictly the slower one.
5. **Sampler uploads the whole token history per sequence per step** —
   `mistralrs-core/src/sampler.rs:481` `Tensor::new(context, logits.device())`
   where `context` is `seq.get_toks().to_vec()`
   (`mistralrs-core/src/pipeline/sampling.rs:499`).
6. **KV marshalling per (layer × sequence)** on the non-paged path V4 uses —
   `mistralrs-core/src/kv_cache/mod.rs:356` loop with `slice_set` at
   `:379-380`, `clone_out_cache` at `:496-509`, `FullCacheManager` cat at
   `:840-865`. `post_op` is unconditionally `CacheInstruction::Out`
   (`mistralrs-core/src/engine/mod.rs:397-399`).
7. **The autonomous-decode context is built per sequence every step and then
   discarded** — `mistralrs-core/src/engine/mod.rs:734-751` builds a block
   table padded to `max_blocks_per_seq` (5,120 `i32` per sequence at
   `max_seq_len=163840, block_size=32`;
   `mistralrs-core/src/paged_attention/kv_cache_manager.rs:426`) before
   calling `autonomous_decode`, which returns `Ok(None)` on V4 (#4).
8. **An unconditional debug cast in the MoE hot path** —
   `mistralrs-core/src/models/deepseek4.rs:2027` evaluates
   `topk_idx.to_dtype(DType::F32)` as an *argument* to `v4_collapse_dbg`, so
   the GPU cast runs on every layer every step even though the function
   early-returns on the env check
   (`deepseek4.rs:2252-2255`).

**Not a factor** (checked, so nobody re-scans): the schedulers do issue **one**
forward per step for the whole running set — `engine/mod.rs:415-425` (default)
and `:893-903` (paged), one input-processor call at `pipeline/mod.rs:1069-1082`
and one `graph_wrapped_forward` at `:1099`. `Sequence::update_time_info`
overwriting group totals is confined to reporting; it feeds
`usage.total_prompt_time_sec`, not scheduling. Prefill scaling 4.8× is
consistent with everything above: prefill runs many tokens per sequence, so it
is always in the `n_tokens > 8` regime for *both* B=1 and B=64 and never
crosses a path boundary.

---

## 5. What was fixed

Only #5's one-line defect — unambiguous, small, and evidenced by a verbatim
production log line.

`arc-cuda-graph/src/flashmlasparse.rs:188` and
`arc-cuda-graph/src/sampling_cuda.rs:218` now delegate to
`arc-cuda-graph/src/weights.rs:128` (`tensor_device_ptr`), which already does
the dtype dispatch correctly and carries the comment "cudarc requires type
match". Two unused `DevicePtr` imports removed.

Regression test: `radix_topk_rows_f32_accepts_f32_scores`
(`arc-cuda-graph/src/flashmlasparse.rs`, `#[cfg(feature = "cuda")]`). It
builds a strictly-decreasing F32 `[1, 4096]` score row and asserts the kernel
returns exactly indices `0..64`. **Without the fix it fails at the `?` with
`unexpected dtype, expected: U8, got: F32`** — the same error the H200 logged.
It no-ops when `Device::new_cuda(0)` is unavailable, so it is a GPU-session
test.

Nothing else was changed. #1–#4 are architectural and are written up here
rather than patched.

---

## 6. What still needs a GPU — one measurement, not an exploration

Run in this order; the first item is the whole ballgame.

1. **Re-run the B-sweep with `ARC_QTIP_ONDEVICE_MOE_MAX_TOKENS=64`.** One env
   var, no rebuild. If #1 is right, the B=32 and B=64 aggregate rows rise and
   the 8→13 token cliff disappears. If they do not move, #1 is wrong and the
   fused path is not actually faster than the fallback at these shapes —
   which is itself the answer, and means the grouped GEMM needs the work.
   Also set `ARC_WARN_DEQUANT_MATERIALIZE=1` so the fallback announces itself
   (`mistralrs-quant/src/qtip/mod.rs:107-127`).
2. **Log which QTIP rung the served artifact actually uses** (LUT
   `QtipLayer` vs `Qtip2bLayer`). #1's severity differs: the LUT rung's >8
   fallback dequantizes to HBM, the 2-bit rung's goes to the grouped GEMM.
   This determines whether the fix is "raise the cap" or "finish the grouped
   kernel".
3. **Confirm the fix**: `cargo test -p arc-cuda-graph --features cuda
   radix_topk_rows_f32_accepts_f32_scores`. Should fail on `master`, pass on
   this branch. Then check the startup warning is gone.
4. **Quantify #3 independently**: serve with `--paged-attn off` vs the current
   default and compare `running`/`waiting` in the engine log. They should be
   identical (V4 already never gets paged attention) — if they differ, link 3
   of the chain is wrong.
5. **Time one decode step at 8 vs 9 tokens** with NSys or a step-timer. #1
   predicts a discontinuity at exactly 9.

Everything above is a single 30-minute server session on one card. No
re-bake, no re-quantize.

---

## 7. Surfaced, not shipped

> **Noticed:** `cuda_tensor_ptr` returns a `usize` derived from
> `t.contiguous()?`, a **local temporary**. When the input is not already
> contiguous, the copy's storage is freed at function exit and the returned
> pointer dangles. Every current caller passes an already-contiguous tensor
> (so `contiguous()` is a no-op clone and the caller keeps the storage alive),
> which is why this has never fired. It is one `bail!` away from being
> impossible. Left alone to keep this change to the confirmed defect.
> Worth a separate change?

> **Noticed:** the roofline denominator used in `docs/BENCHMARKS.md` and
> `wave26-AX` (full artifact bytes per step) is only correct asymptotically
> for a sparse MoE, and it flatters large-B while understating how far b=1 is
> from the bound (§3, Correction A/B). Any public "×N per node" figure derived
> from it inherits the error in both directions.
> Worth a separate change?

> **Noticed:** `ARC_QTIP_ONDEVICE_MOE_MAX_TOKENS`, `ARC_NO_QTIP_GROUPED_MOE`,
> `ARC_NO_QTIP_ONDEVICE_MOE` and `ARC_WARN_DEQUANT_MATERIALIZE` are four
> silent `std::env::var` reads that select between MoE kernels with
> order-of-magnitude performance consequences, and none of them is logged at
> startup or recorded in any results artifact. The measured sweep cannot be
> reproduced from its own record.
> Worth a separate change?
