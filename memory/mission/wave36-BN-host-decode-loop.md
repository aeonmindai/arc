# wave36-BN — the host side of the decode loop

**Date:** 2026-08-16 · **Base:** `master` @ `36ce1699f` · **Branch:** `perf/host-decode-loop`
**No GPU was rented. $0.00.** Everything below is source analysis plus arithmetic
against `wave34-BL-post-fix-sweep.md`'s measured table. Every projection is
labelled `[projection]`; nothing here is a measurement of Arc.

---

## 0. Headline

**The decode step copies the logits from device to host `B` times per step, and
each copy moves the WHOLE batch.** That is the only `O(B²)` term in the step,
and it is the only thing that can make aggregate throughput *fall* as `B` grows
— which is exactly what B=128 → B=256 does (28.86 → 19.02 tok/s).

`mistralrs-core/src/pipeline/mod.rs:1155-1166` (paged) and `:932-943` (default,
**the branch V4 actually takes**) did:

```rust
logits[seq_idx] = Some(raw_logits.index_bs(logit_idx)?);   // a VIEW
...
let logits_on_cpu = logits.len() > 1;
l.to_device(&Device::Cpu)                                   // per sequence
```

Two verified candle facts make that quadratic:

| fact | source |
|---|---|
| `Tensor::narrow` (and therefore `Tensor::i()`) returns a **view**: `storage: self.storage.clone()` | candle `d2d1d07`, `candle-core/src/tensor.rs:906` |
| `Tensor::to_device` copies the **whole storage** and only carries the view's layout across: `Storage::Cpu(storage.to_cpu_storage()?)` → `clone_dtoh(slice)` | `tensor.rs:2379`, `cuda_backend/mod.rs:1788-1791` |

So each of the B per-sequence "row" copies drags all `B × vocab` elements over
PCIe. V4's `vocab_size = 129_280` (`models/deepseek4.rs:4529`) and the logits
come back in the model dtype — BF16 — because `forward_autocast` restores
`original_ty` (`mistralrs-quant/src/lib.rs:1384-1391`).

**Per decode step: `B² × 129_280 × 2` bytes of D2H, plus B fresh host `Vec`
allocations that are all held live simultaneously.**

| B | D2H bytes/step (before) | host RAM held live |
|---|---|---|
| 8 | 265 MB | 265 MB |
| 64 | 1.06 GB | 1.06 GB |
| 128 | 4.24 GB | 4.24 GB |
| **256** | **16.94 GB** | **16.94 GB** |

That is why the box showed **one core at 100 %, GPU at 0-4 % and 121 W**: a
pageable `cudaMemcpyDtoH` is a CPU-blocking driver call, and each call also
`malloc`s + first-touches a fresh 66 MB host buffer. 16.9 GB of that per step is
pure single-threaded host work with no kernel in flight.

---

## 1. The arithmetic, before the fix

Step time is `B / aggregate`, from wave34-BL §2.1:

| B | 1 | 8 | 16 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|---|---|
| agg tok/s | 16.27 | 26.96 | 30.65 | 30.59 | 30.41 | 28.86 | 19.02 |
| **step ms** | 61.5 | 296.7 | 522.0 | 1046.1 | 2104.6 | 4435.2 | **13459.5** |

Fit `step(B) = k + a·B + c·B²` on the three rows B=16/64/256:

```
c = 0.1090 ms   a = 24.26 ms   k = 105.9 ms
check  B=8  : 307 predicted vs 297 measured  (+3%)
       B=32 : 995            vs 1046         (-5%)
       B=128: 5002           vs 4435         (+13%)
```

**The quadratic term, and what it costs:**

| B | `c·B²` | share of the step |
|---|---|---|
| 16 | 28 ms | 5 % |
| 64 | 446 ms | 21 % |
| 128 | 1786 ms | 40 % |
| **256** | **7143 ms** | **53 %** |

**Does the D2H model reproduce `c`?** `c = vocab × bytes_per_elem / BW`, so
`BW = 129_280 × 2 / 1.09e-4 s = 2.37 GB/s`. That is an entirely ordinary
effective rate for pageable-memory D2H *plus* a fresh 66 MB `Vec` allocation and
first-touch on every call. The model is not tuned to fit — `vocab` and the dtype
are read out of the source, and the only free parameter lands on a physically
plausible number.

**Why aggregate must fall.** `aggregate = B / (k + aB + cB²) = 1 / (k/B + a + cB)`.
Any non-zero `c` makes aggregate decline once `cB` overtakes `k/B`. A purely
linear step cost can only plateau, never decline. wave34-BL measured a decline.
**Nothing else in the step is quadratic in B.**

**Cross-check against the power/util observation, at BOTH batch sizes.** At
B=64 the quadratic is 21 % of the step, so most of the timeline still has
kernels in it (420-476 W, 97-100 % util). At B=256 it is 53 % and it is one
contiguous host-side stall, so a sample taken during decode overwhelmingly lands
inside it (0-4 %, 121 W). The same model explains both rows.

---

## 2. ⚠️ The brief's b=1 premise is wrong, and it matters

The brief said this term "dominates the 87× gap at b=1". **It contributes
exactly zero at b=1.** `logits_on_cpu = logits.len() > 1` was `false`, so no
bytes crossed the boundary at all; and wave34-BL's own profile has
`forward_total = 67.24 ms` against a 61.5 ms step — i.e. at B=1 the model
forward *is* the step. **The b=1 gap is GPU-side kernel efficiency (MoE GEMV,
MLA), not host overhead.** The fix below deliberately leaves B=1 on the device,
because `Sampler`'s GPU fast path is gated on `!logits.device().is_cpu()`
(`sampler.rs:1220`) — pulling a B=1 batch to the host would *disable* it.

---

## 3. ⚠️ What the quadratic does NOT explain — the linear term

`a = 24.26 ms per sequence per step` is **not** accounted for by anything in
this report, and it is larger than the quadratic below B≈223. wave34-BL's own
B=64 profile puts `forward_total` at **210 ms of a 2105 ms step — 10 %**. So
~90 % of the B=64 step is outside `forward_inputs`, of which the quadratic is
21 points. **The remaining ~69 points are unattributed.** §4 lists concrete O(B)
suspects, none of which has been shown to sum to 24 ms/seq. Do not let this
report close the question.

> **Caveat on that 10 %, stated because it is load-bearing.** The 210 ms came
> from a **separate serve process** (`ARC_TIME_DECODE=1`) and the 2105 ms from
> the sweep process; the ratio assumes the two ran at the same step rate. Same
> binary, same model, same box — but it is a cross-process comparison, not one
> measurement. Confirming it needs the §8.3 instrumentation on one process.

**Instrumentation gap that makes this hard:** the `STEP_us avg ... TOTAL= fwd=
sample= other=` line only exists in the **PagedAttention** branch
(`pipeline/mod.rs:1300-1324`). V4 has `supports_paged_attention() -> false`
(`loaders/normal_loaders.rs:3231`), so it runs the DefaultScheduler branch,
which logs **no** per-step fwd/host split at all. That is the single cheapest
next instrument to add.

---

## 4. Ranked per-step host costs

| # | cost | `file:line` | order in B | label |
|---|---|---|---|---|
| 1 | **per-sequence D2H of the whole logits batch** | `pipeline/mod.rs:924`, `:1161` | **O(B²) bytes** | FIXABLE-NOW — **FIXED here** |
| 2 | `Tensor::new(ctxt, device)` **inside** the per-seq decode loop → B device allocations + B H2D per step, then a B-way `Tensor::cat` | `pipeline/inputs_processor.rs:538`, cat at `:815` | O(B) allocs + O(B) H2D | FIXABLE-NOW — scoped |
| 3 | `post_op = CacheInstruction::Out` **unconditionally** → `clone_out_cache` every step even when the same sequence set runs next step; rebuilds `layers × B` `KvCache` values, incl. a full `XsRollingCache` struct clone per (layer, seq) | `engine/mod.rs:397-404`; `kv_cache/mod.rs:605-708`, clone at `:701` | O(B × layers) | FIXABLE-NOW — scoped (KV/xs fence) |
| 4 | `seq.get_toks().to_vec()` — clones the whole token history per sequence per step | `pipeline/sampling.rs:499` | O(B × ctx) | FIXABLE-NOW — scoped |
| 5 | the post-sample loop is **serial on the engine task** and `.await`s `responder.send()` per sequence while holding the pipeline mutex; only the sampler half is rayon-parallel (`:503`) | `pipeline/sampling.rs:466-480`; `sequence.rs:1606` | O(B) serial | FIXABLE-NOW — scoped (43 of 44 cores idle) |
| 6 | `update_time_info` takes the group lock **5 separate times**, once per token per sequence via `add_streaming_chunk_choice_to_group`; the group lock is a **spin loop** (`try_lock` + `yield_now`) — the mechanism that converts any contention into 100 % of one core. Also the known BACKLOG bug: assigns (`=`) group totals instead of accumulating | `sequence.rs:1077-1102`, `:1150-1153`; `utils/mod.rs:227-237` | O(B) locks | FIXABLE-NOW — scoped |
| 7 | stop-string search re-scans the **entire** `completion_bytes` for every stop string, every token, every sequence | `sequence.rs:1005` | O(B·L)/step, O(B·L²)/completion | FIXABLE-NOW — scoped. **Zero cost in this sweep (no stop strings) — a customer cliff, not a sweep finding** |
| 8 | the scheduler moves all B `Sequence` structs by value into a `HashMap<BucketKey, Vec<Sequence>>` and back, every step | `scheduler/default_scheduler.rs:174-214`, `:226-239` | O(B) moves | FIXABLE-NOW — small |
| 9 | batched decode **never** uses the GPU sampler: the fast path is gated on `!logits.device().is_cpu()`, and B>1 always lands on the host | `sampler.rs:1220`, CPU path `:1288` | O(B·vocab) CPU | larger project, still not graphs |

**None of these is `NEEDS-GRAPHS`.** CUDA-graph capture attacks kernel-launch
overhead inside the forward; every item above is host work that exists
regardless of how the forward is dispatched. The megakernel remains orthogonal
and correctly deferred.

---

## 5. What was fixed

`mistralrs-core/src/pipeline/mod.rs` — one host copy of the **batch**, hoisted
above the per-sequence split, in both `step()` branches:

```rust
let raw_logits = host_copy_batched_result(raw_logits, input_seqs.len(), &Device::Cpu)?;
for (logit_idx, seq_idx) in seq_indices.into_iter().enumerate() { ... }  // views only
```

`host_copy_batched_result` also preserves two behaviours on purpose:
- **B=1 CausalGeneration stays on the device**, keeping `Sampler`'s GPU argmax /
  radix top-k path alive.
- raw-logit and embedding requests always come back to the host, at any B.

Side effect worth knowing: on the paged branch the logits D2H moves out of
`STEP_us`' `sample` bucket into `other`. Same work, attributed where it happens.

**D2H per step after the fix: `B × vocab × 2` bytes — a factor of B less.**
At B=256 that is 66.2 MB instead of 16.94 GB, and 66.2 MB of live host RAM
instead of 16.94 GB.

---

## 6. The test, and the proof it can fail

`mistralrs-core/src/pipeline/mod.rs` → `mod host_copy_tests`, one test:
`batched_host_copy_is_o1_in_batch_while_the_old_order_was_o_b_squared`.

It asserts, for B ∈ {2, 5, 8, 64}:

1. **The premise, measured against candle rather than assumed.**
   `raw.index_bs(2)` has `elem_count() == V` but
   `storage_elems(...) == b * V`, read through candle's **public**
   `Tensor::storage_and_layout()`. If `Tensor::i()` ever starts copying, this
   assertion fails loudly and the whole O(B²) premise is retired rather than
   quietly surviving as folklore.
2. **Fixed order:** exactly **1** host copy per step at every B (O(1) in B), one
   batch of elements crosses the boundary, and each sequence still gets its own
   correct row.
3. **Mutation control, in the test itself:** the pre-fix order (split, then copy
   per row) through the *same* counter and the *same* helper yields **B** copies
   and `b·b·V` elements, and `old_elems / (b·V) == b`.
4. **B=1 issues zero copies**; raw-logits/embeddings issue one at any B.

**Fixture discrimination (DOCTRINE D12).** `V = 7` is equal to none of the batch
sizes and coprime with all of them, so no assertion can pass by `b·V` aliasing
`b`, `V`, or `V·b`; `assert_ne!(b, V)` is asserted explicitly. The row-contents
check uses a ramp `0..b·V`, so every row differs from every other row *and* from
its own index — a splitter that returned row 0 for everything fails.

**Mutation-proved, not asserted.** The production helper was temporarily changed
to `for _ in 0..n_seqs { counter += 1 }` (the pre-fix behaviour) and the test was
re-run:

```
thread '...batched_host_copy_is_o1_in_batch_while_the_old_order_was_o_b_squared' panicked
assertion `left == right` failed: B=2: one host copy per STEP, not one per sequence
  left: 2
 right: 1
test result: FAILED. 0 passed; 1 failed
```

The mutation was then reverted and the test passes again.

**Honest limit of the test:** it pins the helper, not the call sites. The call
sites are one-line uses of it, but a future edit that re-introduces a
per-sequence `to_device` *outside* the helper would not be caught. Driving
`Pipeline::step` end to end needs a stub pipeline with a real inputs processor
and tokenizer, which was out of scope here.

---

## 7. Projection — **`[projection]`, not measured**

Subtract the fitted quadratic and add back the single copy
(`B·vocab·2 / 2.37 GB/s`):

| B | measured agg | step after (ms) | agg after `[projection]` | range `[projection]` |
|---|---|---|---|---|
| 64 | 30.41 | 1666 | 38.4 | **34 – 40** |
| 128 | 28.86 | 2663 | 48.1 | **40 – 52** |
| 256 | 19.02 | 6345 | 40.3 | **32 – 46** |
| 1 | 16.27 | unchanged | **16.27** | unchanged — see §2 |

The band is ±30 %, sized on the fit's worst residual (13 % at B=128). The shape
claim is the confident part: **aggregate should stop declining and resume rising
with B**, because the only quadratic term is gone. The 24.3 ms/seq linear term
(§3) survives untouched and is what then caps it.

---

## 8. What needs a GPU to confirm

1. The effective D2H rate (`2.37 GB/s` is inferred from the fit, not measured).
   `nsys`/`nvprof` on one decode step at B=256 settles it directly.
2. The post-fix sweep, same protocol as wave34-BL (64 decode tokens, 1 rep,
   `--max-seqs 256`, `--prefix-cache-n 0`), rows B=1/8/16/32/64/128/256.
3. **The unattributed linear term (§3)** — needs a per-step fwd/host split on the
   **DefaultScheduler** branch, which currently logs none. Add the `STEP_us`
   block there before renting.
4. Whether `nvidia-smi`'s 97-100 % at B=64 survives once the stall is gone, or
   whether it was always a coarse-window artefact over a ~10 %-busy timeline.

---

## 9. Surfaced, not shipped

- **Item 2 in §4 is CLAUDE.md pitfall #5, live in the decode hot loop**
  (`inputs_processor.rs:538`). One `Tensor::from_vec` of the whole `[B, 1]` token
  batch replaces B device allocations and B H2D copies per step. Cheap, contained.
- **Item 3**: `clone_out_cache` runs every step by construction
  (`post_op` is `Out` unconditionally), even when `last_completion_ids ==
  current_completion_ids` and the next step will reuse the batched cache in
  place. The `In` side is already guarded that way; the `Out` side is not.
- **Item 6**: `get_mut_group!`/`get_mut_arcmutex!` are **spin loops**
  (`try_lock` + `std::thread::yield_now`). They are the reason any lock
  contention shows up as "100 % of one core, GPU idle" rather than as a blocked
  thread. Worth knowing before diagnosing the next CPU-pegged box.
- **Item 9**: batched decode has *never* used the GPU sampler. Not a regression,
  but it means every sampler optimisation to date only ever applied to B=1.
