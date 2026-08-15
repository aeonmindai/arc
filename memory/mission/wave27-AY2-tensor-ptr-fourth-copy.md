# Wave 27 — AY2: the fourth copy of the `as_cuda_slice::<u8>` defect

Follow-up to PR #52 (`memory/mission/wave27-AY-decode-serialization.md`).
PR #52 fixed three copies. The coordinator found a fourth still on master by
grepping rather than assuming the remaining hits were benign. This wave fixes
it and inventories every remaining site in the workspace.

---

## 1. The defect, restated

`CudaStorage::as_cuda_slice::<T>()` type-checks `T` against the storage's
**runtime** dtype and returns `Error::UnexpectedDType` otherwise
(candle `candle-core/src/cuda_backend/mod.rs:1295-1304`). A generic
"give me the device pointer for an arbitrary `&Tensor`" helper that hardcodes
`::<u8>()` therefore fails on **every tensor that is not U8**.

Correct reference implementation, already in the crate:
`arc-cuda-graph/src/weights.rs:128` `tensor_device_ptr`, which matches on
`tensor.dtype()` and picks the matching `T`. Its own comment says "cudarc
requires type match".

## 2. Copy four — `arc-cuda-graph/src/autonomous.rs:116`

```rust
fn tensor_ptr(t: &Tensor) -> candle_core::Result<usize> {
    let t = t.contiguous()?;
    let (storage, layout) = t.storage_and_layout();
    match &*storage {
        candle_core::Storage::Cuda(cuda_storage) => {
            let slice = cuda_storage.as_cuda_slice::<u8>()?;   // <-- same bug
```

**All 28 call sites in the file fail.** Every buffer `tensor_ptr` is handed is
U32, I64 or BF16 — never U8:

| buffer | dtype | declared at |
|---|---|---|
| `input_ids`, `positions`, `block_tables`, `context_lens` | U32 | `arc-cuda-graph/src/buffers.rs:24-27` |
| `slot_mappings` | I64 | `buffers.rs:28` |
| `sampled_tokens`, `n_generated`, `output_tokens`, `finished`, `loop_condition` | I64 | `buffers.rs:48-52` |
| `logits_buf` | BF16 | `autonomous.rs` (`AutonomousDecodeRunner::new`) |

Call sites: `autonomous.rs:520`, `:523`, `:539-540`, `:565-572`, `:580`,
`:590`, `:599-600`, `:772`, `:780`, `:783`, `:791`, `:794`, `:810`, `:817`,
`:824`, `:829`, `:834`, `:839`. Not one could ever have returned a valid
pointer. The GPU-autonomous decode runner has never taken a single pointer
successfully.

### Why this one is the interesting one

This is the **CUDA-graph autonomous decode path** — a headline roadmap item
(`memory/project_cuda_graph_plan.md`: capture-once → fused sampling → GPU
WHILE loop, targeting ~0 µs/step host overhead). The file's own doc comment
claims "Zero host overhead per token" and "better than vLLM's ~10 µs".

**It is dead twice over:**

1. **Gated off before it runs.** `mistralrs-core/src/pipeline/normal.rs:1841-1847`
   returns `Ok(None)` when `metadata.cache_config` is `None`. On V4 that is
   always: `DeepSeekV4Loader::supports_paged_attention` → `Ok(false)`
   (`mistralrs-core/src/pipeline/loaders/normal_loaders.rs:3231-3237`) →
   `normal.rs:345-347` nulls `paged_attn_config` → no `cache_config`. And the
   bail is `tracing::debug!`, so nothing in a normal server log says so.
2. **Broken if the gate were lifted**, by this dtype bug, at the first
   pointer extraction.

**A path that never executes never reports its own breakage.** That is the
whole explanation for how four copies of one bug survived review, CI, and a
GPU session: three of them sat behind guards that were never satisfied, and
the fourth (the radix sampler) only announced itself because it had an
explicit `tracing::warn!` fallback arm. Without that one warn line, the
family would still be invisible.

The corollary for the roadmap: **"implemented" and "executes" are different
claims**, and only the second one is worth anything. Nothing in the repo
currently distinguishes them for `autonomous.rs`.

## 3. The fix

`autonomous.rs:116` now delegates to `crate::weights::tensor_device_ptr`,
matching the three sites fixed in #52. The unused `DevicePtr` import is
removed. No fifth copy is introduced.

**Regression test** — `tensor_ptr_accepts_every_decode_buffer_dtype`
(`arc-cuda-graph/src/autonomous.rs`, `#[cfg(all(test, feature = "cuda"))]`).
It targets this entry point directly:

- calls `tensor_ptr` on U32, I64 and BF16 tensors — the exact dtype set the
  decode buffers use. Without the fix it fails on the **first** call with
  `unexpected dtype, expected: U8, got: U32`.
- asserts the byte offset of a narrowed row is scaled by the tensor's own
  dtype width (`8 * size_of::<i64>()`), which pins the second half of the
  contract — the old code computed the offset from `t.dtype()` but read the
  base pointer as `u8`, so the two halves disagreed.

It no-ops when `Device::new_cuda(0)` is unavailable, so it is a GPU-session
test like `radix_topk_rows_f32_accepts_f32_scores` from #52.

## 4. Full workspace inventory of `as_cuda_slice::<`

`git grep -n 'as_cuda_slice::<' -- '*.rs'` → ~280 hits across 30 files.
The decision rule applied to every one: **a site is a BUG only if `T` is
hardcoded while the tensor's dtype at that point can vary.** A site is
LEGITIMATE when the dtype is already pinned — by an enclosing
`match x.dtype()` arm, by a `<T: WithDType>` generic, by `map_dtype!`-style
macro dispatch inside a candle `CustomOp1/2/3` impl, by a preceding
check/bail or `to_dtype()`, or by a documented struct-field invariant
(packed U8 blocks, F32 scales).

| file | sites | verdict |
|---|---|---|
| `arc-cuda-graph/src/autonomous.rs` | 1 | **BUG — fixed here** (copy 4) |
| `arc-cuda-graph/src/flashmlasparse.rs` | 1 call + 2 comments | fixed in #52 |
| `arc-cuda-graph/src/sampling_cuda.rs` | 1 comment | fixed in #52 |
| `arc-cuda-graph/src/weights.rs` | 7 | legitimate — each is an arm of the `match tensor.dtype()` in `tensor_device_ptr`; the two `::<u8>` at `:165`/`:171` are the genuine U8 and F8E4M3 arms (F8E4M3 is stored as u8 in cudarc) |
| `mistralrs-core/src/cuda/{gdn,moe,sinkhorn,ssm}.rs` | 32 | legitimate — `match dtype`→`cuda_fwd::<T>` dispatch; the hardcoded-f32 recurrence helpers in `gdn.rs`/`ssm.rs` have a documented f32-only contract their sole callers honour with `to_dtype(F32)` (`models/gdn.rs:579-611`, `models/granite.rs:1046-1063`) |
| `mistralrs-paged-attn/src/cuda/backend/cache.rs` | 1 | **BUG — fixed here** (copy 5) |
| `mistralrs-paged-attn/src/cuda/backend/*.rs` (rest) | 65 | legitimate — `match q.dtype()`→`cuda_fwd_t::<T>`; the `::<u8>` sites are TurboQuant packed caches, `::<f32>` are alibi/scale scalars |
| `mistralrs-quant/src/**` (17 files) | 203 | legitimate — candle `CustomOp` `match dtype` arms, `map_dtype!`-style macro dispatch, explicit `if x.dtype() != ... { bail }` guards, or asserted format invariants (packed U8 blocks, F32 scales) |

### A fifth site — and a correction to my own first answer

**I initially published "zero further bugs" and it was wrong.** Recording the
error, because how it happened is the point of this whole wave.

My argument was structural: the antipattern needs a *dtype-erased pointer
helper*, and `git grep -n 'fn .*ptr.*(.*&Tensor' -- '*.rs'` returns no hits
outside `arc-cuda-graph/`. Two supporting observations were and remain true —
the shared `slice_ptr<T: DeviceRepr>`
(`mistralrs-quant/src/utils/mod.rs:43`,
`mistralrs-paged-attn/src/cuda/backend/mod.rs:22`) is generic over `T` and
adds no assumption of its own; and upstream call sites overwhelmingly pin the
dtype first (`mistralrs-core/src/cuda/gdn.rs:34` is an f32-only entry point;
`mistralrs-quant/src/qtip/cuda_ops.rs:90-98` relies on struct invariants that
are *explicitly asserted* at `cuda_ops.rs:1346-1354`).

**But the premise was wrong.** The defect does not need a named helper. It
needs only a hardcoded `T` at a site where the dtype varies — and that can sit
inline in any function. A grep for `fn .*ptr` cannot see it. A concurrent
per-site audit found exactly that, at:

**`mistralrs-paged-attn/src/cuda/backend/cache.rs:247-251`** — in
`swap_blocks`'s `(Device::Cpu, Device::Cuda)` arm:

```rust
let (dst_ptr, _guard_dst) = slice_ptr(
    dst_storage.as_cuda_slice::<u8>()?,   // always fails
    dst_layout.start_offset(),
);
let src_slice = src_storage.as_slice::<u8>()?;   // CPU twin, same dtype check
```

`swap_blocks` takes bare `Tensor`/`&Tensor` with no dtype parameter and no
guard. The **sibling `(Cuda, Cuda)` arm in the same function**
(`cache.rs:196-215`) pins the intended domain: it matches
`CudaStorageSlice::{BF16, F16, F32}` and bails with *"only f32, f16 and bf16
input data type supported!"*. U8 is not in that set, so the CPU→CUDA path
returns `UnexpectedDType { expected: U8, got: BF16 }` before copying a byte.
Both sides are broken — `CpuStorage::as_slice::<u8>()` has the identical
check.

Same root cause as the four in `arc-cuda-graph`: the author reached for `u8`
because the copy is byte-oriented (the comment at `cache.rs:221` says "u8s
because we copy by bytes"), but candle **type-checks the view rather than
reinterpreting it**. That is the one-sentence statement of the whole bug
family, and it is worth more than my structural argument was.

**Status:** pre-existing upstream mistral.rs code, not Arc code, and currently
**dead** — `git grep swap_blocks -- '*.rs'` finds the definition
(`cache.rs:176`) and two re-exports (`backend/mod.rs:8`, `cuda/mod.rs:9`), no
caller. Zero production impact today; it would fire the moment CPU KV-cache
offload/swap is wired up. Fixed here anyway, since a latent broken path that
nothing exercises is precisely the debt this wave exists to clear.

**The fix** mirrors the sibling arm's verified shape rather than inventing
one: match `&*dst_storage.slice` on BF16/F16/F32 and call `slice_ptr` with the
correctly-typed slice, match `src_storage` on the same three `CpuStorage`
variants, and cast to bytes afterwards via a small `as_byte_slice` helper.
Note the offset semantics this corrects: `slice_ptr(v, lo)` does `v.slice(lo..)`,
so `lo` is in **elements of `T`**. Passing `dst_layout.start_offset()` (an
element count) alongside a `CudaSlice<u8>` conflated elements with bytes; with
the slice correctly typed the two agree, exactly as they already do in the
`(Cuda, Cuda)` arm.

**The lesson, corrected:** the right standing check is not "find the dtype-erased
helpers" — it is `git grep -n 'as_cuda_slice::<'` (plus `as_slice::<`) and a
verdict on *every* site. A structural shortcut that reasons about where a bug
*could* live will miss the instance that lives somewhere else. I reached for
the shortcut because 300 sites looked expensive; the shortcut was wrong, and
the exhaustive pass was what found the fifth.

So: **five copies, not four.** Four in Arc-authored `arc-cuda-graph` generic
pointer helpers (#52 ×3, #53 ×1) and one inline in upstream paged-attn (#53).
The two `::<u8>` arms in `weights.rs:165`/`:171` are the genuine U8 and F8E4M3
arms of the correct dispatch, not copies of the bug.

## 5. The lesson, recorded in BACKLOG as instance seven

> **When you fix a defect in a copy-pasted helper, grep for every copy before
> closing.** Three of the four copies here were found only because someone
> kept looking *after* the "fix" was merged.

Standing check: `git grep -n 'as_cuda_slice::<'` and confirm every site either
dispatches on `t.dtype()` or sits under a pinned dtype.

---

## 6. Surfaced, not shipped

> **Noticed:** `tensor_ptr` / `cuda_tensor_ptr` return a `usize` derived from
> `t.contiguous()?`, a **local temporary**. If the input is not already
> contiguous, the copy's storage is freed at function exit and the returned
> pointer dangles. Every current caller passes an already-contiguous tensor,
> which is why it has never fired. One `bail!` on non-contiguous input makes
> it impossible. Same note as #52 — still unfixed, now in four places' worth
> of call sites.
> Worth a separate change?

> **Noticed:** three helpers take a bare `&Tensor` and hardcode `::<f32>()`
> with **no runtime dtype bail** — `mistralrs-core/src/cuda/gdn.rs:34,41,48,55,62,69`
> and `:147,154,161,168,175,182`, and `mistralrs-core/src/cuda/ssm.rs:64,83`.
> They are correct *today* only because every caller converts with
> `to_dtype(F32)` first. Same shape as the bug family, held together by caller
> discipline. `mistralrs-core/src/cuda/sinkhorn.rs:33-38` does the identical
> job but *does* bail on a dtype mismatch — that is the cheap hardening
> pattern if these should fail loudly rather than silently depend on callers.
> Worth a separate change?

> **Noticed:** `swap_blocks`'s `(Cpu, Cuda)` arm binds `_src_layout` and then
> ignores it — the src offset is computed as `src_block_number *
> block_size_in_bytes` with no `start_offset()` term, so a *sliced* CPU source
> tensor would be read from the wrong address. Pre-existing, orthogonal to the
> dtype bug, and equally dead today. Left alone.
> Worth a separate change?

> **Noticed:** nothing in the repo distinguishes "implemented" from
> "executes". `autonomous.rs` is a headline roadmap item that has never run a
> line in production, and its disable is a `tracing::debug!`. A startup INFO
> line naming which optional fast paths are ACTIVE vs BYPASSED — autonomous
> decode, paged attention, the fused MoE gather, the indexer CUDA kernel —
> would have caught all four copies of this bug, and #52's 8-token MoE cap,
> on the first server boot.
> Worth a separate change?
