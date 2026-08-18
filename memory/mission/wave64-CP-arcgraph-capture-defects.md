# wave64-CP — ArcGraph capture: defect register

Parent system: ArcInfer / ArcGraph, with routed items in ArcKV and ArcQuant.

Everything below is **read from source**, not run. No hardware leg was possible in
this session: `arc-graph-probe`'s lock was held by `arcflash-rung1` (claimed
00:14:36Z, compute-apps empty — a chain mid-load), and `arc-prefill-curve` carried
a live 130 464 MiB compute app (pid 344231) with **no lock file at all** — the
lock lying in the second direction, exactly as `GPU_ACCESS_RULE` warns. Neither
box was touched.

The primary finding (block-table staging stride) is fixed on this branch and
pre-registered in `arc-tools/prereg/arcgraph_blocktable_stride.md`. This file
records what was found around it and NOT folded into that fix.

---

## Corrections to the received brief (all four confirmed by the coordinator)

1. **Two capture machineries, not one.**
   - V4 host heap corruption: `arc-cuda-graph/src/graph.rs` (`CudaGraphRunner`),
     mode **RELAXED**, capturing the whole *candle* forward, driven from
     `mistralrs-core/src/pipeline/normal.rs:1679`.
   - Qwen `reshape_and_cache_kernel.cu:140`: `arc-cuda-graph/src/dedicated.rs:830`
     (`DedicatedDecodePath`), mode **already THREAD_LOCAL**, capturing the
     raw-FFI `decode_forward`.
   The standing suspect-list item "try THREAD_LOCAL instead of RELAXED" is
   therefore already answered for the path that actually reproduces.

2. **`reshape_and_cache_kernel.cu:140` is an innocent bystander.** It is
   `CUDA_CHECK(cudaGetLastError())` — a sticky, thread-wide check — and it is the
   **only** error check anywhere in the captured `decode_forward`. It reports
   whatever error any earlier unchecked launch latched. Same shape as the moving
   glibc abort site; do not anchor on the filename.

3. **`CUDA_CHECK` calls `exit(err)`, not `abort()`** (`pagedattention.cuh:46-52`
   and five sibling copies). `exit()` from one thread of a live multithreaded CUDA
   process runs atexit handlers and stdio cleanup while other threads keep
   allocating. That is a candidate mechanism for the **glibc diagnostic itself**
   on any path that trips a `CUDA_CHECK`, independent of whatever corrupted state
   caused the CUDA error.

4. **`ARC_NO_DEDICATED_DECODE` is confounded and its inference does not carry.**
   On the Qwen repro the flag is not "a decode-path flag suppressing a
   capture-time crash" at all: the capture *lives inside* the dedicated decode
   path (`dedicated.rs:830`), so the flag simply turns the capture off. Nothing
   surprising to explain. On V4 the flag is confounded a second way — the
   extraction it gates retains a large dequantised BF16 weight set, so setting it
   changes VRAM occupancy, allocator fragmentation, and host-heap allocation
   history. **A flag that moves the heap layout can mask a heap-corruption
   diagnostic without touching its cause.**

---

## Registered defect 1 — `ensure_buffers` pins the batch size forever

`arc-cuda-graph/src/dedicated.rs:430-433`

```
fn ensure_buffers(&mut self, batch_size: usize) -> candle_core::Result<()> {
    if self.buffers.is_some() {
        return Ok(());
    }
```

Activation buffers are sized from the **first** batch size ever seen and never
resized or re-checked. `decode_forward` then reads `buffers.batch_size`
(`decode_forward.rs`, `let bs = buffers.batch_size as u64`), not the caller's
batch size, for every kernel's row count.

Consequences, both live:
- **buffers pinned high, later step low** (e.g. pinned at 8, step at 1): the
  kernels process 8 rows while `stage_paged_attn` only refreshed 1 row of block
  tables / context lens / slot mappings. Rows 1..7 carry the *previous* step's
  metadata, or uninitialised memory on the first such step ⇒ wild physical block
  indices ⇒ device-side OOB read. **Not capture-dependent** — eager decode does
  it too, which is a plausible source of the `Xid 31` events that are present but
  unattributable to any capture leg.
- **buffers pinned low, later step high**: silently decodes only the first
  `buffers.batch_size` sequences of the batch. Wrong tokens, no error.

Note this is *independent* of the stride fix: the stride fix makes the staged
rows land where the kernel reads them; this defect is about how many rows are
refreshed at all.

## Registered defect 2 — `fuse_qkv` ignores `WeightPtr::cols`

`arc-cuda-graph/src/dedicated.rs:345-420`, with `WeightPtr` at
`arc-cuda-graph/src/weights.rs:12-16`.

`WeightPtr` carries `{ptr, rows, cols}`. `fuse_qkv` computes every copy length as
`rows * weights.config.hidden_size * 2` and **never reads `cols`**. That is only
correct when every projection's in-features equals `hidden_size`.

It is not correct for MLA (`q_a_proj` is `[q_lora_rank, hidden]`, `q_b_proj` is
`[heads*qk_head_dim, q_lora_rank]`) nor for MoE expert projections. Where
`cols < hidden_size` the D2D copy **over-reads past the end of the source device
buffer**; `cudaMemcpy` does not bounds-check device pointers, so it silently
copies adjacent VRAM whenever that VRAM happens to be mapped. Silent garbage
weights, no error.

This compounds with the positional indexing in
`extract_model_weights` (`weights.rs:242-257`), which assumes exactly 7
projections per layer in the order `q,k,v,o,gate,up,down` and indexes
`layers[1 + i*7 + k]` with no inventory check. For V4 (MLA + MoE + MTP entries)
that index selects unrelated tensors. `check_dense_layer_inventory` on the
unmerged `arcgraph/capture-truth` branch converts this from luck into a contract;
**it is not on master**, so master's V4 extraction is still silently mis-indexed.
That is also why "the architecture guard refuses V4" is true of one binary and
false of another — the two-binaries-fail-differently observation.

## Routed to the candle fork — `clone_htod` is the unfixed half of a fixed bug

`candle-core/src/cuda_backend/device.rs` (fork rev `d2d1d077`, the rev Cargo.lock
pins).

`htod_info` leaks its host source while `capture_mode()` is set, with a comment
stating the reason exactly:

> "a captured `cuMemcpyHtoDAsync` records the host POINTER and re-reads it on
> every (re)launch — a freed Vec yields garbage dims → the reduce kernel indexes
> out of bounds → MMU fault"

```
let leaked: &'static [T] = Vec::leak(src.to_vec());
self.stream.memcpy_htod(leaked, &mut dst).w()?;
```

`clone_htod`, immediately above it, was given the *device-side* half of the same
fix (route the destination through the caching allocator for a stable address)
but **not** the host-side half — it passes the caller's `src` straight to
`memcpy_htod`. `clone_htod` is what backs `storage_from_cpu_storage` /
`storage_from_slice`, i.e. **every `Tensor::from_vec` / `Tensor::new` on a CUDA
device** (`device.rs:754-908`). Any such call inside a captured forward records a
memcpy node pointing at a Vec that drops when the op returns.

CLAUDE.md pitfall #5 ("Never use `Tensor::{from_vec,arange}` in hot loops")
establishes that such calls do occur in forwards. The fix is symmetric with
`htod_info` and is one branch.

**Scope, stated honestly:** this is a *read* of freed host memory. It produces
garbage kernel arguments and thus MMU faults / `Xid 31` /
`CUDA_ERROR_ILLEGAL_ADDRESS` — the signal the brief kept separate. It does **not**
explain `corrupted double-linked list` / `malloc_consolidate()`, which require a
*write* into the host heap. That signal remains unexplained; see below.

## Still unexplained — the host heap corruption

A captured graph can only write host memory through a **D2H memcpy node**. Ruled
out for the V4 path by argument: candle's only D2H route is `to_cpu_storage` →
`clone_dtoh`, which synchronises; a synchronise on a capturing stream errors, and
`normal.rs:1712-1718` has an explicit handler for exactly that
("forward errored DURING capture (likely a host sync)") which did **not** fire in
the failing log — the log shows `V4 forward RECORDED`, i.e. capture succeeded.

Remaining candidates, none tested:
1. An invalid free reaching libcuda's host-side bookkeeping (which lives in the
   process glibc arena) — the previous chain's pool-lifetime theory is dead on
   this path (`cuMemPoolDestroy` count is 0), but the *class* is not.
2. `exit()` from a `CUDA_CHECK` racing other threads — correction 3 above.
3. A plain host-side buffer overflow in the capture path. `CUmemPoolProps`
   (`arc-cuda-graph/src/ffi.rs:134-142`) was checked field-by-field against
   `CUmemPoolProps_v1` and is byte-correct (88 bytes). `CudaGraphNodeParams` is
   over-sized (4008 vs 256), which is safe for an out-param. Not found.

## ArcQuant / startup — the dedicated-decode extraction tax

`mistralrs-core/src/pipeline/normal.rs:1265-1324` → `weights.rs:221-328` →
`dedicated.rs:345-427`. Runs at model load on every start unless
`ARC_NO_DEDICATED_DECODE=1`.

Cost, derived from the code (numbers pending a box):
- `dequantize_w()` is called `1 + 7 * num_layers` times (lm_head + q,k,v,o,gate,
  up,down per layer), plus 2 extra probe calls at `normal.rs:1278` and `:1283`
  that are thrown away. Each materialises a full BF16 copy of that weight.
- `ModelWeights::anchors` then **retains** `lm_head` + `o,gate,up,down` per layer
  + every residual, for the lifetime of the process — deliberately, to keep the
  raw `u64` pointers valid.
- `fuse_qkv` `cudaMalloc`s and D2D-copies a further
  `Σ_layers (rows_q+rows_k+rows_v) * hidden * 2` bytes on top.

So steady-state retention is approximately **the whole model a second time, in
BF16**. Against a QTIP-quantised model that is a large multiple of the served
weights, which is why the comment at `normal.rs:1258-1263` reports it OOMing on
V4 — and the OOM happens *after* the CPU time is spent and after the allocator has
been fragmented by the failed allocations.

Worse, on V4 the result is discarded twice over: the extraction either OOMs or is
refused by the guard, and even when it succeeds V4 routes through
`CacheBackendMetadata::DefaultInstructions`, which never calls
`try_dedicated_decode` at all (`pipeline/mod.rs:880` marks it unreachable). The
path is paid for on every start and can never be used by that model.

Fix shape (not implemented here, needs the measurement first): decide whether the
dedicated path is reachable for this architecture **before** extracting, not
after. The reachability facts — paged-attention support and the dense-layer
inventory — are both known at load time.
