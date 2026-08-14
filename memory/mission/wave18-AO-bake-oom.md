# wave18-AO — the V4-Flash bake OOM

Branch `fix/bake-oom` off `origin/master` (809643552). No GPU used; every claim
below is either read off the source, measured by a CPU test in this branch, or
arithmetic from the two.

---

## 1. What actually happened

Bake: `mistralrs quantize text -m <V4-Flash> -a deepseekv4 --isq qtip2 -o <dir>/`
with `ARC_QTIP_BEAM=256 MISTRALRS_ISQ_SINGLETHREAD=1`, on a 140 GB H200.

| layer | GPU mem | slope since previous sample |
|---|---|---|
| 7 | 21.5 GB | — |
| 14 | 35.0 GB | 1.93 GB/layer |
| 22 | 48.7 GB | 1.71 GB/layer |
| 24 | 57.6 GB | **4.45 GB/layer** |
| 28 | OOM (device is 140 GB) | — |

Per-layer time flat at 223.7–225.7 s for all 28 layers. Host RAM 33/235 GB.
`dmesg` clean. Output directory 4 KB.

## 2. Root cause — two mechanisms, one shape

### 2a. Retention: the bake keeps the whole artifact on the GPU

`FusedExperts::new` (`mistralrs-quant/src/distributed/layers.rs`, INT4-packed
branch ~line 2000) picks

```rust
let target_device = experts_vb.device().clone();   // = the mapped CUDA device
```

and hands it to `apply_isq`. For a 3-D expert stack that lands in
`QtipLayer::quantize_3d_with_bake_config`, where

```rust
let move_back = matches!(quant_device, Device::Cuda(_)) && matches!(device, Device::Cpu);
```

is **false** (device is CUDA), so the packed result is left on the device.

That is correct for a serve. For a **bake** it is pure cost: the model is
constructed only so `QuantizedSerde::serialize` can be called on it, and no
forward pass ever runs.

Arithmetic, V4-Flash (E=256, hidden 4096, moe_intermediate 2048, 43 layers):

- gate/up packed: `256 × 2048 × (4096/4) B` = 537 MB each
- down packed: `256 × 4096 × (2048/4) B` = 537 MB
- **1.61 GB retained per layer**, 69.3 GB over 43 layers — which is the 68 GB
  artifact size, and matches the measured 1.7–1.9 GB/layer baseline growth.

So the measured baseline slope *is* the artifact accumulating on the card.

### 2b. Fragmentation: a 4.3 GiB transient the code never meant to allocate

`quantize_3d_with_bake_config` streams experts in batches of 16
(`ARC_QTIP_EXPERT_BATCH`):

```rust
let chunk   = weight.narrow(0, expert_idx, this_b)?;   // view
let rows_2d = chunk.reshape((this_b * n, k_in))?;      // still a view
… quantize_with_options_cuda(&rows_2d, …)
    → let weight_cuda_f32 = weight.to_device(device)?.to_dtype(DType::F32)?;
```

Candle's `to_device` (verified in the pinned fork,
`candle-core/src/tensor.rs:2368`) copies the **entire backing storage** and
clones the layout, offset included:

```rust
(Storage::Cpu(storage), Device::Cuda(cuda)) => Storage::Cuda(cuda.storage_from_cpu_storage(storage)?),
…
layout: self.layout.clone(),
```

A `narrow`+`reshape` view names 16 experts but still owns all 256. So every
chunk uploaded the **whole 4.295 GiB BF16 stack** instead of its 268 MiB slice —
16 chunks × 3 projections = **48 alloc/free cycles of a 4.3 GiB block per
layer**, on top of 48 cycles of the ~6 GiB beam trace scratch
(`rows_in_flight × num_symbols × width × 4`, capped by the 6 GiB
`ARC_VITERBI_SCRATCH_GB` budget).

Large transients cycling around a resident set that grows by 537 MB blocks is
the textbook shape for `cuMemAllocAsync` pool fragmentation, and it explains the
part the retention alone does not: the constant term crept from ~10 GB (layer 7)
to ~19 GB (layer 24) and then the slope broke from 1.7 to 4.45 GB/layer. Extend
4.45→~20 GB/layer over layers 24→28 and you reach 140 GB, which is where it
died. `trim_cuda_pools_after_isq()` already runs twice per layer and could not
help: `cuMemPoolTrimTo` only releases *fully free* physical chunks, and the
retained packed blocks pin them.

**Verdict: retention is the primary cause and it is real (measured 1.7–1.9
GB/layer ≡ 1.61 GB/layer of packed weights). Fragmentation is the accelerant
that turned "should have finished at ~80 GB" into "OOM at layer 28".** Both are
cured by the same change, because with nothing permanent being interleaved the
pool reaches steady state after layer 1.

### Beam vs exhaustive — not the cause

Session 3 completed a bake with exhaustive search; tonight used beam. Scratch
sizing is nearly identical between them:

- beam: `trace_bytes_per_row = num_symbols × width × 4` = 2 MiB (gate/up),
  `rows_in_flight = 6 GiB / 2 MiB = 3072` → one **6 GiB** alloc.
- exhaustive: `per_row = num_symbols × 4096 + 2 × 65536 × 4` = 8.5 MiB,
  `rows_in_flight = 723` → 6.06 GiB + 2 × 189 MB = **6.4 GiB** in three allocs.

Beam allocates one block where exhaustive allocates three, both ~6 GiB. That is
not an 80 GB difference. What changed between session 3 and tonight is more
likely the 4.3 GiB per-chunk upload (RUN-161 expert-batching) interacting with a
larger card and different pool state. **Beam is exonerated as the cause; it is
not exonerated as a contributor to fragmentation, and neither is exhaustive.**

## 3. The fix

### 3a. Bake output goes to the host (`set_bake_isq_to_host`)

New process-global switch in `mistralrs-quant/src/lib.rs`. Set by the loaders
(`normal.rs`, `vision.rs`, `embedding.rs`) when
`write_uqff.is_some() && from_uqff.is_none() && !has_registered_hooks()`.

In both QTIP rungs' 3-D expert path (`qtip/mod.rs`, `qtip/bitshift.rs`):

```rust
let out_device = if crate::bake_isq_to_host() { Device::Cpu } else { device.clone() };
let move_back  = !quant_device.same_device(&out_device);
```

The quantize still runs on `quant_device` (the mapped GPU) — Arc's moat is
untouched. Only the packed result comes back. This reuses the `move_back` path
that already existed for CPU-target stacks, so it is not a new code path.

Excluded when a post-load hook is registered: arc-engine's TD-MoE compressor
rewrites the quantized layers in place after the load and expects them where the
model was mapped.

Not thread-local (unlike `LOADING_FROM_UQFF`): `apply_immediate_isq_always` can
run a layer's quantize on the immediate-ISQ rayon pool, and a thread-local would
silently read `false` there — a half-applied fix is the worst outcome here.
Pinned by `bake_to_host_flag_is_visible_from_another_thread`.

### 3b. Per-chunk upload carries only its chunk

```rust
let rows_2d = if this_b < e { rows_2d.force_contiguous()? } else { rows_2d };
```

`force_contiguous` allocates a fresh storage sized to the layout and copies the
window into it, so `to_device` ships 268 MiB instead of 4.295 GiB. Applied
unconditionally for partial chunks (not only when a device hop is pending) so a
CPU test can reach it; the copy is one chunk's worth of host bandwidth.

### 3c. Fail-fast budget (`mistralrs-quant/src/utils/bake_budget.rs`)

Armed by the loader with the model's layer count; `note_bake_layer()` is called
once per fused MoE layer from `FusedExperts::new`. Per layer it samples
`cuMemGetInfo` and projects:

```
growth = (used_now − used_4_layers_ago) / 4          # SLOPE_WINDOW = 4
peak   = used_now + (total_layers − layers_done) × growth
budget = device_total × (1 − headroom)               # 8%, ARC_BAKE_HEADROOM
```

Stops the bake only after **5 consecutive** over-budget projections
(`CONSECUTIVE_TRIPS = SLOPE_WINDOW + 1`), then bails with every term printed
plus remediations. The debounce is the point: a single unlucky layer, where the
pool happens to grab a whole scratch block, lifts a windowed slope for exactly
`SLOPE_WINDOW` samples and then falls out of the window, so it can never reach
the count. Sustained growth can. Killing a healthy two-hour bake on allocator
noise would be worse than the failure the guard exists to prevent.

It measures **device usage**, not the bytes we produced, so it sees pool caching
and fragmentation — the exact thing a tensor-size accounting would miss.
Overestimating `total_layers` (dense prefix layers never reach `FusedExperts`)
only makes it conservative, and it only bites when usage is actually growing.

**What it would and would not have caught — stated honestly.**

- The *pre-fix retention shape* on an 80 GB H100 (1.61 GiB/layer from layer 1):
  fires at **layer 6 of 43**, ~22 minutes in, instead of dying at ~layer 40
  after 2.5 hours. Pinned by
  `steady_retention_growth_is_caught_in_the_first_quarter_of_the_bake`.
- **Tonight's actual 140 GB curve: it would not have saved this run.** The
  steady 1.71 GiB/layer stretch through layer 22 genuinely projects to ~85–95 GB
  on a 140 GB card — it *fits*. The run was killed by a late nonlinearity that no
  early projection could see; by the time the trailing window registered it, the
  card was already near the wall. That is pinned as a test
  (`the_measured_v4_flash_trend_projects_to_fit_on_a_140gb_card`) so nobody later
  assumes otherwise. The fix for this particular run is 3a, not the guard.
- The post-fix shape (flat usage) never even reaches `Watching`
  (`a_flat_bake_never_warns`).

### 3d. Streaming UQFF shard writer

`mistralrs-core/src/pipeline/isq.rs` used to serialize every ISQ tensor into one
`Vec<(String, Vec<u8>)>` and only then split it into ≤10 GB shards — 68 GB of
host buffer on top of the tensors it was copying from, and nothing on disk until
the last tensor finished. Replaced with `UqffShardWriter`, which holds at most
one shard and writes each out as it fills.

This became **required**, not optional: moving 68 GB of quantized tensors to the
host means the old buffered serializer would have needed 68 + 68 GB.

### 3e. PagedAttention off for `quantize`

PagedAttention sizes its KV cache from *free* VRAM. Freeing ~68 GB by 3a would
have made the bake allocate a ~120 GB KV cache immediately after writing the
artifact, for a command that never generates a token. One line in
`mistralrs-cli/src/commands/quantize.rs`.

## 4. Projected peak after the fix

Per-layer device working set (V4-Flash, batch 16, beam W=256):

| item | bytes |
|---|---|
| chunk BF16 upload (16 experts) | 0.27 GiB |
| chunk F32 | 0.54 GiB |
| rotated F32 | 0.54 GiB |
| beam trace scratch | 6.00 GiB |
| packed out + row scales | 0.04 GiB |
| **working set** | **~7.4 GiB** |

Resident across the run:

| item | bytes |
|---|---|
| 2-D ISQ layers (MLA, shared experts, router, lm_head ≈ 7B params @ 2 bit) | ~2 GiB |
| CUDA context, cuBLAS handles, kernels | ~1 GiB |

**Projected peak ≈ 10.5 GiB, flat across all 43 layers** (was ~1.61 GiB/layer of
growth plus a ~12 GiB working set plus fragmentation). 13× margin on a 140 GB
H200; it now also fits an 80 GB H100 with 8× margin, and the per-chunk upload
drop removes 48 × 4 GiB of pool churn per layer.

Host side: 68 GB retained quantized tensors + ≤10 GB shard buffer + ~33 GB
baseline ≈ 111 GB of 235 GB.

## 5. Incremental serialization — assessment

Flushing shards during the *serialize* pass: **done** (3d). Bounds host RAM and
puts shards on disk as they complete.

Making a load-time OOM at layer 28 cost only the tail: **not feasible without a
format/API change, and not attempted.** UQFF tensor names are positional indices
into `IsqModel::get_layers()`, which does not exist until the whole model is
constructed — there is no layer object to serialize mid-load, and no name to
give it. Doing it properly means either a two-pass naming scheme or reworking
`get_layers` into a streaming visitor. Since 3a removes the load-time OOM at the
source, that rework is not on the critical path; flagging it rather than forcing
it.

## 6. Tests (all CPU, no GPU)

`mistralrs-quant/src/qtip/bake_memory_tests.rs` — 6 tests:

- `narrowed_expert_chunk_still_owns_the_whole_stack` — states the bug as an
  assertion: a 2-of-16-expert view has `storage_elems == 16·n·k` and shares the
  parent's allocation address.
- `materialised_chunk_owns_only_its_own_experts` — after `force_contiguous`,
  `storage_elems == 2·n·k`, different address, ratio == 8.
- `materialised_chunk_holds_the_same_experts` — window contents preserved at
  three offsets.
- `chunked_bake_is_bit_identical_to_per_expert_bakes` — 18 experts (chunks of
  16 + 2) baked through the production 3-D door; experts 0/15/16/17 must match
  solo bakes byte for byte, and the serialized artifact must match the
  in-memory bake. **Mutation-tested**: changing the chunk window to
  `narrow(0, 0, this_b)` makes it fail at expert 16 with the diff printed.
- `quantized_stack_retains_no_handle_on_its_source` — the produced layer's
  storages do not alias the BF16 source, and total live bytes are packed-size,
  not source-size.
- `bake_to_host_flag_is_visible_from_another_thread` — the switch is
  process-global, so an ISQ-pool thread sees it.

`mistralrs-quant/src/utils/bake_budget.rs` — 8 tests driving the whole state
machine over 43 samples: a flat bake never warns on either card size; the
pre-fix 1.61 GiB/layer retention shape is stopped at layer 6 of 43 on an 80 GB
card; a one-off 20 GiB allocator jump warns but never stops; sustained growth
stops after exactly `CONSECUTIVE_TRIPS` confirmations; and tonight's measured
trend is recorded as *fitting* on 140 GB, so the guard's limits are documented
in code rather than assumed.

Supporting change: `QtipLayer::quantize_3d_concrete_with_bake_config` returns
the concrete layer (`QuantMethod` does not extend `Any`, so there was previously
no way for a test to see what memory a freshly baked 3-D stack holds); the
`Arc<dyn>` entry point is now a two-line wrapper over it.

## 7. Verification

- `cargo check --workspace --tests` — clean.
- `cargo test -p mistralrs-quant -p mistralrs-core` — 192 + 245 + 11 + 2 pass, 0
  fail.
- `cargo clippy --no-deps -p arc-bench -p arc-engine -p arc-cuda-graph -p arc-cli
  -p mistralrs-quant --tests --examples -- -D warnings` — clean.
- The `#[cfg(any(feature = "metal", feature = "cuda"))]` serializer arm was
  type-checked by temporarily flipping the cfg to `all()` (metal kernels do not
  build in this environment); reverted after.
- `rustfmt --check` run only on the two new files and hand-matched on the edited
  hunks — no upstream file was mass-reformatted (fork policy `fab114fe3`).

## 8. What still needs hardware

- Confirm the flat device-usage curve on a real 43-layer bake. The projection in
  §4 is arithmetic from measured tensor sizes; it has not been observed.
- Confirm the guard stays silent on a healthy bake (it should — growth ≈ 0).
- Confirm bit-identity of the artifact against a pre-fix bake. `force_contiguous`
  is a pure copy and `to_device` on the packed result does not touch values, so
  the bytes should be identical; unverified on GPU.
- Wall-clock: removing 48 × 4.3 GiB of H2D per layer should shave time off the
  223.7 s/layer, but the split between PCIe and kernel in that number is unknown.

## 9. Noticed, not shipped

- `expert_stack_quant_device` hardcodes `Device::new_cuda(0)` when promoting a
  CPU-target expert stack to the GPU. On a multi-GPU bake that sends every
  layer's expert quantize to GPU 0 regardless of the layer's mapped device. Not
  touched here (this branch passes the mapped device through, so the hardcode is
  not reached by the new path), but it is a live landmine for multi-GPU bakes.
- The ~6 GiB beam/Viterbi scratch is allocated and freed per
  `quantize_rows_cuda` call — 48 times per layer, always the same size. A
  device-scoped cached scratch buffer would remove the last big alloc/free cycle
  in the bake. Not needed now that nothing permanent is interleaved with it.
