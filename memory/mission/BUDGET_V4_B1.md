# THE V4 B=1 DECODE BUDGET — where the 98× goes

**This file answers "why is V4 at ~1% of the roofline". It is MEASURED, not estimated.
Do not re-derive any number here. Do not reason about V4 decode speed without reading it.**

Provenance: box `arc-v4-stack` (H200), SHA `05af600e7` (= `origin/master`, PR #120 confirmed
ancestor), `--features "cuda flash-attn"`, model `aeonmind/DeepSeek-V4-Flash-UQFF-qtip2` over
`deepseek-ai/DeepSeek-V4-Flash`. Device-timer selftest PASS, ratio 26.8×. Session 8, 2026-08-18.

---

## The headline

| | ms/step | % |
|---|---|---|
| **Measured step** (uninstrumented `mistralrs bench`, B=1, **15.0 tok/s**) | **66.68** | 100% |
| **GPU executing kernels** | **34.04** | **51%** |
| **GPU idle, waiting on the host** | **32.64** | **49%** |
| memory-bandwidth floor (13B active × 2 bit ÷ 4.8 TB/s) | 0.677 | — |

**66.68 / 0.677 = 98×.** It factors as **~2× (GPU idle) × ~50× (the kernels themselves)**.

Profiler-accounted step wall 70.1 ms agrees with the clean 66.68 within 1.5% — that agreement is
what makes the span budget below trustworthy. `ARC_PROFILE` overhead measured at **5–8%**, not 4×.

Span budget (device stream-elapsed, 320 recorded steps, µs/step): `mla_attn` 27,689 (40.7%) ·
`moe` 23,571 (34.7%) · `mhc_ffn_pre` 7,085 (10.4%) · `mhc_attn_pre` 5,993 (8.8%) ·
`mix_post_*` 2,521 · `lm_head` 281. The `layer` node reconciles exactly: `device_self_ns = 0`,
zero violations, zero misnested, zero unresolved.

---

## THE DIAGNOSIS: op count, not slow kernels

- **9,131 kernel launches per token** (212 per layer). Step count N=189 pinned from three agreeing
  anchors (517,120 B logits D2H × 189; `topk_kernel` 40.1/step = 40 MoE layers; `sinkhorn`
  86.2/step = 2 × 43).
- **11,436 `cuMemAllocAsync` + 11,436 `cuMemFreeAsync` per token — 1.25 allocations per kernel.**
  > ✅ **SOLVED (mostly), 2026-08-18** — branch `run-161-arena-counters` @ `dac70e6`, binary `0f1a6125`:
  > | per token | baseline | arena |
  > |---|---|---|
  > | `cuMemFreeAsync` | 11,406.2 | **0.0 (−100%)** |
  > | `cuMemAllocAsync` | 11,406.2 | **~288/step (interval) — −97.5%** |
  > | layout H2D | 2,361.6 | **6.0 (−99.7%)** |
  > | **total host calls** | **25,622** | **~737 (−97%)** |
  >
  > ### 🔑 A CUMULATIVE AVERAGE CANNOT SHOW A STEADY STATE — use the INTERVAL rate
  > The first-quoted `181.4 → 234.5` allocs/step were **run averages**, and *"a run average over a
  > warming cache is the same trap as a distinct-size count"* — **warm-up allocations stay in it
  > forever, decaying like 1/steps toward a floor they never reach.** The honest steady-state figure
  > is the rate **between consecutive reports**: **287.6/step over steps 20→40, and rising.**
  > Interval extractor **falsified both ways**: reports **0/step** where an interval is genuinely
  > empty (the average would have claimed 2.50/step) and **300/step** for a real ongoing cost.
  > ⇒ **Same family as "distinct count ≠ occurrence count" and "absent ≠ stale": a statistic that
  > mixes history with the present cannot answer a question about the present.**
  >
  > **Bit-identical over all 52 steps** (FNV over raw IEEE logit bits), with `A1 vs F DIFFER` proving
  > the comparison returns the negative and `A1 vs A2 MATCH` proving the harness is deterministic.
  > 8,554 cache hits/step, `accounting OK`, `puts_refused 0`, arena high-water **956 MB**.
  > **`driver_frees` is EXACTLY 0** — with the arena on, every `CudaStorage::drop` becomes a cache
  > put, so `cuMemFreeAsync` is never called. **Half the 1,535 `MEM_ALLOC`/`MEM_FREE` capture gate
  > has no remaining source.**
  >
  > ⚠️ **The allocation side is RISING, not decaying — the chain's own prediction refuted, and it
  > said so.** Cumulative 181.4/step at 20 steps → 234.5/step at 40 ⇒ the **interval** rate over
  > 20→40 was **287.6/step**. Bucket-decay model treated as wrong pending the `--gen-len 3000` run.
  > **Two levers:** coarser bucketing for large buffers (1/2-octave: ~45 → ~12 distinct sizes per
  > growing tensor, costs only memory against ~65 GB headroom); and **graph mode's fixed-capacity KV
  > slot makes shapes step-invariant, removing the growing-size family at source** — the standing
  > prediction for how `driver_allocs == 0` is reached *in the capture window specifically*, and the
  > only part still unmeasured.
  >
  > ### 🔴 GATE FAILS IN EAGER MODE — THE ARENA LEAKS, mechanism named (500-token run)
  > | step | arena MB | buffers | `puts_refused` | `driver_frees`/tok |
  > |---|---|---|---|---|
  > | 100 | 2,098 | 44,150 | 0 | **0.0** |
  > | 200 | 4,527 | 97,716 | 0 | **0.0** |
  > | 300 | 6,878 | 156,102 | 0 | **0.0** |
  > | 400 | **8,192 (CAP)** | 212,305 | 15,822 | 39.6 |
  > | 500 | **8,192 (CAP)** | 261,153 | **119,828** | **239.7** |
  >
  > **Interval `driver_allocs` over steps 400→500 = 1,401/step. `GATE driver_allocs == 0: FAIL`.**
  >
  > **Mechanism:** the arena grows **~24 MB and ~520 buffers per step and never returns any.**
  > 1/8-octave bucketing does **not** collapse the growing-`kv_len` family — it hands each step a
  > fresh set of sizes that are **cached forever and never requested again.**
  > **🔑 THE CAP POLICY IS EXACTLY BACKWARDS: under pressure it refuses NEW puts while retaining the
  > OLD buffers — and the old ones, at sizes that will never be requested again, ARE the dead
  > weight.** Past the cap the arena degrades **below its own purpose**: `cuMemFreeAsync` returns
  > from 0.0 to **239.7/token**.
  > **Fix (neither speculative): EVICTION, not refusal** — when the cap binds, drop stale entries and
  > keep fresh ones; plus **power-of-two bucketing for large buffers** (~45 → ~7 distinct sizes per
  > growing tensor).
  >
  > **What still stands:** `cuMemFreeAsync` **0.0/token** and layout H2D **6.0/token** held **flat
  > across all 500 steps until the cap bound**, `accounting OK` throughout, output bit-identical.
  > ⚠️ **And the earlier 287.6/step reading was itself an early-warmup artefact of a RISING curve** —
  > the reason run-averages were abandoned applies to short interval samples too.
  >
  > **`puts_refused` is the only reason this was visible.** Without it the arena reads healthy —
  > `accounting OK`, hits still ~8,200/step — **while silently reverting to driver frees.** Sixth
  > instance of a metric that must be structurally impossible being the one that carries the signal.
  >
  > **Two defects found in this work by instruments, not inspection:** symmetric bucketing filed
  > buffers under keys **larger** than physical size ⇒ pre-arena buffers handed out short ⇒ prefill
  > `CUDA_ERROR_INVALID_VALUE` (**a short buffer is a wrong-numbers bug by nature; it surfaced as a
  > hard error only by luck**); then that fix returned `0` below 128 B ⇒ every small buffer allocated
  > *and* freed through the driver **while the arena reported itself healthy**. Tell: `driver_frees`
  > non-zero at all — a quantity that is supposed to be structurally impossible.
- **2,818 H2D copies per token averaging 72 bytes** — candle's per-op dims/strides upload
  (`clone_htod`). Not model data.
- Kernel duration **mean 3.73 µs, median 1.18 µs**, p90 4.58 µs.
- **~5,300 of the 9,131 launches (58%) are dtype casts, copies and elementwise ops** at 1–3 µs each
  ⇒ **≥8.0 ms/step, ≥24% of all GPU time**. `cast_bf16_f32` + `cast_f32_bf16` alone are
  **1,571 launches/token**, largely MHC's F32 round trip in `hc_pre`/`hc_post`.
- Corroboration on the clean run: **memory-controller utilization 4%**, board power **204 W of
  700 W**. The GPU is barely touching HBM.

---

## THE SYNCS ARE INVISIBLE TO THE OBVIOUS GREP

**`ALL *Synchronize* = 0.0 calls/step.** Anyone counting `cudaStreamSynchronize` concludes
"no syncs" and is wrong.**

The real synchronisation is **44 `cuMemcpyDtoHAsync_v2` per step at ~109 µs each = 4.81 ms**.
An "async" call taking 109 µs of host time is blocking — pageable-memory staged copy.
**43 per layer + 1 logits.**

Source: `dsv4_kv_fp8.rs:260` `e4m3_codes_cpu` — a deliberate device→host→device round trip
**because candle has no CUDA F8E4M3 cast**. At B=1 it moves exactly 448 F32 = **1792 bytes**.
Trace shows exactly two DtoH sizes: 1792 B × 8144 and 517,120 B × 189. `KvQuantMode` defaults to
`CpuExact`, whose own doc says *"at the price of one device sync per layer."*

### 🔑 Removing the 43 syncs makes it SLOWER — and that is the proof of the diagnosis

Interleaved A/B (`ARC_GPU_ACT_QUANT=1`, i.e. `GpuApprox`):
`A1 66.99 · B1 67.71 · A2 68.05 · B2 70.30 ms/T` — **monotonic 3.3% drift exceeding the arm
difference**, so always interleave A-B-A-B on this box and state the drift.
Engagement proven per D32: `kv_fp8_quant` **73.49 → 134.18 µs/call (+83%)**, `kv_fp8_dequant`
50.09 → 37.54. The sync-free path swaps one blocking copy for **~19 extra elementwise kernels
per layer**.

> **In a machine bottlenecked on op count, a sync is cheaper than the 800 launches it takes to
> avoid one.** ⇒ **`GpuApprox` is NOT the fix and must not ship as one.** The fix is a single
> **fused quantize+dequantize device kernel**: the round trip costs 5.31 ms/step (8.0%) under the
> default, 11.1% under GpuApprox, and should cost **~2 µs**.

## ✅ SOLVED — the fused FP8 kernel shipped, and the ~2 µs prediction landed

Branch `arckv-fp8-fused` (9 commits). Measured on `arc-v4-stack`, private target dir, binary md5
`e15259dc…`, 118 symbol hits verified in the measured binary.

| | before | after |
|---|---|---|
| **blocking D2H per step** | **43.89** | **1.00** |
| launches per step | 9,081.8 | **8,404.5** (−677.3) |
| the FP8 round trip | `cuMemcpyDtoHAsync_v2` **48.56 µs/call host = 2.131 ms/step BLOCKING** | quantize **1.98 µs/call** + dequantize **2.34 µs/call** @ 42.93 calls/step = **0.185 ms/step device** |

The **1,792-byte transfer size is absent from the after-trace entirely.** The one surviving D2H is
the 517,120-byte logits readback — present in both arms, not this subsystem's.
`cuLaunchKernel` −773.5, `cuMemAllocAsync` −758.6.

### 🏆 The parity proof — exhaustive over the ENTIRE input domain
**All 2^32 f32 values** swept against NVIDIA's software reference (what `float8` ports, hence
candle's CPU cast) **and** the sm_90 hardware path: **0 mismatches**, with a `visited` counter
confirming all **4,294,967,296** were actually touched. Negative control catches **123,731,850
(2.88%)**. Full-tensor GPU test passes on codes + side bytes, crosswise reconstruction, and the
verbatim pre-packing implementation.
**D33 discharged on the shipped path itself:** pointing it at the truncating mutant produced
`assertion left == right failed: fused E4M3 codes differ from candle's CPU cast`; revert → passed,
worktree clean. The counter was also validated on a known answer first — it reproduced the recorded
44.09 / 9,131.5 / 11,436 / 0.00-Synchronize before being pointed at new data.

### ⚠️ `--use_fast_math` SILENTLY BREAKS `__f*_rn` INTRINSICS
The chain assumed `__f*_rn` were IEEE. **The PTX showed nvcc 13.1 rewriting them to `.ftz`** under
`--use_fast_math` — flush-to-zero denormals inside a *numeric format conversion*. Replaced with
inline PTX; the audit is now **`grep -c '\.ftz\.f32'` = 0**. Third `--use_fast_math` surprise of the
session (it had also already collapsed the trellis transcendentals). **Check the PTX, not the
intrinsic's name.**

### No end-to-end tok/s, deliberately
The chain **deleted its A/B driver rather than ship a number a biased estimator can't support**
(D36). Per-call kernel timings + launch counts only — which is why this result survives a night
that destroyed three end-to-end measurements.

**ArcGraph unblocked from this side:** zero blocking host work remains in the FP8 path; kernels
launch on `dev.cuda_stream().cu_stream()` so they record into a graph captured on candle's stream;
a warm decode step issues no `cuMemAllocAsync` from this op. **The remaining capture blocker is the
NULL-stream binding + the PagedAttention gate.** `GpuApprox` untouched, still reachable, documented
in-code as not the fix.

---

## WHY CUDA GRAPH REPLAY HAS NEVER LAUNCHED — root cause, finally named

> `arc-cuda-graph/src/graph.rs:71` — `cuda_dev.cuda_stream().cu_stream()` is **NULL**, i.e. CUDA's
> **legacy default stream, on which capture is impossible.** Logged live:
> `CUDA graph: NULL stream, capture disabled`.

Three independent confirmations it is off:
1. `Decode path extraction failed: tensor_device_ptr requires CUDA tensor`
2. `paged_attention` **UNREACHABLE** (`pipeline/mod.rs:1088`, DefaultInstructions) — this takes
   `graph_wrapped_forward`, the dedicated decode path, and graph replay with it
3. `cuda_graph.capture_probe` **UNREACHABLE**

**Prerequisite chain (ordering matters — graphs cannot capture a blocking D2H):**
**(a)** move the engine off the legacy default stream → **(b)** remove the per-layer D2H *as a
fused device kernel, NOT as `GpuApprox`* → **(c)** capture.

---

---

## ✅ THE CPU **CAN** BE TAKEN OUT OF THE LOOP — MEASURED ON THIS H200, 2026-08-18

Jish's instruction was *"kill the CPU out of the loop."* **It is achievable on our own hardware
and the number is in.** Measured at `/root/arc-wt/probe`, interleaved A-C-A-C ×4, bench lock held:

| route | **host calls per decode step** |
|---|---|
| today (graph + sync per step) | **2.0** |
| host-side K-replay | 1.0078 |
| **device-side WHILE conditional node** | **0.0156** at N=128 |

The 0.0156 is **2 calls per *generation*** (1 launch + 1 sync) — **not per step** — so it goes to
zero as N grows. Device-loop overhead is **constant in body size**: +2.90 µs (M=1), +2.39 (M=64),
**+1.98 µs (M=256)** — against a 66.68 ms step that is **0.004%**. Drift 0.1–1.3%.
**Falsifiable control that makes it real:** one instantiated graph run with device-side `target=1`
then `target=N` returned counter 1 then N — proving the **device**, not the host, chose the trip
count. Three redundant device counters (counter/checksum/filler) agreed in every arm.

**Residual host work per step once this lands: ZERO issued calls.** Token streaming is already a
host *poll* of pinned memory (`decode_loop.cu:61-66`, `__threadfence_system`), not work issued to
the GPU.

### What the installed toolkit actually supports (quoted, not remembered)
`/usr/local/cuda-13.1/include/` · H200 **cc 9.0** · driver 580.173.02 · `nvcc` V13.1.115 ·
`cuda.h:264 CUDA_VERSION 13010`.
- **WHILE conditional nodes**: `cuda.h:1951 CU_GRAPH_COND_TYPE_WHILE = 1` + device-side
  `cuda_device_runtime_api.h:480 cudaGraphSetConditional` — **measured working**
- body population via `cuda.h:15665 cuStreamBeginCaptureToGraph` (`cuda.h:1976` names it the
  supported way)
- **device-side graph launch**: `cuda_device_runtime_api.h:290 cudaGraphLaunch` +
  `cuda.h:5100 CUDA_GRAPH_INSTANTIATE_FLAG_DEVICE_LAUNCH = 4` — instantiate returned SUPPORTED,
  **but the header bars combining it with `AUTO_FREE_ON_LAUNCH`, which `graph.rs:304` uses today**

### 🔒 HARD DEPENDENCY, confirmed by header not guessed
`cuda.h:1971` permits only *"kernel nodes, empty nodes, child graphs, memsets, memcopies, and
conditionals"* inside a conditional body — **no alloc/free nodes.** ⇒ **the autonomous loop cannot
close until the 11,436 allocations per token are gone.** The workspace-arena chain is the gate.

### Bugs fixed to get there (compile-verified, D33 negative control: injected type error → exit 101)
- **`cuGraphAddNode` declared with 5 args and the first two transposed** against `cuda.h:21829`'s
  6, whose first parameter is an out-pointer — **the driver wrote through a `CUgraph` it thought
  was a `CUgraphNode*`.** ⇒ **this is the cause of the previously unexplained "host heap
  corruption" capture failure already on our record.**
- conditional body was captured to a throwaway graph and **destroyed**, leaving the body empty
- missing `CU_GRAPH_COND_ASSIGN_DEFAULT`; missing `ctx` field
- `DecodeState` allocated **I64** while every kernel signature is `int32_t*`
- `reset()` reallocated buffers, **silently invalidating the captured graph's baked pointers**
- `tensor_device_ptr` had **no I32 arm** ⇒ see the sampling defect below

### ✅✅ THE GATE IS OPEN — 2026-08-18, branch `wt/pagedgate` @ `b137f1e0`, 3 files

| marker | control (`ARC_NO_AUTONOMOUS_DECODE=1`) | gate open |
|---|---|---|
| `cuda_graph.autonomous_decode` | **ABSENT** | **FIRED, calls=22** |
| `cuda_graph.capture_probe` | **UNREACHABLE, calls=0** | **FIRED, calls=22** |

Three independent counters agree **live, not at teardown**: `engine_attempts=25 == pipeline_entered=25
== capture_probe_entered=25`; profiler node calls=22 == decode `forward` calls=22. Binary md5-frozen
(`bff2a901…`), symbol-gated before each run, private `CARGO_TARGET_DIR`, builds under `host.lock -s`.
**Cost of the now-reachable attempt: 780 ns against a 66,680 µs step.** D33: poison injected into the
new code → `RC=101` naming it; restored → clean. Unit tests 5/5 incl. a paged-vs-contiguous negative
control. Xid delta 0.

**VERDICT: INCIDENTAL, proven.** `AutonomousDecodeRunner::capture<F>` is generic over any
`Fn() -> Result<Tensor>` and **mentions paging nowhere**. `block_size`/`max_blocks_per_seq` are
consumed only to size `block_tables` and feed `slot = block_tables[b][pos/bs]*bs + pos%bs`.
**A dense cache is the one-block case of that exact formula** — `block_size=max_seq_len`,
`max_blocks=1`, `block_tables[b]=b` collapses it to `slot = b*max_seq_len + pos`.
**Paging is a parameterisation, not a capability.** Route (b) separately disproven: V4-with-paging
measured **zero tokens (ILLEGAL_ADDRESS)**, and the runner isn't reached even then.

**WHY V4 DEFAULTS TO `paged=false` — premise sound, test untouched and still passing.**
`flashinfer_mla_decode.cu` fixes `HEAD_DIM_CKV=512`/`KPE=64`; **V4's 448 trips
`static_assert(vec_size % 8 == 0)` and will not compile.** It computes dense causal attention with
**no sliding window, no sinks, no second key set — which no V4 layer computes.** Plus
`cache_write_and_gather` returns a varlen pack forcing `bs=1`, and V4's compressor `xs` history lives
in `NormalCache` slots the paged arm never clones. ⇒ **the test stayed and the gate moved.**

**The fix names the real property:** new `DecodeKvGeometry` + `GeneralMetadata::decode_kv_geometry()`;
`normal.rs` gates on **geometry** instead of `cache_config`; `step()`'s DefaultInstructions arm now
routes through `graph_wrapped_forward`; the engine offers decode batches from **both** scheduler arms
via one shared `autonomous_decode_attempt`. Kill-switch/negative control `ARC_NO_AUTONOMOUS_DECODE=1`
and an `autonomous_gave_up` latch so a failed init cannot re-allocate every token.

> ⚠️ **CORRECTION TO THIS FILE'S OWN RECORD — it was ONE cause with FOUR symptoms, and the outermost
> was never on file.** The engine called `autonomous_decode` **only** from
> `SchedulerOutput::PagedAttention`, so for V4 the method never ran and
> **`mark_unreachable("cuda_graph.autonomous_decode")` NEVER EXECUTED.** Falsifier: the marker is
> **ABSENT**, not "unreachable", from `/root/budget-chain/profiles/v4b1.json` and
> `v4b1_gpuactquant.json`. **The `normal.rs:1907` citation below was a SOURCE READ, not an
> observation** — an unreached `mark_unreachable` is invisible, and "absent" and "unreachable" are
> different states (D24, three-state guards).

### 🔴🔴 THE KV-WRITE BUG — a **fake 4.9× speedup**, and it is upstream of TWO chains

**Measured, same card / same session / same binary:**
```
EAGER      600 tokens   17.4 T/s   57.55 ms/T   no errors
GRAPHMODE  600 tokens   85.7 T/s   11.67 ms/T   ERROR at prompt step —
           kv-write: src [1,1,1,1] incompatible with cache [1,1,512,512]
```
**The prompt step FAILS, decode proceeds from unwritten KV, and throughput RISES because the model
skipped its work.** ⇒ **Any A/B on this path measures how much work was skipped.**
🔑 **A replay-consistency check would have PASSED — both replay arms agree with each other while
both are wrong. Only replay-vs-EAGER detects it.**

**Trigger is `ARC_CANDLE_ALLOC_CACHE=1`, NOT the capture probe.** Probe alone → zero kv-write errors
across 7 runs. That one env var does **two unrelated things** inside one `if` at `normal.rs:1605`:
`set_alloc_cache_enabled(true)` **and** `set_graph_mode_positions(...)`. **One flag, two behaviours —
that coupling is what hid this. File them as separate flags.**

> ⚠️ **THE NODE COUNTS WERE MEASURED ON THE BROKEN FORWARD.** Corrected:
> | config | prompt step | nodes | memory nodes |
> |---|---|---|---|
> | probe + alloc-cache | **FAILS** | 14,274 | **1,535** ← *do not quote as the gate* |
> | probe only (clean) | OK | 32,621 | **20,930** |
>
> ✅ **The gate is SEPARABLE after all:** the arena hook at `normal.rs:1585` is **outside** that `if`
> (gated on `ARC_ARENA` + `seq_len == 1`). ⇒ **`ARC_V4_CAPTURE_PROBE=1 ARC_ARENA=1` with
> `ARC_CANDLE_ALLOC_CACHE` UNSET** = capture + allocator cache + **correct forward**.
>
> ### ✅✅ THE CENSUS ON A VALID FORWARD — 20,930 → **687**, and MEM_FREE is **0**
> | config | prompt step | nodes | MEM_ALLOC | MEM_FREE |
> |---|---|---|---|---|
> | probe + `ARC_CANDLE_ALLOC_CACHE` | **FAILS** | 14,274 | 1,535 combined | — |
> | probe only | OK | 32,621 | 20,930 combined | — |
> | **probe + `ARC_ARENA`** | **OK ✅** | **10,351** | **687** | **0** |
>
> **Prompt step asserted, not assumed: 0 kv-write errors, 0 `prompt step - Model failed` lines.**
> Arena engagement printed: `ARC arena: ON (cap 8192 MB, intern 16384 layouts)` ·
> `warmup done after 3 steps — 3407 driver allocs, 97 interned layouts. Counters zeroed.`
> ⇒ **Valid-to-valid: 20,930 → 687 = 96.7% reduction** — *better* than the 93% claimed off the broken
> run, and the discredited 1,535 was never the right baseline. **`MEM_FREE = 0` because the arena
> defers frees entirely, so the ENTIRE residual is 687 allocation nodes. That is the distance to
> capture.**
>
> ⚠️ **RETRACTION by the graph chain, its own:** *"Xid unchanged 1590 → 1590"* held for the
> alloc-cache and probe-only paths, but **the arena run produced 1590 → 1591** —
> `Xid 31, name=mistralrs, MMU Fault … FAULT_PDE ACCESS_TYPE_VIRT_WRITE`, at its run's end.
> **Diagnosis corroborates the earlier one exactly: capture-time allocations are VIRTUAL and never
> materialise unless the graph launches; the eager fallback then writes into them** — plausibly worse
> under the arena because deferred frees leave more such buffers live.
> ⇒ **Refusing to launch is NECESSARY BUT NOT SUFFICIENT. The discarded body's buffers must be
> purged — or better, never created.**
>
> **Provenance (this build is NOT reproducible from git alone):** `arena/decode-zero-host` merged
> into `graphstream` (clean, zero file overlap), plus a `[patch]` pointing candle at
> `/root/arc-wt/candle` **which their `Cargo.toml` never committed**. That checkout was **dirty**
> during the build, so it is pinned by `candle_rev=698dec410`, `device.rs md5=4d5e5b2c`,
> binary `341ecfa2`, verified to contain both the census and `ARC_ARENA` before running.

### Root cause: an OWNERLESS allocation decision (not a semantics disagreement)
⚠️ **RETRACTED first diagnosis — and it is the one everyone will guess:** *"eager writes real V,
graph writes a 1-wide marker, the phases disagree."* **Wrong. Both paths write the identical
1-wide marker** (`append_kv_mqa` and `append_graph_kv_mqa` both via `v4_v_marker(k)`).

**Decode reads NOTHING from the V half** — syntactic, not behavioural: both call sites discard the V
readback (`let (k_cached, _marker_cached)`), the graph arm passes `&k_full` as **both** K and V to
`dsv4_attention`, and there is a test `append_kv_mqa_returns_k_for_both_sides`. V4's absorbed MQA
reconstructs V from the same latent as K. **The marker is NOT the bug.**

**The actual fault:**
> `SingleCache::append` tolerates a mismatched buffer by **reallocating** (`preallocation_fits`).
> `SingleCache::append_graph` **never reallocates** and hard-fails on a width it didn't expect.
> **One design, two tolerances, no owner — and the tolerant one always runs first.**

**Three allocator candidates, not two** — the third found in `append`'s own comment: on a prompt step
with no prefix hit the engine issues `CacheInstruction::Reset { load_preallocated_cache: true }` and
`NormalCacheManager::set_none_cache` installs the sequence's **preallocated** tensor as `all_data`,
**before any append**. Sized uniformly from `head_dim` with no knowledge of the marker convention ⇒
**leading suspect.**

**Structural principle (holds whichever candidate wins): the V half's width must be decided ONCE,
explicitly, at cache construction from model config — never implied by execution order. And
`append_graph` must fail loudly on an unexpected width rather than inherit it.**

### ✅ FIXED AND PROVEN ON HARDWARE — `realloc_lines=0`
Binary md5 `ce7b8e15…` frozen and gated, `SYMBOL_OK ×3`, card `apps=[] free_mb=143156`, `bench rc=0`:
```
prefill  src=[1,1,64,512]  buf=Some([1,1,512,512])  reuse
prefill  src=[1,1,64,  1]  buf=Some([1,1,512,  1])  reuse   ← was REALLOCATE
decode   src=[1,1, 1,512]  buf=Some([1,1,512,512])  reuse
decode   src=[1,1, 1,  1]  buf=Some([1,1,512,  1])  reuse   ← was REALLOCATE
realloc_lines=0   v_half_reuse_lines=3   CARD WAS EXCLUSIVE BEFORE AND AFTER   fail=0
```
**Both V-half `REALLOCATE`s replaced by `reuse` against a 1-wide preallocated buffer, in both
phases. The run COMPLETED — this is a trace from a working forward, not from a corpse.**

**Discrimination, not just counting:** the two remaining `ALLOCATE` lines are **rank-3**
(`[1,16,512]`, `[1,1,512]`) — the xs-history/compressor slots that `set_none_cache` deliberately
resets to `None` because they have no preallocated KV-shaped buffer. **Not KV halves, correctly not
flagged.** A guard that counted `ALLOCATE`s would have failed here; one that classifies them passes
for the right reason.

**Three changes shipped:** `ModelConfigLike::v_cache_head_dim()` + `KvCacheLayout::FusedMqaMarker`,
with `engine/add_request.rs` sizing the pair through one `preallocated_kv_shapes()` — and
**`kv_cache_elements_per_token` left byte-identical so the fix cannot smuggle a context-size change
in beside it**; `append_graph` now refuses a buffer it cannot write, naming both shapes; and
**`ARC_CANDLE_ALLOC_CACHE` (allocator) split from `ARC_GRAPH_MODE_POSITIONS` (device-slot KV write)**.
5 new tests pass, **2 go red when `v_cache_head_dim` reverts to mirroring `v_head_dim`**, and the
sentinel builds its config through the **production** `v4_model_config_metadata()` from V4's verbatim
`config.json` — **so restating the struct cannot keep it green.** 185 existing tests still pass.

⏳ **Still running: the three-arm A/B (`eager` / `alloccc` / `graph`) that must show the 4.9× gone.**
**Pass condition is INVERTED and compiled into the script:** it fails if graph exceeds **1.2× eager**,
if the graph arm yields no decode row, or if the `kv-write: src … incompatible` refusal reappears.

### 🔴 THE TWO REMAINING BLOCKERS — named at the branch that bails, and NEITHER IS PAGING
1. **`cuda_graph.capture_probe.capture`** — candle bound to the **legacy default NULL stream**
   ⇒ **graph-stream chain** (`ARC_CAPTURE_STREAM`).
2. **`cuda_graph.autonomous_decode.capture`** — **no V4 forward closure**:
   `Decode path extraction failed: tensor_device_ptr requires CUDA tensor`, and the dedicated path
   models a **dense q/k/v/o + gate/up/down decoder that cannot express MLA + MoE + mHC**
   ⇒ **autonomous-decode chain must pass the V4 candle forward to the generic `capture<F>`.**
   *(The runner is already generic — this is a plumbing job, not a redesign.)*
Then the **arena chain** (no alloc/free nodes in a conditional body).

⚠️ **OPERATIONAL: V4 maps 140 GB of 143.7 GB.** Two control attempts OOM'd — **any neighbour holding
VRAM kills the model load.** V4 timing runs need the card essentially alone.

---

### (historical) THE GATE THAT BLOCKED ALL OF THIS FOR V4 — found by two chains independently

**V4 does not use PagedAttention.** It takes `DefaultInstructions`, so `cache_config` is `None`,
and **three separate fast paths are constructed only on the PagedAttention arm and therefore never
exist for V4**: `graph_wrapped_forward`, the dedicated decode path, and the autonomous runner
(`normal.rs:1907` → `mark_unreachable("cuda_graph.autonomous_decode")`). The current behaviour is
**pinned by a unit test**: `normal_loaders.rs:5687 v4_supports_paged_attention_defaults_to_false`.
Meanwhile `paged_attention` is itself recorded UNREACHABLE at `pipeline/mod.rs:1088` — the gate is
closed *and* the thing it gates on isn't running.

⇒ **The measured 2.0 → 0.0156 host-calls-per-step win cannot be switched on for V4 until this is
resolved.** Open question for the owning chain: is the coupling **essential or incidental**? The
runners need stable device pointers and statically-shaped buffers; paged block tables are one way
to get that, not obviously the only way — and V4's KV path is ours (compressed KV, sliding window,
MQA `num_key_value_heads=1`). **Find out why V4 defaults to `paged=false` before touching it; do
not invert the test to make something pass.**

### ⚠️ SAMPLING DEFECT — REAL, BUT **NOT LIVE IN SERVING** (proven by count, not by call graph)
- **`sampling.cu:150 fused_top_p_bf16` is NOT nucleus sampling.** It walks the vocab in
  **token-id order** accumulating full-distribution mass to `top_p*u`, **biasing toward low token
  ids**. Its own comment says *"Not quite right — proper top-p needs sorting"*. It documented
  itself and shipped anyway.
- **`CudaSampler::sample()` returned `unsupported dtype I32` on EVERY call** ⇒ the correct
  top-k/top-p sampler has **never once executed on a GPU**. Its tests are a CPU simulator that
  **structurally cannot observe this** — D18 again: the test could not have caught it.
- ✅ **`rng_offset` baking — NOW MEASURED, prediction confirmed exactly.** Identical
  capture-and-replay, 64 replays each: **OLD `fused_top_p_bf16` → 1 distinct token** (the by-value
  `rng_offset` is baked at capture, so every step of an autonomous loop draws the same uniform).
  **NEW `arc_launch_sampler_bf16` → 64 distinct tokens, 64/64 inside the probability mass**;
  device-resident `rng_state` advances across replay. **This is the A/B where a broken instrument
  could have shown "both fine" and did not.**
  ↳ *Retracted by the chain itself:* an earlier claim that the old sampler drew **outside** the
  probability mass — that branch never fired, and its single token (121637) was **inside** the
  support. The vocab-id-order bias is real **by code reading**; this test did not isolate it.

### 🛑 STOP-SHIP FOUND IN THE FIX ITSELF — the CORRECT sampler is 663 ms/token

**`arc_launch_sampler_bf16` measured at 663,116 µs/step = 663 ms per token** — **~10× V4's entire
66.68 ms decode step** — at vocab 129,280 with a flat 12,928-token support and `top_k` disabled.

Mechanism confirmed at **`sampling_kernel.cu:279`**: `cap = vocab` whenever `top_k <= 0`, and the
keep loop runs a **full O(vocab) argmax scan per kept token in a single block** ⇒ cost is
**O(vocab × kept)**. Two honest qualifiers from the chain: this is the **worst case** (real LLM
distributions are peaked, keeping `kept` in the tens), and the support-width cost curve
(1 → 12,928) that would locate the cliff is **compiled and queued, not returned**.
**But mistral.rs's default maps to `top_k = -1`, so nothing bounds `kept` today.**

**⛔ Do NOT enable the autonomous sampler as a default until `kept` is bounded.** It is correct and
replay-safe; it is **not yet fast**.

### 📉 THE COST CURVE — measured, and the fix is a REGRESSION in the common case

⚠️ From the **unguarded** run: treat absolute µs as **provisional**; the *shape* is robust to any
uniform contamination factor. Guarded re-run pending.

| support width | legacy µs | bisect µs | speedup |
|---|---|---|---|
| 1 | 134 | 1994 | **0.1×** |
| 8 | 503 | 1996 | **0.3×** |
| 64 | 3,132 | 1991 | 1.6× |
| 512 | 24,287 | 1997 | 12.2× |
| 4,096 | 205,446 | 1998 | 102.8× |
| 12,928 | 665,694 | 1998 | **333×** |

**Bisect is FLAT at ~1995 µs across four orders of magnitude of support width** — the design
property, confirmed empirically, and the strongest internal-consistency signal in the table.
Legacy scales linearly, exactly as the O(vocab × kept) reading predicted.

**But the crossover is at support ≈ 64, and real LLM distributions are peaked** ⇒ typical `kept` is
in the tens ⇒ **bisect is 3–15× SLOWER in the everyday regime.** Swapping unconditionally trades a
pathological cliff for a permanent regression.

### ✅ THE ANSWER IS A HYBRID, NOT A WINNER (D21 — don't rank the novel system down)
Run **legacy's exact iterative argmax while `kept` stays small; fall back to bisection when it
exceeds a fixed iteration budget.** Peaked case stays at legacy speed; diffuse case is bounded at
**~2 ms instead of 663 ms**. And because the fallback is **exact rather than a truncation**, it
still introduces **no cap and no narrowing** — D4 intact. **Designed, not yet built.**
`CudaSampler::sample` stays on the legacy launcher meanwhile.

### ✅ A2 tie boundary — measured, matches the prediction that was refused as a claim
Exactly-uniform support: legacy **15/16** reachable, bisect **16/16**, **TV = 0.0612** against the
analytic **0.0625**. It errs **wider** — the safe direction. The harness printed `MISMATCH` only
because that run used the old single-threshold bound: **the test artifact the chain had identified
and split out in advance precisely so it would not be mistaken for a sampler defect.** It isn't one.

**SCOPE — proven with counts from the real V4 B=1 trace** (`/root/budget-chain/nsys/v4b1.sqlite`,
**4,913,454 kernel launches across 109 distinct kernels**): `fused_top_p*` **0**, `fused_argmax*`
**0**, `arc_sampler*` **0**, anything matching `sampl` **0**. Statically, zero references to
`CudaSampler`/`fused_top_p_bf16`/`launch_fused_argmax_bf16` outside `arc-cuda-graph`.
**Serving samples through `mistralrs-core/src/sampler.rs`, a separate implementation** —
consistent with the 517,120 B D2H per step (= 129,280 vocab × 4 B). The 21,796 `topk_kernel<float>`
launches are the **MoE router** (40/step × 40 layers), not token selection.
⇒ **Confined to the gated-off autonomous path. NOT a stop-everything.** It *was* live for the
autonomous chain's own work, which is why it mattered.

**FIXED:** `arc_launch_sampler_bf16` with device-resident `rng_state` now runs in the graph body;
`fused_top_p_bf16`/`fused_argmax_bf16` are out of it; the by-value `rng_offset` field is **deleted**
rather than left to bake. `top_k` added to `AutonomousDecodeConfig`; the single `normal.rs`
construction site maps mistral.rs's `<=0` to the kernel's `-1`. Penalties stay an in-place pre-pass
and are zeroed in config, because passing `freq_counts=None` with non-zero penalties **drops them
silently** (`sampling_kernel.cu:167`). `cargo check --features cuda` EXIT 0 for `arc-cuda-graph`
and `mistralrs-core`, with an injected-error control proving the check actually compiles them.

**NEW code-read risk for whoever takes the sampler forward:** phase 4 does a **full O(vocab) argmax
scan per kept token** with `cap = vocab` when `top_k <= 0` (`sampling_kernel.cu:279`), and
mistral.rs's default maps to `top_k = -1`. Peaked distributions stay cheap; **a flat distribution
at 129K vocab is a cliff.**

**AUTO_FREE / DEVICE_LAUNCH resolved explicitly:** the conditional path instantiates via flag-free
`cuGraphInstantiate_v2` with the reason in-comment (`autonomous.rs:518-522`); `graph.rs:276` keeps
`AUTO_FREE_ON_LAUNCH` for the *non-conditional* capture-once runner. **Those two instantiate calls
must never be merged** — the header forbids combining that flag with `DEVICE_LAUNCH`.

**tok/s on V4 is NOT measured and must not be estimated.** Work is on `wt/autonomous-decode` in
`/root/arc-wt/autonomous` (6 files, +357/−95); probes at `/root/arc-wt/probe/`.

**RETRACTED:** the five clashing extern declarations in `gemv_ffi.rs` are **NOT** the same class as
the `cuGraphAddNode` bug. `CUstream = *mut CUstream_st` (cudarc-0.19.4 `driver/sys:1759`), so all
five are *pointer-type* disagreements with **identical arity and identical register layout** —
cosmetic, worth a cleanup pass, **not a corruption risk**. `cuGraphAddNode` was different arity
*plus* a transposed out-pointer. Filing them together would have been a false alarm.

---

## Runner-up by GPU time — the keystone GEMM (ArcKernels/Trellis, the moat)

`qtip_gather_gemv_warp_kernel`: **9.85 ms/step, 129 calls, ~77.3 µs each — 29% of all kernel time.**

> ⚠️ **CORRECTION: it is 29× above its own bandwidth bound, NOT 15×.** The 15× figure divided this
> kernel's ms/step by the *whole model's* 0.677 ms floor. **Its own floor is 129 × 2.62 µs =
> 0.338 ms.** Do not requote 15×.

> ⚠️ **"near its instruction floor for this format" is now QUANTIFIED, not accepted.** The Gaussian
> codeword accounts for **13.0 of the 35.5 inst/weight**, and predicted time from inst/weight tracks
> measured time at **~70% issue efficiency in every arm** ⇒ **this kernel is ISSUE-BOUND and
> inst/weight is the control variable.** That is a transfer function, not a ceiling.

### Measured on `arc-v4-stack`, SHA `05af600e7`, interleaved A-B-A-B ×4, **drift 0.16%**

| | inst/weight | µs/call (gate/up) | ms/step | × its own bw bound |
|---|---|---|---|---|
| **ships today** | 35.52 | 77.31 | **9.85** | 29.2× |
| **+ bit-identical fixes** | 28.91 | 66.19 | **8.59** | 25.4× |
| + `sum2` codebook (**FORMAT CHANGE — see caveat**) | 15.85 | 33.47 | **~4.6** | ~13.6× |

> ⚠️ **THE `sum2` ROW IS TIMING-VALID, NOT A QUALITY CLAIM. Quote it as "2.27× available,
> contingent on a re-bake and Jish's call" — NEVER as "2.27× measured end-to-end".**
> In that arm the harness decodes **Gaussian-baked bytes against the MCG codebook**, so the outputs
> are **numerically meaningless**. Control flow and instruction mix are identical, which is exactly
> why µs and inst/weight transfer — and exactly why accuracy does not.
> The only quality evidence that exists is an in-tree fixture
> (`computed_codebook_quality_is_neutral_as_shipped`: **+0.00017 cos, +0.37% weight NMSE**, against
> `split` at −0.00174 / +3.73%). **A serving-side quality number requires the re-bake
> (`ARC_QTIP_CODEBOOK=mcg`) and has NOT been run.**
>
> **Containment confirmed by construction, not by intent:** `git status --porcelain` shows exactly
> one changed file and it is a `.cu` — `mistralrs-quant/kernels/qtip/qtip_gather_gemv.cu`. No Rust
> file touched, so `QtipCodebook::DEFAULT = Gaussian` (`mod.rs:564`) is untouched and `cuda_mult()`
> still returns `CB_MULT_GAUSSIAN_LUT` = 0 for every shipped artifact. **Nothing in this work can
> flip the format**; the kernel merely keeps supporting both arms through the **pre-existing**
> `cb_mult` ABI. Every `sum2` number came from forcing `cb_mult` via a `GG_CB_MULT` env var that
> exists **only in `bench_gg.cu`** and never reached the model path. Both e2e A/B arms are Gaussian.

**Both projections moved** — gate/up **77.31 → 66.19 µs/call (−14.4%)**, down **74.54 → 67.48
(−9.5%)**, total **9.85 → 8.59 ms/step (−12.8%)**. Static SASS and ncu agree to **0.3%** on the
codeword cost.

**Parity: 0 differing bytes** of 24,576 and 49,152, identical md5/FNV/sum — **and the check is
proven able to fail**: a one-token mutation produced 21,494 and 43,150 differing bytes (D33).
**`GROUP=8` is faster but changes summation order ⇒ deliberately EXCLUDED** — speed that costs
bit-identity is a different decision and doesn't ride along in a parity-preserving PR.

**SHIPPED: PR #129**, one file (`mistralrs-quant/kernels/qtip/qtip_gather_gemv.cu`, +158/−100),
checkout restored to `master` with all seven other chains' modified and untracked work intact.

### What the SASS showed that nobody had named
Loop fully unrolled to GROUP×ROWS = 8 symbol decodes; static count 561 = **35.06 inst/weight vs
ncu's measured 35.52**, so the static histogram is a faithful dynamic profile.
1. **16 × `LD.E.U8` — *generic* byte loads, not `LDS`.** `stage_packed` was a runtime `int`, so
   `rp[r]` was a runtime-selected shared-or-global pointer and the compiler had to emit generic
   loads plus a 64-bit address chain (`LEA.HI.X.SX32`×12, `IMAD.SHL.U32`×12, `IMAD.X`×6,
   `IADD3`×23). **The shared-memory staging optimisation was being defeated at the ISA level** —
   the code was written, and silently degraded by one runtime flag.
2. **Half of those loads are the warm-up replay** — 4 symbols replayed per 4 decoded, per row,
   producing **no weights at all**.

**The three bit-identical fixes:** `STAGED` as a *template* parameter (→ real `LDS`, 32-bit
addressing); the L=16 state **is** the last 4 nibbles of the stream, so the replay collapses to one
aligned `LDS.U16` window plus a nibble reversal; and the GROUP activations become one `LDG.E.128`.

**🔑 The `sum2` codebook row is a FORMAT change — it halves inst/weight and needs Jish's call plus
a quality number and a re-bake.** D17: the byte format is the moat, which is exactly why changing
it is a decision, not an optimisation.

### Trap avoided, worth keeping
The chain's first standalone harness ran **129.5 µs vs the model's 78.4** — it had not matched the
crate's nvcc flags. The crate builds with **`--use_fast_math`, which had already collapsed
`logf`/`sqrtf`/`sincosf` into single `MUFU` ops.** Matching the flags reproduced the in-model number
to **1.4%** (77.27 vs 78.35, regs 40 = trace). **Without that check it would have "optimised"
transcendentals that were not there** — the same shape as the earlier +12–20%-predicted/+3.3%-
delivered miss. **Reproduce the in-model number in your harness before optimising anything.**

---

## SUSPECT SCORECARD (three hypotheses, all resolved)

| # | Hypothesis | Verdict |
|---|---|---|
| 1 | mask `arange`/`zeros`/`full` per layer per step (CLAUDE.md pitfall #5) | **INNOCENT** — those are H2D and non-blocking. Whole `sdpa` span incl. all mask construction = 165.01 µs/call = 7.10 ms/step = 10.6%. The real mechanism is the FP8 CPU cast, and the naive fix for it is *negative*. |
| 2 | CUDA graph replay dead | **CONFIRMED**, root cause named: legacy default stream |
| 3 | unfused decode | **CONFIRMED but bounded at 10.6%** |

## Corrections folded in

- The earlier "**~215 µs unattributed** inside `mla_attn`" was **never unattributed** — the child
  list was incomplete. True residual **21.90 µs/call = 3.4%**. The four missing spans:
  `kv_fp8_quant` 73.49 + `kv_fp8_dequant` 50.09 + `compressed_kv_build` 49.97 +
  `compressor_advance` 27.37 = **200.92**, + 21.90 = **222.8 µs**. Nothing is hiding.
- The earlier brief was internally inconsistent by 2.16× (623.32 µs/call × 43 = 26.8 ms, not the
  stated 58.0 ms/step). Resolved in favour of the per-call figure: 643.94 × 43 = **27.69 ms**.
- **RETRACTED: "GPU is 2–7% busy"** — that sampled `nvidia-smi` during the *profiled* run. Clean
  run reads **69–82%**.
- **RETRACTED: "ARC_PROFILE costs 4×"** — inferred from log timestamps; directly measured at 5–8%.

## Surfaced, not shipped

`v4_nan_dbg(&y, &format!("L{li}..."))` builds the `String` **unconditionally, before** the env gate
short-circuits — ~500 wasted allocations + env lookups per token. Assigned to the ArcGraph chain.
