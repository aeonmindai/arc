# Peak Inference: a 120 tok/s BF16 target

> **This document is a PLAN, not a results page.** Every tok/s figure in it is a
> **[target]** or a **[derived]** roofline. **Nothing here has been run.** No
> B200 has ever been rented for this project, and `deploy/benchmark.py` — cited
> below as the measurement harness — **does not exist in the tree**. For numbers
> that were actually measured, see [BENCHMARKS.md](BENCHMARKS.md).

## Evidence grades

Same grading as the rest of `docs/`. Every number below carries one.

| grade | meaning |
|---|---|
| **[measured]** | someone ran it on hardware; the box and the shape are stated |
| **[derived]** | arithmetic over measured quantities or over published hardware specs |
| **[target]** | a goal we are designing toward. **Not a measurement and not a forecast** — nobody has run it |
| **[published]** | a third party's number, measured against *their* baseline, not ours |

**Target:** Qwen3-32B BF16 on a single NVIDIA B200 at 120 tokens/second per
request. **[target]**

**Roofline — needs no third party:** 65 GB weights ÷ 8 TB/s HBM = 8.125 ms/step,
plus ~0.4 ms non-weight overhead = 8.53 ms → **117 tok/s conservative, 123 tok/s
best case**. **[derived, from published B200 HBM specs and the model's weight
footprint]**

**Where the budget goes** — the whole reason a plan exists **[derived]**:
- Weight reads: 8.125 ms (unavoidable)
- Non-weight compute: ~0.16 ms (attention, norms, activations at short context)
- Fixed costs: ~0.25 ms (LM head, sampling, graph replay)
- **Available budget: 8.53 ms per token → 117 tok/s**

Everything above the 8.53 ms floor is software overhead — kernel launches,
scheduling, allocation — which is what the phases below attack.

### What this document does NOT claim

It previously published a line reading *"Current state: 33 tok/s (Arc), 60 tok/s
(SGLang), 43 tok/s (vLLM). All on same hardware, same model, same benchmark."*
**That comparison was never run**, on any hardware, by anyone on this project.
It has been deleted rather than re-sourced, and no estimated replacement row has
been substituted for it, per DOCTRINE D3.

Two separate defects were in that one line:

1. **The competitor half was fabricated.** We have never benchmarked SGLang or
   vLLM. No third-party engine number in this repo may be presented as a
   head-to-head against Arc unless we ran both sides ourselves and say where.
2. **Arc's own 33 tok/s was equally unmeasured** — it is not in `FACTS.md`, no
   B200 was ever rented, and the harness it was attributed to does not exist.
   Arc's only measured decode figure is **14.58 tok/s** on a *different* model
   and a *different* card (DeepSeek V4 Flash, 1×H200, b=1, no-`cudnn` build)
   **[measured]** — see [BENCHMARKS.md](BENCHMARKS.md). It does not transfer
   here and must not be quoted as this baseline.

**Why no competitor row can be reconstructed for the flagship model.** For
DeepSeek V4 Flash, **every published serving configuration we have been able to
find needs more than one GPU.** What we actually checked, and what it said
**[published — third-party model cards and configs; survey by us, not
exhaustive]**:

| what we checked | what it says |
|---|---|
| the native checkpoint | ≈160 GB — larger than a 141 GB H200 |
| smallest published serving config | **4×H200** |
| the one W4A16 quantization we found | 143 GB; its own model card states *"TP=1 OOMs on a single 141 GB H200"* |
| NVFP4 | Blackwell-only, so not an H200 option |

Arc's ~68 GB artifact (≈1.9 bits/param) is what makes 1×H200 possible for us.

**Scope this claim honestly.** It is "we searched and found no published
single-GPU configuration", not "none exists" — absence of evidence, not evidence
of absence. It would be **refuted** by anyone producing a single-GPU config for
this model, and that is the right way to state it: a claim worth publishing is
one that says what would falsify it.

The comparison we *can* defend is therefore **footprint — one GPU against a
published four — plus $/Mtok per node**, and not a tok/s head-to-head, which we
have never run against any engine.

**Sanity-check against a roofline instead of a competitor.** For V4 Flash on
H200: 68 GB ÷ 4.8 TB/s = 14.2 ms/step ⇒ **~4,500 tok/s at B=64** **[derived]**,
against which the measured 63.5 ms grouped-GEMM microbenchmark sits at **~22% of
roofline** **[measured-kernel]**. That is a real, refutable gap, and it needs no
third party to state.

---

## Phase 1: CUDA Graph Capture — target 90+ tok/s **[target]**

CUDA graphs record a sequence of kernel launches once, then replay the entire sequence with a single API call (~10-20μs vs 3-7ms of individual launches). This is the single largest optimization — it eliminates the dominant source of overhead.

### Architecture

```
First decode step (recording):
  cudaStreamBeginCapture(stream)
  → RMSNorm layer 0
  → QKV GEMM layer 0
  → Attention layer 0
  → O GEMM layer 0
  → RMSNorm layer 0 (post)
  → Gate+Up GEMM layer 0
  → SiLU activation layer 0
  → Down GEMM layer 0
  → ... (repeat for all 64 layers)
  → Final RMSNorm
  → LM Head GEMM
  cudaStreamEndCapture(stream) → CUgraph
  cudaGraphInstantiate(graph) → CUgraphExec

All subsequent decode steps:
  cudaGraphLaunch(graphExec, stream)  // single call, ~10μs
```

### What changes in the codebase

#### Key discovery: cudarc already uses `cuMemAllocAsync`

cudarc (Candle's CUDA backend) checks for `CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED` at device init. On modern GPUs (compute >= 8.0, i.e., Ampere+), ALL allocations go through `cuMemAllocAsync`, which IS compatible with CUDA graph stream capture.

This means we can capture the standard Candle forward pass directly — no custom allocator, no model-specific decode path, no cuBLASLt reimplementation.

#### 1. New crate: `arc-cuda-graph/`

```
arc-cuda-graph/
├── Cargo.toml
├── build.rs       # Links libcuda + libcudart
├── src/
│   ├── lib.rs     # Public API
│   ├── ffi.rs     # Raw CUDA driver FFI (graph capture/replay)
│   └── graph.rs   # CudaGraphRunner (model-agnostic)
```

**`graph.rs` — CudaGraphRunner:**
- Model-agnostic: wraps ANY Candle forward pass
- Captures via `cuStreamBeginCapture` / `cuStreamEndCapture`
- Instantiates with `cuGraphInstantiate_v2`
- Replays with `cuGraphLaunch` (~10μs per step)
- Batch size bucketing (powers of 2) for graph reuse
- Pre-allocated fixed input buffers for stable GPU addresses
  - `hidden_states`: [batch, hidden_size] — main activation buffer (2 ping-pong buffers)
  - `residual`: [batch, hidden_size] — residual connection buffer
  - `qkv_out`: [batch, (num_heads + 2*num_kv_heads) * head_dim] — QKV projection output
  - `attn_out`: [batch, num_heads * head_dim] — attention output
  - `gate_up_out`: [batch, 2 * intermediate_size] — gate+up projection output
  - `down_out`: [batch, hidden_size] — down projection output
  - `logits`: [batch, vocab_size] — LM head output
  - `norm_buf`: [batch, hidden_size] — RMSNorm scratch

**`capture.rs` — Stream capture mode:**
- Sets a thread-local flag indicating we're in capture mode
- During capture: all `dev.alloc()` calls redirect to pool slots
- After capture: pool is frozen, graph replays reuse same memory

#### 2. Modifications to `mistralrs-core/src/pipeline/`

**`mod.rs` — Pipeline trait extension:**
- Add `fn cuda_graph(&self) -> Option<&CudaGraphManager>` with default `None`
- Add `fn supports_cuda_graph(&self) -> bool` with default `false`

**New file: `cuda_graph_manager.rs`:**
- `CudaGraphManager` struct holds:
  - `HashMap<(batch_size, seq_len_bucket), CudaGraphExec>` — cached graphs per shape
  - `MemoryPool` — pre-allocated buffers
  - `max_batch_sizes: Vec<usize>` — [1, 2, 4, 8, 16, 32] pre-captured batch sizes
- On first decode at a given batch size:
  1. Allocate pool for that batch size
  2. Run one decode step under `cuStreamBeginCapture`
  3. Instantiate the graph
  4. Cache it
- On subsequent decodes at same batch size:
  1. Update input pointers (query, KV cache block tables, context lengths)
  2. `cuGraphLaunch` — single call
- Batch size bucketing: round up to nearest power of 2, pad with dummy tokens

#### 3. Modifications to `mistralrs-core/src/engine/mod.rs`

**Decode path changes:**
- After model warmup, trigger graph capture for batch sizes [1, 2, 4, 8, 16, 32]
- In the decode loop:
  1. Check if graph exists for current (padded) batch size
  2. If yes: update graph input/output pointers → `cuGraphLaunch` → read output
  3. If no: fall back to eager execution (prefill always runs eager)

#### 4. Modifications to Candle (or bypass layer)

**The critical challenge:** Candle's `Tensor::matmul()` dispatches to cuBLAS internally. During CUDA graph capture, cuBLAS calls ARE captured automatically (cuBLAS is graph-safe since CUDA 10). So this works without modifying Candle — as long as:
- No dynamic memory allocation inside `matmul` (cuBLAS workspace must be pre-allocated)
- No host-device synchronization inside `matmul`
- Tensor shapes are identical between recording and replay

**cuBLAS workspace pre-allocation:**
- Call `cublasSetWorkspace()` with a pre-allocated buffer before capture
- cuBLAS versions ≥ 11.0 support this
- cudarc already wraps cuBLAS — check if workspace is configurable

**Verification plan:**
- Step 1: Test bare cuBLAS matmul under graph capture with cudarc
- Step 2: Test Candle matmul under graph capture
- Step 3: If Candle does host sync or dynamic alloc during matmul, create a thin bypass that calls cuBLAS directly via cudarc FFI for the decode path only

#### 5. What CANNOT be in the graph

- **Prefill:** Variable sequence length → different graph per length → not worth caching
- **Sampling:** Top-p/top-k involves sorting, which has data-dependent control flow
- **KV cache page allocation:** Happens on host (Rust scheduler), not GPU
- **Block table updates:** Host-side, copied to GPU before graph launch

These run AFTER the graph replay, outside the captured region.

### Implementation steps

1. **Spike: cudarc graph capture test** — Write a standalone test that captures 3 cuBLAS matmuls in a graph and replays them. Confirm cudarc's `CudaStream` supports `cuStreamBeginCapture`. (~1 day)

2. **Spike: Candle matmul under capture** — Create a test that does `Tensor::matmul` inside `cuStreamBeginCapture/EndCapture`. Identify if Candle does any disallowed operations (host sync, malloc). (~1 day)

3. **Build `arc-cuda-graph` crate** — Graph wrapper, memory pool, capture mode. Unit tests for pool allocation and graph record/replay. (~2 days)

4. **Integrate into decode path** — Hook `CudaGraphManager` into the engine's decode loop. Capture on first run, replay on subsequent. (~2 days)

5. **Benchmark** — Deploy to Modal B200, run benchmark. Target: 80-100 tok/s. (~1 day)

6. **Optimize** — Profile with nsys, identify remaining overhead, fix. Target: 100-120 tok/s. (~2 days)

---

## Phase 2: Kernel Fusion — target 110+ tok/s **[target]**

Fused kernels eliminate intermediate memory round-trips and reduce graph node count.

### Fusions

**1. RMSNorm + Residual Add → single kernel**
- Current: residual add writes to HBM, RMSNorm reads it back
- Fused: one kernel reads residual + hidden, computes norm, writes normalized output
- Saves: 1 HBM read + 1 HBM write per layer = 2 × batch × hidden × 2 bytes × 64 layers

**2. SiLU + elementwise multiply (gate mechanism) → single kernel**
- Current: SiLU(gate) written to HBM, then gate * up read back
- Fused: one kernel computes SiLU(gate) * up in registers
- Saves: 1 HBM read + 1 HBM write per layer

**3. Fused QKV projection → single GEMM**
- Current: 3 separate GEMMs for Q, K, V
- Fused: 1 GEMM with concatenated weight matrix [W_Q; W_K; W_V]
- Saves: 2 kernel launches per layer (minor with CUDA graphs, but still saves cuBLAS overhead)

### Implementation

New CUDA kernels in `arc-turbo/src/cuda/`:
- `fused_rmsnorm_residual.cu` — RMSNorm with fused residual add
- `fused_silu_mul.cu` — SiLU activation with elementwise multiply
- Rust FFI wrappers and Candle integration

For QKV fusion: modify model forward pass to concatenate weight matrices at load time.

---

## Phase 3: Memory & Scheduling — target 120 tok/s **[target]**

### Zero-allocation decode path

- Pre-allocate ALL tensors at model load time for max supported batch size
- Decode step does zero `cudaMalloc` / `cudaFree` calls
- KV cache blocks pre-allocated by PagedAttention cache engine (already done)
- Intermediate activation buffers from Phase 1 memory pool

### Scheduling optimization

- Lock-free request queue (replace `Mutex<Scheduler>` with lock-free MPSC)
- Batch formation on dedicated thread, not on engine thread
- Pre-compute block tables and context lengths on CPU while GPU executes previous step
- Double-buffer input tensors: prepare step N+1 inputs while step N runs on GPU

### cuBLAS tuning

- Use `cublasLtMatmul` with algorithm selection heuristic for B200
- Pre-select optimal algorithm per GEMM shape at startup (one-time cost)
- Set math mode to `CUBLAS_TF32_TENSOR_OP_MATH` for non-attention GEMMs where precision allows
- Pre-allocate cuBLAS workspace (required for graph capture anyway)

---

## Phase 4: TurboQuant Kernel Optimization (context scaling)

Once the BF16 baseline hits 120 tok/s, TurboQuant's KV cache compression provides the scaling advantage at long contexts and high batch counts. The kernel optimizations:

### cp_async pipelining

Current TurboQuant kernel does raw global memory reads. Add 2-stage pipeline:
```cuda
// Stage 0: load tile 0 to smem
cp_async::pred_load(smem[0], global_ptr_tile_0);
cp_async::commit_group();

for tile = 0..num_tiles:
  // Start loading tile N+1
  cp_async::pred_load(smem[(tile+1)%2], global_ptr_tile_next);
  cp_async::commit_group();
  // Compute on tile N (already in smem)
  cp_async::wait_group<1>();
  compute_qk(smem[tile%2], ...);
```

### Coalesced V reads

Current: 4 scattered byte reads per 10-in-32 group (each separated by BLOCK_SIZE bytes).
Fix: Change V cache layout to store the 4 bytes of each 10-in-32 word contiguously:
```
Current: vc[group*4*BS + t], vc[group*4*BS + BS + t], vc[group*4*BS + 2*BS + t], vc[group*4*BS + 3*BS + t]
New:     vc[group*BS*4 + t*4 + 0..3]  // 4 consecutive bytes = 1 coalesced 32-bit read
```
This turns 4 scattered reads into 1 coalesced `uint32_t` read.

### Split-K decomposition

For long contexts (1K+ tokens), split the sequence dimension across multiple thread blocks:
- Each block processes a chunk of the KV cache
- Partial results (weighted V sums + log-sum-exp) written to scratch buffer
- Final reduction kernel merges partial results using online softmax (FlashAttention-style m/d/o merge)

---

## Benchmark Targets

**Every row is a [target]. None has been run.** There is no measured baseline
row because no B200 benchmark of Qwen3-32B has ever been executed on this
project — see the caveat at the top of this document.

| Phase | Target tok/s | Target overhead | Target utilization | Status |
|-------|---------------|----------|-------------|---|
| Baseline (Arc, unmeasured) | — | — | — | **Never run.** No B200 rental exists; the previously published "33 tok/s / ~22 ms / 26%" row was not a measurement and has been removed |
| Phase 1 (CUDA graphs) | 90-100 | ~1-2ms | 72-81% | **[target]** |
| Phase 2 (kernel fusion) | 105-115 | ~0.5-1ms | 85-93% | **[target]** |
| Phase 3 (scheduling + cuBLAS) | 115-120 | ~0.3-0.5ms | 93-97% | **[target]** |

**Intended** measurement protocol, once someone runs it: 20 requests, 5
concurrent, 256 max tokens, on a Modal B200. Note that the harness this was
attributed to, `deploy/benchmark.py`, **does not exist in the tree** — only
`deploy/modal_b200.py` does. Writing it is part of the work, not a prerequisite
already satisfied.

**The first honest step for this document is to produce its own baseline row.**
Until that exists, no phase in this plan has a measured starting point, and the
speedup ratios implied by the table are unfounded.

---

## Files to Create/Modify

### New files
- `arc-cuda-graph/Cargo.toml`
- `arc-cuda-graph/src/lib.rs`
- `arc-cuda-graph/src/graph.rs`
- `arc-cuda-graph/src/pool.rs`
- `arc-cuda-graph/src/capture.rs`
- `arc-turbo/src/cuda/fused_rmsnorm_residual.cu`
- `arc-turbo/src/cuda/fused_silu_mul.cu`

### Modified files
- `mistralrs-core/src/engine/mod.rs` — graph manager integration in decode loop
- `mistralrs-core/src/pipeline/mod.rs` — Pipeline trait extension
- `mistralrs-core/src/pipeline/normal.rs` — Graph capture for normal text models
- `mistralrs-paged-attn/src/cuda/turbo_paged_attention.cu` — cp_async, coalesced V, Split-K
- `mistralrs-paged-attn/src/cuda/backend/paged_attention.rs` — graph-compatible dispatch
- `Cargo.toml` (workspace) — add arc-cuda-graph
- `deploy/modal_b200.py` — updated build flags if needed
- `deploy/benchmark.py` — add latency percentiles, warmup improvements
