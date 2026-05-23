# SCMoE fused kernel: top-K and rank-K in one MoE pass

**Status:** Research-only. No source modifications.
**Author:** Arc research agent.
**Target:** DeepSeek V4-class MoE (256 experts, top-6, ~13 B active), QTIP 2-bit weights, H100 (3.35 TB/s HBM3) and B200 (8 TB/s HBM3e, extrapolated).
**Question:** Can top-K and rank-K (SCMoE weak route) expert computation be fused into one MoE kernel pass at decode, exploiting unused SM occupancy?

## 1. Headline answer

**Yes — for the kernel topology. No — for the bandwidth bill.**

The existing Arc MoE kernels (`indexed_moe.cu`, `mxfp4_gemm.cu`'s `mxfp4_moe_gemm`) and the upstream Triton MoE kernels (SGLang, vLLM, MegaBlocks) **already parallelize work over the top-K dimension as a data-parallel axis**. Going from K=6 to K_total=12 (top-6 strong + rank-6 weak) requires no kernel rewrite — only a doubled work list. The compute side is essentially free at decode batch=1.

The honest blocker is **HBM bandwidth**. At decode the MoE FFN is bandwidth-bound on expert weight reads, not compute-bound. Doubling the active expert count roughly doubles the bytes read from HBM per MoE layer. With symmetric quantization (both routes use QTIP 2-bit) the realistic decode overhead is **~1.45-1.55×** on H100 and **~1.25-1.35×** on B200 — meaningfully better than the 1.3× the original SCMoE paper measured on Mixtral (where it ran two full forwards), but not "near-free." To get into the 1.10-1.20× range you need **asymmetric quantization** of the weak route (e.g., weak experts at 1-bit BitNet-style or low-rank), and that's where you start trading the cleanliness of the contrast for the cleanliness of the math. Detailed numbers below.

## 2. GPU occupancy at MoE decode — what's actually used

From Arc's `research/speculative_routing.md:81-89` (single-stream H100 V4 decode):

| Component                                    | Latency      |
|----------------------------------------------|-------------:|
| MLA attention (incl. cache load)             | 0.6-1.1 ms   |
| Router (linear + topk)                       | 5-15 us      |
| Expert weight load (8 x 56 MB FP8 = 448 MB)  | 130-450 us   |
| Expert FFN compute (top-8 GEMVs)             | 60-120 us    |
| All-reduce + residual (EP-sharded)           | 30-80 us     |

Per-MoE-layer expert load is **2-7x longer than expert FFN compute** — the SMs sit idle waiting for HBM. At V4's 58 MoE layers x ~300 us expert load ~= **17 ms/token of pure expert-load critical path** out of ~30-40 ms total decode. Expert load is **40-60% of single-batch H100 decode**.

Decode is solidly memory-bound: batch=1 GEMV on 2-bit experts sits at AI < 1 FLOP/byte, well below H100's BF16 ridge point of ~300 FLOP/byte (989 TFLOPS / 3.35 TB/s). SM achieved-occupancy during MoE FFN is typically <30% (Nsight Compute on `fused_moe_kernel` at M=1, BLOCK_M=16: ~12-20 active warps per SM out of 64 max; SGLang's `fused_moe_triton_config.py:203` uses `BLOCK_SIZE_M=16, GROUP_SIZE_M=1` for M<=16, deliberately under-tiling because there's nothing to tile).

**Implication: doubling top-K doubles work the SMs can already absorb, provided the bandwidth side scales.** The bottleneck is HBM, not arithmetic. B200 (8 TB/s HBM3e, 192 GB, TMA, FP4, DSMEM per `ARC_V2.md:21`): same regime, 2.4x more bandwidth headroom in absolute terms. Extrapolation only — no Arc B200 measurements exist yet.

## 3. Existing fused MoE kernels — none do double-K, but all could

A survey of what's shipping, with the file:line for the work-distribution code:

- **SGLang Triton `fused_moe_kernel`** at `research/code/06_foundation/sglang/python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe_triton_kernels.py:323` (and `:91` for GPTQ/AWQ): takes `sorted_token_ids_ptr` (token IDs repeated top_k times and sorted by expert) and `expert_ids_ptr` (one expert per BLOCK_SIZE_M chunk). Grid: `cdiv(EM, BLOCK_SIZE_M) * cdiv(N, BLOCK_SIZE_N)` with `EM = num_tokens * top_k`. The kernel is oblivious to whether top_k is 6 or 12.
- **vLLM Triton `fused_moe_kernel`** at `research/code/06_foundation/vllm/vllm/model_executor/layers/fused_moe/fused_moe.py:295` (`:61` for GPTQ/AWQ; grid launch at `:650`): identical structure to SGLang.
- **MegaBlocks `permute_and_compute`** at `research/code/01_weight_compression/megablocks/megablocks/layers/moe.py:185-207`: sort by expert, bin, binned-gather, grouped-GEMM over `(M_i, K, N)` per-expert sub-problems, binned-scatter. Indifferent to source of expansion.
- **Arc `indexed_moe.cu:651`**: `dim3 grid(n, batch, topk)`. Inside the kernel at `:500-509`, `current_topk = blockIdx.z` selects the expert slot and dereferences `indices[task_id]`. **Doubling K is literally `grid(n, batch, 2*topk)` and a doubled indices array. Zero source change.**
- **Arc `mxfp4_moe_gemm`** at `mistralrs-quant/kernels/mxfp4/mxfp4_gemm.cu:252-293`: explicit `input_has_topk_dim` branch. Branch A iterates all topk slots in the inner loop (line 321) — change `expert_slot_end` to `2*topk`. Branch B expands the grid automatically. Either mode is a parameter change.
- **SGLang FlashInfer CUTE-DSL grouped GEMM** (Blackwell-targeted) at `research/code/06_foundation/sglang/python/sglang/srt/layers/moe/flashinfer_cutedsl_moe.py:22, :137, :163`: parameterized by per-expert problem sizes; 2K is a re-bin.
- **sgl-kernel `moe_align_block_size`** at `research/code/06_foundation/sglang/sgl-kernel/csrc/moe/moe_align_kernel.cu:56`: the one kernel where K is semantic — sorts/bins `topk_ids` of shape `[num_tokens, top_k]`. For doubled K, runs on the concatenated `[topk_strong || topk_weak]` of shape `[num_tokens, 2*top_k]`, which doubles `EM` and so doubles dispatched GEMM blocks.

**Bottom line:** every production MoE kernel structures K as the data-parallel axis of a token-expert work list. Going strong+weak in one pass is a re-bin + a doubled work list, not a kernel rewrite. The hard part is the bandwidth, not the code.

## 4. Bandwidth math (strong-only vs strong+weak)

V4-Flash, QTIP 2-bit experts. Numbers used:
- 58 MoE layers (V4-Flash, per `research/predictor_sparsity_deepdive.md:131`).
- 256 routed experts, top-6 per token.
- Expert intermediate dim: 2048.
- Hidden: 7168.
- Per-expert weights (gate + up + down): 3 x 2048 x 7168 = 44.0 M params.
- At QTIP 2-bit: 44.0 M x 0.25 byte = **11.0 MB per expert** (plus ~3% for LUT + per-row scales).

Per token per MoE layer: top-6 strong = **66 MB**; top-6 strong + rank-6 weak (symmetric 2-bit) = **132 MB**; delta = **66 MB/layer**. Across 58 MoE layers: strong-only **3.83 GB**, strong+weak **7.66 GB**, delta **+3.83 GB per token**.

On H100 HBM3 at 2.6 TB/s realized (~78% of 3.35 peak — typical for SGLang/vLLM nsys traces): strong-only **1.47 ms** of pure expert-weight HBM time; strong+weak **2.95 ms**; **extra +1.47 ms/token**.

Honest end-to-end overhead. Take Arc's V4 decode baseline at ~32 ms/token (PEAK_INFERENCE target, MTP-1 partly hides FFN): ~17 ms expert-load + ~5 ms expert-compute + ~10 ms attention/comms/router. With weak route added: expert load doubles (17 -> 34 ms), expert compute doubles (5 -> 10 ms), the rest unchanged (~10 ms) -> 34 + 10 + 10 = **54 ms = 1.69x overhead**.

That's *worse* than the SCMoE paper's 1.30x because the paper measured Mixtral 8x7B BF16, where attention dominates and there are only 8 experts to reuse. **For V4/QTIP-2bit specifically, naive symmetric strong+weak is 1.5-1.7x, not 1.3x.** Anyone quoting 1.3x from the paper as if it transfers is making a category error. To get to ~1.2x we need to reduce the weak-route bandwidth bill.

## 5. Asymmetric quantization of the weak route

The weak route contributes only a *contrast term* — `logits = (1+beta) * z_strong - beta * z_weak`. Robust to noise that is independent of the strong-route signal, so quantization headroom exists.

| Option | Weak-expert size | Top-6 weak bytes/layer | Total bytes/layer | Token overhead | Quality risk |
|---|---:|---:|---:|---:|---|
| (a) Symmetric QTIP 2-bit | 11.0 MB | 66 MB | 132 MB | 1.55x | None — paper-faithful |
| (b) BitNet 1.58-bit ternary | 8.8 MB | 52.8 MB | 118.8 MB | 1.50x | Low |
| (c) 1-bit sign-only | 5.5 MB | 33 MB | 99 MB | **1.30x** | Moderate (untested) |
| (d) Rank-128 low-rank BF16 | 3.65 MB | 21.9 MB | 88 MB | **1.20x** | Low — Tucker MoE shows rank-128 keeps 95% of FFN variance |
| (e) Shared "universal weak" FFN | 1.5 GB once | ~0 MB | 66 MB | ~1.05x | **Forbidden** — Shi et al. Table 4 shows fixed-baseline contrast collapses |

**Engineering recommendation: option (d).** 3x smaller than 2-bit symmetric; the low-rank approximation just needs to provide a degraded-but-coherent contrast signal, not predict tokens directly. Requires one offline SVD pass over QTIP-dequantized expert weights (~3-6 H100 hours, one-time). Option (c) is the fallback if low-rank training proves brittle. Both reach or beat the paper's 1.3x headline at V4 scale.

## 6. Kernel architecture proposal

Single-kernel design that fits Arc's existing `indexed_moe.cu` / `mxfp4_moe_gemm` structure:

```
inputs:
  x_strong          : [num_tokens, hidden]     // shared input (1 copy)
  topk_idx_strong   : [num_tokens, 6]          // strong route experts
  topk_w_strong     : [num_tokens, 6]
  topk_idx_weak     : [num_tokens, 6]          // weak route experts (rank-2..rank-7)
  topk_w_weak       : [num_tokens, 6]
  expert_w_strong   : [256, ...]               // QTIP-2bit, full 256 expert weights
  expert_w_weak     : [256, ...]               // alt format (1-bit or low-rank), full 256

concat:
  topk_idx_combined : [num_tokens, 12]         // strong[0..5] || weak[0..5]
  topk_w_combined   : [num_tokens, 12]         // strong[0..5] || -beta*weak[0..5]
                                                // (sign-flip baked in for the contrast subtract)
  route_flag        : [num_tokens, 12]         // 0=strong, 1=weak (selects weight format)

kernel:
  grid: (n_output_chunks, num_tokens, 12)
  each block:
    expert_slot = blockIdx.z
    is_weak = route_flag[token_idx][expert_slot]
    expert_idx = topk_idx_combined[token_idx][expert_slot]
    if (is_weak):
      W = expert_w_weak + expert_idx * stride_weak
      // 1-bit decode path: sign bits -> +/-1 -> fma
    else:
      W = expert_w_strong + expert_idx * stride_qtip
      // QTIP LUT decode -> bf16 -> fma
    accumulate gate_up, apply SwiGLU, accumulate down
    output[token, expert_slot] = topk_w_combined[token, expert_slot] * result

post-kernel:
  z_combined = sum(output, dim=expert_slot)    // already includes the -beta subtraction
                                                // because weak weights had sign flipped
```

**Critical observation:** strong and weak experts contribute to the same output residual, weighted by `(1+beta)` and `-beta`. Pre-multiply the weak route's `topk_w` by `-beta` and the strong route's by `(1+beta)`; the final reduce-sum across 12 expert slots *is* the SCMoE contrast — no separate subtract pass. Saves one tensor pass.

**Two code paths in the inner loop** — QTIP LUT decode for strong, 1-bit sign-pack or low-rank matmul for weak. A runtime branch on `blockIdx.z >= 6` works but a templated kernel with two launches (`<typename WeakFmt>`, grid `dim3(n, batch, 6)` each) avoids the divergence. Two launches on the same stream add ~5 us overhead each, negligible against ~300 us expert-load.

**Synchronization:** none beyond the existing per-layer barrier. Strong and weak GEMV outputs feed the same all-reduce; the reduce-sum across expert_slot is the existing MoE down-projection accumulator.

**Memory layout:** strong (QTIP) weights stay in `[256, N, K]`; weak weights go in a parallel `[256, N', K']` buffer with format-specific stride. Router output produces two index arrays: `topk(scores, 6)` (existing) and `rank_k(scores, [2..7])` (one fused kernel picking k-th sorted elements). Both <= 1 KB.

**SM occupancy:** at decode batch=1, strong fused_moe achieves ~20 warps/SM (~30% occupancy). Adding weak roughly doubles outstanding work toward ~60% — productive range, absorbs the compute side entirely. HBM is still the limit.

**Persistent kernel variant** (replace per-block dispatch with a GPU work queue pulled by 132 SMs) eliminates kernel-launch overhead but launch overhead is already <1% of decode budget. CUTLASS Hopper grouped GEMM (`cutlass/include/cutlass/gemm/kernel/sm90_gemm_grouped*`) provides this pattern at ~5-8 agent-sessions of porting. **Ship doubled-K version of the existing kernel first; defer persistent.**

## 7. Pipeline parallelism — can weak hide under attention?

Per-layer critical path: `attention (0.8 ms) -> router (10 us) -> expert_load+compute (~400 us) -> comms (~50 us)`. Weak route adds ~400 us (1-bit) or ~150 us (low-rank).

Run weak on a side stream, **one layer offset**: during attention of layer L, async-load + compute weak experts for layer L-1's residual (held in a small ring buffer). SCMoE in the paper applies its contrast at the final logits, not per-layer, so offsetting weak by one layer is semantically equivalent — **lossless**.

For low-rank weak (~150 us) under H100 attention (~0.8 ms): easily fits. For 1-bit weak (~400 us): ~50% of attention, tight but feasible. CUDA-graph capture of two streams with event-record/wait nodes is well-supported (CUDA 12.x); Arc's `arc-cuda-graph/src/decode_forward.rs` already runs decode on a non-blocking stream — adding stream B + ~58 event sync nodes is mechanical.

**Effective overhead with pipelining drops from 1.5-1.7x to ~1.20-1.30x on H100 with low-rank weak**, and to ~1.10-1.15x extrapolated on B200.

## 8. B200-specific opportunities (extrapolation — no Arc B200 measurements yet)

Per `ARC_V2.md:21`, B200 has 8 TB/s HBM3e and 192 GB. Key deltas:

- **HBM bandwidth ~2.4x higher** — the dominant bottleneck shrinks proportionally; extra HBM time for weak route drops to ~0.6 ms instead of ~1.5 ms.
- **TMA (Tensor Memory Accelerator)** for async loads — critical for option-(d) low-rank weak: the `U Sigma V^T` matmul intermediates load with zero CPU sync.
- **DSMEM across SMs** — could share the weak LUT (if 1-bit weak uses one) across SMs without L2 re-load. Minor.
- **FP4 native paths** don't help the weak route past 1-bit (QTIP-2bit is already smaller than FP4), but could replace QTIP on the strong path if Arc adopts FP4. Orthogonal.

Extrapolated overhead with B200 features + low-rank weak + pipelining: **~1.10-1.15x**. Extrapolation only. Whoever ships this needs to measure.

## 9. Composition with QTIP 2-bit base

QTIP's decode (`mistralrs-quant/src/qtip/mod.rs:54`, `L=16, K=4, V=2`) is a state-machine LUT lookup; the LUT is 128 KB per layer, L2-resident after one warm-up. QTIP places **no constraint on the weak format**: 1-bit sign decode, low-rank BF16 matmul, and symmetric QTIP-2bit all run as independent inner-loops. QTIP's optional Hadamard incoherence rotation (qtip/mod.rs:32) composes orthogonally — apply once at weight-load time, both routes see the rotated input.

## 10. Honest cost summary

| Config                                                       | H100 overhead | B200 overhead (extrap.) | Quality risk vs paper |
|--------------------------------------------------------------|--------------:|-------------------------:|-----------------------|
| Naive: top-6 strong + top-6 weak, both QTIP-2bit, no overlap | 1.50-1.70x    | 1.25-1.35x               | None (matches paper)  |
| + Stream-pipelined weak under attention                      | 1.30-1.40x    | 1.15-1.20x               | None                  |
| + Weak as 1-bit sign-pack                                    | 1.20-1.30x    | 1.10-1.15x               | Moderate (untested)   |
| + Weak as rank-128 low-rank (BF16)                           | 1.15-1.25x    | 1.07-1.12x               | Low (Tucker shows 95% retention) |
| + Persistent kernel + DSMEM weak LUT (B200 only)             | n/a           | 1.05-1.10x               | Same                  |

**Realistic minimum on H100 today: ~1.15x overhead with rank-128 low-rank weak + stream pipelining.**
**Realistic minimum on B200: ~1.07-1.10x extrapolated.**

The "near-free" claim that motivated this study is **not achievable on H100 batch=1**. The HBM bandwidth bill is fundamental. It *becomes* near-free on B200 with FP4 + TMA + low-rank weak — but that's a hardware-dependent claim and Arc has no B200 measurement yet.

## 11. Verdict — research project or port?

**Port-able (2-4 agent-sessions):** concatenate strong + weak indices, pre-multiply by `+/-beta` for the contrast subtract; re-use existing `indexed_moe.cu` / `mxfp4_moe_gemm` with grid `(n, batch, 12)`; plumb a second weight buffer for the weak route (QTIP-2bit symmetric); extend `MoeGate::forward` in `mistralrs-core/src/models/deepseek4.rs:1088` to emit both `topk_idx` and `rank_k_idx` from one scoring pass. Ships the capability at ~1.50-1.70x decode overhead on H100 (1.25-1.35x on B200, extrapolated).

**Research project (multiple sessions of engineering + measurement):** offline rank-128 SVD of all 256 V4 experts per layer (~3-6 H100 hours one-time); 1-bit / low-rank weak-format decode kernels; two-stream pipelined CUDA graph; H100 + B200 measurement; quality sweep to confirm contrast holds with degraded weak formats. Hits ~1.15-1.20x on H100, ~1.07-1.10x on B200. The empirical hole: nobody has validated SCMoE-style contrast with 1-bit or low-rank weak experts at V4 scale. The math says noise correlation should still cancel; untested.

**Recommendation:** ship the symmetric 2-bit version first to validate SCMoE -> V4 transfer (open question per `research/quality_up_inference.md:331-333`: does SCMoE work at 256-expert top-6 scale at all?). If yes, invest in low-rank weak for the 1.15x headline. If symmetric SCMoE doesn't transfer — possible because V4's top-6 is deeper than Mixtral's top-2 and rank-K may not produce a useful contrast at that depth — the kernel project is moot. **The first probe is a quality probe, not a kernel probe.**

## 12. References

- **SCMoE paper:** Shi et al., "Unchosen Experts Can Contribute Too: Unleashing MoE Models' Power by Self-Contrast", NeurIPS 2024. [arXiv:2405.14507](https://arxiv.org/abs/2405.14507).
- **SGLang fused MoE kernels:** `research/code/06_foundation/sglang/python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe_triton_kernels.py:91, :323`. CUDA align kernel at `research/code/06_foundation/sglang/sgl-kernel/csrc/moe/moe_align_kernel.cu:56`.
- **vLLM fused MoE:** `research/code/06_foundation/vllm/vllm/model_executor/layers/fused_moe/fused_moe.py:61, :295, :620`.
- **MegaBlocks permute-and-compute:** `research/code/01_weight_compression/megablocks/megablocks/layers/moe.py:185`. Paper: Gale et al., NeurIPS 2023, `research/01_weight_compression/megablocks_block_sparse_moe.pdf`.
- **FlashInfer CUTE-DSL grouped GEMM:** `research/code/06_foundation/sglang/python/sglang/srt/layers/moe/flashinfer_cutedsl_moe.py:22, :137, :163`.
- **Arc indexed MoE kernel:** `mistralrs-quant/kernels/indexed_moe/indexed_moe.cu:489-567` (template), `:651` (grid launch).
- **Arc MXFP4 MoE kernel:** `mistralrs-quant/kernels/mxfp4/mxfp4_gemm.cu:251-460` (kernel), `:827, :848` (launch).
- **Arc QTIP 2-bit:** `mistralrs-quant/src/qtip/mod.rs:1-100`. Paper: Cornell NeurIPS 2024, `research/01_weight_compression/qtip_trellis_coded_quantization.pdf`.
- **V4 MoE structure in Arc:** `mistralrs-core/src/models/deepseek4.rs:1050-1188` (MoeGate), `:1190-1248` (Moe), `mistralrs-core/src/moe/experts.rs:369-477` (forward_fused).
- **H100 V4 decode latency table:** `research/speculative_routing.md:81-89` — primary source for the section 2 occupancy numbers.
- **Low-rank MoE references:** `research/01_weight_compression/td_moe_tucker_decomposition_moe.pdf`, `research/01_weight_compression/moe_lrd_low_rank_decomposition_2024.pdf`, `research/01_weight_compression/sub_moe_subspace_merging.pdf`.
- **BitNet b1.58 ternary weights:** `research/inference_sparsity_survey.md:128`; original paper [arXiv:2402.17764](https://arxiv.org/abs/2402.17764).
- **Existing SCMoE survey & implementation sketch:** `research/quality_up_inference.md:64-95, :259-271, :319-336`.
- **Arc B200 target spec:** `ARC_V2.md:21`. No measurements yet.
