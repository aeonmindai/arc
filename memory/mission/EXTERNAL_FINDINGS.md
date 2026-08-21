# EXTERNAL_FINDINGS.md — cross-project intel worth acting on.
# Source: colibrì (JustVugg/colibri) v1.6.1 @df2c248, researched 2026-08-14.
# Colibrì = pure-C MoE engine streaming experts from NVMe (consumer HW).
# Opposite mission to Arc, but it independently implements DeepSeek V4 Flash,
# which makes it a REFERENCE ORACLE for our V4 port.

## F1 (DEFECT, ours) — V4 Lightning Indexer scores wrongly. HIGH VALUE.
Two independent references agree AGAINST Arc: SGLang `indexer.py:78-88`
(vendored in our own research/code/06_foundation/sglang) and colibrì
`deepseek_v4.c` (written from the checkpoint).
REFERENCE: `logit[q,c] = kv_scale_c * Σ_h relu(q_h·k_c) * w_h[q]` → ONE top-k over c.
ARC (dsv4_indexer.rs:366-380, arc-cuda-graph/src/cuda/flashmlasparse/
indexer_score.cu, and its cpu_reference — all self-consistent, all wrong):
`score[h,q,c] = (q_h·k_c) * w_h[q]` → top-k PER HEAD.
Three deltas: (a) no ReLU (negatives cancel positives across heads instead of
clamping → different keys chosen); (b) no sum over heads — per-head selection
also multiplies sparse-gather traffic by n_heads(64), a PERF bug for our fleet
wedge since shared page selection is what makes the gather cheap; (c) no
Hadamard + low-precision QDQ on indexer Q (SGLang fuses it; colibrì does
Hadamard → FP4 QDQ group-32 → bf16 round, bf16 rounding after every stage).
Also omitted: `weight_scale = softmax_scale * n_heads^-0.5` (uniform positive
⇒ selection-neutral; matters only if logits are consumed numerically).
STATUS: DORMANT — dsv4_attention.rs:33-45 attends densely over raw∪compressed,
and CSA top-512 covers all compressed entries while ctx ≤ index_topk*ratio=2048.
Bites the moment the indexer goes live for long context = exactly V4 Flash's
value. Fix now (free) vs later (rented-GPU debug session).

### F1 STATUS: FIXED in PR #26 (2026-08-14, draft).
Verified against SGLang source directly: `fp8_paged_mqa_logits_torch:84-89` =
linear → relu → *q_scale → sum(dim=1) → *kvcache_scale, ONE topk over [batch,seq].
`C4Indexer.__init__:519`: weight_scale = head_dim^-0.5 * n_heads^-0.5 (folded
into weights pre-kernel). kv_scale is the paged-FP8 dequant scale — ≡1 for Arc's
BF16/F32 indexer K, but NOT selection-neutral if/when paged-FP8 indexer K lands.
Arc now: indexer_logits(q,k,w) = matmul→relu→*w→sum-over-heads, out [B,T_q,topk];
CUDA regridded (B·H,T_q)→(B,T_q) with head loop inside the block (warp-owned
column ⇒ atomic-free, deterministic); entry points renamed *_score_*→*_logits_*
so stale callers LINK-ERROR instead of silently computing the old formula.
Dormancy proven (zero `.indexer` reads in deepseek4.rs; dense union exact while
ctx ≤ 512×4=2048). Tests: force-full-selection ⇒ bit-exact dense equivalence
(both Rust + CUDA cpu_reference) + a fixture where old/new pick DIFFERENT keys.
**NEW FINDINGS from that work (act before enabling the sparse path):**
(i) Arc's compressor builds indexer K **per-head** [B,H,T_c,D] while SGLang's
indexer cache is **MQA (num_heads_kv=1)** ⇒ Arc pays ~64× the K memory/compute
for the indexer. (ii) The dead `indexer` field costs weight memory per CSA layer
for a path that never runs today. (iii) Hadamard + FP4-QDQ on indexer Q still
deferred (separate variable).

## F2 (IDEA, maybe) — lossless entropy coding of ALREADY-quantized weights.
Orthogonal to TurboQuant (compresses quantized bytes; does NOT requantize).
`int4-rans256-g0`: static rANS, 256 round-robin-interleaved streams/tensor so
wide decoders get coalesced output; ~0.76 ratio on per-row int4 GLM-5.2,
byte-exact. Measured VRAM tier (DietGPU ANS, warm int4 experts compressed in
VRAM, decoded to scratch before grouped GEMM): 6×5090 host went 9,335 → 10,628
resident experts (+13.9%), decode +4.5%..+15%.
CAVEATS THEY DOCUMENT (we'd inherit): interleave granularity is what makes GPU
decode viable; moving experts between execution paths changed FP accumulation
order enough to diverge a greedy A/B even with byte-exact weights.
NEGATIVE RESULT THAT GATES US: on group-scaled (g64) containers ratio collapses
to ~0.89 (codes better normalized, scales dominate the stream). **If our
TurboQuant/qtip output is group-scaled, entropy coding likely won't pay —
MEASURE THE RATIO ON A REAL CONTAINER BEFORE BUILDING ANYTHING.**

## F3 (EMPIRICAL, saves GPU-hours)
- Quantized draft head kills speculation: int4 MTP head → 0-4% acceptance;
  int8 mandatory. (Relevant if we ever qtip the V4 MTP head — we currently
  ISQ it; VERIFY.)
- Draft and verify MUST run the same kernel family: their SPEC_PIN=1 default
  exists because different kernels for draft vs verify silently destroyed
  acceptance. Arc has multiple GEMV paths → same trap.
- Speculation can be NET NEGATIVE: MTP measured -32% at ~85% expert hit rate.
- Quantization-granularity failures show up as **EOS starvation, not
  perplexity**: per-row int4 vs group-64 ≈9pp worse, manifesting as think-mode
  loops / never-terminating generations. ⇒ add a TERMINATION check to the
  TurboQuant K4/V3 quality gate; perplexity would miss it.

## F4 (TEST METHOD, steal it)
Validate sparse attention by **forcing full-key selection and requiring exact
reproduction of dense attention**. Colibrì validates DSA this way; Arc has no
equivalent (v4_e2e.rs has one synthetic finite/deterministic check). This test
would have caught F1 at ZERO GPU cost. Also: teacher-forcing oracle vs
transformers (they hold 30-32/32); stamp-mandatory format registry (an
entropy-coded tensor has no expected_bytes(O,I) ⇒ size-inference structurally
impossible ⇒ stamp-gated dispatch must precede inference — a real constraint
for any Arc container with data-dependent size).

## F5 (IGNORE) — their CUDA backend
Zero cudaGraph calls, no cuBLAS/CUTLASS, scalar FP32 kernels (one block per
output element, shared-mem tree reductions), only tensor-core path is a dead
wmma s4 fragment. arc-cuda-graph + our QTIP GEMV work is well ahead. Their
fmt=6 E8/IQ3 lattice + block-diagonal FWHT is QuIP#/QTIP territory but below
trellis on rate-distortion. Disk-streaming/dual-SSD/expert-placement JIT =
consumer answers to a problem we don't have. TWO exceptions worth borrowing:
**batch-union** (read each unique expert once per BATCH, not per position —
maps onto batch_load_probe work) and their measurement that **routing is 71.6%
predictable one layer ahead** (prior for any future expert-offload path).

## DISCIPLINE (Jish, 2026-08-14)
**Never test 5 things at once — you can't detect the menace then.** One variable
per experiment, especially on rented GPU time.
