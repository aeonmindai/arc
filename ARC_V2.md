# Arc v2 — The Roadmap

**Reader:** AI coding agent or new engineer arriving cold. By the time you finish this document, you should know exactly what Arc is building, why each piece exists, which paper proves it works, and where the reference code lives.

This document is the source of truth. If anything else in the repo contradicts it, this wins.

---

## What Arc is

Arc is a Rust inference engine, forked from `mistral.rs`. It is built to serve frontier MoE models on radically less hardware by composing **techniques from peer-reviewed published research with public code** — QTIP 2-bit weights, TurboQuant K4/V3 KV, TD-MoE Tucker decomposition, model-native sparse attention.

**The only model Arc has ever served is DeepSeek V4 Flash.** DeepSeek V4 Pro, Kimi K2.6 and GLM 5.1 appear throughout this document as **roadmap targets** — they have never been loaded here. This is a roadmap: unless a number is explicitly labelled *measured*, it is a target.

**What is measured** (protocols and raw-artifact provenance in [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md)):

- **74.19 GB at 2.09 bits/param.** V4 Flash (284B total / 13B active) serves from one H200, where **every published V4-Flash config we could find needs four GPUs** — native checkpoint ≈160 GB, smallest published config 4×H200, and the one W4A16 quant is 143 GB whose own model card states that TP=1 OOMs on a single 141 GB H200. *This is checkable, and we want it checked: a published single-GPU V4-Flash config, or any engine loading it in ≤141 GB, refutes it.*
- **GSM8K 1270/1319 = 96.3% ±1.0pp**, full test set, 0-shot, 0 degenerate / 0 truncated / 0 errors. The reference model's published 90.8 is **8-shot** — a different and easier protocol, so this is *not* a like-for-like win over it. (This supersedes the provisional **87.0%** n=100 subset, which was measured on decode math since changed. The `docs/` pages that still quoted it were synced on 2026-08-17; where 87.0% survives it is explicitly labelled superseded and kept only as the historical record.)
- **111.69 tok/s aggregate decode at B=256, $12.06/Mtok**, rising monotonically from B=1, `effective_B == B` on all seven batch rows, 0 errors across 505 requests.

**No side-by-side run against SGLang, vLLM, or any other engine has ever been performed — not once — and no third-party performance number appears anywhere in this repo.** `arc-bench` is already vendor-abstracted and drives any OpenAI-compatible server, so a genuine baseline is one rental away; until that run exists, this document states what Arc does and what other engines *implement*, never how fast they are.

The thesis: every frontier lab is implementing the same five-feature consensus stack (FP8/FP4 + FlashAttn3 + Paged/Radix KV + EAGLE + TP+EP). They're racing each other on execution speed of one recipe. Meanwhile, ~10 published techniques with public code sit unexploited because they require crossing team boundaries the labs haven't crossed.

**Arc's edge is reading the literature, not inventing new science.** Every bet below has a paper, has code, and is shippable as engineering work — not research.

---

## Hardware target

**Primary**: single DGX-B200 node (4× B200, 768 GB HBM, 32 TB/s aggregate bandwidth, NVLink).
**Stretch**: single B200 for GLM 5.1 and DeepSeek V4 Flash (smaller models that fit).
**Future**: AMD MI300X/MI325X and Apple M3 Ultra (same software, different backends — orthogonal to v2 work).

All numbers below assume a single DGX-B200 node serving DeepSeek V4 Pro (1.6T total / 49B active / 1M context).

---

## The metric targets

| Metric | Arc v2 target |
|---|---|
| Single-user decode tok/s @ 32K context | 5,000–7,000 |
| Single-user decode tok/s @ 1M context | 3,000–5,000 |
| Aggregate tok/s at full concurrency | 250,000–350,000 |
| Concurrent users at 1M context | 120–300 |
| Quality (perplexity drift from BF16) | ~1.5% |
| Model size in HBM | 320 GB |

These are Arc targets only. No apprentice-mode research bet; no Tucker-MoE-at-aggressive-rank hand-waving.

> ⚠️ **A `SGLang today` baseline column and a derived `Multiple` column were removed
> from this table on 2026-08-17.** They carried figures (`~100` and `~30` single-user,
> `~25,000` aggregate, `~10` concurrent users) that **we have never run and cannot
> source**, and every multiple was computed against them. This is the same defect class
> as the fabricated competitor benchmark deleted from `docs/PEAK_INFERENCE.md` this
> week. **Standing rule: name what another engine has; never state how fast it is
> unless we ran it ourselves, on stated hardware, with the command recorded.** If a
> baseline is wanted here, produce it with `arc-bench`, which is already
> vendor-abstracted and can drive any OpenAI-compatible server.

---

## The nine techniques

Each technique is filed under one or more domains. Domains and their headline numbers:

- **Domain 1 — Weight storage** (model footprint in HBM): target 320 GB for V4 Pro
- **Domain 2 — KV memory per user** (concurrent users at long context): target 3 GB/user at 1M context
- **Domain 3 — Per-token decode speed** (single-user tok/s): three sub-bottlenecks
- **Domain 4 — Aggregate throughput** (utilization across all requests): target ~4× over Arc's own measured scheduling baseline (a cross-engine multiple would need a run we have not done)
- **Domain 5 — Quality preservation** (near-lossless stack)

### 1. NVFP4 weights (Domain 1, foundation)

**What it does:** Use Blackwell's native 4-bit floating-point tensor cores for the weight matmul. Hardware-given; no algorithmic choice.

**Why:** Free 4× over BF16. The B200 was designed for this.

**Paper:** None needed — NVIDIA Blackwell whitepaper.

**Code reference:** `research/code/06_foundation/tensorrt_llm/` (NVIDIA's reference implementation).

**Implementation status in Arc:** partially done via Candle + cuBLASLt; needs verification on B200.

### 2. QTIP trellis quantization for weights (Domain 1)

**What it does:** Compresses weights to 2 bits per parameter using trellis-coded quantization on top of incoherence processing (Hadamard rotation). Stacks on NVFP4 → effective 2-bit weights at hardware speed.

**Why:** 2× smaller than NVFP4 alone, validated lossless to 405B.

**Paper:** `research/01_weight_compression/qtip_trellis_coded_quantization.pdf` (Tseng et al., Cornell, NeurIPS 2024).

**Code reference:** `research/code/01_weight_compression/qtip/` — official Cornell-RelaxML implementation.

**Implementation plan:**
1. Read `qtip/qtip/codebook.py` for the bitshift trellis + 3INST/HYB lookup codes
2. Port the Viterbi quantize + dequantize routines to Arc's CUDA kernel layout
3. Integrate as a new ISQ type in `mistralrs-quant/src/qtip/`
4. Validate against published Llama-2/3 perplexity numbers

### 3. TD-MoE Tucker decomposition (Domain 1)

**What it does:** Stacks all MoE experts in a layer into a 3D tensor, applies whitening, then Tucker decomposition. 20% lossless compression on MoE expert weights.

**Why:** Composes with NVFP4 + QTIP. Brings V4 Pro from 400 GB (QTIP only) to 320 GB.

**Paper:** `research/01_weight_compression/td_moe_tucker_decomposition_moe.pdf` (Xu et al., HKUST/Huawei, ICLR 2026).

**Code reference:** No public repo yet. Paper appendix has Algorithm 1 + Algorithm 2 (rank allocation). Uses PyTorch + TensorLy.

**Implementation plan:**
1. Implement Algorithm 1 (cross-expert tensorization → multi-linear whitening → Tucker → re-coloring) at model-load time, not per-request
2. Output: stored core tensor + factor matrices, replacing the original expert tensor
3. Inference path: standard MoE forward, but expert weights are reconstructed via Tucker factors. No runtime overhead.
4. Calibration uses 256 samples per the paper.

### 4. TurboQuant K4/V3 KV cache (Domain 2)

**What it does:** Walsh-Hadamard rotation + Lloyd-Max codebook quantization on K (4-bit) and V (3-bit), with FP16 recent window. Lossless on LongBench.

**Why:** This is your own shipped work. Don't break it. Stack other techniques on top.

**Paper:** `research/02_turboquant_kv/turboquant_arc_iclr2026.pdf`.

**Code reference:** Arc's own `mistralrs-quant/src/turboquant/` + `arc-turbo/`.

**Implementation status: NOT shipped — experimental and off by default.** The
algorithm, codebooks and WHT are implemented and unit-tested, but: the eager KV
path is opt-in via `ARC_TURBOQUANT_KV=1`; the paged kernel exists at **head_dim
128 only** and auto-falls back to the unquantized cache for everything else
(including every MLA model); there is **no kernel at head_dim 512**, so
DeepSeek V4 cannot use it at all; and the prefix cache auto-disables under it.
**No TurboQuant serving run has ever been measured.** The "4.27×" figure that
circulated in this repo was format arithmetic — bytes per token at 3.5 bits
versus BF16 — never a forward pass. Retracted 2026-08-17.

### 5. YOCO (You Only Cache Once) (Domain 2)

**What it does:** Cross-layer KV sharing. Most layers reuse the same compressed KV instead of each holding its own.

**Why:** Multiplies on top of TurboQuant — 2× more concurrent users at same memory budget.

**Paper:** `research/12_long_context/yoco_you_only_cache_once.pdf` (Microsoft).

**Code reference:** `research/code/01_weight_compression/microsoft_unilm/YOCO/`.

**Implementation plan:**
1. Read `unilm/YOCO/yoco/` for the two-stage architecture (self-decoder + cross-decoder)
2. Add a "YOCO mode" flag at model load that re-wires the model graph: half the layers compute KV, the other half reuse
3. Combined with TurboQuant: KV stored at 3.5 bit, shared across N/2 layers → ~9× total compression vs BF16-per-layer

**Caveat:** YOCO requires the model to support cross-layer KV sharing in its architecture. For models that don't (e.g., V4 Pro as released), this is a deploy-time architectural rewrite — borderline of the "no offline conversion" rule. Treat as opt-in for customer-uploaded models that already use YOCO.

### 6. DuoAttention (Domain 2)

**What it does:** Distinguishes "retrieval heads" (need full KV) from "streaming heads" (need only sliding window). Allocates KV memory per head type.

**Why:** Additional 2–4× KV reduction on top of TurboQuant + YOCO.

**Paper:** `research/16_underexploited/duo_attention.pdf` (MIT Han Lab).

**Code reference:** `research/code/03_per_token_speed/duo_attention/` — official MIT.

**Implementation plan:**
1. Read `duo-attention/duo_attn/patch/` for the head categorization logic
2. Run their head-importance optimization (~30 min per model on one A100) at model load
3. At inference time: retrieval heads get full TurboQuant K4/V3; streaming heads get a fixed-size sliding window
4. Track per-head budget; recompose attention output normally

### 7. Turbo Sparse activation pruning (Domain 3a)

**What it does:** dReLU activation function applied to gate AND up projections of the FFN. Pushes activation sparsity from ~75% to ~95% on MoE models. Per-token bandwidth drops 3×.

**Why:** This is the biggest single decode-speed lever for MoE models. Quality measurably improves (TurboSparse-Mixtral-47B beats baseline on OpenLLM).

**Paper:** `research/01_weight_compression/moe_lrd_low_rank_decomposition_2024.pdf` (the file I originally mislabeled — actual title is "Turbo Sparse," Song et al., SJTU-IPADS 2024).

**Code reference:** `research/code/03_per_token_speed/powerinfer/` (PowerInfer engine integrates the model; the dReLU model checkpoints are at huggingface.co/PowerInfer).

**Implementation plan:**
1. Build a small fine-tuner that substitutes SwiGLU with dReLU and continues pretraining for ~150B tokens
2. This is one-time per model architecture (~1 week of compute on a small cluster)
3. At serving time: sparse predictor identifies active neurons per token; Arc loads only those weights from HBM
4. Composes with NVFP4 + QTIP at the storage layer (no conflict)

**Caveat:** This requires a ~1-week fine-tune per model. That's not pure runtime, but it's a one-time per-architecture engineering step, not per-request. Treat as deploy-time work, like the 3-minute Tucker SVD.

### 8. MoBA / NSA sparse attention (Domain 3b)

**What it does:** Each token attends only to a sparse subset of context blocks (MoBA) or a hierarchical sparse pattern (NSA). 5–10× attention speedup at long context with no quality loss.

**Why:** Without this, attention cost grows with context length until it dominates step time at 1M context. The tok/s ceiling that follows is **unmeasured** — on Arc and on every other engine; the `~30` previously printed here came from the unsourced SGLang column struck from the target table and was never run by anyone. With it, attention becomes O(n log n) or O(n √n).

**Papers:**
- `research/12_long_context/moba_mixture_of_block_attention.pdf` (Moonshot, shipped in Kimi K2/K2.6)
- `research/09_sparse_attention/deepseek_native_sparse_attention.pdf` (DeepSeek, shipped in V3+)

**Code references:**
- `research/code/03_per_token_speed/moba/` — official MoonshotAI
- `research/code/03_per_token_speed/deepseek_v3_nsa/` — DeepSeek V3 inference codebase including NSA

**Implementation plan:**
1. Detect at model load whether the model ships with MoBA-style or NSA-style attention (Kimi K2.6 = MoBA, DeepSeek V3+ = NSA)
2. If yes: use the model's native sparse attention path
3. If no: fall back to standard attention. Do NOT attempt to retrofit sparse attention onto non-trained models in v2.

### 9. SageAttention INT8 (Domain 3b)

**What it does:** Quantize the attention computation itself to INT8 — Q, K, V, and the softmax inputs are all INT8. 2× attention throughput, bit-exact equivalence to FP16 attention in published benchmarks.

**Why:** Free 2× on the attention kernel cost, independent of sparse-attention choice. Stacks on MoBA/NSA/standard attention.

**Paper:** `research/16_underexploited/sage_attention_int8.pdf` + `sage_attention_2.pdf` (thu-ml, 2024–2025).

**Code reference:** `research/code/03_per_token_speed/sage_attention/` — official thu-ml.

**Implementation plan:**
1. Read `SageAttention/sageattention/quant_per_block.py` for the per-block INT8 quantization scheme
2. Port the kernel to Arc's attention path
3. Validate bit-exactness against FP16 baseline on Llama-3 prompts

### 10. MTP / EAGLE-3 + MagicDec (Domain 3c)

**What it does:** Reduce the number of forward passes needed per emitted token via speculative decoding.

- **MTP** (Multi-Token Prediction): built into DeepSeek V3, V4 family — auxiliary heads predict 2–4 future tokens, target verifies. ~1.8× decode.
- **EAGLE-3**: trained draft head for models that don't ship MTP. ~2.5× decode.
- **MagicDec**: speculative decoding tuned specifically for long context. Additional 1.5× at 1M context, composes with EAGLE-3.

**Why:** Single biggest forward-pass-amortization lever. Mandatory.

**Papers:**
- `research/11_models/deepseek_v3.pdf` (MTP architecture)
- `research/10_foundation/eagle_v3.pdf`
- `research/15_speculative_extensions/magicdec_speculative_long_context.pdf`

**Code references:**
- MTP: `research/code/03_per_token_speed/deepseek_v3_nsa/` (the V3 repo has MTP)
- EAGLE-3: `research/code/03_per_token_speed/eagle/` — official SafeAILab
- MagicDec: `research/code/03_per_token_speed/magicdec/` — official Infini-AI-Lab

**Implementation plan:**
1. If model ships MTP weights (DeepSeek V3+, V4 family): use them. Adapter into Arc's spec-decoding path. ~1.8× decode.
2. Otherwise: load an EAGLE-3 draft head for the target model. ~2.5× decode.
3. At long context (>64K), layer MagicDec on top of either. Additional 1.5×.

---

## Aggregate throughput techniques (Domain 4)

These stack on top of all the per-token speedups. Each addresses a different source of GPU idle time.

### 11. Two-Batch Overlap (TBO)

**What it does:** Mix prefill and decode batches at the kernel level so they execute concurrently on different SMs.

**Paper:** `research/14_serving_systems/sarathi_serve_chunked_prefill.pdf` (the TBO companion to Sarathi-Serve).

**Code reference:** `research/code/04_aggregate_throughput/sarathi_serve/` (Microsoft) — and the TBO patterns are referenced in `research/code/06_foundation/sglang/python/sglang/srt/batch_overlap/`.

**Implementation plan:** Port the SGLang TBO patterns into Arc's scheduler. Add a `--tbo` flag.

### 12. Cross-request expert affinity batching

**What it does:** When N users are routing to the same MoE expert in the same step, load that expert from HBM once and serve all N. Amortizes expert load cost across users.

**Why:** Expected to improve aggregate throughput on MoE workloads at high concurrency. The `2–3×` previously stated here was **projected, with no paper and no run behind it** — this technique has no canonical paper to cite, so the size of the gain is an open question until Arc measures it.

**Paper:** No single canonical paper; described in DeepSeek V3 § 3.2 (EPLB section) and SGLang scheduling docs.

**Code reference:** `research/code/06_foundation/sglang/python/sglang/srt/eplb/` for the expert-load-balancer logic.

**Implementation plan:** Group decoding users by expert affinity in 50ms windows; serve all users for the same expert from a single weight load.

### 13. Chunked prefill (Sarathi-Serve)

**What it does:** Break long prefill into small chunks so it doesn't block ongoing decoding.

**Paper:** `research/14_serving_systems/sarathi_chunked_prefill.pdf`.

**Code reference:** `research/code/04_aggregate_throughput/sarathi_serve/`.

**Implementation plan:** Read `sarathi-serve/sarathi/core/scheduler/` for the chunking logic; port to Arc's scheduler.

### 14. GPU-autonomous decode (your arc-cuda-graph)

**What it does:** The GPU runs the entire decode loop without per-token CPU sync. Uses CUDA 12.4+ conditional graph nodes.

**Why:** Removes ~20 μs of CPU↔GPU sync per token. Marginal at low tok/s; meaningful at 2000+ tok/s.

**Paper:** No paper; this is Arc's own engineering bet.

**Code reference:** Arc's own `arc-cuda-graph/src/` — already 80% built.

**Implementation status:** Complete the autonomous-loop pass. Add conditional graph nodes for sampling decisions.

### 15. Holographic prefix cache

**What it does:** Encode every conversation prefix as a 10,000-bit hypervector; look up by Hamming similarity instead of exact tree-walk match. Cache hits work on paraphrases.

**Why:** An exact-prefix trie reuses KV only when a request opens with byte-identical text; a similarity index can also hit on paraphrases. **Neither hit rate has been measured** — the `~30%` attributed to RadixAttention and the `~60%` projected for this design were both unsourced, and this technique has no application paper to cite for them. Quantifying the gain requires a hit-rate measurement over a real chat trace, which `arc-bench` can produce.

**Paper:** No single application paper; foundations in `research/07_holographic_cache/` (Plate 1995 HRR, Kanerva SDM, and the 2021 surveys).

**Code reference:** No public LLM application code. Implementation is novel engineering. Start from `research/code/06_foundation/sglang/python/sglang/srt/mem_cache/radix_cache.py` as the structural reference, then swap the trie for a hypervector index.

**Implementation plan:**
1. Implement hypervector bind/bundle/permute ops in Rust
2. Build an HNSW index over conversation-prefix hypervectors
3. Lookup at request time; promote a hit when Hamming distance is below threshold
4. Verify cache hits produce equivalent KV state (or fall back to recompute)

---

## Quality-preserving techniques (Domain 5)

### 16. Differential Transformer

**What it does:** Attention output = softmax(Q1 K1ᵀ) - λ softmax(Q2 K2ᵀ). Cancels noise. Reduces hallucinations measurably.

**Why:** Drop-in quality boost. Validated by Microsoft + Tsinghua at ICLR 2025.

**Paper:** `research/16_underexploited/differential_transformer_microsoft.pdf`.

**Code reference:** `research/code/01_weight_compression/microsoft_unilm/Diff-Transformer/`.

**Implementation plan:** Same caveat as YOCO — requires model to be trained with differential attention or distilled. For v2, ship as opt-in for models that ship native diff-attention; defer retrofit.

---

## What is NOT in v2 (and why)

| Bet | Why we cut it |
|---|---|
| **Apprentice mode** (online Hopfield/TTT side channel) | No published validation of the integrated product. TTT-E2E requires training-time integration. Pure runtime attachment is unvalidated research. Dropped for v2; revisit if budget allows. |
| **Architecture conversion** (Mamba-in-Llama, Mercat) | Requires offline distillation, which violated your "runtime only" constraint. |
| **Aggressive Tucker (8×)** | Validated only at 4×. The 8× claim was hand-waving. |
| **Compressed sensing for KV** | Math is right; no production code. Pure research bet. |
| **xLSTM / RWKV-7 / Hyena / Griffin** | Architecture alternatives. Out of scope until a customer ships a frontier model in one of these architectures. |

---

## Build order

Phase 1 — Validate the foundation (months 1–2):
1. NVFP4 hardware path verified on B200
2. QTIP integrated, perplexity matches paper on Llama-3
3. TD-MoE implemented at model load, perplexity matches paper on Mixtral
4. TurboQuant reaches a real serving path: a kernel beyond head_dim 128, then
   the first measured A/B (quality + throughput) — today it is off by default
   and unmeasured

Phase 2 — Per-token speed (months 2–4):
5. Turbo Sparse fine-tune pipeline running; first MoE model converted
6. MTP / EAGLE-3 integration
7. SageAttention kernel port

Phase 3 — Concurrent users (months 4–6):
8. YOCO + DuoAttention integrated for compatible models
9. MagicDec layered on long-context spec
10. Holographic prefix cache prototype

Phase 4 — Aggregate throughput (months 5–7):
11. TBO + chunked prefill from Sarathi-Serve
12. Cross-request expert affinity batching
13. Finish arc-cuda-graph autonomous decode

Phase 5 — Validation (month 7):
14. End-to-end benchmark vs SGLang on V4 Pro / Kimi K2.6 / GLM 5.1
15. Third-party benchmark publication

---

## How to use this document as an AI agent

1. Pick a technique by number (1–16).
2. Read the corresponding paper in `research/{domain}/{filename}.pdf`.
3. Open the reference code in `research/code/{domain}/{repo}/`.
4. The "Implementation plan" subsection is your starting checklist.
5. The "Caveat" subsections flag where the paper's assumptions might not match Arc's constraints.
6. When done, validate against the perplexity / tok/s numbers in the paper's Table 1.
7. Update `mistralrs-core/CHANGELOG.md` and benchmark in `mistralrs-bench/`.

If a paper's path or code path is wrong in this document, fix it here first. This document is the source of truth.

---

## Source-of-truth files

- `ARC_V2.md` (this file) — what we're building
- `research/FRONTIER_GAPS.md` — why we're building these specific things
- `research/code/CODE_INDEX.md` — paper → code mapping
- `research/INDEX.md` — paper-by-paper rationale
- `CLAUDE.md` — house rules for AI agents in this repo

---

## Quick reference: the nine core techniques

| # | Technique | Domain | Paper | Code |
|---|---|---|---|---|
| 1 | NVFP4 | weight storage | NVIDIA whitepaper | `code/06_foundation/tensorrt_llm/` |
| 2 | QTIP 2-bit | weight storage | `01_weight_compression/qtip_trellis_coded_quantization.pdf` | `code/01_weight_compression/qtip/` |
| 3 | TD-MoE 20% Tucker | weight storage | `01_weight_compression/td_moe_tucker_decomposition_moe.pdf` | (paper only) |
| 4 | TurboQuant K4/V3 | KV memory | `02_turboquant_kv/turboquant_arc_iclr2026.pdf` | Arc private |
| 5 | YOCO | KV memory | `12_long_context/yoco_you_only_cache_once.pdf` | `code/01_weight_compression/microsoft_unilm/YOCO/` |
| 6 | DuoAttention | KV memory | `16_underexploited/duo_attention.pdf` | `code/03_per_token_speed/duo_attention/` |
| 7 | Turbo Sparse | per-token FFN | `01_weight_compression/moe_lrd_low_rank_decomposition_2024.pdf` | `code/03_per_token_speed/powerinfer/` |
| 8 | MoBA / NSA | per-token attention | `12_long_context/moba_mixture_of_block_attention.pdf` + `09_sparse_attention/deepseek_native_sparse_attention.pdf` | `code/03_per_token_speed/moba/` + `code/03_per_token_speed/deepseek_v3_nsa/` |
| 9 | SageAttention | per-token attention | `16_underexploited/sage_attention_int8.pdf` | `code/03_per_token_speed/sage_attention/` |
| 10 | MTP / EAGLE-3 + MagicDec | per-token spec | `11_models/deepseek_v3.pdf` + `10_foundation/eagle_v3.pdf` + `15_speculative_extensions/magicdec_speculative_long_context.pdf` | `code/03_per_token_speed/eagle/` + `code/03_per_token_speed/magicdec/` |
| 11 | TBO + Sarathi | aggregate | `14_serving_systems/sarathi_serve_chunked_prefill.pdf` | `code/04_aggregate_throughput/sarathi_serve/` |
| 12 | Expert affinity | aggregate | DeepSeek V3 §3.2 | `code/06_foundation/sglang/python/sglang/srt/eplb/` |
| 13 | GPU-autonomous decode | aggregate | (own) | `arc-cuda-graph/` |
| 14 | Holographic prefix cache | aggregate | `07_holographic_cache/` surveys | (novel; base from `code/06_foundation/sglang/...mem_cache/`) |
| 15 | Differential Transformer | quality | `16_underexploited/differential_transformer_microsoft.pdf` | `code/01_weight_compression/microsoft_unilm/Diff-Transformer/` |

That's the v2 stack. Ten months of focused engineering. Zero research risk. Every line traces to a paper and a repo.
