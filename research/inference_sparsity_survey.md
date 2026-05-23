# Inference-Time Activation Sparsity for LLMs Without Retraining — Survey

**Author:** Research Agent 1
**Date:** 2026-05-22
**Scope:** Methods that achieve activation-level sparsity at inference time for LLMs, without full retraining of the base model. Calibration on small datasets is allowed.

---

## 1. Executive Summary

| Method | Training requirement | Sparsity achieved | Quality cost | Status | Citation |
|---|---|---|---|---|---|
| **TEAL** | None (offline thresholds from ~128 calib samples) | 25% near-zero loss, 40% minimal, 50% on Llama-2/3, Mistral 7B-70B | 40% Llama-2-7B: 5.07 → 5.22 PPL (+0.15); Llama-3-8B 5.87 → 6.21 (+0.34) | ICLR 2025 Spotlight; public Triton kernels (FP16) | [arXiv 2408.14690](https://arxiv.org/abs/2408.14690) |
| **CATS** | Calibration; *optional* 1%-param fine-tune to recover | 50% MLP sparsity | <1-2% downstream drop; ~15% custom-kernel latency win | COLM 2024; CUDA kernel in repo | [arXiv 2404.08763](https://arxiv.org/abs/2404.08763) |
| **GRIFFIN** | None (zero calibration) | 50% FFN sparsity at sequence level | "Almost no loss" Llama-2-13B; 1.25-1.29x latency | NeurIPS 2024; small reference code | [arXiv 2404.01365](https://arxiv.org/abs/2404.01365) |
| **R-Sparse** | Calibration (SVD-based) | 50% model-level | ~Comparable to dense; 43% end-to-end speedup w/ kernels | 2025; VITA-Group GitHub | [arXiv 2504.19449](https://arxiv.org/abs/2504.19449) |
| **La RoSA** | None (orthogonal rotation) | 40% (LLaMA-2-7B) | 0.17 PPL gap at 40%; 1.30x speedup; beats TEAL by 1.77% on zero-shot | ICML 2025 | [arXiv 2507.01299](https://arxiv.org/abs/2507.01299) |
| **WiSparse** | Calibration (evolutionary search) | 50% (Llama-3.1) | 97% of dense; 21.4% end-to-end speedup | 2026 preprint | [arXiv 2602.14452](https://arxiv.org/abs/2602.14452) |
| **Polar Sparsity** | Calibration; predictors | MLP @ small batch, attention heads @ large batch | "Without compromising accuracy"; 2.2x at scale | NeurIPS 2025; sparsity-aware GPU kernels | [arXiv 2505.14884](https://arxiv.org/abs/2505.14884) |
| **DejaVu** | Predictor MLPs trained from calib data | 75-85% contextual; OPT-175B 2-6x | Reported "no quality loss" on OPT | ICML 2023 Oral; FMInference/DejaVu (older OPT-focused stack) | [arXiv 2310.17157](https://arxiv.org/abs/2310.17157) |
| **PowerInfer** | Adopts DejaVu predictors; needs ReLU/dReLU model | ~90% with ProSparse/Turbo-Sparse weights | Matches dense (paired with sparsified weights) | SOSP 2024; SJTU-IPADS/PowerInfer | [arXiv 2312.12456](https://arxiv.org/abs/2312.12456) |
| **TurboSparse / ProSparse** | **Continued pretraining (out of scope but cited)** | ~90% FFN sparsity | Comparable; 2-5x decode | 2024 papers | [arXiv 2406.05955](https://arxiv.org/abs/2406.05955), [arXiv 2402.13516](https://arxiv.org/abs/2402.13516) |
| **Q-Sparse** | **Training/STE required (semi-out-of-scope)** | Full top-K activation sparsity | "Baseline LLM performance" | Microsoft Research 2024 | [arXiv 2407.10969](https://arxiv.org/abs/2407.10969) |
| **LayerSkip** | **Training (layer-dropout)** required | Per-token early exit | Up to 2.16x on summarization | ACL 2024; facebookresearch/LayerSkip | [arXiv 2404.16710](https://arxiv.org/abs/2404.16710) |
| **EE-LLM** | **Training of exit heads** required | Early exit | Outstanding; no compromise reported | ICML 2024; pan-x-c/EE-LLM (Megatron) | [arXiv 2312.04916](https://arxiv.org/abs/2312.04916) |
| **Mixture of Depths (MoD)** | **Pretraining with MoD routers** (out of scope) | k tokens / layer | Match baseline at lower FLOPs | 2024 DeepMind | [arXiv 2404.02258](https://arxiv.org/abs/2404.02258) |
| **A-MoD (attention-as-router)** | None (parameter-free) | Token routing | Reuses pretrained model | 2024 | [arXiv 2412.20875](https://arxiv.org/abs/2412.20875) |
| **LazyLLM** | None | Long-context token pruning | Maintains accuracy; 2.34x prefill on LLaMA-2-7B MultiDoc-QA | Apple/Meta ICML 2024 Workshop | [arXiv 2407.14057](https://arxiv.org/abs/2407.14057) |
| **FastV** | None | Visual-token pruning (multimodal only) | Minimal drop for LLaVA | ECCV 2024 | [arXiv 2403.06764](https://arxiv.org/abs/2403.06764) |
| **ShadowLLM** | Predictor trained on calib data | Contextual sparsity (attention + neurons) | +15% accuracy and +20% speed vs DejaVu | EMNLP 2024; abdelfattah-lab/shadow_llm | [arXiv 2406.16635](https://arxiv.org/abs/2406.16635) |
| **SparseInfer** | None (training-free sign-bit predictor) | Same regime as ReLU-based sparsity | <1% accuracy loss | 2024 | [arXiv 2411.12692](https://arxiv.org/abs/2411.12692) |
| **TDA / "First Activations Matter"** | None | Dynamic activation across SwiGLU/ReLU/GeLU | 18-25% gen speedup, no significant drop | 2024 | [arXiv 2408.11393](https://arxiv.org/abs/2408.11393) |
| **DuoGPT** | Activation-aware calibration | Dual sparsity (weight + activation) | +9.17% accuracy vs SOTA structured pruning at iso-1.39x speedup | NeurIPS 2025; code public | [arXiv 2506.20194](https://arxiv.org/abs/2506.20194) |
| **Wanda / SparseGPT** | Calibration only | 50% weight sparsity (less relevant — weights not activations) | Mild PPL drop | ICLR 2024 / ICML 2023 | [arXiv 2306.11695](https://arxiv.org/abs/2306.11695) |
| **SqueezeLLM (D&S)** | Calibration | 3-bit dense + 0.45% outlier sparse | <2.1x improvement over baselines | ICML 2024 | [arXiv 2306.07629](https://arxiv.org/abs/2306.07629) |
| **MoE++ / ZEDA** | Self-distillation (light) | MoE skip half experts | Matches w/ small distill cost | 2025 preprints | various |

---

## 2. Top 5 Methods Ranked by ROI for Arc (Least Code Change Wins)

Arc is a Rust/CUDA inference engine forked from mistral.rs targeting BF16 SOTA (120 tok/s on B200). The current bottleneck for single-stream decode is HBM bandwidth on FFN gate/up/down — exactly what activation sparsity attacks. Ranking by *implementation effort vs. expected wall-clock win on B200/H100*:

### #1 — **TEAL** (Training-Free Activation Sparsity)
- **Why first:** *Zero calibration cost on supported models* (pre-baked thresholds for Llama-2/3, Mistral exist in repo), magnitude-based logic, drops cleanly into any SwiGLU/SiLU MLP. Sparsifies all seven projections per block (Q/K/V/O + gate/up/down), so it stacks with attention-side wins.
- **Public code:** `https://github.com/FasterDecoding/TEAL` (Triton reference kernel).
- **Win for Arc:** rewrite the Triton kernel as a Rust+CUDA selective-column-load gemv. The L2-cache + column-major layout described in §3 of the paper is straightforward to port. Expected: 1.5-1.8x decode speedup at 40-50% sparsity per their numbers.
- **Caveat:** Triton ref is FP16-only (no BF16 atomic_add in their build). Arc is BF16-native, but the kernel does not need atomic_add for gemv — it needs splitK reduce. So the BF16 port is easy.

### #2 — **GRIFFIN** (Prompt-Prompted Adaptive Structured Pruning)
- **Why second:** *Zero training, zero calibration, sequence-level expert selection.* The "flocking" observation means you compute the active FFN expert set *once* from the prompt prefix, then reuse for all decode steps. Trivially compatible with CUDA graph capture (Arc's existing strength) because the active set is static after prefill.
- **Public code:** Yes (referenced in [arXiv 2404.01365](https://arxiv.org/abs/2404.01365)).
- **Win for Arc:** lowest-risk way to get 1.25-1.29x latency on Llama-2-13B / Gemma 7B *while keeping the CUDA graph topology fixed*. Combine with #1 for compounding.

### #3 — **La RoSA** (Layerwise Rotated Sparse Activation)
- **Why third:** *No training, no calibration.* Uses a fixed orthogonal Hadamard-style rotation before top-K. Reports beating TEAL by 1.77% accuracy and CATS by 17.14% at the same sparsity — strongest accuracy/sparsity Pareto in the 2025 literature. 1.30x wall-clock speedup at 40% sparsity on LLaMA-2-7B.
- **Public code:** Paper-only as of submission; ICML 2025.
- **Win for Arc:** if TEAL's quality is acceptable, La RoSA is strictly better. The rotation can be absorbed into weight matrices offline (W' = R W, b' = R b) which means the inference cost is identical to TEAL but the threshold can sit higher (more sparsity at equal quality). This is the most-likely "next gen" after a TEAL port.

### #4 — **CATS** (Contextually-Aware Thresholding)
- **Why fourth:** *Best-documented downstream evaluation* (Mistral-7B and Llama-2-7B/13B), Stanford's Scaling Intelligence Lab maintains it. Lighter than TEAL because it only sparsifies the MLP gate output (not all seven projections), which limits ceiling but reduces risk and integrates faster.
- **Public code:** `https://github.com/ScalingIntelligence/CATS` (custom CUDA kernel in repo).
- **Win for Arc:** 18-21% end-to-end token throughput, 99% downstream quality. Use as the *safe baseline* shippable in V1 if TEAL has surprise regressions on a specific model.
- **Caveat:** Needs a 1% LoRA-tune to fully recover quality; without it, 1-2% drop. If Arc demands no-tune at all, prefer TEAL or La RoSA.

### #5 — **SparseInfer** (Sign-Bit Predictor) — niche bet
- **Why fifth:** *Cheapest possible predictor.* Compares only sign bits of inputs and weights to predict ReLU-like zeros. Doesn't help SwiGLU directly, but it's an extremely small kernel that could be fused into the gate projection for *almost-free* sparsity prediction on any model the user has switched to ReLU/dReLU via a checkpoint swap (ReluLLaMA, ProSparse-LLaMA, TurboSparse-Mistral).
- **Public code:** Paper-only as of writing.
- **Win for Arc:** if Arc decides to support *swap-in* ReLU-fied weight files (which Microsoft, Tsinghua and Apple all release on HF), SparseInfer is a one-shot 0.5x compute reduction on FFN with no calibration.

**Honorable mention #6 — Polar Sparsity** for the *batched serving* case (Arc-engine multi-tenant). MLP sparsity wins at batch=1, attention-head sparsity wins at batch≥16. The right answer for a serving stack is *both regimes*. Worth doing after #1-3.

---

## 3. Detailed Write-Up Per Method

### Déjà Vu (Liu et al. 2023, ICML Oral)
- **Mechanism:** Two small MLPs predict per-layer which FFN neurons and attention heads to activate, given the current input. Predictors are trained on activation traces gathered from the base LLM.
- **Sparsity:** "LLMs have up to 85% contextual sparsity for a given input" ([arXiv 2310.17157](https://arxiv.org/abs/2310.17157)).
- **Quality cost:** Reported "without compromising LLM's quality or in-context learning ability" on OPT-175B and BLOOM-176B.
- **Training:** *Predictor MLPs require training on activation traces*. The base model is not retrained.
- **Public code:** `https://github.com/FMInference/DejaVu` (older, OPT-focused, last meaningful PR 2024-ish, issue activity through 2025).
- **MoE:** Not designed for MoE; targets monolithic FFN.
- **Attention vs FFN:** Both — sparsifies FFN neurons *and* attention heads.

### PowerInfer (Song et al. 2023, SOSP 2024)
- **Mechanism:** Hybrid CPU/GPU runtime that statically partitions "hot" vs "cold" neurons from offline activation profiling, then uses *DejaVu-style predictors* + GPU-resident hot neurons + CPU-resident cold neurons.
- **Sparsity:** "Adopted DEJAVU's activation sparsity predictor and achieved a large speedup of up to 11.6× over llama.cpp when the GPU memory could not load the entire model" (PowerInfer SOSP paper, [arXiv 2312.12456](https://arxiv.org/abs/2312.12456)).
- **Quality cost:** Matches dense, but *requires the model to actually be sparse* — works best with ReluLLaMA, ProSparse-LLaMA, or TurboSparse-Mistral.
- **Training:** No, but model weights must come from a ReLU/dReLU variant (which *was* retrained).
- **Public code:** `https://github.com/SJTU-IPADS/PowerInfer`. PowerInfer-2 (June 2024) targets smartphones; 11.68 tok/s on TurboSparse-Mixtral-47B on a phone.
- **MoE:** Yes — works on Mixtral and TurboSparse-Mixtral.
- **Attention vs FFN:** FFN only.

### Turbo Sparse (Song et al. 2024)
- **Mechanism:** Replaces SiLU with **dReLU** (ReLU applied to both gate and up projections), then continues pretraining to recover.
- **Sparsity:** "For TurboSparse-Mistral-7B, on average, 90% of the neurons are inactive in each layer. For TurboSparse-Mixtral-47B, this percentage is slightly lower at 85%" ([arXiv 2406.05955](https://arxiv.org/abs/2406.05955)).
- **Quality cost:** Comparable to original; 2-5x decode speedup.
- **Training:** **Out of scope** — requires continued pretraining (Part B of the paper). Part A (dReLU substitution) without retraining loses quality.
- **Public code:** Sparse-LLM/TurboSparse weights on HF.
- **MoE:** Yes (Mixtral variant).
- **Attention vs FFN:** FFN only.

### CATS (Lee et al. 2024, COLM 2024)
- **Mechanism:** Thresholds the *gate* output of SwiGLU; if `|gate(x)|<τ`, set the corresponding hidden activation to zero and skip the up/down column. τ is set per-layer from a small calibration set.
- **Sparsity:** "CATS achieves 50% activation sparsity for MLP layers with a negligible drop in downstream evaluations" ([arXiv 2404.08763](https://arxiv.org/abs/2404.08763)).
- **Quality cost:** "CATS-based models often achieve downstream task performance within 1-2% of their base models without any fine-tuning and even at activation sparsity levels of 50%". With 1% LoRA tuning, performance fully recovers and beats competing techniques.
- **Training:** Optional 1%-param fine-tune; works without it at 1-2% accuracy cost. Calibration is required.
- **Public code:** `https://github.com/ScalingIntelligence/CATS` — Python (81%) + CUDA (13%); ~15% wall-clock latency improvement from custom kernel.
- **MoE:** Not demonstrated; in principle each expert FFN is a SwiGLU so it should port, but no published results.
- **Attention vs FFN:** FFN only.

### ReLU-fication / ReLU² (Mirzadeh et al. 2023, Zhang et al. 2024)
- **Mechanism:** Replace SiLU/GeLU with ReLU or Squared-ReLU, then continued-pretrain. ReLU² showed in [arXiv 2402.03804](https://arxiv.org/abs/2402.03804) the best sparsity/quality/hardware trade-off.
- **Sparsity:** Up to 90% FFN sparsity.
- **Quality cost:** Approximately matches dense after continued pretraining.
- **Training:** **Yes — out of scope** (continued pretraining).
- **Public code:** ReluLLaMA-7B/70B, relu2-30B on HF (SparseLLM org).
- **MoE:** Possible but not the focus.
- **Attention vs FFN:** FFN only.

### Q-Sparse (Wang et al. 2024, Microsoft)
- **Mechanism:** Top-K sparsification of activations during *both* training and inference, with straight-through estimator.
- **Sparsity:** Full sparsity (e.g., 50% top-K).
- **Quality cost:** "Baseline LLM performance with lower inference costs"; established a scaling law for sparsely-activated LLMs.
- **Training:** **Yes — requires from-scratch or continued training**.
- **Public code:** 3rd-party reimpl `nanowell/Q-Sparse-LLM`.
- **MoE:** Compatible with BitNet b1.58 (which is dense), not specifically targeted at MoE.
- **Attention vs FFN:** Both.

### SqueezeLLM Dense+Sparse (Kim et al. 2024, ICML)
- **Mechanism:** Weight quantization to 3-bit dense + ~0.45% outlier weights in CSR-sparse FP16. Not activation sparsity proper.
- **Sparsity:** Sparse portion is small (<1% of weights).
- **Quality cost:** "3-bit quantization significantly reduces the perplexity gap from the FP16 baseline by up to 2.1x" ([arXiv 2306.07629](https://arxiv.org/abs/2306.07629)); 2.3x speedup on A6000.
- **Training:** Calibration only.
- **Public code:** `https://github.com/SqueezeAILab/SqueezeLLM`.
- **MoE:** Yes in principle.
- **Attention vs FFN:** Both (weight-level).
- **Relevance:** Marginal for our purposes — it's a quantization scheme, not activation sparsity.

### Wanda / SparseGPT (Sun et al. 2024 / Frantar & Alistarh 2023)
- **Mechanism:** Wanda prunes the smallest |w_ij · ||x_j||| per output channel. SparseGPT does the same with iterative weight updates.
- **Sparsity:** 50% unstructured or 2:4 semi-structured.
- **Quality cost:** Mild PPL bump; "Wanda performs competitively with SparseGPT on zero-shot tasks".
- **Training:** Calibration only (Wanda is 300x faster than SparseGPT and needs no weight update).
- **Public code:** `https://github.com/locuslab/wanda`, `https://github.com/IST-DASLab/sparsegpt`.
- **MoE:** Wanda was extended to MoE in follow-ups.
- **Attention vs FFN:** Weight-level — both, but speedups on dense GPUs are weak unless 2:4 hardware is used.
- **Relevance:** Less relevant than activation sparsity because static weight pruning rarely yields >1.3x speedup on a real GPU at 50% unstructured.

### Mixture of Depths (Raposo et al. 2024, DeepMind)
- **Mechanism:** Top-k token routing into each transformer layer; tokens not selected skip the layer.
- **Sparsity:** Token-level skipping; can save up to 50% per layer.
- **Quality cost:** "Match baseline performance for equivalent FLOPs and training time but require significantly fewer FLOPs per forward pass" ([arXiv 2404.02258](https://arxiv.org/abs/2404.02258)). Up to 50% faster post-training sampling.
- **Training:** **Yes — MoD routers must be trained from scratch / continued.**
- **Public code:** No widely-used official release; community reimpls exist.
- **MoE:** Composable with MoE in theory.
- **Attention vs FFN:** Both — entire layer.
- **Relevance:** Out of scope for "no retraining." But see A-MoD below for the training-free variant.

### A-MoD (Attention-as-Router for MoD, 2024)
- **Mechanism:** Repurpose existing attention maps as token-importance scores for MoD-style top-k token selection. *Parameter-free* and works on any pretrained checkpoint.
- **Sparsity:** Configurable; trades quality for fewer tokens per layer.
- **Quality cost:** Reported to work without retraining; quantitative comparison vs vanilla MoD not extensively benchmarked.
- **Training:** None.
- **Public code:** Reference impl from [arXiv 2412.20875](https://arxiv.org/abs/2412.20875).
- **MoE:** Composable.
- **Attention vs FFN:** Both (whole-layer skip).

### FastV (Chen et al. 2024)
- **Mechanism:** In multimodal LLMs (LLaVA-style), prune visual tokens in deeper layers based on layer-2 attention weights.
- **Sparsity:** Drops a fixed fraction of visual tokens.
- **Quality cost:** Minimal on common VQA benchmarks.
- **Training:** None.
- **Public code:** Several reimpls (e.g., `pkunlp-icler/FastV`).
- **MoE:** Yes if LMM backbone is MoE.
- **Attention vs FFN:** Token-level — saves both.
- **Relevance:** Only matters if Arc supports vision LMMs. Vision is in Arc's roadmap (mistralrs-vision crate exists), so this is a candidate for the vision pipeline specifically.

### LazyLLM (Apple/Meta 2024, ICML Workshop)
- **Mechanism:** Long-context token pruning — selectively computes KV-cache for "important" tokens at the prefill stage and defers others, with dynamic re-promotion per generation step.
- **Sparsity:** Long-context only; depends on prompt length.
- **Quality cost:** "Accelerates the prefilling stage of the LLama 2 7B model by 2.34x while maintaining accuracy" on multi-document QA ([arXiv 2407.14057](https://arxiv.org/abs/2407.14057)).
- **Training:** None ("training-free implementation").
- **Public code:** No official Apple release; reimpls exist.
- **MoE:** Should compose.
- **Attention vs FFN:** Both (prunes tokens, so all downstream compute drops).
- **Relevance:** *Prefill-only.* High value for Arc's long-context cases (RAG, code agents).

### TEAL (Liu et al. 2024 → ICLR 2025 Spotlight)
- **Mechanism:** Magnitude-based thresholding of input activations across all 7 projections per transformer block (Q, K, V, O, gate, up, down). Thresholds are layer-dependent, set from offline activation histograms (just need ~128 samples).
- **Sparsity:** "40-50% model-wide activation sparsity with minimal performance degradation" across Llama-2/3 and Mistral 7B-70B.
- **Quality cost (verbatim from Table 1, [arXiv 2408.14690](https://arxiv.org/abs/2408.14690)):**
  - 40% sparsity: Llama-2-7B PPL 5.07 → **5.22** (+0.15); Llama-3-8B 5.87 → **6.21** (+0.34); Mistral-7B 4.92 → **5.13** (+0.21).
  - 50% sparsity: Llama-2-7B → **5.43**; Llama-3-8B → **6.67**.
  - "Near-zero performance degradation at 25%, minimal degradation at 40%."
- **Training:** *None.* Calibration only — pre-baked thresholds for Llama-2/3/Mistral live in the repo's `models/` directory.
- **Public code:** `https://github.com/FasterDecoding/TEAL`. Triton kernel (FP16-only, BF16 atomic_add unsupported in their stack). 21 commits on main.
- **MoE:** *Not demonstrated.* The paper's evaluation excludes Mixtral. Should port in principle (each expert is a SwiGLU), but unverified.
- **Attention vs FFN:** *Both.* Unlike CATS/GRIFFIN which only touch the FFN.
- **Limitations they call out:** "TEAL exhibits substantial sparsity in the low-batch setting...but does not scale as well to higher batch sizes."

### GRIFFIN (Prompt-Prompted, Dong et al. 2024)
- **Mechanism:** *Sequence-level* FFN expert selection. Observes that LLM activations exhibit "flocking" — the same FFN neurons fire repeatedly within a sequence. Picks the active subset *once* from a prompt prefix and reuses for the whole decode.
- **Sparsity:** 50% FFN sparsity.
- **Quality cost:** "Almost no loss in performance" on Llama-2-13B and Gemma-7B; 1.25-1.29x latency improvement.
- **Training:** **Zero training, zero calibration** — runtime only.
- **Public code:** Reference impl at [arXiv 2404.01365](https://arxiv.org/abs/2404.01365).
- **MoE:** Conceptually compatible; not explicitly demonstrated.
- **Attention vs FFN:** FFN only.
- **Why this is structurally important for Arc:** "Compute active set once per sequence" composes perfectly with CUDA graph capture, because the graph topology doesn't change per decode step.

### TDA / "First Activations Matter" (Ma et al. 2024)
- **Mechanism:** Training-free Threshold-based Dynamic Activation. Exploits "history-related activation uncertainty" — the first activations in a sequence determine sparsity for the rest of it.
- **Sparsity:** Comparable to GRIFFIN.
- **Quality cost:** "Accelerate generation speed by 18-25% without significantly compromising task performance" ([arXiv 2408.11393](https://arxiv.org/abs/2408.11393)).
- **Training:** None.
- **Public code:** Paper-only.
- **MoE:** Demonstrated on dense; conceptually portable.
- **Attention vs FFN:** FFN-focused.

### LayerSkip (Elhoushi et al. 2024, Meta, ACL 2024)
- **Mechanism:** Train with layer-dropout (low at early, high at late layers) + early-exit loss. At inference, exit early; verify with the rest of the model in a self-speculative-decoding loop.
- **Sparsity:** Per-token depth.
- **Quality cost:** "Speedups of up to 2.16x on summarization for CNN/DM documents, 1.82x on coding, and 2.0x on TOPv2 semantic parsing task" ([arXiv 2404.16710](https://arxiv.org/abs/2404.16710)).
- **Training:** **Yes — layer-dropout + early-exit loss must be applied during finetune.**
- **Public code:** `https://github.com/facebookresearch/LayerSkip`.
- **MoE:** Compatible.
- **Attention vs FFN:** Both.
- **Relevance for Arc:** Out of "no retraining" scope, but the *inference* code is reusable if we get a LayerSkip-tuned model.

### EE-LLM (Chen et al. 2024, ICML)
- **Mechanism:** Large-scale early-exit framework built on Megatron-LM. Supports both full-parameter training and "EE-Tuning" — light parameter-efficient training of exit heads with the base model frozen.
- **Sparsity:** Per-token early exit.
- **Quality cost:** "Outstanding inference speedup without compromising output quality."
- **Training:** **Yes** — at minimum EE-Tuning of exit heads.
- **Public code:** `https://github.com/pan-x-c/EE-LLM`.
- **MoE:** Megatron supports MoE.
- **Attention vs FFN:** Both.
- **Relevance for Arc:** Out of strict "no retraining" scope.

### ShadowLLM (Akhauri et al. 2024, EMNLP)
- **Mechanism:** A small predictor model that shadows the full model and outputs a sparsity mask. Uses activation-gradient pruning criteria, not magnitude.
- **Sparsity:** Both attention heads and neurons.
- **Quality cost:** "Over 15% improvement in end-to-end accuracy compared to prior methods" and "up to 20% speed-up over DejaVu" on Llama-2 and OPT up to 30B ([arXiv 2406.16635](https://arxiv.org/abs/2406.16635)).
- **Training:** Predictor must be trained on calibration data; base model unchanged.
- **Public code:** `https://github.com/abdelfattah-lab/shadow_llm`.
- **MoE:** Not specifically targeted.
- **Attention vs FFN:** Both.

### SparseInfer (Shin et al. 2024)
- **Mechanism:** Training-free predictor that uses only the *sign bits* of inputs and weights to predict ReLU-style zero activations.
- **Sparsity:** Effective only on ReLU/dReLU-based models.
- **Quality cost:** "Negligible accuracy loss of within 1%p".
- **Training:** None.
- **Public code:** Paper-only as of late 2024.
- **MoE:** Compatible with TurboSparse-Mixtral.
- **Attention vs FFN:** FFN.

### R-Sparse (Zhang et al. 2025)
- **Mechanism:** Rank-aware sparse inference using SVD components of weight matrices. Replaces a linear layer with `low_rank(W) @ x + sparse(W − low_rank(W)) @ x` and exploits both input sparsity and singular-value structure.
- **Sparsity:** 50% model-level sparsity.
- **Quality cost:** Comparable to dense across 10 tasks on Llama-2/3 and Mistral; **43% end-to-end speedup with customized kernels** ([arXiv 2504.19449](https://arxiv.org/abs/2504.19449)).
- **Training:** Calibration only (SVD decomposition is a one-time precompute).
- **Public code:** `https://github.com/VITA-Group/R-Sparse`.
- **MoE:** Not yet demonstrated.
- **Attention vs FFN:** Linear layers — both.

### La RoSA (Liu et al. 2025, ICML)
- **Mechanism:** Apply a fixed *layerwise orthogonal rotation* to input activations *before* top-K. The rotation concentrates magnitude into a few "important" channels so top-K is much more accurate.
- **Sparsity:** 40% (numbers reported).
- **Quality cost:** "For LLaMA2-7B at 40% sparsity, LaRoSA achieves a mere 0.17 perplexity gap with a consistent 1.30x wall-clock time speed-up, and reduces the accuracy gap in zero-shot tasks compared to the dense model to just 0.54%, while surpassing TEAL by 1.77% and CATS by 17.14%" ([arXiv 2507.01299](https://arxiv.org/abs/2507.01299)).
- **Training:** None.
- **Public code:** Paper-only; ICML 2025.
- **MoE:** Not demonstrated.
- **Attention vs FFN:** Both.
- **Key insight:** The rotation can be merged into the *previous* weight matrix offline (R · W_in, W_out · R^T) — zero runtime overhead vs a pure top-K scheme.

### WiSparse (Tsinghua 2026)
- **Mechanism:** Weight-aware mixed-granularity activation sparsity. Combines activation magnitude with precomputed weight L2 norms to score channels. Distributes a global sparsity budget across blocks via evolutionary search.
- **Sparsity:** 50%.
- **Quality cost:** "At 50% sparsity, WiSparse preserves 97% of Llama3.1's dense model performance" with 21.4% end-to-end inference acceleration ([arXiv 2602.14452](https://arxiv.org/abs/2602.14452)).
- **Training:** Calibration only; evolutionary search adds an offline cost.
- **Public code:** Recent preprint.
- **MoE:** Not yet demonstrated.
- **Attention vs FFN:** FFN.

### Polar Sparsity (Shrestha et al. 2025, NeurIPS)
- **Mechanism:** First work to handle the *batched* contextual-sparsity collapse problem. Observation: at small batch, MLP sparsity wins; at large batch, attention-head sparsity wins (because the *union* of active MLP neurons across the batch approaches dense, but per-head attention sparsity is invariant). Trains Selective Head Attention predictors + custom sparse GPU kernels.
- **Sparsity:** Batch-adaptive.
- **Quality cost:** "Up to 2.2× end-to-end speedups for models like OPT, LLaMA-2 & 3, Qwen, Mistral across various batch sizes and sequence lengths without compromising accuracy" ([arXiv 2505.14884](https://arxiv.org/abs/2505.14884)).
- **Training:** Predictor calibration.
- **Public code:** NeurIPS 2025 release imminent.
- **MoE:** Demonstrated on dense and Qwen; not explicit Mixtral results.
- **Attention vs FFN:** Both — explicitly bifurcates the strategy.

### DuoGPT (Yin et al. NeurIPS 2025)
- **Mechanism:** Combines unstructured weight pruning with activation sparsity in one framework, using activation-aware OBC calibration. The output residuals from the dense model are used as correction terms to preserve accuracy.
- **Sparsity:** Dual-sparse — weights *and* activations.
- **Quality cost:** "Outperforms state-of-the-art structured pruning methods by up to 9.17% accuracy at an iso-speedup of 1.39× compared to the baseline dense model" ([arXiv 2506.20194](https://arxiv.org/abs/2506.20194)).
- **Training:** Calibration only (no fine-tune).
- **Public code:** `https://github.com/Intelligent-Computing-Lab-Panda/DuoGPT`.
- **MoE:** Not yet demonstrated; in scope for future work.
- **Attention vs FFN:** Weight-level — both.

### MoE++ / ZEDA / Dynamic Top-K (various 2025 MoE works)
- **Mechanism:** Augment standard FFN experts with zero/copy/constant experts; or post-hoc skip half of the top-K experts via self-distillation.
- **Sparsity:** Up to 50% extra (on top of native top-K MoE).
- **Quality cost:** Self-distillation recovers most quality at low cost (≪ pretraining).
- **Training:** ZEDA needs self-distillation; pure dynamic skipping (top-1 only when expert weight ≫ second) is training-free.
- **Public code:** Reference impls scattered.
- **MoE:** *Yes — this is the only family of methods specifically designed for MoE.*
- **Attention vs FFN:** FFN (expert) only.

---

## 4. Recommendation for Arc

Given Arc's profile (Rust + CUDA, BF16-native, single-stream decode focus on B200, mistral.rs heritage, V2 supports SwiGLU LLaMA/Mistral families, V4 is MoE), the recommended order is:

### Phase A (immediate, low risk):
1. **Port TEAL's selective-column-load gemv kernel to Arc's BF16 CUDA path.** Reuse their published thresholds for Llama-2/3 and Mistral. Expected: 1.5-1.8x decode speedup at 40-50% sparsity, with measurable quality loss <0.5 PPL on Llama-2-7B/Mistral-7B. The Triton kernel architecture (column-major weights, splitK, L2 management) is described in the paper appendix and is straightforward to translate.
2. **Add GRIFFIN-style "sequence-level active set" as a degraded fallback for CUDA-graph compatibility.** When the user requests CUDA-graph-locked decode (Arc's 06_gpu_autonomous angle), GRIFFIN gives a static sparsity pattern compatible with capture-once. This is a ~200-line addition.

### Phase B (next iteration):
3. **Replace TEAL's magnitude threshold with La RoSA's rotation + top-K.** Same kernel, better accuracy ceiling (beats TEAL by 1.77% on zero-shot tasks). The rotation matrices can be merged into adjacent weight tensors offline — zero runtime cost vs Phase A.

### Phase C (MoE / V4):
4. **For V4 (MoE), GRIFFIN-style sequence-level expert pruning + dynamic top-K reduction.** Pure TEAL on MoE is unverified; the safest port is sequence-level FFN sparsity inside each expert PLUS dynamic top-1 routing when the gate confidence is high. Expected gain: 1.2-1.4x on top of native MoE sparsity, no retraining.

### What to skip (for now):
- TurboSparse / ProSparse / ReLU² — require continued pretraining we can't afford.
- LayerSkip / EE-LLM — require model finetune.
- Q-Sparse — requires training.
- Mixture-of-Depths (proper) — requires from-scratch routers.
- FastV — only relevant for the vision pipeline, defer.
- LazyLLM — useful for long-context but more invasive (KV-cache rewrites); defer to long-context work item.

---

## 5. Open Questions / Where the Literature Is Thin

1. **MoE + activation sparsity together.** Only Polar Sparsity (2025) and a few MoE++ papers address this seriously. The interaction between native top-K expert sparsity and per-expert activation sparsity is not well-quantified. For Arc V4 (Mixtral-class), this matters a lot.

2. **TEAL/La RoSA on Mixtral 8x7B / 8x22B / Qwen3-Next.** The published evaluations stop at dense Llama-2/3, Mistral-7B. There is no published TEAL or La RoSA Mixtral number. Arc would be the first to report this.

3. **Batched serving + activation sparsity.** Most activation-sparsity methods explicitly note that benefits collapse at batch≥16 because the union of active neurons across the batch approaches dense. Polar Sparsity's bifurcation (MLP at small batch, attention heads at large batch) is the only credible answer, and its public release is fresh.

4. **Composition with NVFP4 / weight quantization.** TEAL paper says "compatible with weight quantization, enabling further efficiency gains" but does not provide numbers when stacked with 4-bit weights. Arc V2's NVFP4 path *needs* this number — does activation sparsity + NVFP4 multiply (1.8x * 1.6x ≈ 2.9x) or eat into each other?

5. **B200/H200/H100 kernel performance.** Almost all published kernels target A100/A6000. The HBM3e bandwidth on H200/B200 changes the math: when memory bandwidth is 5 TB/s instead of 2 TB/s, the gain from saving HBM reads is larger in absolute time but smaller as a fraction of TPS. This needs Arc-side measurement, not literature lookup.

6. **Top-K predictor cost on tail latency.** None of the papers (TEAL, La RoSA, CATS) measure p99 latency with the predictor in the critical path. Arc's 120 tok/s target is sustained — a 5ms tail in the predictor would kill the win.

7. **Compositional accuracy.** TEAL + GRIFFIN + La RoSA + dynamic top-K MoE: nobody has stacked four of these and measured the joint quality cost. Likely additive in surprising ways. Arc would learn this first.

8. **Attention sparsity outside the contextual-sparsity regime.** The literature splits into (a) "drop neurons in FFN" (TEAL, CATS, GRIFFIN) and (b) "drop tokens from KV cache" (LazyLLM, H2O, StreamingLLM). Combining them — drop neurons *and* tokens simultaneously, ideally based on cross-signal — is unaddressed.

---

## Sources

- [Déjà Vu (arXiv 2310.17157)](https://arxiv.org/abs/2310.17157), [DejaVu GitHub](https://github.com/FMInference/DejaVu)
- [PowerInfer (arXiv 2312.12456)](https://arxiv.org/abs/2312.12456), [PowerInfer GitHub](https://github.com/SJTU-IPADS/PowerInfer)
- [Turbo Sparse (arXiv 2406.05955)](https://arxiv.org/abs/2406.05955)
- [CATS (arXiv 2404.08763)](https://arxiv.org/abs/2404.08763), [CATS GitHub](https://github.com/ScalingIntelligence/CATS), [CATS Stanford blog](https://scalingintelligence.stanford.edu/blogs/cats/)
- [ReLU² (arXiv 2402.03804)](https://arxiv.org/abs/2402.03804)
- [Q-Sparse (arXiv 2407.10969)](https://arxiv.org/abs/2407.10969)
- [SqueezeLLM (arXiv 2306.07629)](https://arxiv.org/abs/2306.07629), [SqueezeLLM GitHub](https://github.com/SqueezeAILab/SqueezeLLM)
- [Wanda (arXiv 2306.11695)](https://arxiv.org/abs/2306.11695), [Wanda GitHub](https://github.com/locuslab/wanda)
- [Mixture of Depths (arXiv 2404.02258)](https://arxiv.org/abs/2404.02258)
- [Attention-as-Router A-MoD (arXiv 2412.20875)](https://arxiv.org/abs/2412.20875)
- [LazyLLM (arXiv 2407.14057)](https://arxiv.org/abs/2407.14057)
- [TEAL (arXiv 2408.14690)](https://arxiv.org/abs/2408.14690), [TEAL GitHub](https://github.com/FasterDecoding/TEAL), [TEAL Together.ai blog](https://www.together.ai/blog/teal-training-free-activation-sparsity-in-large-language-models)
- [GRIFFIN (arXiv 2404.01365)](https://arxiv.org/abs/2404.01365)
- [TDA / First Activations Matter (arXiv 2408.11393)](https://arxiv.org/abs/2408.11393)
- [LayerSkip (arXiv 2404.16710)](https://arxiv.org/abs/2404.16710), [LayerSkip GitHub](https://github.com/facebookresearch/LayerSkip)
- [EE-LLM (arXiv 2312.04916)](https://arxiv.org/abs/2312.04916), [EE-LLM GitHub](https://github.com/pan-x-c/EE-LLM)
- [ShadowLLM (arXiv 2406.16635)](https://arxiv.org/abs/2406.16635), [ShadowLLM GitHub](https://github.com/abdelfattah-lab/shadow_llm)
- [SparseInfer (arXiv 2411.12692)](https://arxiv.org/abs/2411.12692)
- [R-Sparse (arXiv 2504.19449)](https://arxiv.org/abs/2504.19449)
- [La RoSA (arXiv 2507.01299)](https://arxiv.org/abs/2507.01299)
- [WiSparse (arXiv 2602.14452)](https://arxiv.org/abs/2602.14452)
- [Polar Sparsity (arXiv 2505.14884)](https://arxiv.org/abs/2505.14884)
- [DuoGPT (arXiv 2506.20194)](https://arxiv.org/abs/2506.20194), [DuoGPT GitHub](https://github.com/Intelligent-Computing-Lab-Panda/DuoGPT)
- [ProSparse (arXiv 2402.13516)](https://arxiv.org/abs/2402.13516)
- [FastV (arXiv 2403.06764)](https://arxiv.org/abs/2403.06764)
