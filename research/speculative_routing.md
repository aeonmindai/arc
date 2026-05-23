# Speculative Routing for MoE: Draft Router Prefetch on H100

Research target: can a tiny "draft router" predict expert selection ahead of the real router for DeepSeek V4 (256 experts/layer, top-8), enabling expert weight prefetch from HBM while attention runs, and pipelining FFN execution behind it? This is the MoE analog of CPU branch prediction.

Verdict up front: the idea is real, the systems literature on it is roughly 14 months deep and active as of May 2026, multiple groups have demonstrated 88-98% top-k prediction accuracy with sub-200 µs predictor cost, but **almost all published work targets PCIe-bottlenecked CPU-offload regimes**, not the in-HBM, EP-sharded H100 setting Arc actually serves. The HBM regime is the gap. There is a publishable angle for Arc.

---

## 1. State of the art

The relevant literature splits into four buckets.

### 1a. Pre-attention / same-layer prediction (the closest fit for Arc)

**Pre-Attention Expert Prediction and Prefetching for Mixture-of-Experts LLMs**, Zhu, Bohl, Oester, Alonso (ETH Zürich), arXiv 2511.10676, Nov 2025. Two linear layers over the pre-attention hidden state (residual stream entering layer L) predict layer L's expert IDs while attention runs. Hits 93.03% exact-match top-k on DeepSeek V2 Lite, 94.69% on Qwen3-30B, 97.62% on Phi-mini-MoE. Predictor cost: 0.15 ms vs 0.7-1.1 ms for attention, so it executes fully under the attention shadow. Loss is weighted BCE + pairwise ranking (λ=0.3, margin=0.1). Trained on 10M MMLU samples, 30 epochs. Compared against FATE baseline (~79%), they claim +15 pp absolute. Tested V100/A100, no H100, no end-to-end TPOT number — only "expert load time would drop from 0.85-2.01 ms to 0.27-0.64 ms" which is a microbenchmark, not a closed-loop measurement. (https://arxiv.org/abs/2511.10676)

**Fate: Fast Edge Inference of MoE via Cross-Layer Gate**, Fang et al., arXiv 2502.12224, Feb 2025. The predecessor. Uses layer L's *gate input* (not output) to predict layer L+1's experts. No additional model training — just runs the same gate function early on CPU in parallel with GPU. 97.15% prefetch accuracy (using 75th-percentile threshold instead of strict top-k). 4.1-4.5× speedup on RTX 3090, DeepseekMoE 16.4B and Qwen1.5-MoE 14.3B. Targets PCIe offload. (https://arxiv.org/html/2502.12224v1)

**SP-MoE: Speculative Decoding and Prefetching for Accelerating MoE-based Model Inference**, arXiv 2510.10302v2, Oct 2025. Combines token-level spec decoding (Mistral-7B drafts Mixtral 8×7B, etc.) with expert prefetch driven by the draft's attention outputs. 88.94% top-1 expert accuracy on DeepSeek-Lite, ~88% on Mixtral and Phi-3.5-MoE. **Cutoff-layer policy**: prefetch only up to layer L* where "peak non-expert memory + prefetched experts ≤ capacity" and "expert load fully hidden by compute." Speedups 1.07-3.5× TPOT on RTX 3090/4090/A100. (https://arxiv.org/pdf/2510.10302)

**MoE-SpeQ**, arXiv 2511.14102, Nov 2025. Uses an INT4-quantized version of the *same* model as the draft (sharing non-expert params + KV cache). 90.9% prediction accuracy. Adds an "Amortization Roofline Model" that picks draft length per-request based on PCIe bandwidth vs compute roofs. 2.34× over Mixtral-Offloading on A100. (https://arxiv.org/html/2511.14102v1)

**Pre-gated MoE**, Hwang et al., ISCA 2024 (arXiv 2308.12066). The OG: modify the router architecturally so layer L emits *layer L+1's* expert IDs, then prefetch during layer L's MoE block. Requires retraining the gate. Limited adoption because changing the model weights is a non-starter for hosted serving. (https://arxiv.org/pdf/2308.12066)

### 1b. Cache/path-based (no learned predictor)

**MoE-Infinity**, Xue et al., arXiv 2401.14361. Tracks the per-request "Expert Activation Matrix" (EAM) — sparse bitmap of which experts fired at which layer — then prefetches based on cosine-nearest cached EAMs from past requests. 3.1-16.7× latency improvement over DeepSpeed-Inference for offload regime. No learned predictor; pure cache heuristic. (https://arxiv.org/html/2401.14361v3)

**PreScope** (LLaPor predictor), arXiv 2509.23638, Sep 2025. Separate predictor *per layer group* (input/middle/output) — shallow MLP for input/output groups, deeper residual blocks for middle. Top-4 prediction 94-99%. 141% throughput vs Klotski. V100 and A100, Mixtral 8×7B + DeepSeek-V2-Lite + Qwen3-30B + Moonlight-16B. (https://arxiv.org/html/2509.23638v1)

**MoE-Beyond**, arXiv 2508.17137, Aug 2025. Tiny transformer (66M expert-activation traces from DeepSeek V2 Chat Lite) gives 97.5% accuracy, 86.6% F1. Edge focus. (https://arxiv.org/abs/2508.17137)

### 1c. PowerInfer (the conceptual ancestor for "hot vs cold")

**PowerInfer**, Song et al., SOSP 2024 (arXiv 2312.12456). Not MoE — neuron-level sparsity in dense LLaMA-class models. But the architecture is the template: online predictor identifies "hot" units (here: neurons, equivalent to experts for the MoE case), pin them on GPU, predict cold ones for prefetch. 11× over llama.cpp on a single 4090. The MoE-Infinity / Fate / PreScope line is essentially PowerInfer's idea applied to MoE-granularity sparsity. (https://arxiv.org/abs/2312.12456)

### 1d. Distribution-level (not per-token)

**MoE-GPS**, arXiv 2506.07366, Jun 2025. Argues per-token prediction is overkill for EP load balancing. Predicting just the *aggregate token distribution* over experts is enough and saves overhead. +23% E2E vs token-to-expert prediction on Mixtral 8×7B. (https://arxiv.org/pdf/2506.07366)

### 1e. Production systems

No production MoE serving system (vLLM, SGLang, TensorRT-LLM, Fireworks FireAttention) ships speculative expert prefetch as of May 2026, based on public docs. SGLang and vLLM use **expert parallelism** to keep all experts HBM-resident across N GPUs (LMSYS deployed DeepSeek V3 across 96 H100s using PD-disaggregation + DeepEP all-to-all kernels — no prefetch needed because the bottleneck is all-to-all latency, not load). (https://www.lmsys.org/blog/2025-05-05-large-scale-ep/) Fireworks ships proprietary "FireAttention" CUDA kernels for DeepSeek but no public mention of expert prefetch. The single-GPU / single-node serving case where prefetch matters most is exactly the case Arc targets and the production world has largely ceded to research papers.

---

## 2. Best draft router architecture for V4

Given the constraint that DeepSeek V4 is 256 experts/layer top-8 and Arc serves single-GPU (or small-EP) FP4-MoE/FP8-non-MoE on H100, the optimal draft router design — synthesizing the four leading approaches — is:

**Tier 0 (free, baseline):** Reuse the previous layer's actual router output as a draft for the next layer. Cross-layer routing correlation in Mixtral, Phi-MoE, and DeepSeek V2/V3 is documented as "strong sequential dependency" — neighboring layers have >50% expert-set overlap. Zero parameters, zero compute, zero training. Get a free 40-60% prediction accuracy floor. Use this as a sanity baseline before adding any learned component.

**Tier 1 (recommended):** Two-linear-layer probe on the residual stream entering layer L, with ranking-aware loss — the Zhu et al. (2511.10676) architecture. For V4 with hidden dim d≈7168 (assumed similar to V3) and 256 experts, this is `Linear(d, 2048) → SiLU → Linear(2048, 256)` per layer = ~16 MB FP16 per layer × ~60 MoE layers = ~1 GB of predictor weights. Forward cost: ~0.15 ms per layer on H100, executes under the attention shadow. Top-8 exact-match accuracy 93-95% expected based on V2-Lite results. Trained on routing traces (10M samples, 30 epochs is enough; this is small data and a tiny model — under 2 H100 hours total per layer).

**Tier 2 (research):** A GRU/recurrent router across layers (Layerwise Recurrent Router for MoE, arXiv 2408.06793) trained to emit *next-layer* routes — but this requires retraining the base model or LoRA-style fine-tuning, which violates "don't change DeepSeek weights." Not recommended for V1.

The Tier 1 design is the right choice. Two linear layers, pre-attention activations, ranking loss, train offline once per model release. Worth noting: every paper that has tried this architecture on DeepSeek-family models has reported 93%+ top-k accuracy, with cosine-similarity-of-routing in the 0.94+ range for V2 — i.e. routing is highly predictable from the pre-attention hidden state.

---

## 3. Predicted accuracy

For DeepSeek V4 specifically — TBD until measured, but with high prior confidence in the 92-95% top-8 exact-match range:

- DeepSeek V2 Lite (closely related architecture, 64 experts): 93.03% measured. (Zhu et al.)
- DeepSeek V2 Chat Lite (MoE-Beyond predictor): 97.5% top-k accuracy on autoregressive decode traces.
- DeepSeek-Lite under SP-MoE's coarser draft-model-based scheme: 88.94% top-1.

Going from 64 experts to 256 experts hurts marginally (more classes, larger output head), but the hidden-state signal-to-noise is actually *better* for larger V4 because the residual stream is wider and load-balancing nudges (V3-style bias terms instead of aux loss) make routing more deterministic. **Predicted: 92-95% exact-match top-8 accuracy on V4, ~99% top-k coverage (the right experts are in the top 16, just maybe not ranked 1-8).**

For misprediction handling, the over-provisioning trick from Zhu et al. matters: predict top-12 instead of top-8 and prefetch all 12. Even at 93% strict accuracy, top-12 coverage of true top-8 is ~99%. This is the cheap win.

---

## 4. End-to-end speedup estimate on H100

**Honest first check: is the routing cost actually the problem?**

On H100, per-layer DeepSeek-V3 decode breakdown at batch 1 (estimates, FP8 MoE):

| Component                | Latency           | Notes |
|--------------------------|-------------------|-------|
| MLA attention            | 0.6-1.1 ms        | Includes cache load |
| Router (linear + topk)   | 5-15 µs           | Tiny |
| Expert weight load (8 × 56 MB = 448 MB FP8) | 130-450 µs | At 3.35 TB/s peak; often 1.4 ms realized due to non-contiguous |
| Expert FFN compute       | 60-120 µs         | Top-8 GEMVs, bandwidth-bound |
| All-reduce + residual    | 30-80 µs          | EP-sharded |

The router itself is tiny — 5-15 µs. **The thing prefetch hides is expert weight load, not router compute.** That cost is real: 130-450 µs per layer × 60 MoE layers = 8-27 ms per token of pure expert-load time at 3.35 TB/s peak HBM, against ~30-50 ms total decode latency. Expert load is 40-60% of single-batch decode latency on H100. This is the prize.

**The pipeline question**: H100 attention takes ~0.6-1.1 ms (with MLA — fairly cheap). Expert load is 130-450 µs. **Attention provides enough shadow to fully overlap the expert load** if and only if (a) the predictor runs in parallel and (b) the load is issued async on a side stream. With FP4 expert weights (V4 default), per-expert size drops from 56 MB FP8 to ~28 MB FP4 — load becomes ~67 µs per expert, ~535 µs for 8 — easier to hide.

**Speedup ceiling on a perfect predictor:** if expert load is fully hidden by attention, the per-layer critical path becomes max(attention, router+expert_load) + expert_FFN + comms. Going from "attention then expert_load then FFN" (serial) to "attention || expert_load, then FFN" (overlapped) saves the expert_load time on the critical path. At 60 MoE layers × ~300 µs expert load saved per layer → **18 ms/token saved** out of ~40 ms baseline → **1.5-1.8× decode speedup ceiling** with perfect prediction.

**Realistic with 93% predictor + over-provisioning:** ~85-90% of the ideal saving is achievable. **1.4-1.6× decode speedup** on a single-node H100 V4 deployment. This matches SP-MoE's measured 1.3-1.5× on A100 (which is more constrained) and PreScope's 70-140% throughput claims on PCIe-offload setups (which are even more constrained).

**Honest caveat**: with full EP-sharding across 8-96 H100s (SGLang/vLLM setup), all experts are HBM-resident, the bottleneck shifts to all-to-all dispatch latency, and prefetch is moot. **The speculative-routing win is for the single-node / small-cluster regime that Arc targets** — exactly the audience that does *not* have a 96-GPU cluster.

---

## 5. Quality risk analysis (3 modes)

**Mode A — Prefetch-only, real router still runs and decides:** Zero quality drop. The predictor only guides which weights to start loading; the real router selects the actual 8 to compute. If the predictor was right, weights are warm; if wrong, you eat a load stall on the miss but get correct output. This is the recommended mode for Arc V1. Equivalent to a CPU branch predictor that never affects program semantics.

**Mode B — Aggressive (skip real router, trust draft):** Quality drops proportional to mis-route rate. At 93% top-8 exact-match, ~7% of tokens use at least one wrong expert. Empirically (Pre-gated MoE paper, ISCA 2024) this costs 0.5-1.5 PPL on WikiText and ~0.5-2 pp on MMLU. **Not recommended unless throughput pressure is extreme.** Quality drop is "noticeable but tolerable" — a small risk surface but real.

**Mode C — Hybrid verify-and-correct:** Predictor prefetches top-12, real router runs in parallel during attention shadow, then a fast set-membership check rejects the predicted set if the real top-8 isn't fully covered. Cost of verification: one router forward (5-15 µs, negligible) + a max-1-expert "emergency load" (in worst case ~60 µs) when over-provisioning fails. **Zero quality drop, ~98% of Mode A's speedup.** This is the right production target.

---

## 6. Integration with arc-cuda-graph

Arc's `arc-cuda-graph` crate is currently dense-only — `decode_forward.rs`, `dedicated.rs`, the autonomous loop in `autonomous.rs`. No MoE primitives exist yet (`grep moe|expert` returns nothing).

The integration story:

1. **Async expert-load stream**: Arc already runs decode on a "non-blocking stream" (per `lib.rs` docstring) separate from the model-load NULL stream. Add a *second* non-blocking stream — call it `prefetch_stream` — dedicated to expert-weight HBM-to-HBM (or HBM-to-cache) copies. `cudaMemcpyAsync` between two HBM regions is the H100 SDMA path; ~3.35 TB/s effective. Issue prefetches on `prefetch_stream` from the predictor output, then `cudaStreamWaitEvent` on the compute stream before the MoE GEMV launches. CUDA graphs *can* capture multi-stream dependencies via event-record/event-wait nodes — this is straightforward, well-tested H100 territory.

2. **CUDA graph capture**: The predictor MLP is two GEMVs — fully graph-capturable. The dynamic part is *which* experts to load — that's not in the graph topology, just in the pointers. Use cudaGraphExecKernelNodeSetParams (or graph update via memcpy-node param replacement) to swap expert pointers per-decode-step without re-capturing the graph. Arc already does this pattern for KV cache pointers in `graph.rs` / `autonomous.rs`.

3. **Expert pool layout**: Pre-pack all 256 experts per layer into a contiguous HBM region indexed `expert_pool[layer][expert_id]`. Prefetch becomes a `cudaMemcpyAsync` of the predicted 12 expert chunks into a per-layer "warm" buffer that the MoE GEMV reads from. Or, if all 256 are HBM-resident already (FP4 → 256 × 28 MB × 60 layers ≈ 430 GB — too big for a single H100's 80 GB), the prefetch is just a `cudaPrefetchAsync` to L2 / a cache-resident pool. **For DeepSeek V4 FP4, expect ~430 GB total expert weight** — that does *not* fit one H100, so the prefetch use case is real: some experts will be in CPU/SSD or sharded across GPUs, and predicting which 12 of 256 to land in HBM ahead of time saves the load.

4. **Graph topology**: per layer, the captured DAG is:
   `attention → {predictor → prefetch_issue (on stream B)} → router → expert_FFN (waits on prefetch event) → comms → next_layer`
   The predictor and prefetch live on the side stream; the compute stream synchronizes only when the FFN needs the weights. If the predictor was right, the wait is zero. If it was wrong, the wait costs the load latency — the worst case is back to baseline, never worse.

5. **Zero-CPU-sync still holds**: predictor output (expert IDs) is a small int32 vector that lives entirely on-GPU; the topk selection from the predictor's logits is one CUB-topk call. No CPU round-trip. Compatible with Arc's GPU-autonomous WHILE-loop design.

The whole pipeline change is roughly: +1 stream, +1 predictor module per MoE layer, +1 prefetch memcpy node per layer in the captured graph, +1 event sync per layer. No fundamental redesign of `arc-cuda-graph`.

---

## 7. Combination with cross-layer routing correlation (parallel thread)

The cheapest possible draft router is **the previous layer's actual router output**. No new model. Zero training. The correlation literature is unanimous that adjacent-layer expert sets overlap significantly:

- Lina / MoE-Infinity / Fate all exploit cross-layer correlation, reporting "strong sequential dependency."
- Layerwise Recurrent Router (arXiv 2408.06793) shows that explicitly training cross-layer dependencies via a GRU on top of routers improves quality — meaning the latent dependency is already there in vanilla models, just not fully exploited.
- PreScope's "layer-group" finding (input layers cluster, middle layers cluster, output layers cluster, each with its own predictor architecture) is the same story expressed differently.

**Recommended Arc V1 design**: ship the prev-layer-as-draft path first. Measure prediction accuracy on V4. If it's >85% top-8, that may be enough — combined with over-provisioning (load top-16 union of prev layer + heuristic), you might hit 98% coverage *without* training any model. This is the lowest-risk, highest-leverage first step. Add the Tier 1 learned predictor in V2 if measurements show a clear ROI gap.

The two approaches **stack**: use prev-layer routing as a "free" sketch, then a tiny learned residual predictor refines it for the deeper-in-the-network layers where correlation weakens (the middle-layer group in PreScope's taxonomy).

---

## 8. Estimated effort (agent-sessions)

- **Prev-layer-as-draft path** (no training, just pipeline wiring): 1-2 sessions. Add a second CUDA stream + event sync + a memcpy-node in the captured graph. Run an evaluation pass on DeepSeek V2 Lite to measure overlap accuracy and TPOT delta.

- **Tier 1 learned predictor (offline training + inference integration)**: 2-3 sessions. Training a 2-linear-layer predictor on 10M MMLU samples for 30 epochs per layer × 60 layers ≈ a few H100-hours total (small model, small data — this is *not* weeks of training). Plus the predictor inference path in `arc-cuda-graph`. Plus the predictor weight format / loading in Arc's weight loader.

- **CUDA graph multi-stream capture + event sync nodes**: 1-2 sessions. Validate against H100, confirm replay correctness, confirm no CPU sync introduced.

- **End-to-end measurement + tuning of over-provisioning factor**: 1 session. Sweep top-k prefetch sizes (8, 10, 12, 16), measure TPOT and quality.

**Total: 5-8 agent-sessions** for a research-quality implementation that ships in Arc and produces an evaluation table comparable to SP-MoE / Pre-Attention Expert Prediction. Plus a small writeup. The expensive blocker is *measurement infrastructure* (need a stable V4 inference baseline to compare against) — not the implementation itself.

---

## 9. Honest verdict on novelty and ROI

**Novelty (May 2026):**
- The *idea* of expert prefetch via prediction is well-trodden — at least eight papers in the last 18 months (Pre-gated MoE 2024, MoE-Infinity 2024, Fate Feb 2025, MoE-Beyond Aug 2025, PreScope Sep 2025, SP-MoE Oct 2025, Pre-Attention Nov 2025, MoE-SpeQ Nov 2025).
- **The gap that is actually open**: every published evaluation targets PCIe-offloaded inference on V100/A100/3090/4090, with experts streaming from CPU or SSD. **Nobody has published a clean end-to-end TPOT result on H100 for DeepSeek V3/V4 with speculative expert prefetch in the all-HBM regime where the prefetch is HBM-to-cache rather than DRAM-to-HBM.** That is novel.
- Arc's CUDA-graph zero-CPU-sync decode is a clean substrate for this — the multi-stream graph capture pattern is well-defined but nobody has tested it for MoE prefetch on H100 in a published paper.
- The "prev-layer-as-draft + over-provisioning" zero-training baseline is documented in pieces but no paper has shipped it as the simplest possible thing that works. There's a publishable ablation: "how much of the speedup does the learned predictor actually buy over free correlation?"

**ROI:**
- Speedup ceiling 1.5-1.8×, realistic 1.4-1.6× on DeepSeek V4 single-node H100 decode. Combined with TurboQuant K4/V3, Arc's autonomous-decode WHILE-loop, and FP4 expert weights, this puts a 60 tok/s baseline at 90-100 tok/s — meaningful step toward the 120 tok/s peak-inference target.
- Zero quality drop in Mode C (hybrid verify).
- Stacks cleanly with existing Arc work (CUDA graphs, autonomous decode, weight quantization). No reorg of `arc-cuda-graph` needed.

**Recommendation:** ship the prev-layer-as-draft path (1-2 sessions) as a probe. Measure on V2 Lite or V3 to confirm 85%+ correlation. Then decide on the learned-predictor add-on based on actual measurement, not paper claims.

The downside risk is real but bounded: if H100 attention turns out shorter than expected on V4 with MLA + sparse attention (V4 Flash has CSA/HCA which may cut attention to 200-400 µs), the shadow shrinks and the win shrinks with it. Worth measuring before committing to the full learned-predictor build.

---

## Sources

Pre-attention / learned predictor:
- [Pre-Attention Expert Prediction and Prefetching for MoE LLMs (arXiv 2511.10676)](https://arxiv.org/abs/2511.10676)
- [Fate: Fast Edge Inference of MoE via Cross-Layer Gate (arXiv 2502.12224)](https://arxiv.org/html/2502.12224v1)
- [SP-MoE: Speculative Decoding and Prefetching for MoE (arXiv 2510.10302)](https://arxiv.org/pdf/2510.10302)
- [MoE-SpeQ: Speculative Quantized Decoding with Proactive Expert Prefetching (arXiv 2511.14102)](https://arxiv.org/html/2511.14102v1)
- [Pre-gated MoE (ISCA 2024, arXiv 2308.12066)](https://arxiv.org/pdf/2308.12066)
- [PreScope: Prefetching for Resource-Constrained MoE (arXiv 2509.23638)](https://arxiv.org/html/2509.23638v1)
- [MoE-Beyond: Learning-Based Expert Activation Prediction on Edge Devices (arXiv 2508.17137)](https://arxiv.org/abs/2508.17137)

Cache / path / correlation:
- [MoE-Infinity: Activation-Aware Expert Offloading (arXiv 2401.14361)](https://arxiv.org/html/2401.14361v3)
- [MoE-GPS: Prediction Strategy for Dynamic Expert Duplication (arXiv 2506.07366)](https://arxiv.org/pdf/2506.07366)
- [Layerwise Recurrent Router for MoE (arXiv 2408.06793)](https://arxiv.org/html/2408.06793v1)

Branch-prediction analog / PowerInfer:
- [PowerInfer: Fast LLM Serving with Consumer-grade GPU (arXiv 2312.12456)](https://arxiv.org/abs/2312.12456)

Production / DeepSeek deployment context:
- [Deploying DeepSeek with PD Disaggregation and Large-Scale EP on 96 H100 GPUs (LMSYS Blog, 2025)](https://www.lmsys.org/blog/2025-05-05-large-scale-ep/)
- [DeepSeek-V3 Technical Report (arXiv 2412.19437)](https://arxiv.org/pdf/2412.19437)
- [Insights into DeepSeek-V3 Hardware Co-design (arXiv 2505.09343)](https://arxiv.org/html/2505.09343v2)

Speculative decoding × MoE (orthogonal but related):
- [MoE-Spec: Expert Budgeting for Efficient Spec Decoding (arXiv 2602.16052)](https://arxiv.org/html/2602.16052v1)
- [EVICT: Adaptive Verification for MoE Speculative Decoding (arXiv 2605.00342)](https://arxiv.org/abs/2605.00342)
- [Why MoE Models Get More from Speculative Decoding (Cohere)](https://cohere.com/blog/mixture-of-experts-models-get-more-from-speculative-decoding)
