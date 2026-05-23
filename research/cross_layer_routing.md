# Cross-Layer Routing Correlation in MoE Inference

Status: Research-only survey. No Arc source modified.
Audience: Arc inference-engine engineering, V4 decode-path optimization.
Bottom line: Cross-layer routing correlation is **a real, empirically measured, repeatedly published effect**, and the family of "next-layer expert prefetch" techniques is **already a crowded design space** (8+ systems papers since 2023). Arc would not be inventing the idea; the win, if any, is implementation quality on DeepSeek-V4 plus integration with the existing `arc-cuda-graph` zero-sync decode path. Read sections 5 and 7 carefully before scoping work.

---

## 1. Literature findings

### 1.1 Direct measurements of inter-layer routing correlation

There is a substantial published record measuring how strongly layer L's routing predicts layer L+1's. The strongest concrete numbers come from these papers:

1. **ExFlow** (Yao et al., 2024, [arXiv:2401.08383](https://arxiv.org/abs/2401.08383); [GitHub](https://github.com/YJHMITWEB/ExFlow)) measured "inter-layer expert affinity" via conditional probability heatmaps `P(expert_j at layer L+1 | expert_i at layer L)` on GPT-MoE models (350M-1.3B, 8-64 experts). They report that the heatmaps are highly sparse ("only a few columns are red"), and on GPT-MoE-64 with 4 GPUs, exploiting this affinity keeps over half of tokens off the cross-GPU AlltoAll path; with 8 GPUs, 40 percent stay on the same GPU; tokens are 2× more likely to stay within the same node. They did not publish the raw conditional-probability matrix, but the placement gain implies a strongly non-uniform P(L+1 | L).

2. **Cross-Layer Gate** (Fang et al., 2025, [arXiv:2502.12224](https://arxiv.org/html/2502.12224v1)) is the cleanest existing measurement. They reuse the **layer-L gate weights** to score expert preferences using the gate input that flows into layer L+1 (cloned to CPU and run async). On **Qwen1.5-MoE-A2.7B (top-4 of 60)** and **DeepSeekMoE-16B (top-6 of 64)** they report: **78.79 percent top-k recall for single-position prediction**, and **97.15 percent recall when using a 75th-percentile confidence cutoff** (i.e., overfetching the top experts plus the next slice). No fine-tuning, parameter-free.

3. **Speculative MoE (s-MoE)** (Li et al., 2025, [arXiv:2503.04398](https://arxiv.org/html/2503.04398v2)) measured cross-layer routing correlation on **DeepSeek-V2** (small/many experts) and **Mixtral-8x7B** (large/few experts). They report that **"tokens that select specific experts at one layer tend to choose a rather fixed set of experts at the next layer with high probability"** and achieved **~89 percent accuracy speculating token-expert routes** with their probabilistic model. Using **five preceding layers as context** drives prediction confidence to ~70 percent (so 5-layer history is roughly equivalent to looking ahead 1 layer using L's full gate).

4. **PreScope / LayerScope** (Zhang et al., Sep 2025, [arXiv:2509.23638](https://arxiv.org/abs/2509.23638)) measured this on **Mixtral-8x7B, DeepSeek-V2-Lite (64 experts), Qwen3-30B-A3B (128 experts), Moonlight-16B-A3B (64 experts)**. Their layer-aware predictor (LLaPor) hits **>90 percent top-4 prediction accuracy** across all four models, and **>94 percent hit rate for hot experts ranked by token volume**, with token-coverage error under 5 percent. They identified three distinct layer groups (near-input, near-output, middle) with different correlation profiles — near-input and near-output layers show stronger routing correlation and more skewed gating weights; middle layers show higher cosine similarity in gate-input but more diffuse weight.

5. **ExpertFlow** (Yuan et al., 2024, [arXiv:2410.17954](https://arxiv.org/html/2410.17954v2)) trained a T5-style encoder-decoder predictor (RPP) whose decoder emits per-layer expert logits in one pass. On Switch-32 at CS=16 they get **91.90 percent cache hit ratio**, beating LRU by up to 61 points. Tested on Switch Transformer, Mixtral-8×7B, DeepSeek-MoE (top-6 of 64), Qwen1.5-MoE.

6. **Pre-gated MoE** (Hwang et al., ISCA 2024, [arXiv:2308.12066](https://arxiv.org/html/2308.12066v3)) replaces the layer-L gate with a "pre-gate" that emits **layer-L+1's** expert choices instead. Requires fine-tuning (2,048 steps, ~227M tokens). Tested on SwitchTransformer (8/64/128 experts, Switch-Large up to 26.4B). Result: 1.5× e2e throughput, 4.2× peak GPU memory reduction; reaches 81 percent of oracle GPU-only throughput. Notable caveat: model accuracy "varies within acceptable ranges" — a slight quality cost from forcing the predicted routing.

7. **SiDA-MoE** (MLSys 2024, [arXiv:2310.18859](https://arxiv.org/abs/2310.18859)) trains an offline hash function that replaces the router; up to 3.93× throughput, 80 percent GPU memory saving, with "down to 1 percent" accuracy drop.

8. **SP-MoE** (Chen et al., 2025, [arXiv:2510.10302](https://arxiv.org/abs/2510.10302)) uses an **aligned smaller draft MoE** (e.g., Phi-mini-MoE drafting for Phi-3.5-MoE) to predict the target model's routing — attention output of draft layer L feeds the target's layer-L gate. On **DeepSeek-Lite** they get 40 percent prefetch hit rate (vs 21.85 percent for AdapMoE and 17.67 percent for Mixtral-Offloading) and 1.07–3.5× TPOT speedup.

9. **Decoding Knowledge Attribution in MoE** (Wang et al., 2025, [arXiv:2505.24593](https://arxiv.org/pdf/2505.24593)) measured **temporal correlation between attention heads and expert selection: r = 0.68, p < 0.0014** on Mixtral-8×7B — i.e., attention output is a strong signal for current-layer routing. This is a *within*-layer correlation but directly suggests that L's attention output predicts L+1's expert pick (because the residual carries forward).

10. **Synergistic Intra/Cross-Layer Regularization Losses for MoE Expert Specialization** (2026, [arXiv:2602.14159](https://arxiv.org/html/2602.14159)) names the phenomenon explicitly: **"cross-layer coupling"** — "an emergent property in MoE where routing decisions in adjacent layers are strongly correlated, such that the expert activated at layer ℓ is highly predictive of the expert activated at layer ℓ+1." They show this coupling emerges *naturally during training* (not just under their regularizer) and propose to amplify it for better specialization.

11. **MoE-Infinity** ([arXiv:2401.14361](https://arxiv.org/html/2401.14361v2)) builds the prefetch strategy on the empirical observation that "experts within MoE models tend to be activated in groups across different layers" — i.e., expert *co-activation* across layers is the property that makes prefetching work. CPU-GPU offload system; relies on cross-layer prediction to avoid late prefetches.

12. **PROBE** ([arXiv:2602.00509](https://arxiv.org/pdf/2602.00509)) and **DuoServe-MoE** ([arXiv:2509.07379](https://arxiv.org/html/2509.07379v1)) are 2025-2026 systems that build on the same correlation property; DuoServe-MoE reports 54-67 percent top-2 accuracy with a different predictor design.

### 1.2 Routing distribution (hot vs cold) at each layer

- **Snowflake Arctic** activates only 1.5-14.5 percent of experts for batch sizes 1-32 (MoE-Infinity). Implies hot/cold separation.
- **PreScope** found near-input and near-output layers have *more skewed* gating weights (a small set of experts dominates), while middle layers are *more uniform* in expert weight but more correlated in gate-input cosine similarity. This matches the multilingual-MoE literature finding ([arXiv:2510.04694](https://arxiv.org/pdf/2510.04694)) that early/late layers do language-specific processing and middle layers are language-agnostic capacity hubs.
- **DeepSeek-V3 / V4** uses an **auxiliary-loss-free load balancer** (dynamic bias term per expert; [DeepSeek-V3 Technical Report arXiv:2412.19437](https://arxiv.org/abs/2412.19437)). The explicit design target is uniform expert utilization *over a training batch*. This **flattens** the per-token routing distribution compared to Mixtral (no auxiliary load loss) and reduces head-room for prefetch — fewer "always hot" experts on a per-batch basis. This is important for Arc.
- **PowerInfer** ([arXiv:2312.12456](https://arxiv.org/abs/2312.12456)) showed for *dense* models that ~80 percent of activations land on a small "hot" set of neurons. The MoE analog is weaker because of explicit load balancing, but the locality intuition transfers.

### 1.3 GitHub repos that measure this empirically

- [ExFlow](https://github.com/YJHMITWEB/ExFlow) — `solve_affinity.py` runs an integer program over measured expert routing traces.
- [MoE-Infinity](https://github.com/EfficientMoE/MoE-Infinity) — runtime expert prefetcher.
- [PowerInfer](https://github.com/SJTU-IPADS/PowerInfer) — neuron-level hot/cold (dense, but analog).
- DeepSeek-V3 SGLang integration: SGLang already exposes **EPLB (Expert Parallelism Load Balancer)** which analyzes activation statistics to relocate experts ([SGLang Expert Parallelism docs](https://docs.sglang.io/advanced_features/expert_parallelism.html)). EPLB is a *placement* optimizer, not a prefetcher.

### 1.4 "Speculative routing" as a term

Yes, it exists. **s-MoE** ([arXiv:2503.04398](https://arxiv.org/html/2503.04398v2)) coined "Speculative Expert Pre-grouping (s-EG)." **SpecMD** ([arXiv:2602.03921](https://arxiv.org/html/2602.03921)) is titled "A Comprehensive Study On Speculative Expert Prefetching" and is the closest thing to a survey of the design space. **MoE-Spec** ([arXiv:2602.16052](https://arxiv.org/abs/2602.16052)) and **MoE-SpeQ** ([arXiv:2511.14102](https://arxiv.org/abs/2511.14102)) use "speculative" in the same sense. **SP-MoE** ([arXiv:2510.10302](https://arxiv.org/abs/2510.10302)) overlaps speculative *decoding* with speculative *expert prefetch*. The term is established.

---

## 2. Theoretical mechanism — why correlation exists

Three independent reasons:

1. **Residual-stream continuity.** A standard transformer block updates `x_{L+1} = x_L + attn(x_L) + mlp(x_L)`. The residual is preserved across layers, and at decode the MLP/MoE update is bounded in norm relative to the residual (this is the entire reason transformers train at depth). So the gate input at layer L+1 — which is `x_L` plus a bounded update — is similar to gate input at layer L. Since routers are typically linear projections `W_router x`, similar inputs give similar logits. The Cross-Layer Gate paper exploits exactly this fact: feeding L's gate-input to L+1's gate gives 78.79 percent single-position recall.

2. **Expert specialization.** Routers learn to send semantically similar tokens to the same expert. If a token is "Python code about file I/O" at layer 7, that semantics doesn't vanish at layer 8 — the residual still carries it. So the expert most appropriate for that semantic role tends to be selected again. The cross-layer-coupling paper shows this is *amplified* by training dynamics.

3. **Attention-driven semantics.** Wang et al. measured r=0.68 between attention head firings and expert selection in Mixtral. Attention output evolves smoothly across layers (especially deep into the network), so its strong correlation with routing implies routing inherits that smoothness.

**Why correlation might be WEAKER in DeepSeek-V4 than Mixtral:**
- V4 has 256 fine-grained experts per layer vs Mixtral's 8. Each fine-grained expert handles a narrower slice, so the conditional `P(expert_j L+1 | expert_i L)` is spread across more entries — Shannon entropy is higher, raw correlation is lower.
- V4 uses **auxiliary-loss-free load balancing** that injects a dynamic per-expert bias *only at routing time*. This bias term explicitly works *against* per-token correlation (because it pushes underutilized experts up regardless of token content).
- V4's first three MoE layers are **Hash routing** (token-id keyed). Hash routing is by construction uncorrelated with previous layers' learned routing.

This is why I expect **measured correlation on V4 to be lower than the 90+ percent numbers reported for Qwen/Mixtral**, and the offline experiment in section 5 is non-negotiable before committing engineering.

---

## 3. Implementation sketch — a draft router in Arc

Given the literature, three options ranked by effort/risk:

### Option A — Cross-Layer Gate (parameter-free, lowest risk)
- During decode at layer L, after the gate computes top-k for L, also speculatively run **layer L+1's gate function** on layer L's gate-input (`x_L`).
- This produces a predicted top-k for L+1.
- Issue a CUDA async copy / `cudaMemcpyAsync` (or DeepEP dispatch) for the 6 predicted V4 experts of layer L+1 into HBM if not resident.
- When layer L+1 actually runs, the gate computes the *real* top-k; on a hit, expert weights are already warm in cache; on a miss, fall back to on-demand load.
- **Quality impact: zero.** This is a pure prefetch — the actual computation uses the real router output.
- Predictor cost: one extra GEMV per layer (`W_router_{L+1}` × `x_L`), dim 7168 × 256 = 1.8M elements ≈ 3.6 MB BF16 traffic per layer per token = trivial.

### Option B — Predicted-only fast path (medium risk)
- Same prediction as Option A, but if predicted == real, **skip the second router GEMV entirely** and reuse the prediction's logits.
- Saves one tiny GEMV. Negligible decode speedup at V4 sizes (~µs per layer); not worth complexity.

### Option C — Pre-gated style (high risk, quality cost)
- Train an additional pre-gate per layer that predicts layer L+1's experts from layer L's input. Use the prediction as the *actual* routing decision (no verification against ground truth).
- Requires fine-tuning. Pre-gated MoE paper reports "acceptable" quality drop — non-zero.
- Not recommended for Arc.

**Where it lives in Arc:**
- DeepSeek-V4 MoE forward is in `mistralrs-core/src/models/deepseek4.rs` lines ~1050-1250 (`MoeGate`, `MoeBlock`).
- Prefetch hook would live just after the gate computation in `MoeBlock::forward`. Need to thread access to layer L+1's gate weights into the layer-L step. Cleanest factoring: precompute the chain at model construction so each layer holds a borrow on the next layer's gate.
- Integration with `arc-cuda-graph`: prefetch must be **captured in the graph** (no host-side scheduling). Use `cudaMemcpyAsync` on a side stream that the graph captures; if the experts already live on-GPU (no offload), the prefetch collapses into an L2-cache warming hint — still useful for non-resident expert weights under expert-parallel sharding.

### Critical caveat for V4 + Arc

V4 has 256 experts × 43 MoE layers. At BF16, each expert is `2 × moe_intermediate_size × hidden_size × 2 bytes = 2 × 2048 × 7168 × 2 = ~56 MB per expert (gate + up + down)`. Total MoE weight = 256 × 43 × ~84 MB ≈ ~924 GB at BF16. So **V4 always runs under expert parallelism** across many GPUs (or with quantization). The relevance of prefetch depends entirely on **where the experts live**:
- **All experts resident in HBM on this rank (no offload, no remote EP)**: prefetch buys nothing measurable — the weights are already 3.35 TB/s away. (You'd just be warming L2.)
- **Experts offloaded to CPU pinned memory**: prefetch over PCIe (~64 GB/s) is huge — this is the regime PreScope/MoE-Infinity targets, 70+ percent latency reduction.
- **Expert parallelism across nodes (DeepEP all-to-all)**: prefetch can be a **routing-decision hint** to start the dispatch earlier on a side stream. SGLang's two-batch-overlap already does this for whole batches. Per-token speculative dispatch is an Arc-novel angle (see section 5).

---

## 4. Memory bandwidth math

Assumptions stated up front:
- V4 Flash: 43 MoE layers, 256 experts/layer, top-6 (V4 Flash per `deepseek4.rs:2030`; the prompt said top-8 — that's V3, not V4-Flash).
- BF16 weights, per-expert size: `~84 MB` (gate+up+down at moe_intermediate_size=2048, hidden=7168).
- 6 fired experts/layer × 43 layers × 84 MB = **21.7 GB** of expert weight traffic per token at decode.
- H100 HBM3: 3.35 TB/s.
- Time-lower-bound at 100 percent BW: 21.7 GB / 3.35 TB/s = **6.5 ms / token** just for expert weights.
- B200 HBM3e: 8 TB/s → 2.7 ms/token lower bound.

**Case 1: All experts in HBM on local rank (no offload).** Prefetch hides nothing — weights are already on the bus the moment the matmul issues. Expected speedup: **0 percent**, possibly negative from added scheduling overhead. *Prefetch is dead in this case.*

**Case 2: Expert parallelism with DeepEP all-to-all (multi-GPU/multi-node).** Each fired expert's *activation* (not weight) is dispatched over NVLink/IB. NVLink 4 = 900 GB/s, IB = 400 Gbps = 50 GB/s. Dispatch payload per layer per token = ~7168 × 2 bytes × 6 experts = ~86 KB. Dispatch latency is dominated by *fixed-cost AlltoAll setup*, not bandwidth. Speculative pre-dispatch on top-k-predicted = potentially overlap with attention compute of layer L+1. **Realistic decode speedup estimate: 5-15 percent**, dependent on EP topology. SGLang's "single-batch-overlap" already covers some of this win.

**Case 3: CPU offload (e.g., consumer GPU running V4).** This is the regime where every paper above (PreScope, MoE-Infinity, ExpertFlow, Pre-gated MoE) reports 40-141 percent throughput gain. **Realistic Arc-on-CPU-offload speedup: 1.5-2.5×**. But this is not Arc's primary target (Arc targets B200/H100 at peak performance, not single-GPU offload).

**Bottom line on bandwidth math:** Speculative expert prefetch is *huge* under PCIe offload, *modest* under multi-GPU EP, and *zero* in fully-resident HBM. Arc's target deployment of V4 on B200 with EP is Case 2 — the realistic win is **single-digit-percent decode latency**, not a step-change.

---

## 5. Gap assessment — novel or done?

**The base mechanism is fully published.** Cross-layer prefetch with reused gates (Cross-Layer Gate), trained predictors (ExpertFlow, SiDA-MoE, Pre-gated MoE), and draft-model-aligned prediction (SP-MoE) are all in the literature. Throughput claims of 1.5×-3.5× exist. Naming "speculative routing" is also done.

**What is *not* in the literature and *would* be novel for Arc:**

1. **Empirical cross-layer correlation on DeepSeek-V4 specifically.** Every paper measures Mixtral, DeepSeek-V2/V2-Lite, Qwen, OLMoE, Phi-MoE, Switch. **None measure V4.** V4 has structural reasons (aux-loss-free balancing, hash routing in early layers, 256 fine-grained experts) to behave differently. **This is a real measurement gap.**

2. **Cross-layer prefetch integrated with a capture-once CUDA graph.** All published systems use stream-level scheduling with host-side glue. Arc has a *zero-host-sync* decode loop (`arc-cuda-graph`). Capturing a side-stream `cudaMemcpyAsync` inside a graph that runs autonomously is non-trivial and not described in the literature. The interaction with DeepEP all-to-all under graphs is also novel territory.

3. **Speculative routing combined with V4's MTP head and DSA (DeepSeek Sparse Attention).** V4 has multi-token-prediction; prefetching for the MTP-drafted continuation would be a 2-step lookahead that nobody has tried at scale.

4. **Cross-layer routing as a *signal to skip* layer L+1's gate entirely** (Option B in section 3), measuring how often router decisions are *identical* in V4 — could plausibly elide a microscopic but cumulative cost. Nobody has measured this on V4.

**Net assessment:** The conceptual gap is small (~3 percent novelty); the empirical-on-V4 gap is real (~80 percent novelty); the graph-integration engineering gap is moderate (~50 percent novelty). Arc would not be publishing a research paper, but a well-executed Arc-on-V4 prefetcher could be a real engineering win over SGLang.

---

## 6. Estimated effort to ship in Arc

Calibrated in agent-sessions:

| Step | Description | Estimate |
|------|-------------|----------|
| 1 | Offline empirical study: instrument V4 to log per-token routing (layer, top-k expert IDs); compute P(L+1 | L), top-k overlap, conditional entropy across all 43 layers; compare aux-loss-free balancing era vs hash-routed first 3 layers | ~2 hours |
| 2 | Decision gate: if measured top-6 overlap < 50 percent, **kill the project** and write it up as "V4 prefetch is bandwidth-irrelevant" | ~30 min |
| 3 | Prototype Cross-Layer Gate (Option A) in `deepseek4.rs` MoeBlock forward; non-graph path first | ~2-3 hours |
| 4 | Integrate prefetch with side-stream `cudaMemcpyAsync` captured inside the `arc-cuda-graph` decode graph | ~3-4 hours |
| 5 | Benchmark on B200 single-GPU and 8×B200 EP setup; compare against SGLang two-batch-overlap baseline | ~2-3 hours |
| 6 | (Optional) DeepEP speculative dispatch — kick off all-to-all on predicted experts before real gate fires | ~4-6 hours |

**Total to a measured yes/no on V4: ~5 hours of agent time. Total to ship a working production path if positive: ~10-15 hours of agent time. Step 2 is the critical decision point — do not skip it.**

---

## 7. Honest verdict

**The cross-layer routing correlation is real and empirically established.** Reproducing it on V4 is a legitimate exercise. But: (a) the design space is crowded; (b) DeepSeek-V4's auxiliary-loss-free balancing and hash-routed early layers *probably* weaken the effect compared to the literature's Mixtral/Qwen numbers — magnitude unknown until measured; (c) on Arc's target deployment of B200 + EP with all experts in HBM, the bandwidth win is single-digit percent at best, not the 1.5-2.5× the offload literature shows.

**Recommended path:**
1. Run the offline measurement (section 6, step 1). It's cheap (~2 hours agent time) and the answer is load-bearing.
2. If top-6 next-layer overlap > 60 percent on V4, ship Option A (parameter-free, zero quality risk). The prefetch behaves like an L2-warming hint at worst and a real bandwidth saver at best.
3. If overlap < 40 percent, the auxiliary-loss-free balancer has done its job, and there is no signal to exploit — drop the project, save the engineering for fused kernels and CUDA graph extensions where the bandwidth math is more compelling.
4. **Do not pursue Option C (pre-gated style) under any circumstances** — quality risk on a frontier model is not worth a marginal decode speedup.

**What this is NOT:** This is not a novel research direction. There are >10 systems-paper precedents. It's an engineering integration play. The opportunity is "do this carefully on V4 with Arc's capture-once decode loop" — that's worth doing iff the measurement comes back positive.

> Noticed (per Aeonmind principle 3): the *real* untapped lever from this survey is **SGLang's existing EPLB + Arc's CUDA-graph integration**. EPLB does placement; Arc does graphs. Marrying them — graph-captured EP dispatch that respects EPLB's expert placement — is plausibly a bigger win than prefetch, and orthogonal to the prefetch question. Worth a separate change.

---

## Citations and links

### Direct primary sources
- ExFlow — [arXiv:2401.08383](https://arxiv.org/abs/2401.08383), [code](https://github.com/YJHMITWEB/ExFlow)
- Cross-Layer Gate — [arXiv:2502.12224](https://arxiv.org/html/2502.12224v1) (Fang et al., 2025)
- Speculative MoE (s-MoE) — [arXiv:2503.04398](https://arxiv.org/html/2503.04398v2) (Li et al., 2025)
- PreScope / LayerScope — [arXiv:2509.23638](https://arxiv.org/abs/2509.23638) (Sep 2025)
- ExpertFlow — [arXiv:2410.17954](https://arxiv.org/html/2410.17954v2) (Yuan et al., 2024)
- Pre-gated MoE (ISCA 2024) — [arXiv:2308.12066](https://arxiv.org/html/2308.12066v3), [Microsoft camera-ready](https://www.microsoft.com/en-us/research/wp-content/uploads/2024/05/isca24_pregated_moe_camera_ready.pdf)
- SiDA-MoE (MLSys 2024) — [arXiv:2310.18859](https://arxiv.org/abs/2310.18859)
- SP-MoE — [arXiv:2510.10302](https://arxiv.org/pdf/2510.10302) (Chen et al., 2025)
- SpecMD survey — [arXiv:2602.03921](https://arxiv.org/html/2602.03921)
- MoE-Spec — [arXiv:2602.16052](https://arxiv.org/abs/2602.16052) (Feb 2026, McDanel)
- MoE-SpeQ — [arXiv:2511.14102](https://arxiv.org/abs/2511.14102)
- MoE-Infinity — [arXiv:2401.14361](https://arxiv.org/html/2401.14361v2)
- Pre-Attention Expert Prediction (ETH thesis) — [research-collection.ethz.ch PDF](https://www.research-collection.ethz.ch/server/api/core/bitstreams/59a0a37d-f1b5-4d3a-9b25-c65814ab0aab/content)
- PROBE — [arXiv:2602.00509](https://arxiv.org/pdf/2602.00509)
- DuoServe-MoE — [arXiv:2509.07379](https://arxiv.org/html/2509.07379v1)
- Cross-layer regularization for specialization — [arXiv:2602.14159](https://arxiv.org/html/2602.14159)
- Decoding Knowledge Attribution (Mixtral attn-routing r=0.68) — [arXiv:2505.24593](https://arxiv.org/pdf/2505.24593)
- Multilingual Routing in MoE — [arXiv:2510.04694](https://arxiv.org/pdf/2510.04694)
- Opportunistic Expert Activation — [arXiv:2511.02237](https://arxiv.org/abs/2511.02237)
- Expert Load Distribution Stabilization — [arXiv:2404.16914](https://arxiv.org/html/2404.16914)
- TIDE (diffusion MoE temporal locality) — [arXiv:2605.20179](https://arxiv.org/html/2605.20179v1)

### Model/system references
- DeepSeek-V3 Technical Report — [arXiv:2412.19437](https://arxiv.org/abs/2412.19437)
- DeepSeek-V3.2 Technical Report — [arXiv:2512.02556](https://arxiv.org/pdf/2512.02556)
- DeepSeekMoE (256-expert origin) — [arXiv:2401.06066](https://arxiv.org/pdf/2401.06066)
- DeepSeek hardware insights — [arXiv:2505.09343](https://arxiv.org/html/2505.09343v1)
- PowerInfer (hot/cold neuron analog) — [arXiv:2312.12456](https://arxiv.org/abs/2312.12456), [code](https://github.com/SJTU-IPADS/PowerInfer)
- DeepSpeed-MoE — [arXiv:2201.05596](https://arxiv.org/pdf/2201.05596)
- SGLang Expert Parallelism docs — [docs.sglang.io](https://docs.sglang.io/advanced_features/expert_parallelism.html)
- SGLang DeepSeek V3 large-scale EP deployment — [LMSYS blog](https://www.lmsys.org/blog/2025-05-05-large-scale-ep/)
- Mixtral of Experts (Mistral AI) — original paper at [arXiv:2401.04088](https://arxiv.org/abs/2401.04088)

### Arc internal references
- `/Users/jish/Documents/GitHub/arc/mistralrs-core/src/models/deepseek4.rs` — V4 model implementation, MoE block at lines ~1050-1250, V4 Flash config at ~2024-2030 (`n_routed_experts=256`, `num_experts_per_tok=6`, `num_hidden_layers=43`, `moe_intermediate_size=2048`).
- `/Users/jish/Documents/GitHub/arc/research/inference_sparsity_survey.md` — related Arc sparsity work
- `/Users/jish/Documents/GitHub/arc/research/predictor_sparsity_deepdive.md` — predictor design context
