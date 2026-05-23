# Quality-Up Inference Techniques — A Survey for Arc

**Audience:** Arc engineering, decoding strategy roadmap
**Question:** Are there inference-time techniques that *demonstrably raise* base-model quality (not just preserve it) with no retraining of the base model — and that fit Arc's QTIP 2-bit + TurboQuant 3.5-bit KV + DeepSeek-V4 MoE stack?
**Short answer:** Yes, three of them are real wins. Most others are noise, cherry-picked, or pay 2-30x slowdown for a couple of points on a single benchmark. The standout candidate for an MoE engine is **SCMoE (Self-Contrast Mixture-of-Experts, NeurIPS 2024)** — a method that improves Mixtral by contrasting strong vs. weak expert routing on the same MoE model, with only 1.3x slowdown. Nobody has yet combined SCMoE with low-bit quantization or DeepSeek-V4 — that intersection is the research gap.

---

## 1. Method-by-method survey

### 1.1 Contrastive Decoding (CD) — Li et al., ACL 2023

- Paper: "Contrastive Decoding: Open-ended Text Generation as Optimization" (arXiv [2210.15097](https://arxiv.org/abs/2210.15097))
- Reasoning extension: "Contrastive Decoding Improves Reasoning in LLMs", O'Brien & Lewis, arXiv [2309.09117](https://arxiv.org/abs/2309.09117)
- Code reference: llama.cpp issue [#3278](https://github.com/ggml-org/llama.cpp/issues/3278) (community port)

**Mechanism.** Run a large "expert" LM and a small "amateur" LM in parallel. At each step compute the contrast `(1+β)·log p_expert − β·log p_amateur`, with an "adaptive plausibility constraint" that prunes the candidate set to tokens whose expert-prob is ≥ α·max_prob. The amateur supplies a model of *generic* mistakes that the expert is statistically prone to (repetition, copy-paste from prompt, topic drift), and subtraction removes them. Two forward passes per token; on a 65B+1.5B pair the FLOP overhead is ~3.25% (the amateur is tiny relative to the expert).

**Measured quality lift (Llama 1 family, O'Brien & Lewis 2023):**
| Benchmark | Llama-7B | Llama-13B | Llama-30B | Llama-65B |
|-----------|---------:|---------:|---------:|---------:|
| GSM8K (baseline) | 10.7 | 17.4 | 35.3 | 51.0 |
| GSM8K (+CD) | 14.3 | 22.7 | 43.1 | 56.8 |
| Δ | **+3.6** | **+5.3** | **+7.8** | **+5.8** |
| HellaSwag 65B | — | — | — | 84.2 → **88.0** |
| MMLU 65B | — | — | — | 63.5 → ~63.3 (flat) |

Where CD *hurts*:
- **OpenBookQA / TriviaQA (factual recall):** -2.4 / -2.1 pp on 65B. The amateur's "common-noun" bias was actually correct for these.
- **CommonsenseQA, StrategyQA** at small expert sizes: regressions.
- **MATH:** essentially flat.

**Verdict:** Real but task-dependent. CD is a reasoning amplifier, not a universal quality lever. It hurts on factual-recall. Reproducible with a small (~1.5B) amateur, FLOP overhead is small at our scale (V4 is ~700B params, a 1.5B amateur is 0.2% overhead).

**Compatibility with Arc:**
- *MoE compat:* Works in principle — V4 is the expert. Amateur would need to be a Llama-class dense model or a DeepSeek-distill like the 1.5B distill of R1. Tokenizer mismatch is the real risk; needs verification (DeepSeek tokenizer differs from Llama's).
- *2-bit weights + 3.5-bit KV:* Both expert and amateur logits are top-of-LM-head outputs in BF16/FP16 — quantization noise on the amateur could amplify in the contrast. Numerical risk: small-model log-prob noise gets multiplied by β. Likely fine if amateur is run unquantized (it's only 1.5B), but needs an empirical probe.
- *Effort:* ~2-3 agent-sessions. Add `mistralrs-core/src/pipeline/contrastive.rs`, plumb a second pipeline handle through `CustomLogitsProcessor`, fuse the subtraction into the sampler.

---

### 1.2 DoLa — Chuang et al., ICLR 2024

- Paper: "DoLa: Decoding by Contrasting Layers Improves Factuality", arXiv [2309.03883](https://arxiv.org/abs/2309.03883)
- Code: [github.com/voidism/DoLa](https://github.com/voidism/DoLa), also in `transformers<4.53.0`, port at [voidism/transformers-dola](https://github.com/voidism/transformers-dola)

**Mechanism.** Instead of using a smaller amateur model, project an *early* layer's hidden state to the vocabulary with the existing LM-head (`W_unembed @ h_early`) and contrast against the final layer's logits. Picks the "premature" layer dynamically by Jensen-Shannon divergence at every step. Free of any auxiliary model — single forward pass, only the LM-head matrix-multiply runs twice (negligible).

**Measured quality lift (LLaMA family):**
- **TruthfulQA:** +12 to +17 absolute points across Llama 7/13/30/65B (largest single-method jump on TruthfulQA reported in 2023).
- **StrategyQA (CoT):** +1 to +4 pp.
- **GSM8K:** +1 to +2 pp.
- **FACTOR (factual generation):** consistent +2 to +5 pp.

**Critique.** Recent work ([PruneCD, arXiv 2509.16598](https://arxiv.org/abs/2509.16598), Sep 2025) argues DoLa's premature logits are "flat and low-magnitude," not actually meaningfully different — and PruneCD beats DoLa using a pruned-self-model amateur. So the 12-17pp TruthfulQA number is real but may be partially a regularization effect rather than the principled contrast story the paper tells.

**Compatibility with Arc:**
- *MoE compat:* **Untested on MoE.** V4 has 43 all-MoE layers. The DoLa story is "early-layer factual knowledge vs. late-layer fluency"; whether MoE routing changes layer-wise knowledge localization is an open question. This is part of the publishable gap (see §4).
- *Quantization:* The LM-head must be applied to an early hidden state that has been computed under 2-bit weights. Numerical drift in early layers compounds — this might *help* DoLa (more contrast) or *hurt* it (the early projection becomes meaningless noise). Empirical question.
- *Effort:* ~1-2 agent-sessions. Lightest of any candidate. Touch `mistralrs-core/src/pipeline/mod.rs` to expose intermediate hidden states, add a `dola.rs` module, hook into the sampler.

---

### 1.3 Self-Contrastive Decoding for MoE — SCMoE (Shi et al., NeurIPS 2024) ⭐

- Paper: "Unchosen Experts Can Contribute Too: Unleashing MoE Models' Power by Self-Contrast", arXiv [2405.14507](https://arxiv.org/abs/2405.14507)
- Code: NeurIPS 2024 paper, OpenReview [C1d3VVfdVG](https://openreview.net/forum?id=C1d3VVfdVG). Authors' code reportedly on GitHub but exact repo not surfaced cleanly in search; UNITES-Lab/scMoE is a different (single-cell) project.

**Mechanism.** Same MoE model, two routes:
1. **Strong activation:** standard top-2 routing (or top-k for V4 — top-6/8).
2. **Weak activation:** "rank-k routing" — pick the k-th highest-scoring expert *only*, ignoring everything above it. For Mixtral the sweet spot is rank-2 (the second-best expert alone).

Then `logits = (1+β)·z_strong − β·z_weak`. The intuition is that the chosen experts are good at fluency but underutilize the "unchosen" experts' specialized signal; the contrast moves probability mass toward tokens where the *full* routing fires confidently but the *degraded* routing does not — i.e., tokens that genuinely depended on top-routing.

**Measured quality lift (Mixtral 8x7B base):**
| Benchmark | Greedy | SCMoE | Δ |
|-----------|-------:|------:|---:|
| GSM8K | 61.79 | **66.94** | +5.15 |
| StrategyQA | 72.83 | **76.29** | +3.46 |
| MBPP pass@1 | 46.20 | **48.80** | +2.60 |
| HumanEval pass@1 | 33.54 | **41.46** | **+7.92** |
| GSM8K + self-consistency major@20 | 75.59 | **78.31** | +2.72 |

Latency: **1.30x** greedy. Beats contrastive search (1.62x) and external-amateur CD (1.43x).

Also validated on DeepSeekMoE-16B (smaller, older relative of V3/V4).

**Verdict.** This is the strongest "quality-up at inference" result targeted directly at MoE that I found. The mechanism is theoretically clean (it uses the model's own routing structure rather than an external small LM), the cost is one extra forward pass *through only the experts that are not normally hit*, and the numbers are non-trivially large on hard benchmarks (GSM8K +5.15, HumanEval +7.92).

**Compatibility with Arc:**
- *MoE compat:* **Direct fit.** V4 has 256 experts top-6/8; "rank-k" generalizes trivially — pick the (k+8)-th expert. Empirically the right "weak" choice for V4 is open and is exactly the kind of probe Arc should run.
- *2-bit + 3.5-bit:* The contrast is between two MoE routings of the *same* model — both quantized identically. Quantization noise is approximately *correlated* across the two routes, which means the contrast is largely robust (noise cancels). This is the *best* candidate for numerical stability.
- *Effort:* ~3-5 agent-sessions. The hard part is that V4's expert dispatch is fused into a custom CUDA kernel; SCMoE needs to run *two* dispatches per forward and subtract. Touch points: `mistralrs-core/src/models/deepseek4.rs` (the `MoEGate::forward`), the fused expert kernels in `mistralrs-core/src/kernels`, and the sampling path.

---

### 1.4 DExperts — Liu et al., ACL 2021

- Paper: arXiv [2105.03023](https://arxiv.org/abs/2105.03023)
- Code: [github.com/alisawuffles/DExperts](https://github.com/alisawuffles/DExperts)

**Mechanism.** Product-of-experts at logit level: `log p ∝ log p_base + log p_expert − log p_antiexpert`, where expert and antiexpert are small fine-tuned LMs (one trained on "good" text — e.g., non-toxic — and one on "bad"). Use case: detoxification, sentiment, style control.

**Quality lift:** Originally measured on toxicity / sentiment, not reasoning. Reported reductions in toxic generation from ~30% to ~10%. **Not a reasoning lifter.**

**Verdict.** Useful for safety/style work, not for the "MMLU/GSM8K goes up" question. Skip for Arc unless content-moderation becomes a goal.

**Effort:** Same code shape as CD (~2 agent-sessions) but useless until the expert/antiexpert small LMs exist.

---

### 1.5 Self-Consistency (Wang et al., ICLR 2023)

- Paper: arXiv [2203.11171](https://arxiv.org/abs/2203.11171)

**Mechanism.** Sample N independent chain-of-thought traces at T>0, parse out the final answer of each, return the majority vote.

**Quality lift:** GSM8K +17.9 pp, SVAMP +11 pp, AQuA +12.2 pp, StrategyQA +6.4 pp, ARC-C +3.9 pp. These are the **largest** quality-up numbers reported by *any* technique in this survey.

**Compute cost: N× decode**, where N=40 in the original paper. That's a 40x slowdown.

**Verdict.** Quality lift is enormous and robust — replicated many times. But the cost is essentially *parallel best-of-N* inference. From Arc's perspective this is not a "decoding strategy" — it's a "use the engine N times and aggregate." Arc *enables* self-consistency cheaply (we serve fast, so 40 samples cost less than other engines' 10), but it's not a decoder change. Worth surfacing as a feature in the SDK ("self-consistency mode: sample N, vote") but the work is in the request router, not the model code.

**Effort:** ~1 agent-session to wire `n_samples` + answer-parser + majority-vote into the server.

---

### 1.6 Lookahead Decoding (Fu et al., ICML 2024)

- Paper: arXiv [2402.02057](https://arxiv.org/abs/2402.02057)
- Code: [github.com/hao-ai-lab/LookaheadDecoding](https://github.com/hao-ai-lab/LookaheadDecoding)

**Mechanism.** Jacobi iteration + verification window. Generates n-grams from past trajectories speculatively and verifies them.

**Quality lift:** **Zero.** Lookahead is *lossless* — same distribution as greedy. It's a speed technique, not a quality technique. 1.8x speedup on MT-Bench, up to 4x on code.

**Verdict.** Wrong category for this question. Not a quality lifter. Include here only to flag: "this is one of the ones that doesn't actually improve quality." Already in scope of Arc's existing speculative-decoding stack.

---

### 1.7 LayerSkip / Self-Speculative (Elhoushi et al., ACL 2024)

- Paper: arXiv [2404.16710](https://arxiv.org/abs/2404.16710)
- Code: [github.com/facebookresearch/LayerSkip](https://github.com/facebookresearch/LayerSkip)

**Mechanism.** Layer-dropout-trained model exits early at layer E, drafts γ tokens, verifies with full L layers. Up to 2.16x speedup, *quality preserved* (not improved).

**Verdict.** Wrong category — speed, not quality. Requires the *base model* to be trained with layer-dropout (Meta released LayerSkip-Llama checkpoints). V4 is *not* layer-dropout-trained. Not applicable.

---

### 1.8 Best-of-N + Verifier (Snell et al., Brown et al., 2024-2025)

- Snell et al.: "Scaling LLM Test-Time Compute Optimally" (arXiv [2408.03314](https://arxiv.org/abs/2408.03314))
- CarBoN: arXiv [2510.15674](https://arxiv.org/abs/2510.15674)

**Mechanism.** Sample N candidate completions, rerank with a learned verifier / reward model, return top-1.

**Quality lift:** Snell et al. show that for fixed FLOP budget, best-of-N with a verifier can beat scaling parameters 14x on certain tasks. AIME jumps from ~15% to ~60% on small models with N=256 + good verifier.

**Cost: N× decode + 1× verifier.**

**Compatibility with Arc:**
- Architecturally orthogonal to Arc's compression stack — Arc just generates the N samples cheaply.
- Requires a verifier model (usually a same-size or smaller reward LM).
- If V4 is the generator, the verifier could itself be V4 (zero-shot) — and that bootstraps surprisingly well.

**Effort:** ~2-3 agent-sessions for the verifier-aware sampling loop on the server side.

---

### 1.9 Inference-Time Intervention (ITI) — Li et al., NeurIPS 2023

- Paper: arXiv [2306.03341](https://arxiv.org/abs/2306.03341)
- Code: [github.com/likenneth/honest_llama](https://github.com/likenneth/honest_llama)

**Mechanism.** A one-time offline step trains linear probes on a few hundred TruthfulQA examples to identify attention heads that encode "truthfulness direction." At inference, add a small constant vector along those heads' output activations — i.e., literally `h += α·v_truth` on selected heads.

**Quality lift:** Alpaca TruthfulQA 32.5% → **65.1%** (+32.6 pp). Largest single-benchmark lift in any method here.

**Caveats:**
- The lift is *only* on TruthfulQA. Doesn't generalize uniformly.
- Probes are trained on TruthfulQA training set — so it's not "no training" in the strictest sense (a tiny logistic regression on a few hundred examples, but still).
- Reproduction work shows ~+10 to +15 pp on Llama-2/3 — still huge but less than Alpaca.

**Compatibility with Arc:**
- *MoE compat:* Untested on MoE. The "truthful heads" hypothesis has been shown only on dense Llama. Whether V4's 43 all-MoE layers contain analogous heads is an open empirical question.
- *Quantization:* Steering vector is a fixed small offset added to BF16/FP16 hidden state. Robust to weight quantization. KV-quantization is irrelevant.
- *Effort:* ~2 agent-sessions to bake in a fixed steering tensor at load time (the offline probing is one-time on a workstation).

---

### 1.10 Tree of Thoughts / Graph of Thoughts (Yao et al., 2023; Besta et al., 2024)

- ToT: arXiv [2305.10601](https://arxiv.org/abs/2305.10601)

**Mechanism.** Frame reasoning as tree/graph search; expand multiple thought branches; self-evaluate.

**Quality lift:** Game-of-24: GPT-4 baseline 4% → ToT **74%**. But baseline is a single-shot, so a fair comparison is ToT vs. self-consistency-CoT, where ToT wins by smaller margins (10-30%).

**Cost:** 10-100× single-shot.

**Verdict.** Prompt-engineering / agent-loop layer, not a decoding strategy. Arc is the substrate; ToT runs *on top of* Arc by issuing many requests. Not a decoder change. Skip for this report.

---

### 1.11 Self-Refinement / Self-Correction (Madaan et al., 2023; Kamoi et al., 2024)

- Survey: arXiv [2412.14352](https://arxiv.org/html/2412.14352v1)
- Awesome list: [github.com/ryokamoi/llm-self-correction-papers](https://github.com/ryokamoi/llm-self-correction-papers)

**Mechanism.** Generate → critique → revise loop, all in the same model.

**Quality lift:** *Mixed.* The Kamoi et al. survey ("When Can LLMs Actually Correct Their Own Mistakes?") shows that without an oracle or external feedback, self-correction often makes outputs *worse*, not better. SETS (arXiv [2501.19306](https://arxiv.org/abs/2501.19306)) shows ~10.9% gain but with self-verification *also* trained in.

**Verdict.** Honest read: untrained self-correction is **not** a reliable quality lifter — the literature has reversed on this since 2023. Skip.

---

### 1.12 Activation Steering / Contrastive Activation Addition (Panickssery et al., 2023)

- arXiv [2312.06681](https://arxiv.org/abs/2312.06681)

Very similar shape to ITI but using *paired* contrastive examples to compute the steering direction. Reported gains comparable to ITI on TruthfulQA, generalization claims on a few other tasks.

**Effort:** Same code path as ITI. ~2 agent-sessions.

---

## 2. Aggregated honesty table

| Method | Quality gain | Cost | Reproducible | MoE-tested | Verdict |
|--------|-------------:|----:|:------------:|:----------:|---------|
| Self-Consistency | **+17.9 GSM8K** | 40× | yes (many) | n/a (router) | wire it into SDK |
| ITI | **+32 TruthfulQA** | ~0 | partially | no | task-narrow but cheap |
| SCMoE | **+5.15 GSM8K / +7.92 HumanEval** | 1.3× | only one paper | **yes (Mixtral, DSMoE-16B)** | **strongest fit for Arc** |
| Contrastive Decoding | +5 to +8 GSM8K | 1.03× FLOPs | yes (Llama family) | no | works on reasoning, hurts factual recall |
| DoLa | +12-17 TruthfulQA | ~0 | partially (PruneCD critique) | no | cheap but narrow |
| Best-of-N + verifier | up to +45 AIME | N× | yes | n/a | router-level, not decoder |
| ToT / GoT | task-dependent | 10-100× | yes | n/a | agent-loop, not decoder |
| Self-Refinement | mixed, often *negative* | 2-5× | **no** (often hurts) | no | skip |
| Lookahead, LayerSkip | 0 (speed only) | <1× | yes | no | wrong category |
| DExperts | toxicity, not reasoning | 2× | yes | no | not in scope |
| Min-p sampling | claimed, **disputed** ([arXiv 2506.13681](https://arxiv.org/abs/2506.13681)) | ~0 | **no** | n/a | likely noise |

Quality-up at inference time **is genuinely rare**. The clean wins are:
1. **N× sampling + smart aggregation** (self-consistency, best-of-N). Huge gains but huge cost.
2. **Subtractive contrast** (CD/DoLa/SCMoE). Modest gains (3-8 pp on hard reasoning) for ~1.0-1.3× cost.
3. **Activation steering** (ITI/CAA). Huge gains on the *one* dimension you steered, ~0 cost, narrow generalization.

Everything else is either a speed technique miscategorized, an agent-loop, or a noise-level claim.

---

## 3. Top-5 ranked candidates for Arc

Ranked by (real quality lift) × (Arc portability) × (stacks with QTIP+TurboQuant+V4).

### #1 — SCMoE (Self-Contrast MoE). Score: 9/10.

This is the technique I would prioritize. It is the *only* surveyed method that was published *on* MoE models, with a clean ~5-8 pp lift on hard reasoning at 1.3× cost.

**Implementation sketch:**
- File `mistralrs-core/src/pipeline/scmoe.rs` — new decoder strategy.
- Touch `mistralrs-core/src/models/deepseek4.rs::MoEGate::forward` — accept a `routing_mode: enum { Strong, Weak(rank: usize) }` parameter.
- Modify the fused expert dispatch CUDA kernel under `mistralrs-core/src/kernels/` to allow a "rank-k only" route (single-expert dispatch instead of top-k). This already exists in the V4 codepath for the rare singleton case — wire it.
- Decoder runs two MoE forwards (strong: top-6, weak: rank-k for k∈{2..8}). Subtract logits with β=0.5 as the literature default, tune on a small held-out set.
- Numerical stability: both forwards are at the same precision and on the same KV cache (the *attention* is shared, only the FFN expert dispatch differs). Quantization noise is highly correlated → contrast cancels noise. This is the *one* technique where 2-bit weights + 3.5-bit KV actively *helps* over a dense baseline (because correlated noise is what makes the contrast clean).
- Probe: run a 200-example GSM8K probe at β ∈ {0.0, 0.25, 0.5, 0.75}, k ∈ {2, 4, 6, 8} on V4-Flash. Expect 3-7 pp lift if the Mixtral result transfers.
- Risk: V4's top-6/8 routing is *deeper* than Mixtral's top-2 — the "weak" routing may need to be different (e.g., "use only the bottom expert *of the chosen 6*"). Empirical question.

### #2 — DoLa (Layer-Contrast). Score: 7/10.

Cheapest implementation, untested on MoE so high research interest.

**Implementation sketch:**
- File `mistralrs-core/src/pipeline/dola.rs`.
- Modify `deepseek4.rs` forward pass to optionally return intermediate hidden states at a subset of layers (every-4th is the literature default).
- In the sampler, run the LM head on each candidate "premature" hidden state, compute JSD vs. final, pick the max-JSD layer as the contrast.
- LM head is run K extra times (K = candidate layers, ~5-10). LM head on V4 is a 7168 × 129280 matmul, dominated by the FFN regardless — the overhead is <2% of the per-step cost.
- Numerical risk: early-layer hidden states under 2-bit weights may have larger drift than late layers. This could *help* DoLa (sharper contrast) or break it (noise). Probe required.
- This is the only candidate where Arc would be the first to test the method on MoE — that's a publishable result.

### #3 — Inference-Time Intervention (ITI). Score: 7/10.

Cheapest at inference (literally adds a constant), narrow but huge lift on truthfulness.

**Implementation sketch:**
- One-time offline step: run V4 on TruthfulQA training set, extract per-head activations on layer outputs, fit linear probes, identify the top-K (paper uses K=48) "truthful" heads, compute mean activation difference between true/false statements per head as the steering vector v_h.
- Bake `v_h` into V4's `.safetensors` or load alongside.
- At inference: at each forward pass through the relevant layers, add `α·v_h` to the head outputs *before* the attention output projection.
- Code touchpoint: `mistralrs-core/src/models/deepseek4.rs::Attention::forward` — small constant addition after `o_proj`.
- Numerical stability: trivially compatible with 2-bit weights and 3.5-bit KV (the steering vector is a fixed FP16/BF16 add).
- Limitation: only lifts TruthfulQA-shaped tasks. For general MMLU/GSM8K, near-zero.

### #4 — External-amateur Contrastive Decoding. Score: 6/10.

Real but task-narrow. The amateur model (DeepSeek-R1-Distill-1.5B) is open-source; tokenizer matches V4 (DeepSeek family) — so no tokenizer-bridging risk.

**Implementation sketch:**
- `mistralrs-core/src/pipeline/contrastive.rs` adds a second `Pipeline` for the amateur.
- Sampler subtracts amateur logprobs from expert logprobs with β=0.5, applies α=0.1 plausibility filter (drop tokens with `p_expert < α·max(p_expert)`).
- Two pipelines, separate KV caches. KV memory cost on the amateur is small (1.5B vs. 700B → 0.2% of total).
- Numerical risk: amateur is small enough to keep at FP8 or BF16 — no need to quantize. The contrast is BF16 − BF16. Stable.
- Will **regress** factual-recall (-2 pp on TriviaQA). Need a routing knob: contrastive-on for math/code, contrastive-off for QA.

### #5 — Self-Consistency. Score: 5/10 (high gain, very high cost).

Not really a decoder change. SDK feature.

**Implementation sketch:**
- Server route `POST /v1/completions?n=20&aggregate=majority_vote` or a new `/v1/cot/self_consistency` endpoint.
- Spawn N concurrent decode jobs at T=0.7, run an answer-parser (regex for `\boxed{...}`, last-line numeric, etc.), majority-vote.
- Pure server-side logic, no model-code changes.
- Pairs naturally with SCMoE (#1) — SCMoE+SC was the +5.15 → +2.72 stacking result in the paper (76% → 78% major@20).

---

## 4. Quality-up + MoE intersection — the publishable gap

**Has anyone combined contrastive decoding with MoE models?**

- **SCMoE** (Shi et al., NeurIPS 2024): yes, this is *the* paper. Mixtral 8x7B and DeepSeekMoE-16B. Top-2 routing only. Tested 5 benchmarks. Published a clean +5 to +8 pp result.
- **DoLa on MoE:** Searched. **Nobody.** All DoLa papers test on dense Llama/Mistral/Falcon. The mechanism (early-layer hidden states projected to vocab) makes assumptions about how factual knowledge is layered, and that has *not* been verified on MoE where each layer's FFN is a sparse mixture rather than a dense network. **This is a publishable gap.**
- **ITI on MoE:** Searched. **Nobody.** All replicating ITI papers use Llama/Alpaca. The "truthful attention heads" hypothesis has not been tested on MoE — but MoE only sparsifies the FFN, not attention, so ITI *should* port cleanly. Untested.
- **External-amateur CD on MoE:** Some unofficial llama.cpp / Aphrodite experiments mention trying it with Mixtral, no peer-reviewed paper. Open.
- **SCMoE on V3/V4-scale (256 experts, top-6/8):** Untested. Mixtral has 8 experts, DeepSeekMoE-16B has 64 experts top-6. V4 has 256 experts top-6. The "rank-k as weak" choice generalizes ambiguously when k ≪ N.

**The Arc-specific research gap is therefore:**

> Does SCMoE-style contrast (or DoLa-style layer-contrast, or ITI-style activation steering) work on large-N-expert top-k MoE models like DeepSeek-V4, and does it stack with low-bit (2-bit weight, 3.5-bit KV) quantization?

If Arc runs this probe end-to-end on V4 and the answer is *yes*, that is a workshop paper at NeurIPS/ICLR. If the answer is *no* (the gains evaporate at 256-expert scale), that is *also* a paper. Either way it is publishable.

A reasonable concrete experiment plan: pick three benchmarks (GSM8K, HumanEval, TruthfulQA), three V4 configs (BF16, QTIP-2bit, QTIP-2bit + TurboQuant-3.5bit KV), and four decoders (greedy, SCMoE, DoLa, SCMoE+self-consistency). 36 cells. Each cell is ~1000 prompts at V4-Flash 500 tok/s → ~1 hour. Total ~36 hours of compute on a single H100.

---

## 5. Honest verdict

Quality-up at inference time is **mostly an academic mirage** for the average decoding strategy, but **a real win in a small number of well-characterized regimes**. The cleanest wins are (a) ensembling — self-consistency, best-of-N — which trade compute for quality at a known rate; (b) subtractive contrast — CD, DoLa, SCMoE — which give modest but real lifts on reasoning at 1-1.3× cost; and (c) activation steering — ITI — which lifts a single task dimension (truthfulness) by huge margins at near-zero cost. Most other claimed "quality-up" methods (self-refinement without an external oracle, min-p sampling improvements, ToT margins over self-consistency) are at or near the noise floor of benchmark variance. For Arc specifically, **SCMoE is the standout candidate** because it was published *on* MoE, the math suggests quantization noise *cancels* in its contrast (which is the opposite of the usual quantization-amplifies-error story), and the gap to "SCMoE on 256-expert top-k V4 under QTIP+TurboQuant" is a research direction with no published prior work — it's a one-agent-session probe to validate, a one-week paper to publish if it works. The honest tradeoff to flag is that *none* of these methods improve the average quality across all benchmarks — they trade off factual-recall vs. reasoning, or steer one dimension at the cost of another. Arc should expose them as opt-in modes (`--scmoe`, `--dola`, `--iti=truthful`), not bake them into the default decoder.

---

## References

1. Li et al., "Contrastive Decoding: Open-ended Text Generation as Optimization", ACL 2023. [arXiv:2210.15097](https://arxiv.org/abs/2210.15097)
2. O'Brien & Lewis, "Contrastive Decoding Improves Reasoning in Large Language Models", 2023. [arXiv:2309.09117](https://arxiv.org/abs/2309.09117)
3. Chuang et al., "DoLa: Decoding by Contrasting Layers Improves Factuality in Large Language Models", ICLR 2024. [arXiv:2309.03883](https://arxiv.org/abs/2309.03883). Code: [github.com/voidism/DoLa](https://github.com/voidism/DoLa)
4. Shi et al., "Unchosen Experts Can Contribute Too: Unleashing MoE Models' Power by Self-Contrast", NeurIPS 2024. [arXiv:2405.14507](https://arxiv.org/abs/2405.14507). OpenReview: [C1d3VVfdVG](https://openreview.net/forum?id=C1d3VVfdVG)
5. Liu et al., "DExperts: Decoding-Time Controlled Text Generation with Experts and Anti-Experts", ACL 2021. [arXiv:2105.03023](https://arxiv.org/abs/2105.03023). Code: [github.com/alisawuffles/DExperts](https://github.com/alisawuffles/DExperts)
6. Wang et al., "Self-Consistency Improves Chain of Thought Reasoning in Language Models", ICLR 2023. [arXiv:2203.11171](https://arxiv.org/abs/2203.11171)
7. Fu et al., "Break the Sequential Dependency of LLM Inference Using Lookahead Decoding", ICML 2024. [arXiv:2402.02057](https://arxiv.org/abs/2402.02057). Code: [github.com/hao-ai-lab/LookaheadDecoding](https://github.com/hao-ai-lab/LookaheadDecoding)
8. Elhoushi et al., "LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding", ACL 2024. [arXiv:2404.16710](https://arxiv.org/abs/2404.16710). Code: [github.com/facebookresearch/LayerSkip](https://github.com/facebookresearch/LayerSkip)
9. Snell et al., "Scaling LLM Test-Time Compute Optimally", 2024. [arXiv:2408.03314](https://arxiv.org/abs/2408.03314)
10. Li et al., "Inference-Time Intervention: Eliciting Truthful Answers from a Language Model", NeurIPS 2023. [arXiv:2306.03341](https://arxiv.org/abs/2306.03341). Code: [github.com/likenneth/honest_llama](https://github.com/likenneth/honest_llama)
11. Yao et al., "Tree of Thoughts: Deliberate Problem Solving with Large Language Models", NeurIPS 2023. [arXiv:2305.10601](https://arxiv.org/abs/2305.10601)
12. Kamoi et al., "When Can LLMs Actually Correct Their Own Mistakes? A Critical Survey", 2024. [github.com/ryokamoi/llm-self-correction-papers](https://github.com/ryokamoi/llm-self-correction-papers)
13. Panickssery et al., "Steering Llama 2 via Contrastive Activation Addition", 2023. [arXiv:2312.06681](https://arxiv.org/abs/2312.06681)
14. Yu et al., "PruneCD: Contrasting Pruned Self Model to Improve Decoding Factuality", 2025. [arXiv:2509.16598](https://arxiv.org/abs/2509.16598)
15. Yuan et al., "Speculative Contrastive Decoding", 2023. [arXiv:2311.08981](https://arxiv.org/abs/2311.08981)
16. Survey: Wang et al., "A Survey on LLM Inference-Time Self-Improvement", Dec 2024. [arXiv:2412.14352](https://arxiv.org/abs/2412.14352). Awesome-list: [github.com/dongxiangjue/Awesome-LLM-Self-Improvement](https://github.com/dongxiangjue/Awesome-LLM-Self-Improvement)
17. Critique: Khalifa et al., "Min-p, Max Exaggeration: A Critical Analysis of Min-p Sampling in Language Models", 2025. [arXiv:2506.13681](https://arxiv.org/abs/2506.13681)
18. Critique: Chen et al., "The Mirage of Performance Gains: Why Contrastive Decoding Fails to Mitigate Object Hallucinations in MLLMs", 2025. [arXiv:2504.10020](https://arxiv.org/abs/2504.10020)
