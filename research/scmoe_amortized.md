# Amortized SCMoE: Can the Self-Contrast Correction Be Reused Across Tokens?

**Research investigation — May 2026**

**Verdict up front:** *Marginal-to-viable, with significant unknowns.* The amortization idea is well-motivated by analogous techniques (DeepCache, LoRA-Drop, ActLCD) and by indirect evidence on MoE routing stickiness, but the SCMoE paper itself reports zero direct measurements of cross-token correction stability. We can confidently project N=2 (≈1.15× overhead) at minimal quality loss, plausibly defend N=4 (≈1.075×) on prose-heavy generation, and should *not* trust fixed N≥8 without offline measurement. The strongest path forward is **entropy-gated refresh**, not fixed-stride.

---

## 1. SCMoE primer (verified numbers)

From arXiv 2405.14507 ([HTML version](https://arxiv.org/html/2405.14507)):

* **Latency overhead:** *1.30× greedy* — exactly 65.47 s vs 50.32 s for 512 tokens on Mixtral 8x7B. This sits between Dynamic Routing (1.09×) and external Contrastive Decoding (1.43×) / Contrastive Search (1.62×).
* **Quality lift (Mixtral 8x7B):** GSM8K 61.79 → **66.94** (+5.15), HumanEval 33.54 → **41.46** (+7.92), StrategyQA 72.83 → **76.29** (+3.46), MBPP 46.20 → **48.80** (+2.60).
* **Formula (Eq. 5):** `z_sc = (1+β)·z_top-k − β·z_rank-k`. Default `β = 0.5`, with a plausibility-mask `α = 0.1` (declared "robust", not ablated).
* **"Weak" = rank-k single-expert dispatch**, not bottom-k or random. Main experiments use **rank-2** (second-ranked expert only).
* **Token-position analysis: NONE.** The paper KL-decomposes by token *category* ("Expression" vs "Stopword" vs "All") in the appendix but never measures how the correction `(z_strong − z_weak)` evolves across consecutive positions. **This is the empirical gap the amortization hypothesis hinges on.**

The paper does, however, surface one secondhand signal: *function/stopword tokens show markedly lower routing KLD between strong and weak* than reasoning tokens. That is the seed of any token-type-adaptive schedule (§7).

---

## 2. Has anyone done cross-token contrastive-correction reuse?

**Direct prior art on amortized CD: essentially none.** Searches for "amortized contrastive decoding", "every-Nth contrastive correction", and "cached contrastive logits" return no direct hits. The closest published work is:

* **Speculative Contrastive Decoding (SCD)** — Yuhui Li et al., ACL 2024 ([2311.08981](https://arxiv.org/abs/2311.08981)). Speeds CD up by *parallelizing* (amateur drafts γ=4 tokens, expert verifies in one pass), reaching 1.78–3.32× speedup. But it **recomputes the contrastive delta at every position**; nothing is reused. Acceleration comes from parallel verification, not amortization.
* **Active Layer-Contrastive Decoding (ActLCD)** — ([2505.23657](https://arxiv.org/pdf/2505.23657)). Reinforcement-learned binary policy decides per-step whether to apply DoLa's layer contrast or skip. The paper reports "minimal latency overhead, increasing decoding time over DoLa by only 3–5%" — i.e. the savings come from *skipping* contrast on tokens the policy deems unnecessary, not from interpolation. This is the most direct philosophical prior art for *gated* amortization. It does **not** quantify "what fraction of tokens skip" in the published abstract / review.
* **Distillation Contrastive Decoding (DCD)** — ([2402.14874](https://arxiv.org/abs/2402.14874)). Removes the amateur entirely via dropout/quantization-simulated contrast. Different orthogonal axis (avoid weak forward, not reuse it).
* **LoRA-Drop** — ([2601.02569](https://arxiv.org/abs/2601.02569)). **The closest mechanical analog.** Reuses *previous-token hidden state* in a fixed subset of layers, applies a low-rank correction, periodically (every k+1 tokens) executes the full forward. This is structurally the same pattern we are proposing for SCMoE.

LoRA-Drop's reported numbers are *the most empirically informative anchor we have* for the proposed Arc design, and §5 below grounds the projected curve in them.

---

## 3. MoE routing stability across consecutive tokens (key empirical evidence)

This is the strongest *indirect* evidence for amortization, because if the weak (rank-k) expert set is itself stable across positions, the weak logits are stable, and the correction is stable.

**From Mixtral paper Table 5** ([2401.04088](https://arxiv.org/abs/2401.04088)), proportion of *consecutive* tokens routed to the **same first-choice expert** (selected layers):

| Domain        | Layer 0 | Layer 15 | Layer 31 |
| ------------- | ------: | -------: | -------: |
| ArXiv         |   14.0% |    27.9% |    22.7% |
| DM Math       |   14.1% |    28.4% |    19.7% |
| GitHub        |   14.9% |    28.1% |    19.7% |
| Gutenberg     |   13.9% |    26.1% |    26.3% |
| Wikipedia     |   14.4% |    23.6% |    25.3% |

Random baseline: 12.5% (first-choice), ≈46% (first-or-second). **First-or-second choice repetition** sits at ~62–67% in mid-layers. So *one of the same two experts* fires again on **~2/3 of token transitions**.

In Mixtral, the weak path's top-1 expert *changes* on roughly 73–86% of token transitions. Naive "freeze the weak logits, reuse them" is unjustified — the underlying activation differs most of the time.

The right framing, though, is not "do the same experts fire?" but "**does the correction direction (z_strong − z_weak) stay similar in vocab space?**" — softer claim. Two pieces of indirect evidence:

1. **Adjacent-token hidden-state cosine similarity runs 0.6–0.85 in LLMs** (diffusion-LLM KV-drift literature [2510.14973](https://arxiv.org/pdf/2510.14973), EntropyCache [2603.18489](https://arxiv.org/pdf/2603.18489)). If hidden states are this similar, vocabulary projections — and *differences* between two projections — are similar too.
2. **DoLa overhead is 1.01–1.08×** ([2309.03883](https://arxiv.org/abs/2309.03883)): the contrastive layer choice is recomputed per token but is *similar* most steps. The contrast structure itself drifts slowly.

DeepSeek V3/V4 routing is intentionally more specialized than Mixtral (256 routed + 1 shared, top-8 in V3, top-6 in V4-Flash, sequence-wise balance loss; [2412.19437](https://arxiv.org/abs/2412.19437)). Higher specialization with shared-expert factoring suggests **more stickiness for the shared component** but **less for routed picks** — net effect on amortization unknown, requires measurement.

---

## 4. Analogies that *do* work — projecting a credible quality curve

LoRA-Drop is the cleanest published analog. It defines "temporal window" k = consecutive tokens generated before a full refresh.

LoRA-Drop reported, at fixed sparsity ρ=0.5 ([2601.02569v1](https://arxiv.org/html/2601.02569v1)):

| Refresh k | Speedup   | GSM8K Δ (LLaMA2-7B) | HumanEval Δ |
| --------: | --------- | ------------------: | ----------: |
|         1 | ~1.35–1.45× |              − ~0.1 pp |     −0.3 pp |
|         2 | ~1.45–1.55× |              − ~0.2 pp |     −0.4 pp |
|         3 | ~1.68–1.73× |              − ~0.2 pp |     −0.3 pp |
|         5 | ~1.75–1.85× |       notable drop |  notable drop |

The "safe zone" is **k ≤ 3 at ρ ≤ 0.5**, retaining within 0.5 pp of baseline accuracy. Above k=3, drift becomes measurable on math/code.

DeepCache ([2312.00858](https://arxiv.org/abs/2312.00858)) is the diffusion-model analog of exactly this pattern: cache U-Net high-level features, reuse for k consecutive timesteps, refresh at "key" timesteps. Achieves 2.3× speedup for SD-1.5 at −0.05 CLIP, 4.1× for LDM-4-G at −0.22 FID. Stable Diffusion is a *much harder* stability test than autoregressive LLM decoding because the diffusion trajectory compounds error. If DeepCache works there, fixed-stride amortization in LLM decoding has structural support.

**Translating to SCMoE**, the projection (extrapolated — not measured on SCMoE itself):

| Refresh N | Avg overhead vs greedy | Quality retention vs full-SCMoE (projected) |   Confidence |
| --------: | ---------------------: | ------------------------------------------: | -----------: |
|         1 (always) | 1.30× | 100% (5.15 pp GSM8K, 7.92 pp HumanEval) | Measured |
|         2 |  1.15× |  ~90–95% (≈4.6 pp / ≈7.1 pp) | Medium |
|         3 |  1.10× |  ~80–88% (≈4.2 pp / ≈6.5 pp) | Medium-low |
|         4 | 1.075× |  ~70–82% (≈3.6 pp / ≈5.8 pp) | Low |
|         8 | 1.038× |  ~45–60% (≈2.3 pp / ≈4.0 pp) | Very low; do not trust |
|        16 | 1.019× |  ~20–35% (≈1.0 pp / ≈1.6 pp) | Speculative |

Confidence levels are honest. Anything past N=4 should be treated as untested.

---

## 5. KV-cache analogy (limits)

KV cache reuse works because *past* K/V is mathematically unchanged by future tokens — there is **no approximation**. SCMoE amortization is approximation: we cache something that *would* differ if recomputed, and accept the error. The right analogy is therefore **not** KV cache but **MagicDec/TriForce-style draft caches** (in the existing Arc reference set under `/Users/jish/Documents/GitHub/arc/research/15_speculative_extensions/`) where staleness has bounded effect.

Building a "correction cache" alongside KV is feasible: store the last refreshed `(z_strong − z_weak)` vector (vocab-sized, ~129k floats for V4) and apply it as a fixed bias for the next N steps. Memory cost is trivial. Compute cost on reuse steps is one extra vocab-sized add (~free).

---

## 6. Interpolation strategies (analysis)

Four options, in increasing complexity:

1. **Hold-last** (cheapest): `z_t = z_top-k(t) + β · δ_last`. Zero extra compute, zero memory above 1 vocab vector. *Baseline for amortization.*
2. **Linear interpolation** between two stamped corrections at t₀ and t₀+N: `δ_t = δ_{t₀} + (t − t₀)/N · (δ_{t₀+N} − δ_{t₀})`. Requires lookahead, breaks causal streaming. **Not viable for online decoding** unless paired with speculative decoding.
3. **Exponential decay**: `δ_t = γ^{t−t₀} · δ_{t₀}` with γ ∈ (0.9, 0.99). Cheap, smoothly weakens correction over the reuse window. Plausible — DeepCache uses similar fading on features.
4. **Learned interpolation**: tiny MLP on `(h_t, h_{t₀}, δ_{t₀})` → `δ_t`. Requires offline training and adds dependency on training data. Best quality, worst engineering footprint.

**Recommendation:** Start with hold-last (option 1); if quality unacceptable, add exponential decay (option 3) — γ tuned offline on GSM8K. Skip linear interpolation entirely (breaks streaming) and skip learned (engineering cost outweighs Arc's near-term return).

---

## 7. Token-type-adaptive ("gated") refresh — the *better* idea

Fixed-N amortization wastes refreshes on prose and starves them on reasoning. The SCMoE paper itself identifies the asymmetry: **stopword/function-word tokens show low routing KLD; reasoning tokens (math/code) show high KLD**. So the marginal value of a fresh correction is itself token-dependent.

Two cheap online signals to gate refresh:

* **Top-k logit entropy** of the current step's strong forward. Low entropy ⇒ predictable token, reuse stale correction. High entropy ⇒ refresh. This is the same signal AdaDec ([2506.08980](https://arxiv.org/pdf/2506.08980)) and EntropyCache ([2603.18489](https://arxiv.org/pdf/2603.18489)) use for *their* gating decisions, both reporting it as a robust staleness proxy.
* **Router-bias delta**: |gate_logits_t − gate_logits_{t−1}|. Cheap to compute; spikes correlate with semantic shifts (`(\n`, sentence boundaries, equation entry).

ActLCD already validates the principle: a binary gate over contrast vs no-contrast adds only 3–5% latency over DoLa while *improving* factuality over always-on, because skipping low-value contrast removes a noise source. We expect the same here.

**Composition with the parallel "gated SCMoE" research thread:** the two ideas multiply. Gated SCMoE decides *whether* to do contrast at all; amortized SCMoE decides *whether to recompute* the correction when contrast is enabled. Combined: average overhead becomes
`p_contrast · ( (1/N) · 1.30× + (1 − 1/N) · 1.0× ) + (1 − p_contrast) · 1.0×`.
At p_contrast = 0.5, N = 4: average overhead = 0.5 · 1.075 + 0.5 · 1.0 = **1.038×**. The two ideas compound rather than collide.

---

## 8. Failure modes (honest)

1. **Long-context drift.** Every analog above (LoRA-Drop, DeepCache, AB-Cache) flags drift accumulation past their safe-zone refresh interval. For SCMoE this manifests as the correction "anchoring" the model to stale evidence as context evolves. *Hard mitigation:* force refresh on chat-template/turn boundaries and every K context tokens regardless of stride.
2. **Multi-turn dialogue.** Context switch on a new turn invalidates any cached correction. Trivial mitigation: clear correction cache when the chat template emits `<|user|>` / equivalent.
3. **Generation diversity collapse.** Repeated reuse of the same correction biases toward the same direction across multiple decisions — like fixed-temperature sampling but worse. This is the most plausible reason the projected quality curve in §4 droops fast past N=4. Hard to predict without measurement.
4. **Math/code regressions.** Mixtral Table 5 shows DM Math and GitHub have the *highest* mid-layer routing stickiness (28.4% / 28.1%) but the *lowest* layer-31 stickiness (19.7% / 19.7%). Stickiness is non-monotone across depth; this suggests math/code corrections drift faster at output-adjacent layers and amortization will hurt these benchmarks more than prose. Expect HumanEval to degrade faster than StrategyQA.
5. **DeepSeek V4 specifically untested.** Mixtral has 8 experts top-2. V4 has 256 routed top-6 with a shared expert. The "rank-k as weak" definition is itself ambiguous at large N. **Compounding two untested generalizations (SCMoE on V4 *and* amortization) is a high-variance bet.** Arc should land vanilla SCMoE first and measure correction stability *before* shipping amortization.

---

## 9. Proposed refresh schedule

Recommended Arc design, ordered from least to most aggressive:

1. **Baseline:** vanilla SCMoE, N=1, 1.30× overhead. Land first, measure quality.
2. **Conservative amortization:** fixed N=2 with hold-last interpolation. Expected avg overhead 1.15×, retain ~92% of lift. Low engineering risk.
3. **Standard amortization:** entropy-gated refresh with min stride 2 and max stride 4. Use top-3 logit entropy threshold τ tuned on a 200-sample GSM8K dev set. Expected avg overhead ~1.08–1.10× at ~85% lift retention. **Best engineering/quality ratio.**
4. **Aggressive:** combine §7 (gated SCMoE) and entropy-gated amortization. Projected 1.04–1.06× at ~75–80% lift retention. Ship only after #3 is verified on V4.

Schedule choice should be re-derived per model — Arc cannot port Mixtral-derived thresholds to V4 without measurement.

---

## 10. The measurement gap

The single most important thing this survey concludes is: **no public source measures cross-token stability of the SCMoE correction.** The SCMoE paper does not. No follow-up paper does. Mixtral's Table 5 measures *expert* stability, not *correction* stability — these are different quantities (expert switching can produce similar logit corrections if the routing collapses to similar subspaces).

The offline experiment Arc needs, before committing to a fixed N:

1. Generate 1000 tokens of GSM8K-style reasoning with vanilla SCMoE on V4.
2. Log `δ_t = z_top-6(t) − z_rank-k(t)` for every position.
3. Compute cosine similarity matrix `S[i,j] = cos(δ_i, δ_j)` for offsets j−i ∈ {1, 2, 4, 8, 16}.
4. The *integral* `mean_t cos(δ_t, δ_{t+N})` is the direct quality predictor.

If `cos(δ_t, δ_{t+2}) > 0.95`, N=2 is safe. If `cos(δ_t, δ_{t+4}) > 0.90`, N=4 is safe. **Below 0.85 average for N=4, fixed amortization fails and we must move to entropy-gated.** This experiment is ~1 GPU-hour and is the prerequisite to shipping anything past N=2.

---

## 11. Final verdict

* **N=2 fixed amortization**: **viable**. Drops overhead to ~1.15×, projected quality loss <10% of SCMoE lift. Ship after offline cosine measurement confirms `cos(δ_t, δ_{t+1}) > 0.9` average.
* **N=4 fixed**: **marginal**. Only viable if (a) the cosine measurement above holds at offset 4, or (b) it is *gated* rather than fixed-stride.
* **Entropy-gated adaptive refresh (N∈[2, 4])**: **the recommended target**. It compounds with the parallel gated-SCMoE thread, has a published analog in ActLCD with similar overhead (3–5%), and is robust to V4's untested routing behavior because it self-regulates on the model's own uncertainty signal.
* **N≥8 fixed**: **unworkable until proven otherwise**. The MoE routing data does not support reusing a single correction across 8+ positions in reasoning-heavy generation.

Net: the headline claim "1.075× at N=4" *might* hold on prose, *probably* won't hold on GSM8K/HumanEval at fixed stride, and *should* hold under an entropy gate. Arc should land vanilla SCMoE → instrument the cosine measurement → ship the entropy-gated variant. Skip the fixed-N=4 step.

---

## Citations

* SCMoE: Shi et al., *Unchosen Experts Can Contribute Too: Unleashing MoE Models' Power by Self-Contrast*, NeurIPS 2024 — [arXiv:2405.14507](https://arxiv.org/abs/2405.14507) | [HTML](https://arxiv.org/html/2405.14507)
* Mixtral routing analysis (Table 5): Jiang et al., *Mixtral of Experts* — [arXiv:2401.04088](https://arxiv.org/abs/2401.04088) | [ar5iv HTML](https://ar5iv.labs.arxiv.org/html/2401.04088)
* DeepSeek-V3 technical report — [arXiv:2412.19437](https://arxiv.org/abs/2412.19437)
* Speculative Contrastive Decoding: Yuhui Li et al., ACL 2024 — [arXiv:2311.08981](https://arxiv.org/abs/2311.08981)
* Active Layer-Contrastive Decoding (ActLCD) — [arXiv:2505.23657](https://arxiv.org/pdf/2505.23657) | [review](https://www.themoonlight.io/en/review/active-layer-contrastive-decoding-reduces-hallucination-in-large-language-model-generation)
* DoLa (layer contrastive) — [arXiv:2309.03883](https://arxiv.org/abs/2309.03883)
* Contrastive Decoding (original): O'Brien & Lewis, *Contrastive Decoding Improves Reasoning* — [arXiv:2309.09117](https://arxiv.org/abs/2309.09117)
* Distillation Contrastive Decoding (DCD) — [arXiv:2402.14874](https://arxiv.org/abs/2402.14874)
* LoRA-Drop (temporal LoRA decoding, *the* mechanical analog) — [arXiv:2601.02569](https://arxiv.org/abs/2601.02569) | [HTML](https://arxiv.org/html/2601.02569v1)
* DeepCache (diffusion-model feature reuse) — [arXiv:2312.00858](https://arxiv.org/abs/2312.00858)
* AB-Cache (training-free diffusion acceleration) — [arXiv:2504.10540](https://arxiv.org/pdf/2504.10540)
* EntropyCache (token-entropy gated cache for diffusion LLMs) — [arXiv:2603.18489](https://arxiv.org/pdf/2603.18489)
* Diffusion LLM KV-drift (hidden state cosine similarity 0.6–0.85) — [arXiv:2510.14973](https://arxiv.org/pdf/2510.14973)
* AdaDec (entropy-triggered re-evaluation) — [arXiv:2506.08980](https://arxiv.org/pdf/2506.08980)
* Routing in Sparsely-Gated LMs responds to context — [arXiv:2409.14107](https://arxiv.org/pdf/2409.14107)
* A Closer Look into MoE in LLMs (router behavior, KLD layer norm) — [arXiv:2406.18219](https://arxiv.org/abs/2406.18219)
* MoE-SpeQ (speculative decoding × MoE) — [arXiv:2511.14102](https://arxiv.org/pdf/2511.14102)
* Stable Entropy Hypothesis (decoding stability) — [arXiv:2302.06784](https://arxiv.org/pdf/2302.06784)
* Existing Arc research note on SCMoE prioritization: `/Users/jish/Documents/GitHub/arc/research/quality_up_inference.md`
