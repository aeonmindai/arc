# Distilled SCMoE Corrector Head — Research Survey

**Mission.** Investigate whether a small trained head can approximate the SCMoE contrastive output delta from strong-pass features alone, eliminating the second forward pass. Target: full quality lift at <5% overhead.

## TL;DR Verdict

**No prior work directly distills SCMoE-style MoE contrastive decoding into a single-pass head.** It would be novel. However, four neighbouring lines — Cross-Model Control, CASAL, Jakiro, and the EAGLE/Medusa drafter family — show that a small (~50–500M) head trained on a frozen base can produce useful logit-space deltas. Quality fidelity in those cases ranges ~50–91%, with <5% overhead.

The dominant risk: SCMoE's lift is **routing-specific** — the delta contrasts top-K vs rank-K *experts*. The "amateur" signal is not generic miscalibration but the model's own counterfactual MoE behaviour, which is information not present in the strong-pass features alone. This is the central unresolved question.

This is a **research bet** with reasonable adjacent precedent. A 1–10M-param linear/MLP head will almost certainly fail; a 50–200M-param transformer-block head conditioned on multi-layer features *and router probabilities* has a plausible path to ~50–75% of SCMoE's lift.

---

## 1. The SCMoE Signal — What Is the Head Predicting?

SCMoE (Shi et al., NeurIPS 2024, [arXiv:2405.14507](https://arxiv.org/abs/2405.14507)) computes:

```
z_sc(x_t) = (1 + β) · z_strong(x_t) − β · z_weak(x_t)
```

where `z_strong` uses top-2 routing and `z_weak` uses rank-k (k≈3–5). Vocabulary is masked to `V_valid = { i : z_strong(i) ≥ log(α) + max_j z_strong(j) }` with α=0.1, β∈[0.1,0.9]. Reported: +5.15 GSM8K, +3.46 StrategyQA, +7.92 HumanEval on Mixtral 8x7B; 1.30× latency.

The **corrector target** is the delta `Δ(x_t) = z_sc − z_strong = β·(z_strong − z_weak)`. This depends on (a) strong-pass logits, (b) what the weak/rank-k experts *would have* produced, and (c) the mask. (a) and (c) are recoverable from the strong pass; (b) is the hard part — counterfactual MoE behaviour.

Two framings:
- **Calibration framing:** weak experts produce a generic "noisier" distribution biased toward fluent-but-wrong tokens; subtracting is a learnable de-bias.
- **Counterfactual framing:** the weak pass routes to genuinely different experts holding information the strong experts don't; contrast exposes specific disagreement. The head needs router state to reproduce it.

The SCMoE paper doesn't disambiguate. That rank-k (not random) experts work best is moderate evidence for the counterfactual framing.

## 2. Prior Work — Direct Search Results

### Has anyone distilled contrastive decoding into a single forward pass?

**No, not cleanly.** Two papers come closest:

**Distillation Contrastive Decoding (DCD)** — Phan et al., [arXiv:2402.14874](https://arxiv.org/abs/2402.14874), [code](https://github.com/pphuc25/distil-cd). Eliminates the *separate* amateur but still requires two passes through the same model (one normal, one with dropout/quantization). **No learned head.** Llama2-7B/GSM8K: +2.96 vs greedy, +1.89 vs CD. FLOPs unchanged (~2× greedy). DCD demonstrates the amateur signal can come from same-model perturbation — mildly encouraging.

**DoLa** — Chuang et al., [arXiv:2309.03883](https://arxiv.org/abs/2309.03883). Replaces amateur with early-exit projection: ~3.6% memory and 1.01–1.08× latency on LLaMA-7B. +12–17 pp on TruthfulQA but hallucination-focused, not reasoning-focused. Notably, Anil et al. observed CD with a 7B amateur can **hurt** large-model math accuracy — a real warning that miscalibrated contrast injects noise. Not a learned head, just intermediate-layer reuse.

**Verdict:** literal search for "distilled SCMoE" or "amortized contrastive decoding for MoE" returns nothing. The space is open.

### Adjacent — learned logit-delta heads on frozen LLMs

**Cross-Model Control (CMC)** — Wu et al., [arXiv:2410.17599](https://arxiv.org/abs/2410.17599). Most directly relevant. A **TinyLlama-110M** delta model trained alongside a frozen template LLM; at inference, `z_final = LogSoftmax(z_base) + z_delta`. Delta runs *in parallel*. CMC-Llama2-70B AlpacaEval: 49.81% (vs 11.55% vanilla; LoRA upper bound 68–79%). **~50–70% of fine-tuning lift retained** at <1/640 parameter overhead. Training: 52K GPT4-Alpaca examples, 4 epochs. Strongest evidence that a tiny learned delta head can substitute for an expensive inference operation.

**CASAL** — [arXiv:2510.02324](https://arxiv.org/abs/2510.02324). Bakes contrastive activation steering into model weights by training one transformer-layer submodule. Hallucination reduction 30–40%, **30× compute-efficient and 20× data-efficient vs LoRA-SFT/DPO**. Critically: "first steering-based training method shown to be effective for both dense and MoE models." Proves the amortization principle works for contrastive steering and MoE bases.

**Proxy-Tuning** — Liu et al., [arXiv:2401.08565](https://arxiv.org/abs/2401.08565). Two separate passes (small expert + anti-expert) produce a logit delta added to a frozen large base. Closes **88–91% of the fine-tuning gap**. Three forward passes — not single-pass — but logit-delta substitution can retain ~90% of full fine-tuning.

**ThinkLogit** — [arXiv:2507.12759](https://arxiv.org/abs/2507.12759). A 1.5B guider produces logit shifts to elicit long-CoT in a frozen 32B base. +26% relative pass@1 on math. Sequential, not single-pass.

### Adjacent — drafter heads (most architecturally relevant)

**EAGLE-3** — Li et al., [arXiv:2503.01840](https://arxiv.org/abs/2503.01840). Single Transformer decoder layer + projections, **2–5% of target params**. Inputs: fused low/mid/high target features (3k→k dims). Training: ShareGPT 68K + UltraChat 464K, ~2 weeks on 16× A100s for 70B targets (**~5,400 GPU-hours**). Acceptance rate nearly flat across positions — head captures enough strong-base info to predict the *correct base distribution*. Imitating-base is easier than predicting-counterfactual.

**Medusa** — Cai et al., [arXiv:2401.10774](https://arxiv.org/abs/2401.10774). Multiple lightweight heads (typically <1% of target), 5 heads recommended, base frozen during training. Predicts future tokens; not contrastive.

**Jakiro** — Huang et al., [arXiv:2502.06282](https://arxiv.org/abs/2502.06282). **Most thematically aligned.** MoE-decoupled multi-head with contrastive operation between the two highest-scoring expert heads. Drafter 0.35–1.87B params. Training: 68K ShareGPT, 2–3 days on 4× A100s for 70B. "Contrastive operations between activated expert heads, introducing almost no additional latency." Not distilling SCMoE, but existence proof that contrast-aware heads over expert features work at near-zero overhead.

## 3. Proposed Corrector Architecture

Given the evidence, the design space converges. Below is the most defensible proposal, sized for DeepSeek V4 (256 experts, ~7168 hidden, 60+ layers, vocab ~129K).

### Inputs (ranked by importance)

1. **Strong-pass final hidden state** `h_L ∈ R^7168`. Free.
2. **Router probabilities per layer** `π_l ∈ R^256` over ~10 chosen layers = 2560 dims. The signal SCMoE relies on — without this, the head is calibration-only.
3. **Top-K/rank-K expert-id encodings** per layer — explicit feature for the contrast. Learned 64-dim embeddings × top-12 × L layers.
4. **Strong-pass logits themselves** — the SCMoE mask depends on them.
5. **Mid-layer hidden states** (à la EAGLE-3) — lets the head see features the final state discarded.

### Architecture

**Recommend: single transformer decoder block (~150–300M params) + delta projection.**

- Input projection: concat features → 4096-dim residual.
- 1 transformer decoder layer over recent context (window 8–16 tokens) — EAGLE-3-style "training-time test."
- Delta head: `R^4096 → R^vocab` initialized to zero. Output `Δ ∈ R^vocab`.
- Final logits: `z_strong + Δ`, then SCMoE adaptive plausibility mask from `z_strong`.

A 1–10M-param linear probe is almost certainly inadequate: Next-Token Perception Score work ([arXiv:2505.17169](https://arxiv.org/abs/2505.17169)) shows linear probes on frozen LLMs hit ceilings well below full-fidelity prediction. The lift comes specifically from suppressing fluent-but-wrong tokens — exactly the hard pattern for a low-capacity probe.

### Training Target

**Direct delta regression** is right given CMC's success:
```
L = KL(softmax(z_strong + Δ_pred) || softmax(z_SCMoE))   # forward KL
  + λ · MSE(Δ_pred, Δ_teacher)                            # auxiliary
```

Forward KL keeps the post-correction distribution faithful; MSE stabilises per-token. Mask outside the SCMoE α-set (target Δ=0 for masked tokens — the head shouldn't predict noise).

### Training Recipe

- **Data:** ~500M tokens of completions from base running SCMoE, diverse mix (30% math/code, rest SlimPajama/UltraChat-style). EAGLE-3 used <1B; CMC used 52K examples. Start at 100M, scale if quality saturates.
- **Loss:** β as a head input, not fixed — head learns conditional behaviour.
- **Optimizer:** AdamW, LR 1e-4, cosine, ~2 epochs.
- **GPU-hours:** ~200–500 H100-hours for 200M-param head on 500M tokens.
- **Critical:** generate teacher with the **same QTIP 2-bit + TurboQuant 3.5-bit** stack Arc ships, so the head learns the deployed noise floor.

## 4. Quality Fidelity — Realistic Expectations

No published number exists for "distilled SCMoE % lift retained." Triangulating:

- CMC retains ~50–70% of fine-tuning lift at 110M params.
- EAGLE-3 recovers >95% of base behaviour (acceptance) with 2–5% extra params — but imitating the base is easier than predicting a counterfactual contrast.
- Proxy-Tuning closes 88–91% of the FT gap, but uses actual contrasting model logits at inference — the very pass we eliminate.
- CASAL retains 30–40% hallucination reduction when amortized, with no clear steering-baseline comparison.

**Realistic targets** for a 150–300M head on 500M tokens:
- **Optimistic (60–80%):** if the calibration framing is mostly right.
- **Realistic (30–50%):** if some lift requires actually running rank-K experts.
- **Failure (<10%, possibly negative):** if hard-reasoning lift (the +7.92 HumanEval) comes from low-frequency decisions where rank-K experts inject info absent from the strong pass — a noisy Δ then *degrades* baseline. Anil et al.'s observation that 7B-amateur CD can harm large-model reasoning is the warning here.

## 5. Inference Cost

For a 200M head over 284B/13B-active MoE on H100:
- Strong pass: ~25–40 ms/tok with Arc's planned graph/fusion stack.
- Head: 200M params at ~700 GB/s effective DRAM → memory-bound ~0.3 ms; compute-bound <1 ms.
- **Overhead: 1–3% of total step latency**, well inside the <5% target.

Sharing the strong-pass CUDA graph and reading features from registers/SMEM drops it under 1 ms.

## 6. Risks & Failure Modes

| Risk | Severity | Mitigation |
|---|---|---|
| Head cannot recover counterfactual MoE behaviour from strong features | **High** — kills the project | Include explicit router probabilities and expert-id embeddings as input. Add a router-disagreement scalar as a side feature. Train on enough data (>500M tokens) to surface rare patterns. |
| Δ becomes a learned noise injector, degrades baseline | High | Initialize the delta projection to zero. Train with a residual gate `α(h)·Δ` so the head can choose to do nothing. Validate that disabling Δ never hurts. |
| Compression noise from QTIP 2-bit corrupts the training signal | Moderate | Generate teacher with the same quantized base. Confirm SCMoE itself still lifts under quantization before training (this should be measured first — if SCMoE on quantized V4 gives <2 pp GSM8K, distillation is moot). |
| Adversarial overfitting — head over-corrects on training distribution and underperforms on OOD prompts | Moderate | Use a heavy, diverse training mix. Hold out reasoning benchmarks during training. Monitor a robust eval set. |
| The lift turns out to be small after compression anyway | Real, but orthogonal | First experiment must be: does SCMoE on Arc's quantized V4 reproduce a meaningful lift? If yes, distillation is worth the cost. If no, abandon. |
| Distribution drift across α/β contrast settings | Low–moderate | Condition the head on (α, β) so a single trained head covers the inference-time hyperparameter sweep. |

## 7. Composition with Arc's Quantized Base

Two concerns:
1. **Teacher must use the production quantization stack.** Training against FP16 SCMoE outputs and serving a 2-bit base means the head corrects a base it never sees. One-line training fix; biggest source of silent fidelity loss if missed.
2. **Rank-K behaviour under 2-bit is unmeasured.** SCMoE paper doesn't study quantized bases. Plausible that 2-bit compresses the top-K/rank-K gap (both noisier), weakening SCMoE *before* distillation. **Pre-experiment:** run SCMoE on quantized V4, measure GSM8K. If lift drops from +5.15 to <+1.5, nothing to distill — pivot.

## 8. Honest Verdict

**Viable research direction, no direct precedent.** Supporting evidence — CMC (50–70% retained, 110M params), CASAL (30–40% retained, MoE-effective), Jakiro (contrastive heads, near-zero latency), EAGLE-3 (~95% of an easier target) — is collectively strong enough to expect *some* fraction of SCMoE's lift to be amortizable. Not strong enough to predict whether it'll be 20% or 80%.

Dominant uncertainty: the calibration-vs-counterfactual question. If SCMoE's lift is fundamentally routing-counterfactual, a head cannot fully replicate it from a single strong pass. Router state and expert-id features are the key architectural levers to push from calibration-only toward counterfactual-aware.

**Recommended sequencing:** (1) verify SCMoE lift on Arc's quantized V4 — one weekend, blocks everything; (2) ablation: perturb only router probabilities at the weak pass, measure surviving lift — tests calibration framing cheaply; (3) only then commit ~500 H100-hours to training the head.

If (1) and (2) survive, this is a publishable single-pass quality-lift technique for MoE inference. If not, the safe fallback is sharing more state in the second pass — keep strong-pass attention/KV and only re-route through different experts, cutting overhead from 1.30× toward ~1.10× with zero training risk.

---

## Key References

- Shi et al., "Unchosen Experts Can Contribute Too: Unleashing MoE Models' Power by Self-Contrast," NeurIPS 2024. [arXiv:2405.14507](https://arxiv.org/abs/2405.14507). [code](https://github.com/DavidFanzz/SCMoE).
- Phan et al., "Distillation Contrastive Decoding," 2024. [arXiv:2402.14874](https://arxiv.org/abs/2402.14874). [code](https://github.com/pphuc25/distil-cd).
- Chuang et al., "DoLa: Decoding by Contrasting Layers," ICLR 2024. [arXiv:2309.03883](https://arxiv.org/abs/2309.03883).
- Wu et al., "Cross-Model Control: Improving Multiple Large Language Models in One-time Training," 2024. [arXiv:2410.17599](https://arxiv.org/abs/2410.17599).
- "Hallucination Reduction with CASAL: Contrastive Activation Steering for Amortized Learning," 2025. [arXiv:2510.02324](https://arxiv.org/abs/2510.02324).
- Liu et al., "Tuning Language Models by Proxy," 2024. [arXiv:2401.08565](https://arxiv.org/abs/2401.08565).
- Li et al., "EAGLE-3: Scaling up Inference Acceleration via Training-Time Test," NeurIPS 2025. [arXiv:2503.01840](https://arxiv.org/abs/2503.01840). [code](https://github.com/SafeAILab/EAGLE).
- Cai et al., "Medusa: Simple LLM Inference Acceleration Framework," 2024. [arXiv:2401.10774](https://arxiv.org/abs/2401.10774).
- Huang et al., "Jakiro: Boosting Speculative Decoding with Decoupled Multi-Head via MoE," 2025. [arXiv:2502.06282](https://arxiv.org/abs/2502.06282).
- Yi et al., "Speculative Contrastive Decoding," ACL 2024. [arXiv:2311.08981](https://arxiv.org/abs/2311.08981).
- Li et al., "Contrastive Decoding: Open-ended Text Generation as Optimization," ACL 2023. [arXiv:2210.15097](https://arxiv.org/abs/2210.15097).
- Anil et al., "Contrastive Decoding Improves Reasoning in Large Language Models," 2023. [arXiv:2309.09117](https://arxiv.org/abs/2309.09117).
- "Logit Arithmetic Elicits Long Reasoning Capabilities Without Training" (ThinkLogit), 2025. [arXiv:2507.12759](https://arxiv.org/abs/2507.12759).
- "SLED: Self Logits Evolution Decoding," 2024. [arXiv:2411.02433](https://arxiv.org/abs/2411.02433). [code](https://github.com/JayZhang42/SLED).
- "Next Token Perception Score," 2025. [arXiv:2505.17169](https://arxiv.org/abs/2505.17169) — quantifies linear probe ceilings on frozen LLMs.
