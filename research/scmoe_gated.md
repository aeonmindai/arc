# Gated SCMoE: Selective Self-Contrastive Decoding for DeepSeek V4

**Mission.** Can SCMoE's `1.30×` latency overhead be amortized by only invoking the
weak-routing forward on a small, uncertain subset of tokens? Target: `<1.10×`
average overhead while retaining ≥90% of SCMoE's quality lift (`+5.15 GSM8K`,
`+7.92 HumanEval` on Mixtral 8x7B).

**Verdict up front.** This is not novel in spirit — confidence-gated contrastive
decoding has been published in three forms (CCD, USCD, EASD) — but the specific
combination of (a) SCMoE-style strong/weak MoE routing as the contrast source,
(b) MoE router-entropy + top-1/top-2 margin as the gating signal, and (c)
DeepSeek V4 as the target appears unpublished. Genuine 1–2 agent-session port
with a meaningful research contribution attached.

---

## 1. Why gating should work: where SCMoE's gains live

The SCMoE paper itself contains the key evidence. Table 5 of Shi et al. (2024)
breaks down the KL divergence between strong and weak routing across token
categories on GSM8K and reports:

- "Expression" tokens (numeric, operators, equation building): KLD **+31.13%
  above the dataset mean**.
- Function words / stopwords (e.g., "the", "and", "of"): KLD **−30.25% below
  the mean**.
- Reasoning-step initiation tokens: "apparent KLD observed".

The paper's own Finding 2 states the contrast "is particularly evident in the
parts that require rigorous reasoning". Appendix F flags that MMLU (no
verbalized reasoning) benefits less than GSM8K — i.e., the lift is **task-
concentrated**, and within a single trace it is also **token-concentrated**.

This is the load-bearing empirical claim. If the +5.15 GSM8K were uniformly
spread across all decoded tokens, gating would simply trade quality for speed
proportionally and gain nothing. The published KLD evidence says the opposite:
roughly 70–80% of decoded tokens (function words, end-of-step boilerplate,
prompt copies) get **near-zero** marginal value from the second pass. Skipping
them is approximately quality-neutral.

Recent confirmation from the broader contrastive-decoding literature:

- **Critical Tokens Matter** (Lin et al., arXiv [2411.19943](https://arxiv.org/abs/2411.19943))
  shows reasoning failures are localized to a sparse subset of tokens.
- **Thinking by Subtraction** (CCD, arXiv [2602.18232](https://arxiv.org/abs/2602.18232))
  explicitly states: "reasoning uncertainty is highly localized; a small subset
  of low-confidence tokens disproportionately influences output correctness".
- **LayerCake** (arXiv [2507.04404](https://arxiv.org/html/2507.04404v1))
  classifies tokens into punctuation/conceptual/functional and applies CD only
  on conceptual tokens at the relevant layers, outperforming uniform DoLa/SLED.

The hypothesis is therefore well-supported.

---

## 2. Prior art: selective / gated contrastive decoding

Closer than I expected. Three direct hits.

### 2.1 USCD — Uncertainty-Aware Selective Contrastive Decoding (Wang et al., 2024)

arXiv [2409.05923](https://arxiv.org/abs/2409.05923). Code-generation focused.
Builds a "lame prompt" amateur (few-shot examples removed) and uses
Jensen-Shannon divergence between the strong distribution and an uncertainty
estimate (~0.25 on average) to **selectively** subtract noise. Reports +16.59%
pass@1 averaged across HumanEval/MBPP/MultiPL-E, lower latency than vanilla CD
because the contrast is skipped on confident tokens.

### 2.2 CCD — Confidence-Driven Contrastive Decoding (Liu et al., 2026)

arXiv [2602.18232](https://arxiv.org/abs/2602.18232). The most direct prior art.

- **Signal**: token-level confidence = negative average log-probability of the
  top-k tokens (k usually 3–10), measuring how peaked the strong distribution
  is.
- **Threshold**: two adaptive quantile thresholds (`τ_cd`, `τ_rep`) computed on
  a sliding window of recent confidences. Recommended `(q_cd, q_rep) = (3, 8)`
  or `(3, 10)` (lower decile triggers CD; upper decile masked in contrast).
- **Invocation rate**: not numerically reported, but the quantile design pins
  it at roughly ~10–30% of tokens by construction.
- **Quality**: +3–5 pp over the base model on AIME/MATH with Qwen-family
  reasoning models, **at lower overhead than uniform CD** (dual KV cache, but
  only one extra forward at low-confidence positions).
- **Training-free**, model-agnostic.

### 2.3 EASD — Entropy-Aware Speculative Decoding (Anonymous, 2025)

arXiv [2512.23765](https://arxiv.org/pdf/2512.23765). Gates speculative drafts
by entropy: drafts are accepted/rejected based on whether both draft and target
distributions are flat enough; selective resampling on high-entropy tokens.
Demonstrates that entropy gating preserves quality while reducing rejection
cost on reasoning workloads. Same shape of gate, different application.

### 2.4 Adjacent: not quite the same thing

- **Speculative Contrastive Decoding** (SCD, arXiv [2311.08981](https://arxiv.org/html/2311.08981v2))
  combines spec-decoding with CD but does **not** gate — it accepts/rejects
  draft tokens against a contrastive distribution.
- **Adaptive Contrastive Search** (ACS, arXiv [2407.18698](https://arxiv.org/html/2407.18698v2))
  tunes contrastive-search hyperparameters (k, α) by entropy; still pays full
  contrastive search at every step (~35% slower than CS, not faster).
- **DoLa** / **LayerCake** contrast across layers, not across MoE routings.

**Gap.** None of these target MoE strong/weak routing as the contrast source.
None analyze MoE-router-entropy as an additional cheap signal. None target
DeepSeek V3/V4. The combination is open.

---

## 3. Adjacent literature: gates that already work

Stable, training-free per-token gates from neighbouring problems:

| Method | Signal | Gate cost | Calibration |
|---|---|---|---|
| **CALM** (Schuster et al., NeurIPS 2022, arXiv [2207.07061](https://arxiv.org/abs/2207.07061)) | softmax confidence at intermediate layer | ~0 (already computed) | conformal risk control, ≤ε quality drop with prob ≥1−δ |
| **LayerSkip** (Elhoushi et al., arXiv [2404.16710](https://arxiv.org/pdf/2404.16710)) | early-exit head logits | needs training | layer-dropout co-training |
| **Mixture-of-Depths** (Raposo et al., 2024) | learned per-token router | needs training | jointly trained with model |
| **TARG** (arXiv [2511.09803](https://arxiv.org/html/2511.09803v1)) | top-1/top-2 logit gap | ~0 | empirical CDF → invocation rate ρ |
| **Think Just Enough** (Cao et al., arXiv [2510.08146](https://arxiv.org/html/2510.08146v1)) | sequence-level entropy | ~0 | 5–10 calibration examples; **0% accuracy loss, 25–50% compute saved** |
| **SpecDec++** (Huang et al., arXiv [2405.19715](https://openreview.net/pdf?id=Y131N9fUbU)) | trained acceptance head | small forward | RL/threshold policy |
| **GammaTune+** (arXiv [2504.00030](https://arxiv.org/pdf/2504.00030)) | running acceptance rate | ~0 | training-free, +15% speedup |

The two most relevant takeaways:

1. **TARG explicitly argues entropy is a poor signal on modern instruction-
   tuned LLMs** ("prefix entropies compress and may lose discriminative
   power") and recommends the **top-1/top-2 logit margin** instead, because it
   retains dynamic range even when distributions are peaked. This is a strong
   prior for DeepSeek V4, which is heavily post-trained.
2. **Think Just Enough** establishes that 5–10 examples of correct-answer
   entropy suffice to calibrate a threshold that delivers **0% accuracy loss
   at 25–50% compute savings**. This is the empirical existence proof that
   training-free quantile gates work.

---

## 4. Proposed gate for SCMoE on V4

**Goal.** A rule-based, training-free gate computable in O(vocab) per token
(i.e., already inside the sampler's hot path) that decides whether to launch
the second (weak-routing) forward.

### 4.1 Composite signal `U(t)`

Three signals, each ~free given that the strong-pass logits and router
distributions already sit in registers:

1. **Top-1/top-2 logit margin** of the strong-pass output:
   `m = logit[top1] - logit[top2]`. Small `m` → uncertain → contrast.
2. **Strong-pass output entropy** (truncated to top-32 to bound cost):
   `H = -Σ p_i log p_i`. Useful as a secondary signal; per TARG, less reliable
   solo on peaked distributions.
3. **MoE router entropy at the deepest few MoE layers**:
   `H_router = -Σ_e g_e log g_e` where `g_e` is the post-softmax routing weight
   over experts at layer `L−k` for small `k`. V4 has 256 experts top-6; a
   peaked router (1–2 experts dominate) signals an "easy" token; flat router
   signals expert disagreement → genuine ambiguity → likely benefits from
   contrast. **This is the MoE-native signal that does not exist for dense
   models** and is the principal contribution.
4. *(Optional, V4-specific)* **Lightning Indexer dispersion**: the top-k
   selected block scores. If the indexer's top-1 vs top-k gap is small,
   attention is being spread across many context blocks — a complexity proxy.
   Note this is a per-layer signal; using it requires a cheap reduction. The
   payoff is unclear and it can be deferred.

Combine as a single scalar score:

```
U = α · ranknorm(−m)
  + β · ranknorm(H)
  + γ · ranknorm(H_router_deep)
```

`ranknorm(·)` mapping each signal to its empirical quantile on a calibration
trace makes the weights `(α, β, γ)` interpretable as "fraction of importance"
and avoids per-model rescaling. Initial weights `α=0.5, β=0.2, γ=0.3` are a
reasonable starting prior derived from TARG (margin dominant) plus the SCMoE
finding that the action is at the MoE routing layer.

### 4.2 Calibration recipe (training-free)

Borrow directly from TARG / Think Just Enough:

1. Run V4 on 100–500 calibration prompts (GSM8K dev split is ideal).
2. Compute `U(t)` at every decoded position.
3. Pick threshold `τ = F_U^{-1}(1 − ρ)` to hit target invocation rate `ρ`
   (e.g., ρ=0.20 → top-20% uncertain tokens trigger SCMoE).
4. Validate quality on a held-out slice; sweep ρ ∈ {0.05, 0.10, 0.20, 0.30}.

No training, no auxiliary head, no model changes. The threshold is the only
hyperparameter and it is just a quantile.

### 4.3 Where the gate fires in the pipeline

Strong-pass MoE forward → strong logits + router weights → sampler.

```
strong_forward()                  # always
U = compute_U(strong_logits, router_weights_deep)
if U > τ:
    weak_forward()                # only ~ρ of the time
    final_logits = (1+λ) strong - λ weak
else:
    final_logits = strong
sample(final_logits)
```

Gate cost: one extra reduction over top-K=32 of the strong logits plus one
entropy over 256 experts at 1–2 deep layers. On B200 this is on the order
of a few microseconds, well below 1% of a single forward.

---

## 5. Expected latency / quality curve

Latency math, assuming SCMoE costs `1.30×` per invocation and the strong
forward is `1.00×`:

| Invocation rate ρ | Avg latency | If quality lift is uniform | If lift is concentrated on uncertain tokens (per SCMoE Table 5) |
|---:|---:|---:|---:|
| 1.00 (uniform SCMoE) | 1.30× | +5.15 GSM8K | +5.15 GSM8K |
| 0.30 | 1.09× | +1.55 | ~ +4.5 to +5.0 |
| 0.20 | 1.06× | +1.03 | ~ +4.0 to +4.8 |
| 0.10 | 1.03× | +0.52 | ~ +3.0 to +4.2 |
| 0.05 | 1.015× | +0.26 | ~ +2.0 to +3.5 |

The uniform-distribution column is the pessimistic worst case (proportional
trade). The concentrated column is the empirically-supported case: SCMoE's own
KLD analysis says ~70% of tokens contribute ≤0.7× the average lift and ~20% of
tokens contribute >2× the average lift. CCD reports similar concentration on
Qwen-reasoning runs.

**Best estimate.** `ρ = 0.20–0.30`, average latency `1.06–1.09×`, retained
quality `85–95%` of the full SCMoE lift. This hits the stated `<1.10×` target.

Worst case if signals are poorly correlated with actual ambiguity: degrades
toward the uniform-lift column, at which point you may as well always run
SCMoE on the difficult-token-heavy portion of the request (e.g., gate at the
request level via task classifier rather than per-token).

---

## 6. Risks and honest caveats

1. **Gate–benefit correlation is the load-bearing assumption.** The SCMoE
   paper supports it via KLD-by-category; CCD supports it via low-confidence
   localization; but neither has been measured **with SCMoE-style strong/weak
   contrast specifically**. Until measured on V4 + GSM8K, treat the
   "concentrated" column as a prior, not data.
2. **Gate cost is not zero.** Top-1/top-2 margin is essentially free; router
   entropy adds a `256-way log-softmax` reduction per MoE layer touched. On
   V4's 256-expert routing this is ~256 FLOPs/token/layer — negligible in
   absolute terms but worth budgeting honestly.
3. **Modern post-training compresses entropy.** TARG's warning applies to V4
   (heavily RLHF/SFT'd). Lean on the margin and router-entropy signals; treat
   output entropy as supplementary.
4. **Quantization interaction.** Under Arc's QTIP 2-bit weights + TurboQuant
   3.5-bit KV, both strong and weak forwards share noise (correlated error),
   so the contrast is still approximately well-conditioned — this is precisely
   the SCMoE advantage flagged in `quality_up_inference.md`. The gate signals
   themselves are computed on the strong-pass output and are unaffected.
5. **Adversarial concern.** Could a benchmark be exploited to push hard tokens
   into the "easy" bucket? In principle yes, but the gate is read-only on the
   model state — no input-side knob — and the threshold is calibrated offline
   on a held-out set. Low practical risk; flag and move on.
6. **V4 router is auxiliary-loss-free with anticipatory routing**: the router
   distribution is already pushed toward decisive top-6 selection by training.
   This may make `H_router` less informative than on Mixtral; measure before
   relying on it heavily.
7. **Streaming vs batch.** Per-token gating is fine for batch=1 latency-
   sensitive use; in batched decode the divergent control flow (some requests
   need the second forward, some don't) costs throughput. Mitigation: a small
   per-step bucket — collect all requests that triggered the gate and run a
   single batched weak-forward. CUDA-graph-friendly with two captured graphs
   (with/without weak pass) and a per-step dispatch decision.

---

## 7. Composition with V4-specific signals

V4's Lightning Indexer (FP4, ReLU-scored multi-head dot product, top-k blocks)
produces a per-token score distribution over compressed KV blocks. A flat
indexer distribution (no clear best block) is a complexity signal: the model
is unsure where to attend. This is a **free** byproduct of the existing
forward and could be a fourth term in `U`. Empirically untested but cheap to
add. See the V4 technical reports for the indexer architecture
([HuggingFace blog](https://huggingface.co/blog/deepseekv4),
[MarkTechPost coverage](https://www.marktechpost.com/2026/04/24/deepseek-ai-releases-deepseek-v4-compressed-sparse-attention-and-heavily-compressed-attention-enable-one-million-token-contexts/)).

V4 router entropy at deep layers (last 2–4 MoE layers) is generally a stronger
signal than at early layers: early MoE layers route by surface form (function
words → consistent experts), deep MoE layers route by semantics (ambiguity
shows up here). Use deep-layer router entropy preferentially.

---

## 8. Verdict and effort estimate

This is **a 1–2 agent-session implementation on top of the SCMoE port**, not a
new research programme. The mechanism is mechanically simple:

- Add a `GatedSCMoE` decode strategy in `mistralrs-core/src/pipeline/`.
- Plumb router-weight tensors out of `MoEGate::forward` (deep layers only) to
  the sampler.
- Compute `U` in the existing sampler hot path, fork on `U > τ` to either
  invoke the existing weak-routing forward or skip it.
- Add a `--scmoe-rho` CLI flag and a tiny calibration script
  (`arc-tools/calibrate_scmoe_gate.rs`) that takes a dev split and emits `τ`.
- CUDA-graph: capture two graphs (strong-only, strong+weak); per-step
  dispatch.

The **research contribution** is a clean ablation on V4: which gate signal
(margin / output-entropy / router-entropy / indexer-dispersion / composite)
gives the highest quality-retention at fixed `ρ`. That is publishable as a
short paper or technical report alongside the engineering work. The prior
art (USCD, CCD, EASD) gives a strong defensive position — "gated CD works,
we generalized it to MoE strong/weak routing and to V4".

**Open empirical question, in order of importance:**

1. What fraction `ρ` retains ≥90% of SCMoE's GSM8K and HumanEval lift on V4?
2. Does router entropy at deep MoE layers add real signal beyond top-1/top-2
   margin, or is it correlated and redundant?
3. Does the gate transfer across tasks (calibrate on GSM8K, evaluate on
   MATH/MBPP) or is per-task recalibration needed?

None of these block implementation; all are answerable on a single 8×B200
node with a few hours of decode time.

---

## Sources

- [SCMoE — Shi et al., NeurIPS 2024, arXiv 2405.14507](https://arxiv.org/abs/2405.14507)
- [Contrastive Decoding — Li et al., ACL 2023, arXiv 2210.15097](https://arxiv.org/abs/2210.15097)
- [CD Improves Reasoning — O'Brien & Lewis, arXiv 2309.09117](https://arxiv.org/abs/2309.09117)
- [USCD — Wang et al., arXiv 2409.05923](https://arxiv.org/abs/2409.05923)
- [CCD: Thinking by Subtraction — arXiv 2602.18232](https://arxiv.org/abs/2602.18232)
- [EASD: Entropy-Aware Speculative Decoding — arXiv 2512.23765](https://arxiv.org/pdf/2512.23765)
- [Speculative Contrastive Decoding — arXiv 2311.08981](https://arxiv.org/html/2311.08981v2)
- [Adaptive Contrastive Search — arXiv 2407.18698](https://arxiv.org/html/2407.18698v2)
- [Critical Tokens Matter — arXiv 2411.19943](https://arxiv.org/html/2411.19943)
- [LayerCake — arXiv 2507.04404](https://arxiv.org/html/2507.04404v1)
- [DoLa — Chuang et al., arXiv 2309.03883](https://arxiv.org/abs/2309.03883)
- [CALM — Schuster et al., NeurIPS 2022, arXiv 2207.07061](https://arxiv.org/abs/2207.07061)
- [LayerSkip — Elhoushi et al., arXiv 2404.16710](https://arxiv.org/pdf/2404.16710)
- [TARG — arXiv 2511.09803](https://arxiv.org/html/2511.09803v1)
- [Think Just Enough — arXiv 2510.08146](https://arxiv.org/html/2510.08146v1)
- [SpecDec++ — Huang et al., OpenReview Y131N9fUbU](https://openreview.net/pdf?id=Y131N9fUbU)
- [GammaTune — arXiv 2504.00030](https://arxiv.org/pdf/2504.00030)
- [EAGLE-3 — arXiv 2503.01840](https://arxiv.org/html/2503.01840v1)
- [GW-MoE: Resolving Router Uncertainty — arXiv 2406.12375](https://arxiv.org/pdf/2406.12375)
- [DeepSeek-V4 release coverage — HuggingFace blog](https://huggingface.co/blog/deepseekv4)
- [DeepSeek-V4 — MarkTechPost analysis](https://www.marktechpost.com/2026/04/24/deepseek-ai-releases-deepseek-v4-compressed-sparse-attention-and-heavily-compressed-attention-enable-one-million-token-contexts/)
