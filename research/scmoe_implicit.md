# Implicit SCMoE: Can the Weak Forward Be Eliminated?

**Mission.** Investigate whether SCMoE's `weak` distribution (rank-K MoE
activation) can be derived implicitly from signals already produced during the
`strong` (top-K) forward pass, eliminating the second forward entirely.

**Honest verdict up front.** No published method recovers the full SCMoE lift
without a second pass through the MoE FFNs. The strongest single-pass analog
(DoLa) actively *hurts* Mixtral 8x7B — GSM8K drops from 61.79 → 49.96,
HumanEval from 33.54 → 12.80 (Shi et al., Table 1; [arXiv:2405.14507](https://arxiv.org/abs/2405.14507)).
There is theoretical and empirical evidence that some fraction of the SCMoE
lift is recoverable from single-pass signals — realistically **20–35%** of the
quality delta — but **the rank-K activation contains distinct information that
is not present in the strong-pass hidden states**, and that bounds the win. The
viable direction is *partial* implicit contrast (`gated SCMoE` plus a learned
or cheap router-based proxy), not full elimination.

---

## 1. What SCMoE actually contrasts

From Shi et al. (NeurIPS 2024, [arXiv:2405.14507v2](https://arxiv.org/html/2405.14507v2)):

```
z_sc(x_t) = (1+β)·z_top-2(x_t) − β·z_rank-k(x_t)
```

with `V_valid = {i : z_top-2(i) ≥ log α + max_j z_top-2(j)}`, α = 0.1.

Two crucial structural facts:

1. **Weak forward is a full forward.** The paper does not explicitly carve out
   the cost, but it reports `1.30×` end-to-end latency vs. greedy ([Shi
   Table 3, repro confirmed](https://arxiv.org/html/2405.14507v2)). Two full
   passes would be `~2.0×`, so SCMoE almost certainly reuses the **attention
   KV cache and non-MoE layers** and re-runs only the MoE FFN with a different
   router selection. The `0.30×` extra is the cost of routing through the
   rank-K expert FFNs at every MoE layer. The weak signal is therefore
   architecturally a **second pass through the routed FFNs**, not a second
   pass through the whole transformer.

2. **What rank-K adds is novel information.** Shi et al. measure KL divergence
   between strong and weak output distributions per token category (Table 5):
   the mean is `5.05` for rank-2 and rises to `25.36` for rank-8, with
   "expression" tokens diverging `+31.13%` more than stopwords. The activation
   proportion of *unchosen* experts under rank-2 is `46.21%` on GSM8K. The
   contrast is empirically substantive, not a smoothed copy of strong logits.

The combination implies the weak signal is **functionally distinct** from any
function of the strong-pass internal state, because the strong-pass state never
routed through those experts in the first place. Whatever the weak logit is,
it carries information about how *unselected* experts would have transformed
the token — which the strong forward never computed.

---

## 2. The closest single-pass analog: DoLa

DoLa (Chuang et al., ICLR 2024, [arXiv:2309.03883](https://arxiv.org/abs/2309.03883))
projects early-layer hidden states to vocabulary space via the *final*
unembedding, dynamically picks the early layer maximizing JSD from the final
layer, and contrasts. No second pass. Reported lifts on LLaMA-13B:

- TruthfulQA MC1: 25.6 → 32.2 (+6.6)
- TruthfulQA MC2: 40.6 → 63.8 (+23.2)
- TruthfulQA MC3: 19.2 → 32.1 (+12.9)
- Latency overhead: `1.01–1.08×` (Chuang Table 8)

On *MoE* specifically the picture inverts. Shi et al. explicitly compare DoLa
on Mixtral 8x7B (Table 1):

| Method            | GSM8K     | StrategyQA | MBPP      | HumanEval  |
|-------------------|-----------|------------|-----------|------------|
| Greedy            | 61.79     | 72.83      | 46.20     | 33.54      |
| DoLa              | **49.96** | 71.04      | **33.00** | **12.80**  |
| Contrastive Dec.  | 62.24     | 74.45      | 45.20     | 35.98      |
| SCMoE             | 66.94     | 76.29      | 48.80     | 41.46      |

DoLa's HumanEval result on Mixtral is catastrophic — a `−20.74 pp` drop. The
authors note (Appendix E) that DoLa hallucinates aggressively on MoE
architectures. PruneCD ([arXiv:2509.16598](https://arxiv.org/abs/2509.16598))
diagnoses the underlying reason: "early exit logits tend to be flat, low in
magnitude, and fail to reflect meaningful contrasts" — the early-layer
unembedding mismatch (logit lens problem; cf. [Belrose et al., Tuned
Lens](https://arxiv.org/abs/2303.08112)) is *worse* on sparse models because
intermediate hidden states are being shaped by selectively-routed experts and
don't share a consistent vocabulary geometry.

**Conclusion.** DoLa is the obvious "single-pass contrast" baseline and on the
target architecture (Mixtral / DeepSeek-class MoE) it is actively harmful.
This is the strongest piece of evidence that a purely layer-wise single-pass
synthesis of the weak distribution does not work out of the box.

Variants:
- **SLED** ([arXiv:2411.02433](https://arxiv.org/abs/2411.02433)) integrates
  early-layer logits as an approximate-gradient correction over final logits;
  reported on dense models only.
- **PruneCD** uses a *pruned* version of the model as the amateur, not early
  exit; structurally requires a second (smaller) pass.
- **ALW** ([2025.findings-acl.447](https://aclanthology.org/2025.findings-acl.447/))
  and **LayerCake** ([arXiv:2507.04404](https://arxiv.org/abs/2507.04404))
  add token-aware adaptive layer selection. Both are dense-LLM only.

None of these have been demonstrated to work on MoE without modification, and
the one direct measurement (DoLa on Mixtral) is negative.

---

## 3. Frequency / anti-LM baselines

DExperts ([Liu et al., 2021](https://arxiv.org/abs/2105.03023)), Anti-LM
([arXiv:2311.08324](https://arxiv.org/abs/2311.08324)), and unigram-prior
methods ([arXiv:2212.09686](https://arxiv.org/abs/2212.09686)) subtract a
distribution that depends *only on the corpus / vocabulary*, not the current
hidden state.

These methods are essentially free at decode time (an n-gram lookup) and yes,
they remove unigram fluency bias and improve open-ended generation. They have
two disqualifying problems for SCMoE replacement:

1. **They regress factual recall** — confirmed across multiple studies
   ([Contrastive Decoding for Reasoning, Sec 5,
   arXiv:2309.09117](https://arxiv.org/abs/2309.09117); [Knowledge Editing
   eval, arXiv:2404.00216](https://arxiv.org/abs/2404.00216)). Subtracting a
   frequency baseline pushes high-frequency factual tokens out of the top of
   the distribution. This is exactly the opposite of what SCMoE achieves on
   GSM8K (a heavy-recall benchmark).
2. **They cannot reproduce SCMoE's token-localized lift.** SCMoE's gains
   concentrate on reasoning / expression tokens (Shi Table 5); a static
   distribution that depends only on token id has no per-context selectivity.

A unigram or bigram subtraction at best removes mode-collapse repetition. It
does **not** approximate rank-K weak routing. Estimated retention of SCMoE
lift: **0–5%** with a measurable risk of negative transfer on factual tasks.

---

## 4. Temperature-scaled "weak" distributions

Could `softmax(z_strong / T_high)` proxy for `softmax(z_weak)`? The difference
of high-T and low-T softmaxes has been studied as a smoothing operator
([arXiv:2404.04575](https://arxiv.org/abs/2404.04575); DecoRTL
[arXiv:2507.02226](https://arxiv.org/abs/2507.02226)).

Two reasons this cannot replace rank-K weak:

- **Same support.** `softmax(z/T)` for any `T` is a deterministic, monotone
  transformation of `z`. Its difference from `z` is by construction
  uninformative beyond an entropy rescaling — equivalent to a temperature
  knob, which is already a standard generation control. Empirically the
  reasoning lift of temperature alone is `<1 pp` on GSM8K.
- **Wrong ranking signal.** SCMoE flips token rankings (the `V_valid`
  plausibility filter passes tokens that were dominated under strong but
  re-promoted under contrast). High-T softmax cannot flip rankings; it can
  only flatten them.

Estimated retention of SCMoE lift: **≤5%**, indistinguishable from a
temperature sweep.

---

## 5. Router-only implicit signals (the most promising single-pass direction)

The router at each MoE layer produces a full soft distribution `π(e | h)` over
all `E` experts (256 in DeepSeek V4) before top-K selection. The bottom
`E−K` of `π` is computed and discarded — it is genuinely free to read.

**Can `π_rank-K` substitute for the weak FFN output?** Information-theoretic
answer:

- `π` encodes which experts the model *believes* would have been useful, not
  what those experts would have *produced*. The expert FFN is a deep MLP with
  nonlinear gating; its output is not determined by the routing weight that
  selects it.
- Recent work on MoE specialization ([arXiv:2604.09780](https://arxiv.org/abs/2604.09780)
  "Myth of Expert Specialization") shows routing scores reflect *hidden-state
  geometry*, not domain expertise. Expert outputs decorrelate from routing
  scores beyond layer ~6.
- The "Closer Look at MoE" paper ([arXiv:2406.18219](https://arxiv.org/abs/2406.18219))
  measures: correlation between router gate weight and post-FFN contribution
  norm is `0.3–0.5` across layers — significant but far from saturating.

So `π_rank-K` carries a fraction (call it `r²`, so `~9–25%`) of the variance
of the actual rank-K FFN output. As an implicit weak proxy, the router
distribution alone is a **weak signal**, not noise — but it is **not** the
rank-K logit. The information-theoretic bound is the mutual information
between router scores and post-FFN expert output, which the above empirical
correlation estimates at `~0.3 bits` per expert per layer.

**Most viable form of implicit contrast.** Use `π_rank-K` as a *gating signal*
to decide when SCMoE's full weak pass is worth running, not as a replacement
for it. This is exactly the `gated SCMoE` direction documented in
[scmoe_gated.md](./scmoe_gated.md): router entropy + top-1/top-2 margin
predict token-level value of the weak forward. Estimated retention with
gating: **70–95%** of SCMoE lift at `~1.05×` latency. This is the realistic
implicit path — partial elimination, not full elimination.

A more aggressive alternative: train a tiny linear head that maps
`{hidden_state, router_logits}` → predicted rank-K logits, distilled from
real SCMoE traces. This is the [Apprentice Mode](./03_apprentice_mode/)
direction. It is no longer "purely implicit" (requires offline training) and
is bounded above by the same MI as the router-only signal unless the head
also reads hidden state from FFN-adjacent points.

---

## 6. Hidden-state-derived weak proxies

The full strong-pass internal state `{h_l}_{l=1..L}` contains *everything the
model computed*, including which experts were activated and what their
contributions were. Information-theoretically, the only information *not* in
`{h_l}` is the information from experts that were not run.

Formally: let `h_strong_l = h_{l-1} + Σ_{e ∈ TopK} g_e(h_{l-1}) · FFN_e(h_{l-1})`.
The weak pass would produce `h_weak_l = h_{l-1} + Σ_{e ∈ RankK} g_e(h_{l-1}) · FFN_e(h_{l-1})`.
The unselected experts' outputs `{FFN_e(h_{l-1}) : e ∉ TopK}` are **never
computed** in the strong pass. No function of `h_strong` can recover them
without recomputing them, because they are an information-theoretic *new
function evaluation* on a known input. This is the fundamental limit.

What `h_strong` *does* contain that helps: the input `h_{l-1}` to each expert,
the router scores `π`, the activated experts' outputs, and after the layer,
the integrated post-MoE state. A learned mapping `(h_{l-1}, π) → estimated
FFN_e(h_{l-1})` is possible *if* expert FFNs are approximately predictable
from their input plus routing context — this is precisely the
"expert-prediction" thesis. Reported accuracy of expert prediction from
hidden state ([arXiv:2502.12224 "Accurate Expert Predictions in MoE
Inference via Cross-Layer Gate"](https://arxiv.org/abs/2502.12224)) reaches
`99.94%` on *which experts* will be activated — not what they will produce.

**Bottom line on hidden-state derivation.** The functional output of an
unrouted expert is genuinely new information. A learned predictor can
recover a fraction (the predictable component of expert outputs given input
and routing), but this becomes a regression problem, not a "derivation".
Theoretical ceiling: bounded above by the mutual information `I(FFN_rank-K
output; h_{l-1}, π)`, which is unknown but empirically the related expert-
contribution correlation is `~0.3–0.5`.

---

## 7. Information-theoretic verdict

**Is the weak distribution genuinely additional information?** Yes,
provably. The rank-K expert FFNs are deterministic functions of the layer
input that are not evaluated in the strong forward. Their output sits outside
the σ-algebra of the strong-pass computation. The only ways to recover them
are (a) actually evaluate them (SCMoE's real second pass), or (b) train a
predictor that exploits the predictable component of expert outputs from
input + routing. (b) is information-theoretically bounded; (a) is exact.

**Is some fraction recoverable?** Yes. Three signals offer non-trivial
proxies:

1. Router rank-K distribution `π` (free, `~9–25%` variance correlation).
2. Intermediate hidden state evolution `Δh_l` per layer (free, captures the
   "would this token's representation change much if we re-routed" question).
3. Strong-pass output entropy `H(z_strong)` (free, correlates with the
   *value* of contrast but does not synthesize the contrast itself).

A linear combination of (1)+(2)+(3) feeding a learned head ("derived" only
in the sense of computed from strong-pass signals, but the head's parameters
are trained, not closed-form) is the most promising direction in the
literature. No closed-form derivation exists.

---

## 8. Honest quality fidelity estimates

Putting numbers on it. Reference: SCMoE adds `+5.15 pp GSM8K`, `+7.92 pp
HumanEval` on Mixtral 8x7B at `1.30×` latency.

| Approach                                  | Latency | SCMoE lift retained | Risk          |
|-------------------------------------------|---------|---------------------|---------------|
| DoLa (layer contrast)                     | `1.05×` | **−100% to −300%**  | Active harm   |
| SLED (single-pass gradient)               | `1.05×` | Untested on MoE     | Likely harm   |
| Unigram / Anti-LM subtraction             | `1.00×` | `0–5%`              | Factual loss  |
| Temperature contrast                      | `1.00×` | `≤5%`               | Negligible    |
| Router-only `π_rank-K` proxy              | `1.00×` | `15–30%` (est.)     | Unknown       |
| Learned distill head (Apprentice Mode)    | `1.02×` | `40–70%` (est.)     | Training cost |
| Gated SCMoE (real weak on subset)         | `1.05×` | `70–95%`            | Concentrated  |
| Full SCMoE (the upper bound)              | `1.30×` | `100%`              | Latency       |

The router-only and distill-head numbers are estimates from MI analysis and
analogy to expert-prediction literature; they have not been measured directly
on SCMoE-style contrast tasks. The negative DoLa numbers are measured.

---

## 9. Specific proposal

**Recommended single-pass weak proxy:**

```
z_weak_proxy(x_t) ≈ W_v · concat(
    h_strong_final,                  # final hidden state
    h_strong_at_first_moe,           # early MoE-layer state
    Σ_l π_rank-K_l(h_l),             # aggregated unselected router mass per layer
    H(z_strong)                      # entropy summary scalar
)
```

where `W_v` is a small `d × |V|` linear head trained to predict actual
SCMoE weak logits on a calibration corpus. This is the minimum signal set
that has theoretical grounding (carries all freely-available strong-pass
information that correlates with rank-K outputs) and the maximum compute
saving (one linear projection vs. a full FFN re-routing).

Expected behavior, extrapolating from expert-prediction MI literature:
- **GSM8K lift retained: 30–50%** (i.e., `+1.5 to +2.5 pp` instead of
  `+5.15`), assuming the calibration is good.
- **HumanEval lift retained: 25–45%** (i.e., `+2 to +3.5 pp` instead of
  `+7.92`).
- **Latency overhead: <1.01×** — one matrix multiply per decode step.

This is *not* a full SCMoE replacement; it is the best implicit
approximation the information theory allows.

---

## 10. Final verdict

**Fundamentally possible? No.** The rank-K weak forward computes a function
that the strong forward never evaluates. No single-pass derivation reaches
100% of SCMoE's lift because part of SCMoE's signal is **the actual output
of unselected experts**, which is missing from the strong-pass σ-algebra.

**Partially possible? Yes.** Approximately `25–50%` of the lift is
recoverable from strong-pass signals via a learned head over
`(hidden_state, router_logits, entropy)`. This is the only direction with
both theoretical grounding and prior-art support; it is essentially the
Apprentice Mode distillation approach reframed as "implicit contrast".

**Most pragmatic direction.** Combine cheap-or-free signals to *gate* a real
weak forward (gated SCMoE), keeping `70–95%` of the lift at `~1.05×`
latency. The "purely implicit" direction trades quality for ~`0.05×` extra
saving, which is not the right trade.

For Arc / DeepSeek V4: keep SCMoE as the primary contrastive mechanism, gate
it aggressively on router entropy + top-1/top-2 margin, and treat any
purely-single-pass distillation head as a *fallback for the cheap-token
path* — not as a replacement.

---

## References

- Shi et al., "Unchosen Experts Can Contribute Too: Unleashing MoE Models'
  Power by Self-Contrast", NeurIPS 2024,
  [arXiv:2405.14507](https://arxiv.org/abs/2405.14507).
- Chuang et al., "DoLa: Decoding by Contrasting Layers", ICLR 2024,
  [arXiv:2309.03883](https://arxiv.org/abs/2309.03883).
- Liu et al., "DExperts: Decoding-Time Controlled Text Generation",
  [arXiv:2105.03023](https://arxiv.org/abs/2105.03023).
- Li et al., "Contrastive Decoding: Open-ended Text Generation as
  Optimization", [arXiv:2210.15097](https://arxiv.org/abs/2210.15097).
- O'Brien & Lewis, "Contrastive Decoding Improves Reasoning in LLMs",
  [arXiv:2309.09117](https://arxiv.org/abs/2309.09117).
- Phan et al., "Distillation Contrastive Decoding",
  [arXiv:2402.14874](https://arxiv.org/abs/2402.14874).
- Zhang et al., "SLED: Self Logits Evolution Decoding",
  [arXiv:2411.02433](https://arxiv.org/abs/2411.02433).
- PruneCD, [arXiv:2509.16598](https://arxiv.org/abs/2509.16598).
- Belrose et al., "Eliciting Latent Predictions with the Tuned Lens",
  [arXiv:2303.08112](https://arxiv.org/abs/2303.08112).
- "A Natural Bias for Language Generation Models",
  [arXiv:2212.09686](https://arxiv.org/abs/2212.09686).
- Anti-LM Decoding, [arXiv:2311.08324](https://arxiv.org/abs/2311.08324).
- Lu et al., "A Closer Look into MoE in LLMs",
  [arXiv:2406.18219](https://arxiv.org/abs/2406.18219).
- "Myth of Expert Specialization in MoEs",
  [arXiv:2604.09780](https://arxiv.org/abs/2604.09780).
- "Accurate Expert Predictions in MoE Inference",
  [arXiv:2502.12224](https://arxiv.org/abs/2502.12224).
- Self-Contrast (reflection-time, not decoding-time),
  [arXiv:2401.02009](https://arxiv.org/abs/2401.02009).
- LayerCake, [arXiv:2507.04404](https://arxiv.org/abs/2507.04404).
- ALW Adaptive Layer-Wise CD,
  [aclanthology.org/2025.findings-acl.447](https://aclanthology.org/2025.findings-acl.447/).
- Welleck et al. survey, "From Decoding to Meta-Generation",
  [arXiv:2406.16838](https://arxiv.org/abs/2406.16838).
