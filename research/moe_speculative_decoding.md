# MoE-Aware Speculative Decoding for Arc / DeepSeek V4

**Status:** Research-only. No source modifications.
**Author:** Arc research agent
**Target model:** DeepSeek V4-Flash (256 routed + 1 shared expert, top-6, 37 B active / 671 B total)
**Baseline:** Arc's existing `MtpSpeculativePipeline` (Tier A, greedy, depth 1, ~50% acceptance on V4-flash) and `SpeculativePipeline` (dense-style draft+target, MoE-agnostic).

---

## 1. Survey of speculative decoding methods (relevance to MoE)

| Method | Year | Mechanism | Dense speedup (reported) | MoE compat. notes |
|---|---|---|---|---|
| **Vanilla SD** (Leviathan et al.) | 2023 | Small dense draft + target verification w/ rejection sampling | 2–3× | Works on MoE but draft routing ≠ target routing breaks the acceptance assumption probabilistically; verification cost inflates because draft tree activates many experts (Cascade [arXiv:2506.20675] reports 1.5× *slowdown* on some MoE setups). |
| **Medusa** (Cai et al., ICML'24) | 2024 | Multiple feed-forward heads attached to target final hidden | ~2× | Heads operate post-MoE on the residual stream → MoE invariant. Lightweight to attach but doesn't exploit routing. |
| **EAGLE-1 / 2 / 3** (Li et al., NeurIPS'25) | 2024–25 | Auto-regressive draft transformer that consumes target's hidden features + token embedding; tree drafting in EAGLE-2/3; multi-layer feature fusion + training-time test in EAGLE-3 | 3.0–6.5× (Vicuna-13B → 70B; T=0) | EAGLE-3 *supports* Mixtral-8×7B and DeepSeek-R1-Distill-LLaMA-8B but the SafeAILab repo and the EAGLE-3 paper explicitly note no public 405B/671B results; SGLang's V4 deploy uses EAGLE-3/1/4 (`num-steps=3, eagle-topk=1, num-draft-tokens=4`) *replacing* V4 MTP, achieving ~2.5 accept length [LMSYS V4 blog]. |
| **Hydra** (Ankner et al., COLM'24) | 2024 | Sequentially-dependent Medusa heads | ~2.5× | Same MoE-invariance argument as Medusa; better than Medusa baseline but still ignores routing. |
| **Sequoia** (Chen et al., NeurIPS'24) | 2024 | DP-optimal draft tree topology, robust to high temperature | 2–4× | Orthogonal to MoE; tree shape applies on top of any draft model. Tree size inflates expert activation set, however. |
| **LayerSkip** (Elhoushi et al., ACL'24) | 2024 | Self-speculative: first E layers of target = draft, remaining layers = verify; trained with layer dropout + early-exit loss | 1.3–2× | MoE-relevant: draft passes only the bottom-E *experts* of the same MoE → routing-coherent by construction. Public LayerSkip is dense Llama only; no published MoE training recipe. |
| **MTP** (DeepSeek V3/V4) | 2024–26 | Native head trained jointly: predicts token at position t+k from hidden state(s); reused at inference as a 1-step draft | V3 reports ~1.8× w/ 85–90% accept on MTP-1 [DeepWiki V3]; V4-Flash MTP depth = 1 [TAAC V4 paper] | This *is* the MoE-native baseline. Acceptance comes for free because the MTP head was trained on the same data and shares the LM head; arc/`mtp_pipeline.rs` already wires this for V4. |
| **MoE-SpeQ** (arXiv 2511.14102, 2026) | 2026 | 4-bit quantized version of target = draft; draft also produces "Expert Lookahead Buffer" for proactive expert prefetch in offload setting | 2.5–4.8× (Phi-MoE, Qwen2-MoE, DeepSeekV2-Lite); 2.34× over best offloading baseline | >90% acceptance because draft is literally a quantized target. Targets offload regime; doesn't compose with V4 MTP. |
| **SP-MoE** (arXiv 2510.10302, 2025) | 2025 | Maps draft attention outputs to target gating networks; cosine sim 56–95% between layers; SD-aware prefetch | 1.07–3.5× TPOT, peak 3.5× on DeepSeek-Lite + A100 | **97% acceptance reported on DeepSeek-Lite-AWQ pairing.** Closest published analogue to "routing-aware" SD; orientation is still I/O latency hiding, not target verification quality. |
| **MoE-Spec** (arXiv 2602.16052, 2026) | 2026 | Same EAGLE-3 draft, but caps verification to top-B experts by aggregate routing weight | 10–30% throughput over EAGLE-3 baseline at matched quality | Heavy-tail observation: top-32/64 = 93% routing mass on OLMoE; paper explicitly flags V3/V4-style sigmoid routing as future work. |
| **Cascade** (arXiv 2506.20675, 2025) | 2025 | Utility-driven on/off + adaptive K | Limits slowdown to 5% (vs 1.5× worst-case) + 7–14% throughput | The pessimistic case study: MoE break-even acceptance is fundamentally higher because verification cost scales with *unique experts in draft tree*, not just FLOPs. |
| **Jakiro** (Hu et al., ACL'26, arXiv 2502.06282) | 2025 | EAGLE-style draft but routes the *draft heads* through an MoE so the K tree candidates are produced by *different* experts → decorrelated drafts | "Highest acceptance length" across tested models | The MoE is in the draft, not the target — orthogonal to V4's target-side MoE. Could be stacked. |
| **MoESD** (arXiv 2505.19645, 2025) | 2025 | Theoretical: SD on MoE benefits *more* than dense in a moderate-batch sweet spot | 2.29× on Qwen2-57B-A14B at BS≈4 | Reframes Cascade's pessimism: BS=1 is hostile to MoE+SD; BS=2-8 is the sweet spot because verification reuses experts already loaded for decode. Routing not used dynamically. |

**Bottom line for the survey:** EAGLE-3 sets the dense ceiling (~5–6.5×) but its published evaluation does not include DeepSeek-V3/V4 671B. On large MoE in production (SGLang's V4 deploy), EAGLE-3 in practice yields ~2.5 accept length — *worse* than the speedup ratio EAGLE-3 reaches on 70 B dense. V4's native MTP gets 1.8× at ~85% accept on MTP-1 essentially for free because the head is co-trained. There is no published method that uses target MoE routing as a draft-time signal for *acceptance-rate* improvement — every MoE-aware SD paper so far targets *I/O scheduling* (prefetch) or *verification cost* (expert budget).

## 2. MoE-aware variants in literature — the actual gap

The literature does not lack MoE-aware speculative work; what it lacks is MoE-aware work targeting the *acceptance-rate / draft-quality* axis on large frontier MoE. The body of work clusters into three buckets:

1. **Prefetch / offload latency hiding** — SP-MoE, MoE-SpeQ, SpecMD, "Speculating Experts" (arXiv 2603.19289). Draft tokens or draft attention outputs predict *which experts* the target will activate; the predictions feed a prefetcher. Acceptance rate is high (90–98%) but only because the draft is essentially the target at lower precision. These methods solve a different problem (CPU↔GPU bandwidth on memory-constrained setups), not one Arc has on B200 with full V4 in HBM.

2. **Verification-cost shaping** — Cascade (turn SD on/off, adapt K), MoE-Spec (cap expert budget). Both reduce the *cost side* of `SU = (1+γα)/(1+γc/Cb)`. Neither improves α (acceptance).

3. **Draft architecture using MoE** — Jakiro (MoE in the draft heads to decorrelate candidates), and the broader EAGLE family pretending MoE doesn't exist. Jakiro is the only paper that lets MoE structure improve α, and it does so by *adding* MoE to the draft, not by exploiting the target's routing.

**The conspicuous gap:** *no published paper uses the target's expert routing as input to the draft model to predict the next token's routing → next token.* The closest is SP-MoE's layer-to-layer attention mapping, but that's used for prefetch, not next-token logits. This is the publishable hole.

## 3. The MoE-aware draft architecture proposal

### 3.1 Routing signature as draft input

V4 emits, for each token at each of 58 MoE layers (V4-Flash), a top-6 selection over 256 experts plus the gate logits. That's 58 × (6 indices + 6 weights) ≈ 696 floats per token — a 696-dim "routing fingerprint" that is highly informative about what the model is doing on this token (topic, register, syntactic role; the DeepSeekMoE specialization paper [arXiv 2401.06066] shows experts specialize). A draft that consumes this fingerprint, in addition to the residual hidden state, should produce a much better next-token distribution than one that only sees hidden state, because routing is the model's already-computed compressed representation of "what kind of token this is."

Architecture sketch (call it **EAGLE-MoE** for now):

```
target step t produces:
    h_t            : [hidden]       residual
    r_t = {top6_idx, top6_w}_{L}    routing signature, shape [58 × 12]
    e_t            : [hidden]       last-token embedding

draft step t+1..t+γ:
    f_t = h_proj(h_t) + e_proj(e_t) + r_proj(r_t)    # add routing channel
    f_t = single attention block over recent f's
    logits_t = lm_head(f_t)
    sample token, embed, loop (EAGLE-style autoregression)
```

The added cost is `r_proj`: a `[58*12 → hidden]` linear, run once per draft step. On B200 this is ~70 µs at hidden=7168, negligible against draft attn.

### 3.2 Where to get `r_t` for past tokens during training

Target rollouts on ShareGPT-style data, with hidden states *and* per-layer top-k indices logged. This is a one-time pre-pass; SafeAILab's EAGLE-3 training recipe already needs hidden state rollouts, so the marginal cost is just dumping `topk_idx` and `topk_w` per MoE layer. Expected: a few TB of side data per 100k conversations. Cheap.

### 3.3 Why this beats EAGLE-3 on V4 specifically

EAGLE-3's training-time test trick simulates 5-step autoregression at training time. Its draft's failure mode is *off-manifold drift*: the more steps it autoregresses, the more its hidden states diverge from where the target's would land. Routing fingerprints are a *compact target oracle* the draft can lean on for the first token (where the target *just* computed `r_{t-1}`) and which the draft must *predict* for t+2..t+γ — which is the same problem EAGLE-3 already solves for hidden states but in a far lower-dimensional, semantically tighter space (top-6-of-256 routing is closer to a few hundred bits than to 7168 floats). The hypothesis: this drastically tightens drift across γ ≥ 3.

### 3.4 Alternative: "shrunk MoE" draft

Instead of routing-aware dense draft, build a *small MoE draft* — e.g., 32 experts top-2 with the same router head as the target's first MoE layer, distilled. This is what one could naively call "draft is also MoE." The honest assessment: the per-token compute of a 32-expert top-2 draft is similar to a 1B dense draft; the *latency* advantage over a dense draft of the same active size is zero on a single B200. Worth it only if (a) expert-sharing between draft and target enables cache reuse, or (b) routing entropy matches target better than dense draft. SP-MoE de facto explored (a) for offload and reported 97% accept; on HBM-resident V4 (b) is the open question. I'd ship the routing-fingerprint dense draft first — it's a smaller code change and the architecture is cleaner.

## 4. Quality improvement mechanism — honest assessment

Standard speculative decoding (Leviathan rejection sampling, EAGLE accept-or-reject) **provably preserves the target distribution token-for-token**. It does not improve quality. Period. Anyone claiming otherwise has confused speedup with quality.

There are two genuine "speculation-as-quality" mechanisms in the literature:

1. **Reward-shifted speculative sampling** (arXiv 2508.15044, EMNLP'25) — train an aligned draft, leave the target unaligned; modify the acceptance criterion so accepted samples are drawn from a distribution closer to the RLHF-optimal aligned target. Reports +0.6 on OPT-6.7B reward over Best-of-N at 2.9–5.4× speed. *This works because acceptance criterion is intentionally biased; it's not the standard Leviathan/EAGLE acceptance.* Strictly speaking, this is "alignment via draft", not "quality from speculation."

2. **Speculative Rejection** for Best-of-N (Sun et al.) — use draft+rejection to drop bad branches early in a BoN search, dropping the cost of BoN ~16–32×. Quality up because BoN is up; speculation just makes BoN affordable. Composes with reward models.

**MoE-specific angle: none.** The routing signature could in principle rank candidates by "routing similarity to a known-good response," but this is post-hoc and would require a reference. There's no published mechanism by which MoE routing improves *standard* speculative quality, and the verification math says there can't be — the target gets the final word at every position. Calling MoE-aware drafting a quality improvement would be a category error.

## 5. Speedup analysis on V4 vs MTP baseline

Speculative speedup formula (Leviathan 2023, Cascade 2025 form):
`SU = (1 + γα) / (1 + γ·c/C_b)`
where γ = draft length, α = mean acceptance, c = draft latency / token, C_b = target latency / token.

For V4-Flash on B200 (Arc's target), single-stream decode:
- C_b ≈ 8.3 ms/tok (Arc's measured TPS ~120 → 8.3 ms; matches V4 deploy numbers)
- V4 MTP head: c_MTP ≈ 0.5 ms, α_MTP ≈ 0.85 at depth 1, γ=1 → SU ≈ (1 + 0.85)/(1 + 0.06) ≈ 1.74× ✓ matches DeepSeek's reported 1.8×.

Now project EAGLE-MoE on V4 (routing-fingerprint draft, depth 3):
- Draft is a single 8-layer transformer w/ hidden=2048: c ≈ 0.6 ms (B200, BF16, hidden=2048)
- Hypothesized α at γ=3: 0.75 (EAGLE-3 reports ~0.7–0.8 on dense; routing signal *should* match or beat that)
- SU = (1 + 3·0.75) / (1 + 3·0.6/8.3) = 3.25 / 1.217 ≈ **2.67×** vs MTP's 1.74× → ~1.5× *additional* on top of V4 MTP if stacked (see §8).

Bounds:
- Upper bound (α=1, ignoring all cost): SU → 1+γ = 4× at γ=3.
- Realistic floor: if routing fingerprint adds nothing over hidden state, α stays at EAGLE-3's ~0.7 dense ⇒ ~2.4×; still > MTP's 1.74×.

This is the headline number worth chasing. **Caveat — verification cost on MoE:** V4's target verification of 3 draft tokens may activate more unique experts than vanilla decode, inflating C_b's effective term. Cascade reports 2–3× verification inflation. For B200 with full model in HBM (Arc's case), expert "loading" cost is zero — experts are *already* there; the inflation is just extra matmuls. At V4-Flash's 37 B active and B200's ~2 PB/s HBM, the extra activations are ~3× the unique-expert count: with top-6 over 58 layers and 3 candidate tokens, at random 64% of experts get hit somewhere in the tree (analytical estimate; MoESD's `1 − ((E−K)/E)^t` formula). That's ~165/256 experts touched, vs 6×58=348 expert-positions for single-token decode anyway. On HBM, this is a 1.1–1.3× verification penalty, not 2–3×. Cascade's 2–3× number is for offload setups, not B200 HBM.

So the corrected formula with HBM-MoE penalty:
`SU = (1 + γα) / (β + γ·c/C_b)` where β ≈ 1.15 (HBM-MoE verification inflation)
- EAGLE-MoE on V4 at γ=3, α=0.75: SU ≈ 3.25 / 1.367 ≈ **2.38×**.
- Still > V4 MTP baseline of 1.74×.

## 6. Composition with TEAL / DSA / xKV / adaptive top-k

| Other technique | Conflict / compound with EAGLE-MoE |
|---|---|
| **TEAL** (FFN activation sparsity) | Compound. TEAL prunes elementwise inside expert FFNs; orthogonal to routing fingerprint. Verification cost shrinks ~1.3–1.5× from TEAL → β drops → SU goes up. |
| **DSA Indexer** (top-k sparse attention) | Compound during target verification: shorter attn matmul per token. Acceptance unaffected. Multiplicative speedup. Draft can also use DSA; tiny absolute saving (draft attn is small). |
| **xKV / TurboQuant KV** (K4/V3 codebook KV) | Compound. KV decode mems shrink → β shrinks → SU up. Draft has its own KV; either skip quant on draft (negligible mem) or share K4/V3. Acceptance unaffected. |
| **Adaptive expert top-k** | *Potential conflict.* If target dynamically picks top-4 instead of top-6 on some tokens, the routing fingerprint distribution shifts; draft trained on top-6 routing degrades. Fix: train draft with k jittered, or feed `k_actual` as a token in the fingerprint. Manageable. |
| **Speculative routing prediction** (predict expert routing one layer ahead) | Compound + synergy. The same routing-fingerprint pipeline used for the draft head can also feed a per-layer routing predictor; both share the `r_proj` infrastructure. Notable in §3.4. |
| **V4 native MTP head** | See §8 below. Two stacking options. |
| **Sequoia tree shape** | Compound. Tree topology is orthogonal to draft model. EAGLE-MoE produces a logit stream; Sequoia DP picks the optimal tree. Yields the canonical 1.2–1.4× over linear draft. |

The composition picture is favorable: every other Arc optimization either shrinks β or has no interaction, except adaptive top-k which requires draft training to be top-k-aware. Nothing on the list conflicts terminally.

## 7. Integration sketch with Arc's `SpeculativePipeline`

Arc currently has two relevant pieces:

- `mistralrs-core/src/pipeline/speculative.rs` (967 lines) — dense draft+target with rejection sampling; **zero MoE awareness** (verified by grep: no occurrences of `expert`, `MoE`, or `gate` in this file). Implements the standard Leviathan path with γ tokens drafted then verified against target logits via `sample_target_sequence_speculative`. KV-cache rollback on rejection is handled for `Full`, `Normal`, and `Hybrid` caches.

- `mistralrs-core/src/pipeline/mtp_pipeline.rs` (767 lines) — V4 MTP head wrapper. `MtpDecodeKit` carries `{embed_tokens, lm_head, h_proj, e_proj}` (the four V4 MTP tensors) and steps with `fused = h_proj(h_prev) + e_proj(e_last)`; greedy verification only (Tier A). Acceptance counters exposed for telemetry.

Integration shape for EAGLE-MoE:

1. **New struct `MoeAwareDraftKit`** alongside `MtpDecodeKit`. Fields: `embed_tokens, lm_head, h_proj, e_proj, r_proj` and a small `transformer: Vec<TransformerBlock>` (single attn block, sized hidden=2048). All `Arc<dyn QuantMethod>` to play nicely with ISQ.
2. **Trait extension on `NormalModel`**: add `fn routing_fingerprint(&self, last_step: &ForwardCtx) -> Option<Tensor>` returning `[batch, 58*12]`. V4 implements it by reading the saved `topk_idx`/`topk_w` per MoE layer that the model already computes; other models return `None`. This is a target-side change but localized — V4's MoE forward already has `topk_idx, topk_w` in scope.
3. **New `EagleMoePipeline`** in a new file `eagle_moe_pipeline.rs`, mirroring the structure of `MtpSpeculativePipeline` but with `depth` > 1 and tree drafting (Sequoia-shape optional, linear γ default). Verification reuses Arc's existing `sample_target_sequence_speculative` for stochastic, or a greedy fast path mirroring `MtpSpeculativePipeline`.
4. **Loader**: A `EagleMoeLoader` wrapping a `target: Box<dyn Loader>` and pointing at a draft checkpoint path on disk. Draft is a separate small checkpoint, not part of V4. Add the `--draft-eagle-moe <path>` CLI flag in `mistralrs-cli/src/main.rs` and the corresponding `ModelKind::EagleMoe` enum branch.
5. **Training is out of scope for Arc.** Adapt SafeAILab's EAGLE-3 trainer; add routing-fingerprint logging during target rollout (~20 LoC patch in the trainer's hidden-state-dump path). The output is a draft checkpoint Arc can load.

Effort estimate, agent-session calibration (per CLAUDE.md):
- New pipeline + loader + glue, mirroring existing patterns: ~1.5–2 hours.
- V4 model patch to expose routing fingerprint: ~30 min (search for `topk_idx` in `models/deepseek_v4.rs`, plumb a side channel into `ForwardCtx`).
- Tests (mirror `mtp_pipeline.rs` test stanza, sizes hidden=64 vocab=64): ~30 min.
- Training fork + first checkpoint on V4-Flash: **outside agent time**; this is the GPU rental + training-data-rollout step. Calibrate against EAGLE-3's training recipe — typical published reports are ~24–48 GPU-hours for a Vicuna-13B EAGLE-3 head; V4-Flash with routing input is wider data but same head depth, plausibly 100–200 H100-hours for a usable head.

**Total Arc-side engineering: ~3 hours of focused agent work for the integration, with training as a separate offline pipeline.**

## 8. Stacking with V4 MTP

Three patterns possible:

1. **Replace MTP with EAGLE-MoE.** What SGLang does today. EAGLE-MoE alone, target uses no MTP head. SU = 2.4× over plain V4.
2. **Sequential stack: MTP draft, EAGLE-MoE refines.** MTP produces 1 token cheaply (0.5 ms, 85% accept); when MTP rejects, EAGLE-MoE proposes γ-1 alternatives. The gain over either alone is small because once MTP accepts, EAGLE-MoE's slot disappears.
3. **Parallel stack: both contribute candidates into a tree.** MTP token at position +1 + EAGLE-MoE tokens at +1..+γ form a tree, target verifies the union. This is the Sequoia-on-top-of-multiple-drafts pattern. Probably the highest-ceiling option; published precedent: HASS, Glide, ParallelSpec do versions of this for dense. Estimated SU ~2.7–3.0× combined.

Recommendation: ship pattern 1 first (clean ablation against MTP baseline), then add pattern 3 if there's headroom.

## 9. Honest verdict — gap or marginal?

**Verdict: real but narrow gap, of mixed publishability.**

Three honest claims:

1. **The "use target MoE routing as draft input" architectural hole is real and unfilled.** SP-MoE uses routing for prefetch, MoE-Spec uses routing for verification budgeting, MoE-SpeQ uses quantized-target similarity (not routing per se). Jakiro adds MoE to the draft heads. Nobody publishes "EAGLE-MoE" as I've sketched it. If the paper presents +1 token in acceptance length on V4-class over EAGLE-3 by adding the routing fingerprint, that's a publishable ablation.

2. **Practical Arc speedup over V4 native MTP is bounded.** MTP gives 1.74× free. EAGLE-MoE plausibly stacks to 2.4×–2.7×. That's a 40–55% improvement, not a 3× improvement. It's worth shipping but it isn't the marquee Arc result.

3. **The frontier ceiling is held by combining MoE-Spec verification budgeting + EAGLE-MoE draft + Sequoia tree shape.** No single one is a moonshot; the combination is the only path to a clean 3×+ over V4 MTP, and even then verification cost on MoE (β) caps absolute SU below dense EAGLE-3's 6× because MoE verification inherently activates more unique experts.

What I'd *not* claim:

- Not a quality improvement. Standard speculative decoding preserves the target distribution. Routing-aware drafting doesn't change that. If Arc's marketing wants a "quality story" it has to come from reward-shifted draft (orthogonal alignment trick) or from the underlying V4 model itself, not from MoE-aware speculation.

- Not redundant with MTP. V4 MTP is a single-step head; EAGLE-MoE goes γ=3+ deep with autoregression. They genuinely cover different ranges of γ; stacking is sensible.

- Not "easy 5×". Cascade's break-even analysis is real; the verification-cost inflation on MoE caps the *theoretical* speedup well below dense EAGLE-3's headline numbers. Be honest in the README.

---

## Citations (canonical)

- Leviathan, Y., et al. *Fast Inference from Transformers via Speculative Decoding.* ICML 2023. <https://arxiv.org/abs/2211.17192>
- Cai, T., et al. *Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads.* ICML 2024. <https://arxiv.org/abs/2401.10774>
- Li, Y., et al. *EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty.* ICML 2024. <https://arxiv.org/abs/2401.15077>
- Li, Y., et al. *EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test.* NeurIPS 2025. <https://arxiv.org/abs/2503.01840>
- Ankner, Z., et al. *Hydra: Sequentially-Dependent Draft Heads for Medusa Decoding.* COLM 2024. <https://arxiv.org/abs/2402.05109>
- Chen, Z., et al. *Sequoia: Scalable and Robust Speculative Decoding.* NeurIPS 2024. <https://arxiv.org/abs/2402.12374>
- Elhoushi, M., et al. *LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding.* ACL 2024. <https://aclanthology.org/2024.acl-long.681/>
- DeepSeek-AI. *DeepSeek-V3 Technical Report.* 2024. <https://arxiv.org/abs/2412.19437>
- DeepSeek-AI. *DeepSeek-V4 Technical Notes (TAAC 2026).* <https://puiching-memory.github.io/TAAC_2026/papers/deepseek-v4/>
- Hu, H., et al. *Jakiro: Boosting Speculative Decoding with Decoupled Multi-Head via MoE.* ACL 2026. <https://arxiv.org/abs/2502.06282>
- Cohere Engineering Blog. *Why MoE models get more from speculative decoding.* <https://cohere.com/blog/mixture-of-experts-models-get-more-from-speculative-decoding>
- LMSYS. *DeepSeek-V4 on Day 0: SGLang and Miles.* April 2026. <https://www.lmsys.org/blog/2026-04-25-deepseek-v4/>
- *Utility-Driven Speculative Decoding for Mixture-of-Experts* (Cascade). 2025. <https://arxiv.org/abs/2506.20675>
- *SP-MoE: Speculative Decoding and Prefetching for Accelerating MoE-based Model Inference.* 2025. <https://arxiv.org/abs/2510.10302>
- *MoE-Spec: Expert Budgeting for Efficient Speculative Decoding.* 2026. <https://arxiv.org/abs/2602.16052>
- *MoE-SpeQ: Speculative Quantized Decoding with Proactive Expert Prefetching and Offloading for MoE.* 2026. <https://arxiv.org/abs/2511.14102>
- *MoESD: Unveil Speculative Decoding's Potential for Accelerating Sparse MoE.* 2025. <https://arxiv.org/abs/2505.19645>
- *Speculating Experts Accelerates Inference for Mixture-of-Experts.* 2026. <https://arxiv.org/abs/2603.19289>
- *Speculative MoE: Communication Efficient Parallel MoE Inference with Speculative Token and Expert Pre-scheduling.* 2025. <https://arxiv.org/abs/2503.04398>
- Li, R., et al. *Reward-Shifted Speculative Sampling Is An Efficient Test-Time Weak-to-Strong Aligner.* EMNLP 2025. <https://arxiv.org/abs/2508.15044>
- SafeAILab. *EAGLE official implementation.* <https://github.com/SafeAILab/EAGLE>
- FasterDecoding. *Medusa.* <https://github.com/FasterDecoding/Medusa>
- facebookresearch. *LayerSkip.* <https://github.com/facebookresearch/LayerSkip>
- Infini-AI-Lab. *Sequoia.* <https://github.com/Infini-AI-Lab/Sequoia>
- haiduo. *Jakiro.* <https://github.com/haiduo/Jakiro>
- DeepSeek-AI. *DeepSeek-V3 reference implementation.* <https://github.com/deepseek-ai/DeepSeek-V3>
- DeepSeekMoE: *DeepSeekMoE: Towards Ultimate Expert Specialization in MoE Language Models.* 2024. <https://arxiv.org/abs/2401.06066>

## Files referenced in Arc

- `/Users/jish/Documents/GitHub/arc/mistralrs-core/src/pipeline/speculative.rs` — existing dense SpeculativePipeline (967 LoC, no MoE awareness)
- `/Users/jish/Documents/GitHub/arc/mistralrs-core/src/pipeline/mtp_pipeline.rs` — V4 MTP wrapper (767 LoC, Tier A greedy)
- `/Users/jish/Documents/GitHub/arc/research/15_speculative_extensions/` — existing PDFs (Lookahead, MagicDec, Medusa, REST, Sequoia, SpecInfer, Triforce) — none MoE-specific
