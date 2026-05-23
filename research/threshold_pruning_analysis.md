# Inference-Time Threshold & Magnitude Pruning for FFN Activations

**Scope:** Pure inference-time methods — no retraining, no learned predictor, no fine-tuning required. Statistical calibration on a small dataset is allowed (most methods need 10-128 sequences) but the LLM weights are untouched.

**TL;DR:**
- The honest training-free ceiling for SwiGLU LLMs (Llama-3, Mistral) is **~40-50% model-wide hidden-state sparsity** with <2pt downstream-task drop, yielding **1.5-1.8× decode speedup** (TEAL, Liu et al. ICLR 2025).
- Going past 50% on SwiGLU models without retraining causes a perplexity cliff — Llama-3-8B PPL jumps from 5.87 → 6.67 at 50% but balloons to 9.06 at 65% ([TEAL Table 1](https://arxiv.org/html/2408.14690v1)).
- Methods that claim ≥85% sparsity (ProSparse, TurboSparse, ReLU Strikes Back) **all require ≥30B-token retraining**. They are *not* training-free.
- Top-k (Spark Transformer) and channel-thresholding (CHESS) are minor variations on the same Pareto curve.
- The factor that determines headroom for Arc isn't the threshold method — it's the **batch size** and the **W_down vs W_gate vs W_up** distribution. Single-batch decode is the sweet spot.

---

## A. Magnitude Thresholding

### A.1 The training-free SOTA: TEAL

The right anchor is **TEAL** ([Liu, Wang et al., ICLR 2025, arXiv:2408.14690](https://arxiv.org/html/2408.14690v1)) because it is the only widely-cited method that genuinely requires zero retraining and reports complete sparsity-quality numbers on modern SwiGLU stacks.

**Method.** For every linear input (hidden state of Q/K/V/O/gate/up/down projections), TEAL:
1. Calibrates an empirical CDF of `|x_i|` per-tensor using **10 sequences of length 2048 from C4**.
2. At inference, picks a per-tensor target sparsity `p` (from a block-wise greedy allocation), looks up threshold `t_p` from the CDF, and zeros entries where `|x_i| ≤ t_p`.
3. The downstream GEMV skips the corresponding *weight column*, saving bandwidth.

**Thresholds are static (set once at calibration) and per-tensor**, not per-head, not per-token, not input-dependent.

**Real sparsity-vs-PPL numbers** (WikiText-2 perplexity, single-batch decode) ([TEAL Table 1](https://arxiv.org/html/2408.14690v1)):

| Model | Base | 25% | 40% | 50% | 65% |
|---|---|---|---|---|---|
| Llama-2-7B | 5.07 | 5.09 | 5.22 | 5.43 | (cliff) |
| Llama-3-8B | 5.87 | 5.94 | 6.21 | 6.67 | **9.06** |
| Llama-2-70B | 3.12 | — | — | 3.50 | 4.28 |

**Downstream-task averages** (MMLU / ARC / HellaSwag / GSM8K / PiQA / Winogrande, Llama-3-8B): 68.07 base → 67.73 at 25% → 66.21 at 40% → 63.42 at 50% → 52.59 at 65%.

**Per-tensor sparsity at 50% model-wide target** (Llama-3-70B, [TEAL §4 / Fig 7](https://arxiv.org/html/2408.14690v1)):
- `W_down` (FFN output proj): **~60% sparsity** — most sparsifiable
- `W_gate`: moderate (~50%)
- `W_up`: ~35-40% — least sparsifiable, because its input is the residual stream and is approximately Gaussian rather than Laplacian
- `W_q`, `W_k`: high at block 0 (70%+), declining sharply with depth
- `W_o`: 50-60% baseline, peaks 80-90% mid-model

The reason `W_down` is the most sparsifiable: its input is `SiLU(gate(x)) * up(x)`, and `SiLU(z) ≈ 0` for `z << 0`, so half of the FFN-intermediate is already near-zero by construction. Thresholding finds the long tail cheaply.

### A.2 What about static, per-layer-only thresholds (CATS / CHESS)?

**CATS** ([Lee et al., Stanford, arXiv:2404.08763](https://arxiv.org/html/2404.08763v1)):
- Applies magnitude thresholding to the **gate output** only (`SiLU(W_gate x)`), not to all hidden states.
- At 50% MLP-layer sparsity on Mistral-7B and Llama-2-7B: **~1-2% downstream drop without fine-tuning**, ~15% end-to-end latency improvement.
- At 70% sparsity: needs ~500 LoRA fine-tuning steps to close the gap.
- At 90% sparsity: average downstream accuracy **drops from 69.94% → 43.68%** on Mistral — full cliff.

**CHESS** ([He et al., arXiv:2409.01366](https://arxiv.org/abs/2409.01366)):
- Per-channel thresholds in FFN + selective sparsification in attention.
- Up to **1.27× speedup** with lower performance degradation than CATS.
- Granularity (per-channel) helps a bit but the marginal win over TEAL is small.

TEAL outperforms both because (a) it sparsifies all seven linear inputs in the block, not just FFN, and (b) it sparsifies through `x` for `W_up` rather than through `SiLU(gate)` — the gate output has a fat tail; the input `x` is more Gaussian and prunes cleaner.

### A.3 Methods that claim ≥85% — and require retraining

These are **not** training-free and should not be confused with TEAL/CATS:

| Method | Sparsity | Retraining cost | Activation function |
|---|---|---|---|
| ReLU Strikes Back ([Mirzadeh et al., NeurIPS-ENLSP 2023](https://arxiv.org/html/2310.04564)) | 62-90% | 30B + 50B tokens (~3-5% of pretrain) | ReLU |
| ProSparse ([Song et al., COLING 2025](https://arxiv.org/html/2402.13516v7)) | **89.32%** Llama-2-7B | 54.53B tokens (2.73% of pretrain) | FATReLU (shifted-threshold ReLU) |
| TurboSparse ([Song et al., arXiv:2406.05955](https://arxiv.org/html/2406.05955v1)) | **90%** Mistral-7B, **97%** Mixtral-47B-per-expert | dReLU retraining | dReLU |

ProSparse on Llama-2-7B with 89.32% sparsity preserves base quality (MMLU 45.48, GSM8K 12.13, HumanEval 19.42) and delivers up to **4.52× speedup with PowerInfer**. But the 54B-token retraining is the price.

This is the actual gap between training-free and trained: ~40% sparsity training-free vs ~90% with two days of GPU retraining. There's no inference-time-only path to 90% on a SwiGLU base model.

### A.4 Static vs input-dependent thresholds

All the major methods use **static thresholds calibrated once**. Adaptive per-input thresholds (GLASS, [Wang et al., arXiv:2508.14302, Aug 2025](https://arxiv.org/html/2508.14302v1)) combine local (prompt-window) and global (model-intrinsic) importance signals. GLASS is "training-free" but the rank-aggregation adds non-trivial CPU overhead per token, and its biggest claim is robustness on **short-prompt-long-generation** workloads where TEAL's static thresholds drift. Speedup numbers are not reported in the GLASS paper — only perplexity/KL — which is suspicious for a paper claiming inference acceleration.

For Arc's target workload (decode-heavy on single-batch), TEAL's static thresholds are the right baseline.

---

## B. Top-k FFN Selection

Top-k by absolute value of the FFN intermediate is the natural alternative to thresholding. The relationship to thresholding is dual: a static threshold approximately implements a static top-k *in expectation*, but per-token top-k gives a guarantee on neuron count.

**Spark Transformer** ([Wang et al., arXiv:2506.06644, June 2025](https://arxiv.org/pdf/2506.06644)) implements top-k masking in both FFN and attention. Spark needs pretraining (it changes the model's training dynamics), so it's not directly applicable as an inference-time tool — but it confirms that top-k masking is *information-equivalent* to magnitude thresholding when k is matched to the average # of survivors. The advantage of top-k is **predictable compute** (constant work per token); the disadvantage is **per-token sort/selection overhead** (typically a partial radix or bitonic top-k, ~10-30μs on a 4096-d vector on H100/B200).

For Arc's V4-class model with `intermediate = 4096` per expert:
- A bitonic top-1024 (25% sparsity) on a 4096 vector is ~5-15μs on B200 — non-trivial vs ~50μs per FFN forward at BF16.
- A static threshold compare-and-mask is ~1-2μs — clearly preferred unless we need the guarantee.

**Verdict: top-k is not worth the overhead vs magnitude threshold for Arc.** The exception is MoE routing, where per-token expert selection is already a top-k operation and adding intra-expert top-k is free.

---

## C. Top-p (Nucleus) on FFN

I found **no paper** that applies nucleus-style (cumulative-mass) thresholding to FFN activations as a published method. The closest analog is GLASS's "rank-aggregation," but it ranks by importance score, not cumulative mass.

Why hasn't anyone tried it? Two reasons:
1. **Cost.** Nucleus requires a sort over the FFN intermediate, then a prefix sum. That's `O(n log n)` per token per layer — strictly worse than top-k.
2. **No quality gain.** Cumulative-mass thresholding selects roughly the same set as top-k or magnitude threshold when the distribution is heavy-tailed (which FFN activations are). The empirical CDF used by TEAL already captures the cumulative-mass intuition statically.

If Arc wants to try it, the cheap implementation is: precompute the cumulative-mass curve once at calibration → emit a target nuclear-mass `p` → look up the corresponding threshold `t_p`. This is *exactly TEAL* with a different way of choosing `t_p`. There's no orthogonal quality gain.

---

## D. Pattern Analysis on Real Models

**OPT-30B (ReLU)** has ~80% FFN sparsity intrinsically. In some layers of OPT-66B, **>70% of FFN neurons are completely dead** — they never activate on any input ([Voita et al., 2023, "Neurons in LLMs"](https://lena-voita.github.io/posts/neurons_in_llms_dead_ngram_positional.html)).

**SwiGLU models** (Llama-3, Mistral) have **near-zero intrinsic sparsity** at activation = 0. The SiLU function is smooth and never produces exact zeros. But the magnitude distribution is Laplacian-with-tails:

- ~50% of `W_down` inputs have `|x| < 0.15` on Mistral-7B ([CATS Fig 1](https://arxiv.org/html/2404.08763v1)).
- TurboSparse (after retraining to dReLU) achieves 90% inactive per layer on Mistral-7B → confirms the distribution has heavy mass near zero, the smooth gate just doesn't snap to it.

**Hot/cold neurons (PowerInfer).** Of MLP neurons accounting for **80% of total activations** ([PowerInfer §3, arXiv:2312.12456](https://arxiv.org/html/2312.12456v2)):
- OPT-30B (ReLU): **26%** of neurons are "hot"
- Llama-2-70B (ReGLU): 43%
- **Llama-2-70B (SwiGLU): 69%** — SwiGLU spreads activations across most neurons, killing the hot/cold dichotomy that makes PowerInfer's CPU/GPU split work.

This is the key result: **PowerInfer's reported 11× speedup is gated on ReLU-family activations.** On native SwiGLU Llama/Mistral, you get ~2× max from a pure activation-sparsity approach, period.

---

## E. Combined with Weight Quantization

This is the question Arc actually cares about: does TEAL stack with QTIP 2-bit?

**Direct evidence from TEAL Fig 4 + §5 ([arXiv:2408.14690](https://arxiv.org/html/2408.14690v1)):**
- TEAL was tested compounded with 8-bit RTN, 4-bit AWQ, and **2/3-bit QuIP#**.
- "Errors from activation sparsity and weight quantization compound somewhat independently" — quality curves are roughly parallel.
- Specialized sparse+quantized kernels are **future work** — TEAL's own kernel uses dense quantized weights with sparse column skipping.

**Independent confirmation:** [arXiv:2511.04477 "Enabling Dynamic Sparsity in Quantized LLM Inference"](https://arxiv.org/pdf/2511.04477) reports **2-4× combined speedup** vs dense FP16 when dynamic sparsity is layered on INT4, with "minimal perplexity degradation."

**Theoretical caveat:** QuIP-style trellis quantization shifts the weight distribution, so the threshold calibrated on FP16 weights may need re-calibration on quantized weights. TEAL does the calibration *on activations*, not weights — so this is approximately invariant. But for trellis schemes that distort the residual stream (e.g., random Hadamard rotations before quantization), re-calibration is mandatory.

**Practical answer for Arc:** Yes, TEAL stacks with QTIP. Recalibrate the per-tensor thresholds after the Hadamard rotation, **on 10-32 short C4 sequences post-rotation**. Expect compounding to ~1.5×_sparse × ~2×_quant = ~3× total, not the multiplicative 1.8 × 4 = 7× theoretical max — because the bandwidth savings overlap.

---

## F. Theoretical Quality Bound

There is no clean theoretical bound. The closest results are:

- **Lazy Neuron Phenomenon** ([Li et al., 2022, arXiv:2210.06313](https://arxiv.org/pdf/2210.06313)): Transformer FFN neurons follow a power-law activation distribution where the top-k neurons explain most of the output. The implied bound: top-k sparsity is achievable at a quality cost ~`O(1/k^α)` where `α` depends on the architecture's tail index.
- **Information-theoretic floor:** For attention sinks and routing tokens, you cannot sparsify below the entropy of the routing distribution. Empirically this floor is at ~10-15% kept neurons for the W_down ([TEAL §6 failure modes](https://arxiv.org/html/2408.14690v1)).

**Empirical "1% quality drop" thresholds for modern models** (all training-free, single-batch):
- Llama-2-7B: **40% sparsity** → 0.4 PPL increase, ~1% downstream avg drop
- Llama-3-8B: **25% sparsity** → 0.07 PPL increase, 0.3% downstream drop (Llama-3 is markedly less compressible)
- Mistral-7B: **40-50% sparsity** is the 1% threshold
- Llama-2-70B: **50% sparsity** before 1% drop

Llama-3 is the harder model. The TEAL authors observe that "quantization techniques are less effective on newer models" — same dynamic applies to sparsity. Llama-3 was trained on more tokens per parameter, and the result is denser weight utilization with less redundancy.

---

## Pareto Frontier Summary

The honest, training-free Pareto frontier across SwiGLU models, single-batch decode:

```
sparsity   |  ΔPPL    |  Δdownstream  |  decode speedup  |  recommended
-----------|----------|---------------|------------------|---------------
0%         |  0       |  0            |  1.00×            |  baseline
25%        |  +0.01-0.07 | -0.3%      |  1.20×            |  safe everywhere
40%        |  +0.15-0.35 | -1 to -2%  |  1.53×            |  the sweet spot
50%        |  +0.4-0.8  | -3 to -5%   |  1.80×            |  acceptable for chat/RAG
65%        |  +1.5-3.2  | -10 to -15% |  ~2.0× (limited)  |  quality cliff — avoid
80%+       | (cliff)    | -30%+       |  (irrelevant)     |  requires retraining
```

This frontier is robust across TEAL, CATS, CHESS, and GLASS — they trade <2% on each other inside the green zone.

---

## Per-Model Behavior

| Model family | Activation | 1%-drop ceiling | 5%-drop ceiling | Cliff |
|---|---|---|---|---|
| Llama-2-7B | SwiGLU | 40% | 50% | ~60% |
| Llama-2-70B | SwiGLU | 50% | 60% | 65% |
| Llama-3-8B | SwiGLU | 25-30% | 40-45% | 55% |
| Mistral-7B | SwiGLU | 40-50% | 55% | 65% |
| Mixtral-8x7B | SwiGLU + MoE | per-expert: 50% | per-expert: 60% | per-expert: 70% |
| OPT-30B | ReLU | ~80% intrinsic | n/a | n/a |
| Phi-3 | SwiGLU (deep+narrow) | ~30% (less studied) | ~40% | ~50% |

**MoE caveat:** Mixtral and DeepSeek already do per-token top-k expert routing (top-2 of 8 for Mixtral, top-8 of 256 for DeepSeek-V3). Adding intra-expert activation sparsity stacks: TurboSparse-Mixtral-47B claims 97% intra-expert sparsity after retraining; **training-free**, expect ~50% intra-expert with the same quality budget as a dense model.

---

## Decode-Time Implementation Cost

For Arc on a B200 single-batch decode, per-layer cost of applying TEAL-style thresholding:

| Operation | Time per FFN block | Notes |
|---|---|---|
| `abs(x) > t` mask | ~0.5μs | fused into the matmul launch |
| Column skip in W_down GEMV | "free" | bandwidth saved is the speedup |
| Threshold storage | 4 floats × 32 layers × 7 tensors = 896 bytes | trivial |
| Calibration (one-time) | ~30 seconds on 10×2048 C4 sequences | acceptable |
| Per-layer CPU overhead | **~0 CPU cycles** | thresholds are CUDA constants |

Compare to a learned-predictor approach (Deja Vu): ~20-50μs of predictor MLP per token per layer, eating most of the speedup on small batch. **Static threshold is the only viable path for sub-1ms-per-token decode.**

The TEAL kernel ([github.com/FasterDecoding/TEAL](https://github.com/FasterDecoding/TEAL)) uses column-major storage of weights and SplitK atomics; on A6000 it's 1.53× at 40% sparsity. On A100 the dense torch.compile baseline is closer, so the relative speedup is smaller; expect B200 to behave like A6000 or better due to HBM3e bandwidth headroom.

**Failure mode at B200 scale:** if the GEMV is already compute-bound (large model, low context), sparsity doesn't help. Decode at small batch (1-4) on a memory-bound model is the only regime where TEAL pays off — which is exactly Arc's target.

---

## Specific Recommendations for Arc

1. **Adopt TEAL-style static per-tensor magnitude thresholding** as the default, calibrated on 10 C4 sequences post-QTIP-Hadamard. Stage 1 deliverable.

2. **Target 40% model-wide sparsity** as the default. This is the proven 1.53× single-batch speedup with <2% downstream drop on every SwiGLU model. Expose as `--ffn-sparsity 0.40`.

3. **Per-tensor sparsity allocation** (greedy from TEAL Algorithm 1):
   - `W_down`: 60% (Laplacian input from `SiLU·up`)
   - `W_gate`: 40%
   - `W_up`: 30% (Gaussian input from residual)
   - `W_q`, `W_k`, `W_v`: 30-40% (depth-dependent — high in early blocks, low in late)
   - `W_o`: 50%

4. **Skip sparsifying** the LM head and the first 2 transformer blocks. The LM head is too small to benefit, and early blocks contain attention-sink behavior that breaks under sparsification ([TEAL §6](https://arxiv.org/html/2408.14690v1)).

5. **Recalibrate per quantization scheme.** When stacking with QTIP/QuIP# 2-bit, the post-Hadamard activation distribution differs from FP16 by a non-trivial rotation. Recalibrate per quantization config.

6. **Single-batch only.** At batch ≥4, TEAL's effective sparsity drops by ~22pt (W_down 60% → 38%); the speedup vanishes. **Disable sparsification when batch > 1** unless we add per-sample mask handling, which is non-trivial.

7. **Stack with QTIP 2-bit.** Expected combined decode throughput: ~3× over dense BF16 baseline. Toward Arc's 120 tok/s BF16 target on B200, this is the realistic multiplier from this dimension. Do not double-count.

8. **Don't promise 4-5× from this dimension alone.** That number comes from ProSparse + PowerInfer with 30B-token retraining. Without retraining, the ceiling is ~1.8×.

---

## Failure Modes

1. **Long-generation drift** ([GLASS §2](https://arxiv.org/html/2508.14302v1)): static thresholds calibrated on prompt-like distributions degrade when generation runs hundreds of tokens. **Mitigation:** recalibrate every N tokens, or use GLASS-style local+global aggregation if quality regression appears.

2. **Attention sinks** ([TEAL §6](https://arxiv.org/html/2408.14690v1)): the first few tokens carry outsized attention mass. Sparsifying prefill on initial tokens breaks the sink, producing perplexity spikes. **Mitigation:** only sparsify the second half of prefill, or skip the first 4 tokens entirely.

3. **Batched inference** ([TEAL §5.2](https://arxiv.org/html/2408.14690v1)): per-sample masks union to a dense mask at batch ≥ 4, killing the bandwidth win. **Mitigation:** disable for batch > 1, or accept lower per-tensor sparsity (38% for W_down at batch 4 vs 60% at batch 1).

4. **Llama-3 / dense newer models:** the 1% ceiling drops to 25% from Llama-2's 40%. **Mitigation:** model-specific threshold tables. Don't assume "40% is safe" — verify per model.

5. **MoE per-expert sparsification breaks load balancing.** Routers expect roughly equal activation count per expert; aggressive intra-expert sparsity can flip the router's selections. **Mitigation:** apply sparsity *after* routing decisions are committed.

6. **Recalibration drift.** If you change the calibration set (C4 vs RedPajama vs domain data), thresholds shift by 5-20%. **Mitigation:** ship the calibration dataset hash alongside the threshold table.

7. **Sparse kernel overhead on small models / large GPU.** On Llama-2-7B at A100/B200 with `torch.compile`'d dense GEMV, the dense baseline is so fast that TEAL's sparse kernel only wins at ≥50% sparsity. **Mitigation:** benchmark per (model, GPU, batch) before enabling by default.

---

## Bottom Line for Arc

Activation thresholding is a real **1.5-1.8× single-batch decode lever** that stacks with QTIP 2-bit, costs ~1ms of integration time per model, and has no training requirement. It is not the path to 5×. The numbers that justify 5× all require retraining the activation function (ReLU/FATReLU/dReLU), which is out of scope for "pure inference-time."

**Expected stacked contribution to the 120 tok/s B200 target:** ~1.5-1.7× on decode, on top of QTIP weight compression and CUDA graph autonomy. Treat this as a stage-3 optimization once the CUDA-graph decode loop is stable, not a stage-1 priority.

---

## Sources

- [TEAL: Training-Free Activation Sparsity in LLMs (ICLR 2025, arXiv:2408.14690)](https://arxiv.org/html/2408.14690v1) — primary anchor for training-free numbers
- [TEAL blog post (together.ai)](https://www.together.ai/blog/teal-training-free-activation-sparsity-in-large-language-models)
- [CATS: Contextually-Aware Thresholding (Stanford, arXiv:2404.08763)](https://arxiv.org/html/2404.08763v1)
- [CATS blog (Scaling Intelligence Lab, Stanford)](https://scalingintelligence.stanford.edu/blogs/cats/)
- [CHESS: Channel-Wise Thresholding (arXiv:2409.01366)](https://arxiv.org/abs/2409.01366)
- [Deja Vu (Liu et al., ICML 2023, arXiv:2310.17157)](https://arxiv.org/abs/2310.17157) — learned predictor, not training-free
- [PowerInfer (Song et al., SOSP 2024, arXiv:2312.12456)](https://arxiv.org/html/2312.12456v2)
- [ProSparse (Song et al., COLING 2025, arXiv:2402.13516)](https://arxiv.org/html/2402.13516v7) — requires retraining
- [TurboSparse / dReLU (Song et al., arXiv:2406.05955)](https://arxiv.org/html/2406.05955v1) — requires retraining
- [ReLU Strikes Back (Mirzadeh et al., NeurIPS-ENLSP 2023, arXiv:2310.04564)](https://arxiv.org/html/2310.04564) — requires retraining
- [Spark Transformer (arXiv:2506.06644)](https://arxiv.org/pdf/2506.06644) — requires pretraining
- [GLASS: Global-Local Aggregation (arXiv:2508.14302)](https://arxiv.org/html/2508.14302v1) — training-free, adaptive thresholds
- [SparseInfer: Training-Free Prediction of Activation Sparsity (arXiv:2411.12692)](https://arxiv.org/pdf/2411.12692)
- [Neurons in LLMs: Dead, N-gram, Positional (Voita et al., ACL 2024)](https://lena-voita.github.io/posts/neurons_in_llms_dead_ngram_positional.html)
- [The Lazy Neuron Phenomenon (arXiv:2210.06313)](https://arxiv.org/pdf/2210.06313)
- [LExI: Layer-Adaptive Active Experts for MoE (arXiv:2509.02753)](https://arxiv.org/abs/2509.02753)
- [Enabling Dynamic Sparsity in Quantized LLM Inference (arXiv:2511.04477)](https://arxiv.org/pdf/2511.04477)
- [TEAL GitHub: github.com/FasterDecoding/TEAL](https://github.com/FasterDecoding/TEAL)
- [Deja Vu GitHub: github.com/FMInference/DejaVu](https://github.com/FMInference/DejaVu)
