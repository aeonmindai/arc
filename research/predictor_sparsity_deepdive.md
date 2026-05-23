# Predictor-Based Inference-Time Sparsity — Deep Dive

**Scope.** PowerInfer (SOSP'24), Déjà Vu (ICML'23), CATS (COLM'24), plus the
training-free siblings (TEAL, SparseInfer). Question: **does a tiny learned
router save more compute than it costs on a V4-class MoE, and what's the
training cost?**

Headline:

> Predictor-based sparsity on **dense ReLU/SiLU LLMs** is well-trodden: ~2×
> wall-clock speedup, 5–10% predictor parameter tax, one-time training cost
> ~24-200 A100-hours from a 400k-sample calibration set ($30–340 cloud).
> Predictor-based sparsity on **MoE V4-class models** is essentially
> unpublished. The closest analogues (SP-MoE, MoE-SpeQ) predict which
> **experts** fire, not which **neurons within those experts** fire. There is
> a clean novel angle: a routing-conditional predictor sized at ~0.05% of
> the model exploiting top-k routing as a free side-channel.

---

## 1. Predictor cost-benefit table

| Method | Predictor architecture | Predictor size | Training data | Train cost | Sparsity | Quality cost | Predictor overhead | Models |
|---|---|---|---|---|---|---|---|---|
| **Déjà Vu** | 2-layer MLP, low-rank D=1000, one per layer, separate MLP- and attention-head predictors | OPT-175B: ~5.84B params (≈3.3% of LLM); OPT-30B: 0.25B (≈0.8%) | 400,000 (hidden state, label) pairs **per layer** collected by running OPT on C4 | ~28 GPU-h on 8×A100 (data collection + training); ~$340 cloud | Up to **75% MLP + attention sparsity** without accuracy drop (OPT-175B) | ≤0 perplexity drop on WikiText/C4 at 75%, larger drop ≥85% | "Low-cost" — quantified by paper as small fraction of saved compute (2× speedup net) | OPT-30/66/175B only; needs ReLU-ish activations |
| **PowerInfer** | 1-hidden-layer MLP per layer, **adaptively sized** per layer's skewness (target ≥95% recall) | 6.71% (OPT-13B) → 8.68% (Falcon-40B) → 8.08% (Llama-70B-ReGLU) of base model params; ~$5–10 GB fp16 at 70B | C4 + Wikipedia activations; not quantified beyond "substantial amount" | "Several hours" multi-GPU; pre-trained predictors shipped on HF for ReluLLaMA/Falcon | **~80–95% MLP-neuron sparsity** by load (hot 17–26% of neurons carry 80% of activation mass) | Negligible: OPT-175B PIQA 79.26 vs 79.65 baseline, Winogrande 72.38 vs 72.77 | <10% of total inference time (Fig. 17), avg 5–8% | **ReLU/ReGLU/Squared-ReLU only** — NOT Mistral/Llama/Qwen with SiLU |
| **CATS** | **No learned predictor.** Per-layer percentile threshold over training-data activation magnitudes | 0 extra params (just one fp32 threshold per MLP layer) | "Random subset of training data" for threshold; optional 500-step fine-tune on 1% of params (Q/K + W_gate + W_down) | Threshold calibration: ~15 min on 8×L40S (~$3). Optional fine-tune: ~6h on 8×L40S (~$60) | 50% MLP sparsity at near-zero cost; 70% with mild degrade; 90% breaks | Mistral-7B: 1.5% avg drop on 8 zero-shot tasks **without fine-tune** at 50% | Threshold compare is ~free (single ABS + compare per element) | Mistral-7B, Llama2-7B (SiLU/SwiGLU) |
| **TEAL** (training-free, ICLR'25) | None — magnitude-threshold on activations | 0 | "Distributional shape fitting" on a calibration set (Gaussian/Laplacian shapes); no training | Effectively free | 40–50% **model-wide** sparsity | Near-zero degrade at 25%, minimal at 40% on Llama-2/3, Mistral 7B–70B | Free (threshold compare) | Llama 2/3, Mistral, SiLU family |
| **SparseInfer** (training-free, '24) | None — sign-bit XOR predictor | 0 (337 MB activation cache vs PowerInfer's 1480 MB) | None | Free | Matches PowerInfer sparsity rates within 1% accuracy | 1.5–2% degrade at 50% | 3.66× faster than PowerInfer's predictor (2.2M ops/layer vs 19.4M) | ReLU-fied Llama/Falcon/OPT |

**Takeaway.** Training-free methods (CATS, TEAL, SparseInfer) match or beat
trained-predictor methods on quality-vs-sparsity at the same hardware tier.
The case for trained predictors is mostly historical and ergonomic
(predictor weights ship with the model).

---

## 2. Calibration data requirements quantified

### Déjà Vu (the canonical predictor)

From `sparse_predictor/main_mlp.py`:

```python
CONFIG = {'175b': {'num_layer': 95, 'd': 12288, 'h': 96, 'N': 400000}, ...}
```

- **400,000 (hidden, label) pairs per layer**, collected on C4. Per layer
  that's 400k × (d + 4d) × 2 bytes; for OPT-175B ≈ 49 GB × 95 layers ≈
  4.6 TB of activation traces stored as fp16 mmap files
  (`mlp_x_{l}.mmap`, `mlp_label_{l}.mmap`).
- 400k samples ≈ 400k tokens — minuscule vs OPT's 180 B-token training
  corpus.
- **20 epochs, batch 1024, AdamW lr=1e-3 wd=0.01,** early-stop on recall
  ≥0.99 (`trainer_mlp.py:74-138`).
- Layers train in parallel: `run_c4_mlp.sh` runs 8 GPUs × 1 layer each, 12
  waves for OPT-175B.

End-to-end on 8×A100: ~1 day, $200–400 cloud.

### PowerInfer

Paper says "C4 and Wikipedia" activations, "several hours" of training. The
README lists "Release predictor training code" as a TODO; only pre-trained
predictors ship on HF. Reverse-engineering the size from Table 6:
Llama-70B-ReGLU 8.08% × 70B = 5.66B params, 80 layers × (d·D + D·FFN) with
d=8192, FFN=28672 → **D ≈ 2000**. So PowerInfer ≈ Déjà Vu + adaptive D per
layer + hot/cold ILP solver.

### CATS

Threshold is empirical-CDF inverse: *t = min{t' : F(t') ≥ k}* over
|activation| on a RefinedWeb subset (~1M tokens, 3 OOM smaller than
ProSparse's 34.6B / Turbo Sparse B's 150B). Optional fine-tune: 500 steps on
1% of params (Q/K + W_gate + W_down) on 8 L40S in BF16/DeepSpeed, ~$60.

### TEAL / SparseInfer

Few-thousand-sample fit of Gaussian/Laplacian shapes or sign-bit
statistics; <1 min on a single A100, negligible cost.

### Comparison vs Turbo Sparse Part B's 150B-token retraining

| Method | Tokens | A100-h | $ cloud | Ratio vs Turbo B |
|---|---:|---:|---:|---:|
| TEAL / SparseInfer | <1 M | <0.1 | <$0.20 | **1 : 100,000+** |
| CATS threshold calibration | ~1 M | ~0.5 | ~$1 | **1 : 21,000** |
| CATS + 500-step fine-tune | ~1 M + 500 steps | ~48 | ~$60 | **1 : 350** |
| Déjà Vu OPT-175B predictor | 400k traces × 95 layers | ~200 | ~$340 | **1 : 62** |
| PowerInfer (per paper) | C4+Wiki, "substantial" | ~24-100 | ~$36-150 | **1 : 140-600** |
| **Turbo Sparse Part B** | **150 B** | **~14 000** | **~$21 000** | **1 : 1** |

The trained-predictor methods are **62–600× cheaper** than full Turbo-Sparse
ReLUfication; the training-free methods are **another ~100× cheaper still**.

---

## 3. Predictor inference overhead — when does it pay for itself?

Reading off `main_mlp.py`'s architecture (Linear(d→D), Linear(D→4d)), each
predictor call costs:

> FLOPs_predictor = 2 · (d·D + D·4d) = 10 · d · D

For Llama-70B (d=8192, D=2000): 10·8192·2000 = **164 MFLOPs/layer/token**
For a full FFN forward (gate + up + down): 6 · d · 4d = 24·d² = **1.61 GFLOPs/layer/token**
Predictor is ~**10% of FFN compute**. At 50% sparsity the FFN saves 805 MFLOPs,
so the net win is 805 − 164 = **641 MFLOPs/layer/token (≈40% speedup)**.

**Break-even sparsity:**
> sparsity_breakeven = D / (4d) (the predictor outputs 4d masks at cost D·4d)

For Llama-70B with D=2000, breakeven is 2000/32768 = **6.1% sparsity** — any
sparsity above 6% pays for the predictor itself.

For Déjà Vu's reported 75% sparsity → ~10× FFN cost reduction net of
predictor. Paper reports 2× wall-clock; the gap is memory-bandwidth limits
and that attention isn't sparsified the same way.

**SparseInfer's claim that PowerInfer's predictor uses 19.4M ops vs 2.2M for
sign-bit XOR** suggests the trained predictor is the bottleneck on
memory-bound workloads. **At V4-class scale where the FFN per active expert
is small (2048-dim intermediate), a trained predictor's overhead becomes a
larger fraction of expert compute.**

---

## 4. MoE-specific opportunities — what predictors can do on V4-class models

V4-class architecture (per public DeepSeek V3 + reported V4 deltas):
hidden=7168, **256 routed experts × top-8 (V3) or top-16 (V4)**, expert
intermediate=2048, 58 MoE layers, 1 shared expert. The routing decision
already gives you which 8 (of 256) experts fire — that's ~3% expert-level
sparsity *for free*.

The novel question: **given the routing decision {e₁, …, e_k}, can a tiny
predictor cheaply identify which down-proj rows in those k experts matter?**

### Option A: Per-expert independent predictor (naïve Déjà-Vu transplant)

- One predictor per (layer, expert) → 256 × 58 = **14,848 predictors**
- Each: 7168→128→2048, ~1.18 M params, ~2.36 MB fp16
- Total: **17.5 B params ≈ 35 GB fp16 ≈ 2.6% of full 671B model**
- Per token: only 8 predictors fire → 8 × 1.18M × 2 = **18.9 MFLOPs/layer**
  vs 8 × 2 × 7168 × 2048 = **470 MFLOPs/layer** of expert FFN compute
- **Predictor overhead: 4% of expert compute.** Breakeven at 4% sparsity.

This is the "obvious" port. It works but each expert seeing only its own
slice means the predictor can't exploit cross-expert structure.

### Option B: Shared routing-conditional predictor (novel)

- **One predictor per layer**, takes `[hidden_state ‖ routing_mask]` as input
  and outputs an `8·2048 = 16384`-dim sparse-neuron mask, scattered to the
  active experts' indices.
- Input dim = 7168 + 256 = 7424, low-rank D=256, output 16384
- Per-layer params: `7424·256 + 256·16384 = 6.09 M`
- **Total: 6.09 M × 58 layers = 353 M params ≈ 707 MB fp16 ≈ 0.05% of V3**
- Per-token compute: ~24 MFLOPs vs 470 MFLOPs expert FFN → **5% overhead**
- The routing-conditional design lets the predictor learn **expert
  interactions** — "if experts {17, 42, 91} fire, then expert 17's neurons
  {100–500} are likely co-active." Each expert pair has its own learned
  joint sparsity pattern.

### Option C: Reuse the router itself

Even cheaper: **the affinity scores from the top-k gate are already a
soft-importance signal.** The hypothesis: rank the 8 active experts by gate
weight; for the bottom-half experts (low weight), drop the 50% lowest-mag
down-proj rows. This is **predictor-free** — it uses signal V4 already
computes for routing — but is **MoE-specific** and has not been published as
far as we found.

### Has this been done?

We checked the recent MoE-sparsity literature (SP-MoE, MoE-SpeQ, ExpertFlow,
Speculative MoE, Mixture of Neuron Experts arXiv 2510.05781). All current
work predicts **which experts fire** (for prefetch / offload), not **which
neurons within the firing experts fire**. Mixture of Neuron Experts is the
closest — it does neuron-level deactivation inside MoE but the paper's
description suggests it's still per-expert, not routing-conditional. **This
gap is a clean novel target for Arc.**

---

## 5. Concrete recommendation for Arc

For a **V4-class deployment (671B-class MoE, SiLU/SwiGLU experts):**

**Tier 1 — ship first, no predictor training.**

- Implement **TEAL-style per-expert magnitude threshold** at 25-40% within-
  expert sparsity. Threshold per (layer, expert) fit on a 1M-token
  calibration of the deployment domain. Cost: **~1 A100-hour, $2**, no
  training. Wall-clock benefit: ~1.4× speedup on FFN compute, which is
  bottlenecked on activated-expert memory bandwidth.
- This is a strict superset of the V4 native routing sparsity (256→8) — we
  add within-expert sparsity on top of cross-expert sparsity.

**Tier 2 — when Tier 1 stops giving headroom, train Option B predictor.**

- Single routing-conditional predictor per layer: 353 M params total, 707 MB
  fp16. Trains in **~8–16 GPU-hours on 4×H100** using 400k calibration
  samples — same recipe as Déjà Vu but on aggregated MoE traces.
- Cost estimate: **~$50–120 in cloud GPU time** (4 H100s × 16h @ $2.50/h).
- Expected sparsity: 50–70% within-expert + 97% cross-expert = effective
  98%+ FFN sparsity total.
- Predictor overhead: ~5% of expert compute → breakeven trivially met.

**What Arc gets that's novel.**

1. **Routing-conditional predictor** — exploits the cross-expert structure
   that single-expert predictors can't see. Cleanly publishable.
2. **Predictor co-design with TurboQuant codebooks** — the predictor's
   output mask can be used to gate codebook lookups *before* dequant,
   reducing memory traffic. We'd need to verify the kernel fusion.
3. **Calibration data domain-mixing** — for Runcrate (Arc's target
   workload), calibrating the predictor on Runcrate traces should give
   strictly better in-domain quality than the C4-trained Déjà Vu
   approach.

**What Arc should *not* spend time on.**

- Reproducing PowerInfer end-to-end: ReLU-only, doesn't help SiLU/SwiGLU
  V4-class models.
- Training-from-scratch 150B-token Turbo Sparse-style ReLUfication: 100×
  more expensive than Option B and there's no evidence it's needed if
  CATS/TEAL achieve 40% sparsity at near-zero quality cost.

---

## 6. Code references — predictor train + inference loops

### Déjà Vu (FMInference/DejaVu, branch `master`)

| File | What's in it |
|---|---|
| `sparse_predictor/main_mlp.py:101-115` | Predictor architecture: `nn.Sequential(Linear(d→D), Linear(D→4d))` — that's it, two linear layers, no bias, no activation between them |
| `sparse_predictor/main_mlp.py:11-37` | CONFIG dict with d/h/N per model size |
| `sparse_predictor/main_mlp.py:54-71` | Memmap loader for `mlp_x_{l}.mmap` and `mlp_label_{l}.mmap` |
| `sparse_predictor/trainer_mlp.py:78-138` | Training loop: AdamW lr=1e-3 wd=0.01, weighted BCE, early stop on recall ≥0.99 |
| `sparse_predictor/trainer_mlp.py:14-17` | `generate_label`: any neuron with `y > 0` is positive (ReLU sparsity oracle) |
| `sparse_predictor/run_c4_mlp.sh` | Parallel launcher, 8 GPUs, 1 layer/GPU |
| `Decentralized_FM_alpha/run_infer_opt_175b_collect_sp_data.sh` | Data collection (400k samples per layer) |

### PowerInfer (SJTU-IPADS/PowerInfer)

- Predictor training code **not released** (README TODO).
- Pre-trained predictors on HuggingFace: `PowerInfer/ReluLLaMA-7B-Predictor`,
  `PowerInfer/ReluFalcon-40B-Predictor`, `PowerInfer/Bamboo-base-v0.1`.
- Conversion: `convert.py`, `convert-hf-to-powerinfer-gguf.py` package
  predictor weights into the PowerInfer GGUF format alongside base weights.
- Inference dispatch is in the forked llama.cpp — predictor MLP is run before
  each FFN call and gates which neuron rows get loaded from CPU/RAM.

### CATS (ScalingIntelligence/CATS)

- `experiments/` — threshold calibration scripts
- `flash_gemv/` — the 15%-speedup custom CUDA kernel that fuses threshold +
  GEMV
- `inference.py` — the runtime that does `mask = abs(act) > threshold; act
  *= mask` before W_down GEMM
- `reproduction_script.sh` — end-to-end repro of paper numbers

### TEAL (Together AI)

- Threshold calibration: ~30 lines, computes per-layer activation magnitude
  percentile from a small calibration loader.
- Linked from the Together AI blog and ICLR'25 paper (2408.14690).

---

## 7. References & sources

Primary papers and repos consulted:

- Déjà Vu — Liu et al., ICML 2023 — [arxiv 2310.17157](https://arxiv.org/abs/2310.17157), [code](https://github.com/FMInference/DejaVu)
- PowerInfer — Song et al., SOSP 2024 — [arxiv 2312.12456](https://arxiv.org/abs/2312.12456), [SOSP'24 PDF](https://ipads.se.sjtu.edu.cn/_media/publications/song-sosp24.pdf), [code](https://github.com/SJTU-IPADS/PowerInfer)
- CATS — Lee et al., COLM 2024 — [arxiv 2404.08763](https://arxiv.org/abs/2404.08763), [code](https://github.com/ScalingIntelligence/CATS), [blog](https://scalingintelligence.stanford.edu/blogs/cats/)
- TEAL — Liu et al., ICLR 2025 — [arxiv 2408.14690](https://arxiv.org/abs/2408.14690)
- SparseInfer — 2024 — [arxiv 2411.12692](https://arxiv.org/abs/2411.12692)
- ProSparse — Song et al., COLING 2025 — [arxiv 2402.13516](https://arxiv.org/abs/2402.13516)
- ShadowLLM — [arxiv 2406.16635](https://arxiv.org/abs/2406.16635)
- SP-MoE (expert-prefetch predictor for MoE) — [arxiv 2510.10302](https://arxiv.org/abs/2510.10302)
- MoE-SpeQ — [arxiv 2511.14102](https://arxiv.org/abs/2511.14102)
- Mixture of Neuron Experts — [arxiv 2510.05781](https://arxiv.org/abs/2510.05781)

### Noticed during research — worth a separate change?

> The closest published MoE neuron-sparsity work (Mixture of Neuron Experts,
> 2510.05781) deactivates neurons inside experts but not conditioned on the
> co-firing expert set. A routing-conditional predictor as in §4 Option B
> appears genuinely novel and well-aligned with V4's top-k routing. Worth
> spinning into its own research thread once Tier 1 (TEAL-style) ships.

> The codepath for predictor mask → codebook-lookup gating in TurboQuant
> hasn't been explored anywhere we found. Combining a routing-conditional
> mask with TurboQuant's WHT-based row addressing should let us skip
> codebook dequant entirely for masked-out neurons — needs a quick kernel
> sketch to confirm the addressing math.
