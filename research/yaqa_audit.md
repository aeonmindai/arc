# YAQA Audit — Should Arc Adopt Cornell-RelaxML's YAQA Quantization?

**Date:** 2026-05-23
**Scope:** Pure literature + code survey. No implementation.
**Verdict (TL;DR):** **Skip the kernel port. Port the algorithm only if we need to squeeze more quality out of QTIP 2-bit; treat as an Apprentice-Mode-style offline upgrade to our existing trellis pipeline.**

---

## 1. What YAQA Is

YAQA ("Yet Another Quantization Algorithm") is the post-training-quantization algorithm published as *Model-Preserving Adaptive Rounding* by Albert Tseng, Zhaofeng Sun and Christopher De Sa, Cornell University, arXiv 2505.22988 (May 2025; v2 thread Sept 2025).[^paper][^twitter] It is **not** a new format, **not** a new codebook, and **not** a new inference kernel. It is a smarter **rounding objective** that replaces QTIP's LDLQ pass during quantization with a Kronecker-factored approximation of each linear layer's Hessian taken with respect to the **full-model KL divergence** rather than just that layer's local activation reconstruction error.[^paper][^readme]

Concretely:

- QTIP's LDLQ minimizes `||W·X - Ŵ·X||²` locally (one L matrix from a Cholesky of `X·Xᵀ`).
- YAQA's `LDLQ_2hess` minimizes a Kronecker approximation of `∇²L_KL(model‖model_q)` over the *whole* network's output, expressed as two L factors `Lin` and `Lout` from sketches of the true Hessian.[^ldlq]
- "Sketch A" = layerwise Hessian under token independence assumption + power iteration (var. O(1/T), biased).
- "Sketch B" (recommended) = one round of power iteration on `∇W²L` from identity init, in a single dataset pass.[^hessreadme]

The paper provides the **first end-to-end error bound** for adaptive rounding, with convergence governed by the cosine similarity between the Hessian sketch and the true Hessian.[^paper] Author Tseng claims on X (Sept 2025): *"YAQA is still the only PTQ algorithm that beats QAT, and still the only PTQ algorithm with end to end error bounds."*[^twitter]

The repo itself is the QTIP codebase forked at one point, with the rounding routine replaced and a new `hessian_llama/` directory containing FSDP-based Sketch-A/B Hessian collection scripts; the README explicitly says *"To quantize models, follow the instructions in the QTIP codebase."*[^readme]

---

## 2. Comparison Table: YAQA vs QTIP-paper vs EXL3

| Dimension                  | **YAQA** (Cornell-RelaxML, 2025)             | **QTIP** (Cornell-RelaxML, ICLR 2025)         | **EXL3** (turboderp, 2025)                            |
|----------------------------|----------------------------------------------|-----------------------------------------------|-------------------------------------------------------|
| **Tile shape**             | 16x16 (kernel-tied: `td_x = td_y = 16`)[^ldlq] | 16x16                                         | 16x16 (Tensor-Core-aligned)[^exl3wiki]                |
| **Codebook**               | Identical to QTIP: `bitshift_codebook` with `1mad`/`2mad`/`3inst`/`quantlut`/`quantlut_sym` modes, L=16, K∈{2,3,4}, V∈{1,2}.[^bitshift] | Same — trellis-coded with computed (Gaussian/Hadamard-mapped) LUTs | QTIP-derived; uses `3INST` parameter family (community-tuned variant in PR #26)[^exl3pr] |
| **RHT / incoherence**      | Hadamard rotation, identical `matmul_had.py`[^bitshift] | Hadamard rotation (paper) | Hadamard transform + LDL[^exl3wiki]                   |
| **LDL / rounding**         | **LDLQ_2hess** (two Hessian factors, Kronecker)[^ldlq] | LDLQ (one Cholesky of layerwise `X·Xᵀ`) | LDL-based GPTQ-style                                  |
| **Calibration cost**       | Sketch-A ≈ 30 GPU-hours / 10B params / 20M-token corpus; Sketch-B ≈ <50 GPU-hours / 10B params / 64K×2K-token corpus[^paper] (8×80GB node for ≤20B, 2 nodes for 70B[^hessreadme]) | Hours on a single 8×80GB node (QTIP paper)   | Minutes for small models; LDL+RHT only, no full Hessian sketch |
| **Bitrate floor**          | 2-, 3-, 4-bit demonstrated (Tables 1–4)[^paperhtml] | 2-, 3-, 4-bit                                | 1.6 → 8 bpw[^exl3wiki]                                |
| **PPL Llama-2-7B WT2 @ 4b**| 5.59 (YAQA-B, INT4 quantizer)[^paperhtml]    | 5.61–5.66 (QuIP#)                            | Near-FP16 ≥ 4 bpw[^exl3wiki]                          |
| **KL@2b Llama-3.1-8B-Inst**| **0.241** YAQA-B vs **0.356** LDLQ (−32%)[^paperhtml] | 0.356 (LDLQ baseline shown in YAQA paper)    | not reported in same units                            |
| **KL@4b Llama-3.1-8B-Inst**| **0.013** YAQA-B vs **0.019** LDLQ (−32%)[^paperhtml] | 0.019                                        | not reported                                          |
| **Beats QAT?**             | Yes — Gemma-3 12B INT4: YAQA-B DKL=0.056 vs Google's QAT recipe DKL=0.089[^paperhtml] | No claim                                     | No claim                                              |
| **License**                | **GPL v3**[^readme]                          | non-commercial / academic (QTIP repo)        | MIT (exllamav3)[^exl3wiki]                            |
| **Kernel availability**    | **None new** — repo ships QTIP's `inference.cu` (byte-identical, 472 lines)[^kernelcmp] plus a handful of extra shape registrations in `wrapper.cpp` | QTIP repo ships the same `inference.cu` (research-grade, fixed model shapes per template instantiation) | **Production-grade EXL3 kernels** with autotuning, multi-GPU, runtime shape support[^exl3wiki] |

**Critical observation:** The kernel paths in `yaqa-quantization/qtip-kernels/src/inference.cu` and `qtip/qtip-kernels/src/inference.cu` are byte-identical (`wc -l` confirms both are 472 lines, and `diff -q` flags no difference).[^kernelcmp] The only kernel-side changes are extra `decompress_matvec_*` shape registrations in `wrapper.cpp` and changes in `qtip_torch.cu` (the latter relates to Hessian computation, not inference). **YAQA introduces zero new inference kernels.**

---

## 3. Same Bitrate — Better Quality? Better Speed?

**Quality:** Yes, materially. At the **same** bitrate using the **same** QTIP kernel, YAQA-B reduces KL-to-original by ~30% across the board:

| Model & bits          | LDLQ DKL | YAQA-B DKL | Reduction |
|-----------------------|----------|------------|-----------|
| Llama-3.1-8B-Inst 2b  | 0.356    | 0.241      | −32%      |
| Llama-3.1-8B-Inst 3b  | 0.069    | 0.044      | −36%      |
| Llama-3.1-8B-Inst 4b  | 0.019    | 0.013      | −32%      |
| Llama-3.1-70B-Inst 2b | 0.497    | 0.335      | −33%      |
| Llama-3.1-70B-Inst 3b | 0.138    | 0.094      | −32%      |
| Llama-3.1-70B-Inst 4b | 0.045    | 0.030      | −33%      |

Source: Table 1, *Model-Preserving Adaptive Rounding*.[^paperhtml] On INT4 (LLama-3.1-8B), 0-shot average improves from 67.99 (LDLQ) → 68.92 (YAQA-B), PPL 6.76 → 6.72.[^paperhtml] On Gemma-3 12B INT4, YAQA-B (DKL 0.056) **beats Google's QAT recipe** (DKL 0.089) — a meaningful claim because QAT is normally the upper bound for PTQ methods.[^paperhtml]

**Speed:** **No change.** Inference uses the same kernel, same codebook, same tile shape, same RHT block. At eval time there is literally no path difference — a YAQA-quantized weight is bit-compatible with a QTIP-quantized weight of the same `(L, K, V, tlut_bits)` configuration. The decode side cannot tell which rounding algorithm produced the trellis.

So: **YAQA is QTIP with a smarter offline rounding step.** It pays a one-time, much larger calibration bill (30–50 GPU-hours per ~10B params for the Hessian sketch) in exchange for a permanent ~⅓ KL reduction at every bitrate, with **zero runtime cost**.

---

## 4. MoE Applicability

**Strong negative finding.** The repo has **no MoE support whatsoever**:

- `grep -rli "mixtral\|moe\|expert\|mixture"` across the entire repo returns **zero hits** in `.py` or `.md` files.
- `hessian_llama/llama_hess.py` is a hardcoded fork of HF's `modeling_llama.py` (one giant ~55KB file with `LlamaDecoderLayer`, `LlamaMLP`, `LlamaAttention`, etc.) — there is no expert routing, no `MixtralSparseMoeBlock`, no gating logic. The Hessian collection runs **per dense linear layer** with FSDP sharding tuned for monolithic Llama blocks.[^hessreadme]
- The paper's empirical section evaluates **Llama-3.1-1B/3.2B/8B/70B-Instruct and Gemma-3 12B**. No Mixtral. No DeepSeek-V2/V3. No Qwen-MoE. No OLMoE.[^paperhtml]
- The Hessian-sketch theory is layerwise. For an MoE, "layerwise" is ambiguous — is the Hessian per-expert, per-router-output, or per-token-routed-path? The paper does not address this.

To use YAQA on **DeepSeek V4 Flash** (256 experts, 6 active per token), we would have to:
1. Reimplement Hessian collection to capture per-expert activations under routed traffic.
2. Decide whether to use a shared Hessian sketch across experts (risk: ignores expert specialization) or per-expert (risk: blow up calibration cost by 256×).
3. Handle the long-tail of cold experts where Hessian samples will be sparse.

This is **non-trivial research**, not engineering. The MoE literature has separate methods specifically for this (MoEQuant, EAQuant, MILO, QuantMoE-Bench).[^moeq] **Arc would be on the bleeding edge if we tried this — there is no published precedent.**

---

## 5. Recommendation for Arc

**Skip the kernel. Conditionally port the algorithm. Wait on MoE.**

**(a) Skip the kernel — definitive.** Arc already has `mistralrs-quant/src/qtip/` with `mod.rs` + `viterbi.rs`, and we just shipped the Hadamard incoherence rotation (commit `faeebb605`). YAQA ships **the same inference kernel** as QTIP. There is nothing in the YAQA repo's `qtip-kernels/src/inference.cu` that Arc does not already (or could not already) have from upstream QTIP. **Porting the YAQA repo's kernel buys us zero inference speed and zero quality.**

**(b) Conditionally port the algorithm.** If Arc TurboQuant 2-bit (K=4 / V=3) lands and we measure a KL/PPL gap to FP16 that hurts model quality, the YAQA rounding algorithm is the highest-leverage offline improvement available without changing the on-disk format or the inference path. It would slot into Apprentice-Mode (offline quant pipeline) as a drop-in replacement for our current LDLQ. The trellis it produces is bit-compatible with our existing decoder.

The catch: YAQA is **GPL v3**. Arc is MIT/Apache-style. We cannot link the YAQA Python directly into Arc's quant pipeline. We would have to **re-implement** `LDLQ_2hess` and the Sketch-B Hessian collection from the paper, citing it but not copying GPL'd code. The algorithm itself is described in the paper with enough fidelity that a clean-room implementation is feasible.

**(c) Wait on MoE.** YAQA has **not been tested on any MoE model**. For DeepSeek V4 Flash specifically, the Hessian-sketch design must be re-derived for routed experts. That is a separate research project — not a port. If anyone in the trellis-quantization community (turboderp, Cornell-RelaxML, MoEQuant authors) publishes MoE-YAQA, revisit. Until then, the safer path is:
1. Ship QTIP 2-bit on V4 (current trajectory).
2. Validate PPL/KL against FP8 baseline.
3. If PPL gap > acceptable threshold, **then** invest in YAQA-style Hessian sketches per-expert.

---

## 6. Effort Estimate (If We Port)

**Engineering scope — algorithm port only, no MoE extension, no kernel work:**

| Task                                                                              | Agent-sessions |
|-----------------------------------------------------------------------------------|----------------|
| Read paper + YAQA code in detail; write clean-room spec doc                       | 1              |
| Implement Sketch-B Hessian collection in Rust (FSDP equivalent via Candle/ndarray; reuse Arc's existing per-layer hook plumbing) | 2–3            |
| Implement `LDLQ_2hess` rounding pass (replaces `viterbi.rs`'s offline calibration; trellis output unchanged) | 2              |
| Numerical parity test vs YAQA Python reference on Llama-3.1-8B (cosine ≥ 0.99 on a few layers' L matrices) | 1              |
| Integration into Arc apprentice-mode pipeline + CLI flag (`--quant-algo yaqa`)    | 1              |
| End-to-end PPL/KL eval on a real model                                            | 1              |
| **Total**                                                                         | **8–9 sessions** |

**Cost on top:** 30–50 GPU-hours of H100 time per ~10B params for the Hessian sketch *per model*. For DeepSeek V4 Flash (256-expert MoE), this number is unknown and probably much larger if we go per-expert.

**Hard prerequisites if we ever extend to MoE:** an additional **3–5 agent-sessions** of research design (deciding per-expert vs shared, sample budget per expert, cold-expert handling) before any code is written. This is the bit we should not estimate as "porting" — it is novel research.

---

## 7. Side Notes Worth Flagging

> *Noticed: EXL3 (turboderp's open-source, MIT-licensed, production-grade QTIP variant) already ships a tuned inference kernel with runtime shape support and multi-GPU, and supports 1.6 → 8 bpw. The YAQA kernel path is the same as QTIP's research-grade fixed-template kernel. If Arc ever wants a robust trellis kernel for a wider set of shapes, **studying EXL3's kernel is likely a bigger win than porting YAQA's kernel** (which is just QTIP's). Worth a separate audit.*[^exl3wiki][^exl3pr]

> *Noticed: YAQA's calibration cost (30–50 GPU-hours per 10B params) is high enough that it materially constrains experiment velocity. For Arc's "ship quickly, iterate" loop, this is a real friction. A 256-expert MoE at that rate would be days of H100 time per quantization run — that needs to be priced into any MoE port plan.*

---

## Citations

[^paper]: Tseng, Sun, De Sa. *Model-Preserving Adaptive Rounding*. arXiv:2505.22988, 2025. https://arxiv.org/abs/2505.22988
[^paperhtml]: HTML rendering of the paper containing all numerical tables. https://arxiv.org/html/2505.22988v1
[^readme]: `Cornell-RelaxML/yaqa-quantization` README. https://github.com/Cornell-RelaxML/yaqa-quantization/blob/main/README.md
[^hessreadme]: `Cornell-RelaxML/yaqa-quantization` `hessian_llama/README.md`. Local clone path `/tmp/yaqa-investigate/yaqa-quantization/hessian_llama/README.md`. Documents Sketch-A and Sketch-B `torchrun` invocations and resource requirements.
[^ldlq]: `LDLQ_2hess` in `lib/algo/ldlq.py` of YAQA repo. Local clone path `/tmp/yaqa-investigate/yaqa-quantization/lib/algo/ldlq.py`. Compare with upstream QTIP `LDLQ(Wr, L, cb, args, ...)` in `/tmp/yaqa-investigate/qtip/lib/algo/ldlq.py`.
[^bitshift]: `lib/codebook/bitshift.py` in YAQA repo (identical to QTIP). Local clone path `/tmp/yaqa-investigate/yaqa-quantization/lib/codebook/bitshift.py`. Contains `bitshift_codebook` class with `1mad`/`2mad`/`3inst`/`quantlut`/`quantlut_sym` decode modes and the Viterbi search.
[^kernelcmp]: Verified by `diff -rq /tmp/yaqa-investigate/qtip/qtip-kernels/src/ /tmp/yaqa-investigate/yaqa-quantization/qtip-kernels/src/` — only `wrapper.cpp` (added shape registrations) and `qtip_torch.cu` (Hessian-side, not inference) differ. `inference.cu` and `inference.h` are byte-identical at 472 and 56 lines respectively.
[^twitter]: Albert Tseng (@tsengalb99) on X, Sept 2025: *"An updated version of YAQA is now available on arXiv, now with some nice new theory on what types of Hessian approximations admit fast quantization algorithms. YAQA is still the only PTQ algorithm that beats QAT, and still the only PTQ algorithm with end to end error bounds."* https://x.com/tsengalb99/status/1972940864516444547
[^exl3wiki]: DeepWiki summary of turboderp-org/exllamav3 (EXL3). https://deepwiki.com/turboderp-org/exllamav3/4-exl3-quantization-system — describes EXL3 as QTIP-based with 1.6–8 bpw range, near-lossless at 4 bpw, MIT license, production kernels.
[^exl3pr]: turboderp-org/exllamav3 PR #26 ("Superior 3INST parameters") by louiehelm — community-tuned variant of the QTIP `3INST` codebook decoder used in EXL3. https://github.com/turboderp-org/exllamav3/pull/26
[^moeq]: MoE PTQ literature surveyed: *QuantMoE-Bench* (arXiv 2406.08155), *MoEQuant* (arXiv 2505.03804), *EAQuant* (arXiv 2506.13329), *MILO* (MLSys 2025). None apply YAQA-style Hessian sketches to expert-routed weights.

---

**Final disposition:** YAQA is a real ~⅓-KL improvement over QTIP/LDLQ at the same bitrate and zero inference cost — but it is **not** a successor format, **not** a faster kernel, and **not** MoE-validated. For Arc's current trajectory (QTIP 2-bit on V4 MoE, just shipped Hadamard incoherence), the right move is to ship the existing QTIP path, measure quality on V4, and **only then** decide whether the ~8 agent-session + 50 GPU-hour-per-10B-params investment in a YAQA algorithm port is justified by the quality delta we observe.
