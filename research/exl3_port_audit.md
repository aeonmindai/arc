# EXL3 Port Audit — Should Arc Replace RUN-158 QTIP?

**Mission:** Audit turboderp's ExLlamaV3 (EXL3) as a potential upgrade path for Arc's just-shipped RUN-158 Rust QTIP stack. Pure research, no implementation.

**Date:** 2026-05-23 · **Branch baseline:** `master` @ commit `4860dff94` · **RUN-158 commit:** `faeebb605`

---

## 1. EXL3 in one paragraph

EXL3 is the only production-grade open-source inference stack that ships a fused-Viterbi QTIP-style trellis quantizer with a tensor-core GEMM. It is a deliberate streamlining of Cornell QTIP (NeurIPS 2024): same 16×16 tile shape that matches `mma.m16n8k16`, same Hadamard incoherence rotation (locked to 128-element blocks with `1/sqrt(128) = 0.088388347648f`), same K=2…8 trellis bit-rates, but ditching QTIP's learned LUTs in favor of three **procedural codebooks** (`cb=0` LCG with constants `89226354u`/`64248484u`; `cb=1` MCG with multiplier `0xCBAC1FEDu`; `cb=2` 2INST `0x83DCD12Du` + `vabsdiff4` with accumulator `0x6400u`) that decode in registers via integer-pipe MADs running concurrently with the tensor cores. Quantization runs **single-pass with on-the-fly Hessian computation** (block-LDL sweeping in reverse to compensate accumulated error) and a parallel Viterbi over 16-element vectors per tile. The shipped GEMM is **Marlin-inspired** with templated `SH_STAGES` (global→shared async cp.async pipeline) and `FRAG_STAGES` (register double-buffering 1–5 deep) plus a register-level `dq_dispatch<bits, cb>()` that dequantizes uint32 packed-bit chunks directly into `FragB` tensor-core operand fragments. Repo is MIT-licensed, 891 stars, actively maintained, ships in ~36 Python architecture files covering every major dense + MoE LLM through GLM-4.7-Flash and at least one user has an open issue (#210) requesting DeepSeek-V4-Flash support.

---

## 2. EXL3 vs Arc RUN-158 — Side-by-side

| Dimension                    | EXL3 (turboderp, MIT)                                                              | Arc RUN-158 (`mistralrs-quant/src/qtip/`)                                                    |
| ---------------------------- | ---------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| **Tile shape**               | 16×16 (Tensor-Core-locked, m16n8k16) — `TILESIZE_M == 16` static_assert            | None (pure-Rust row-by-row over packed nibbles)                                              |
| **Hadamard block size**      | **Fixed 128** — `had_ff_r_128_kernel`, `TORCH_CHECK_DIV(input, 1, 128)`            | **Up to 128, power-of-2 cap** (`QTIP_ROTATION_MAX_BLOCK = 128`) ([mod.rs:79](../mistralrs-quant/src/qtip/mod.rs#L79)) |
| **Hadamard scale**           | `0.088388347648f` = `1/√128`                                                       | `1/√n` per block via [`fwht_inplace`](../mistralrs-quant/src/turboquant/wht.rs)              |
| **Codebook**                 | Procedural: MCG `0xCBAC1FEDu`, 2INST `0x83DCD12Du`+`vabsdiff4`, LCG variants — **decoded in int-pipe registers, zero VRAM cost** | Static Box-Muller Gaussian LUT, 2^L = **65,536 × 2 entries** stored on GPU ([mod.rs:170-180](../mistralrs-quant/src/qtip/mod.rs#L170)) |
| **Trellis K (bits/sym)**     | 1, 2, 3, 4, 5, 6, 7, 8 (eight comp units `exl3_comp_unit_{1..8}.cu`)               | **K=4 only** (`pub const K: u32 = 4;` → 2 bpw) ([mod.rs:150](../mistralrs-quant/src/qtip/mod.rs#L150)) |
| **L (state width)**          | 16-bit state, `edges = 65536 >> K`                                                 | `L=16` ([mod.rs:148](../mistralrs-quant/src/qtip/mod.rs#L148))                               |
| **V (reproduction dim)**     | 2 (same)                                                                            | `V=2` ([mod.rs:152](../mistralrs-quant/src/qtip/mod.rs#L152))                                |
| **Bitrate range**            | **1.0 → 8.0 bpw** with per-tensor overrides + `-hb`/`--head_bits` 1–8              | **2 bpw locked** (no mixed-precision, no head-bits override)                                 |
| **Quantizer**                | Fused-Viterbi CUDA kernel with on-the-fly Hessian + block-LDL reverse sweep, parallel over tiles | Rayon-parallel **CPU Viterbi** ([viterbi.rs](../mistralrs-quant/src/qtip/viterbi.rs), 507 lines), no Hessian, no LDL |
| **GEMM kernel**              | Marlin-derived `exl3_gemm_kernel` + `exl3_gemv_kernel` (batch≤8 decode path), templated `<bits, cb, TILESIZE_M, TILESIZE_K, TILESIZE_N, SH_STAGES, FRAG_STAGES>`, `cudaLaunchCooperativeKernel` for grid sync | **Pure-Rust dequant-then-matmul** through Candle (`dequantize_w()` → BF16 GEMM); no fused kernel |
| **Async pipeline**           | `cp.async` 4–6 stages G→S, 2–5 stages register frag double-buffer                  | None                                                                                          |
| **Tensor Core MMA**          | `ptx_mma_m16n8k16` consumes dequantized `FragB` directly                           | None — round-trips through full BF16 weights                                                 |
| **MoE kernel**               | `exl3_moe` / batched `exl3_mgemm`, fused gate+up+activation+down, 9 K-instances × 2 N-alignments = 18 templates | None — `gather_forward` on QtipLayer goes through the trait but no fused MoE path implemented |
| **Per-module bitrate**       | Yes (`-l2`/`-l3` flags, `optimize.py` driven by per-layer KL-div measurements)     | No (uniform 2-bit)                                                                            |
| **CUDA cap target**          | Architecture-agnostic dispatch; `__CUDA_ARCH__ > 890` branch in `exl3_gemm_kernel.cuh` chooses `group_barrier()` (sm_90+) vs `grid.sync()` (older); core kernel runs sm_70+ with `cudaLaunchCooperativeKernel` (sm_60+) | n/a — CPU only |
| **License**                  | MIT                                                                                | (Arc repo license)                                                                            |
| **HF-compatible tensor naming** | Yes (`trellis` / `suh` / `svh` / `mcg`)                                          | Custom format (`blocks` / `row_scales` / `lut`)                                              |
| **Lines of CUDA**            | ~226 KB total in `exllamav3/exllamav3_ext/quant/` (gemm_inner 21.8 KB, gemv_kernel 12.7 KB, moe_kernel 10.5 KB, dq 10.3 KB, kernel_map 10.5 KB) | 0                                                                                             |
| **Lines of Rust**            | 0                                                                                  | 1,553 (`mod.rs`) + 507 (`viterbi.rs`) = **2,060 lines**                                       |

Sources for EXL3 column: `exllamav3/exllamav3_ext/quant/codebook.cuh`, `exl3_gemm_inner.cuh`, `exl3_gemm_kernel.cuh`, `exl3_moe.cu`, `exl3_moe_kernel.cuh`, `hadamard.cu`, `quantize.cu`; [DeepWiki EXL3 page](https://deepwiki.com/turboderp-org/exllamav3/4-exl3-quantization-system).

---

## 3. MoE-specific evidence (the part that matters for V4 Flash)

### 3.1 `exl3_moe` kernel surface

The MoE entry point ([`exl3_moe.cuh`](https://github.com/turboderp-org/exllamav3/blob/master/exllamav3/exllamav3_ext/quant/exl3_moe.cuh)) takes **separate K parameters for gate / up / down** (`K_gate`, `K_up`, `K_down`), three `at::Tensor` pointer-arrays per projection (`*_ptrs_trellis`, `*_ptrs_suh`, `*_ptrs_svh`), token-sorted-by-expert routing, `expert_count` bincount, and routing weights. It supports the procedural MCG codebook only (`gate_mcg`, `up_mcg`, `down_mcg`) — the `mul1` codebook is rejected for MoE. Importantly **gate and up are NOT pre-fused into a single `gate_up_proj`** like Arc's `gpt_oss.rs` does — they are kept as separate calls but the kernel fuses gate+up GEMMs, Hadamard transform, gating multiplication, activation, down-projection GEMM, output Hadamard, and scatter-add into one launch.

### 3.2 Expert dispatch model

```
exl3_moe_max_concurrency = num_sms / MOE_SMS_PER_EXPERT
grid_dim = (MOE_SMS_PER_EXPERT, 1, concurrency)
```

EXL3 partitions SMs across experts at launch time. The kernel iterates `for (; expert_idx < num_experts; ++expert_idx)` with `expert_idx_assign++ % concurrency != group_idx` routing experts to thread groups round-robin. **No hard cap on num_experts** — the GLM-4.7-Flash architecture file (`glm4_moe.py`) configures `"n_routed_experts": 128, "n_shared_experts": 1, "num_experts_per_tok": 8` and works without code changes. The selection table is `kernel = exl3_moe_kernel_instances[2 * K + N_off]` where `K ∈ {0..8}` and `N_off ∈ {0, 1}` (1 if both `hidden_dim` and `intermediate_dim` are divisible by 256). 18 kernel instances precompiled (`exl3_moe_inst_{0..8}.cu` + `_128` + `_256` variants for shape `0`).

### 3.3 Will it work on DeepSeek-V4 Flash's 256-expert top-6 layout?

**Architecturally yes, but unvalidated.** EXL3's MoE dispatcher has no expert-count limit, doesn't care about top-k (token-sorted dispatch is k-agnostic), and supports shared experts via separate `GatedMLP`. Issue #210 ("Support for Deepseek V4 flash", opened 2026-05-12) explicitly requests it; no implementation yet. The architecture file (`dflash.py`) exists but appears to be a draft-model spec, not the full V4 MoE. Porting V4 to EXL3 mainly requires writing a new `architecture/deepseek4.py` mirroring `glm4_moe.py` (~few hundred lines of routing config + weight mapping). The kernel itself already handles 256 routed + 1 shared experts with top-6 routing because nothing in `exl3_moe_kernel.cuh` hardcodes those counts.

### 3.4 Bitrate floor on MoE — CRITICAL HONEST FINDING

**The 1.6 bpw "coherent" claim is dense-only.** It comes from turboderp's Llama-3.1-70B-Instruct-exl3 model card, which lists a `1.60bpw_H3` variant for that **dense** model. The MoE evidence is much more pessimistic:

| Model                          | Type       | Lowest validated bpw         | Size at floor | KL-div (q→fp16) |
| ------------------------------ | ---------- | ---------------------------- | ------------- | --------------- |
| Llama-3.1-70B-Instruct         | Dense      | **1.60 bpw** (H3 head)       | ~14 GiB       | (not published) |
| GLM-4.6 (355B MoE)             | MoE        | **3.00 bpw** (3bpw)          | 124 GiB       | 0.326           |
| GLM-4.7-Flash (355B MoE)       | MoE        | **2.00 bpw** (`2bpw-H6`)     | 83 GiB        | **0.651**       |
| GLM-4.7-Flash 2.10bpw-tuned    | MoE        | 2.10 bpw                     | 86 GiB        | 0.544           |

Source: [mratsim/GLM-4.7-EXL3](https://huggingface.co/mratsim/GLM-4.7-EXL3), [mratsim/GLM-4.6-EXL3](https://huggingface.co/mratsim/GLM-4.6-EXL3).

KL-div 0.651 at 2 bpw on GLM-4.7-Flash is **not coherent** by usual standards — that's "noticeable but possibly tolerable" territory. The community manual-tuning recipe explicitly recommends keeping `self_attn` and `shared_experts` at 6–8 bpw and only dropping routed experts to 2–3 bpw. This is **mixed-precision quantization driven by per-tensor KL-div measurement**, not uniform 1.6 bpw.

**Implication for V4 Flash (280B/A13B):** the realistic EXL3 floor on V4 is **~2.0–2.5 bpw mixed** (routed experts at 2 bpw, attention + shared at 4–6 bpw), landing at **~72–90 GiB**. That's still far better than the BF16 ~560 GiB or any uniform 4-bit scheme, but it is **not** "1.6 bpw → ~57 GiB" as the brief speculates. The 1.6 bpw figure simply does not transfer to MoE in any current evidence.

---

## 4. Quantified upgrade value: what EXL3 actually adds

Apples-to-apples comparison, same model (hypothetical DeepSeek-V4-Flash), same hardware (H100 80GB):

| Capability                              | RUN-158 (today)                              | EXL3 ported                                   | Delta                                       |
| --------------------------------------- | -------------------------------------------- | --------------------------------------------- | ------------------------------------------- |
| **Bitrate**                             | 2 bpw uniform                                | 2 bpw routed / 4–6 bpw shared+attn (mixed)    | Quality gain at fixed bpw                   |
| **V4 Flash footprint** (estimate)       | ~70 GiB (uniform 2 bpw across all weights) — but **broken on attention** at this bitrate without per-tensor overrides | ~75–90 GiB (mixed)                          | RUN-158 fits, but probably won't be coherent on V4 attention. EXL3 mixed-precision lands in the same envelope with much better quality. |
| **Matmul kernel**                       | Pure-Rust dequant → BF16 matmul (Candle)     | Marlin-derived TC GEMM, memory-bound at 4 bpw | **~10–30× decode throughput** (estimate; EXL3 reports memory-bound at 4bpw on RTX 4090) |
| **MoE forward**                         | Sequential per-expert via `gather_forward`, dequant + Candle matmul, no fusion | Single fused `exl3_moe` launch, gate+up+act+down fused, expert-parallel across SMs | **Big** — Arc's current MoE quant path is the bottleneck for V4 |
| **Codebook in VRAM**                    | 65,536 × 2 fp32 LUT = **512 KB / layer**     | 0 (procedural decode in int pipe)             | -512 KB / layer (negligible on 80 GB; large win on registers + cache) |
| **Quantizer wallclock**                 | Rayon CPU Viterbi (~minutes per 70B layer)    | Fused CUDA Viterbi (~minutes for entire 70B model) | **~10–100× faster calibration**             |
| **Per-tensor bitrate**                  | No                                            | Yes, KL-div-driven `optimize.py`              | Enables the only known path to coherent <3 bpw MoE |
| **HF tensor naming**                    | Custom                                        | Compatible (`trellis`/`suh`/`svh`)            | Reuses existing community-quantized models (~half the time) |

### What RUN-158 keeps even after porting

- The **Hadamard rotation infrastructure** ([`turboquant/wht.rs`](../mistralrs-quant/src/turboquant/wht.rs)) is portable and Arc-specific because TurboQuant KV uses the same WHT primitive.
- The **`QuantMethod` trait** ([`lib.rs:955`](../mistralrs-quant/src/lib.rs#L955)) is already MoE-aware via `gather_forward` — EXL3's `exl3_moe` API maps onto it cleanly.
- The **Viterbi CPU implementation** ([`viterbi.rs`](../mistralrs-quant/src/qtip/viterbi.rs)) is useful as a **reference oracle** for tests, even if production calibration shifts to CUDA.
- The Marlin-FFI plumbing already in [`mistralrs-quant/src/gptq/marlin_ffi.rs`](../mistralrs-quant/src/gptq/marlin_ffi.rs) is the **exact pattern** EXL3 FFI needs to follow — Arc already ships a Marlin kernel for GPTQ-AWQ, so the build infrastructure (`cudaforge::KernelBuilder`, build.rs glob) is reusable.

### What RUN-158 work becomes obsolete

- The Rust forward path (`forward` impl on `QtipLayer`, `dequantize_weights_rotated_f32` and dense-matmul fallback) is superseded by the fused TC GEMM.
- The CPU Viterbi probably ships only as a verification reference, not the production calibration path.
- The 65,536-entry Box-Muller Gaussian LUT goes away entirely — EXL3's procedural codebook is strictly better (smaller, faster, just as accurate).

**Estimate: ~60% of RUN-158 code (Viterbi + Hadamard + tests + serde + trait wiring) survives as scaffolding; ~40% (LUT, packed-nibble layout, dequant path) gets replaced.**

---

## 5. Porting cost

### 5.1 Kernel surface to vendor

Total: ~226 KB CUDA + headers, plus generated comp-unit files. Critical files:

| File                                                              | Size    | Role                                                       |
| ----------------------------------------------------------------- | ------- | ---------------------------------------------------------- |
| `quant/codebook.cuh`                                              | 5.1 KB  | Procedural codebooks (MCG, LCG, 2INST)                     |
| `quant/exl3_dq.cuh`                                               | 10.3 KB | Register-level `dq_dispatch<bits,cb>` for 1–8 bpw           |
| `quant/exl3_gemm_inner.cuh`                                       | 21.9 KB | Templated GEMM core (16×TILESIZE_K×TILESIZE_N tiles)       |
| `quant/exl3_gemm_kernel.cuh`                                      | 8.7 KB  | Outer kernel launcher                                       |
| `quant/exl3_gemv_kernel.cuh`                                      | 12.7 KB | Decode path (m ≤ 8) — **single-token decode**              |
| `quant/exl3_moe_kernel.cuh`                                       | 10.5 KB | Fused MoE forward (gate+up+act+down)                       |
| `quant/exl3_moe.cu` + `.cuh`                                      | 10.6 KB | MoE dispatch and entry                                     |
| `quant/exl3_kernel_map.cu` + `.cuh`                               | 15.2 KB | Compute-cap-aware shape selection                          |
| `quant/exl3_devctx.cu` + `.cuh`                                   | 3.3 KB  | SM count caching                                            |
| `quant/coop_autotune.cu` + `.cuh`                                 | 17.7 KB | Cooperative-launch autotune (mostly self-contained)        |
| `quant/hadamard.cu` + `hadamard_inner.cuh`                        | 22.7 KB | 128-block H transform                                       |
| `quant/quantize.cu`                                               | 15.9 KB | Fused-Viterbi quantizer (256-element tiles)                |
| `quant/reconstruct.cu`                                            | 4.0 KB  | Dequantize for debugging                                    |
| `quant/pack.cu`                                                   | 6.0 KB  | Pack symbols into storage layout                            |
| `quant/comp_units/exl3_comp_unit_{1..8}.{cu,cuh}`                 | ~3 KB   | One TU per bit-width (compile-time fan-out)                |
| `quant/comp_units/exl3_moe_inst_{0..8}*.cu`                       | ~2 KB   | MoE kernel instantiations                                   |
| `ptx.cuh`, `util.cuh`, `compat.cuh`                               | ~5 KB   | PTX wrappers and compat helpers                             |

**Vendoring estimate: ~70 files, mostly headers + thin `.cu` wrappers.**

### 5.2 Rust FFI surface

Pattern is already proven by [`marlin_ffi.rs`](../mistralrs-quant/src/gptq/marlin_ffi.rs). EXL3 needs roughly:

```rust
// extern "C" wrappers around four C++ entrypoints:
exl3_gemm(a, b, c, size_m, size_k, size_n, k_bits, codebook, suh, svh);
exl3_gemv(...);              // batch=1 decode fast path
exl3_moe(...);               // fused MoE
exl3_quantize(...);          // calibration-time only
```

Plus a few utility calls: `exl3_devctx_init`, `exl3_max_concurrency`. Roughly **~400 lines of FFI + ~300 lines of safe Rust wrappers + ~200 lines of QuantMethod integration = ~900 lines new Rust**.

### 5.3 Build integration

Arc's [`mistralrs-quant/build.rs`](../mistralrs-quant/build.rs) already uses `cudaforge::KernelBuilder` and globs `kernels/marlin/*.cu`. Adding `kernels/exl3/*.cu` follows the **exact same pattern**. The `__CUDA_ARCH__ > 890` branch in EXL3 means we get a sm_90 (H100) fast path for free; the cooperative-launch fallback works on sm_70+.

### 5.4 Agent-session cost estimate

Following the Aeonmind calibration anchors (porting well-debugged Python/CUDA → Rust + tests, ~30–60 min per ~500 lines of source):

| Phase                                                       | Estimate              | Notes                                                |
| ----------------------------------------------------------- | --------------------- | ---------------------------------------------------- |
| **P1** Vendor kernel tree + add to `build.rs`               | 1–2 sessions          | Mostly mechanical; ~70 files, follow Marlin precedent |
| **P2** Write FFI bindings (`exl3_ffi.rs`) + safe wrappers   | 2–3 sessions          | ~900 lines new Rust, well-typed entry points         |
| **P3** Implement `QuantMethod for Exl3Layer` (dense)        | 1–2 sessions          | Forward / dequantize_w / serde — straight-line work  |
| **P4** Implement `gather_forward` via `exl3_moe`            | 2–3 sessions          | Pre-sort tokens by expert, route to fused kernel     |
| **P5** Calibration glue: HF safetensors → trellis indices   | 1–2 sessions          | Reuse turboderp's `convert.py` semantics             |
| **P6** Numerical equivalence tests vs EXL3 Python reference | 1–2 sessions          | Vendor a small Llama-3.1-8B exl3 checkpoint as fixture |
| **P7** Wire into Arc plain pipeline + MoE pipeline          | 2 sessions            | Loader detection, model integration                  |
| **Total**                                                   | **10–16 sessions** | **~1–2 days of agent-time**                          |

This is significantly **less** than RUN-158 took to design from scratch (estimated ~5–8 sessions for the Rust QTIP from greenfield + viterbi + RUN-158 rotation work) because **the algorithm is solved** — we are vendoring, not researching.

### 5.5 Risks

1. **Sample MoE coverage gap.** No one has shipped EXL3 on DeepSeek V4 specifically; we'd be the first. The architecture file needs to be written. Mitigation: the MoE kernel is expert-count-agnostic, so this is "writing a 200-line Python descriptor" not "writing a new kernel." Risk: **low**.
2. **H100 throughput unverified.** EXL3 reports "memory-bound at 4 bpw on RTX 4090." The sm_90 branch is a single `group_barrier` change — there is no Hopper-optimized TMA pipeline. EXL3 on H100 will probably run **well but not Hopper-optimal** (no WGMMA, no TMA). Mitigation: we can profile and add a TMA path post-port; the existing Marlin-style kernel is already faster than what Arc has today. Risk: **medium** (we won't hit theoretical peak, but we'll beat current Arc by ≥10× on quantized matmul).
3. **`cudaLaunchCooperativeKernel` + CUDA Graphs interaction.** Arc's `project_cuda_graph_plan` envisions GPU-autonomous decode. Cooperative kernels can be captured into CUDA graphs but the locks pattern (`int* locks` arg used for cross-block reduction sync) may complicate graph reuse. Mitigation: GEMV path (batch ≤ 8) uses normal `<<<...>>>` launch — that's the decode path, which is what graphs matter for. Risk: **low–medium**.
4. **License/header preservation.** MIT is permissive but every vendored file needs an SPDX header preserving turboderp's copyright. Mitigation: trivial — preserve headers as-is. Risk: **none**.
5. **Marlin-on-Hopper underperformance.** Public discussion ([Red Hat / Machete blog](https://developers.redhat.com/articles/2024/10/14/introducing-machete-mixed-input-gemm-kernel)) confirms Marlin "struggles on Hopper GPUs (namely H100)." Machete was designed to fix that. EXL3's GEMM is Marlin-derived. Mitigation: this is a known gap, not a regression — we are still much faster than the current pure-Rust path. We can add a Machete-style TMA path as a future M-milestone. Risk: **medium** for absolute peak, **low** for "much faster than RUN-158."

---

## 6. Recommendation

**Port EXL3 — but staged, and only after rental verification of RUN-158 on real V4-Flash weights.**

Reasoning:

1. **RUN-158 alone will not ship V4 Flash competitively.** The current pure-Rust forward path through Candle has no chance of being memory-bound on H100, and the uniform 2-bit quantization will almost certainly fail on V4's MLA attention (per all MoE EXL3 evidence, attention needs ≥4 bpw). RUN-158 was a correctness milestone (cos sim ≥0.97 on Gaussian) — production deployment on V4 needs mixed-precision and a real GEMM.
2. **EXL3 is the most production-ready QTIP-style stack that exists.** It is the only one with:
   - Fused Viterbi quantizer (vs Arc's CPU Rayon)
   - Tensor-core GEMM (vs Arc's Candle BF16)
   - Fused MoE forward (vs Arc's per-expert dispatch)
   - Per-tensor mixed-bitrate (vs Arc's uniform 2 bpw)
   - HF-compatible naming (instant access to community quants)
3. **Porting cost (10–16 sessions, 1–2 days) is small** relative to building the same surface from scratch (which would be a multi-week research project). Arc's existing Marlin-FFI infrastructure pre-pays the build integration. The QuantMethod trait already has `gather_forward` for MoE — the wiring is straightforward.
4. **But verify first.** Before sinking 10+ sessions into porting, **rent an H100 and test EXL3 directly** with `mratsim/GLM-4.7-EXL3 2bpw_H6` or similar to confirm the throughput numbers are real and the MoE path is solid. Quick 1-session validation prevents porting something that doesn't deliver.
5. **The 1.6 bpw → 57 GB MoE dream is not real.** Don't sell that internally. The realistic V4 Flash target is **2.0–2.5 bpw mixed → ~75–90 GiB**, which **is** the difference between "fits a single H100" and "doesn't fit," and that is the win worth chasing.

**Staged approach:**

- **Stage 1 (1 session):** Rent H100, run `pip install exllamav3` + load GLM-4.7-Flash 2bpw, measure throughput + perplexity. Decision gate.
- **Stage 2 (P1-P3, 4–7 sessions):** Vendor kernel tree, write FFI, implement dense `Exl3Layer` with `QuantMethod`.
- **Stage 3 (P4, 2–3 sessions):** MoE forward via `exl3_moe`, validate on Mixtral-8x7B or Qwen3-MoE first (community checkpoints exist).
- **Stage 4 (P5-P7, 4–6 sessions):** DeepSeek-V4 architecture file, loader integration, e2e test.

If Stage 1 doesn't show ≥5× speedup over Arc's current path on the same H100, abandon the port and double down on Hopper-native kernel work (TMA + WGMMA, Machete-style) instead.

---

## 7. Linear ticket structure (if porting)

Suggested milestone: **M-QTIP-EXL3** (parent), under the existing weight-compression workstream.

| Ticket            | Title                                                                  | Depends on     | Est sessions |
| ----------------- | ---------------------------------------------------------------------- | -------------- | ------------ |
| RUN-EXL3-00       | H100 rental validation: run EXL3 GLM-4.7 2bpw, measure tok/s + ppl     | —              | 1            |
| RUN-EXL3-01       | Vendor kernel tree → `mistralrs-quant/kernels/exl3/`, build.rs wiring  | RUN-EXL3-00    | 1–2          |
| RUN-EXL3-02       | `exl3_ffi.rs` — extern "C" bindings for gemm / gemv / moe / quantize    | RUN-EXL3-01    | 2–3          |
| RUN-EXL3-03       | `Exl3Layer: QuantMethod` (dense forward + dequantize_w + serde)        | RUN-EXL3-02    | 1–2          |
| RUN-EXL3-04       | `Exl3Layer::gather_forward` via `exl3_moe`                             | RUN-EXL3-03    | 2–3          |
| RUN-EXL3-05       | Calibration converter: `safetensors` → trellis indices (Rust)          | RUN-EXL3-02    | 1–2          |
| RUN-EXL3-06       | Numerical equivalence tests vs EXL3 Python (Llama-3.1-8B exl3 fixture) | RUN-EXL3-03    | 1–2          |
| RUN-EXL3-07       | Loader auto-detect + plain pipeline integration                        | RUN-EXL3-03    | 1            |
| RUN-EXL3-08       | MoE pipeline integration (DeepSeek-V4 + Mixtral arch)                  | RUN-EXL3-04, 07 | 2            |
| RUN-EXL3-09       | KL-div-driven per-tensor bitrate (`optimize.py` port to Rust)          | RUN-EXL3-05    | 2 (deferred) |
| RUN-EXL3-10       | Deprecate RUN-158 forward path; keep CPU Viterbi as test oracle        | RUN-EXL3-03    | 1            |

**Critical path: 00 → 01 → 02 → 03 → 06 → 07** (≈ 7–10 sessions to dense feature parity)
**Full MoE delivery (V4 Flash ready): + 04 + 08** (≈ +4–5 sessions)

---

## Citations

- Arc QTIP source: [`mistralrs-quant/src/qtip/mod.rs`](../mistralrs-quant/src/qtip/mod.rs#L62-L160), [`viterbi.rs`](../mistralrs-quant/src/qtip/viterbi.rs)
- Arc QuantMethod trait: [`mistralrs-quant/src/lib.rs:955-1039`](../mistralrs-quant/src/lib.rs#L955-L1039)
- Arc existing Marlin: [`mistralrs-quant/kernels/marlin/`](../mistralrs-quant/kernels/marlin/), [`marlin_ffi.rs`](../mistralrs-quant/src/gptq/marlin_ffi.rs)
- Arc build pattern: [`mistralrs-quant/build.rs`](../mistralrs-quant/build.rs), [`mistralrs-paged-attn/build.rs`](../mistralrs-paged-attn/build.rs)
- Arc MoE usage: [`mistralrs-core/src/moe/experts.rs:498-517`](../mistralrs-core/src/moe/experts.rs#L498), [`mistralrs-core/src/models/gpt_oss.rs:451-480`](../mistralrs-core/src/models/gpt_oss.rs)
- EXL3 repo: [github.com/turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3) — MIT, 891 stars, master branch
- EXL3 format doc: [`doc/exl3.md`](https://github.com/turboderp-org/exllamav3/blob/master/doc/exl3.md)
- EXL3 conversion doc: [`doc/convert.md`](https://github.com/turboderp-org/exllamav3/blob/master/doc/convert.md)
- EXL3 kernel surface: `exllamav3/exllamav3_ext/quant/` (codebook.cuh, exl3_gemm_inner.cuh, exl3_moe.cu, exl3_dq.cuh, hadamard.cu, quantize.cu)
- DeepWiki EXL3 page: [deepwiki.com/turboderp-org/exllamav3](https://deepwiki.com/turboderp-org/exllamav3/4-exl3-quantization-system)
- GLM-4.6 EXL3: [huggingface.co/mratsim/GLM-4.6-EXL3](https://huggingface.co/mratsim/GLM-4.6-EXL3)
- GLM-4.7 EXL3: [huggingface.co/mratsim/GLM-4.7-EXL3](https://huggingface.co/mratsim/GLM-4.7-EXL3)
- Llama-3.1-70B EXL3 (1.6 bpw dense reference): [huggingface.co/turboderp/Llama-3.1-70B-Instruct-exl3](https://huggingface.co/turboderp/Llama-3.1-70B-Instruct-exl3)
- DeepSeek-V4 support tracking issue: [github.com/turboderp-org/exllamav3/issues/210](https://github.com/turboderp-org/exllamav3/issues/210)
- QTIP paper: [arxiv.org/abs/2406.11235](https://arxiv.org/abs/2406.11235) (Tseng et al., NeurIPS 2024)
- Arc prior research: [`research/yaqa_audit.md`](./yaqa_audit.md) (touches on EXL3 in §2)
- Marlin-on-Hopper gap: [Machete blog, Red Hat 2024](https://developers.redhat.com/articles/2024/10/14/introducing-machete-mixed-input-gemm-kernel)
