# wave62-CM — FlashInfer as an Arc substrate: evaluation and verdict

**Date:** 2026-08-17
**Evidence base:** FlashInfer git clone, `version.txt` = **0.6.18**, commit
`e77a4a0d276367895c3b50a642fd8f326c03fb72`, dated 2026-08-16 (one day before this
memo). All file:line citations below are against that commit. No GPU was used;
every claim is a source-tree fact or an explicit arch guard, never a benchmark.

**Verdict: COMPLEMENT — not ADOPT.** FA4→MLA stands. Rung 2 is unblocked.

---

## Q1 — Sinks on SM90 (Hopper). **YES. The "Blackwell-only" premise was wrong.**

The search result that prompted this evaluation ("sinks exist on SM12x XQA and
SM100 trtllm-gen") is **incorrect as a statement about FlashInfer's sink
support**. Those are two of several sink paths, not the whole set.

- `flashinfer/jit/attention/variants.py:168-171` defines
  `attention_sink_decl = {"fa2": ..., "fa3": ...}`. **`fa3` is the Hopper/SM90
  backend**, confirmed at `flashinfer/jit/attention/modules.py:198-199`
  (`if backend == "fa3": extra_cuda_cflags += sm90a_nvcc_flags`) and
  `:1740-1814` (`batch_prefill_sm90_*` templates, `is_sm90_template=True`).
- `attention_sink_fa3_decl` (`variants.py:55-166`) is a full CUTLASS/CuTe
  `OnlineSoftmaxWithSink` — the sink is folded into `m`/`d` at `:114-116`,
  exactly the per-head scalar-in-the-denominator semantics V4 needs.
- Tests run this **on SM90**: `tests/attention/test_attention_sink.py:157`
  parametrizes `backend` over `["fa2", "fa3"]`, and `:163-164` skips fa3 only
  when `is_sm90a_supported(device)` is false. Sliding window is parametrized at
  `:155` (`window_left` ∈ {-1, 128}) — V4's `sliding_window=128`.

Why a naive grep says otherwise: `grep -r sink include/flashinfer/attention/hopper/`
returns **nothing**. The sink variant is a JIT-injected code string, not a
checked-in header. Anyone re-running this check must grep
`flashinfer/jit/attention/variants.py`, not the kernel headers.

**Q1 does not block adoption.** It also is not, on its own, a reason to adopt —
see Q2.

## Q2 — head_dim 512 on SM90. **Split answer, and the split is what decides this.**

First, a shape correction that matters: **V4's attention is symmetric 512/512**,
not the asymmetric MLA 576/512. RoPE is applied *in place* to the last 64 of the
512 dims (`mistralrs-core/src/models/deepseek4.rs:15-20`), and
`q_head_dim() == head_dim == 512` (`:444-445`). So FlashInfer's MLA kernel family
(`head_dim_ckv=512`, `head_dim_kpe=64`) is **the wrong family for V4**. The right
family is the generic FA2/FA3 MHA path.

| Path | head_dim 512 on SM90? | Evidence |
|---|---|---|
| **FA2, 16-bit (BF16/FP16) KV** | **YES** | `modules.py:1205-1213`: `supported_major_versions=[8, 9, 10, 11, 12]` — **9 is present** |
| **FA2, 1-byte (FP8) KV** | **NO** | same block, `:1206-1210`: `[10, 11, 12]` — SM90 excluded |
| **FA3 (Hopper-native WGMMA/TMA)** | **NO** | `flashinfer/utils.py:458-462`: `is_fa3_prefill_head_dim_supported` returns `head_dim_qk in {64, 128, 256}` for symmetric, else only `(192,128)` |

Supporting evidence for the FA2 512 path being real and shipped, not aspirational:
- `include/flashinfer/utils.cuh:214-218` — `DISPATCH_HEAD_DIM` has an explicit
  `case 512`.
- `include/flashinfer/utils.cuh:415-422` — `FA2DetermineCtaTileQ` has a
  "True VO-split (VO >= 512)" branch with register-pressure reasoning.
- `tests/attention/test_batch_prefill_kernels.py:313` and `:847` —
  `test_batch_prefill_with_paged_kv_cache_head_dim_512` /
  `..._ragged_kv_cache_head_dim_512`. Decode:
  `tests/attention/test_batch_decode_kernels.py:102` parametrizes
  `head_dim` ∈ [128, 256, **512**].
- The gate is `head_dim_512_supported()` = compute capability major **>= 8**
  (`test_batch_prefill_kernels.py:28-30`), i.e. SM80+, which includes SM90.

**This resolves the "when head dimension support exceeds 512" ambiguity: shipped,
not aspirational — but only on the FA2 kernel.**

### What the split means

On an H200, V4's 512/512 attention would land on the **FA2 Ampere-era large-head
kernel**: no WGMMA, no TMA, no Hopper warp specialization. FA3 — the kernel that
actually uses Hopper's hardware — **refuses head_dim 512 by allowlist**. Note
this is not a maturity judgement or a performance guess; it is a four-line
allowlist at `utils.py:458-462`.

Second, **head_dim 512 is not AOT-precompiled at all**: the default config is
`"fa2_head_dim": [(64,64), (128,128), (256,256)]` with the comment
"head_dim=512 (FA2 prefill/decode, SM100+) excluded to reduce space in the
jit-cache wheel" (`flashinfer/aot.py:1017-1021`). `"fa3_head_dim"` has no 512
entry at all. So 512 is a runtime JIT compile.

Third, **sinks + head_dim 512 together is untested**. Both are FA2 features and
the sink hook is head_dim-agnostic (it touches only `m`/`d` in
`transform_output`, `include/flashinfer/attention/prefill.cuh:1834-1836`), so the
combination should compile. But **every sink test hard-codes head_dim = 128**
(`test_attention_sink.py:41, 166, 380`). Arc would be the first user of that
cell.

Fourth, and decisive for Arc's FP8 ambitions: the FP8-KV large-head path is
**SM100+ only** (`modules.py:1206-1210`). Arc's `ARC_V4_FP8_KV` work
(wave43-BU/wave49-BZ) has no FlashInfer counterpart on Hopper.

---

## The finding that actually settles it

FlashInfer **has already built V4's exact attention kernel** — and explicitly
refuses to run it on Hopper.

`flashinfer/cute_dsl/attention/wrappers/batch_hca.py:402-419` —
`cute_dsl_hca_decode(query, window_kv_cache, compressed_kv_cache, ...,
window_indices, compressed_block_tables, ..., sinks=...)`. Query is
`[B, Q, H, 512]`; two disjoint pools (raw window + compressed) under one softmax;
per-head `sinks`; `hca_compress_ratio=128`. That is DeepSeek-V4 Heavily
Compressed Attention, verbatim — the same two-region read PR #90 exists to
express.

Its arch guard, `batch_hca.py:435-440`:

```python
if compute_capability not in ((10, 0), (10, 3)):
    raise ValueError("CuTe DSL HCA requires SM100/SM103, got ...")
```

Same story across the V4-shaped kernels:
- `flashinfer/mla/_core.py:1038-1064` — `trtllm_batch_decode_sparse_mla_dsv4`
  backend resolver **raises** on anything that is not SM100/SM103 or SM120/121.
- `include/flashinfer/attention/sparse_mla_sm120/decode_dsv4_kernel.cuh` —
  the `attn_sink`-aware DSV4 decode kernel is, per its directory name and
  `flashinfer/mla/_sparse_mla_sm120.py:424`
  (`@supported_compute_capability([120, 121])`), SM120-only.
- On SM90 FlashInfer offers only **dense** MLA (`_core.py:2203`).

So this is a REJECT resting on a **specific arch guard and a specific unsupported
shape** — `batch_hca.py:435-440` and `utils.py:458-462` — not on "probably not
mature enough." Per `CEILINGS.json` ANTI_PESSIMISM_PROTOCOL, that is the required
standard, and it is met.

**Conversely, the anti-pessimism protocol also cuts the other way here**, and
this is the part worth acting on: `flashinfer/cute_dsl/attention/dsa/hca_fp8.py`
is **4,001 lines of readable CuTeDSL source** implementing that kernel
(`arch_str = "sm_100"`, `:130-131`), under **BSD-3-Clause**
(`hca_fp8.py:1-2`). The whole `flashinfer/cute_dsl/attention` tree is **47,013
lines** of CuTeDSL, 52 files BSD-3-Clause / 4 Apache-2.0.

Arc is building a CuTeDSL head_dim-512 MQA-with-sinks kernel. FlashInfer has one,
in CuTeDSL, open, permissively licensed — aimed at sm_100. **Jish's stated reason
for choosing CuTeDSL over hand-written CUDA was precisely that it retargets.**
This is the strongest available evidence for that thesis, and it converts FA4→MLA
from "write a kernel from scratch" into "port a working reference across an arch
boundary." That is a materially smaller task, and it is the single most valuable
thing this evaluation found.

---

## Secondary questions

**Q3 — C/C++ API for Rust. Usable, via headers, not via the shipped `.so`.**
- No libtorch anywhere in the compiled artifacts: zero `TORCH_LIBRARY` /
  `torch/extension.h` hits in `csrc/` or `include/`; link line is
  `-shared -lcudart -lcuda` (`flashinfer/jit/cpp_ext.py:254-258`).
- Bindings are **TVM-FFI**, a C ABI over DLPack — 296 `TVM_FFI_DLL_EXPORT_TYPED_FUNC`
  sites, loaded by `dlopen`/`dlsym` (`flashinfer/jit/core.py:430`), not a CPython
  extension. Callable from Rust in principle, but requires vendoring
  `apache-tvm-ffi` headers, and there is **no `.a`, no `CMakeLists.txt` anywhere,
  and no C header exposing the ops**.
- **The clean path is header-only.** Only 2 of 224 headers under `include/`
  touch tvm; **zero touch torch**.
  `include/flashinfer/attention/prefill.cuh:19-35` includes only CUDA runtime +
  in-tree `.cuh`; `hopper/prefill_sm90.cuh:16-32` adds CUTLASS/CuTe. So an
  `arc-flashinfer` build.rs would: vendor CUTLASS/CCCL, add `include/` to an
  nvcc/`cc::Build` compile, and instantiate the templates behind Arc's own
  `extern "C"` shims. No Python, no libtorch.
- **Licence: mixed, and the good parts are clean.** Root is Apache-2.0
  (`LICENSE`), `NOTICE` present. SPDX census over `csrc/ include/ flashinfer/`:
  210 BSD-3-Clause, 127 Apache-2.0, 11 MIT — **and 61 files under the
  "NVIDIA TensorRT Source Code License Agreement" plus 1
  `LicenseRef-NvidiaProprietary`**. The restricted files are confined to
  `csrc/fmha_v2/**` (e.g. `csrc/fmha_v2/fmha/hopper/gmma_descriptor.h:1-11`,
  "…without an express license agreement from NVIDIA … is strictly prohibited")
  and `csrc/cudnn_sdpa_utils.h:3`. **Avoidable** — nothing Arc would take
  depends on them.
- **"Partly open-source" = confirmed, and it is the more important caveat.**
  Whole kernel families ship **only as prebuilt cubins with no source in-tree**
  (trtllm-gen FMHA/GEMM, cuDNN SDPA, DeepGEMM), fetched at runtime from NVIDIA
  Artifactory (`flashinfer/jit/cubin_loader.py:36-38`) via a **ctypes callback
  from C++ into Python** (`include/flashinfer/cubin_loader.h:34-44`). Those ops
  therefore require a live Python interpreter at runtime. Arc must not depend on
  them.

**Q4 — Grouped GEMM for MoE. NO, and Arc is already ahead.**
- Generic segment GEMM is **bf16/fp16 only** — the JIT emits exactly
  `["__nv_bfloat16", "half"]` (`flashinfer/jit/gemm/core.py:94,192,209`);
  runtime dispatch is `DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16`
  (`csrc/tvm_ffi_utils.h:86-96`). MoE weight formats are a closed trtllm enum
  (`flashinfer/tllm_enums.py:162-173`) with an explicit allowlist
  (`flashinfer/fused_moe/core.py:392-408`). **No 2-bit format anywhere.**
- **No weight-decode hook.** The mainloop is stock CUTLASS with
  `ComplexTransform::kNone` and no converter template parameter
  (`include/flashinfer/gemm/group_gemm.cuh:63,67`). Dequant is expressible only
  as block-scale operands, which cannot represent a trellis state→value map. A
  caller cannot inject a functor.
- Every *quantized* grouped GEMM is SM100/SM120-only by filename
  (`group_gemm_{fp8,mxfp4,nvfp4}_groupwise_sm1xx`); W4A16 is Blackwell-only.
  On Hopper the only grouped option is plain bf16/fp16.
- Arc's format is a bitshift trellis, 2 bits/weight, decode is **computed, not a
  LUT**, and **random-access** (`mistralrs-quant/kernels/qtip/qtip2b_common.cuh:33-34,
  47-61`) — therefore fusable into a mainloop. Arc already ships
  `mistralrs-quant/kernels/qtip/qtip_grouped_gemm.cu` (514 lines: expert sort →
  ragged tile map → cp.async double-buffered persistent CTAs → in-register decode
  → `mma.sync m16n8k16`) on **sm_80/sm_90**.
- Routing through FlashInfer would require decoding to BF16 first, destroying the
  4× HBM-byte advantage that is the entire point of the 2-bit path. **Task #6's
  trellis grouped-GEMM keystone is unaffected by this evaluation.**

**Q5 — Block-sparse KV vs PR #90. Complements; does not subsume.**
- The BSR format **can** express a two-region read. `(indptr, indices)` per
  Q-block-row (`flashinfer/sparse.py:284, 443-447`), the only validation is an
  upper bound (`:782`) — no monotonicity or contiguity requirement — and the
  gather is a pure indirection (`include/flashinfer/page.cuh:223-225`). So
  `[window blocks] ++ [compressed blocks]` in one list is legal. **This is
  independent third-party validation that PR #90's list-of-runs design is
  sound.**
- **But not at Arc's shapes on SM90.** `plan()` passes one `head_dim` as both qk
  and vo (`sparse.py:868-869, 884-885`), and on SM90 that resolves to FA3, which
  allows only {64,128,256}. Every VSA backend is SM100/103 or SM120/121 with
  head_dim ≤ 128 (`sparse.py:526-531, 625-631, 698-704`).
- PR #90 remains necessary: it is a **layout descriptor, not a kernel**
  (`segment.rs:27-38`, `:401-408`), and its near-term value is deleting the
  `Tensor::cat(&[k, comp], 2)` at `dsv4_attention.rs:431`. Nothing in FlashInfer
  provides the host-side run bookkeeping, `slide`, or per-region rollback it
  implements.

**Q6 — SM90 maturity. First-class, arguably co-primary. Not a risk.**
- **H100 is the only datacenter GPU in PR CI** (`.github/workflows/pr-test.yml:176-205`;
  other runners are A10G/T4). There is **no B200/SM100 runner in PR CI at all**.
- `9.0a` is in the release arch list in all four CUDA-version branches
  (`.github/workflows/release.yml:195`).
- SM90 has *exclusive* kernels still being added: FA3 MLA is SM90-only
  (`csrc/batch_mla_sm90_*.cu`, `include/flashinfer/attention/mla_hopper.cuh`),
  and **MonoMoE is Hopper-exclusive and new**
  (`flashinfer/fused_moe/monomoe.py:252` — `@supported_compute_capability([90])`).
- **No deprecation signals** attach to any sm90/hopper/fa3 path.
- Caveat: the *newest* work (NVFP4/MXFP8 GEMM, DSv4 sparse MLA, cute-dsl MoE,
  cutlass FMHA) is Blackwell-only. Hopper gets a maintained, CI-gated attention
  stack, not the feature frontier. **The gap is feature coverage, not decay.**

---

## Verdict: COMPLEMENT

FlashInfer does **not** solve V4's attention on our hardware. It clears Q1
(sinks on SM90 — the Blackwell-only premise was simply wrong), but on Q2 it
routes V4's 512/512 shape to the FA2 Ampere-era kernel because FA3 allowlists
head_dim to {64,128,256} (`utils.py:458-462`), and it refuses the actual V4
two-region kernel on Hopper outright (`batch_hca.py:435-440`). Adopting it as the
attention substrate would mean shipping V4 on a non-Hopper-native kernel, in an
untested (sinks × 512) cell, with FP8 KV unavailable on SM90.

### What changes for FA4→MLA

**Nothing is cancelled. Rung 2 (vanilla FA4 Rust binding) is unblocked — proceed.**
FlashInfer cannot make it redundant, because the kernel Arc needs does not exist
for SM90 in FlashInfer.

**One thing changes, and it is a scope reduction, not a scope cut.** Before
writing Arc's HCA kernel from scratch, read
`flashinfer/cute_dsl/attention/dsa/hca_fp8.py` (4,001 lines, BSD-3-Clause). It is
V4 HCA in CuTeDSL — two-region, head_dim 512, per-head sinks, FP8 — differing
from Arc's target essentially in `arch_str`. Retargeting sm_100 → sm_90 is a real
engineering task (2-CTA MMA and tcgen05/tmem constructs are Blackwell-specific
and have no direct Hopper equivalent, so the mainloop needs genuine rework), but
it is a **port with a working oracle**, not a blank page. Treat it as the
reference implementation for FA4→MLA and cite it in the NOTICE if any of it is
carried across.

Also reusable immediately and independent of the port: the sink fold in
`variants.py:39-46` (fa2) and `:114-116` (fa3) is a compact, correct log2-domain
formulation Arc can check its own sink math against.

### What changes for PR #90

**Nothing — proceed as designed, with one piece of validating evidence.**
FlashInfer's BSR gather (`page.cuh:223-225`) accepts arbitrary, non-monotonic
block lists, which is exactly PR #90's list-of-runs contract. An independent
production engine reaching the same design is confirmation, not competition.
FlashInfer does not subsume #90 on SM90 (wrong head_dim), and does not implement
the host-side layer #90 provides at all.

### Do not re-raise unless one of these changes

1. `is_fa3_prefill_head_dim_supported` (`flashinfer/utils.py:458-462`) gains 512.
2. `cute_dsl_hca_decode`'s arch guard (`batch_hca.py:435-440`) admits `(9, 0)`.
3. Arc's fleet moves to SM100/SM103 — at which point **re-open immediately**:
   on Blackwell, `cute_dsl_hca_decode` is V4's attention kernel, ready-made, and
   this verdict inverts to ADOPT.

### Surfaced, not shipped

- `mistralrs-quant/kernels/qtip/qtip_grouped_gemm.cu` already targets sm_80/sm_90
  with 2-bit-class weights end-to-end; 4-bit descriptor plumbing is stubbed
  (`mistralrs-quant/src/qtip/grouped.rs:9-10, 65-78`). Unrelated to FlashInfer,
  noted because Q4 required reading it.
- If we ever want the FA2 head_dim-512 path as a **correctness oracle** for Arc's
  own kernel on SM90 (BF16 KV only), it is one JIT call. Cheap, and it would give
  the sinks × 512 cell its first execution anywhere. Worth ~15 min inside an
  existing rental; not worth a rental of its own.
