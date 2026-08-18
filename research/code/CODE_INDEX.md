# Code Index — Paper to Repo Mapping

54 repos cloned (shallow `--depth=1`), 4.8 GB total. Each entry below maps the technique → paper PDF in `research/` → cloned code in `research/code/`.

**Where the code actually lives:** the 4.8 GB of git clones live in `../../arc-research-code/` (a sibling directory to the arc repo, outside git). The entries under `research/code/<domain>/<repo>/` are **symlinks** pointing into that shadow location. This keeps the arc git repo small while preserving the structure.

**To repopulate the shadow location** (e.g., after fresh clone of arc):

```bash
./research/code/CLONE_REPOS.sh
```

That script re-clones every repo into `../arc-research-code/`. The symlinks then resolve automatically.

---

## Frontier model coverage in cloned reference engines (SGLang + vLLM)

Critical insight: **the model loaders and attention backends for our four target families already exist** as production Python code in SGLang and vLLM. Arc's job is to port these to Rust, not invent them. SGLang is the canonical V4 reference; vLLM lags on V4 but has full coverage of the others.

### DeepSeek V4 Pro / V4 Flash (1.6T / 284B; CSA + HCA attention; 1M context)

| Concern | SGLang path | vLLM path |
|---|---|---|
| Main loader | `code/06_foundation/sglang/python/sglang/srt/models/deepseek_v4.py` (1,816 lines, extends `deepseek_v2.py`) | ❌ not yet (only test fixture `tests/models/test_deepseek_v4_mega_moe.py`) |
| MTP / NextN | `code/06_foundation/sglang/python/sglang/srt/models/deepseek_v4_nextn.py` (280 lines) | `code/06_foundation/vllm/vllm/model_executor/models/deepseek_mtp.py` |
| **CSA + HCA attention backend** | `code/06_foundation/sglang/python/sglang/srt/layers/attention/dsv4/` (10 files): `compressor.py` + `compressor_v2.py` (CSA K compression), `indexer.py` (top-k select), `metadata.py` + `metadata_kernel.py`, `quant_k_cache.py`, **`tilelang_kernel.py`** (the actual attention kernel) | — |
| V3-style NSA / DSA sibling | `code/06_foundation/sglang/python/sglang/srt/layers/attention/dsa/` (8 files) | — |
| Top-level dispatch | `code/06_foundation/sglang/python/sglang/srt/layers/attention/deepseek_v4_backend.py` + `_backend_hip_radix.py` for AMD | — |
| Config schema | `code/06_foundation/sglang/python/sglang/srt/configs/deepseek_v4.py` (43 layers, 256+1 experts, kv_lora_rank=512, qk_nope_head_dim=448, index_topk=512) | — |
| Original DeepSeek inference | `code/03_per_token_speed/deepseek_v3_nsa/inference/model.py` (V3 only; V4 weights load via SGLang) | — |

### Kimi K2.5 / K2.6 (1T / 32B active; MLA; 256K context; MoonViT vision)

| Concern | SGLang path | vLLM path |
|---|---|---|
| Main loader (text) | `code/06_foundation/sglang/python/sglang/srt/models/kimi_k25.py` (859 lines, **extends `DeepseekV3ForCausalLM`**) | `code/06_foundation/vllm/vllm/model_executor/models/kimi_k25.py` |
| EAGLE-3 speculative head | `code/06_foundation/sglang/python/sglang/srt/models/kimi_k25_eagle3.py` | — |
| Vision tower (MoonViT) | `code/06_foundation/sglang/python/sglang/srt/models/kimi_vl.py` + `kimi_vl_moonvit.py` | — |
| Linear-attn variant | `code/06_foundation/sglang/python/sglang/srt/models/kimi_linear.py` | `code/06_foundation/vllm/vllm/model_executor/models/kimi_linear.py` |
| Audio | — | `code/06_foundation/vllm/vllm/model_executor/models/kimi_audio.py` |
| Config schema | `code/06_foundation/sglang/python/sglang/srt/configs/kimi_k25.py` | — |
| MoBA reference paper code | `code/03_per_token_speed/moba/moba/moba_efficient.py` (Moonshot research; **NOT used by K2.6 production — K2.6 uses MLA**) | — |

**Important:** K2.5 inherits `DeepseekV3ForCausalLM` directly. Arc's existing `mistralrs-core/src/models/deepseek3.rs` is ~90% of what K2.5/K2.6 needs — just add Kimi-specific config + tokenizer + MoonViT for vision.

### GLM 5.0 / 5.1 (754B / 40B active; DSA attention; 200K context)

**SGLang serves GLM-5 via `GlmMoeDsaForCausalLM`** (a subclass of `DeepseekV2ForCausalLM`) inside `glm4_moe.py`. No separate `glm5.py` file is needed — GLM-5 reuses the GLM-4-MoE codepath with DSA attention.

| Concern | SGLang path | vLLM path |
|---|---|---|
| Main loader — covers **GLM-4.5, GLM-4.6, GLM-4.7, GLM-5.0, GLM-5.1** | `code/06_foundation/sglang/python/sglang/srt/models/glm4_moe.py` (1,499 lines). Two entry classes: `Glm4MoeForCausalLM` (plain GLM-4 MoE) and **`GlmMoeDsaForCausalLM` extends `DeepseekV2ForCausalLM` (the GLM-5 path)**. | `code/06_foundation/vllm/vllm/model_executor/models/glm4_moe.py` |
| Lite variant | `code/06_foundation/sglang/python/sglang/srt/models/glm4_moe_lite.py` (792 lines) | `code/06_foundation/vllm/vllm/model_executor/models/glm4_moe_lite.py` |
| MTP / NextN | `code/06_foundation/sglang/python/sglang/srt/models/glm4_moe_nextn.py` (168 lines) | `code/06_foundation/vllm/vllm/model_executor/models/glm4_moe_mtp.py` + `glm4_moe_lite_mtp.py` |
| Vision (GLM-4V MoE) | `code/06_foundation/sglang/python/sglang/srt/models/glm4v.py` (820 lines) + `glm4v_moe.py` (285 lines) | `code/06_foundation/vllm/vllm/model_executor/models/glm4v.py` + `glm4_1v.py` |
| OCR | `code/06_foundation/sglang/python/sglang/srt/models/glm_ocr.py` + `glm_ocr_nextn.py` | `code/06_foundation/vllm/vllm/model_executor/models/glm_ocr.py` + `glm_ocr_mtp.py` |
| ASR | `code/06_foundation/sglang/python/sglang/srt/models/glmasr.py` | `code/06_foundation/vllm/vllm/model_executor/models/glmasr.py` |
| Function calling | `code/06_foundation/sglang/python/sglang/srt/function_call/glm47_moe_detector.py` (handles GLM-4.7 + GLM-5) | — |
| Top-k routing for GLM-5 | In `code/06_foundation/sglang/python/sglang/srt/models/deepseek_v2.py:549` — uses grouped-noaux-tc top-k (same as V3/V3.2/GLM-4-MoE-Lite) | — |

**Important:** Arc can serve GLM-5.0 and GLM-5.1 through a single `glm_moe.rs` loader that handles GLM-4.5/4.6/4.7/5.0/5.1 — same code path with different config. The architecture is stable across the line; per Zhipu's pattern, version bumps come from post-training.

### Attention backends registered in SGLang (21 total)

Located at `code/06_foundation/sglang/python/sglang/srt/layers/attention/`:

| Backend | What it is | When to use |
|---|---|---|
| `flashinfer` | FlashInfer GEMM + paged KV | Default GPU path for most models |
| `trtllm_mla` / `cutlass_mla` / `flashmla` | MLA-specific attention | DeepSeek V2/V3, Kimi K2.x, GLM-5 |
| `dsa` | DeepSeek Sparse Attention (V3-style NSA) | V3, V3.2, **GLM-5** |
| `dsv4` | **V4's CSA + HCA hybrid attention** | DeepSeek V4 Pro/Flash |
| `nsa` | Generic NSA path | Any NSA-trained model |
| `fa3` / `fa4` | FlashAttention 3 / 4 | Generic SDPA |
| `triton` / `torch_native` / `flex_attention` | Software fallbacks | Reference / CPU |
| `aiter` | AMD ROCm path | AMD GPUs |
| `wave`, `ascend`, `intel_amx`, `intel_xpu`, `trtllm_mha` | Vendor-specific paths | Non-NVIDIA hardware |
| `tokenspeed_mla`, `dual_chunk_flash_attn` | Specialized variants | — |

Backend registry: `code/06_foundation/sglang/python/sglang/srt/layers/attention/attention_registry.py` (uses `@register_attention_backend(name)` decorator).

### Arc → SGLang porting map for new model loaders

| Arc work | SGLang reference to port from | Estimated effort |
|---|---|---|
| `mistralrs-core/src/models/deepseek4.rs` | `sglang/python/sglang/srt/models/deepseek_v4.py` | 2 weeks |
| `mistralrs-core/src/models/kimi_k2.rs` | `sglang/python/sglang/srt/models/kimi_k25.py` | 1 week (90% reuse from `deepseek3.rs`) |
| `mistralrs-core/src/models/glm_moe.rs` (covers GLM-4.5+, GLM-5.x) | `sglang/python/sglang/srt/models/glm4_moe.py` | 1.5 weeks |
| `mistralrs-quant/src/dsv4/` (CSA + HCA) | `sglang/python/sglang/srt/layers/attention/dsv4/*.py` | 2-3 weeks (TileLang kernel → port to Triton/CUDA or emit CPU reference for Tier A) |

> ⚠️ **What Arc actually runs for V4 attention today — read before assuming the
> kernel row above is done.**
>
> - **There is no fused head_dim-512 attention kernel in the tree.** Only
>   feasibility probes exist. Do not describe one as shipped.
> - **V4 cannot use FlashAttention.** Sinks are set on **all 43 layers**, and the
>   fused flash-with-sinks kernels support head_dim ∈ **{64, 80, 96, 112, 128,
>   192, 256}** only (`mistralrs-core/src/attention/backends/sinks.rs:79`). V4's
>   head_dim is **512**, so it runs the **unfused matmul + `softmax_with_sinks`**
>   path.
> - **Expert parallelism is not implemented** — `Comm::Dummy`, world_size 1.
>
> Also **not in the tree, and never to be described as shipped**: LDLQ, EoRA, a
> KV segment allocator, and the xKV cross-layer pool.

---

## Domain 1: Weight compression (`01_weight_compression/`)

| Technique | Paper PDF | Code repo | Status |
|---|---|---|---|
| **QTIP (trellis quantization)** | `01_weight_compression/qtip_trellis_coded_quantization.pdf` | `code/01_weight_compression/qtip/` | ✓ official Cornell-RelaxML |
| **QuIP#** | `01_weight_compression/quip_sharp_lattice_quant.pdf` | `code/01_weight_compression/quip_sharp/` | ✓ official |
| **AQLM** | `01_weight_compression/aqlm_additive_quantization.pdf` | `code/01_weight_compression/aqlm/` | ✓ official |
| **GPTQ** | `01_weight_compression/gptq_quantization.pdf` | `code/01_weight_compression/gptq/` | ✓ IST-DASLab |
| **AWQ** | `01_weight_compression/awq_activation_aware_quantization.pdf` | `code/01_weight_compression/awq/` | ✓ MIT Han Lab |
| **SmoothQuant** | `01_weight_compression/smoothquant_int8.pdf` | `code/01_weight_compression/smoothquant/` | ✓ MIT Han Lab |
| **HQQ** | `01_weight_compression/hqq_half_quadratic_quantization.pdf` | `code/01_weight_compression/hqq/` | ✓ Mobius ML |
| **BitNet b1.58** | `01_weight_compression/bitnet_b1_58_1bit_weights.pdf` | `code/01_weight_compression/bitnet/` | ✓ Microsoft |
| **SparseGPT** | `01_weight_compression/sparsegpt_pruning.pdf` | `code/01_weight_compression/sparsegpt/` | ✓ IST-DASLab |
| **Wanda** | `01_weight_compression/wanda_pruning.pdf` | `code/01_weight_compression/wanda/` | ✓ Locuslab |
| **SliceGPT** | `01_weight_compression/slicegpt_dimension_reduction.pdf` | `code/01_weight_compression/slicegpt/` | ✓ Microsoft (in TransformerCompression) |
| **MegaBlocks (sparse MoE kernels)** | `01_weight_compression/megablocks_block_sparse_moe.pdf` | `code/01_weight_compression/megablocks/` | ✓ Stanford |
| **Expert pruning** | (various MoE compression PDFs) | `code/01_weight_compression/expert_sparsity/` | ✓ |
| **YOCO** | `12_long_context/yoco_you_only_cache_once.pdf` | `code/01_weight_compression/microsoft_unilm/YOCO/` | ✓ in unilm |
| **Differential Transformer** | `16_underexploited/differential_transformer_microsoft.pdf` | `code/01_weight_compression/microsoft_unilm/Diff-Transformer/` | ✓ in unilm |
| **TD-MoE (Tucker)** | `01_weight_compression/td_moe_tucker_decomposition_moe.pdf` | not on GitHub yet (ICLR 2026, OpenReview only) | ⚠ paper only |
| **TurboSparse / PowerInfer dReLU** | `01_weight_compression/moe_lrd_low_rank_decomposition_2024.pdf` (Turbo Sparse) | `code/03_per_token_speed/powerinfer/` | ✓ PowerInfer engine |

## Domain 2: KV cache compression (`02_kv_compression/`)

| Technique | Paper PDF | Code repo | Status |
|---|---|---|---|
| **TurboQuant (Arc's own)** | `02_turboquant_kv/turboquant_arc_iclr2026.pdf` | (your own private code; not in this dir) | ⚠ **in Arc, OFF by default — not "shipped"** (see note below) |
| **QuaRot (rotation-based)** | `02_turboquant_kv/quarot_rotation_quantization.pdf` | `code/02_kv_compression/quarot/` | ✓ SPCL |
| **SpinQuant** | `02_turboquant_kv/spinquant_learned_rotations.pdf` | `code/02_kv_compression/spinquant/` | ✓ Meta |
| **KIVI (2-bit lossless)** | `02_turboquant_kv/kivi_2bit_kv_cache.pdf` | `code/02_kv_compression/kivi/` | ✓ Jy-yuan |
| **KVQuant** | `02_turboquant_kv/kvquant_outlier_kv_quantization.pdf` | `code/02_kv_compression/kvquant/` | ✓ SqueezeAILab |
| **Atom (INT4 serving)** | `02_turboquant_kv/atom_int4_inference.pdf` | `code/02_kv_compression/atom/` | ✓ EFeslab |
| **SnapKV** | `02_turboquant_kv/snapkv_compression.pdf` | `code/02_kv_compression/snapkv/` | ✓ FasterDecoding |
| **PyramidKV** | `02_turboquant_kv/pyramidkv_layer_kv.pdf` | `code/02_kv_compression/pyramidkv/` | ✓ Zefan-Cai |
| **Quest (query-aware KV)** | `12_long_context/quest_query_aware_kv_selection.pdf` | `code/02_kv_compression/quest/` | ✓ MIT Han Lab |
| **InfiniGen (KV offload)** | `12_long_context/infinigen_kv_offload.pdf` | `code/02_kv_compression/infinigen/` | ✓ SNU Comparch |
| **DuoAttention (per-head)** | `16_underexploited/duo_attention.pdf` | `code/03_per_token_speed/duo_attention/` | ✓ MIT Han Lab |

> ⚠️ **Correction — TurboQuant is not shipped in Arc.** An earlier revision of
> this row read "✓ shipped in Arc". It is not on any default path:
>
> - The eager KV path is **opt-in via `ARC_TURBOQUANT_KV=1`**, default **off**
>   (`mistralrs-core/src/kv_cache/mod.rs`).
> - The paged kernel exists at **head_dim 128 only**. Off-envelope requests fall
>   back to `Auto` with a warning.
> - There is **no kernel at head_dim 512**, so **DeepSeek V4 cannot use
>   TurboQuant at all**.
> - **No TurboQuant forward pass has ever been benchmarked.** The compression
>   ratio sometimes quoted for it (bytes/token at 3.5 bits vs BF16) is **format
>   arithmetic, not a measurement** — it has never been produced by a forward
>   pass. Never present it as measured.
>
> Read every "✓" in the Status columns of this document as **"the reference code
> is cloned and present here"**, which is what it means everywhere else in the
> file. It is **not** a statement that Arc ships the technique.

## Domain 3: Per-token speed (`03_per_token_speed/`)

| Technique | Paper PDF | Code repo | Status |
|---|---|---|---|
| **MoBA (block attention routing)** | `12_long_context/moba_mixture_of_block_attention.pdf` | `code/03_per_token_speed/moba/` | ✓ MoonshotAI |
| **NSA (native sparse attention)** | `09_sparse_attention/deepseek_native_sparse_attention.pdf` | `code/03_per_token_speed/deepseek_v3_nsa/` | ✓ in V3 inference code |
| **SageAttention (INT8 attention)** | `16_underexploited/sage_attention_int8.pdf` + `sage_attention_2.pdf` | `code/03_per_token_speed/sage_attention/` | ✓ thu-ml |
| **EAGLE-3 (speculative)** | `10_foundation/eagle_v3.pdf` | `code/03_per_token_speed/eagle/` | ✓ SafeAILab |
| **Medusa (multi-head spec)** | `15_speculative_extensions/medusa_multi_head.pdf` | `code/03_per_token_speed/medusa/` | ✓ FasterDecoding |
| **MagicDec (long-ctx spec)** | `15_speculative_extensions/magicdec_speculative_long_context.pdf` | `code/03_per_token_speed/magicdec/` | ✓ Infini-AI-Lab |
| **Lookahead Decoding** | `15_speculative_extensions/lookahead_decoding.pdf` | `code/03_per_token_speed/lookahead/` | ✓ hao-ai-lab |
| **PowerInfer 2 (sparse exec)** | `10_foundation/powerinfer_2_smartphone.pdf` | `code/03_per_token_speed/powerinfer/` | ✓ SJTU-IPADS |
| **MInference (dynamic sparse)** | `16_underexploited/minference_dynamic_sparse.pdf` | `code/03_per_token_speed/minference/` | ✓ Microsoft |
| **FlashAttention** | `10_foundation/flashattention_v1/v2/v3.pdf` | `code/03_per_token_speed/flash_attention/` and `code/06_foundation/flash_attention_dao/` | ✓ Dao-AILab |
| **Mixture of Depths** | `13_alt_architectures/mixture_of_depths_adaptive.pdf` | no public code | ⚠ paper only |
| **MTP (in DeepSeek V3)** | `11_models/deepseek_v3.pdf` | `code/03_per_token_speed/deepseek_v3_nsa/` (V3 includes MTP) | ✓ in V3 weights |

## Domain 4: Aggregate throughput / serving systems (`04_aggregate_throughput/`)

| Technique | Paper PDF | Code repo | Status |
|---|---|---|---|
| **Sarathi-Serve (chunked prefill)** | `14_serving_systems/sarathi_serve_chunked_prefill.pdf` | `code/04_aggregate_throughput/sarathi_serve/` | ✓ Microsoft |
| **vAttention (dynamic KV)** | `14_serving_systems/vattention_dynamic_kv_management.pdf` | `code/04_aggregate_throughput/vattention/` | ✓ Microsoft |
| **LMCache (KV reuse across requests)** | `14_serving_systems/lmcache_kv_reuse.pdf` | `code/04_aggregate_throughput/lmcache/` | ✓ LMCache |
| **DeepSpeed-MII** | `14_serving_systems/deepspeed_fastgen.pdf` | `code/04_aggregate_throughput/deepspeed_mii/` | ✓ Microsoft |
| **ChunkAttention** | `14_serving_systems/chunk_attention.pdf` | `code/04_aggregate_throughput/chunk_attention/` | ✓ Microsoft |
| **LLMLingua (prompt compression)** | `12_long_context/llmlingua_prompt_compression.pdf` | `code/04_aggregate_throughput/llmlingua/` | ✓ Microsoft |
| **Splitwise (P/D disagg)** | `14_serving_systems/splitwise_pd_disaggregation.pdf` | code not standalone (in vLLM/DeepSpeed-MII) | ⚠ ref impl in MII |
| **DistServe** | `14_serving_systems/distserve_pd_disaggregation.pdf` | code at LMSYS sglang (alt impl) | ⚠ |

## Foundation engines (`06_foundation/`)

| Component | Why it's here | Code repo |
|---|---|---|
| **vLLM** | Reference serving engine, PagedAttention origin | `code/06_foundation/vllm/` |
| **SGLang** | Competitor + RadixAttention reference | `code/06_foundation/sglang/` |
| **FlashInfer** | The attention kernel library Arc should integrate | `code/06_foundation/flashinfer/` |
| **FlashAttention (Dao)** | The base FA implementation | `code/06_foundation/flash_attention_dao/` |
| **FlashMLA** | DeepSeek's MLA kernel — required for V4 Pro | `code/06_foundation/flashmla/` |
| **TensorRT-LLM** | NVIDIA's reference for kernel quality | `code/06_foundation/tensorrt_llm/` |
| **DeepSpeed** | Microsoft's training/inference stack reference | `code/06_foundation/deepspeed/` |

## Supporting / alternative architectures (`99_supporting/`)

| Architecture | Why it's here | Code repo |
|---|---|---|
| **Hopfield Layers** | The math foundation we cited (now research-only) | `code/99_supporting/hopfield_layers/` |
| **Mamba** | State-space model reference | `code/99_supporting/mamba/` |
| **RWKV** | Linear-attention alternative | `code/99_supporting/rwkv/` |
| **xLSTM** | Extended LSTM (Hochreiter's sequel) | `code/99_supporting/xlstm/` |
| **Lightning Attention** | Faster linear attention | `code/99_supporting/lightning_attention/` |

---

## Papers we cite but don't yet have code for

| Paper | Why no code | Workaround |
|---|---|---|
| TD-MoE (ICLR 2026) | OpenReview-only at this date; no GitHub release | Implementation guided by paper Algorithm 1; uses PyTorch + TensorLy |
| Mixture of Depths (DeepMind) | DeepMind research, no public release | Reimplement from paper |
| Differential Transformer | In `unilm/Diff-Transformer/` ✓ | (have it) |
| YOCO | In `unilm/YOCO/` ✓ | (have it) |
| TurboQuant (Arc's own) | Lives in Arc's private codebase | (own) — **off by default, head_dim 128 kernel only, never benchmarked**; see the correction under Domain 2 |
| Apprentice Mode | Not yet validated by any paper | Killed from roadmap |

## Repo size summary

| Domain | Repos | Size |
|---|---|---|
| Weight compression | 14 | ~660 MB |
| KV compression | 10 | ~1.4 GB |
| Per-token speed | 12 | ~830 MB |
| Aggregate throughput | 6 | ~450 MB |
| Foundation | 7 | ~1.3 GB |
| Supporting | 5 | ~120 MB |
| **Total** | **54** | **~4.8 GB** |
