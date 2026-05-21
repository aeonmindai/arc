# Code Index — Paper to Repo Mapping

54 repos cloned (shallow `--depth=1`), 4.8 GB total. Each entry below maps the technique → paper PDF in `research/` → cloned code in `research/code/`.

**Where the code actually lives:** the 4.8 GB of git clones live in `../../arc-research-code/` (a sibling directory to the arc repo, outside git). The entries under `research/code/<domain>/<repo>/` are **symlinks** pointing into that shadow location. This keeps the arc git repo small while preserving the structure.

**To repopulate the shadow location** (e.g., after fresh clone of arc):

```bash
./research/code/CLONE_REPOS.sh
```

That script re-clones every repo into `../arc-research-code/`. The symlinks then resolve automatically.

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
| **TurboQuant (Arc's own)** | `02_turboquant_kv/turboquant_arc_iclr2026.pdf` | (your own private code; not in this dir) | ✓ shipped in Arc |
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
| TurboQuant (Arc's own) | Lives in Arc's private codebase | (own) |
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
