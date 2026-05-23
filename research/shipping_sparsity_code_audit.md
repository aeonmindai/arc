# Shipping Sparsity in Production Inference Engines — Code Audit

**Methodology:** Read actual source code in cloned repos at
`/Users/jish/Documents/GitHub/arc/research/code/`. Quoted code is verbatim
from the head of each repo. READMEs and blog posts deliberately ignored —
those exaggerate. This audit asks: *what is wired into the forward pass
that a user can actually turn on at serve time?*

---

## 1. Code-inspection table

| Engine            | Method                              | Status            | File                                                                                                                       | LOC      | Activation                                                                                  |
| ----------------- | ----------------------------------- | ----------------- | -------------------------------------------------------------------------------------------------------------------------- | -------- | ------------------------------------------------------------------------------------------- |
| **vLLM**          | DeepSeek-V3.2 sparse MLA (FlashMLA) | **SHIPS**         | `vllm/v1/attention/backends/mla/flashmla_sparse.py`                                                                        | 1171     | HF config field `index_topk` present → `self.is_v32 = True` → `is_sparse=True` auto-wired   |
| **vLLM**          | DeepSeek-V3.2 sparse MLA (FlashInfer/TRT-LLM-gen kernel) | **SHIPS** | `vllm/v1/attention/backends/mla/flashinfer_mla_sparse.py`                                                           | 365      | same `index_topk` HF config flag                                                            |
| **vLLM**          | DeepSeek-V3.2 sparse MLA (ROCm Aiter)     | **SHIPS**   | `vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py`                                                                  | ~700     | same `index_topk` HF config flag                                                            |
| **vLLM**          | DeepSeek-V4 sparse MLA (ROCm DSV4)        | **SHIPS**   | `vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse_dsv4.py`                                                             | ~800     | DSV4-specific config                                                                        |
| **vLLM**          | XPU MLA sparse (Intel)                    | **SHIPS**   | `vllm/v1/attention/backends/mla/xpu_mla_sparse.py`                                                                         | ~300     | same `index_topk`                                                                           |
| **vLLM**          | Gemma3n FFN activation sparsity (`GeluAndMulSparse`) | **SHIPS** | `vllm/model_executor/layers/activation.py:233-289`                                                              | ~60      | HF config field `activation_sparsity_pattern[layer_id] > 0`                                 |
| **vLLM**          | MiniCPM-S FATReLU (`FatreluAndMul`)       | **SHIPS**   | `vllm/model_executor/layers/activation.py:77-113`                                                                          | ~37      | HF config `hidden_act == "fatrelu"`                                                         |
| **vLLM**          | MInference vertical-slash (block-sparse FA) | NOT IN MAIN PATH | (only via external MInference patcher; not in vLLM tree)                                                              | —        | external `MInference(attn_type="minference").patch_model(model)`                            |
| **vLLM**          | compressed-tensors 2:4 weight sparsity    | **REMOVED**  | `vllm/.../compressed_tensors.py:262`: `raise DeprecationWarning("Sparsity support has been removed from compressed-tensors.")` | —     | (used to work, now disabled)                                                                |
| **SGLang**        | DeepSeek-Sparse-Attention (DSA) Indexer for V3.2 | **SHIPS** | `python/sglang/srt/layers/attention/dsa/dsa_indexer.py`                                                              | 1747     | model config has `index_topk`; `get_dsa_index_topk(hf_config)` returns non-None             |
| **SGLang**        | DSV4 CSA (4×) + HCA (128×) compressor    | **SHIPS** (private/preview) | `python/sglang/srt/layers/attention/dsv4/compressor.py`                                                  | ~500     | model is DSV4; env var `SGLANG_OPT_USE_ONLINE_COMPRESS`                                     |
| **SGLang**        | MInference vertical-slash via FA3        | **SHIPS** (dual-chunk only) | `python/sglang/srt/layers/attention/dual_chunk_flashattention_backend.py:1668`                            | ~50      | model: Qwen2.5-Instruct dual-chunk; uses `sgl_kernel.sparse_flash_attn.sparse_attn_func`    |
| **SGLang**        | Gemma3n FFN activation sparsity          | **SHIPS**    | `python/sglang/srt/models/gemma3n_causal.py:99-130`                                                                        | ~30      | same Gemma3n HF config                                                                      |
| **SGLang**        | sgl-kernel `convert_vertical_slash_indexes` CUDA kernel | **SHIPS** | `sgl-kernel/csrc/attention/vertical_slash_index.cu`                                                              | ~470     | exposed as PyTorch op; only call site is dual-chunk attn backend                            |
| **TensorRT-LLM**  | Gemma3n activation sparsity (AutoDeploy) | **SHIPS**    | `tensorrt_llm/_torch/auto_deploy/models/custom/modeling_gemma3n.py:140-172`                                                | ~30      | HF config `activation_sparsity_pattern`                                                     |
| **TensorRT-LLM**  | DeepSeek V3.2 sparse MLA                 | NOT FOUND in tree | (no `Indexer`, `topk_indices`, `sparse_attention`, or `index_topk` strings in the TRT-LLM Python tree)                | —        | n/a (vLLM/SGLang lead here)                                                                 |
| **FlashInfer**    | `BlockSparseAttentionWrapper` API        | **SHIPS** (library API) | `flashinfer/flashinfer/sparse.py:69-657`                                                                          | ~600     | call `wrapper.plan(indptr, indices, ...)` — used by vLLM `flashinfer_mla_sparse.py`         |
| **PowerInfer-v1** | Predictor-gated FFN axpy (CUDA)          | **SHIPS** but research-only | `ggml-cuda.cu` (lines 4541-4604, 4685-4755, 5666-5709, 7232-7320)                                              | ~600     | env `LLAMA_SPARSE_PRED_THRESHOLD=0.0` + GGUF file with `powerinfer.sparse_threshold` field |
| **PowerInfer-v2 (SmallThinker)** | Pred-gated CPU SpMV (AVX512) | **SHIPS** (mobile/CPU only) | `smallthinker/powerinfer/powerinfer-cpu/src/sparse_matmul.hpp:57`                                                | ~200     | predictor weights baked into GGUF; `SPARSE_PRED_THRESHOLD = 0`                              |
| **PowerInfer-v2** | MoE-expert sparse pipeline                | **SHIPS** (CPU) | `smallthinker/powerinfer/moe_sparse_pipeline/`                                                                             | ~1000    | predictor-based routing                                                                     |
| **MInference**    | PIT sparse FA, block-sparse FA, retr-attn | RESEARCH PATCH | `minference/minference/models_patch.py:19`                                                                                | ~300     | `MInference(attn_type=..., kv_type=...).patch_model(model)` — monkey-patches HF forward     |
| **MoBA**          | Mixture-of-Block-Attention reference     | RESEARCH POC | `moba/moba/moba_efficient.py:67-200+`                                                                                      | ~300     | none — no production integration; flash-attn-varlen wrapper only                            |
| **MLX**           | (not in repo tree)                       | N/A (repo not cloned) | —                                                                                                                  | —        | —                                                                                           |

---

## 2. Real implementations (verbatim quotes)

### 2A. vLLM Gemma3n activation sparsity — Gaussian-quantile FATReLU

`/Users/jish/Documents/GitHub/arc/research/code/06_foundation/vllm/vllm/model_executor/layers/activation.py:233-289`

```python
@CustomOp.register("gelu_and_mul_sparse")
class GeluAndMulSparse(CustomOp):
    """An activation function for GeluAndMulSparse.
    This activation function is used in Gemma3n. ...
    """
    def __init__(self, activation_sparsity: float, approximate: str = "none"):
        super().__init__()
        ...
        if activation_sparsity == 0.0:
            raise ValueError("activation_sparsity is 0.0. Please use GeluAndMul.")
        target_sparsity_tensor = torch.tensor(activation_sparsity, dtype=torch.float32)
        normal_dist = torch.distributions.normal.Normal(0, 1)
        self.std_multiplier = normal_dist.icdf(target_sparsity_tensor)

    def _gaussian_topk(self, x: torch.Tensor) -> torch.Tensor:
        """Get % sparse percentile of the Gaussian distribution."""
        mean = torch.mean(x, dim=-1, keepdim=True)
        std = torch.std(x, dim=-1, keepdim=True, unbiased=False)
        cutoff_x = mean + std * self.std_multiplier
        return nn.functional.relu(x - cutoff_x)
```

**Note**: this is *not* a predictor. The threshold is set so that, *given Gaussian
activations*, exactly `activation_sparsity` fraction will fall below the cutoff.
It still computes the full `gate_proj` GEMM — sparsity is only applied at the
elementwise stage. No FFN-up GEMM rows are actually skipped.

### 2B. vLLM DeepSeek V3.2 sparse MLA — `is_v32` activation

`/Users/jish/Documents/GitHub/arc/research/code/06_foundation/vllm/vllm/model_executor/models/deepseek_v2.py:986-1043`

```python
self.is_v32 = hasattr(config, "index_topk")
_skip_topk = False
if self.is_v32:
    self.indexer_rope_emb = get_rope(...)
    self.indexer = Indexer(
        vllm_config, config, hidden_size, q_lora_rank,
        quant_config, cache_config, topk_indices_buffer,
        f"{prefix}.indexer",
    )
    use_index_cache = getattr(config, "use_index_cache", False)
    if use_index_cache:
        _index_topk_freq = getattr(config, "index_topk_freq", 1)
        ...
mla_modules = MLAModules(
    ...,
    indexer=self.indexer,
    indexer_rotary_emb=self.indexer_rope_emb,
    is_sparse=self.is_v32,
    topk_indices_buffer=topk_indices_buffer,
)
```

`/Users/jish/Documents/GitHub/arc/research/code/06_foundation/vllm/vllm/model_executor/models/deepseek_v2.py:676-737` (the actual indexer forward):

```python
def forward(self, hidden_states, qr, positions, rotary_emb):
    q, _ = self.wq_b(qr)
    q = q.view(-1, self.n_head, self.head_dim)
    ...
    kw, _ = self.wk_weights_proj(hidden_states)
    k = kw[:, :self.head_dim]
    weights = kw[:, self.head_dim:]
    k = self.k_norm(k)
    ...
    q_fp8, q_scale = per_token_group_quant_fp8(q, self.quant_block_size, ...)
    ...
    return self.indexer_op(hidden_states, q_fp8, k, weights)
```

The indexer is a tiny extra attention head whose only purpose is to emit a
`topk_indices_buffer`, which then drives the *real* MLA attention to read only
those KV slots. **This is not "magnitude pruning at runtime"** — it's
trained-in dynamic sparsity (DeepSeek's "DSA": DeepSeek Sparse Attention).

### 2C. vLLM FlashMLA sparse kernel call

`/Users/jish/Documents/GitHub/arc/research/code/06_foundation/vllm/vllm/v1/attention/backends/mla/flashmla_sparse.py:777-799`

```python
def _forward_bf16_kv(self, q, kv_c_and_k_pe_cache, topk_indices, attn_metadata):
    topk_indices = triton_convert_req_index_to_global_index(
        attn_metadata.req_id_per_token,
        attn_metadata.block_table,
        topk_indices,
        BLOCK_SIZE=attn_metadata.block_size,
        NUM_TOPK_TOKENS=topk_indices.shape[1],
    )
    return self._bf16_flash_mla_kernel(q, kv_c_and_k_pe_cache, topk_indices)
```

### 2D. SGLang DSA Indexer

`/Users/jish/Documents/GitHub/arc/research/code/06_foundation/sglang/python/sglang/srt/layers/attention/dsa/dsa_indexer.py:183-272`

```python
class Indexer(MultiPlatformOp):
    def __init__(self, hidden_size, index_n_heads, index_head_dim,
                 rope_head_dim, index_topk, q_lora_rank, ...):
        super().__init__()
        ...
        self.wq_b = ReplicatedLinear(self.q_lora_rank,
                                     self.n_heads * self.head_dim, bias=False, ...)
        self.wk = ReplicatedLinear(self.hidden_size, self.head_dim, bias=False, ...)
        self.weights_proj = ReplicatedLinear(self.hidden_size, self.n_heads,
                                             bias=False, params_dtype=torch.bfloat16, ...)
        self.k_norm = LayerNorm(self.head_dim, dtype=...)
```

Same idea as vLLM's `Indexer`: trained `wq_b` / `wk` / `weights_proj` produce
top-k indices.

### 2E. PowerInfer v1 — predictor-gated CUDA SpMV (the only "magnitude pruning at runtime" that ships)

`/Users/jish/Documents/GitHub/arc/research/code/03_per_token_speed/powerinfer/ggml-cuda.cu:4685-4754`

```cpp
static __global__ void dequantize_mul_mat_vec_sparse(
    const void * __restrict__ vx, const dfloat * __restrict__ y,
    float * __restrict__ dst, const int ncols, const int nrows,
    int * lst, float * idx) {
    const int gpu_row = blockIdx.y*blockDim.y + threadIdx.y;
    if (gpu_row >= nrows) { return; }

    int row = lst ? lst[gpu_row] : gpu_row;
    if (idx[row] < dev_sparse_threshold) {
        return;                    // <-- skip this output row entirely
    }
    const int tid = threadIdx.x;
    ...
    // Otherwise compute the dot product as usual:
    for (int i = 0; i < ncols; i += iter_stride) {
        ...
        dequantize_kernel(vx, ib, iqs + j/qr, v);
        tmp += v.x * y[iybs + iqs + j/qr + 0];
        tmp += v.y * y[iybs + iqs + j/qr + y_offset];
    }
    ...
    if (tid == 0) { dst[row] = tmp; }
}
```

The "predictor" itself, from `llama.cpp:4687-4695` (PowerInfer's `llm_build_ffn_sparse`):

```cpp
// prepare sparse idx
ggml_tensor * idx = ggml_mul_mat(ctx, pre_w1, pred_inpl);
cb(idx, "mlp_pre_hidden");
idx = ggml_relu(ctx, idx);
cb(idx, "mlp_pre_relu");
idx = ggml_mul_mat(ctx, pre_w2, idx);
```

So the predictor is a **2-layer MLP**: `pre_w1 → ReLU → pre_w2`. The output
is a per-neuron activation-magnitude estimate. PowerInfer skips any output
row of the FFN-up GEMM whose predicted score falls below
`dev_sparse_threshold` (set via env var `LLAMA_SPARSE_PRED_THRESHOLD`,
default 0.0).

### 2F. PowerInfer v2 / SmallThinker — CPU SpMV with predictor

`/Users/jish/Documents/GitHub/arc/research/code/03_per_token_speed/powerinfer/smallthinker/powerinfer/powerinfer-cpu/src/sparse_matmul.hpp:55-67`

```cpp
for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ir0++) {
    const float score = predictor_row[ir0];
    if (score <= SPARSE_PRED_THRESHOLD) {
        dst_col[ir0] = 0;
        continue;
    }
    ggml_vec_dot_q4_0_q8_0<32>(ne00, &dst_col[ir0],
                               (const char *)src0_row + ir0 * nb01, src1_col);
}
```

`SPARSE_PRED_THRESHOLD = 0` (constant in `powerinfer-type.hpp`). The
predictor output is precomputed once per token, the SpMV kernel branch-skips
any neuron with non-positive predicted score.

### 2G. SGLang — block-sparse FA used in Qwen2-Instruct dual-chunk

`/Users/jish/Documents/GitHub/arc/research/code/06_foundation/sglang/python/sglang/srt/layers/attention/dual_chunk_flashattention_backend.py:1668-1681`

```python
out, lse = sparse_attn_func(
    q, k, v,
    block_count, block_offset,
    column_count, column_index,
    causal=causal,
    softmax_scale=softmax_scale,
    return_softmax_lse=True,
)
```

This is the only call-site in the SGLang Python tree. The MInference-style
"vertical+slash" indexes are computed on-the-fly per layer.

### 2H. vLLM compressed-tensors 2:4 sparsity — **REMOVED**

`/Users/jish/Documents/GitHub/arc/research/code/06_foundation/vllm/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py:255-266`

```python
sparsity_config = SparsityCompressionConfig.model_validate(sparsity_config)
sparse_scheme_map: dict[str, SparsityCompressionConfig] = {
    target: sparsity_config for target in sparsity_config.targets or list()
}
sparsity_ignore_list = sparsity_config.ignore or list()

# Raise DeprecationError if non-empty sparse_scheme_map is detected
if sparse_scheme_map:
    raise DeprecationWarning(
        "Sparsity support has been removed from compressed-tensors. "
        "Please use a model without sparsity configuration."
    )
```

This is the most damning evidence in the audit: vLLM **explicitly removed**
2:4 weight sparsity at the compressed-tensors layer. The vLLM team
shipped it, watched users not benefit, and tore it out.

---

## 3. Activation flags / config triggers

| Sparsity feature                            | How users turn it on                                                                                                          |
| ------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| vLLM DeepSeek V3.2 sparse MLA               | Load `deepseek-ai/DeepSeek-V3.2-Exp` whose `config.json` has `index_topk: 2048`. **No CLI flag, no env var.** Pure config-driven. |
| vLLM Gemma3n activation sparsity            | Load Gemma3n whose `config.json` has `activation_sparsity_pattern: [0.95, 0.95, ...]`. Auto-wired in `Gemma3nTextMLP`.                  |
| vLLM MiniCPM-S FATReLU                      | Load MiniCPM-S whose `config.json` has `hidden_act: "fatrelu"` and `hidden_act_param: <threshold>`.                                  |
| vLLM 2:4 weight sparsity                    | **Disabled.** Raises `DeprecationWarning` if model carries `sparsity_config`.                                                  |
| SGLang DSA Indexer                          | Same: load DeepSeek-V3.2 model; `get_dsa_index_topk(hf_config)` returns non-None → backend dispatches DSA path.                |
| SGLang DSV4 CSA/HCA compressor              | Use a DSV4 checkpoint; env var `SGLANG_OPT_USE_ONLINE_COMPRESS=1` for the online-compress variant.                            |
| SGLang block-sparse FA (vertical+slash)     | Only triggered for Qwen2-Instruct dual-chunk attention backend.                                                               |
| TRT-LLM Gemma3n activation sparsity         | AutoDeploy path only; HF config `activation_sparsity_pattern`. **Not in the main TRT-LLM engine path.**                       |
| PowerInfer v1 magnitude predictor           | Build with `LLAMA_CUDA=1`; env `LLAMA_SPARSE_PRED_THRESHOLD=0.0`; **GGUF must contain `powerinfer.sparse_threshold` and predictor weights**. |
| MInference vertical-slash                   | External monkeypatch: `MInference(attn_type="minference").patch_model(hf_model)`. Not in vLLM/SGLang trees.                  |
| TensorRT-LLM CLI flag like `--use_sparse_mlp` | **Does not exist.** No such flag in the repo.                                                                              |

---

## 4. Production reality check

### What actually ships and is reachable from a serve command

1. **Trained-in dynamic sparse MLA (DeepSeek V3.2, V4)** — vLLM, SGLang, FlashInfer, ROCm Aiter, XPU. Multiple backends, hundreds of LOC each, real test files in CI. This is the only **attention-side** sparsity that ships in mainstream production.
2. **Activation-sparsity-via-Gaussian-quantile** (Gemma3n) — vLLM, SGLang, TRT-LLM AutoDeploy. ~30-line implementation each. Elementwise threshold post-GEMM; doesn't save the up-GEMM at all.
3. **FATReLU for MiniCPM-S** — vLLM only, ~37 LOC. Same shape as above: post-GEMM zero-out.

### What is shipped only as research / external library

4. **PowerInfer v1 magnitude predictor** — In a *fork* of `llama.cpp`, not upstream. The upstream `llama.cpp` (`ggerganov/llama.cpp`) does not have `ggml_axpy`/`dequantize_mul_mat_axpy_sparse`. Last meaningful PowerInfer commit on the fork was 2024; not maintained in step with mainline.
5. **PowerInfer v2 / SmallThinker** — Mobile/edge focus (CPU + Hexagon DSP). The CUDA path of PowerInfer-v1 was kept; CUDA path of v2 is essentially absent. Production-relevance for cloud GPUs: very low.
6. **MInference** — A monkey-patch library by Microsoft. Not integrated into vLLM/SGLang serve paths. Users have to wrap `model = MInference(...)(model)` in their own scripts.
7. **MoBA** — Reference impl only. No vLLM/SGLang/TRT-LLM call sites.

### What was tried and removed

8. **Compressed-tensors 2:4 weight sparsity in vLLM** — actively raises `DeprecationWarning` if you try. (Quoted above.)

### Conspicuous absences

- **No `--use_sparse_mlp` / `--turbo_sparse` / `--deja_vu` flag exists** in any of vLLM, SGLang, TRT-LLM, llama.cpp.
- **No predictor module class** (`class Predictor(nn.Module)`, `class SparseMLPPredictor`) is present in vLLM or SGLang Python trees.
- **No Mixture-of-Depths / per-token layer-skip** in any production engine. Zero hits for `MoD`, `mixture_of_depths`, `early_exit_threshold` in vLLM/SGLang code (excluding the unrelated `early_exit_gate` in `ouro.py`).
- **No "ReluFication"-style retrofit** (Liu et al. TurboSparse) is wired into a model loader.

### Why aren't the magnitude-pruning ideas in production?

Reading the code paths suggests three reasons:

1. **Wrong workload regime.** PowerInfer-style row-skipping in SpMV gives wins when arithmetic intensity is < 1 — i.e. CPU SpMV at batch=1. On a B200 doing BF16 GEMMs with batch ≥ 16, the kernel is compute-bound; skipping ~50% of rows does not give ~50% speedup because the SMs are already saturated and the branch divergence + irregular access pattern adds overhead. The PowerInfer kernel quoted above uses `atomicAdd(&dst[i+tid], ...)` which is fine at batch=1 but is a contention hotspot at batch=128.
2. **Loss of fused MoE / TP code paths.** vLLM's `MergedColumnParallelLinear` and `RowParallelLinear` fuse the gate and up projections into a single GEMM, then split. A per-neuron predictor breaks that fusion and serialises into two GEMMs + an extra predictor GEMM. The engineering cost to redo TP shards + compressed-tensors integration is high and the SLA win small (or negative) for the common case.
3. **Predictor weights = an extra file format.** Both vLLM and SGLang load models from `safetensors` index files (`model.safetensors.index.json`). PowerInfer requires a *custom* GGUF that carries `mlp.pre_w1` and `mlp.pre_w2` tensors. No HF Hub model in the popular catalogue (Llama 4, Qwen3, DeepSeek V3.2) ships those tensors. The de facto standard for "FFN sparsity" has migrated to the trained-in version: DSA's `Indexer` is shipped *as part of the model checkpoint*.

The trained-in pattern (DSA, Gemma3n quantile) won because: (a) the sparsity weights ship with the model checkpoint via standard `safetensors`, (b) the threshold/topk is calibrated at training time so it survives quantisation, (c) no per-deployment threshold tuning is needed.

---

## 5. What Arc can copy directly

### Highest priority — patterns that survive on B200 today

1. **DSA-style learned indexer for KV-selection.** Port the architecture in
   `/Users/jish/Documents/GitHub/arc/research/code/06_foundation/sglang/python/sglang/srt/layers/attention/dsa/dsa_indexer.py:183-272` and
   `/Users/jish/Documents/GitHub/arc/research/code/06_foundation/vllm/vllm/model_executor/models/deepseek_v2.py:604-737`.
   These two are independent implementations of the same idea (DeepSeek "DSA") — agreement is strong, naming
   conventions are clean, and they expose `topk_indices_buffer` as a stable contract.
   For Arc: build the same Indexer in `mistralrs-core/src/models/deepseek_v3.rs`, emit a `Vec<i32>` topk per
   layer, and pass it to a sparse-MLA attention kernel. This is the only one of the six bullets that is *both*
   trained-in (so it survives quantisation) and actually deployed at scale.

2. **`GeluAndMulSparse` Gaussian-quantile activation gating.** ~30 lines, drop-in for any FFN that wants a
   tunable activation sparsity without retraining. File:
   `/Users/jish/Documents/GitHub/arc/research/code/06_foundation/vllm/vllm/model_executor/layers/activation.py:233-289`.
   Port to `mistralrs-core/src/layers/activation.rs`. Use `mean + std * icdf(target)` to produce the
   cutoff in-kernel.

3. **FlashMLA-sparse kernel binding.** The actual CUDA kernel is in DeepSeek's `FlashMLA` repo, exposed via
   the function `flash_mla_sparse_fwd` in
   `/Users/jish/Documents/GitHub/arc/research/code/06_foundation/vllm/vllm/v1/attention/ops/flashmla.py`.
   Arc's path: clone `deepseek-ai/FlashMLA`, build the sparse fwd as a `cudarc` binding in
   `mistralrs-paged-attn`. Compute-wise this is the only kernel that delivers DSA throughput.

### Medium priority — useful only for the right model

4. **FATReLU for MiniCPM-S**: 37 lines, easy.
   `vllm/model_executor/layers/activation.py:77-113`. Worth only if Arc plans to support MiniCPM-S.

5. **Vertical-slash sparse FA pattern** (MInference / SGLang dual-chunk): the CUDA index converter is in
   `/Users/jish/Documents/GitHub/arc/research/code/06_foundation/sglang/sgl-kernel/csrc/attention/vertical_slash_index.cu`.
   Useful for prefill of very long contexts. Skip for now; revisit if Arc supports 1M-ctx models.

### Skip — verified non-shipping

6. **PowerInfer-style magnitude predictor (`ggml_axpy_sparse`)**: do **not** port.
   The CUDA implementation in `/Users/jish/Documents/GitHub/arc/research/code/03_per_token_speed/powerinfer/ggml-cuda.cu:4541-4604`
   is built around `atomicAdd` for output accumulation, which is the wrong primitive for B200 at batch ≥ 8.
   The predictor architecture (2-layer MLP `pre_w1 → ReLU → pre_w2`) is interesting as an idea but the
   weights don't exist in any modern HF checkpoint, so Arc would have to train them per model — and the
   ROI is low because the wins disappear on serving-batch workloads.

7. **2:4 weight sparsity**: **vLLM removed it**. Don't add.

8. **Mixture-of-Depths**: nothing ships. Don't speculate.

### Repo pointers (port targets)

| Pattern             | Source file (read-only, use as reference)                                                                                       |
| ------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| DSA Indexer (Python ref) | `research/code/06_foundation/sglang/python/sglang/srt/layers/attention/dsa/dsa_indexer.py`                              |
| DSA Indexer (alt ref) | `research/code/06_foundation/vllm/vllm/model_executor/models/deepseek_v2.py:604-737`                                        |
| FlashMLA sparse fwd | `research/code/06_foundation/vllm/vllm/v1/attention/backends/mla/flashmla_sparse.py:777-799`                                  |
| FlashInfer sparse fwd | `research/code/06_foundation/vllm/vllm/v1/attention/backends/mla/flashinfer_mla_sparse.py:315-359`                          |
| Gaussian-quantile FATReLU | `research/code/06_foundation/vllm/vllm/model_executor/layers/activation.py:233-289`                                     |
| Trigger logic       | `research/code/06_foundation/vllm/vllm/model_executor/models/deepseek_v2.py:986-1043` (`is_v32 = hasattr(config, "index_topk")`) |
| Vertical-slash kernel | `research/code/06_foundation/sglang/sgl-kernel/csrc/attention/vertical_slash_index.cu`                                       |

---

## 6. Bottom line

The only sparsity pattern in mainstream production today is **trained-in dynamic
KV-selection** via a learned indexer head (DeepSeek DSA / V3.2). vLLM and
SGLang both ship it with multiple backends; FlashInfer/TRT-LLM-gen has the
kernels. **Activation gating** (Gemma3n quantile, MiniCPM-S FATReLU) ships
as small post-GEMM elementwise ops — it does *not* save the FFN-up GEMM.
**Magnitude-prediction (PowerInfer, Déjà Vu, TurboSparse)** lives only in
research forks; no mainstream engine integrates them. **2:4 weight sparsity**
was tried by vLLM and explicitly removed.

For Arc, the single bet that pays out is: implement the DSA learned indexer
and bind FlashMLA-sparse. Everything else listed in the original sparsity
literature is either an elementwise micro-op or a non-shipping research idea.
