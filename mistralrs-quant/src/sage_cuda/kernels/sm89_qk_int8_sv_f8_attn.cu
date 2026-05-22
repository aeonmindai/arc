/*
 * Copyright (c) 2024 by SageAttention team. Vendored into Arc.
 *
 * Torch-free FFI entry point. Mirrors the original
 * `sm89_qk_int8_sv_f8_accum_f32_attn_inst_buf` torch wrapper.
 *
 * All shape and contiguity checks are the responsibility of the Rust caller.
 *
 * Licensed under the Apache License, Version 2.0 (the "License").
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "qk_int_sv_f8_cuda_sm89.cuh"

// Forward declaration of the template kernel lives in the .cuh header.

namespace {

// SageAttention SM89 INT8 QK + FP8 SV launcher
//
// Layout `tensor_layout`:
//   0 = (B, S, H, D)  — "BSHD"
//   1 = (B, H, S, D)  — "BHSD"
//
// All strides are in elements (not bytes).
//
// `out_dtype`: 0 = half (FP16), 1 = nv_bfloat16
template <typename DTypeOut, int HEAD_DIM, bool IS_CAUSAL, int QK_QUANT_GRAN,
          bool RETURN_LSE, bool USE_INST_BUF>
void launch_sm89_qk_int8_sv_f8(
    const int8_t* query, const int8_t* key, const int8_t* value,
    DTypeOut* output, float* lse, const float* query_scale,
    const float* key_scale,
    int batch_size, int qo_len, int kv_len,
    int num_qo_heads, int num_kv_heads,
    int stride_bz_q, int stride_seq_q, int stride_h_q,
    int stride_bz_k, int stride_seq_k, int stride_h_k,
    int stride_bz_v, int stride_h_v, int stride_d_v,
    int stride_bz_o, int stride_seq_o, int stride_h_o,
    float sm_scale, cudaStream_t stream) {
  constexpr int CTA_Q = 128;
  constexpr int CTA_K = 64;
  constexpr int WARP_Q = 32;
  constexpr int WARP_K = 64;

  constexpr MaskMode mask_mode = IS_CAUSAL ? MaskMode::kCausal : MaskMode::kNone;

  size_t smem_max = std::max(
      CTA_Q * HEAD_DIM * sizeof(int8_t)
          + CTA_K * HEAD_DIM * sizeof(int8_t)
          + CTA_K * HEAD_DIM * sizeof(int8_t),
      CTA_Q * HEAD_DIM * sizeof(half));

  auto kernel_func = qk_int_sv_f8_attn_kernel<
      CTA_Q, CTA_K, WARP_Q, WARP_K, HEAD_DIM,
      DataType::kInt8,
      static_cast<QuantGranularity>(QK_QUANT_GRAN),
      static_cast<QuantGranularity>(QK_QUANT_GRAN),
      float, USE_INST_BUF, DTypeOut, ComputeUnit::kCudaCore,
      mask_mode, RETURN_LSE,
      /*fuse_v_scale=*/false,
      /*fuse_v_mean=*/false,
      /*use_pv_fp16_accu=*/false>;

  cudaFuncSetAttribute(kernel_func,
                       cudaFuncAttributeMaxDynamicSharedMemorySize, smem_max);

  const int num_kv_groups = num_qo_heads / num_kv_heads;

  dim3 grid(div_ceil(qo_len, CTA_Q), num_qo_heads, batch_size);
  dim3 block(32, (CTA_Q / WARP_Q) * (CTA_K / WARP_K));

  kernel_func<<<grid, block, smem_max, stream>>>(
      const_cast<int8_t*>(query),
      const_cast<int8_t*>(key),
      const_cast<int8_t*>(value),
      output, lse,
      const_cast<float*>(query_scale),
      const_cast<float*>(key_scale),
      nullptr, nullptr,
      static_cast<uint32_t>(qo_len),
      static_cast<uint32_t>(kv_len),
      static_cast<uint32_t>(num_kv_groups),
      static_cast<uint32_t>(stride_bz_q),
      static_cast<uint32_t>(stride_seq_q),
      static_cast<uint32_t>(stride_h_q),
      static_cast<uint32_t>(stride_bz_k),
      static_cast<uint32_t>(stride_seq_k),
      static_cast<uint32_t>(stride_h_k),
      static_cast<uint32_t>(stride_bz_v),
      static_cast<uint32_t>(stride_h_v),
      static_cast<uint32_t>(stride_d_v),
      static_cast<uint32_t>(stride_bz_o),
      static_cast<uint32_t>(stride_seq_o),
      static_cast<uint32_t>(stride_h_o),
      sm_scale);
}

// Single dispatch helper templated only on output dtype + USE_INST_BUF, with
// remaining specialization done by runtime branching on small dispatches.
template <typename DTypeOut, bool USE_INST_BUF>
int dispatch_inner(
    const int8_t* query, const int8_t* key, const int8_t* value,
    DTypeOut* output, float* lse, const float* query_scale,
    const float* key_scale,
    int batch_size, int qo_len, int kv_len,
    int num_qo_heads, int num_kv_heads,
    int head_dim, int is_causal, int qk_quant_gran, int return_lse,
    int stride_bz_q, int stride_seq_q, int stride_h_q,
    int stride_bz_k, int stride_seq_k, int stride_h_k,
    int stride_bz_v, int stride_h_v, int stride_d_v,
    int stride_bz_o, int stride_seq_o, int stride_h_o,
    float sm_scale, cudaStream_t stream) {
  try {
    DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
      DISPATCH_CAUSAL(is_causal, IS_CAUSAL, {
        DISPATCH_QK_QUANT_GRAN(qk_quant_gran, QK_QUANT_GRAN, {
          DISPATCH_RETURN_LSE(return_lse, RETURN_LSE, {
            launch_sm89_qk_int8_sv_f8<
                DTypeOut, HEAD_DIM, IS_CAUSAL, QK_QUANT_GRAN, RETURN_LSE,
                USE_INST_BUF>(
                query, key, value, output, lse, query_scale, key_scale,
                batch_size, qo_len, kv_len, num_qo_heads, num_kv_heads,
                stride_bz_q, stride_seq_q, stride_h_q,
                stride_bz_k, stride_seq_k, stride_h_k,
                stride_bz_v, stride_h_v, stride_d_v,
                stride_bz_o, stride_seq_o, stride_h_o,
                sm_scale, stream);
          });
        });
      });
    });
  } catch (const std::invalid_argument& e) {
    return 1;
  }
  return 0;
}

} // anonymous namespace

extern "C" {

// `out_dtype`: 0 = half, 1 = bf16
// Returns 0 on success, non-zero on dispatch error.
int sage_attention_sm89_int8_f8_accum_f32(
    const int8_t* query, const int8_t* key, const int8_t* value,
    void* output, float* lse, const float* query_scale,
    const float* key_scale,
    int batch_size, int qo_len, int kv_len,
    int num_qo_heads, int num_kv_heads,
    int head_dim, int tensor_layout,
    int is_causal, int qk_quant_gran, int return_lse, int out_dtype,
    int stride_bz_q, int stride_seq_q, int stride_h_q,
    int stride_bz_k, int stride_seq_k, int stride_h_k,
    int stride_bz_v, int stride_h_v, int stride_d_v,
    int stride_bz_o, int stride_seq_o, int stride_h_o,
    float sm_scale, cudaStream_t stream) {
  (void)tensor_layout; // strides already encode layout
  if (out_dtype == 0) {
    return dispatch_inner<half, /*USE_INST_BUF=*/false>(
        query, key, value, reinterpret_cast<half*>(output), lse, query_scale,
        key_scale, batch_size, qo_len, kv_len, num_qo_heads, num_kv_heads,
        head_dim, is_causal, qk_quant_gran, return_lse,
        stride_bz_q, stride_seq_q, stride_h_q,
        stride_bz_k, stride_seq_k, stride_h_k,
        stride_bz_v, stride_h_v, stride_d_v,
        stride_bz_o, stride_seq_o, stride_h_o,
        sm_scale, stream);
  } else if (out_dtype == 1) {
    return dispatch_inner<nv_bfloat16, /*USE_INST_BUF=*/false>(
        query, key, value, reinterpret_cast<nv_bfloat16*>(output), lse,
        query_scale, key_scale, batch_size, qo_len, kv_len, num_qo_heads,
        num_kv_heads, head_dim, is_causal, qk_quant_gran, return_lse,
        stride_bz_q, stride_seq_q, stride_h_q,
        stride_bz_k, stride_seq_k, stride_h_k,
        stride_bz_v, stride_h_v, stride_d_v,
        stride_bz_o, stride_seq_o, stride_h_o,
        sm_scale, stream);
  }
  return 2; // unsupported dtype
}

// "inst_buf" variant: the original sm89 kernel with use_inst_buffer=true,
// which gives better register reuse on Ada.
int sage_attention_sm89_int8_f8_accum_f32_inst_buf(
    const int8_t* query, const int8_t* key, const int8_t* value,
    void* output, float* lse, const float* query_scale,
    const float* key_scale,
    int batch_size, int qo_len, int kv_len,
    int num_qo_heads, int num_kv_heads,
    int head_dim, int tensor_layout,
    int is_causal, int qk_quant_gran, int return_lse, int out_dtype,
    int stride_bz_q, int stride_seq_q, int stride_h_q,
    int stride_bz_k, int stride_seq_k, int stride_h_k,
    int stride_bz_v, int stride_h_v, int stride_d_v,
    int stride_bz_o, int stride_seq_o, int stride_h_o,
    float sm_scale, cudaStream_t stream) {
  (void)tensor_layout;
  if (out_dtype == 0) {
    return dispatch_inner<half, /*USE_INST_BUF=*/true>(
        query, key, value, reinterpret_cast<half*>(output), lse, query_scale,
        key_scale, batch_size, qo_len, kv_len, num_qo_heads, num_kv_heads,
        head_dim, is_causal, qk_quant_gran, return_lse,
        stride_bz_q, stride_seq_q, stride_h_q,
        stride_bz_k, stride_seq_k, stride_h_k,
        stride_bz_v, stride_h_v, stride_d_v,
        stride_bz_o, stride_seq_o, stride_h_o,
        sm_scale, stream);
  } else if (out_dtype == 1) {
    return dispatch_inner<nv_bfloat16, /*USE_INST_BUF=*/true>(
        query, key, value, reinterpret_cast<nv_bfloat16*>(output), lse,
        query_scale, key_scale, batch_size, qo_len, kv_len, num_qo_heads,
        num_kv_heads, head_dim, is_causal, qk_quant_gran, return_lse,
        stride_bz_q, stride_seq_q, stride_h_q,
        stride_bz_k, stride_seq_k, stride_h_k,
        stride_bz_v, stride_h_v, stride_d_v,
        stride_bz_o, stride_seq_o, stride_h_o,
        sm_scale, stream);
  }
  return 2; // unsupported dtype
}

} // extern "C"
