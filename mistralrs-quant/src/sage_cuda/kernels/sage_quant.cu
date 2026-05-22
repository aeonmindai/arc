/*
 * Copyright (c) 2024 by SageAttention team. Vendored into Arc.
 *
 * Torch-free FFI entry points for the per-block INT8 quantization kernels
 * used to prepare Q and K tensors for the SageAttention kernel.
 *
 * Licensed under the Apache License, Version 2.0 (the "License").
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "fused_kernels.cuh"

namespace {

template <typename c_type, int BLOCK_SIZE, int HEAD_DIM, bool HAS_SM_SCALE>
void launch_quant_per_block_int8(
    const c_type* input, int8_t* output, float* scale, float sm_scale,
    int batch_size, int num_tokens, int num_heads,
    int stride_bz_input, int stride_seq_input, int stride_h_input,
    int stride_bz_output, int stride_seq_output, int stride_h_output,
    int stride_bz_scale, int stride_h_scale, cudaStream_t stream) {
  dim3 grid((num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE, num_heads, batch_size);

  constexpr int num_pack_per_thread =
      (BLOCK_SIZE * (HEAD_DIM / 8) + 1023) / 1024;
  dim3 block(BLOCK_SIZE * (HEAD_DIM / 8) / num_pack_per_thread);

  QuantInt8Kernel<HEAD_DIM, BLOCK_SIZE, num_pack_per_thread, HAS_SM_SCALE,
                  /*sub_mean=*/false, c_type>
      <<<grid, block, 0, stream>>>(
          const_cast<c_type*>(input),
          /*mean=*/nullptr,
          output,
          scale,
          sm_scale,
          static_cast<uint32_t>(num_tokens),
          static_cast<uint32_t>(stride_bz_input),
          static_cast<uint32_t>(stride_seq_input),
          static_cast<uint32_t>(stride_h_input),
          /*stride_bz_mean=*/0,
          /*stride_h_mean=*/0,
          static_cast<uint32_t>(stride_bz_output),
          static_cast<uint32_t>(stride_seq_output),
          static_cast<uint32_t>(stride_h_output),
          static_cast<uint32_t>(stride_bz_scale),
          static_cast<uint32_t>(stride_h_scale));
}

template <typename c_type, bool HAS_SM_SCALE>
int dispatch_quant_per_block(
    const c_type* input, int8_t* output, float* scale, float sm_scale,
    int batch_size, int num_tokens, int num_heads,
    int head_dim, int block_size,
    int stride_bz_input, int stride_seq_input, int stride_h_input,
    int stride_bz_output, int stride_seq_output, int stride_h_output,
    int stride_bz_scale, int stride_h_scale, cudaStream_t stream) {
  try {
    DISPATCH_BLOCK_SIZE(block_size, BLOCK_SIZE, {
      DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
        launch_quant_per_block_int8<c_type, BLOCK_SIZE, HEAD_DIM,
                                    HAS_SM_SCALE>(
            input, output, scale, sm_scale, batch_size, num_tokens, num_heads,
            stride_bz_input, stride_seq_input, stride_h_input,
            stride_bz_output, stride_seq_output, stride_h_output,
            stride_bz_scale, stride_h_scale, stream);
      });
    });
  } catch (const std::invalid_argument& e) {
    return 1;
  }
  return 0;
}

} // anonymous namespace

extern "C" {

// `input_dtype`: 0 = half (FP16), 1 = nv_bfloat16
// `tensor_layout`: 0 = BSHD, 1 = BHSD — already encoded in caller-provided
// strides, so this is informational.
int sage_quant_per_block_int8(
    const void* input, int8_t* output, float* scale,
    int batch_size, int num_tokens, int num_heads,
    int head_dim, int block_size,
    int tensor_layout, int input_dtype,
    int stride_bz_input, int stride_seq_input, int stride_h_input,
    int stride_bz_output, int stride_seq_output, int stride_h_output,
    int stride_bz_scale, int stride_h_scale, cudaStream_t stream) {
  (void)tensor_layout;
  if (input_dtype == 0) {
    return dispatch_quant_per_block<half, /*HAS_SM_SCALE=*/false>(
        reinterpret_cast<const half*>(input), output, scale,
        /*sm_scale=*/0.0f, batch_size, num_tokens, num_heads, head_dim,
        block_size,
        stride_bz_input, stride_seq_input, stride_h_input,
        stride_bz_output, stride_seq_output, stride_h_output,
        stride_bz_scale, stride_h_scale, stream);
  } else if (input_dtype == 1) {
    return dispatch_quant_per_block<nv_bfloat16, /*HAS_SM_SCALE=*/false>(
        reinterpret_cast<const nv_bfloat16*>(input), output, scale,
        /*sm_scale=*/0.0f, batch_size, num_tokens, num_heads, head_dim,
        block_size,
        stride_bz_input, stride_seq_input, stride_h_input,
        stride_bz_output, stride_seq_output, stride_h_output,
        stride_bz_scale, stride_h_scale, stream);
  }
  return 2; // unsupported dtype
}

// Variant that pre-multiplies the input by `sm_scale` before quantizing.
// Useful for folding the softmax 1/sqrt(d_head) factor into the quantization.
int sage_quant_per_block_int8_with_scale(
    const void* input, int8_t* output, float* scale, float sm_scale,
    int batch_size, int num_tokens, int num_heads,
    int head_dim, int block_size,
    int tensor_layout, int input_dtype,
    int stride_bz_input, int stride_seq_input, int stride_h_input,
    int stride_bz_output, int stride_seq_output, int stride_h_output,
    int stride_bz_scale, int stride_h_scale, cudaStream_t stream) {
  (void)tensor_layout;
  if (input_dtype == 0) {
    return dispatch_quant_per_block<half, /*HAS_SM_SCALE=*/true>(
        reinterpret_cast<const half*>(input), output, scale,
        sm_scale, batch_size, num_tokens, num_heads, head_dim, block_size,
        stride_bz_input, stride_seq_input, stride_h_input,
        stride_bz_output, stride_seq_output, stride_h_output,
        stride_bz_scale, stride_h_scale, stream);
  } else if (input_dtype == 1) {
    return dispatch_quant_per_block<nv_bfloat16, /*HAS_SM_SCALE=*/true>(
        reinterpret_cast<const nv_bfloat16*>(input), output, scale,
        sm_scale, batch_size, num_tokens, num_heads, head_dim, block_size,
        stride_bz_input, stride_seq_input, stride_h_input,
        stride_bz_output, stride_seq_output, stride_h_output,
        stride_bz_scale, stride_h_scale, stream);
  }
  return 2; // unsupported dtype
}

} // extern "C"
