/**
 * Parent system: ArcKernels.
 *
 * Link-only stubs for the WMMA blockwise-FP8 GEMM on compute capability < 8.0.
 *
 * UNVERIFIED ON HARDWARE -- never run. (Nothing in this wave has executed.)
 *
 * WHY THIS FILE EXISTS
 * --------------------
 * `build.rs` excludes `*_wmma.cu` from the kernel set when the compute cap is
 * below 8.0 -- BF16 WMMA fragments need sm_80 -- and excludes `*_dummy.cu`
 * when it is 8.0 or above. So exactly one of this file and
 * blockwise_fp8_gemm_wmma.cu is compiled, and the `extern "C"` symbols the
 * Rust FFI declares resolve either way.
 *
 * Nothing here is ever reached at runtime: `has_blockwise_fp8_kernels` is
 * only emitted for cc >= 8.0, so `HAVE_BLOCKWISE_GEMM_KERNELS` is false on
 * the builds that link these stubs and the Rust dispatcher never calls them.
 * They exist because an rlib build does not link, so a missing symbol here
 * would surface only when a binary is finally linked -- on a rented box.
 *
 * Mirrors blockwise_fp8_gemm_dummy.cu, which does the same job for the scalar
 * kernels next door.
 */

#include <cstdint>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>

extern "C" void launch_fp8_matmul_wmma_f16(const __half *input,
                                           const void *weight, // __nv_fp8_e4m3*
                                           const float *weight_scale,
                                           __half *output, int M, int N, int K,
                                           int scale_row_stride,
                                           int block_size_y, int block_size_x,
                                           cudaStream_t stream) {
  fprintf(stderr, "FP8 WMMA matmul not supported on this GPU (requires compute "
                  "capability >= 8.0)\n");
}

extern "C" void launch_fp8_matmul_wmma_bf16(
    const __nv_bfloat16 *input,
    const void *weight, // __nv_fp8_e4m3*
    const float *weight_scale, __nv_bfloat16 *output, int M, int N, int K,
    int scale_row_stride, int block_size_y, int block_size_x,
    cudaStream_t stream) {
  fprintf(stderr, "FP8 WMMA matmul not supported on this GPU (requires compute "
                  "capability >= 8.0)\n");
}

// Must still report the tiling the Rust dispatcher asks about at startup. The
// values are the ones in blockwise_fp8_gemm_wmma.cu; they are only used to
// decide eligibility, and on this build the kernel is never selected anyway.
extern "C" void fp8_matmul_wmma_tile_dims(int *m_blk, int *n_blk, int *k_blk) {
  *m_blk = 64;
  *n_blk = 64;
  *k_blk = 128;
}
