// EXHAUSTIVE proof that arc_kvquant.cu's E4M3 conversion IS NVIDIA's.
//
// Parent system: ArcQuant / TurboQuant (ArcInfer / ArcKV / Fp8).
//
// D33 — a check you have not seen fail is not a check. This sweeps ALL 2^32
// f32 bit patterns (not a sample) and compares three implementations:
//
//   arc::arc_f32_to_e4m3_code(x)                      the transcription the
//                                                     fused kernel ships
//   __nv_cvt_double_to_fp8((double)x, SATFINITE, E4M3) NVIDIA's SOFTWARE
//                                                     reference — the routine
//                                                     the Rust `float8` crate
//                                                     ports as `convert_to_fp8`,
//                                                     hence exactly what
//                                                     `F8E4M3::from_f32`, and
//                                                     therefore candle's CPU
//                                                     cast, computes
//   __nv_cvt_float_to_fp8(x, SATFINITE, E4M3)          NVIDIA's HARDWARE path
//                                                     (cvt.rn.satfinite.e4m3x2.f32
//                                                     on sm_89+)
//
// One mismatch against the software reference is a bug in the kernel. Software
// vs hardware is counted separately so the two can never be confused.
//
// Two guards make a green result mean something:
//   * `visited` counts the values actually processed; a sweep that silently
//     ran zero iterations would otherwise report "0 mismatches" and look
//     identical to a pass.
//   * `-DMUTANT=1` runs the same sweep against the deliberately-wrong
//     truncating variant. If THAT also reports 0 mismatches the sweep is
//     vacuous and the exit code says so.
//
// Build + run (needs only nvcc + a GPU, not the cargo build):
//   nvcc -std=c++17 -O3 -arch=sm_90 --use_fast_math --expt-relaxed-constexpr \
//     -U__CUDA_NO_BFLOAT16_CONVERSIONS__ -DMUTANT=0 \
//     -I<repo>/mistralrs-quant/kernels/arc_kvquant e4m3_exhaustive.cu -o sweep0
// Exit 0 pass, 1 mismatch, 2 environment/vacuous.
#ifndef MUTANT
#define MUTANT 0
#endif

#include <cuda_fp8.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "arc_kvquant.cu"

__global__ void sweep(unsigned long long base, unsigned long long n,
                      unsigned long long *mismatch_sw,
                      unsigned long long *mismatch_hw, unsigned *first_bad,
                      unsigned long long *visited) {
  for (unsigned long long i =
           blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
       i < n; i += (unsigned long long)gridDim.x * blockDim.x) {
    const unsigned bits = (unsigned)(base + i);
    const float x = __uint_as_float(bits);
    const uint8_t mine = MUTANT ? arc::arc_f32_to_e4m3_code_truncating(x)
                                : arc::arc_f32_to_e4m3_code(x);
    const uint8_t sw =
        (uint8_t)__nv_cvt_double_to_fp8((double)x, __NV_SATFINITE, __NV_E4M3);
    const uint8_t hw =
        (uint8_t)__nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3);
    if (mine != sw) {
      if (atomicAdd(mismatch_sw, 1ULL) == 0ULL) {
        *first_bad = bits;
      }
    }
    if (sw != hw) {
      atomicAdd(mismatch_hw, 1ULL);
    }
    atomicAdd(visited, 1ULL);
  }
}

int main() {
  unsigned long long *d_sw, *d_hw, *d_vis;
  unsigned *d_first;
  if (cudaMalloc(&d_sw, 8) != cudaSuccess || cudaMalloc(&d_hw, 8) != cudaSuccess ||
      cudaMalloc(&d_vis, 8) != cudaSuccess ||
      cudaMalloc(&d_first, 4) != cudaSuccess) {
    printf("FATAL cudaMalloc\n");
    return 2;
  }
  cudaMemset(d_sw, 0, 8);
  cudaMemset(d_hw, 0, 8);
  cudaMemset(d_vis, 0, 8);
  cudaMemset(d_first, 0, 4);

  const unsigned long long TOTAL = 1ULL << 32;
  const unsigned long long CHUNK = 1ULL << 28;
  for (unsigned long long base = 0; base < TOTAL; base += CHUNK) {
    sweep<<<8192, 256>>>(base, CHUNK, d_sw, d_hw, d_first, d_vis);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
      printf("FATAL cuda: %s\n", cudaGetErrorString(e));
      return 2;
    }
  }

  unsigned long long sw = 0, hw = 0, vis = 0;
  unsigned first = 0;
  cudaMemcpy(&sw, d_sw, 8, cudaMemcpyDeviceToHost);
  cudaMemcpy(&hw, d_hw, 8, cudaMemcpyDeviceToHost);
  cudaMemcpy(&vis, d_vis, 8, cudaMemcpyDeviceToHost);
  cudaMemcpy(&first, d_first, 4, cudaMemcpyDeviceToHost);

  printf("MUTANT                %d\n", MUTANT);
  printf("visited               %llu of %llu f32 bit patterns\n", vis, TOTAL);
  if (vis != TOTAL) {
    printf("FATAL the sweep did not visit every value; a 0 here means nothing\n");
    return 2;
  }
  printf("mismatch vs NVIDIA sw %llu\n", sw);
  printf("mismatch sw vs hw     %llu\n", hw);

  if (MUTANT) {
    if (sw == 0) {
      printf("FATAL VACUOUS: the deliberately-wrong kernel also matched on "
             "every input, so this sweep cannot fail and proves nothing\n");
      return 2;
    }
    printf("NEGATIVE CONTROL OK: the wrong kernel is caught on %llu inputs "
           "(%.2f%%)\n",
           sw, 100.0 * (double)sw / (double)TOTAL);
    return 0;
  }
  if (sw) {
    float x;
    memcpy(&x, &first, 4);
    printf("first bad input       0x%08x (%.9g)\n", first, x);
    return 1;
  }
  printf("RESULT: arc_f32_to_e4m3_code is bit-identical to NVIDIA's E4M3 "
         "conversion on every one of the 2^32 f32 values.\n");
  return 0;
}
