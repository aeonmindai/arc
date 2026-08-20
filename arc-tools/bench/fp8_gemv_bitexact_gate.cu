// Merge gate for ad1c738c0 / c59e7a5df: is fp8_gemv_warp_mlp bit-identical to
// the original fused fp8_gemv_warp?
//
// Compared at F32 BEFORE any narrowing: both kernels here write the raw f32
// accumulator, not the bf16 the production kernels emit. bf16 has 8 mantissa
// bits and would swallow exactly the reassociation this gate exists to catch.
//
// Negative control: a single-element 1-ULP perturbation of the REFERENCE
// OUTPUT at a known index. Deliberately NOT a perturbation of a shared input --
// a shared input moves both paths together by construction, so its control can
// never fire and the gate reads green while testing nothing.
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
  fprintf(stderr,"ENVFAIL %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(2);} }while(0)

__device__ __forceinline__ float fp8_to_float(__nv_fp8_e4m3 v) {
  return __half2float(__nv_cvt_fp8_to_halfraw(v.__x, __NV_E4M3));
}

// ---- ORIGINAL: one 128 B request per warp, consumed immediately -----------
template <int ROWS_PER_BLOCK>
__global__ void gemv_orig(const __nv_bfloat16 *__restrict__ input,
                          const __nv_fp8_e4m3 *__restrict__ weight,
                          const float *__restrict__ weight_scale,
                          float *__restrict__ out, int M, int N, int K,
                          int scale_row_stride, int block_size_y, int block_size_x) {
  const int lane = threadIdx.x & 31;
  const int n = blockIdx.x * ROWS_PER_BLOCK + (threadIdx.x >> 5);
  const int m = blockIdx.y;
  if (n >= N || m >= M) return;
  const __nv_fp8_e4m3 *w_row = weight + (size_t)n * K;
  const __nv_bfloat16 *in_row = input + (size_t)m * K;
  const int scale_row_offset = (n / block_size_y) * scale_row_stride;
  float acc = 0.0f;
  const int K_aligned = (K / 128) * 128;
  for (int k_base = 0; k_base < K_aligned; k_base += 128) {
    const int k = k_base + lane * 4;
    const uint32_t w4 = __ldg(reinterpret_cast<const uint32_t *>(&w_row[k]));
    const __nv_bfloat162 b01 = __ldg(reinterpret_cast<const __nv_bfloat162 *>(&in_row[k]));
    const __nv_bfloat162 b23 = __ldg(reinterpret_cast<const __nv_bfloat162 *>(&in_row[k + 2]));
    float i0=__bfloat162float(b01.x), i1=__bfloat162float(b01.y);
    float i2=__bfloat162float(b23.x), i3=__bfloat162float(b23.y);
    __nv_fp8_e4m3 w0,w1,w2,w3;
    w0.__x=(w4>>0)&0xFF; w1.__x=(w4>>8)&0xFF; w2.__x=(w4>>16)&0xFF; w3.__x=(w4>>24)&0xFF;
    const float scale = __ldg(&weight_scale[scale_row_offset + k / block_size_x]);
    acc += scale * (i0*fp8_to_float(w0) + i1*fp8_to_float(w1) +
                    i2*fp8_to_float(w2) + i3*fp8_to_float(w3));
  }
#pragma unroll
  for (int off = 16; off > 0; off /= 2) acc += __shfl_down_sync(0xffffffff, acc, off);
  if (lane == 0) out[(size_t)m * N + n] = acc;
}

// ---- MLP: U independent requests in flight, consumed u-ascending ----------
template <int ROWS_PER_BLOCK, int U>
__global__ void gemv_mlp(const __nv_bfloat16 *__restrict__ input,
                         const __nv_fp8_e4m3 *__restrict__ weight,
                         const float *__restrict__ weight_scale,
                         float *__restrict__ out, int M, int N, int K,
                         int scale_row_stride, int block_size_y, int block_size_x) {
  const int lane = threadIdx.x & 31;
  const int n = blockIdx.x * ROWS_PER_BLOCK + (threadIdx.x >> 5);
  const int m = blockIdx.y;
  if (n >= N || m >= M) return;
  const __nv_fp8_e4m3 *w_row = weight + (size_t)n * K;
  const __nv_bfloat16 *in_row = input + (size_t)m * K;
  const int scale_row_offset = (n / block_size_y) * scale_row_stride;
  float acc = 0.0f;
  const int K_aligned = (K / 128) * 128;
  const int K_deep = (K / (128 * U)) * (128 * U);
  for (int k_base = 0; k_base < K_deep; k_base += 128 * U) {
    uint32_t w4v[U]; float iv[U][4]; float sc[U];
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const int k = k_base + u * 128 + lane * 4;
      w4v[u] = __ldg(reinterpret_cast<const uint32_t *>(&w_row[k]));
      const __nv_bfloat162 b01 = __ldg(reinterpret_cast<const __nv_bfloat162 *>(&in_row[k]));
      const __nv_bfloat162 b23 = __ldg(reinterpret_cast<const __nv_bfloat162 *>(&in_row[k + 2]));
      iv[u][0]=__bfloat162float(b01.x); iv[u][1]=__bfloat162float(b01.y);
      iv[u][2]=__bfloat162float(b23.x); iv[u][3]=__bfloat162float(b23.y);
      sc[u] = __ldg(&weight_scale[scale_row_offset + k / block_size_x]);
    }
#pragma unroll
    for (int u = 0; u < U; ++u) {
      __nv_fp8_e4m3 w0,w1,w2,w3;
      w0.__x=(w4v[u]>>0)&0xFF; w1.__x=(w4v[u]>>8)&0xFF;
      w2.__x=(w4v[u]>>16)&0xFF; w3.__x=(w4v[u]>>24)&0xFF;
      acc += sc[u] * (iv[u][0]*fp8_to_float(w0) + iv[u][1]*fp8_to_float(w1) +
                      iv[u][2]*fp8_to_float(w2) + iv[u][3]*fp8_to_float(w3));
    }
  }
  for (int k_base = K_deep; k_base < K_aligned; k_base += 128) {
    const int k = k_base + lane * 4;
    const uint32_t w4 = __ldg(reinterpret_cast<const uint32_t *>(&w_row[k]));
    const __nv_bfloat162 b01 = __ldg(reinterpret_cast<const __nv_bfloat162 *>(&in_row[k]));
    const __nv_bfloat162 b23 = __ldg(reinterpret_cast<const __nv_bfloat162 *>(&in_row[k + 2]));
    float i0=__bfloat162float(b01.x), i1=__bfloat162float(b01.y);
    float i2=__bfloat162float(b23.x), i3=__bfloat162float(b23.y);
    __nv_fp8_e4m3 w0,w1,w2,w3;
    w0.__x=(w4>>0)&0xFF; w1.__x=(w4>>8)&0xFF; w2.__x=(w4>>16)&0xFF; w3.__x=(w4>>24)&0xFF;
    const float scale = __ldg(&weight_scale[scale_row_offset + k / block_size_x]);
    acc += scale * (i0*fp8_to_float(w0) + i1*fp8_to_float(w1) +
                    i2*fp8_to_float(w2) + i3*fp8_to_float(w3));
  }
#pragma unroll
  for (int off = 16; off > 0; off /= 2) acc += __shfl_down_sync(0xffffffff, acc, off);
  if (lane == 0) out[(size_t)m * N + n] = acc;
}

__global__ void fill_w(__nv_fp8_e4m3 *w, size_t n) {
  for (size_t i = blockIdx.x*(size_t)blockDim.x+threadIdx.x; i < n; i += (size_t)gridDim.x*blockDim.x) {
    unsigned s = (unsigned)(i*2654435761u+12345u);
    w[i] = __nv_fp8_e4m3(((float)((s >> 9) & 0xFF) / 128.0f) - 1.0f);
  }
}
__global__ void fill_in(__nv_bfloat16 *x, size_t n) {
  for (size_t i = blockIdx.x*(size_t)blockDim.x+threadIdx.x; i < n; i += (size_t)gridDim.x*blockDim.x) {
    unsigned s = (unsigned)(i*1664525u+1013904223u);
    x[i] = __float2bfloat16(((float)((s >> 8) & 0x1FF) / 256.0f) - 1.0f);
  }
}
__global__ void fill_s(float *s_, size_t n) {
  for (size_t i = blockIdx.x*(size_t)blockDim.x+threadIdx.x; i < n; i += (size_t)gridDim.x*blockDim.x) {
    unsigned s = (unsigned)(i*69069u+1u);
    s_[i] = 0.5f + (float)((s >> 12) & 0xFF) / 512.0f;
  }
}

static inline uint32_t bits(float f){ uint32_t u; memcpy(&u,&f,4); return u; }

int main(int argc, char** argv) {
  const int M = argc > 1 ? atoi(argv[1]) : 1;
  const int N = 4096, K = 4096, BSY = 128, BSX = 128;
  const int srs = (K + BSX - 1) / BSX;
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p,0));
  printf("GPU %s  M=%d N=%d K=%d  (flags: -O3 --use_fast_math, per build.rs)\n", p.name, M, N, K);

  __nv_fp8_e4m3 *w; __nv_bfloat16 *x; float *sc, *o_ref, *o_mlp;
  CK(cudaMalloc(&w,(size_t)N*K)); CK(cudaMalloc(&x,(size_t)M*K*2));
  CK(cudaMalloc(&sc,(size_t)((N+BSY-1)/BSY)*srs*4));
  CK(cudaMalloc(&o_ref,(size_t)M*N*4)); CK(cudaMalloc(&o_mlp,(size_t)M*N*4));
  fill_w<<<512,256>>>(w,(size_t)N*K); fill_in<<<512,256>>>(x,(size_t)M*K);
  fill_s<<<64,256>>>(sc,(size_t)((N+BSY-1)/BSY)*srs);
  CK(cudaDeviceSynchronize()); CK(cudaGetLastError());

  dim3 grid((N+3)/4, M);
  gemv_orig<4><<<grid,4*32>>>(x,w,sc,o_ref,M,N,K,srs,BSY,BSX);
  gemv_mlp<4,4><<<grid,4*32>>>(x,w,sc,o_mlp,M,N,K,srs,BSY,BSX);
  CK(cudaDeviceSynchronize()); CK(cudaGetLastError());

  const size_t NEL = (size_t)M*N;
  float *hr=(float*)malloc(NEL*4), *hm=(float*)malloc(NEL*4);
  CK(cudaMemcpy(hr,o_ref,NEL*4,cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(hm,o_mlp,NEL*4,cudaMemcpyDeviceToHost));

  size_t nz=0; for(size_t i=0;i<NEL;++i) if(hr[i]!=0.f) nz++;
  printf("engagement: %zu/%zu reference outputs nonzero\n", nz, NEL);
  if(nz*10 < NEL){ fprintf(stderr,"ENVFAIL: reference mostly zero, gate would be vacuous\n"); return 2; }

  size_t bad=0; size_t firstbad=(size_t)-1;
  for(size_t i=0;i<NEL;++i) if(bits(hr[i])!=bits(hm[i])){ if(bad==0) firstbad=i; bad++; }
  printf("[1] F32 BIT-PATTERN COMPARE  mismatches %zu / %zu  (%.4f%%)\n",
         bad, NEL, 100.0*(double)bad/NEL);
  if(bad) printf("    first at %zu: ref %08x  mlp %08x\n", firstbad, bits(hr[firstbad]), bits(hm[firstbad]));

  // [2] negative control on the OUTPUT, single element, known index.
  size_t idx = NEL/3;
  uint32_t b = bits(hr[idx]) + 1u; float saved = hr[idx]; memcpy(&hr[idx], &b, 4);
  size_t bad2=0; for(size_t i=0;i<NEL;++i) if(bits(hr[i])!=bits(hm[i])) bad2++;
  printf("[2] NEGATIVE CONTROL  1 ULP on OUTPUT index %zu -> %zu -> %zu (delta %ld, expect 1)\n",
         idx, bad, bad2, (long)bad2-(long)bad);
  memcpy(&hr[idx], &saved, 4);
  if((long)bad2-(long)bad != 1){ fprintf(stderr,"ENVFAIL: control did not fire exactly once\n"); return 2; }
  printf("    CONTROL FIRED — comparator is live\n");

  printf("[3] VERDICT: %s\n", bad==0 ? "BIT-IDENTICAL — gate PASSES"
                                     : "NOT bit-identical — gate FAILS");
  return bad==0 ? 0 : 1;
}
