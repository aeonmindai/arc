// Arc SPEED lane — A/B for the cb_mult==0 decode arm on the SHIPPED qtip2
// (LUT) rung: computed Box-Muller (what shipped) vs gathering the baked 512 KB
// table (what every other kernel does, and what 54e326b41 restores).
//
// Compiled with the crate's REAL flags (mistralrs-quant/build.rs:273-281),
// --use_fast_math included, because that is the whole question: under fast-math
// logf/sincosf become __logf/__sincosf and the device result is NOT the host
// gaussian_lut() it claimed to reproduce.
//
// Reports, for all 2^16 states:
//   1. F32 BIT-PATTERN exactness vs the host reference (compared as u32 BEFORE
//      any narrowing), plus max ULP error.
//   2. A 1-ULP negative control that MUST fire, and must move the mismatch
//      count by EXACTLY ONE -- chosen so it cannot cancel or be absorbed.
//   3. Decode cost of each arm over a realistic scattered-state stream.
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
  fprintf(stderr,"ENVFAIL %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(2);} }while(0)

static const int L = 16;
static const int LUT_N = 1 << L;          // 65536 states
static const float DENOM = 4294967296.0f; // (u32::MAX as f32 + 2.0) == 2^32

// ---- host reference: exact mirror of qtip/mod.rs gaussian_lut() ------------
static void host_lut(float* out) {
  for (uint32_t s = 0; s < (uint32_t)LUT_N; ++s) {
    uint64_t z = (uint64_t)s * 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z ^= z >> 31;
    uint32_t hi = (uint32_t)(z >> 32), lo = (uint32_t)(z & 0xFFFFFFFFu);
    float u1 = ((float)hi + 1.0f) / DENOM;
    float u2 = ((float)lo + 1.0f) / DENOM;
    float r = sqrtf(-2.0f * logf(u1));
    float th = (2.0f * 3.14159265358979323846f) * u2;
    out[2*s+0] = r * cosf(th);
    out[2*s+1] = r * sinf(th);
  }
}

// ---- device: the shipped computed arm (qtip_gather_gemv.cu qtip_decode_state)
__device__ __forceinline__ float2 dev_boxmuller(uint32_t state) {
  unsigned long long z = (unsigned long long)state * 0x9E3779B97F4A7C15ULL;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  z ^= z >> 31;
  const uint32_t hi = (uint32_t)(z >> 32), lo = (uint32_t)(z & 0xFFFFFFFFu);
  const float u1 = ((float)hi + 1.0f) / DENOM;
  const float u2 = ((float)lo + 1.0f) / DENOM;
  const float r = sqrtf(-2.0f * logf(u1));
  const float th = (2.0f * 3.14159265358979323846f) * u2;
  float s, c; sincosf(th, &s, &c);
  return make_float2(r * c, r * s);
}
__global__ void emit_boxmuller(float* o) {
  for (int s = blockIdx.x*blockDim.x+threadIdx.x; s < LUT_N; s += gridDim.x*blockDim.x) {
    float2 v = dev_boxmuller((uint32_t)s); o[2*s]=v.x; o[2*s+1]=v.y;
  }
}

// ---- decode cost: scattered states, Box-Muller vs LUT gather ---------------
__global__ void cost_bm(const uint32_t* st, float* sink, long n) {
  float a = 0.f;
  for (long i = blockIdx.x*(long)blockDim.x+threadIdx.x; i < n; i += (long)gridDim.x*blockDim.x) {
    float2 v = dev_boxmuller(st[i]); a += v.x + v.y;
  }
  atomicAdd(sink, a);
}
__global__ void cost_lut(const uint32_t* st, const float* __restrict__ lut, float* sink, long n) {
  float a = 0.f;
  for (long i = blockIdx.x*(long)blockDim.x+threadIdx.x; i < n; i += (long)gridDim.x*blockDim.x) {
    uint32_t s = st[i];
    a += __ldg(lut + 2*(size_t)s) + __ldg(lut + 2*(size_t)s + 1);
  }
  atomicAdd(sink, a);
}
__global__ void mkstates(uint32_t* st, long n) {
  for (long i = blockIdx.x*(long)blockDim.x+threadIdx.x; i < n; i += (long)gridDim.x*blockDim.x)
    st[i] = (uint32_t)((i * 2654435761u + 12345u) & (LUT_N - 1));
}

static inline uint32_t bits(float f){ uint32_t u; memcpy(&u,&f,4); return u; }
static int ulp(float a, float b){
  int32_t x=(int32_t)bits(a), y=(int32_t)bits(b);
  if(x<0) x=0x80000000-x; if(y<0) y=0x80000000-y;
  long d=(long)x-(long)y; return (int)(d<0?-d:d);
}
static long compare(const float* ref, const float* dev, int* maxulp){
  long bad=0; *maxulp=0;
  for(int i=0;i<LUT_N*2;++i){
    if(bits(ref[i])!=bits(dev[i])){ bad++; int u=ulp(ref[i],dev[i]); if(u>*maxulp)*maxulp=u; }
  }
  return bad;
}

int main(){
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p,0));
  printf("GPU %s sm_%d%d  (flags: -O3 --use_fast_math, per mistralrs-quant/build.rs)\n",
         p.name,p.major,p.minor);

  float* ref=(float*)malloc(LUT_N*2*sizeof(float)); host_lut(ref);
  float *d_o,*h_o=(float*)malloc(LUT_N*2*sizeof(float));
  CK(cudaMalloc(&d_o,LUT_N*2*sizeof(float)));
  emit_boxmuller<<<256,256>>>(d_o); CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
  CK(cudaMemcpy(h_o,d_o,LUT_N*2*sizeof(float),cudaMemcpyDeviceToHost));

  int mu=0; long bad=compare(ref,h_o,&mu);
  long tot=(long)LUT_N*2;
  printf("\n[1] BIT-EXACTNESS  device Box-Muller vs host gaussian_lut(), F32 bit patterns\n");
  printf("    exact %ld / %ld = %.2f%%   mismatched %ld   max ULP %d\n",
         tot-bad,tot,100.0*(double)(tot-bad)/tot,bad,mu);

  // [2] negative control: perturb exactly one reference value by 1 ULP and
  // require the comparator to notice, moving the count by EXACTLY one. A
  // whole-array or XOR-style control could cancel; a single-element delta on a
  // known-good index cannot.
  printf("\n[2] NEGATIVE CONTROL (must fire)\n");
  long idx=-1; for(int i=0;i<LUT_N*2;++i) if(bits(ref[i])==bits(h_o[i])){ idx=i; break; }
  if(idx<0){ fprintf(stderr,"ENVFAIL: no bit-exact element to perturb\n"); return 2; }
  float save=ref[idx]; uint32_t b=bits(save)+1u; memcpy(&ref[idx],&b,4);
  int mu2=0; long bad2=compare(ref,h_o,&mu2);
  printf("    perturbed index %ld by 1 ULP -> mismatches %ld -> %ld (delta %ld, expect 1)\n",
         idx,bad,bad2,bad2-bad);
  if(bad2-bad!=1){ fprintf(stderr,"ENVFAIL: control did not fire exactly once\n"); return 2; }
  printf("    CONTROL FIRED — the comparator is live\n");
  ref[idx]=save;

  // [3] decode cost of the two arms
  const long N=1L<<26;
  uint32_t* d_st; CK(cudaMalloc(&d_st,N*sizeof(uint32_t)));
  mkstates<<<1024,256>>>(d_st,N); CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
  float *d_lut,*d_sink; CK(cudaMalloc(&d_lut,LUT_N*2*sizeof(float))); CK(cudaMalloc(&d_sink,4));
  CK(cudaMemcpy(d_lut,ref,LUT_N*2*sizeof(float),cudaMemcpyHostToDevice));
  cudaEvent_t a,bb; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&bb));
  float ms_bm=0,ms_lut=0; float h_bm=0,h_lut=0; const int IT=20;
  int blocks=p.multiProcessorCount*8;

  CK(cudaMemset(d_sink,0,4));
  cost_bm<<<blocks,256>>>(d_st,d_sink,N); CK(cudaDeviceSynchronize());
  CK(cudaEventRecord(a));
  for(int i=0;i<IT;++i) cost_bm<<<blocks,256>>>(d_st,d_sink,N);
  CK(cudaEventRecord(bb)); CK(cudaEventSynchronize(bb)); CK(cudaGetLastError());
  CK(cudaEventElapsedTime(&ms_bm,a,bb)); ms_bm/=IT;
  CK(cudaMemcpy(&h_bm,d_sink,4,cudaMemcpyDeviceToHost));

  CK(cudaMemset(d_sink,0,4));
  cost_lut<<<blocks,256>>>(d_st,d_lut,d_sink,N); CK(cudaDeviceSynchronize());
  CK(cudaEventRecord(a));
  for(int i=0;i<IT;++i) cost_lut<<<blocks,256>>>(d_st,d_lut,d_sink,N);
  CK(cudaEventRecord(bb)); CK(cudaEventSynchronize(bb)); CK(cudaGetLastError());
  CK(cudaEventElapsedTime(&ms_lut,a,bb)); ms_lut/=IT;
  CK(cudaMemcpy(&h_lut,d_sink,4,cudaMemcpyDeviceToHost));

  printf("\n[3] DECODE COST  %ld scattered states/launch (512 KB LUT, L2-resident)\n",N);
  printf("    Box-Muller (shipped cb_mult==0 arm) %8.3f ms   sink=%.6g\n",ms_bm,h_bm);
  printf("    LUT gather (54e326b41 restores)     %8.3f ms   sink=%.6g\n",ms_lut,h_lut);
  printf("    SPEEDUP %.2fx\n", ms_bm/ms_lut);
  if(h_bm==0.f||h_lut==0.f){ fprintf(stderr,"ENVFAIL: a decode sink is zero\n"); return 2; }
  return 0;
}
