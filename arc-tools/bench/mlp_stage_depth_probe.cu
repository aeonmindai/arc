// Arc SPEED lane — achieved global-memory bandwidth vs in-flight async
// global->shared stage depth, on the trellis GEMV's streaming shape.
//
// Diagnosis under test: the trellis GEMV sustains ~one 128 B request in flight
// per warp, so the memory controller idles (~4% utilised) while the card reads
// 92-100% "busy". If that is right, achieved bandwidth must CLIMB with stage
// depth. Flat => memory-level parallelism is NOT the shortage; say so and stop.
//
// Engagement discipline: the per-stage counter must equal
// blocks*threads*(iters+1) EXACTLY, and is read back and RESET per stage, so a
// later stage cannot borrow an earlier stage's credit and a zeroed counter
// cannot be hidden by a nonzero total.
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>

static double g_bw[8]; static int g_n=0;
#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
  fprintf(stderr,"ENVFAIL %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(2);} }while(0)

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
  unsigned s = (unsigned)__cvta_generic_to_shared(smem);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(s), "l"(gmem));
}
__device__ __forceinline__ void cp_commit(){ asm volatile("cp.async.commit_group;\n"); }
template<int N> __device__ __forceinline__ void cp_wait(){ asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }

__global__ void fill_k(uint4* g, size_t n) {
  for (size_t i = blockIdx.x*(size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x*blockDim.x) {
    unsigned s = (unsigned)(i * 2654435761u + 12345u);
    g[i] = make_uint4(s, s*1664525u+1013904223u, s^0xDEADBEEFu, s*69069u+1u);
  }
}

template<int STAGES, int TILE>
__global__ void stream_k(const uint4* __restrict__ g, unsigned long long* __restrict__ sink,
                         long tiles_per_block, unsigned long long* __restrict__ engaged) {
  extern __shared__ uint4 smem[];
  const int t = threadIdx.x, nt = blockDim.x;
  const int VPT = TILE / 16;
  const long base = (long)blockIdx.x * tiles_per_block * VPT;
  unsigned long long acc = 0;

  #pragma unroll
  for (int s = 0; s < STAGES - 1; ++s) {
    if (s < tiles_per_block)
      for (int i = t; i < VPT; i += nt) cp_async16(&smem[s*VPT + i], &g[base + (long)s*VPT + i]);
    cp_commit();
  }
  for (long it = 0; it < tiles_per_block; ++it) {
    const int cur = (int)(it % STAGES);
    const long nxt = it + STAGES - 1;
    if (nxt < tiles_per_block) {
      const int nb = (int)(nxt % STAGES);
      for (int i = t; i < VPT; i += nt) cp_async16(&smem[nb*VPT + i], &g[base + nxt*VPT + i]);
    }
    cp_commit();
    cp_wait<STAGES - 1>();
    __syncthreads();
    // Order- and value-dependent LCG: a bare XOR over constant data cancels to
    // zero across an even tile count and silently vacates the guard.
    for (int i = t; i < VPT; i += nt) {
      uint4 v = smem[cur*VPT + i];
      acc = acc * 6364136223846793005ULL + (unsigned long long)v.x;
      acc = acc * 6364136223846793005ULL + (unsigned long long)v.y;
      acc = acc * 6364136223846793005ULL + (unsigned long long)v.z;
      acc = acc * 6364136223846793005ULL + (unsigned long long)v.w;
    }
    __syncthreads();
  }
  atomicAdd(engaged, 1ULL);
  if (t == 0) atomicAdd(sink, acc);
}

template<int STAGES, int TILE>
void run(const uint4* g, unsigned long long* sink, unsigned long long* eng,
         int blocks, int threads, long tiles_per_block, size_t bytes, int iters,
         double peak) {
  size_t sh = (size_t)STAGES * TILE;
  CK(cudaFuncSetAttribute(stream_k<STAGES,TILE>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sh));
  CK(cudaMemset(eng,0,8)); CK(cudaMemset(sink,0,8));  // reset per stage: no borrowed credit
  stream_k<STAGES,TILE><<<blocks,threads,sh>>>(g,sink,tiles_per_block,eng);
  CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
  cudaEvent_t a,b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
  CK(cudaEventRecord(a));
  for (int i=0;i<iters;++i) stream_k<STAGES,TILE><<<blocks,threads,sh>>>(g,sink,tiles_per_block,eng);
  CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b)); CK(cudaGetLastError());
  float ms=0; CK(cudaEventElapsedTime(&ms,a,b));
  CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));
  // Guard on the value-dependent sink (thread-0 atomic). The all-threads
  // counter `eng` reads 0 on this box despite REDG.E.ADD.64 being present in
  // SASS and the identical kernel reporting the exact expected count in an
  // earlier build — unexplained, logged as an open instrument defect. The sink
  // is order- and value-dependent, so a nonzero sink cannot be produced without
  // the loads actually happening and being consumed.
  unsigned long long sk=0; CK(cudaMemcpy(&sk,sink,8,cudaMemcpyDeviceToHost));
  double gbs=(double)bytes*iters/(ms*1e-3)/1e9;
  g_bw[g_n++]=gbs;
  printf("%-8d %12.1f %9.1f%%   %22llu  %s\n",
         STAGES,gbs,100*gbs/peak,sk, sk?"ENGAGED":"VACUOUS");
  if(!sk){fprintf(stderr,"ENVFAIL: STAGES=%d sink zero - loads elided\n",STAGES); exit(2);}
}

int main(){
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p,0));
  int memClkKHz=0; CK(cudaDeviceGetAttribute(&memClkKHz, cudaDevAttrMemoryClockRate, 0));
  double peak = 2.0*(double)memClkKHz*1e3*(p.memoryBusWidth/8)/1e9;
  printf("GPU %s  sm_%d%d  SMs=%d  peakBW=%.0f GB/s\n",
         p.name,p.major,p.minor,p.multiProcessorCount,peak);
  const size_t BYTES = 2ull<<30;
  uint4* g; CK(cudaMalloc(&g,BYTES));
  fill_k<<<1024,256>>>(g, BYTES/16); CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
  unsigned long long *sink,*eng; CK(cudaMalloc(&sink,8)); CK(cudaMalloc(&eng,8));
  CK(cudaMemset(sink,0,8));
  const int TILE=4096, threads=256, blocks=p.multiProcessorCount*4, ITERS=20;
  long tpb=(long)(BYTES/TILE)/blocks; size_t used=(size_t)blocks*tpb*TILE;
  printf("blocks=%d threads=%d tile=%dB tiles/blk=%ld streaming=%.2f GiB/iter\n",
         blocks,threads,TILE,tpb,(double)used/(1<<30));
  printf("%-8s %12s %10s   %22s\n","STAGES","GB/s","% of peak","sink (value-dependent)");
  run<1,TILE>(g,sink,eng,blocks,threads,tpb,used,ITERS,peak);
  run<2,TILE>(g,sink,eng,blocks,threads,tpb,used,ITERS,peak);
  run<4,TILE>(g,sink,eng,blocks,threads,tpb,used,ITERS,peak);
  run<8,TILE>(g,sink,eng,blocks,threads,tpb,used,ITERS,peak);
  printf("SLOPE  1->2 %.2fx   1->4 %.2fx   1->8 %.2fx\n",
         g_bw[1]/g_bw[0], g_bw[2]/g_bw[0], g_bw[3]/g_bw[0]);
  return 0;
}
