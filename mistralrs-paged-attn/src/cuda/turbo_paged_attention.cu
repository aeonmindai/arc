/**
 * TurboQuant Paged Attention — Optimized CUDA Kernels
 *
 * 4-bit K (nibble packed), 3-bit V (10-in-32 packed)
 * Warp-parallel structure matching vLLM paged attention.
 */

#include <cuda_fp16.h>
#include <cstdint>
#include <cmath>
#include <cfloat>

#include "turbo_paged_attention.cuh"

#define TQ_WARP 32
#define TQ_DIVUP(a,b) (((a)+(b)-1)/(b))

static __constant__ float CB4[16] = {
    -0.237664013127f,-0.180836062501f,-0.141805261760f,-0.110288414632f,
    -0.082828489390f,-0.057772320256f,-0.034151583096f,-0.011302500645f,
     0.011302500645f, 0.034151583096f, 0.057772320256f, 0.082828489390f,
     0.110288414632f, 0.141805261760f, 0.180836062501f, 0.237664013127f,
};
static __constant__ float BD4[17] = {
    -1.0f,-0.209250037814f,-0.161320662130f,-0.126046838196f,
    -0.096558452011f,-0.070300404823f,-0.045961951676f,-0.022727041871f,
     0.0f, 0.022727041871f, 0.045961951676f, 0.070300404823f,
     0.096558452011f, 0.126046838196f, 0.161320662130f, 0.209250037814f, 1.0f,
};
static __constant__ float CB3[8] = {
    -0.188397319183f,-0.118139828402f,-0.066585638471f,-0.021604320011f,
     0.021604320011f, 0.066585638471f, 0.118139828402f, 0.188397319183f,
};
static __constant__ float BD3[9] = {
    -1.0f,-0.153268573792f,-0.092362733436f,-0.044094979241f,
     0.0f, 0.044094979241f, 0.092362733436f, 0.153268573792f, 1.0f,
};
static __constant__ float SGN[128] = {
    -1,-1, 1,-1, 1,-1,-1, 1,-1,-1, 1, 1,-1,-1,-1,-1,
    -1, 1,-1,-1, 1, 1, 1, 1, 1, 1,-1,-1,-1, 1, 1,-1,
    -1,-1, 1,-1, 1,-1,-1,-1, 1,-1,-1,-1, 1,-1,-1,-1,
    -1,-1, 1,-1,-1,-1, 1,-1,-1,-1,-1,-1, 1,-1, 1, 1,
    -1,-1,-1, 1,-1,-1, 1, 1,-1,-1,-1,-1, 1,-1, 1,-1,
     1, 1, 1,-1, 1, 1, 1, 1, 1, 1, 1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1, 1, 1, 1,-1,-1,-1, 1, 1, 1,-1,-1,-1,
    -1,-1,-1,-1, 1, 1,-1, 1, 1,-1,-1, 1, 1,-1, 1,-1,
};

// WHT that works with any number of active threads (uses only __syncthreads)
__device__ void wht128(float* d, int tid, int nthreads) {
    for (int h = 1; h < 128; h *= 2) {
        __syncthreads();
        for (int i = tid; i < 64; i += nthreads) {
            int bs = (i/h)*(h*2), off = i%h;
            float a = d[bs+off], b = d[bs+off+h];
            d[bs+off] = a+b; d[bs+off+h] = a-b;
        }
    }
    __syncthreads();
}

__device__ void rotate128(float* d, int tid, int nthreads) {
    if (tid < 128) d[tid] *= SGN[tid];
    __syncthreads();
    wht128(d, tid, nthreads);
    if (tid < 128) d[tid] *= 0.08838834764831845f;
    __syncthreads();
    if (tid < 128) d[tid] *= SGN[tid];
    __syncthreads();
}

__device__ uint8_t q4(float x) {
    if (x<=BD4[1]) return 0; if (x>=BD4[16]) return 15;
    int lo=1,hi=16; while(lo<hi){int m=(lo+hi)>>1;if(x<BD4[m])hi=m;else lo=m+1;} return lo-1;
}
__device__ uint8_t q3(float x) {
    if (x<=BD3[1]) return 0; if (x>=BD3[8]) return 7;
    int lo=1,hi=8; while(lo<hi){int m=(lo+hi)>>1;if(x<BD3[m])hi=m;else lo=m+1;} return lo-1;
}

// ============================================================================
// reshape_and_cache
// ============================================================================

__global__ void tq_cache_k(
    const __half* in, uint8_t* cache, __half* norms, const int64_t* slots,
    int nh, int hs, int bs, int in_stride, int cbs, int chs, int nbs, int nhs
) {
    int vid = blockIdx.x, tid = threadIdx.x;
    int tok = vid/nh, head = vid%nh;
    int64_t slot = slots[tok]; if (slot<0) return;
    int bi = slot/bs, bo = slot%bs;
    __shared__ float s[128]; __shared__ uint8_t ix[128];
    if (tid<hs) s[tid] = __half2float(in[tok*in_stride+head*hs+tid]);
    __syncthreads();
    __shared__ float nb[128];
    nb[tid] = (tid<hs)?s[tid]*s[tid]:0.f; __syncthreads();
    for(int r=64;r>0;r>>=1){if(tid<r)nb[tid]+=nb[tid+r];__syncthreads();}
    float nm = sqrtf(nb[0]);
    if(tid==0) norms[bi*nbs+head*nhs+bo]=__float2half(nm);
    if(tid<hs&&nm>1e-10f)s[tid]/=nm; __syncthreads();
    rotate128(s,tid,128);
    if(tid<hs)ix[tid]=q4(s[tid]); __syncthreads();
    if(tid<hs/2){
        uint8_t p=(ix[2*tid]&0xF)|((ix[2*tid+1]&0xF)<<4);
        int x=16,by=tid;
        cache[bi*cbs+head*chs+(by/x)*bs*x+bo*x+(by%x)]=p;
    }
}

__global__ void tq_cache_v(
    const __half* in, uint8_t* cache, __half* norms, const int64_t* slots,
    int nh, int hs, int bs, int in_stride, int vbs, int vhs, int nbs, int nhs
) {
    int vid = blockIdx.x, tid = threadIdx.x;
    int tok = vid/nh, head = vid%nh;
    int64_t slot = slots[tok]; if (slot<0) return;
    int bi = slot/bs, bo = slot%bs;
    __shared__ float s[128]; __shared__ uint8_t ix[128];
    if (tid<hs) s[tid] = __half2float(in[tok*in_stride+head*hs+tid]);
    __syncthreads();
    __shared__ float nb[128];
    nb[tid] = (tid<hs)?s[tid]*s[tid]:0.f; __syncthreads();
    for(int r=64;r>0;r>>=1){if(tid<r)nb[tid]+=nb[tid+r];__syncthreads();}
    float nm = sqrtf(nb[0]);
    if(tid==0) norms[bi*nbs+head*nhs+bo]=__float2half(nm);
    if(tid<hs&&nm>1e-10f)s[tid]/=nm; __syncthreads();
    rotate128(s,tid,128);
    if(tid<hs)ix[tid]=q3(s[tid]); __syncthreads();
    int ng=(hs+9)/10;
    if(tid<ng){
        int base=tid*10; uint32_t w=0;
        int cnt=min(10,hs-base);
        for(int j=0;j<cnt;j++) w|=((uint32_t)ix[base+j]&7)<<(j*3);
        int bb=bi*vbs+head*vhs+tid*4*bs+bo;
        cache[bb]=(uint8_t)w; cache[bb+bs]=(uint8_t)(w>>8);
        cache[bb+2*bs]=(uint8_t)(w>>16); cache[bb+3*bs]=(uint8_t)(w>>24);
    }
}

extern "C" void turbo_reshape_and_cache(
    const void* key, const void* value,
    void* kc, void* vc, void* kn, void* vn,
    const int64_t* slots,
    int nt, int nh, int hs, int bs,
    int ks, int vs, int kbs, int khs, int nbs, int nhs,
    cudaStream_t stream, uint32_t dtype
) {
    if(hs!=128)return;
    dim3 g(nt*nh),b(128);
    tq_cache_k<<<g,b,0,stream>>>((const __half*)key,(uint8_t*)kc,(__half*)kn,slots,nh,hs,bs,ks,kbs,khs,nbs,nhs);
    int vpd=(hs+9)/10*4; int vvbs=nh*vpd*bs; int vvhs=vpd*bs;
    tq_cache_v<<<g,b,0,stream>>>((const __half*)value,(uint8_t*)vc,(__half*)vn,slots,nh,hs,bs,vs,vvbs,vvhs,nbs,nhs);
}

// ============================================================================
// Attention kernel — optimized, no __syncthreads in divergent paths
//
// Structure:
// 1. Load + rotate Q (all 128 threads, __syncthreads is safe)
// 2. Q·K: each warp handles blocks, each lane does full dot product for one
//    token per iteration (no __syncthreads needed — warp shuffles only)
// 3. Softmax: all threads participate (safe __syncthreads)
// 4. V accumulation: each thread accumulates 4 dims, warps process blocks
// 5. Output: write accs to shared, rotate, write out (all threads participate)
// ============================================================================

template<int BLOCK_SIZE>
__global__ void tq_attn(
    float* __restrict__ out,
    const __half* __restrict__ query,
    const uint8_t* __restrict__ kc,
    const uint8_t* __restrict__ vc,
    const __half* __restrict__ kn,
    const __half* __restrict__ vn,
    const uint32_t* __restrict__ bt,
    const uint32_t* __restrict__ cl,
    int nkvh, int mbps, int nh, float scale,
    int kbs, int khs, int vbs, int vhs, int nbs, int nhs
) {
    constexpr int HS = 128;
    constexpr int NT = 128;
    constexpr int NW = NT / TQ_WARP;
    constexpr int EPT = HS / TQ_WARP; // 4 elements per thread for dot product

    const int hidx = blockIdx.x, sidx = blockIdx.y;
    const int tid = threadIdx.x;
    const int warp = tid / TQ_WARP, lane = tid % TQ_WARP;
    const uint32_t clen = cl[sidx];
    if (clen == 0) return;
    const int kvh = hidx / (nh / nkvh);

    // 1. Load + rotate Q
    __shared__ float qr[HS];
    if (tid < HS) qr[tid] = __half2float(query[sidx*nh*HS + hidx*HS + tid]);
    __syncthreads();
    rotate128(qr, tid, NT);
    // After this: qr is rotated, all threads synced

    // 2. Q·K — compute logits
    extern __shared__ char shmem[];
    float* logits = (float*)shmem;
    // Initialize logits to 0
    for (int i = tid; i < (int)clen; i += NT) logits[i] = 0.f;
    __syncthreads();

    float qk_max = -FLT_MAX;
    const uint32_t* sbt = bt + sidx * mbps;
    const int nblocks = TQ_DIVUP(clen, BLOCK_SIZE);

    // Each warp processes blocks in parallel.
    // All 32 lanes collaborate on ONE token at a time.
    // Each lane reads 2 packed bytes = 4 elements of the 128-dim K vector.
    // Warp shuffle reduces the 32 partial sums into one dot product.
    for (int bi = warp; bi < nblocks; bi += NW) {
        int pb = sbt[bi];
        int tib = min(BLOCK_SIZE, (int)clen - bi*BLOCK_SIZE);

        for (int t = 0; t < tib; t++) {
            int tpos = bi * BLOCK_SIZE + t;

            // Each lane reads its 2 packed bytes (4 elements)
            float qk = 0.f;
            #pragma unroll
            for (int e = 0; e < EPT/2; e++) {
                int byidx = lane * (EPT/2) + e; // 0..63
                int x = 16;
                int koff = pb*kbs + kvh*khs + (byidx/x)*BLOCK_SIZE*x + t*x + (byidx%x);
                uint8_t pk = kc[koff];
                int d0 = byidx*2, d1 = byidx*2+1;
                qk += qr[d0] * CB4[pk & 0xF];
                qk += qr[d1] * CB4[(pk>>4) & 0xF];
            }

            // Warp reduction: sum 32 partial dot products
            #pragma unroll
            for (int mask = TQ_WARP/2; mask > 0; mask >>= 1)
                qk += __shfl_xor_sync(0xffffffff, qk, mask);

            float knorm = __half2float(kn[pb*nbs + kvh*nhs + t]);
            qk *= knorm * scale;

            if (lane == 0) logits[tpos] = qk;
            qk_max = fmaxf(qk_max, qk);
        }
    }

    // Reduce qk_max across all threads
    #pragma unroll
    for (int mask = TQ_WARP/2; mask >= 1; mask /= 2)
        qk_max = fmaxf(qk_max, __shfl_xor_sync(0xffffffff, qk_max, mask));
    __shared__ float red[8];
    if (lane == 0) red[warp] = qk_max;
    __syncthreads();
    qk_max = (lane < NW) ? red[lane] : -FLT_MAX;
    #pragma unroll
    for (int mask = NW/2; mask >= 1; mask /= 2)
        qk_max = fmaxf(qk_max, __shfl_xor_sync(0xffffffff, qk_max, mask));
    qk_max = __shfl_sync(0xffffffff, qk_max, 0);

    // 3. Softmax
    float esum = 0.f;
    for (int i = tid; i < (int)clen; i += NT) {
        float v = __expf(logits[i] - qk_max);
        logits[i] = v;
        esum += v;
    }
    #pragma unroll
    for (int mask = TQ_WARP/2; mask >= 1; mask /= 2)
        esum += __shfl_xor_sync(0xffffffff, esum, mask);
    if (lane == 0) red[NW + warp] = esum;
    __syncthreads();
    esum = (lane < NW) ? red[NW + lane] : 0.f;
    #pragma unroll
    for (int mask = NW/2; mask >= 1; mask /= 2)
        esum += __shfl_xor_sync(0xffffffff, esum, mask);
    float inv = __fdividef(1.f, __shfl_sync(0xffffffff, esum, 0) + 1e-6f);
    for (int i = tid; i < (int)clen; i += NT) logits[i] *= inv;
    __syncthreads();

    // 4. V accumulation
    // Each thread accumulates EPT=4 output dimensions
    // thread tid handles dims: for Q·K we used lane-based, for V we use tid-based
    // tid handles dims [tid*EPT .. tid*EPT+3] (but tid goes 0..127, EPT=4 would be 512 dims)
    // Actually: 128 threads, 128 dims, each thread handles 1 dim
    // Simpler: each thread handles dim = tid (one dim per thread)

    float acc = 0.f;
    if (tid < HS) {
        int dim = tid;
        int group = dim / 10;
        int pos = dim % 10;

        for (int bi = 0; bi < nblocks; bi++) {
            int pb = sbt[bi];
            int tib = min(BLOCK_SIZE, (int)clen - bi*BLOCK_SIZE);

            for (int t = 0; t < tib; t++) {
                float w = logits[bi*BLOCK_SIZE + t];
                if (w < 1e-8f) continue;

                int bb = pb*vbs + kvh*vhs + group*4*BLOCK_SIZE + t;
                uint32_t word = (uint32_t)vc[bb] |
                               ((uint32_t)vc[bb+BLOCK_SIZE]<<8) |
                               ((uint32_t)vc[bb+2*BLOCK_SIZE]<<16) |
                               ((uint32_t)vc[bb+3*BLOCK_SIZE]<<24);
                uint8_t vi = (word>>(pos*3))&7;
                float vnorm = __half2float(vn[pb*nbs+kvh*nhs+t]);
                acc += w * CB3[vi] * vnorm;
            }
        }
    }

    // 5. Inverse rotation + output
    __shared__ float obuf[HS];
    if (tid < HS) obuf[tid] = acc;
    __syncthreads();
    rotate128(obuf, tid, NT);

    if (tid < HS)
        out[sidx*nh*HS + hidx*HS + tid] = obuf[tid];
}

extern "C" void turbo_paged_attention_v1_f16(
    void* out, const void* query,
    const void* kc, const void* vc,
    const void* kn, const void* vn,
    int nkvh, float scale, float softcapping,
    const uint32_t* bt, const uint32_t* cl,
    int bs, int mcl, int ns, int nh, int hs,
    int mbps, int qs, int kbs, int khs,
    int nbs, int nhs, cudaStream_t stream
) {
    if (hs != 128) return;
    int vpd=(hs+9)/10*4;
    int vvbs=nkvh*vpd*bs, vvhs=vpd*bs;
    int padded = TQ_DIVUP(mcl,bs)*bs;
    int smem = padded * sizeof(float);
    // Ensure enough for output buffer too (128 floats = 512 bytes, always < logits)
    dim3 grid(nh, ns, 1);
    dim3 block(128);

    switch(bs){
    case 8: tq_attn<8><<<grid,block,smem,stream>>>((float*)out,(const __half*)query,(const uint8_t*)kc,(const uint8_t*)vc,(const __half*)kn,(const __half*)vn,bt,cl,nkvh,mbps,nh,scale,kbs,khs,vvbs,vvhs,nbs,nhs); break;
    case 16: tq_attn<16><<<grid,block,smem,stream>>>((float*)out,(const __half*)query,(const uint8_t*)kc,(const uint8_t*)vc,(const __half*)kn,(const __half*)vn,bt,cl,nkvh,mbps,nh,scale,kbs,khs,vvbs,vvhs,nbs,nhs); break;
    case 32: tq_attn<32><<<grid,block,smem,stream>>>((float*)out,(const __half*)query,(const uint8_t*)kc,(const uint8_t*)vc,(const __half*)kn,(const __half*)vn,bt,cl,nkvh,mbps,nh,scale,kbs,khs,vvbs,vvhs,nbs,nhs); break;
    }
}
