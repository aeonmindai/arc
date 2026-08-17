/**
 * TurboQuant Paged Attention — Optimized CUDA Kernels
 *
 * 4-bit K (nibble packed), 3-bit V (10-in-32 packed)
 * Warp-parallel structure matching vLLM paged attention.
 */

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cmath>
#include <cfloat>
#include <type_traits>

// Generic float conversion — works with __half OR __nv_bfloat16
__device__ __forceinline__ float tq_to_f(__half x) { return __half2float(x); }
__device__ __forceinline__ float tq_to_f(__nv_bfloat16 x) { return __bfloat162float(x); }

// Byte-for-byte the same `fast_tanh` the mainline paged-attention kernel uses
// (pagedattention.cuh:94). Duplicated rather than included because that header
// drags in the whole vLLM attention-utils tree, which this translation unit has
// no other need for. Renamed so the two can never collide if a future TU takes
// both. Logit softcapping must agree numerically between the two kernels or a
// Gemma-2-style model would produce different logits depending only on whether
// its KV cache happens to be TurboQuant-compressed.
__device__ __forceinline__ float tq_fast_tanh(float x) {
#if defined(__CUDA_ARCH__)
#if (__CUDACC_VER_MAJOR__ >= 11) && (__CUDA_ARCH__ >= 750)
    float y;
    asm volatile("tanh.approx.f32 %0, %1; " : "=f"(y) : "f"(x));
    return y;
#else
    return ::tanhf(x);
#endif
#else
    return std::tanh(x);
#endif
}

#include "turbo_paged_attention.cuh"

#define TQ_WARP 32
#define TQ_DIVUP(a,b) (((a)+(b)-1)/(b))

// ============================================================================
// reshape_and_cache
// ============================================================================

// Per-head-vector compression into the paged cache.
//
// One block per (token, head); NT threads cooperate on one HS-wide vector.
// HS is a template parameter so the shared buffers, the rotation order and the
// codebook are all fixed at compile time. At HS == NT == 128 the emitted work
// is the same as the original hand-written 128-only kernel.
template<int HS, typename T>
__global__ void tq_cache_k(
    const T* in, uint8_t* cache, __half* norms, const int64_t* slots,
    int nh, int bs, int in_stride, int cbs, int chs, int nbs, int nhs
) {
    constexpr int NT = 128;
    const int vid = blockIdx.x, tid = threadIdx.x;
    const int tok = vid/nh, head = vid%nh;
    const int64_t slot = slots[tok]; if (slot<0) return;
    const int bi = slot/bs, bo = slot%bs;

    __shared__ float s[HS];
    __shared__ uint8_t ix[HS];
    for (int i=tid;i<HS;i+=NT) s[i] = tq_to_f(in[tok*in_stride+head*HS+i]);
    __syncthreads();

    // L2 norm: strided partial per thread, then a flat block reduction.
    float p = 0.f;
    for (int i=tid;i<HS;i+=NT) p += s[i]*s[i];
    __shared__ float nb[NT];
    nb[tid]=p; __syncthreads();
    for(int r=NT/2;r>0;r>>=1){ if(tid<r) nb[tid]+=nb[tid+r]; __syncthreads(); }
    const float nm = sqrtf(nb[0]);
    if(tid==0) norms[bi*nbs+head*nhs+bo]=__float2half(nm);
    if(nm>1e-10f){ for(int i=tid;i<HS;i+=NT) s[i]/=nm; }
    __syncthreads();

    tq_rotate<HS,NT>(s,tid);
    for(int i=tid;i<HS;i+=NT) ix[i]=tq_q4<HS>(s[i]);
    __syncthreads();

    // 4-bit nibble pack, x=16 swizzled so a decode lane's run stays contiguous.
    for(int i=tid;i<HS/2;i+=NT){
        const uint8_t pk=(ix[2*i]&0xF)|((ix[2*i+1]&0xF)<<4);
        constexpr int x=16;
        cache[(long long)bi*cbs+(long long)head*chs+(i/x)*bs*x+bo*x+(i%x)]=pk;
    }
}

template<int HS, typename T>
__global__ void tq_cache_v(
    const T* in, uint8_t* cache, __half* norms, const int64_t* slots,
    int nh, int bs, int in_stride, int vbs, int vhs, int nbs, int nhs
) {
    constexpr int NT = 128;
    const int vid = blockIdx.x, tid = threadIdx.x;
    const int tok = vid/nh, head = vid%nh;
    const int64_t slot = slots[tok]; if (slot<0) return;
    const int bi = slot/bs, bo = slot%bs;

    __shared__ float s[HS];
    __shared__ uint8_t ix[HS];
    for (int i=tid;i<HS;i+=NT) s[i] = tq_to_f(in[tok*in_stride+head*HS+i]);
    __syncthreads();

    float p = 0.f;
    for (int i=tid;i<HS;i+=NT) p += s[i]*s[i];
    __shared__ float nb[NT];
    nb[tid]=p; __syncthreads();
    for(int r=NT/2;r>0;r>>=1){ if(tid<r) nb[tid]+=nb[tid+r]; __syncthreads(); }
    const float nm = sqrtf(nb[0]);
    if(tid==0) norms[bi*nbs+head*nhs+bo]=__float2half(nm);
    if(nm>1e-10f){ for(int i=tid;i<HS;i+=NT) s[i]/=nm; }
    __syncthreads();

    tq_rotate<HS,NT>(s,tid);
    for(int i=tid;i<HS;i+=NT) ix[i]=tq_q3<HS>(s[i]);
    __syncthreads();

    // 3-bit pack, 10 indices per 32-bit word, bytes strided by block size.
    constexpr int NG = (HS+9)/10;
    for(int i=tid;i<NG;i+=NT){
        const int base=i*10; uint32_t w=0;
        const int cnt = (HS-base) < 10 ? (HS-base) : 10;
        for(int j=0;j<cnt;j++) w |= ((uint32_t)ix[base+j]&7)<<(j*3);
        const long long bb=(long long)bi*vbs+(long long)head*vhs+i*4*bs+bo;
        cache[bb]=(uint8_t)w; cache[bb+bs]=(uint8_t)(w>>8);
        cache[bb+2*bs]=(uint8_t)(w>>16); cache[bb+3*bs]=(uint8_t)(w>>24);
    }
}

// Clocked variants of tq_cache_k/v: per-block stamps at:
//   phase 0 = entry
//   phase 1 = end of input load
//   phase 2 = end of norm computation
//   phase 3 = end of rotate+quantize
//   phase 4 = end of pack+write
template<int HS, typename T>
__global__ void tq_cache_k_clocked(
    const T* in, uint8_t* cache, __half* norms, const int64_t* slots,
    unsigned long long* clocks,
    int nh, int bs, int in_stride, int cbs, int chs, int nbs, int nhs
) {
    constexpr int NT = 128;
    const int vid = blockIdx.x, tid = threadIdx.x;
    if (tid == 0) clocks[vid * 5 + 0] = clock64();
    const int tok = vid/nh, head = vid%nh;
    const int64_t slot = slots[tok]; if (slot<0) { if (tid==0) clocks[vid*5+4]=clock64(); return; }
    const int bi = slot/bs, bo = slot%bs;

    __shared__ float s[HS];
    __shared__ uint8_t ix[HS];
    for (int i=tid;i<HS;i+=NT) s[i] = tq_to_f(in[tok*in_stride+head*HS+i]);
    __syncthreads();
    if (tid == 0) clocks[vid * 5 + 1] = clock64();

    float p = 0.f;
    for (int i=tid;i<HS;i+=NT) p += s[i]*s[i];
    __shared__ float nb[NT];
    nb[tid]=p; __syncthreads();
    for(int r=NT/2;r>0;r>>=1){ if(tid<r) nb[tid]+=nb[tid+r]; __syncthreads(); }
    const float nm = sqrtf(nb[0]);
    if(tid==0) norms[bi*nbs+head*nhs+bo]=__float2half(nm);
    if(nm>1e-10f){ for(int i=tid;i<HS;i+=NT) s[i]/=nm; }
    __syncthreads();
    if (tid == 0) clocks[vid * 5 + 2] = clock64();

    tq_rotate<HS,NT>(s,tid);
    for(int i=tid;i<HS;i+=NT) ix[i]=tq_q4<HS>(s[i]);
    __syncthreads();
    if (tid == 0) clocks[vid * 5 + 3] = clock64();

    for(int i=tid;i<HS/2;i+=NT){
        const uint8_t pk=(ix[2*i]&0xF)|((ix[2*i+1]&0xF)<<4);
        constexpr int x=16;
        cache[(long long)bi*cbs+(long long)head*chs+(i/x)*bs*x+bo*x+(i%x)]=pk;
    }
    if (tid == 0) clocks[vid * 5 + 4] = clock64();
}

template<int HS, typename T>
__global__ void tq_cache_v_clocked(
    const T* in, uint8_t* cache, __half* norms, const int64_t* slots,
    unsigned long long* clocks,
    int nh, int bs, int in_stride, int vbs, int vhs, int nbs, int nhs
) {
    constexpr int NT = 128;
    const int vid = blockIdx.x, tid = threadIdx.x;
    if (tid == 0) clocks[vid * 5 + 0] = clock64();
    const int tok = vid/nh, head = vid%nh;
    const int64_t slot = slots[tok]; if (slot<0) { if (tid==0) clocks[vid*5+4]=clock64(); return; }
    const int bi = slot/bs, bo = slot%bs;

    __shared__ float s[HS];
    __shared__ uint8_t ix[HS];
    for (int i=tid;i<HS;i+=NT) s[i] = tq_to_f(in[tok*in_stride+head*HS+i]);
    __syncthreads();
    if (tid == 0) clocks[vid * 5 + 1] = clock64();

    float p = 0.f;
    for (int i=tid;i<HS;i+=NT) p += s[i]*s[i];
    __shared__ float nb[NT];
    nb[tid]=p; __syncthreads();
    for(int r=NT/2;r>0;r>>=1){ if(tid<r) nb[tid]+=nb[tid+r]; __syncthreads(); }
    const float nm = sqrtf(nb[0]);
    if(tid==0) norms[bi*nbs+head*nhs+bo]=__float2half(nm);
    if(nm>1e-10f){ for(int i=tid;i<HS;i+=NT) s[i]/=nm; }
    __syncthreads();
    if (tid == 0) clocks[vid * 5 + 2] = clock64();

    tq_rotate<HS,NT>(s,tid);
    for(int i=tid;i<HS;i+=NT) ix[i]=tq_q3<HS>(s[i]);
    __syncthreads();
    if (tid == 0) clocks[vid * 5 + 3] = clock64();

    constexpr int NG = (HS+9)/10;
    for(int i=tid;i<NG;i+=NT){
        const int base=i*10; uint32_t w=0;
        const int cnt = (HS-base) < 10 ? (HS-base) : 10;
        for(int j=0;j<cnt;j++) w |= ((uint32_t)ix[base+j]&7)<<(j*3);
        const long long bb=(long long)bi*vbs+(long long)head*vhs+i*4*bs+bo;
        cache[bb]=(uint8_t)w; cache[bb+bs]=(uint8_t)(w>>8);
        cache[bb+2*bs]=(uint8_t)(w>>16); cache[bb+3*bs]=(uint8_t)(w>>24);
    }
    if (tid == 0) clocks[vid * 5 + 4] = clock64();
}

// Head dimensions the kernels are instantiated for. `TURBOQUANT_CUDA_HEAD_DIMS`
// in mistralrs-quant mirrors this list and a unit test pins the two together,
// so a head dim can never reach a launcher that has no instantiation for it.
template<int HS, typename T>
static void tq_launch_cache(
    const void* key, const void* value, void* kc, void* vc, void* kn, void* vn,
    const int64_t* slots, int nt, int nh, int bs,
    int ks, int vs, int kbs, int khs, int nbs, int nhs, cudaStream_t stream
) {
    dim3 g(nt*nh), b(128);
    constexpr int VPD = ((HS+9)/10)*4;
    const int vvbs = nh*VPD*bs, vvhs = VPD*bs;
    tq_cache_k<HS><<<g,b,0,stream>>>((const T*)key,(uint8_t*)kc,(__half*)kn,slots,nh,bs,ks,kbs,khs,nbs,nhs);
    tq_cache_v<HS><<<g,b,0,stream>>>((const T*)value,(uint8_t*)vc,(__half*)vn,slots,nh,bs,vs,vvbs,vvhs,nbs,nhs);
}

template<int HS, typename T>
static void tq_launch_cache_clocked(
    const void* key, const void* value, void* kc, void* vc, void* kn, void* vn,
    const int64_t* slots, void* clocks_k, void* clocks_v, int nt, int nh, int bs,
    int ks, int vs, int kbs, int khs, int nbs, int nhs, cudaStream_t stream
) {
    dim3 g(nt*nh), b(128);
    constexpr int VPD = ((HS+9)/10)*4;
    const int vvbs = nh*VPD*bs, vvhs = VPD*bs;
    tq_cache_k_clocked<HS><<<g,b,0,stream>>>((const T*)key,(uint8_t*)kc,(__half*)kn,slots,(unsigned long long*)clocks_k,nh,bs,ks,kbs,khs,nbs,nhs);
    tq_cache_v_clocked<HS><<<g,b,0,stream>>>((const T*)value,(uint8_t*)vc,(__half*)vn,slots,(unsigned long long*)clocks_v,nh,bs,vs,vvbs,vvhs,nbs,nhs);
}

// dtype: 1 = BF16 input, anything else = F16.
#define TQ_CACHE_BY_DTYPE(FN, HS_, ...)                       \
    do {                                                      \
        if (dtype == 1) FN<HS_, __nv_bfloat16>(__VA_ARGS__);  \
        else            FN<HS_, __half>(__VA_ARGS__);         \
    } while (0)

#define TQ_CACHE_BY_HEAD_DIM(FN, ...)                                       \
    switch (hs) {                                                           \
        case 64:  TQ_CACHE_BY_DTYPE(FN, 64,  __VA_ARGS__); break;           \
        case 128: TQ_CACHE_BY_DTYPE(FN, 128, __VA_ARGS__); break;           \
        case 256: TQ_CACHE_BY_DTYPE(FN, 256, __VA_ARGS__); break;           \
        case 512: TQ_CACHE_BY_DTYPE(FN, 512, __VA_ARGS__); break;           \
        default: break;                                                     \
    }

extern "C" void turbo_reshape_and_cache_clocked(
    const void* key, const void* value,
    void* kc, void* vc, void* kn, void* vn,
    const int64_t* slots,
    void* clocks_k, void* clocks_v,
    int nt, int nh, int hs, int bs,
    int ks, int vs, int kbs, int khs, int nbs, int nhs,
    cudaStream_t stream, uint32_t dtype
) {
    TQ_CACHE_BY_HEAD_DIM(tq_launch_cache_clocked,
        key, value, kc, vc, kn, vn, slots, clocks_k, clocks_v,
        nt, nh, bs, ks, vs, kbs, khs, nbs, nhs, stream);
}

extern "C" void turbo_reshape_and_cache(
    const void* key, const void* value,
    void* kc, void* vc, void* kn, void* vn,
    const int64_t* slots,
    int nt, int nh, int hs, int bs,
    int ks, int vs, int kbs, int khs, int nbs, int nhs,
    cudaStream_t stream, uint32_t dtype
) {
    TQ_CACHE_BY_HEAD_DIM(tq_launch_cache,
        key, value, kc, vc, kn, vn, slots,
        nt, nh, bs, ks, vs, kbs, khs, nbs, nhs, stream);
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

// OutT = float (default) or __nv_bfloat16 (saves a separate F32→BF16 cast kernel
// in the dedicated decode path).
//
// HS is the head dimension. Everything that depends on it — the rotation order,
// the sign diagonal, the codebooks, the packed stride, how many output dims each
// thread owns — is resolved at compile time. NT stays 128 for every HS so the
// warp-reduction topology (4 warps, `red[8]`) is unchanged; wider heads give
// each thread more dims rather than adding threads.
template<int HS, int BLOCK_SIZE, typename QT, typename OutT = float>
__global__ void tq_attn(
    OutT* __restrict__ out,
    const QT* __restrict__ query,
    const uint8_t* __restrict__ kc,
    const uint8_t* __restrict__ vc,
    const __half* __restrict__ kn,
    const __half* __restrict__ vn,
    const uint32_t* __restrict__ bt,
    const uint32_t* __restrict__ cl,
    int nkvh, int mbps, int nh, float scale, float softcapping, int qs,
    int kbs, int khs, int vbs, int vhs, int nbs, int nhs
) {
    constexpr int NT = 128;
    constexpr int NW = NT / TQ_WARP;
    constexpr int BPL = TQ_BYTES_PER_LANE(HS);     // packed K bytes per lane
    constexpr int DPT = (HS + NT - 1) / NT;        // output dims per thread

    const int hidx = blockIdx.x, sidx = blockIdx.y;
    const int tid = threadIdx.x;
    const int warp = tid / TQ_WARP, lane = tid % TQ_WARP;
    const uint32_t clen = cl[sidx];
    if (clen == 0) return;
    const int kvh = hidx / (nh / nkvh);

    const float* cb4 = tq_cb4<HS>();
    const float* cb3 = tq_cb3<HS>();

    // 1. Load + rotate Q. Attention runs in the rotated domain, so this is the
    //    only rotation the Q·K half needs.
    //
    //    `qs` is the query's sequence stride in elements; the head and dim
    //    strides are still assumed contiguous, which the Rust wrapper checks
    //    before it launches. For a contiguous query qs == nh*HS, so this is the
    //    same address every previous caller computed.
    __shared__ float qr[HS];
    for (int i = tid; i < HS; i += NT) qr[i] = tq_to_f(query[sidx*qs + hidx*HS + i]);
    __syncthreads();
    tq_rotate<HS, NT>(qr, tid);

    // 2. Q·K — compute logits
    extern __shared__ char shmem[];
    float* logits = (float*)shmem;
    for (int i = tid; i < (int)clen; i += NT) logits[i] = 0.f;
    __syncthreads();

    float qk_max = -FLT_MAX;
    const uint32_t* sbt = bt + sidx * mbps;
    const int nblocks = TQ_DIVUP(clen, BLOCK_SIZE);
    // Softcapping is a launch-wide scalar, so this test is uniform across the
    // whole grid and the default (1.0 == no cap) path keeps the token loop
    // exactly as it was. Hoisted here so the comparison itself never runs per
    // token.
    const bool do_cap = (softcapping != 1.0f);

    // Each warp takes whole blocks; all 32 lanes collaborate on one token at a
    // time, each owning BPL packed bytes (2*BPL coordinates). A warp shuffle
    // folds the 32 partial dot products into the logit.
    for (int bi = warp; bi < nblocks; bi += NW) {
        const int pb = sbt[bi];
        const int tib = min(BLOCK_SIZE, (int)clen - bi*BLOCK_SIZE);
        if (tib <= 0) continue;

        // This lane's packed run: contiguous, and inside one x=16 group.
        const int byidx0 = lane * BPL;
        const uint8_t* kbase = kc + (long long)pb*kbs + (long long)kvh*khs
                             + (byidx0 / 16) * BLOCK_SIZE * 16 + (byidx0 % 16);

        uint64_t cur = tq_load_klane<BPL>(kbase);

        for (int t = 0; t < tib; t++) {
#if TQ_PREFETCH > 1
            // Hopper/Blackwell: stage the next token's bytes while reducing this
            // one. Same arithmetic, one fewer stall.
            const uint64_t nxt = (t + 1 < tib)
                ? tq_load_klane<BPL>(kbase + (t + 1) * 16) : 0ull;
#endif
            float qk = 0.f;
            #pragma unroll
            for (int j = 0; j < BPL; j++) {
                const uint8_t pk = (uint8_t)((cur >> (8 * j)) & 0xFFull);
                const int d0 = (byidx0 + j) * 2;
                qk += qr[d0]     * cb4[pk & 0xF];
                qk += qr[d0 + 1] * cb4[(pk >> 4) & 0xF];
            }

            #pragma unroll
            for (int mask = TQ_WARP/2; mask > 0; mask >>= 1)
                qk += __shfl_xor_sync(0xffffffff, qk, mask);

            const float knorm = __half2float(kn[pb*nbs + kvh*nhs + t]);
            qk *= knorm * scale;

            // Logit softcapping, applied after the scale multiply and before
            // both the store and the running max — the same position and the
            // same arithmetic as the mainline kernel (pagedattention.cuh:276).
            // Capping only one of the two would leave the softmax renormalising
            // against a maximum no stored logit can reach.
            if (do_cap) qk = tq_fast_tanh(qk / softcapping) * softcapping;

            if (lane == 0) logits[bi*BLOCK_SIZE + t] = qk;
            qk_max = fmaxf(qk_max, qk);

#if TQ_PREFETCH > 1
            cur = nxt;
#else
            if (t + 1 < tib) cur = tq_load_klane<BPL>(kbase + (t + 1) * 16);
#endif
        }
    }

    // Reduce qk_max across all threads
    #pragma unroll
    for (int mask = TQ_WARP/2; mask >= 1; mask /= 2)
        qk_max = fmaxf(qk_max, __shfl_xor_sync(0xffffffff, qk_max, mask));
    __shared__ float red[2 * NW];
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
        const float v = __expf(logits[i] - qk_max);
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
    const float inv = __fdividef(1.f, __shfl_sync(0xffffffff, esum, 0) + 1e-6f);
    for (int i = tid; i < (int)clen; i += NT) logits[i] *= inv;
    __syncthreads();

    // 4. V accumulation — each thread owns DPT output dims, strided by NT.
    float acc[DPT];
    #pragma unroll
    for (int r = 0; r < DPT; r++) acc[r] = 0.f;

    for (int bi = 0; bi < nblocks; bi++) {
        const int pb = sbt[bi];
        const int tib = min(BLOCK_SIZE, (int)clen - bi*BLOCK_SIZE);

        for (int t = 0; t < tib; t++) {
            const float w = logits[bi*BLOCK_SIZE + t];
            if (w < 1e-8f) continue;
            const float vnorm = __half2float(vn[pb*nbs + kvh*nhs + t]);

            #pragma unroll
            for (int r = 0; r < DPT; r++) {
                const int dim = tid + r * NT;
                if (dim >= HS) continue;
                const int group = dim / 10, pos = dim % 10;
                const long long bb = (long long)pb*vbs + (long long)kvh*vhs + group*4*BLOCK_SIZE + t;
                const uint32_t word = (uint32_t)vc[bb]
                                    | ((uint32_t)vc[bb + BLOCK_SIZE] << 8)
                                    | ((uint32_t)vc[bb + 2*BLOCK_SIZE] << 16)
                                    | ((uint32_t)vc[bb + 3*BLOCK_SIZE] << 24);
                const uint8_t vi = (word >> (pos * 3)) & 7;
                acc[r] += w * cb3[vi] * vnorm;
            }
        }
    }

    // 5. Inverse rotation + output. V was quantized in the rotated basis, so the
    //    accumulator lands there too; D·H·D is its own inverse.
    __shared__ float obuf[HS];
    #pragma unroll
    for (int r = 0; r < DPT; r++) {
        const int dim = tid + r * NT;
        if (dim < HS) obuf[dim] = acc[r];
    }
    __syncthreads();
    tq_rotate<HS, NT>(obuf, tid);

    for (int i = tid; i < HS; i += NT) {
        const int oi = sidx*nh*HS + hidx*HS + i;
        const float v = obuf[i];
        if constexpr (std::is_same<OutT, __nv_bfloat16>::value) {
            out[oi] = __float2bfloat16(v);
        } else if constexpr (std::is_same<OutT, __half>::value) {
            out[oi] = __float2half(v);
        } else {
            out[oi] = v;
        }
    }
}

// Clocked variant of tq_attn: per-block stamps at:
//   phase 0 = entry
//   phase 1 = after Q load+rotate
//   phase 2 = after QK dot products
//   phase 3 = after softmax
//   phase 4 = after V accumulate
//   phase 5 = after inverse rotate + write
template<int HS, int BLOCK_SIZE, typename QT>
__global__ void tq_attn_clocked(
    __nv_bfloat16* __restrict__ out,
    const QT* __restrict__ query,
    const uint8_t* __restrict__ kc,
    const uint8_t* __restrict__ vc,
    const __half* __restrict__ kn,
    const __half* __restrict__ vn,
    const uint32_t* __restrict__ bt,
    const uint32_t* __restrict__ cl,
    unsigned long long* __restrict__ clocks,
    int nkvh, int mbps, int nh, float scale, float softcapping, int qs,
    int kbs, int khs, int vbs, int vhs, int nbs, int nhs
) {
    constexpr int NT = 128;
    constexpr int NW = NT / TQ_WARP;
    constexpr int BPL = TQ_BYTES_PER_LANE(HS);
    constexpr int DPT = (HS + NT - 1) / NT;
    const float* cb4 = tq_cb4<HS>();
    const float* cb3 = tq_cb3<HS>();

    const int hidx = blockIdx.x, sidx = blockIdx.y;
    const int tid = threadIdx.x;
    const int warp = tid / TQ_WARP, lane = tid % TQ_WARP;
    const int block_id = sidx * nh + hidx;
    if (tid == 0) clocks[block_id * 6 + 0] = clock64();

    const uint32_t clen = cl[sidx];
    if (clen == 0) {
        if (tid == 0) for (int p = 1; p < 6; p++) clocks[block_id*6+p] = clock64();
        return;
    }
    const int kvh = hidx / (nh / nkvh);

    __shared__ float qr[HS];
    for (int i = tid; i < HS; i += NT) qr[i] = tq_to_f(query[sidx*qs + hidx*HS + i]);
    __syncthreads();
    tq_rotate<HS, NT>(qr, tid);
    if (tid == 0) clocks[block_id * 6 + 1] = clock64();

    extern __shared__ char shmem[];
    float* logits = (float*)shmem;
    for (int i = tid; i < (int)clen; i += NT) logits[i] = 0.f;
    __syncthreads();

    float qk_max = -FLT_MAX;
    const uint32_t* sbt = bt + sidx * mbps;
    const int nblocks = TQ_DIVUP(clen, BLOCK_SIZE);
    const bool do_cap = (softcapping != 1.0f);
    // This loop must stay a mirror of tq_attn's: the whole point of the clocked
    // variant is that its phase-2 stamps describe the kernel that actually
    // runs. A scalar gather here against a vectorized one there would have made
    // the timings describe a kernel nobody launches.
    for (int bi = warp; bi < nblocks; bi += NW) {
        int pb = sbt[bi];
        int tib = min(BLOCK_SIZE, (int)clen - bi*BLOCK_SIZE);
        if (tib <= 0) continue;

        const int byidx0 = lane * BPL;
        const uint8_t* kbase = kc + (long long)pb*kbs + (long long)kvh*khs
                             + (byidx0 / 16) * BLOCK_SIZE * 16 + (byidx0 % 16);
        uint64_t cur = tq_load_klane<BPL>(kbase);

        for (int t = 0; t < tib; t++) {
#if TQ_PREFETCH > 1
            const uint64_t nxt = (t + 1 < tib)
                ? tq_load_klane<BPL>(kbase + (t + 1) * 16) : 0ull;
#endif
            float qk = 0.f;
            #pragma unroll
            for (int j = 0; j < BPL; j++) {
                const uint8_t pk = (uint8_t)((cur >> (8 * j)) & 0xFFull);
                const int d0 = (byidx0 + j) * 2;
                qk += qr[d0]     * cb4[pk & 0xF];
                qk += qr[d0 + 1] * cb4[(pk >> 4) & 0xF];
            }
            #pragma unroll
            for (int mask = TQ_WARP/2; mask > 0; mask >>= 1)
                qk += __shfl_xor_sync(0xffffffff, qk, mask);
            float knorm = __half2float(kn[pb*nbs + kvh*nhs + t]);
            qk *= knorm * scale;
            if (do_cap) qk = tq_fast_tanh(qk / softcapping) * softcapping;
            if (lane == 0) logits[bi * BLOCK_SIZE + t] = qk;
            qk_max = fmaxf(qk_max, qk);
#if TQ_PREFETCH > 1
            cur = nxt;
#else
            if (t + 1 < tib) cur = tq_load_klane<BPL>(kbase + (t + 1) * 16);
#endif
        }
    }
    if (tid == 0) clocks[block_id * 6 + 2] = clock64();

    #pragma unroll
    for (int mask = TQ_WARP/2; mask >= 1; mask /= 2)
        qk_max = fmaxf(qk_max, __shfl_xor_sync(0xffffffff, qk_max, mask));
    __shared__ float red[2 * NW];
    if (lane == 0) red[warp] = qk_max;
    __syncthreads();
    qk_max = (lane < NW) ? red[lane] : -FLT_MAX;
    #pragma unroll
    for (int mask = NW/2; mask >= 1; mask /= 2)
        qk_max = fmaxf(qk_max, __shfl_xor_sync(0xffffffff, qk_max, mask));
    qk_max = __shfl_sync(0xffffffff, qk_max, 0);

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
    if (tid == 0) clocks[block_id * 6 + 3] = clock64();

    float acc[DPT];
    #pragma unroll
    for (int r = 0; r < DPT; r++) acc[r] = 0.f;
    for (int bi = 0; bi < nblocks; bi++) {
        int pb = sbt[bi];
        int tib = min(BLOCK_SIZE, (int)clen - bi*BLOCK_SIZE);
        for (int t = 0; t < tib; t++) {
            float w = logits[bi*BLOCK_SIZE + t];
            if (w < 1e-8f) continue;
            float vnorm = __half2float(vn[pb*nbs+kvh*nhs+t]);
            #pragma unroll
            for (int r = 0; r < DPT; r++) {
                const int dim = tid + r * NT;
                if (dim >= HS) continue;
                const int group = dim / 10, pos = dim % 10;
                const long long bb = (long long)pb*vbs + (long long)kvh*vhs + group*4*BLOCK_SIZE + t;
                const uint32_t word = (uint32_t)vc[bb] |
                               ((uint32_t)vc[bb+BLOCK_SIZE]<<8) |
                               ((uint32_t)vc[bb+2*BLOCK_SIZE]<<16) |
                               ((uint32_t)vc[bb+3*BLOCK_SIZE]<<24);
                const uint8_t vi = (word>>(pos*3))&7;
                acc[r] += w * cb3[vi] * vnorm;
            }
        }
    }
    if (tid == 0) clocks[block_id * 6 + 4] = clock64();

    __shared__ float obuf[HS];
    #pragma unroll
    for (int r = 0; r < DPT; r++) {
        const int dim = tid + r * NT;
        if (dim < HS) obuf[dim] = acc[r];
    }
    __syncthreads();
    tq_rotate<HS, NT>(obuf, tid);
    for (int i = tid; i < HS; i += NT) {
        out[sidx*nh*HS + hidx*HS + i] = __float2bfloat16(obuf[i]);
    }
    if (tid == 0) clocks[block_id * 6 + 5] = clock64();
}

// ============================================================================
// Launch wrappers
// ============================================================================

/// Launch one instantiation, opting in to the arch's real shared-memory budget
/// first.
///
/// The logits array is dynamic shared memory sized by the padded context
/// length, so it grows without bound as context grows. CUDA caps dynamic shared
/// memory at 48 KB per block unless a kernel explicitly opts in — 164 KB on
/// Ampere, 227 KB on Hopper (SM90) and Blackwell (SM100/SM103). Without this
/// call, any context past roughly 12k tokens fails to launch. `granted` is a
/// per-instantiation static, so the driver call happens once per kernel rather
/// than once per decode step.
template<int HS, int BLOCK_SIZE, typename QT, typename OutT>
static void tq_launch_attn(
    void* out, const void* query, const void* kc, const void* vc,
    const void* kn, const void* vn, const uint32_t* bt, const uint32_t* cl,
    int nkvh, int mbps, int nh, float scale, float softcapping, int qs,
    int kbs, int khs, int vbs, int vhs, int nbs, int nhs,
    dim3 grid, dim3 block, int smem, cudaStream_t stream
) {
    // Not cached: the requestable maximum differs by architecture, so a single
    // process-wide flag would be wrong the moment two devices differ. A failure
    // here means the request exceeds the device's opt-in cap; the launch below
    // then fails with cudaErrorInvalidValue, which surfaces on the next sync.
    if (smem > 48 * 1024) {
        (void)cudaFuncSetAttribute(
            (const void*)(tq_attn<HS, BLOCK_SIZE, QT, OutT>),
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    }
    tq_attn<HS, BLOCK_SIZE, QT, OutT><<<grid, block, smem, stream>>>(
        (OutT*)out, (const QT*)query, (const uint8_t*)kc, (const uint8_t*)vc,
        (const __half*)kn, (const __half*)vn, bt, cl,
        nkvh, mbps, nh, scale, softcapping, qs, kbs, khs, vbs, vhs, nbs, nhs);
}

template<int HS, int BLOCK_SIZE, typename QT>
static void tq_launch_attn_clocked(
    void* out_bf16, const void* query, const void* kc, const void* vc,
    const void* kn, const void* vn, const uint32_t* bt, const uint32_t* cl,
    void* clocks, int nkvh, int mbps, int nh, float scale, float softcapping, int qs,
    int kbs, int khs, int vbs, int vhs, int nbs, int nhs,
    dim3 grid, dim3 block, int smem, cudaStream_t stream
) {
    // Not cached: the requestable maximum differs by architecture, so a single
    // process-wide flag would be wrong the moment two devices differ. A failure
    // here means the request exceeds the device's opt-in cap; the launch below
    // then fails with cudaErrorInvalidValue, which surfaces on the next sync.
    if (smem > 48 * 1024) {
        (void)cudaFuncSetAttribute(
            (const void*)(tq_attn_clocked<HS, BLOCK_SIZE, QT>),
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    }
    tq_attn_clocked<HS, BLOCK_SIZE, QT><<<grid, block, smem, stream>>>(
        (__nv_bfloat16*)out_bf16, (const QT*)query,
        (const uint8_t*)kc, (const uint8_t*)vc,
        (const __half*)kn, (const __half*)vn, bt, cl,
        (unsigned long long*)clocks,
        nkvh, mbps, nh, scale, softcapping, qs, kbs, khs, vbs, vhs, nbs, nhs);
}

// Paged block sizes the cache engine emits.
#define TQ_ATTN_BY_BS(HS_, QT, OUTT, ...)                                    \
    switch (bs) {                                                            \
        case 8:  tq_launch_attn<HS_, 8,  QT, OUTT>(__VA_ARGS__); break;      \
        case 16: tq_launch_attn<HS_, 16, QT, OUTT>(__VA_ARGS__); break;      \
        case 32: tq_launch_attn<HS_, 32, QT, OUTT>(__VA_ARGS__); break;      \
        default: break;                                                      \
    }

// Head dimensions with codebooks and sign diagonals in turbo_paged_attention.cuh.
// `TURBOQUANT_CUDA_HEAD_DIMS` in mistralrs-quant mirrors this set and a unit
// test pins the two together, so the Rust side refuses a head dim before it can
// reach a launcher with no instantiation for it.
#define TQ_ATTN_BY_HS(QT, OUTT, ...)                                         \
    switch (hs) {                                                            \
        case 64:  TQ_ATTN_BY_BS(64,  QT, OUTT, __VA_ARGS__) break;           \
        case 128: TQ_ATTN_BY_BS(128, QT, OUTT, __VA_ARGS__) break;           \
        case 256: TQ_ATTN_BY_BS(256, QT, OUTT, __VA_ARGS__) break;           \
        case 512: TQ_ATTN_BY_BS(512, QT, OUTT, __VA_ARGS__) break;           \
        default: break;                                                      \
    }

#define TQ_ATTN_CLOCKED_BY_BS(HS_, QT, ...)                                  \
    switch (bs) {                                                            \
        case 8:  tq_launch_attn_clocked<HS_, 8,  QT>(__VA_ARGS__); break;    \
        case 16: tq_launch_attn_clocked<HS_, 16, QT>(__VA_ARGS__); break;    \
        case 32: tq_launch_attn_clocked<HS_, 32, QT>(__VA_ARGS__); break;    \
        default: break;                                                      \
    }

#define TQ_ATTN_CLOCKED_BY_HS(QT, ...)                                       \
    switch (hs) {                                                            \
        case 64:  TQ_ATTN_CLOCKED_BY_BS(64,  QT, __VA_ARGS__) break;         \
        case 128: TQ_ATTN_CLOCKED_BY_BS(128, QT, __VA_ARGS__) break;         \
        case 256: TQ_ATTN_CLOCKED_BY_BS(256, QT, __VA_ARGS__) break;         \
        case 512: TQ_ATTN_CLOCKED_BY_BS(512, QT, __VA_ARGS__) break;         \
        default: break;                                                      \
    }

// Geometry shared by every entry point below.
#define TQ_ATTN_PROLOGUE()                                                   \
    const int vpd = (hs + 9) / 10 * 4;                                       \
    const int vvbs = nkvh * vpd * bs, vvhs = vpd * bs;                       \
    const int padded = TQ_DIVUP(mcl, bs) * bs;                               \
    const int smem = padded * (int)sizeof(float);                            \
    dim3 grid(nh, ns, 1);                                                    \
    dim3 block(128);

extern "C" void turbo_paged_attention_v1_bf16out_clocked(
    void* out_bf16, const void* query,
    const void* kc, const void* vc,
    const void* kn, const void* vn,
    int nkvh, float scale, float softcapping,
    const uint32_t* bt, const uint32_t* cl,
    void* clocks,
    int bs, int mcl, int ns, int nh, int hs,
    int mbps, int qs, int kbs, int khs,
    int nbs, int nhs, cudaStream_t stream, uint32_t qdtype
) {
    TQ_ATTN_PROLOGUE()
    if (qdtype == 1) {
        TQ_ATTN_CLOCKED_BY_HS(__nv_bfloat16,
            out_bf16, query, kc, vc, kn, vn, bt, cl, clocks,
            nkvh, mbps, nh, scale, softcapping, qs, kbs, khs, vvbs, vvhs, nbs, nhs,
            grid, block, smem, stream)
    } else {
        TQ_ATTN_CLOCKED_BY_HS(__half,
            out_bf16, query, kc, vc, kn, vn, bt, cl, clocks,
            nkvh, mbps, nh, scale, softcapping, qs, kbs, khs, vvbs, vvhs, nbs, nhs,
            grid, block, smem, stream)
    }
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
    TQ_ATTN_PROLOGUE()
    TQ_ATTN_BY_HS(__half, float,
        out, query, kc, vc, kn, vn, bt, cl,
        nkvh, mbps, nh, scale, softcapping, qs, kbs, khs, vvbs, vvhs, nbs, nhs,
        grid, block, smem, stream)
}

// Dtype-generic entry point for the dedicated decode path
extern "C" void turbo_paged_attention_v1(
    void* out, const void* query,
    const void* kc, const void* vc,
    const void* kn, const void* vn,
    int nkvh, float scale, float softcapping,
    const uint32_t* bt, const uint32_t* cl,
    int bs, int mcl, int ns, int nh, int hs,
    int mbps, int qs, int kbs, int khs,
    int nbs, int nhs, cudaStream_t stream, uint32_t dtype
) {
    TQ_ATTN_PROLOGUE()
    if (dtype == 1) {
        TQ_ATTN_BY_HS(__nv_bfloat16, float,
            out, query, kc, vc, kn, vn, bt, cl,
            nkvh, mbps, nh, scale, softcapping, qs, kbs, khs, vvbs, vvhs, nbs, nhs,
            grid, block, smem, stream)
    } else {
        TQ_ATTN_BY_HS(__half, float,
            out, query, kc, vc, kn, vn, bt, cl,
            nkvh, mbps, nh, scale, softcapping, qs, kbs, khs, vvbs, vvhs, nbs, nhs,
            grid, block, smem, stream)
    }
}

// BF16-output entry point: writes BF16 directly, eliminating the F32→BF16 cast
// kernel that the dedicated decode path used to need after this attention call.
extern "C" void turbo_paged_attention_v1_bf16out(
    void* out_bf16, const void* query,
    const void* kc, const void* vc,
    const void* kn, const void* vn,
    int nkvh, float scale, float softcapping,
    const uint32_t* bt, const uint32_t* cl,
    int bs, int mcl, int ns, int nh, int hs,
    int mbps, int qs, int kbs, int khs,
    int nbs, int nhs, cudaStream_t stream, uint32_t qdtype
) {
    TQ_ATTN_PROLOGUE()
    if (qdtype == 1) {
        TQ_ATTN_BY_HS(__nv_bfloat16, __nv_bfloat16,
            out_bf16, query, kc, vc, kn, vn, bt, cl,
            nkvh, mbps, nh, scale, softcapping, qs, kbs, khs, vvbs, vvhs, nbs, nhs,
            grid, block, smem, stream)
    } else {
        TQ_ATTN_BY_HS(__half, __nv_bfloat16,
            out_bf16, query, kc, vc, kn, vn, bt, cl,
            nkvh, mbps, nh, scale, softcapping, qs, kbs, khs, vvbs, vvhs, nbs, nhs,
            grid, block, smem, stream)
    }
}
