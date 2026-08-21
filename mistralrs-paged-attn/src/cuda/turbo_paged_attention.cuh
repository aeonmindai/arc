/**
 * TurboQuant device tables and generic rotation — shared by every TurboQuant
 * CUDA kernel in this crate.
 *
 * WHAT THIS FILE IS
 * -----------------
 * TurboQuant compresses a KV head vector by (1) recording its L2 norm, (2)
 * applying the involutory rotation D·H·D to the unit vector, and (3) scalar
 * quantizing each rotated coordinate against a Lloyd-Max codebook. Attention
 * then runs *in the rotated domain*: because D·H·D is orthogonal, q·k is
 * preserved under it, so the K dot product needs no inverse rotation at all.
 * Only the V accumulation, which lands in the rotated basis, is rotated back
 * once at the end.
 *
 * DIMENSION SUPPORT
 * -----------------
 * The tables below cover head_dim 64, 128, 256 and 512. Both `D` (the sign
 * diagonal) and the codebook are functions of the dimension, so a kernel
 * cannot simply reuse the 128 tables at another width — the codebook centroids
 * shrink as sqrt(1/d) because the rotated coordinates concentrate.
 *
 * PROVENANCE — DO NOT HAND-EDIT THE TABLES
 * ----------------------------------------
 * Every `__constant__` array below was emitted by
 *   cargo run -p mistralrs-quant --example emit_turboquant_cuda_tables -- 64 128 256 512
 * which calls the same `turboquant::wht::generate_signs` and
 * `turboquant::codebook::get_codebook` that the Rust compressor uses. The
 * values are therefore correct by construction rather than by transcription.
 *
 * `turboquant::cuda_tables` re-derives them at test time and diffs against this
 * file, so a hand-edit here fails `cargo test -p mistralrs-quant`. That test is
 * the only thing standing between a silent sign-table drift and a cache that
 * decodes to noise, which is why it parses this header directly.
 *
 * ARCH SPECIALISATION (D16)
 * -------------------------
 * `TQ_PREFETCH` selects the software-pipeline depth of the packed-K gather.
 * Hopper and Blackwell have deeper memory latency to hide and enough registers
 * to hide it, so they run a 2-deep pipeline; pre-Hopper parts run the flat
 * loop. Both produce identical arithmetic — this is a scheduling choice, not a
 * numerical one.
 */

#pragma once

#include <stdint.h>
#include <cuda_fp16.h>
#include <float.h>

// ---------------------------------------------------------------------------
// Arch specialisation
// ---------------------------------------------------------------------------
// Hopper (SM90) and Blackwell (SM100/SM103) hide global latency better with a
// software-pipelined gather; older parts are latency-bound on the extra
// registers instead. Identical results either way.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
#define TQ_PREFETCH 2
#else
#define TQ_PREFETCH 1
#endif

// ---------------------------------------------------------------------------
// Tables — generated, see PROVENANCE above.
// ---------------------------------------------------------------------------

// ==== head_dim = 64 (seed 42) ====
static __constant__ float TQ_SGN_64[64] = {
     1,-1, 1,-1, 1, 1,-1,-1,-1,-1, 1, 1, 1,-1, 1, 1,
     1,-1, 1,-1,-1,-1, 1, 1, 1,-1, 1,-1, 1, 1, 1, 1,
     1,-1,-1, 1,-1,-1, 1, 1,-1, 1, 1,-1,-1,-1, 1,-1,
    -1,-1,-1, 1, 1, 1, 1,-1, 1,-1, 1,-1, 1, 1, 1, 1,
};
static __constant__ float TQ_CB3_64[8] = {
    -0.263914018869f,-0.166167899966f,-0.093832284212f,-0.030469184741f,
    0.030469184741f,0.093832284212f,0.166167899966f,0.263914018869f,
};
static __constant__ float TQ_BD3_64[9] = {
    -1.000000000000f,-0.215040951967f,-0.130000084639f,-0.062150731683f,
    0.000000000000f,0.062150731683f,0.130000084639f,0.215040951967f,
    1.000000000000f,
};
static __constant__ float TQ_CB4_64[16] = {
    -0.330796420574f,-0.252913773060f,-0.198856174946f,-0.154925554991f,
    -0.116486772895f,-0.081311784685f,-0.048089791089f,-0.015919025987f,
    0.015919025987f,0.048089791089f,0.081311784685f,0.116486772895f,
    0.154925554991f,0.198856174946f,0.252913773060f,0.330796420574f,
};
static __constant__ float TQ_BD4_64[17] = {
    -1.000000000000f,-0.291855096817f,-0.225884988904f,-0.176890864968f,
    -0.135706171393f,-0.098899275064f,-0.064700789750f,-0.032004408538f,
    0.000000000000f,0.032004408538f,0.064700789750f,0.098899275064f,
    0.135706171393f,0.176890864968f,0.225884988904f,0.291855096817f,
    1.000000000000f,
};

// ==== head_dim = 128 (seed 42) ====
static __constant__ float TQ_SGN_128[128] = {
    -1,-1, 1,-1, 1,-1,-1, 1,-1,-1, 1, 1,-1,-1,-1,-1,
    -1, 1,-1,-1, 1, 1, 1, 1, 1, 1,-1,-1,-1, 1, 1,-1,
    -1,-1, 1,-1, 1,-1,-1,-1, 1,-1,-1,-1, 1,-1,-1,-1,
    -1,-1, 1,-1,-1,-1, 1,-1,-1,-1,-1,-1, 1,-1, 1, 1,
    -1,-1,-1, 1,-1,-1, 1, 1,-1,-1,-1,-1, 1,-1, 1,-1,
     1, 1, 1,-1, 1, 1, 1, 1, 1, 1, 1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1, 1, 1, 1,-1,-1,-1, 1, 1, 1,-1,-1,-1,
    -1,-1,-1,-1, 1, 1,-1, 1, 1,-1,-1, 1, 1,-1, 1,-1,
};
static __constant__ float TQ_CB3_128[8] = {
    -0.188397318125f,-0.118139825761f,-0.066585637629f,-0.021604320034f,
    0.021604320034f,0.066585637629f,0.118139825761f,0.188397318125f,
};
static __constant__ float TQ_BD3_128[9] = {
    -1.000000000000f,-0.153268575668f,-0.092362731695f,-0.044094979763f,
    0.000000000000f,0.044094979763f,0.092362731695f,0.153268575668f,
    1.000000000000f,
};
static __constant__ float TQ_CB4_128[16] = {
    -0.237664014101f,-0.180836066604f,-0.141805261374f,-0.110288411379f,
    -0.082828491926f,-0.057772319764f,-0.034151583910f,-0.011302500963f,
    0.011302500963f,0.034151583910f,0.057772319764f,0.082828491926f,
    0.110288411379f,0.141805261374f,0.180836066604f,0.237664014101f,
};
static __constant__ float TQ_BD4_128[17] = {
    -1.000000000000f,-0.209250032902f,-0.161320656538f,-0.126046836376f,
    -0.096558451653f,-0.070300407708f,-0.045961949974f,-0.022727042437f,
    0.000000000000f,0.022727042437f,0.045961949974f,0.070300407708f,
    0.096558451653f,0.126046836376f,0.161320656538f,0.209250032902f,
    1.000000000000f,
};

// ==== head_dim = 256 (seed 42) ====
static __constant__ float TQ_SGN_256[256] = {
     1,-1,-1, 1,-1, 1,-1, 1,-1, 1,-1,-1,-1, 1, 1, 1,
     1,-1,-1,-1, 1,-1,-1,-1,-1, 1,-1, 1,-1, 1, 1, 1,
     1, 1, 1, 1,-1, 1, 1,-1,-1,-1, 1, 1, 1,-1,-1, 1,
    -1,-1, 1,-1, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 1,
    -1,-1, 1,-1, 1, 1, 1, 1,-1, 1, 1,-1,-1, 1,-1, 1,
     1,-1,-1, 1,-1, 1,-1,-1, 1,-1,-1,-1,-1,-1,-1,-1,
     1, 1, 1, 1, 1, 1,-1,-1, 1,-1,-1,-1,-1,-1, 1, 1,
     1,-1,-1,-1, 1, 1,-1, 1, 1, 1, 1,-1, 1,-1, 1,-1,
    -1, 1,-1, 1,-1, 1, 1,-1,-1, 1,-1, 1,-1, 1,-1, 1,
    -1,-1,-1, 1, 1,-1,-1, 1,-1,-1,-1,-1, 1,-1,-1, 1,
    -1, 1, 1, 1, 1,-1, 1,-1,-1, 1, 1, 1, 1, 1, 1,-1,
    -1, 1, 1,-1, 1,-1,-1,-1,-1, 1,-1, 1,-1, 1, 1,-1,
     1, 1,-1, 1, 1,-1, 1, 1, 1,-1, 1,-1, 1, 1,-1, 1,
    -1,-1, 1, 1,-1,-1, 1, 1, 1, 1, 1, 1,-1,-1,-1, 1,
    -1,-1,-1,-1,-1,-1, 1, 1, 1,-1, 1,-1,-1, 1,-1, 1,
     1, 1, 1, 1,-1, 1,-1, 1,-1,-1, 1, 1,-1,-1,-1,-1,
};
static __constant__ float TQ_CB3_256[8] = {
    -0.133854493499f,-0.083765551448f,-0.047166757286f,-0.015297502279f,
    0.015297502279f,0.047166757286f,0.083765551448f,0.133854493499f,
};
static __constant__ float TQ_BD3_256[9] = {
    -1.000000000000f,-0.108810022473f,-0.065466158092f,-0.031232129782f,
    0.000000000000f,0.031232129782f,0.065466158092f,0.108810022473f,
    1.000000000000f,
};
static __constant__ float TQ_CB4_256[16] = {
    -0.169410735369f,-0.128588363528f,-0.100698113441f,-0.078249387443f,
    -0.058732159436f,-0.040949229151f,-0.024200897664f,-0.008008380421f,
    0.008008380421f,0.024200897664f,0.040949229151f,0.058732159436f,
    0.078249387443f,0.100698113441f,0.128588363528f,0.169410735369f,
};
static __constant__ float TQ_BD4_256[17] = {
    -1.000000000000f,-0.148999556899f,-0.114643231034f,-0.089473746717f,
    -0.068490773439f,-0.049840692431f,-0.032575063407f,-0.016104638577f,
    0.000000000000f,0.016104638577f,0.032575063407f,0.049840692431f,
    0.068490773439f,0.089473746717f,0.114643231034f,0.148999556899f,
    1.000000000000f,
};

// ==== head_dim = 512 (seed 42) ====
static __constant__ float TQ_SGN_512[512] = {
     1,-1,-1, 1, 1,-1, 1, 1, 1,-1,-1, 1, 1, 1,-1, 1,
    -1, 1, 1,-1,-1,-1,-1, 1, 1,-1,-1,-1,-1,-1,-1, 1,
    -1,-1,-1, 1,-1, 1,-1, 1,-1,-1,-1, 1,-1,-1, 1,-1,
     1,-1, 1, 1,-1, 1,-1,-1,-1, 1,-1,-1, 1,-1, 1,-1,
     1, 1,-1, 1, 1, 1, 1, 1,-1, 1, 1, 1,-1, 1, 1,-1,
    -1,-1,-1,-1, 1,-1,-1,-1,-1,-1,-1, 1, 1,-1, 1,-1,
     1, 1,-1,-1, 1,-1,-1, 1,-1, 1,-1, 1,-1,-1,-1,-1,
     1, 1, 1, 1, 1, 1,-1,-1,-1,-1,-1, 1,-1,-1,-1,-1,
    -1, 1,-1,-1, 1, 1, 1,-1,-1, 1,-1,-1, 1, 1, 1, 1,
    -1,-1, 1, 1, 1, 1, 1,-1,-1, 1,-1,-1, 1,-1, 1,-1,
     1,-1,-1, 1,-1, 1,-1,-1, 1,-1, 1,-1,-1, 1,-1, 1,
     1,-1,-1,-1,-1,-1,-1, 1, 1, 1,-1,-1,-1, 1,-1, 1,
    -1, 1, 1,-1,-1,-1, 1,-1, 1,-1, 1, 1,-1, 1,-1,-1,
    -1, 1,-1, 1,-1, 1, 1,-1, 1,-1,-1, 1, 1,-1, 1,-1,
     1, 1, 1, 1,-1,-1,-1,-1, 1, 1, 1,-1,-1,-1,-1, 1,
     1, 1,-1, 1, 1, 1,-1,-1, 1,-1,-1, 1, 1, 1, 1, 1,
    -1,-1,-1, 1, 1,-1,-1,-1,-1,-1, 1, 1,-1, 1,-1,-1,
     1, 1,-1, 1, 1,-1,-1, 1, 1,-1, 1,-1,-1, 1, 1, 1,
     1,-1,-1, 1, 1, 1, 1,-1,-1, 1,-1,-1,-1, 1,-1, 1,
     1, 1,-1,-1, 1, 1, 1, 1, 1,-1,-1, 1, 1, 1,-1,-1,
     1,-1, 1, 1,-1, 1, 1,-1,-1, 1, 1, 1, 1,-1, 1, 1,
     1, 1,-1,-1, 1, 1,-1, 1,-1, 1, 1,-1, 1, 1, 1, 1,
    -1, 1,-1,-1, 1,-1, 1,-1,-1,-1, 1, 1, 1,-1, 1, 1,
    -1, 1, 1,-1, 1,-1,-1,-1,-1, 1, 1, 1,-1, 1, 1,-1,
     1, 1, 1, 1,-1, 1, 1, 1,-1, 1, 1, 1,-1, 1,-1,-1,
    -1,-1, 1,-1, 1,-1,-1,-1, 1,-1,-1,-1,-1, 1,-1, 1,
     1, 1,-1,-1,-1,-1, 1,-1, 1,-1, 1, 1, 1, 1, 1,-1,
     1,-1, 1, 1,-1,-1,-1, 1, 1, 1,-1,-1, 1,-1,-1, 1,
     1,-1, 1,-1, 1,-1, 1, 1,-1,-1,-1, 1,-1, 1, 1, 1,
    -1, 1,-1, 1,-1, 1, 1,-1, 1,-1,-1,-1,-1, 1, 1, 1,
     1,-1, 1, 1,-1, 1, 1, 1,-1,-1,-1, 1,-1,-1, 1, 1,
    -1, 1,-1,-1, 1,-1,-1,-1, 1,-1, 1,-1,-1, 1, 1,-1,
};
static __constant__ float TQ_CB3_512[8] = {
    -0.094875931740f,-0.059311967343f,-0.033381462097f,-0.010824348778f,
    0.010824348778f,0.033381462097f,0.059311967343f,0.094875931740f,
};
static __constant__ float TQ_BD3_512[9] = {
    -1.000000000000f,-0.077093951404f,-0.046346712857f,-0.022102905437f,
    -0.000000000000f,0.022102905437f,0.046346712857f,0.077093951404f,
    1.000000000000f,
};
static __constant__ float TQ_CB4_512[16] = {
    -0.120276063681f,-0.091181337833f,-0.071355909109f,-0.055424209684f,
    -0.041587848216f,-0.028990162537f,-0.017131028697f,-0.005668541417f,
    0.005668541417f,0.017131028697f,0.028990162537f,0.041587848216f,
    0.055424209684f,0.071355909109f,0.091181337833f,0.120276063681f,
};
static __constant__ float TQ_BD4_512[17] = {
    -1.000000000000f,-0.105728700757f,-0.081268623471f,-0.063390061259f,
    -0.048506028950f,-0.035289004445f,-0.023060595617f,-0.011399785057f,
    -0.000000000000f,0.011399785057f,0.023060595617f,0.035289004445f,
    0.048506028950f,0.063390061259f,0.081268623471f,0.105728700757f,
    1.000000000000f,
};


// ---------------------------------------------------------------------------
// Table selection by head dimension
// ---------------------------------------------------------------------------
// `if constexpr` chains rather than a class template: __constant__ arrays decay
// to __constant__-space pointers, so this compiles to a compile-time address
// with no indirection at run time.

template <int HS>
__device__ __forceinline__ const float* tq_signs() {
    if constexpr (HS == 64) return TQ_SGN_64;
    else if constexpr (HS == 128) return TQ_SGN_128;
    else if constexpr (HS == 256) return TQ_SGN_256;
    else return TQ_SGN_512;
}

template <int HS>
__device__ __forceinline__ const float* tq_cb4() {
    if constexpr (HS == 64) return TQ_CB4_64;
    else if constexpr (HS == 128) return TQ_CB4_128;
    else if constexpr (HS == 256) return TQ_CB4_256;
    else return TQ_CB4_512;
}

template <int HS>
__device__ __forceinline__ const float* tq_cb3() {
    if constexpr (HS == 64) return TQ_CB3_64;
    else if constexpr (HS == 128) return TQ_CB3_128;
    else if constexpr (HS == 256) return TQ_CB3_256;
    else return TQ_CB3_512;
}

template <int HS>
__device__ __forceinline__ const float* tq_bd4() {
    if constexpr (HS == 64) return TQ_BD4_64;
    else if constexpr (HS == 128) return TQ_BD4_128;
    else if constexpr (HS == 256) return TQ_BD4_256;
    else return TQ_BD4_512;
}

template <int HS>
__device__ __forceinline__ const float* tq_bd3() {
    if constexpr (HS == 64) return TQ_BD3_64;
    else if constexpr (HS == 128) return TQ_BD3_128;
    else if constexpr (HS == 256) return TQ_BD3_256;
    else return TQ_BD3_512;
}

/// 1/sqrt(HS) — the normalisation that makes the Hadamard transform, and
/// therefore all of D·H·D, its own inverse. A plain constant expression rather
/// than a constexpr function so it needs no relaxed-constexpr support.
#define TQ_WHT_SCALE(HS) ((HS) == 64    ? 0.125f                \
                        : (HS) == 128   ? 0.08838834764831845f  \
                        : (HS) == 256   ? 0.0625f               \
                        : 0.04419417382415922f)

// ---------------------------------------------------------------------------
// Rotation
// ---------------------------------------------------------------------------

/// In-place D·H·D over `HS` contiguous floats in shared memory, cooperatively
/// across `NT` threads.
///
/// Self-inverse, so the same call rotates a query forward and rotates the V
/// accumulator back. The butterfly and the 1/sqrt(HS) scaling are ordered to
/// match `turboquant::wht::fwht_inplace` exactly (all butterflies, then one
/// scale) — reordering them would still be orthogonal but would no longer
/// agree with the cache written by the Rust compressor.
///
/// At HS == NT == 128 this emits the same sequence of operations the original
/// hand-written `rotate128` did.
template <int HS, int NT>
__device__ __forceinline__ void tq_rotate(float* d, int tid) {
    const float* sgn = tq_signs<HS>();

    for (int i = tid; i < HS; i += NT) d[i] *= sgn[i];
    __syncthreads();

    for (int h = 1; h < HS; h *= 2) {
        __syncthreads();
        for (int i = tid; i < HS / 2; i += NT) {
            const int bs = (i / h) * (h * 2), off = i % h;
            const float a = d[bs + off], b = d[bs + off + h];
            d[bs + off] = a + b;
            d[bs + off + h] = a - b;
        }
    }
    __syncthreads();

    constexpr float s = TQ_WHT_SCALE(HS);
    for (int i = tid; i < HS; i += NT) d[i] *= s;
    __syncthreads();

    for (int i = tid; i < HS; i += NT) d[i] *= sgn[i];
    __syncthreads();
}

// ---------------------------------------------------------------------------
// Scalar quantizers
// ---------------------------------------------------------------------------
// Binary search over the codebook's decision boundaries. `boundaries[0]` and
// `boundaries[n]` are the sentinels -1 and +1, so a unit-sphere coordinate can
// never fall outside.

template <int HS>
__device__ __forceinline__ uint8_t tq_q4(float x) {
    const float* bd = tq_bd4<HS>();
    if (x <= bd[1]) return 0;
    if (x >= bd[16]) return 15;
    int lo = 1, hi = 16;
    while (lo < hi) { const int m = (lo + hi) >> 1; if (x < bd[m]) hi = m; else lo = m + 1; }
    return (uint8_t)(lo - 1);
}

template <int HS>
__device__ __forceinline__ uint8_t tq_q3(float x) {
    const float* bd = tq_bd3<HS>();
    if (x <= bd[1]) return 0;
    if (x >= bd[8]) return 7;
    int lo = 1, hi = 8;
    while (lo < hi) { const int m = (lo + hi) >> 1; if (x < bd[m]) hi = m; else lo = m + 1; }
    return (uint8_t)(lo - 1);
}

// ---------------------------------------------------------------------------
// Packed-K gather
// ---------------------------------------------------------------------------

/// Bytes of packed 4-bit K each lane owns: (HS/2 bytes per head) / 32 lanes.
/// 1 at HS=64, 2 at 128, 4 at 256, 8 at 512.
#define TQ_BYTES_PER_LANE(HS) ((HS) / 64)

/// Load a lane's whole packed-K run as one vector access.
///
/// The run is contiguous and naturally aligned: the cache's x=16 swizzle keeps
/// each group of 16 bytes contiguous, and a lane's run starts at `lane*BPL`
/// with BPL a power of two dividing 16, so it never straddles a group.
template <int BPL>
__device__ __forceinline__ uint64_t tq_load_klane(const uint8_t* p) {
    if constexpr (BPL == 1) {
        return (uint64_t)(*p);
    } else if constexpr (BPL == 2) {
        return (uint64_t)(*reinterpret_cast<const uint16_t*>(p));
    } else if constexpr (BPL == 4) {
        return (uint64_t)(*reinterpret_cast<const uint32_t*>(p));
    } else {
        const uint2 v = *reinterpret_cast<const uint2*>(p);
        return ((uint64_t)v.y << 32) | (uint64_t)v.x;
    }
}
