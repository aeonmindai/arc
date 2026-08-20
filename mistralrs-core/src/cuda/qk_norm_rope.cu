// Parent system: ArcInfer / ArcAttention
//
// Fused DeepSeek-V4 Q/K pre-attention block: head-transpose + per-head Q
// RMS-normalisation + adjacent-pair RoPE + the NoPE/PE recombination, in ONE
// kernel.
//
// ---------------------------------------------------------------------------
// WHY
// ---------------------------------------------------------------------------
// At b=1 the V4 decode step is launch-bound. After the mHC/router fusions the
// step is 5,566 launches, of which 1,910.8 (34.3%) are PURE DATA MOVEMENT:
// ucopy_bf16 514, copy2d_bf16 418, cast_bf16_f32 356, cast_f32_bf16 221,
// cast_u32_f32 203. They move kilobytes and cost microseconds each.
//
// The Q/K pre-attention block in `deepseek4.rs::Attention::forward` is the
// largest single contributor of `ucopy`/`copy2d` in that list. It spells a
// transpose, an RMS-norm and a RoPE as SIXTEEN candle launches per layer, of
// which TEN are pure copies:
//
//   q.reshape.transpose(1,2).contiguous()      1  copy   (materialised transpose)
//   q.sqr                                      1
//   .mean_keepdim  = fast_sum + affine(1/D)    2
//   .affine(1.0, eps)                          1
//   .recip                                     1
//   .sqrt                                      1
//   q.broadcast_mul(inv_rms)                   1
//   k.reshape.transpose(1,2).contiguous()      1  copy   (free at T=1)
//   q_pe.contiguous()   (inside rope_i)        1  copy
//   rope_i(q_pe)                               1
//   k_pe.contiguous()   (inside rope_i)        1  copy
//   rope_i(k_pe)                               1
//   cat([q_nope, q_pe])                        2  copies
//   cat([k_nope, k_pe])                        2  copies
//
// Sixteen launches whose entire data footprint at decode is 64 KB of Q and
// 1 KB of K. This file does the whole thing in one launch of 65 blocks.
//
// The intermediate tensors that the copies exist to materialise are never
// built: the transpose is expressed as an output ADDRESS, the NoPE/PE split
// and re-`cat` as two write ranges of the same row, and the RMS statistic is
// kept in a register between the reduction and the scaling.
//
// ---------------------------------------------------------------------------
// BIT-IDENTITY CONTRACT
// ---------------------------------------------------------------------------
// Q feeds attention scores and K is written into the KV cache, so a
// reassociated sum or a contracted FMA does not merely perturb a logit — it
// perturbs every future token that reads that cache entry. This kernel is
// therefore BIT-IDENTICAL to the candle chain, not an approximation of it, and
// `ARC_QK_VERIFY=1` proves that on real tensors at every layer of every step,
// with a negative control.
//
// Three things make that achievable rather than merely hoped for:
//
//  1. **The reduction tree shape is copied, not chosen.** candle's `fast_sum`
//     (candle-kernels/src/reduce.cu) accumulates in T -- NOT in f32; see the
//     standing TODO at the top of that file -- into `__shared__ T shr[]`, one
//     element per thread, then halves with `for (s = blockDim.x/2; s > 0;
//     s >>= 1)`. The result depends on `blockDim.x`, which candle sets to
//     `min(1024, el_to_sum_per_block).next_power_of_two()`. `HEAD_DIM` here is
//     that same block width (asserted host-side in `qk_norm_rope.rs`), so the
//     tree has the same shape and the same rounding. A row length whose block
//     width does not equal the row length is REFUSED, not approximated.
//
//  2. **The FMA contraction is REPRODUCED, not avoided.** This was got wrong
//     once and the `ARC_QK_VERIFY` harness caught it, so the correction is
//     recorded here with its measurement. bf16 arithmetic is NOT immune to
//     `--fmad`: nvcc contracts one of the two products of `a*c - b*s` into an
//     `fma.rn.bf16`, so that product is rounded once with the subtraction
//     instead of twice. Measured on an H200 for head 0 / pair 0 at position 1
//     (a = -5.65625, b = -3.546875, cos = 0x3f0a, sin = 0x3f57, and the
//     diagnostic confirmed BOTH sides read the same table row):
//
//         round(round(a*c) - round(b*s))  = -0.062500      (--fmad=false)
//         round_f32(a*c - b*s)            = -0.069824      (full f32)
//         round(a*c - round(b*s))         = -0.064453125   (CANDLE)
//
//     The third is an exact tie -- significand 1.02734375, midway between
//     1.0234375 and 1.03125 -- resolved by round-half-to-even. Building this
//     file with `--fmad=false` therefore made 306 of the 4,096 RoPE outputs
//     per layer differ, all on the low-j (high-frequency) pairs, because
//     high-j pairs have cos ~ 1 / sin ~ 0 and round the same either way.
//     `build.rs` consequently gives this file its OWN builder carrying
//     candle-kernels' exact flags (`-O3 --expt-relaxed-constexpr -std=c++17`,
//     default `-fmad=true`, no fast math) rather than the IEEE builder that
//     sinkhorn.cu and hc_fused.cu need. The f32 path is still REFUSED below:
//     the same reasoning would have to be re-established for f32's own
//     contraction behaviour, and it has not been measured.
//
//  3. **The scalars are converted host-side by the same code candle uses.**
//     candle's `affine(mul, add)` narrows f64 -> T before the kernel sees it;
//     `1/512` and `rms_norm_eps` arrive here as bf16 bit patterns produced by
//     `half::bf16::from_f64`, so no rounding decision is taken twice.
//
// This file is compiled by the DEDICATED IEEE BUILDER in build.rs, alongside
// sinkhorn.cu and hc_fused.cu, and carries the same guard against being
// re-globbed under `--use_fast_math`.
// ---------------------------------------------------------------------------

#ifdef __CUDA_ARCH__
#if defined(__CUDACC_RTC__)
#error "qk_norm_rope.cu must not be compiled with NVRTC"
#endif
#endif

// Guard against this file being moved back under the fast-math glob in
// build.rs. `--use_fast_math` defines __CUDA_FAST_MATH__ and would rewrite the
// f32 helpers used by hsqrt/__hdiv's approximation sandwich.
#ifdef __CUDA_FAST_MATH__
#error "qk_norm_rope.cu is bit-identity critical and must be built WITHOUT --use_fast_math (see build.rs: it belongs to the IEEE builder, not the fast-math glob)"
#endif

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>

// dtype tags shared with qk_norm_rope.rs. Only BF16 is accepted; see (2) above.
#define QK_DTYPE_BF16 1

namespace {

// candle `urecip_bf16` == `recipg(x)` == `__nv_bfloat16 one = 1.0; one / a`
// (candle-kernels/src/cuda_utils.cuh). `operator/` is `__hdiv`, a div.rn.f32
// sandwich. Spelled the same way here so the same PTX is emitted.
__device__ __forceinline__ __nv_bfloat16 qk_recip(__nv_bfloat16 a) {
    __nv_bfloat16 one = 1.0;
    return one / a;
}

// candle `usqrt_bf16` == `sqrtg(x)` == `hsqrt(a)`.
__device__ __forceinline__ __nv_bfloat16 qk_sqrt(__nv_bfloat16 a) { return hsqrt(a); }

}  // namespace

// ---------------------------------------------------------------------------
// The fused kernel.
//
// Grid  : (H_Q + 1, B * T)   -- one block per output row; the +1 row is K.
// Block : HEAD_DIM threads   -- one thread per element, which is also exactly
//                               candle's fast_sum block width for this row
//                               length, so the reduction tree matches.
//
// Q in  : [B, T, H, D] contiguous, straight off the q_proj GEMM (NOT
//         transposed -- the transpose is this kernel's output addressing).
// K in  : [B, T, 1, D] contiguous, straight off kv_norm.
// Q out : [B, H, T, D] contiguous.
// K out : [B, 1, T, D] contiguous.
//
// `cos`/`sin` are the FULL [max_pos, D_rope/2] tables; `pos_offset` is
// `seqlen_offsets[0]`, i.e. candle's `self.cos.narrow(0, offset, seq_len)`
// expressed as a base offset instead of a view.
// ---------------------------------------------------------------------------
template <int HEAD_DIM, int ROPE_DIM>
__global__ void qk_norm_rope_kernel(
    const __nv_bfloat16 *__restrict__ q_in,
    const __nv_bfloat16 *__restrict__ k_in,
    const __nv_bfloat16 *__restrict__ cos_tab,
    const __nv_bfloat16 *__restrict__ sin_tab,
    __nv_bfloat16 *__restrict__ q_out,
    __nv_bfloat16 *__restrict__ k_out,
    const int n_heads,
    const int seq_len,
    const __nv_bfloat16 inv_n,   // bf16(1.0 / HEAD_DIM)   -- affine mul
    const __nv_bfloat16 zero,    // bf16(0.0)              -- affine add
    const __nv_bfloat16 one,     // bf16(1.0)              -- affine mul
    const __nv_bfloat16 eps,     // bf16(rms_norm_eps)     -- affine add
    const int pos_offset) {
    constexpr int NOPE = HEAD_DIM - ROPE_DIM;
    constexpr int HALF_ROPE = ROPE_DIM / 2;

    const int row = blockIdx.x;          // 0..n_heads-1 => Q head; n_heads => K
    const int bt = blockIdx.y;           // flattened (b, t)
    const int t = bt % seq_len;
    const int b = bt / seq_len;
    const int tid = threadIdx.x;
    const bool is_k = (row == n_heads);

    const __nv_bfloat16 *src;
    __nv_bfloat16 *dst;
    if (is_k) {
        // K: [B, T, 1, D] -> [B, 1, T, D]
        src = k_in + (size_t)bt * HEAD_DIM;
        dst = k_out + (size_t)((size_t)b * seq_len + t) * HEAD_DIM;
    } else {
        // Q: [B, T, H, D] -> [B, H, T, D]   (the materialised transpose)
        src = q_in + ((size_t)bt * n_heads + row) * HEAD_DIM;
        dst = q_out + (((size_t)b * n_heads + row) * seq_len + t) * HEAD_DIM;
    }

    // -- 1. Q only: the RMS statistic ---------------------------------------
    //
    // candle: `q.sqr()?.mean_keepdim(-1)?.affine(1.0, eps)?.recip()?.sqrt()?`
    // then `q.broadcast_mul(&inv_rms)`. K does NOT get this -- it has already
    // been through `kv_norm` (a weighted RMSNorm module) before reaching here.
    // Initialised to exactly 1.0 only so it is never read uninitialised on the
    // K row; every use is guarded by `!is_k`, because K has already been
    // through `kv_norm` and must NOT be scaled again.
    __nv_bfloat16 inv_rms = one;
    if (!is_k) {
        __shared__ __nv_bfloat16 shr[HEAD_DIM];
        // `usqr_bf16` is `x*x`, rounded to bf16 BEFORE fast_sum reads it, so
        // the multiply and the accumulate are two separate roundings here too.
        __nv_bfloat16 x = src[tid];
        shr[tid] = x * x;
        // candle fast_sum's tree, verbatim: __syncthreads() at the TOP of the
        // body, and the halving order it implies.
        for (int s = HEAD_DIM / 2; s > 0; s >>= 1) {
            __syncthreads();
            if (tid < s) shr[tid] += shr[tid + s];
        }
        // candle stops here (only tid 0 reads shr[0]); every thread needs the
        // statistic to scale its own element, so one more barrier is added.
        // A barrier changes no arithmetic.
        __syncthreads();
        __nv_bfloat16 sum = shr[0];

        // `mean_keepdim` == `sum_keepdim` then `* (1/n)`, which candle lowers
        // to `affine(mul = 1/n, add = 0)`: `x * mul + add`.
        __nv_bfloat16 mean = sum * inv_n + zero;
        // `.affine(1.0, eps)`
        __nv_bfloat16 denom = mean * one + eps;
        inv_rms = qk_sqrt(qk_recip(denom));
    }

    // -- 2. NoPE dims: scale (Q) or copy (K) --------------------------------
    //
    // This is the `cat([nope, pe])` low half. candle materialises it with a
    // copy2d out of the normalised tensor; here it is written straight to the
    // transposed destination.
    for (int i = tid; i < NOPE; i += HEAD_DIM) {
        dst[i] = is_k ? src[i] : (__nv_bfloat16)(src[i] * inv_rms);
    }

    // -- 3. RoPE on the trailing ROPE_DIM dims ------------------------------
    //
    // candle `ropei` (candle-kernels/src/reduce.cu):
    //     dst[2i]   = src[2i] * c - src[2i+1] * s
    //     dst[2i+1] = src[2i] * s + src[2i+1] * c
    // all in T. `rope_idx` there resolves to `t * (ROPE_DIM/2) + j` against the
    // narrowed table, i.e. `(pos_offset + t) * (ROPE_DIM/2) + j` against the
    // full one.
    const __nv_bfloat16 *cos_row = cos_tab + (size_t)(pos_offset + t) * HALF_ROPE;
    const __nv_bfloat16 *sin_row = sin_tab + (size_t)(pos_offset + t) * HALF_ROPE;
    for (int j = tid; j < HALF_ROPE; j += HEAD_DIM) {
        // The rotation reads the NORMALISED value, matching candle's order
        // (broadcast_mul happens before apply_rope_inplace splits the tensor).
        __nv_bfloat16 a = src[NOPE + 2 * j];
        __nv_bfloat16 bb = src[NOPE + 2 * j + 1];
        if (!is_k) {
            a = a * inv_rms;
            bb = bb * inv_rms;
        }
        const __nv_bfloat16 c = cos_row[j];
        const __nv_bfloat16 s = sin_row[j];
        // candle's `ropei` spells this `src[2i]*c - src[2i+1]*s`, and its PTX
        // is contracted by the driver JIT so that the FIRST product is fused
        // into the add/sub (one rounding) while the SECOND is rounded on its
        // own. Written out explicitly because the operator form compiles to
        // inline PTX here, which nvcc cannot contract at any `--fmad` setting.
        // See the bit-identity contract at the top of this file for the
        // measurement that pins this down.
        const __nv_bfloat16 bs = bb * s;  // round(b*s)
        const __nv_bfloat16 bc = bb * c;  // round(b*c)
        dst[NOPE + 2 * j] = __hfma(a, c, __hneg(bs));      // round(a*c - round(b*s))
        dst[NOPE + 2 * j + 1] = __hfma(a, s, bc);          // round(a*s + round(b*c))
    }
}

extern "C" {

// Returns 0 if the kernel was launched, non-zero if this (dtype, head_dim,
// rope_dim) combination is outside the specialised set. A non-zero return means
// the caller MUST fall back to the eager chain: producing a wrong answer from a
// generic slow path would be worse than the launches it saves.
int arc_qk_norm_rope_bf16_v2(
    const void *q_in,
    const void *k_in,
    const void *cos_tab,
    const void *sin_tab,
    void *q_out,
    void *k_out,
    int n_heads,
    int batch,
    int seq_len,
    int head_dim,
    int rope_dim,
    int pos_offset,
    int dtype,
    // Widened from uint16_t to a full word each. The bf16 bit patterns are
    // carried in the low 16 bits. Sub-word arguments packed behind five other
    // stack arguments are exactly the kind of boundary that is cheaper to
    // remove than to reason about.
    unsigned int inv_n_bits,
    unsigned int zero_bits,
    unsigned int one_bits,
    unsigned int eps_bits,
    long long stream) {
    if (dtype != QK_DTYPE_BF16) return 1;

    __nv_bfloat16 inv_n, zero, one, eps;
    unsigned short b;
    b = (unsigned short)(inv_n_bits & 0xFFFFu); memcpy(&inv_n, &b, sizeof(b));
    b = (unsigned short)(zero_bits & 0xFFFFu);  memcpy(&zero, &b, sizeof(b));
    b = (unsigned short)(one_bits & 0xFFFFu);   memcpy(&one, &b, sizeof(b));
    b = (unsigned short)(eps_bits & 0xFFFFu);   memcpy(&eps, &b, sizeof(b));

    // One-shot probe: print what actually crossed the FFI boundary, once, so a
    // wrong value is OBSERVED rather than inferred from output symptoms.
    static int probed = 0;
    if (!probed) {
        probed = 1;
        fprintf(stderr,
                "[qk_norm_rope FFI] n_heads=%d batch=%d seq_len=%d head_dim=%d "
                "rope_dim=%d pos_offset=%d dtype=%d inv_n=0x%04x zero=0x%04x "
                "one=0x%04x eps=0x%04x\n",
                n_heads, batch, seq_len, head_dim, rope_dim, pos_offset, dtype,
                inv_n_bits & 0xFFFFu, zero_bits & 0xFFFFu, one_bits & 0xFFFFu,
                eps_bits & 0xFFFFu);
        fflush(stderr);
    }

    dim3 grid((unsigned)(n_heads + 1), (unsigned)(batch * seq_len), 1);
    cudaStream_t s = (cudaStream_t)stream;

#define QK_LAUNCH(HD, RD)                                                       \
    qk_norm_rope_kernel<HD, RD><<<grid, HD, 0, s>>>(                            \
        (const __nv_bfloat16 *)q_in, (const __nv_bfloat16 *)k_in,               \
        (const __nv_bfloat16 *)cos_tab, (const __nv_bfloat16 *)sin_tab,         \
        (__nv_bfloat16 *)q_out, (__nv_bfloat16 *)k_out, n_heads, seq_len,       \
        inv_n, zero, one, eps, pos_offset)

    // Only shapes whose row length IS candle's fast_sum block width are
    // specialised -- see the bit-identity contract, point (1). V4 Flash is
    // (512, 64); the others are here so a config change fails loudly at the
    // dispatch rather than silently rounding differently.
    if (head_dim == 512 && rope_dim == 64) {
        QK_LAUNCH(512, 64);
    } else if (head_dim == 256 && rope_dim == 64) {
        QK_LAUNCH(256, 64);
    } else if (head_dim == 192 && rope_dim == 64) {
        // 192 is not a power of two: candle's block width would be 256, not
        // 192, so the tree shape would differ. Refuse.
        return 2;
    } else {
        return 2;
    }
#undef QK_LAUNCH

    return 0;
}

}  // extern "C"
