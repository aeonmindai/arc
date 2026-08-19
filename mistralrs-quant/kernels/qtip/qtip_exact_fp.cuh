// Per-operation IEEE-754 f32 arithmetic for the QTIP trellis quantizers.
//
// WHY THIS FILE EXISTS
// --------------------
// `mistralrs-quant/build.rs` compiles every kernel in `kernels/*/*.cu` with
// `--use_fast_math`, which implies (nvcc docs, "Options for Steering GPU Code
// Generation"):
//
//   * `-fmad=true`  — the compiler contracts `a*b + c` into a single-rounded
//     FMA. `d0*d0 + d1*d1` therefore becomes `fmaf(d1, d1, d0*d0)` and rounds
//     ONCE where the Rust reference rounds twice.
//   * `-prec-div=false` — the `/` operator becomes an approximate
//     reciprocal-multiply, so `1.0f / scale` is not the IEEE quotient.
//
// The Rust trellis reference (`qtip/viterbi.rs::decode_error`, and the row
// scaling in `qtip/mod.rs`) evaluates every operation separately in IEEE f32:
// rustc performs no FMA contraction and no reciprocal substitution without
// explicit fast-math opt-in. So a GPU bake and a CPU bake of the *same* weights
// could previously disagree in the last ulp of the branch metric — enough to
// flip a Viterbi tie and emit a different symbol. That is precisely the silent
// CPU/GPU divergence wave13-AD's bake header exists to make impossible.
//
// `__fadd_rn` / `__fsub_rn` / `__fmul_rn` are documented as "This operation
// will never be merged into a single multiply-add instruction"
// (CUDA Math API, Single Precision Intrinsics), and `__fdiv_rn` is the
// IEEE round-to-nearest division intrinsic — unaffected by `-prec-div`.
// Writing the branch metric with them makes the CUDA trellis search
// bit-identical to the Rust one.
//
// Denormals: `--use_fast_math` also implies `-ftz=true`, which DOES affect
// these intrinsics. It cannot bite inside the metric. The targets and LUT
// values live in roughly [-4, 4], where the gap between adjacent f32 is
// ~2.4e-7, so a nonzero `d = l - t` has |d| >= 2.4e-7 and d*d >= 5.7e-14 —
// twenty-four orders of magnitude above the smallest normal (1.18e-38). The
// only other outcome is d == 0 exactly, and 0*0 == 0 either way. The one
// residual is an input weight that is itself denormal (|w| < 1.18e-38): the
// GPU would flush `w * inv_scale` to zero where the CPU would not. Trained
// weights never reach that magnitude, and an exact zero (which is common)
// behaves identically on both sides.

#pragma once

#include <cstdint>

// IEEE reciprocal of the per-row scale. Mirrors `let inv_scale = 1.0 / scale;`
// in `QtipLayer::quantize_with_options_concrete_calibrated`.
__device__ __forceinline__ float qtip_inv_scale_exact(float scale) {
    return __fdiv_rn(1.0f, scale);
}

// One element of the scaled target row: `w * inv_scale`.
__device__ __forceinline__ float qtip_scaled_target_exact(float w, float inv_scale) {
    return __fmul_rn(w, inv_scale);
}

// Squared error of decoding trellis state `s` against the V=2 target vector,
// evaluated exactly as `qtip/viterbi.rs::decode_error` does:
//
//     let mut err = 0f32;
//     for v in 0..2 { let d = lut[off + v] - target[v]; err += d * d; }
//
// `0.0 + x == x` in IEEE for every x that can appear here (err >= 0), so the
// accumulator start is a no-op and the expression reduces to two independent
// products followed by one add.
__device__ __forceinline__ float qtip_decode_err_exact(
    const float* __restrict__ lut, unsigned int s, float t0, float t1
) {
    const float l0 = lut[(size_t)s * 2u + 0u];
    const float l1 = lut[(size_t)s * 2u + 1u];
    const float d0 = __fsub_rn(l0, t0);
    const float d1 = __fsub_rn(l1, t1);
    return __fadd_rn(__fmul_rn(d0, d0), __fmul_rn(d1, d1));
}

// Same, but the two LUT values are already in registers (the beam kernel reads
// a group's 16 successors as one contiguous 128-byte run).
__device__ __forceinline__ float qtip_decode_err_exact_lv(
    float l0, float l1, float t0, float t1
) {
    const float d0 = __fsub_rn(l0, t0);
    const float d1 = __fsub_rn(l1, t1);
    return __fadd_rn(__fmul_rn(d0, d0), __fmul_rn(d1, d1));
}

// V-generic form of the same metric, for rungs where the reproduction vector is
// not two-dimensional (K=8/V=4/L=12). The Rust reference accumulates
//
//     let mut err = 0f32;
//     for v in 0..V { let d = c[v] - target[v]; err += d * d; }
//
// which is a LEFT fold. `0.0 + x == x` exactly for every `x = d*d` (a square is
// `+0.0` or larger, never `-0.0`), so seeding the accumulator with the v=0 term
// instead of `0.0f` is the identical value and saves an add. At V == 2 the
// expansion is `__fadd_rn(__fmul_rn(d0,d0), __fmul_rn(d1,d1))` — instruction for
// instruction `qtip_decode_err_exact_lv` above, which is what keeps the K=4/V=2
// rung byte-identical through this generalisation.
template <uint32_t V>
__device__ __forceinline__ float qtip_decode_err_exact_vec(
    const float* __restrict__ c, const float* __restrict__ t
) {
    const float d0 = __fsub_rn(c[0], t[0]);
    float acc = __fmul_rn(d0, d0);
    #pragma unroll
    for (uint32_t v = 1; v < V; ++v) {
        const float d = __fsub_rn(c[v], t[v]);
        acc = __fadd_rn(acc, __fmul_rn(d, d));
    }
    return acc;
}

// Total order over f32 matching Rust's `f32::total_cmp`, as an unsigned key:
// ordering the keys ascending orders the floats by `total_cmp` ascending.
// (`total_cmp` maps bits to i32 and flips the low 31 bits of negatives; adding
// the sign flip that turns a signed compare into an unsigned one gives exactly
// the classic sign-magnitude-to-unsigned transform below.)
__device__ __forceinline__ unsigned int qtip_total_order_key(float x) {
    const unsigned int b = __float_as_uint(x);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

// Exact inverse of `qtip_total_order_key`. The map is a bijection on the 2^32
// bit patterns, so a candidate can be carried through the selection as its
// ordered key alone and turned back into the identical float at the end —
// which is what lets the kernel stop rebuilding a key from a float on every
// radix pass (wave17-AF). Round-trip identity is pinned by
// `order_key_round_trips_bitwise` on the Rust side.
__device__ __forceinline__ float qtip_key_to_float(unsigned int k) {
    const unsigned int b = (k & 0x80000000u) ? (k & 0x7FFFFFFFu) : ~k;
    return __uint_as_float(b);
}
