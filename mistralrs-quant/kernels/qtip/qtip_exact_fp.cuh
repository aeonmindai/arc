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

// Total order over f32 matching Rust's `f32::total_cmp`, as an unsigned key:
// ordering the keys ascending orders the floats by `total_cmp` ascending.
// (`total_cmp` maps bits to i32 and flips the low 31 bits of negatives; adding
// the sign flip that turns a signed compare into an unsigned one gives exactly
// the classic sign-magnitude-to-unsigned transform below.)
__device__ __forceinline__ unsigned int qtip_total_order_key(float x) {
    const unsigned int b = __float_as_uint(x);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}
