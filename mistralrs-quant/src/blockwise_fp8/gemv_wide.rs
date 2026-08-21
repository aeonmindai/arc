//! Parent system: ArcQuant / ArcKernels.
//!
//! Host-side gate and CPU mirror for `fp8_gemv_wide`, the 128-bit-load
//! rewrite of the b=1 decode GEMV in
//! `kernels/blockwise_fp8/blockwise_fp8_gemm.cu`.
//!
//! # Why a CPU mirror exists
//!
//! The wide kernel is **not bit-identical** to `fp8_gemv_warp` and cannot be:
//! a `uint4` load forces lane `L` to own the 16 contiguous weights at
//! `k_base + 16L`, where the 32-bit kernel gives it the 4 at `k_base + 4L`. A
//! different lane→k partition means different per-lane partial sums, hence
//! different f32 rounding.
//!
//! What *is* claimed, and what this module exists to hold to account:
//!
//! 1. Both kernels partition `[0, K)` into the **same** aligned 4-element
//!    groups, and evaluate each group through the identical expression.
//! 2. The scale multiplying each group is **bit-identical**, because
//!    `k >> log2(bsx)` and `k / bsx` agree exactly for non-negative `k` and
//!    power-of-two `bsx`.
//! 3. The kernels therefore differ **only** in the order the group
//!    contributions are accumulated: one f32 accumulator per lane versus
//!    four. That is f32 re-association error, nothing else.
//!
//! [`accumulate_warp32`] and [`accumulate_wide128`] below reproduce those two
//! accumulation orders in Rust, so claims 1-3 are checked by `cargo test` on
//! any machine, with no GPU. They model the **order**, not nvcc's FMA
//! contraction — contraction is identical in both kernels and so cancels out
//! of the comparison.
//!
//! # Why the gate is opt-in
//!
//! 🔴 The kernel is UNVERIFIED ON HARDWARE. [`wide_enabled`] reads
//! `ARC_FP8_GEMV_WIDE` and is true **only** for the exact value `"1"`.
//! `ARC_FP8_GEMV_WIDE=0`, `=off`, `=true` and unset all leave it OFF. That
//! strictness is deliberate: this crate already ships `ARC_NO_FP8_GEMV`, read
//! with `std::env::var(..).is_ok()` (`blockwise_fp8/ops.rs`), where `=0`
//! *enables* the disable — the polarity bug this form cannot have.

// This module is exactly two things: a gate that only a CUDA build calls, and
// a CPU model of the kernel that only `mod tests` calls. On a non-CUDA build
// neither has a ship-path caller, so `dead_code` fires on the whole file —
// but the model is load-bearing for `cargo test` on any machine, which is the
// entire point of writing it. Blanket-allow rather than sprinkle `cfg_attr`.
#![allow(dead_code)]

use float8::F8E4M3;

/// Number of K-elements one warp of `fp8_gemv_wide` covers per iteration:
/// 32 lanes x 16 bytes.
pub(crate) const WIDE_K_PER_ITER: usize = 512;
/// Number of K-elements one warp of `fp8_gemv_warp` covers per iteration:
/// 32 lanes x 4 bytes.
pub(crate) const WARP_K_PER_ITER: usize = 128;
/// K-elements one lane of `fp8_gemv_wide` owns per iteration.
pub(crate) const WIDE_K_PER_LANE: usize = 16;

/// `ARC_FP8_GEMV_WIDE` — opt-in switch for the wide b=1 GEMV.
///
/// Split from the env read so the polarity is unit-testable. Only the exact
/// string `"1"` enables.
pub(crate) fn wide_enabled_from(value: Option<&str>) -> bool {
    value == Some("1")
}

/// `ARC_FP8_GEMV_WIDE=1` enables `fp8_gemv_wide` for eligible b=1 shapes.
/// Default OFF. Read once; the decode hot loop never touches the env.
#[cfg(feature = "cuda")]
pub(crate) fn wide_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| wide_enabled_from(std::env::var("ARC_FP8_GEMV_WIDE").ok().as_deref()))
}

/// The kernel's `scale_shift` argument, or `None` if this `block_size_x`
/// cannot use the wide kernel.
///
/// Two requirements, both load-bearing:
///
/// * **power of two** — the kernel indexes scales with `k >> scale_shift`
///   where the 32-bit kernel divides by `block_size_x`. Those agree exactly
///   iff `block_size_x` is a power of two (and `k >= 0`, which it is).
/// * **a multiple of 16** — a lane owns the 16 contiguous weights at
///   `k_base + 16*lane`. One scale load per lane is only correct if that whole
///   slice sits inside one scale block. See
///   [`tests::lane_slice_never_straddles_a_scale_block`].
///
/// DeepSeek blockwise FP8 ships `weight_block_size = [128, 128]`, which
/// satisfies both. Anything else keeps `fp8_gemv_warp`.
pub(crate) fn scale_shift_for(block_size_x: usize) -> Option<i32> {
    if block_size_x < WIDE_K_PER_LANE
        || !block_size_x.is_power_of_two()
        || !block_size_x.is_multiple_of(WIDE_K_PER_LANE)
    {
        return None;
    }
    Some(block_size_x.trailing_zeros() as i32)
}

/// Full host-side eligibility test for `fp8_gemv_wide`.
///
/// `weight_addr` / `input_addr` are the **device pointers actually passed to
/// the kernel** (post `start_offset`), not the base allocations: a `uint4`
/// load traps on a misaligned address, and a tensor sliced out of a larger
/// buffer can land anywhere. Checking the real pointers is exact and costs
/// two ANDs on the host.
///
/// `k % 16 == 0` covers both remaining alignment obligations at once: it makes
/// every weight row start 16-byte aligned relative to the base (rows are `k`
/// bytes apart), and it makes the activation row offset `m*k` a multiple of 8
/// elements = 16 bytes.
///
/// Returns the `scale_shift` to pass to the kernel.
pub(crate) fn wide_eligible(
    k: usize,
    block_size_x: usize,
    weight_addr: usize,
    input_addr: usize,
) -> Option<i32> {
    if !k.is_multiple_of(WIDE_K_PER_LANE) {
        return None;
    }
    if !weight_addr.is_multiple_of(16) || !input_addr.is_multiple_of(16) {
        return None;
    }
    scale_shift_for(block_size_x)
}

// ---------------------------------------------------------------------------
// CPU mirror
// ---------------------------------------------------------------------------

/// Deliberate defects, used by [`tests::the_guard_is_live`] to prove the
/// numerical assertions can actually fail. Not reachable from any ship path.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Mutation {
    /// The kernel as written.
    None,
    /// Kernel no-ops: returns the zero it initialised `acc` to.
    NoOpKernel,
    /// `k >> (shift + 1)` — the index bug a wrong `scale_shift` would cause.
    WrongShift,
    /// The `[K_wide, K)` tail is dropped.
    DropTail,
    /// The final combine forgets `acc3`, i.e. one group slot in four is lost.
    DropOneAccumulator,
}

/// One FP8 weight, decoded exactly as `fp8_to_float` does on device.
///
/// The device path is FP8 -> f16 -> f32. Every E4M3 value (max magnitude 448,
/// 3 mantissa bits, subnormals included) is exactly representable in f16, so
/// that hop is lossless and the result equals a direct FP8 -> f32 decode.
#[inline]
fn deq(bits: u8) -> f32 {
    F8E4M3::from_bits(bits).to_f32()
}

/// The 4-element group both kernels share, in the association order nvcc
/// produces for `i0*f0 + i1*f1 + i2*f2 + i3*f3` (one FMUL then three FFMAs,
/// left to right). Identical in both accumulation orders, so any difference
/// between them is re-association of the *groups*, never of a group's insides.
#[inline]
fn group4(w: &[u8], x: &[f32], k: usize) -> f32 {
    let g = x[k] * deq(w[k]);
    let g = f32::mul_add(x[k + 1], deq(w[k + 1]), g);
    let g = f32::mul_add(x[k + 2], deq(w[k + 2]), g);
    f32::mul_add(x[k + 3], deq(w[k + 3]), g)
}

/// The `__shfl_down_sync` butterfly both kernels end with, offsets 16..1.
/// Returns what lane 0 holds afterwards.
fn warp_reduce(mut acc: [f32; 32]) -> f32 {
    let mut offset = 16usize;
    while offset > 0 {
        for lane in 0..32 {
            let src = lane + offset;
            if src < 32 {
                acc[lane] += acc[src];
            }
        }
        offset /= 2;
    }
    acc[0]
}

/// `fp8_gemv_warp`'s accumulation order: 4 weight bytes per lane per
/// iteration, ONE f32 accumulator per lane, groups consumed in ascending `k`.
pub(crate) fn accumulate_warp32(w: &[u8], x: &[f32], scales: &[f32], block_size_x: usize) -> f32 {
    let k_len = w.len();
    let k_aligned = (k_len / WARP_K_PER_ITER) * WARP_K_PER_ITER;
    let mut acc = [0f32; 32];

    for (lane, a) in acc.iter_mut().enumerate() {
        let mut k_base = 0usize;
        while k_base < k_aligned {
            let k = k_base + lane * 4;
            let scale = scales[k / block_size_x];
            *a = f32::mul_add(scale, group4(w, x, k), *a);
            k_base += WARP_K_PER_ITER;
        }
        // Scalar remainder, verbatim from the kernel: `acc += scale*in*f(w)`.
        let mut k = k_aligned + lane;
        while k < k_len {
            let scale = scales[k / block_size_x];
            *a = f32::mul_add(scale * x[k], deq(w[k]), *a);
            k += 32;
        }
    }

    warp_reduce(acc)
}

/// `fp8_gemv_wide`'s accumulation order: 16 weight bytes per lane per
/// iteration, FOUR f32 accumulators per lane (one per group slot), then the
/// shared tail, then `((a0+a1)+(a2+a3))+tail`.
pub(crate) fn accumulate_wide128(
    w: &[u8],
    x: &[f32],
    scales: &[f32],
    block_size_x: usize,
    mutation: Mutation,
) -> f32 {
    if mutation == Mutation::NoOpKernel {
        return 0.0;
    }
    let shift = scale_shift_for(block_size_x).expect("wide path requires an eligible block_size_x");
    let shift = match mutation {
        Mutation::WrongShift => shift + 1,
        _ => shift,
    } as usize;

    let k_len = w.len();
    let k_wide = (k_len / WIDE_K_PER_ITER) * WIDE_K_PER_ITER;
    let k_aligned = k_wide + ((k_len - k_wide) / WARP_K_PER_ITER) * WARP_K_PER_ITER;
    let mut acc = [0f32; 32];

    for (lane, out) in acc.iter_mut().enumerate() {
        let mut a = [0f32; 4];

        let mut k_base = 0usize;
        while k_base < k_wide {
            let k = k_base + lane * WIDE_K_PER_LANE;
            let scale = scales[k >> shift];
            for (g, slot) in a.iter_mut().enumerate() {
                *slot = f32::mul_add(scale, group4(w, x, k + 4 * g), *slot);
            }
            k_base += WIDE_K_PER_ITER;
        }

        let mut tail = 0f32;
        if mutation != Mutation::DropTail {
            let mut k_base = k_wide;
            while k_base < k_aligned {
                let k = k_base + lane * 4;
                let scale = scales[k >> shift];
                tail = f32::mul_add(scale, group4(w, x, k), tail);
                k_base += WARP_K_PER_ITER;
            }
            let mut k = k_aligned + lane;
            while k < k_len {
                let scale = scales[k >> shift];
                tail = f32::mul_add(scale * x[k], deq(w[k]), tail);
                k += 32;
            }
        }

        let a3 = if mutation == Mutation::DropOneAccumulator {
            0.0
        } else {
            a[3]
        };
        *out = ((a[0] + a[1]) + (a[2] + a3)) + tail;
    }

    warp_reduce(acc)
}

/// f64 ground truth: the same dot product with no rounding worth speaking of.
pub(crate) fn reference_f64(w: &[u8], x: &[f32], scales: &[f32], block_size_x: usize) -> f64 {
    let mut acc = 0f64;
    for k in 0..w.len() {
        acc += scales[k / block_size_x] as f64 * x[k] as f64 * deq(w[k]) as f64;
    }
    acc
}

/// Every `k` the wide kernel touches, in the order it touches them. Used to
/// prove the mapping is an exact cover of `[0, K)`.
///
/// This — not the numerical bound — is the guard that catches a dropped or
/// mis-strided tail. On an ill-conditioned row the tail can be a few percent
/// of the sum while the f32 re-association bound is larger than that, so the
/// numerical comparison is blind to it by construction. Coverage is not.
pub(crate) fn wide_k_cover(k_len: usize, mutation: Mutation) -> Vec<usize> {
    let k_wide = (k_len / WIDE_K_PER_ITER) * WIDE_K_PER_ITER;
    let k_aligned = k_wide + ((k_len - k_wide) / WARP_K_PER_ITER) * WARP_K_PER_ITER;
    let mut out = Vec::with_capacity(k_len);
    if mutation == Mutation::NoOpKernel {
        return out;
    }

    for lane in 0..32usize {
        let mut k_base = 0usize;
        while k_base < k_wide {
            let k = k_base + lane * WIDE_K_PER_LANE;
            let groups = if mutation == Mutation::DropOneAccumulator {
                3
            } else {
                4
            };
            for g in 0..groups {
                for e in 0..4 {
                    out.push(k + 4 * g + e);
                }
            }
            k_base += WIDE_K_PER_ITER;
        }
        if mutation == Mutation::DropTail {
            continue;
        }
        let mut k_base = k_wide;
        while k_base < k_aligned {
            let k = k_base + lane * 4;
            for e in 0..4 {
                out.push(k + e);
            }
            k_base += WARP_K_PER_ITER;
        }
        let mut k = k_aligned + lane;
        while k < k_len {
            out.push(k);
            k += 32;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic, dependency-free PRNG so the numerical rows are
    /// reproducible byte for byte on any machine.
    struct Lcg(u64);
    impl Lcg {
        fn next_u32(&mut self) -> u32 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (self.0 >> 32) as u32
        }
        /// Uniform in [-1, 1), then rounded through bf16 — the dtype the b=1
        /// decode path actually carries.
        fn next_bf16_as_f32(&mut self) -> f32 {
            let u = (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32;
            let v = u * 2.0 - 1.0;
            f32::from_bits(v.to_bits() & 0xFFFF_0000)
        }
        /// Approximately standard normal (sum of 6 uniforms, Irwin-Hall).
        /// Good enough to stand in for a weight matrix; no rand dependency.
        fn next_normal(&mut self) -> f32 {
            let s: f32 = (0..6)
                .map(|_| (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32)
                .sum();
            (s - 3.0) * std::f32::consts::SQRT_2
        }
    }

    /// A realistic blockwise-FP8 row: weights are a normal sample rounded
    /// through E4M3 (what quantizing a real weight matrix produces), not
    /// uniform bit patterns — those would mix 448.0 with subnormals and make
    /// the dot product so ill-conditioned that every numerical bound is
    /// vacuous.
    fn make_case(k_len: usize, block_size_x: usize, seed: u64) -> (Vec<u8>, Vec<f32>, Vec<f32>) {
        let mut rng = Lcg(seed);
        let w: Vec<u8> = (0..k_len)
            .map(|_| F8E4M3::from_f32(rng.next_normal()).to_bits())
            .collect();
        let x: Vec<f32> = (0..k_len).map(|_| rng.next_bf16_as_f32()).collect();
        let scales: Vec<f32> = (0..k_len.div_ceil(block_size_x))
            .map(|_| 0.5 + (rng.next_u32() >> 8) as f32 / (1u32 << 24) as f32)
            .collect();
        (w, x, scales)
    }

    /// `sum |scale * x_k * dequant(w_k)|` in f64 — the conditioning term in
    /// the textbook f32 summation bound. Using this instead of `|sum|` makes
    /// the bound loose exactly when the problem is ill-conditioned, and tight
    /// when it is not, rather than being loose by a constant everywhere.
    fn abs_sum_f64(w: &[u8], x: &[f32], scales: &[f32], block_size_x: usize) -> f64 {
        (0..w.len())
            .map(|k| (scales[k / block_size_x] as f64 * x[k] as f64 * deq(w[k]) as f64).abs())
            .sum()
    }

    /// Textbook bound on how far two orderings of the same n-term f32 sum can
    /// diverge: each is within `(n-1) * u * sum|term|` of the exact value with
    /// `u = 2^-24`, so they are within twice that of each other. `n` is the
    /// number of 4-element GROUPS, since a group is atomic in both orders.
    /// The 4x is headroom for the FMA structure inside a group.
    fn reassociation_bound(k_len: usize, abs_sum: f64) -> f64 {
        let n_groups = (k_len / 4) as f64;
        4.0 * n_groups * (f32::EPSILON as f64 / 2.0) * abs_sum
    }

    /// CLAIM 2, the only bit-exactness claim the wide kernel makes: replacing
    /// the runtime `k / block_size_x` with `k >> log2(block_size_x)` changes
    /// nothing. Exhaustive over a full 1M-element K for the shipped 128, and
    /// over every other eligible block size.
    #[test]
    fn shift_is_exactly_the_division_it_replaces() {
        for bsx in [16usize, 32, 64, 128, 256, 512, 1024, 4096] {
            let shift = scale_shift_for(bsx).expect("power of two, multiple of 16") as usize;
            assert_eq!(1usize << shift, bsx, "shift {shift} does not encode {bsx}");
            for k in 0..(1usize << 20) {
                assert_eq!(
                    k >> shift,
                    k / bsx,
                    "k={k} bsx={bsx}: shift gave {} but divide gave {}",
                    k >> shift,
                    k / bsx
                );
            }
        }
    }

    /// The wide kernel issues ONE scale load per lane per iteration. That is
    /// only correct if a lane's 16 contiguous weights never straddle a scale
    /// block boundary.
    #[test]
    fn lane_slice_never_straddles_a_scale_block() {
        for bsx in [16usize, 32, 64, 128, 256, 512] {
            let shift = scale_shift_for(bsx).unwrap() as usize;
            for k_base in (0..65_536).step_by(WIDE_K_PER_ITER) {
                for lane in 0..32usize {
                    let k = k_base + lane * WIDE_K_PER_LANE;
                    assert_eq!(
                        k >> shift,
                        (k + WIDE_K_PER_LANE - 1) >> shift,
                        "bsx={bsx} lane={lane} k={k}: slice spans two scale blocks"
                    );
                }
            }
        }
    }

    /// CLAIM 1: the wide mapping (wide loop + 128-wide tail + scalar
    /// remainder) covers every k in [0, K) exactly once. A stride, offset or
    /// tail-bound bug shows up here and nowhere else.
    #[test]
    fn wide_mapping_is_an_exact_cover() {
        for k_len in [
            512usize, 1024, 1536, 2048, 4096, 7168, 640, 528, 16, MUTATION_K,
        ] {
            let uncovered = cover_defect(k_len, Mutation::None);
            assert!(
                uncovered.is_none(),
                "K={k_len}: {}",
                uncovered.unwrap_or_default()
            );
        }
    }

    /// `None` if the wide mapping covers `[0, K)` exactly once; otherwise a
    /// description of the first defect found.
    fn cover_defect(k_len: usize, mutation: Mutation) -> Option<String> {
        let cover = wide_k_cover(k_len, mutation);
        if cover.len() != k_len {
            return Some(format!("touched {} of {k_len} k-positions", cover.len()));
        }
        let mut seen = vec![0u8; k_len];
        for k in cover {
            if k >= k_len {
                return Some(format!("touched out-of-range k={k}"));
            }
            seen[k] += 1;
        }
        let bad: Vec<usize> = (0..k_len).filter(|&k| seen[k] != 1).collect();
        if bad.is_empty() {
            None
        } else {
            Some(format!(
                "{} k not covered exactly once, first {:?}",
                bad.len(),
                &bad[..bad.len().min(8)]
            ))
        }
    }

    /// The K used by the mutation harness. Chosen so BOTH tail loops run:
    /// 4240 = 8*512 (wide) + 128 (the 32-bit 128-wide loop) + 16 (the scalar
    /// remainder), and 4240 % 16 == 0 so the shape is eligible.
    const MUTATION_K: usize = 4240;

    /// CLAIM 3: the two orders differ only by f32 re-association, and the
    /// four-accumulator order is not systematically worse against f64 truth.
    #[test]
    fn wide_and_warp_orders_agree_and_wide_is_no_worse() {
        let bsx = 128usize;
        let mut sq_wide = 0f64;
        let mut sq_warp = 0f64;
        let mut cases = 0usize;

        for (i, &k_len) in [512usize, 1024, 2048, 4096, 7168, 1536, 640, MUTATION_K]
            .iter()
            .enumerate()
        {
            for seed in 0..8u64 {
                let (w, x, scales) = make_case(k_len, bsx, 0x5EED_0000 + (i as u64) * 97 + seed);
                let warp = accumulate_warp32(&w, &x, &scales, bsx);
                let wide = accumulate_wide128(&w, &x, &scales, bsx, Mutation::None);
                let truth = reference_f64(&w, &x, &scales, bsx);
                let bound = reassociation_bound(k_len, abs_sum_f64(&w, &x, &scales, bsx));

                // The load-bearing assertion: the two orders cannot differ by
                // more than f32 re-association of the same group values.
                let diff = (wide as f64 - warp as f64).abs();
                assert!(
                    diff <= bound,
                    "K={k_len} seed={seed}: wide={wide} warp={warp} diff={diff} \
                     exceeds re-association bound {bound}"
                );
                // Neither order may drift further from truth than that either.
                assert!(
                    (wide as f64 - truth).abs() <= bound,
                    "K={k_len} seed={seed}: wide={wide} vs f64 truth {truth}, bound {bound}"
                );

                sq_wide += (wide as f64 - truth).powi(2);
                sq_warp += (warp as f64 - truth).powi(2);
                cases += 1;
            }
        }

        // Four accumulators means chains a quarter as long, so RMS error
        // should be about half. Assert only that it is not WORSE — the
        // directional claim, without pretending to a constant we did not
        // derive.
        let rms_wide = (sq_wide / cases as f64).sqrt();
        let rms_warp = (sq_warp / cases as f64).sqrt();
        assert!(
            rms_wide <= 1.25 * rms_warp,
            "over {cases} rows the 4-accumulator order was less accurate than the \
             1-accumulator order: rms_wide={rms_wide} rms_warp={rms_warp}"
        );
    }

    /// MUTATION HARNESS. Every assertion above is worthless if it cannot go
    /// red. Each defect here is a plausible way to get `fp8_gemv_wide` wrong,
    /// and each must be caught by the guard that is *supposed* to catch it.
    ///
    /// The split is deliberate and is itself a finding. `DropTail` is worth
    /// 144 of 4240 terms; on an ill-conditioned row that is smaller than the
    /// legitimate f32 re-association bound, so the numerical comparison is
    /// blind to it **by construction** and only the coverage test sees it.
    /// A harness that asserted "every mutation breaks the numerical bound"
    /// would have to be tuned until it did — which is how a test stops
    /// testing anything.
    #[test]
    fn the_guard_is_live() {
        let bsx = 128usize;
        let (w, x, scales) = make_case(MUTATION_K, bsx, 0xC0FFEE);
        let warp = accumulate_warp32(&w, &x, &scales, bsx);
        let bound = reassociation_bound(MUTATION_K, abs_sum_f64(&w, &x, &scales, bsx));

        let clean = accumulate_wide128(&w, &x, &scales, bsx, Mutation::None);
        let clean_diff = (clean as f64 - warp as f64).abs();
        assert!(
            clean_diff <= bound,
            "unmutated wide kernel already fails: diff={clean_diff} bound={bound}"
        );
        println!(
            "clean: wide={clean} warp={warp} diff={clean_diff} bound={bound} \
             (headroom {:.1}x)",
            bound / clean_diff.max(f64::MIN_POSITIVE)
        );

        // Numerical defects: must break the CLAIM-3 comparison.
        for m in [
            Mutation::NoOpKernel,
            Mutation::WrongShift,
            Mutation::DropOneAccumulator,
        ] {
            let got = accumulate_wide128(&w, &x, &scales, bsx, m);
            let diff = (got as f64 - warp as f64).abs();
            println!("mutation {m:?}: wide={got} warp={warp} diff={diff} bound={bound}");
            assert!(
                diff > bound,
                "MUTATION {m:?} SLIPPED THROUGH: wide={got} warp={warp} diff={diff} \
                 is still inside bound {bound} — the CLAIM-3 assertion cannot fail"
            );
        }

        // Structural defects: must break the coverage test.
        for m in [
            Mutation::NoOpKernel,
            Mutation::DropTail,
            Mutation::DropOneAccumulator,
        ] {
            let defect = cover_defect(MUTATION_K, m);
            println!("mutation {m:?}: coverage says {defect:?}");
            assert!(
                defect.is_some(),
                "MUTATION {m:?} SLIPPED THROUGH: the coverage of [0, {MUTATION_K}) is \
                 still exact — `wide_mapping_is_an_exact_cover` cannot fail"
            );
        }
    }

    /// `Mutation::DropTail` is only a defect if the K it runs on has a tail,
    /// and it should exercise BOTH tail loops. A K that is a multiple of 512
    /// would silently turn `the_guard_is_live` into a test of nothing.
    #[test]
    fn the_mutation_harness_k_exercises_both_tail_loops() {
        let k_wide = (MUTATION_K / WIDE_K_PER_ITER) * WIDE_K_PER_ITER;
        let k_aligned = k_wide + ((MUTATION_K - k_wide) / WARP_K_PER_ITER) * WARP_K_PER_ITER;
        assert!(
            k_wide > 0,
            "K={MUTATION_K} has no wide section, so the wide loop is untested"
        );
        assert!(
            k_aligned > k_wide,
            "K={MUTATION_K} has no 128-wide tail, so that loop is untested"
        );
        assert!(
            MUTATION_K > k_aligned,
            "K={MUTATION_K} has no scalar remainder, so that loop is untested"
        );
        assert_eq!(
            MUTATION_K % WIDE_K_PER_LANE,
            0,
            "K={MUTATION_K} is not eligible"
        );
    }

    /// Env polarity: only the exact string "1" turns the kernel on. Guards
    /// against the `ARC_NO_DEDICATED_DECODE` failure, where `var_os(..)
    /// .is_some()` makes `=0` *enable* the flag.
    #[test]
    fn only_literal_one_enables() {
        assert!(wide_enabled_from(Some("1")));
        for v in [
            "0", "", "on", "off", "true", "false", "yes", "no", "2", " 1", "1 ",
        ] {
            assert!(
                !wide_enabled_from(Some(v)),
                "`ARC_FP8_GEMV_WIDE={v}` must not enable"
            );
        }
        assert!(!wide_enabled_from(None), "unset must not enable");
    }

    /// The gate must reject every shape whose preconditions the kernel relies
    /// on. Each rejection below corresponds to a specific way a `uint4` load
    /// or a shifted scale index would be wrong.
    #[test]
    fn gate_rejects_ineligible_shapes() {
        // The shipped DeepSeek shape passes.
        assert_eq!(wide_eligible(7168, 128, 0x1000, 0x2000), Some(7));

        // K not a multiple of 16: rows are not 16-byte aligned to each other.
        assert_eq!(wide_eligible(7160, 128, 0x1000, 0x2000), None);
        // Misaligned weight pointer: `uint4` would trap.
        assert_eq!(wide_eligible(7168, 128, 0x1008, 0x2000), None);
        // Misaligned input pointer.
        assert_eq!(wide_eligible(7168, 128, 0x1000, 0x2004), None);
        // Non-power-of-two block: `>>` is not `/`.
        assert_eq!(wide_eligible(7168, 96, 0x1000, 0x2000), None);
        assert_eq!(wide_eligible(7168, 192, 0x1000, 0x2000), None);
        // Block smaller than a lane's 16-wide slice: one scale load is wrong.
        assert_eq!(wide_eligible(7168, 8, 0x1000, 0x2000), None);
        assert_eq!(wide_eligible(7168, 4, 0x1000, 0x2000), None);

        // `scale_shift_for` in isolation.
        assert_eq!(scale_shift_for(16), Some(4));
        assert_eq!(scale_shift_for(128), Some(7));
        assert_eq!(scale_shift_for(12), None);
        assert_eq!(scale_shift_for(0), None);
    }
}
