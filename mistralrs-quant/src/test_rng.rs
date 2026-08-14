//! Deterministic test fixtures for numeric-tolerance tests.
//!
//! **Why this module exists.** Three separate tests in this workspace have now
//! been found asserting a tight numeric bound against an *unseeded*
//! `Tensor::randn`, so every run drew a fresh input and the assertion was a
//! coin flip on the tail of a distribution nobody had measured:
//!
//! | test | old bound | draws over it (n = 20 000) | fixed in |
//! |---|---|---|---|
//! | `dsv4_indexer::make_indexer_tensors` (mistralrs-core) | — | — | wave3-rebase |
//! | `vector_fp8::test_fp8_vector_quant_cpu` | `< 0.27` | 108 = 0.54% | wave14-AK |
//! | `vector_fp8::test_fp8_vector_quant_dequant_roundtrip` | `< 0.24` | 851 = 4.26% | wave14-AL |
//! | `blockwise_fp8::test_fp8_blockwise_quant_dequant_roundtrip` | `< 0.16` | 4517 = 22.6% | wave14-AL |
//!
//! The last two are `#[cfg(feature = "cuda")]`, so they never run in CI or on a
//! developer laptop — they run on a rented GPU box, where a spurious red costs
//! money and tempts someone into debugging on the clock (DOCTRINE D4b). The
//! blockwise one fails better than one run in five.
//!
//! The fix in every case is the same and is deliberately kept in ONE place so
//! the fourth instance has somewhere to go: pin the input, measure the real
//! error on *that* input, and set the bound from the measurement with a stated
//! margin. Never widen a bound until the flake stops — every bound replaced
//! this way has ended up tighter than the one before it.

/// Deterministic pseudo-Gaussian values (SplitMix64 + Box-Muller): the same
/// seed yields bit-identical `f32`s on every run and every platform.
///
/// SplitMix64 is exact integer arithmetic, so the only cross-platform freedom
/// is the last bit of `ln` / `sin_cos` in the Box-Muller transform. Tests built
/// on this should therefore carry a margin that is generous against last-bit
/// drift and tight against everything else; the drift is measurable by
/// perturbing the output and re-running the metric.
#[cfg(test)]
pub(crate) fn det_randn(seed: u64, mean: f32, std: f32, n: usize) -> Vec<f32> {
    let mut state = seed;
    let mut next = move || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    let mut vals = Vec::with_capacity(n + 1);
    while vals.len() < n {
        // Box-Muller from two uniforms in (0, 1].
        let u1 = ((next() >> 11) as f64 + 1.0) / (1u64 << 53) as f64;
        let u2 = ((next() >> 11) as f64 + 1.0) / (1u64 << 53) as f64;
        let r = (-2.0 * u1.ln()).sqrt();
        let (s, c) = (2.0 * std::f64::consts::PI * u2).sin_cos();
        vals.push((mean as f64 + std as f64 * r * c) as f32);
        vals.push((mean as f64 + std as f64 * r * s) as f32);
    }
    vals.truncate(n);
    vals
}

#[cfg(test)]
mod tests {
    use super::det_randn;

    /// The whole point of the helper: same seed ⇒ same bytes, and a different
    /// seed ⇒ different bytes. A generator that quietly returned a constant
    /// would make every bound built on it vacuous.
    #[test]
    fn det_randn_is_reproducible_and_not_constant() {
        let a = det_randn(0xF8E4_0001, 0.0, 2.0, 256);
        let b = det_randn(0xF8E4_0001, 0.0, 2.0, 256);
        assert_eq!(
            a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            b.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "the same seed must reproduce bit-identical values"
        );

        let c = det_randn(0xF8E4_0002, 0.0, 2.0, 256);
        assert_ne!(a, c, "different seeds must produce different draws");

        // Non-degenerate and roughly N(0, 2): a helper that returned zeros, or
        // that lost the scale, would silently defang every bound built on it.
        assert_eq!(a.len(), 256);
        let mean = a.iter().sum::<f32>() / a.len() as f32;
        let var = a.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / a.len() as f32;
        assert!(mean.abs() < 0.5, "mean {mean} is not near 0");
        assert!(
            (2.0..=6.0).contains(&var),
            "variance {var} is not near 4 (std 2)"
        );

        // Odd lengths must not over-run: Box-Muller produces values in pairs.
        assert_eq!(det_randn(1, 0.0, 1.0, 7).len(), 7);
    }
}
