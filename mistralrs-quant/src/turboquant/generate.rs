//! Runtime Lloyd–Max codebook generation for arbitrary TurboQuant block dims.
//!
//! # Why this exists
//!
//! [`super::codebook::get_codebook`] shipped with hand-pasted tables for
//! `dim ∈ {64, 128, 256}` only, and panicked for anything else. That table —
//! not the mathematics, not the kernels — is what made "TurboQuant runs on
//! head_dim 64/128/256" true and "TurboQuant runs on Llama-3.2 (64) but not
//! Phi-2 (80), Gemma (256 ✓), DeepSeek-V4 (512 ✗)" true. The distribution the
//! codebook is optimal for is a closed-form function of the dimension, so the
//! table was never a mathematical limit; it was three rows of a function
//! nobody had written down in Rust.
//!
//! # The distribution
//!
//! After the randomized Walsh–Hadamard rotation `D·H·D`, a unit-norm input
//! vector is (to the accuracy the flattening lemma gives) a uniformly random
//! point on the sphere `S^{d-1}`. The marginal density of **one coordinate**
//! of such a point is
//!
//! ```text
//!     f_d(x) ∝ (1 - x²)^((d-3)/2),      x ∈ [-1, 1]
//! ```
//!
//! i.e. `(x+1)/2 ~ Beta((d-1)/2, (d-1)/2)`. The module doc on
//! [`super::codebook`] says "Beta(d/2, d/2)"; that is off by one half in both
//! shape parameters. The exponent above is the one that reproduces the shipped
//! tables — see `generator_reproduces_shipped_tables` in this file, which pins
//! `mse_per_coord` to the shipped constants to 1e-9 for all nine
//! `(dim, bits)` pairs. `(d-2)/2` and `(d-1)/2` were both tried and both miss.
//!
//! # The algorithm
//!
//! Textbook Lloyd–Max (Lloyd's algorithm for a known continuous density):
//!
//! 1. boundaries are midpoints of adjacent centroids (nearest-neighbour rule),
//! 2. each centroid becomes the conditional mean of its cell (centroid rule),
//! 3. repeat to a fixed point.
//!
//! The centroid rule's **numerator is analytic** — `∫ x (1-x²)^m dx` has the
//! closed form `-(1-x²)^(m+1) / (2(m+1))` — so only the cell mass needs
//! quadrature. That is why this converges to ~1e-7 on the centroids with a few
//! hundred Simpson nodes per cell instead of the ~1e-4 a plain histogram gets
//! with 400k global grid points.
//!
//! Cost is ~10 ms per `(dim, bits)` pair, paid once per process and memoised.

use super::codebook::Codebook;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

/// Smallest block dimension TurboQuant will rotate and quantize.
///
/// Below this the flattening lemma has no room to work: a `d`-coordinate
/// rotation spreads energy over `d` buckets, and with `d < 8` the rotated
/// coordinates are not concentrated enough for a scalar codebook to beat
/// simply storing the numbers. It is also the point where the density
/// `(1-x²)^((d-3)/2)` stops being unimodal-and-bounded (`d = 3` is uniform,
/// `d < 3` is unbounded at ±1), so the generator itself would need special
/// cases.
pub const MIN_BLOCK_DIM: usize = 8;

/// Largest block dimension the generator will produce a codebook for.
///
/// Not a mathematical limit — a guard so a nonsense `head_dim` from a corrupt
/// config cannot make the process spend minutes in Lloyd iterations.
pub const MAX_BLOCK_DIM: usize = 8192;

/// Simpson's rule over `[a, b]` with `n` (even) intervals.
fn simpson<F: Fn(f64) -> f64>(f: F, a: f64, b: f64, n: usize) -> f64 {
    debug_assert!(n.is_multiple_of(2));
    if b <= a {
        return 0.0;
    }
    let h = (b - a) / n as f64;
    let mut s = f(a) + f(b);
    for i in 1..n {
        let w = if i % 2 == 1 { 4.0 } else { 2.0 };
        s += w * f(a + i as f64 * h);
    }
    s * h / 3.0
}

/// Unnormalised density `(1 - x²)^m` evaluated stably.
#[inline]
fn dens(x: f64, m: f64) -> f64 {
    let t = 1.0 - x * x;
    if t <= 0.0 {
        0.0
    } else {
        (m * t.ln()).exp()
    }
}

/// Closed form of `∫_a^b x (1-x²)^m dx` (unnormalised).
#[inline]
fn moment1(a: f64, b: f64, m: f64) -> f64 {
    let p = |x: f64| {
        let t = 1.0 - x * x;
        if t <= 0.0 {
            0.0
        } else {
            ((m + 1.0) * t.ln()).exp()
        }
    };
    (p(a) - p(b)) / (2.0 * (m + 1.0))
}

/// The generated codebook for one `(dim, bits)` pair.
pub struct GeneratedCodebook {
    pub centroids: Vec<f32>,
    pub boundaries: Vec<f32>,
    pub mse_per_coord: f32,
}

/// Compute the Lloyd–Max codebook for the `S^{dim-1}` coordinate marginal.
///
/// Deterministic: no RNG, no platform-dependent iteration order. The same
/// `(dim, bits)` always yields bit-identical output on any machine with IEEE
/// doubles, which is what lets the CUDA side upload these values and stay in
/// agreement with the CPU reference.
pub fn generate_codebook(dim: usize, bits: u32) -> GeneratedCodebook {
    assert!(
        (MIN_BLOCK_DIM..=MAX_BLOCK_DIM).contains(&dim),
        "TurboQuant codebook dim {dim} outside [{MIN_BLOCK_DIM}, {MAX_BLOCK_DIM}]"
    );
    assert!(
        (2..=4).contains(&bits),
        "TurboQuant codebook bits {bits} outside [2, 4]"
    );

    let n = 1usize << bits;
    let m = (dim as f64 - 3.0) / 2.0;
    // Simpson nodes per cell. 512 puts the cell-mass error far below the
    // 1e-13 fixed-point tolerance below for every dim in range.
    const NODES: usize = 512;

    // Initialise centroids spread over ±0.9 sd. The fixed point is unique for
    // a log-concave density, so the initialisation only affects iteration
    // count, not the answer; this one converges in <60 iterations everywhere.
    let sd = 1.0 / (dim as f64).sqrt();
    let mut cent: Vec<f64> = (0..n)
        .map(|k| {
            let t = (2 * k) as f64 - (n - 1) as f64;
            (t * 0.9 * sd).clamp(-1.0, 1.0)
        })
        .collect();

    let bounds_of = |c: &[f64]| -> Vec<f64> {
        let mut b = Vec::with_capacity(n + 1);
        b.push(-1.0);
        for i in 0..n - 1 {
            b.push((c[i] + c[i + 1]) / 2.0);
        }
        b.push(1.0);
        b
    };

    for _ in 0..500 {
        let b = bounds_of(&cent);
        let mut next = Vec::with_capacity(n);
        let mut delta: f64 = 0.0;
        for k in 0..n {
            let mass = simpson(|x| dens(x, m), b[k], b[k + 1], NODES);
            let c = if mass > 0.0 {
                moment1(b[k], b[k + 1], m) / mass
            } else {
                cent[k]
            };
            delta = delta.max((c - cent[k]).abs());
            next.push(c);
        }
        cent = next;
        if delta < 1e-13 {
            break;
        }
    }

    // Distortion: E[(X - Q(X))²] under the normalised density.
    let b = bounds_of(&cent);
    let z = simpson(|x| dens(x, m), -1.0, 1.0, 8192);
    let mut mse = 0.0;
    for k in 0..n {
        let i0 = simpson(|x| dens(x, m), b[k], b[k + 1], NODES);
        let i1 = moment1(b[k], b[k + 1], m);
        let i2 = simpson(|x| x * x * dens(x, m), b[k], b[k + 1], NODES);
        mse += i2 - 2.0 * cent[k] * i1 + cent[k] * cent[k] * i0;
    }

    GeneratedCodebook {
        centroids: cent.iter().map(|&c| c as f32).collect(),
        boundaries: b.iter().map(|&x| x as f32).collect(),
        mse_per_coord: (mse / z) as f32,
    }
}

type CodebookCache = Mutex<HashMap<(usize, u32), Codebook>>;
static CACHE: OnceLock<CodebookCache> = OnceLock::new();

/// Memoised [`generate_codebook`], returning the `'static` view [`Codebook`]
/// needs.
///
/// The leak is bounded and intentional: the key space is
/// `(block dims a model can have) × {2,3,4}`, which is a handful of entries
/// for the process lifetime, and a codebook that outlives every cache that
/// points at it is exactly the lifetime we want.
pub fn cached_generated_codebook(dim: usize, bits: u32) -> Codebook {
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = cache.lock().unwrap();
    *guard.entry((dim, bits)).or_insert_with(|| {
        let g = generate_codebook(dim, bits);
        Codebook {
            centroids: Box::leak(g.centroids.into_boxed_slice()),
            boundaries: Box::leak(g.boundaries.into_boxed_slice()),
            mse_per_coord: g.mse_per_coord,
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::turboquant::codebook::get_codebook;

    /// The generator must land on the same optimum as the three hand-pasted
    /// tables that shipped — otherwise every dim it newly unlocks is quantized
    /// against a codebook nobody validated.
    ///
    /// `mse_per_coord` is the discriminating check: it is the quantity the
    /// codebook is optimal *for*, and it is pinned to 1e-9. Centroid positions
    /// are checked more loosely (2e-4) because distortion is second-order in
    /// centroid displacement near the optimum, so the shipped tables' own
    /// generator precision shows up there and not in the MSE.
    #[test]
    fn generator_reproduces_shipped_tables() {
        for dim in [64usize, 128, 256] {
            for bits in [2u32, 3, 4] {
                let shipped = get_codebook(dim, bits);
                let gen = generate_codebook(dim, bits);
                assert_eq!(gen.centroids.len(), shipped.centroids.len());
                assert_eq!(gen.boundaries.len(), shipped.boundaries.len());
                assert!(
                    (gen.mse_per_coord - shipped.mse_per_coord).abs() < 1e-9,
                    "dim={dim} bits={bits}: generated mse {} vs shipped {}",
                    gen.mse_per_coord,
                    shipped.mse_per_coord
                );
                for i in 0..gen.centroids.len() {
                    assert!(
                        (gen.centroids[i] - shipped.centroids[i]).abs() < 2e-4,
                        "dim={dim} bits={bits} centroid[{i}]: {} vs {}",
                        gen.centroids[i],
                        shipped.centroids[i]
                    );
                }
            }
        }
    }

    /// The distortion of an optimal `b`-bit scalar quantizer for a unit-norm
    /// vector is `≈ D_b` *independent of the dimension* — that is the whole
    /// reason TurboQuant's error bound does not degrade as head_dim grows, and
    /// the direct refutation of "512 is too big for TurboQuant".
    ///
    /// `D_b` are the Lloyd–Max Gaussian distortions (0.1175 / 0.03454 /
    /// 0.009497 for 2/3/4 bits); the sphere marginal is sub-Gaussian so the
    /// true values sit just under, approaching from below as `dim → ∞`.
    #[test]
    fn total_distortion_is_dimension_free() {
        for &(bits, limit) in &[(2u32, 0.1175f32), (3, 0.03454), (4, 0.009497)] {
            let mut prev = 0.0f32;
            for dim in [8usize, 16, 32, 64, 80, 96, 112, 128, 192, 256, 512, 1024] {
                let g = generate_codebook(dim, bits);
                let total = g.mse_per_coord * dim as f32;
                assert!(
                    total <= limit,
                    "dim={dim} bits={bits}: total distortion {total} exceeds the \
                     Gaussian Lloyd-Max limit {limit}"
                );
                assert!(
                    total > 0.7 * limit,
                    "dim={dim} bits={bits}: total distortion {total} implausibly \
                     below the limit {limit} — generator is probably wrong"
                );
                // Monotone in dim: the marginal tends to Gaussian from below.
                assert!(
                    total >= prev - 1e-6,
                    "dim={dim} bits={bits}: distortion {total} < previous {prev}"
                );
                prev = total;
            }
        }
    }

    #[test]
    fn generated_codebooks_are_sorted_and_symmetric() {
        for dim in [8usize, 80, 96, 112, 192, 512] {
            for bits in [2u32, 3, 4] {
                let g = generate_codebook(dim, bits);
                let n = g.centroids.len();
                for i in 1..n {
                    assert!(
                        g.centroids[i] > g.centroids[i - 1],
                        "dim={dim} bits={bits} centroids not sorted"
                    );
                }
                for i in 1..g.boundaries.len() {
                    assert!(
                        g.boundaries[i] > g.boundaries[i - 1],
                        "dim={dim} bits={bits} boundaries not sorted"
                    );
                }
                for i in 0..n / 2 {
                    assert!(
                        (g.centroids[i] + g.centroids[n - 1 - i]).abs() < 1e-6,
                        "dim={dim} bits={bits} asymmetric at {i}"
                    );
                }
                assert_eq!(g.boundaries[0], -1.0);
                assert_eq!(g.boundaries[g.boundaries.len() - 1], 1.0);
            }
        }
    }

    /// Memoisation must not change the answer.
    #[test]
    fn cached_matches_fresh() {
        let a = cached_generated_codebook(192, 4);
        let b = generate_codebook(192, 4);
        assert_eq!(a.centroids, b.centroids.as_slice());
        assert_eq!(a.boundaries, b.boundaries.as_slice());
        // Second call hits the cache and must be the same allocation.
        let c = cached_generated_codebook(192, 4);
        assert!(std::ptr::eq(a.centroids, c.centroids));
    }
}
