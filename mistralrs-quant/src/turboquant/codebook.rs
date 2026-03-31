/// Lloyd-Max optimal codebooks for TurboQuant KV cache compression.
///
/// These are pre-computed centroids and decision boundaries for the Beta(d/2, d/2)
/// distribution on [-1, 1], which is the marginal distribution of a single coordinate
/// of a uniformly random point on the unit hypersphere S^{d-1} (Lemma 1, TurboQuant paper).
///
/// After a randomized Walsh-Hadamard rotation, every coordinate of the input vector
/// follows this distribution regardless of the original data, making these codebooks
/// universally optimal (data-oblivious).
///
/// Reference: TurboQuant (ICLR 2026), arXiv:2504.19874, Section 3.1, Eq. (4).

/// A codebook for a specific (dimension, bit-width) pair.
#[derive(Debug, Clone, Copy)]
pub struct Codebook {
    /// Reconstruction levels (centroids), sorted ascending. Length = 2^bits.
    pub centroids: &'static [f32],
    /// Decision boundaries. Length = 2^bits + 1. boundaries[0] = -1.0, boundaries[last] = 1.0.
    /// A value x maps to centroid i if boundaries[i] <= x < boundaries[i+1].
    pub boundaries: &'static [f32],
    /// MSE per coordinate for this codebook.
    pub mse_per_coord: f32,
}

impl Codebook {
    /// Quantize a single scalar value to its nearest centroid index.
    /// Uses binary search on sorted boundaries.
    #[inline]
    pub fn quantize(&self, x: f32) -> u8 {
        // Binary search: find the bucket where boundaries[i] <= x < boundaries[i+1]
        let n = self.centroids.len();
        // Clamp to valid range
        if x <= self.boundaries[1] {
            return 0;
        }
        if x >= self.boundaries[n] {
            return (n - 1) as u8;
        }
        // Binary search over interior boundaries
        let mut lo = 1usize;
        let mut hi = n;
        while lo < hi {
            let mid = (lo + hi) / 2;
            if x < self.boundaries[mid] {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        (lo - 1) as u8
    }

    /// Dequantize an index back to its centroid value.
    #[inline]
    pub fn dequantize(&self, idx: u8) -> f32 {
        self.centroids[idx as usize]
    }

    /// Number of bits per element for this codebook.
    pub fn bits(&self) -> u32 {
        match self.centroids.len() {
            4 => 2,
            8 => 3,
            16 => 4,
            _ => unreachable!(),
        }
    }
}

// ============================================================================
// d=64 codebooks
// ============================================================================

static D64_2BIT_CENTROIDS: [f32; 4] = [
    -0.187496894581, -0.056514879971, 0.056514879971, 0.187496894581,
];
static D64_2BIT_BOUNDARIES: [f32; 5] = [
    -1.0, -0.122005887276, 0.0, 0.122005887276, 1.0,
];

static D64_3BIT_CENTROIDS: [f32; 8] = [
    -0.263914011139, -0.166167896845, -0.093832282234, -0.030469184818,
     0.030469184818,  0.093832282234,  0.166167896845,  0.263914011139,
];
static D64_3BIT_BOUNDARIES: [f32; 9] = [
    -1.0, -0.215040953992, -0.130000089540, -0.062150733526,
     0.0,  0.062150733526,  0.130000089540,  0.215040953992, 1.0,
];

static D64_4BIT_CENTROIDS: [f32; 16] = [
    -0.330796418715, -0.252913787303, -0.198856181058, -0.154925556401,
    -0.116486772790, -0.081311782388, -0.048089791275, -0.015919025204,
     0.015919025204,  0.048089791275,  0.081311782388,  0.116486772790,
     0.154925556401,  0.198856181058,  0.252913787303,  0.330796418715,
];
static D64_4BIT_BOUNDARIES: [f32; 17] = [
    -1.0, -0.291855103009, -0.225884984180, -0.176890868729,
    -0.135706164595, -0.098899277589, -0.064700786832, -0.032004408240,
     0.0,  0.032004408240,  0.064700786832,  0.098899277589,
     0.135706164595,  0.176890868729,  0.225884984180,  0.291855103009, 1.0,
];

// ============================================================================
// d=128 codebooks
// ============================================================================

static D128_2BIT_CENTROIDS: [f32; 4] = [
    -0.133041590561, -0.039991612341, 0.039991612341, 0.133041590561,
];
static D128_2BIT_BOUNDARIES: [f32; 5] = [
    -1.0, -0.086516601451, 0.0, 0.086516601451, 1.0,
];

static D128_3BIT_CENTROIDS: [f32; 8] = [
    -0.188397319183, -0.118139828402, -0.066585638471, -0.021604320011,
     0.021604320011,  0.066585638471,  0.118139828402,  0.188397319183,
];
static D128_3BIT_BOUNDARIES: [f32; 9] = [
    -1.0, -0.153268573792, -0.092362733436, -0.044094979241,
     0.0,  0.044094979241,  0.092362733436,  0.153268573792, 1.0,
];

static D128_4BIT_CENTROIDS: [f32; 16] = [
    -0.237664013127, -0.180836062501, -0.141805261760, -0.110288414632,
    -0.082828489390, -0.057772320256, -0.034151583096, -0.011302500645,
     0.011302500645,  0.034151583096,  0.057772320256,  0.082828489390,
     0.110288414632,  0.141805261760,  0.180836062501,  0.237664013127,
];
static D128_4BIT_BOUNDARIES: [f32; 17] = [
    -1.0, -0.209250037814, -0.161320662130, -0.126046838196,
    -0.096558452011, -0.070300404823, -0.045961951676, -0.022727041871,
     0.0,  0.022727041871,  0.045961951676,  0.070300404823,
     0.096558452011,  0.126046838196,  0.161320662130,  0.209250037814, 1.0,
];

// ============================================================================
// d=256 codebooks
// ============================================================================

static D256_2BIT_CENTROIDS: [f32; 4] = [
    -0.094237888048, -0.028288624543, 0.028288624543, 0.094237888048,
];
static D256_2BIT_BOUNDARIES: [f32; 5] = [
    -1.0, -0.061263256296, 0.0, 0.061263256296, 1.0,
];

static D256_3BIT_CENTROIDS: [f32; 8] = [
    -0.133854499678, -0.083765552399, -0.047166757397, -0.015297502119,
     0.015297502119,  0.047166757397,  0.083765552399,  0.133854499678,
];
static D256_3BIT_BOUNDARIES: [f32; 9] = [
    -1.0, -0.108810026038, -0.065466154898, -0.031232129758,
     0.0,  0.031232129758,  0.065466154898,  0.108810026038, 1.0,
];

static D256_4BIT_CENTROIDS: [f32; 16] = [
    -0.169410740456, -0.128588358534, -0.100698110176, -0.078249384982,
    -0.058732157584, -0.040949230131, -0.024200896741, -0.008008380402,
     0.008008380402,  0.024200896741,  0.040949230131,  0.058732157584,
     0.078249384982,  0.100698110176,  0.128588358534,  0.169410740456,
];
static D256_4BIT_BOUNDARIES: [f32; 17] = [
    -1.0, -0.148999549495, -0.114643234355, -0.089473747579,
    -0.068490771283, -0.049840693858, -0.032575063436, -0.016104638571,
     0.0,  0.016104638571,  0.032575063436,  0.049840693858,
     0.068490771283,  0.089473747579,  0.114643234355,  0.148999549495, 1.0,
];

// ============================================================================
// Lookup function
// ============================================================================

/// Get the codebook for a given head dimension and bit-width.
///
/// Supported dimensions: 64, 128, 256.
/// Supported bit-widths: 2, 3, 4.
///
/// Panics if an unsupported (dim, bits) combination is requested.
pub fn get_codebook(dim: usize, bits: u32) -> Codebook {
    match (dim, bits) {
        (64, 2) => Codebook {
            centroids: &D64_2BIT_CENTROIDS,
            boundaries: &D64_2BIT_BOUNDARIES,
            mse_per_coord: 0.0017894641,
        },
        (64, 3) => Codebook {
            centroids: &D64_3BIT_CENTROIDS,
            boundaries: &D64_3BIT_BOUNDARIES,
            mse_per_coord: 0.0005217316,
        },
        (64, 4) => Codebook {
            centroids: &D64_4BIT_CENTROIDS,
            boundaries: &D64_4BIT_BOUNDARIES,
            mse_per_coord: 0.0001426799,
        },
        (128, 2) => Codebook {
            centroids: &D128_2BIT_CENTROIDS,
            boundaries: &D128_2BIT_BOUNDARIES,
            mse_per_coord: 0.0009062496,
        },
        (128, 3) => Codebook {
            centroids: &D128_3BIT_CENTROIDS,
            boundaries: &D128_3BIT_BOUNDARIES,
            mse_per_coord: 0.0002653586,
        },
        (128, 4) => Codebook {
            centroids: &D128_4BIT_CENTROIDS,
            boundaries: &D128_4BIT_BOUNDARIES,
            mse_per_coord: 0.0000727717,
        },
        (256, 2) => Codebook {
            centroids: &D256_2BIT_CENTROIDS,
            boundaries: &D256_2BIT_BOUNDARIES,
            mse_per_coord: 0.0004560148,
        },
        (256, 3) => Codebook {
            centroids: &D256_3BIT_CENTROIDS,
            boundaries: &D256_3BIT_BOUNDARIES,
            mse_per_coord: 0.0001338123,
        },
        (256, 4) => Codebook {
            centroids: &D256_4BIT_CENTROIDS,
            boundaries: &D256_4BIT_BOUNDARIES,
            mse_per_coord: 0.0000367481,
        },
        _ => panic!(
            "TurboQuant codebook not available for dim={dim}, bits={bits}. \
             Supported: dim={{64,128,256}}, bits={{2,3,4}}"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_codebook_symmetry() {
        // All codebooks should be symmetric around 0
        for dim in [64, 128, 256] {
            for bits in [2, 3, 4] {
                let cb = get_codebook(dim, bits);
                let n = cb.centroids.len();
                for i in 0..n / 2 {
                    let diff = (cb.centroids[i] + cb.centroids[n - 1 - i]).abs();
                    assert!(
                        diff < 1e-6,
                        "Asymmetric centroids at dim={dim}, bits={bits}, i={i}: {} vs {}",
                        cb.centroids[i],
                        cb.centroids[n - 1 - i]
                    );
                }
            }
        }
    }

    #[test]
    fn test_codebook_sorted() {
        for dim in [64, 128, 256] {
            for bits in [2, 3, 4] {
                let cb = get_codebook(dim, bits);
                for i in 1..cb.centroids.len() {
                    assert!(
                        cb.centroids[i] > cb.centroids[i - 1],
                        "Centroids not sorted at dim={dim}, bits={bits}"
                    );
                }
                for i in 1..cb.boundaries.len() {
                    assert!(
                        cb.boundaries[i] > cb.boundaries[i - 1],
                        "Boundaries not sorted at dim={dim}, bits={bits}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_quantize_dequantize() {
        let cb = get_codebook(128, 4);
        // Each centroid should quantize to itself
        for (i, &c) in cb.centroids.iter().enumerate() {
            assert_eq!(cb.quantize(c), i as u8);
            assert_eq!(cb.dequantize(i as u8), c);
        }
    }

    #[test]
    fn test_quantize_boundary_values() {
        let cb = get_codebook(128, 3);
        // Value at 0 should map to centroid index 3 or 4 (straddling zero)
        let idx = cb.quantize(0.0);
        assert!(idx == 3 || idx == 4, "0.0 mapped to index {idx}");
        // Large negative should map to index 0
        assert_eq!(cb.quantize(-0.5), 0);
        // Large positive should map to last index
        assert_eq!(cb.quantize(0.5), 7);
    }

    #[test]
    fn test_mse_matches_paper() {
        // Paper Theorem 1: for b=2,3,4, Dmse ≈ 0.117, 0.03, 0.009
        let cb2 = get_codebook(128, 2);
        let cb3 = get_codebook(128, 3);
        let cb4 = get_codebook(128, 4);
        let total_mse_2 = cb2.mse_per_coord * 128.0;
        let total_mse_3 = cb3.mse_per_coord * 128.0;
        let total_mse_4 = cb4.mse_per_coord * 128.0;
        assert!((total_mse_2 - 0.117).abs() < 0.005, "2-bit MSE: {total_mse_2}");
        assert!((total_mse_3 - 0.03).abs() < 0.005, "3-bit MSE: {total_mse_3}");
        assert!((total_mse_4 - 0.009).abs() < 0.001, "4-bit MSE: {total_mse_4}");
    }
}
