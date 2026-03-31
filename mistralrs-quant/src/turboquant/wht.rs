//! Fast Walsh-Hadamard Transform (WHT) for TurboQuant rotation.
//!
//! The WHT is used to randomly rotate input vectors so that each coordinate
//! follows a concentrated Beta distribution, enabling optimal scalar quantization
//! with pre-computed codebooks.
//!
//! The full rotation is D·H·D where:
//! - D is a diagonal matrix of random ±1 signs (generated deterministically from a seed)
//! - H is the Hadamard matrix (applied via the fast butterfly algorithm)
//!
//! Complexity: O(d log d) vs O(d²) for a dense random rotation matrix.
//! For d=128, this is 7×64 = 448 multiply-adds vs 16,384 — a 36× reduction.
//!
//! Reference: TurboQuant paper Section 3.1; spiritbuun/llama-cpp-turboquant-cuda.

/// Generate deterministic random ±1 sign vector from a seed.
///
/// Uses a simple xorshift64 PRNG seeded with `seed ^ dim_salt` to produce
/// `dim` sign values. The same (seed, dim) always produces the same signs.
pub fn generate_signs(seed: u64, dim: usize) -> Vec<f32> {
    let mut state = seed ^ (dim as u64).wrapping_mul(0x9E3779B97F4A7C15);
    let mut signs = Vec::with_capacity(dim);
    for _ in 0..dim {
        // xorshift64
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        signs.push(if state & 1 == 0 { 1.0f32 } else { -1.0f32 });
    }
    signs
}

/// In-place Fast Walsh-Hadamard Transform (butterfly algorithm).
///
/// Transforms `data` of length `n` (must be a power of 2) in O(n log n).
/// After the transform, each element is scaled by 1/sqrt(n).
///
/// The butterfly computes: for each pair (a, b) at distance h,
///   a' = a + b
///   b' = a - b
/// across log2(n) stages with doubling stride.
pub fn fwht_inplace(data: &mut [f32]) {
    let n = data.len();
    debug_assert!(n.is_power_of_two(), "WHT requires power-of-2 length, got {n}");

    let mut h = 1;
    while h < n {
        let mut i = 0;
        while i < n {
            for j in i..i + h {
                let a = data[j];
                let b = data[j + h];
                data[j] = a + b;
                data[j + h] = a - b;
            }
            i += h * 2;
        }
        h *= 2;
    }

    // Normalize by 1/sqrt(n)
    let scale = 1.0 / (n as f32).sqrt();
    for x in data.iter_mut() {
        *x *= scale;
    }
}

/// Apply the full TurboQuant rotation: D·H·D (sign flip, Hadamard, sign flip).
///
/// This transforms arbitrary input vectors into vectors whose coordinates
/// follow Beta(d/2, d/2), enabling optimal scalar quantization.
///
/// `signs` should be generated once via `generate_signs()` and reused.
pub fn rotate_forward(data: &mut [f32], signs: &[f32]) {
    let n = data.len();
    debug_assert_eq!(n, signs.len());

    // First sign flip: D · x
    for i in 0..n {
        data[i] *= signs[i];
    }

    // Hadamard transform: H · (D · x)
    fwht_inplace(data);

    // Second sign flip: D · H · D · x
    for i in 0..n {
        data[i] *= signs[i];
    }
}

/// Apply the inverse rotation: D·H·D (same as forward — the rotation is its own inverse).
///
/// Since D and H are both symmetric and orthogonal, and D·H·D is also orthogonal,
/// applying it twice returns to the original vector (up to floating-point precision).
/// However, note that H applied twice gives n·I, and our fwht_inplace normalizes
/// by 1/sqrt(n), so H_normalized is self-inverse: H_norm · H_norm = I.
pub fn rotate_inverse(data: &mut [f32], signs: &[f32]) {
    // D·H·D is its own inverse when H is normalized by 1/sqrt(n)
    rotate_forward(data, signs);
}

/// Quantize a single vector using TurboQuant MSE-optimal algorithm (Algorithm 1).
///
/// Steps:
/// 1. Compute and store the L2 norm
/// 2. Normalize to unit sphere
/// 3. Apply WHT rotation
/// 4. Quantize each coordinate using the codebook
///
/// Returns (indices, norm).
pub fn quantize_vector(
    data: &[f32],
    signs: &[f32],
    codebook: &super::codebook::Codebook,
) -> (Vec<u8>, f32) {
    let n = data.len();

    // Step 1: Compute L2 norm
    let norm: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm < 1e-10 {
        // Zero vector — return all-zeros
        return (vec![codebook.quantize(0.0); n], 0.0);
    }

    // Step 2: Normalize to unit sphere
    let mut rotated: Vec<f32> = data.iter().map(|x| x / norm).collect();

    // Step 3: WHT rotation
    rotate_forward(&mut rotated, signs);

    // Step 4: Scalar quantize each coordinate
    let indices: Vec<u8> = rotated.iter().map(|&x| codebook.quantize(x)).collect();

    (indices, norm)
}

/// Dequantize a vector from indices and norm.
///
/// Steps:
/// 1. Look up centroid values from codebook
/// 2. Apply inverse WHT rotation
/// 3. Scale by stored norm
pub fn dequantize_vector(
    indices: &[u8],
    norm: f32,
    signs: &[f32],
    codebook: &super::codebook::Codebook,
) -> Vec<f32> {
    // Step 1: Reconstruct from centroids
    let mut data: Vec<f32> = indices.iter().map(|&idx| codebook.dequantize(idx)).collect();

    // Step 2: Inverse rotation
    rotate_inverse(&mut data, signs);

    // Step 3: Scale by norm
    for x in data.iter_mut() {
        *x *= norm;
    }

    data
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::turboquant::codebook::get_codebook;

    #[test]
    fn test_fwht_identity() {
        // Applying FWHT twice should return the original (since normalized H is involutory)
        let original = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut data = original.clone();
        fwht_inplace(&mut data);
        fwht_inplace(&mut data);
        for (a, b) in data.iter().zip(original.iter()) {
            assert!(
                (a - b).abs() < 1e-5,
                "FWHT not involutory: {a} vs {b}"
            );
        }
    }

    #[test]
    fn test_rotation_preserves_norm() {
        let signs = generate_signs(42, 128);
        let mut data: Vec<f32> = (0..128).map(|i| (i as f32) * 0.01).collect();
        let original_norm: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();

        rotate_forward(&mut data, &signs);
        let rotated_norm: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();

        assert!(
            (original_norm - rotated_norm).abs() < 1e-4,
            "Rotation changed norm: {original_norm} -> {rotated_norm}"
        );
    }

    #[test]
    fn test_rotation_roundtrip() {
        let signs = generate_signs(42, 128);
        let original: Vec<f32> = (0..128).map(|i| (i as f32 - 64.0) * 0.01).collect();
        let mut data = original.clone();

        rotate_forward(&mut data, &signs);
        rotate_inverse(&mut data, &signs);

        for (a, b) in data.iter().zip(original.iter()) {
            assert!(
                (a - b).abs() < 1e-4,
                "Roundtrip failed: {a} vs {b}"
            );
        }
    }

    #[test]
    fn test_quantize_dequantize_quality() {
        let signs = generate_signs(42, 128);
        let codebook = get_codebook(128, 4);

        // Create a random-ish vector
        let data: Vec<f32> = (0..128)
            .map(|i| ((i * 7 + 3) % 100) as f32 / 100.0 - 0.5)
            .collect();
        let original_norm: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();

        let (indices, norm) = quantize_vector(&data, &signs, &codebook);
        let reconstructed = dequantize_vector(&indices, norm, &signs, &codebook);

        // Check norm is preserved
        assert!(
            (norm - original_norm).abs() < 1e-5,
            "Norm mismatch: {norm} vs {original_norm}"
        );

        // Check reconstruction MSE is within expected bounds
        // Paper: 4-bit MSE ≈ 0.009 for unit-norm vectors
        let mse: f32 = data
            .iter()
            .zip(reconstructed.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f32>();
        let normalized_mse = mse / (norm * norm);
        assert!(
            normalized_mse < 0.02, // Allow some slack for finite-precision effects
            "Normalized MSE too high: {normalized_mse} (expected < 0.02 for 4-bit)"
        );
    }

    #[test]
    fn test_signs_deterministic() {
        let s1 = generate_signs(42, 128);
        let s2 = generate_signs(42, 128);
        assert_eq!(s1, s2, "Same seed should produce same signs");

        let s3 = generate_signs(43, 128);
        assert_ne!(s1, s3, "Different seeds should produce different signs");
    }
}
