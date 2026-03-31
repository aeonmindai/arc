//! TurboQuant KV cache compression.
//!
//! Implements the TurboQuant algorithm (ICLR 2026, arXiv:2504.19874) for
//! compressing LLM key-value caches to 2-4 bits per coordinate with near-zero
//! quality degradation.
//!
//! The algorithm:
//! 1. Normalize the input vector and store its L2 norm as fp16.
//! 2. Apply a randomized Walsh-Hadamard rotation (D·H·D where D is a diagonal
//!    of random ±1 signs and H is the Hadamard matrix). This makes every
//!    coordinate follow a Beta(d/2, d/2) distribution regardless of input.
//! 3. Quantize each coordinate independently using pre-computed Lloyd-Max
//!    optimal codebooks for the Beta distribution.
//! 4. Pack the quantized indices into sub-byte storage (4-bit nibble or
//!    3-bit 10-in-32 packing).
//!
//! During attention, the query is rotated once and the dot product is computed directly
//! against codebook centroids looked up from the packed indices — no full
//! dequantization needed.

#[allow(clippy::excessive_precision)]
pub mod codebook;
pub mod wht;

use codebook::Codebook;
use serde::{Deserialize, Serialize};

/// TurboQuant compression preset.
///
/// Each preset specifies the bit-width for keys and values independently.
/// Keys are more sensitive (they determine attention focus via softmax)
/// and benefit from higher precision. Values are averaged by attention
/// weights so errors cancel out.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum TurboQuantPreset {
    /// Default: 4-bit keys, 3-bit values (3.5 bits avg). Lossless quality.
    /// LongBench 50.06 = identical to FP16 50.06 on Llama-3.1-8B.
    #[default]
    Default,
    /// Balanced: 3-bit keys, 3-bit values (3.0 bits avg). ~0.1% quality loss.
    Balanced,
    /// Aggressive: 3-bit keys, 2-bit values (2.5 bits avg). ~1.2% quality loss.
    /// Opt-in only for cost-sensitive deployments.
    Aggressive,
}

impl TurboQuantPreset {
    /// Bit-width for key vectors.
    pub fn key_bits(&self) -> u32 {
        match self {
            Self::Default => 4,
            Self::Balanced => 3,
            Self::Aggressive => 3,
        }
    }

    /// Bit-width for value vectors.
    pub fn value_bits(&self) -> u32 {
        match self {
            Self::Default => 3,
            Self::Balanced => 3,
            Self::Aggressive => 2,
        }
    }

    /// Average bits per coordinate across keys and values.
    pub fn avg_bits(&self) -> f32 {
        (self.key_bits() + self.value_bits()) as f32 / 2.0
    }

    /// Approximate memory compression ratio vs FP16.
    pub fn compression_ratio(&self, head_dim: usize) -> f32 {
        let fp16_bytes = (head_dim * 2 * 2) as f32; // K + V, 2 bytes each
        let k_bytes = packed_size(head_dim, self.key_bits());
        let v_bytes = packed_size(head_dim, self.value_bits());
        let tq_bytes = (k_bytes + v_bytes + 4) as f32; // +4 for two fp16 norms
        fp16_bytes / tq_bytes
    }
}

impl std::fmt::Display for TurboQuantPreset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Default => write!(f, "turboquant (K4/V3, 3.5-bit, lossless)"),
            Self::Balanced => write!(f, "turboquant-3 (K3/V3, 3.0-bit)"),
            Self::Aggressive => write!(f, "turboquant-aggressive (K3/V2, 2.5-bit)"),
        }
    }
}

/// Full configuration for TurboQuant KV cache compression.
#[derive(Debug, Clone)]
pub struct TurboQuantConfig {
    /// Compression preset (determines key/value bit-widths).
    pub preset: TurboQuantPreset,
    /// Head dimension. Must be 64, 128, or 256.
    pub head_dim: usize,
    /// Number of recent tokens to keep in full FP16 precision.
    /// These tokens are not compressed. Default: 128.
    pub fp16_window: usize,
    /// Random seed for the Walsh-Hadamard sign vectors.
    /// Deterministic: same seed always produces the same rotation.
    pub seed: u64,
}

impl TurboQuantConfig {
    pub fn new(head_dim: usize) -> Self {
        Self {
            preset: TurboQuantPreset::Default,
            head_dim,
            fp16_window: 128,
            seed: 42,
        }
    }

    pub fn with_preset(mut self, preset: TurboQuantPreset) -> Self {
        self.preset = preset;
        self
    }

    pub fn with_fp16_window(mut self, window: usize) -> Self {
        self.fp16_window = window;
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Get the codebook for key vectors.
    pub fn key_codebook(&self) -> Codebook {
        codebook::get_codebook(self.head_dim, self.preset.key_bits())
    }

    /// Get the codebook for value vectors.
    pub fn value_codebook(&self) -> Codebook {
        codebook::get_codebook(self.head_dim, self.preset.value_bits())
    }
}

/// Compute the packed byte size for `dim` elements at the given bit-width.
///
/// - 4-bit: nibble packing, 2 values per byte → dim/2 bytes.
/// - 3-bit: 10-in-32 packing, 10 values per 4 bytes → ceil(dim/10)*4 bytes.
/// - 2-bit: 4 values per byte → dim/4 bytes.
pub fn packed_size(dim: usize, bits: u32) -> usize {
    match bits {
        2 => dim / 4,
        3 => (dim.div_ceil(10)) * 4,
        4 => dim / 2,
        _ => panic!("Unsupported bit-width: {bits}"),
    }
}

/// Pack quantized indices into a byte buffer.
///
/// - 4-bit: two nibbles per byte (low nibble = even index, high nibble = odd).
/// - 3-bit: 10 values per u32, each 3 bits, packed LSB-first.
/// - 2-bit: four values per byte, packed from MSB.
pub fn pack_indices(indices: &[u8], bits: u32, output: &mut [u8]) {
    match bits {
        4 => {
            assert!(indices.len().is_multiple_of(2));
            for i in 0..indices.len() / 2 {
                output[i] = (indices[2 * i] & 0xF) | ((indices[2 * i + 1] & 0xF) << 4);
            }
        }
        3 => {
            // 10 values per u32 (30 bits used, 2 bits wasted)
            let chunks = indices.chunks(10);
            let mut out_idx = 0;
            for chunk in chunks {
                let mut packed: u32 = 0;
                for (j, &val) in chunk.iter().enumerate() {
                    packed |= (val as u32 & 0x7) << (j * 3);
                }
                let bytes = packed.to_le_bytes();
                output[out_idx..out_idx + 4].copy_from_slice(&bytes);
                out_idx += 4;
            }
        }
        2 => {
            assert!(indices.len().is_multiple_of(4));
            for i in 0..indices.len() / 4 {
                output[i] = ((indices[4 * i] & 0x3) << 6)
                    | ((indices[4 * i + 1] & 0x3) << 4)
                    | ((indices[4 * i + 2] & 0x3) << 2)
                    | (indices[4 * i + 3] & 0x3);
            }
        }
        _ => panic!("Unsupported bit-width: {bits}"),
    }
}

/// Unpack quantized indices from a byte buffer.
///
/// Inverse of `pack_indices`.
pub fn unpack_indices(packed: &[u8], bits: u32, count: usize, output: &mut [u8]) {
    match bits {
        4 => {
            for i in 0..count / 2 {
                output[2 * i] = packed[i] & 0xF;
                output[2 * i + 1] = (packed[i] >> 4) & 0xF;
            }
        }
        3 => {
            let mut out_idx = 0;
            let mut pack_idx = 0;
            while out_idx < count {
                let word = u32::from_le_bytes([
                    packed[pack_idx],
                    packed[pack_idx + 1],
                    packed[pack_idx + 2],
                    packed[pack_idx + 3],
                ]);
                let n = std::cmp::min(10, count - out_idx);
                for j in 0..n {
                    output[out_idx + j] = ((word >> (j * 3)) & 0x7) as u8;
                }
                out_idx += 10;
                pack_idx += 4;
            }
        }
        2 => {
            for i in 0..count / 4 {
                output[4 * i] = (packed[i] >> 6) & 0x3;
                output[4 * i + 1] = (packed[i] >> 4) & 0x3;
                output[4 * i + 2] = (packed[i] >> 2) & 0x3;
                output[4 * i + 3] = packed[i] & 0x3;
            }
        }
        _ => panic!("Unsupported bit-width: {bits}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_packed_size() {
        assert_eq!(packed_size(128, 4), 64);
        assert_eq!(packed_size(128, 3), 52); // ceil(128/10)*4 = 13*4
        assert_eq!(packed_size(128, 2), 32);
        assert_eq!(packed_size(256, 4), 128);
        assert_eq!(packed_size(256, 3), 104); // ceil(256/10)*4 = 26*4
        assert_eq!(packed_size(256, 2), 64);
    }

    #[test]
    fn test_pack_unpack_4bit() {
        let indices: Vec<u8> = (0..16).collect();
        let mut packed = vec![0u8; packed_size(16, 4)];
        pack_indices(&indices, 4, &mut packed);
        let mut unpacked = vec![0u8; 16];
        unpack_indices(&packed, 4, 16, &mut unpacked);
        assert_eq!(indices, unpacked);
    }

    #[test]
    fn test_pack_unpack_3bit() {
        let indices: Vec<u8> = (0..20).map(|i| i % 8).collect();
        let mut packed = vec![0u8; packed_size(20, 3)];
        pack_indices(&indices, 3, &mut packed);
        let mut unpacked = vec![0u8; 20];
        unpack_indices(&packed, 3, 20, &mut unpacked);
        assert_eq!(indices, unpacked);
    }

    #[test]
    fn test_pack_unpack_2bit() {
        let indices: Vec<u8> = (0..16).map(|i| i % 4).collect();
        let mut packed = vec![0u8; packed_size(16, 2)];
        pack_indices(&indices, 2, &mut packed);
        let mut unpacked = vec![0u8; 16];
        unpack_indices(&packed, 2, 16, &mut unpacked);
        assert_eq!(indices, unpacked);
    }

    #[test]
    fn test_compression_ratio() {
        let default = TurboQuantPreset::Default;
        let ratio = default.compression_ratio(128);
        // K4/V3: 64 + 52 + 4 = 120 bytes vs 512 FP16 bytes (128*2*2)
        let k_bytes = packed_size(128, 4); // 64
        let v_bytes = packed_size(128, 3); // 52
        let expected = 512.0 / (k_bytes + v_bytes + 4) as f32;
        assert!(
            (ratio - expected).abs() < 0.01,
            "ratio: {ratio}, expected: {expected}"
        );

        let aggressive = TurboQuantPreset::Aggressive;
        let ratio = aggressive.compression_ratio(128);
        let k_bytes = packed_size(128, 3); // 52
        let v_bytes = packed_size(128, 2); // 32
        let expected = 512.0 / (k_bytes + v_bytes + 4) as f32;
        assert!(
            (ratio - expected).abs() < 0.01,
            "ratio: {ratio}, expected: {expected}"
        );
    }
}
