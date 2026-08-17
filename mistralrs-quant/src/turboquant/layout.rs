//! Block decomposition: TurboQuant for head dimensions that are not powers of
//! two.
//!
//! # The constraint, and what it actually is
//!
//! The rotation at the heart of TurboQuant is `D·H·D`, and `H` — the Hadamard
//! matrix applied by the fast butterfly — exists only for power-of-two orders.
//! That is a real mathematical constraint, and it is the *only* one. It is
//! **not** the reason the shipped code stopped at `{64, 128, 256}`: 512 is a
//! power of two and was excluded anyway, because the limit was a lookup table
//! (see [`super::generate`]) and a `static_assert(HEAD_SIZE == 128)` in the
//! CUDA kernels, not the transform.
//!
//! # The answer for non-powers-of-two
//!
//! Split the vector into consecutive power-of-two **blocks** and run the whole
//! TurboQuant pipeline — normalise, rotate, quantize — independently per
//! block, storing one norm per block:
//!
//! ```text
//!   head_dim  80 →  64 + 16
//!   head_dim  96 →  64 + 32
//!   head_dim 112 →  64 + 32 + 16
//!   head_dim 192 → 128 + 64
//! ```
//!
//! A residue smaller than [`MIN_BLOCK_DIM`] becomes one zero-padded
//! [`MIN_BLOCK_DIM`] block (`head_dim 100 → 64 + 32 + 8`, of which 4
//! coordinates are padding).
//!
//! ## Why this is the right split, and what it costs
//!
//! The alternative is to pad the whole vector to the next power of two
//! (`80 → 128`) and mask. Both are correct; block decomposition is strictly
//! cheaper:
//!
//! | head_dim | pad-to-pow2 | block decomposition |
//! |---|---|---|
//! | 80  | 128 coords + 1 norm | **80 coords + 2 norms** |
//! | 96  | 128 coords + 1 norm | **96 coords + 2 norms** |
//! | 112 | 128 coords + 1 norm | **112 coords + 3 norms** |
//! | 192 | 256 coords + 1 norm | **192 coords + 2 norms** |
//!
//! At 4 bits a coordinate is half a byte and a norm is 2 bytes, so `head_dim
//! 80` costs 40 + 4 = 44 bytes blocked against 64 + 2 = 66 bytes padded — a
//! third less. The gap widens as the head dim moves further from a power of
//! two.
//!
//! **Accuracy is not the trade.** TurboQuant's distortion for a unit-norm
//! vector is `≈ D_b` (0.0095 at 4 bits) essentially independent of dimension
//! — `total_distortion_is_dimension_free` in [`super::generate`] pins this
//! across `dim ∈ [8, 1024]`. Splitting a unit vector into blocks with norms
//! `n_j` (`Σ n_j² = 1`) gives total squared error `Σ n_j² · D_b = D_b`: the
//! *same* bound, because each block is quantized against a codebook generated
//! for its own dimension. What the split actually costs is the extra norms,
//! and the slightly weaker flattening of a 16-coordinate rotation versus a
//! 128-coordinate one — which the per-block codebook already accounts for,
//! since a wider marginal produces wider centroids.
//!
//! `roundtrip_error_is_bounded_for_every_head_dim` measures this end to end.

use super::codebook::{try_get_codebook, Codebook};
use super::generate::MIN_BLOCK_DIM;
use super::wht::{fwht_inplace, generate_signs};
use super::{pack_indices, packed_size, unpack_indices};

/// One power-of-two rotation block within a head vector.
#[derive(Debug, Clone)]
pub struct BlockSpec {
    /// Rotation order. A power of two, `>= MIN_BLOCK_DIM`.
    pub dim: usize,
    /// Offset of this block's first coordinate in the logical head vector.
    pub offset: usize,
    /// Real coordinates this block carries. Equals `dim` except for a final
    /// zero-padded block, where `len < dim`.
    pub len: usize,
    /// Offset of this block's packed bytes within the vector's packed buffer.
    pub byte_offset: usize,
    /// Packed bytes this block occupies.
    pub bytes: usize,
    /// The `±1` diagonal of `D·H·D` for this block.
    pub signs: Vec<f32>,
    /// Lloyd–Max codebook for this block's dimension and bit-width.
    pub codebook: Codebook,
}

/// The full packed layout of one head vector at one bit-width.
#[derive(Debug, Clone)]
pub struct TurboQuantLayout {
    /// Logical head dimension.
    pub dim: usize,
    /// Bits per coordinate (2, 3 or 4).
    pub bits: u32,
    pub blocks: Vec<BlockSpec>,
    /// Packed bytes per head vector, summed over blocks.
    pub packed_bytes: usize,
}

/// Decompose `head_dim` into power-of-two rotation orders.
///
/// Returns `(rotation_dim, real_coordinates)` pairs, most significant first.
/// A power-of-two `head_dim` yields exactly one block equal to itself, which
/// is what keeps 64/128/256 bit-identical to the pre-block implementation.
pub fn block_plan(head_dim: usize) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut rem = head_dim;
    while rem > 0 {
        if rem < MIN_BLOCK_DIM {
            // Tail residue: one zero-padded minimum block. Padding `rem` up to
            // 8 costs at most 7 coordinates; merging it into the previous
            // block would cost that block's size again (64 + 4 would have to
            // become a 128-order rotation).
            out.push((MIN_BLOCK_DIM, rem));
            rem = 0;
        } else {
            // Largest power of two that fits.
            let d = if rem.is_power_of_two() {
                rem
            } else {
                rem.next_power_of_two() >> 1
            };
            out.push((d, d));
            rem -= d;
        }
    }
    out
}

impl TurboQuantLayout {
    /// Build the layout for `head_dim` at `bits`, deriving each block's signs
    /// from `seed`.
    ///
    /// Block `j` uses `seed + j`, so a single-block (power-of-two) layout uses
    /// `seed` unchanged and reproduces `generate_signs(seed, head_dim)`
    /// exactly.
    pub fn new(head_dim: usize, bits: u32, seed: u64) -> Result<Self, String> {
        // A head vector narrower than one block is refused outright rather
        // than zero-padded up to `MIN_BLOCK_DIM`. Padding is correct for a
        // *residue* of a wider vector (it costs at most 7 coordinates against
        // 64+), but padding the whole vector would inflate a 1-wide column to
        // 8 coordinates and quietly turn a marker into a quantized rotation.
        // DeepSeek-V4's V half is exactly such a 1-wide marker.
        if head_dim < MIN_BLOCK_DIM {
            return Err(format!(
                "TurboQuant head_dim={head_dim} is narrower than one rotation block \
                 ({MIN_BLOCK_DIM}). There is nothing for D·H·D to flatten at this \
                 width; store it uncompressed."
            ));
        }
        let plan = block_plan(head_dim);
        let mut blocks = Vec::with_capacity(plan.len());
        let mut offset = 0usize;
        let mut byte_offset = 0usize;
        for (j, (dim, len)) in plan.into_iter().enumerate() {
            let codebook = try_get_codebook(dim, bits)?;
            let bytes = packed_size(dim, bits);
            blocks.push(BlockSpec {
                dim,
                offset,
                len,
                byte_offset,
                bytes,
                signs: generate_signs(seed.wrapping_add(j as u64), dim),
                codebook,
            });
            offset += len;
            byte_offset += bytes;
        }
        Ok(Self {
            dim: head_dim,
            bits,
            blocks,
            packed_bytes: byte_offset,
        })
    }

    /// Norms stored per head vector — one per block.
    pub fn num_norms(&self) -> usize {
        self.blocks.len()
    }

    /// Bytes per head vector including the `f16`-width norms, which is what
    /// the compression ratio is actually measured against.
    pub fn stored_bytes(&self) -> usize {
        self.packed_bytes + 2 * self.num_norms()
    }

    /// Quantize one head vector.
    ///
    /// `vector.len() == self.dim`, `packed.len() == self.packed_bytes`,
    /// `norms.len() == self.num_norms()`.
    pub fn quantize_into(&self, vector: &[f32], packed: &mut [u8], norms: &mut [f32]) {
        debug_assert_eq!(vector.len(), self.dim);
        debug_assert_eq!(packed.len(), self.packed_bytes);
        debug_assert_eq!(norms.len(), self.blocks.len());

        let mut scratch = vec![0f32; self.blocks.iter().map(|b| b.dim).max().unwrap_or(0)];
        let mut indices = vec![0u8; scratch.len()];

        for (j, b) in self.blocks.iter().enumerate() {
            let buf = &mut scratch[..b.dim];
            buf[..b.len].copy_from_slice(&vector[b.offset..b.offset + b.len]);
            // Zero the padding tail (only non-empty on a final short block).
            buf[b.len..].fill(0.0);

            let norm = buf.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm < 1e-10 {
                // A zero block: emit the index that decodes closest to zero so
                // dequantization returns zeros rather than a centroid bias.
                let zero_idx = b.codebook.quantize(0.0);
                indices[..b.dim].fill(zero_idx);
                norms[j] = 0.0;
            } else {
                let inv = 1.0 / norm;
                for x in buf.iter_mut() {
                    *x *= inv;
                }
                rotate_block(buf, &b.signs);
                for (i, &x) in buf.iter().enumerate() {
                    indices[i] = b.codebook.quantize(x);
                }
                norms[j] = norm;
            }
            pack_indices(
                &indices[..b.dim],
                self.bits,
                &mut packed[b.byte_offset..b.byte_offset + b.bytes],
            );
        }
    }

    /// Reconstruct one head vector. Inverse of [`Self::quantize_into`].
    pub fn dequantize_into(&self, packed: &[u8], norms: &[f32], out: &mut [f32]) {
        debug_assert_eq!(out.len(), self.dim);
        debug_assert_eq!(packed.len(), self.packed_bytes);
        debug_assert_eq!(norms.len(), self.blocks.len());

        let maxdim = self.blocks.iter().map(|b| b.dim).max().unwrap_or(0);
        let mut scratch = vec![0f32; maxdim];
        let mut indices = vec![0u8; maxdim];

        for (j, b) in self.blocks.iter().enumerate() {
            let norm = norms[j];
            if norm == 0.0 {
                out[b.offset..b.offset + b.len].fill(0.0);
                continue;
            }
            unpack_indices(
                &packed[b.byte_offset..b.byte_offset + b.bytes],
                self.bits,
                b.dim,
                &mut indices[..b.dim],
            );
            let buf = &mut scratch[..b.dim];
            for (i, x) in buf.iter_mut().enumerate() {
                *x = b.codebook.dequantize(indices[i]);
            }
            // `D·H·D` with a 1/sqrt(n)-normalised H is its own inverse.
            rotate_block(buf, &b.signs);
            for (i, x) in out[b.offset..b.offset + b.len].iter_mut().enumerate() {
                *x = buf[i] * norm;
            }
        }
    }
}

/// `D·H·D` on one power-of-two block.
#[inline]
fn rotate_block(data: &mut [f32], signs: &[f32]) {
    debug_assert_eq!(data.len(), signs.len());
    for (x, s) in data.iter_mut().zip(signs.iter()) {
        *x *= s;
    }
    fwht_inplace(data);
    for (x, s) in data.iter_mut().zip(signs.iter()) {
        *x *= s;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::turboquant::wht::{dequantize_vector, quantize_vector};

    /// Head dimensions that actually occur in shipped models, plus the two the
    /// tree excluded. 64: Llama-3.2-1B/3B. 80: Phi-2. 96: several 1.5B-class
    /// models. 112: MPT-style. 128: Llama-3.1, Qwen2/3, Mistral. 192: Qwen-VL
    /// variants. 256: Gemma-2/3. 512: DeepSeek-V4-Flash.
    const REAL_HEAD_DIMS: [usize; 8] = [64, 80, 96, 112, 128, 192, 256, 512];

    fn probe_vector(dim: usize, seed: u64) -> Vec<f32> {
        // Deterministic, heavy-tailed-ish, and NOT symmetric — a symmetric or
        // constant probe would hide sign-vector bugs.
        let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15) | 1;
        (0..dim)
            .map(|i| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                let u = ((s >> 40) as f32) / (1u32 << 24) as f32 - 0.5;
                // A few outlier channels, which is the situation the rotation
                // exists to fix.
                if i % 37 == 0 {
                    u * 12.0
                } else {
                    u
                }
            })
            .collect()
    }

    #[test]
    fn plan_covers_the_head_dim_with_power_of_two_blocks() {
        for dim in REAL_HEAD_DIMS {
            let plan = block_plan(dim);
            assert_eq!(
                plan.iter().map(|(_, l)| l).sum::<usize>(),
                dim,
                "plan for {dim} does not cover it: {plan:?}"
            );
            for (d, l) in &plan {
                assert!(d.is_power_of_two(), "block {d} is not a power of two");
                assert!(*d >= MIN_BLOCK_DIM, "block {d} below MIN_BLOCK_DIM");
                assert!(l <= d, "block real len {l} exceeds rotation dim {d}");
            }
        }
        assert_eq!(block_plan(80), vec![(64, 64), (16, 16)]);
        assert_eq!(block_plan(96), vec![(64, 64), (32, 32)]);
        assert_eq!(block_plan(112), vec![(64, 64), (32, 32), (16, 16)]);
        assert_eq!(block_plan(192), vec![(128, 128), (64, 64)]);
        // Tail residue smaller than MIN_BLOCK_DIM becomes one padded block.
        assert_eq!(block_plan(100), vec![(64, 64), (32, 32), (8, 4)]);
        // Powers of two stay single-block — this is what makes 64/128/256
        // bit-identical to the pre-block implementation.
        for d in [64usize, 128, 256, 512] {
            assert_eq!(block_plan(d), vec![(d, d)]);
        }
    }

    /// A power-of-two layout must agree **bit for bit** with the original
    /// `quantize_vector`/`dequantize_vector` pair, so generalising the layout
    /// cannot have perturbed the three head dims that were already validated.
    #[test]
    fn single_block_layout_is_bit_identical_to_the_original_path() {
        for dim in [64usize, 128, 256] {
            for bits in [2u32, 3, 4] {
                let layout = TurboQuantLayout::new(dim, bits, 42).unwrap();
                assert_eq!(layout.blocks.len(), 1);
                let v = probe_vector(dim, dim as u64 * 7 + bits as u64);

                let signs = generate_signs(42, dim);
                let cb = try_get_codebook(dim, bits).unwrap();
                let (idx_ref, norm_ref) = quantize_vector(&v, &signs, &cb);
                let ref_out = dequantize_vector(&idx_ref, norm_ref, &signs, &cb);

                let mut packed = vec![0u8; layout.packed_bytes];
                let mut norms = vec![0f32; layout.num_norms()];
                layout.quantize_into(&v, &mut packed, &mut norms);
                let mut out = vec![0f32; dim];
                layout.dequantize_into(&packed, &norms, &mut out);

                assert_eq!(norms[0].to_bits(), norm_ref.to_bits(), "dim={dim} b={bits}");
                for i in 0..dim {
                    assert_eq!(
                        out[i].to_bits(),
                        ref_out[i].to_bits(),
                        "dim={dim} bits={bits} coord {i}: {} vs {}",
                        out[i],
                        ref_out[i]
                    );
                }
            }
        }
    }

    /// The headline claim, measured: for **every** real head dim, the relative
    /// reconstruction error is at or below the theoretical `sqrt(D_b)`, with
    /// slack for the fact that the flattening lemma is probabilistic.
    #[test]
    fn roundtrip_error_is_bounded_for_every_head_dim() {
        // sqrt of the Lloyd-Max distortions, times a 1.5x allowance for
        // finite-dimension flattening. Anything worse than this means the
        // rotation or the codebook is wrong, not merely lossy.
        let budget = |bits: u32| -> f32 {
            match bits {
                2 => 0.1175f32.sqrt() * 1.5,
                3 => 0.03454f32.sqrt() * 1.5,
                4 => 0.009497f32.sqrt() * 1.5,
                _ => unreachable!(),
            }
        };
        for dim in REAL_HEAD_DIMS {
            for bits in [2u32, 3, 4] {
                let layout = TurboQuantLayout::new(dim, bits, 42).unwrap();
                let mut packed = vec![0u8; layout.packed_bytes];
                let mut norms = vec![0f32; layout.num_norms()];
                let mut out = vec![0f32; dim];

                let mut worst = 0f32;
                for trial in 0..24u64 {
                    let v = probe_vector(dim, trial * 1013 + dim as u64);
                    layout.quantize_into(&v, &mut packed, &mut norms);
                    layout.dequantize_into(&packed, &norms, &mut out);

                    let num: f32 = v.iter().zip(&out).map(|(a, b)| (a - b) * (a - b)).sum();
                    let den: f32 = v.iter().map(|x| x * x).sum();
                    worst = worst.max((num / den).sqrt());
                }
                assert!(
                    worst <= budget(bits),
                    "dim={dim} bits={bits}: relative error {worst} exceeds budget {}",
                    budget(bits)
                );
                // And it must not be suspiciously perfect — a round trip that
                // returns its input exactly means nothing was quantized.
                assert!(
                    worst > 1e-4,
                    "dim={dim} bits={bits}: relative error {worst} is too good to \
                     be a {bits}-bit quantizer — the codec is a no-op"
                );
            }
        }
    }

    /// Storage must actually shrink, at every head dim, including the
    /// non-powers-of-two where extra norms are the price of the split.
    #[test]
    fn every_head_dim_compresses() {
        for dim in REAL_HEAD_DIMS {
            let k = TurboQuantLayout::new(dim, 4, 42).unwrap();
            let v = TurboQuantLayout::new(dim, 3, 42).unwrap();
            let fp16 = dim * 2 * 2; // K + V at 2 bytes
            let tq = k.stored_bytes() + v.stored_bytes();
            let ratio = fp16 as f32 / tq as f32;
            assert!(
                ratio > 1.9,
                "head_dim {dim}: K4/V3 compression only {ratio:.2}x ({tq} vs {fp16} bytes)"
            );
        }
    }

    /// MUTATION GUARD — break the butterfly.
    ///
    /// If `fwht_inplace` stopped being involutory (e.g. a stage dropped, or the
    /// 1/sqrt(n) normalisation applied once instead of twice), the round trip
    /// would still *run* and still return plausible numbers. This asserts the
    /// property the whole codec rests on, at every block order the planner can
    /// emit.
    #[test]
    fn wht_is_involutory_at_every_block_order() {
        for order in [8usize, 16, 32, 64, 128, 256, 512] {
            let orig = probe_vector(order, order as u64);
            let mut d = orig.clone();
            fwht_inplace(&mut d);
            // A transform that did nothing would pass the round trip below.
            let moved = orig
                .iter()
                .zip(&d)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(moved > 1e-3, "order {order}: FWHT changed nothing");
            fwht_inplace(&mut d);
            for i in 0..order {
                assert!(
                    (d[i] - orig[i]).abs() < 1e-3 * (1.0 + orig[i].abs()),
                    "order {order} coord {i}: {} vs {}",
                    d[i],
                    orig[i]
                );
            }
        }
    }

    /// MUTATION GUARD — break the codebook lookup.
    ///
    /// Quantizing a centroid must return that centroid's own index, and
    /// dequantizing must return the centroid. An off-by-one in the boundary
    /// binary search (the `Ok(i) => i + 1` class of bug) shifts every index by
    /// one and this catches it at every generated dimension, not just the three
    /// that had hand-checked tables.
    #[test]
    fn codebook_lookup_is_an_identity_on_centroids() {
        for dim in REAL_HEAD_DIMS {
            for bits in [2u32, 3, 4] {
                let cb = try_get_codebook(dim, bits).unwrap();
                for (i, &c) in cb.centroids.iter().enumerate() {
                    assert_eq!(
                        cb.quantize(c) as usize,
                        i,
                        "dim={dim} bits={bits}: centroid {i} ({c}) quantized to {}",
                        cb.quantize(c)
                    );
                    assert_eq!(cb.dequantize(i as u8), c);
                }
            }
        }
    }

    /// MUTATION GUARD — an unsupported dimension must be refused, loudly and
    /// specifically, not silently degraded.
    #[test]
    fn sub_minimum_dims_are_refused_with_a_specific_message() {
        for dim in [0usize, 1, 4, 7] {
            let err = try_get_codebook(dim, 4).unwrap_err();
            assert!(
                err.contains("outside") && err.contains(&MIN_BLOCK_DIM.to_string()),
                "dim={dim} produced an unhelpful error: {err}"
            );
        }
        assert!(try_get_codebook(64, 5).is_err());
        assert!(try_get_codebook(64, 1).is_err());
        // A 1-wide vector must not silently become an 8-wide one either.
        // `TurboQuantLayout::new` is where a model config lands, so it must
        // reject rather than pad a marker column into a rotation — V4's V half
        // is exactly such a marker.
        for d in [0usize, 1, 4, 7] {
            let err = TurboQuantLayout::new(d, 4, 42).unwrap_err();
            assert!(
                err.contains("narrower than one rotation block"),
                "head_dim {d}: unhelpful error {err}"
            );
        }
        assert!(TurboQuantLayout::new(MIN_BLOCK_DIM, 4, 42).is_ok());
    }

    /// The layout is the storage contract; if `packed_bytes` disagreed with
    /// what `quantize_into` writes, the cache would interleave vectors.
    #[test]
    fn packed_bytes_matches_what_blocks_consume() {
        for dim in REAL_HEAD_DIMS.iter().copied().chain([100, 8, 33]) {
            for bits in [2u32, 3, 4] {
                let l = TurboQuantLayout::new(dim, bits, 7).unwrap();
                let summed: usize = l.blocks.iter().map(|b| b.bytes).sum();
                assert_eq!(summed, l.packed_bytes, "dim={dim} bits={bits}");
                let last = l.blocks.last().unwrap();
                assert_eq!(last.byte_offset + last.bytes, l.packed_bytes);
                assert_eq!(
                    l.blocks.iter().map(|b| b.len).sum::<usize>(),
                    dim,
                    "dim={dim} bits={bits}: blocks do not cover the head vector"
                );
            }
        }
    }
}
