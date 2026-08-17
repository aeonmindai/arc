//! Host-side companions of the trellis grouped GEMM
//! (`kernels/qtip/qtip_grouped_gemm.cu`) — Arc Stage 4, batched 2-bit MoE
//! serving.
//!
//! Three things live here:
//!
//! 1. **The per-expert bit-width descriptor table** ([`ExpertBpwTable`]).
//!    The grouped kernel dispatches one template instantiation per
//!    bit-width class; this PR ships the 2-bit class end-to-end and the
//!    descriptor plumbing so 4-bit experts slot in next (mixed-bpw stacks
//!    partition their expert groups by class and launch per class).
//!
//! 2. **CPU mirrors of the on-device routing math** so the scheduler logic
//!    is unit-testable on macOS: [`build_group_tile_map`] mirrors the
//!    `qtip2b_moe_build` kernel's scans + ragged tile flattening, and
//!    [`window_state_2b`] mirrors the kernel's random-access state
//!    reconstruction (`state(t) = pair_reverse_16(stream_window_16(t))`),
//!    the identity that lets the GEMM decode any (row, k) weight in ~4 ALU
//!    ops with no sequential warm-up replay.
//!
//! 3. **Tile-geometry constants** shared with the `.cu` file. Keep in sync
//!    with `QG_TILE_M` / `QG_TILE_N` / `QG_TILE_K` there.

/// Pairs per m-tile (one `mma.m16n8k16` M dimension). Mirrors `QG_TILE_M`.
///
/// Public because the amortization a benchmark can legitimately claim is
/// bounded by it: each expert's weights are staged once per m-tile, so the
/// asymptotic weight-read saving over a per-pair GEMV is at most `TILE_M`.
pub const GROUPED_TILE_M: usize = 16;
/// Weight rows per n-tile (4 warps x 16). Mirrors `QG_TILE_N`.
pub const GROUPED_TILE_N: usize = 64;
/// Symbols per k-chunk (16 packed bytes per row). Mirrors `QG_TILE_K`.
/// The grouped path requires `in_features % GROUPED_TILE_K == 0`.
///
/// Public so a harness can assert its fixture actually reaches the grouped
/// kernel: when this divisibility fails, `gather_forward` silently takes the
/// CPU reference instead and a "grouped" timing measures something else
/// entirely (DOCTRINE D12).
pub const GROUPED_TILE_K: usize = 64;

/// Host-side upper bound on the grouped GEMM's m-tile count: every expert
/// group contributes at most one partial tile beyond its full tiles, so
/// `ceil(P / TILE_M) + E` bounds the ragged total. The kernel grid is sized
/// from this bound; the exact count stays on the device (zero host syncs).
#[cfg_attr(not(feature = "cuda"), allow(dead_code))] // CUDA host wrapper + tests
pub(crate) fn grouped_max_m_tiles(n_pairs: usize, num_experts: usize) -> usize {
    n_pairs.div_ceil(GROUPED_TILE_M) + num_experts
}

// ---------------------------------------------------------------------------
// Per-expert bit-width descriptors
// ---------------------------------------------------------------------------

/// Bits-per-weight class of one expert's trellis tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrellisBpw {
    /// K=2 bitshift trellis (the `qtip2b` rung) — 2 bits per weight.
    B2,
    // B4 (K=4 bitshift trellis) lands with the 4-bit expert rung; the
    // grouped dispatch partitions expert groups by class and launches the
    // matching kernel template per class.
}

/// Descriptor table: one [`TrellisBpw`] per expert tensor in a stacked MoE
/// layer. Today every constructor produces a uniform 2-bit table; the
/// grouped dispatch consults it so a future mixed-precision stack (hot
/// experts at 4 bits, cold at 2) only has to extend the enum and the
/// per-class launch loop.
#[derive(Debug, Clone)]
pub struct ExpertBpwTable {
    bpw: Vec<TrellisBpw>,
}

impl ExpertBpwTable {
    /// A uniform 2-bit table for `num_experts` experts.
    pub fn uniform_2bit(num_experts: usize) -> Self {
        Self {
            bpw: vec![TrellisBpw::B2; num_experts],
        }
    }

    /// Bit-width class of expert `e`.
    pub fn bpw(&self, e: usize) -> TrellisBpw {
        self.bpw[e]
    }

    /// Number of experts described.
    pub fn len(&self) -> usize {
        self.bpw.len()
    }

    /// True when the table is empty.
    pub fn is_empty(&self) -> bool {
        self.bpw.is_empty()
    }

    /// True when every expert is 2-bit — the single-launch fast case (and
    /// the only class implemented end-to-end today).
    pub fn is_uniform_2bit(&self) -> bool {
        self.bpw.iter().all(|b| matches!(b, TrellisBpw::B2))
    }

    /// The expert-parallel slice: the bit-width classes of `ids`, in order.
    ///
    /// This has to travel with the weight slice rather than be rebuilt as a
    /// uniform table — a mixed-precision stack (4-bit hot experts) would
    /// otherwise be silently re-declared as uniform 2-bit and mis-dispatched.
    pub fn select(&self, ids: &[usize]) -> candle_core::Result<Self> {
        let mut bpw = Vec::with_capacity(ids.len());
        for &e in ids {
            let Some(&b) = self.bpw.get(e) else {
                candle_core::bail!(
                    "ExpertBpwTable::select: expert {e} is out of range for {} experts",
                    self.bpw.len()
                );
            };
            bpw.push(b);
        }
        Ok(Self { bpw })
    }
}

// ---------------------------------------------------------------------------
// CPU mirrors of the on-device routing / decode math
// ---------------------------------------------------------------------------

/// One m-tile of the grouped schedule: `(expert, row_start)` into the
/// expert-sorted pair array. `rows = min(TILE_M, offsets[e+1] - row_start)`.
#[allow(dead_code)] // kernel mirror; exercised by the colocated tests
pub(crate) type GroupTile = (u32, u32);

/// Host mirror of the `qtip2b_moe_build` kernel: exclusive pair-count scan
/// plus the flattened ragged tile list, with per-expert TILE_M rounding so
/// a tile is always a full scheduling unit (ragged groups cannot strand a
/// CTA mid-group). Returns `(offsets[E+1], tiles)`.
#[allow(dead_code)] // kernel mirror; exercised by the colocated tests
pub(crate) fn build_group_tile_map(counts: &[u32], tile_m: usize) -> (Vec<u32>, Vec<GroupTile>) {
    let mut offsets = Vec::with_capacity(counts.len() + 1);
    let mut tiles = Vec::new();
    let mut acc = 0u32;
    for (e, &c) in counts.iter().enumerate() {
        offsets.push(acc);
        let nt = (c as usize).div_ceil(tile_m);
        for i in 0..nt {
            tiles.push((e as u32, acc + (i * tile_m) as u32));
        }
        acc += c;
    }
    offsets.push(acc);
    (offsets, tiles)
}

/// Host mirror of the kernel's random-access state reconstruction.
///
/// The bitshift trellis state is a sliding 16-bit window over the symbol
/// stream: after symbol `t`, state pair `j` (bits `2j+1..2j`) is
/// `sym[t-j]`, while the packed stream stores `sym[t-7+j]` at window pair
/// `j`. Hence
///
/// ```text
/// state(t) = pair_reverse_16(stream_bits[2t-14 .. 2t+1])
/// ```
///
/// with bits below stream position 0 reading as zero (the all-zero initial
/// state). The kernel computes this with an aligned u32 window load,
/// `__brev`, and one adjacent-bit swap; this mirror follows the same
/// bit-reverse + swap route and is property-tested against the sequential
/// recurrence below.
#[allow(dead_code)] // kernel mirror; exercised by the colocated tests
pub(crate) fn window_state_2b(packed: &[u8], t: usize) -> u32 {
    let bit_start = 2 * t as i64 - 14;
    let mut win: u32 = 0;
    for j in 0..16i64 {
        let b = bit_start + j;
        if b >= 0 {
            let byte = packed[(b / 8) as usize];
            win |= (((byte >> (b % 8)) & 1) as u32) << j;
        }
    }
    // Full 16-bit bit-reversal, then swap adjacent bits => pair reversal.
    let r = (win as u16).reverse_bits() as u32;
    ((r & 0x5555) << 1) | ((r >> 1) & 0x5555)
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- descriptor table ---------------------------------------------------

    #[test]
    fn descriptor_table_uniform_2bit() {
        let t = ExpertBpwTable::uniform_2bit(8);
        assert_eq!(t.len(), 8);
        assert!(!t.is_empty());
        assert!(t.is_uniform_2bit());
        for e in 0..8 {
            assert_eq!(t.bpw(e), TrellisBpw::B2);
        }
    }

    // -- tile map -----------------------------------------------------------

    /// The flattened tile list must exactly partition every expert group
    /// into TILE_M-rounded units and respect the ceil(P/TILE_M)+E bound.
    #[test]
    fn tile_map_partitions_ragged_groups() {
        let tile_m = GROUPED_TILE_M;
        let cases: Vec<Vec<u32>> = vec![
            // 1 giant expert + many tiny.
            vec![1000, 1, 2, 3, 0, 1, 5, 7],
            // All tokens on one expert (others empty).
            vec![0, 0, 640, 0],
            // Empty experts interleaved, incl. trailing.
            vec![0, 16, 0, 17, 0, 15, 0],
            // Single expert, single pair.
            vec![1],
            // Nothing routed at all.
            vec![0, 0, 0],
            // Exact-multiple group sizes (no partial tiles).
            vec![32, 16, 48],
        ];
        for counts in cases {
            let n_pairs: u32 = counts.iter().sum();
            let (offsets, tiles) = build_group_tile_map(&counts, tile_m);

            // Offsets are an exclusive scan.
            assert_eq!(offsets.len(), counts.len() + 1);
            assert_eq!(offsets[0], 0);
            assert_eq!(*offsets.last().unwrap(), n_pairs);
            for (e, &c) in counts.iter().enumerate() {
                assert_eq!(offsets[e + 1] - offsets[e], c);
            }

            // The kernel-side upper bound holds.
            assert!(
                tiles.len() <= grouped_max_m_tiles(n_pairs as usize, counts.len()),
                "tile count {} exceeds bound for counts {counts:?}",
                tiles.len()
            );

            // Every pair slot is covered by exactly one tile, tiles never
            // cross group boundaries, and each tile has 1..=TILE_M rows.
            let mut covered = vec![0u32; n_pairs as usize];
            for &(e, row_start) in &tiles {
                let e = e as usize;
                assert!(row_start >= offsets[e] && row_start < offsets[e + 1]);
                assert_eq!(
                    (row_start - offsets[e]) as usize % tile_m,
                    0,
                    "tile start not TILE_M-aligned within its group"
                );
                let rows = (offsets[e + 1] - row_start).min(tile_m as u32);
                assert!(rows >= 1);
                for r in row_start..row_start + rows {
                    covered[r as usize] += 1;
                }
            }
            assert!(
                covered.iter().all(|&c| c == 1),
                "tile coverage not a partition for counts {counts:?}"
            );
        }
    }

    // -- window/state identity ---------------------------------------------

    /// THE load-bearing identity of the grouped kernel: for every position
    /// `t`, the pair-reversed 16-bit stream window equals the state the
    /// sequential recurrence `state = ((state << 2) | sym) & 0xFFFF`
    /// reaches — including the zero-filled warm-up region t < 7.
    #[test]
    fn window_state_matches_sequential_recurrence() {
        // Deterministic pseudo-random symbol streams (splitmix-ish).
        for (len, seed) in [(8usize, 1u64), (64, 2), (256, 3), (1024, 4)] {
            let mut syms = vec![0u8; len];
            let mut z = seed;
            for s in syms.iter_mut() {
                z = z
                    .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                    .wrapping_add(0xBF58_476D_1CE4_E5B9);
                *s = ((z >> 33) & 0x3) as u8;
            }
            // Pack LSB-first, 4 symbols per byte (the production layout).
            let mut packed = vec![0u8; len / 4];
            for (t, &s) in syms.iter().enumerate() {
                packed[t / 4] |= (s & 0x3) << (2 * (t % 4));
            }

            let mut state: u32 = 0;
            for (t, &s) in syms.iter().enumerate() {
                state = ((state << 2) | s as u32) & 0xFFFF;
                let via_window = window_state_2b(&packed, t);
                assert_eq!(
                    state, via_window,
                    "state mismatch at t={t} (len={len}, seed={seed})"
                );
            }
        }
    }
}
