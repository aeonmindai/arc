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
// Runtime kernel-variant selection (ONE BINARY, ONE VARIABLE)
// ---------------------------------------------------------------------------

/// The grouped GEMM as first shipped and measured (FACTS wave38: 8.2 us per
/// m-tile unit against the fused GEMV's 4.4). Mirrors
/// `QTIP_GROUPED_VARIANT_BASELINE` in `qtip_grouped_gemm.cu`.
pub const QTIP_GROUPED_VARIANT_BASELINE: i32 = 0;
/// Bit-identical output, cheaper route to the same weights: conflict-free
/// staged strides plus a once-per-row pair-reversal that hoists `__brev` and
/// the pair-swap out of the per-weight decode. Mirrors
/// `QTIP_GROUPED_VARIANT_TUNED`.
pub const QTIP_GROUPED_VARIANT_TUNED: i32 = 1;
/// Variant 1 plus two instruction-level collapses, both bit-exact: one packed
/// f32->16-bit convert in place of `q2b_pack2`'s two scalar converts + shift +
/// or, and one `ldmatrix.x4` in place of the four `LDS.32` A-fragment loads.
/// `ldmatrix` is only conflict-free here BECAUSE variant 1 already padded the
/// staged activation stride to 144 B. Mirrors `QTIP_GROUPED_VARIANT_LDST`.
pub const QTIP_GROUPED_VARIANT_LDST: i32 = 2;
/// Number of selectable variants; also the launch-counter array length.
/// DIAGNOSTIC ONLY — produces wrong weights. Bounds the trellis decode's share
/// of grouped-GEMM runtime. Selectable only by `ARC_QTIP_GROUPED_VARIANT=stub`.
pub const QTIP_GROUPED_VARIANT_DECODESTUB: i32 = 3;

pub const QTIP_GROUPED_VARIANT_COUNT: usize = 4;

/// Env override for [`grouped_variant`]. Accepts `baseline`/`0`, `tuned`/`1`
/// or `ldst`/`2`. Latched on first read, then overridable in-process by
/// [`set_grouped_variant`].
pub const QTIP_GROUPED_VARIANT_ENV: &str = "ARC_QTIP_GROUPED_VARIANT";

const GROUPED_VARIANT_UNINIT: i32 = -1;
static GROUPED_VARIANT: std::sync::atomic::AtomicI32 =
    std::sync::atomic::AtomicI32::new(GROUPED_VARIANT_UNINIT);

/// Per-variant launch counters, indexed by variant id.
///
/// These exist because "assert the arms differ" must be answered by the
/// RUNTIME, not by the binary: both variants compile in, so a mode switch that
/// silently fails to take would otherwise produce a table of one kernel
/// measured twice — the exact defect that voided the previous grouped-GEMM
/// microbench (FACTS, wave35-BM).
static GROUPED_LAUNCHES: [std::sync::atomic::AtomicU64; QTIP_GROUPED_VARIANT_COUNT] = [
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
];

/// Select the grouped-GEMM kernel variant for every subsequent launch.
///
/// Deliberately an atomic and NOT a `LazyLock`/`OnceLock` env read: a memoized
/// switch is what turned the last grouped-vs-GEMV comparison into the same
/// kernel measured twice.
pub fn set_grouped_variant(v: i32) {
    GROUPED_VARIANT.store(v, std::sync::atomic::Ordering::SeqCst);
}

/// The variant the next launch will use (env-latched on first call).
pub fn grouped_variant() -> i32 {
    use std::sync::atomic::Ordering;
    let cur = GROUPED_VARIANT.load(Ordering::SeqCst);
    if cur != GROUPED_VARIANT_UNINIT {
        return cur;
    }
    let init = match std::env::var(QTIP_GROUPED_VARIANT_ENV) {
        Ok(s) => match s.trim().to_ascii_lowercase().as_str() {
            "tuned" | "1" => QTIP_GROUPED_VARIANT_TUNED,
            "ldst" | "2" => QTIP_GROUPED_VARIANT_LDST,
            // Not reachable by "3": spelling it out keeps a stray numeric env
            // value from silently selecting a wrong-numerics kernel.
            "stub" => {
                tracing::error!(
                    "ARC_QTIP_GROUPED_VARIANT=stub: DIAGNOSTIC decode-stub kernel selected. \
                     Weights are WRONG by construction; this measures decode share only."
                );
                QTIP_GROUPED_VARIANT_DECODESTUB
            }
            _ => QTIP_GROUPED_VARIANT_BASELINE,
        },
        Err(_) => QTIP_GROUPED_VARIANT_BASELINE,
    };
    let _ = GROUPED_VARIANT.compare_exchange(
        GROUPED_VARIANT_UNINIT,
        init,
        Ordering::SeqCst,
        Ordering::SeqCst,
    );
    GROUPED_VARIANT.load(Ordering::SeqCst)
}

/// Record one grouped-GEMM launch of `variant`. Called from the CUDA host
/// wrapper immediately before the launch.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) fn note_grouped_launch(variant: i32) {
    if let Some(c) = GROUPED_LAUNCHES.get(variant.max(0) as usize) {
        c.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
}

/// Launches so far, indexed by variant id. A harness that reports a
/// per-variant timing MUST show this counter advancing on the variant it
/// claims to have measured, and NOT advancing on the others, or the number
/// describes some other kernel.
pub fn grouped_launch_counts() -> [u64; QTIP_GROUPED_VARIANT_COUNT] {
    let mut out = [0u64; QTIP_GROUPED_VARIANT_COUNT];
    for (i, o) in out.iter_mut().enumerate() {
        *o = GROUPED_LAUNCHES[i].load(std::sync::atomic::Ordering::Relaxed);
    }
    out
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

/// CPU mirror of the tuned kernel's staged-row transform
/// (`q2b_reverse_row` + `q2b_states_rev` in `qtip_grouped_gemm.cu`).
///
/// Takes the 32-byte staged buffer for one k-chunk of one weight row — bytes
/// `[12,16)` = the 4-byte history prefix, `[16,32)` = the chunk — and returns
/// the five reversed words the kernel decodes from. Reversing the sixteen
/// 2-bit groups of a word turns the sliding state window into a plain shift,
/// so `state(ts) = (R >> (126 - 2*ts)) & 0xFFFF` with NO per-weight `__brev`.
///
/// `pair_rev` is injected so the colocated test can prove the property test
/// rejects a wrong permutation (D12: a check never seen red is decoration).
#[allow(dead_code)] // kernel mirror; exercised by the colocated tests
pub(crate) fn reverse_staged_row_2b_with(
    staged: &[u8; 32],
    pair_rev: fn(u32) -> u32,
) -> [u32; GROUPED_REV_WORDS] {
    let mut w = [0u32; 8];
    for (i, word) in w.iter_mut().enumerate() {
        *word = u32::from_le_bytes(staged[i * 4..i * 4 + 4].try_into().expect("4 bytes"));
    }
    let mut r = [0u32; GROUPED_REV_WORDS];
    for (j, out) in r.iter_mut().enumerate() {
        *out = pair_rev(w[7 - j]);
    }
    r
}

/// Reversed words the tuned grouped kernel decodes from (bytes 12..32 of the
/// staged row, one word each). Mirrors `QG_REV_WORDS`.
pub(crate) const GROUPED_REV_WORDS: usize = 5;
/// Reversed-bit origin of chunk-local symbol 0. Mirrors `QG_REV_BIT_BASE`.
pub(crate) const GROUPED_REV_BIT_BASE: i32 = 126;

/// Reverse the sixteen 2-bit groups of a word, bits inside a group in order.
/// Mirrors `q2b_pair_reverse_32`.
#[allow(dead_code)]
pub(crate) fn pair_reverse_32(x: u32) -> u32 {
    let r = x.reverse_bits();
    ((r & 0x5555_5555) << 1) | ((r >> 1) & 0x5555_5555)
}

/// Read the trellis state of chunk-local symbol `ts` out of the reversed
/// words. Mirrors `q2b_states_rev`.
#[allow(dead_code)]
pub(crate) fn state_from_reversed_2b(r: &[u32; GROUPED_REV_WORDS], ts: usize) -> u32 {
    let bit = GROUPED_REV_BIT_BASE - 2 * ts as i32;
    let (q, sh) = ((bit >> 5) as usize, (bit & 31) as u32);
    let lo = r[q];
    let hi = if q + 1 < GROUPED_REV_WORDS {
        r[q + 1]
    } else {
        0
    };
    let f = if sh == 0 {
        lo
    } else {
        (lo >> sh) | (hi << (32 - sh))
    };
    f & 0xFFFF
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

    /// Build the 32-byte staged buffer for k-chunk `c` of `packed`, exactly as
    /// `q2b_stage_chunk` does: bytes [12,16) = the 4-byte history prefix
    /// (zero for chunk 0), bytes [16,32) = this chunk's 16 bytes.
    fn stage_chunk(packed: &[u8], c: usize) -> [u8; 32] {
        let mut buf = [0u8; 32];
        let base = c * (GROUPED_TILE_K / 4);
        if c > 0 {
            buf[12..16].copy_from_slice(&packed[base - 4..base]);
        }
        buf[16..32].copy_from_slice(&packed[base..base + 16]);
        buf
    }

    /// The tuned grouped kernel's decode is only allowed to exist because the
    /// pair-reversal is a PURE PERMUTATION: it must reproduce the sequential
    /// trellis recurrence exactly, for every chunk-local symbol including the
    /// zero-filled warm-up region of chunk 0.
    #[test]
    fn reversed_staged_row_matches_sequential_recurrence() {
        let (len, mut z) = (1024usize, 0x5EED_2B00u64);
        let mut syms = vec![0u8; len];
        for s in syms.iter_mut() {
            z = z
                .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                .wrapping_add(0xBF58_476D_1CE4_E5B9);
            *s = ((z >> 33) & 0x3) as u8;
        }
        let mut packed = vec![0u8; len / 4];
        for (t, &s) in syms.iter().enumerate() {
            packed[t / 4] |= (s & 0x3) << (2 * (t % 4));
        }

        let mut checked = 0usize;
        for c in 0..len / GROUPED_TILE_K {
            let staged = stage_chunk(&packed, c);
            let r = reverse_staged_row_2b_with(&staged, pair_reverse_32);
            // `window_state_2b` is itself property-tested against the
            // sequential recurrence above, so comparing to it covers every
            // chunk-local ts including chunk 0's zero-filled warm-up.
            for ts in 0..GROUPED_TILE_K {
                let t = c * GROUPED_TILE_K + ts;
                let want = window_state_2b(&packed, t);
                assert_eq!(
                    want,
                    state_from_reversed_2b(&r, ts),
                    "reversed-window state mismatch at chunk {c}, ts {ts} (t={t})"
                );
                checked += 1;
            }
        }
        assert_eq!(
            checked, len,
            "the sweep must cover every symbol, or a broken region could hide"
        );
    }

    /// The `ldmatrix.x4` address formula in `q2b_ldmatrix_x4` is a DERIVATION,
    /// and it is the part of variant 2 most likely to be silently wrong — a
    /// wrong lane->address map still loads real data, still runs, and just
    /// computes the wrong GEMM. The PTX ISA's `ldmatrix` section is truncated
    /// in the published HTML, so the derivation could not be read off a doc.
    ///
    /// This simulates the instruction from its two structural rules and
    /// asserts the result lands exactly on the `mma.m16n8k16` A fragment:
    ///   * `.x4` sub-matrix `i` takes its 8 row addresses from lanes `8i..8i+7`;
    ///   * within an 8x8 `.b16` matrix, lane `l` receives row `l/4`,
    ///     columns `(l%4)*2 .. +1`.
    ///
    /// The A fragment it must reproduce (g = lane>>2, tig = lane&3):
    ///
    /// ```text
    /// a0 = (row g,   k 2tig..+1)    a1 = (row g+8, k 2tig..+1)
    /// a2 = (row g,   k 2tig+8..9)   a3 = (row g+8, k 2tig+8..9)
    /// ```
    #[test]
    fn ldmatrix_x4_address_map_lands_on_the_mma_a_fragment() {
        const X_STRIDE: usize = 72; // variant 1's padded stride, in elements
        let elem = |row: usize, k: usize| row * 1000 + k;

        for kb in [0usize, 16, 32, 48] {
            // What each lane hands the instruction (q2b_ldmatrix_x4's caller).
            let addr = |lane: usize| (lane & 15) * X_STRIDE + kb + if lane >= 16 { 8 } else { 0 };

            let mut got = [[0usize; 2]; 32 * 4];
            for sub in 0..4usize {
                let row_start: Vec<usize> = (0..8).map(|r| addr(sub * 8 + r)).collect();
                for lane in 0..32usize {
                    let base = row_start[lane / 4];
                    for (c, slot) in got[lane * 4 + sub].iter_mut().enumerate() {
                        let off = base + (lane % 4) * 2 + c;
                        *slot = elem(off / X_STRIDE, off % X_STRIDE);
                    }
                }
            }

            for lane in 0..32usize {
                let (g, tig) = (lane >> 2, lane & 3);
                let want = [
                    [elem(g, kb + tig * 2), elem(g, kb + tig * 2 + 1)],
                    [elem(g + 8, kb + tig * 2), elem(g + 8, kb + tig * 2 + 1)],
                    [elem(g, kb + tig * 2 + 8), elem(g, kb + tig * 2 + 9)],
                    [elem(g + 8, kb + tig * 2 + 8), elem(g + 8, kb + tig * 2 + 9)],
                ];
                for (i, w) in want.iter().enumerate() {
                    assert_eq!(
                        &got[lane * 4 + i],
                        w,
                        "ldmatrix.x4 register a{i} for lane {lane} (g={g}, tig={tig}, kb={kb}) \
                         does not land on the mma A fragment"
                    );
                }
            }
        }
    }

    /// `ldmatrix` is only safe here BECAUSE variant 1 padded the staged
    /// activation stride. At the original 128 B stride every row of a phase
    /// lands on ONE bank; the check must show that difference, or it is not
    /// evidence for the padded stride.
    #[test]
    fn ldmatrix_phase_is_conflict_free_only_at_the_padded_stride() {
        let distinct_banks = |stride_elems: usize| -> usize {
            let mut seen = std::collections::HashSet::new();
            for row in 0..8usize {
                seen.insert(((row * stride_elems * 2) / 4) % 32);
            }
            seen.len()
        };
        assert_eq!(
            distinct_banks(64),
            1,
            "the ORIGINAL 64-element (128 B) stride must be shown to collide, or this test is \
             not evidence that the padding is what makes ldmatrix safe"
        );
        assert_eq!(
            distinct_banks(72),
            8,
            "variant 1's 72-element (144 B) stride must spread one ldmatrix phase over 8 banks"
        );
    }

    /// D12 negative control: the property test above must be ABLE to fail.
    /// A plain 32-bit reversal (no adjacent-bit swap) is the most plausible
    /// wrong permutation — it must be rejected, or the test proves nothing.
    #[test]
    fn reversed_staged_row_check_rejects_a_wrong_permutation() {
        fn brev_only(x: u32) -> u32 {
            x.reverse_bits()
        }
        let mut packed = vec![0u8; 64];
        let mut z = 0xA5A5_1234u64;
        for b in packed.iter_mut() {
            z = z
                .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                .wrapping_add(0xBF58_476D_1CE4_E5B9);
            *b = (z >> 40) as u8;
        }
        let staged = stage_chunk(&packed, 1);
        let good = reverse_staged_row_2b_with(&staged, pair_reverse_32);
        let bad = reverse_staged_row_2b_with(&staged, brev_only);

        let mismatches = (0..GROUPED_TILE_K)
            .filter(|&ts| {
                let t = GROUPED_TILE_K + ts;
                state_from_reversed_2b(&bad, ts) != window_state_2b(&packed, t)
            })
            .count();
        assert!(
            mismatches > 0,
            "GUARD IS BLIND: a bit-reversal WITHOUT the adjacent-bit swap reproduced every state, \
             so reversed_staged_row_matches_sequential_recurrence could not have caught a wrong \
             permutation."
        );
        assert_ne!(
            good, bad,
            "the two permutations must actually differ on this fixture"
        );
        for ts in 0..GROUPED_TILE_K {
            let t = GROUPED_TILE_K + ts;
            assert_eq!(
                state_from_reversed_2b(&good, ts),
                window_state_2b(&packed, t)
            );
        }
    }
}

// ---------------------------------------------------------------------------
// MEASURED 2026-08-21, H200, V4-Flash qtip2b — what the grouped GEMM's 100% SM
// is actually executing, and the ceiling on fixing it.
//
// TWO INDEPENDENT METHODS, agreeing to one point:
//
//  1. Static SASS mix of the shipping kernel (variant 0, bf16, 512 instrs,
//     `cuobjdump -sass`): trellis-decode-class opcodes 48.2%, fp math 3.7%,
//     memory 10.5%. Innermost loop (166 instrs): LOP3 46, SHF 24, LDS 18,
//     HADD2 16, IMAD 11, BREV 8, FADD 8, FMUL 8 — 54% decode vs 19% arithmetic.
//     LIMIT: static counts, not dynamic execution counts.
//
//  2. Dynamic ablation (diagnostic variant 3, decode stubbed to a symbol read,
//     same loads / loop / packing / accumulate, TUNED strides held fixed):
//       variant 1 (tuned, correct)      806,193 ns/call
//       variant 3 (tuned, no decode)    425,842 ns/call
//     => the trellis decode is 47.2% of this kernel's time. LIMIT: a LOWER
//     bound — the stub still pays a byte read and a convert.
//
// SO THE CEILING ON ANY DECODE OPTIMISATION IS 806,193/425,842 = 1.89x ON THIS
// KERNEL. Decode is the single largest item and worth fixing; it is not, and
// cannot be, a 70x.
//
// Tensor-core instructions in the whole kernel: 2. Despite the surrounding
// comments describing "a persistent tensor-core tile loop" and "the mma.sync
// pipeline is the point", the shipped inner loop does the math in scalar
// HADD2/FMUL/FADD. That is a separate, unclaimed defect.
//
// VARIANT SELECTION: `grouped_variant()` falls back to BASELINE and nothing in
// the serving path ever calls `set_grouped_variant`, so serving has been using
// variant 0 — the untuned kernel — the whole time. Measured:
//     grouped kernel   1,614,399 -> 806,193 ns/call   (2.00x)
//     prefill@256           303.2 -> 356.3 tok/s      (+17.5%)
//     B=256 serving        193.46 -> 219.23 tok/s     (+13.3%)
// Greedy output is token-identical on 3/3 prompts at temperature 0 — evidence,
// NOT a quality gate. Flipping the default should be gated on a ppl/greedy
// run, which is why this comment recommends it rather than doing it.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// THE B=256 PURE-DECODE KERNEL TABLE (H200, V4-Flash qtip2b, variant 0)
//
// METHOD: the server ran under `nsys --delay 330 --duration 25` so the capture
// window opens long after model load and prefill and closes while all 256
// sequences are still generating. Steady state asserted from the server's own
// logger inside the window: "Throughput (T/s) 204.80 ... 256 running". A
// whole-run profile is prefill-dominated and says nothing about a decode step;
// this is why the earlier prefill-shaped profile was misleading.
//
//   share  launches/step  kernel
//   45.0%          301    fp8_gemm::fp8_matmul_tiled          <-- largest
//   25.0%          129    qtip2b_grouped_gemm_kernel (MoE)
//   10.2%       22,794    ucopy_bf16
//    5.3%            1    sm90_xmma_gemm
//    5.0%       33,610    copy2d_bf16
//    3.9%       33,054    rope_i_bf16
//    0.8%       11,008    uneg_bf16
//
// SO THE MoE LANE IS FINISHED FOR THROUGHPUT. Amdahl on its 25.0%:
//    tuned variant (2.00x, measured)                 1.14x   (measured 1.133x)
//    + free decode (1.89x ceiling, measured)         1.23x
//    + full tensor cores (16.3x more, computed)      1.33x
// A PERFECT MoE kernel -- free decode AND full tensor-core rates -- buys 1.33x
// at B=256 decode. `fp8_matmul_tiled` made free would buy 1.82x.
//
// WHY THE MoE KERNEL IS ON THE WRONG ROOFLINE (arithmetic bound, computed from
// measured time + shapes anchored to the byte probe's 2 MiB/expert):
//   per launch at n_tokens=256: 1536 pairs x 8,388,608 weights = 2.577e10 FLOP
//     variant 0        1614 us ->  16.0 TFLOP/s
//     tuned             806 us ->  32.0 TFLOP/s = 47.8% of scalar FP32 peak
//     decode stubbed    426 us ->  60.5 TFLOP/s = 90.4% of scalar FP32 peak
//   H200: scalar FP32 FMA ~67 TFLOP/s, dense BF16 tensor core ~989 TFLOP/s.
// Decode-free, this kernel is essentially AT the scalar FP32 roofline. It is
// not inefficient -- it is using the wrong one, and there are 2 tensor-core
// instructions in the whole kernel. Tensor-core headroom beyond a free decode
// is ~16.3x. LIMITS: 989 TFLOP/s is a published dense peak (real kernels reach
// 60-80% of it); the SASS uses packed HADD2.F32, which may issue at 2x the
// FP32 lane rate, in which case "90% of peak" is nearer 45%; and this bounds
// arithmetic only -- it assumes a decode that can deliver operands in MMA
// register layout, which is exactly TCFRAG's unverified claim.
//
// THE STRUCTURAL ANOMALY IS NOT THE MoE: ~100,000 launches per decode step, of
// which rope_i_bf16 is 129 PER SEQUENCE PER STEP (= 43 layers x 3), i.e. RoPE
// and the bf16 copies are launched per-sequence and never batched across the
// cohort. 19.9% of step time for 99% of the launches. That is ArcKV/ArcAttention
// territory, not ArcQuant.
//
// Step count is 18.56 from the MoE gather counter vs 20.00 from throughput
// (window clipping partial steps), so per-step launch figures carry ~7%.
// ---------------------------------------------------------------------------
