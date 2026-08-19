//! Parent system: ArcQuant / QTIP
//!
//! The **V=4 / L=12** trellis decode family — a 32,768 B bf16 reproduction
//! table, with **K as a parameter**.
//!
//! # What is fixed, and what varies
//!
//! `L=12` and `V=4` fix the table: `2^12 × 4` bf16 values = **32,768 B**, which
//! is under the 48 KiB static `__shared__` limit, so a kernel stages the whole
//! thing per block with no `cudaFuncSetAttribute` opt-in and every lookup is an
//! LDS. That is the entire point of the family, and **the table does not read K
//! at all** — measured across a codebook sweep, so K can move without
//! re-baking or re-designing the codebook.
//!
//! What K changes is **symbol extraction, and nothing else**:
//!
//! | | K=8 | K=9 | K=10 |
//! |---|---|---|---|
//! | bits per weight (`K/V`) | 2.00 | **2.25** | 2.50 |
//! | symbol ↔ byte | **a symbol IS a byte** | spans a byte boundary | spans a byte boundary |
//! | bytes read per symbol | 1 | 2 | 2–3 |
//! | table, its size, its layout | identical | identical | identical |
//! | warmup `ceil(L/K)` | 2 | 2 | 2 |
//!
//! # Why this is a parameter and not a constant
//!
//! K=8 was built first and is now the **control**: it is the geometry whose
//! compiled-probe figures (5.375 inst/weight, 4.375 with the row-scale hoist,
//! vs 15.125 for the shipped K=4/V=2/L=16 rung) anchor the comparison.
//!
//! It is not expected to be the geometry that ships. Two CPU quality sweeps put
//! K=8/V=4/L=12 at **Δw_cos −0.00698** (random codebook) and **−0.00307**
//! (converged trellis-Lloyd) against a **±0.0008** ship threshold, and six
//! codebook designs failed to recover it — with the trellis off, LBG halves the
//! loss against random; with the trellis on, LBG *loses* to random. Trellis
//! freedom binds, not codebook coverage. **K=9 at 2.25 bpw measured +0.00402,
//! 5× better than the shipped control.** Those numbers are the coordinator's
//! CPU sweeps, not this module's.
//!
//! So K=9 must be a *specialisation*, not a rewrite. Hence [`Rung`].
//!
//! # The one thing that could still kill K=9
//!
//! K=8 is byte-aligned; K=9 is not. A 9-bit field spans a byte boundary, so
//! extraction costs a second byte read, a shift and a mask where K=8 costs one
//! `LDG`. **Whether that penalty eats the 3.46× is UNMEASURED here.** A SASS
//! pass over the compiled kernel is what answers it.
//!
//! Two things about that measurement, so it is not misread:
//!
//! 1. Nothing in this module or `kernels/qtip/qtip_gemv_v4l12.cu` has been
//!    compiled with `nvcc` or run on a GPU by the author. Every instruction
//!    count here is unestablished, at every K.
//! 2. The kernel's multi-byte read **clamps its byte index** so it cannot run
//!    past the last row. That clamp is a real per-symbol cost and it is
//!    *removable* — padding the row stride by `MAX_BYTES − 1` bytes deletes it.
//!    A K=9 measurement that includes the clamp is therefore an upper bound on
//!    the alignment penalty, not the penalty itself.
//!
//! # Bit layout — this is format
//!
//! Symbol `t` occupies bits `[t·K, t·K + K)` of the row's bitstream,
//! **LSB-first**: bit `j` lives in byte `j / 8` at bit position `j % 8`, with
//! position 0 the least significant.
//!
//! This is the same convention the shipped K=4/V=2 rung already uses — there,
//! symbol `2b` is the low nibble of byte `b` and symbol `2b+1` the high nibble,
//! which is exactly what this rule produces at K=4. Pinned by
//! [`tests::the_bit_layout_matches_the_shipped_k4_rungs_nibble_order`].
//!
//! # How to read an inst/weight census number against this kernel
//!
//! The alignment census measures `inst/weight` as a **differential over unroll
//! depth**. That cancels prologue and epilogue *exactly*, which is what makes
//! it robust — and it is also precisely why it is **blind to per-row and
//! per-block overhead**.
//!
//! Concretely: this kernel's move to one warp per row deleted a cross-warp
//! butterfly, a `warp_sums` shared array, and **both `__syncthreads` from the
//! row loop**. **None of that can ever appear in a census number.** The
//! instrument can neither credit nor charge it.
//!
//! So a flat census result does **not** mean the reshape did nothing. What the
//! census measures is steady-state inner-loop cost per weight; the reshape
//! shows up there only *indirectly*, through the extraction rate it changes
//! ([`Rung::extractions_per_weight_x10000`]). The barrier removal is a real
//! change in a regime that instrument does not observe, and it will only
//! surface in `ncu` or in wall-clock on a card.
//!
//! [`Rung::structural_per_row_overhead`] exists so that half has an accounting
//! of its own — structural counts, taken from the source, never presented as
//! instruction counts.
//!
//! And when the census does report a per-route cost, multiply it by **this
//! rung's own extraction rate at the shape in question**
//! ([`Rung::route_cost_per_weight_x10000`]), not by an illustrative endpoint.
//! At `in_features=4096` this kernel runs at 0.2651 extractions per weight, not
//! at 0.25 and not at 0.375.
//!
//! # The row-scale hoist
//!
//! [`RowScaleHoist`] is a separate switch from everything else because it is
//! the one lever that costs bit-exactness. Folding the per-row scale out of the
//! inner loop saves a measured 1.000 inst/weight, but it reassociates the
//! floating-point accumulation — `Σ (cb·s)·x` becomes `s · Σ cb·x` — and those
//! are not the same f32. Parity is therefore tested with the hoist **off**.

use half::bf16;

// ===========================================================================
// Fixed geometry: the table
// ===========================================================================

/// Trellis state width in bits. Fixed for this family.
pub const L: u32 = 12;
/// Reproduction values produced per symbol. Fixed for this family.
pub const V: u32 = 4;

/// `(1 << L) - 1`.
pub const STATE_MASK: u32 = (1u32 << L) - 1;
/// Number of distinct trellis states: `2^L`.
pub const LUT_STATES: usize = 1usize << L;
/// Total bf16 values in the table: `2^L × V`.
pub const LUT_ENTRIES: usize = LUT_STATES * V as usize;
/// Table footprint in bytes.
///
/// **32,768 B exactly**, which is what makes this family possible: it is under
/// the 48 KiB static `__shared__` limit, so the kernel needs no
/// `cudaFuncSetAttribute` opt-in to stage the whole table per block.
///
/// Contrast the shipped K=4/V=2/L=16 rung's table: `2^16 × 2` **f32** =
/// 524,288 B. That does not fit shared memory at any occupancy, so every
/// decoded symbol pays a dependent, data-scattered load to L2 — measured at
/// 388 GB/s, about 8% of H200 HBM, and it is the decode limiter. Note that
/// this is the *shipped default* path: `QtipCodebook::DEFAULT` is `Gaussian`,
/// which gathers from the stored table. (`qtip/mod.rs`'s claim that "the GPU
/// decode paths compute instead" holds only for the `Mcg` codebook, which is
/// not the default.) **Killing that 512 KiB gather is what the 32 KiB table is
/// for.**
///
/// Pinned by [`tests::lut_is_exactly_32768_bytes`] and by a `static_assert` in
/// `kernels/qtip/qtip_gemv_v4l12.cu`.
pub const LUT_BYTES: usize = LUT_ENTRIES * core::mem::size_of::<bf16>();

/// Compile-time twin of the `static_assert`s in
/// `kernels/qtip/qtip_gemv_v4l12.cu`. A geometry edit that pushes the table
/// over the static `__shared__` limit fails the Rust build too, instead of
/// waiting for a CUDA build that most contributors cannot run.
const _: () = {
    assert!(LUT_BYTES == 32_768);
    assert!(
        LUT_BYTES <= 48 * 1024,
        "the table must fit static shared memory without a cudaFuncSetAttribute opt-in"
    );
};

/// Threads per block the CUDA launcher uses. Mirrors `THREADS` in
/// `kernels/qtip/qtip_gemv_v4l12.cu`.
///
/// This is a *block* size — it sets how fast the 32 KiB table is staged and how
/// many rows a block works on at once. It is **not** how many threads share a
/// row; see [`KERNEL_LANES_PER_ROW`].
pub const KERNEL_THREADS: usize = 128;

/// One warp.
const WARP: usize = 32;

/// Threads that share one output row: **one warp**.
///
/// Not `KERNEL_THREADS`, and the difference is worth real instructions. Every
/// thread replays [`Rung::warmup_syms`] symbols before decoding its own slice,
/// so the extraction cost per weight is
///
/// ```text
///   (S + W) / (S · V)      S = symbols per lane, W = warmup, V = 4
/// ```
///
/// against an ideal of `1/V`. The tax is `1 + W/S`, so it is set entirely by
/// **slice length**, and slice length is `num_symbols / lanes_per_row`. Spreading
/// a row across a whole 128-thread block makes slices short and the tax large:
///
/// | `in_features` | 128 lanes/row | 32 lanes/row |
/// |---|---|---|
/// | 4096 | S=8, **1.25×** | S=32, **1.06×** |
/// | 1024 | S=2, **1.99×** | S=8, **1.24×** |
/// | 512  | S=1, **2.98×** | S=4, **1.48×** |
///
/// So a warp owns a row and a block owns `KERNEL_THREADS / 32` rows at once.
/// Same parallelism, four times the slice length. It also deletes the
/// cross-warp reduction and both `__syncthreads` from the row loop, since a
/// warp reduces with shuffles alone.
///
/// This is **K-independent**: it multiplies the extraction count at every K, so
/// it scales the alignment penalty too. Pinned by
/// [`tests::the_warmup_tax_is_bounded_by_slice_length`].
pub const KERNEL_LANES_PER_ROW: usize = WARP;

// ===========================================================================
// The K parameter
// ===========================================================================

/// The K this family was first built and probed at: **byte-aligned**, 2.00 bpw.
///
/// Kept as the control. Its extraction is a single `LDG` with no shift and no
/// mask, so it is the floor any non-byte-aligned K is measured against.
pub const K_CONTROL: u32 = 8;

/// Every K this family implements.
///
/// K=9 is the quality winner at 2.25 bpw; K=10 is carried so the alignment
/// penalty can be measured on *two* non-byte-aligned points rather than one,
/// which is what separates "9 is unlucky" from "non-alignment costs C".
pub const K_SUPPORTED: [u32; 3] = [8, 9, 10];

/// One packed row plus the two things needed to interpret it.
///
/// `packed`, `num_symbols` and `scale` always travel together and are
/// meaningless apart: the byte length alone does not determine the symbol count
/// (at K=9 `ceil(n·9/8)` is not invertible), and the scale is what turns table
/// values into weights. Grouping them keeps the decode entry points at a
/// readable arity and makes "a row" a thing the API names.
#[derive(Debug, Clone, Copy)]
pub struct Row<'a> {
    /// The packed symbol bitstream. Must be at least
    /// [`Rung::packed_bytes`]`(num_symbols)` long.
    pub packed: &'a [u8],
    /// How many symbols `packed` holds. Explicit because it cannot be
    /// recovered from the byte length at a non-byte-aligned K.
    pub num_symbols: usize,
    /// Per-row scale, applied to every decoded table value.
    pub scale: f32,
}

impl<'a> Row<'a> {
    /// Describe a packed row.
    pub fn new(packed: &'a [u8], num_symbols: usize, scale: f32) -> Self {
        Row {
            packed,
            num_symbols,
            scale,
        }
    }
}

/// A K=`k` / V=4 / L=12 decode rung.
///
/// The whole K-dependence of the family lives behind this type: symbol
/// extraction, the packed size, the state shift width, and the warmup depth.
/// The table and every path that touches it are K-independent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rung {
    k: u32,
}

impl Rung {
    /// The byte-aligned control rung, K=8.
    pub const CONTROL: Rung = Rung { k: K_CONTROL };

    /// Construct a rung, refusing a K this family has no decoder for.
    ///
    /// Refuses rather than clamping: K is an explicit field in the UQFF
    /// geometry section, so an unsupported value means an artifact from a
    /// newer Arc, and decoding its bitstream at the wrong K would produce
    /// plausible garbage rather than a fault.
    pub fn new(k: u32) -> Result<Self, String> {
        if K_SUPPORTED.contains(&k) {
            Ok(Rung { k })
        } else {
            Err(format!(
                "qtip v4l12: K={k} is not implemented (have {K_SUPPORTED:?}). Refusing rather \
                 than extracting this bitstream at the wrong symbol width."
            ))
        }
    }

    /// Bits per symbol.
    #[inline]
    pub fn k(self) -> u32 {
        self.k
    }

    /// Bits per weight × 100, as an integer so it can be compared exactly.
    /// K=8 → 200, K=9 → 225, K=10 → 250.
    #[inline]
    pub fn bpw_x100(self) -> u32 {
        self.k * 100 / V
    }

    /// True when a symbol lands exactly on byte boundaries, i.e. `K % 8 == 0`.
    ///
    /// The single fact that separates the control from every other K: at K=8
    /// extraction is one `LDG`, and at every other supported K it is a
    /// multi-byte read plus a shift and a mask.
    #[inline]
    pub fn is_byte_aligned(self) -> bool {
        self.k.is_multiple_of(8)
    }

    /// Symbols of history a thread consumes before decoding its own slice:
    /// `ceil(L / K)`. **2 at every supported K.**
    ///
    /// Two shifts move `2·K ≥ 2·8 = 16 ≥ L = 12` bits through the register, so
    /// the state after them is independent of whatever preceded — that is what
    /// lets a thread start at an arbitrary symbol offset.
    /// [`tests::state_is_determined_by_the_last_warmup_symbols`] proves it over
    /// the whole state space at every K rather than asserting it.
    ///
    /// **This is one symbol more than strictly necessary, deliberately.** The
    /// state that decodes symbol `t` keeps only `L − K` bits of `s_{t-1}`, and
    /// those bits are themselves supplied by `packed[t-1]`, so a *single* prior
    /// symbol already reconstructs it — pinned by
    /// [`tests::one_prior_symbol_is_the_true_minimum_warmup`]. `ceil(L/K)` is
    /// kept because it is the same definition `kernels/qtip/qtip_gemv.cu` uses
    /// at K=4/L=16 (where it walks 4 prior symbols and is likewise one over),
    /// and one extra read per thread per row is not worth two rungs disagreeing
    /// about what the constant means.
    #[inline]
    pub fn warmup_syms(self) -> usize {
        (L as usize).div_ceil(self.k as usize)
    }

    /// Advance the trellis by one symbol:
    /// `state_t = ((state_{t-1} << K) | sym_t) & ((1 << L) - 1)`.
    #[inline]
    pub fn next_state(self, state: u32, sym: u32) -> u32 {
        ((state << self.k) | sym) & STATE_MASK
    }

    /// Trellis symbols in a row of `in_features` weights.
    ///
    /// **K-independent** — it depends only on V. This is why K can move without
    /// the table or the activation indexing moving with it.
    #[inline]
    pub fn num_symbols(self, in_features: usize) -> usize {
        in_features / V as usize
    }

    /// Packed bytes holding `num_symbols` symbols: `ceil(num_symbols · K / 8)`.
    ///
    /// The general form, not `num_symbols / (8 / K)`. At K=9 there is no whole
    /// number of symbols per byte and the naive form divides by zero.
    #[inline]
    pub fn packed_bytes(self, num_symbols: usize) -> usize {
        (num_symbols * self.k as usize).div_ceil(8)
    }

    /// Packed bytes in a row of `in_features` weights.
    #[inline]
    pub fn packed_len(self, in_features: usize) -> usize {
        self.packed_bytes(self.num_symbols(in_features))
    }

    /// Bytes a single extraction may touch, worst case over the bit offsets
    /// that are actually **reachable**.
    ///
    /// The bit offset of symbol `t` is `(t·K) mod 8`, which ranges over the
    /// multiples of `gcd(K, 8)` and nothing else. The worst reachable offset is
    /// therefore `8 − gcd(K, 8)`, not 7, and the bound is
    /// `ceil((8 − gcd(K,8) + K) / 8)`.
    ///
    /// That distinction is not cosmetic. At **K=10** the offsets are only
    /// `{0, 2, 4, 6}`, so `off + K ≤ 16` and **two bytes always suffice** — the
    /// naive `ceil((7+K)/8)` says three and would have the kernel read, mask
    /// and discard a wasted byte on every symbol. 1 at K=8, 2 at K=9, 2 at
    /// K=10.
    ///
    /// **Two bytes is EXACT, not merely sufficient, for every K this family
    /// implements.** K=8 needs 1 and K=9/K=10 need 2, so the general 3-byte
    /// route is unreachable for any supported rung — it survives in the
    /// template only to price the unnecessary generality. A K that needed
    /// three (K=11, K=13, …) would have to be added deliberately; the kernel
    /// `static_assert`s against instantiating one, and
    /// [`tests::two_bytes_is_exact_for_every_supported_k`] pins it here.
    ///
    /// Mirrors `QtipSymExtract<K>::MAX_BYTES` in the kernel, and is also the
    /// number of bytes the row stride would have to be padded by to delete that
    /// kernel's tail clamp.
    #[inline]
    pub fn max_bytes_per_symbol(self) -> usize {
        let g = gcd8(self.k as usize);
        (8 - g + self.k as usize).div_ceil(8)
    }

    /// Symbols each lane decodes when a row is split across
    /// [`KERNEL_LANES_PER_ROW`] lanes.
    #[inline]
    pub fn syms_per_lane(self, num_symbols: usize) -> usize {
        num_symbols.div_ceil(KERNEL_LANES_PER_ROW)
    }

    /// Convert a per-EXTRACTION cost into a per-WEIGHT cost at a given shape.
    ///
    /// The census reports route costs per symbol extraction. Turning that into
    /// a per-weight number requires **this rung's actual extraction rate at
    /// the shape in question**, which is what
    /// [`Rung::extractions_per_weight_x10000`] gives. Quoting one of the
    /// census's illustrative endpoints (0.25 / 0.375 / 0.50) instead would be
    /// wrong for this kernel in both directions — it runs at 0.2651 at
    /// `in_features=4096` and 0.3105 at 1024.
    ///
    /// `inst_per_extraction_x100` is the route cost ×100 (so 175 means 1.75
    /// instructions per extraction). Result is ×10,000, matching
    /// [`Rung::extractions_per_weight_x10000`].
    pub fn route_cost_per_weight_x10000(
        self,
        num_symbols: usize,
        inst_per_extraction_x100: usize,
    ) -> usize {
        self.extractions_per_weight_x10000(num_symbols) * inst_per_extraction_x100 / 100
    }

    /// Per-row work the census **cannot see**, as structural counts.
    ///
    /// Returns `(block_barriers, warp_butterflies, shared_roundtrips)` for one
    /// output row.
    ///
    /// These are counts of *constructs in the source*, not instruction counts.
    /// Nothing here has been compiled, and turning any of it into a cycle or
    /// instruction figure requires `ncu` or a card. They exist because the
    /// census's differential-over-unroll-depth method cancels exactly this
    /// category, so the warp-per-row reshape would otherwise have no
    /// accounting at all.
    ///
    /// Before the reshape (a row split across a whole block): `(2, 2, 2)` — two
    /// `__syncthreads`, an intra-warp butterfly plus a cross-warp one, and a
    /// `warp_sums` store and load. After: `(0, 1, 0)`.
    pub fn structural_per_row_overhead(self) -> (usize, usize, usize) {
        // A warp owns a row, so the reduction never leaves the warp.
        (0, 1, 0)
    }

    /// Symbol extractions per decoded weight, ×10,000, for a row of
    /// `num_symbols` symbols split across [`KERNEL_LANES_PER_ROW`] lanes.
    ///
    /// The ideal is `1/V` = 0.25 (2500). The excess is the warmup replay: each
    /// lane but the first re-extracts [`Rung::warmup_syms`] symbols it will not
    /// accumulate. Reported as an integer so it can be asserted exactly.
    pub fn extractions_per_weight_x10000(self, num_symbols: usize) -> usize {
        if num_symbols == 0 {
            return 0;
        }
        let s = self.syms_per_lane(num_symbols);
        let active = num_symbols.div_ceil(s);
        let total = num_symbols + (active - 1) * self.warmup_syms();
        total * 10_000 / (num_symbols * V as usize)
    }

    /// Extract symbol `t` from a packed row.
    ///
    /// Bits `[t·K, t·K + K)`, LSB-first — see the module docs. Reads exactly
    /// the bytes those bits fall in, so a row of [`Rung::packed_bytes`] length
    /// is always in bounds.
    #[inline]
    pub fn extract(self, packed: &[u8], t: usize) -> u32 {
        let bit = t * self.k as usize;
        let b0 = bit / 8;
        let off = (bit % 8) as u32;
        let need = (off as usize + self.k as usize).div_ceil(8);
        let mut w = 0u32;
        for i in 0..need {
            w |= (packed[b0 + i] as u32) << (8 * i);
        }
        (w >> off) & ((1u32 << self.k) - 1)
    }

    /// Pack symbols into a bitstream — the inverse of [`Rung::extract`].
    ///
    /// Exists so tests can build fixtures at any K, and so the round-trip is
    /// something that can be asserted rather than assumed.
    pub fn pack(self, syms: &[u32]) -> Vec<u8> {
        let mut out = vec![0u8; self.packed_bytes(syms.len())];
        let mask = (1u64 << self.k) - 1;
        for (t, &s) in syms.iter().enumerate() {
            let bit = t * self.k as usize;
            let b0 = bit / 8;
            let off = (bit % 8) as u32;
            let v = (s as u64 & mask) << off;
            let need = (off as usize + self.k as usize).div_ceil(8);
            for i in 0..need {
                out[b0 + i] |= ((v >> (8 * i)) & 0xFF) as u8;
            }
        }
        out
    }

    /// Seed a thread's trellis state from the [`Rung::warmup_syms`] symbols
    /// preceding `sym_start`, exactly as the kernel's warmup loop does.
    ///
    /// Symbols before index 0 do not exist, so a thread starting inside the
    /// first few symbols walks from 0 with `state = 0`, which is the true
    /// initial state rather than an approximation of it.
    pub fn warmup_state(self, packed: &[u8], sym_start: usize) -> u32 {
        let warm_start = sym_start.saturating_sub(self.warmup_syms());
        let mut state = 0u32;
        for t in warm_start..sym_start {
            state = self.next_state(state, self.extract(packed, t));
        }
        state
    }

    fn check(self, packed: &[u8], num_symbols: usize, lut: &[bf16]) -> Result<(), String> {
        if lut.len() != LUT_ENTRIES {
            return Err(format!(
                "qtip v4l12: table has {} entries, expected {LUT_ENTRIES} (2^{L} × {V})",
                lut.len()
            ));
        }
        let want = self.packed_bytes(num_symbols);
        if packed.len() < want {
            return Err(format!(
                "qtip v4l12 K={}: {num_symbols} symbols need {want} packed bytes, got {}",
                self.k,
                packed.len()
            ));
        }
        Ok(())
    }

    /// Decode one packed row into `out`, in the rotated frame and scaled.
    ///
    /// This is the dequantize reference: a GPU dequantize of this geometry must
    /// reproduce it exactly (bf16→f32 widening is exact, and `cb · scale` is a
    /// single correctly-rounded f32 multiply on both sides).
    ///
    /// # Errors
    ///
    /// Returns an error rather than panicking on a length or table mismatch:
    /// this runs on artifact-derived data, where a wrong length means a corrupt
    /// or wrong-geometry payload, and a panic in a loader is not a diagnosis.
    pub fn decode_row_into(
        self,
        row: Row<'_>,
        lut: &[bf16],
        out: &mut [f32],
    ) -> Result<(), String> {
        let Row {
            packed,
            num_symbols,
            scale,
        } = row;
        self.check(packed, num_symbols, lut)?;
        if out.len() != num_symbols * V as usize {
            return Err(format!(
                "qtip v4l12: {num_symbols} symbols decode to {} weights, but out has {}",
                num_symbols * V as usize,
                out.len()
            ));
        }
        let mut state: u32 = 0;
        for t in 0..num_symbols {
            state = self.next_state(state, self.extract(packed, t));
            let off = state as usize * V as usize;
            let dst = t * V as usize;
            for v in 0..V as usize {
                out[dst + v] = lut[off + v].to_f32() * scale;
            }
        }
        Ok(())
    }

    /// Decode one packed row and accumulate `Σ w·x` — the CPU model of the
    /// fused decode+gemv kernel's arithmetic, operation for operation.
    ///
    /// The order of operations is load-bearing, not incidental. With
    /// [`RowScaleHoist::Off`] this evaluates, per weight,
    ///
    /// ```text
    ///   w   = cb * scale      // one f32 multiply, no contraction
    ///   acc = fma(w, x, acc)  // one fused multiply-add, single rounding
    /// ```
    ///
    /// which is exactly what `qtip_gemv_v4l12.cu` emits — the kernel writes the
    /// multiply as `__fmul_rn` so `--use_fast_math` cannot contract it into the
    /// following FMA, for the same reason `kernels/qtip/qtip_exact_fp.cuh`
    /// exists. Rust's [`f32::mul_add`] is the single-rounding FMA that matches
    /// `fmaf`.
    pub fn gemv_row(
        self,
        row: Row<'_>,
        lut: &[bf16],
        x: &[f32],
        hoist: RowScaleHoist,
    ) -> Result<f32, String> {
        let Row {
            packed,
            num_symbols,
            scale,
        } = row;
        self.check(packed, num_symbols, lut)?;
        if x.len() != num_symbols * V as usize {
            return Err(format!(
                "qtip v4l12: {num_symbols} symbols decode to {} weights, but x has {}",
                num_symbols * V as usize,
                x.len()
            ));
        }
        let mut state: u32 = 0;
        let mut acc = 0.0f32;
        for t in 0..num_symbols {
            state = self.next_state(state, self.extract(packed, t));
            let off = state as usize * V as usize;
            let xo = t * V as usize;
            for v in 0..V as usize {
                let cb = lut[off + v].to_f32();
                acc = match hoist {
                    RowScaleHoist::Off => (cb * scale).mul_add(x[xo + v], acc),
                    RowScaleHoist::On => cb.mul_add(x[xo + v], acc),
                };
            }
        }
        Ok(match hoist {
            RowScaleHoist::Off => acc,
            RowScaleHoist::On => acc * scale,
        })
    }

    /// Bit-exact CPU model of `qtip_fused_gemv_v4_l12_kernel` for one row.
    ///
    /// **This is the parity reference the GPU is held to, not an approximation
    /// of it.** [`Rung::gemv_row`] models the arithmetic; this models the
    /// arithmetic *and* the parallelism — the per-thread slicing, the warmup
    /// seeding, the warp butterfly, and the cross-warp butterfly — so a GPU
    /// result may be compared to it with `==` on the bits rather than with a
    /// tolerance.
    ///
    /// That distinction matters. A 32-way split reassociates the sum, so
    /// the kernel cannot be bit-equal to a single-threaded dot product and
    /// never could be; a tolerance-based gate would then be hiding a real class
    /// of bug (a mis-seeded thread) behind a bound chosen for a benign one
    /// (reassociation). Modelling the tree removes the tolerance entirely.
    pub fn gemv_row_gpu_model(
        self,
        row: Row<'_>,
        lut: &[bf16],
        x: &[f32],
        hoist: RowScaleHoist,
    ) -> Result<f32, String> {
        let Row {
            packed,
            num_symbols,
            scale,
        } = row;
        self.check(packed, num_symbols, lut)?;
        if x.len() != num_symbols * V as usize {
            return Err(format!(
                "qtip v4l12: {num_symbols} symbols decode to {} weights, but x has {}",
                num_symbols * V as usize,
                x.len()
            ));
        }
        // One warp owns the row; there is no cross-warp step to model because
        // the kernel does not have one.
        let sym_per_thread = self.syms_per_lane(num_symbols);

        // ---- per-lane accumulation ----
        let mut per_thread = [0f32; WARP];
        for (tid, slot) in per_thread.iter_mut().enumerate() {
            let sym_start = tid * sym_per_thread;
            if sym_start >= num_symbols {
                continue;
            }
            let sym_end = (sym_start + sym_per_thread).min(num_symbols);
            let mut state = self.warmup_state(packed, sym_start);
            let mut acc = 0f32;
            for t in sym_start..sym_end {
                state = self.next_state(state, self.extract(packed, t));
                let off = state as usize * V as usize;
                let xo = t * V as usize;
                for v in 0..V as usize {
                    let cb = lut[off + v].to_f32();
                    acc = match hoist {
                        RowScaleHoist::Off => (cb * scale).mul_add(x[xo + v], acc),
                        RowScaleHoist::On => cb.mul_add(x[xo + v], acc),
                    };
                }
            }
            *slot = acc;
        }

        // ---- the single warp butterfly ----
        let mut lanes = per_thread;
        warp_butterfly_sum(&mut lanes);

        Ok(match hoist {
            RowScaleHoist::Off => lanes[0],
            RowScaleHoist::On => lanes[0] * scale,
        })
    }
}

/// `gcd(k, 8)`. Small and total; `k` is a symbol width, never zero in practice,
/// and 0 maps to 8 which is the mathematically correct `gcd(0, 8)`.
const fn gcd8(k: usize) -> usize {
    let mut a = k % 8;
    let mut b = 8usize;
    while a != 0 {
        let t = b % a;
        b = a;
        a = t;
    }
    b
}

/// The XOR-shuffle butterfly the kernel's `warp_reduce_sum` performs.
///
/// `v[i] += v[i ^ off]` for `off` in 16, 8, 4, 2, 1, all lanes in lockstep.
/// Every lane ends holding the same value (float addition is commutative, so
/// lane `i` and lane `i^off` compute the identical sum at every step), and the
/// *order* of those additions is fixed — which is what makes the whole
/// reduction reproducible off-GPU.
fn warp_butterfly_sum(v: &mut [f32; WARP]) {
    for off in [16usize, 8, 4, 2, 1] {
        let prev = *v;
        for (i, slot) in v.iter_mut().enumerate() {
            *slot = prev[i] + prev[i ^ off];
        }
    }
}

// ===========================================================================
// The table (K-independent)
// ===========================================================================

/// FNV-1a over the table's raw little-endian bf16 bytes.
///
/// The table *is* format: symbols baked against one set of reproduction values
/// cannot be decoded against another, so a change to [`gaussian_lut_bf16`] is a
/// change to what every V=4/L=12 artifact means, at every K. Shape and summary
/// statistics do not catch that — a mutation that rounds every value to the
/// nearest integer leaves the count, the mean and σ inside tolerance while
/// destroying the codebook (measured: mutation M6 of this module's build-out
/// passed all 16 tests before this digest existed). Pinned by
/// [`tests::lut_contents_are_pinned`].
pub const LUT_DIGEST: u64 = 0x3366_F467_FEEA_F6FD;

/// FNV-1a 64 over the table's wire bytes, so the digest is over what would be
/// serialized rather than over an in-memory representation.
pub fn lut_digest(lut: &[bf16]) -> u64 {
    let mut h: u64 = 0xCBF2_9CE4_8422_2325;
    for v in lut {
        for b in v.to_bits().to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01B3);
        }
    }
    h
}

/// Splitmix-style hash to two well-distributed uniforms in (0, 1).
///
/// Byte-for-byte the mixing the K=4/V=2 rung's `gaussian_lut` uses
/// (`qtip/mod.rs::hash_to_two_uniforms`), restated here rather than shared so
/// this family's table is pinned by its own test and cannot drift when the
/// other rung's is retuned. A table change is an artifact-format change; the
/// two must be able to move independently.
fn hash_to_two_uniforms(seed: u32) -> (f32, f32) {
    let mut z = (seed as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    let hi = (z >> 32) as u32;
    let lo = (z & 0xFFFF_FFFF) as u32;
    let u1 = (hi as f32 + 1.0) / (u32::MAX as f32 + 2.0);
    let u2 = (lo as f32 + 1.0) / (u32::MAX as f32 + 2.0);
    (u1, u2)
}

/// Standard Box-Muller: two uniforms → two i.i.d. standard normals.
fn box_muller(u1: f32, u2: f32) -> (f32, f32) {
    let r = (-2.0_f32 * u1.ln()).sqrt();
    let theta = 2.0 * std::f32::consts::PI * u2;
    (r * theta.cos(), r * theta.sin())
}

/// The `[2^L, V]` bf16 reproduction table, in row-major `state`-major order.
///
/// **K-independent by construction**, which is what makes K a free parameter:
/// nothing here reads a symbol width. The coordinator's codebook sweep confirms
/// the same thing empirically at fixed L=12/V=4.
///
/// Each state draws **two** Box-Muller pairs, seeded by `2·state` and
/// `2·state + 1`. Distinct seeds across the whole table (state < 4096, so the
/// seeds occupy 0..8192 without collision), and the "V=4 is two V=2 draws"
/// structure stays visible instead of being buried in a bespoke mixer.
///
/// Values are stored as bf16 — the table's whole reason for fitting in shared
/// memory. bf16 keeps 8 explicit mantissa bits, which is coarser than the
/// shipped rung's f32 table, but the table is a *codebook*: its job is to place
/// 16,384 reproduction points, not to represent any particular weight
/// precisely, and the row scale is applied after the lookup. The quality
/// consequence of bf16 codebook values against f32 ones is **not measured** by
/// this module.
pub fn gaussian_lut_bf16() -> Vec<bf16> {
    let mut lut = Vec::with_capacity(LUT_ENTRIES);
    for state in 0..LUT_STATES as u32 {
        let (u1, u2) = hash_to_two_uniforms(2 * state);
        let (g0, g1) = box_muller(u1, u2);
        let (u3, u4) = hash_to_two_uniforms(2 * state + 1);
        let (g2, g3) = box_muller(u3, u4);
        lut.push(bf16::from_f32(g0));
        lut.push(bf16::from_f32(g1));
        lut.push(bf16::from_f32(g2));
        lut.push(bf16::from_f32(g3));
    }
    debug_assert_eq!(lut.len(), LUT_ENTRIES);
    lut
}

// ===========================================================================
// Row-scale policy
// ===========================================================================

/// Whether the per-row scale is applied inside the accumulation or hoisted out.
///
/// **This is the only knob in the family that changes results.** Everything
/// else about the decode is bit-determined; this one reassociates the sum, so
/// it is a separate type rather than a `bool` buried in an argument list, and
/// the parity gate runs with [`RowScaleHoist::Off`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RowScaleHoist {
    /// `acc += (cb · scale) · x` per weight — one extra multiply per weight,
    /// and the arithmetic the parity gate pins. The default, because a rung
    /// whose output depends on an optimisation flag is not a format.
    #[default]
    Off,
    /// `acc += cb · x`, then `y = scale · acc` once per row.
    ///
    /// Saves a measured 1.000 instruction per weight on the compiled probe.
    /// Costs bit-exactness with [`RowScaleHoist::Off`]: `Σ (cb·s)·x` and
    /// `s · Σ cb·x` differ in f32 by rounding, not by algebra.
    On,
}

impl RowScaleHoist {
    /// The ABI value the CUDA launcher takes. Kept explicit so the two sides
    /// cannot drift into disagreeing about which integer means which policy.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub(crate) fn as_abi(self) -> i32 {
        match self {
            RowScaleHoist::Off => 0,
            RowScaleHoist::On => 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic pseudo-random symbols in `[0, 2^k)`.
    fn fixture_symbols(n: usize, seed: u64, k: u32) -> Vec<u32> {
        let mut z = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        (0..n)
            .map(|_| {
                z ^= z << 13;
                z ^= z >> 7;
                z ^= z << 17;
                ((z >> 24) as u32) & ((1u32 << k) - 1)
            })
            .collect()
    }

    fn fixture_x(n: usize, seed: u64) -> Vec<f32> {
        let mut z = seed.wrapping_mul(0xD1B5_4A32_D192_ED03) | 1;
        (0..n)
            .map(|_| {
                z ^= z << 13;
                z ^= z >> 7;
                z ^= z << 17;
                ((z >> 40) as f32 / 8_388_608.0) - 1.0
            })
            .collect()
    }

    /// `Σ |w_i · x_i|` — the scale a reassociation bound must be stated
    /// against.
    ///
    /// Comparing two summation orders *relative to their own result* is wrong
    /// when the sum cancels: at 4095 symbols the terms are O(0.02) and the
    /// total is O(0.065), so a rounding difference of 8.5e-7 is 1.3e-5 of the
    /// result but 5e-9 of the work done. The L1 magnitude is the
    /// condition-aware denominator, and it keeps the guard sharp — a genuinely
    /// dropped slice moves the answer by O(1) of L1, six orders above the bound.
    fn l1_magnitude(
        r: Rung,
        packed: &[u8],
        n_sym: usize,
        scale: f32,
        lut: &[bf16],
        x: &[f32],
    ) -> f32 {
        let mut w = vec![0f32; n_sym * V as usize];
        r.decode_row_into(Row::new(packed, n_sym, scale), lut, &mut w)
            .unwrap();
        w.iter().zip(x.iter()).map(|(a, b)| (a * b).abs()).sum()
    }

    fn rungs() -> Vec<Rung> {
        K_SUPPORTED.iter().map(|&k| Rung::new(k).unwrap()).collect()
    }

    // ---- the K parameter ------------------------------------------------

    #[test]
    fn supported_rungs_and_their_bit_rates() {
        assert_eq!(Rung::CONTROL.k(), 8);
        assert_eq!(Rung::new(8).unwrap().bpw_x100(), 200);
        assert_eq!(Rung::new(9).unwrap().bpw_x100(), 225);
        assert_eq!(Rung::new(10).unwrap().bpw_x100(), 250);
        // The control is the only byte-aligned one; that is the whole reason
        // the alignment penalty needs measuring.
        assert!(Rung::CONTROL.is_byte_aligned());
        assert!(!Rung::new(9).unwrap().is_byte_aligned());
        assert!(!Rung::new(10).unwrap().is_byte_aligned());
    }

    #[test]
    fn an_unsupported_k_is_refused_not_clamped() {
        for k in [0u32, 1, 4, 7, 11, 16, 255] {
            let err = Rung::new(k).unwrap_err();
            assert!(err.contains("not implemented"), "K={k}: {err}");
        }
    }

    #[test]
    fn warmup_is_two_symbols_at_every_supported_k() {
        for r in rungs() {
            assert_eq!(r.warmup_syms(), 2, "K={}", r.k());
            // ceil(L/K), not L/K: the floor would be 1 here and would silently
            // leave stale state in every mid-row restart.
            assert!(r.warmup_syms() * r.k() as usize >= L as usize);
        }
    }

    #[test]
    fn packed_size_follows_the_bit_rate_at_every_k() {
        for r in rungs() {
            for k_in in [512usize, 1024, 4096, 7168] {
                let n_sym = r.num_symbols(k_in);
                // num_symbols depends only on V, never on K.
                assert_eq!(n_sym, k_in / V as usize);
                // ceil, not floor: at K=9 a row rarely lands on a byte.
                assert_eq!(r.packed_len(k_in), (n_sym * r.k() as usize).div_ceil(8));
                // ...and the packed size is the bit rate, to the byte.
                let bits = k_in * r.bpw_x100() as usize;
                assert_eq!(r.packed_len(k_in), bits.div_ceil(800));
            }
        }
    }

    #[test]
    fn the_control_and_the_quality_winner_differ_in_packed_size() {
        // K=8 is 2.00 bpw and K=9 is 2.25, so a K=9 row is 12.5% larger. This
        // is a real format difference — unlike K=8 vs the shipped K=4/V=2 rung,
        // where both are 2 bpw and the byte counts are identical. Worth
        // asserting: it means a K=9 artifact cannot be silently read as either.
        let k8 = Rung::new(8).unwrap();
        let k9 = Rung::new(9).unwrap();
        for k_in in [512usize, 4096, 7168] {
            assert!(k9.packed_len(k_in) > k8.packed_len(k_in));
            assert_eq!(k9.packed_len(k_in) * 8, k8.packed_len(k_in) * 9);
        }
    }

    // ---- bit layout -----------------------------------------------------

    #[test]
    fn pack_and_extract_round_trip_at_every_k() {
        for r in rungs() {
            for n in [1usize, 2, 7, 8, 63, 64, 1000, 1024] {
                for seed in 0..4u64 {
                    let syms = fixture_symbols(n, seed, r.k());
                    let packed = r.pack(&syms);
                    assert_eq!(packed.len(), r.packed_bytes(n), "K={} n={n}", r.k());
                    for (t, &want) in syms.iter().enumerate() {
                        assert_eq!(
                            r.extract(&packed, t),
                            want,
                            "K={} n={n} seed={seed} symbol {t}",
                            r.k()
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn extraction_never_reads_past_the_packed_row() {
        // The kernel clamps its byte index because it reads a compile-time
        // count; the reference reads exactly what it needs and must therefore
        // stay in bounds on its own. A panic here would be an OOB read in
        // production.
        for r in rungs() {
            for n in [1usize, 3, 8, 9, 17, 255, 256] {
                let syms = fixture_symbols(n, 5, r.k());
                let packed = r.pack(&syms);
                for t in 0..n {
                    let bit = t * r.k() as usize;
                    let need = ((bit % 8) + r.k() as usize).div_ceil(8);
                    assert!(
                        bit / 8 + need <= packed.len(),
                        "K={} n={n} symbol {t} would read past a {}-byte row",
                        r.k(),
                        packed.len()
                    );
                    let _ = r.extract(&packed, t);
                }
            }
        }
    }

    #[test]
    fn the_bit_layout_matches_the_shipped_k4_rungs_nibble_order() {
        // The LSB-first rule must reproduce, at K=4, exactly what
        // `qtip/mod.rs::dequantize_weights_rotated_f32` and `qtip_gemv.cu` do:
        // "byte b stores sym 2b in the low nibble, sym 2b+1 in the high
        // nibble". K=4 is not a supported rung here, so drive the rule
        // directly rather than through `Rung`.
        let syms: Vec<u32> = vec![0x3, 0xA, 0x5, 0xC];
        // Hand-packed by the shipped rung's rule.
        let want: Vec<u8> = vec![0xA3, 0xC5];
        let mut got = vec![0u8; 2];
        for (t, &s) in syms.iter().enumerate() {
            let bit = t * 4;
            got[bit / 8] |= ((s as u8) & 0x0F) << (bit % 8);
        }
        assert_eq!(
            got, want,
            "LSB-first packing must reproduce the K=4 nibble order"
        );
        // And the extraction rule inverts it.
        for (t, &s) in syms.iter().enumerate() {
            let bit = t * 4;
            let w = got[bit / 8] as u32;
            assert_eq!((w >> (bit % 8)) & 0x0F, s);
        }
    }

    #[test]
    fn the_control_extraction_is_exactly_a_byte_load() {
        // The property the K=8 kernel specialisation relies on: at K=8 the bit
        // offset is always 0 and symbol t IS byte t. If this ever stops being
        // true the specialisation is silently wrong.
        let r = Rung::CONTROL;
        let syms = fixture_symbols(64, 3, 8);
        let packed = r.pack(&syms);
        assert_eq!(packed.len(), 64);
        for (t, &s) in syms.iter().enumerate() {
            assert_eq!(t * r.k() as usize % 8, 0, "K=8 offsets must all be zero");
            assert_eq!(packed[t] as u32, s, "symbol {t} must be byte {t}");
            assert_eq!(r.extract(&packed, t), s);
        }
    }

    // ---- table ----------------------------------------------------------

    #[test]
    fn lut_is_exactly_32768_bytes() {
        assert_eq!(LUT_STATES, 4096);
        assert_eq!(LUT_ENTRIES, 16_384);
        assert_eq!(LUT_BYTES, 32_768);
        // The `<= 48 KiB` half is a `const _: () = { assert!(..) }` at module
        // scope, so it fails the build rather than a test run.
    }

    #[test]
    fn lut_contents_are_pinned() {
        // Not a shape check and not a statistics check: the exact bytes. Any
        // edit to `gaussian_lut_bf16` — including one that leaves the
        // distribution looking right — has to come here and change this
        // constant, which is the point. Every artifact baked in this family
        // decodes against these 16,384 values and no others, at every K.
        let lut = gaussian_lut_bf16();
        assert_eq!(
            lut_digest(&lut),
            LUT_DIGEST,
            "the V=4/L=12 reproduction table changed. This is a FORMAT change: symbols baked \
             against the old table decode to different weights against the new one. If the \
             change is intended, re-bake and update LUT_DIGEST in the same commit."
        );
    }

    #[test]
    fn lut_is_well_formed_and_deterministic() {
        let a = gaussian_lut_bf16();
        let b = gaussian_lut_bf16();
        assert_eq!(a.len(), LUT_ENTRIES);
        assert_eq!(a, b, "table must be deterministic across calls");
        assert!(
            a.iter().all(|v| v.to_f32().is_finite()),
            "table must contain no NaN/Inf"
        );
        let vals: Vec<f32> = a.iter().map(|v| v.to_f32()).collect();
        let mean = vals.iter().sum::<f32>() / vals.len() as f32;
        let var = vals.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / vals.len() as f32;
        assert!(mean.abs() < 0.05, "table mean {mean} should be ~0");
        assert!(
            (0.85..1.15).contains(&var.sqrt()),
            "table sigma {} should be ~1",
            var.sqrt()
        );
    }

    #[test]
    fn lut_uses_distinct_seeds_per_value_group() {
        let lut = gaussian_lut_bf16();
        let identical = (0..LUT_STATES)
            .filter(|s| {
                let o = s * V as usize;
                lut[o] == lut[o + 2] && lut[o + 1] == lut[o + 3]
            })
            .count();
        assert!(
            identical < LUT_STATES / 100,
            "{identical}/{LUT_STATES} states have duplicated value pairs — the two draws \
             collapsed to one seed"
        );
    }

    // ---- trellis --------------------------------------------------------

    #[test]
    fn state_is_determined_by_the_last_warmup_symbols() {
        // Proven over the entire state space at every K, not sampled: for
        // EVERY prior state and every pair of symbols, the state after
        // warmup_syms shifts must be identical. This is what lets threads
        // start at arbitrary symbol offsets, and it is what a wrong
        // warmup_syms breaks.
        for r in rungs() {
            assert_eq!(r.warmup_syms(), 2);
            let hi = (1u32 << r.k()) - 1;
            for a in [0u32, 1, 15, 16, hi / 2, hi - 1, hi] {
                for b in [0u32, 1, 15, 16, hi / 2, hi - 1, hi] {
                    let mut seen: Option<u32> = None;
                    for prior in 0..LUT_STATES as u32 {
                        let s = r.next_state(r.next_state(prior, a), b);
                        match seen {
                            None => seen = Some(s),
                            Some(expect) => assert_eq!(
                                s,
                                expect,
                                "K={}: state after 2 symbols still depends on prior state \
                                 {prior} (syms {a},{b})",
                                r.k()
                            ),
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn next_state_stays_in_range() {
        for r in rungs() {
            let hi = (1u32 << r.k()) - 1;
            for prior in 0..LUT_STATES as u32 {
                for sym in [0u32, 1, hi / 2, hi] {
                    assert!(r.next_state(prior, sym) < LUT_STATES as u32);
                }
            }
        }
    }

    #[test]
    fn one_prior_symbol_is_the_true_minimum_warmup() {
        // Documents WHY warmup_syms is one over the minimum, so nobody "fixes"
        // it in the wrong direction. Seeding from a single prior symbol
        // reproduces the decode state exactly at every supported K; seeding
        // from ZERO prior symbols does not, and this pins both halves.
        for r in rungs() {
            let syms = fixture_symbols(64, 5, r.k());
            let packed = r.pack(&syms);
            let mut full = 0u32;
            let mut zero_warmup_wrong = 0;
            for (t, &sym) in syms.iter().enumerate() {
                let truth = r.next_state(full, sym);
                if t >= 1 {
                    let one_prior = r.next_state(r.next_state(0, syms[t - 1]), sym);
                    assert_eq!(
                        one_prior,
                        truth,
                        "K={}: one prior symbol should suffice at symbol {t}",
                        r.k()
                    );
                }
                if r.next_state(0, sym) != truth {
                    zero_warmup_wrong += 1;
                }
                full = truth;
            }
            let _ = &packed;
            assert!(
                zero_warmup_wrong > 0,
                "K={}: zero warmup symbols would have to be wrong somewhere",
                r.k()
            );
        }
    }

    #[test]
    fn warmup_state_matches_a_full_walk_from_the_row_start() {
        for r in rungs() {
            let syms = fixture_symbols(97, 11, r.k());
            let packed = r.pack(&syms);
            let mut full = 0u32;
            for (t, &sym) in syms.iter().enumerate() {
                assert_eq!(
                    r.warmup_state(&packed, t),
                    full,
                    "K={}: restart at symbol {t} disagrees with the full walk",
                    r.k()
                );
                full = r.next_state(full, sym);
            }
            assert_eq!(r.warmup_state(&packed, syms.len()), full);
        }
    }

    // ---- decode ---------------------------------------------------------

    #[test]
    fn decode_produces_v_weights_per_symbol_at_the_scaled_table_value() {
        let lut = gaussian_lut_bf16();
        for r in rungs() {
            let n = 64usize;
            let syms = fixture_symbols(n, 7, r.k());
            let packed = r.pack(&syms);
            let scale = 0.017_5f32;
            let mut out = vec![0f32; n * V as usize];
            r.decode_row_into(Row::new(&packed, n, scale), &lut, &mut out)
                .unwrap();

            // Recompute independently of the loop under test.
            let mut state = 0u32;
            for (t, &sym) in syms.iter().enumerate() {
                state = ((state << r.k()) | sym) & 0xFFF;
                for v in 0..4usize {
                    let want = lut[state as usize * 4 + v].to_f32() * scale;
                    assert_eq!(
                        out[t * 4 + v],
                        want,
                        "K={} weight {v} of symbol {t} (state {state:#05x})",
                        r.k()
                    );
                }
            }
        }
    }

    #[test]
    fn decode_rejects_a_wrong_sized_table() {
        let r = Rung::CONTROL;
        let short = vec![bf16::ZERO; LUT_ENTRIES - 1];
        let mut out = vec![0f32; 4];
        let err = r
            .decode_row_into(Row::new(&[3u8], 1, 1.0), &short, &mut out)
            .unwrap_err();
        assert!(
            err.contains("16384"),
            "error must name the expected size: {err}"
        );

        let k4v2_sized = vec![bf16::ZERO; 65_536 * 2];
        let err = r
            .decode_row_into(Row::new(&[3u8], 1, 1.0), &k4v2_sized, &mut out)
            .unwrap_err();
        assert!(err.contains("expected"), "{err}");
    }

    #[test]
    fn decode_rejects_a_short_packed_row_and_a_wrong_sized_output() {
        let lut = gaussian_lut_bf16();
        for r in rungs() {
            // out too small
            let syms = fixture_symbols(4, 1, r.k());
            let packed = r.pack(&syms);
            let mut out = vec![0f32; 3];
            assert!(r
                .decode_row_into(Row::new(&packed, 4, 1.0), &lut, &mut out)
                .is_err());
            // packed too short for the claimed symbol count
            let mut out = vec![0f32; 40 * V as usize];
            let err = r
                .decode_row_into(Row::new(&packed, 40, 1.0), &lut, &mut out)
                .unwrap_err();
            assert!(err.contains("packed bytes"), "K={}: {err}", r.k());
        }
    }

    // ---- gemv parity ----------------------------------------------------

    #[test]
    fn gemv_without_the_hoist_is_bit_exact_against_decode_then_dot() {
        // THE parity contract, at every K. `gemv_row(.., Off)` must equal
        // decode-then-dot evaluated in the same order, to the bit — this is
        // the CPU side of the gate the GPU kernel has to pass.
        let lut = gaussian_lut_bf16();
        for r in rungs() {
            for (seed, n_sym) in [(1u64, 16usize), (2, 64), (3, 257), (4, 1024)] {
                let syms = fixture_symbols(n_sym, seed, r.k());
                let packed = r.pack(&syms);
                let x = fixture_x(n_sym * V as usize, seed ^ 0xABCD);
                let scale = 0.031_25f32 * (seed as f32 + 1.0);

                let mut w = vec![0f32; n_sym * V as usize];
                r.decode_row_into(Row::new(&packed, n_sym, scale), &lut, &mut w)
                    .unwrap();
                let mut want = 0f32;
                for i in 0..w.len() {
                    want = w[i].mul_add(x[i], want);
                }

                let got = r
                    .gemv_row(
                        Row::new(&packed, n_sym, scale),
                        &lut,
                        &x,
                        RowScaleHoist::Off,
                    )
                    .unwrap();
                assert_eq!(
                    got.to_bits(),
                    want.to_bits(),
                    "K={} seed {seed}: fused gemv {got:e} != decode-then-dot {want:e}",
                    r.k()
                );
            }
        }
    }

    #[test]
    fn the_hoist_changes_the_bits_and_is_therefore_not_the_parity_path() {
        // A guard that would be vacuous if it only checked closeness: the
        // point is that the hoist DOES change the result, which is why the
        // parity gate runs with it off. If this ever stops finding a
        // difference, the switch has stopped doing anything.
        let lut = gaussian_lut_bf16();
        for r in rungs() {
            let mut differed = 0;
            for seed in 0..24u64 {
                let syms = fixture_symbols(512, seed, r.k());
                let packed = r.pack(&syms);
                let x = fixture_x(512 * V as usize, seed ^ 0x51D);
                let scale = 0.021_37f32;
                let off = r
                    .gemv_row(Row::new(&packed, 512, scale), &lut, &x, RowScaleHoist::Off)
                    .unwrap();
                let on = r
                    .gemv_row(Row::new(&packed, 512, scale), &lut, &x, RowScaleHoist::On)
                    .unwrap();
                if off.to_bits() != on.to_bits() {
                    differed += 1;
                }
                let denom = off.abs().max(on.abs()).max(1e-6);
                assert!(
                    (off - on).abs() / denom < 1e-4,
                    "K={} seed {seed}: hoist changed the answer materially: {off:e} vs {on:e}",
                    r.k()
                );
            }
            assert!(
                differed > 0,
                "K={}: the row-scale hoist produced bit-identical results on every fixture — \
                 either the switch is dead or the reassociation is not happening",
                r.k()
            );
        }
    }

    // ---- the GPU model --------------------------------------------------

    #[test]
    fn a_single_active_lane_reduces_to_the_scalar_reference_to_the_bit() {
        // When only lane 0 has work the butterfly adds nothing but exact
        // zeros, so the model must collapse onto `gemv_row` bit-for-bit. This
        // is the anchor that ties the parallel model back to the arithmetic
        // reference; without it the model could be self-consistently wrong.
        let lut = gaussian_lut_bf16();
        for r in rungs() {
            for seed in 0..8u64 {
                let syms = fixture_symbols(1, seed, r.k());
                let packed = r.pack(&syms);
                let x = fixture_x(V as usize, seed ^ 0x99);
                let scale = 0.0231f32 * (seed as f32 + 1.0);
                for hoist in [RowScaleHoist::Off, RowScaleHoist::On] {
                    let modelled = r
                        .gemv_row_gpu_model(Row::new(&packed, 1, scale), &lut, &x, hoist)
                        .unwrap();
                    let scalar = r
                        .gemv_row(Row::new(&packed, 1, scale), &lut, &x, hoist)
                        .unwrap();
                    assert_eq!(
                        modelled.to_bits(),
                        scalar.to_bits(),
                        "K={} seed {seed} {hoist:?}: single-lane model {modelled:e} != scalar \
                         {scalar:e}",
                        r.k()
                    );
                }
            }
        }
    }

    #[test]
    fn the_butterfly_sums_all_lanes_exactly_once() {
        // The reduction must be a sum, not a sum-with-duplicates: a butterfly
        // written with the wrong offsets still terminates and still returns
        // "a number", so pin the value.
        let mut lanes = [0f32; WARP];
        for (i, l) in lanes.iter_mut().enumerate() {
            *l = (i + 1) as f32;
        }
        warp_butterfly_sum(&mut lanes);
        let want = (WARP * (WARP + 1) / 2) as f32; // 528
        for (i, l) in lanes.iter().enumerate() {
            assert_eq!(
                *l, want,
                "lane {i} holds {l}, expected every lane to hold {want}"
            );
        }
    }

    #[test]
    fn the_gpu_model_agrees_with_the_scalar_reference_within_reassociation() {
        // The model must not be free to return anything: a 128-way split only
        // reassociates the sum, so it stays close. This is what catches a
        // model whose thread slicing or warmup seeding is wrong, since those
        // produce WRONG table entries, not merely a different summation order.
        //
        // The symbol counts include several NOT divisible by KERNEL_THREADS.
        // That is deliberate and was learned the hard way: with only
        // power-of-two counts, `sym_per_thread` computed with a floor instead
        // of a ceil silently drops the tail of every row and every test still
        // passed (mutation M11 of this module's build-out).
        let lut = gaussian_lut_bf16();
        for r in rungs() {
            for n_sym in [1024usize, 1000, 129, 4095, 257] {
                for seed in 0..4u64 {
                    let syms = fixture_symbols(n_sym, seed, r.k());
                    let packed = r.pack(&syms);
                    let x = fixture_x(n_sym * V as usize, seed ^ 0x1234);
                    let scale = 0.0177f32;
                    let modelled = r
                        .gemv_row_gpu_model(
                            Row::new(&packed, n_sym, scale),
                            &lut,
                            &x,
                            RowScaleHoist::Off,
                        )
                        .unwrap();
                    let scalar = r
                        .gemv_row(
                            Row::new(&packed, n_sym, scale),
                            &lut,
                            &x,
                            RowScaleHoist::Off,
                        )
                        .unwrap();
                    let l1 = l1_magnitude(r, &packed, n_sym, scale, &lut, &x).max(1e-6);
                    assert!(
                        (modelled - scalar).abs() <= 1e-5 * l1,
                        "K={} n_sym={n_sym} seed {seed}: model {modelled:e} vs scalar \
                         {scalar:e} — |diff| {:e} exceeds 1e-5 of L1 {l1:e}",
                        r.k(),
                        (modelled - scalar).abs()
                    );
                }
            }
        }
    }

    #[test]
    fn the_gpu_model_covers_every_symbol_when_the_row_does_not_divide_evenly() {
        // Direct, arithmetic-free statement of the thing M11 broke: the union
        // of the threads' slices must be exactly [0, num_symbols). Checked as
        // coverage rather than as a float comparison, so it cannot be masked
        // by a tolerance.
        for n_sym in [1usize, 127, 128, 129, 1000, 1024, 4095] {
            for threads in [WARP] {
                let per = n_sym.div_ceil(threads);
                let mut covered = vec![0usize; n_sym];
                for tid in 0..threads {
                    let start = tid * per;
                    if start >= n_sym {
                        continue;
                    }
                    for slot in covered
                        .iter_mut()
                        .take((start + per).min(n_sym))
                        .skip(start)
                    {
                        *slot += 1;
                    }
                }
                assert!(
                    covered.iter().all(|c| *c == 1),
                    "n_sym={n_sym} threads={threads}: some symbol is covered != 1 times"
                );
            }
        }
    }

    #[test]
    fn the_gpu_model_is_deterministic_and_actually_models_a_tree() {
        // Determinism is what lets the CUDA gate compare bits. Differing from
        // the sequential reference is the anti-vacuity half: if the model
        // returned the same bits as `gemv_row`, it would not be modelling the
        // 32-way reduction at all and the CUDA comparison would pass against a
        // stub that just did a serial dot product.
        let lut = gaussian_lut_bf16();
        for r in rungs() {
            let n = 2048usize;
            let syms = fixture_symbols(n, 31, r.k());
            let packed = r.pack(&syms);
            let x = fixture_x(n * V as usize, 32);
            let scale = 0.0143f32;
            let a = r
                .gemv_row_gpu_model(Row::new(&packed, n, scale), &lut, &x, RowScaleHoist::Off)
                .unwrap();
            let b = r
                .gemv_row_gpu_model(Row::new(&packed, n, scale), &lut, &x, RowScaleHoist::Off)
                .unwrap();
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "K={}: model is not deterministic",
                r.k()
            );

            let seq = r
                .gemv_row(Row::new(&packed, n, scale), &lut, &x, RowScaleHoist::Off)
                .unwrap();
            assert_ne!(
                a.to_bits(),
                seq.to_bits(),
                "K={}: the 32-lane model returned the same bits as the sequential reference — \
                 it is not modelling the reduction tree, and the CUDA gate would pass against \
                 a serial stub",
                r.k()
            );
        }
    }

    #[test]
    fn the_warmup_tax_is_bounded_by_slice_length() {
        // The reason a warp owns a row instead of a block. Extractions per
        // weight are (S + W)/(S*V) against an ideal of 1/V = 0.25, so the tax
        // is 1 + W/S, and S is num_symbols / lanes_per_row. Splitting a row
        // across all 128 threads would quarter S and multiply the tax.
        //
        // Pinned as exact integers, because "we made it faster" is not a fact
        // and these are.
        let r = Rung::CONTROL;
        assert_eq!(KERNEL_LANES_PER_ROW, 32);
        assert_eq!(r.warmup_syms(), 2);

        // in_features = 4096 -> 1024 symbols -> S = 32
        assert_eq!(r.syms_per_lane(1024), 32);
        assert_eq!(r.extractions_per_weight_x10000(1024), 2651); // 1.06x ideal

        // in_features = 1024 -> 256 symbols -> S = 8
        assert_eq!(r.syms_per_lane(256), 8);
        assert_eq!(r.extractions_per_weight_x10000(256), 3105); // 1.24x ideal

        // in_features = 512 -> 128 symbols -> S = 4
        assert_eq!(r.extractions_per_weight_x10000(128), 3710); // 1.48x ideal

        // Splitting the same row across a whole 128-thread block would have
        // been materially worse — this is the change stated as a number.
        let s_block = 1024usize.div_ceil(KERNEL_THREADS);
        assert_eq!(s_block, 8);
        let tax_block = (s_block + r.warmup_syms()) * 10_000 / (s_block * V as usize);
        assert_eq!(tax_block, 3125); // 1.25x ideal, vs 1.06x at one warp per row
        assert!(tax_block > r.extractions_per_weight_x10000(1024));

        // The tax is never below the ideal and always above it while W > 0.
        for n in [32usize, 128, 256, 1024, 4096] {
            let e = r.extractions_per_weight_x10000(n);
            assert!(e > 2500, "n={n}: {e} must exceed the 0.25 ideal");
            assert!(
                e <= 7500,
                "n={n}: {e} is worse than one extraction per symbol"
            );
        }
        // ...and it is K-independent: the same tax multiplies every K, so it
        // scales the alignment penalty rather than competing with it.
        for rr in rungs() {
            assert_eq!(
                rr.extractions_per_weight_x10000(1024),
                r.extractions_per_weight_x10000(1024),
                "the warmup tax must not depend on K"
            );
        }
    }

    #[test]
    fn two_bytes_is_exact_for_every_supported_k() {
        // The gcd bound makes two bytes EXACT for K=9 and K=10, not just
        // adequate — so the general 3-byte extraction route is dead code for
        // every K anyone has. If a future K needs three, this test is where
        // that becomes a deliberate decision rather than a silent cost.
        for r in rungs() {
            assert!(
                r.max_bytes_per_symbol() <= 2,
                "K={}: needs {} bytes per symbol. The 3-byte route is supposed to be \
                 unreachable for every supported rung — adding a K that needs it is a real \
                 cost decision, not a detail.",
                r.k(),
                r.max_bytes_per_symbol()
            );
        }
        // ...and a K that WOULD need three exists, so the bound is not vacuous.
        let g = gcd8(11);
        assert_eq!(g, 1);
        assert_eq!(
            (8 - g + 11usize).div_ceil(8),
            3,
            "K=11 would need three bytes"
        );
    }

    #[test]
    fn the_per_row_overhead_the_census_cannot_see_is_accounted_separately() {
        // The census measures inst/weight as a differential over unroll depth,
        // which cancels prologue and epilogue exactly — and is therefore blind
        // to everything counted here. A flat census number does NOT mean the
        // warp-per-row reshape did nothing; the reshape's reduction half is
        // invisible to that instrument by construction.
        //
        // Structural counts from the source, NOT instruction counts. Nothing
        // here has been compiled.
        let r = Rung::CONTROL;
        let (barriers, butterflies, shared_roundtrips) = r.structural_per_row_overhead();
        assert_eq!(
            barriers, 0,
            "a warp owns a row; nothing needs a block barrier"
        );
        assert_eq!(
            butterflies, 1,
            "one intra-warp butterfly, no cross-warp step"
        );
        assert_eq!(shared_roundtrips, 0, "no `warp_sums` store/load");

        // What it was before the reshape, so the delta is on the record rather
        // than in a commit message.
        let (was_barriers, was_butterflies, was_shared) = (2usize, 2usize, 2usize);
        assert!(barriers < was_barriers);
        assert!(butterflies < was_butterflies);
        assert!(shared_roundtrips < was_shared);

        // It is K-independent, like the warmup tax.
        for rr in rungs() {
            assert_eq!(rr.structural_per_row_overhead(), (0, 1, 0));
        }
    }

    #[test]
    fn a_census_route_cost_is_multiplied_by_this_rungs_own_rate() {
        // Pins the arithmetic the census result will be run through, so the
        // step where an illustrative endpoint gets substituted for the real
        // rate cannot happen silently.
        let r = Rung::CONTROL;
        // in_features=4096 -> 1024 symbols -> 0.2651 extractions/weight.
        assert_eq!(r.extractions_per_weight_x10000(1024), 2651);
        // A hypothetical 1.75 inst/extraction route costs 0.4639 inst/weight
        // here — NOT 0.4375 (the 0.25 endpoint) and NOT 0.65625 (the 0.375 one).
        assert_eq!(r.route_cost_per_weight_x10000(1024, 175), 4639);
        assert_ne!(r.route_cost_per_weight_x10000(1024, 175), 2500 * 175 / 100);
        assert_ne!(r.route_cost_per_weight_x10000(1024, 175), 3750 * 175 / 100);
        // in_features=1024 -> 256 symbols -> 0.3105.
        assert_eq!(r.extractions_per_weight_x10000(256), 3105);
        assert_eq!(r.route_cost_per_weight_x10000(256, 175), 5433);
        // A zero-cost route costs nothing; a 1.00 route is exactly the rate.
        assert_eq!(r.route_cost_per_weight_x10000(1024, 0), 0);
        assert_eq!(r.route_cost_per_weight_x10000(1024, 100), 2651);
    }

    #[test]
    fn the_reachable_offset_bound_saves_a_byte_at_k10() {
        // `(t*K) mod 8` only ever takes the multiples of gcd(K, 8), so the
        // worst reachable offset is 8 - gcd(K,8), not 7. At K=10 that is the
        // difference between reading 2 bytes and 3 on EVERY symbol.
        for (k, want) in [(8u32, 1usize), (9, 2), (10, 2)] {
            let r = Rung::new(k).unwrap();
            assert_eq!(r.max_bytes_per_symbol(), want, "K={k}");
            // Prove the bound rather than trusting the formula: enumerate the
            // reachable offsets and check none needs more bytes.
            let mut worst = 0usize;
            for t in 0..64usize {
                let off = (t * k as usize) % 8;
                worst = worst.max((off + k as usize).div_ceil(8));
            }
            assert_eq!(worst, want, "K={k}: enumerated worst case disagrees");
        }
        // The naive all-offsets bound would have said 3 at K=10.
        assert_eq!((7 + 10usize).div_ceil(8), 3);
    }

    #[test]
    fn hoist_abi_values_are_pinned() {
        assert_eq!(RowScaleHoist::Off.as_abi(), 0);
        assert_eq!(RowScaleHoist::On.as_abi(), 1);
        assert_eq!(RowScaleHoist::default(), RowScaleHoist::Off);
    }

    #[test]
    fn the_rust_model_and_the_cuda_kernel_agree_on_their_shared_constants() {
        // The one guard that can keep an un-compilable kernel honest in a
        // no-GPU lane. `gemv_row_gpu_model` is only a model of the kernel at
        // KERNEL_THREADS; if the .cu changes its block size, or its fixed
        // geometry, or the size of its shared table, or which K values it
        // instantiates, the bit-exact CUDA gate silently starts comparing
        // against the wrong thing. So read the source.
        const SRC: &str = include_str!("../../kernels/qtip/qtip_gemv_v4l12.cu");

        // Whitespace-normalised on both sides. The kernel's declarations are
        // column-aligned today and `make fmt` runs clang-format over
        // `kernels/**/*.cu`, so matching raw text would make this guard break
        // on a reformat — a false alarm that teaches people to delete guards.
        fn squeeze(s: &str) -> String {
            s.split_whitespace().collect::<Vec<_>>().join(" ")
        }
        let src = squeeze(SRC);

        let want = [
            (
                format!("constexpr int THREADS = {KERNEL_THREADS};"),
                "block size",
            ),
            (format!("constexpr uint32_t QV4_L = {L};"), "L"),
            (format!("constexpr uint32_t QV4_V = {V};"), "V"),
            (
                "constexpr uint32_t QV4_LUT_ENTRIES = QV4_LUT_STATES * QV4_V;".to_string(),
                "table entry count",
            ),
            (
                format!("sizeof(__nv_bfloat16) == {LUT_BYTES}"),
                "table byte size static_assert",
            ),
            (
                "constexpr int WARMUP_SYMS = (QV4_L + K - 1) / K;".to_string(),
                "ceil(L/K) warmup",
            ),
            (
                "if (row_scale_hoist != 0)".to_string(),
                "nonzero-means-hoist ABI",
            ),
            (
                "static constexpr int MAX_BYTES = (8 - GCD8 + K + 7) / 8;".to_string(),
                "reachable-offset bytes-per-symbol bound",
            ),
            (
                "const int sym_per_thread = (num_symbols + 31) / 32;".to_string(),
                "one warp per row (the warmup-tax fix)",
            ),
            (
                "row += gridDim.x * N_WARPS".to_string(),
                "grid stride over rows in units of warps",
            ),
            (
                "static_assert(QtipSymExtract<K>::MAX_BYTES <= 2,".to_string(),
                "compile-time refusal of a K needing a 3-byte window",
            ),
        ];
        for (needle, what) in want {
            assert!(
                src.contains(&squeeze(&needle)),
                "kernels/qtip/qtip_gemv_v4l12.cu no longer states the {what} this module \
                 models. Expected to find:\n    {needle}\nIf the kernel changed deliberately, \
                 change this module to match — `gemv_row_gpu_model` is a model of that kernel, \
                 and the CUDA parity gate compares bits against it."
            );
        }

        // Every supported K must be instantiated, or the launcher's runtime
        // switch silently returns without launching and the output stays zero.
        //
        // The case label alone is not enough: `case 9: LAUNCH_K(T, 8, ..)` is a
        // one-character copy-paste away and would decode a K=9 bitstream at
        // K=8 — in bounds, no fault, wrong weights. So require the label and
        // the template argument to agree, in both hoist arms.
        for k in K_SUPPORTED {
            for hoist in ["true", "false"] {
                let needle = squeeze(&format!(
                    "case {k}: QTIP_V4L12_LAUNCH_K(T, {k}, {hoist}); break;"
                ));
                assert!(
                    src.contains(&needle),
                    "the launcher does not dispatch K={k} (hoist={hoist}) to the K={k} \
                     instantiation. Expected:\n    {needle}\nA missing case means the kernel \
                     never launches and `y` keeps whatever it held; a mismatched template \
                     argument means it decodes at the wrong symbol width."
                );
            }
        }
        // And nothing may claim a K the Rust side does not implement, which
        // would be a kernel the parity gate never covers.
        for k in [2u32, 3, 4, 5, 6, 7, 11, 12, 16] {
            assert!(
                !K_SUPPORTED.contains(&k),
                "K_SUPPORTED gained {k} without this guard being updated"
            );
            assert!(
                !src.contains(&squeeze(&format!("case {k}:"))),
                "the launcher dispatches K={k}, which K_SUPPORTED does not list — that \
                 instantiation has no CPU model and no parity gate"
            );
        }
        // ...and the byte-aligned control must keep its single-LDG
        // specialisation, which is the floor every other K is measured against.
        assert!(
            src.contains(&squeeze("struct QtipSymExtract<8>")),
            "the K=8 control lost its byte-aligned extraction specialisation"
        );
        // The warp-per-row shape removes the cross-warp reduction entirely. If
        // a `warp_sums` array comes back, the model's single butterfly is no
        // longer what the kernel does and the bit-exact gate would be comparing
        // against the wrong tree.
        assert!(
            !src.contains("__shared__ float warp_sums"),
            "the kernel regained a cross-warp reduction; `gemv_row_gpu_model` models ONE warp \
             butterfly and would no longer be bit-exact"
        );
        // Exactly ONE `__syncthreads` STATEMENT in the whole kernel: the one
        // after staging the table. Any barrier inside the row loop is per-row
        // overhead that the inst/weight census cannot see, so it would never
        // show up as a regression there — this is the only thing that watches
        // for it.
        //
        // Comments are stripped first. Counting raw occurrences also counts the
        // two places the header *describes* barriers, which made this guard
        // report 2 against a kernel that has 1 — a false alarm whose obvious
        // "fix" is to assert 2, at which point it would never catch a real
        // barrier again.
        let code_only: String = SRC
            .lines()
            .map(|l| match l.find("//") {
                Some(i) => &l[..i],
                None => l,
            })
            .collect::<Vec<_>>()
            .join("\n");
        let barriers = code_only.matches("__syncthreads();").count();
        assert_eq!(
            barriers, 1,
            "expected exactly one __syncthreads statement (staging the table), found \
             {barriers}. A barrier in the row loop is per-row cost that the \
             unroll-differential census is blind to by construction."
        );
    }

    // ---- the GPU gate ---------------------------------------------------

    /// `qtip_gemv_v4l12.cu` must reproduce [`Rung::gemv_row_gpu_model`] **to
    /// the bit**, at every supported K, with the row-scale hoist off.
    ///
    /// No tolerance, deliberately. The model already contains the kernel's
    /// reduction tree, so every legitimate source of float divergence has been
    /// accounted for and anything left is a defect. A tolerance here would be
    /// a place for a mis-seeded warmup or an off-by-one thread slice to hide.
    ///
    /// **This has never been run.** Nothing in this change has touched a GPU;
    /// the kernel has not been compiled. Treat a first green run of this test
    /// as the moment the kernel becomes real, not as a confirmation of
    /// something already established.
    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_v4l12_gemv_matches_the_cpu_model_bit_for_bit() -> candle_core::Result<()> {
        use crate::qtip::cuda_ops;
        use candle_core::{DType, Device, Tensor};

        let cuda = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(_) => {
                eprintln!("CUDA not available; skipping cuda_v4l12_gemv_matches_the_cpu_model");
                return Ok(());
            }
        };

        let lut = gaussian_lut_bf16();
        let lut_t = Tensor::from_slice(&lut, (LUT_STATES, V as usize), &cuda)?;

        for r in rungs() {
            // in_features chosen so that num_symbols = in_features/4 is
            // sometimes NOT a multiple of KERNEL_THREADS — that is the case a
            // floored sym_per_thread drops the tail of. The last two also
            // leave the row's final symbol mid-byte at K=9/K=10, which is what
            // exercises the kernel's tail clamp.
            for (n_rows, in_features) in [
                (8usize, 256usize),
                (64, 1024),
                (33, 4096),
                (7, 4000),
                (5, 516),
            ] {
                let n_sym = r.num_symbols(in_features);
                let row_bytes = r.packed_bytes(n_sym);
                let mut packed = Vec::with_capacity(n_rows * row_bytes);
                let mut scales = Vec::with_capacity(n_rows);
                for row in 0..n_rows {
                    let syms = fixture_symbols(n_sym, 1000 + row as u64, r.k());
                    packed.extend(r.pack(&syms));
                    scales.push(0.011f32 * (row as f32 + 1.0));
                }
                assert_eq!(packed.len(), n_rows * row_bytes);
                let x = fixture_x(in_features, 4242);

                let blocks_t = Tensor::from_slice(&packed, (n_rows, row_bytes), &cuda)?;
                let scales_t = Tensor::from_slice(&scales, n_rows, &cuda)?;
                let x_t = Tensor::from_slice(&x, in_features, &cuda)?.to_dtype(DType::F32)?;

                let y = cuda_ops::fused_gemv_v4l12_cuda(
                    &blocks_t,
                    &scales_t,
                    &lut_t,
                    &x_t,
                    in_features,
                    r,
                    RowScaleHoist::Off,
                )?;
                let got: Vec<f32> = y.flatten_all()?.to_vec1()?;
                assert_eq!(got.len(), n_rows);

                for row in 0..n_rows {
                    let slice = &packed[row * row_bytes..(row + 1) * row_bytes];
                    let want = r
                        .gemv_row_gpu_model(
                            Row::new(slice, n_sym, scales[row]),
                            &lut,
                            &x,
                            RowScaleHoist::Off,
                        )
                        .expect("model");
                    assert_eq!(
                        got[row].to_bits(),
                        want.to_bits(),
                        "K={} n_rows={n_rows} k={in_features} row {row}: kernel {:e} != model \
                         {want:e}",
                        r.k(),
                        got[row]
                    );
                }
            }
        }
        Ok(())
    }

    /// The launcher must refuse an artifact whose table is not this family's.
    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_v4l12_gemv_refuses_a_wrong_geometry_artifact() -> candle_core::Result<()> {
        use crate::qtip::cuda_ops;
        use candle_core::{Device, Tensor};

        let cuda = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(_) => {
                eprintln!("CUDA not available; skipping cuda_v4l12_gemv_refuses_wrong_geometry");
                return Ok(());
            }
        };

        let r = Rung::CONTROL;
        let (n_rows, in_features) = (4usize, 256usize);
        let n_sym = r.num_symbols(in_features);
        let row_bytes = r.packed_bytes(n_sym);
        let packed: Vec<u8> = (0..(n_rows * row_bytes))
            .map(|i| (i * 37 % 251) as u8)
            .collect();
        let scales = vec![0.02f32; n_rows];
        let x = fixture_x(in_features, 8);
        let blocks_t = Tensor::from_slice(&packed, (n_rows, row_bytes), &cuda)?;
        let scales_t = Tensor::from_slice(&scales, n_rows, &cuda)?;
        let x_t = Tensor::from_slice(&x, in_features, &cuda)?;
        let good_lut = Tensor::from_slice(&gaussian_lut_bf16(), (LUT_STATES, V as usize), &cuda)?;

        // Sanity: the good case is accepted, so the refusals below are about
        // the geometry and not about the fixture being broken.
        cuda_ops::fused_gemv_v4l12_cuda(
            &blocks_t,
            &scales_t,
            &good_lut,
            &x_t,
            in_features,
            r,
            RowScaleHoist::Off,
        )?;

        // An F32 table is the K=4/V=2 rung's.
        let f32_lut = Tensor::zeros((LUT_STATES, V as usize), candle_core::DType::F32, &cuda)?;
        let err = cuda_ops::fused_gemv_v4l12_cuda(
            &blocks_t,
            &scales_t,
            &f32_lut,
            &x_t,
            in_features,
            r,
            RowScaleHoist::Off,
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("BF16"), "expected a dtype refusal, got: {err}");

        // A 2^16 × 2 table is the K=4/V=2 rung's size.
        let big_lut = Tensor::zeros((65_536, 2), candle_core::DType::BF16, &cuda)?;
        let err = cuda_ops::fused_gemv_v4l12_cuda(
            &blocks_t,
            &scales_t,
            &big_lut,
            &x_t,
            in_features,
            r,
            RowScaleHoist::Off,
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("16384"), "expected a size refusal, got: {err}");

        // A row sized for a different K.
        let k9 = Rung::new(9).unwrap();
        let err = cuda_ops::fused_gemv_v4l12_cuda(
            &blocks_t,
            &scales_t,
            &good_lut,
            &x_t,
            in_features,
            k9,
            RowScaleHoist::Off,
        )
        .unwrap_err()
        .to_string();
        assert!(
            err.contains("packed") || err.contains("bytes"),
            "expected a row-size refusal, got: {err}"
        );

        Ok(())
    }
}
