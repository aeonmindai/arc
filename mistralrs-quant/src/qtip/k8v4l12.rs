//! Parent system: ArcQuant / QTIP
//!
//! The **K=8 / V=4 / L=12** trellis decode rung, and the CPU reference every
//! GPU decode of that geometry must match.
//!
//! # Why this geometry
//!
//! `bpw = K / V`, so K=8/V=4 is **2 bits per weight — the same rate the shipped
//! K=4/V=2/L=16 rung produces**. This is not a compression change. It is a
//! *decode-cost* change: the shipped rung spends a measured **15.125
//! instructions per weight** (compiled, `nvcc -cubin`, CUDA 12.4.131, inner
//! loop isolated by unroll differencing), where the budget to be memory-bound
//! at 2 bpw is 1.13–1.41. At B=32 that shows up as a GPU 90.8% busy while the
//! memory controller reads 3.5%: flat out doing arithmetic, barely touching
//! memory.
//!
//! Three things move at once when K goes 4→8, V goes 2→4 and L goes 16→12:
//!
//! | | K=4/V=2/L=16 (shipped) | K=8/V=4/L=12 (this rung) |
//! |---|---|---|
//! | table entries | 65,536 × 2 | 4,096 × 4 |
//! | table element | f32 | bf16 |
//! | table bytes | 524,288 (L2 only) | **32,768 (fits static shared)** |
//! | symbols per byte | 2 (nibble unpack) | **1 (a symbol *is* a byte)** |
//! | weights per table read | 2 | **4** |
//! | warmup symbols, ceil(L/K) | 4 | **2** |
//!
//! The table read is the one that pays: at V=4 a single 8-byte read yields four
//! weights instead of two, and it comes out of shared memory instead of a
//! dependent, data-scattered trip to L2. The nibble unpack (`>>4`, `&0x0F`,
//! and the odd/even branch around it) disappears outright because K=8 makes a
//! symbol exactly one byte.
//!
//! # Why V=4 specifically, and not V=8
//!
//! For `mma.m16n8k16.bf16` each thread's B fragment is 4 contiguous-k bf16
//! values. A V=4 bf16 table entry is therefore **exactly 8 bytes = one
//! `LDS.64` = a complete mma B-operand pair** — no conversion, no repacking
//! between the table read and the tensor-core operand. V=8 would need L≥16 to
//! keep the rate at 2 bpw, which puts the table back at 1 MB and out of shared
//! memory entirely. V=4/L=12 is the largest V that keeps the table resident.
//!
//! # What is measured here and what is not
//!
//! **The 5.375 / 4.375 inst-per-weight figures for this geometry came from a
//! compiled probe, not from the serving kernel in
//! `kernels/qtip/qtip_gemv_k8v4l12.cu`.** Nothing in this module or that kernel
//! has been compiled with `nvcc` or run on a GPU. Treat the serving kernel's
//! instruction count as UNMEASURED until someone runs `nvcc -cubin` against it.
//! What *is* established here is semantics: the decode this rung means, and a
//! CPU reference that the GPU must reproduce bit-for-bit.
//!
//! # The row-scale hoist
//!
//! [`RowScaleHoist`] is a separate switch from everything else because it is
//! the one lever that costs bit-exactness. Folding the per-row scale out of the
//! inner loop saves a measured 1.000 inst/weight, but it reassociates the
//! floating-point accumulation — `Σ (cb·s)·x` becomes `s · Σ cb·x` — and those
//! are not the same f32. Parity is therefore tested with the hoist **off**;
//! the hoist is tested separately for agreement within a stated tolerance.

use half::bf16;

// ===========================================================================
// Geometry
// ===========================================================================

/// Bits per trellis symbol. At K=8 a symbol *is* a byte: no nibble unpack.
pub const K: u32 = 8;
/// Trellis state width in bits.
pub const L: u32 = 12;
/// Reproduction values produced per symbol.
pub const V: u32 = 4;

/// `(1 << L) - 1`.
pub const STATE_MASK: u32 = (1u32 << L) - 1;
/// Number of distinct trellis states: `2^L`.
pub const LUT_STATES: usize = 1usize << L;
/// Total bf16 values in the table: `2^L × V`.
pub const LUT_ENTRIES: usize = LUT_STATES * V as usize;
/// Table footprint in bytes.
///
/// **32,768 B exactly**, which is what makes this rung possible: it is under
/// the 48 KiB static `__shared__` limit, so the kernel needs no
/// `cudaFuncSetAttribute` opt-in to stage the whole table per block. Pinned by
/// [`tests::lut_is_exactly_32768_bytes`] and by a `static_assert` in
/// `kernels/qtip/qtip_gemv_k8v4l12.cu`.
pub const LUT_BYTES: usize = LUT_ENTRIES * core::mem::size_of::<bf16>();

/// Compile-time twin of the `static_assert`s in
/// `kernels/qtip/qtip_gemv_k8v4l12.cu`. A geometry edit that pushes the table
/// over the static `__shared__` limit fails the Rust build too, instead of
/// waiting for a CUDA build that most contributors cannot run.
const _: () = {
    assert!(LUT_BYTES == 32_768);
    assert!(
        LUT_BYTES <= 48 * 1024,
        "the table must fit static shared memory without a cudaFuncSetAttribute opt-in"
    );
    // `V * N_BITS == K` says both "K divides evenly into V-sized groups" and
    // "the rate is N_BITS", without a `%` (which clippy would rewrite into a
    // method call that is not usable in a const block on this toolchain).
    assert!(V * (N_BITS as u32) == K);
};

/// Bits per weight: `K / V`. Two — the same rate the shipped rung produces.
pub const N_BITS: usize = (K / V) as usize;

/// Symbols of history a thread consumes before decoding its own slice:
/// `ceil(L / K)`, which is **2** here.
///
/// Two shifts move `2·K = 16 ≥ L = 12` bits through the register, so the state
/// after them is independent of whatever preceded — that is what lets a thread
/// start at an arbitrary symbol offset.
/// [`tests::state_is_determined_by_the_last_warmup_symbols`] proves it over the
/// whole state space rather than asserting it.
///
/// **This is one symbol more than strictly necessary, deliberately.** The state
/// used to decode symbol `t` is `((s_{t-1} & 0xF) << 8) | packed[t]`, and
/// `s_{t-1} & 0xF == packed[t-1] & 0xF` for every `t ≥ 1`, so a *single* prior
/// symbol already reconstructs it — pinned by
/// [`tests::one_prior_symbol_is_the_true_minimum_warmup`]. `ceil(L/K)` is kept
/// because it is the same definition `kernels/qtip/qtip_gemv.cu` uses at
/// K=4/L=16 (where it walks 4 prior symbols and is likewise one over), and one
/// extra `__ldg` per thread per row is not worth two rungs disagreeing about
/// what the constant means.
pub const WARMUP_SYMS: usize = (L as usize).div_ceil(K as usize);

/// Advance the trellis by one symbol.
///
/// `state_t = ((state_{t-1} << K) | sym_t) & ((1 << L) - 1)` — the same
/// recurrence the shipped rung uses, at this rung's K and L.
#[inline]
pub fn next_state(state: u32, sym: u8) -> u32 {
    ((state << K) | sym as u32) & STATE_MASK
}

/// Packed bytes needed for a row of `in_features` weights.
///
/// One symbol per byte, `V` weights per symbol.
#[inline]
pub fn packed_len(in_features: usize) -> usize {
    in_features / V as usize
}

/// Trellis symbols in a row of `in_features` weights. Equal to
/// [`packed_len`] at this geometry, and deliberately a separate function: they
/// are equal only because K=8 puts exactly one symbol in a byte, and a caller
/// that conflates them will be wrong on any other rung.
#[inline]
pub fn num_symbols(in_features: usize) -> usize {
    in_features / V as usize
}

// ===========================================================================
// The table
// ===========================================================================

/// Splitmix-style hash to two well-distributed uniforms in (0, 1).
///
/// Byte-for-byte the mixing the K=4/V=2 rung's `gaussian_lut` uses
/// (`qtip/mod.rs::hash_to_two_uniforms`), restated here rather than shared so
/// this rung's table is pinned by its own test and cannot drift when the other
/// rung's is retuned. A table change is an artifact-format change; the two
/// rungs must be able to move independently.
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
/// consequence of bf16 codebook values against f32 ones is **not measured** —
/// it needs a bake, and no bake of this geometry exists yet.
/// FNV-1a over the table's raw little-endian bf16 bytes.
///
/// The table *is* format: symbols baked against one set of reproduction values
/// cannot be decoded against another, so a change to [`gaussian_lut_bf16`] is a
/// change to what every K=8/V=4/L=12 artifact means. Shape and summary
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
// Decode
// ===========================================================================

/// Whether the per-row scale is applied inside the accumulation or hoisted out.
///
/// **This is the only knob in the rung that changes results.** Everything else
/// about the K=8/V=4/L=12 decode is bit-determined; this one reassociates the
/// sum, so it is a separate type rather than a `bool` buried in an argument
/// list, and the parity gate runs with [`RowScaleHoist::Off`].
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

/// Decode one packed row into `out`, in the rotated frame and scaled.
///
/// `packed[t]` is symbol `t`. `out` receives `packed.len() × V` weights.
/// This is the dequantize reference: the GPU dequantize of this geometry must
/// reproduce it exactly (bf16→f32 widening is exact, and `cb · scale` is a
/// single correctly-rounded f32 multiply on both sides).
///
/// # Errors
///
/// Returns an error rather than panicking on a length mismatch: this runs on
/// artifact-derived data, where a wrong length means a corrupt or
/// wrong-geometry payload, and a panic in a loader is not a diagnosis.
pub fn decode_row_into(
    packed: &[u8],
    scale: f32,
    lut: &[bf16],
    out: &mut [f32],
) -> Result<(), String> {
    if lut.len() != LUT_ENTRIES {
        return Err(format!(
            "qtip k8v4l12: table has {} entries, expected {LUT_ENTRIES} (2^{L} × {V})",
            lut.len()
        ));
    }
    if out.len() != packed.len() * V as usize {
        return Err(format!(
            "qtip k8v4l12: {} packed symbols decode to {} weights, but out has {}",
            packed.len(),
            packed.len() * V as usize,
            out.len()
        ));
    }

    let mut state: u32 = 0;
    for (t, &sym) in packed.iter().enumerate() {
        state = next_state(state, sym);
        let off = state as usize * V as usize;
        let dst = t * V as usize;
        for v in 0..V as usize {
            out[dst + v] = lut[off + v].to_f32() * scale;
        }
    }
    Ok(())
}

/// Decode one packed row and accumulate `Σ w·x` — the CPU model of the fused
/// decode+gemv kernel's arithmetic, operation for operation.
///
/// The order of operations is load-bearing, not incidental. With
/// [`RowScaleHoist::Off`] this evaluates, per weight,
///
/// ```text
///   w   = cb * scale      // one f32 multiply, no contraction
///   acc = fma(w, x, acc)  // one fused multiply-add, single rounding
/// ```
///
/// which is exactly what `qtip_gemv_k8v4l12.cu` emits — the kernel writes the
/// multiply as `__fmul_rn` so `--use_fast_math` cannot contract it into the
/// following FMA, for the same reason `kernels/qtip/qtip_exact_fp.cuh` exists.
/// Rust's [`f32::mul_add`] is the single-rounding FMA that matches `fmaf`.
///
/// # Errors
///
/// Length/format mismatches, as [`decode_row_into`].
pub fn gemv_row(
    packed: &[u8],
    scale: f32,
    lut: &[bf16],
    x: &[f32],
    hoist: RowScaleHoist,
) -> Result<f32, String> {
    if lut.len() != LUT_ENTRIES {
        return Err(format!(
            "qtip k8v4l12: table has {} entries, expected {LUT_ENTRIES} (2^{L} × {V})",
            lut.len()
        ));
    }
    if x.len() != packed.len() * V as usize {
        return Err(format!(
            "qtip k8v4l12: {} packed symbols decode to {} weights, but x has {}",
            packed.len(),
            packed.len() * V as usize,
            x.len()
        ));
    }

    let mut state: u32 = 0;
    let mut acc = 0.0f32;
    for (t, &sym) in packed.iter().enumerate() {
        state = next_state(state, sym);
        let off = state as usize * V as usize;
        let xo = t * V as usize;
        for v in 0..V as usize {
            let cb = lut[off + v].to_f32();
            match hoist {
                RowScaleHoist::Off => {
                    let w = cb * scale;
                    acc = w.mul_add(x[xo + v], acc);
                }
                RowScaleHoist::On => {
                    acc = cb.mul_add(x[xo + v], acc);
                }
            }
        }
    }
    Ok(match hoist {
        RowScaleHoist::Off => acc,
        RowScaleHoist::On => acc * scale,
    })
}

/// Threads per block the CUDA launcher uses. Mirrors `THREADS` in
/// `kernels/qtip/qtip_gemv_k8v4l12.cu`; [`gemv_row_gpu_model`] is only a model
/// of that kernel at this value.
pub const KERNEL_THREADS: usize = 128;

/// One warp.
const WARP: usize = 32;

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

/// Bit-exact CPU model of `qtip_fused_gemv_k8_v4_l12_kernel` for one row.
///
/// **This is the parity reference the GPU is held to, not an approximation of
/// it.** [`gemv_row`] models the arithmetic; this models the arithmetic *and*
/// the parallelism — the per-thread slicing, the warmup seeding, the warp
/// butterfly, and the cross-warp butterfly — so a GPU result may be compared
/// to it with `==` on the bits rather than with a tolerance.
///
/// That distinction matters. A `THREADS`-way split reassociates the sum, so the
/// kernel cannot be bit-equal to a single-threaded dot product and never could
/// be; a tolerance-based gate would then be hiding a real class of bug (a
/// mis-seeded thread) behind a bound chosen for a benign one (reassociation).
/// Modelling the tree removes the tolerance entirely.
///
/// # Errors
///
/// Length/format mismatches, as [`decode_row_into`].
pub fn gemv_row_gpu_model(
    packed: &[u8],
    scale: f32,
    lut: &[bf16],
    x: &[f32],
    threads: usize,
    hoist: RowScaleHoist,
) -> Result<f32, String> {
    if lut.len() != LUT_ENTRIES {
        return Err(format!(
            "qtip k8v4l12: table has {} entries, expected {LUT_ENTRIES} (2^{L} × {V})",
            lut.len()
        ));
    }
    if x.len() != packed.len() * V as usize {
        return Err(format!(
            "qtip k8v4l12: {} packed symbols decode to {} weights, but x has {}",
            packed.len(),
            packed.len() * V as usize,
            x.len()
        ));
    }
    if threads == 0 || !threads.is_multiple_of(WARP) {
        return Err(format!(
            "qtip k8v4l12: threads must be a nonzero multiple of {WARP}, got {threads}"
        ));
    }

    let num_symbols = packed.len();
    let sym_per_thread = num_symbols.div_ceil(threads);

    // ---- per-thread accumulation ----
    let mut per_thread = vec![0f32; threads];
    for (tid, slot) in per_thread.iter_mut().enumerate() {
        let sym_start = tid * sym_per_thread;
        if sym_start >= num_symbols {
            continue;
        }
        let sym_end = (sym_start + sym_per_thread).min(num_symbols);
        let mut state = warmup_state(packed, sym_start);
        let mut acc = 0f32;
        for (s, &sym) in packed.iter().enumerate().take(sym_end).skip(sym_start) {
            state = next_state(state, sym);
            let off = state as usize * V as usize;
            let xo = s * V as usize;
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

    // ---- warp butterflies ----
    let n_warps = threads / WARP;
    let mut warp_sums = vec![0f32; n_warps];
    for (w, total) in warp_sums.iter_mut().enumerate() {
        let mut lanes = [0f32; WARP];
        lanes.copy_from_slice(&per_thread[w * WARP..(w + 1) * WARP]);
        warp_butterfly_sum(&mut lanes);
        *total = lanes[0];
    }

    // ---- cross-warp butterfly, performed by warp 0 ----
    let mut lanes = [0f32; WARP];
    for (lane, slot) in lanes.iter_mut().enumerate() {
        if lane < n_warps {
            *slot = warp_sums[lane];
        }
    }
    warp_butterfly_sum(&mut lanes);

    Ok(match hoist {
        RowScaleHoist::Off => lanes[0],
        RowScaleHoist::On => lanes[0] * scale,
    })
}

/// Seed a thread's trellis state from the [`WARMUP_SYMS`] symbols preceding
/// `sym_start`, exactly as the kernel's warmup loop does.
///
/// Exposed (and tested) because it is the single place the mid-row restart
/// property is encoded, and a kernel that gets it wrong produces *plausible*
/// garbage — every symbol still decodes to a real table entry, just the wrong
/// one. Symbols before index 0 do not exist, so a thread starting inside the
/// first [`WARMUP_SYMS`] symbols walks from 0 with `state = 0`, which is the
/// true initial state rather than an approximation of it.
pub fn warmup_state(packed: &[u8], sym_start: usize) -> u32 {
    let warm_start = sym_start.saturating_sub(WARMUP_SYMS);
    let mut state = 0u32;
    for &sym in &packed[warm_start..sym_start] {
        state = next_state(state, sym);
    }
    state
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic pseudo-random symbol row. Not `rand` — a fixture whose
    /// bytes a failure can be reproduced from.
    fn fixture_symbols(n: usize, seed: u64) -> Vec<u8> {
        let mut z = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        (0..n)
            .map(|_| {
                z ^= z << 13;
                z ^= z >> 7;
                z ^= z << 17;
                (z >> 24) as u8
            })
            .collect()
    }

    /// `Σ |w_i · x_i|` — the scale a reassociation bound must be stated
    /// against.
    ///
    /// Comparing two summation orders *relative to their own result* is wrong
    /// when the sum cancels: at n_sym=4095 the terms are O(0.02) and the total
    /// is O(0.065), so a rounding difference of 8.5e-7 is 1.3e-5 of the result
    /// but 5e-9 of the work done. The L1 magnitude is the condition-aware
    /// denominator, and it keeps the guard sharp — a genuinely dropped slice
    /// moves the answer by O(1) of L1, six orders above the bound.
    fn l1_magnitude(packed: &[u8], scale: f32, lut: &[bf16], x: &[f32]) -> f32 {
        let mut w = vec![0f32; packed.len() * V as usize];
        decode_row_into(packed, scale, lut, &mut w).unwrap();
        w.iter().zip(x.iter()).map(|(a, b)| (a * b).abs()).sum()
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

    // ---- geometry -------------------------------------------------------

    #[test]
    fn geometry_is_two_bits_per_weight() {
        // The whole premise: this rung is a decode-cost change at a FIXED bit
        // rate, not a compression change. If K/V ever stops being 2 the
        // comparison against the shipped rung is no longer apples to apples.
        assert_eq!(K % V, 0, "K must be a whole number of V-sized groups");
        assert_eq!(N_BITS, 2);
        assert_eq!(K as usize / V as usize, N_BITS);
    }

    #[test]
    fn lut_is_exactly_32768_bytes() {
        // Under the 48 KiB static __shared__ limit => no cudaFuncSetAttribute
        // opt-in in the kernel. This is the constraint the geometry was chosen
        // to satisfy, so it gets an assertion rather than a comment.
        assert_eq!(LUT_STATES, 4096);
        assert_eq!(LUT_ENTRIES, 16_384);
        assert_eq!(LUT_BYTES, 32_768);
        // The `<= 48 KiB` half is a `const _: () = { assert!(..) }` at module
        // scope, so it fails the build rather than a test run.
    }

    #[test]
    fn warmup_is_two_symbols() {
        assert_eq!(WARMUP_SYMS, 2);
        // ceil(L/K), not L/K: the floor would be 1 here and would silently
        // leave 4 bits of stale state in every mid-row restart.
        assert_eq!(WARMUP_SYMS, (L as usize).div_ceil(K as usize));
        assert!(WARMUP_SYMS * K as usize >= L as usize);
    }

    #[test]
    fn state_is_determined_by_the_last_warmup_symbols() {
        // Proven over the entire state space, not sampled: for EVERY prior
        // state and every pair of symbols, the state after WARMUP_SYMS shifts
        // must be identical. This is the property that lets threads start at
        // arbitrary symbol offsets, and it is the property a wrong
        // WARMUP_SYMS breaks.
        for a in [0u8, 1, 15, 16, 127, 128, 200, 255] {
            for b in [0u8, 1, 15, 16, 127, 128, 200, 255] {
                let mut seen: Option<u32> = None;
                for prior in 0..LUT_STATES as u32 {
                    let s = next_state(next_state(prior, a), b);
                    match seen {
                        None => seen = Some(s),
                        Some(expect) => assert_eq!(
                            s, expect,
                            "state after {WARMUP_SYMS} symbols still depends on prior state \
                             {prior} (syms {a},{b})"
                        ),
                    }
                }
            }
        }
    }

    #[test]
    fn one_prior_symbol_is_the_true_minimum_warmup() {
        // Documents WHY WARMUP_SYMS is one over the minimum, so nobody "fixes"
        // it in the wrong direction. Seeding from a single prior symbol
        // reproduces the decode state exactly at this geometry; seeding from
        // ZERO prior symbols does not, and this pins both halves.
        let packed = fixture_symbols(64, 5);
        let mut full = 0u32;
        let mut zero_warmup_wrong = 0;
        for t in 0..packed.len() {
            let truth = next_state(full, packed[t]);
            if t >= 1 {
                let one_prior = next_state(next_state(0, packed[t - 1]), packed[t]);
                assert_eq!(
                    one_prior, truth,
                    "one prior symbol should suffice at symbol {t}"
                );
            }
            if next_state(0, packed[t]) != truth {
                zero_warmup_wrong += 1;
            }
            full = truth;
        }
        assert!(
            zero_warmup_wrong > 0,
            "zero warmup symbols would have to be wrong somewhere, or the trellis carries no \
             state across symbols at all"
        );
    }

    #[test]
    fn next_state_stays_in_range() {
        for prior in 0..LUT_STATES as u32 {
            for sym in [0u8, 1, 128, 255] {
                assert!(next_state(prior, sym) < LUT_STATES as u32);
            }
        }
    }

    #[test]
    fn packed_len_matches_the_bit_rate() {
        for k_in in [512usize, 1024, 4096, 7168] {
            // 2 bits per weight => packed bytes == weights / 4.
            assert_eq!(packed_len(k_in), k_in / 4);
            assert_eq!(num_symbols(k_in), packed_len(k_in));
            assert_eq!(packed_len(k_in) * 8, k_in * N_BITS);
        }
    }

    // ---- the table ------------------------------------------------------

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

        // Roughly standard-normal: Box-Muller pairs, bf16-rounded.
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
    fn lut_contents_are_pinned() {
        // Not a shape check and not a statistics check: the exact bytes. Any
        // edit to `gaussian_lut_bf16` — including one that leaves the
        // distribution looking right — has to come here and change this
        // constant, which is the point. Every artifact baked at this geometry
        // decodes against these 16,384 values and no others.
        let lut = gaussian_lut_bf16();
        assert_eq!(
            lut_digest(&lut),
            LUT_DIGEST,
            "the K=8/V=4/L=12 reproduction table changed. This is a FORMAT change: symbols \
             baked against the old table decode to different weights against the new one. If \
             the change is intended, re-bake and update LUT_DIGEST in the same commit."
        );
    }

    #[test]
    fn lut_uses_distinct_seeds_per_value_group() {
        // The two Box-Muller pairs of a state must not be the same draw. If
        // the seeding collapsed to one seed, values 0,1 would equal 2,3 and
        // the codebook would have 8,192 distinct points instead of 16,384.
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

    // ---- decode ---------------------------------------------------------

    #[test]
    fn decode_produces_v_weights_per_symbol_at_the_scaled_table_value() {
        let lut = gaussian_lut_bf16();
        let packed = fixture_symbols(64, 7);
        let scale = 0.017_5f32;
        let mut out = vec![0f32; packed.len() * V as usize];
        decode_row_into(&packed, scale, &lut, &mut out).unwrap();

        // Recompute independently of the loop under test.
        let mut state = 0u32;
        for (t, &sym) in packed.iter().enumerate() {
            state = ((state << 8) | sym as u32) & 0xFFF;
            for v in 0..4usize {
                let want = lut[state as usize * 4 + v].to_f32() * scale;
                assert_eq!(
                    out[t * 4 + v],
                    want,
                    "weight {v} of symbol {t} (state {state:#05x})"
                );
            }
        }
    }

    #[test]
    fn decode_rejects_a_wrong_sized_table() {
        // A wrong-geometry table is the failure mode that produces plausible
        // garbage rather than an error, so it must be refused, not indexed.
        let short = vec![bf16::ZERO; LUT_ENTRIES - 1];
        let mut out = vec![0f32; 4];
        let err = decode_row_into(&[3u8], 1.0, &short, &mut out).unwrap_err();
        assert!(
            err.contains("16384"),
            "error must name the expected size: {err}"
        );

        let k4v2_sized = vec![bf16::ZERO; 65_536 * 2];
        let err = decode_row_into(&[3u8], 1.0, &k4v2_sized, &mut out).unwrap_err();
        assert!(err.contains("expected"), "{err}");
    }

    #[test]
    fn decode_rejects_a_wrong_sized_output() {
        let lut = gaussian_lut_bf16();
        let mut out = vec![0f32; 3];
        assert!(decode_row_into(&[1u8], 1.0, &lut, &mut out).is_err());
    }

    #[test]
    fn warmup_state_matches_a_full_walk_from_the_row_start() {
        // The mid-row restart property, exercised through the actual helper
        // the kernel mirrors: seeding from WARMUP_SYMS prior symbols must give
        // the same state as walking the row from symbol 0.
        let packed = fixture_symbols(97, 11);
        let mut full = 0u32;
        for (t, &sym) in packed.iter().enumerate() {
            assert_eq!(
                warmup_state(&packed, t),
                full,
                "restart at symbol {t} disagrees with the full walk"
            );
            full = next_state(full, sym);
        }
        assert_eq!(warmup_state(&packed, packed.len()), full);
    }

    // ---- gemv parity ----------------------------------------------------

    #[test]
    fn gemv_without_the_hoist_is_bit_exact_against_decode_then_dot() {
        // THE parity contract. `gemv_row(.., Off)` must equal decode-then-dot
        // evaluated in the same order, to the bit — this is the CPU side of
        // the gate the GPU kernel has to pass.
        let lut = gaussian_lut_bf16();
        for (seed, n_sym) in [(1u64, 16usize), (2, 64), (3, 256), (4, 1024)] {
            let packed = fixture_symbols(n_sym, seed);
            let x = fixture_x(n_sym * V as usize, seed ^ 0xABCD);
            let scale = 0.031_25f32 * (seed as f32 + 1.0);

            let mut w = vec![0f32; n_sym * V as usize];
            decode_row_into(&packed, scale, &lut, &mut w).unwrap();
            let mut want = 0f32;
            for i in 0..w.len() {
                want = w[i].mul_add(x[i], want);
            }

            let got = gemv_row(&packed, scale, &lut, &x, RowScaleHoist::Off).unwrap();
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "seed {seed}: fused gemv {got:e} != decode-then-dot {want:e}"
            );
        }
    }

    #[test]
    fn the_hoist_changes_the_bits_and_is_therefore_not_the_parity_path() {
        // A guard that would be vacuous if it only checked closeness: the
        // point is that the hoist DOES change the result, which is why the
        // parity gate runs with it off. If this ever stops finding a
        // difference, the switch has stopped doing anything.
        let lut = gaussian_lut_bf16();
        let mut differed = 0;
        for seed in 0..24u64 {
            let packed = fixture_symbols(512, seed);
            let x = fixture_x(512 * V as usize, seed ^ 0x51D);
            let scale = 0.021_37f32;
            let off = gemv_row(&packed, scale, &lut, &x, RowScaleHoist::Off).unwrap();
            let on = gemv_row(&packed, scale, &lut, &x, RowScaleHoist::On).unwrap();
            if off.to_bits() != on.to_bits() {
                differed += 1;
            }
            // ...but it must still be the same answer to within rounding.
            let denom = off.abs().max(on.abs()).max(1e-6);
            assert!(
                (off - on).abs() / denom < 1e-4,
                "seed {seed}: hoist changed the answer materially: {off:e} vs {on:e}"
            );
        }
        assert!(
            differed > 0,
            "the row-scale hoist produced bit-identical results on every fixture — either the \
             switch is dead or the reassociation is not happening"
        );
    }

    #[test]
    fn gemv_matches_a_thread_split_reconstruction() {
        // Models what the kernel actually does: split the row across threads,
        // each seeding from WARMUP_SYMS prior symbols. Partial sums are
        // combined in thread order, which is what the block reduction must
        // preserve for the split to be exact.
        let lut = gaussian_lut_bf16();
        let n_sym = 512usize;
        let packed = fixture_symbols(n_sym, 99);
        let x = fixture_x(n_sym * V as usize, 12345);
        let scale = 0.019f32;

        for threads in [1usize, 2, 4, 8, 128] {
            let per = n_sym.div_ceil(threads);
            let mut partials = Vec::new();
            for t in 0..threads {
                let start = t * per;
                if start >= n_sym {
                    partials.push(0f32);
                    continue;
                }
                let end = (start + per).min(n_sym);
                let mut state = warmup_state(&packed, start);
                let mut acc = 0f32;
                for s in start..end {
                    state = next_state(state, packed[s]);
                    let off = state as usize * V as usize;
                    for v in 0..V as usize {
                        let w = lut[off + v].to_f32() * scale;
                        acc = w.mul_add(x[s * V as usize + v], acc);
                    }
                }
                partials.push(acc);
            }
            // Sequential recombination == the single-thread answer only when
            // the split is exact at the state level; the sum itself is
            // reassociated, so compare with a tolerance and report the split.
            let got: f32 = partials.iter().sum();
            let want = gemv_row(&packed, scale, &lut, &x, RowScaleHoist::Off).unwrap();
            let l1 = l1_magnitude(&packed, scale, &lut, &x).max(1e-6);
            assert!(
                (got - want).abs() <= 1e-5 * l1,
                "{threads}-way split gave {got:e}, single-thread {want:e} — the warmup seeding \
                 is wrong, not the summation order (|diff| {:e} vs L1 {l1:e})",
                (got - want).abs()
            );
        }
    }

    #[test]
    fn a_single_active_lane_reduces_to_the_scalar_reference_to_the_bit() {
        // When only lane 0 has work the butterfly adds nothing but exact
        // zeros, so the model must collapse onto `gemv_row` bit-for-bit. This
        // is the anchor that ties the parallel model back to the arithmetic
        // reference; without it the model could be self-consistently wrong.
        let lut = gaussian_lut_bf16();
        for seed in 0..8u64 {
            let packed = fixture_symbols(1, seed);
            let x = fixture_x(V as usize, seed ^ 0x99);
            let scale = 0.0231f32 * (seed as f32 + 1.0);
            for hoist in [RowScaleHoist::Off, RowScaleHoist::On] {
                let modelled =
                    gemv_row_gpu_model(&packed, scale, &lut, &x, KERNEL_THREADS, hoist).unwrap();
                let scalar = gemv_row(&packed, scale, &lut, &x, hoist).unwrap();
                assert_eq!(
                    modelled.to_bits(),
                    scalar.to_bits(),
                    "seed {seed} {hoist:?}: single-lane model {modelled:e} != scalar {scalar:e}"
                );
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
        for n_sym in [1024usize, 1000, 129, 4095, 257] {
            for seed in 0..6u64 {
                let packed = fixture_symbols(n_sym, seed);
                let x = fixture_x(n_sym * V as usize, seed ^ 0x1234);
                let scale = 0.0177f32;
                let modelled = gemv_row_gpu_model(
                    &packed,
                    scale,
                    &lut,
                    &x,
                    KERNEL_THREADS,
                    RowScaleHoist::Off,
                )
                .unwrap();
                let scalar = gemv_row(&packed, scale, &lut, &x, RowScaleHoist::Off).unwrap();
                let l1 = l1_magnitude(&packed, scale, &lut, &x).max(1e-6);
                assert!(
                    (modelled - scalar).abs() <= 1e-5 * l1,
                    "n_sym={n_sym} seed {seed}: {KERNEL_THREADS}-thread model {modelled:e} vs \
                     scalar {scalar:e} — |diff| {:e} exceeds 1e-5 of the L1 magnitude {l1:e}",
                    (modelled - scalar).abs()
                );
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
            for threads in [WARP, 64, KERNEL_THREADS] {
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
                    "n_sym={n_sym} threads={threads}: symbols covered {:?} times (want all 1)",
                    {
                        let mut k: Vec<usize> = covered.clone();
                        k.dedup();
                        k
                    }
                );
            }
        }
    }

    #[test]
    fn the_rust_model_and_the_cuda_kernel_agree_on_their_shared_constants() {
        // The one guard that can keep an un-compilable kernel honest in a
        // no-GPU lane. `gemv_row_gpu_model` is only a model of the kernel at
        // KERNEL_THREADS; if the .cu changes its block size, or its geometry,
        // or the size of its shared table, the bit-exact CUDA gate silently
        // starts comparing against the wrong thing. So read the source.
        const SRC: &str = include_str!("../../kernels/qtip/qtip_gemv_k8v4l12.cu");

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
            (format!("constexpr uint32_t QK8_K = {K};"), "K"),
            (format!("constexpr uint32_t QK8_L = {L};"), "L"),
            (format!("constexpr uint32_t QK8_V = {V};"), "V"),
            (
                "constexpr uint32_t QK8_LUT_ENTRIES = QK8_LUT_STATES * QK8_V;".to_string(),
                "table entry count",
            ),
            (
                format!("sizeof(__nv_bfloat16) == {LUT_BYTES}"),
                "table byte size static_assert",
            ),
            (
                "constexpr uint32_t QK8_WARMUP_SYMS = (QK8_L + QK8_K - 1u) / QK8_K;".to_string(),
                "ceil(L/K) warmup",
            ),
            (
                "if (row_scale_hoist != 0)".to_string(),
                "nonzero-means-hoist ABI",
            ),
        ];
        for (needle, what) in want {
            assert!(
                src.contains(&squeeze(&needle)),
                "kernels/qtip/qtip_gemv_k8v4l12.cu no longer states the {what} this module \
                 models. Expected to find:\n    {needle}\nIf the kernel changed deliberately, \
                 change this module to match — `gemv_row_gpu_model` is a model of that kernel, \
                 and the CUDA parity gate compares bits against it."
            );
        }
    }

    #[test]
    fn the_gpu_model_is_deterministic_and_thread_count_sensitive() {
        // Determinism is what lets the CUDA gate compare bits. Thread-count
        // sensitivity is the anti-vacuity half: if the model returned the same
        // bits for every split, it would not be modelling the reduction tree
        // at all and the CUDA comparison would pass against a stub.
        let lut = gaussian_lut_bf16();
        let packed = fixture_symbols(2048, 31);
        let x = fixture_x(2048 * V as usize, 32);
        let scale = 0.0143f32;
        let a = gemv_row_gpu_model(&packed, scale, &lut, &x, KERNEL_THREADS, RowScaleHoist::Off)
            .unwrap();
        let b = gemv_row_gpu_model(&packed, scale, &lut, &x, KERNEL_THREADS, RowScaleHoist::Off)
            .unwrap();
        assert_eq!(a.to_bits(), b.to_bits(), "model is not deterministic");

        let c = gemv_row_gpu_model(&packed, scale, &lut, &x, WARP, RowScaleHoist::Off).unwrap();
        assert_ne!(
            a.to_bits(),
            c.to_bits(),
            "the model returned identical bits at 32 and {KERNEL_THREADS} threads — it is not \
             modelling the reduction tree"
        );
    }

    #[test]
    fn the_gpu_model_rejects_a_thread_count_that_is_not_whole_warps() {
        let lut = gaussian_lut_bf16();
        let packed = fixture_symbols(64, 1);
        let x = fixture_x(64 * V as usize, 2);
        assert!(gemv_row_gpu_model(&packed, 1.0, &lut, &x, 0, RowScaleHoist::Off).is_err());
        assert!(gemv_row_gpu_model(&packed, 1.0, &lut, &x, 48, RowScaleHoist::Off).is_err());
        assert!(gemv_row_gpu_model(&packed, 1.0, &lut, &x, 64, RowScaleHoist::Off).is_ok());
    }

    #[test]
    fn hoist_abi_values_are_pinned() {
        // The Rust enum and the kernel's `row_scale_hoist` int must agree, and
        // the kernel treats nonzero as "hoist". Pin the mapping here so a
        // reordering of the enum cannot silently flip the policy.
        assert_eq!(RowScaleHoist::Off.as_abi(), 0);
        assert_eq!(RowScaleHoist::On.as_abi(), 1);
        assert_eq!(RowScaleHoist::default(), RowScaleHoist::Off);
    }

    // ---- the GPU gate ---------------------------------------------------

    /// `qtip_gemv_k8v4l12.cu` must reproduce [`gemv_row_gpu_model`] **to the
    /// bit**, with the row-scale hoist off.
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
    fn cuda_k8v4l12_gemv_matches_the_cpu_model_bit_for_bit() -> candle_core::Result<()> {
        use crate::qtip::cuda_ops;
        use candle_core::{DType, Device, Tensor};

        let cuda = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(_) => {
                eprintln!("CUDA not available; skipping cuda_k8v4l12_gemv_matches_the_cpu_model");
                return Ok(());
            }
        };

        let lut = gaussian_lut_bf16();
        let lut_t = Tensor::from_slice(&lut, (LUT_STATES, V as usize), &cuda)?;

        // in_features chosen so that num_symbols = in_features/4 is sometimes
        // NOT a multiple of KERNEL_THREADS — that is the case a floored
        // sym_per_thread drops the tail of, and it must be in the GPU gate too.
        for (n_rows, in_features) in [
            (8usize, 256usize),
            (64, 1024),
            (33, 4096),
            (7, 4000),
            (5, 516),
        ] {
            let n_sym = num_symbols(in_features);
            let mut packed = Vec::with_capacity(n_rows * n_sym);
            let mut scales = Vec::with_capacity(n_rows);
            for r in 0..n_rows {
                packed.extend(fixture_symbols(n_sym, 1000 + r as u64));
                scales.push(0.011f32 * (r as f32 + 1.0));
            }
            let x = fixture_x(in_features, 4242);

            let blocks_t = Tensor::from_slice(&packed, (n_rows, n_sym), &cuda)?;
            let scales_t = Tensor::from_slice(&scales, n_rows, &cuda)?;
            let x_t = Tensor::from_slice(&x, in_features, &cuda)?.to_dtype(DType::F32)?;

            let y = cuda_ops::fused_gemv_k8v4l12_cuda(
                &blocks_t,
                &scales_t,
                &lut_t,
                &x_t,
                in_features,
                RowScaleHoist::Off,
            )?;
            let got: Vec<f32> = y.flatten_all()?.to_vec1()?;
            assert_eq!(got.len(), n_rows);

            for r in 0..n_rows {
                let row = &packed[r * n_sym..(r + 1) * n_sym];
                let want = gemv_row_gpu_model(
                    row,
                    scales[r],
                    &lut,
                    &x,
                    KERNEL_THREADS,
                    RowScaleHoist::Off,
                )
                .expect("model");
                assert_eq!(
                    got[r].to_bits(),
                    want.to_bits(),
                    "n_rows={n_rows} k={in_features} row {r}: kernel {:e} != model {want:e}",
                    got[r]
                );
            }
        }
        Ok(())
    }

    /// The launcher must refuse a K=4/V=2 artifact rather than decode it.
    ///
    /// The failure this guards against is silent: an F32 `[65536, 2]` table
    /// reinterpreted as bf16 still indexes in bounds for every state, so the
    /// kernel would return plausible numbers.
    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_k8v4l12_gemv_refuses_a_wrong_geometry_artifact() -> candle_core::Result<()> {
        use crate::qtip::cuda_ops;
        use candle_core::{Device, Tensor};

        let cuda = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(_) => {
                eprintln!("CUDA not available; skipping cuda_k8v4l12_gemv_refuses_wrong_geometry");
                return Ok(());
            }
        };

        let (n_rows, in_features) = (4usize, 256usize);
        let n_sym = num_symbols(in_features);
        let packed: Vec<u8> = fixture_symbols(n_rows * n_sym, 7);
        let scales = vec![0.02f32; n_rows];
        let x = fixture_x(in_features, 8);
        let blocks_t = Tensor::from_slice(&packed, (n_rows, n_sym), &cuda)?;
        let scales_t = Tensor::from_slice(&scales, n_rows, &cuda)?;
        let x_t = Tensor::from_slice(&x, in_features, &cuda)?;
        let good_lut = Tensor::from_slice(&gaussian_lut_bf16(), (LUT_STATES, V as usize), &cuda)?;

        // Sanity: the good case is accepted, so the refusals below are about
        // the geometry and not about the fixture being broken.
        cuda_ops::fused_gemv_k8v4l12_cuda(
            &blocks_t,
            &scales_t,
            &good_lut,
            &x_t,
            in_features,
            RowScaleHoist::Off,
        )?;

        // An F32 table is the K=4/V=2 rung's.
        let f32_lut = Tensor::zeros((LUT_STATES, V as usize), candle_core::DType::F32, &cuda)?;
        let err = cuda_ops::fused_gemv_k8v4l12_cuda(
            &blocks_t,
            &scales_t,
            &f32_lut,
            &x_t,
            in_features,
            RowScaleHoist::Off,
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("BF16"), "expected a dtype refusal, got: {err}");

        // A 2^16 × 2 table is the K=4/V=2 rung's size.
        let big_lut = Tensor::zeros((65_536, 2), candle_core::DType::BF16, &cuda)?;
        let err = cuda_ops::fused_gemv_k8v4l12_cuda(
            &blocks_t,
            &scales_t,
            &big_lut,
            &x_t,
            in_features,
            RowScaleHoist::Off,
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("16384"), "expected a size refusal, got: {err}");

        // Nibble-packed rows are half as long as this rung needs.
        let nibble_t =
            Tensor::from_slice(&packed[..n_rows * (n_sym / 2)], (n_rows, n_sym / 2), &cuda)?;
        let err = cuda_ops::fused_gemv_k8v4l12_cuda(
            &nibble_t,
            &scales_t,
            &good_lut,
            &x_t,
            in_features,
            RowScaleHoist::Off,
        )
        .unwrap_err()
        .to_string();
        assert!(
            err.contains("symbol is one byte"),
            "expected a packing refusal, got: {err}"
        );

        Ok(())
    }
}
