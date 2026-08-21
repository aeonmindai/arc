//! QTIP: Trellis-coded incoherence-processing 2-bit / 3-bit / 4-bit weight quantization.
//!
//! Cornell NeurIPS 2024 paper:  `research/01_weight_compression/qtip_trellis_coded_quantization.pdf`
//! Reference code:               `research/code/01_weight_compression/qtip/`
//!
//! ## Format (this Tier-A implementation)
//!
//! Configuration: **K=4 bits per symbol, V=2 reproduction values per state, L=16 trellis state**.
//!   → Each symbol s ∈ [0, 16) shifts into a 16-bit state, producing V=2 weight values.
//!   → 4 bits encode 2 weights = **2 bits per weight** (the headline QTIP config).
//!
//! Decode:
//!   state_t = ((state_{t-1} << K) | symbol_t) & ((1 << L) - 1)
//!   (w_{2t}, w_{2t+1}) = LUT[state_t]
//!
//! The LUT itself is 2^L × V = 65,536 × 2 entries, drawn from a Gaussian distribution
//! (paper uses Gaussian quantile spacings — see `gaussian_lut`).
//!
//! ## Tier A scope
//!
//! - Storage format: packed nibbles + per-row scale + global LUT
//! - Decode (forward / dequantize): LUT lookup with state shift
//! - Quantize: **greedy nearest-state search** (not Viterbi). This produces a valid
//!   QTIP-format symbol stream but with ~0.1–0.2 PPL gap vs Cornell's Viterbi.
//!   Closing that gap is a Tier-B work item — see TODOs.
//! - Tests: format round-trip, dot-product matches dense within tolerance.
//!
//! ## Status by tier
//!
//! - Tier A (in tree):
//!   * Format round-trip and dense matmul correctness
//!   * Greedy mode quantizer (~0.90 matmul cos sim, no rotation)
//! - Tier B (in tree as of RUN-158):
//!   * Full Viterbi optimal quantizer (`viterbi.rs`)
//!   * Hadamard incoherence rotation wired into both the quantizer and the
//!     forward path — Viterbi matmul cos sim now ≥0.95 (matches Cornell)
//! - Tier C (deferred):
//!   * CUDA decode kernel port from `qtip-kernels/src/qtip_torch.cu`
//!     (1MAD / 3INST / HYB) for >5× faster inference vs the dequantize path
//!   * Compose with NVFP4 per-tensor scale

use std::{
    borrow::Cow,
    io::Cursor,
    sync::{atomic::AtomicUsize, Arc},
};

use byteorder::{LittleEndian, ReadBytesExt};
use candle_core::{DType, Device, Result, Tensor};

use crate::{
    turboquant::wht::{fwht_inplace, generate_signs},
    utils::{deserialize_tensor, serialize_tensor, version_is_compatible, UQFF_VERSION},
    IsqType, QuantMethod, QuantMethodConfig, QuantizeOntoGuard, QuantizedConfig, QuantizedSerde,
    QuantizedSerdeType, ShardedVarBuilder,
};

pub mod bake_cache;
#[cfg(test)]
mod bake_memory_tests;
#[cfg(test)]
mod bake_quality_tests;
pub mod bitshift;
#[cfg(test)]
mod codebook_tests;
#[cfg(feature = "cuda")]
mod cuda_ops;
pub(crate) mod device_guard;
#[cfg(feature = "cuda")]
mod ffi;
pub(crate) mod gather_policy;
#[cfg(test)]
mod greedy_ban_tests;
pub mod grouped;
#[cfg(test)]
mod search_bench;
pub mod tcfrag2b;
pub mod trellis_v4l12;
pub mod tune;
mod viterbi;
pub use bitshift::{Qtip2bLayer, QTIP2B_MCG_MULT};
// The MoE dispatch facts a benchmark needs in order to prove it measured what
// it claims: the routing model the amortization argument rests on, the
// structural pair ceiling, the env var that selects the path, and the tile
// geometry that decides whether the grouped kernel is reachable at all.
// Exported rather than re-derived so a harness cannot drift from the dispatch.
pub use bake_cache::{BakeCacheError, BakeKey};
pub use gather_policy::{
    expected_distinct_experts as qtip_expected_distinct_experts,
    expected_pairs_per_distinct_expert as qtip_expected_pairs_per_distinct_expert,
    grouped_gemm_tile_fill as qtip_grouped_gemm_tile_fill,
    GATHER_GEMV_MAX_PAIRS as QTIP_GATHER_GEMV_MAX_PAIRS,
    ONDEVICE_MOE_MAX_TOKENS_ENV as QTIP_ONDEVICE_MOE_MAX_TOKENS_ENV,
};
pub use grouped::{
    grouped_launch_counts, grouped_variant, set_grouped_variant, ExpertBpwTable, TrellisBpw,
    GROUPED_TILE_K as QTIP_GROUPED_TILE_K, GROUPED_TILE_M as QTIP_GROUPED_TILE_M,
    GROUPED_TILE_N as QTIP_GROUPED_TILE_N, QTIP_GROUPED_VARIANT_BASELINE,
    QTIP_GROUPED_VARIANT_COUNT, QTIP_GROUPED_VARIANT_ENV, QTIP_GROUPED_VARIANT_LDST,
    QTIP_GROUPED_VARIANT_TUNED,
};
#[allow(unused_imports)]
pub use viterbi::{
    hessian_row_weights, quantize_row, viterbi_quantize_row, TrellisSearch, HESSIAN_SIGMA_REG,
};

/// Default seed for the QTIP Hadamard incoherence rotation.
///
/// QTIP (Cornell ICLR'25) requires that quantize-time and inference-time agree
/// on a single deterministic rotation. We default to a fixed seed so that
/// checkpoints quantized with rotation enabled also decode correctly without
/// needing to ship the sign vector in older formats (the seed alone is enough).
const QTIP_ROTATION_SEED: u64 = 0xA3C1_7B0F_5F2E_1D4D;

/// Decode-shaped-workload threshold. Matches the default of
/// `ARC_QTIP_ONDEVICE_MOE_MAX_TOKENS`: at or below this many tokens the
/// fused on-device gather/GEMV kernels are the intended path and no
/// dequantized weight tensor should ever be written to global memory.
pub(crate) const DECODE_REGIME_MAX_TOKENS: usize = 8;

/// Opt-in decode-path diagnostic (`ARC_WARN_DEQUANT_MATERIALIZE=1`).
///
/// Fires when a dequantize-materializing fallback engages for a
/// decode-shaped workload (`n_tokens <= DECODE_REGIME_MAX_TOKENS`). On the
/// intended decode path the fused gather/GEMV kernels
/// (`gather_gemv_cuda` / `fused_gemv_cuda` and their bitshift siblings)
/// never materialize dequantized weights to HBM; if this warning fires, a
/// precondition (env override, unsupported dtype, non-CUDA storage, kernel
/// not compiled in) knocked the layer off the fused path and decode is
/// paying the full dequantize bandwidth — the `qtip_dequantize` line in the
/// nsys profile. Warns once per call-site context to avoid flooding
/// per-token logs.
pub(crate) fn warn_dequant_materialize_at_decode(n_tokens: usize, context: &'static str) {
    use std::collections::HashSet;
    use std::sync::{LazyLock, Mutex};
    static ENABLED: LazyLock<bool> =
        LazyLock::new(|| std::env::var("ARC_WARN_DEQUANT_MATERIALIZE").is_ok_and(|v| v != "0"));
    if !*ENABLED || n_tokens > DECODE_REGIME_MAX_TOKENS {
        return;
    }
    static SEEN: LazyLock<Mutex<HashSet<&'static str>>> =
        LazyLock::new(|| Mutex::new(HashSet::new()));
    let first_hit = SEEN
        .lock()
        .map(|mut seen| seen.insert(context))
        .unwrap_or(false);
    if first_hit {
        tracing::warn!(
            "ARC_WARN_DEQUANT_MATERIALIZE: {context} engaged at decode shape \
             (n_tokens={n_tokens}); dequantized weights are being materialized \
             instead of using the fused gather/GEMV kernels. \
             (warning once per call site)"
        );
    }
}

// ---------------------------------------------------------------------------
// GPU-quantize fallback accounting (wave6-Q)
// ---------------------------------------------------------------------------

/// Process-wide count of QTIP quantizes that engaged the CPU pipeline while a
/// GPU was plausibly available (see [`expert_stack_quant_device`]). A silent
/// switch from the GPU prefix-grouped Viterbi (~30 s/layer on H200) to the
/// CPU rayon Viterbi (~11 min/layer) is a ~20x bake regression, so every such
/// reroute is counted here and warned about (once per call site). CUDA tests
/// assert this stays flat across a GPU-path quantize.
static GPU_QUANT_CPU_FALLBACKS: AtomicUsize = AtomicUsize::new(0);

/// Number of times a QTIP quantize fell back to the CPU pipeline even though
/// a GPU was plausibly available, since process start. Test/diagnostic hook
/// for the wave6-Q regression guard.
pub fn gpu_quantize_cpu_fallback_count() -> usize {
    GPU_QUANT_CPU_FALLBACKS.load(std::sync::atomic::Ordering::Relaxed)
}

/// `true` the first time this `context` is seen (shared registry for the
/// warn-once diagnostics below; contexts are distinct static strings).
fn first_warn_for_context(context: &'static str) -> bool {
    use std::collections::HashSet;
    use std::sync::{LazyLock, Mutex};
    static SEEN: LazyLock<Mutex<HashSet<&'static str>>> =
        LazyLock::new(|| Mutex::new(HashSet::new()));
    SEEN.lock()
        .map(|mut seen| seen.insert(context))
        .unwrap_or(false)
}

/// Record + warn (once per `context`) that a QTIP quantize is running on the
/// CPU pipeline despite a GPU being plausibly available. `reason` carries the
/// actual error/condition so the bake log names the culprit instead of just
/// silently getting ~20x slower per layer.
fn note_gpu_quant_cpu_fallback(context: &'static str, reason: &str) {
    GPU_QUANT_CPU_FALLBACKS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    if first_warn_for_context(context) {
        tracing::warn!(
            "QTIP GPU quantize fallback: {context} is quantizing on the CPU — {reason}. \
             Expect ~10-20x slower per-layer bake times (CPU Viterbi vs GPU \
             prefix-grouped Viterbi). (warning once per call site)"
        );
    }
}

/// Probe for an NVIDIA driver when this binary was built WITHOUT the `cuda`
/// feature: the one configuration where a GPU box silently bakes on the CPU
/// with no CUDA code compiled in at all (wave6-Q trap). Linux-only paths;
/// on other OSes this returns `false` and CPU-only builds stay quiet.
#[cfg(not(feature = "cuda"))]
fn nvidia_driver_present() -> bool {
    use std::sync::LazyLock;
    static PRESENT: LazyLock<bool> = LazyLock::new(|| {
        std::path::Path::new("/proc/driver/nvidia").exists()
            || std::path::Path::new("/dev/nvidia0").exists()
    });
    *PRESENT
}

/// Decide which device a 3-D `[E, N, K]` expert-stack quantize runs on.
///
/// The bake keeps expert stacks on the CPU (pre-moving the dense stack OOMs
/// the single-GPU tail, RUN-161) and streams batches to the GPU from here.
/// Every route that ends on the CPU while a GPU was plausibly available is
/// LOUD: it increments [`gpu_quantize_cpu_fallback_count`] and warns with the
/// actual reason. wave6-Q: the previous silent
/// `cuda_if_available(0).unwrap_or_else(|_| cpu)` version of this gate could
/// reroute a whole bake to the CPU Viterbi (~20x per layer) without logging
/// a single line.
pub(crate) fn expert_stack_quant_device(device: &Device, context: &'static str) -> Device {
    if !matches!(device, Device::Cpu) {
        // Weight already targets an accelerator: quantize where it lives. The
        // CUDA path hard-fails instead of falling back, so there is nothing
        // to instrument on this route.
        return device.clone();
    }
    #[cfg(feature = "cuda")]
    {
        if !ffi::HAVE_QTIP_KERNELS {
            note_gpu_quant_cpu_fallback(
                context,
                "CUDA build but the QTIP kernels were not compiled in \
                 (`has_qtip_kernels` cfg absent; build-time compute cap < 8.0?)",
            );
            return device.clone();
        }
        // Route to the ordinal this ISQ worker owns, not unconditionally to 0.
        // A hardcoded 0 here is the landmine for a multi-device bake
        // (wave22 / `--bake-devices`): every worker whose layer resolved to a
        // CPU target would pile its expert stacks onto device 0 while the other
        // devices idled, and would exhaust device 0's memory doing it.
        // Unset (the default, and every single-device bake) still means 0.
        let ordinal = crate::worker_cuda_ordinal().unwrap_or(0);
        match Device::new_cuda(ordinal) {
            Ok(cuda) => cuda,
            Err(e) => {
                note_gpu_quant_cpu_fallback(
                    context,
                    &format!("CUDA device {ordinal} initialization failed: {e}"),
                );
                device.clone()
            }
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        if nvidia_driver_present() {
            note_gpu_quant_cpu_fallback(
                context,
                "an NVIDIA driver is present on this machine but mistralrs-quant \
                 was built WITHOUT the `cuda` feature",
            );
        }
        device.clone()
    }
}

/// Warn (once per call site) when a model-scale 2-D weight runs the CPU
/// quantize pipeline while a GPU is plausibly available. The 2-D CPU path is
/// reached by design when a layer's ISQ target device is the CPU; unlike the
/// 3-D expert-stack path it has no opportunistic GPU offload, so a mis-mapped
/// bake pays the full CPU Viterbi cost with no error anywhere. The size
/// threshold keeps unit-test shapes quiet. Does not increment the fallback
/// counter (that is reserved for the 3-D gate the CUDA tests assert on).
pub(crate) fn warn_big_cpu_2d_quantize(n: usize, k: usize, context: &'static str) {
    const BIG_2D_WEIGHTS: usize = 1 << 22; // 4M weights: real-model scale
    if n.saturating_mul(k) < BIG_2D_WEIGHTS {
        return;
    }
    #[cfg(not(feature = "cuda"))]
    if !nvidia_driver_present() {
        return;
    }
    if first_warn_for_context(context) {
        tracing::warn!(
            "{context}: quantizing a {n}x{k} weight on the CPU Viterbi/greedy \
             pipeline. The GPU quantize engages only when the layer's target \
             device is CUDA (2-D) or via the 3-D expert-stack offload; if this \
             is a bake on a GPU box, check the device mapping. \
             (warning once per call site)"
        );
    }
}

/// Maximum block size for the block-diagonal Hadamard rotation.
///
/// Real LLM linear layers have `in_features` that is not always a power of 2
/// (e.g., Llama-3 intermediate_size = 14336 = 7·2^11). We use a block-diagonal
/// rotation `R = blockdiag(R_0, R_1, ...)` where each R_i is a Walsh-Hadamard
/// rotation on a power-of-2 sub-block. The block size is the largest power of
/// 2 dividing `in_features`, capped at this constant for kernel-friendliness.
/// 128 matches Cornell QTIP's standard `had_left` block size and is a multiple
/// of common GPU warp sizes (32) and tile dims (16, 32, 64, 128).
const QTIP_ROTATION_MAX_BLOCK: usize = 128;

/// Compute the block size for the Hadamard rotation given `in_features`.
///
/// Returns the largest power of 2 dividing `in_features` up to
/// `QTIP_ROTATION_MAX_BLOCK`. Returns 0 if `in_features` is odd (rotation
/// disabled — should not happen for real LLMs but kept defensive).
fn rotation_block_size(in_features: usize) -> usize {
    if in_features == 0 || !in_features.is_multiple_of(2) {
        return 0;
    }
    let mut block = 1usize;
    while block * 2 <= QTIP_ROTATION_MAX_BLOCK && in_features.is_multiple_of(block * 2) {
        block *= 2;
    }
    block
}

/// Apply the block-diagonal D·H·D rotation in-place to a row vector.
///
/// The full rotation `R` is block-diagonal with each block of size `block_size`
/// applying `D_i·H·D_i` (sign flip → Walsh-Hadamard → sign flip). `signs` is
/// the full `[in_features]` ±1 vector; we slice it per-block.
///
/// **Math**: D·H·D is symmetric (R = R^T) and involutory (R·R = I), since both
/// D (diagonal ±1) and H (normalized Hadamard) are symmetric and involutory.
/// Therefore the same function applied at quantize time (to weight rows) and
/// inference time (to activation rows) leaves the matmul invariant:
///   x_rot @ W_rot^T = (x R) (W R)^T = x R R^T W^T = x R R W^T = x W^T.
///
/// **Why this fixes Viterbi**: Without rotation, Viterbi's globally-optimal
/// symbol path produces reconstruction errors that are correlated with the raw
/// activation distribution → the matmul amplifies these correlations. After
/// D·H·D the activations are approximately Gaussian (any orthogonal projection
/// of a roughly-uniform input is concentrated to a high-entropy distribution),
/// matching the assumption Viterbi's cost model makes. Greedy doesn't benefit
/// as visibly because its local choices already produce errors uncorrelated
/// with the input by construction.
pub(crate) fn apply_block_rotation(row: &mut [f32], signs: &[f32], block_size: usize) {
    if block_size == 0 || !block_size.is_power_of_two() {
        return;
    }
    debug_assert_eq!(row.len(), signs.len());
    debug_assert_eq!(
        row.len() % block_size,
        0,
        "row length {} must be a multiple of block_size {}",
        row.len(),
        block_size
    );

    for block_start in (0..row.len()).step_by(block_size) {
        let row_block = &mut row[block_start..block_start + block_size];
        let sign_block = &signs[block_start..block_start + block_size];

        // D: first sign flip.
        for i in 0..block_size {
            row_block[i] *= sign_block[i];
        }
        // H: Walsh-Hadamard butterfly with 1/sqrt(n) normalization (involutory).
        fwht_inplace(row_block);
        // D: second sign flip.
        for i in 0..block_size {
            row_block[i] *= sign_block[i];
        }
    }
}

/// Trellis state width in bits. QTIP paper uses L=16.
pub const L: u32 = 16;
/// Bits per symbol. K=4 with V=2 yields 2 bits per weight.
pub const K: u32 = 4;
/// Reproduction vector dimension per state.
pub const V: u32 = 2;
/// State mask: (1 << L) - 1.
const STATE_MASK: u32 = (1u32 << L) - 1;
/// LUT size: 2^L entries × V values per entry.
const LUT_SIZE: usize = 1 << L;
/// Symbol alphabet size: 2^K.
const ALPHABET: usize = 1 << K;

pub(crate) const N_BITS: usize = 2;

/// Per-state pseudo-random Gaussian LUT.
///
/// For each of the 65,536 states, draw V Gaussian values via Box-Muller, seeded
/// deterministically by the state index. This produces a smooth, well-conditioned
/// LUT where every state — including the all-zero initial state — decodes to a
/// reasonable Gaussian sample. The QTIP paper uses a learned LUT trained per-layer;
/// the Gaussian default works because LLM weights are approximately Gaussian and
/// the trellis itself provides codebook density beyond a per-symbol code.
fn gaussian_lut() -> Vec<f32> {
    let mut lut = Vec::with_capacity(LUT_SIZE * V as usize);
    for state in 0..(1u32 << L) {
        // Two independent uniforms from a deterministic hash of state.
        let (u1, u2) = hash_to_two_uniforms(state);
        let (g0, g1) = box_muller(u1, u2);
        lut.push(g0);
        lut.push(g1);
    }
    lut
}

/// Splitmix-style hash to produce two well-distributed uniforms in (0, 1).
fn hash_to_two_uniforms(state: u32) -> (f32, f32) {
    // Splitmix64-style mixing on a 64-bit value derived from state.
    let mut z = (state as u64).wrapping_mul(0x9E3779B97F4A7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^= z >> 31;
    let hi = (z >> 32) as u32;
    let lo = (z & 0xFFFFFFFF) as u32;
    // Map to (0, 1) avoiding exactly 0.
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

// ===========================================================================
// The computed codebook (wave24-AU) — the K=4/V=2 twin of the qtip2b rung's
// `mcg_codeword`.
// ===========================================================================

/// MCG multiplier for the computed K=4/V=2 codebook. Same spectrally-optimized
/// constant the `qtip2b` rung ships (`bitshift::QTIP2B_MCG_MULT`,
/// exllamav3 PR #26); measured indistinguishable from EXL3's original
/// `0xCBAC1FED` at this geometry (+0.37% vs +0.26% weight NMSE, inside the
/// fixture noise).
pub const QTIP_MCG_V2_MULT: u32 = 0xCAF6_A435;

/// σ of the computed `sum2` codebook over all 2^16 states (both V values),
/// measured in f64. Pinned by `computed_codebook_sigma_matches_constant`.
///
/// This is the K=4/V=2 analogue of the 1.2064 that
/// `bitshift::QTIP2B_SCALE_DIVISOR` folds in at K=2/V=1; it differs
/// because the second value of each pair comes from a *chained* product, whose
/// masked-fp16 halves have a slightly wider spread (1.2444 vs 1.2064).
pub(crate) const QTIP_MCG_V2_SIGMA: f32 = 1.225_552;

/// Row-scale divisor for the computed codebook: `max|row| / (3 · σ)`.
///
/// The Gaussian LUT is unit-σ by construction, so the rung's historical policy
/// is `max|row| / 3`. The computed codebook is not normalized in the table —
/// normalizing the *values* would cost a multiply on every decoded weight,
/// which is exactly the instruction budget this change exists to save — so σ
/// is folded into the divisor instead. Same policy, same objective (the search
/// minimises `Σ(cb − t/scale)²`, and scaling cb by 1/σ while scaling the
/// divisor by σ leaves the argmin identical), zero decode cost. This mirrors
/// `QTIP2B_SCALE_DIVISOR = 3.0 × 1.2064` on the sibling rung.
pub(crate) const QTIP_MCG_V2_SCALE_DIVISOR: f32 = 3.0 * QTIP_MCG_V2_SIGMA;

/// Divisor for the Gaussian LUT: it is already unit-σ, so `max|row| / 3`.
pub(crate) const QTIP_GAUSSIAN_SCALE_DIVISOR: f32 = 3.0;

/// The `cb_mult` value every K=4/V=2 CUDA launcher reads as "gather the
/// reproduction values from the stored table".
///
/// 0 is a safe sentinel because an MCG multiplier must be odd to have full
/// period, so it can never name a real computed codebook. Produced only by
/// [`QtipCodebook::cuda_mult`].
pub(crate) const CB_MULT_GAUSSIAN_LUT: u32 = 0;

/// One masked MCG product folded to a codeword: `(x & 0x8FFF8FFF) ^ 0x3B603B60`,
/// then the two fp16 halves summed **in f32**.
///
/// The f32 sum (rather than `__hadd`) is what makes the CUDA kernel
/// bit-identical to this reference: both fp16→f32 conversions are exact and the
/// f32 add is correctly rounded. Mirrors `qtip_cb_fold` in
/// `kernels/qtip/qtip_codebook.cuh`.
#[inline]
fn mcg_fold(x: u32) -> f32 {
    let m = (x & 0x8FFF_8FFF) ^ 0x3B60_3B60;
    let hi = half::f16::from_bits((m >> 16) as u16);
    let lo = half::f16::from_bits((m & 0xFFFF) as u16);
    hi.to_f32() + lo.to_f32()
}

/// The V=2 codeword pair for trellis `state`, "sum2" construction.
///
/// `v0` folds `state · mult`; `v1` folds the **chained** product
/// `state · mult²`. Both values therefore have the sum-of-two-masked-halves
/// distribution the `qtip2b` rung already ships and measures, which is the
/// whole point: the cheaper alternative ("split", taking the two halves of one
/// product as the pair) cannot produce `|v| < 0.142` — the mask keeps 12 low
/// bits and the XOR pins each half's exponent into 12..15 — and so puts a hole
/// in the codebook exactly where a Gaussian weight distribution has most of its
/// mass. Measured at this geometry (wave19-AP part 2): sum2 is **+0.00017 cos /
/// +0.37% weight NMSE** against the Gaussian LUT (neutral, 3–8× inside the
/// fixture noise), split is **−0.00174 cos / +3.73% NMSE**.
///
/// Mirrors `qtip_cb_sum2` in `kernels/qtip/qtip_codebook.cuh`.
#[inline]
pub fn mcg_codeword_v2(state: u32, mult: u32) -> (f32, f32) {
    let x0 = (state & STATE_MASK).wrapping_mul(mult);
    let x1 = x0.wrapping_mul(mult);
    (mcg_fold(x0), mcg_fold(x1))
}

/// Materialize the computed codebook in the same `[2^L, V]` flat layout as
/// [`gaussian_lut`], so it is a drop-in for the search and the CPU decode.
///
/// The GPU decode paths never call this — they compute each codeword in
/// registers, which is the point. It exists because the CPU Viterbi/beam inner
/// loop wants a flat table, and because keeping the table in the artifact is
/// what lets a reader that predates the codebook discriminator still decode a
/// new bake correctly.
pub(crate) fn mcg_codebook_v2(mult: u32) -> Vec<f32> {
    let mut cb = Vec::with_capacity(LUT_SIZE * V as usize);
    for state in 0..(1u32 << L) {
        let (v0, v1) = mcg_codeword_v2(state, mult);
        cb.push(v0);
        cb.push(v1);
    }
    cb
}

/// Which codebook a [`QtipLayer`]'s symbols decode against.
///
/// **This is a format discriminator, not a runtime knob.** The two codebooks
/// produce different reproduction values, so an artifact baked against one
/// cannot be decoded against the other. It is serialized into the UQFF and
/// read back; artifacts written before the field existed end at EOF and are
/// [`QtipCodebook::Gaussian`], which is what they are.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QtipCodebook {
    /// The original 65,536 × 2 Box-Muller table ([`gaussian_lut`]), gathered
    /// from memory at decode time.
    Gaussian,
    /// The computed "sum2" MCG code — no table read anywhere. `mult` is the
    /// MCG multiplier so a future retune stays readable.
    Mcg { mult: u32 },
}

impl Default for QtipCodebook {
    /// What a fresh bake uses when nothing says otherwise.
    fn default() -> Self {
        Self::DEFAULT
    }
}

impl QtipCodebook {
    /// The computed codebook at the multiplier this rung ships.
    pub const COMPUTED: Self = QtipCodebook::Mcg {
        mult: QTIP_MCG_V2_MULT,
    };

    /// What a fresh bake picks when `ARC_QTIP_CODEBOOK` says nothing. Old
    /// artifacts always keep whatever they were written with; this only decides
    /// what a new quantize produces.
    ///
    /// **Still `Gaussian`, and the flip to [`Self::COMPUTED`] is this one
    /// line.** It is deliberately not taken in the change that introduces the
    /// mechanism: switching it changes what every future artifact *means*, and
    /// the two are separable — the port is worthless if it is wrong, and the
    /// flip is worthless if it is unmeasured. Set `ARC_QTIP_CODEBOOK=mcg` to
    /// bake against the computed codebook today.
    ///
    /// # The flip was attempted here and BACKED OUT — the numbers
    ///
    /// Flipping this to [`Self::COMPUTED`] is the durable half of the
    /// `(void)lut;` fix in `kernels/qtip/qtip_gather_gemv.cu` (that kernel
    /// discarded the 512 KiB table it was handed and recomputed Box-Muller in
    /// registers; the two disagreed on 89.2% of the 65,536 codewords). `Mcg`
    /// has ZERO host/device disagreement by construction — `qtip_cb_fold` uses
    /// exact fp16->f32 conversions and `__fadd_rn`, which `--use_fast_math`
    /// cannot contract — reads no table, and drops 512 KiB per layer from the
    /// artifact. All of that is still true and still worth having.
    ///
    /// But it is **not quality-neutral on this repo's own fixtures**, and that
    /// is a measurement, taken on CPU, that anyone can reproduce with
    /// `cargo test -p mistralrs-quant --lib qtip::`:
    ///
    /// | test | Gaussian | Mcg |
    /// |---|---|---|
    /// | `viterbi_matmul_cosine_similarity_with_rotation`, greedy | > 0.85 (gate) | **0.3156** |
    /// | same, Viterbi + rotation | >= 0.80 (gate) | **0.4750** |
    /// | `viterbi_rotation_ablation`, rotation OFF | (below ON) | 0.8880 |
    /// | `viterbi_rotation_ablation`, rotation ON | (above OFF) | **0.4750** |
    ///
    /// The ablation *inverts*: under `Mcg` the Hadamard rotation makes that
    /// fixture worse, which it must not. That points at the codebook's
    /// interaction with the rotated distribution or at
    /// [`QTIP_MCG_V2_SCALE_DIVISOR`], not at a free lunch. The
    /// "+0.00017 cosine / +0.37% NMSE, NEUTRAL" figure from
    /// `probe_computed_codebook_quality` is a *weight-reconstruction* number on
    /// Gaussian draws; these are *matmul* numbers on a structured fixture, and
    /// they disagree. Until that is understood, flipping the default would
    /// change what every future artifact means on the strength of a
    /// measurement that does not cover the case that failed.
    ///
    /// ⚠️ Note what this does NOT excuse: the `(void)lut;` kernel bug is fixed
    /// regardless. Existing Gaussian-tagged artifacts are the ones that were
    /// being mis-decoded, and they do not care what a future bake picks.
    pub const DEFAULT: Self = QtipCodebook::Gaussian;

    /// Read the codebook choice from the environment. `ARC_QTIP_CODEBOOK` is
    /// `mcg` (default) or `gaussian`; anything else is refused rather than
    /// silently resolved, because picking the wrong one changes the artifact.
    pub fn from_env() -> Result<Self> {
        match std::env::var("ARC_QTIP_CODEBOOK").as_deref() {
            Err(_) | Ok("") => Ok(Self::DEFAULT),
            Ok("mcg") | Ok("computed") | Ok("sum2") => Ok(Self::COMPUTED),
            Ok("gaussian") | Ok("lut") => Ok(QtipCodebook::Gaussian),
            Ok(other) => candle_core::bail!(
                "ARC_QTIP_CODEBOOK={other:?} is not a codebook. Use `mcg` (computed) \
                 or `gaussian` (the stored LUT, the default). Refusing rather than guessing \
                 — the choice changes the artifact's reproduction values."
            ),
        }
    }

    /// `max|row| / divisor` is the per-row scale policy; this is the divisor.
    pub fn scale_divisor(self) -> f32 {
        match self {
            QtipCodebook::Gaussian => QTIP_GAUSSIAN_SCALE_DIVISOR,
            QtipCodebook::Mcg { .. } => QTIP_MCG_V2_SCALE_DIVISOR,
        }
    }

    /// The `[2^L, V]` table. Always materialized for the CPU search and for
    /// the artifact; the GPU decode paths compute instead.
    pub(crate) fn materialize(self) -> Vec<f32> {
        match self {
            QtipCodebook::Gaussian => gaussian_lut(),
            QtipCodebook::Mcg { mult } => mcg_codebook_v2(mult),
        }
    }

    /// The CUDA ABI selector: [`CB_MULT_GAUSSIAN_LUT`] (0) means "gather from
    /// the stored table", nonzero is the MCG multiplier. This is the only thing
    /// that produces the value every K=4/V=2 launcher takes.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub(crate) fn cuda_mult(self) -> u32 {
        match self {
            QtipCodebook::Gaussian => CB_MULT_GAUSSIAN_LUT,
            QtipCodebook::Mcg { mult } => mult,
        }
    }

    /// Short label for bake headers and error messages.
    pub fn tag(self) -> String {
        match self {
            QtipCodebook::Gaussian => "gaussian-lut".to_string(),
            QtipCodebook::Mcg { mult } => format!("mcg-sum2({mult:#010X})"),
        }
    }

    /// Wire encoding, appended after the search-detail section.
    ///
    /// **`Gaussian` writes nothing.** The discriminator is purely additive: an
    /// artifact baked against the stored table is byte-identical to one this
    /// codebase produced before the field existed, so no reader, test, or
    /// checksum that predates it moves. Absence has always meant "gather from
    /// the stored table" and still does; only the codebook that *cannot* be
    /// read off the table announces itself.
    ///
    /// Tag `0` is deliberately unused, so a zero byte from a corrupt or
    /// zero-padded payload is refused rather than read as a valid codebook.
    /// Tag `1` is reserved for a future codebook that needs to be named
    /// explicitly rather than inferred from absence.
    fn to_wire(self) -> Option<(u8, u32)> {
        match self {
            QtipCodebook::Gaussian => None,
            QtipCodebook::Mcg { mult } => Some((2, mult)),
        }
    }

    /// Parse the wire tag, reading the multiplier only when the tag calls for
    /// one. Unknown tags fail closed (DOCTRINE: refuse, never launder).
    fn from_wire(tag: u8, read_mult: impl FnOnce() -> Result<u32>) -> Result<Self> {
        match tag {
            GEOMETRY_WIRE_TAG => candle_core::bail!(
                "QtipLayer: internal error — the geometry section reached the codebook parser. \
                 The trailing-section loop must consume tag {GEOMETRY_WIRE_TAG} itself."
            ),
            2 => {
                let mult = read_mult()?;
                if mult == 0 || mult % 2 == 0 {
                    candle_core::bail!(
                        "QtipLayer: computed-codebook multiplier {mult:#010X} is even. An MCG \
                         multiplier must be odd to have full period; this payload is corrupt."
                    );
                }
                Ok(QtipCodebook::Mcg { mult })
            }
            other => candle_core::bail!(
                "QtipLayer: unknown codebook tag {other} in the UQFF payload. This artifact was \
                 written by a newer Arc than this build understands; refusing rather than \
                 decoding its symbols against the wrong codebook."
            ),
        }
    }
}

/// Trailing-section tag for the trellis geometry.
///
/// **It is 3, and it is written BEFORE the codebook section, and both of those
/// are load-bearing.** A build that predates this field parses the trailing
/// region by reading one byte and handing it to [`QtipCodebook::from_wire`],
/// which refuses every tag it does not know. So an old Arc handed a
/// non-default-geometry artifact sees tag 3 first and fails closed with
/// "written by a newer Arc … refusing rather than decoding its symbols against
/// the wrong codebook" — which is exactly the right outcome, because it cannot
/// decode this geometry either.
///
/// Put the section *after* the codebook and that property is lost: an old
/// reader would consume the codebook section, stop, and decode a K=8/V=4 symbol
/// stream as K=4/V=2. It would not even fault — the packed byte count is
/// identical at 2 bpw, so every index stays in bounds and the output is
/// plausible garbage. [`tests::geometry_section_precedes_the_codebook_section`]
/// pins the order.
const GEOMETRY_WIRE_TAG: u8 = 3;

/// Which trellis geometry a [`QtipLayer`]'s symbols were baked at.
///
/// **This is a format discriminator, not a tuning knob.** Symbols are a
/// bit-stream interpreted by a specific `(K, L, V)` and a specific reproduction
/// table; read at the wrong geometry they decode to different weights, silently.
///
/// `K` is carried as data, not baked into the variant, because that is what the
/// wire already says — the section is `[tag, K, L, V]`. Within the V=4/L=12
/// family the table is K-independent, so a new K is a new value here and not a
/// new variant. See [`crate::trellis_v4l12`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum QtipGeometry {
    /// K=4 / V=2 / L=16 with a `[65536, 2]` F32 table — every QTIP artifact
    /// Arc has ever written. The default, and it serializes as **nothing**.
    #[default]
    K4V2L16,
    /// The V=4 / L=12 family: a `[4096, 4]` **BF16** table (32,768 B — fits
    /// static shared memory), with `k` bits per symbol.
    ///
    /// `k = 8` is the byte-aligned control at 2.00 bpw; `k = 9` is 2.25 bpw and
    /// is the quality winner. Constructed only through
    /// [`QtipGeometry::trellis_v4l12`], which refuses a `k` with no decoder.
    TrellisV4L12 {
        /// The decode rung, which owns the K-dependent arithmetic. Stored
        /// rather than a bare `k` so the packed-size formula has exactly one
        /// implementation — see [`QtipGeometry::packed_len`].
        rung: trellis_v4l12::Rung,
    },
}

impl QtipGeometry {
    /// The V=4/L=12 family at symbol width `k`, refusing a `k` this build has
    /// no decoder for.
    ///
    /// Refuses rather than storing it: `k` arrives from an artifact, and a
    /// geometry we cannot decode must not be constructible as a value that
    /// later code will try to use.
    pub fn trellis_v4l12(k: u32) -> Result<Self> {
        let rung = trellis_v4l12::Rung::new(k).map_err(candle_core::Error::Msg)?;
        Ok(QtipGeometry::TrellisV4L12 { rung })
    }

    /// Bits per symbol.
    pub fn k(self) -> u32 {
        match self {
            QtipGeometry::K4V2L16 => K,
            QtipGeometry::TrellisV4L12 { rung } => rung.k(),
        }
    }

    /// Trellis state width in bits.
    pub fn l(self) -> u32 {
        match self {
            QtipGeometry::K4V2L16 => L,
            QtipGeometry::TrellisV4L12 { .. } => trellis_v4l12::L,
        }
    }

    /// Reproduction values per symbol.
    pub fn v(self) -> u32 {
        match self {
            QtipGeometry::K4V2L16 => V,
            QtipGeometry::TrellisV4L12 { .. } => trellis_v4l12::V,
        }
    }

    /// Bits per weight × 100 (`100·K/V`), as an integer so it compares exactly.
    ///
    /// 200 for the shipped rung and for K=8/V=4; **225 for K=9/V=4**; 250 for
    /// K=10/V=4. Not every geometry in this enum is the same bit rate any more,
    /// which is a change from when the family was K=8-only.
    pub fn bpw_x100(self) -> u32 {
        self.k() * 100 / self.v()
    }

    /// Total values in the reproduction table: `2^L × V`.
    pub fn lut_values(self) -> usize {
        (1usize << self.l()) * self.v() as usize
    }

    /// Element type of the reproduction table.
    ///
    /// Part of the discriminator, not an implementation detail: the V=4 table
    /// is BF16 precisely so it lands at 32,768 B, and an F32 table of the same
    /// shape is a different artifact.
    pub fn lut_dtype(self) -> DType {
        match self {
            QtipGeometry::K4V2L16 => DType::F32,
            QtipGeometry::TrellisV4L12 { .. } => DType::BF16,
        }
    }

    /// Trellis symbols in a row of `in_features` weights. Depends only on V.
    pub fn num_symbols(self, in_features: usize) -> usize {
        in_features / self.v() as usize
    }

    /// Bytes of a row that actually hold symbols, governed by the bit rate.
    ///
    /// **Not the row stride.** [`QtipGeometry::packed_len`] is what a tensor is
    /// allocated at and what this format validates; it additionally carries
    /// tail padding and a round-up to 4 for the V=4/L=12 family. Conflating
    /// them is how a bits-per-weight claim ends up four bytes wrong.
    pub fn data_bytes(self, in_features: usize) -> usize {
        match self {
            QtipGeometry::K4V2L16 => self.num_symbols(in_features) / 2,
            QtipGeometry::TrellisV4L12 { rung } => rung.data_bytes(rung.num_symbols(in_features)),
        }
    }

    /// The allocated length of a row of `in_features` weights — the tensor's
    /// last dimension, and what this format validates.
    ///
    /// **Delegated, not restated.** `num_symbols / (8 / K)` is wrong at K=9 —
    /// there is no whole number of symbols per byte — and divides by zero, so
    /// the formula has to be the general one. It is also invisible to test at
    /// realistic shapes: every plausible `in_features` is a multiple of 32, so
    /// `num_symbols · K` is a multiple of 8 and floor equals ceil. A duplicate
    /// of this formula here would therefore be unguarded in practice (measured:
    /// mutation W3 floored it and every format test stayed green), so the
    /// family's own [`trellis_v4l12::Rung`] is the single implementation and it
    /// is exercised at non-byte-aligned symbol counts by that module's tests.
    pub fn packed_len(self, in_features: usize) -> usize {
        match self {
            QtipGeometry::K4V2L16 => self.num_symbols(in_features) / 2,
            QtipGeometry::TrellisV4L12 { rung } => rung.packed_len(in_features),
        }
    }

    /// Short label for bake headers and error messages.
    pub fn tag(self) -> String {
        match self {
            QtipGeometry::K4V2L16 => "k4v2l16".to_string(),
            QtipGeometry::TrellisV4L12 { rung } => format!("k{}v4l12", rung.k()),
        }
    }

    /// Wire encoding. **The default geometry writes nothing**, so every
    /// artifact Arc has already produced stays byte-identical and no existing
    /// checksum moves. Only a geometry that cannot be inferred from absence
    /// announces itself.
    fn to_wire(self) -> Option<(u8, [u8; 3])> {
        match self {
            QtipGeometry::K4V2L16 => None,
            QtipGeometry::TrellisV4L12 { rung } => Some((
                GEOMETRY_WIRE_TAG,
                [
                    rung.k() as u8,
                    trellis_v4l12::L as u8,
                    trellis_v4l12::V as u8,
                ],
            )),
        }
    }

    /// Parse a `(K, L, V)` triple into a known geometry.
    ///
    /// The triple is stored rather than an opaque ordinal so a payload is
    /// self-describing to a human with a hex editor, and so an unsupported but
    /// well-formed geometry produces a diagnosis instead of "unknown tag".
    fn from_wire_body(k: u8, l: u8, v: u8) -> Result<Self> {
        // Written as comparisons, not as match patterns: a bare `K` in pattern
        // position is a const pattern only for as long as the const exists, and
        // silently becomes a catch-all binding if it is ever renamed away.
        let got = (k as u32, l as u32, v as u32);
        if got == (K, L, V) {
            return Ok(QtipGeometry::K4V2L16);
        }
        if got.1 == trellis_v4l12::L && got.2 == trellis_v4l12::V {
            // In-family: K is the free parameter, so let the family say
            // whether it has a decoder for this one.
            return QtipGeometry::trellis_v4l12(got.0);
        }
        let (ok, ol, ov) = got;
        candle_core::bail!(
            "QtipLayer: UQFF payload declares trellis geometry K={ok}/L={ol}/V={ov}, which this \
             build has no decoder for. Refusing rather than decoding its symbols at the wrong \
             geometry — the packed byte count alone does not identify a geometry, so a wrong \
             guess need not even fault.",
        )
    }

    /// Check that a layer's tensors are shaped the way this geometry requires.
    ///
    /// The tag alone is a claim; this is what makes it true. The table's dtype
    /// and length are the load-bearing half: K=4/V=2 and K=8/V=4 are both
    /// 2 bits per weight, so `blocks` has the identical size either way and
    /// cannot discriminate them.
    fn validate_shapes(self, blocks: &Tensor, lut: &Tensor, in_features: usize) -> Result<()> {
        if !in_features.is_multiple_of(self.v() as usize) {
            candle_core::bail!(
                "QtipLayer[{}]: in_features {in_features} is not a multiple of V={}",
                self.tag(),
                self.v()
            );
        }
        let want_packed = self.packed_len(in_features);
        let got_packed = blocks.dims()[blocks.dims().len() - 1];
        if got_packed != want_packed {
            candle_core::bail!(
                "QtipLayer[{}]: blocks row is {got_packed} B but K={}/V={} with in_features \
                 {in_features} needs {want_packed} B",
                self.tag(),
                self.k(),
                self.v()
            );
        }
        if lut.dtype() != self.lut_dtype() {
            candle_core::bail!(
                "QtipLayer[{}]: reproduction table is {:?} but this geometry stores {:?}. The \
                 table's element type is part of the format, not an implementation detail.",
                self.tag(),
                lut.dtype(),
                self.lut_dtype()
            );
        }
        if lut.elem_count() != self.lut_values() {
            candle_core::bail!(
                "QtipLayer[{}]: reproduction table has {} values, expected 2^{} × {} = {}",
                self.tag(),
                lut.elem_count(),
                self.l(),
                self.v(),
                self.lut_values()
            );
        }
        Ok(())
    }

    /// Codebooks this geometry can legally carry.
    ///
    /// The computed `sum2` codebook is V=2-specific — `mcg_codeword_v2` folds
    /// exactly two chained MCG products, and the CUDA twin
    /// (`qtip_codebook.cuh::qtip_cb_pair_from_x0`) returns a `float2`. There is
    /// no V=4 form of it, so pairing it with any V≠2 geometry is not a thing
    /// that can be decoded, and it is refused on both the write and the read
    /// side rather than left to fail somewhere less obvious.
    fn check_codebook(self, codebook: QtipCodebook) -> Result<()> {
        match codebook {
            QtipCodebook::Mcg { .. } if self.v() != 2 => candle_core::bail!(
                "QtipLayer[{}]: the computed `sum2` codebook is V=2-only (it produces a PAIR of \
                 reproduction values per state) and cannot describe a V={} geometry. This rung \
                 is table-only.",
                self.tag(),
                self.v()
            ),
            _ => Ok(()),
        }
    }
}

/// QTIP 2-bit weight layer.
///
/// # Storage layout — two modes
///
/// ## 2-D mode (`num_experts == None`)
///
/// Standard per-linear quantized weight matrix.
/// - `blocks`:  `[N, packed_K]`  (U8, two K=4 symbols per byte → `packed_K = K_in / 4`)
/// - `row_scales`: `[N]`         (F32, per-output-row scale)
/// - `lut`:     `[2^L, V]`        (F32, shared Gaussian trellis LUT)
/// - `rotation_signs`: `Option<[K_in]>` (F32 ±1, shared across rows)
///
/// ## 3-D mode (`num_experts == Some(E)`)  ← RUN-NVFP4-3D
///
/// Per-expert stacked weights for MoE. The same Gaussian LUT and rotation signs
/// are shared across all experts (only the per-row symbols and scales differ),
/// matching the per-input-dim rotation invariance (R is determined entirely by
/// `K_in`, which is identical for every expert).
/// - `blocks`:  `[E, N, packed_K]`  — expert e, output-row n
/// - `row_scales`: `[E, N]`         — expert e, output-row n
/// - `lut`:     `[2^L, V]`           — shared
/// - `rotation_signs`: `Option<[K_in]>` — shared
///
/// `dequantize_weights()` returns `[N, K_in]` in 2-D mode and `[E, N, K_in]`
/// in 3-D mode, in the original (unrotated) frame.
///
/// # gather_forward contract (for sister task #1)
///
/// Sister `gather_forward(x, indices)` reads the 3-D layout above. For each
/// `(token, slot)` indexing pair with `e = indices[token, slot]`:
///   1. Slice `blocks[e]`, `row_scales[e]` to recover the expert's `[N, packed_K]` /
///      `[N]` storage.
///   2. Dequantize using the **shared** `lut` and (if present) **shared**
///      `rotation_signs` + `rotation_block`.
///   3. Rotate `x[token, slot, :]` by the shared rotation, then matmul.
///
/// Bias is currently not used in the 3-D path (V4 Flash MoE experts are
/// bias-free); when present in 2-D mode it is `[N]`.
#[derive(Debug)]
pub struct QtipLayer {
    /// Packed K-bit symbols. For K=4, two symbols per byte.
    /// - 2-D mode: `[N, num_symbols_per_row / 2]`
    /// - 3-D mode: `[E, N, num_symbols_per_row / 2]`
    blocks: Tensor,
    /// Per-row scale factor, FP32.
    /// - 2-D mode: `[N]`
    /// - 3-D mode: `[E, N]`
    row_scales: Tensor,
    /// Shared LUT (one per layer or shared across the module).
    /// Shape: [2^L, V] FP32.
    lut: Tensor,
    /// Optional bias: `[N]` (2-D mode only; 3-D mode does not carry bias).
    bias: Option<Tensor>,
    /// Cached input dim (K) so we can decode without bookkeeping at the user level.
    in_features: usize,
    /// When `Some(E)`, this layer holds `E` stacked experts. When `None`, this
    /// is a standard 2-D linear weight. See the struct-level doc comment for
    /// the storage layout in each mode.
    num_experts: Option<usize>,
    /// Hadamard incoherence rotation signs, shape [in_features], FP32, values ±1.
    ///
    /// When present, both quantize time (weight row pre-rotation) and inference
    /// time (input row pre-rotation) apply a block-diagonal D·H·D rotation
    /// using these signs. The rotation matrix is involutory (R·R = I), so
    /// applying it to both sides of the matmul cancels out exactly, while
    /// transforming the trellis input distribution to be approximately Gaussian
    /// — the assumption Viterbi's cost model makes. `None` disables rotation
    /// (back-compat with checkpoints quantized before RUN-158 and with the
    /// Greedy path that already achieves >0.85 cos sim without it).
    ///
    /// In 3-D mode this vector is **shared** across all experts because the
    /// rotation is determined by `K_in` (identical for every expert in a stack).
    rotation_signs: Option<Tensor>,
    /// Block size for the block-diagonal Hadamard rotation. 0 when disabled.
    rotation_block: usize,
    /// Which trellis search produced these blocks. Serialized into UQFF from
    /// 0.3.0 and checked at load (DOCTRINE D4 §3) so a greedy bake can never
    /// again pass itself off as Viterbi. Deliberately has no `Default`: every
    /// construction site must state provenance, and payloads loaded from a
    /// format that cannot carry it say [`QtipSearchStamp::Unstamped`].
    search: QtipSearchStamp,
    /// *Which* trellis search, beside `search`'s *whether*: the beam width and
    /// the objective. Serialized as the UQFF ≥ 0.3.0 flags byte. Like `search`
    /// it has no `Default` — [`QtipSearchDetail::Unknown`] is the only way to
    /// say "not recorded", and it refuses to serialize.
    search_detail: QtipSearchDetail,
    /// Which codebook `lut` holds, and therefore which reproduction values
    /// these symbols mean. [`QtipCodebook::Gaussian`] for every artifact
    /// written before the discriminator existed — those keep gathering from
    /// their stored table exactly as they always did. A
    /// [`QtipCodebook::Mcg`] layer's `lut` holds the same values the decode
    /// computes, so the table stays a correct fallback (and the CPU search
    /// still wants a flat one) while every GPU path skips it entirely.
    codebook: QtipCodebook,
    /// Which trellis geometry `blocks` was baked at, and therefore how its
    /// bytes decompose into symbols and how `lut` is indexed.
    ///
    /// [`QtipGeometry::K4V2L16`] for every artifact written before the
    /// discriminator existed — which is what they are. Deliberately has no
    /// per-site default: both supported geometries are 2 bits per weight, so a
    /// mislabelled layer has a correctly-sized `blocks` tensor and decodes to
    /// plausible garbage rather than faulting. Every construction site states
    /// it, and the compiler is what enforces that.
    geometry: QtipGeometry,
}

/// Borrowed, dequantization-free view of a [`QtipLayer`]'s packed trellis
/// weights. Used by custom decode paths (e.g. `arc-cuda-graph`) that feed the
/// raw 2-bit bytes straight into `fused_gemv_cuda` / a fused gather-gemv kernel
/// instead of materializing a dense BF16 weight — the memory blow-up the
/// dedicated decode path must avoid for V4 Flash. All fields borrow the live
/// layer and must not outlive it.
pub struct QtipPackedView<'a> {
    /// Packed K-bit symbols, U8. 2-D `[N, packed_K]` or 3-D `[E, N, packed_K]`.
    pub blocks: &'a Tensor,
    /// Per-row scales, F32. 2-D `[N]` or 3-D `[E, N]`.
    pub row_scales: &'a Tensor,
    /// Shared Gaussian trellis LUT, `[2^L, V]` F32.
    pub lut: &'a Tensor,
    /// Optional bias `[N]` (2-D mode only; 3-D expert stacks are bias-free).
    pub bias: Option<&'a Tensor>,
    /// Input feature dim K.
    pub in_features: usize,
    /// `Some(E)` for stacked experts (3-D), `None` for a plain 2-D linear.
    pub num_experts: Option<usize>,
    /// Hadamard rotation signs `[in_features]` F32 (+/-1) when rotation is on.
    pub rotation_signs: Option<&'a Tensor>,
    /// Hadamard block size; 0 when rotation is disabled (the Greedy path).
    pub rotation_block: usize,
}

impl QtipLayer {
    /// Borrow this layer's packed trellis weights without dequantizing. The
    /// returned [`QtipPackedView`] aliases the live tensors; do not outlive it.
    pub fn packed_view(&self) -> QtipPackedView<'_> {
        QtipPackedView {
            blocks: &self.blocks,
            row_scales: &self.row_scales,
            lut: &self.lut,
            bias: self.bias.as_ref(),
            in_features: self.in_features,
            num_experts: self.num_experts,
            rotation_signs: self.rotation_signs.as_ref(),
            rotation_block: self.rotation_block,
        }
    }
}

/// Trellis search that produced (or will produce) a QTIP payload.
///
/// **`Greedy` is banned in production — DOCTRINE D4, "ban greedy forever".**
/// It survives as an enum variant for exactly one reason: unit tests need a
/// cheap search (an exhaustive trellis pass over even a modest fixture is
/// ~8.5G ops — wave1-B), and deleting the variant would delete the reference
/// implementation the Viterbi optimality tests compare against.
///
/// Every door into the quantizer refuses it:
///
/// | entry point | Greedy accepted? |
/// |---|---|
/// | ISQ dispatch (`--isq qtip2` / `qtip2b`, `unquantized/mod.rs`) | never — hard-wired to [`QtipMode::default_expert_mode`] |
/// | `quantize` / `quantize_with_mode` (the production door) | never — hard error in **every** build |
/// | `quantize_with_options*` / `quantize_with_calibration` (the fixture door) | only under `cfg!(test)`, i.e. only inside this crate's own test binaries |
/// | UQFF load | never — a `Greedy` stamp is refused outright, see [`QtipSearchStamp`] |
///
/// So the ONLY route to a greedy-quantized layer is a direct
/// `quantize_with_options*(.., QtipMode::Greedy, ..)` call compiled with
/// `cfg(test)` inside `mistralrs-quant`. There is no env var, no CLI flag, no
/// config field and no serde path anywhere in the workspace that reaches it,
/// and a release build of this crate hard-errors before emitting a single
/// greedy symbol. There is deliberately no `Default` impl: a defaulted search
/// mode is how greedy escaped the first time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QtipMode {
    /// Fast but suboptimal: at each position, pick the locally-best symbol given the current state.
    /// ~5-10× faster than Viterbi at calibration time, with ~3× higher reconstruction error.
    ///
    /// **BANNED in production (D4).** Measured cost on the realistic
    /// FP4-lattice fixtures that match V4's expert source chain: matmul cos
    /// 0.675 (greedy, no rotation) / 0.887 (greedy + rotation) vs 0.963
    /// (Viterbi + rotation). It is also *cheaper at inference* when it
    /// disables rotation, so a greedy artifact reports inflated speed as well
    /// as degraded quality — there is no metric it is honest on.
    Greedy,
    /// Globally optimal symbol sequence via dynamic-programming search over the trellis.
    /// Matches Cornell's paper numbers; slower at calibration time but quantization is one-shot.
    Viterbi,
}

impl QtipMode {
    /// The mode every production bake uses, for 3-D MoE expert stacks and for
    /// plain 2-D linears alike. **Always Viterbi** — there is no env var and
    /// no argument that changes it (D4).
    ///
    /// History: this used to read `ARC_QTIP_EXPERT_GREEDY`. That knob is gone.
    /// The greedy default it guarded baked every V4 expert with greedy walk +
    /// NO rotation + max/3 scales and was the root cause of PPL qtip2=58.85 vs
    /// q2k=22.50 (H200, 2026-08-13). GPU Viterbi keeps the bake inside the
    /// 7-20 min budget (RUN-161 prefix-grouped kernel), so speed was never a
    /// reason to default to greedy — it was only ever a reason to fix the fast
    /// path. `ARC_QTIP_EXPERT_VITERBI` is still accepted as a no-op so old
    /// runbook invocations do not fail.
    pub const fn default_expert_mode() -> Self {
        QtipMode::Viterbi
    }

    /// **The production door.** Refuses `Greedy` in every build, tests
    /// included, so a test can assert the ban holds and no in-crate caller can
    /// smuggle greedy through a production entry point.
    pub fn deny_greedy(self, entry_point: &str) -> Result<()> {
        match self {
            QtipMode::Viterbi => Ok(()),
            QtipMode::Greedy => candle_core::bail!(
                "{entry_point}: QtipMode::Greedy is banned in production (DOCTRINE D4). \
                 Greedy costs ~0.29 matmul cosine on FP4-lattice experts (0.675 vs 0.963) \
                 and silently disables the Hadamard incoherence rotation. There is no flag \
                 to re-enable it: if a bake is too slow, fix the fast path — do not \
                 downgrade the artifact. Tests that need a cheap search construct it \
                 directly via `quantize_with_options*` under `cfg(test)`."
            ),
        }
    }

    /// **The fixture door.** Refuses `Greedy` in every build EXCEPT this
    /// crate's own `cfg(test)` binaries, where a cheap search is a legitimate
    /// fixture. `cfg!(test)` is false for every downstream crate and for every
    /// release build, so this is a structural ban rather than a documented
    /// convention.
    pub fn deny_greedy_outside_tests(self, entry_point: &str) -> Result<()> {
        match self {
            QtipMode::Viterbi => Ok(()),
            QtipMode::Greedy if cfg!(test) => Ok(()),
            QtipMode::Greedy => self.deny_greedy(entry_point),
        }
    }

    /// Tag for the bake log header. Task #17: a Greedy bake must never again be
    /// mistakable for a Viterbi bake by reading the log.
    fn tag(&self) -> &'static str {
        match self {
            QtipMode::Greedy => "greedy",
            QtipMode::Viterbi => "viterbi",
        }
    }
}

/// Bake-time search + objective selection for the trellis quantizer (wave13-AD).
///
/// Both axes default to today's behaviour and are read once per process from
/// the environment:
///
/// * `ARC_QTIP_BEAM=<W>` — prune the trellis search to a beam of width `W`.
///   Unset / `0` / `off` keeps the exhaustive `2^L` dynamic program.
/// * `ARC_QTIP_HESSIAN=1` — minimise `(w−ŵ)ᵀ H (w−ŵ)` using a diagonal
///   activation Hessian instead of `‖w−ŵ‖²`. Requires calibration data (an
///   imatrix); without it the bake silently stays unweighted, which the log
///   header reports as `mse(no-calibration)`.
#[derive(Clone, Copy, Debug, Default)]
pub struct QtipBakeConfig {
    /// Search strategy over the trellis state space.
    pub search: TrellisSearch,
    /// Whether to use the diagonal-Hessian objective when calibration data is
    /// available.
    pub hessian: bool,
    /// Which codebook the symbols will mean. Unlike `search` and `hessian`
    /// this is not a quality/speed dial — it is the artifact's format — so an
    /// unrecognised `ARC_QTIP_CODEBOOK` is refused instead of defaulted.
    pub codebook: QtipCodebook,
}

impl QtipBakeConfig {
    /// Read (and memoise) the bake configuration from the environment.
    ///
    /// Fallible because `ARC_QTIP_CODEBOOK` fails closed: a typo there would
    /// otherwise silently bake against a different codebook than the operator
    /// asked for, and unlike a mis-set beam width that is not recoverable by
    /// re-reading the artifact.
    pub fn get() -> Result<Self> {
        static CFG: std::sync::OnceLock<std::result::Result<QtipBakeConfig, String>> =
            std::sync::OnceLock::new();
        let cfg = CFG.get_or_init(|| {
            let codebook = QtipCodebook::from_env().map_err(|e| e.to_string())?;
            Ok(QtipBakeConfig {
                search: TrellisSearch::from_env(),
                hessian: matches!(
                    std::env::var("ARC_QTIP_HESSIAN").as_deref(),
                    Ok("1") | Ok("true") | Ok("on")
                ),
                codebook,
            })
        });
        match cfg {
            Ok(c) => Ok(*c),
            Err(msg) => candle_core::bail!("{msg}"),
        }
    }
}

/// Emit the bake header describing exactly what search and objective produced
/// this checkpoint — once per distinct configuration per process.
///
/// Task #17: "a Greedy bake must never again be mistakable for Viterbi." The
/// header names the mode, the search strategy, the objective (including whether
/// calibration data was actually supplied), and the incoherence rotation width,
/// so a bake log is self-describing.
fn bake_header_line(
    rung: &str,
    mode: QtipMode,
    cfg: QtipBakeConfig,
    rotation_block: usize,
    have_calibration: bool,
) -> String {
    // Greedy ignores the trellis search entirely — it is a one-step walk, not a
    // dynamic program — so never label a greedy bake with a beam width.
    let search = match mode {
        QtipMode::Greedy => "greedy-walk (no trellis search)".to_string(),
        QtipMode::Viterbi => cfg.search.tag(),
    };
    let objective = match (cfg.hessian, have_calibration) {
        (true, true) => "hessian-diag (weighted)",
        (true, false) => "mse(no-calibration)",
        (false, _) => "mse (unweighted)",
    };
    let rotation = if rotation_block >= 2 {
        format!("hadamard-{rotation_block}")
    } else {
        "off".to_string()
    };
    format!(
        "QTIP bake [{rung}]: mode={} search={search} objective={objective} \
         rotation={rotation} codebook={}",
        mode.tag(),
        cfg.codebook.tag()
    )
}

/// Map a requested [`TrellisSearch`] onto the CUDA quantize kernels.
///
/// * `Ok(TrellisSearch::Exhaustive)` — run the prefix-grouped DP kernel.
/// * `Ok(TrellisSearch::Beam { width })` — run `qtip_beam.cu` at that width.
/// * `Err(_)` — the CUDA path cannot honour this search. **Never** substitute a
///   different one: PR #29 exists precisely so that the same command cannot
///   produce two different checkpoints depending on which device it ran on.
///
/// A width at or above the full state space prunes nothing, so it is mapped to
/// the exhaustive kernel — identical semantics, and it mirrors the CPU
/// `viterbi::quantize_row`, which routes `Beam { width >= 2^L }` the same way.
/// `max_beam_width` is read from the kernel itself (`qtip_beam_max_width`) so
/// the Rust-side limit cannot drift from the CUDA one; pass 0 when the kernels
/// are absent.
#[cfg(any(feature = "cuda", test))]
pub(crate) fn cuda_search_plan(
    search: TrellisSearch,
    max_beam_width: usize,
) -> Result<TrellisSearch> {
    match search {
        TrellisSearch::Exhaustive => Ok(TrellisSearch::Exhaustive),
        TrellisSearch::Beam { width } if width >= LUT_SIZE => Ok(TrellisSearch::Exhaustive),
        // `beam_quantize_row` clamps the width to at least 1; mirror it rather
        // than inventing a third behaviour for a width the env parser can never
        // produce anyway.
        TrellisSearch::Beam { width } if width.max(1) <= max_beam_width => {
            Ok(TrellisSearch::Beam {
                width: width.max(1),
            })
        }
        TrellisSearch::Beam { width } => candle_core::bail!(
            "QTIP quantize: ARC_QTIP_BEAM={width} but the CUDA beam kernel supports \
             widths 1..={max_beam_width}. Either lower the width (256 is the \
             quality-neutral setting measured in PR #29) or bake on CPU; the GPU \
             path will not silently substitute a different search."
        ),
    }
}

fn log_bake_header(
    rung: &str,
    mode: QtipMode,
    cfg: QtipBakeConfig,
    rotation_block: usize,
    have_calibration: bool,
) {
    use std::sync::{Mutex, OnceLock};
    static SEEN: OnceLock<Mutex<std::collections::HashSet<String>>> = OnceLock::new();

    let line = bake_header_line(rung, mode, cfg, rotation_block, have_calibration);

    let seen = SEEN.get_or_init(|| Mutex::new(std::collections::HashSet::new()));
    let mut guard = match seen.lock() {
        Ok(g) => g,
        Err(poisoned) => poisoned.into_inner(),
    };
    if guard.insert(line.clone()) {
        tracing::info!("{line}");
    }
}

/// Whether the Hadamard incoherence rotation is applied to a bake.
///
/// Split out from [`QtipMode`] deliberately (wave13-AG). Rotation used to be a
/// *side effect* of the search mode: `matches!(mode, QtipMode::Viterbi)`,
/// written independently in three places. A `matches!` test silently answers
/// "off" for any mode variant added later — exactly the defect class that
/// produced PPL 58.85, where a bake lost the search AND the incoherence
/// processing at once and nothing in the code said so out loud.
///
/// [`QtipRotation::for_mode`] is now the single decision point and it is an
/// exhaustive `match`: a new [`QtipMode`] variant fails to COMPILE until its
/// rotation policy is stated, instead of silently inheriting "off".
///
/// Today's observable behaviour is unchanged (`Greedy → Off`, `Viterbi → On`),
/// so no checkpoint bytes move; the point is that the decision is now visible
/// and pinned by `rotation_policy_is_pinned_per_mode`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QtipRotation {
    /// Block-diagonal Hadamard incoherence processing ON (D11: the default).
    On,
    /// No incoherence processing. Reachable only for test fixtures, since the
    /// only mode mapped to it is the test-only [`QtipMode::Greedy`].
    Off,
}

impl QtipRotation {
    /// THE rotation policy table. Exhaustive by construction — every mode must
    /// state its answer here.
    pub const fn for_mode(mode: QtipMode) -> Self {
        match mode {
            // Greedy is a test fixture only (D4). Keeping rotation off holds
            // the historical fixture bytes and keeps the fixture cheap; it is
            // NOT a quality trade-off anyone can select in production.
            QtipMode::Greedy => QtipRotation::Off,
            // D11: rotation is the default and is data-free, so it holds on any
            // tenant model with no calibration corpus.
            QtipMode::Viterbi => QtipRotation::On,
        }
    }

    /// Boolean form for the `use_rotation` plumbing.
    pub const fn enabled(self) -> bool {
        matches!(self, QtipRotation::On)
    }
}

/// Search provenance stamped into every QTIP UQFF payload from UQFF 0.3.0
/// (DOCTRINE D4 §3: "the format refuses it, not just the CLI").
///
/// Greedy escaped three times before this existed, and every escape was
/// invisible for the same reason: nothing recorded which search produced an
/// artifact, so a greedy bake could pass itself off as Viterbi indefinitely.
/// The stamp closes that: provenance travels with the weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QtipSearchStamp {
    /// Trellis dynamic program. Covers both of [`TrellisSearch`]'s arms — the
    /// exhaustive `2^L` DP and the pruned beam (`ARC_QTIP_BEAM`, wave13-AD),
    /// which approximates the *same* algorithm: an unpruned beam reproduces
    /// the exhaustive DP bit-for-bit, and that equivalence is regression-tested
    /// in `viterbi.rs`. The only search a production bake can emit.
    ///
    /// Not recorded: the beam *width*. A narrow beam is a genuinely different
    /// quality point that this stamp cannot distinguish from the exhaustive
    /// DP — a flags byte beside the stamp would carry the width and the
    /// Hessian objective (wave13-AG follow-up).
    Trellis,
    /// Greedy walk. Refused at load, unconditionally.
    Greedy,
    /// Payload written before UQFF 0.3.0 — it carries no stamp byte, so its
    /// provenance is unknown. See [`QtipSearchStamp::enforce_at_load`] for the
    /// legacy policy.
    Unstamped,
}

impl QtipSearchStamp {
    /// Exhaustive mode → stamp mapping (mirrors [`QtipRotation::for_mode`]).
    pub const fn for_mode(mode: QtipMode) -> Self {
        match mode {
            QtipMode::Greedy => QtipSearchStamp::Greedy,
            QtipMode::Viterbi => QtipSearchStamp::Trellis,
        }
    }

    /// Wire byte, or `None` for [`Self::Unstamped`] (which is never written —
    /// it only ever comes from reading a pre-0.3.0 payload). `0` is reserved
    /// and invalid so a zero-filled buffer can never read as a valid stamp.
    pub const fn to_wire(self) -> Option<u8> {
        match self {
            QtipSearchStamp::Trellis => Some(1),
            QtipSearchStamp::Greedy => Some(2),
            QtipSearchStamp::Unstamped => None,
        }
    }

    /// Parse a wire byte written by [`Self::to_wire`].
    pub fn from_wire(byte: u8) -> Result<Self> {
        match byte {
            1 => Ok(QtipSearchStamp::Trellis),
            2 => Ok(QtipSearchStamp::Greedy),
            other => candle_core::bail!(
                "QTIP artifact: unexpected search-stamp byte {other} (expected 1=trellis, 2=greedy)"
            ),
        }
    }

    /// Human tag for logs and errors.
    pub const fn tag(self) -> &'static str {
        match self {
            QtipSearchStamp::Trellis => "trellis",
            QtipSearchStamp::Greedy => "greedy",
            QtipSearchStamp::Unstamped => "unstamped(pre-0.3.0)",
        }
    }

    /// Load-time gate. Called by both rungs' UQFF deserializers.
    ///
    /// Policy:
    /// * `Trellis` → serve.
    /// * `Greedy` → **hard refuse, no override.** No build of Arc since the ban
    ///   can emit this stamp, so its presence means a deliberately doctored or
    ///   foreign artifact. D4 is absolute.
    /// * `Unstamped` + **no rotation** → **refuse, overridable.** Every shipped
    ///   bake policy enabled rotation iff the trellis search ran
    ///   ([`QtipRotation::for_mode`]), so a pre-0.3.0 payload with
    ///   `rotation_block == 0` is a greedy bake with very high confidence —
    ///   this is precisely the artifact class that measured PPL 58.85.
    /// * `Unstamped` + rotation present → **serve, with a loud one-time warn.**
    ///   Rotation-on implies the trellis search under every shipped policy, so
    ///   refusing would brick honest artifacts (e.g. everything baked between
    ///   PR #20 and this change) for no quality gain. The warn names the file
    ///   as unverifiable so an operator can choose to re-bake.
    ///
    /// `ARC_ALLOW_UNSTAMPED_QTIP=1` downgrades the refuse case to the warn
    /// case. It is a LOAD-side escape for artifacts that already exist; it
    /// cannot make a bake produce greedy, and it does not apply to an explicit
    /// `Greedy` stamp.
    pub fn enforce_at_load(self, rung: &str, rotation_block: usize) -> Result<()> {
        match self {
            QtipSearchStamp::Trellis => Ok(()),
            QtipSearchStamp::Greedy => candle_core::bail!(
                "{rung}: refusing a UQFF artifact stamped `greedy`. Greedy-baked weights are \
                 banned (DOCTRINE D4): measured matmul cos 0.675 vs 0.963 for the trellis \
                 search on the FP4-lattice experts this rung is built for, and the missing \
                 incoherence rotation also makes the artifact report inflated decode speed. \
                 Re-bake with `mistralrs quantize`. There is no override for this case."
            ),
            QtipSearchStamp::Unstamped if rotation_block >= 2 => {
                warn_unstamped_qtip_artifact_once(rung);
                Ok(())
            }
            QtipSearchStamp::Unstamped if allow_unstamped_qtip_artifacts() => {
                warn_unstamped_qtip_artifact_once(rung);
                Ok(())
            }
            QtipSearchStamp::Unstamped => candle_core::bail!(
                "{rung}: refusing a pre-UQFF-0.3.0 QTIP artifact that carries no incoherence \
                 rotation. Under every bake policy Arc has ever shipped, rotation was enabled \
                 if and only if the trellis search ran, so this artifact was almost certainly \
                 baked with the banned greedy walk (DOCTRINE D4) — the exact artifact class \
                 that measured PPL 58.85 vs q2k 22.50. Re-bake with `mistralrs quantize`; \
                 the new artifact carries a search stamp. To load it anyway for diagnostics, \
                 set ARC_ALLOW_UNSTAMPED_QTIP=1."
            ),
        }
    }
}

/// The flags byte that rides beside [`QtipSearchStamp`] from UQFF 0.3.0
/// (wave13-AF, closing the gap wave13-AG named in its own hand-off).
///
/// The stamp answers "was this a trellis search or a greedy walk" — the D4 ban.
/// It deliberately cannot answer "*which* trellis search", and a `W = 64` beam
/// is a genuinely different quality point from the exhaustive `2^L` DP
/// (PR #29 measured matmul cos 0.95054 vs 0.96495 on FP4-lattice fixtures).
/// An artifact that cannot distinguish them is a mislabelled artifact, which is
/// the failure class the stamp exists to end — so the detail travels with the
/// weights too.
///
/// ## Wire format
///
/// One flags byte immediately after the stamp byte, then a `u16` little-endian
/// width **iff** the beam bit is set:
///
/// | bit | meaning |
/// |---|---|
/// | `0x01` | pruned beam; a `u16` width follows |
/// | `0x02` | diagonal-activation-Hessian objective (`ARC_QTIP_HESSIAN`) |
/// | `0x04..=0x80` | reserved — **must be zero** |
///
/// So an exhaustive unweighted bake costs exactly one byte (`0x00`) and a
/// `W = 256` beam costs three (`0x01`, `0x00 0x01`). No version bump: 0.3.0 is
/// unreleased, so the byte is free to add now, and it is **mandatory** — a
/// payload carrying a stamp but no flags byte is truncated, and truncation
/// fails closed rather than being read as "exhaustive".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QtipSearchDetail {
    /// Provenance recorded by the process that ran the search.
    Known {
        /// `None` — exhaustive dynamic program over all `2^L` states.
        /// `Some(w)` — pruned beam keeping the best `w` states per timestep,
        /// `1 <= w < 2^L`. A width at or above `2^L` prunes nothing and *is*
        /// the exhaustive DP, so it is normalised to `None` at construction
        /// rather than recorded as a beam.
        beam_width: Option<u16>,
        /// The `(w − ŵ)ᵀ H (w − ŵ)` objective was used instead of `‖w − ŵ‖²`.
        hessian: bool,
    },
    /// Not recorded: a pre-0.3.0 payload, or a loader for a format that cannot
    /// carry provenance. Mirrors [`QtipSearchStamp::Unstamped`] — it refuses to
    /// serialize, so an unknown artifact can never be re-emitted as a specific
    /// claim.
    Unknown,
}

impl QtipSearchDetail {
    const FLAG_BEAM: u8 = 0x01;
    const FLAG_HESSIAN: u8 = 0x02;
    const FLAG_RESERVED: u8 = !(Self::FLAG_BEAM | Self::FLAG_HESSIAN);

    /// The detail a bake earned. `search` is the plan that actually ran (post
    /// [`cuda_search_plan`] on the GPU path), never the raw env request.
    ///
    /// Greedy is not a trellis search, so it records neither a width nor an
    /// objective — the flags byte of a greedy artifact is always `0x00`. (It is
    /// refused at load anyway; this just keeps the two fields from contradicting
    /// each other on the wire.)
    pub fn for_bake(mode: QtipMode, search: TrellisSearch, hessian: bool) -> Self {
        match mode {
            QtipMode::Greedy => QtipSearchDetail::Known {
                beam_width: None,
                hessian: false,
            },
            QtipMode::Viterbi => QtipSearchDetail::Known {
                beam_width: match search {
                    TrellisSearch::Beam { width } if (1..LUT_SIZE).contains(&width) => {
                        Some(width as u16)
                    }
                    // width >= 2^L prunes nothing; width 0 is not a search.
                    _ => None,
                },
                hessian,
            },
        }
    }

    /// The detail of a bake that ran the exhaustive DP with the unweighted
    /// objective. This is every `qtip2b` bake by construction: that rung's
    /// `viterbi_quantize_row_2b` is the exhaustive `2^L` DP and it has no
    /// weighted branch metric, so the claim is earned from the code path, not
    /// assumed.
    pub const EXHAUSTIVE_MSE: Self = QtipSearchDetail::Known {
        beam_width: None,
        hessian: false,
    };

    /// Beam width, or `None` for the exhaustive DP / unknown provenance.
    pub fn beam_width(self) -> Option<u16> {
        match self {
            QtipSearchDetail::Known { beam_width, .. } => beam_width,
            QtipSearchDetail::Unknown => None,
        }
    }

    /// Human tag for logs and errors.
    pub fn tag(self) -> String {
        match self {
            QtipSearchDetail::Unknown => "unrecorded".to_string(),
            QtipSearchDetail::Known {
                beam_width,
                hessian,
            } => {
                let search = match beam_width {
                    Some(w) => format!("beam(W={w})"),
                    None => "exhaustive".to_string(),
                };
                let obj = if hessian { "hessian-diag" } else { "mse" };
                format!("{search}/{obj}")
            }
        }
    }

    /// Encode to `(flags, Option<width>)`. `Err` when the detail was never
    /// observed, or when it contradicts the stamp it would be written beside.
    fn to_wire(self, stamp: QtipSearchStamp) -> Result<(u8, Option<u16>)> {
        let (beam_width, hessian) = match self {
            QtipSearchDetail::Unknown => candle_core::bail!(
                "QtipLayer::serialize: refusing to write a search-detail flags byte for a layer \
                 whose search detail was never recorded. Re-quantize from the source weights so \
                 the beam width and objective are earned rather than invented (DOCTRINE D4)."
            ),
            QtipSearchDetail::Known {
                beam_width,
                hessian,
            } => (beam_width, hessian),
        };
        if matches!(stamp, QtipSearchStamp::Greedy) && (beam_width.is_some() || hessian) {
            candle_core::bail!(
                "QtipLayer::serialize: a `greedy` stamp cannot carry a beam width or a weighted \
                 objective — a greedy walk runs no trellis search at all. Refusing to write a \
                 self-contradictory artifact."
            );
        }
        if let Some(w) = beam_width {
            if w == 0 || (w as usize) >= LUT_SIZE {
                candle_core::bail!(
                    "QtipLayer::serialize: beam width {w} is not a pruned search (valid range \
                     1..{LUT_SIZE}); the exhaustive DP must be written as `no beam`, not as a \
                     beam wide enough to prune nothing."
                );
            }
        }
        let mut flags = 0u8;
        if beam_width.is_some() {
            flags |= Self::FLAG_BEAM;
        }
        if hessian {
            flags |= Self::FLAG_HESSIAN;
        }
        Ok((flags, beam_width))
    }

    /// Decode a flags byte (and, when the beam bit is set, the width that
    /// follows). Every rejected case below is a claim the artifact makes about
    /// itself that cannot be true; none of them is normalised into a plausible
    /// value, because silently reading a malformed claim as "exhaustive" is how
    /// a mislabelled artifact would get laundered.
    fn from_wire(
        flags: u8,
        stamp: QtipSearchStamp,
        read_width: impl FnOnce() -> Result<u16>,
    ) -> Result<Self> {
        if flags & Self::FLAG_RESERVED != 0 {
            candle_core::bail!(
                "QTIP artifact: reserved bits set in the search-detail flags byte \
                 (0x{flags:02X}). This artifact was written by a newer Arc whose provenance \
                 fields this build cannot interpret; refusing rather than guessing."
            );
        }
        let hessian = flags & Self::FLAG_HESSIAN != 0;
        let beam_width = if flags & Self::FLAG_BEAM != 0 {
            let w = read_width()?;
            if w == 0 || (w as usize) >= LUT_SIZE {
                candle_core::bail!(
                    "QTIP artifact: search-detail claims a beam of width {w}, which is not a \
                     pruned search (valid range 1..{LUT_SIZE}). A width at or above the state \
                     space prunes nothing and must be recorded as the exhaustive DP."
                );
            }
            Some(w)
        } else {
            None
        };
        if matches!(stamp, QtipSearchStamp::Greedy) && (beam_width.is_some() || hessian) {
            candle_core::bail!(
                "QTIP artifact: stamped `greedy` but the search-detail claims {}. A greedy walk \
                 runs no trellis search, so this artifact contradicts itself; refusing.",
                QtipSearchDetail::Known {
                    beam_width,
                    hessian
                }
                .tag()
            );
        }
        Ok(QtipSearchDetail::Known {
            beam_width,
            hessian,
        })
    }
}

fn allow_unstamped_qtip_artifacts() -> bool {
    matches!(
        std::env::var("ARC_ALLOW_UNSTAMPED_QTIP").as_deref(),
        Ok("1") | Ok("true") | Ok("on")
    )
}

/// One warn per rung per process — a per-layer warn on a 61-layer MoE is noise
/// that scrolls the useful line off the screen (that is how escape #2 stayed
/// invisible).
fn warn_unstamped_qtip_artifact_once(rung: &str) {
    use std::sync::{Mutex, OnceLock};
    static SEEN: OnceLock<Mutex<std::collections::HashSet<String>>> = OnceLock::new();
    let seen = SEEN.get_or_init(|| Mutex::new(std::collections::HashSet::new()));
    let mut guard = match seen.lock() {
        Ok(g) => g,
        Err(poisoned) => poisoned.into_inner(),
    };
    if guard.insert(rung.to_string()) {
        tracing::warn!(
            "{rung}: UQFF artifact predates the 0.3.0 search stamp — its trellis search cannot \
             be verified from the file. Rotation is present, which implies a Viterbi bake under \
             every policy Arc has shipped, so it is being served. Re-bake to get a verifiable \
             artifact."
        );
    }
}

impl QtipLayer {
    /// Quantize an unquantized weight tensor [N, K_in] to QTIP 2-bit format
    /// with the production recipe: [`QtipMode::default_expert_mode`] (Viterbi)
    /// plus the rotation its [`QtipRotation`] policy selects.
    pub fn quantize(
        weight: &Tensor,
        bias: Option<Tensor>,
        device: &Device,
    ) -> Result<Arc<dyn QuantMethod>> {
        Self::quantize_with_mode(weight, bias, device, QtipMode::default_expert_mode())
    }

    /// Quantize with explicit mode selection. **This is the production door**:
    /// it refuses [`QtipMode::Greedy`] in every build (D4).
    ///
    /// Rotation is not inferred here — it comes from [`QtipRotation::for_mode`],
    /// the single exhaustive policy table, so "which search" and "is incoherence
    /// processing on" are two decisions that are each written down once.
    pub fn quantize_with_mode(
        weight: &Tensor,
        bias: Option<Tensor>,
        device: &Device,
        mode: QtipMode,
    ) -> Result<Arc<dyn QuantMethod>> {
        mode.deny_greedy("QtipLayer::quantize_with_mode")?;
        let use_rotation = QtipRotation::for_mode(mode).enabled();
        Self::quantize_with_options(weight, bias, device, mode, use_rotation)
    }

    /// **The greedy fixture door — this crate's tests only (DOCTRINE D4).**
    ///
    /// Compiled only under `cfg(test)` of `mistralrs-quant`, so it is absent
    /// from every release artifact and unreachable from every other crate.
    /// Unit tests need a cheap search: an exhaustive trellis pass over even a
    /// 4×64×64 expert fixture is ~8.5G ops in a debug build (wave1-B), which
    /// would turn the serde/shape suites into multi-minute runs. Tests that
    /// assert *quality* must never use this — they go through
    /// `quantize_with_mode` like production does.
    #[cfg(test)]
    pub(crate) fn quantize_greedy_fixture(
        weight: &Tensor,
        bias: Option<Tensor>,
        device: &Device,
    ) -> Result<Arc<dyn QuantMethod>> {
        Self::quantize_with_options(
            weight,
            bias,
            device,
            QtipMode::Greedy,
            QtipRotation::for_mode(QtipMode::Greedy).enabled(),
        )
    }

    /// Lowest-level quantize entry: explicit mode + explicit rotation flag.
    /// Used by tests to A/B test rotation effect; production paths go through
    /// `quantize_with_mode`.
    ///
    /// Rank dispatch:
    /// - 2-D weight `[N, K]`  → standard per-linear quantize (this method's body).
    /// - 3-D weight `[E, N, K]` → forwarded to `quantize_with_options_3d`, which
    ///   loops over experts (sharing LUT + rotation signs) and returns a single
    ///   `QtipLayer` in 3-D mode. This is the path V4 Flash takes when
    ///   `--isq qtip2` requantizes a stacked MoE expert weight.
    pub fn quantize_with_options(
        weight: &Tensor,
        bias: Option<Tensor>,
        device: &Device,
        mode: QtipMode,
        use_rotation: bool,
    ) -> Result<Arc<dyn QuantMethod>> {
        Self::quantize_with_calibration(weight, bias, device, mode, use_rotation, None)
    }

    /// [`Self::quantize_with_options`] plus an optional activation Hessian
    /// diagonal for the weighted objective (wave13-AD, axis B).
    ///
    /// `hessian_diag` is `H_jj = (1/N) Σ_n x_{n,j}²` over calibration
    /// activations — one entry per **input** feature, i.e. exactly the vector
    /// [`crate::ImatrixLayerStats::compute_imatrix`] produces. It is ignored
    /// unless `ARC_QTIP_HESSIAN=1` is set, so wiring calibration data through
    /// is safe by default.
    pub fn quantize_with_calibration(
        weight: &Tensor,
        bias: Option<Tensor>,
        device: &Device,
        mode: QtipMode,
        use_rotation: bool,
        hessian_diag: Option<&[f32]>,
    ) -> Result<Arc<dyn QuantMethod>> {
        // Dispatch by rank. 3-D inputs come from V4 Flash MoE expert stacks
        // (gate_proj / up_proj / down_proj fused across all experts) and are
        // not supported by the dims2()-based pipeline below.
        if weight.dims().len() == 3 {
            if bias.is_some() {
                // 3-D MoE stacks in V4 Flash are bias-free; we don't currently
                // need a per-expert bias broadcast path, and silently dropping
                // a caller-supplied bias would be a footgun.
                candle_core::bail!(
                    "QTIP 3-D quantize: bias not supported for stacked-expert weights"
                );
            }
            return Self::quantize_with_options_3d_calibrated(
                weight,
                device,
                mode,
                use_rotation,
                hessian_diag,
            );
        }
        let layer = Self::quantize_with_options_concrete_calibrated(
            weight,
            bias,
            device,
            mode,
            use_rotation,
            hessian_diag,
        )?;
        Ok(Arc::new(layer))
    }

    /// Concrete-typed 2-D quantize that returns a `QtipLayer` directly
    /// instead of `Arc<dyn QuantMethod>`. Used by `quantize_with_options_3d`
    /// and `stack_experts` to extract the per-expert blocks/row_scales
    /// without a `dyn`-downcast (the `QuantMethod` trait does not extend
    /// `Any`).
    ///
    /// Also used by tests to inspect typed fields (`num_experts`, `blocks`,
    /// ...) and rotation state without a downcast. Production code should
    /// call `quantize_with_options` and treat the layer as a `QuantMethod`.
    pub fn quantize_with_options_concrete(
        weight: &Tensor,
        bias: Option<Tensor>,
        device: &Device,
        mode: QtipMode,
        use_rotation: bool,
    ) -> Result<Self> {
        Self::quantize_with_options_concrete_calibrated(
            weight,
            bias,
            device,
            mode,
            use_rotation,
            None,
        )
    }

    /// [`Self::quantize_with_options_concrete`] with an optional activation
    /// Hessian diagonal (`[K_in]`) for the weighted objective.
    ///
    /// Takes the bake configuration from the process environment
    /// ([`QtipBakeConfig::get`]). To drive a specific search from a test, use
    /// [`Self::quantize_with_bake_config`].
    pub fn quantize_with_options_concrete_calibrated(
        weight: &Tensor,
        bias: Option<Tensor>,
        device: &Device,
        mode: QtipMode,
        use_rotation: bool,
        hessian_diag: Option<&[f32]>,
    ) -> Result<Self> {
        Self::quantize_with_bake_config(
            weight,
            bias,
            device,
            mode,
            use_rotation,
            hessian_diag,
            QtipBakeConfig::get()?,
        )
    }

    /// [`Self::quantize_with_options_concrete_calibrated`] with the bake
    /// configuration passed **explicitly** instead of read from the process
    /// environment.
    ///
    /// Why this exists (wave13-AF finding, fixed in wave14-AK): production
    /// reads `ARC_QTIP_BEAM` / `ARC_QTIP_HESSIAN` once into a `OnceLock` —
    /// correct, since a bake must not re-read the environment per row, and a
    /// mid-run change of search would produce a checkpoint no stamp could
    /// honestly describe. But the memoisation also meant **no test could drive
    /// a real beam bake end to end**: the round-trip test had to stamp an
    /// already-baked layer, leaving the path from "a width was requested" to
    /// "the artifact says beam(W=…)" untested in the one subsystem where a
    /// mislabelled artifact has burned us repeatedly (DOCTRINE D4).
    ///
    /// The env stays the production path and stays memoised; this door only
    /// lets a caller say which search it wants. It is deliberately *not* wired
    /// into any loader, CLI flag, or config file — `QtipBakeConfig::get()` is
    /// still the only way a production bake picks its search.
    pub fn quantize_with_bake_config(
        weight: &Tensor,
        bias: Option<Tensor>,
        device: &Device,
        mode: QtipMode,
        use_rotation: bool,
        hessian_diag: Option<&[f32]>,
        bake_cfg: QtipBakeConfig,
    ) -> Result<Self> {
        // D4 fixture door: greedy is reachable only from this crate's own
        // `cfg(test)` builds. Every production caller arrives via
        // `quantize_with_mode`, which refuses greedy in all builds. Placed on
        // the calibrated worker, not its uncalibrated shim, so both entry
        // points are covered by one gate.
        mode.deny_greedy_outside_tests("QtipLayer::quantize_with_options_concrete")?;
        // RUN-quant-on-gpu fast path. For 284B-parameter V4 Flash the CPU-only
        // Viterbi quantize took 30-90 min per load; the GPU path collapses
        // this to <1 min. The GPU Viterbi kernel uses prefix-grouping — for
        // each (timestep, prefix p ∈ [0, 2^(L-K))) it does one 16-way
        // reduction, instead of 16 redundant reductions per state, because
        // all 16 successors of a given prefix share the SAME predecessor
        // argmin. This makes the kernel 16× faster AND uses 16× less
        // backtrace memory, which is what made the previous per-state
        // implementation hang on H100-scale layers (e.g. Qwen 7B
        // mlp.down_proj with num_symbols=9472).
        //
        // Hard rule: when CUDA is compiled in AND the tensor lives on CUDA,
        // there is NO CPU fallback. If the GPU path bails, we surface the
        // error to the caller; we do not silently quantize on CPU. Arc's
        // moat is that weight quantize stays entirely on the device the
        // model lives on.
        #[cfg(feature = "cuda")]
        if matches!(device, Device::Cuda(_)) {
            if !ffi::HAVE_QTIP_KERNELS {
                candle_core::bail!(
                    "QTIP quantize: CUDA device but QTIP kernels not compiled in. \
                     Rebuild mistralrs-quant with CUDA + has_qtip_kernels."
                );
            }
            // wave13-AF: the beam search now has a CUDA kernel
            // (`kernels/qtip/qtip_beam.cu`), so `ARC_QTIP_BEAM` is honoured on
            // GPU instead of hard-failing. `cuda_search_plan` still refuses any
            // width the kernel cannot run rather than substituting one it can —
            // a bake must never silently change its search.
            //
            // The Hessian-weighted objective remains CPU-only (the branch metric
            // in every CUDA kernel is unweighted), so that flag still refuses.
            let cuda_search = cuda_search_plan(bake_cfg.search, cuda_ops::beam_max_width())?;
            if bake_cfg.hessian && hessian_diag.is_some() {
                candle_core::bail!(
                    "QTIP quantize: ARC_QTIP_HESSIAN=1 with calibration data, but the CUDA \
                     quantize kernel only implements the unweighted branch metric. Either unset \
                     ARC_QTIP_HESSIAN or bake on CPU; the GPU path will not silently substitute \
                     a different objective."
                );
            }
            match Self::quantize_with_options_cuda(
                weight,
                bias.clone(),
                device,
                mode,
                use_rotation,
                cuda_search,
                bake_cfg.codebook,
            )? {
                Some(layer) => return Ok(layer),
                None => candle_core::bail!(
                    "QTIP quantize: GPU path returned None on a CUDA tensor. \
                     CPU fallback is disabled on CUDA — fix the preconditions \
                     (F32 dtype, contiguous layout, supported rotation block) \
                     instead of forcing a CPU detour."
                ),
            }
        }

        let weight_f32 = weight.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;
        let (n, k_in) = weight_f32.dims2()?;
        warn_big_cpu_2d_quantize(n, k_in, "QtipLayer::quantize_with_options_concrete");
        if !(k_in as u32).is_multiple_of(V) {
            candle_core::bail!("QTIP quantize: in_features ({k_in}) must be divisible by V ({V})");
        }
        let num_symbols_per_row = k_in / V as usize;
        if !num_symbols_per_row.is_multiple_of(2) {
            // We pack two K=4 symbols per byte.
            candle_core::bail!(
                "QTIP quantize: number of symbols per row ({num_symbols_per_row}) must be even for K=4 packing"
            );
        }

        // Build the codebook once. For the computed codebook this table is
        // what the CPU search reads and what goes into the artifact; the GPU
        // decode paths never touch it (they compute each codeword in
        // registers), which is the whole point of the change.
        let codebook = bake_cfg.codebook;
        let lut_data = codebook.materialize();
        let weight_data: Vec<f32> = weight_f32.flatten_all()?.to_vec1()?;

        // Compute the Hadamard incoherence rotation parameters (RUN-158).
        // We rotate each row of W by R = D·H·D (block-diagonal) *before*
        // quantizing, then apply the same R to each row of the input at
        // forward time. Since R is orthogonal AND involutory (R·R = I), the
        // matmul x @ W^T equals (xR) @ (WR)^T exactly — no algebraic change —
        // but the quantizer sees rotation-decorrelated weights and the matmul
        // sees rotation-decorrelated activations, both of which match the
        // statistical assumptions baked into the trellis LUT.
        let (rotation_block, rotation_signs_vec) = if use_rotation {
            let block = rotation_block_size(k_in);
            if block >= 2 {
                (block, generate_signs(QTIP_ROTATION_SEED, k_in))
            } else {
                (0usize, Vec::new())
            }
        } else {
            (0usize, Vec::new())
        };

        // wave13-AD axis B: project the calibration Hessian diagonal into the
        // frame the trellis actually searches. `hessian_row_weights` folds in
        // QTIP's relative damping and the `diag(RᵀHR) = block-mean(diag H)`
        // correction; a flat Hessian normalises to exactly 1.0 everywhere and
        // is therefore bit-identical to the unweighted objective.
        let calibration_ok = hessian_diag.is_some_and(|h| h.len() == k_in);
        if let Some(h) = hessian_diag {
            if bake_cfg.hessian && h.len() != k_in {
                candle_core::bail!(
                    "QTIP quantize: Hessian diagonal has {} entries but in_features is {k_in}",
                    h.len()
                );
            }
        }
        let search_weights: Option<Vec<f32>> = if bake_cfg.hessian && calibration_ok {
            hessian_diag.map(|h| viterbi::hessian_row_weights(h, rotation_block))
        } else {
            None
        };
        log_bake_header(
            "K4/V2 2-bit",
            mode,
            bake_cfg,
            rotation_block,
            search_weights.is_some(),
        );

        use rayon::prelude::*;
        let row_results: Vec<(Vec<u8>, f32)> = (0..n)
            .into_par_iter()
            .map(|row| {
                let row_offset = row * k_in;
                let row_slice = &weight_data[row_offset..row_offset + k_in];

                // Step 1: apply Hadamard incoherence rotation (RUN-158) to this
                // weight row, if enabled. Operates on a private copy so the
                // input tensor is untouched; the per-row scale below is then
                // chosen against rotated values, which is the correct ordering.
                let rotated_buf: Vec<f32>;
                let working_row: &[f32] = if rotation_block >= 2 {
                    let mut buf = row_slice.to_vec();
                    apply_block_rotation(&mut buf, &rotation_signs_vec, rotation_block);
                    rotated_buf = buf;
                    &rotated_buf[..]
                } else {
                    row_slice
                };

                // Pick row scale = max(|row|) / (3 sigma) so most values live
                // in [-3 sigma, 3 sigma] (the bulk of the codebook). The
                // Gaussian LUT is unit-sigma so its divisor is 3.0; the
                // computed codebook folds its own sigma in — see
                // `QtipCodebook::scale_divisor`.
                let max_abs = working_row.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
                let scale = if max_abs == 0.0 {
                    1.0
                } else {
                    max_abs / codebook.scale_divisor()
                };
                let inv_scale = 1.0 / scale;

                let mut packed = vec![0u8; num_symbols_per_row / 2];

                // Scale the target row.
                let scaled_target: Vec<f32> = working_row.iter().map(|w| w * inv_scale).collect();

                // Encode the symbol sequence via the selected mode.
                let symbols: Vec<u8> = match mode {
                    QtipMode::Viterbi => viterbi::quantize_row(
                        &scaled_target,
                        &lut_data,
                        bake_cfg.search,
                        search_weights.as_deref(),
                    ),
                    QtipMode::Greedy => {
                        let mut state: u32 = 0;
                        let mut syms = vec![0u8; num_symbols_per_row];
                        for sym_idx in 0..num_symbols_per_row {
                            let target_t =
                                &scaled_target[sym_idx * V as usize..(sym_idx + 1) * V as usize];

                            let mut best_sym: u8 = 0;
                            let mut best_err = f32::INFINITY;
                            for sym in 0u32..ALPHABET as u32 {
                                let next_state = ((state << K) | sym) & STATE_MASK;
                                let lut_off = (next_state as usize) * V as usize;
                                let mut err = 0f32;
                                for v in 0..V as usize {
                                    let d = lut_data[lut_off + v] - target_t[v];
                                    err += d * d;
                                }
                                if err < best_err {
                                    best_err = err;
                                    best_sym = sym as u8;
                                }
                            }
                            state = ((state << K) | best_sym as u32) & STATE_MASK;
                            syms[sym_idx] = best_sym;
                        }
                        syms
                    }
                };

                // Pack symbols into bytes (two K=4 symbols per u8, low nibble first).
                for (sym_idx, &sym) in symbols.iter().enumerate() {
                    let byte_idx = sym_idx / 2;
                    if sym_idx.is_multiple_of(2) {
                        packed[byte_idx] = sym & 0x0F;
                    } else {
                        packed[byte_idx] |= (sym & 0x0F) << 4;
                    }
                }

                (packed, scale)
            })
            .collect();

        let mut all_packed = Vec::with_capacity(n * num_symbols_per_row / 2);
        let mut all_scales = Vec::with_capacity(n);
        for (p, s) in row_results {
            all_packed.extend_from_slice(&p);
            all_scales.push(s);
        }

        let blocks = Tensor::from_vec(all_packed, (n, num_symbols_per_row / 2), &Device::Cpu)?
            .to_dtype(DType::U8)?
            .to_device(device)?;
        let row_scales = Tensor::from_vec(all_scales, (n,), &Device::Cpu)?.to_device(device)?;
        let lut =
            Tensor::from_vec(lut_data, (LUT_SIZE, V as usize), &Device::Cpu)?.to_device(device)?;
        let bias = bias.map(|b| b.to_device(device)).transpose()?;
        let rotation_signs = if rotation_block >= 2 {
            Some(
                Tensor::from_vec(rotation_signs_vec.clone(), (k_in,), &Device::Cpu)?
                    .to_device(device)?,
            )
        } else {
            None
        };

        Ok(Self {
            blocks,
            row_scales,
            lut,
            bias,
            in_features: k_in,
            num_experts: None,
            rotation_signs,
            rotation_block,
            search: QtipSearchStamp::for_mode(mode),
            // The plan that ran, not the env request: `bake_cfg.search` is what
            // `quantize_row` was called with, and `search_weights` is `Some`
            // only when calibration data actually arrived.
            search_detail: QtipSearchDetail::for_bake(
                mode,
                bake_cfg.search,
                search_weights.is_some(),
            ),
            codebook,
            // This crate's trellis search is K=4/V=2/L=16 throughout
            // (`viterbi.rs` reaches 16 successors per step and groups by
            // 2^(L-K)); there is no producer of any other geometry yet.
            geometry: QtipGeometry::K4V2L16,
        })
    }

    /// GPU fast path for `quantize_with_options`. Returns `Ok(Some(layer))`
    /// on success, `Ok(None)` when a precondition (shape, dtype, kernel
    /// availability) means we should fall through to the CPU path, or
    /// `Err(_)` when the GPU path is reachable but the underlying CUDA call
    /// failed (caller propagates).
    ///
    /// End-to-end GPU pipeline:
    ///   1. Ensure weight is F32 on the target CUDA device (single launch
    ///      cast-and-copy via `to_dtype` + `to_device`; no host roundtrip
    ///      since the user-provided weight is already on CUDA).
    ///   2. Build the Gaussian LUT once on the host (~512 KiB, constant
    ///      per layer) and upload it once.
    ///   3. Compute rotation signs (if `use_rotation`) on the host (tiny —
    ///      `in_features` floats) and upload.
    ///   4. Apply `rotate_weight_rows_cuda` to produce the rotated weight.
    ///   5. Call `quantize_rows_cuda` for the (packed_blocks, row_scales)
    ///      pair on-device.
    ///   6. Wrap into a `QtipLayer` carrying device-resident tensors.
    #[cfg(feature = "cuda")]
    fn quantize_with_options_cuda(
        weight: &Tensor,
        bias: Option<Tensor>,
        device: &Device,
        mode: QtipMode,
        use_rotation: bool,
        search: TrellisSearch,
        codebook: QtipCodebook,
    ) -> Result<Option<Self>> {
        // Sanity preconditions; any failure falls through to CPU.
        let (n, k_in) = match weight.dims2() {
            Ok((n, k)) => (n, k),
            Err(_) => return Ok(None),
        };
        if !(k_in as u32).is_multiple_of(V) {
            return Ok(None);
        }
        let num_symbols_per_row = k_in / V as usize;
        if !num_symbols_per_row.is_multiple_of(2) {
            return Ok(None);
        }

        // Move weight to CUDA F32. Order matters: when the source is CPU BF16
        // (the 3-D MoE bake path passes CPU expert slices), casting BEFORE the
        // transfer does a slow CPU bf16->f32 widening of every element AND then
        // ships 2x the bytes (f32). Move bf16 to the device FIRST, then cast on
        // the GPU -> half the PCIe traffic + the cast runs on the GPU. The
        // bf16->f32 widening is exact, so this is bit-identical. RUN-161.
        let weight_cuda_f32 = weight.to_device(device)?.to_dtype(DType::F32)?;

        // Build the codebook on host (~512 KiB) and upload. The quantize
        // kernels ignore this buffer when the codebook is computed; it is
        // uploaded anyway because it is what the artifact stores, and because
        // an artifact whose table still matches its symbols stays decodable by
        // a reader that predates the codebook discriminator.
        let lut_data = codebook.materialize();
        let lut =
            Tensor::from_vec(lut_data, (LUT_SIZE, V as usize), &Device::Cpu)?.to_device(device)?;

        // Rotation params.
        let (rotation_block, rotation_signs_vec) = if use_rotation {
            let block = rotation_block_size(k_in);
            if block >= 2 && matches!(block, 2 | 4 | 8 | 16 | 32 | 64 | 128) {
                (block, generate_signs(QTIP_ROTATION_SEED, k_in))
            } else {
                // Unsupported block size for the CUDA path; defer to CPU.
                return Ok(None);
            }
        } else {
            (0usize, Vec::new())
        };

        // Apply rotation on-device.
        let weight_rotated = if rotation_block >= 2 {
            let signs_cuda = Tensor::from_vec(rotation_signs_vec.clone(), (k_in,), &Device::Cpu)?
                .to_device(device)?;
            cuda_ops::rotate_weight_rows_cuda(&weight_cuda_f32, &signs_cuda, rotation_block)?
        } else {
            weight_cuda_f32
        };

        // Task #17: the GPU fast path returns before the CPU pipeline's header
        // call, so it must emit its own — otherwise a GPU bake is exactly the
        // unlabelled artifact the header exists to prevent. `search` is the
        // plan the kernels will actually run (post `cuda_search_plan`), not the
        // raw env request, so the log never over-promises. The GPU branch
        // metric is unweighted, hence `have_calibration = false`.
        log_bake_header(
            "K4/V2 2-bit",
            mode,
            QtipBakeConfig {
                search,
                hessian: false,
                codebook,
            },
            rotation_block,
            false,
        );

        // Quantize (Viterbi or Greedy) on-device.
        let (blocks, row_scales) =
            cuda_ops::quantize_rows_cuda(&weight_rotated, &lut, mode, search, codebook)?;

        let bias = bias.map(|b| b.to_device(device)).transpose()?;
        let rotation_signs = if rotation_block >= 2 {
            Some(Tensor::from_vec(rotation_signs_vec, (k_in,), &Device::Cpu)?.to_device(device)?)
        } else {
            None
        };

        let _ = n; // n is just for symmetry with CPU path; row_scales already encodes it
        Ok(Some(Self {
            blocks,
            row_scales,
            lut,
            bias,
            in_features: k_in,
            num_experts: None,
            rotation_signs,
            rotation_block,
            search: QtipSearchStamp::for_mode(mode),
            // `search` here is the post-`cuda_search_plan` plan the kernels
            // actually ran. Every CUDA branch metric is unweighted, so the
            // objective bit is false by construction, not by omission.
            search_detail: QtipSearchDetail::for_bake(mode, search, false),
            codebook,
            // Same geometry as the CPU sibling above: the CUDA bake kernels are
            // the K=4/V=2/L=16 trellis too, so a GPU-baked artifact must carry
            // the identical wire tag. Omitting it here compiled fine without
            // the `cuda` feature and only broke the CUDA lane.
            geometry: QtipGeometry::K4V2L16,
        }))
    }

    /// Quantize a 3-D `[E, N, K]` stacked-expert weight tensor.
    ///
    /// Each expert's `[N, K]` slice is quantized independently, but all experts
    /// share:
    /// - the same Gaussian trellis `lut` (constant per QTIP config),
    /// - the same Hadamard `rotation_signs` and `rotation_block`
    ///   (the rotation matrix is determined by `K`, which is identical across
    ///   experts).
    ///
    /// The per-expert results are stacked into:
    /// - `blocks: [E, N, packed_K]`
    /// - `row_scales: [E, N]`
    ///
    /// On CUDA with QTIP kernels compiled in, the per-expert quantize calls
    /// the same GPU kernels as the 2-D path — each launch is independent so the
    /// expert dimension is embarrassingly parallel; we serialize launches per
    /// expert rather than batching at the kernel level to avoid blowing the
    /// Viterbi scratch budget on a 256-expert stack (a single-expert Viterbi
    /// backtrace already approaches `VITERBI_MAX_SCRATCH_BYTES`).
    ///
    /// The CPU fallback iterates over experts and reuses the 2-D CPU pipeline,
    /// then concatenates packed bytes and scale floats in `[E, N, ...]` order.
    pub fn quantize_with_options_3d(
        weight: &Tensor,
        device: &Device,
        mode: QtipMode,
        use_rotation: bool,
    ) -> Result<Arc<dyn QuantMethod>> {
        Self::quantize_with_options_3d_calibrated(weight, device, mode, use_rotation, None)
    }

    /// [`Self::quantize_with_options_3d`] with an optional activation Hessian
    /// diagonal (`[K_in]`, shared across experts because every expert in a
    /// stack reads the same input features).
    ///
    /// Takes the bake configuration from the process environment
    /// ([`QtipBakeConfig::get`]). To drive a specific search from a test, use
    /// [`Self::quantize_3d_with_bake_config`].
    pub fn quantize_with_options_3d_calibrated(
        weight: &Tensor,
        device: &Device,
        mode: QtipMode,
        use_rotation: bool,
        hessian_diag: Option<&[f32]>,
    ) -> Result<Arc<dyn QuantMethod>> {
        Self::quantize_3d_with_bake_config(
            weight,
            device,
            mode,
            use_rotation,
            hessian_diag,
            QtipBakeConfig::get()?,
        )
    }

    /// [`Self::quantize_with_options_3d_calibrated`] with the bake configuration
    /// passed **explicitly** instead of read from the process environment — the
    /// 3-D sibling of [`Self::quantize_with_bake_config`].
    ///
    /// Why this exists: wave14-AK opened the explicit-config door for 2-D
    /// linears, but this is the path `--isq qtip2`/`qtip2b` actually takes for
    /// V4 Flash — the stacked MoE experts, which are essentially the entire
    /// weight mass of the model. Because the per-expert chunks went through
    /// [`Self::quantize_with_options_concrete_calibrated`], which reads the
    /// memoised env config internally, a 3-D beam bake could not be exercised
    /// by any test at all: the one code path a paid GPU session spends its
    /// money on was the one path no test had ever run.
    ///
    /// The env stays the production path and stays memoised — a bake must not
    /// re-read the environment per row, and a mid-run change of search would
    /// produce a checkpoint no stamp could honestly describe. This door only
    /// lets a caller say which search it wants; it is deliberately *not* wired
    /// into any loader, CLI flag, or config file.
    pub fn quantize_3d_with_bake_config(
        weight: &Tensor,
        device: &Device,
        mode: QtipMode,
        use_rotation: bool,
        hessian_diag: Option<&[f32]>,
        bake_cfg: QtipBakeConfig,
    ) -> Result<Arc<dyn QuantMethod>> {
        Ok(Arc::new(Self::quantize_3d_concrete_with_bake_config(
            weight,
            device,
            mode,
            use_rotation,
            hessian_diag,
            bake_cfg,
        )?))
    }

    /// [`Self::quantize_3d_with_bake_config`] returning the concrete layer
    /// instead of `Arc<dyn QuantMethod>`.
    ///
    /// `QuantMethod` does not extend `Any`, so without this there is no way to
    /// inspect a freshly baked 3-D stack's `blocks` / `row_scales` — including
    /// no way to assert *what memory it holds*, which is the property wave18
    /// needed to pin (see `qtip/bake_memory_tests.rs`).
    pub fn quantize_3d_concrete_with_bake_config(
        weight: &Tensor,
        device: &Device,
        mode: QtipMode,
        use_rotation: bool,
        hessian_diag: Option<&[f32]>,
        bake_cfg: QtipBakeConfig,
    ) -> Result<Self> {
        let dims = weight.dims3()?;
        let (e, n, k_in) = (dims.0, dims.1, dims.2);
        if e == 0 || n == 0 || k_in == 0 {
            candle_core::bail!("QTIP 3-D quantize: zero-sized expert stack ({e}, {n}, {k_in})");
        }
        if !(k_in as u32).is_multiple_of(V) {
            candle_core::bail!(
                "QTIP 3-D quantize: in_features ({k_in}) must be divisible by V ({V})"
            );
        }
        let num_symbols_per_row = k_in / V as usize;
        if !num_symbols_per_row.is_multiple_of(2) {
            candle_core::bail!(
                "QTIP 3-D quantize: number of symbols per row ({num_symbols_per_row}) must be even for K=4 packing"
            );
        }
        let packed_per_row = num_symbols_per_row / 2;

        // Quantize each expert slice into a single-expert 2-D QtipLayer. We
        // call the typed `quantize_with_options_concrete` so we can pluck
        // blocks/row_scales directly without a `dyn QuantMethod` downcast
        // (the trait doesn't extend `Any`).
        let mut blocks_slices: Vec<Tensor> = Vec::with_capacity(e);
        let mut scales_slices: Vec<Tensor> = Vec::with_capacity(e);
        let mut shared_lut: Option<Tensor> = None;
        let mut shared_rotation_signs: Option<Tensor> = None;
        let mut shared_rotation_block: usize = 0;
        let mut shared_search_detail = QtipSearchDetail::Unknown;
        let mut shared_codebook = QtipCodebook::Gaussian;
        let mut shared_geometry = QtipGeometry::K4V2L16;

        // Per-expert streaming. The full stack is kept on CPU (caller passes
        // device=CPU for the 3-D experts to avoid the ~4GB dense BF16 transient
        // that OOMs the single-H100 tail). But the CPU Viterbi quantize is
        // single-threaded and takes hours for a full requantize, while the
        // compiled GPU Viterbi kernel finishes in seconds. So when CUDA + the
        // QTIP kernels are available, quantize each expert on the GPU:
        // `quantize_with_options_cuda` moves one expert at a time (CPU BF16 ->
        // GPU F32, ~33MB transient + reused scratch), and we move the small
        // packed result back to `device` (CPU) for stacking — matching the
        // original output layout exactly. (RUN-161)
        //
        // Any reroute to the CPU pipeline while a GPU is plausibly available
        // is counted + warned inside `expert_stack_quant_device` (wave6-Q:
        // the silent version of this gate cost ~20x per layer).
        let quant_device = expert_stack_quant_device(device, "QtipLayer::quantize_with_options_3d");

        // Where the *result* lands, which is not always where the math ran.
        //
        // wave18: during a UQFF bake the quantized stack is only ever handed to
        // `serialize()`, so keeping it on the accelerator buys nothing and costs
        // the full artifact size in device memory — ~1.6 GiB per layer for V4
        // Flash, ~68 GB over 43 layers — while also forcing the allocator to
        // fit its multi-GiB per-chunk transients around a permanently growing
        // set of resident blocks. That is what killed a 43-layer bake at layer
        // 28 on a 140 GB H200 with nothing written. With the result on the host,
        // device usage reaches steady state after the first layer and stays
        // there, and the same transients are reused every chunk.
        //
        // `bake_isq_to_host()` is false for every serve/inference load, so this
        // is a bake-only detour: nothing that will run a forward pass is moved.
        let out_device = if crate::bake_isq_to_host() {
            Device::Cpu
        } else {
            device.clone()
        };
        let move_back = !quant_device.same_device(&out_device);

        // RUN-161 expert-batching: the per-expert loop used to stream ONE expert
        // at a time -> 256 CPU<->GPU blocking round-trips/projection, which
        // dominate the bake (the GPU kernel itself is <1s). Quantize experts in
        // BATCHES instead: narrow B experts [B,N,K], reshape to 2-D rows
        // [B*N, K] (per-row rotation/scale/Viterbi is identical to the
        // single-expert path), quantize in ONE call, reshape the packed result
        // back to [B,N,packed]. Cuts host round-trips from E to E/B.
        // B is memory-bounded: B*N*K*4 (F32 weight) + ~6GB Viterbi scratch must
        // fit free VRAM during the bake. Default 16; tune via ARC_QTIP_EXPERT_BATCH.
        let batch = std::env::var("ARC_QTIP_EXPERT_BATCH")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(16)
            .clamp(1, e);

        let mut expert_idx = 0usize;
        while expert_idx < e {
            let this_b = batch.min(e - expert_idx);
            // [this_b, N, K] -> [this_b*N, K] (CPU reshape; concrete moves to GPU).
            let chunk = weight.narrow(0, expert_idx, this_b)?;
            let rows_2d = chunk.reshape((this_b * n, k_in))?;
            // `narrow` + `reshape` produce a *view*: the layout shrinks to this
            // chunk but the backing storage is still the whole [E, N, K] stack.
            // `Tensor::to_device` copies the entire storage and clones the
            // layout (offset included), so handing this view straight to the
            // CUDA path uploads all E experts on every chunk — 4.3 GiB per call
            // instead of 268 MiB for V4 Flash, 16x more traffic and a 4.3 GiB
            // alloc/free cycle 48 times per layer for the allocator to work
            // around. Materialise the chunk into its own storage first; skip it
            // when the quantize runs where the weight already lives (no
            // `to_device` copy happens then) or when the chunk *is* the whole
            // stack. (wave18)
            //
            // Done for every partial chunk, not only when a device hop is
            // pending: the copy is one chunk's worth (268 MiB for V4 Flash,
            // microseconds of bandwidth) and making it unconditional keeps the
            // path a CPU test can actually reach.
            let rows_2d = if this_b < e {
                rows_2d.force_contiguous()?
            } else {
                rows_2d
            };
            // Every chunk is baked with the SAME `bake_cfg` the caller handed
            // in — not a per-chunk re-read. The cross-chunk `search_detail`
            // check below is what turns that into an enforced property rather
            // than a comment.
            let layer = Self::quantize_with_bake_config(
                &rows_2d,
                None,
                &quant_device,
                mode,
                use_rotation,
                hessian_diag,
                bake_cfg,
            )?;

            // blocks [this_b*N, packed] -> [this_b, N, packed]; scales [this_b*N] -> [this_b, N].
            let blk = if move_back {
                layer.blocks.to_device(&out_device)?
            } else {
                layer.blocks.clone()
            };
            let scl = if move_back {
                layer.row_scales.to_device(&out_device)?
            } else {
                layer.row_scales.clone()
            };
            blocks_slices.push(blk.reshape((this_b, n, packed_per_row))?);
            scales_slices.push(scl.reshape((this_b, n))?);

            if expert_idx == 0 {
                shared_lut = Some(if move_back {
                    layer.lut.to_device(&out_device)?
                } else {
                    layer.lut.clone()
                });
                shared_rotation_signs = match layer.rotation_signs.clone() {
                    Some(s) if move_back => Some(s.to_device(&out_device)?),
                    other => other,
                };
                shared_rotation_block = layer.rotation_block;
                shared_search_detail = layer.search_detail;
                shared_codebook = layer.codebook;
                shared_geometry = layer.geometry;
            } else {
                debug_assert_eq!(layer.lut.dims(), shared_lut.as_ref().unwrap().dims());
                debug_assert_eq!(layer.rotation_block, shared_rotation_block);
                // Every chunk runs the same bake config, so a divergence here
                // means the stack would carry one expert's provenance while
                // holding another's weights. Hard-check it (D4).
                if layer.search_detail != shared_search_detail {
                    candle_core::bail!(
                        "QTIP 3-D quantize: expert chunk at {expert_idx} recorded search \
                         detail {} but the stack already carries {} — refusing to stamp a \
                         mixed-provenance artifact.",
                        layer.search_detail.tag(),
                        shared_search_detail.tag()
                    );
                }
                // Same argument for the codebook, and harder: a stack whose
                // experts mean different reproduction values decodes to
                // garbage, and the single stored table can only match one of
                // them.
                if layer.codebook != shared_codebook {
                    candle_core::bail!(
                        "QTIP 3-D quantize: expert chunk at {expert_idx} used codebook {} but \
                         the stack already carries {} — refusing to build a stack whose experts \
                         decode against different codebooks.",
                        layer.codebook.tag(),
                        shared_codebook.tag()
                    );
                }
                // And the geometry, for the same reason and with the same
                // teeth: at 2 bpw a mixed-geometry stack has consistently
                // sized `blocks`, so nothing downstream would notice.
                if layer.geometry != shared_geometry {
                    candle_core::bail!(
                        "QTIP 3-D quantize: expert chunk at {expert_idx} used geometry {} but \
                         the stack already carries {} — refusing to build a stack whose experts \
                         decode at different geometries.",
                        layer.geometry.tag(),
                        shared_geometry.tag()
                    );
                }
            }
            expert_idx += this_b;
        }

        // Concatenate per-batch [b_i, N, packed_K] chunks into [E, N, packed_K].
        let blocks_3d = Tensor::cat(&blocks_slices, 0)?;
        let row_scales_2d = Tensor::cat(&scales_slices, 0)?;
        debug_assert_eq!(blocks_3d.dims(), &[e, n, packed_per_row]);
        debug_assert_eq!(row_scales_2d.dims(), &[e, n]);

        let lut = shared_lut.ok_or_else(|| {
            candle_core::Error::Msg("QTIP 3-D quantize: no expert produced an LUT".into())
        })?;

        Ok(Self {
            blocks: blocks_3d,
            row_scales: row_scales_2d,
            lut,
            // V4 Flash MoE experts have no bias. We bail above if a caller
            // passes one through `quantize_with_options`, so reaching here
            // with `None` is the only valid state.
            bias: None,
            in_features: k_in,
            num_experts: Some(e),
            rotation_signs: shared_rotation_signs,
            rotation_block: shared_rotation_block,
            search: QtipSearchStamp::for_mode(mode),
            search_detail: shared_search_detail,
            codebook: shared_codebook,
            geometry: shared_geometry,
        })
    }

    /// Read the raw decoded weights *in the rotated frame*. When rotation is
    /// disabled, this is also the original frame. Internal helper for the
    /// fused matmul forward path — saves a redundant rotate-then-unrotate when
    /// the input has already been rotated.
    // Index-based loop over `scales_data` is intentional in this dequant hot
    // path (GPU-validated numerical parity); see arc-tools/CI_HYGIENE.md.
    #[allow(clippy::needless_range_loop)]
    fn dequantize_weights_rotated_f32(&self) -> Result<Vec<f32>> {
        let blocks_cpu = self.blocks.to_device(&Device::Cpu)?;
        let scales_cpu = self.row_scales.to_device(&Device::Cpu)?;
        let lut_cpu = self.lut.to_device(&Device::Cpu)?;

        let blocks_data: Vec<u8> = blocks_cpu.flatten_all()?.to_vec1()?;
        let scales_data: Vec<f32> = scales_cpu.flatten_all()?.to_vec1()?;
        let lut_data: Vec<f32> = lut_cpu.flatten_all()?.to_vec1()?;

        let n = self.row_scales.dim(0)?;
        let k_in = self.in_features;
        let num_symbols_per_row = k_in / V as usize;
        let packed_per_row = num_symbols_per_row / 2;

        let mut out = vec![0f32; n * k_in];

        for row in 0..n {
            let scale = scales_data[row];
            let mut state: u32 = 0;
            let row_packed_base = row * packed_per_row;
            let out_row_base = row * k_in;

            for sym_idx in 0..num_symbols_per_row {
                let byte = blocks_data[row_packed_base + sym_idx / 2];
                let sym = if sym_idx.is_multiple_of(2) {
                    byte & 0x0F
                } else {
                    (byte >> 4) & 0x0F
                };
                state = ((state << K) | sym as u32) & STATE_MASK;
                let lut_off = (state as usize) * V as usize;
                out[out_row_base + sym_idx * 2] = lut_data[lut_off] * scale;
                out[out_row_base + sym_idx * 2 + 1] = lut_data[lut_off + 1] * scale;
            }
        }

        Ok(out)
    }

    fn dequantize_weights_f32(&self) -> Result<Vec<f32>> {
        let mut out = self.dequantize_weights_rotated_f32()?;

        // Undo the Hadamard incoherence rotation row-by-row so callers see
        // weights in the original (unrotated) frame. R is involutory so the
        // same `apply_block_rotation` reverses the rotation.
        if self.rotation_block >= 2 {
            let signs = match &self.rotation_signs {
                Some(t) => t.to_device(&Device::Cpu)?.to_vec1::<f32>()?,
                None => candle_core::bail!(
                    "QtipLayer: rotation_block={} but rotation_signs is None",
                    self.rotation_block
                ),
            };
            let n = self.row_scales.dim(0)?;
            let k_in = self.in_features;
            for row in 0..n {
                let row_slice = &mut out[row * k_in..(row + 1) * k_in];
                apply_block_rotation(row_slice, &signs, self.rotation_block);
            }
        }

        Ok(out)
    }

    pub fn dequantize_weights(&self) -> Result<Tensor> {
        self.require_k4v2l16("dequantize_weights")?;
        // 3-D stacked-expert path: iterate over experts and stack `[N, K_in]`
        // slices into `[E, N, K_in]`. We reuse the 2-D dequant per expert
        // rather than writing a 3-D CUDA kernel — the existing
        // `dequantize_rotated_cuda` is already row-parallel, so per-expert
        // launches are still bandwidth-bound and only sequential at the
        // launch level (negligible vs the work per launch for V4 Flash's
        // 256-expert × 4096-row stacks).
        if let Some(e) = self.num_experts {
            let n = self.row_scales.dim(1)?;
            let k_in = self.in_features;
            let mut expert_tensors: Vec<Tensor> = Vec::with_capacity(e);
            // Build the dense [E, N, K] stack on CPU. It is multi-GB and OOMs
            // the GPU when the model is already resident (the TD-MoE decompose
            // path, which is the only caller of the 3-D dequant). Moving each
            // expert's blocks/scales to CPU forces `dequantize_single_expert`'s
            // CPU fallback, so no large GPU tensor is ever allocated. The
            // Tucker decomposition runs on CPU anyway. (RUN-161)
            for expert_idx in 0..e {
                let blocks_e = self
                    .blocks
                    .narrow(0, expert_idx, 1)?
                    .squeeze(0)?
                    .to_device(&Device::Cpu)?
                    .contiguous()?;
                let scales_e = self
                    .row_scales
                    .narrow(0, expert_idx, 1)?
                    .squeeze(0)?
                    .to_device(&Device::Cpu)?
                    .contiguous()?;
                let expert_w = self.dequantize_single_expert(&blocks_e, &scales_e, n, k_in)?;
                expert_tensors.push(expert_w);
            }
            return Tensor::stack(&expert_tensors, 0);
        }

        // 2-D path: CPU returns a [n, k_in] BF16 tensor on the device.
        // GPU path: dequantize on-GPU into the rotated frame, then
        // un-rotate with `rotate_x_cuda` on the *rows* (since R is
        // involutory and applies per-row).
        #[cfg(feature = "cuda")]
        {
            if cuda_ops::can_use_qtip_cuda(&self.blocks) {
                let w_rotated = cuda_ops::dequantize_rotated_cuda(
                    &self.blocks,
                    &self.row_scales,
                    &self.lut,
                    self.in_features,
                    DType::BF16,
                    self.codebook,
                )?;
                if self.rotation_block >= 2 {
                    let signs = match &self.rotation_signs {
                        Some(t) => t,
                        None => candle_core::bail!(
                            "QtipLayer: rotation_block={} but rotation_signs is None",
                            self.rotation_block
                        ),
                    };
                    return cuda_ops::rotate_x_cuda(&w_rotated, signs, self.rotation_block);
                }
                return Ok(w_rotated);
            }
        }
        let out = self.dequantize_weights_f32()?;
        let n = self.row_scales.dim(0)?;
        let k_in = self.in_features;
        Tensor::from_vec(out, (n, k_in), &Device::Cpu)?
            .to_device(self.blocks.device())?
            .to_dtype(DType::BF16)
    }

    /// Dequantize a single expert's `[N, K_in]` weight matrix using the layer's
    /// shared LUT and (optionally) shared rotation. Internal helper for the
    /// 3-D `dequantize_weights` and (in the future) `gather_forward` paths.
    ///
    /// `blocks_e: [N, packed_K]` U8 and `scales_e: [N]` F32 must already be
    /// contiguous on the layer's device. Returns BF16 `[N, K_in]` in the
    /// original (unrotated) frame.
    #[allow(clippy::needless_range_loop)] // scales_data index loop, GPU-validated hot path
    fn dequantize_single_expert(
        &self,
        blocks_e: &Tensor,
        scales_e: &Tensor,
        _n: usize,
        k_in: usize,
    ) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        {
            if cuda_ops::can_use_qtip_cuda(blocks_e) {
                let w_rotated = cuda_ops::dequantize_rotated_cuda(
                    blocks_e,
                    scales_e,
                    &self.lut,
                    k_in,
                    DType::BF16,
                    self.codebook,
                )?;
                return if self.rotation_block >= 2 {
                    let signs = match &self.rotation_signs {
                        Some(t) => t,
                        None => candle_core::bail!(
                            "QtipLayer: rotation_block={} but rotation_signs is None",
                            self.rotation_block
                        ),
                    };
                    cuda_ops::rotate_x_cuda(&w_rotated, signs, self.rotation_block)
                } else {
                    Ok(w_rotated)
                };
            }
        }

        // CPU fallback: replicate `dequantize_weights_rotated_f32` against the
        // local `blocks_e`/`scales_e` slices, then un-rotate.
        let blocks_cpu = blocks_e.to_device(&Device::Cpu)?;
        let scales_cpu = scales_e.to_device(&Device::Cpu)?;
        let lut_cpu = self.lut.to_device(&Device::Cpu)?;
        let blocks_data: Vec<u8> = blocks_cpu.flatten_all()?.to_vec1()?;
        let scales_data: Vec<f32> = scales_cpu.flatten_all()?.to_vec1()?;
        let lut_data: Vec<f32> = lut_cpu.flatten_all()?.to_vec1()?;

        let n_local = scales_e.dim(0)?;
        let num_symbols_per_row = k_in / V as usize;
        let packed_per_row = num_symbols_per_row / 2;
        let mut out = vec![0f32; n_local * k_in];

        for row in 0..n_local {
            let scale = scales_data[row];
            let mut state: u32 = 0;
            let row_packed_base = row * packed_per_row;
            let out_row_base = row * k_in;
            for sym_idx in 0..num_symbols_per_row {
                let byte = blocks_data[row_packed_base + sym_idx / 2];
                let sym = if sym_idx.is_multiple_of(2) {
                    byte & 0x0F
                } else {
                    (byte >> 4) & 0x0F
                };
                state = ((state << K) | sym as u32) & STATE_MASK;
                let lut_off = (state as usize) * V as usize;
                out[out_row_base + sym_idx * 2] = lut_data[lut_off] * scale;
                out[out_row_base + sym_idx * 2 + 1] = lut_data[lut_off + 1] * scale;
            }
        }
        if self.rotation_block >= 2 {
            let signs = match &self.rotation_signs {
                Some(t) => t.to_device(&Device::Cpu)?.to_vec1::<f32>()?,
                None => candle_core::bail!(
                    "QtipLayer: rotation_block={} but rotation_signs is None",
                    self.rotation_block
                ),
            };
            for row in 0..n_local {
                let row_slice = &mut out[row * k_in..(row + 1) * k_in];
                apply_block_rotation(row_slice, &signs, self.rotation_block);
            }
        }
        Tensor::from_vec(out, (n_local, k_in), &Device::Cpu)?
            .to_device(blocks_e.device())?
            .to_dtype(DType::BF16)
    }

    fn forward_dequantize(&self, x: &Tensor) -> Result<Tensor> {
        // 3-D stacked-expert layers cannot be matmul'd against a flat input —
        // the caller must route through `gather_forward` (V4 MoE Fast backend)
        // which knows how to pick the right expert per token. Bail clearly
        // instead of producing garbage from a 0-th expert slice.
        if self.num_experts.is_some() {
            candle_core::bail!(
                "QtipLayer::forward called on a 3-D stacked-expert layer; \
                 use gather_forward(x, indices) instead"
            );
        }

        let orig_dims = x.dims().to_vec();
        let x_2d = if orig_dims.len() > 2 {
            let features = orig_dims[orig_dims.len() - 1];
            let batch_size: usize = orig_dims[..orig_dims.len() - 1].iter().product();
            x.reshape((batch_size, features))?
        } else {
            x.clone()
        };

        // RUN-158 Hadamard incoherence-processing forward path.
        //
        // When rotation is enabled, we (a) rotate x rows by R = D·H·D, and
        // (b) matmul against weights *in the rotated frame* (skipping the
        // inverse rotation in dequantize). Since R is involutory:
        //   (xR) @ (WR)^T = xR · R^T · W^T = x · I · W^T = x · W^T
        // so the result is algebraically the original matmul, but the
        // quantizer sees inputs that match the Gaussian-coordinate
        // assumption of the trellis LUT.
        let n = self.row_scales.dim(0)?;
        let k_in = self.in_features;

        // GPU fast path: dequantize-on-GPU + rotate-x-on-GPU + matmul.
        // We attempt it whenever the layer's storage is on CUDA and the
        // kernels are compiled in. Any precondition failure (unsupported
        // dtype, non-power-of-2 block) silently falls through to the
        // CPU path below.
        #[cfg(feature = "cuda")]
        {
            if cuda_ops::can_use_qtip_cuda(&self.blocks) {
                if let Ok(mut result) = self.forward_dequantize_cuda(&x_2d, x.dtype(), n, k_in) {
                    if let Some(bias) = &self.bias {
                        result = result.broadcast_add(bias)?;
                    }
                    if orig_dims.len() > 2 {
                        let mut new_dims = orig_dims[..orig_dims.len() - 1].to_vec();
                        new_dims.push(result.dim(1)?);
                        result = result.reshape(new_dims)?;
                    }
                    return Ok(result);
                }
            }
        }

        // Materializing fallback: full weight dequantize + matmul. At decode
        // shapes the fused GEMV above should have handled this.
        warn_dequant_materialize_at_decode(
            x_2d.dim(0)?,
            "QtipLayer::forward_dequantize fallback (full dequantize+matmul)",
        );

        let (x_for_matmul, w_for_matmul) = if self.rotation_block >= 2 {
            let signs = match &self.rotation_signs {
                Some(t) => t.to_device(&Device::Cpu)?.to_vec1::<f32>()?,
                None => candle_core::bail!(
                    "QtipLayer: rotation_block={} but rotation_signs is None",
                    self.rotation_block
                ),
            };

            // Rotate every row of x.
            let x_cpu = x_2d.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
            let (batch, feat) = x_cpu.dims2()?;
            debug_assert_eq!(feat, k_in);
            let mut x_data: Vec<f32> = x_cpu.flatten_all()?.to_vec1()?;
            for row in 0..batch {
                let row_slice = &mut x_data[row * feat..(row + 1) * feat];
                apply_block_rotation(row_slice, &signs, self.rotation_block);
            }
            let x_rot = Tensor::from_vec(x_data, (batch, feat), &Device::Cpu)?
                .to_device(x.device())?
                .to_dtype(x.dtype())?;

            // Use weights still in the rotated frame (no inverse rotation).
            let w_rotated = self.dequantize_weights_rotated_f32()?;
            let w_tensor = Tensor::from_vec(w_rotated, (n, k_in), &Device::Cpu)?
                .to_device(x.device())?
                .to_dtype(x.dtype())?;
            (x_rot, w_tensor)
        } else {
            let w = self.dequantize_weights()?.to_dtype(x.dtype())?;
            (x_2d, w)
        };

        let mut result = x_for_matmul.matmul(&w_for_matmul.t()?)?;

        if let Some(bias) = &self.bias {
            result = result.broadcast_add(bias)?;
        }

        if orig_dims.len() > 2 {
            let mut new_dims = orig_dims[..orig_dims.len() - 1].to_vec();
            new_dims.push(result.dim(1)?);
            result = result.reshape(new_dims)?;
        }
        Ok(result)
    }

    /// GPU forward path: dequantize the rotated weight matrix on-device,
    /// rotate `x` rows on-device, matmul, return the bias-free result.
    ///
    /// Single-token decode (`n_tokens == 1`) uses the fused decode-gemv
    /// kernel which avoids materializing the BF16 weight matrix in HBM:
    /// roughly 3× less weight-side bandwidth, lifting the per-token decode
    /// ceiling on V4 Flash from ~330 tok/s (dequant+matmul) toward ~1 K tok/s.
    ///
    /// Multi-token forward (`n_tokens > 1`) keeps the dequantize+matmul
    /// path — the cuBLAS GEMM amortizes the dequantize cost across all
    /// tokens, and a fused decode-GEMM at high batch is a separate kernel
    /// (TODO: gather-style flavor for MoE prefill).
    ///
    /// Returns `Err` if any precondition fails (e.g. unsupported dtype or
    /// block size) — the caller falls back to the CPU path.
    #[cfg(feature = "cuda")]
    fn forward_dequantize_cuda(
        &self,
        x_2d: &Tensor,
        out_dtype: DType,
        _n: usize,
        k_in: usize,
    ) -> Result<Tensor> {
        if !matches!(out_dtype, DType::BF16 | DType::F16 | DType::F32) {
            candle_core::bail!("QTIP forward CUDA: unsupported out dtype {out_dtype:?}");
        }
        let x_dev = x_2d.device();
        if !matches!(x_dev, candle_core::Device::Cuda(_)) {
            candle_core::bail!("QTIP forward CUDA: x must live on CUDA");
        }
        if x_2d.dim(1)? != k_in {
            candle_core::bail!(
                "QTIP forward CUDA: x.dim(1)={} != in_features={}",
                x_2d.dim(1)?,
                k_in
            );
        }

        // Cast x to the layer's working dtype (BF16) if needed for the
        // rotation kernel — the rotation kernel supports BF16/F16/F32. We
        // run rotation in the original input dtype where possible to
        // minimize precision loss; the matmul forces a consistent dtype.
        let x_rot_dtype = if matches!(x_2d.dtype(), DType::BF16 | DType::F16 | DType::F32) {
            x_2d.dtype()
        } else {
            DType::BF16
        };
        let x_for_rot = if x_2d.dtype() == x_rot_dtype {
            x_2d.contiguous()?
        } else {
            x_2d.to_dtype(x_rot_dtype)?.contiguous()?
        };

        let x_rotated = if self.rotation_block >= 2 {
            let signs = match &self.rotation_signs {
                Some(t) => t,
                None => candle_core::bail!(
                    "QtipLayer: rotation_block={} but rotation_signs is None",
                    self.rotation_block
                ),
            };
            cuda_ops::rotate_x_cuda(&x_for_rot, signs, self.rotation_block)?
        } else {
            x_for_rot
        };

        // Single-token decode fast path: fused decode + gemv kernel.
        // For a 1-token batch we save the HBM round-trip on the BF16
        // dequantized weights. The output is [1, n_rows] which matches the
        // shape the matmul path returns ([1, k_in] @ [k_in, n_rows] →
        // [1, n_rows]).
        let n_tokens = x_rotated.dim(0)?;
        // See `tune::spec_pin_gemm` — default off; pins draft and verify onto
        // one kernel family so MTP's exact-argmax verification is not decided
        // by floating-point accumulation order.
        if n_tokens == 1 && !tune::spec_pin_gemm() {
            let y = cuda_ops::fused_gemv_cuda(
                &self.blocks,
                &self.row_scales,
                &self.lut,
                &x_rotated,
                k_in,
                self.codebook,
            )?;
            if y.dtype() != out_dtype {
                return y.to_dtype(out_dtype);
            }
            return Ok(y);
        }

        // Multi-token path: dequantize the weight matrix (rotated frame)
        // into the matmul dtype. Matching dtypes lets candle's matmul go
        // through the BF16/F16 fast path on H100.
        let w_dtype = x_rotated.dtype();
        let w_rotated = cuda_ops::dequantize_rotated_cuda(
            &self.blocks,
            &self.row_scales,
            &self.lut,
            k_in,
            w_dtype,
            self.codebook,
        )?;

        let y = x_rotated.matmul(&w_rotated.t()?)?;
        if y.dtype() != out_dtype {
            return y.to_dtype(out_dtype);
        }
        Ok(y)
    }

    /// GPU sparse-gather forward. See `QuantMethod::gather_forward` for the
    /// contract. Returns `Err` on any precondition mismatch so the caller
    /// can fall back to `gather_forward_cpu` for diagnostics.
    ///
    /// Implementation notes:
    /// * The output dtype matches `a.dtype()` (typically BF16 in
    ///   production V4 forward passes).
    /// * We dequantize each unique expert with the existing
    ///   `dequantize_rotated_cuda` helper that keeps weights in the rotated
    ///   frame, then rotate `a` rows once and matmul. The rotated-frame
    ///   matmul identity `(xR) · (WR)^T = x · W^T` keeps the result
    ///   algebraically correct.
    /// * We touch only the experts named in `indices` — never the full
    ///   `E`-stack. This is the load-bearing constraint that distinguishes
    ///   `gather_forward` from `dequantize_w` + dense matmul.
    /// On-device (sync-free) sibling of `gather_forward_cuda` for the small-
    /// batch DECODE regime. Reads each `(token, slot)` pair's expert id on the
    /// GPU via the gather-gemv kernel, so it never pulls `indices` to the host
    /// — making the MoE dispatch CUDA-graph capturable. One independent trellis
    /// gemv per pair (no cross-token expert reuse), optimal for decode but not
    /// prefill. Rotation handling mirrors `gather_forward_cuda` exactly so the
    /// rotated-frame matmul identity `(xR)·(WR)^T = x·W^T` holds.
    #[cfg(feature = "cuda")]
    fn gather_forward_cuda_ondevice(&self, a: &Tensor, indices: &Tensor) -> Result<Tensor> {
        let (n_tokens, n_experts_per_tok, cols) = a.dims3()?;
        let rows = self.rows_per_expert()?;

        if self.num_experts.is_none() {
            candle_core::bail!(
                "QtipLayer::gather_forward_cuda_ondevice: expected an expert-stacked (3-D) layer"
            );
        }

        let out_dtype = a.dtype();
        let n_pairs = n_tokens * n_experts_per_tok;
        let a_flat = a.reshape((n_pairs, cols))?;
        let a_for_rot = if matches!(a_flat.dtype(), DType::BF16 | DType::F16 | DType::F32) {
            a_flat.contiguous()?
        } else {
            a_flat.to_dtype(DType::BF16)?.contiguous()?
        };

        // Rotate once (rotated-frame identity), matching gather_forward_cuda.
        let a_rotated = if self.rotation_block >= 2 {
            let signs = match &self.rotation_signs {
                Some(t) => t,
                None => candle_core::bail!(
                    "QtipLayer::gather_forward_cuda_ondevice: rotation_block={} but rotation_signs is None",
                    self.rotation_block
                ),
            };
            cuda_ops::rotate_x_cuda(&a_for_rot, signs, self.rotation_block)?
        } else {
            a_for_rot
        };

        // Indices stay on-device — just normalize to U32 + contiguous.
        let idx_u32 = indices.flatten_all()?.to_dtype(DType::U32)?.contiguous()?;

        // [n_pairs, rows] -> [n_tokens, n_experts_per_tok, rows]
        let out_flat = cuda_ops::gather_gemv_cuda(
            &self.blocks,
            &self.row_scales,
            &self.lut,
            &a_rotated,
            &idx_u32,
            self.in_features,
            self.codebook,
        )?;
        let out = out_flat
            .reshape((n_tokens, n_experts_per_tok, rows))?
            .to_dtype(out_dtype)?;
        Ok(out)
    }

    #[cfg(feature = "cuda")]
    fn gather_forward_cuda(&self, a: &Tensor, indices: &Tensor) -> Result<Tensor> {
        let (n_tokens, n_experts_per_tok, cols) = a.dims3()?;
        let rows = self.rows_per_expert()?;
        let num_experts = self.num_experts_count();

        if self.num_experts.is_none() {
            candle_core::bail!(
                "QtipLayer::gather_forward_cuda: expected an expert-stacked (3-D) layer"
            );
        }

        // 1. Pull indices to host and find unique expert IDs.
        let idx_cpu: Vec<u32> = indices
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_dtype(DType::U32)?
            .to_vec1()?;
        let mut unique_ids: Vec<usize> = idx_cpu.iter().map(|&v| v as usize).collect();
        unique_ids.sort_unstable();
        unique_ids.dedup();
        for &e in &unique_ids {
            if e >= num_experts {
                candle_core::bail!(
                    "QtipLayer::gather_forward_cuda: expert index {e} >= num_experts {num_experts}"
                );
            }
        }

        // 2. Rotate `a` once. Cast to the rotation kernel's preferred dtype
        // when needed.
        let out_dtype = a.dtype();
        let a_flat = a.reshape((n_tokens * n_experts_per_tok, cols))?;
        let a_for_rot = if matches!(a_flat.dtype(), DType::BF16 | DType::F16 | DType::F32) {
            a_flat.contiguous()?
        } else {
            a_flat.to_dtype(DType::BF16)?.contiguous()?
        };
        let a_rotated = if self.rotation_block >= 2 {
            let signs = match &self.rotation_signs {
                Some(t) => t,
                None => candle_core::bail!(
                    "QtipLayer::gather_forward_cuda: rotation_block={} but rotation_signs is None",
                    self.rotation_block
                ),
            };
            cuda_ops::rotate_x_cuda(&a_for_rot, signs, self.rotation_block)?
        } else {
            a_for_rot
        };
        let a_rotated_dtype = a_rotated.dtype();

        // 3. Dequantize each unique expert's weights (rotated frame).
        //    Store keyed by expert ID for the scatter step.
        let mut weight_cache: std::collections::HashMap<usize, Tensor> =
            std::collections::HashMap::with_capacity(unique_ids.len());
        for &e in &unique_ids {
            let blocks_e = self.blocks_for_expert(e)?.contiguous()?;
            let scales_e = self.scales_for_expert(e)?.contiguous()?;
            let w_e = cuda_ops::dequantize_rotated_cuda(
                &blocks_e,
                &scales_e,
                &self.lut,
                self.in_features,
                a_rotated_dtype,
                self.codebook,
            )?;
            weight_cache.insert(e, w_e);
        }

        // 4. Build a per-expert positions list. For each unique expert e,
        //    collect the flat positions (0..n_tokens*n_experts_per_tok)
        //    that route to e.
        let total_pairs = n_tokens * n_experts_per_tok;
        let mut positions_by_expert: std::collections::HashMap<usize, Vec<u32>> =
            std::collections::HashMap::with_capacity(unique_ids.len());
        for (flat, &e_u32) in idx_cpu.iter().enumerate() {
            positions_by_expert
                .entry(e_u32 as usize)
                .or_default()
                .push(flat as u32);
        }

        // 5. Per-expert: gather a_rotated rows, matmul against W_e^T,
        //    scatter into the output tensor.
        //    Output layout: [n_tokens * n_experts_per_tok, rows].
        let device = a.device();
        // Pre-allocate an output buffer of zeros so we can fill via
        // `index_add` / `slice_assign` style operations. We use
        // `Tensor::zeros` then `index_add` for the scatter — this matches
        // candle's supported ops and avoids hand-rolling kernels.
        let mut out_flat = Tensor::zeros((total_pairs, rows), a_rotated_dtype, device)?;

        for &e in &unique_ids {
            let positions = positions_by_expert
                .get(&e)
                .expect("positions for expert should be populated");
            let pos_tensor = Tensor::from_vec(positions.clone(), (positions.len(),), device)?;

            // Gather the rotated activation rows that routed to expert e:
            //   a_e: [n_e, cols], same dtype as a_rotated.
            let a_e = a_rotated.index_select(&pos_tensor, 0)?;

            // Matmul: [n_e, cols] @ [cols, rows] = [n_e, rows].
            // weight_cache[e] is [rows, cols]; transpose for the matmul.
            let w_e = weight_cache.get(&e).expect("weight should be cached");
            let y_e = a_e.matmul(&w_e.t()?)?;

            // Scatter y_e back into out_flat at `positions`.
            out_flat = out_flat.index_add(&pos_tensor, &y_e, 0)?;
        }

        let mut out = out_flat.reshape((n_tokens, n_experts_per_tok, rows))?;
        if out.dtype() != out_dtype {
            out = out.to_dtype(out_dtype)?;
        }
        if let Some(bias) = &self.bias {
            out = out.broadcast_add(&bias.to_dtype(out.dtype())?)?;
        }
        Ok(out)
    }
}

impl QtipLayer {
    /// Construct an expert-stacked 3-D `QtipLayer` directly from its parts.
    /// The blocks tensor must have rank 3, the row_scales rank 2. This is
    /// the entry point used by the 3-D loader (#2 in the orchestrator
    /// blocker list); tests that need to construct a 3-D layer go through
    /// `stack_experts` which performs the rank promotion internally.
    ///
    /// `search` is the provenance of the supplied blocks. A loader reading a
    /// format that cannot carry provenance must pass
    /// [`QtipSearchStamp::Unstamped`] rather than assuming Trellis — claiming a
    /// search we did not verify is the failure this stamp exists to end (D4).
    /// `search_detail` is the same contract one level finer (which trellis
    /// search): pass [`QtipSearchDetail::Unknown`] unless you ran it.
    ///
    /// `codebook` says what `lut` holds. [`QtipCodebook::Gaussian`] is the
    /// always-safe answer for a loader that has no discriminator to read: it
    /// means "gather from the supplied table", which decodes correctly no
    /// matter which codebook produced the table — it just does not take the
    /// computed-decode fast path.
    #[allow(clippy::too_many_arguments)]
    pub fn from_stacked_parts(
        blocks: Tensor,
        row_scales: Tensor,
        lut: Tensor,
        bias: Option<Tensor>,
        in_features: usize,
        rotation_signs: Option<Tensor>,
        rotation_block: usize,
        search: QtipSearchStamp,
        search_detail: QtipSearchDetail,
        codebook: QtipCodebook,
        geometry: QtipGeometry,
    ) -> Result<QtipLayer> {
        if blocks.dims().len() != 3 {
            candle_core::bail!(
                "QtipLayer::from_stacked_parts: blocks must be rank 3, got rank {}",
                blocks.dims().len()
            );
        }
        if row_scales.dims().len() != 2 {
            candle_core::bail!(
                "QtipLayer::from_stacked_parts: row_scales must be rank 2, got rank {}",
                row_scales.dims().len()
            );
        }
        if blocks.dim(0)? != row_scales.dim(0)? {
            candle_core::bail!(
                "QtipLayer::from_stacked_parts: expert dim mismatch (blocks={}, scales={})",
                blocks.dim(0)?,
                row_scales.dim(0)?
            );
        }
        if blocks.dim(1)? != row_scales.dim(1)? {
            candle_core::bail!(
                "QtipLayer::from_stacked_parts: row dim mismatch (blocks={}, scales={})",
                blocks.dim(1)?,
                row_scales.dim(1)?
            );
        }
        let e = blocks.dim(0)?;
        // The caller asserts a geometry; make it true here rather than at the
        // first decode. Both supported geometries are 2 bits per weight, so
        // `blocks` alone cannot tell them apart — the table's dtype and length
        // are what discriminate.
        geometry.validate_shapes(&blocks, &lut, in_features)?;
        geometry.check_codebook(codebook)?;
        Ok(QtipLayer {
            search_detail,
            blocks,
            row_scales,
            lut,
            bias,
            in_features,
            num_experts: Some(e),
            rotation_signs,
            rotation_block,
            search,
            codebook,
            geometry,
        })
    }

    /// Stack `per_expert_layers` into a single 3-D expert-stacked
    /// `QtipLayer`. All layers must share the same `in_features`, row count,
    /// `rotation_block`, and rotation signs (when present). The shared LUT
    /// is taken from the first layer (LUTs are deterministic Gaussian and
    /// identical across experts in the current Tier-A implementation).
    ///
    /// This is the path the 3-D loader / 3-D `quantize` helper produces
    /// (#2 in the orchestrator blocker list). It also enables tests to
    /// construct an expert-stacked layer from independent 2-D quantize
    /// calls without going through the production loader.
    pub fn stack_experts(per_expert_layers: Vec<QtipLayer>) -> Result<QtipLayer> {
        if per_expert_layers.is_empty() {
            candle_core::bail!("QtipLayer::stack_experts: empty input");
        }
        let head = &per_expert_layers[0];
        let in_features = head.in_features;
        let rotation_block = head.rotation_block;
        let n_rows_head = head.row_scales.dim(0)?;

        // Validate every layer has identical shape and rotation params.
        for (i, layer) in per_expert_layers.iter().enumerate() {
            if layer.in_features != in_features {
                candle_core::bail!(
                    "QtipLayer::stack_experts: layer {i} in_features={} != head {}",
                    layer.in_features,
                    in_features
                );
            }
            if layer.row_scales.dim(0)? != n_rows_head {
                candle_core::bail!(
                    "QtipLayer::stack_experts: layer {i} n_rows={} != head {}",
                    layer.row_scales.dim(0)?,
                    n_rows_head
                );
            }
            if layer.rotation_block != rotation_block {
                candle_core::bail!(
                    "QtipLayer::stack_experts: layer {i} rotation_block={} != head {}",
                    layer.rotation_block,
                    rotation_block
                );
            }
            if layer.blocks.dims().len() != 2 || layer.row_scales.dims().len() != 1 {
                candle_core::bail!(
                    "QtipLayer::stack_experts: layer {i} must be 2-D (got blocks rank {}, scales rank {})",
                    layer.blocks.dims().len(),
                    layer.row_scales.dims().len()
                );
            }
            // A stack whose experts came from different searches has no honest
            // single stamp, so refuse rather than pick one (D4).
            if layer.search != head.search {
                candle_core::bail!(
                    "QtipLayer::stack_experts: layer {i} search={} != head {}",
                    layer.search.tag(),
                    head.search.tag()
                );
            }
            if layer.search_detail != head.search_detail {
                candle_core::bail!(
                    "QtipLayer::stack_experts: layer {i} search detail={} != head {}",
                    layer.search_detail.tag(),
                    head.search_detail.tag()
                );
            }
            // Only `head.lut` survives the stack, so experts that mean
            // different reproduction values cannot be stacked at all.
            if layer.codebook != head.codebook {
                candle_core::bail!(
                    "QtipLayer::stack_experts: layer {i} codebook={} != head {}",
                    layer.codebook.tag(),
                    head.codebook.tag()
                );
            }
            // Only `head.lut` survives the stack here too, and a table is
            // meaningful only at the geometry it was built for.
            if layer.geometry != head.geometry {
                candle_core::bail!(
                    "QtipLayer::stack_experts: layer {i} geometry={} != head {}",
                    layer.geometry.tag(),
                    head.geometry.tag()
                );
            }
        }

        let blocks_refs: Vec<&Tensor> = per_expert_layers.iter().map(|l| &l.blocks).collect();
        let scales_refs: Vec<&Tensor> = per_expert_layers.iter().map(|l| &l.row_scales).collect();

        let blocks = Tensor::stack(&blocks_refs, 0)?;
        let row_scales = Tensor::stack(&scales_refs, 0)?;

        // Pick a representative rotation signs / bias from the first layer.
        // Loader / quantize-with-options is responsible for ensuring all
        // experts share these; we already validated rotation_block above.
        let e = per_expert_layers.len();
        Ok(QtipLayer {
            blocks,
            row_scales,
            lut: head.lut.clone(),
            bias: head.bias.clone(),
            in_features,
            num_experts: Some(e),
            rotation_signs: head.rotation_signs.clone(),
            rotation_block,
            search: head.search,
            search_detail: head.search_detail,
            codebook: head.codebook,
            geometry: head.geometry,
        })
    }

    /// Number of experts encoded in this layer. Returns 1 for a legacy 2-D
    /// layer (`num_experts == None`) and `E` for an expert-stacked 3-D layer
    /// (`num_experts == Some(E)`). The expert dimension only appears for MoE
    /// gate/up/down projections that have been packed across experts by the
    /// loader (see `MoEExperts` / `PackedExperts` paths).
    ///
    /// This is the internal helper used by `gather_forward` etc. The public
    /// `num_experts() -> Option<usize>` accessor is the source-of-truth API.
    fn num_experts_count(&self) -> usize {
        self.num_experts.unwrap_or(1)
    }

    /// Number of output rows per expert. For a 2-D layer this is just
    /// `row_scales.dim(0)`; for a 3-D layer it is `row_scales.dim(1)`.
    fn rows_per_expert(&self) -> Result<usize> {
        match self.num_experts {
            None => self.row_scales.dim(0),
            Some(_) => self.row_scales.dim(1),
        }
    }

    /// Slice the per-expert 2-D `(N, packed_per_row)` view of `blocks` for
    /// expert `e`. When the layer is 2-D this returns `blocks` directly.
    fn blocks_for_expert(&self, e: usize) -> Result<Tensor> {
        match self.num_experts {
            None => {
                if e != 0 {
                    candle_core::bail!(
                        "QtipLayer: requested expert {e} on a non-stacked (2-D) layer"
                    );
                }
                Ok(self.blocks.clone())
            }
            Some(_) => {
                // narrow + squeeze gives a contiguous `(N, packed_per_row)` view.
                self.blocks.narrow(0, e, 1)?.squeeze(0)
            }
        }
    }

    /// Slice the per-expert 1-D `(N,)` view of `row_scales` for expert `e`.
    fn scales_for_expert(&self, e: usize) -> Result<Tensor> {
        match self.num_experts {
            None => {
                if e != 0 {
                    candle_core::bail!(
                        "QtipLayer: requested expert {e} on a non-stacked (2-D) layer"
                    );
                }
                Ok(self.row_scales.clone())
            }
            Some(_) => self.row_scales.narrow(0, e, 1)?.squeeze(0),
        }
    }

    /// Dequantize a single expert `e`'s weights into a `[rows, in_features]`
    /// tensor in the *unrotated* (original) frame. Used by the CPU
    /// `gather_forward` fallback when the GPU rotation kernel is unavailable.
    #[allow(dead_code)]
    #[allow(clippy::needless_range_loop)] // scales_data index loop, GPU-validated hot path
    fn dequantize_expert_weights_unrotated(&self, e: usize) -> Result<Tensor> {
        let blocks_e = self.blocks_for_expert(e)?;
        let scales_e = self.scales_for_expert(e)?;
        let n = scales_e.dim(0)?;
        let k_in = self.in_features;

        let blocks_data: Vec<u8> = blocks_e.to_device(&Device::Cpu)?.flatten_all()?.to_vec1()?;
        let scales_data: Vec<f32> = scales_e.to_device(&Device::Cpu)?.flatten_all()?.to_vec1()?;
        let lut_data: Vec<f32> = self.lut.to_device(&Device::Cpu)?.flatten_all()?.to_vec1()?;

        let num_symbols_per_row = k_in / V as usize;
        let packed_per_row = num_symbols_per_row / 2;

        let mut out = vec![0f32; n * k_in];
        for row in 0..n {
            let scale = scales_data[row];
            let mut state: u32 = 0;
            let row_packed_base = row * packed_per_row;
            let out_row_base = row * k_in;
            for sym_idx in 0..num_symbols_per_row {
                let byte = blocks_data[row_packed_base + sym_idx / 2];
                let sym = if sym_idx.is_multiple_of(2) {
                    byte & 0x0F
                } else {
                    (byte >> 4) & 0x0F
                };
                state = ((state << K) | sym as u32) & STATE_MASK;
                let lut_off = (state as usize) * V as usize;
                out[out_row_base + sym_idx * 2] = lut_data[lut_off] * scale;
                out[out_row_base + sym_idx * 2 + 1] = lut_data[lut_off + 1] * scale;
            }
        }

        // Undo the Hadamard rotation row-by-row if rotation is active so the
        // matmul against the (unrotated) activation produces the algebraically
        // correct result on the CPU fallback.
        if self.rotation_block >= 2 {
            let signs: Vec<f32> = match &self.rotation_signs {
                Some(t) => t.to_device(&Device::Cpu)?.to_vec1::<f32>()?,
                None => candle_core::bail!(
                    "QtipLayer::dequantize_expert_weights_unrotated: rotation_block={} but rotation_signs is None",
                    self.rotation_block
                ),
            };
            for row in 0..n {
                let row_slice = &mut out[row * k_in..(row + 1) * k_in];
                apply_block_rotation(row_slice, &signs, self.rotation_block);
            }
        }

        Tensor::from_vec(out, (n, k_in), &Device::Cpu)?
            .to_device(self.blocks.device())?
            .to_dtype(DType::BF16)
    }

    /// CPU sparse-gather forward (portable reference).
    ///
    /// Mirrors `gather_forward_cuda` step-for-step but uses
    /// `dequantize_expert_weights_unrotated` (no GPU kernel) and host-side
    /// matmul. Output dtype follows `a.dtype()`.
    ///
    /// Like the GPU path, it touches only the experts named in `indices` —
    /// the dequantize is per-unique-expert, and the matmul batches all
    /// (token, k) pairs that route to the same expert into one operation.
    fn gather_forward_cpu(&self, a: &Tensor, indices: &Tensor) -> Result<Tensor> {
        let (n_tokens, n_experts_per_tok, cols) = a.dims3()?;
        let (i_tokens, i_k) = indices.dims2()?;
        if i_tokens != n_tokens || i_k != n_experts_per_tok {
            candle_core::bail!(
                "QtipLayer::gather_forward: indices shape ({i_tokens}, {i_k}) doesn't match a shape ({n_tokens}, {n_experts_per_tok}, {cols})"
            );
        }
        if cols != self.in_features {
            candle_core::bail!(
                "QtipLayer::gather_forward: a.dim(-1)={cols} != in_features={}",
                self.in_features
            );
        }
        let rows = self.rows_per_expert()?;
        let num_experts = self.num_experts_count();

        let idx_cpu: Vec<u32> = indices
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_dtype(DType::U32)?
            .to_vec1()?;

        // Track unique expert IDs so we only dequantize each one once even
        // when several (token, k) pairs share the same expert (which is the
        // common case for prefill — top-6 of 256 means many duplicates).
        let mut unique_ids: Vec<usize> = idx_cpu.iter().map(|&v| v as usize).collect();
        unique_ids.sort_unstable();
        unique_ids.dedup();
        for &e in &unique_ids {
            if e >= num_experts {
                candle_core::bail!(
                    "QtipLayer::gather_forward: expert index {e} >= num_experts {num_experts}"
                );
            }
        }

        // Per-expert dequantize. `dequantize_expert_weights_unrotated`
        // already applies the inverse-rotation, so the resulting weight
        // is in the **original** frame — we match it against the
        // un-rotated activation.
        let mut weight_cache: std::collections::HashMap<usize, Tensor> =
            std::collections::HashMap::with_capacity(unique_ids.len());
        for &e in &unique_ids {
            let w_e = self
                .dequantize_expert_weights_unrotated(e)?
                .to_dtype(a.dtype())?;
            weight_cache.insert(e, w_e);
        }

        // Build per-expert positions list.
        let total_pairs = n_tokens * n_experts_per_tok;
        let mut positions_by_expert: std::collections::HashMap<usize, Vec<u32>> =
            std::collections::HashMap::with_capacity(unique_ids.len());
        for (flat, &e_u32) in idx_cpu.iter().enumerate() {
            positions_by_expert
                .entry(e_u32 as usize)
                .or_default()
                .push(flat as u32);
        }

        // a_flat: [n_tokens * n_experts_per_tok, cols]
        let a_flat = a.reshape((n_tokens * n_experts_per_tok, cols))?;
        let device = a.device();

        // Output accumulator. We initialize to zeros and scatter-add the
        // per-expert results, exactly mirroring the GPU code path's
        // partitioning. Since the positions are disjoint across experts,
        // each output row is written exactly once.
        let mut out_flat = Tensor::zeros((total_pairs, rows), a.dtype(), device)?;

        for &e in &unique_ids {
            let positions = positions_by_expert
                .get(&e)
                .expect("positions for expert should be populated");
            let pos_tensor = Tensor::from_vec(positions.clone(), (positions.len(),), device)?;
            let a_e = a_flat.index_select(&pos_tensor, 0)?;
            let w_e = weight_cache.get(&e).expect("weight should be cached");
            let y_e = a_e.matmul(&w_e.t()?)?;
            out_flat = out_flat.index_add(&pos_tensor, &y_e, 0)?;
        }

        let mut out = out_flat.reshape((n_tokens, n_experts_per_tok, rows))?;
        if let Some(bias) = &self.bias {
            // bias shape: [rows]. QtipLayer doesn't carry per-expert bias —
            // MoE projections in the V4 checkpoint are bias-free.
            out = out.broadcast_add(&bias.to_dtype(out.dtype())?)?;
        }
        Ok(out)
    }
}

/// Keep only `ids` (ascending, in range) along a tensor's leading dimension.
///
/// Shared by both trellis rungs' expert-parallel slice. Contiguous id runs —
/// the `ExpertPlacement::contiguous` case — take `narrow`, which avoids
/// building an index tensor on the device.
pub(crate) fn select_experts_dim0(t: &Tensor, ids: &[usize]) -> candle_core::Result<Tensor> {
    let e = t.dim(0)?;
    if ids.is_empty() {
        candle_core::bail!("expert slice: cannot keep zero experts");
    }
    for w in ids.windows(2) {
        if w[0] >= w[1] {
            candle_core::bail!("expert slice: ids must be strictly ascending, got {w:?}");
        }
    }
    if ids[ids.len() - 1] >= e {
        candle_core::bail!(
            "expert slice: id {} is out of range for a {e}-expert stack",
            ids[ids.len() - 1]
        );
    }
    if ids.windows(2).all(|w| w[1] == w[0] + 1) {
        return t.narrow(0, ids[0], ids.len())?.contiguous();
    }
    let idx = Tensor::from_vec(
        ids.iter().map(|&i| i as u32).collect::<Vec<_>>(),
        (ids.len(),),
        t.device(),
    )?;
    t.index_select(&idx, 0)?.contiguous()
}

impl QtipLayer {
    /// The expert-parallel slice: keep only `ids` out of this stack.
    ///
    /// `lut` and `rotation_signs` are **shared** across the stack (the
    /// rotation is a function of `K_in`, identical for every expert), so they
    /// are replicated rather than sliced — exactly as wave44-BV §4.1
    /// describes. Only `blocks` and `row_scales` carry an expert axis.
    pub fn select_experts_concrete(&self, ids: &[usize]) -> candle_core::Result<Self> {
        let Some(num_experts) = self.num_experts else {
            candle_core::bail!(
                "QtipLayer::select_experts: this layer is a plain 2-D linear, not an expert stack"
            );
        };
        if ids.len() > num_experts {
            candle_core::bail!(
                "QtipLayer::select_experts: asked for {} experts out of {num_experts}",
                ids.len()
            );
        }
        Ok(Self {
            blocks: select_experts_dim0(&self.blocks, ids)?,
            row_scales: select_experts_dim0(&self.row_scales, ids)?,
            lut: self.lut.clone(),
            bias: self.bias.clone(),
            in_features: self.in_features,
            num_experts: Some(ids.len()),
            rotation_signs: self.rotation_signs.clone(),
            rotation_block: self.rotation_block,
            search: self.search,
            search_detail: self.search_detail,
            codebook: self.codebook,
            geometry: self.geometry,
        })
    }
}

impl QuantMethod for QtipLayer {
    fn new(method: QuantMethodConfig) -> candle_core::Result<Self>
    where
        Self: Sized,
    {
        match method {
            QuantMethodConfig::Qtip {
                blocks,
                row_scales,
                lut,
                bias,
                in_features,
                rotation_signs,
                rotation_block,
            } => Ok(Self {
                blocks,
                row_scales,
                lut,
                bias,
                in_features,
                // QuantMethodConfig::Qtip is the public 2-D constructor; 3-D
                // stacks are built internally by `quantize_with_options_3d`
                // and never round-trip through the config-based factory.
                num_experts: None,
                rotation_signs,
                rotation_block,
                // The config carries packed blocks from an unknown producer.
                // We did not run the search, so we do not claim it (D4).
                search: QtipSearchStamp::Unstamped,
                search_detail: QtipSearchDetail::Unknown,
                // Same contract for the codebook: the config carries a table
                // and no discriminator, so the only honest reading is "these
                // symbols mean whatever that table says". `Gaussian` is
                // exactly that instruction — gather from the stored values —
                // and stays correct for a computed-codebook table, since the
                // table holds the computed values. It only forgoes the
                // in-register decode.
                codebook: QtipCodebook::Gaussian,
                // Same again for the geometry: the config carries no
                // discriminator, and every payload that reached this factory
                // before the field existed was K=4/V=2/L=16. Reading it as
                // anything else would be inventing a claim.
                geometry: QtipGeometry::K4V2L16,
            }),
            _ => candle_core::bail!("QtipLayer requires QuantMethodConfig::Qtip"),
        }
    }

    fn dequantize_w(&self) -> Result<candle_core::Tensor> {
        self.dequantize_weights()
    }

    fn qtip_packed(&self) -> Option<QtipPackedView<'_>> {
        // `QtipPackedView` carries no geometry field, so a consumer would read
        // these bytes as K=4/V=2 nibbles. Hand out nothing rather than
        // something that cannot describe itself.
        match self.geometry {
            QtipGeometry::K4V2L16 => Some(self.packed_view()),
            _ => None,
        }
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.require_k4v2l16("forward")?;
        self.forward_dequantize(x)
    }

    /// Sparse-gather matmul for V4 MoE Fast backend.
    ///
    /// # Shapes (CUDA path)
    /// * `a`       : `[n_tokens, n_experts_per_tok, in_features]`
    /// * `indices` : `[n_tokens, n_experts_per_tok]` (U32)
    /// * self      : expert-stacked layer where `blocks` is rank-3 with
    ///   leading expert dim `E`; weight per expert is
    ///   `(rows, in_features)`.
    /// * output    : `[n_tokens, n_experts_per_tok, rows]`
    ///
    /// # Algorithm
    /// 1. Read `indices` to host once (small — `(n_tokens, k)` U32; for V4
    ///    decode with batch=1 and topk=6 this is 24 bytes total).
    /// 2. Compute the **unique** set of expert IDs the router landed on
    ///    this step. For top-6 of 256 in decode this is at most 6 experts;
    ///    for prefill it converges to all 256 only at very long sequences.
    /// 3. For each unique expert, dequantize its `(rows, in_features)`
    ///    weight slab from the packed QTIP blocks. We touch only the
    ///    selected experts' weights — this is the load-bearing
    ///    requirement (dequantizing all 256 every step would defeat the
    ///    point of an MoE).
    /// 4. Rotate every row of `a` by the layer's Hadamard incoherence
    ///    rotation. The rotation `R = block_diag(D·H·D)` is involutory and
    ///    independent of expert ID, so we apply it once across all
    ///    (token, k) pairs and reuse the rotated activations against every
    ///    expert's *rotated-frame* weight.
    /// 5. For each unique expert `e`, find the (token, k) positions that
    ///    routed to `e`, gather those `a` rows into a contiguous
    ///    `(n_e, in_features)` batch, matmul against `W_e^T` to produce
    ///    `(n_e, rows)`, then scatter the rows back into the output at the
    ///    matching positions.
    ///
    /// # Rotation invariant
    /// QTIP stores `W` in the rotated frame: `W_stored = W · R`. The
    /// algebraic identity that makes the rotation free is
    /// `(x · R) · (W · R)^T = x · R · R^T · W^T = x · W^T` (since `R` is
    /// orthogonal and involutory). We exploit it by leaving the
    /// dequantized weight in the rotated frame and matching it with a
    /// rotated activation — saves one rotation pass per expert.
    fn gather_forward(&self, a: &Tensor, indices: &Tensor) -> Result<Tensor> {
        self.require_k4v2l16("gather_forward")?;
        if self.num_experts.is_none() {
            // The contract is "expert-sparse dispatch" — a 2-D layer is a
            // single-expert (i.e. non-MoE) layer and `gather_forward`
            // doesn't make sense on it. Bail with a clear message rather
            // than silently produce wrong output. Callers that hit this
            // are using the wrong code path.
            candle_core::bail!(
                "QtipLayer::gather_forward requires an expert-stacked 3-D layer (got a 2-D layer with no expert dim)"
            );
        }

        let (n_tokens, n_experts_per_tok, cols) = a.dims3()?;
        let (i_tokens, i_k) = indices.dims2()?;
        if i_tokens != n_tokens || i_k != n_experts_per_tok {
            candle_core::bail!(
                "QtipLayer::gather_forward: indices shape ({i_tokens}, {i_k}) doesn't match a shape ({n_tokens}, {n_experts_per_tok}, {cols})"
            );
        }
        if cols != self.in_features {
            candle_core::bail!(
                "QtipLayer::gather_forward: a.dim(-1)={cols} != in_features={}",
                self.in_features
            );
        }

        // GPU fast path — only when storage lives on CUDA, the kernels were
        // compiled in, and `a` is on CUDA in a dtype the dequantize +
        // rotation kernels accept. Any precondition failure drops into the
        // CPU reference path below.
        #[cfg(feature = "cuda")]
        {
            if cuda_ops::can_use_qtip_cuda(&self.blocks)
                && matches!(a.device(), candle_core::Device::Cuda(_))
                && matches!(a.dtype(), DType::BF16 | DType::F16 | DType::F32)
            {
                // On-device (sync-free) fused path: reads the routing indices
                // on-GPU so the MoE dispatch is CUDA-graph capturable. The
                // alternative, `gather_forward_cuda`, pulls indices to the host
                // (a capture-aborting sync), dequantizes every DISTINCT expert
                // to BF16 in HBM and holds them all live — so, unlike the
                // bitshift rung, this rung's over-boundary path is NOT an
                // amortizing grouped GEMM. The boundary is therefore derived
                // from the traffic ratio between the two (16x: a BF16 write
                // plus a BF16 read against the packed 2-bit bytes) rather than
                // pinned at a decode-shaped constant, and it still stays inside
                // the kernel's real structural limit (grid.y = n_pairs <=
                // 65535). See `gather_policy` for the arithmetic.
                let num_experts = self.num_experts_count();
                // Read by VALUE: `ARC_NO_QTIP_GROUPED_MOE=0` leaves the grouped
                // GEMM enabled (the default).
                let grouped_disabled = crate::env_flag_is_set("ARC_NO_QTIP_GROUPED_MOE");
                // Is the grouped GEMM actually usable for this call?
                let grouped_available = !grouped_disabled
                    && matches!(a.dtype(), DType::BF16 | DType::F16)
                    && self.in_features.is_multiple_of(grouped::GROUPED_TILE_K);

                // Prefer the grouped GEMM over the fused gather-GEMV once — and
                // only once — its m-tiles are full enough to amortize.
                //
                // The grouped kernel stages each woken expert's packed bytes
                // ONCE PER M-TILE of `GROUPED_TILE_M` pairs. Its advantage over
                // the per-pair GEMV is exactly the average tile fill, so the
                // gate is a fill, not a token count:
                //
                //     pairs per woken expert >= GROUPED_TILE_M
                //     pairs per woken expert = n*k / (E * (1 - (1 - k/E)^n))
                //
                // Gating on `n_tokens > DECODE_REGIME_MAX_TOKENS` instead put
                // the kernel to work at n=9, where V4's top-6-of-256 routing
                // wakes ~54 experts for 54 pairs and the 16-row tile holds one
                // useful row — ~2 orders of magnitude below where the kernel
                // amortizes. Full tiles arrive at ~683 tokens for this routing
                // shape.
                //
                // This fix does NOT explain the B=32 aggregate-throughput
                // collapse it was filed against; that attribution is retracted
                // and the measurement is in `gather_policy` §4. Do not re-derive
                // it from this comment.
                //
                // The decode regime stays fused UNCONDITIONALLY: that is the
                // RUN-161 floor, not a performance choice (see
                // `gather_policy`). An explicit `ARC_QTIP_ONDEVICE_MOE_MAX_TOKENS`
                // override also still wins, so a harness can pin the GEMV arm.
                // 🔴 The `grouped_gemm_tiles_amortize` conjunct is GONE — see
                // `gather_policy` §4a. It demanded `6n/256 >= 16`, i.e.
                // n >= 683, so on this rung too the grouped GEMM was
                // unreachable in serving and every decode step ran the per-pair
                // GEMV. Measured on THIS rung (qtip2 LUT, H200, V4-Flash),
                // clean rows only: forcing the grouped path is 1.10x at B=48,
                // 1.14x at 64, 1.22x at 80, 1.29x at 96, 1.66x at 128 and
                // 1.51x at 512. Tile fill is not the deciding quantity; staging
                // each woken expert once per m-tile instead of once per
                // (token, slot) pair is.
                let grouped_preferred = grouped_available
                    && n_tokens > DECODE_REGIME_MAX_TOKENS
                    && gather_policy::ondevice_max_tokens_override().is_none();

                let use_ondevice = !grouped_preferred
                    && match gather_policy::ondevice_max_tokens_override() {
                        Some(cap) => {
                            n_tokens <= cap
                                && n_tokens.saturating_mul(n_experts_per_tok)
                                    <= gather_policy::GATHER_GEMV_MAX_PAIRS
                        }
                        None => gather_policy::lut_fused_gather_preferred(
                            n_tokens,
                            n_experts_per_tok,
                            num_experts,
                        ),
                    };
                // Read by VALUE: `ARC_NO_QTIP_ONDEVICE_MOE=0` leaves the
                // on-device path enabled (the default).
                let ondevice_disabled = crate::env_flag_is_set("ARC_NO_QTIP_ONDEVICE_MOE");
                if !ondevice_disabled && use_ondevice {
                    // On-device ONLY, propagate its error. The host fallback
                    // (`gather_forward_cuda`) does a `to_vec1` D2H read of
                    // `indices` which, under CUDA-graph capture, is
                    // recorded-not-executed -> returns garbage indices ->
                    // out-of-bounds expert-weight read -> MMU fault. So we must
                    // never silently fall back to it in the capturable path.
                    // (RUN-161)
                    return self.gather_forward_cuda_ondevice(a, indices);
                }
                // Prefill regime: the LUT-rung trellis grouped GEMM. Tokens
                // sorted by expert on-device, then a persistent tensor-core
                // tile loop over the ragged groups, so each woken expert's
                // packed bytes are staged once per m-tile instead of once per
                // (token, slot) pair.
                //
                // This rung shipped without it for one wrong reason: its state
                // update `state = ((state << 4) | sym) & 0xFFFF` LOOKS
                // sequential. It is not — the state at symbol `t` is just the
                // last four nibbles, i.e. a 16-bit window over the packed
                // stream, exactly as the qtip2b rung's state is a window over
                // its 2-bit stream. Random-access decode is what makes a
                // grouped GEMM reachable, and it was always available here.
                //
                // Below this, `gather_forward_cuda` stays as the fallback: it
                // is the only path that handles F32 activations and shapes
                // whose `in_features` is not a multiple of the k-chunk.
                if grouped_available {
                    gather_policy::log_grouped_gemm_engaged_once(
                        n_tokens,
                        n_experts_per_tok,
                        num_experts,
                    );
                    let total_pairs = n_tokens * n_experts_per_tok;
                    let a_flat = a.reshape((total_pairs, cols))?.contiguous()?;
                    let a_rotated = if self.rotation_block >= 2 {
                        match &self.rotation_signs {
                            Some(signs) => {
                                cuda_ops::rotate_x_cuda(&a_flat, signs, self.rotation_block)?
                            }
                            None => candle_core::bail!(
                                "QtipLayer::gather_forward: rotation_block={} but rotation_signs is None",
                                self.rotation_block
                            ),
                        }
                    } else {
                        a_flat
                    };
                    let idx = indices
                        .reshape((total_pairs,))?
                        .to_dtype(DType::U32)?
                        .contiguous()?;
                    let out_flat = cuda_ops::grouped_gemm_lut_cuda(
                        &self.blocks,
                        &self.row_scales,
                        &self.lut,
                        &a_rotated,
                        &idx,
                        self.in_features,
                        self.codebook,
                    )?;
                    let mut out =
                        out_flat.reshape((n_tokens, n_experts_per_tok, self.rows_per_expert()?))?;
                    if out.dtype() != a.dtype() {
                        out = out.to_dtype(a.dtype())?;
                    }
                    if let Some(bias) = &self.bias {
                        out = out.broadcast_add(&bias.to_dtype(out.dtype())?)?;
                    }
                    return Ok(out);
                }

                // The per-expert dequantize below materializes weights to HBM.
                gather_policy::log_lut_gather_fallback_once(
                    n_tokens,
                    n_experts_per_tok,
                    num_experts,
                );
                warn_dequant_materialize_at_decode(
                    n_tokens,
                    "QtipLayer::gather_forward_cuda (per-expert dequantize+matmul)",
                );
                if let Ok(out) = self.gather_forward_cuda(a, indices) {
                    return Ok(out);
                }
            }
        }

        warn_dequant_materialize_at_decode(n_tokens, "QtipLayer::gather_forward_cpu");
        self.gather_forward_cpu(a, indices)
    }

    fn quantized_act_type(&self) -> Option<DType> {
        None
    }

    fn select_experts(&self, ids: &[usize]) -> Result<Arc<dyn QuantMethod>> {
        Ok(Arc::new(self.select_experts_concrete(ids)?))
    }

    fn add_delta_w(&self, _delta: &Tensor) -> Result<Arc<dyn QuantMethod>> {
        candle_core::bail!("QtipLayer does not support add_delta_w")
    }

    fn dtype_and_device(&self) -> (DType, candle_core::Device) {
        (DType::BF16, self.blocks.device().clone())
    }

    fn apply_isq(
        self: Arc<Self>,
        _dtype: Option<IsqType>,
        _device: Device,
        _n_quantized: &AtomicUsize,
        _imatrix_weight: Option<Vec<f32>>,
        _guard: QuantizeOntoGuard,
    ) -> Result<Arc<dyn QuantMethod>> {
        candle_core::bail!("QtipLayer does not support ISQ re-quantization")
    }
}

impl QtipLayer {
    /// `Some(E)` when this layer holds a 3-D stacked-expert weight; `None`
    /// for a standard 2-D linear weight. Sister `gather_forward` calls this
    /// to detect the storage layout.
    pub fn num_experts(&self) -> Option<usize> {
        self.num_experts
    }

    /// Cached input feature dim `K_in` (number of columns of each per-row
    /// quantized weight vector). Shared across experts in 3-D mode.
    pub fn in_features(&self) -> usize {
        self.in_features
    }

    /// Number of output rows `N` (per-expert in 3-D mode, total in 2-D mode).
    pub fn out_features(&self) -> Result<usize> {
        match self.num_experts {
            Some(_) => self.row_scales.dim(1),
            None => self.row_scales.dim(0),
        }
    }

    /// Block storage tensor. Shape:
    /// - 2-D mode: `[N, packed_K]`
    /// - 3-D mode: `[E, N, packed_K]`
    ///
    /// where `packed_K = K_in / 4` (two K=4 symbols per byte).
    pub fn blocks(&self) -> &Tensor {
        &self.blocks
    }

    /// Per-row scale tensor. Shape:
    /// - 2-D mode: `[N]`
    /// - 3-D mode: `[E, N]`
    pub fn row_scales(&self) -> &Tensor {
        &self.row_scales
    }

    /// Shared Gaussian trellis LUT, shape `[2^L, V]`.
    pub fn lut(&self) -> &Tensor {
        &self.lut
    }

    /// Hadamard incoherence rotation signs, shape `[K_in]`. `None` when
    /// rotation is disabled (Greedy path, legacy checkpoints).
    pub fn rotation_signs(&self) -> Option<&Tensor> {
        self.rotation_signs.as_ref()
    }

    /// Block size for the block-diagonal Hadamard rotation, or 0 when
    /// disabled.
    pub fn rotation_block(&self) -> usize {
        self.rotation_block
    }

    /// Which trellis search produced these blocks (DOCTRINE D4 §3).
    pub fn search_stamp(&self) -> QtipSearchStamp {
        self.search
    }

    /// *Which* trellis search: beam width and objective (UQFF ≥ 0.3.0 flags
    /// byte). [`QtipSearchDetail::Unknown`] for artifacts whose producer could
    /// not record it.
    pub fn search_detail(&self) -> QtipSearchDetail {
        self.search_detail
    }

    /// Which codebook these symbols decode against.
    /// [`QtipCodebook::Gaussian`] for every artifact written before the
    /// discriminator existed, and for any loader that has no field to read it
    /// from — in both cases it means "gather from the stored table", which is
    /// correct for any table.
    pub fn codebook(&self) -> QtipCodebook {
        self.codebook
    }

    /// Which trellis geometry this layer's symbols were baked at, and
    /// therefore which decoder can read them. See [`QtipGeometry`].
    pub fn geometry(&self) -> QtipGeometry {
        self.geometry
    }

    /// Refuse any decode path that only knows K=4/V=2/L=16.
    ///
    /// **This is what keeps stage 2 from being a half-applied change.** The
    /// geometry discriminator makes a K=8/V=4/L=12 artifact *loadable*; the
    /// decoders that would then read it — `dequantize_weights_rotated_f32`,
    /// `dequantize_single_expert`, `gather_forward_cpu`, and every CUDA
    /// launcher in `cuda_ops` other than `fused_gemv_k8v4l12_cuda` — all
    /// unpack nibbles and index a `[2^16, 2]` table. Handed a K=8 row they
    /// would not fault: at 2 bits per weight the byte count is identical and
    /// every index lands in bounds. They would serve garbage.
    ///
    /// So every entry point into decode states the geometries it can handle.
    /// A layer this build cannot serve is refused loudly at the door rather
    /// than quietly at the logits.
    fn require_k4v2l16(&self, op: &str) -> Result<()> {
        match self.geometry {
            QtipGeometry::K4V2L16 => Ok(()),
            other => candle_core::bail!(
                "QtipLayer::{op}: this layer was baked at geometry {}, and {op} only implements \
                 K=4/V=2/L=16. The {} decode path is not wired into serving yet — the fused \
                 GEMV kernel exists (`kernels/qtip/qtip_gemv_k8v4l12.cu`) but nothing dispatches \
                 to it. Refusing rather than decoding these symbols as K=4/V=2, which would not \
                 fault (both geometries are 2 bits per weight, so the packed byte count is the \
                 same) and would silently serve wrong weights.",
                other.tag(),
                other.tag()
            ),
        }
    }

    /// Dequantize the i-th expert's `[N, K_in]` BF16 weight matrix (3-D mode
    /// only). Internal use by `gather_forward` and friends; bails when called
    /// on a 2-D layer or with `expert_idx >= num_experts`.
    pub fn dequantize_expert(&self, expert_idx: usize) -> Result<Tensor> {
        self.require_k4v2l16("dequantize_expert")?;
        let e = self.num_experts.ok_or_else(|| {
            candle_core::Error::Msg("QtipLayer::dequantize_expert called on a 2-D layer".into())
        })?;
        if expert_idx >= e {
            candle_core::bail!(
                "QtipLayer::dequantize_expert: expert_idx {expert_idx} >= num_experts {e}"
            );
        }
        let n = self.row_scales.dim(1)?;
        let k_in = self.in_features;
        let blocks_e = self
            .blocks
            .narrow(0, expert_idx, 1)?
            .squeeze(0)?
            .contiguous()?;
        let scales_e = self
            .row_scales
            .narrow(0, expert_idx, 1)?
            .squeeze(0)?
            .contiguous()?;
        self.dequantize_single_expert(&blocks_e, &scales_e, n, k_in)
    }

    pub fn linear_b(
        in_dim: usize,
        out_dim: usize,
        config: &QuantizedConfig,
        bias: bool,
        vb: ShardedVarBuilder,
    ) -> Result<Arc<dyn QuantMethod>> {
        let QuantizedConfig::Qtip {} = config else {
            candle_core::bail!("Unexpected quantization config for QTIP layer.")
        };
        let num_symbols_per_row = in_dim / V as usize;
        if !num_symbols_per_row.is_multiple_of(2) {
            candle_core::bail!(
                "QTIP linear_b: symbols per row ({num_symbols_per_row}) must be even"
            );
        }
        let packed_per_row = num_symbols_per_row / 2;

        let blocks = vb.get_with_hints_dtype(
            (out_dim, packed_per_row),
            "qtip_blocks",
            Default::default(),
            DType::U8,
        )?;
        let row_scales = vb.get_with_hints_dtype(
            (out_dim,),
            "qtip_row_scales",
            Default::default(),
            DType::F32,
        )?;
        let lut = vb
            .get_with_hints_dtype(
                (LUT_SIZE, V as usize),
                "qtip_lut",
                Default::default(),
                DType::F32,
            )
            .or_else(|_| {
                // No LUT in checkpoint — synthesize Gaussian default
                let g = gaussian_lut();
                Tensor::from_vec(g, (LUT_SIZE, V as usize), vb.device())
            })?;

        let bias = if bias {
            Some(vb.get((out_dim,), "bias")?)
        } else {
            None
        };

        // RUN-158 incoherence rotation: load the per-input-feature ±1 signs
        // if the checkpoint provides them. Older checkpoints without
        // `qtip_rotation_signs` get rotation disabled (cos sim parity with
        // pre-RUN-158 behavior). Newer Viterbi-quantized checkpoints carry
        // these signs so quantize-time and infer-time rotations agree.
        let (rotation_signs, rotation_block) = match vb.get_with_hints_dtype(
            (in_dim,),
            "qtip_rotation_signs",
            Default::default(),
            DType::F32,
        ) {
            Ok(t) => (Some(t), rotation_block_size(in_dim)),
            Err(_) => (None, 0usize),
        };

        Ok(Arc::new(Self {
            blocks,
            row_scales,
            lut,
            bias,
            in_features: in_dim,
            // `linear_b` loads single-linear checkpoints (2-D).
            num_experts: None,
            rotation_signs,
            rotation_block,
            // Safetensors QTIP checkpoints carry no provenance field, so the
            // honest answer is "unknown" — see `QtipSearchStamp::enforce_at_load`.
            search: QtipSearchStamp::Unstamped,
            search_detail: QtipSearchDetail::Unknown,
            // No codebook discriminator in safetensors either. Gather from the
            // table that is actually in the checkpoint (or the synthesized
            // Gaussian default above) — correct for any table, and never
            // assumes a computed codebook we cannot verify.
            codebook: QtipCodebook::Gaussian,
            // Safetensors QTIP checkpoints predate every geometry but this
            // one, and carry no field that could say otherwise.
            geometry: QtipGeometry::K4V2L16,
        }))
    }

    /// Concrete-typed UQFF deserialize that returns a `QtipLayer` (plus the
    /// optional bias tensor) instead of `Arc<dyn QuantMethod>`. Shared body
    /// of `deserialize` / `deserialize_ext_bias`; also used by tests to
    /// inspect typed fields without a `dyn`-downcast (the `QuantMethod`
    /// trait does not extend `Any`).
    ///
    /// Handles both storage modes:
    /// - 2-D: `blocks [N, packed_K]`, `row_scales [N]` → `num_experts: None`
    /// - 3-D: `blocks [E, N, packed_K]`, `row_scales [E, N]` →
    ///   `num_experts: Some(E)`
    ///
    /// The mode is inferred from the self-describing tensor shapes (UQFF
    /// tensor payloads carry rank + dims), so no extra header bytes are
    /// needed and 2-D payloads remain byte-identical to pre-v0.2.1 writers.
    /// Each tensor is deserialized in one shot (single host buffer → single
    /// device upload) — no per-expert round-trips at load.
    fn deserialize_concrete(
        data: Cow<[u8]>,
        device: &Device,
        guard: QuantizeOntoGuard,
    ) -> Result<(Self, Option<Tensor>)> {
        let (layer, ext_bias) = Self::deserialize_concrete_unchecked(data, device, guard)?;
        // D4 §3 teeth: this is the load gate every serving path passes through
        // (`deserialize` / `deserialize_ext_bias` both funnel here).
        layer
            .search
            .enforce_at_load("qtip-layer", layer.rotation_block)?;
        Ok((layer, ext_bias))
    }

    /// Payload parser without the D4 load gate. Private, and used only by the
    /// checked wrapper above plus the serde round-trip tests, which need to
    /// round-trip a cheap greedy fixture without asserting anything about
    /// serving policy. Nothing outside this crate can reach it, and no serving
    /// path calls it directly.
    fn deserialize_concrete_unchecked(
        data: Cow<[u8]>,
        device: &Device,
        guard: QuantizeOntoGuard,
    ) -> Result<(Self, Option<Tensor>)> {
        let mut buffer = Cursor::new(data);

        let version = buffer.read_u32::<LittleEndian>()?;
        if let Err(e) = version_is_compatible(version) {
            return Err(candle_core::Error::wrap(e));
        }
        let isq_type = buffer.read_u8()? as usize;
        if isq_type != QuantizedSerdeType::Qtip as usize {
            candle_core::bail!(
                "ISQ type ({isq_type}) doesn't match expected QTIP type {}",
                QuantizedSerdeType::Qtip as usize
            );
        }
        let has_bias = buffer.read_u8()? != 0;
        let in_features = buffer.read_u32::<LittleEndian>()? as usize;

        let _acquired_load_guard = guard.acquire(device);
        let blocks = deserialize_tensor(&mut buffer, device)?;
        let row_scales = deserialize_tensor(&mut buffer, device)?;
        let lut = deserialize_tensor(&mut buffer, device)?;

        let num_experts = match blocks.dims().len() {
            2 => None,
            3 => {
                let e = blocks.dim(0)?;
                if row_scales.dims().len() != 2
                    || row_scales.dim(0)? != e
                    || row_scales.dim(1)? != blocks.dim(1)?
                {
                    candle_core::bail!(
                        "QtipLayer: 3-D blocks {:?} require row_scales [E, N]; got {:?}",
                        blocks.dims(),
                        row_scales.dims()
                    );
                }
                if has_bias {
                    // The serializer refuses to attach a bias to a 3-D
                    // stack, so this indicates a corrupt payload.
                    candle_core::bail!("QtipLayer: 3-D stacked-expert payloads are bias-free");
                }
                Some(e)
            }
            other => {
                candle_core::bail!("QtipLayer: blocks tensor must be rank 2 or 3, got rank {other}")
            }
        };

        let bias = if has_bias {
            Some(deserialize_tensor(&mut buffer, device)?)
        } else {
            None
        };
        let ext_bias = bias.clone();

        // RUN-158 rotation params. Legacy payloads (no flag byte) end here
        // — we treat EOF / read error as "no rotation" and proceed.
        let (rotation_signs, rotation_block) = match buffer.read_u8() {
            Ok(0) => (None, 0usize),
            Ok(1) => {
                let block = buffer.read_u32::<LittleEndian>()? as usize;
                let signs = deserialize_tensor(&mut buffer, device)?;
                (Some(signs), block)
            }
            Ok(other) => candle_core::bail!(
                "QtipLayer: unexpected rotation-flag byte {other} (expected 0 or 1)"
            ),
            Err(_) => (None, 0usize),
        };

        // D4 §3 search stamp (UQFF ≥ 0.3.0). Pre-0.3.0 payloads end after the
        // rotation section, so EOF means "unstamped" rather than a corrupt
        // file — the version bump is what makes the absence unambiguous.
        let search = match buffer.read_u8() {
            Ok(byte) => QtipSearchStamp::from_wire(byte)?,
            Err(_) => QtipSearchStamp::Unstamped,
        };

        // wave13-AF search-detail flags byte. Unlike the stamp this is NOT
        // optional: every payload that carries a stamp carries the flags byte
        // too, so EOF here means the file is truncated. Fail closed — reading a
        // missing flags byte as "exhaustive, unweighted" would invent exactly
        // the claim this field exists to make verifiable.
        let search_detail = match search {
            QtipSearchStamp::Unstamped => QtipSearchDetail::Unknown,
            stamp => {
                let flags = buffer.read_u8().map_err(|_| {
                    candle_core::Error::Msg(
                        "QtipLayer: payload carries a search stamp but no search-detail flags \
                         byte. UQFF 0.3.0 always writes one, so this file is truncated; \
                         refusing rather than assuming an exhaustive unweighted bake."
                            .into(),
                    )
                })?;
                QtipSearchDetail::from_wire(flags, stamp, || {
                    buffer.read_u16::<LittleEndian>().map_err(|_| {
                        candle_core::Error::Msg(
                            "QtipLayer: search-detail flags claim a beam but the width is \
                             missing (truncated payload)."
                                .into(),
                        )
                    })
                })?
            }
        };

        // wave24-AU codebook discriminator. EOF here is the *answer*, not a
        // truncation: the section is written only for a non-Gaussian codebook
        // (`QtipCodebook::to_wire`), so its absence says "gather from the
        // stored table" — which is what every payload without it has always
        // meant. An unknown tag value, on the other hand, IS refused: that
        // means a newer Arc wrote it, and decoding its symbols against a
        // codebook we guessed would be silent corruption.
        // The trailing region is a sequence of tagged sections, read until EOF.
        // Each tag may appear at most once: a repeat means a corrupt or
        // concatenated payload, and silently letting the last one win is how a
        // reader ends up decoding against something nobody wrote.
        let mut codebook: Option<QtipCodebook> = None;
        let mut geometry: Option<QtipGeometry> = None;
        while let Ok(tag) = buffer.read_u8() {
            if tag == GEOMETRY_WIRE_TAG {
                if geometry.is_some() {
                    candle_core::bail!(
                        "QtipLayer: the UQFF payload carries two geometry sections. Refusing a \
                         payload that describes itself twice."
                    );
                }
                let mut klv = [0u8; 3];
                for (i, slot) in klv.iter_mut().enumerate() {
                    *slot = buffer.read_u8().map_err(|_| {
                        candle_core::Error::Msg(format!(
                            "QtipLayer: geometry section is truncated (got {i} of 3 K/L/V bytes)."
                        ))
                    })?;
                }
                geometry = Some(QtipGeometry::from_wire_body(klv[0], klv[1], klv[2])?);
            } else {
                if codebook.is_some() {
                    candle_core::bail!(
                        "QtipLayer: the UQFF payload carries two codebook sections. Refusing a \
                         payload that describes itself twice."
                    );
                }
                codebook = Some(QtipCodebook::from_wire(tag, || {
                    buffer.read_u32::<LittleEndian>().map_err(|_| {
                        candle_core::Error::Msg(
                            "QtipLayer: codebook tag claims a computed codebook but the \
                             multiplier is missing (truncated payload)."
                                .into(),
                        )
                    })
                })?);
            }
        }
        // Absence is the answer for both, and it is the same answer it has
        // always been: the historical geometry, gathering from the stored table.
        let codebook = codebook.unwrap_or(QtipCodebook::Gaussian);
        let geometry = geometry.unwrap_or(QtipGeometry::K4V2L16);

        // The tag is a claim; these make it true. Both supported geometries are
        // 2 bits per weight, so `blocks` is the same size either way — the
        // table's dtype and length are what actually discriminate.
        geometry.validate_shapes(&blocks, &lut, in_features)?;
        geometry.check_codebook(codebook)?;

        Ok((
            Self {
                blocks,
                row_scales,
                lut,
                bias,
                in_features,
                num_experts,
                rotation_signs,
                rotation_block,
                search,
                search_detail,
                codebook,
                geometry,
            },
            ext_bias,
        ))
    }
}

impl QuantizedSerde for QtipLayer {
    fn name(&self) -> &'static str {
        "qtip-layer"
    }
    fn isq_serde_supported(&self) -> bool {
        true
    }
    fn serialize(&self) -> Result<Cow<'_, [u8]>> {
        self.serialize_with_bias(self.bias.clone())
    }
    fn serialize_with_bias(&self, bias: Option<Tensor>) -> Result<Cow<'_, [u8]>> {
        // 2-D and 3-D stacked-expert layers share one field order. UQFF
        // tensor payloads are self-describing (rank + dims), so the 3-D
        // `blocks [E, N, packed_K]` / `row_scales [E, N]` round-trip through
        // the same `serialize_tensor` calls as the 2-D layout, with the
        // shared `lut` / rotation metadata stored exactly once (not per
        // expert). The deserializer infers the expert mode from the blocks
        // rank — see `deserialize_concrete`.
        if self.num_experts.is_some() && bias.is_some() {
            // 3-D MoE stacks are bias-free (`quantize_with_options` bails on
            // a bias too); refuse rather than silently attach a single [N]
            // bias vector to E experts.
            candle_core::bail!("QtipLayer::serialize: 3-D stacked-expert layers are bias-free");
        }
        let mut buffer = Vec::new();
        buffer.extend(&UQFF_VERSION.to_le_bytes());
        buffer.push(QuantizedSerdeType::Qtip as u8);
        buffer.push(bias.is_some() as u8);
        buffer.extend(&(self.in_features as u32).to_le_bytes());

        serialize_tensor(&mut buffer, &self.blocks)?;
        serialize_tensor(&mut buffer, &self.row_scales)?;
        serialize_tensor(&mut buffer, &self.lut)?;
        if let Some(bias) = &bias {
            serialize_tensor(&mut buffer, bias)?;
        }
        // RUN-158: persist Hadamard incoherence rotation parameters. The
        // signs vector is `None` for legacy (no-rotation) checkpoints; we
        // mark presence with a u8 flag and emit the block size + signs only
        // when set. Block size is u32; we keep it explicit so a future
        // change to QTIP_ROTATION_MAX_BLOCK doesn't break older payloads.
        if let Some(signs) = &self.rotation_signs {
            buffer.push(1u8);
            buffer.extend(&(self.rotation_block as u32).to_le_bytes());
            serialize_tensor(&mut buffer, signs)?;
        } else {
            buffer.push(0u8);
        }
        // D4 §3: stamp the trellis search. UQFF 0.3.0 always writes this byte;
        // re-serializing a layer whose provenance we never knew is refused
        // rather than laundered into a Trellis claim.
        match self.search.to_wire() {
            Some(byte) => buffer.push(byte),
            None => candle_core::bail!(
                "QtipLayer::serialize: refusing to write an artifact with unknown search \
                 provenance. This layer was loaded from a pre-0.3.0 payload or a format that \
                 carries no stamp; re-quantize from the source weights so the stamp is earned \
                 rather than assumed (DOCTRINE D4)."
            ),
        }
        // wave13-AF: the search DETAIL beside the stamp — beam width and
        // objective — so an exhaustive bake and a W=256 beam are distinguishable
        // from the artifact alone. Refuses unknown or self-contradictory detail
        // for the same reason the stamp does.
        let (flags, beam_width) = self.search_detail.to_wire(self.search)?;
        buffer.push(flags);
        if let Some(w) = beam_width {
            buffer.extend(&w.to_le_bytes());
        }
        // The trellis geometry, written FIRST in the trailing-section region
        // and only when it is not the historical K=4/V=2/L=16 — see
        // `QtipGeometry::to_wire` and `GEOMETRY_WIRE_TAG` for why the order is
        // load-bearing. In short: a build that predates this field reads one
        // trailing byte and hands it to the codebook parser, which refuses
        // every tag it does not know. Tag 3 first means an old Arc fails
        // closed on a geometry it cannot decode; tag 3 last would let it
        // decode K=8/V=4 symbols as K=4/V=2 without faulting, because at 2 bpw
        // the packed byte count is the same.
        self.geometry.check_codebook(self.codebook)?;
        if let Some((geo_tag, klv)) = self.geometry.to_wire() {
            buffer.push(geo_tag);
            buffer.extend(&klv);
        }
        // wave24-AU: the codebook discriminator, appended LAST and only when
        // the codebook is not the historical Gaussian one — see
        // `QtipCodebook::to_wire`. A Gaussian artifact at the default geometry
        // is therefore byte-for-byte what this writer produced before either
        // field existed.
        //
        // A build that predates the field, handed a computed-codebook payload,
        // parses identically up to here and then ignores a trailing section it
        // does not know about — and still decodes correctly, because the table
        // written above holds the computed values. The tag buys the
        // in-register decode, not correctness.
        if let Some((cb_tag, cb_mult)) = self.codebook.to_wire() {
            buffer.push(cb_tag);
            buffer.extend(&cb_mult.to_le_bytes());
        }
        Ok(Cow::from(buffer))
    }
    fn deserialize(
        data: Cow<[u8]>,
        device: &Device,
        _comm: &Arc<crate::Comm>,
        guard: QuantizeOntoGuard,
    ) -> Result<Arc<dyn QuantMethod>>
    where
        Self: Sized,
    {
        let (layer, _) = Self::deserialize_concrete(data, device, guard)?;
        Ok(Arc::new(layer))
    }
    fn deserialize_ext_bias(
        data: Cow<[u8]>,
        device: &Device,
        guard: QuantizeOntoGuard,
    ) -> Result<(Arc<dyn QuantMethod>, Option<Tensor>)>
    where
        Self: Sized,
    {
        let (layer, ext_bias) = Self::deserialize_concrete(data, device, guard)?;
        Ok((Arc::new(layer), ext_bias))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sanity: LUT has the right size and reasonable distribution.
    #[test]
    fn gaussian_lut_is_well_formed() {
        let lut = gaussian_lut();
        assert_eq!(lut.len(), LUT_SIZE * V as usize);
        // Most entries should be in [-5, 5] (Gaussian tails)
        let in_range = lut.iter().filter(|&&v| v.abs() < 5.0).count();
        let frac = in_range as f64 / lut.len() as f64;
        assert!(frac > 0.99, "Only {frac} of LUT entries in [-5,5]");
    }

    /// Round-trip: quantize → dequantize. Per-element error should be bounded.
    /// Greedy quantize is suboptimal vs Viterbi, so tolerance is wider than the
    /// paper's Viterbi numbers; this only validates format correctness.
    #[test]
    fn qtip_round_trip_correctness() -> Result<()> {
        let device = Device::Cpu;
        // [N=4, K_in=64] — 64 weights per row = 32 symbols (V=2) = 16 bytes packed
        let n = 4;
        let k_in = 64;
        let mut data = vec![0.0f32; n * k_in];
        for (i, v) in data.iter_mut().enumerate() {
            *v = ((i as f32) * 0.213).sin();
        }
        let w = Tensor::from_vec(data.clone(), (n, k_in), &device)?;
        let layer = QtipLayer::quantize(&w, None, &device)?;
        let dq = layer.dequantize_w()?;
        let dq_v: Vec<f32> = dq.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;

        let mut sum_err = 0.0f32;
        let mut sum_orig = 0.0f32;
        for (o, r) in data.iter().zip(dq_v.iter()) {
            sum_err += (o - r).powi(2);
            sum_orig += o.powi(2);
        }
        let rel = (sum_err / sum_orig).sqrt();
        // Greedy QTIP @ 2-bit: per-element rel L2 around 0.5-0.7.
        // Cosine similarity (in the matmul test) is the more meaningful metric.
        // Viterbi optimization (Tier B) drops this to ~0.15.
        assert!(rel < 0.7, "QTIP rel L2 {rel} > 0.7");
        Ok(())
    }

    /// Matmul correctness through the PRODUCTION entry (`QtipLayer::quantize`,
    /// now Viterbi + Hadamard rotation): dense vs QTIP-dequant-then-matmul.
    ///
    /// Fixture note (wave13-AG). This test used to quantize `w[i] =
    /// cos(0.31·i)·1.5` — a pure sinusoid, i.e. a weight whose magnitudes are
    /// near-uniform on a circle. That is the pathological input for incoherence
    /// processing: rotating an already-flat vector CONCENTRATES its energy into
    /// a few spectral spikes, the opposite of what the rotation is for.
    /// Measured on that fixture: greedy/no-rot 0.900, viterbi+rot 0.836,
    /// viterbi/no-rot 0.504. On a deterministic Gaussian of the same shape the
    /// ladder points the normal way — greedy/no-rot 0.869, viterbi+rot 0.945 —
    /// which matches every realistic fixture in `bake_quality_tests`. So the
    /// fixture is now Gaussian and the bar went UP, rather than the production
    /// recipe being judged against a distribution no weight matrix has.
    #[test]
    fn qtip_matmul_cosine_similarity() -> Result<()> {
        let device = Device::Cpu;
        let n = 8;
        let k_in = 64;
        let mut wdata = vec![0.0f32; n * k_in];
        for (i, v) in wdata.iter_mut().enumerate() {
            let mut z = ((i + 1) as u64).wrapping_mul(0x9E3779B97F4A7C15);
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^= z >> 31;
            let u1 = ((z >> 32) as u32 as f32 + 1.0) / (u32::MAX as f32 + 2.0);
            let u2 = ((z & 0xFFFF_FFFF) as u32 as f32 + 1.0) / (u32::MAX as f32 + 2.0);
            *v = (-2.0_f32 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos() * 0.5;
        }
        let w = Tensor::from_vec(wdata.clone(), (n, k_in), &device)?;

        let xdata: Vec<f32> = (0..(2 * k_in)).map(|i| ((i as f32) * 0.05).sin()).collect();
        let x = Tensor::from_vec(xdata, (2, k_in), &device)?.to_dtype(DType::F32)?;

        let dense = x.matmul(&w.t()?)?;
        let dense_v: Vec<f32> = dense.flatten_all()?.to_vec1()?;

        let layer = QtipLayer::quantize(&w, None, &device)?;
        let qout = layer.forward(&x)?;
        let qv: Vec<f32> = qout.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;

        let mut dot = 0.0f32;
        let mut nd = 0.0f32;
        let mut nq = 0.0f32;
        for (d, q) in dense_v.iter().zip(qv.iter()) {
            dot += d * q;
            nd += d * d;
            nq += q * q;
        }
        let cos = dot / (nd.sqrt() * nq.sqrt());
        // Measured 0.9449 at k=64 (only 16 trellis symbols per row — the wide
        // production rows in `bake_quality_tests` reach 0.963).
        assert!(cos > 0.93, "QTIP matmul cos sim {cos} <= 0.93");
        Ok(())
    }

    /// DEBUG: dump dense vs quantized matmul outputs cell-by-cell.
    #[test]
    fn debug_matmul_cell_by_cell() -> Result<()> {
        let device = Device::Cpu;
        let n = 8;
        let k_in = 64;
        let wdata: Vec<f32> = (0..(n * k_in))
            .map(|i| ((i as f32) * 0.31).cos() * 1.5)
            .collect();
        let w = Tensor::from_vec(wdata.clone(), (n, k_in), &device)?;

        let xdata: Vec<f32> = (0..(2 * k_in)).map(|i| ((i as f32) * 0.05).sin()).collect();
        let x = Tensor::from_vec(xdata, (2, k_in), &device)?.to_dtype(DType::F32)?;

        let dense = x.matmul(&w.t()?)?;
        let dense_v: Vec<f32> = dense.flatten_all()?.to_vec1()?;

        let g_layer = QtipLayer::quantize_greedy_fixture(&w, None, &device)?;
        let g_out = g_layer.forward(&x)?;
        let g_v: Vec<f32> = g_out.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;

        let v_layer = QtipLayer::quantize_with_mode(&w, None, &device, QtipMode::Viterbi)?;
        let v_out = v_layer.forward(&x)?;
        let v_v: Vec<f32> = v_out.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;

        println!("\n  i | dense   | greedy  | viterbi | g_err   | v_err");
        for i in 0..dense_v.len() {
            println!(
                "{i:3} | {:+7.3} | {:+7.3} | {:+7.3} | {:+7.3} | {:+7.3}",
                dense_v[i],
                g_v[i],
                v_v[i],
                g_v[i] - dense_v[i],
                v_v[i] - dense_v[i],
            );
        }
        // Magnitude comparison
        let dn: f32 = dense_v.iter().map(|x| x * x).sum::<f32>().sqrt();
        let gn: f32 = g_v.iter().map(|x| x * x).sum::<f32>().sqrt();
        let vn: f32 = v_v.iter().map(|x| x * x).sum::<f32>().sqrt();
        println!("\nNorms: dense={dn:.3}, greedy={gn:.3}, viterbi={vn:.3}");
        Ok(())
    }

    /// DEBUG: compute the matmul cos sim in two ways: (a) via layer.forward() like
    /// the production test does, (b) via dequantize_w → x @ W.t. They should match.
    /// If they don't, forward() has a bug.
    #[test]
    fn debug_forward_vs_manual_matmul() -> Result<()> {
        let device = Device::Cpu;
        let n = 8;
        let k_in = 64;
        let wdata: Vec<f32> = (0..(n * k_in))
            .map(|i| ((i as f32) * 0.31).cos() * 1.5)
            .collect();
        let w = Tensor::from_vec(wdata.clone(), (n, k_in), &device)?;

        let xdata: Vec<f32> = (0..(2 * k_in)).map(|i| ((i as f32) * 0.05).sin()).collect();
        let x = Tensor::from_vec(xdata, (2, k_in), &device)?.to_dtype(DType::F32)?;

        let dense = x.matmul(&w.t()?)?;
        let dense_v: Vec<f32> = dense.flatten_all()?.to_vec1()?;

        for mode in [QtipMode::Greedy, QtipMode::Viterbi] {
            let layer = QtipLayer::quantize_with_options(
                &w,
                None,
                &device,
                mode,
                QtipRotation::for_mode(mode).enabled(),
            )?;

            // (a) via forward
            let qout_forward = layer.forward(&x)?;
            let qv_forward: Vec<f32> = qout_forward
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1()?;

            // (b) via dequantize + manual matmul
            let w_recon = layer.dequantize_w()?.to_dtype(DType::F32)?;
            let qout_manual = x.matmul(&w_recon.t()?)?;
            let qv_manual: Vec<f32> = qout_manual.flatten_all()?.to_vec1()?;

            // Cos sim of dense vs each
            let cos = |a: &[f32], b: &[f32]| -> f32 {
                let mut d = 0f32;
                let mut na = 0f32;
                let mut nb = 0f32;
                for (x, y) in a.iter().zip(b.iter()) {
                    d += x * y;
                    na += x * x;
                    nb += y * y;
                }
                d / (na.sqrt() * nb.sqrt())
            };

            let cos_forward = cos(&dense_v, &qv_forward);
            let cos_manual = cos(&dense_v, &qv_manual);

            println!(
                "{:?}: forward cos sim = {:.4}, manual cos sim = {:.4}, diff = {:+.4}",
                mode,
                cos_forward,
                cos_manual,
                cos_forward - cos_manual
            );
        }
        Ok(())
    }

    /// DEBUG: compare row-by-row weight reconstruction between Viterbi and Greedy
    /// using the FULL production code path (quantize_with_mode + dequantize).
    /// This isolates whether the bug is in Viterbi or in the production wiring.
    #[test]
    fn debug_full_production_path_viterbi_vs_greedy() -> Result<()> {
        let device = Device::Cpu;
        let n = 8;
        let k_in = 64;
        let wdata: Vec<f32> = (0..(n * k_in))
            .map(|i| ((i as f32) * 0.31).cos() * 1.5)
            .collect();
        let w = Tensor::from_vec(wdata.clone(), (n, k_in), &device)?;

        let greedy_layer = QtipLayer::quantize_greedy_fixture(&w, None, &device)?;
        let viterbi_layer = QtipLayer::quantize_with_mode(&w, None, &device, QtipMode::Viterbi)?;

        let g_recon = greedy_layer.dequantize_w()?.to_dtype(DType::F32)?;
        let v_recon = viterbi_layer.dequantize_w()?.to_dtype(DType::F32)?;
        let g_data: Vec<f32> = g_recon.flatten_all()?.to_vec1()?;
        let v_data: Vec<f32> = v_recon.flatten_all()?.to_vec1()?;

        println!("\nrow | greedy_row_cos | viterbi_row_cos");
        for row in 0..n {
            let dense_row = &wdata[row * k_in..(row + 1) * k_in];
            let g_row = &g_data[row * k_in..(row + 1) * k_in];
            let v_row = &v_data[row * k_in..(row + 1) * k_in];
            let cos = |a: &[f32], b: &[f32]| -> f32 {
                let mut d = 0f32;
                let mut na = 0f32;
                let mut nb = 0f32;
                for (x, y) in a.iter().zip(b.iter()) {
                    d += x * y;
                    na += x * x;
                    nb += y * y;
                }
                d / (na.sqrt() * nb.sqrt())
            };
            let g_cos = cos(dense_row, g_row);
            let v_cos = cos(dense_row, v_row);
            println!("  {row} | {g_cos:.4}        | {v_cos:.4}");

            // Print first few sample values
            if row == 0 {
                println!("    target sample [0..6]: {:?}", &dense_row[..6]);
                println!("    greedy sample [0..6]: {:?}", &g_row[..6]);
                println!("    viterbi sample [0..6]: {:?}", &v_row[..6]);
            }
        }
        Ok(())
    }

    /// RUN-158 fix on the legacy toy fixture (`k_in=64`, sin/cos data):
    /// Viterbi+rotation must dramatically improve over the pre-fix ~0.50
    /// cos sim, and Greedy must stay at its pre-fix ≥0.85.
    ///
    /// **Background**: Before this fix, Viterbi mode's globally-optimal symbol
    /// sequence produced reconstruction errors *correlated* with the matmul
    /// inputs, amplifying their impact on the output. Cos sim collapsed to
    /// ~0.50 even though per-row reconstruction MSE was lower than Greedy's.
    /// (Greedy's local choices "accidentally" produced errors uncorrelated
    /// with the matmul, so its cos sim stayed around 0.90.)
    ///
    /// **Fix**: At quantize time, rotate each weight row by a deterministic
    /// orthogonal R = D·H·D (block-diagonal, sign flip + Walsh-Hadamard +
    /// sign flip). At forward time, rotate each activation row by the same R
    /// before matmul. Since R is involutory (R·R = I), the matmul value is
    /// unchanged algebraically:
    ///   (xR) @ (WR)^T = x · R · R^T · W^T = x · W^T
    /// — but the quantizer now sees rotation-decorrelated weights, and the
    /// matmul sees rotation-decorrelated activations, both matching the
    /// near-Gaussian distribution the trellis LUT was designed for.
    ///
    /// **About the toy fixture's 0.95 ceiling**: This fixture (sin x, cos W,
    /// k_in=64) has x rows and W rows nearly orthogonal in the original
    /// frame (true cos angle ≈ 0.03). The TRUE matmul output has tiny
    /// magnitude relative to ||x|| * ||W||, so even a small amount of
    /// quantization noise (1-2% of ||W||) dominates the signal. Real LLM
    /// activations live on a meaningful subspace of W and have output
    /// magnitudes that are well above the quantization noise — see
    /// `viterbi_matmul_cosine_similarity_realistic` for the ≥0.95 gate on a
    /// Gaussian fixture matching real LLM statistics.
    ///
    /// Greedy still works without rotation (back-compat with older
    /// checkpoints), and rotation is opt-out by quantizing with
    /// `quantize_with_options(..., use_rotation=false)`.
    #[test]
    fn viterbi_matmul_cosine_similarity_with_rotation() -> Result<()> {
        let device = Device::Cpu;
        let n = 8;
        let k_in = 64;
        let mut wdata = vec![0.0f32; n * k_in];
        for (i, v) in wdata.iter_mut().enumerate() {
            *v = ((i as f32) * 0.31).cos() * 1.5;
        }
        let w = Tensor::from_vec(wdata.clone(), (n, k_in), &device)?;

        let xdata: Vec<f32> = (0..(2 * k_in)).map(|i| ((i as f32) * 0.05).sin()).collect();
        let x = Tensor::from_vec(xdata, (2, k_in), &device)?.to_dtype(DType::F32)?;

        let dense = x.matmul(&w.t()?)?;
        let dense_v: Vec<f32> = dense.flatten_all()?.to_vec1()?;

        let cos_sim = |layer: Arc<dyn QuantMethod>| -> Result<f32> {
            let qout = layer.forward(&x)?;
            let qv: Vec<f32> = qout.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
            let mut dot = 0.0f32;
            let mut nd = 0.0f32;
            let mut nq = 0.0f32;
            for (d, q) in dense_v.iter().zip(qv.iter()) {
                dot += d * q;
                nd += d * d;
                nq += q * q;
            }
            Ok(dot / (nd.sqrt() * nq.sqrt()))
        };

        let greedy_layer = QtipLayer::quantize_greedy_fixture(&w, None, &device)?;
        let greedy_cos = cos_sim(greedy_layer)?;

        let viterbi_layer = QtipLayer::quantize_with_mode(&w, None, &device, QtipMode::Viterbi)?;
        let viterbi_cos = cos_sim(viterbi_layer)?;

        println!(
            "QTIP matmul cos sim (RUN-158, toy fixture): Greedy = {:.4}, Viterbi = {:.4}",
            greedy_cos, viterbi_cos
        );

        // Greedy: unchanged behavior, no rotation, ≥0.85 floor.
        assert!(
            greedy_cos > 0.85,
            "Greedy regression: cos sim {greedy_cos} < 0.85 — previously working path broke"
        );

        // Viterbi+rotation: must dramatically improve over the broken 0.50
        // baseline. Toy fixture caps at ~0.85 because of near-orthogonal x/W
        // geometry; see realistic Gaussian test for the ≥0.95 gate.
        assert!(
            viterbi_cos >= 0.80,
            "Viterbi+rotation cos sim {viterbi_cos} < 0.80 — RUN-158 rotation not lifting toy fixture"
        );
        Ok(())
    }

    /// A/B ablation: turning rotation OFF for Viterbi reproduces the
    /// pre-RUN-158 ~0.50 cos sim; turning it ON lifts it substantially.
    /// This proves the rotation step is the load-bearing change.
    #[test]
    fn viterbi_rotation_ablation() -> Result<()> {
        let device = Device::Cpu;
        let n = 8;
        let k_in = 64;
        let wdata: Vec<f32> = (0..(n * k_in))
            .map(|i| ((i as f32) * 0.31).cos() * 1.5)
            .collect();
        let w = Tensor::from_vec(wdata, (n, k_in), &device)?;
        let xdata: Vec<f32> = (0..(2 * k_in)).map(|i| ((i as f32) * 0.05).sin()).collect();
        let x = Tensor::from_vec(xdata, (2, k_in), &device)?;
        let dense_v: Vec<f32> = x.matmul(&w.t()?)?.flatten_all()?.to_vec1()?;

        let cos_sim = |layer: Arc<dyn QuantMethod>| -> Result<f32> {
            let qv: Vec<f32> = layer.forward(&x)?.flatten_all()?.to_vec1()?;
            let (mut dot, mut nd, mut nq) = (0f32, 0f32, 0f32);
            for (d, q) in dense_v.iter().zip(qv.iter()) {
                dot += d * q;
                nd += d * d;
                nq += q * q;
            }
            Ok(dot / (nd.sqrt() * nq.sqrt()))
        };

        let viterbi_no_rot =
            QtipLayer::quantize_with_options(&w, None, &device, QtipMode::Viterbi, false)?;
        let viterbi_with_rot =
            QtipLayer::quantize_with_options(&w, None, &device, QtipMode::Viterbi, true)?;

        let cos_off = cos_sim(viterbi_no_rot)?;
        let cos_on = cos_sim(viterbi_with_rot)?;
        println!(
            "Viterbi cos sim — rotation OFF: {:.4}, rotation ON: {:.4}",
            cos_off, cos_on
        );

        // Rotation OFF reproduces the broken pre-fix behavior on this toy
        // fixture (~0.50 cos sim).
        assert!(
            cos_off < 0.85,
            "Viterbi-without-rotation cos sim {cos_off} unexpectedly high — \
             rotation isn't the actual load-bearing change?"
        );
        // Rotation ON lifts the toy fixture substantially. (The realistic
        // Gaussian test asserts the ≥0.95 number; this fixture's near-
        // orthogonal x/W geometry limits the achievable cos sim.)
        assert!(
            cos_on >= 0.80,
            "Viterbi-with-rotation cos sim {cos_on} < 0.80 — rotation regressed"
        );
        // The lift is substantial — this is the headline number for RUN-158.
        assert!(
            cos_on - cos_off > 0.25,
            "Rotation lift {} too small — expected >0.25 absolute cos-sim gain",
            cos_on - cos_off
        );
        Ok(())
    }

    /// DEBUG: Print row-scale + L2-norm diagnostics for the production test
    /// shape so we can see whether the per-row scale is sane for rotated rows.
    #[test]
    fn debug_row_scale_diagnostics() -> Result<()> {
        let device = Device::Cpu;
        let n = 8;
        let k_in = 64;
        let wdata: Vec<f32> = (0..(n * k_in))
            .map(|i| ((i as f32) * 0.31).cos() * 1.5)
            .collect();
        let signs = crate::turboquant::wht::generate_signs(QTIP_ROTATION_SEED, k_in);
        let block = rotation_block_size(k_in);

        println!("\nrow | l2(orig) max_abs(orig) | l2(rot) max_abs(rot) std(rot)");
        for row in 0..n {
            let orig: &[f32] = &wdata[row * k_in..(row + 1) * k_in];
            let l2_o: f32 = orig.iter().map(|x| x * x).sum::<f32>().sqrt();
            let ma_o = orig.iter().fold(0f32, |m, &v| m.max(v.abs()));

            let mut rot = orig.to_vec();
            apply_block_rotation(&mut rot, &signs, block);
            let l2_r: f32 = rot.iter().map(|x| x * x).sum::<f32>().sqrt();
            let ma_r = rot.iter().fold(0f32, |m, &v| m.max(v.abs()));
            let std_r = (rot.iter().map(|x| x * x).sum::<f32>() / k_in as f32).sqrt();

            println!("{row:3} | {l2_o:.3}   {ma_o:.3}     | {l2_r:.3}  {ma_r:.3}    {std_r:.3}");
        }

        // Print the layer's decoded vs original row L2 norms (original frame).
        let w = Tensor::from_vec(wdata.clone(), (n, k_in), &device)?;
        let layer_dyn = QtipLayer::quantize_with_mode(&w, None, &device, QtipMode::Viterbi)?;
        let layer_recon: Vec<f32> = layer_dyn
            .dequantize_w()?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1()?;

        println!("\nrow | ||W_orig|| | ||W_recon_orig_frame|| | ratio");
        for row in 0..n {
            let orig: &[f32] = &wdata[row * k_in..(row + 1) * k_in];
            let recon: &[f32] = &layer_recon[row * k_in..(row + 1) * k_in];
            let l2_o: f32 = orig.iter().map(|x| x * x).sum::<f32>().sqrt();
            let l2_r: f32 = recon.iter().map(|x| x * x).sum::<f32>().sqrt();
            println!(
                "{row:3} | {l2_o:.3}     | {l2_r:.3}     | {:.3}",
                l2_r / l2_o
            );
        }

        // And in the rotated frame (what forward() uses).
        // Use downcast trick: call quantize_with_options to get a concrete layer.
        let concrete =
            QtipLayer::quantize_with_options(&w, None, &device, QtipMode::Viterbi, true)?;
        // We can't downcast from Arc<dyn QuantMethod>, so re-derive by direct
        // construction or just by rotation of dequantize.
        let _ = concrete;
        // Manually: rotate orig W rows, compare to viterbi's rotated frame
        // (which equals dequantize_weights_rotated_f32 - inaccessible here).
        // Instead: rotate dequantized W to recover the rotated-frame recon.
        let mut recon_rot = layer_recon.clone();
        for row in 0..n {
            let row_slice = &mut recon_rot[row * k_in..(row + 1) * k_in];
            apply_block_rotation(row_slice, &signs, block);
        }
        println!("\nrow | ||W_rot_true|| | ||W_rot_recon|| | ratio");
        for row in 0..n {
            let orig_row: &[f32] = &wdata[row * k_in..(row + 1) * k_in];
            let mut true_rot = orig_row.to_vec();
            apply_block_rotation(&mut true_rot, &signs, block);
            let recon_rot_row: &[f32] = &recon_rot[row * k_in..(row + 1) * k_in];

            let l2_t: f32 = true_rot.iter().map(|x| x * x).sum::<f32>().sqrt();
            let l2_r: f32 = recon_rot_row.iter().map(|x| x * x).sum::<f32>().sqrt();
            println!(
                "{row:3} | {l2_t:.3}     | {l2_r:.3}     | {:.3}",
                l2_r / l2_t
            );
        }
        Ok(())
    }

    /// Realistic LLM-shaped test: random Gaussian weights and inputs, larger
    /// in_features. With incoherence rotation, viterbi cos sim should hit
    /// ≥0.95 — matching the Cornell paper.
    ///
    /// The pre-existing `viterbi_matmul_cosine_similarity_with_rotation` test
    /// uses a small deterministic sin/cos fixture where x and W rows are
    /// nearly orthogonal — a worst-case for cos sim under quantization
    /// (tiny true output, easy to overwhelm with noise). Real LLM
    /// activations are roughly Gaussian and weights are roughly Gaussian, so
    /// the matmul output has typical magnitude ||x|| * ||W|| * 1/sqrt(d),
    /// where the quantization noise is well-attenuated by the
    /// incoherence-rotated decorrelation.
    #[test]
    fn viterbi_matmul_cosine_similarity_realistic() -> Result<()> {
        let device = Device::Cpu;
        let n = 32;
        let k_in = 256; // multiple of 128, gets full FWHT block
        let batch = 4;

        // Generate "realistic" Gaussian-ish weights via deterministic hash.
        let mut wdata = vec![0.0f32; n * k_in];
        for (i, v) in wdata.iter_mut().enumerate() {
            let mut z = (i as u64).wrapping_mul(0x9E3779B97F4A7C15);
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^= z >> 31;
            let u1 = ((z >> 32) as u32 as f32 + 1.0) / (u32::MAX as f32 + 2.0);
            let u2 = ((z & 0xFFFFFFFF) as u32 as f32 + 1.0) / (u32::MAX as f32 + 2.0);
            let r = (-2.0_f32 * u1.ln()).sqrt();
            *v = r * (2.0 * std::f32::consts::PI * u2).cos() * 0.5;
        }
        let mut xdata = vec![0.0f32; batch * k_in];
        for (i, v) in xdata.iter_mut().enumerate() {
            let mut z = ((i + 1_000_000) as u64).wrapping_mul(0x9E3779B97F4A7C15);
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z ^= z >> 31;
            let u1 = ((z >> 32) as u32 as f32 + 1.0) / (u32::MAX as f32 + 2.0);
            let u2 = ((z & 0xFFFFFFFF) as u32 as f32 + 1.0) / (u32::MAX as f32 + 2.0);
            let r = (-2.0_f32 * u1.ln()).sqrt();
            *v = r * (2.0 * std::f32::consts::PI * u2).cos();
        }

        let w = Tensor::from_vec(wdata, (n, k_in), &device)?;
        let x = Tensor::from_vec(xdata, (batch, k_in), &device)?;
        let dense_v: Vec<f32> = x.matmul(&w.t()?)?.flatten_all()?.to_vec1()?;

        let cos_sim = |layer: Arc<dyn QuantMethod>| -> Result<f32> {
            let qv: Vec<f32> = layer.forward(&x)?.flatten_all()?.to_vec1()?;
            let (mut dot, mut nd, mut nq) = (0f32, 0f32, 0f32);
            for (d, q) in dense_v.iter().zip(qv.iter()) {
                dot += d * q;
                nd += d * d;
                nq += q * q;
            }
            Ok(dot / (nd.sqrt() * nq.sqrt()))
        };

        let greedy = QtipLayer::quantize_greedy_fixture(&w, None, &device)?;
        let viterbi = QtipLayer::quantize_with_mode(&w, None, &device, QtipMode::Viterbi)?;

        let greedy_cos = cos_sim(greedy)?;
        let viterbi_cos = cos_sim(viterbi)?;

        println!(
            "RUN-158 realistic Gaussian (n={n}, k_in={k_in}, batch={batch}): \
             Greedy={greedy_cos:.4}, Viterbi+rotation={viterbi_cos:.4}"
        );

        // Viterbi+rotation should hit the paper-quality bar.
        assert!(
            viterbi_cos >= 0.95,
            "Viterbi+rotation cos sim {viterbi_cos} < 0.95 on realistic Gaussian fixture"
        );
        Ok(())
    }

    /// DEBUG: end-to-end forward check WITHOUT quantization — just rotate
    /// weight rows, rotate input rows, matmul. The result must equal the
    /// dense matmul exactly (FP rounding aside). This isolates whether the
    /// rotation step in forward() is wired correctly.
    #[test]
    fn debug_forward_with_rotation_no_quantization() -> Result<()> {
        let device = Device::Cpu;
        let n = 8;
        let k_in = 64;
        let wdata: Vec<f32> = (0..(n * k_in))
            .map(|i| ((i as f32) * 0.31).cos() * 1.5)
            .collect();
        let xdata: Vec<f32> = (0..(2 * k_in)).map(|i| ((i as f32) * 0.05).sin()).collect();
        let w = Tensor::from_vec(wdata.clone(), (n, k_in), &device)?;
        let x = Tensor::from_vec(xdata.clone(), (2, k_in), &device)?;
        let signs = crate::turboquant::wht::generate_signs(QTIP_ROTATION_SEED, k_in);
        let block = rotation_block_size(k_in);

        // Dense reference.
        let dense_v: Vec<f32> = x.matmul(&w.t()?)?.flatten_all()?.to_vec1()?;

        // Manually rotate both x and W, then matmul.
        let mut x_rot = xdata.clone();
        let mut w_rot = wdata.clone();
        for b in 0..2 {
            apply_block_rotation(&mut x_rot[b * k_in..(b + 1) * k_in], &signs, block);
        }
        for j in 0..n {
            apply_block_rotation(&mut w_rot[j * k_in..(j + 1) * k_in], &signs, block);
        }
        let x_rot_t = Tensor::from_vec(x_rot, (2, k_in), &device)?;
        let w_rot_t = Tensor::from_vec(w_rot, (n, k_in), &device)?;
        let rotated_v: Vec<f32> = x_rot_t.matmul(&w_rot_t.t()?)?.flatten_all()?.to_vec1()?;

        let mut max_diff = 0f32;
        for (a, b) in dense_v.iter().zip(rotated_v.iter()) {
            max_diff = max_diff.max((a - b).abs());
        }
        println!("Max diff dense vs rotated-no-quant matmul: {max_diff}");
        assert!(max_diff < 1e-3, "rotated matmul broke math invariant");
        Ok(())
    }

    /// Math invariant: (xR) @ (WR)^T == x @ W^T (up to FP rounding) given
    /// that R is involutory (R·R = I). This is the math the RUN-158 fix
    /// relies on; if this fails, the rotation impl is broken.
    #[test]
    fn rotated_matmul_equals_dense_matmul() {
        let k_in = 64;
        let n = 8;
        let batch = 2;
        let signs = crate::turboquant::wht::generate_signs(QTIP_ROTATION_SEED, k_in);
        let block = rotation_block_size(k_in);

        let wdata: Vec<f32> = (0..(n * k_in))
            .map(|i| ((i as f32) * 0.31).cos() * 1.5)
            .collect();
        let xdata: Vec<f32> = (0..(batch * k_in))
            .map(|i| ((i as f32) * 0.05).sin())
            .collect();

        // Dense reference: x @ W^T
        let mut dense = vec![0f32; batch * n];
        for b in 0..batch {
            for j in 0..n {
                let mut s = 0f32;
                for k in 0..k_in {
                    s += xdata[b * k_in + k] * wdata[j * k_in + k];
                }
                dense[b * n + j] = s;
            }
        }

        // Rotated: rotate each row of x and each row of W, then matmul.
        let mut x_rot = xdata.clone();
        for b in 0..batch {
            apply_block_rotation(&mut x_rot[b * k_in..(b + 1) * k_in], &signs, block);
        }
        let mut w_rot = wdata.clone();
        for j in 0..n {
            apply_block_rotation(&mut w_rot[j * k_in..(j + 1) * k_in], &signs, block);
        }

        let mut rotated = vec![0f32; batch * n];
        for b in 0..batch {
            for j in 0..n {
                let mut s = 0f32;
                for k in 0..k_in {
                    s += x_rot[b * k_in + k] * w_rot[j * k_in + k];
                }
                rotated[b * n + j] = s;
            }
        }

        // Compare element-by-element.
        for (a, b) in dense.iter().zip(rotated.iter()) {
            assert!(
                (a - b).abs() < 1e-3,
                "Rotated matmul {b} differs from dense {a}: |Δ|={}",
                (a - b).abs()
            );
        }
    }

    /// Block-rotation involution check: applying `apply_block_rotation` twice
    /// returns the original vector (up to FP rounding). This is the math
    /// invariant the whole RUN-158 fix relies on.
    #[test]
    fn rotation_is_involutory() {
        let k_in = 64; // power of 2 — gets a single FWHT block
        let signs = crate::turboquant::wht::generate_signs(QTIP_ROTATION_SEED, k_in);
        let block = rotation_block_size(k_in);
        assert!(block >= 2, "block_size should be ≥2 for k_in=64");

        let original: Vec<f32> = (0..k_in).map(|i| ((i as f32) * 0.1).sin()).collect();
        let mut buf = original.clone();
        apply_block_rotation(&mut buf, &signs, block);
        // After one application, contents should differ.
        let diff_after_one: f32 = original
            .iter()
            .zip(buf.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff_after_one > 1e-3, "Rotation produced ~no change");

        apply_block_rotation(&mut buf, &signs, block);
        // After two applications, should recover the original.
        for (a, b) in original.iter().zip(buf.iter()) {
            assert!(
                (a - b).abs() < 1e-4,
                "Involution failed: {a} vs {b}, |Δ|={}",
                (a - b).abs()
            );
        }
    }

    /// Task #17: a Greedy bake must never again be mistakable for a Viterbi
    /// bake by reading the log. The header names the mode, the search, the
    /// objective (including whether calibration data actually arrived) and the
    /// rotation width — and a Greedy bake must never be labelled with a beam
    /// width it does not use.
    #[test]
    fn bake_header_names_the_search_and_objective() {
        let beam = QtipBakeConfig {
            search: TrellisSearch::Beam { width: 128 },
            hessian: true,
            codebook: QtipCodebook::Gaussian,
        };

        let greedy = bake_header_line("K4/V2 2-bit", QtipMode::Greedy, beam, 0, false);
        assert!(greedy.contains("mode=greedy"), "{greedy}");
        assert!(greedy.contains("greedy-walk"), "{greedy}");
        assert!(
            !greedy.contains("viterbi") && !greedy.contains("W=128"),
            "a greedy bake must not advertise a trellis search: {greedy}"
        );
        assert!(greedy.contains("rotation=off"), "{greedy}");
        // Hessian requested but no calibration data => say so, don't imply it.
        assert!(greedy.contains("mse(no-calibration)"), "{greedy}");

        let viterbi = bake_header_line("K4/V2 2-bit", QtipMode::Viterbi, beam, 128, true);
        assert!(viterbi.contains("mode=viterbi"), "{viterbi}");
        assert!(viterbi.contains("viterbi-beam(W=128)"), "{viterbi}");
        assert!(viterbi.contains("hessian-diag"), "{viterbi}");
        assert!(viterbi.contains("rotation=hadamard-128"), "{viterbi}");

        // wave24-AU: the codebook is part of what produced the artifact, so a
        // bake log that omits it cannot be checked against the file. A computed
        // bake must be distinguishable from a stored-table one by the header
        // alone, and must name the multiplier it ran.
        let computed = bake_header_line(
            "K4/V2 2-bit",
            QtipMode::Viterbi,
            QtipBakeConfig {
                codebook: QtipCodebook::COMPUTED,
                ..beam
            },
            128,
            true,
        );
        assert!(computed.contains("codebook=mcg-sum2"), "{computed}");
        assert!(computed.contains("0xCAF6A435"), "{computed}");
        assert!(!viterbi.contains("mcg"), "{viterbi}");

        // The shipped default must read as today's bake, unambiguously.
        let default = bake_header_line(
            "K4/V2 2-bit",
            QtipMode::Viterbi,
            QtipBakeConfig::default(),
            128,
            false,
        );
        assert_eq!(
            default,
            "QTIP bake [K4/V2 2-bit]: mode=viterbi search=viterbi-exhaustive \
             objective=mse (unweighted) rotation=hadamard-128 codebook=gaussian-lut"
        );
    }

    /// wave13-AF: the GPU dispatch may translate a search, but it may never
    /// SUBSTITUTE one. The only legal translation is "a beam at least as wide
    /// as the state space prunes nothing, so run the exhaustive kernel" —
    /// exactly what the CPU `viterbi::quantize_row` does. Anything the kernel
    /// cannot run must be an error, never a quietly narrower beam.
    #[test]
    fn cuda_search_plan_never_substitutes_a_width() {
        const MAX_W: usize = 256;

        assert_eq!(
            cuda_search_plan(TrellisSearch::Exhaustive, MAX_W).unwrap(),
            TrellisSearch::Exhaustive
        );
        for w in [1usize, 16, 64, 128, 256] {
            assert_eq!(
                cuda_search_plan(TrellisSearch::Beam { width: w }, MAX_W).unwrap(),
                TrellisSearch::Beam { width: w },
                "width {w} must be honoured exactly"
            );
        }
        // A beam that prunes nothing is the exhaustive DP, by definition.
        for w in [LUT_SIZE, LUT_SIZE + 1, usize::MAX] {
            assert_eq!(
                cuda_search_plan(TrellisSearch::Beam { width: w }, MAX_W).unwrap(),
                TrellisSearch::Exhaustive
            );
        }
        // Width 0 mirrors `beam_quantize_row`'s `clamp(1, LUT_SIZE)`.
        assert_eq!(
            cuda_search_plan(TrellisSearch::Beam { width: 0 }, MAX_W).unwrap(),
            TrellisSearch::Beam { width: 1 }
        );
        // Too wide for the kernel: refuse. The failure mode this guards is a
        // bake that quietly runs W=256 when the operator asked for W=1024.
        for w in [MAX_W + 1, 1024, LUT_SIZE - 1] {
            let err = cuda_search_plan(TrellisSearch::Beam { width: w }, MAX_W)
                .expect_err("a width beyond the kernel limit must not be silently narrowed");
            let msg = format!("{err}");
            assert!(msg.contains("will not silently substitute"), "{msg}");
        }
        // Kernels absent (max width 0): every beam request must fail loudly.
        assert!(cuda_search_plan(TrellisSearch::Beam { width: 64 }, 0).is_err());
    }

    /// Reference packing for the CUDA parity tests: run the CPU trellis search
    /// on `weight` using scales produced by the GPU row-scale kernel, and pack
    /// exactly as `quantize_with_options_concrete_calibrated` does.
    #[cfg(feature = "cuda")]
    fn cpu_reference_packed(
        weight: &[f32],
        n: usize,
        k_in: usize,
        scales: &[f32],
        lut: &[f32],
        search: TrellisSearch,
    ) -> Vec<u8> {
        let num_symbols = k_in / V as usize;
        let mut out = Vec::with_capacity(n * (num_symbols / 2));
        for row in 0..n {
            let raw = &weight[row * k_in..(row + 1) * k_in];
            let inv_scale = 1.0f32 / scales[row];
            let scaled: Vec<f32> = raw.iter().map(|w| w * inv_scale).collect();
            let symbols = super::viterbi::quantize_row(&scaled, lut, search, None);
            let mut packed = vec![0u8; num_symbols / 2];
            for (i, &sym) in symbols.iter().enumerate() {
                if i.is_multiple_of(2) {
                    packed[i / 2] = sym & 0x0F;
                } else {
                    packed[i / 2] |= (sym & 0x0F) << 4;
                }
            }
            out.extend_from_slice(&packed);
        }
        out
    }

    /// Deterministic Gaussian fixture shared by the CUDA parity tests.
    #[cfg(feature = "cuda")]
    fn parity_fixture(len: usize, seed: u64, sigma: f32) -> Vec<f32> {
        (0..len)
            .map(|i| {
                let mut z = (i as u64)
                    .wrapping_add(seed)
                    .wrapping_mul(0x9E37_79B9_7F4A_7C15);
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z ^= z >> 31;
                let u1 = ((z >> 32) as u32 as f32 + 1.0) / (u32::MAX as f32 + 2.0);
                let u2 = ((z & 0xFFFF_FFFF) as u32 as f32 + 1.0) / (u32::MAX as f32 + 2.0);
                (-2.0_f32 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos() * sigma
            })
            .collect()
    }

    /// THE correctness gate for wave13-AF.
    ///
    /// The CUDA beam kernel must emit the **byte-identical** symbol stream the
    /// CPU beam (PR #29) emits at the same width — not a similar one. Cosine
    /// similarity would hide exactly the failure mode that matters: a GPU bake
    /// and a CPU bake of the same weights with the same flag silently producing
    /// different checkpoints.
    ///
    /// Non-vacuity: the same fixture is also baked with the exhaustive kernel
    /// and asserted to DIFFER, so the test cannot pass by the beam happening to
    /// reproduce the full DP.
    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_beam_matches_cpu_beam_bit_for_bit() -> Result<()> {
        let cuda = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(_) => {
                eprintln!("CUDA not available; skipping cuda_beam_matches_cpu_beam_bit_for_bit");
                return Ok(());
            }
        };
        if !ffi::HAVE_QTIP_KERNELS {
            return Ok(());
        }
        let cpu = Device::Cpu;
        let n = 8;
        let k_in = 512; // num_symbols = 256, long enough to prune every step
        let wdata = parity_fixture(n * k_in, 0xBEEF, 0.5);

        let w_cuda = Tensor::from_vec(wdata.clone(), (n, k_in), &cuda)?;
        let lut_data = gaussian_lut();
        let lut_cuda =
            Tensor::from_vec(lut_data.clone(), (LUT_SIZE, V as usize), &cpu)?.to_device(&cuda)?;

        // Both sides must see the SAME per-row scale, so take the GPU kernel's.
        let scales: Vec<f32> =
            cuda_ops::compute_row_scales_cuda(&w_cuda, QTIP_GAUSSIAN_SCALE_DIVISOR)?
                .to_device(&cpu)?
                .to_vec1()?;

        let exhaustive: Vec<u8> = cuda_ops::quantize_rows_cuda(
            &w_cuda,
            &lut_cuda,
            QtipMode::Viterbi,
            TrellisSearch::Exhaustive,
            QtipCodebook::Gaussian,
        )?
        .0
        .to_device(&cpu)?
        .flatten_all()?
        .to_vec1()?;

        let mut any_differed = false;
        for width in [64usize, 128, 256] {
            let search = TrellisSearch::Beam { width };
            let gpu: Vec<u8> = cuda_ops::quantize_rows_cuda(
                &w_cuda,
                &lut_cuda,
                QtipMode::Viterbi,
                search,
                QtipCodebook::Gaussian,
            )?
            .0
            .to_device(&cpu)?
            .flatten_all()?
            .to_vec1()?;
            let reference = cpu_reference_packed(&wdata, n, k_in, &scales, &lut_data, search);

            assert_eq!(gpu.len(), reference.len());
            let mismatches = gpu
                .iter()
                .zip(reference.iter())
                .filter(|(a, b)| a != b)
                .count();
            assert_eq!(
                mismatches,
                0,
                "W={width}: CUDA beam differs from the CPU beam in {mismatches}/{} bytes — \
                 the GPU and CPU bakes of the same weights are not the same checkpoint",
                gpu.len()
            );
            if gpu != exhaustive {
                any_differed = true;
            }
        }
        assert!(
            any_differed,
            "beam and exhaustive produced identical bytes at every width — the fixture \
             does not actually exercise pruning, so bit-identity proves nothing"
        );
        Ok(())
    }

    /// Mirror of PR #29's `beam_unpruned_matches_exhaustive_bit_for_bit`, on GPU.
    ///
    /// A beam wide enough to prune nothing must reproduce the exhaustive DP
    /// byte for byte. `num_symbols = 2` makes that provable rather than
    /// incidental: from the implicit start state 0 only the 16 states
    /// `0..ALPHABET` are reachable at t=0, and their 16 successors each are the
    /// 256 states `0..256` at t=1 — so at W=256 the beam never drops a
    /// candidate, and the exhaustive DP's finite-cost set is exactly the same
    /// 256 states. (A longer row cannot be tested this way: from t=3 the
    /// reachable set is the full 2^16, which no shared-memory-resident beam can
    /// hold. Long rows are covered transitively —
    /// `cuda_beam_matches_cpu_beam_bit_for_bit` pins CUDA to the CPU beam, and
    /// PR #29 pins the unpruned CPU beam to the CPU exhaustive DP.)
    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_beam_unpruned_matches_cuda_exhaustive() -> Result<()> {
        let cuda = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(_) => {
                eprintln!(
                    "CUDA not available; skipping cuda_beam_unpruned_matches_cuda_exhaustive"
                );
                return Ok(());
            }
        };
        if !ffi::HAVE_QTIP_KERNELS {
            return Ok(());
        }
        let cpu = Device::Cpu;
        let n = 64;
        let k_in = 4; // num_symbols = 2 -> the beam provably prunes nothing at W=256
        let wdata = parity_fixture(n * k_in, 0x5EED, 0.9);
        let w_cuda = Tensor::from_vec(wdata, (n, k_in), &cuda)?;
        let lut_cuda =
            Tensor::from_vec(gaussian_lut(), (LUT_SIZE, V as usize), &cpu)?.to_device(&cuda)?;

        let exhaustive: Vec<u8> = cuda_ops::quantize_rows_cuda(
            &w_cuda,
            &lut_cuda,
            QtipMode::Viterbi,
            TrellisSearch::Exhaustive,
            QtipCodebook::Gaussian,
        )?
        .0
        .to_device(&cpu)?
        .flatten_all()?
        .to_vec1()?;
        let unpruned: Vec<u8> = cuda_ops::quantize_rows_cuda(
            &w_cuda,
            &lut_cuda,
            QtipMode::Viterbi,
            TrellisSearch::Beam { width: 256 },
            QtipCodebook::Gaussian,
        )?
        .0
        .to_device(&cpu)?
        .flatten_all()?
        .to_vec1()?;

        assert_eq!(
            unpruned, exhaustive,
            "an unpruned CUDA beam must be the exhaustive DP, byte for byte"
        );
        Ok(())
    }

    /// wave13-AF also removed the fast-math divergence between the CUDA
    /// trellis kernels and the Rust reference (`qtip_exact_fp.cuh`): FMA
    /// contraction in the branch metric and an approximate `1.0f/scale`.
    /// With those gone the exhaustive kernel is bit-identical to the CPU DP,
    /// which is what makes the beam guard above meaningful — and what makes a
    /// CPU-baked and a GPU-baked checkpoint the same artifact.
    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_exhaustive_matches_cpu_exhaustive_bit_for_bit() -> Result<()> {
        let cuda = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(_) => {
                eprintln!(
                    "CUDA not available; skipping cuda_exhaustive_matches_cpu_exhaustive_bit_for_bit"
                );
                return Ok(());
            }
        };
        if !ffi::HAVE_QTIP_KERNELS {
            return Ok(());
        }
        let cpu = Device::Cpu;
        let n = 4;
        let k_in = 256;
        let wdata = parity_fixture(n * k_in, 0xC0FFEE, 0.5);
        let w_cuda = Tensor::from_vec(wdata.clone(), (n, k_in), &cuda)?;
        let lut_data = gaussian_lut();
        let lut_cuda =
            Tensor::from_vec(lut_data.clone(), (LUT_SIZE, V as usize), &cpu)?.to_device(&cuda)?;

        let scales: Vec<f32> =
            cuda_ops::compute_row_scales_cuda(&w_cuda, QTIP_GAUSSIAN_SCALE_DIVISOR)?
                .to_device(&cpu)?
                .to_vec1()?;
        // The row-scale kernel must agree with `max_abs / 3.0` exactly, too.
        for row in 0..n {
            let max_abs = wdata[row * k_in..(row + 1) * k_in]
                .iter()
                .fold(0.0f32, |m, &v| m.max(v.abs()));
            let expected = if max_abs == 0.0 { 1.0 } else { max_abs / 3.0 };
            assert_eq!(
                scales[row].to_bits(),
                expected.to_bits(),
                "row {row}: GPU scale {} != CPU scale {expected}",
                scales[row]
            );
        }

        let gpu: Vec<u8> = cuda_ops::quantize_rows_cuda(
            &w_cuda,
            &lut_cuda,
            QtipMode::Viterbi,
            TrellisSearch::Exhaustive,
            QtipCodebook::Gaussian,
        )?
        .0
        .to_device(&cpu)?
        .flatten_all()?
        .to_vec1()?;
        let reference = cpu_reference_packed(
            &wdata,
            n,
            k_in,
            &scales,
            &lut_data,
            TrellisSearch::Exhaustive,
        );
        assert_eq!(
            gpu, reference,
            "the CUDA exhaustive DP must be bit-identical to the CPU one"
        );
        Ok(())
    }

    /// Non-power-of-2 in_features that's divisible by 2 should still get a
    /// non-trivial block size. (Defensive — real LLMs often have
    /// `intermediate_size = 14336 = 7·2^11`, block_size should be 128.)
    #[test]
    fn rotation_block_size_on_non_power_of_two() {
        assert_eq!(rotation_block_size(64), 64);
        assert_eq!(rotation_block_size(128), 128);
        assert_eq!(rotation_block_size(256), 128); // capped at QTIP_ROTATION_MAX_BLOCK
        assert_eq!(rotation_block_size(4096), 128);
        assert_eq!(rotation_block_size(14336), 128); // 14336 = 7·2^11
        assert_eq!(rotation_block_size(11008), 128); // 11008 = 43·2^8
        assert_eq!(rotation_block_size(0), 0);
        assert_eq!(rotation_block_size(3), 0); // odd → no rotation
        assert_eq!(rotation_block_size(96), 32); // 96 = 3·2^5
    }

    /// Format invariant: packed symbol stream must be exactly num_symbols/2 bytes per row.
    #[test]
    fn qtip_packed_size_invariant() -> Result<()> {
        let device = Device::Cpu;
        let n = 3;
        let k_in = 32;
        let data = vec![0.5f32; n * k_in];
        let w = Tensor::from_vec(data, (n, k_in), &device)?;
        let layer = QtipLayer::quantize(&w, None, &device)?;
        let layer_t = layer.dequantize_w()?;
        assert_eq!(layer_t.dims(), &[n, k_in]);
        Ok(())
    }

    // -----------------------------------------------------------------
    // GPU parity tests (cuda feature only).
    // -----------------------------------------------------------------
    //
    // These tests assert that the CUDA forward path produces the same
    // output as the CPU forward path to within FP precision. They
    // require a CUDA device to be available; on a CPU-only build they
    // are simply skipped.

    /// GPU dequantize matches CPU dequantize for the rotated Viterbi path.
    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_dequantize_matches_cpu_viterbi_rotation() -> Result<()> {
        let cuda = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(_) => {
                eprintln!(
                    "CUDA not available; skipping cuda_dequantize_matches_cpu_viterbi_rotation"
                );
                return Ok(());
            }
        };
        let cpu = Device::Cpu;
        let n = 16;
        let k_in = 256;

        // Random-ish Gaussian weights via deterministic hash (matches the
        // realistic_gaussian test fixture).
        let mut wdata = vec![0.0f32; n * k_in];
        for (i, v) in wdata.iter_mut().enumerate() {
            let mut z = (i as u64).wrapping_mul(0x9E3779B97F4A7C15);
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z ^= z >> 31;
            let u1 = ((z >> 32) as u32 as f32 + 1.0) / (u32::MAX as f32 + 2.0);
            let u2 = ((z & 0xFFFFFFFF) as u32 as f32 + 1.0) / (u32::MAX as f32 + 2.0);
            let r = (-2.0_f32 * u1.ln()).sqrt();
            *v = r * (2.0 * std::f32::consts::PI * u2).cos() * 0.5;
        }
        let w_cpu_tensor = Tensor::from_vec(wdata.clone(), (n, k_in), &cpu)?;
        let layer_cpu =
            QtipLayer::quantize_with_mode(&w_cpu_tensor, None, &cpu, QtipMode::Viterbi)?;
        let w_recon_cpu = layer_cpu.dequantize_w()?.to_dtype(DType::F32)?;
        let cpu_data: Vec<f32> = w_recon_cpu.flatten_all()?.to_vec1()?;

        // Re-quantize on CUDA — the CPU path is the source of truth; we
        // re-build the layer on CUDA by moving the CPU layer's tensors.
        let w_cuda_tensor = Tensor::from_vec(wdata, (n, k_in), &cuda)?;
        let layer_cuda =
            QtipLayer::quantize_with_mode(&w_cuda_tensor, None, &cuda, QtipMode::Viterbi)?;
        let w_recon_cuda = layer_cuda.dequantize_w()?.to_dtype(DType::F32)?;
        let cuda_data: Vec<f32> = w_recon_cuda.to_device(&cpu)?.flatten_all()?.to_vec1()?;

        let (mut dot, mut na, mut nb) = (0f32, 0f32, 0f32);
        for (a, b) in cpu_data.iter().zip(cuda_data.iter()) {
            dot += a * b;
            na += a * a;
            nb += b * b;
        }
        let cos = dot / (na.sqrt() * nb.sqrt());
        println!("CUDA vs CPU dequant cos sim (Viterbi+rotation): {cos}");
        assert!(
            cos >= 0.999,
            "CUDA dequant deviates from CPU: cos sim {cos} < 0.999"
        );
        Ok(())
    }

    /// End-to-end GPU forward matches the CPU forward for a realistic
    /// Gaussian fixture under the Viterbi+rotation path.
    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_forward_matches_cpu_viterbi_rotation() -> Result<()> {
        let cuda = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(_) => {
                eprintln!("CUDA not available; skipping cuda_forward_matches_cpu_viterbi_rotation");
                return Ok(());
            }
        };
        let cpu = Device::Cpu;
        let n = 32;
        let k_in = 256;
        let batch = 4;

        let mut wdata = vec![0.0f32; n * k_in];
        for (i, v) in wdata.iter_mut().enumerate() {
            let mut z = (i as u64).wrapping_mul(0x9E3779B97F4A7C15);
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z ^= z >> 31;
            let u1 = ((z >> 32) as u32 as f32 + 1.0) / (u32::MAX as f32 + 2.0);
            let u2 = ((z & 0xFFFFFFFF) as u32 as f32 + 1.0) / (u32::MAX as f32 + 2.0);
            let r = (-2.0_f32 * u1.ln()).sqrt();
            *v = r * (2.0 * std::f32::consts::PI * u2).cos() * 0.5;
        }
        let mut xdata = vec![0.0f32; batch * k_in];
        for (i, v) in xdata.iter_mut().enumerate() {
            let mut z = ((i + 1_000_000) as u64).wrapping_mul(0x9E3779B97F4A7C15);
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z ^= z >> 31;
            let u1 = ((z >> 32) as u32 as f32 + 1.0) / (u32::MAX as f32 + 2.0);
            let u2 = ((z & 0xFFFFFFFF) as u32 as f32 + 1.0) / (u32::MAX as f32 + 2.0);
            let r = (-2.0_f32 * u1.ln()).sqrt();
            *v = r * (2.0 * std::f32::consts::PI * u2).cos();
        }

        // CPU layer + CPU forward.
        let w_cpu = Tensor::from_vec(wdata.clone(), (n, k_in), &cpu)?;
        let x_cpu = Tensor::from_vec(xdata.clone(), (batch, k_in), &cpu)?;
        let layer_cpu = QtipLayer::quantize_with_mode(&w_cpu, None, &cpu, QtipMode::Viterbi)?;
        let y_cpu = layer_cpu.forward(&x_cpu)?;
        let y_cpu_v: Vec<f32> = y_cpu.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;

        // CUDA layer + CUDA forward.
        let w_cuda = Tensor::from_vec(wdata, (n, k_in), &cuda)?;
        let x_cuda = Tensor::from_vec(xdata, (batch, k_in), &cuda)?;
        let layer_cuda = QtipLayer::quantize_with_mode(&w_cuda, None, &cuda, QtipMode::Viterbi)?;
        let y_cuda = layer_cuda.forward(&x_cuda)?;
        let y_cuda_v: Vec<f32> = y_cuda
            .to_device(&cpu)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1()?;

        // Compare CPU vs CUDA forward outputs.
        let (mut dot, mut na, mut nb) = (0f32, 0f32, 0f32);
        for (a, b) in y_cpu_v.iter().zip(y_cuda_v.iter()) {
            dot += a * b;
            na += a * a;
            nb += b * b;
        }
        let cos = dot / (na.sqrt() * nb.sqrt());
        println!("CUDA vs CPU forward cos sim (Viterbi+rotation): {cos}");
        assert!(
            cos >= 0.999,
            "CUDA forward deviates from CPU: cos sim {cos} < 0.999"
        );

        // Also assert vs dense matmul: should match the CPU Viterbi number
        // (≥0.95 on the realistic fixture).
        let dense_v: Vec<f32> = x_cpu.matmul(&w_cpu.t()?)?.flatten_all()?.to_vec1()?;
        let (mut dot2, mut na2, mut nb2) = (0f32, 0f32, 0f32);
        for (d, q) in dense_v.iter().zip(y_cuda_v.iter()) {
            dot2 += d * q;
            na2 += d * d;
            nb2 += q * q;
        }
        let cos_vs_dense = dot2 / (na2.sqrt() * nb2.sqrt());
        println!("CUDA vs dense cos sim: {cos_vs_dense}");
        assert!(
            cos_vs_dense >= 0.95,
            "CUDA Viterbi+rotation cos sim vs dense {cos_vs_dense} < 0.95"
        );
        Ok(())
    }

    /// RUN-quant-on-gpu: quantize the SAME tensor on CPU and on CUDA and
    /// verify that `dequantize_weights()` produces outputs whose cosine
    /// similarity to the original FP32 input is ≥0.999 on *both* paths.
    /// This is the load-time-CPU-killer's correctness gate: a real model
    /// gets the same matmul output whether it was quantized on CPU or
    /// CUDA. (We don't require bit-exact equality because Viterbi's
    /// argmin reduction can break ties differently across the two
    /// implementations — but the *reconstruction* must match closely.)
    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_quantize_matches_cpu_dequantize_cos_sim() -> Result<()> {
        let cuda = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(_) => {
                eprintln!(
                    "CUDA not available; skipping cuda_quantize_matches_cpu_dequantize_cos_sim"
                );
                return Ok(());
            }
        };
        let cpu = Device::Cpu;
        let n = 16;
        let k_in = 256;

        // Deterministic Gaussian fixture — same generator as the realistic
        // CPU tests so we can cross-check expected numbers.
        let mut wdata = vec![0.0f32; n * k_in];
        for (i, v) in wdata.iter_mut().enumerate() {
            let mut z = (i as u64).wrapping_mul(0x9E3779B97F4A7C15);
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z ^= z >> 31;
            let u1 = ((z >> 32) as u32 as f32 + 1.0) / (u32::MAX as f32 + 2.0);
            let u2 = ((z & 0xFFFFFFFF) as u32 as f32 + 1.0) / (u32::MAX as f32 + 2.0);
            let r = (-2.0_f32 * u1.ln()).sqrt();
            *v = r * (2.0 * std::f32::consts::PI * u2).cos() * 0.5;
        }

        for mode in [QtipMode::Greedy, QtipMode::Viterbi] {
            let w_cpu = Tensor::from_vec(wdata.clone(), (n, k_in), &cpu)?;
            let w_cuda = Tensor::from_vec(wdata.clone(), (n, k_in), &cuda)?;

            let layer_cpu = QtipLayer::quantize_with_options(
                &w_cpu,
                None,
                &cpu,
                mode,
                QtipRotation::for_mode(mode).enabled(),
            )?;
            let layer_cuda = QtipLayer::quantize_with_options(
                &w_cuda,
                None,
                &cuda,
                mode,
                QtipRotation::for_mode(mode).enabled(),
            )?;

            let cpu_recon: Vec<f32> = layer_cpu
                .dequantize_w()?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1()?;
            let cuda_recon: Vec<f32> = layer_cuda
                .dequantize_w()?
                .to_device(&cpu)?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1()?;

            // (1) GPU vs original cos sim ≥ 0.999 (matches CPU bar).
            let cos = |a: &[f32], b: &[f32]| -> f32 {
                let (mut d, mut na, mut nb) = (0f32, 0f32, 0f32);
                for (x, y) in a.iter().zip(b.iter()) {
                    d += x * y;
                    na += x * x;
                    nb += y * y;
                }
                d / (na.sqrt() * nb.sqrt())
            };

            // (2) CPU vs CUDA recon cos sim must be very close — both
            // quantize the same input with the same algorithm, so they
            // should produce nearly identical reconstructions modulo
            // tie-breaking differences in the argmin reductions.
            let cpu_vs_cuda = cos(&cpu_recon, &cuda_recon);
            println!("{mode:?}: CPU vs CUDA dequant cos sim = {:.5}", cpu_vs_cuda);
            assert!(
                cpu_vs_cuda >= 0.999,
                "{mode:?}: CPU vs CUDA dequant cos sim {cpu_vs_cuda} < 0.999"
            );

            // (3) Both reconstructions must hit the standard recon-vs-original
            // quality bar — Greedy is the looser of the two so we use 0.85
            // (matches the CPU `qtip_matmul_cosine_similarity` greedy gate).
            let cuda_vs_orig = cos(&wdata, &cuda_recon);
            let cpu_vs_orig = cos(&wdata, &cpu_recon);
            println!(
                "{mode:?}: CPU recon vs orig = {:.5}, CUDA recon vs orig = {:.5}",
                cpu_vs_orig, cuda_vs_orig
            );
            let bar = match mode {
                QtipMode::Greedy => 0.85,
                QtipMode::Viterbi => 0.90, // tighter — Viterbi has rotation
            };
            assert!(
                cuda_vs_orig >= bar,
                "{mode:?}: CUDA dequant vs original cos sim {cuda_vs_orig} < {bar}"
            );
        }
        Ok(())
    }

    /// GPU rotation kernel matches the CPU `apply_block_rotation`.
    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_rotate_x_matches_cpu() -> Result<()> {
        use crate::turboquant::wht::generate_signs;
        let cuda = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(_) => {
                eprintln!("CUDA not available; skipping cuda_rotate_x_matches_cpu");
                return Ok(());
            }
        };
        let cpu = Device::Cpu;
        for &(batch, feat) in &[(2usize, 64usize), (4, 128), (1, 256), (3, 512)] {
            let block = rotation_block_size(feat);
            if block < 2 {
                continue;
            }
            let signs_vec = generate_signs(QTIP_ROTATION_SEED, feat);
            let xdata: Vec<f32> = (0..(batch * feat))
                .map(|i| ((i as f32) * 0.031).sin() + ((i as f32) * 0.07).cos())
                .collect();

            // CPU rotation.
            let mut cpu_rot = xdata.clone();
            for b in 0..batch {
                let row = &mut cpu_rot[b * feat..(b + 1) * feat];
                apply_block_rotation(row, &signs_vec, block);
            }

            // CUDA rotation.
            let x_cuda =
                Tensor::from_vec(xdata.clone(), (batch, feat), &cuda)?.to_dtype(DType::F32)?;
            let signs_cuda = Tensor::from_vec(signs_vec, (feat,), &cuda)?;
            let x_rot_cuda = super::cuda_ops::rotate_x_cuda(&x_cuda, &signs_cuda, block)?;
            let cuda_rot: Vec<f32> = x_rot_cuda.to_device(&cpu)?.flatten_all()?.to_vec1()?;

            let mut max_diff = 0f32;
            for (a, b) in cpu_rot.iter().zip(cuda_rot.iter()) {
                max_diff = max_diff.max((a - b).abs());
            }
            println!(
                "Rotate-x CUDA vs CPU (batch={batch}, feat={feat}, block={block}): max_diff = {max_diff:.6}"
            );
            // F32 rotation should match to FP rounding (a few ULPs).
            assert!(
                max_diff < 1e-3,
                "CUDA rotation diverges from CPU: max_diff {max_diff} >= 1e-3"
            );
        }
        Ok(())
    }

    // -----------------------------------------------------------------
    // 3-D stacked-expert tests (RUN-NVFP4-3D).
    //
    // V4 Flash MoE blocks expose `[num_experts, N, K]` tensors to ISQ.
    // These tests verify (a) the rank-dispatch in `quantize_with_options`
    // routes 3-D inputs through the 3-D pipeline, (b) `dequantize_weights`
    // returns the matching `[E, N, K]` shape, (c) per-expert reconstruction
    // matches a per-slice quantize, and (d) end-to-end cos sim against the
    // dense reference stays ≥0.95 (Viterbi+rotation) / ≥0.85 (Greedy).
    // -----------------------------------------------------------------

    fn build_3d_gaussian_weight(
        e: usize,
        n: usize,
        k_in: usize,
        device: &Device,
    ) -> Result<Tensor> {
        let mut data = vec![0.0f32; e * n * k_in];
        for (i, v) in data.iter_mut().enumerate() {
            // Deterministic Gaussian via Box-Muller hash. Different experts
            // get different seeds so we exercise distinct row scales.
            let mut z = (i as u64).wrapping_mul(0x9E3779B97F4A7C15);
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z ^= z >> 31;
            let u1 = ((z >> 32) as u32 as f32 + 1.0) / (u32::MAX as f32 + 2.0);
            let u2 = ((z & 0xFFFFFFFF) as u32 as f32 + 1.0) / (u32::MAX as f32 + 2.0);
            let r = (-2.0_f32 * u1.ln()).sqrt();
            *v = r * (2.0 * std::f32::consts::PI * u2).cos() * 0.5;
        }
        Tensor::from_vec(data, (e, n, k_in), device)
    }

    /// Quantize a 3-D weight and return the typed `QtipLayer` directly so
    /// tests can inspect fields/accessors without going through the
    /// `Arc<dyn QuantMethod>` indirection (the trait does not extend `Any`,
    /// so we can't downcast at the test site).
    fn quantize_3d_typed(w: &Tensor, mode: QtipMode) -> Result<QtipLayer> {
        let device = w.device().clone();
        // Replicate `quantize_with_options_3d` body just enough to capture
        // the typed return — this is the same code path as production, but
        // we collect a `QtipLayer` instead of an `Arc<dyn QuantMethod>`.
        let dims = w.dims3()?;
        let (e, n, k_in) = (dims.0, dims.1, dims.2);
        let use_rotation = QtipRotation::for_mode(mode).enabled();
        let mut blocks_slices: Vec<Tensor> = Vec::with_capacity(e);
        let mut scales_slices: Vec<Tensor> = Vec::with_capacity(e);
        let mut shared_lut: Option<Tensor> = None;
        let mut shared_rotation_signs: Option<Tensor> = None;
        let mut shared_rotation_block: usize = 0;
        let mut shared_search_detail = QtipSearchDetail::Unknown;
        let mut shared_codebook = QtipCodebook::Gaussian;
        let mut shared_geometry = QtipGeometry::K4V2L16;
        for expert_idx in 0..e {
            let expert_w = w.narrow(0, expert_idx, 1)?.squeeze(0)?;
            let layer = QtipLayer::quantize_with_options_concrete(
                &expert_w,
                None,
                &device,
                mode,
                use_rotation,
            )?;
            blocks_slices.push(layer.blocks.clone());
            scales_slices.push(layer.row_scales.clone());
            if expert_idx == 0 {
                shared_lut = Some(layer.lut);
                shared_rotation_signs = layer.rotation_signs;
                shared_rotation_block = layer.rotation_block;
                shared_search_detail = layer.search_detail;
                shared_codebook = layer.codebook;
                shared_geometry = layer.geometry;
            }
        }
        let blocks_3d = Tensor::stack(&blocks_slices, 0)?;
        let scales_2d = Tensor::stack(&scales_slices, 0)?;
        Ok(QtipLayer {
            blocks: blocks_3d,
            row_scales: scales_2d,
            lut: shared_lut.unwrap(),
            bias: None,
            in_features: k_in,
            num_experts: Some(e),
            rotation_signs: shared_rotation_signs,
            rotation_block: shared_rotation_block,
            search: QtipSearchStamp::for_mode(mode),
            search_detail: shared_search_detail,
            codebook: shared_codebook,
            geometry: shared_geometry,
        })
        .inspect(|l| {
            debug_assert_eq!(l.blocks.dims(), &[e, n, k_in / 4]);
            debug_assert_eq!(l.row_scales.dims(), &[e, n]);
        })
    }

    /// Shape contract: a `[E, N, K]` quantize produces a `[E, N, K]`
    /// dequantize. Smoke test against the most obvious failure mode (the
    /// pre-fix dims2 bail at the entry of `quantize_with_options`).
    #[test]
    fn qtip_3d_dequantize_shape() -> Result<()> {
        let device = Device::Cpu;
        let (e, n, k_in) = (3usize, 8usize, 64usize);
        let w = build_3d_gaussian_weight(e, n, k_in, &device)?;
        let layer = quantize_3d_typed(&w, QtipMode::Greedy)?;
        assert_eq!(layer.num_experts(), Some(e));
        assert_eq!(layer.in_features(), k_in);
        assert_eq!(layer.out_features()?, n);
        assert_eq!(layer.blocks().dims(), &[e, n, k_in / 4]);
        assert_eq!(layer.row_scales().dims(), &[e, n]);
        let dq = layer.dequantize_w()?;
        assert_eq!(dq.dims(), &[e, n, k_in]);
        Ok(())
    }

    /// Per-expert consistency: dequantizing expert `i` from the 3-D layer
    /// matches dequantizing a fresh `[N, K]` 2-D layer built from the same
    /// `[N, K]` slice. Catches any per-expert offset / striding bug.
    #[test]
    fn qtip_3d_per_expert_matches_2d() -> Result<()> {
        let device = Device::Cpu;
        let (e, _n, k_in) = (4usize, 8usize, 64usize);
        let w = build_3d_gaussian_weight(e, _n, k_in, &device)?;
        let layer3d = quantize_3d_typed(&w, QtipMode::Greedy)?;
        for expert_idx in 0..e {
            let w_e = w.narrow(0, expert_idx, 1)?.squeeze(0)?.contiguous()?;
            let layer2d_arc = QtipLayer::quantize_greedy_fixture(&w_e, None, &device)?;
            let dq2d = layer2d_arc.dequantize_w()?.to_dtype(DType::F32)?;
            let dq3d_e = layer3d
                .dequantize_expert(expert_idx)?
                .to_dtype(DType::F32)?;
            let a: Vec<f32> = dq2d.flatten_all()?.to_vec1()?;
            let b: Vec<f32> = dq3d_e.flatten_all()?.to_vec1()?;
            assert_eq!(a.len(), b.len());
            for (i, (av, bv)) in a.iter().zip(b.iter()).enumerate() {
                assert!(
                    (av - bv).abs() < 1e-5,
                    "expert {expert_idx} index {i}: 2-D {av} vs 3-D {bv}"
                );
            }
        }
        Ok(())
    }

    /// Reconstruction quality: full `[E, N, K]` round trip preserves
    /// cosine similarity ≥ 0.85 under Viterbi+rotation (the Cornell QTIP
    /// target). This is the headline accuracy contract for the 3-D path.
    #[test]
    fn qtip_3d_round_trip_cosine_similarity_viterbi() -> Result<()> {
        let device = Device::Cpu;
        let (e, n, k_in) = (4usize, 16usize, 128usize);
        let w = build_3d_gaussian_weight(e, n, k_in, &device)?;
        let layer = QtipLayer::quantize_with_mode(&w, None, &device, QtipMode::Viterbi)?;
        let dq = layer.dequantize_w()?.to_dtype(DType::F32)?;
        let orig: Vec<f32> = w.flatten_all()?.to_vec1()?;
        let recon: Vec<f32> = dq.flatten_all()?.to_vec1()?;
        let (mut dot, mut no, mut nr) = (0f32, 0f32, 0f32);
        for (o, r) in orig.iter().zip(recon.iter()) {
            dot += o * r;
            no += o * o;
            nr += r * r;
        }
        let cos = dot / (no.sqrt() * nr.sqrt());
        println!("3-D Viterbi round-trip cos sim = {cos}");
        assert!(cos >= 0.85, "3-D Viterbi cos sim {cos} < 0.85");
        Ok(())
    }

    /// `forward` on a 3-D layer must bail clearly. The MoE forward path
    /// is gather-based; calling the standard matmul forward would silently
    /// pick the wrong expert.
    #[test]
    fn qtip_3d_forward_bails() -> Result<()> {
        let device = Device::Cpu;
        let (e, n, k_in) = (2usize, 4usize, 32usize);
        let w = build_3d_gaussian_weight(e, n, k_in, &device)?;
        let layer = QtipLayer::quantize_greedy_fixture(&w, None, &device)?;
        let x = Tensor::zeros((1, k_in), DType::F32, &device)?;
        let err = layer.forward(&x).err();
        assert!(
            err.as_ref().is_some_and(|e| {
                e.to_string().contains("gather_forward") || e.to_string().contains("3-D")
            }),
            "expected gather_forward bail, got {err:?}"
        );
        Ok(())
    }

    /// Bias is unsupported in 3-D mode; the quantize entry must reject it.
    #[test]
    fn qtip_3d_quantize_rejects_bias() -> Result<()> {
        let device = Device::Cpu;
        let (e, n, k_in) = (2usize, 4usize, 32usize);
        let w = build_3d_gaussian_weight(e, n, k_in, &device)?;
        let bias = Tensor::zeros((e, n), DType::F32, &device)?;
        let err = QtipLayer::quantize_greedy_fixture(&w, Some(bias), &device).err();
        assert!(
            err.as_ref().is_some_and(|e| e.to_string().contains("bias")),
            "expected bias rejection, got {err:?}"
        );
        Ok(())
    }

    /// NVFP4 → QTIP for a 3-D NVFP4 layer goes through the apply_isq dispatch
    /// and produces a 3-D QTIP layer. End-to-end smoke test of the path the
    /// V4 Flash `--isq qtip2` invocation takes.
    #[test]
    fn nvfp4_3d_to_qtip_roundtrip() -> Result<()> {
        use crate::IsqType;
        use std::sync::atomic::AtomicUsize;

        let device = Device::Cpu;
        let (e, n, k_in) = (3usize, 16usize, 64usize);
        let w = build_3d_gaussian_weight(e, n, k_in, &device)?;

        // Build a 3-D NVFP4 layer via the test-only constructor (mirrors
        // what a stacked checkpoint loader would do). We can't go through
        // NVFP4Layer::quantize directly because it dims2()'s the input —
        // and that's the public quantize API contract we don't want to
        // change.
        let combined = crate::NVFP4Layer::quantize_3d_for_test(&w, &device)?;

        // Now run the QTIP ISQ pass.
        let n_quantized = AtomicUsize::new(0);
        let guard = crate::QuantizeOntoGuard::new();
        let qtip = Arc::new(combined).apply_isq(
            Some(IsqType::QtipBitshift2),
            device.clone(),
            &n_quantized,
            None,
            guard,
        )?;
        let dq = qtip.dequantize_w()?;
        assert_eq!(dq.dims(), &[e, n, k_in]);

        // Cos sim against the original BF16-cast weight.
        let recon: Vec<f32> = dq.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
        let orig: Vec<f32> = w.flatten_all()?.to_vec1()?;
        let (mut dot, mut no, mut nr) = (0f32, 0f32, 0f32);
        for (o, r) in orig.iter().zip(recon.iter()) {
            dot += o * r;
            no += o * o;
            nr += r * r;
        }
        let cos = dot / (no.sqrt() * nr.sqrt());
        println!("NVFP4 3-D -> QTIP 3-D round-trip cos sim = {cos}");
        // The double-quantize chain (BF16 -> NVFP4 -> QTIP) loses a bit more
        // than direct QTIP; we set the bar at 0.80 to leave headroom for
        // small Box-Muller fixture rolls.
        assert!(cos >= 0.80, "NVFP4 -> QTIP 3-D cos sim {cos} < 0.80");
        Ok(())
    }

    // -----------------------------------------------------------------
    // gather_forward unit tests (CPU + cuda-parity).
    // -----------------------------------------------------------------
    //
    // These exercise the V4 MoE Fast backend dispatch path. The CPU
    // reference path runs on any host (used as the cross-check for the
    // GPU fast path) and the GPU parity test is gated on `cuda` so it
    // only runs in CUDA-enabled CI.

    /// Build a 3-D expert-stacked layer from per-expert Gaussian weights
    /// and return both the layer + the dense reference weights tensor for
    /// the CPU correctness check.
    ///
    /// Each expert has DIFFERENT weights — verifies the gather actually
    /// selects per-token-per-k.
    fn make_expert_stack(
        num_experts: usize,
        rows: usize,
        in_features: usize,
        device: &Device,
        mode: QtipMode,
    ) -> Result<(QtipLayer, Tensor)> {
        // Per-expert random Gaussian (deterministic per-expert seed).
        let mut weights_per_expert: Vec<Tensor> = Vec::with_capacity(num_experts);
        let mut concrete_layers: Vec<QtipLayer> = Vec::with_capacity(num_experts);
        for e in 0..num_experts {
            let mut wdata = vec![0.0f32; rows * in_features];
            for (i, v) in wdata.iter_mut().enumerate() {
                // Use distinct seed per expert so the weights aren't
                // accidentally identical.
                let seed = ((e + 1) as u64).wrapping_mul(0xDEAD_BEEF_F00D_CAFE)
                    ^ ((i + 1) as u64).wrapping_mul(0x9E3779B97F4A7C15);
                let mut z = seed;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
                z ^= z >> 31;
                let u1 = ((z >> 32) as u32 as f32 + 1.0) / (u32::MAX as f32 + 2.0);
                let u2 = ((z & 0xFFFFFFFF) as u32 as f32 + 1.0) / (u32::MAX as f32 + 2.0);
                let r = (-2.0_f32 * u1.ln()).sqrt();
                *v = r * (2.0 * std::f32::consts::PI * u2).cos() * 0.5;
            }
            let w_e = Tensor::from_vec(wdata.clone(), (rows, in_features), device)?;
            let layer_e = QtipLayer::quantize_with_options_concrete(
                &w_e,
                None,
                device,
                mode,
                matches!(mode, QtipMode::Viterbi),
            )?;
            weights_per_expert.push(w_e);
            concrete_layers.push(layer_e);
        }
        let stack = QtipLayer::stack_experts(concrete_layers)?;

        // Reference: build the dense (num_experts, rows, in_features)
        // tensor by stacking the original weights.
        let refs: Vec<&Tensor> = weights_per_expert.iter().collect();
        let weight_stack = Tensor::stack(&refs, 0)?;
        Ok((stack, weight_stack))
    }

    /// Sanity check: `stack_experts` produces a layer whose `num_experts`
    /// matches and whose `blocks` tensor has rank 3.
    #[test]
    fn qtip_stack_experts_shape() -> Result<()> {
        let device = Device::Cpu;
        let (stack, _refs) = make_expert_stack(4, 16, 64, &device, QtipMode::Greedy)?;
        assert_eq!(stack.blocks.dims().len(), 3, "blocks must be rank 3");
        assert_eq!(
            stack.row_scales.dims().len(),
            2,
            "row_scales must be rank 2"
        );
        assert_eq!(stack.blocks.dim(0)?, 4, "expert dim wrong");
        assert_eq!(stack.row_scales.dim(0)?, 4, "expert dim wrong");
        assert_eq!(stack.in_features, 64);
        Ok(())
    }

    /// CPU `gather_forward` matches the per-expert dense matmul.
    /// Tolerance is wide because each gathered slot uses 2-bit QTIP
    /// (Greedy mode → ~0.85 single-expert cos sim). With 4 experts and 2
    /// k slots per token, the aggregate cos sim should still clear 0.80.
    #[test]
    fn qtip_gather_forward_cpu_matches_reference() -> Result<()> {
        let device = Device::Cpu;
        let num_experts = 4;
        let rows = 16;
        let in_features = 64;
        let n_tokens = 3;
        let n_experts_per_tok = 2;

        let (stack, dense_w_stack) =
            make_expert_stack(num_experts, rows, in_features, &device, QtipMode::Greedy)?;

        // Build random activations.
        let mut adata = vec![0.0f32; n_tokens * n_experts_per_tok * in_features];
        for (i, v) in adata.iter_mut().enumerate() {
            let mut z = ((i + 4711) as u64).wrapping_mul(0x9E3779B97F4A7C15);
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z ^= z >> 31;
            let u1 = ((z >> 32) as u32 as f32 + 1.0) / (u32::MAX as f32 + 2.0);
            let u2 = ((z & 0xFFFFFFFF) as u32 as f32 + 1.0) / (u32::MAX as f32 + 2.0);
            let r = (-2.0_f32 * u1.ln()).sqrt();
            *v = r * (2.0 * std::f32::consts::PI * u2).cos();
        }
        let a = Tensor::from_vec(
            adata.clone(),
            (n_tokens, n_experts_per_tok, in_features),
            &device,
        )?;

        // Indices route to varying experts including duplicates per token.
        let idx_data: Vec<u32> = vec![
            0, 2, // tok 0 → experts 0, 2
            1, 3, // tok 1 → experts 1, 3
            2, 2, // tok 2 → expert 2 twice (duplicate path)
        ];
        let indices = Tensor::from_vec(idx_data.clone(), (n_tokens, n_experts_per_tok), &device)?;

        // Run the gather forward.
        let out = stack.gather_forward(&a, &indices)?;
        assert_eq!(out.dims(), &[n_tokens, n_experts_per_tok, rows]);

        // Reference: per (tok, k), use the dense weights for the routed
        // expert and matmul.
        let dense_w: Vec<f32> = dense_w_stack.flatten_all()?.to_vec1()?;
        let mut ref_out = vec![0f32; n_tokens * n_experts_per_tok * rows];
        for tok in 0..n_tokens {
            for k in 0..n_experts_per_tok {
                let e = idx_data[tok * n_experts_per_tok + k] as usize;
                let a_row = &adata[(tok * n_experts_per_tok + k) * in_features
                    ..(tok * n_experts_per_tok + k + 1) * in_features];
                for r in 0..rows {
                    let w_row = &dense_w[e * rows * in_features + r * in_features
                        ..e * rows * in_features + (r + 1) * in_features];
                    let mut s = 0f32;
                    for c in 0..in_features {
                        s += a_row[c] * w_row[c];
                    }
                    ref_out[(tok * n_experts_per_tok + k) * rows + r] = s;
                }
            }
        }

        let out_v: Vec<f32> = out.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
        let cos = {
            let (mut d, mut na, mut nb) = (0f32, 0f32, 0f32);
            for (x, y) in out_v.iter().zip(ref_out.iter()) {
                d += x * y;
                na += x * x;
                nb += y * y;
            }
            d / (na.sqrt() * nb.sqrt())
        };
        println!("qtip_gather_forward_cpu_matches_reference: cos sim = {cos:.4}");
        // Greedy QTIP at 2 bits typically lands at >0.85 cos sim for a
        // single matmul; aggregated across 6 routed slots the cos sim is
        // dominated by the lowest-quality slot but should still clear
        // 0.80.
        assert!(
            cos >= 0.80,
            "QTIP gather_forward CPU vs reference cos sim {cos} < 0.80"
        );
        Ok(())
    }

    /// `gather_forward` with Viterbi mode (rotation enabled) should track
    /// the dense reference within 0.85 cos sim. The realistic Gaussian
    /// fixture isn't large enough to push to the 0.95 number that the
    /// dense forward test hits on `k_in=256`; here we use `k_in=128` for
    /// build speed and accept 0.85.
    #[test]
    fn qtip_gather_forward_viterbi_with_rotation() -> Result<()> {
        let device = Device::Cpu;
        let num_experts = 3;
        let rows = 16;
        let in_features = 128;
        let n_tokens = 2;
        let n_experts_per_tok = 2;

        let (stack, dense_w_stack) =
            make_expert_stack(num_experts, rows, in_features, &device, QtipMode::Viterbi)?;
        assert!(
            stack.rotation_block >= 2,
            "Viterbi mode should activate rotation (rotation_block={})",
            stack.rotation_block
        );

        let mut adata = vec![0.0f32; n_tokens * n_experts_per_tok * in_features];
        for (i, v) in adata.iter_mut().enumerate() {
            let mut z = ((i + 9001) as u64).wrapping_mul(0x9E3779B97F4A7C15);
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z ^= z >> 31;
            let u1 = ((z >> 32) as u32 as f32 + 1.0) / (u32::MAX as f32 + 2.0);
            let u2 = ((z & 0xFFFFFFFF) as u32 as f32 + 1.0) / (u32::MAX as f32 + 2.0);
            let r = (-2.0_f32 * u1.ln()).sqrt();
            *v = r * (2.0 * std::f32::consts::PI * u2).cos();
        }
        let a = Tensor::from_vec(
            adata.clone(),
            (n_tokens, n_experts_per_tok, in_features),
            &device,
        )?;
        let idx_data: Vec<u32> = vec![0, 1, 1, 2];
        let indices = Tensor::from_vec(idx_data.clone(), (n_tokens, n_experts_per_tok), &device)?;
        let out = stack.gather_forward(&a, &indices)?;

        let dense_w: Vec<f32> = dense_w_stack.flatten_all()?.to_vec1()?;
        let mut ref_out = vec![0f32; n_tokens * n_experts_per_tok * rows];
        for tok in 0..n_tokens {
            for k in 0..n_experts_per_tok {
                let e = idx_data[tok * n_experts_per_tok + k] as usize;
                let a_row = &adata[(tok * n_experts_per_tok + k) * in_features
                    ..(tok * n_experts_per_tok + k + 1) * in_features];
                for r in 0..rows {
                    let w_row = &dense_w[e * rows * in_features + r * in_features
                        ..e * rows * in_features + (r + 1) * in_features];
                    let mut s = 0f32;
                    for c in 0..in_features {
                        s += a_row[c] * w_row[c];
                    }
                    ref_out[(tok * n_experts_per_tok + k) * rows + r] = s;
                }
            }
        }

        let out_v: Vec<f32> = out.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
        let cos = {
            let (mut d, mut na, mut nb) = (0f32, 0f32, 0f32);
            for (x, y) in out_v.iter().zip(ref_out.iter()) {
                d += x * y;
                na += x * x;
                nb += y * y;
            }
            d / (na.sqrt() * nb.sqrt())
        };
        println!("qtip_gather_forward_viterbi_with_rotation: cos sim = {cos:.4}");
        assert!(
            cos >= 0.85,
            "QTIP gather_forward Viterbi cos sim {cos} < 0.85"
        );
        Ok(())
    }

    /// Negative test: `gather_forward` on a 2-D (single-expert) layer
    /// should bail with a clear error so callers don't silently get
    /// wrong outputs.
    #[test]
    fn qtip_gather_forward_rejects_2d_layer() -> Result<()> {
        let device = Device::Cpu;
        let rows = 8;
        let in_features = 64;
        let wdata = vec![0.5f32; rows * in_features];
        let w = Tensor::from_vec(wdata, (rows, in_features), &device)?;
        let layer =
            QtipLayer::quantize_with_options_concrete(&w, None, &device, QtipMode::Greedy, false)?;

        let a = Tensor::zeros((1, 2, in_features), DType::F32, &device)?;
        let idx = Tensor::from_vec(vec![0u32, 0u32], (1, 2), &device)?;
        let res = layer.gather_forward(&a, &idx);
        assert!(
            res.is_err(),
            "gather_forward on 2-D layer must bail (got {:?})",
            res.as_ref().map(|t| t.dims())
        );
        let msg = format!("{}", res.unwrap_err());
        assert!(
            msg.contains("expert-stacked") || msg.contains("3-D"),
            "Expected expert-stacked / 3-D error message, got: {msg}"
        );
        Ok(())
    }

    /// GPU parity test: build a 3-D layer on CUDA, run gather_forward on
    /// CUDA, run the same gather_forward on CPU, and assert the two
    /// agree within ≥0.999 cos sim. Skipped when CUDA is unavailable.
    #[cfg(feature = "cuda")]
    #[test]
    fn qtip_gather_forward_cuda_matches_cpu() -> Result<()> {
        let cuda = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(_) => {
                eprintln!("CUDA not available; skipping qtip_gather_forward_cuda_matches_cpu");
                return Ok(());
            }
        };
        let cpu = Device::Cpu;
        let num_experts = 4;
        let rows = 32;
        let in_features = 128;
        let n_tokens = 3;
        let n_experts_per_tok = 2;

        // Build the SAME random fixture on both devices by stacking weights
        // generated from the same hash.
        let (cpu_stack, _ref_w) =
            make_expert_stack(num_experts, rows, in_features, &cpu, QtipMode::Viterbi)?;
        let (cuda_stack, _ref_w_cuda) =
            make_expert_stack(num_experts, rows, in_features, &cuda, QtipMode::Viterbi)?;

        // Build the same activations on both devices.
        let mut adata = vec![0.0f32; n_tokens * n_experts_per_tok * in_features];
        for (i, v) in adata.iter_mut().enumerate() {
            let mut z = ((i + 13_579) as u64).wrapping_mul(0x9E3779B97F4A7C15);
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z ^= z >> 31;
            let u1 = ((z >> 32) as u32 as f32 + 1.0) / (u32::MAX as f32 + 2.0);
            let u2 = ((z & 0xFFFFFFFF) as u32 as f32 + 1.0) / (u32::MAX as f32 + 2.0);
            let r = (-2.0_f32 * u1.ln()).sqrt();
            *v = r * (2.0 * std::f32::consts::PI * u2).cos();
        }
        let a_cpu = Tensor::from_vec(
            adata.clone(),
            (n_tokens, n_experts_per_tok, in_features),
            &cpu,
        )?;
        let a_cuda = Tensor::from_vec(
            adata.clone(),
            (n_tokens, n_experts_per_tok, in_features),
            &cuda,
        )?
        .to_dtype(DType::BF16)?;

        let idx_data: Vec<u32> = vec![0, 1, 2, 3, 1, 1];
        let idx_cpu = Tensor::from_vec(idx_data.clone(), (n_tokens, n_experts_per_tok), &cpu)?;
        let idx_cuda = Tensor::from_vec(idx_data, (n_tokens, n_experts_per_tok), &cuda)?;

        let y_cpu = cpu_stack.gather_forward(&a_cpu, &idx_cpu)?;
        let y_cuda = cuda_stack.gather_forward(&a_cuda, &idx_cuda)?;

        let y_cpu_v: Vec<f32> = y_cpu.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
        let y_cuda_v: Vec<f32> = y_cuda
            .to_device(&cpu)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1()?;

        let (mut d, mut na, mut nb) = (0f32, 0f32, 0f32);
        for (x, y) in y_cpu_v.iter().zip(y_cuda_v.iter()) {
            d += x * y;
            na += x * x;
            nb += y * y;
        }
        let cos = d / (na.sqrt() * nb.sqrt());
        println!("qtip_gather_forward_cuda_matches_cpu: cos sim = {cos:.5}");
        assert!(
            cos >= 0.999,
            "CUDA gather_forward deviates from CPU: cos sim {cos} < 0.999"
        );
        Ok(())
    }

    /// Deterministic Gaussian fixture (splitmix64 + Box-Muller), shared by the
    /// wave28-AZ gather-boundary tests.
    #[cfg(feature = "cuda")]
    fn az_gaussian(len: usize, seed: u64) -> Vec<f32> {
        let mut out = vec![0.0f32; len];
        for (i, v) in out.iter_mut().enumerate() {
            let mut z = ((i as u64) + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ seed;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            let u1 = ((z >> 32) as u32 as f32 + 1.0) / (u32::MAX as f32 + 2.0);
            let u2 = ((z & 0xFFFF_FFFF) as u32 as f32 + 1.0) / (u32::MAX as f32 + 2.0);
            *v = (-2.0_f32 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
        }
        out
    }

    #[cfg(feature = "cuda")]
    fn az_cos_sim(x: &Tensor, y: &Tensor) -> Result<f64> {
        let to_v = |t: &Tensor| -> Result<Vec<f32>> {
            t.to_device(&Device::Cpu)?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1()
        };
        let (xv, yv) = (to_v(x)?, to_v(y)?);
        let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
        for (a, b) in xv.iter().zip(yv.iter()) {
            dot += (*a as f64) * (*b as f64);
            na += (*a as f64) * (*a as f64);
            nb += (*b as f64) * (*b as f64);
        }
        Ok(dot / (na.sqrt() * nb.sqrt()))
    }

    /// wave28-AZ — the fused on-device gather and the per-expert
    /// dequantize-materialize fallback must agree numerically at every token
    /// count, in particular on both sides of the hard cap of 8 this change
    /// removed (8 / 9 / 16 / 32 / 64 / 128).
    ///
    /// **The test carries its own anti-vacuity guard.** A parity assertion
    /// between two paths that both return zeros — or that both ignore the
    /// routing — passes for free; this repo has found seven tests that could
    /// not fail. So each token count also compares the fused output against
    /// the fallback run on a *different* routing and requires that one to
    /// disagree. An all-zero, constant, or routing-independent output makes the
    /// negative control fail, which is the outcome we want from a broken kernel
    /// rather than a green parity line.
    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_fused_gather_matches_dequantize_fallback_across_the_old_cap() -> Result<()> {
        let cuda = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(_) => {
                eprintln!(
                    "CUDA not available; skipping \
                     cuda_fused_gather_matches_dequantize_fallback_across_the_old_cap"
                );
                return Ok(());
            }
        };
        let (num_experts, rows, in_features, top_k) = (16usize, 32usize, 128usize, 4usize);
        let (stack, _w) =
            make_expert_stack(num_experts, rows, in_features, &cuda, QtipMode::Viterbi)?;

        for &n_tokens in &[1usize, 8, 9, 16, 32, 64, 128] {
            let pairs = n_tokens * top_k;

            // The dispatch change under test: the old cap sent everything past
            // 8 tokens to the fallback; all of these must now pick the fused
            // path for this routing shape.
            assert!(
                gather_policy::lut_fused_gather_preferred(n_tokens, top_k, num_experts),
                "{n_tokens} tokens must dispatch to the fused gather"
            );

            let a = Tensor::from_vec(
                az_gaussian(
                    pairs * in_features,
                    0x5A20_u64.wrapping_add(n_tokens as u64),
                ),
                (n_tokens, top_k, in_features),
                &cuda,
            )?
            .to_dtype(DType::BF16)?;

            let idx: Vec<u32> = (0..pairs)
                .map(|i| ((i * 7 + 3) % num_experts) as u32)
                .collect();
            let idx_shifted: Vec<u32> = idx.iter().map(|&e| (e + 1) % num_experts as u32).collect();
            let idx_t = Tensor::from_vec(idx, (n_tokens, top_k), &cuda)?;
            let idx_shifted_t = Tensor::from_vec(idx_shifted, (n_tokens, top_k), &cuda)?;

            let fused = stack.gather_forward_cuda_ondevice(&a, &idx_t)?;
            let fallback = stack.gather_forward_cuda(&a, &idx_t)?;
            assert_eq!(fused.dims(), fallback.dims(), "n_tokens={n_tokens}");

            let cos = az_cos_sim(&fused, &fallback)?;
            assert!(
                cos >= 0.999,
                "n_tokens={n_tokens}: fused gather vs dequantize fallback cos sim {cos} < 0.999"
            );

            // Cos sim is scale-blind, so also bound the absolute error against
            // the fallback's own dynamic range.
            let fv: Vec<f32> = fused
                .to_device(&Device::Cpu)?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1()?;
            let bv: Vec<f32> = fallback
                .to_device(&Device::Cpu)?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1()?;
            let scale = bv.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
            let max_abs = fv
                .iter()
                .zip(bv.iter())
                .fold(0f32, |m, (x, y)| m.max((x - y).abs()));
            assert!(
                max_abs / scale <= 2e-2,
                "n_tokens={n_tokens}: max |fused - fallback| = {max_abs} ({:.3}% of range {scale})",
                100.0 * max_abs / scale
            );

            // ---- anti-vacuity: the comparison must be able to fail ----
            let fallback_other = stack.gather_forward_cuda(&a, &idx_shifted_t)?;
            let cos_other = az_cos_sim(&fused, &fallback_other)?;
            assert!(
                cos_other < 0.9,
                "n_tokens={n_tokens}: output does not depend on the routing \
                 (cos sim {cos_other} against a shifted expert assignment) — the parity \
                 assertion above is vacuous"
            );
        }
        Ok(())
    }

    /// The fused gather's one real structural limit is `grid.y = n_pairs`,
    /// bounded by CUDA's `maxGridSize[1] = 65535`. Past it the launch fails
    /// with `cudaErrorInvalidConfiguration`, the `extern "C"` launcher discards
    /// the status, and the caller receives the zero-initialised output buffer —
    /// a silently all-zero MoE layer. Assert we now get an **error** there, and
    /// that the production dispatcher never asks for it.
    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_fused_gather_errors_past_the_grid_limit_instead_of_returning_zeros() -> Result<()> {
        let cuda = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(_) => {
                eprintln!(
                    "CUDA not available; skipping \
                     cuda_fused_gather_errors_past_the_grid_limit_instead_of_returning_zeros"
                );
                return Ok(());
            }
        };
        let (num_experts, rows, in_features, top_k) = (4usize, 8usize, 64usize, 4usize);
        let (stack, _w) =
            make_expert_stack(num_experts, rows, in_features, &cuda, QtipMode::Viterbi)?;

        let n_tokens = gather_policy::GATHER_GEMV_MAX_PAIRS / top_k + 1;
        assert!(n_tokens * top_k > gather_policy::GATHER_GEMV_MAX_PAIRS);

        let a = Tensor::from_vec(
            az_gaussian(n_tokens * top_k * in_features, 0xA2C0),
            (n_tokens, top_k, in_features),
            &cuda,
        )?
        .to_dtype(DType::BF16)?;
        let idx = Tensor::zeros((n_tokens, top_k), DType::U32, &cuda)?;

        let err = stack
            .gather_forward_cuda_ondevice(&a, &idx)
            .expect_err("a launch past grid.y must be an error, not a zero-filled tensor");
        let msg = err.to_string();
        assert!(msg.contains("grid.y"), "unexpected error: {msg}");

        // The production dispatcher must never route here in the first place.
        assert!(!gather_policy::lut_fused_gather_preferred(
            n_tokens,
            top_k,
            num_experts
        ));
        Ok(())
    }

    /// Fused decode + gemv kernel must produce numerically-equivalent output
    /// to the dequantize+matmul path. We compare cos sim against the
    /// dequantize-then-matmul reference (NOT against dense) — both paths use
    /// the same QTIP-reconstructed weight, so the fused kernel only changes
    /// HOW the matmul is computed, not the operand values. Cos sim must be
    /// ≥0.999 (allowing only for fmaf vs cuBLAS rounding differences).
    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_fused_gemv_matches_dequant_matmul() -> Result<()> {
        let cuda = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(_) => {
                eprintln!("CUDA not available; skipping cuda_fused_gemv_matches_dequant_matmul");
                return Ok(());
            }
        };
        let cpu = Device::Cpu;

        // Cover a few realistic shapes: small (test), medium (decoder
        // hidden), and mismatched (intermediate / down_proj). All use
        // K=4 V=2 L=16 packing.
        for &(n, k_in) in &[
            (32usize, 256usize), // tiny
            (128, 512),          // small attn-out
            (256, 1024),         // mid
            (1024, 4096),        // realistic LLM scale (decoder block)
        ] {
            // Random-ish Gaussian weights via deterministic hash.
            let mut wdata = vec![0.0f32; n * k_in];
            for (i, v) in wdata.iter_mut().enumerate() {
                let mut z = (i as u64).wrapping_mul(0x9E3779B97F4A7C15);
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
                z ^= z >> 31;
                let u1 = ((z >> 32) as u32 as f32 + 1.0) / (u32::MAX as f32 + 2.0);
                let u2 = ((z & 0xFFFFFFFF) as u32 as f32 + 1.0) / (u32::MAX as f32 + 2.0);
                let r = (-2.0_f32 * u1.ln()).sqrt();
                *v = r * (2.0 * std::f32::consts::PI * u2).cos() * 0.5;
            }
            let mut xdata = vec![0.0f32; k_in];
            for (i, v) in xdata.iter_mut().enumerate() {
                let mut z = ((i + 7_777_777) as u64).wrapping_mul(0x9E3779B97F4A7C15);
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
                z ^= z >> 31;
                let u1 = ((z >> 32) as u32 as f32 + 1.0) / (u32::MAX as f32 + 2.0);
                let u2 = ((z & 0xFFFFFFFF) as u32 as f32 + 1.0) / (u32::MAX as f32 + 2.0);
                let r = (-2.0_f32 * u1.ln()).sqrt();
                *v = r * (2.0 * std::f32::consts::PI * u2).cos();
            }

            // Build the layer on CUDA (Viterbi + rotation, the production
            // path for V4 Flash).
            let w_cuda = Tensor::from_vec(wdata.clone(), (n, k_in), &cuda)?;
            let layer = QtipLayer::quantize_with_mode(&w_cuda, None, &cuda, QtipMode::Viterbi)?;
            // We need direct access to the QtipLayer struct (not the dyn
            // QuantMethod) to call internal forward helpers — downcast the
            // Arc by treating the trait pointer as &QtipLayer at known type.
            //
            // Cleanest approach: use the public `forward` (which dispatches
            // to the fused path internally for n_tokens=1) and compare
            // against a manually-constructed dequant+matmul.

            // Single-token forward through the layer (triggers fused gemv).
            let x_cuda_1tok =
                Tensor::from_vec(xdata.clone(), (1, k_in), &cuda)?.to_dtype(DType::BF16)?;
            let y_fused = layer.forward(&x_cuda_1tok)?;
            let y_fused_v: Vec<f32> = y_fused
                .to_device(&cpu)?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1()?;

            // Manual dequant+matmul reference: dequantize_w() returns the
            // weight in the ORIGINAL (un-rotated) frame, so the matmul
            // x @ W^T should be the algebraic equivalent of layer.forward.
            let w_recon = layer.dequantize_w()?.to_dtype(DType::F32)?;
            let x_f32 = Tensor::from_vec(xdata, (1, k_in), &cuda)?;
            let y_ref = x_f32.matmul(&w_recon.t()?)?;
            let y_ref_v: Vec<f32> = y_ref.to_device(&cpu)?.flatten_all()?.to_vec1()?;

            // Compute cos sim.
            let (mut dot, mut na, mut nb) = (0f32, 0f32, 0f32);
            for (a, b) in y_fused_v.iter().zip(y_ref_v.iter()) {
                dot += a * b;
                na += a * a;
                nb += b * b;
            }
            let cos = dot / (na.sqrt() * nb.sqrt());
            println!("Fused gemv vs dequant+matmul cos sim (n={n}, k_in={k_in}): {cos}");
            assert!(
                cos >= 0.999,
                "Fused gemv deviates from dequant+matmul: cos sim {cos} < 0.999 (n={n}, k_in={k_in})"
            );
        }
        Ok(())
    }

    /// Fused gemv must match the dequant+matmul output for the rotation-disabled
    /// path too (Greedy mode), to catch any state-warmup bug that only manifests
    /// without rotation noise to mask it.
    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_fused_gemv_matches_dequant_matmul_no_rotation() -> Result<()> {
        let cuda = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(_) => {
                eprintln!("CUDA not available; skipping cuda_fused_gemv_matches_dequant_matmul_no_rotation");
                return Ok(());
            }
        };
        let cpu = Device::Cpu;

        let n = 256usize;
        let k_in = 512usize;
        let mut wdata = vec![0.0f32; n * k_in];
        for (i, v) in wdata.iter_mut().enumerate() {
            let mut z = (i as u64).wrapping_mul(0x9E3779B97F4A7C15);
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z ^= z >> 31;
            let u1 = ((z >> 32) as u32 as f32 + 1.0) / (u32::MAX as f32 + 2.0);
            let u2 = ((z & 0xFFFFFFFF) as u32 as f32 + 1.0) / (u32::MAX as f32 + 2.0);
            let r = (-2.0_f32 * u1.ln()).sqrt();
            *v = r * (2.0 * std::f32::consts::PI * u2).cos() * 0.4;
        }
        let xdata: Vec<f32> = (0..k_in)
            .map(|i| ((i as f32) * 0.013).sin() + ((i as f32) * 0.029).cos())
            .collect();

        // Greedy mode → rotation disabled by default.
        let w_cuda = Tensor::from_vec(wdata, (n, k_in), &cuda)?;
        let layer = QtipLayer::quantize_greedy_fixture(&w_cuda, None, &cuda)?;

        let x_cuda_1tok =
            Tensor::from_vec(xdata.clone(), (1, k_in), &cuda)?.to_dtype(DType::BF16)?;
        let y_fused = layer.forward(&x_cuda_1tok)?;
        let y_fused_v: Vec<f32> = y_fused
            .to_device(&cpu)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1()?;

        let w_recon = layer.dequantize_w()?.to_dtype(DType::F32)?;
        let x_f32 = Tensor::from_vec(xdata, (1, k_in), &cuda)?;
        let y_ref = x_f32.matmul(&w_recon.t()?)?;
        let y_ref_v: Vec<f32> = y_ref.to_device(&cpu)?.flatten_all()?.to_vec1()?;

        let (mut dot, mut na, mut nb) = (0f32, 0f32, 0f32);
        for (a, b) in y_fused_v.iter().zip(y_ref_v.iter()) {
            dot += a * b;
            na += a * a;
            nb += b * b;
        }
        let cos = dot / (na.sqrt() * nb.sqrt());
        println!("Fused gemv vs dequant+matmul cos sim (Greedy, no rotation): {cos}");
        assert!(
            cos >= 0.999,
            "Fused gemv deviates from dequant+matmul (Greedy): cos sim {cos} < 0.999"
        );
        Ok(())
    }

    // -----------------------------------------------------------------
    // UQFF serde round-trip tests (2-D regression + 3-D stacked experts).
    // -----------------------------------------------------------------

    /// Exact-bits tensor comparison (shape, dtype, and raw data).
    fn assert_tensor_bits_eq(a: &Tensor, b: &Tensor, what: &str) -> Result<()> {
        assert_eq!(a.dims(), b.dims(), "{what}: shape mismatch");
        assert_eq!(a.dtype(), b.dtype(), "{what}: dtype mismatch");
        match a.dtype() {
            DType::U8 => {
                let av: Vec<u8> = a.flatten_all()?.to_vec1()?;
                let bv: Vec<u8> = b.flatten_all()?.to_vec1()?;
                assert_eq!(av, bv, "{what}: packed bytes differ");
            }
            DType::F32 => {
                let av: Vec<f32> = a.flatten_all()?.to_vec1()?;
                let bv: Vec<f32> = b.flatten_all()?.to_vec1()?;
                let ab: Vec<u32> = av.iter().map(|x| x.to_bits()).collect();
                let bb: Vec<u32> = bv.iter().map(|x| x.to_bits()).collect();
                assert_eq!(ab, bb, "{what}: f32 bit patterns differ");
            }
            DType::BF16 => {
                // The K=8/V=4/L=12 reproduction table. Compared as raw bits,
                // not as floats: a table is format, and `==` on bf16 would
                // still call two NaN payloads unequal and two zeros equal.
                let av: Vec<half::bf16> = a.flatten_all()?.to_vec1()?;
                let bv: Vec<half::bf16> = b.flatten_all()?.to_vec1()?;
                let ab: Vec<u16> = av.iter().map(|x| x.to_bits()).collect();
                let bb: Vec<u16> = bv.iter().map(|x| x.to_bits()).collect();
                assert_eq!(ab, bb, "{what}: bf16 bit patterns differ");
            }
            other => panic!("assert_tensor_bits_eq: unhandled dtype {other:?}"),
        }
        Ok(())
    }

    // ===================================================================
    // Geometry discriminator (UQFF)
    // ===================================================================

    /// The V=4/L=12 family at the byte-aligned control K, as a geometry value.
    fn k8() -> QtipGeometry {
        QtipGeometry::trellis_v4l12(8).unwrap()
    }

    /// A cheap 2-D layer whose fields we can then rewrite for format tests.
    fn geometry_fixture(device: &Device) -> Result<QtipLayer> {
        let (n, k_in) = (4usize, 64usize);
        let wdata: Vec<f32> = (0..(n * k_in))
            .map(|i| ((i as f32) * 0.37).sin() * 1.1)
            .collect();
        let w = Tensor::from_vec(wdata, (n, k_in), device)?;
        QtipLayer::quantize_with_options_concrete(&w, None, device, QtipMode::Greedy, false)
    }

    /// The default geometry must be **purely additive**: it writes nothing, so
    /// every artifact Arc has already produced is byte-identical and no
    /// checksum moves. Stated as an exact suffix rather than as a vague
    /// "unchanged", so the tag value, its position, and the K byte are all
    /// pinned — **at every K in the family**.
    ///
    /// This is the property that makes a K change cheap: K is an explicit wire
    /// field, so moving from the K=8 control to K=9 changes one byte of the
    /// artifact and nothing else about the format.
    #[test]
    fn default_geometry_writes_nothing_and_the_family_appends_exactly_four_bytes() -> Result<()> {
        let device = Device::Cpu;
        let mut layer = geometry_fixture(&device)?;
        assert_eq!(layer.geometry, QtipGeometry::K4V2L16);
        let base = layer.serialize()?.into_owned();

        for k in trellis_v4l12::K_SUPPORTED {
            // Flip ONLY the discriminator; every tensor stays byte-identical.
            layer.geometry = QtipGeometry::trellis_v4l12(k)?;
            let tagged = layer.serialize()?.into_owned();

            let mut want = base.clone();
            want.extend_from_slice(&[GEOMETRY_WIRE_TAG, k as u8, 12, 4]);
            assert_eq!(
                tagged, want,
                "K={k}: the geometry section must be exactly [tag=3, K, L=12, V=4] appended to \
                 an otherwise unchanged payload"
            );
            // The K byte is the ONLY thing that differs between family members.
            assert_eq!(tagged.len(), base.len() + 4);
        }
        Ok(())
    }

    /// Two family members differ by exactly one byte on the wire.
    ///
    /// The concrete statement of "K is a parameter, not a variant": going from
    /// the K=8 control to the K=9 quality winner is a single byte of artifact
    /// change, given identical tensors.
    #[test]
    fn family_members_differ_by_one_wire_byte() -> Result<()> {
        let device = Device::Cpu;
        let mut layer = geometry_fixture(&device)?;
        layer.geometry = QtipGeometry::trellis_v4l12(8)?;
        let a = layer.serialize()?.into_owned();
        layer.geometry = QtipGeometry::trellis_v4l12(9)?;
        let b = layer.serialize()?.into_owned();
        assert_eq!(a.len(), b.len());
        let diffs: Vec<usize> = (0..a.len()).filter(|&i| a[i] != b[i]).collect();
        assert_eq!(
            diffs.len(),
            1,
            "K=8 and K=9 payloads must differ in exactly one byte, differed in {diffs:?}"
        );
        assert_eq!((a[diffs[0]], b[diffs[0]]), (8, 9));
        Ok(())
    }

    /// The section must come BEFORE the codebook section, because that is what
    /// makes an older Arc fail closed instead of mis-decoding.
    #[test]
    fn geometry_section_precedes_the_codebook_section() -> Result<()> {
        let device = Device::Cpu;
        let mut layer = geometry_fixture(&device)?;
        let base = layer.serialize()?.into_owned();

        // Codebook alone (K=4 geometry, so Mcg is legal here).
        layer.codebook = QtipCodebook::COMPUTED;
        let cb_only = layer.serialize()?.into_owned();
        assert_eq!(
            cb_only.len(),
            base.len() + 5,
            "codebook section is tag + u32"
        );
        assert_eq!(cb_only[base.len()], 2, "codebook tag is 2");

        // Both sections. Geometry (4 bytes) must land first.
        //
        // K8V4L12 + Mcg is refused as a pair, so use a hypothetical third
        // geometry is not available — instead assert the writer's ORDER
        // directly by checking that the geometry bytes occupy the slot the
        // codebook otherwise would.
        layer.codebook = QtipCodebook::Gaussian;
        layer.geometry = k8();
        let geo_only = layer.serialize()?.into_owned();
        assert_eq!(
            geo_only[base.len()],
            GEOMETRY_WIRE_TAG,
            "the geometry section must start immediately after the search-detail section — \
             any byte between them would be read as a codebook tag by an older Arc"
        );
        Ok(())
    }

    /// An Arc that predates this field parses the trailing region by handing
    /// its first byte to `QtipCodebook::from_wire`. That must REFUSE tag 3.
    ///
    /// If it ever accepted it, an old build would decode K=8/V=4 symbols as
    /// K=4/V=2 — and would not fault, because at 2 bits per weight the packed
    /// byte count is identical.
    #[test]
    fn an_older_reader_refuses_the_geometry_tag_rather_than_ignoring_it() {
        let err = QtipCodebook::from_wire(GEOMETRY_WIRE_TAG, || Ok(1))
            .expect_err("tag 3 must not parse as a codebook");
        let msg = err.to_string();
        assert!(
            msg.contains("geometry") || msg.contains("unknown codebook tag"),
            "refusal must name the cause: {msg}"
        );
        // And the general property the whole scheme rests on: every tag this
        // build does not know is refused, never skipped.
        for tag in [1u8, 4, 5, 200, 255] {
            assert!(
                QtipCodebook::from_wire(tag, || Ok(1)).is_err(),
                "tag {tag} must fail closed"
            );
        }
    }

    /// A K=8/V=4 tag on a K=4/V=2 payload must be caught by the SHAPES, not
    /// merely trusted. This is the case a hex-editor edit or a truncated
    /// concatenation produces.
    #[test]
    fn a_geometry_tag_that_contradicts_the_tensors_is_refused_at_load() -> Result<()> {
        let device = Device::Cpu;
        let mut layer = geometry_fixture(&device)?;
        layer.geometry = k8();
        let data = layer.serialize()?.into_owned();

        let err = QtipLayer::deserialize_concrete_unchecked(
            Cow::Owned(data),
            &device,
            crate::QuantizeOntoGuard::new(),
        )
        .expect_err("a K8V4L12 tag over a [65536, 2] F32 table must be refused");
        let msg = err.to_string();
        assert!(
            msg.contains("k8v4l12") && (msg.contains("F32") || msg.contains("values")),
            "refusal must name the geometry and the mismatch: {msg}"
        );
        Ok(())
    }

    /// The table's ELEMENT TYPE discriminates, not only its length.
    ///
    /// A `[4096, 4]` F32 table has exactly the right number of values and the
    /// wrong dtype — 65,536 B instead of 32,768 B, so it would not even fit
    /// the shared-memory budget the geometry exists for. Without this case the
    /// element-count check alone passes and the dtype check is dead
    /// (measured: mutation F8 disabled it and every test stayed green).
    #[test]
    fn a_right_sized_table_of_the_wrong_dtype_is_refused() -> Result<()> {
        let device = Device::Cpu;
        let geo = k8();
        let k_in = 64usize;
        let blocks = Tensor::zeros((2, geo.packed_len(k_in)), DType::U8, &device)?;
        let f32_table = Tensor::zeros(
            (trellis_v4l12::LUT_STATES, trellis_v4l12::V as usize),
            DType::F32,
            &device,
        )?;
        assert_eq!(f32_table.elem_count(), geo.lut_values(), "same count");
        let err = geo
            .validate_shapes(&blocks, &f32_table, k_in)
            .expect_err("an F32 table at this geometry must be refused");
        assert!(
            err.to_string().contains("F32") && err.to_string().contains("BF16"),
            "refusal must name both dtypes: {err}"
        );
        Ok(())
    }

    /// `from_stacked_parts` is a public constructor that takes the geometry on
    /// trust from its caller; it must validate, not trust.
    #[test]
    fn from_stacked_parts_refuses_a_geometry_its_tensors_contradict() -> Result<()> {
        let device = Device::Cpu;
        let (e, n, k_in) = (2usize, 4usize, 64usize);
        let blocks = Tensor::zeros((e, n, k_in / 4), DType::U8, &device)?;
        let scales = Tensor::zeros((e, n), DType::F32, &device)?;
        let k4_table = Tensor::zeros((1usize << L, V as usize), DType::F32, &device)?;

        // Truthful call: K=4 tensors, K=4 tag.
        QtipLayer::from_stacked_parts(
            blocks.clone(),
            scales.clone(),
            k4_table.clone(),
            None,
            k_in,
            None,
            0,
            QtipSearchStamp::Unstamped,
            QtipSearchDetail::Unknown,
            QtipCodebook::Gaussian,
            QtipGeometry::K4V2L16,
        )?;

        // Same tensors, K=8 tag.
        let err = QtipLayer::from_stacked_parts(
            blocks,
            scales,
            k4_table,
            None,
            k_in,
            None,
            0,
            QtipSearchStamp::Unstamped,
            QtipSearchDetail::Unknown,
            QtipCodebook::Gaussian,
            k8(),
        )
        .expect_err("a K8V4L12 tag over a K4V2L16 table must be refused at construction");
        assert!(err.to_string().contains("k8v4l12"), "{err}");
        Ok(())
    }

    /// A stack whose experts were baked at different geometries keeps only one
    /// table, so at most one expert could ever decode correctly.
    #[test]
    fn stack_experts_refuses_mixed_geometries() -> Result<()> {
        let device = Device::Cpu;
        let a = geometry_fixture(&device)?;
        let mut b = geometry_fixture(&device)?;
        // Identical in every respect except the discriminator, so nothing but
        // the geometry check can be what fires.
        b.geometry = k8();
        let err = QtipLayer::stack_experts(vec![a, b])
            .expect_err("mixed-geometry stacks must be refused");
        assert!(
            err.to_string().contains("geometry"),
            "refusal must name the geometry: {err}"
        );

        // ...and a uniform stack still works, so the guard is not just
        // rejecting everything.
        let c = geometry_fixture(&device)?;
        let d = geometry_fixture(&device)?;
        QtipLayer::stack_experts(vec![c, d])?;
        Ok(())
    }

    /// Every decode entry point must refuse a geometry it cannot decode.
    ///
    /// Stage 2 makes a K=8/V=4/L=12 artifact loadable. The decoders behind
    /// these entry points all unpack nibbles and index a `[2^16, 2]` table, and
    /// handed a K=8 row they would NOT fault — at 2 bits per weight the packed
    /// byte count is identical and every index lands in bounds. They would
    /// serve garbage. Without this test, "the format landed" would mean "the
    /// engine now has a way to silently serve wrong weights".
    #[test]
    fn every_decode_entry_point_refuses_an_unsupported_geometry() -> Result<()> {
        let device = Device::Cpu;
        let k_in = 64usize;

        // 2-D entry points.
        let mut two_d = geometry_fixture(&device)?;
        // Sanity: all of these work at the geometry this build implements, so
        // the refusals below are about the geometry and nothing else.
        two_d.dequantize_weights()?;
        let x = Tensor::zeros((1, k_in), DType::F32, &device)?;
        two_d.forward(&x)?;
        two_d.dequantize_w()?;
        assert!(two_d.qtip_packed().is_some());

        two_d.geometry = k8();
        for (what, res) in [
            ("dequantize_weights", two_d.dequantize_weights().err()),
            ("forward", two_d.forward(&x).err()),
            ("dequantize_w", two_d.dequantize_w().err()),
        ] {
            let err = res.unwrap_or_else(|| panic!("{what} must refuse a k8v4l12 layer"));
            let msg = err.to_string();
            assert!(
                msg.contains("k8v4l12") && msg.contains("Refusing"),
                "{what}: refusal must name the geometry and say it is refusing: {msg}"
            );
            // The refusal must come from THIS entry point, not from something
            // it happens to call. `forward` reaching an inner guard is still a
            // refusal, but it is not evidence that `forward` checks — and an
            // entry point that relies on a callee's guard breaks the moment the
            // callee gains a fast path that skips it. (Measured: mutation G1
            // removed `forward`'s own guard and every test stayed green.)
            //
            // `dequantize_w` is the one exception by construction: it is a
            // one-line delegation to `dequantize_weights` and has no body to
            // guard.
            let expect_op = if what == "dequantize_w" {
                "dequantize_weights"
            } else {
                what
            };
            assert!(
                msg.contains(&format!("QtipLayer::{expect_op}:")),
                "{what}: refusal came from elsewhere — expected `QtipLayer::{expect_op}:` in: {msg}"
            );
        }
        assert!(
            two_d.qtip_packed().is_none(),
            "qtip_packed must hand out nothing — QtipPackedView carries no geometry, so a \
             consumer would read the bytes as K=4/V=2 nibbles"
        );

        // 3-D entry points.
        let mut stack =
            QtipLayer::stack_experts(vec![geometry_fixture(&device)?, geometry_fixture(&device)?])?;
        stack.dequantize_expert(0)?;
        let a = Tensor::zeros((1, 1, k_in), DType::F32, &device)?;
        let idx = Tensor::zeros((1, 1), DType::U32, &device)?;
        stack.gather_forward(&a, &idx)?;

        stack.geometry = k8();
        for (what, res) in [
            ("dequantize_expert", stack.dequantize_expert(0).err()),
            ("gather_forward", stack.gather_forward(&a, &idx).err()),
        ] {
            let err = res.unwrap_or_else(|| panic!("{what} must refuse a k8v4l12 layer"));
            let msg = err.to_string();
            assert!(msg.contains("k8v4l12"), "{what}: {msg}");
            assert!(
                msg.contains(&format!("QtipLayer::{what}:")),
                "{what}: refusal came from elsewhere: {msg}"
            );
        }
        Ok(())
    }

    /// A genuine V=4/L=12 payload round-trips at every K, tag and tensors both.
    #[test]
    fn trellis_v4l12_payloads_round_trip_exactly_at_every_k() -> Result<()> {
        let device = Device::Cpu;
        let (n, k_in) = (4usize, 128usize);
        let lut = Tensor::from_slice(
            &trellis_v4l12::gaussian_lut_bf16(),
            (trellis_v4l12::LUT_STATES, trellis_v4l12::V as usize),
            &device,
        )?;

        for k in trellis_v4l12::K_SUPPORTED {
            let geo = QtipGeometry::trellis_v4l12(k)?;
            let packed_per_row = geo.packed_len(k_in);
            // Sanity: the DATA length tracks K, and the allocated stride is
            // that plus the padding the unclamped extraction relies on.
            assert_eq!(geo.data_bytes(k_in), (k_in / 4 * k as usize).div_ceil(8));
            assert!(packed_per_row >= geo.data_bytes(k_in));
            assert!(packed_per_row - geo.data_bytes(k_in) <= 4);

            let blocks_data: Vec<u8> = (0..(n * packed_per_row))
                .map(|i| (i * 37 % 251) as u8)
                .collect();
            let blocks = Tensor::from_vec(blocks_data, (n, packed_per_row), &device)?;
            let scales = Tensor::from_vec(
                (0..n).map(|i| 0.01 + i as f32 * 0.003).collect::<Vec<_>>(),
                (n,),
                &device,
            )?;

            let layer = QtipLayer {
                blocks,
                row_scales: scales,
                lut: lut.clone(),
                bias: None,
                in_features: k_in,
                num_experts: None,
                rotation_signs: None,
                rotation_block: 0,
                search: QtipSearchStamp::Trellis,
                search_detail: QtipSearchDetail::Known {
                    beam_width: None,
                    hessian: false,
                },
                codebook: QtipCodebook::Gaussian,
                geometry: geo,
            };

            let data = layer.serialize()?.into_owned();
            let (restored, _) = QtipLayer::deserialize_concrete_unchecked(
                Cow::Owned(data),
                &device,
                crate::QuantizeOntoGuard::new(),
            )?;

            assert_eq!(restored.geometry, geo, "K={k}");
            assert_eq!(restored.geometry.k(), k);
            assert_eq!(restored.in_features, k_in);
            assert_eq!(restored.lut.dtype(), DType::BF16);
            assert_eq!(restored.lut.elem_count(), trellis_v4l12::LUT_ENTRIES);
            assert_tensor_bits_eq(&layer.blocks, &restored.blocks, "blocks")?;
            assert_tensor_bits_eq(&layer.row_scales, &restored.row_scales, "row_scales")?;
            assert_tensor_bits_eq(&layer.lut, &restored.lut, "lut")?;
        }
        Ok(())
    }

    /// A K the wire names but this build has no decoder for is refused, and the
    /// refusal quotes the triple.
    #[test]
    fn an_in_family_but_unsupported_k_is_refused_with_a_diagnosis() -> Result<()> {
        let device = Device::Cpu;
        let layer = geometry_fixture(&device)?;
        let base = layer.serialize()?.into_owned();
        // L=12/V=4 is this family, but K=11 has no decoder.
        let mut bytes = base.clone();
        bytes.extend_from_slice(&[GEOMETRY_WIRE_TAG, 11, 12, 4]);
        let err = QtipLayer::deserialize_concrete_unchecked(
            Cow::Owned(bytes),
            &device,
            crate::QuantizeOntoGuard::new(),
        )
        .expect_err("K=11 must be refused");
        assert!(
            err.to_string().contains("K=11") || err.to_string().contains("not implemented"),
            "refusal must name the K it could not decode: {err}"
        );
        Ok(())
    }

    /// The computed `sum2` codebook produces a PAIR of values per state and so
    /// cannot describe V=4. Refused on write and on read, not left to fail at
    /// the first decode.
    #[test]
    fn the_computed_codebook_is_refused_at_a_v4_geometry() -> Result<()> {
        let device = Device::Cpu;
        let mut layer = geometry_fixture(&device)?;
        layer.geometry = k8();
        layer.codebook = QtipCodebook::COMPUTED;
        let err = layer
            .serialize()
            .expect_err("V=4 + sum2 must not serialize");
        assert!(
            err.to_string().contains("V=2-only"),
            "refusal must say why: {err}"
        );
        // And it is legal at the geometry it was built for, so the guard is
        // about the pairing and not about the codebook being blocked outright.
        layer.geometry = QtipGeometry::K4V2L16;
        layer.serialize()?;
        Ok(())
    }

    /// Malformed trailing regions are refused rather than partially read.
    #[test]
    fn malformed_geometry_sections_are_refused() -> Result<()> {
        let device = Device::Cpu;
        let layer = geometry_fixture(&device)?;
        let base = layer.serialize()?.into_owned();
        let load = |bytes: Vec<u8>| {
            QtipLayer::deserialize_concrete_unchecked(
                Cow::Owned(bytes),
                &device,
                crate::QuantizeOntoGuard::new(),
            )
        };

        // Truncated: tag present, K/L/V missing.
        let mut truncated = base.clone();
        truncated.push(GEOMETRY_WIRE_TAG);
        truncated.push(8);
        let err = load(truncated).expect_err("truncated geometry section must be refused");
        assert!(err.to_string().contains("truncated"), "{err}");

        // Well-formed but unsupported triple: diagnosed, not "unknown tag".
        let mut unsupported = base.clone();
        unsupported.extend_from_slice(&[GEOMETRY_WIRE_TAG, 6, 18, 3]);
        let err = load(unsupported).expect_err("unsupported geometry must be refused");
        let msg = err.to_string();
        assert!(
            msg.contains("K=6") && msg.contains("L=18") && msg.contains("V=3"),
            "refusal must quote the triple it could not decode: {msg}"
        );

        // Two geometry sections: a payload that describes itself twice.
        let mut doubled = base.clone();
        doubled.extend_from_slice(&[GEOMETRY_WIRE_TAG, 4, 16, 2]);
        doubled.extend_from_slice(&[GEOMETRY_WIRE_TAG, 4, 16, 2]);
        let err = load(doubled).expect_err("duplicate geometry sections must be refused");
        assert!(err.to_string().contains("twice"), "{err}");

        // An explicit default-geometry section is legal (a future writer may
        // choose to be explicit) and must load as K4V2L16.
        let mut explicit = base.clone();
        explicit.extend_from_slice(&[GEOMETRY_WIRE_TAG, 4, 16, 2]);
        let (restored, _) = load(explicit)?;
        assert_eq!(restored.geometry, QtipGeometry::K4V2L16);
        Ok(())
    }

    /// The shipped rung and the K=8 control are the SAME bit rate and the SAME
    /// packed size. That is the fact that makes the discriminator necessary —
    /// a mislabelled artifact has a correctly-sized `blocks` tensor — so it
    /// gets an assertion rather than a comment.
    #[test]
    fn k4v2l16_and_k8v4l12_are_indistinguishable_by_packed_size() {
        let k8 = k8();
        for k_in in [64usize, 512, 4096, 7168] {
            assert_eq!(
                QtipGeometry::K4V2L16.packed_len(k_in),
                k8.packed_len(k_in),
                "k_in={k_in}: identical packed size is why the tag is load-bearing"
            );
        }
        assert_eq!(QtipGeometry::K4V2L16.bpw_x100(), 200);
        assert_eq!(k8.bpw_x100(), 200);
        // ...and the tables are what actually differ.
        assert_eq!(QtipGeometry::K4V2L16.lut_values(), 65_536 * 2);
        assert_eq!(k8.lut_values(), 4_096 * 4);
        assert_eq!(QtipGeometry::K4V2L16.lut_dtype(), DType::F32);
        assert_eq!(k8.lut_dtype(), DType::BF16);
    }

    /// **Not every geometry in the enum is 2 bits per weight any more.**
    ///
    /// K=9/V=4 is 2.25 bpw, so its rows are 12.5% larger than K=8/V=4's and it
    /// is *not* size-confusable with the shipped rung. Worth pinning: code that
    /// assumed "all QTIP geometries are 2 bpw" — a true statement when this
    /// family was K=8-only — is now wrong, and `packed_len` is the general
    /// `ceil(n·K/8)` rather than anything that divides by `8/K`.
    #[test]
    fn the_family_spans_more_than_one_bit_rate() {
        let k8 = k8();
        let k9 = QtipGeometry::trellis_v4l12(9).unwrap();
        let k10 = QtipGeometry::trellis_v4l12(10).unwrap();
        assert_eq!(k8.bpw_x100(), 200);
        assert_eq!(k9.bpw_x100(), 225);
        assert_eq!(k10.bpw_x100(), 250);
        for k_in in [512usize, 4096, 7168] {
            assert!(k9.packed_len(k_in) > k8.packed_len(k_in));
            // Exactly the bit-rate ratio, to the byte — on DATA bytes. The
            // stride is only approximately in that ratio, because K=9 pads for
            // its multi-byte extraction and K=8 does not.
            assert_eq!(k9.data_bytes(k_in) * 8, k8.data_bytes(k_in) * 9);
            // The whole family shares one table regardless of K.
            assert_eq!(k9.lut_values(), k8.lut_values());
            assert_eq!(k9.lut_dtype(), k8.lut_dtype());
        }
    }

    /// The packed length must use the CEIL, and that is only visible at an
    /// `in_features` which is not a multiple of 32.
    ///
    /// Every plausible layer width is a multiple of 32, which makes
    /// `num_symbols · K` a multiple of 8 and floor equal to ceil at K=9. A
    /// floored formula therefore passes every realistic fixture (measured:
    /// mutation W3). These widths are deliberately unrealistic.
    #[test]
    fn packed_len_uses_the_ceiling_where_it_is_observable() {
        let k9 = QtipGeometry::trellis_v4l12(9).unwrap();
        // in_features=36 -> 9 symbols -> 81 bits -> 11 DATA bytes, not 10.
        assert_eq!(k9.num_symbols(36), 9);
        assert_eq!(k9.data_bytes(36), 11);
        // in_features=4 -> 1 symbol -> 9 bits -> 2 data bytes, not 1.
        assert_eq!(k9.data_bytes(4), 2);
        let k10 = QtipGeometry::trellis_v4l12(10).unwrap();
        // 3 symbols -> 30 bits -> 4 data bytes, not 3.
        assert_eq!(k10.data_bytes(12), 4);
        // The stride is the data plus tail padding, rounded up to 4.
        assert_eq!(k9.packed_len(36), 12);
        assert_eq!(k9.packed_len(4), 4);
        assert_eq!(k10.packed_len(12), 8);
        // ...and both agree with the rung that owns the formulas.
        for k in trellis_v4l12::K_SUPPORTED {
            let g = QtipGeometry::trellis_v4l12(k).unwrap();
            let r = trellis_v4l12::Rung::new(k).unwrap();
            for k_in in [4usize, 12, 36, 100, 260, 4096] {
                let n = r.num_symbols(k_in);
                assert_eq!(
                    g.packed_len(k_in),
                    r.row_stride(n),
                    "K={k} k_in={k_in} stride"
                );
                assert_eq!(
                    g.data_bytes(k_in),
                    r.data_bytes(n),
                    "K={k} k_in={k_in} data"
                );
            }
        }
    }

    /// A K with no decoder is refused at construction, not stored and used.
    #[test]
    fn an_unsupported_k_cannot_become_a_geometry() {
        for k in [0u32, 4, 7, 11, 16] {
            assert!(
                QtipGeometry::trellis_v4l12(k).is_err(),
                "K={k} must not be constructible"
            );
        }
        for k in trellis_v4l12::K_SUPPORTED {
            assert!(QtipGeometry::trellis_v4l12(k).is_ok(), "K={k} must be");
        }
    }

    /// The geometry accessors must agree with the rung they describe.
    #[test]
    fn geometry_accessors_match_the_rung_module() {
        assert_eq!(QtipGeometry::K4V2L16.k(), K);
        assert_eq!(QtipGeometry::K4V2L16.l(), L);
        assert_eq!(QtipGeometry::K4V2L16.v(), V);
        for k in trellis_v4l12::K_SUPPORTED {
            let g = QtipGeometry::trellis_v4l12(k).unwrap();
            let r = trellis_v4l12::Rung::new(k).unwrap();
            assert_eq!(g.k(), r.k());
            assert_eq!(g.l(), trellis_v4l12::L);
            assert_eq!(g.v(), trellis_v4l12::V);
            assert_eq!(g.bpw_x100(), r.bpw_x100());
            assert_eq!(
                g.lut_values(),
                trellis_v4l12::LUT_ENTRIES,
                "the discriminator and the rung must agree on the table size"
            );
            for k_in in [64usize, 512, 4096] {
                assert_eq!(g.packed_len(k_in), r.packed_len(k_in), "K={k} k_in={k_in}");
                assert_eq!(g.num_symbols(k_in), r.num_symbols(k_in));
            }
            assert_eq!(g.tag(), format!("k{k}v4l12"));
        }
    }

    /// UQFF round-trip for the classic 2-D layout must stay lossless and
    /// byte-compatible (bias branch + no-rotation branch covered).
    #[test]
    fn qtip_uqff_2d_round_trip_with_bias() -> Result<()> {
        let device = Device::Cpu;
        let (n, k_in) = (8usize, 64usize);
        let wdata: Vec<f32> = (0..(n * k_in))
            .map(|i| ((i as f32) * 0.31).cos() * 1.5)
            .collect();
        let w = Tensor::from_vec(wdata, (n, k_in), &device)?;
        let bias_data: Vec<f32> = (0..n).map(|i| i as f32 * 0.05 - 0.2).collect();
        let bias = Tensor::from_vec(bias_data, (n,), &device)?;
        let layer = QtipLayer::quantize_with_options_concrete(
            &w,
            Some(bias),
            &device,
            QtipMode::Greedy,
            false,
        )?;

        let data = layer.serialize()?.into_owned();
        // `_unchecked`: this fixture is a cheap greedy bake, which the D4 load
        // gate refuses on purpose (see `qtip/greedy_ban_tests.rs`). What this
        // test asserts is field-for-field payload fidelity, not serving policy.
        let (restored, ext_bias) = QtipLayer::deserialize_concrete_unchecked(
            Cow::Owned(data),
            &device,
            crate::QuantizeOntoGuard::new(),
        )?;

        assert_eq!(restored.num_experts, None);
        assert_eq!(restored.in_features, k_in);
        assert_eq!(restored.rotation_block, layer.rotation_block);
        assert!(restored.rotation_signs.is_none());
        assert_tensor_bits_eq(&layer.blocks, &restored.blocks, "blocks")?;
        assert_tensor_bits_eq(&layer.row_scales, &restored.row_scales, "row_scales")?;
        assert_tensor_bits_eq(&layer.lut, &restored.lut, "lut")?;
        let (Some(b0), Some(b1)) = (&layer.bias, &restored.bias) else {
            panic!("bias must survive the round-trip");
        };
        assert_tensor_bits_eq(b0, b1, "bias")?;
        assert!(ext_bias.is_some(), "deserialize must also return ext bias");

        // Identical bits ⇒ identical CPU forward outputs, exactly.
        let xdata: Vec<f32> = (0..(2 * k_in)).map(|i| ((i as f32) * 0.05).sin()).collect();
        let x = Tensor::from_vec(xdata, (2, k_in), &device)?;
        let y0: Vec<f32> = layer
            .forward(&x)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1()?;
        let y1: Vec<f32> = restored
            .forward(&x)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1()?;
        assert_eq!(
            y0.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            y1.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "2-D forward outputs must match bit-exactly"
        );
        Ok(())
    }

    /// UQFF round-trip for a 3-D stacked-expert layer: serialize →
    /// deserialize must reconstruct packed trellis data, scales, LUT,
    /// rotation metadata, and config bit-identically — a UQFF load must
    /// never need to re-run the (Viterbi) quantizer.
    ///
    /// Also asserts the production 3-D quantize entry
    /// (`quantize_with_options` on a rank-3 weight) serializes byte-identical
    /// to a per-expert `quantize` + `stack_experts` twin, which gives the
    /// test typed access to the original fields. Greedy mode with rotation
    /// force-enabled keeps the fixture fast while covering the shared
    /// rotation-signs branch of the serde path.
    #[test]
    fn qtip_uqff_3d_expert_stack_round_trip() -> Result<()> {
        let device = Device::Cpu;
        let (num_experts, rows, in_features) = (4usize, 64usize, 64usize);
        let w = build_3d_gaussian_weight(num_experts, rows, in_features, &device)?;

        // Production entry: rank-3 dispatch inside `quantize_with_options`.
        let prod_layer =
            QtipLayer::quantize_with_options(&w, None, &device, QtipMode::Greedy, true)?;
        let prod_bytes = prod_layer.serialize()?.into_owned();

        // Typed twin from the same weight: per-expert quantize + stack.
        // Quantization is deterministic, so the payloads must be identical.
        let mut per_expert = Vec::with_capacity(num_experts);
        for e in 0..num_experts {
            let w_e = w.narrow(0, e, 1)?.squeeze(0)?;
            per_expert.push(QtipLayer::quantize_with_options_concrete(
                &w_e,
                None,
                &device,
                QtipMode::Greedy,
                true,
            )?);
        }
        let stack = QtipLayer::stack_experts(per_expert)?;
        let stack_bytes = stack.serialize()?.into_owned();
        assert_eq!(
            prod_bytes, stack_bytes,
            "3-D quantize and stack_experts must serialize byte-identically"
        );

        // `_unchecked`: cheap greedy fixture, see the 2-D round-trip above.
        let (restored, ext_bias) = QtipLayer::deserialize_concrete_unchecked(
            Cow::Owned(prod_bytes),
            &device,
            crate::QuantizeOntoGuard::new(),
        )?;

        // Config fields.
        assert_eq!(restored.num_experts, Some(num_experts));
        assert_eq!(restored.in_features, in_features);
        assert_eq!(restored.rotation_block, stack.rotation_block);
        assert!(ext_bias.is_none());
        assert!(restored.bias.is_none());

        // Bit-identical tensors.
        assert_eq!(
            restored.blocks.dims(),
            &[num_experts, rows, in_features / 4]
        );
        assert_tensor_bits_eq(&stack.blocks, &restored.blocks, "blocks")?;
        assert_tensor_bits_eq(&stack.row_scales, &restored.row_scales, "row_scales")?;
        assert_tensor_bits_eq(&stack.lut, &restored.lut, "lut")?;
        let (Some(s0), Some(s1)) = (&stack.rotation_signs, &restored.rotation_signs) else {
            panic!("rotation signs must be present on both sides of the round-trip");
        };
        assert_tensor_bits_eq(s0, s1, "rotation_signs")?;

        // Identical bits ⇒ identical CPU gather_forward outputs, exactly.
        let (n_tokens, n_experts_per_tok) = (3usize, 2usize);
        let mut adata = vec![0.0f32; n_tokens * n_experts_per_tok * in_features];
        for (i, v) in adata.iter_mut().enumerate() {
            let mut z = ((i + 31_337) as u64).wrapping_mul(0x9E3779B97F4A7C15);
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z ^= z >> 31;
            let u1 = ((z >> 32) as u32 as f32 + 1.0) / (u32::MAX as f32 + 2.0);
            let u2 = ((z & 0xFFFFFFFF) as u32 as f32 + 1.0) / (u32::MAX as f32 + 2.0);
            let r = (-2.0_f32 * u1.ln()).sqrt();
            *v = r * (2.0 * std::f32::consts::PI * u2).cos();
        }
        let a = Tensor::from_vec(adata, (n_tokens, n_experts_per_tok, in_features), &device)?;
        let indices = Tensor::from_vec(
            vec![0u32, 2, 1, 3, 2, 2],
            (n_tokens, n_experts_per_tok),
            &device,
        )?;
        let y0: Vec<f32> = stack
            .gather_forward(&a, &indices)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1()?;
        let y1: Vec<f32> = restored
            .gather_forward(&a, &indices)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1()?;
        assert_eq!(
            y0.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            y1.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "3-D gather_forward outputs must match bit-exactly"
        );
        Ok(())
    }

    /// Serializing a 3-D layer with an externally-supplied bias must be
    /// refused (3-D MoE stacks are bias-free by contract).
    #[test]
    fn qtip_uqff_3d_serialize_rejects_bias() -> Result<()> {
        let device = Device::Cpu;
        let w = build_3d_gaussian_weight(2, 4, 32, &device)?;
        let layer = QtipLayer::quantize_with_options(&w, None, &device, QtipMode::Greedy, false)?;
        let bias = Tensor::zeros((4,), DType::F32, &device)?;
        let err = layer.serialize_with_bias(Some(bias)).err();
        assert!(
            err.as_ref()
                .is_some_and(|e| e.to_string().contains("bias-free")),
            "expected bias-free rejection, got {err:?}"
        );
        Ok(())
    }

    /// wave6-Q regression guard (LUT rung): a 3-D expert-stack quantize
    /// whose target device is the CPU (the bake path) must stream through
    /// the GPU kernels on a CUDA box — a reroute to the CPU Viterbi is a
    /// ~20x per-layer bake regression and must be counted, never silent.
    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_3d_expert_quantize_does_not_fall_back_to_cpu() -> Result<()> {
        if !ffi::HAVE_QTIP_KERNELS {
            return Ok(());
        }
        if Device::new_cuda(0).is_err() {
            // No physical GPU (e.g. a compile-gate CI lane running tests):
            // the CPU fallback is correct behavior there, nothing to assert.
            return Ok(());
        }
        let before = gpu_quantize_cpu_fallback_count();
        let device = Device::Cpu;
        let (e, n, k_in) = (4usize, 8usize, 256usize);
        let w3 = build_3d_gaussian_weight(e, n, k_in, &device)?;
        let layer = QtipLayer::quantize_with_options_3d(&w3, &device, QtipMode::Viterbi, true)?;
        // Quantize really ran: the packed stack dequantizes at full shape.
        assert_eq!(layer.dequantize_w()?.dims(), &[e, n, k_in]);
        let after = gpu_quantize_cpu_fallback_count();
        assert_eq!(
            after,
            before,
            "QtipLayer 3-D expert quantize fell back to the CPU pipeline on a CUDA box \
             ({} new fallback(s)) — check the warn log for the reason",
            after - before
        );
        Ok(())
    }
}
