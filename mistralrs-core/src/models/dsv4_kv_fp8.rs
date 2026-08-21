//! FP8 storage for DeepSeek-V4's fused MQA key cache.
//!
//! # Why this is a serialization change, not a precision change
//!
//! V4 is FP8-QAT: the reference (`inference/model.py:506`,
//! `act_quant(kv[..., :-rd], 64, ..., inplace=True)`) round-trips the
//! **non-rope** dims of every K through block-wise E4M3 on every forward.
//! Arc does the same — [`quantize_k`] is the only implementation of it — and
//! before this module the round-tripped value was then widened back to BF16
//! and *stored* at BF16. Every stored nope element was therefore one of 256
//! E4M3 values times a per-block scale, carried in 16 bits.
//!
//! Storing the 8-bit code plus the block's `amax` instead is **bit-exact**:
//! [`V4PackedK::dequant`] reproduces the BF16 tensor the old code stored,
//! bit pattern for bit pattern, because
//!
//! * the E4M3 code decodes through [`e4m3_lut`], whose entry `i` is exactly
//!   `F8E4M3::from_bits(i).to_f32()` — the same conversion candle's
//!   `F8E4M3 -> F32` cast performs; and
//! * the block scale is recomputed from the stored `amax` with the *same two
//!   ops* the quantizer used (`amax / 448.0`, then `+ 1e-12`), and `amax` is
//!   stored losslessly because it **is one of the block's own elements** and
//!   therefore already exactly representable in the activation dtype.
//!
//! `kv_fp8_roundtrip_is_bit_exact_vs_reference` pins this against a verbatim
//! copy of the pre-change implementation.
//!
//! # Layout and cost
//!
//! Per token, per layer, for V4-Flash (`head_dim = 512`, `qk_rope_head_dim =
//! 64`, `num_key_value_heads = 1`, BF16 activations):
//!
//! | half | contents | bytes |
//! |---|---|---|
//! | K | 448 E4M3 codes (U8) | 448 |
//! | V | 64 rope (BF16) + 7 block `amax` (BF16) | 142 |
//! | | | **590** |
//!
//! against **1,026** for the previous layout (512 BF16 + a 1-wide marker) and
//! **584** for SGLang's DSV4 pool, which stores 7 UE8M0 scales + 1 pad byte
//! where this stores 7 BF16 `amax`. Those 12 bytes are the price of bit
//! exactness: a UE8M0 scale is a *different* scale, so the round trip through
//! it would not reproduce what Arc stores today.
//!
//! The rope tail keeps its BF16 storage on purpose — RoPE is applied *before*
//! the quantizer and only to those dims, which is exactly why the reference
//! keeps them BF16 too.

use candle_core::{DType, Device, Result, Tensor, D};
use float8::F8E4M3;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{OnceLock, RwLock};

/// Block width of the activation quantizer, matching the reference's
/// `act_quant(..., 64, ...)`.
pub(crate) const KV_QUANT_BLOCK: usize = 64;

/// E4M3 max magnitude; the quantizer scales each block so its largest
/// magnitude lands on it.
const E4M3_MAX: f64 = 448.0;

/// Which arithmetic produces the E4M3 code.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum KvQuantMode {
    /// One fused device kernel per direction (`ArcQuant`,
    /// `mistralrs_quant::arc_kvquant`). **The default.** Bit-identical to
    /// [`Self::CpuExact`] — see that kernel's header for how each arithmetic
    /// step is pinned to the candle op it replaces — but it is one launch
    /// instead of ~11, and it removes the blocking device→host copy entirely.
    ///
    /// Falls back to [`Self::CpuExact`] when the tensor is not on CUDA, which
    /// is what keeps the CPU unit tests below meaningful.
    FusedDevice,
    /// Exact E4M3 via candle's CPU cast (candle has no CUDA `F8E4M3` cast —
    /// "named symbol not found"), at the price of one device sync per layer.
    /// `ARC_KV_FP8_IMPL=cpu`.
    CpuExact,
    /// On-device float arithmetic that reproduces E4M3's value grid with
    /// round-half-away-from-zero instead of round-half-to-even. Removes the
    /// sync; the FP8-trained model tolerates the sub-ULP-of-FP8 difference.
    /// `ARC_GPU_ACT_QUANT=1`. (RUN-161 throughput.)
    ///
    /// **Kept reachable, but it is NOT the fix and must not be shipped as one.**
    /// Measured on H200 at B=1 it is *slower* than the sync it removes
    /// (interleaved A1 66.99 / B1 67.71 / A2 68.05 / B2 70.30 ms/token, with
    /// `kv_fp8_quant` 73.49 → 134.18 µs/call): it trades one blocking copy for
    /// ~19 extra elementwise launches per layer, and this machine is bottlenecked
    /// on op count, not on kernel speed. That measurement is the reason
    /// [`Self::FusedDevice`] is one kernel rather than a chain of candle ops.
    GpuApprox,
}

/// The legal values of `ARC_KV_FP8_IMPL`, for error messages and for the test
/// that pins the table.
pub(crate) const KV_FP8_IMPL_VALUES: &[&str] = &[
    "fused",
    "fused_device",
    "cpu",
    "cpu_exact",
    "gpu",
    "gpu_approx",
];

impl KvQuantMode {
    /// Pure half of [`Self::from_env`], so the table can be tested without
    /// mutating the process environment (which is racy across test threads and
    /// `unsafe` since the 2024 edition).
    ///
    /// `raw` is the `ARC_KV_FP8_IMPL` value (`None` when unset); `legacy_gpu`
    /// is whether the pre-existing `ARC_GPU_ACT_QUANT` is set. Returns the mode
    /// and, when the input was not understood, the complaint to shout about it.
    pub(crate) fn parse_impl(raw: Option<&str>, legacy_gpu: bool) -> (Self, Option<String>) {
        let default = if legacy_gpu {
            // Preserved from before this flag existed: `ARC_GPU_ACT_QUANT=1`
            // selects the approximate on-device path. It only applies when
            // `ARC_KV_FP8_IMPL` says nothing.
            Self::GpuApprox
        } else {
            Self::FusedDevice
        };
        match raw.map(|s| s.trim()) {
            None | Some("") => (default, None),
            Some(v) => match v.to_ascii_lowercase().as_str() {
                "cpu" | "cpu_exact" => (Self::CpuExact, None),
                "gpu" | "gpu_approx" => (Self::GpuApprox, None),
                "fused" | "fused_device" => (Self::FusedDevice, None),
                // NOT `default`. Falling through to `default` here is how a
                // typo used to select `GpuApprox` — the one variant documented
                // as *not* bit-exact and "must not be shipped" — on any box
                // that still had `ARC_GPU_ACT_QUANT` set from an earlier
                // experiment. An unparsable value now lands on the bit-exact
                // default and says so.
                _ => (
                    Self::FusedDevice,
                    Some(format!(
                        "ARC_KV_FP8_IMPL={v:?} is not a legal value (expected one of {}); \
                         falling back to the default fused-device kernel. \
                         NOTE: this flag selects WHICH ARITHMETIC produces the E4M3 code. \
                         It has no 'off' — V4 is FP8-QAT and the quantize/dequantize round \
                         trip is the model's numerics, not an optimisation. The flag that \
                         gates FP8 KV *storage* is ARC_V4_FP8_KV.",
                        KV_FP8_IMPL_VALUES.join(", ")
                    )),
                ),
            },
        }
    }

    /// Resolved once per process: this is called per attention layer per
    /// forward, and `deepseek4` has already been bitten by per-call
    /// `std::env::var_os` in exactly that position (~390 environment scans per
    /// forward, wave33).
    ///
    /// Reads `ARC_KV_FP8_IMPL`. The former spelling `ARC_KV_FP8_MODE` is still
    /// honoured — loudly — because renaming it silently would turn every
    /// operator's muscle memory into a new silent failure, which is the same
    /// disease the rename cures.
    pub(crate) fn from_env() -> Self {
        static MODE: OnceLock<KvQuantMode> = OnceLock::new();
        *MODE.get_or_init(|| {
            // Read by VALUE. `ARC_GPU_ACT_QUANT=0` now means what it says. Under
            // the presence test it selected `GpuApprox` — the one variant this
            // module documents as not bit-exact and "must not be shipped".
            let legacy_gpu = mistralrs_quant::env_flag_is_set("ARC_GPU_ACT_QUANT");
            let new = std::env::var("ARC_KV_FP8_IMPL").ok();
            let old = std::env::var("ARC_KV_FP8_MODE").ok();
            if old.is_some() {
                // eprintln! as well as tracing: a rented box often runs before
                // a subscriber is installed, and a deprecation nobody sees is
                // not a deprecation.
                let msg = "ARC_KV_FP8_MODE has been renamed ARC_KV_FP8_IMPL (it selects the \
                           quantizer ARITHMETIC; it never had an 'off'). The old name still \
                           works for now — please update.";
                eprintln!("[arc-kv-fp8] {msg}");
                tracing::warn!("{msg}");
            }
            let raw = new.or(old);
            let (mode, complaint) = Self::parse_impl(raw.as_deref(), legacy_gpu);
            if let Some(c) = complaint {
                eprintln!("[arc-kv-fp8] {c}");
                tracing::error!("{c}");
            }
            mode
        })
    }
}

/// How many times the fused device kernels actually ran.
///
/// D18/D33: a parity test that passes while these stay at zero has proved
/// nothing — it compared the candle path to itself. Every test that asserts the
/// fused path is correct also asserts these moved, and the fused path is the
/// only thing that touches them.
static FUSED_QUANT_CALLS: AtomicU64 = AtomicU64::new(0);
static FUSED_DEQUANT_CALLS: AtomicU64 = AtomicU64::new(0);

/// Number of `quantize_k` calls served by the fused device kernel.
pub(crate) fn fused_quantize_calls() -> u64 {
    FUSED_QUANT_CALLS.load(Ordering::Relaxed)
}

/// Number of `V4PackedK::dequant` calls served by the fused device kernel.
pub(crate) fn fused_dequantize_calls() -> u64 {
    FUSED_DEQUANT_CALLS.load(Ordering::Relaxed)
}

/// The 256 E4M3 values, indexed by code byte. Entry `i` is exactly what
/// candle's `F8E4M3 -> F32` cast yields for the byte `i`, which is what makes
/// [`V4PackedK::dequant`] bit-exact against the pre-packing implementation.
fn e4m3_values() -> &'static [f32; 256] {
    static VALUES: OnceLock<[f32; 256]> = OnceLock::new();
    VALUES.get_or_init(|| {
        let mut v = [0f32; 256];
        for code in 0..=u8::MAX {
            v[usize::from(code)] = F8E4M3::from_bits(code).to_f32();
        }
        v
    })
}

/// Per-device cached decode table. Rebuilding it per layer per step would be a
/// 1 KB host->device copy 43 times a token (CLAUDE.md pitfall #5); it is a
/// constant, so it is built once per device.
fn e4m3_lut(device: &Device) -> Result<Tensor> {
    static CACHE: OnceLock<RwLock<Vec<(Device, Tensor)>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| RwLock::new(Vec::new()));
    if let Ok(guard) = cache.read() {
        for (dev, lut) in guard.iter() {
            if dev.same_device(device) {
                return Ok(lut.clone());
            }
        }
    }
    let lut = Tensor::from_slice(e4m3_values().as_slice(), 256, device)?;
    if let Ok(mut guard) = cache.write() {
        if !guard.iter().any(|(dev, _)| dev.same_device(device)) {
            guard.push((device.clone(), lut.clone()));
        }
    }
    Ok(lut)
}

/// One layer's cached keys in the packed layout.
///
/// `codes` and `side` are the two halves of a [`crate::kv_cache::KvCache`]
/// `Normal` slot, so every cache-manager path (batch clone-in/clone-out,
/// truncation, prefix reuse) keeps working untouched: both halves stay
/// populated and length-synchronised, they simply have different widths and
/// dtypes.
#[derive(Debug, Clone)]
pub(crate) struct V4PackedK {
    /// `[B, H, T, nope]` U8 — one E4M3 code byte per non-rope dim.
    pub(crate) codes: Tensor,
    /// `[B, H, T, rope_dim + n_blocks]` in the activation dtype:
    /// `[0, rope_dim)` is the RoPE'd tail verbatim, the rest is each 64-wide
    /// block's `amax`.
    pub(crate) side: Tensor,
    /// Width of the RoPE'd tail kept at activation precision.
    pub(crate) rope_dim: usize,
}

impl V4PackedK {
    /// Number of cached tokens.
    pub(crate) fn seq_len(&self) -> Result<usize> {
        self.codes.dim(2)
    }

    /// Restrict to `[base, base + len)` along the sequence dim *without*
    /// reconstructing anything. This is the point of the packed layout: decode
    /// reconstructs only the sliding window it can actually reach, so the
    /// dequant cost is `O(window)` per step, not `O(context)`.
    pub(crate) fn narrow(&self, base: usize, len: usize) -> Result<Self> {
        Ok(Self {
            codes: self.codes.narrow(2, base, len)?,
            side: self.side.narrow(2, base, len)?,
            rope_dim: self.rope_dim,
        })
    }

    /// Rebuild the `[B, H, T, head_dim]` key tensor in `out_dtype`.
    ///
    /// Bit-identical to what the pre-packing `act_quant_kv_nope` stored — see
    /// the module docs and `kv_fp8_roundtrip_is_bit_exact_vs_reference`.
    ///
    /// Served by the fused device kernel when it can be (default), and by the
    /// candle op chain otherwise. The two produce the same bits, so the choice
    /// is invisible to callers; it is exposed to tests through
    /// [`Self::dequant_with`] so parity can be checked *between* them.
    pub(crate) fn dequant(&self, out_dtype: DType) -> Result<Tensor> {
        self.dequant_with(KvQuantMode::from_env(), out_dtype)
    }

    /// [`Self::dequant`] with the path chosen explicitly.
    pub(crate) fn dequant_with(&self, mode: KvQuantMode, out_dtype: DType) -> Result<Tensor> {
        if mode == KvQuantMode::FusedDevice
            && out_dtype == self.side.dtype()
            && mistralrs_quant::arc_kvquant::kv_fp8_fused_available(self.codes.device())
        {
            let nope = self.codes.dim(D::Minus1)?;
            let lut = e4m3_lut(self.codes.device())?;
            let out = mistralrs_quant::arc_kvquant::kv_fp8_dequantize(
                &self.codes,
                &self.side,
                &lut,
                self.rope_dim,
                KV_QUANT_BLOCK,
            )?;
            debug_assert_eq!(out.dim(D::Minus1)?, nope + self.rope_dim);
            FUSED_DEQUANT_CALLS.fetch_add(1, Ordering::Relaxed);
            return Ok(out);
        }
        self.dequant_candle(out_dtype)
    }

    /// The candle op chain: ~13 launches, and the reference the fused kernel is
    /// pinned against. Kept because it is the only implementation that runs on
    /// CPU, and because a fused kernel with nothing to compare to is not a
    /// verified kernel.
    fn dequant_candle(&self, out_dtype: DType) -> Result<Tensor> {
        let (b, h, t, nope) = self.codes.dims4()?;
        let n_blocks = nope / KV_QUANT_BLOCK;
        let dev = self.codes.device();

        // Codes -> exact E4M3 values, via the 256-entry table.
        let lut = e4m3_lut(dev)?;
        let idx = self.codes.flatten_all()?.to_dtype(DType::U32)?;
        let rt = lut
            .index_select(&idx, 0)?
            .reshape((b, h, t, n_blocks, KV_QUANT_BLOCK))?;

        // amax -> scale, with the *same two ops* the quantizer used.
        let amax = self
            .side
            .narrow(D::Minus1, self.rope_dim, n_blocks)?
            .to_dtype(DType::F32)?
            .reshape((b, h, t, n_blocks, 1))?;
        let scale = block_scale(&amax)?;

        let k_nope = rt
            .broadcast_mul(&scale)?
            .reshape((b, h, t, nope))?
            .to_dtype(out_dtype)?;
        let k_rope = self
            .side
            .narrow(D::Minus1, 0, self.rope_dim)?
            .to_dtype(out_dtype)?;
        Tensor::cat(&[&k_nope, &k_rope], D::Minus1)?.contiguous()
    }
}

/// `amax -> scale`, isolated so the quantizer and the dequantizer provably
/// share it. Two candle ops, in this order, on an F32 tensor: changing either
/// breaks bit-exactness with what is already stored.
fn block_scale(amax: &Tensor) -> Result<Tensor> {
    (amax / E4M3_MAX)?.affine(1.0, 1e-12)
}

/// Block-wise E4M3 quantization of K's non-rope dims — the FP8-QAT round trip
/// V4 was trained with (reference `inference/model.py:506`).
///
/// Returns `None` when the geometry has no non-rope dims or they do not divide
/// into whole 64-wide blocks; the caller then keeps `k` unquantized, exactly as
/// the pre-packing implementation did.
pub(crate) fn quantize_k(
    k: &Tensor,
    rope_dim: usize,
    mode: KvQuantMode,
) -> Result<Option<V4PackedK>> {
    let (b, h, t, head_dim) = k.dims4()?;
    let nope = head_dim - rope_dim;
    if nope == 0 || !nope.is_multiple_of(KV_QUANT_BLOCK) {
        return Ok(None);
    }
    let n_blocks = nope / KV_QUANT_BLOCK;

    // The whole point of this module's rewrite: one launch, no device→host
    // round trip, and therefore nothing that blocks CUDA graph capture. Falls
    // through to the candle chain on CPU (and on any dtype the kernel does not
    // carry), which is what the parity tests compare against.
    if mode == KvQuantMode::FusedDevice
        && mistralrs_quant::arc_kvquant::kv_fp8_fused_available(k.device())
        && matches!(k.dtype(), DType::BF16 | DType::F16 | DType::F32)
    {
        let (codes, side) =
            mistralrs_quant::arc_kvquant::kv_fp8_quantize(k, rope_dim, KV_QUANT_BLOCK)?;
        FUSED_QUANT_CALLS.fetch_add(1, Ordering::Relaxed);
        return Ok(Some(V4PackedK {
            codes,
            side,
            rope_dim,
        }));
    }

    let k_nope = k.narrow(D::Minus1, 0, nope)?;
    let k_rope = k.narrow(D::Minus1, nope, rope_dim)?;

    let kb = k_nope
        .to_dtype(DType::F32)?
        .reshape((b, h, t, n_blocks, KV_QUANT_BLOCK))?;
    let amax = kb.abs()?.max_keepdim(D::Minus1)?;
    let scale = block_scale(&amax)?;
    let scaled = kb.broadcast_div(&scale)?;

    let codes = match mode {
        // `FusedDevice` only reaches here when the kernel could not serve the
        // call (CPU tensor, or a dtype it does not carry). The exact path is
        // the right fallback: it is what `FusedDevice` is bit-identical to.
        KvQuantMode::FusedDevice | KvQuantMode::CpuExact => e4m3_codes_cpu(&scaled)?,
        KvQuantMode::GpuApprox => e4m3_codes_arith(&scaled)?,
    }
    .reshape((b, h, t, nope))?;

    // `amax` is one of the block's own elements, so it is already exactly
    // representable in the activation dtype — this cast loses nothing, which is
    // what lets `dequant` rebuild the identical scale.
    let amax_stored = amax.reshape((b, h, t, n_blocks))?.to_dtype(k.dtype())?;
    let side = Tensor::cat(&[&k_rope, &amax_stored], D::Minus1)?.contiguous()?;

    Ok(Some(V4PackedK {
        codes,
        side,
        rope_dim,
    }))
}

/// Exact E4M3 codes via candle's CPU cast. Flattened to a `[N]` U8 tensor on
/// `scaled`'s device.
///
/// This keeps the device sync the pre-packing path already paid (the cast has
/// no CUDA kernel), and trades an F32 round trip for a U8 one — a quarter of
/// the bytes over the bus.
fn e4m3_codes_cpu(scaled: &Tensor) -> Result<Tensor> {
    let dev = scaled.device().clone();
    let cpu = scaled
        .flatten_all()?
        .to_device(&Device::Cpu)?
        .to_dtype(DType::F8E4M3)?;
    let bits: Vec<u8> = cpu
        .to_vec1::<F8E4M3>()?
        .into_iter()
        .map(|v| v.to_bits())
        .collect();
    let n = bits.len();
    Tensor::from_vec(bits, n, &Device::Cpu)?.to_device(&dev)
}

/// E4M3 codes from on-device float arithmetic (no sync).
///
/// For `x` clamped to `+-448`, with `e = clamp(floor(log2|x|), -6, 8)` and
/// `step = 2^(e-3)`, the rounded magnitude `m = round(|x| / step)` gives the
/// code `(e + 6) * 8 + m`, saturating at `0x7E` (`0x7F` is NaN). That single
/// expression covers subnormals (`e` clamped to `-6`, `m` in `[0, 7]`),
/// normals (`m` in `[8, 15]`) and mantissa carry (`m == 16`, which lands
/// exactly on the next exponent's code), because an E4M3 value is always
/// `m * 2^(e-3)`.
fn e4m3_codes_arith(scaled: &Tensor) -> Result<Tensor> {
    let ln2 = std::f64::consts::LN_2;
    let x = scaled.flatten_all()?.clamp(-448f32, 448f32)?;
    let ax = x.abs()?.clamp(1e-30f32, 1e30f32)?;
    let e = ax
        .log()?
        .affine(1.0 / ln2, 0.0)?
        .floor()?
        .clamp(-6f32, 8f32)?;
    let step = e.affine(ln2, -3.0 * ln2)?.exp()?;
    let m = ax.broadcast_div(&step)?.round()?;
    let magnitude = (e.affine(8.0, 48.0)? + m)?.clamp(0f32, 126f32)?;
    let sign = x.lt(0f32)?.to_dtype(DType::F32)?.affine(128.0, 0.0)?;
    (magnitude + sign)?.to_dtype(DType::U8)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use half::bf16;

    /// **Verbatim copy of the pre-packing `act_quant_kv_nope`** (deepseek4.rs
    /// before wave43). The bit-exactness claim is a claim about *this* function,
    /// so it lives here rather than being paraphrased.
    fn reference_act_quant_kv_nope(k: &Tensor, rope_dim: usize, gpu_path: bool) -> Result<Tensor> {
        let head_dim = k.dim(D::Minus1)?;
        let nope = head_dim - rope_dim;
        const BLOCK: usize = 64;
        if nope == 0 || nope % BLOCK != 0 {
            return Ok(k.clone());
        }
        let k_nope = k.narrow(D::Minus1, 0, nope)?;
        let k_rope = k.narrow(D::Minus1, nope, rope_dim)?;
        let orig_dims = k_nope.dims().to_vec();
        let mut blk = orig_dims.clone();
        let last = blk.len() - 1;
        blk[last] = nope / BLOCK;
        blk.push(BLOCK);
        let kb = k_nope.to_dtype(DType::F32)?.reshape(blk)?;
        let amax = kb.abs()?.max_keepdim(D::Minus1)?;
        let scale = (amax / 448.0)?.affine(1.0, 1e-12)?;
        let scaled = kb.broadcast_div(&scale)?;
        let dev = scaled.device().clone();
        let rt = if gpu_path {
            let ln2 = std::f64::consts::LN_2;
            let x = scaled.clamp(-448f32, 448f32)?;
            let ax = x.abs()?.clamp(1e-30f32, 1e30f32)?;
            let e = ax
                .log()?
                .affine(1.0 / ln2, 0.0)?
                .floor()?
                .clamp(-6f32, 8f32)?;
            let step = e.affine(ln2, -3.0 * ln2)?.exp()?;
            x.broadcast_div(&step)?.round()?.broadcast_mul(&step)?
        } else {
            scaled
                .to_device(&Device::Cpu)?
                .to_dtype(DType::F8E4M3)?
                .to_dtype(DType::F32)?
                .to_device(&dev)?
        };
        let q = rt.broadcast_mul(&scale)?;
        let k_nope_q = q.reshape(orig_dims)?.to_dtype(k.dtype())?;
        Tensor::cat(&[&k_nope_q, &k_rope], D::Minus1)?.contiguous()
    }

    /// A K tensor with V4-Flash geometry and enough dynamic range across and
    /// within blocks that both E4M3 subnormals and the saturating top of the
    /// range are exercised.
    fn fixture(b: usize, t: usize, head_dim: usize) -> Result<Tensor> {
        let dev = Device::Cpu;
        let n = b * t * head_dim;
        let vals: Vec<f32> = (0..n)
            .map(|i| {
                let i = i as f32;
                // Per-64-block magnitude swing of ~2^10 so scales differ block
                // to block, plus a sign flip and a within-block ramp.
                let block = ((i as usize / 64) % 11) as f32;
                let sign = if (i as usize) % 7 == 0 { -1.0 } else { 1.0 };
                sign * (2f32.powf(block - 5.0)) * (0.03 + (i % 64.0) / 64.0)
            })
            .collect();
        Tensor::from_vec(vals, (b, 1, t, head_dim), &dev)?.to_dtype(DType::BF16)
    }

    fn bits(t: &Tensor) -> Result<Vec<u16>> {
        Ok(t.flatten_all()?
            .to_vec1::<bf16>()?
            .into_iter()
            .map(|v| v.to_bits())
            .collect())
    }

    /// **The bit-exactness proof.** Packing and reconstructing reproduces the
    /// pre-packing tensor bit pattern for bit pattern, on both quantizer
    /// arithmetics — so moving the KV cache to the packed layout stores exactly
    /// what it stored before, in fewer bytes.
    #[test]
    fn kv_fp8_roundtrip_is_bit_exact_vs_reference() -> Result<()> {
        const HEAD_DIM: usize = 512;
        const ROPE: usize = 64;
        let k = fixture(2, 9, HEAD_DIM)?;

        for (mode, gpu_path) in [
            (KvQuantMode::CpuExact, false),
            (KvQuantMode::GpuApprox, true),
        ] {
            let reference = reference_act_quant_kv_nope(&k, ROPE, gpu_path)?;
            let packed = quantize_k(&k, ROPE, mode)?.expect("448 nope dims are 7 whole blocks");
            let rebuilt = packed.dequant(k.dtype())?;

            assert_eq!(rebuilt.dims(), reference.dims());
            assert_eq!(
                bits(&rebuilt)?,
                bits(&reference)?,
                "{mode:?}: packed round trip must be bit-identical to the stored-BF16 round trip"
            );
        }
        Ok(())
    }

    /// Fixture discrimination (DOCTRINE D12): the equality above is only
    /// evidence if the tensor it compares is not trivially reproducible. Assert
    /// that quantization actually *changes* the input (so an identity `dequant`
    /// would fail), that more than one distinct code is used (so a constant
    /// `dequant` would fail), and that the blocks carry different scales (so a
    /// single-global-scale `dequant` would fail).
    #[test]
    fn kv_fp8_fixture_discriminates() -> Result<()> {
        const HEAD_DIM: usize = 512;
        const ROPE: usize = 64;
        let k = fixture(2, 9, HEAD_DIM)?;
        let packed = quantize_k(&k, ROPE, KvQuantMode::CpuExact)?.unwrap();

        assert_ne!(
            bits(&packed.dequant(k.dtype())?)?,
            bits(&k)?,
            "fixture cannot discriminate: quantization is a no-op on it, so an \
             identity dequant would pass the bit-exactness test"
        );

        let codes: Vec<u8> = packed.codes.flatten_all()?.to_vec1()?;
        let mut distinct: Vec<u8> = codes.clone();
        distinct.sort_unstable();
        distinct.dedup();
        assert!(
            distinct.len() > 32,
            "fixture cannot discriminate: only {} distinct E4M3 codes used",
            distinct.len()
        );

        let n_blocks = (HEAD_DIM - ROPE) / KV_QUANT_BLOCK;
        let amax: Vec<f32> = packed
            .side
            .narrow(D::Minus1, ROPE, n_blocks)?
            .flatten_all()?
            .to_dtype(DType::F32)?
            .to_vec1()?;
        let mut scales = amax.clone();
        scales.sort_by(|a, b| a.partial_cmp(b).unwrap());
        scales.dedup();
        assert!(
            scales.len() > 4,
            "fixture cannot discriminate: {} distinct block scales, a single \
             global scale would be indistinguishable",
            scales.len()
        );
        Ok(())
    }

    /// Narrowing before reconstructing must give the same rows as
    /// reconstructing and then narrowing — this is what makes the decode-time
    /// dequant `O(window)` rather than `O(context)`.
    #[test]
    fn kv_fp8_narrow_then_dequant_matches_dequant_then_narrow() -> Result<()> {
        const ROPE: usize = 64;
        let k = fixture(2, 17, 512)?;
        let packed = quantize_k(&k, ROPE, KvQuantMode::CpuExact)?.unwrap();

        let full = packed.dequant(k.dtype())?;
        for (base, len) in [(0usize, 17usize), (5, 4), (16, 1), (9, 8)] {
            let narrowed = packed.narrow(base, len)?.dequant(k.dtype())?;
            assert_eq!(
                bits(&narrowed)?,
                bits(&full.narrow(2, base, len)?.contiguous()?)?,
                "narrow({base}, {len}) then dequant must equal dequant then narrow"
            );
        }
        Ok(())
    }

    /// Geometries with no whole 64-wide nope blocks decline quantization, and
    /// the caller then stores `k` untouched — the fallback every synthetic
    /// fixture in `deepseek4` relies on, and the reason the reference
    /// implementation short-circuits on them too.
    #[test]
    fn kv_fp8_declines_geometries_without_whole_blocks() -> Result<()> {
        let k = fixture(1, 3, 96)?;
        assert!(quantize_k(&k, 64, KvQuantMode::CpuExact)?.is_none());
        assert!(quantize_k(&k, 96, KvQuantMode::CpuExact)?.is_none());
        assert_eq!(
            bits(&reference_act_quant_kv_nope(&k, 64, false)?)?,
            bits(&k)?,
            "declining and passing through must agree"
        );
        Ok(())
    }

    /// **The fused-kernel bit-parity proof, on hardware.**
    ///
    /// D14: this test is meaningless without a GPU, so it exits 2 (environment
    /// failure) rather than passing when there is no CUDA device — a green run
    /// with no device would be exactly the silent success this project has been
    /// bitten by 13 times.
    ///
    /// D33: three independent quantities have to agree, and the mutant below
    /// proves the comparison can fail.
    ///
    /// 1. `codes`/`side` bytes: fused kernel vs the candle chain + CPU cast.
    /// 2. The reconstructed key tensor, computed **crosswise** — fused-quant
    ///    then candle-dequant against candle-quant then fused-dequant — so a
    ///    matching error would have to be present in both halves of two
    ///    different implementations.
    /// 3. The full round trip against `reference_act_quant_kv_nope`, the
    ///    verbatim pre-packing implementation, run entirely on CPU.
    ///
    /// Plus the engagement assertion D18 demands: the fused call counters must
    /// move, and must move only for the fused arm. A parity check on this exact
    /// subsystem has already passed vacuously by comparing an implementation to
    /// itself; only a launch counter caught it.
    #[cfg(feature = "cuda")]
    #[test]
    fn kv_fp8_fused_is_bit_identical_to_cpu_exact() -> Result<()> {
        const HEAD_DIM: usize = 512;
        const ROPE: usize = 64;

        let dev = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("ENVFAIL: kv_fp8 fused parity needs a CUDA device: {e}");
                std::process::exit(2);
            }
        };

        let k_cpu = fixture(2, 9, HEAD_DIM)?;
        let k = k_cpu.to_device(&dev)?;

        let q0 = fused_quantize_calls();
        let packed_cpu = quantize_k(&k, ROPE, KvQuantMode::CpuExact)?.unwrap();
        assert_eq!(
            fused_quantize_calls(),
            q0,
            "CpuExact must not reach the fused kernel — if it does, the two arms \
             are the same code and this test proves nothing"
        );
        let packed_fused = quantize_k(&k, ROPE, KvQuantMode::FusedDevice)?.unwrap();
        assert_eq!(
            fused_quantize_calls(),
            q0 + 1,
            "the fused arm did not reach the kernel: it silently fell back, so \
             any equality below is the candle path compared with itself"
        );

        // (1) The stored bytes.
        let codes_cpu: Vec<u8> = packed_cpu.codes.flatten_all()?.to_vec1()?;
        let codes_fused: Vec<u8> = packed_fused.codes.flatten_all()?.to_vec1()?;
        assert_eq!(
            codes_fused, codes_cpu,
            "fused E4M3 codes differ from candle's CPU cast"
        );
        assert_eq!(
            bits(&packed_fused.side.to_device(&Device::Cpu)?)?,
            bits(&packed_cpu.side.to_device(&Device::Cpu)?)?,
            "fused `side` (rope tail + block amax) differs from the candle chain"
        );

        // (2) Crosswise reconstruction: each half of one implementation against
        // the other half of the other.
        let d0 = fused_dequantize_calls();
        let fused_then_candle = packed_fused.dequant_candle(DType::BF16)?;
        assert_eq!(
            fused_dequantize_calls(),
            d0,
            "dequant_candle must not reach the fused kernel"
        );
        let candle_then_fused = packed_cpu.dequant_with(KvQuantMode::FusedDevice, DType::BF16)?;
        assert_eq!(
            fused_dequantize_calls(),
            d0 + 1,
            "the fused dequant silently fell back — nothing was tested"
        );
        assert_eq!(
            bits(&candle_then_fused.to_device(&Device::Cpu)?)?,
            bits(&fused_then_candle.to_device(&Device::Cpu)?)?,
            "fused dequant differs from the candle dequant chain"
        );

        // (3) Against the verbatim pre-packing implementation, on CPU.
        let reference = reference_act_quant_kv_nope(&k_cpu, ROPE, false)?;
        assert_eq!(
            bits(&candle_then_fused.to_device(&Device::Cpu)?)?,
            bits(&reference)?,
            "the fused round trip does not reproduce what the pre-packing \
             implementation stored"
        );

        // D33: the comparison above must be able to fail. Same kernel, same
        // scale, same layout, same reduction — only round-to-nearest-even
        // replaced by truncation.
        let (mutant_codes, mutant_side) =
            mistralrs_quant::arc_kvquant::kv_fp8_quantize_mutant_for_test(
                &k.contiguous()?,
                ROPE,
                KV_QUANT_BLOCK,
            )?;
        let mutant_codes: Vec<u8> = mutant_codes.flatten_all()?.to_vec1()?;
        assert_eq!(
            bits(&mutant_side.to_device(&Device::Cpu)?)?,
            bits(&packed_cpu.side.to_device(&Device::Cpu)?)?,
            "the mutant must differ ONLY in the rounding — if `side` differs too \
             it is not isolating the property under test"
        );
        assert_ne!(
            mutant_codes, codes_cpu,
            "the negative control produced identical codes: this parity test \
             cannot fail and therefore proves nothing (D33)"
        );
        let differing = mutant_codes
            .iter()
            .zip(codes_cpu.iter())
            .filter(|(a, b)| a != b)
            .count();
        assert!(
            differing * 20 > codes_cpu.len(),
            "the negative control only perturbed {differing}/{} codes; a \
             comparison that weak could pass on luck",
            codes_cpu.len()
        );
        Ok(())
    }

    /// The decode table must be the E4M3 grid itself, not an approximation of
    /// it: this is the single assumption `dequant`'s exactness rests on.
    #[test]
    fn e4m3_table_is_the_cast() {
        let values = e4m3_values();
        for code in 0u16..256 {
            let v = F8E4M3::from_bits(code as u8);
            if v.to_f32().is_nan() {
                assert!(values[code as usize].is_nan());
            } else {
                assert_eq!(values[code as usize].to_bits(), v.to_f32().to_bits());
            }
        }
        assert_eq!(values[0], 0.0);
        assert_eq!(values[126], 448.0);
    }

    /// The parse table, pinned. Before this, `ARC_KV_FP8_MODE` had a bare
    /// `_ => Self::FusedDevice` catch-all preceded by an
    /// `_ if ARC_GPU_ACT_QUANT is set => Self::GpuApprox` arm, so a MISSPELLED
    /// value on a box that still had `ARC_GPU_ACT_QUANT` exported silently
    /// selected `GpuApprox` — the one variant this module documents as not
    /// bit-exact and "must not be shipped as one".
    #[test]
    fn kv_fp8_impl_parse_table() {
        use KvQuantMode::*;

        // Unset: the bit-exact fused kernel, with and without trailing space.
        assert_eq!(KvQuantMode::parse_impl(None, false).0, FusedDevice);
        assert_eq!(KvQuantMode::parse_impl(Some(""), false).0, FusedDevice);
        assert_eq!(KvQuantMode::parse_impl(Some("  "), false).0, FusedDevice);

        // Every legal value round-trips, case- and whitespace-insensitively,
        // and none of them complains.
        for (raw, want) in [
            ("fused", FusedDevice),
            ("fused_device", FusedDevice),
            ("FUSED", FusedDevice),
            (" cpu ", CpuExact),
            ("cpu_exact", CpuExact),
            ("gpu", GpuApprox),
            ("gpu_approx", GpuApprox),
        ] {
            let (got, complaint) = KvQuantMode::parse_impl(Some(raw), false);
            assert_eq!(got, want, "{raw:?}");
            assert!(complaint.is_none(), "{raw:?} should parse silently");
        }

        // Every value the error message advertises as legal really is legal.
        for v in KV_FP8_IMPL_VALUES {
            assert!(
                KvQuantMode::parse_impl(Some(v), false).1.is_none(),
                "{v:?} is advertised as legal but is rejected"
            );
        }

        // The legacy var still selects GpuApprox, but only when the new flag
        // is silent.
        assert_eq!(KvQuantMode::parse_impl(None, true).0, GpuApprox);
        assert_eq!(KvQuantMode::parse_impl(Some(""), true).0, GpuApprox);
        assert_eq!(KvQuantMode::parse_impl(Some("cpu"), true).0, CpuExact);

        // THE REGRESSION. A typo must be loud, and must NOT reach GpuApprox
        // even with the legacy var set.
        for legacy in [false, true] {
            let (got, complaint) = KvQuantMode::parse_impl(Some("fusd"), legacy);
            assert_eq!(got, FusedDevice, "typo must land on the bit-exact default");
            let c = complaint.expect("a typo must produce a complaint");
            assert!(c.contains("fusd"), "the complaint must quote the bad value");
            assert!(
                c.contains("ARC_V4_FP8_KV"),
                "the complaint must name the flag that actually gates storage"
            );
        }
    }
}
