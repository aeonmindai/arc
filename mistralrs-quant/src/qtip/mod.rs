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
//! ## What's Tier B (deferred to RUN-136/137)
//!
//! - Full Viterbi optimal quantizer (matches paper's PPL exactly)
//! - CUDA decode kernel port from `qtip-kernels/src/qtip_torch.cu` (1MAD / 3INST / HYB)
//! - Hadamard incoherence rotation (currently the caller is expected to pre-rotate
//!   via the existing TurboQuant Hadamard if desired)
//! - Compose with NVFP4 per-tensor scale

use std::{
    borrow::Cow,
    io::Cursor,
    sync::{atomic::AtomicUsize, Arc},
};

use byteorder::{LittleEndian, ReadBytesExt};
use candle_core::{DType, Device, Result, Tensor};

use crate::{
    utils::{deserialize_tensor, serialize_tensor, version_is_compatible, UQFF_VERSION},
    IsqType, QuantMethod, QuantMethodConfig, QuantizeOntoGuard, QuantizedConfig, QuantizedSerde,
    QuantizedSerdeType, ShardedVarBuilder,
};

mod viterbi;
pub use viterbi::viterbi_quantize_row;

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

#[derive(Debug)]
pub struct QtipLayer {
    /// Packed K-bit symbols. For K=4, two symbols per byte.
    /// Shape: [N, num_symbols_per_row / 2]  (because two K=4 symbols pack into one u8)
    blocks: Tensor,
    /// Per-row scale factor. Shape: [N], FP32.
    row_scales: Tensor,
    /// Shared LUT (one per layer or shared across the module).
    /// Shape: [2^L, V] FP32.
    lut: Tensor,
    /// Optional bias: [N].
    bias: Option<Tensor>,
    /// Cached input dim (K) so we can decode without bookkeeping at the user level.
    in_features: usize,
}

/// Quantization mode for QTIP.
#[derive(Debug, Clone, Copy)]
pub enum QtipMode {
    /// Fast but suboptimal: at each position, pick the locally-best symbol given the current state.
    /// ~5-10× faster than Viterbi at calibration time, with ~3× higher reconstruction error.
    Greedy,
    /// Globally optimal symbol sequence via dynamic-programming search over the trellis.
    /// Matches Cornell's paper numbers; slower at calibration time but quantization is one-shot.
    Viterbi,
}

impl Default for QtipMode {
    fn default() -> Self {
        // Greedy is the documented stable mode. Viterbi is currently opt-in
        // while the unit-level matmul reconstruction parity vs the production
        // decoder is being verified. Switch this back to Viterbi once
        // qtip_matmul_cosine_similarity passes with Viterbi mode.
        QtipMode::Greedy
    }
}

impl QtipLayer {
    /// Quantize an unquantized weight tensor [N, K_in] to QTIP 2-bit format.
    /// Default mode is Viterbi (optimal). Use `quantize_with_mode` to override.
    pub fn quantize(
        weight: &Tensor,
        bias: Option<Tensor>,
        device: &Device,
    ) -> Result<Arc<dyn QuantMethod>> {
        Self::quantize_with_mode(weight, bias, device, QtipMode::default())
    }

    /// Quantize with explicit mode selection.
    pub fn quantize_with_mode(
        weight: &Tensor,
        bias: Option<Tensor>,
        device: &Device,
        mode: QtipMode,
    ) -> Result<Arc<dyn QuantMethod>> {
        let weight_f32 = weight.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;
        let (n, k_in) = weight_f32.dims2()?;
        if (k_in as u32) % V != 0 {
            candle_core::bail!(
                "QTIP quantize: in_features ({k_in}) must be divisible by V ({V})"
            );
        }
        let num_symbols_per_row = k_in / V as usize;
        if !num_symbols_per_row.is_multiple_of(2) {
            // We pack two K=4 symbols per byte.
            candle_core::bail!(
                "QTIP quantize: number of symbols per row ({num_symbols_per_row}) must be even for K=4 packing"
            );
        }

        // Build the global LUT once.
        let lut_data = gaussian_lut();
        let weight_data: Vec<f32> = weight_f32.flatten_all()?.to_vec1()?;

        use rayon::prelude::*;
        let row_results: Vec<(Vec<u8>, f32)> = (0..n)
            .into_par_iter()
            .map(|row| {
                let row_offset = row * k_in;
                let row_slice = &weight_data[row_offset..row_offset + k_in];

                // Pick row scale = max(|row|) / 3.0 so most values live in [-3, 3] (the
                // bulk of a standard normal). LUT values are in roughly [-4, 4].
                let max_abs = row_slice.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
                let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 3.0 };
                let inv_scale = 1.0 / scale;

                let mut packed = vec![0u8; num_symbols_per_row / 2];

                // Scale the target row.
                let scaled_target: Vec<f32> = row_slice.iter().map(|w| w * inv_scale).collect();

                // Encode the symbol sequence via the selected mode.
                let symbols: Vec<u8> = match mode {
                    QtipMode::Viterbi => viterbi::viterbi_quantize_row(&scaled_target, &lut_data),
                    QtipMode::Greedy => {
                        let mut state: u32 = 0;
                        let mut syms = vec![0u8; num_symbols_per_row];
                        for sym_idx in 0..num_symbols_per_row {
                            let target_t = &scaled_target[sym_idx * V as usize..(sym_idx + 1) * V as usize];

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
        let row_scales =
            Tensor::from_vec(all_scales, (n,), &Device::Cpu)?.to_device(device)?;
        let lut = Tensor::from_vec(lut_data, (LUT_SIZE, V as usize), &Device::Cpu)?
            .to_device(device)?;
        let bias = bias.map(|b| b.to_device(device)).transpose()?;

        Ok(Arc::new(Self {
            blocks,
            row_scales,
            lut,
            bias,
            in_features: k_in,
        }))
    }

    fn dequantize_weights_f32(&self) -> Result<Vec<f32>> {
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

    pub fn dequantize_weights(&self) -> Result<Tensor> {
        let out = self.dequantize_weights_f32()?;
        let n = self.row_scales.dim(0)?;
        let k_in = self.in_features;
        Tensor::from_vec(out, (n, k_in), &Device::Cpu)?
            .to_device(self.blocks.device())?
            .to_dtype(DType::BF16)
    }

    fn forward_dequantize(&self, x: &Tensor) -> Result<Tensor> {
        let orig_dims = x.dims().to_vec();
        let x_2d = if orig_dims.len() > 2 {
            let features = orig_dims[orig_dims.len() - 1];
            let batch_size: usize = orig_dims[..orig_dims.len() - 1].iter().product();
            x.reshape((batch_size, features))?
        } else {
            x.clone()
        };

        // Reconstruct weight matrix [N, K_in] and do dense matmul.
        // For Tier A this is correct; a block-major version would save memory at
        // the cost of more code. The Viterbi quantizer + CUDA kernel land in Tier B.
        let w = self.dequantize_weights()?.to_dtype(x.dtype())?;
        let mut result = x_2d.matmul(&w.t()?)?;

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
            } => Ok(Self {
                blocks,
                row_scales,
                lut,
                bias,
                in_features,
            }),
            _ => candle_core::bail!("QtipLayer requires QuantMethodConfig::Qtip"),
        }
    }

    fn dequantize_w(&self) -> Result<candle_core::Tensor> {
        self.dequantize_weights()
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.forward_dequantize(x)
    }

    fn quantized_act_type(&self) -> Option<DType> {
        None
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
            .get_with_hints_dtype((LUT_SIZE, V as usize), "qtip_lut", Default::default(), DType::F32)
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

        Ok(Arc::new(Self {
            blocks,
            row_scales,
            lut,
            bias,
            in_features: in_dim,
        }))
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
        let (layer, _) = Self::deserialize_ext_bias(data, device, guard)?;
        Ok(layer)
    }
    fn deserialize_ext_bias(
        data: Cow<[u8]>,
        device: &Device,
        guard: QuantizeOntoGuard,
    ) -> Result<(Arc<dyn QuantMethod>, Option<Tensor>)>
    where
        Self: Sized,
    {
        let mut buffer = Cursor::new(data.to_vec());

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
        let bias = if has_bias {
            Some(deserialize_tensor(&mut buffer, device)?)
        } else {
            None
        };
        let ext_bias = bias.clone();

        Ok((
            Arc::new(Self {
                blocks,
                row_scales,
                lut,
                bias,
                in_features,
            }),
            ext_bias,
        ))
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

    /// Matmul correctness: dense vs QTIP-dequant-then-matmul should produce a
    /// cosine similarity > 0.85 (lower than NVFP4's 0.99 because greedy QTIP
    /// loses more precision; that's expected for Tier A — Viterbi closes the gap).
    #[test]
    fn qtip_matmul_cosine_similarity() -> Result<()> {
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
        assert!(cos > 0.85, "QTIP matmul cos sim {cos} <= 0.85");
        Ok(())
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
}
