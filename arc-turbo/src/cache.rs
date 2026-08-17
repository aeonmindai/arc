use candle_core::{DType, Device, Result, Tensor};
use mistralrs_quant::turboquant::{TurboQuantConfig, TurboQuantLayout};

/// TurboQuant-compressed KV cache for a single key or value stream.
///
/// Stores packed quantized indices + norms for compressed tokens, and keeps
/// the most recent `fp16_window` tokens in full precision.
///
/// # Head dimensions
///
/// Any width in `[MIN_BLOCK_DIM, MAX_BLOCK_DIM]`, powers of two or not — the
/// [`TurboQuantLayout`] splits non-powers-of-two into power-of-two rotation
/// blocks. K and V may have **different** widths, which is why each holds its
/// own layout rather than sharing one taken from a single `head_dim`.
///
/// # Cost model — read this before enabling it on a GPU model
///
/// [`Self::current_data`] reconstructs **every compressed token** on the host
/// and ships the result back to the device on every call, i.e. once per layer
/// per decode step. That is `O(compressed_len × head_dim)` of host work and a
/// full host→device transfer per step, and it is why this path is opt-in
/// rather than the default: it trades memory for a decode-time cost that no
/// measurement has shown to be worth paying.
///
/// The paged path does not have this problem — `turbo_paged_attention.cu`
/// consumes the packed blocks directly and never materialises the dequantized
/// cache. `V4CachedK::span` in `mistralrs-core/src/models/deepseek4.rs` shows
/// the shape of the fix for this path too (reconstruct only the window the
/// step actually reads), and it needs a device-side dequant kernel to be worth
/// doing.
#[derive(Debug, Clone)]
pub struct TurboQuantSingleCache {
    /// Dimension index for the sequence length (typically 2 for [batch, heads, seq, dim]).
    dim: usize,
    /// Packed layout for this stream: block plan, signs, codebooks, byte offsets.
    layout: TurboQuantLayout,

    /// Packed quantized indices for compressed tokens, `layout.packed_bytes`
    /// per (batch, head, token), in that iteration order.
    packed_data: Vec<u8>,
    /// Block norms for compressed tokens, `layout.num_norms()` per vector.
    norms: Vec<f32>,

    /// Full-precision recent tokens (the FP16 window).
    /// This is a standard Candle tensor with shape [batch, heads, window_len, head_dim].
    recent_data: Option<Tensor>,

    /// Number of compressed tokens stored so far.
    compressed_seq_len: usize,
    /// Maximum tokens to keep in full precision (the FP16 window).
    fp16_window: usize,
    /// Total number of tokens seen (compressed + recent).
    total_seq_len: usize,

    /// Number of attention heads (needed for reshaping).
    num_heads: usize,
    /// Batch size (typically 1 for normal cache).
    batch_size: usize,
}

impl TurboQuantSingleCache {
    pub fn new(dim: usize, layout: TurboQuantLayout, fp16_window: usize) -> Self {
        Self {
            dim,
            layout,
            packed_data: Vec::new(),
            norms: Vec::new(),
            recent_data: None,
            compressed_seq_len: 0,
            fp16_window,
            total_seq_len: 0,
            num_heads: 0,
            batch_size: 0,
        }
    }

    /// Logical head dimension this stream compresses.
    pub fn head_dim(&self) -> usize {
        self.layout.dim
    }

    /// Tokens that have actually been compressed (i.e. left the FP16 window).
    ///
    /// Tests must assert this is non-zero, otherwise they are exercising a
    /// plain FP16 buffer and proving nothing about the codec.
    pub fn compressed_seq_len(&self) -> usize {
        self.compressed_seq_len
    }

    /// Bytes held for compressed tokens, packed indices plus norms.
    pub fn compressed_bytes(&self) -> usize {
        self.packed_data.len() + self.norms.len() * std::mem::size_of::<f32>()
    }

    /// Append new tokens to the cache.
    ///
    /// Input tensor shape: [batch, heads, new_seq_len, head_dim]
    /// The dim parameter indicates which dimension is the sequence dimension.
    ///
    /// New tokens go into the FP16 window. When the window exceeds `fp16_window`,
    /// the oldest tokens are compressed and moved to the packed store.
    pub fn append(&mut self, src: &Tensor) -> Result<()> {
        let src = src.contiguous()?;
        let shape = src.dims();
        if shape.len() != 4 {
            candle_core::bail!(
                "TurboQuantCache::append expects a rank-4 [B, H, T, D] tensor, got {shape:?}"
            );
        }
        if shape[3] != self.layout.dim {
            candle_core::bail!(
                "TurboQuantCache::append head_dim mismatch: cache was built for {}, \
                 got {} (shape {shape:?})",
                self.layout.dim,
                shape[3]
            );
        }
        if self.num_heads == 0 {
            // First append: learn the batch/head geometry.
            self.batch_size = shape[0];
            self.num_heads = shape[1];
        } else if shape[0] != self.batch_size || shape[1] != self.num_heads {
            candle_core::bail!(
                "TurboQuantCache::append geometry changed: cache holds [B={}, H={}], \
                 got {shape:?}",
                self.batch_size,
                self.num_heads
            );
        }

        let new_seq_len = shape[self.dim];

        // Concatenate with existing recent data
        let combined = if let Some(ref recent) = self.recent_data {
            Tensor::cat(&[recent, &src], self.dim)?
        } else {
            src.clone()
        };

        let combined_seq_len = combined.dim(self.dim)?;

        // If combined exceeds the FP16 window, compress the overflow
        if combined_seq_len > self.fp16_window {
            let to_compress = combined_seq_len - self.fp16_window;

            // Extract tokens to compress: [batch, heads, to_compress, head_dim]
            let compress_slice = combined.narrow(self.dim, 0, to_compress)?;
            self.compress_tokens(&compress_slice)?;

            // Keep the rest as the new FP16 window
            let keep_slice = combined
                .narrow(self.dim, to_compress, self.fp16_window)?
                .contiguous()?;
            self.recent_data = Some(keep_slice);
        } else {
            self.recent_data = Some(combined);
        }

        self.total_seq_len += new_seq_len;
        Ok(())
    }

    /// Compress tokens from FP16 into the packed store.
    fn compress_tokens(&mut self, tokens: &Tensor) -> Result<()> {
        let tokens = tokens.to_dtype(DType::F32)?.contiguous()?;
        let shape = tokens.dims(); // [batch, heads, seq, head_dim]
        let seq_len = shape[self.dim];
        let data = tokens.flatten_all()?.to_vec1::<f32>()?;

        let head_dim = self.layout.dim;
        let bytes_per_vec = self.layout.packed_bytes;
        let norms_per_vec = self.layout.num_norms();

        let vectors = self.batch_size * self.num_heads * seq_len;
        let base_packed = self.packed_data.len();
        let base_norms = self.norms.len();
        self.packed_data
            .resize(base_packed + vectors * bytes_per_vec, 0);
        self.norms.resize(base_norms + vectors * norms_per_vec, 0.0);

        for v in 0..vectors {
            let src = &data[v * head_dim..(v + 1) * head_dim];
            let p0 = base_packed + v * bytes_per_vec;
            let n0 = base_norms + v * norms_per_vec;
            self.layout.quantize_into(
                src,
                &mut self.packed_data[p0..p0 + bytes_per_vec],
                &mut self.norms[n0..n0 + norms_per_vec],
            );
        }

        self.compressed_seq_len += seq_len;
        Ok(())
    }

    /// Get the current full sequence data, dequantizing compressed tokens.
    ///
    /// Returns the full KV tensor: [batch, heads, total_seq_len, head_dim]
    pub fn current_data(&self) -> Result<Option<Tensor>> {
        if self.total_seq_len == 0 {
            return Ok(None);
        }

        let device = self
            .recent_data
            .as_ref()
            .map(|t| t.device().clone())
            .unwrap_or(Device::Cpu);
        let dtype = self
            .recent_data
            .as_ref()
            .map(|t| t.dtype())
            .unwrap_or(DType::F32);

        let mut parts = Vec::new();

        // Dequantize compressed tokens if any
        if self.compressed_seq_len > 0 {
            let decompressed = self.decompress_all(&device, dtype)?;
            parts.push(decompressed);
        }

        // Add recent FP16 window
        if let Some(ref recent) = self.recent_data {
            parts.push(recent.clone());
        }

        if parts.is_empty() {
            return Ok(None);
        }

        let result = if parts.len() == 1 {
            parts.into_iter().next().unwrap()
        } else {
            Tensor::cat(&parts, self.dim)?
        };

        Ok(Some(result))
    }

    /// Dequantize all compressed tokens into a tensor.
    fn decompress_all(&self, device: &Device, dtype: DType) -> Result<Tensor> {
        let head_dim = self.layout.dim;
        let bytes_per_vec = self.layout.packed_bytes;
        let norms_per_vec = self.layout.num_norms();
        let total_vectors = self.batch_size * self.num_heads * self.compressed_seq_len;

        let mut all_data = vec![0f32; total_vectors * head_dim];
        for v in 0..total_vectors {
            let p0 = v * bytes_per_vec;
            let n0 = v * norms_per_vec;
            self.layout.dequantize_into(
                &self.packed_data[p0..p0 + bytes_per_vec],
                &self.norms[n0..n0 + norms_per_vec],
                &mut all_data[v * head_dim..(v + 1) * head_dim],
            );
        }

        // Reshape to [batch, heads, compressed_seq_len, head_dim]
        let tensor = Tensor::from_vec(
            all_data,
            &[
                self.batch_size,
                self.num_heads,
                self.compressed_seq_len,
                head_dim,
            ],
            device,
        )?
        .to_dtype(dtype)?;

        Ok(tensor)
    }

    pub fn current_seq_len(&self) -> usize {
        self.total_seq_len
    }

    pub fn reset(&mut self) {
        self.packed_data.clear();
        self.norms.clear();
        self.recent_data = None;
        self.compressed_seq_len = 0;
        self.total_seq_len = 0;
        self.num_heads = 0;
        self.batch_size = 0;
    }

    pub fn set_len(&mut self, len: usize) -> candle_core::Result<()> {
        if len == 0 {
            self.reset();
            return Ok(());
        }
        if len > self.total_seq_len {
            candle_core::bail!(
                "TurboQuantCache: cannot set len {len} > current {}",
                self.total_seq_len
            );
        }
        if len < self.total_seq_len {
            // Truncation would have to un-compress and re-pack, and every
            // caller that needs it (prefix cacher, MTP rollback, speculative
            // rejection) is a correctness path. Refuse loudly rather than
            // silently keeping stale tokens.
            candle_core::bail!(
                "TurboQuantCache: truncation to {len} not supported (current {}). \
                 Disable TurboQuant KV for workloads that roll the cache back.",
                self.total_seq_len
            );
        }
        Ok(())
    }
}

/// A paired TurboQuant cache for both keys and values.
#[derive(Debug, Clone)]
pub struct TurboQuantCache {
    pub k: TurboQuantSingleCache,
    pub v: TurboQuantSingleCache,
}

impl TurboQuantCache {
    /// Build a cache for `config`, or explain why the geometry is unsupported.
    pub fn try_new(config: &TurboQuantConfig) -> std::result::Result<Self, String> {
        Ok(Self {
            k: TurboQuantSingleCache::new(2, config.key_layout()?, config.fp16_window),
            v: TurboQuantSingleCache::new(2, config.value_layout()?, config.fp16_window),
        })
    }

    /// Panicking [`Self::try_new`]. Kept for call sites that have already
    /// validated the config.
    pub fn new(config: &TurboQuantConfig) -> Self {
        Self::try_new(config).expect("TurboQuantCache::new on an unsupported geometry")
    }

    pub fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        let k_contig = k.contiguous()?;
        let v_contig = v.contiguous()?;

        self.k.append(&k_contig)?;
        self.v.append(&v_contig)?;

        // Return dequantized full tensors for attention
        let k_out = match self.k.current_data()? {
            Some(t) => t,
            None => {
                let mut shape = k_contig.dims().to_vec();
                shape[2] = 0;
                Tensor::zeros(shape, k_contig.dtype(), k_contig.device())?
            }
        };
        let v_out = match self.v.current_data()? {
            Some(t) => t,
            None => {
                let mut shape = v_contig.dims().to_vec();
                shape[2] = 0;
                Tensor::zeros(shape, v_contig.dtype(), v_contig.device())?
            }
        };

        Ok((k_out, v_out))
    }

    pub fn current_seq_len(&self) -> usize {
        self.k.current_seq_len()
    }

    pub fn reset(&mut self) {
        self.k.reset();
        self.v.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::IndexOp;
    use mistralrs_quant::turboquant::TurboQuantPreset;

    const HEADS: usize = 2;

    fn cfg(head_dim: usize, v_head_dim: usize, window: usize) -> TurboQuantConfig {
        TurboQuantConfig::try_new(head_dim, v_head_dim)
            .unwrap()
            .with_fp16_window(window)
    }

    /// Deterministic, outlier-bearing activations. A constant or symmetric
    /// fixture cannot disagree with a broken rotation, which is the whole
    /// failure mode these tests exist to catch.
    fn activations(b: usize, h: usize, t: usize, d: usize, seed: u64) -> Tensor {
        let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15) | 1;
        let n = b * h * t * d;
        let data: Vec<f32> = (0..n)
            .map(|i| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                let u = ((s >> 40) as f32) / (1u32 << 24) as f32 - 0.5;
                if i % 53 == 0 {
                    u * 9.0
                } else {
                    u
                }
            })
            .collect();
        Tensor::from_vec(data, &[b, h, t, d], &Device::Cpu).unwrap()
    }

    fn rel_err(a: &Tensor, b: &Tensor) -> f32 {
        let a: Vec<f32> = a.flatten_all().unwrap().to_vec1().unwrap();
        let b: Vec<f32> = b.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(a.len(), b.len());
        let num: f32 = a.iter().zip(&b).map(|(x, y)| (x - y) * (x - y)).sum();
        let den: f32 = a.iter().map(|x| x * x).sum::<f32>().max(1e-20);
        (num / den).sqrt()
    }

    /// Prefill-then-decode against a real cache, at every head dim the wild
    /// actually uses — including the two the tree refused (192 as a
    /// non-power-of-two, 512 as V4's).
    ///
    /// The window is deliberately small so tokens genuinely leave it. The
    /// `compressed_seq_len > 0` assertion is the D12 guard: with the shipped
    /// default window of 128 and a short test, nothing would be compressed and
    /// this test would pass against a cache that never called the codec.
    #[test]
    fn prefill_then_decode_roundtrips_at_every_head_dim() {
        for head_dim in [64usize, 80, 96, 112, 128, 192, 256, 512] {
            let mut cache = TurboQuantCache::try_new(&cfg(head_dim, head_dim, 4)).unwrap();
            let prefill = activations(1, HEADS, 12, head_dim, head_dim as u64);
            let (k_out, _) = cache.append(&prefill, &prefill).unwrap();

            assert_eq!(k_out.dims(), &[1, HEADS, 12, head_dim]);
            assert!(
                cache.k.compressed_seq_len() >= 8,
                "head_dim {head_dim}: only {} tokens compressed — the fixture \
                 never left the FP16 window, so this test proves nothing",
                cache.k.compressed_seq_len()
            );

            // The reconstruction must be close to the input but NOT equal to
            // it: equality would mean the codec was bypassed.
            let e = rel_err(&prefill, &k_out);
            assert!(e < 0.30, "head_dim {head_dim}: prefill rel err {e}");
            assert!(
                e > 1e-4,
                "head_dim {head_dim}: rel err {e} — nothing was quantized"
            );

            // The FP16 window must be exact, which is only true if the window
            // bookkeeping is right.
            let win = k_out.i((.., .., 8.., ..)).unwrap();
            let win_ref = prefill.i((.., .., 8.., ..)).unwrap();
            assert!(
                rel_err(&win_ref, &win) < 1e-6,
                "head_dim {head_dim}: the last {} tokens should be untouched FP16",
                4
            );

            // Decode steps.
            for step in 0..6 {
                let tok = activations(1, HEADS, 1, head_dim, 900 + step);
                let (k_out, v_out) = cache.append(&tok, &tok).unwrap();
                let want = 13 + step as usize;
                assert_eq!(k_out.dims(), &[1, HEADS, want, head_dim]);
                assert_eq!(v_out.dims(), &[1, HEADS, want, head_dim]);
                assert_eq!(cache.current_seq_len(), want);
                // The newest token is inside the window, so it must come back
                // bit-exact — this catches an off-by-one in the window split.
                let last = k_out.i((.., .., want - 1.., ..)).unwrap();
                assert!(
                    rel_err(&tok, &last) < 1e-6,
                    "head_dim {head_dim} step {step}"
                );
            }
        }
    }

    /// K and V at different widths — the case `cache_engine.rs` refuses and
    /// MLA-style models need.
    #[test]
    fn k_and_v_head_dims_may_differ() {
        let mut cache = TurboQuantCache::try_new(&cfg(128, 64, 2)).unwrap();
        let k = activations(1, HEADS, 9, 128, 11);
        let v = activations(1, HEADS, 9, 64, 12);
        let (k_out, v_out) = cache.append(&k, &v).unwrap();
        assert_eq!(k_out.dims(), &[1, HEADS, 9, 128]);
        assert_eq!(v_out.dims(), &[1, HEADS, 9, 64]);
        assert!(cache.k.compressed_seq_len() > 0 && cache.v.compressed_seq_len() > 0);
        assert!(rel_err(&k, &k_out) < 0.30);
        assert!(rel_err(&v, &v_out) < 0.40); // V is 3-bit
        assert_eq!(cache.k.head_dim(), 128);
        assert_eq!(cache.v.head_dim(), 64);
    }

    /// MUTATION GUARD — feeding the wrong head dim must be refused by name,
    /// not silently reinterpreted.
    #[test]
    fn head_dim_mismatch_is_refused() {
        let mut cache = TurboQuantCache::try_new(&cfg(128, 128, 4)).unwrap();
        let wrong = activations(1, HEADS, 4, 64, 3);
        let err = cache.k.append(&wrong).unwrap_err().to_string();
        assert!(
            err.contains("head_dim mismatch") && err.contains("128"),
            "unhelpful error: {err}"
        );
    }

    /// MUTATION GUARD — a head dim TurboQuant cannot serve must fail at
    /// construction with a message that names the bound, so a model config
    /// cannot silently fall into a degraded codec.
    #[test]
    fn unsupported_head_dims_fail_at_construction() {
        for d in [0usize, 1, 4, 7] {
            let err = TurboQuantConfig::try_new(d, d).unwrap_err();
            assert!(err.contains("TurboQuant"), "unhelpful error for {d}: {err}");
        }
        assert!(TurboQuantConfig::try_new(512, 512).is_ok());
        assert!(TurboQuantConfig::try_new(80, 80).is_ok());
    }

    /// Truncation is a correctness path (prefix cache, MTP rollback,
    /// speculative rejection). It must refuse, not silently keep stale tokens.
    #[test]
    fn truncation_refuses_loudly() {
        let mut cache = TurboQuantCache::try_new(&cfg(64, 64, 4)).unwrap();
        let t = activations(1, HEADS, 10, 64, 5);
        cache.append(&t, &t).unwrap();
        let err = cache.k.set_len(6).unwrap_err().to_string();
        assert!(err.contains("truncation"), "unhelpful error: {err}");
        // set_len to the current length is a no-op and must succeed.
        cache.k.set_len(10).unwrap();
        // reset() is the supported way back to zero.
        cache.reset();
        assert_eq!(cache.current_seq_len(), 0);
        assert_eq!(cache.k.compressed_seq_len(), 0);
    }

    /// The point of the exercise: compressed tokens must actually occupy less
    /// memory, at every preset, measured from the buffers the cache holds.
    #[test]
    fn compressed_bytes_beat_fp16_at_every_preset() {
        for preset in [
            TurboQuantPreset::Default,
            TurboQuantPreset::Balanced,
            TurboQuantPreset::Aggressive,
        ] {
            for head_dim in [64usize, 80, 128, 192, 512] {
                let c = cfg(head_dim, head_dim, 0).with_preset(preset);
                let mut cache = TurboQuantCache::try_new(&c).unwrap();
                let t = activations(1, HEADS, 32, head_dim, 77);
                cache.append(&t, &t).unwrap();
                assert_eq!(cache.k.compressed_seq_len(), 32);

                let vectors = HEADS * 32;
                let fp16 = vectors * head_dim * 2 * 2; // K + V
                let got = cache.k.compressed_bytes() + cache.v.compressed_bytes();
                // Norms are f32 in this cache; the ratio the paper quotes uses
                // f16 norms, so this is the conservative measurement.
                assert!(
                    (fp16 as f32 / got as f32) > 1.5,
                    "preset {preset:?} head_dim {head_dim}: only {:.2}x ({got} vs {fp16})",
                    fp16 as f32 / got as f32
                );
            }
        }
    }
}
