use candle_core::{DType, Device, Result, Tensor};
use mistralrs_quant::turboquant::{
    codebook::Codebook, pack_indices, packed_size, unpack_indices, wht, TurboQuantConfig,
};

/// TurboQuant-compressed KV cache for a single key or value stream.
///
/// Stores packed quantized indices + fp16 norms for compressed tokens,
/// and keeps the most recent `fp16_window` tokens in full precision.
///
/// During attention, `current_data()` returns fully dequantized tensors.
/// This is the naive (non-fused) path — the fused attention kernel in
/// `pagedattention.cuh`/`.metal` operates directly on compressed data.
#[derive(Debug, Clone)]
pub struct TurboQuantSingleCache {
    /// Dimension index for the sequence length (typically 2 for [batch, heads, seq, dim]).
    dim: usize,
    /// Head dimension (64, 128, or 256).
    head_dim: usize,
    /// Codebook for this cache (key or value).
    codebook: Codebook,
    /// Bit-width (2, 3, or 4).
    bits: u32,
    /// Pre-computed WHT sign vector.
    signs: Vec<f32>,

    /// Packed quantized indices for compressed tokens.
    /// Shape conceptually: [batch, heads, compressed_seq_len, packed_bytes_per_head]
    /// Stored as a flat Vec<u8> for simplicity; actual tensor integration comes with fused kernels.
    packed_data: Vec<u8>,
    /// L2 norms for compressed tokens, one per token per head.
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
    pub fn new(
        dim: usize,
        head_dim: usize,
        codebook: Codebook,
        bits: u32,
        signs: Vec<f32>,
        fp16_window: usize,
    ) -> Self {
        Self {
            dim,
            head_dim,
            codebook,
            bits,
            signs,
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

        // Infer layout from shape
        if self.num_heads == 0 {
            // First append: learn the shape
            self.batch_size = shape[0];
            self.num_heads = shape[1];
            assert_eq!(
                shape[3], self.head_dim,
                "Head dim mismatch: expected {}, got {}",
                self.head_dim, shape[3]
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
            let keep_slice = combined.narrow(self.dim, to_compress, self.fp16_window)?;
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

        let bytes_per_head = packed_size(self.head_dim, self.bits);

        for b in 0..self.batch_size {
            for h in 0..self.num_heads {
                for s in 0..seq_len {
                    let offset =
                        ((b * self.num_heads + h) * seq_len + s) * self.head_dim;
                    let vector = &data[offset..offset + self.head_dim];

                    // Quantize using TurboQuant algorithm
                    let (indices, norm) =
                        wht::quantize_vector(vector, &self.signs, &self.codebook);

                    // Pack indices
                    let mut packed = vec![0u8; bytes_per_head];
                    pack_indices(&indices, self.bits, &mut packed);

                    self.packed_data.extend_from_slice(&packed);
                    self.norms.push(norm);
                }
            }
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
        let bytes_per_head = packed_size(self.head_dim, self.bits);
        let total_vectors = self.batch_size * self.num_heads * self.compressed_seq_len;

        let mut all_data = Vec::with_capacity(total_vectors * self.head_dim);

        for i in 0..total_vectors {
            let packed_offset = i * bytes_per_head;
            let packed = &self.packed_data[packed_offset..packed_offset + bytes_per_head];
            let norm = self.norms[i];

            // Unpack indices
            let mut indices = vec![0u8; self.head_dim];
            unpack_indices(packed, self.bits, self.head_dim, &mut indices);

            // Dequantize
            let vector =
                wht::dequantize_vector(&indices, norm, &self.signs, &self.codebook);
            all_data.extend_from_slice(&vector);
        }

        // Reshape to [batch, heads, compressed_seq_len, head_dim]
        let tensor = Tensor::from_vec(
            all_data,
            &[
                self.batch_size,
                self.num_heads,
                self.compressed_seq_len,
                self.head_dim,
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
                "TurboQuantCache: cannot set len {len} > current {}", self.total_seq_len
            );
        }
        if len < self.total_seq_len {
            // Truncation: for now, reset and note this is a limitation
            // Full implementation would selectively truncate compressed + recent
            candle_core::bail!(
                "TurboQuantCache: truncation to {len} not yet supported (current {})",
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
    pub fn new(config: &TurboQuantConfig) -> Self {
        let k_codebook = config.key_codebook();
        let v_codebook = config.value_codebook();
        let signs = wht::generate_signs(config.seed, config.head_dim);

        Self {
            k: TurboQuantSingleCache::new(
                2, // seq dim
                config.head_dim,
                k_codebook,
                config.preset.key_bits(),
                signs.clone(),
                config.fp16_window,
            ),
            v: TurboQuantSingleCache::new(
                2, // seq dim
                config.head_dim,
                v_codebook,
                config.preset.value_bits(),
                signs,
                config.fp16_window,
            ),
        }
    }

    pub fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        let k_contig = k.contiguous()?;
        let v_contig = v.contiguous()?;

        self.k.append(&k_contig)?;
        self.v.append(&v_contig)?;

        // Return dequantized full tensors for attention
        let k_out = self.k.current_data()?.unwrap_or_else(|| {
            let mut shape = k_contig.dims().to_vec();
            shape[2] = 0;
            Tensor::zeros(shape, k_contig.dtype(), k_contig.device()).unwrap()
        });
        let v_out = self.v.current_data()?.unwrap_or_else(|| {
            let mut shape = v_contig.dims().to_vec();
            shape[2] = 0;
            Tensor::zeros(shape, v_contig.dtype(), v_contig.device()).unwrap()
        });

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
