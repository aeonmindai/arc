//! DeepSeek V4 hybrid attention: CSA (Compressed Sparse Attention) + HCA (Heavily
//! Compressed Attention) + sliding-window local branch.
//!
//! Paper:    `research/11_models/deepseek_v3.pdf` (V3+) and V4 tech report on HuggingFace
//! Repo (canonical reference):
//!   `research/code/06_foundation/sglang/python/sglang/srt/layers/attention/dsv4/`
//!
//! ## Architecture (per V4 tech report)
//!
//! V4 interleaves three attention modes across layers:
//!
//! - **Standard (compress_ratio = 0):** plain MLA attention — used in dense layers
//! - **CSA (compress_ratio = 4):** compress KV by 4× via learned compressor,
//!   then top-1024 sparse selection over compressed entries.
//! - **HCA (compress_ratio = 128):** compress KV by 128×, run dense attention
//!   over the heavily compressed sequence (no top-k needed since the compressed
//!   sequence is already small).
//!
//! Both CSA and HCA pair with a sliding-window local-attention branch (default
//! ~512 tokens) so recent context is always attended to at full resolution.
//!
//! Result per V4: 73% FLOPs reduction vs V3.2; only 10% of V3.2's KV cache at 1M context.
//!
//! ## Tier A scope (this module)
//!
//! - `Compressor`: applies a learned linear (W_compress, b_compress) that reduces
//!   the sequence dim from T to T/ratio. Tests verify shape + numerical sanity.
//! - `top_k_select_indices`: scores queries against compressed entries and returns
//!   the indices of the top-k highest-scoring entries.
//! - `csa_attention`: compress → score → top-k → sparse attention.
//! - `hca_attention`: heavy-compress → dense attention.
//! - `sliding_window_attention`: local-only attention over the most recent N tokens.
//! - `dsv4_attention`: top-level dispatch that combines the right branches for
//!   a given `compress_ratio`.
//!
//! ## Tier B (deferred to RUN-136)
//!
//! - TileLang or Triton/CUDA fused kernel matching SGLang's `tilelang_kernel.py`
//! - FP8 KV cache integration (Arc has TurboQuant K4/V3 — different scheme)
//! - Distributed (CP / EP) parallelism
//! - Real 1M-context throughput vs V3.2 baseline

use candle_core::{DType, Device, Result, Tensor, D};

/// Compression ratios supported by V4: 0 = no compression (standard MLA),
/// 4 = CSA, 128 = HCA.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressRatio {
    Standard,
    Csa,
    Hca,
}

impl CompressRatio {
    /// Numeric compression factor.
    pub fn factor(self) -> usize {
        match self {
            CompressRatio::Standard => 1,
            CompressRatio::Csa => 4,
            CompressRatio::Hca => 128,
        }
    }

    /// Parse from the integer used in V4's per-layer `compress_ratios` config field.
    pub fn from_i32(v: i32) -> Result<Self> {
        match v {
            0 => Ok(CompressRatio::Standard),
            4 => Ok(CompressRatio::Csa),
            128 => Ok(CompressRatio::Hca),
            other => candle_core::bail!(
                "V4 compress_ratio must be 0, 4, or 128; got {other}"
            ),
        }
    }
}

/// Configuration for V4 hybrid attention at one layer.
#[derive(Debug, Clone, Copy)]
pub struct Dsv4Config {
    pub compress_ratio: CompressRatio,
    /// Top-k size for CSA. Default 1024 (per `DeepSeekV4Config.index_topk`).
    pub csa_topk: usize,
    /// Sliding-window size for the local branch. Default 512.
    pub sliding_window: usize,
}

impl Default for Dsv4Config {
    fn default() -> Self {
        Self {
            compress_ratio: CompressRatio::Standard,
            csa_topk: 1024,
            sliding_window: 512,
        }
    }
}

/// Learned KV compressor — reduces the sequence dim by `ratio` via a linear
/// projection that takes `ratio` consecutive token KVs as a flat vector and
/// produces a single compressed entry.
///
/// Input: K tensor of shape `[batch, heads, seq, head_dim]` where seq must be
/// divisible by ratio.
/// Output: compressed K of shape `[batch, heads, seq / ratio, head_dim]`.
///
/// Mathematically (matches SGLang's `dsv4/compressor.py`):
///   c_t[d_out] = sum_{i=0..ratio} sum_{d_in=0..head_dim} W[i*head_dim + d_in, d_out] * K[t*ratio+i, d_in]
///
/// The compressor weight is a `[ratio * head_dim, head_dim]` linear layer. The
/// `Compressor::uniform()` constructor produces a weight matrix that recovers
/// simple averaging behavior for use in tests when no checkpoint weights exist.
#[derive(Debug, Clone)]
pub struct Compressor {
    /// 2D weight matrix of shape `[ratio * head_dim, head_dim]` — a linear
    /// projection from concatenated grouped input to compressed output. At
    /// inference time, these come from the checkpoint
    /// (`model.layers.<i>.self_attn.compressor.weight`).
    pub weights: Tensor,
    /// Compression ratio (4 for CSA, 128 for HCA).
    pub ratio: usize,
    /// Head dimension (the last axis of the input K/V).
    pub head_dim: usize,
}

impl Compressor {
    /// Create a compressor with weights that recover the simple averaging
    /// behavior (uniform 1/ratio across grouped tokens). Used for tests and as
    /// a fallback before checkpoint weights are loaded.
    ///
    /// The weight matrix is `[ratio * head_dim, head_dim]` where block `i` (of
    /// shape `[head_dim, head_dim]`) is `(1/ratio) * I` — i.e., averaging the
    /// d-th input dim of each grouped token to produce the d-th output dim.
    pub fn uniform(ratio: usize, head_dim: usize, device: &Device) -> Result<Self> {
        let mut w = vec![0f32; ratio * head_dim * head_dim];
        let inv = 1.0 / ratio as f32;
        // For each of `ratio` input blocks, place `inv * I_{head_dim}` along the
        // diagonal of the [head_dim, head_dim] slice.
        for i in 0..ratio {
            for d in 0..head_dim {
                let row = i * head_dim + d;
                let col = d;
                w[row * head_dim + col] = inv;
            }
        }
        let weights = Tensor::from_vec(w, (ratio * head_dim, head_dim), device)?;
        Ok(Self {
            weights,
            ratio,
            head_dim,
        })
    }

    /// Create with explicit weights of shape `[ratio * head_dim, head_dim]`.
    pub fn from_weights(weights: Tensor, ratio: usize, head_dim: usize) -> Result<Self> {
        let dims = weights.dims();
        if dims != [ratio * head_dim, head_dim] {
            candle_core::bail!(
                "Compressor weights must be shape [{}, {}], got {:?}",
                ratio * head_dim,
                head_dim,
                dims
            );
        }
        Ok(Self {
            weights,
            ratio,
            head_dim,
        })
    }

    /// Compress a KV tensor along the sequence dimension.
    ///
    /// Input: `[B, H, T, D]` where T % ratio == 0 and D == self.head_dim.
    /// Output: `[B, H, T/ratio, D]`.
    pub fn forward(&self, kv: &Tensor) -> Result<Tensor> {
        let dims = kv.dims();
        if dims.len() != 4 {
            candle_core::bail!("Compressor expects [B, H, T, D], got shape {:?}", dims);
        }
        let (b, h, t, d) = (dims[0], dims[1], dims[2], dims[3]);
        if d != self.head_dim {
            candle_core::bail!(
                "Compressor head_dim mismatch: configured {}, input has {d}",
                self.head_dim
            );
        }
        if t % self.ratio != 0 {
            candle_core::bail!(
                "seq_len {t} not divisible by compress ratio {}",
                self.ratio
            );
        }
        let t_new = t / self.ratio;

        // Reshape: [B, H, T, D] → [B*H*T/r, r*D] (flatten each group to a 2D matrix)
        let kv_flat = kv.reshape((b * h * t_new, self.ratio * d))?;
        // Matmul: [B*H*T/r, r*D] @ [r*D, D] → [B*H*T/r, D]
        let compressed = kv_flat.matmul(&self.weights)?;
        // Reshape back: [B*H*T/r, D] → [B, H, T/r, D]
        compressed.reshape((b, h, t_new, d))
    }
}

/// Score queries against compressed K entries and return top-k indices per query.
///
/// Inputs:
/// - `q`: `[B, H, T_q, D]` — queries
/// - `compressed_k`: `[B, H, T_c, D]` — compressed K entries
///
/// Outputs:
/// - `indices`: `[B, H, T_q, top_k]` — indices of the top-k compressed entries
///   per query (in descending score order).
///
/// Score = dot product of q · compressed_k.
pub fn top_k_select_indices(
    q: &Tensor,
    compressed_k: &Tensor,
    top_k: usize,
) -> Result<Tensor> {
    let q_dims = q.dims();
    let k_dims = compressed_k.dims();
    if q_dims.len() != 4 || k_dims.len() != 4 {
        candle_core::bail!(
            "top_k_select_indices expects rank-4 q and k, got {:?} and {:?}",
            q_dims,
            k_dims
        );
    }
    let (b, h, t_q, d) = (q_dims[0], q_dims[1], q_dims[2], q_dims[3]);
    let (_, _, t_c, _) = (k_dims[0], k_dims[1], k_dims[2], k_dims[3]);
    let top_k = top_k.min(t_c);

    // scores: [B, H, T_q, T_c] = q @ compressed_k.T
    let scores = q.matmul(&compressed_k.transpose(D::Minus2, D::Minus1)?)?;
    let _ = d;

    // For each (b, h, t_q), find the top_k indices along the T_c dim.
    // Candle has no direct top-k; do it on CPU with sort + slice.
    let scores_cpu = scores.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;
    let scores_data: Vec<f32> = scores_cpu.flatten_all()?.to_vec1()?;

    let mut indices_out = vec![0u32; b * h * t_q * top_k];
    for bi in 0..b {
        for hi in 0..h {
            for qi in 0..t_q {
                let row_start = bi * (h * t_q * t_c) + hi * (t_q * t_c) + qi * t_c;
                let row = &scores_data[row_start..row_start + t_c];
                let mut indexed: Vec<(usize, f32)> = row.iter().copied().enumerate().collect();
                indexed.sort_unstable_by(|a, b| {
                    b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                });
                let out_off = bi * (h * t_q * top_k) + hi * (t_q * top_k) + qi * top_k;
                for k in 0..top_k {
                    indices_out[out_off + k] = indexed[k].0 as u32;
                }
            }
        }
    }

    Tensor::from_vec(indices_out, (b, h, t_q, top_k), &Device::Cpu)?
        .to_device(q.device())
}

/// CSA: compress K + V by 4×, select top-k compressed entries per query, run
/// attention over the selected entries.
///
/// Returns attention output of shape `[B, H, T_q, D]`.
pub fn csa_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    compressor_k: &Compressor,
    compressor_v: &Compressor,
    top_k: usize,
) -> Result<Tensor> {
    let q_dims = q.dims();
    let (b, h, t_q, d) = (q_dims[0], q_dims[1], q_dims[2], q_dims[3]);

    // Compress K and V by ratio=4.
    let k_c = compressor_k.forward(k)?;
    let v_c = compressor_v.forward(v)?;

    // Top-k select compressed entries per query.
    let indices = top_k_select_indices(q, &k_c, top_k)?;
    let t_c = k_c.dim(2)?;
    let top_k_actual = top_k.min(t_c);

    // Gather the selected K and V entries.
    let k_selected = gather_selected_entries(&k_c, &indices, b, h, t_q, top_k_actual, d)?;
    let v_selected = gather_selected_entries(&v_c, &indices, b, h, t_q, top_k_actual, d)?;
    // k_selected, v_selected: [B, H, T_q, top_k, D]

    // Per-query attention: q[b,h,t_q,:] vs selected_k[b,h,t_q,:,:] → top_k logits.
    // We compute as element-wise multiplication + sum since each query attends to
    // a different subset of K entries.
    let q_data: Vec<f32> = q.to_dtype(DType::F32)?
        .to_device(&Device::Cpu)?
        .flatten_all()?
        .to_vec1()?;
    let k_sel_data: Vec<f32> = k_selected.to_dtype(DType::F32)?
        .to_device(&Device::Cpu)?
        .flatten_all()?
        .to_vec1()?;
    let v_sel_data: Vec<f32> = v_selected.to_dtype(DType::F32)?
        .to_device(&Device::Cpu)?
        .flatten_all()?
        .to_vec1()?;

    let scale = 1.0_f32 / (d as f32).sqrt();
    let mut out = vec![0f32; b * h * t_q * d];

    for bi in 0..b {
        for hi in 0..h {
            for qi in 0..t_q {
                // Compute scores against the selected entries for this query.
                let mut scores = vec![0f32; top_k_actual];
                for ki in 0..top_k_actual {
                    let mut s = 0f32;
                    for di in 0..d {
                        let q_idx = bi * (h * t_q * d) + hi * (t_q * d) + qi * d + di;
                        let k_idx = bi * (h * t_q * top_k_actual * d)
                            + hi * (t_q * top_k_actual * d)
                            + qi * (top_k_actual * d)
                            + ki * d
                            + di;
                        s += q_data[q_idx] * k_sel_data[k_idx];
                    }
                    scores[ki] = s * scale;
                }
                // Softmax over scores.
                let max_s = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum_exp = 0f32;
                let mut exp_scores = scores.iter().map(|&s| {
                    let e = (s - max_s).exp();
                    sum_exp += e;
                    e
                }).collect::<Vec<_>>();
                if sum_exp > 0.0 {
                    for s in exp_scores.iter_mut() {
                        *s /= sum_exp;
                    }
                }
                // Weighted sum of selected V.
                for di in 0..d {
                    let mut acc = 0f32;
                    for ki in 0..top_k_actual {
                        let v_idx = bi * (h * t_q * top_k_actual * d)
                            + hi * (t_q * top_k_actual * d)
                            + qi * (top_k_actual * d)
                            + ki * d
                            + di;
                        acc += exp_scores[ki] * v_sel_data[v_idx];
                    }
                    let out_idx = bi * (h * t_q * d) + hi * (t_q * d) + qi * d + di;
                    out[out_idx] = acc;
                }
            }
        }
    }

    Tensor::from_vec(out, (b, h, t_q, d), &Device::Cpu)?
        .to_device(q.device())
}

/// HCA: compress K + V by 128×, run dense attention over the compressed sequence.
///
/// Returns `[B, H, T_q, D]`.
pub fn hca_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    compressor_k: &Compressor,
    compressor_v: &Compressor,
) -> Result<Tensor> {
    let k_c = compressor_k.forward(k)?;
    let v_c = compressor_v.forward(v)?;
    let d = q.dim(D::Minus1)?;
    let scale_factor = 1.0_f32 / (d as f32).sqrt();
    // Standard scaled-dot-product over compressed seq.
    let scores = q.matmul(&k_c.transpose(D::Minus2, D::Minus1)?)?;
    let scaled = (scores * scale_factor as f64)?;
    let weights = candle_nn::ops::softmax_last_dim(&scaled)?;
    weights.matmul(&v_c)
}

/// Sliding-window local attention over the most recent `window` tokens.
///
/// Returns `[B, H, T_q, D]`. Queries see only the last `window` K/V entries
/// (causal-style, anchored on the end of the sequence).
pub fn sliding_window_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    window: usize,
) -> Result<Tensor> {
    let k_dims = k.dims();
    let t_k = k_dims[2];
    let window = window.min(t_k);
    let start = t_k - window;
    let k_local = k.narrow(2, start, window)?;
    let v_local = v.narrow(2, start, window)?;
    let d = q.dim(D::Minus1)?;
    let scale = 1.0_f32 / (d as f32).sqrt();
    let scores = q.matmul(&k_local.transpose(D::Minus2, D::Minus1)?)?;
    let scaled = (scores * scale as f64)?;
    let weights = candle_nn::ops::softmax_last_dim(&scaled)?;
    weights.matmul(&v_local)
}

/// Top-level V4 hybrid attention dispatch. Combines compressed-branch output
/// with the sliding-window local branch.
///
/// The mixing weight (paper §3.2 default 0.5) controls the blend; for V4 we
/// follow the published recipe and use a learned scalar at inference time, but
/// for Tier A we use a uniform 0.5 mix.
pub fn dsv4_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    compressor_k: &Compressor,
    compressor_v: &Compressor,
    cfg: Dsv4Config,
) -> Result<Tensor> {
    let main = match cfg.compress_ratio {
        CompressRatio::Standard => {
            // Standard MLA — dense scaled dot-product.
            let d = q.dim(D::Minus1)?;
            let scale = 1.0_f32 / (d as f32).sqrt();
            let scores = q.matmul(&k.transpose(D::Minus2, D::Minus1)?)?;
            let scaled = (scores * scale as f64)?;
            let weights = candle_nn::ops::softmax_last_dim(&scaled)?;
            weights.matmul(v)?
        }
        CompressRatio::Csa => {
            csa_attention(q, k, v, compressor_k, compressor_v, cfg.csa_topk)?
        }
        CompressRatio::Hca => hca_attention(q, k, v, compressor_k, compressor_v)?,
    };

    if cfg.compress_ratio == CompressRatio::Standard {
        return Ok(main);
    }

    // Add sliding-window local branch and blend.
    let local = sliding_window_attention(q, k, v, cfg.sliding_window)?;
    // Uniform 0.5 mix between compressed branch and local window.
    ((main * 0.5)? + (local * 0.5)?)
}

/// Gather selected K (or V) entries by their indices.
///
/// `compressed`: `[B, H, T_c, D]`
/// `indices`:    `[B, H, T_q, top_k]` (each entry is an index into T_c)
/// Returns: `[B, H, T_q, top_k, D]`
fn gather_selected_entries(
    compressed: &Tensor,
    indices: &Tensor,
    b: usize,
    h: usize,
    t_q: usize,
    top_k: usize,
    d: usize,
) -> Result<Tensor> {
    let compressed_cpu = compressed
        .to_dtype(DType::F32)?
        .to_device(&Device::Cpu)?;
    let indices_cpu = indices
        .to_dtype(DType::U32)?
        .to_device(&Device::Cpu)?;
    let comp_data: Vec<f32> = compressed_cpu.flatten_all()?.to_vec1()?;
    let idx_data: Vec<u32> = indices_cpu.flatten_all()?.to_vec1()?;
    let t_c = compressed.dim(2)?;

    let mut out = vec![0f32; b * h * t_q * top_k * d];
    for bi in 0..b {
        for hi in 0..h {
            for qi in 0..t_q {
                for ki in 0..top_k {
                    let sel_idx = idx_data[bi * (h * t_q * top_k)
                        + hi * (t_q * top_k)
                        + qi * top_k
                        + ki] as usize;
                    debug_assert!(sel_idx < t_c, "indexer produced out-of-range index");
                    for di in 0..d {
                        let in_idx = bi * (h * t_c * d) + hi * (t_c * d) + sel_idx * d + di;
                        let out_idx = bi * (h * t_q * top_k * d)
                            + hi * (t_q * top_k * d)
                            + qi * (top_k * d)
                            + ki * d
                            + di;
                        out[out_idx] = comp_data[in_idx];
                    }
                }
            }
        }
    }
    Tensor::from_vec(out, (b, h, t_q, top_k, d), &Device::Cpu)?
        .to_device(compressed.device())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Compressor reduces the sequence dim by the ratio and preserves other dims.
    #[test]
    fn compressor_shape_invariant() -> Result<()> {
        let device = Device::Cpu;
        let b = 2;
        let h = 4;
        let t = 64;
        let d = 16;
        let data: Vec<f32> = (0..(b * h * t * d)).map(|i| (i as f32) * 0.01).collect();
        let kv = Tensor::from_vec(data, (b, h, t, d), &device)?;

        let compressor = Compressor::uniform(4, 16, &device)?;
        let out = compressor.forward(&kv)?;
        assert_eq!(out.dims(), &[b, h, t / 4, d]);
        Ok(())
    }

    /// Uniform compressor at ratio=4 averages 4 consecutive entries.
    #[test]
    fn uniform_compressor_averages_groups() -> Result<()> {
        let device = Device::Cpu;
        // KV: [1, 1, 8, 2] — 8 timesteps, 2-dim features.
        // Token i has values [i, i+0.5]
        let mut data = Vec::with_capacity(8 * 2);
        for i in 0..8 {
            data.push(i as f32);
            data.push(i as f32 + 0.5);
        }
        let kv = Tensor::from_vec(data, (1, 1, 8, 2), &device)?;
        let compressor = Compressor::uniform(4, 2, &device)?;
        let out = compressor.forward(&kv)?;
        let v: Vec<f32> = out.flatten_all()?.to_vec1()?;
        // First group avg: tokens 0..3 → mean of (0,1,2,3) = 1.5; (0.5, 1.5, 2.5, 3.5) = 2.0
        assert!((v[0] - 1.5).abs() < 1e-5, "group 0 dim 0: {} ≠ 1.5", v[0]);
        assert!((v[1] - 2.0).abs() < 1e-5, "group 0 dim 1: {} ≠ 2.0", v[1]);
        // Second group avg: tokens 4..7 → mean of (4,5,6,7) = 5.5; (4.5, 5.5, 6.5, 7.5) = 6.0
        assert!((v[2] - 5.5).abs() < 1e-5);
        assert!((v[3] - 6.0).abs() < 1e-5);
        Ok(())
    }

    /// Top-k selection picks the highest-scoring compressed entries.
    #[test]
    fn top_k_picks_highest_scores() -> Result<()> {
        let device = Device::Cpu;
        // Single query, 1 head, 8 compressed entries, D=4
        // Query points strongly at K entry index 3 (highest dot product)
        let q = Tensor::from_vec(vec![1.0f32, 0.0, 0.0, 0.0], (1, 1, 1, 4), &device)?;
        // 8 K entries with varying magnitudes
        let mut k_data = Vec::with_capacity(8 * 4);
        for i in 0..8 {
            // entry i has first-dim value = (i as f32), rest 0
            // so dot(q, k_i) = i → entry 7 should be top-1, then 6, then 5...
            k_data.push(i as f32);
            k_data.push(0.0);
            k_data.push(0.0);
            k_data.push(0.0);
        }
        let k = Tensor::from_vec(k_data, (1, 1, 8, 4), &device)?;
        let indices = top_k_select_indices(&q, &k, 3)?;
        let v: Vec<u32> = indices.flatten_all()?.to_vec1()?;
        assert_eq!(v, vec![7u32, 6, 5]);
        Ok(())
    }

    /// CSA forward produces output of correct shape on a synthetic input.
    #[test]
    fn csa_forward_shape() -> Result<()> {
        let device = Device::Cpu;
        let b = 1;
        let h = 2;
        let t = 16; // must be divisible by 4
        let d = 8;
        let nq = 4;

        let mk = |seed: f32, len: usize| -> Result<Tensor> {
            let data: Vec<f32> = (0..(b * h * len * d))
                .map(|i| ((i as f32) * seed).sin())
                .collect();
            Tensor::from_vec(data, (b, h, len, d), &device)
        };
        let q = mk(0.1, nq)?;
        let k = mk(0.2, t)?;
        let v = mk(0.3, t)?;

        let comp_k = Compressor::uniform(4, 8, &device)?;
        let comp_v = Compressor::uniform(4, 8, &device)?;
        let out = csa_attention(&q, &k, &v, &comp_k, &comp_v, 2)?;
        assert_eq!(out.dims(), &[b, h, nq, d]);
        // No NaNs / infinities
        let data: Vec<f32> = out.flatten_all()?.to_vec1()?;
        assert!(data.iter().all(|v| v.is_finite()));
        Ok(())
    }

    /// HCA at very large compression still produces finite output.
    #[test]
    fn hca_forward_finite_output() -> Result<()> {
        let device = Device::Cpu;
        let b = 1;
        let h = 1;
        let t = 128; // divisible by 128 → compressed to 1 entry
        let d = 8;

        let mk = |seed: f32, len: usize| -> Result<Tensor> {
            let data: Vec<f32> = (0..(b * h * len * d))
                .map(|i| ((i as f32) * seed).cos())
                .collect();
            Tensor::from_vec(data, (b, h, len, d), &device)
        };
        let q = mk(0.05, 4)?;
        let k = mk(0.15, t)?;
        let v = mk(0.25, t)?;

        let comp_k = Compressor::uniform(128, 8, &device)?;
        let comp_v = Compressor::uniform(128, 8, &device)?;
        let out = hca_attention(&q, &k, &v, &comp_k, &comp_v)?;
        assert_eq!(out.dims(), &[b, h, 4, d]);
        let data: Vec<f32> = out.flatten_all()?.to_vec1()?;
        assert!(data.iter().all(|v| v.is_finite()));
        Ok(())
    }

    /// Sliding window restricts to the last `window` tokens.
    #[test]
    fn sliding_window_uses_recent_tokens() -> Result<()> {
        let device = Device::Cpu;
        // T=10 tokens; window=3 means we attend to tokens 7, 8, 9.
        // K is set so that token i has dim 0 = i. V is identity.
        let b = 1;
        let h = 1;
        let t = 10;
        let d = 4;
        let mut k_data = Vec::with_capacity(t * d);
        let mut v_data = Vec::with_capacity(t * d);
        for i in 0..t {
            k_data.push(i as f32);
            k_data.push(0.0);
            k_data.push(0.0);
            k_data.push(0.0);
            v_data.push(i as f32);
            v_data.push(0.0);
            v_data.push(0.0);
            v_data.push(0.0);
        }
        let k = Tensor::from_vec(k_data, (b, h, t, d), &device)?;
        let v = Tensor::from_vec(v_data, (b, h, t, d), &device)?;
        // Query that strongly prefers token 9 (highest dim 0)
        let q = Tensor::from_vec(vec![10.0f32, 0.0, 0.0, 0.0], (b, h, 1, d), &device)?;
        let out = sliding_window_attention(&q, &k, &v, 3)?;
        let v_data: Vec<f32> = out.flatten_all()?.to_vec1()?;
        // Result should be close to (token 9's value) since the softmax is sharply
        // peaked on the highest dim-0 score (token 9 = 90/sqrt(4) before scaling).
        // Within the window of [7, 8, 9], the weighted sum strongly favors 9.
        assert!(v_data[0] > 8.5, "expected near token 9 (val=9), got {}", v_data[0]);
        Ok(())
    }

    /// dsv4_attention with Standard ratio matches plain scaled-dot-product attention.
    #[test]
    fn standard_dispatch_matches_dense_attention() -> Result<()> {
        let device = Device::Cpu;
        let b = 1;
        let h = 2;
        let t = 8;
        let d = 4;
        let mk = |seed: f32| -> Result<Tensor> {
            let data: Vec<f32> = (0..(b * h * t * d))
                .map(|i| ((i as f32) * seed).sin())
                .collect();
            Tensor::from_vec(data, (b, h, t, d), &device)
        };
        let q = mk(0.1)?;
        let k = mk(0.2)?;
        let v = mk(0.3)?;

        // Reference: standard scaled dot-product.
        let scale = 1.0_f32 / (d as f32).sqrt();
        let scores = q.matmul(&k.transpose(D::Minus2, D::Minus1)?)?;
        let scaled = (scores * scale as f64)?;
        let weights = candle_nn::ops::softmax_last_dim(&scaled)?;
        let expected = weights.matmul(&v)?;

        let comp = Compressor::uniform(4, 4, &device)?;
        let cfg = Dsv4Config {
            compress_ratio: CompressRatio::Standard,
            csa_topk: 2,
            sliding_window: 4,
        };
        let actual = dsv4_attention(&q, &k, &v, &comp, &comp, cfg)?;

        let expected_v: Vec<f32> = expected.flatten_all()?.to_vec1()?;
        let actual_v: Vec<f32> = actual.flatten_all()?.to_vec1()?;
        for (e, a) in expected_v.iter().zip(actual_v.iter()) {
            assert!((e - a).abs() < 1e-5, "{} ≠ {}", e, a);
        }
        Ok(())
    }

    /// With random 2D learned weights, the compressor output is a deterministic
    /// matmul of the grouped input — exercises the full learned path, not just
    /// the uniform shortcut.
    #[test]
    fn learned_compressor_matches_manual_matmul() -> Result<()> {
        let device = Device::Cpu;
        let ratio = 4;
        let head_dim = 8;
        let b = 1;
        let h = 1;
        let t = 16;

        // Build pseudo-random learned weights deterministically.
        let mut w_data = Vec::with_capacity(ratio * head_dim * head_dim);
        for i in 0..(ratio * head_dim * head_dim) {
            w_data.push(((i as f32) * 0.013).sin() * 0.5);
        }
        let w = Tensor::from_vec(w_data.clone(), (ratio * head_dim, head_dim), &device)?;
        let comp = Compressor::from_weights(w, ratio, head_dim)?;

        // Build a deterministic input.
        let kv_data: Vec<f32> = (0..(b * h * t * head_dim))
            .map(|i| ((i as f32) * 0.027).cos())
            .collect();
        let kv = Tensor::from_vec(kv_data.clone(), (b, h, t, head_dim), &device)?;

        let out = comp.forward(&kv)?;
        assert_eq!(out.dims(), &[b, h, t / ratio, head_dim]);

        // Manual reference: flatten groups, matmul, reshape.
        let t_new = t / ratio;
        let mut expected = vec![0f32; b * h * t_new * head_dim];
        for bi in 0..b {
            for hi in 0..h {
                for ti in 0..t_new {
                    for d_out in 0..head_dim {
                        let mut acc = 0f32;
                        for i in 0..ratio {
                            for d_in in 0..head_dim {
                                let in_idx = bi * (h * t * head_dim)
                                    + hi * (t * head_dim)
                                    + (ti * ratio + i) * head_dim
                                    + d_in;
                                let w_idx = (i * head_dim + d_in) * head_dim + d_out;
                                acc += kv_data[in_idx] * w_data[w_idx];
                            }
                        }
                        expected[bi * (h * t_new * head_dim)
                            + hi * (t_new * head_dim)
                            + ti * head_dim
                            + d_out] = acc;
                    }
                }
            }
        }

        let out_data: Vec<f32> = out.flatten_all()?.to_vec1()?;
        for (e, a) in expected.iter().zip(out_data.iter()) {
            assert!(
                (e - a).abs() < 1e-4,
                "Compressor output {a} ≠ manual matmul reference {e}"
            );
        }
        Ok(())
    }

    /// CompressRatio enum parses from the V4 config integer.
    #[test]
    fn compress_ratio_from_i32() {
        assert_eq!(CompressRatio::from_i32(0).unwrap(), CompressRatio::Standard);
        assert_eq!(CompressRatio::from_i32(4).unwrap(), CompressRatio::Csa);
        assert_eq!(CompressRatio::from_i32(128).unwrap(), CompressRatio::Hca);
        assert!(CompressRatio::from_i32(8).is_err());
        assert!(CompressRatio::from_i32(-1).is_err());
    }

    /// Compressor errors clearly when seq_len is not divisible by ratio.
    #[test]
    fn compressor_misaligned_seq_errors() {
        let device = Device::Cpu;
        let data = vec![0f32; 1 * 1 * 7 * 4]; // T=7, ratio=4 → misaligned
        let kv = Tensor::from_vec(data, (1, 1, 7, 4), &device).unwrap();
        let compressor = Compressor::uniform(4, 4, &device).unwrap();
        assert!(compressor.forward(&kv).is_err());
    }

    /// CSA composes with TurboQuant-style (uniform) compressors without crashing on
    /// a large-ish synthetic input — sanity check for memory + iteration patterns.
    #[test]
    fn csa_handles_larger_synthetic_input() -> Result<()> {
        let device = Device::Cpu;
        let b = 1;
        let h = 2;
        let t = 256; // larger seq
        let d = 8;
        let nq = 4;
        let mk = |seed: f32, len: usize| -> Result<Tensor> {
            let data: Vec<f32> = (0..(b * h * len * d))
                .map(|i| ((i as f32) * seed).sin())
                .collect();
            Tensor::from_vec(data, (b, h, len, d), &device)
        };
        let q = mk(0.1, nq)?;
        let k = mk(0.2, t)?;
        let v = mk(0.3, t)?;
        let comp = Compressor::uniform(4, 8, &device)?;
        let out = csa_attention(&q, &k, &v, &comp, &comp, 16)?;
        assert_eq!(out.dims(), &[b, h, nq, d]);
        let data: Vec<f32> = out.flatten_all()?.to_vec1()?;
        assert!(data.iter().all(|v| v.is_finite()));
        Ok(())
    }
}
