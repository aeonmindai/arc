#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::{f32::consts::PI, ops::Mul, str::FromStr, sync::Arc};

use candle_core::{
    quantized::{QMatMul, QTensor},
    Context, DType, Device, IndexOp, Result, Tensor, D,
};
use candle_nn::{
    BatchNorm, BatchNormConfig, Conv1d, Conv1dConfig, Conv2d, Conv2dConfig, Embedding, GroupNorm,
    LayerNorm, LayerNormConfig, Linear, Module,
};
use float8::F8E4M3;
use half::{bf16, f16};
use mistralrs_quant::{
    AfqLayer, ColumnParallelLayer, Convolution, QuantMethod, QuantizedConfig, RowParallelLayer,
    ShardedVarBuilder,
};
use serde::{Deserialize, Serialize};

pub use crate::attention::Sdpa;
pub use crate::layers_masker::CausalMasker;
pub use crate::layers_utils::repeat_kv;
use crate::{
    amoe::{AnyMoeTrainableLayer, MlpLayer},
    embedding_models::embedding_gemma::EmbeddingGemmaConfig,
    gguf::Content,
    models::{llama, smollm3},
    ops::SplitOp,
    vision_models::{
        gemma3::config::Gemma3TextConfig,
        gemma3n::config::Gemma3nTextConfig,
        llama4,
        mllama::{MLlamaRopeScaling, MLlamaRopeType, MLlamaTextConfig},
        phi4::Phi4MMConfig,
    },
};

pub use mistralrs_quant::MatMul;

pub fn embedding(
    in_size: usize,
    out_size: usize,
    vb: ShardedVarBuilder,
    config: &Option<QuantizedConfig>,
) -> Result<Embedding> {
    // AFQ quantized applies quantization to the embeddings.
    let embeddings = if let Some(QuantizedConfig::Afq { .. }) = config {
        let afq_layer =
            AfqLayer::afq_linear_b(out_size, in_size, config.as_ref().unwrap(), false, vb)?;
        afq_layer.dequantize_w()?
    } else {
        vb.get_with_hints((in_size, out_size), "weight", Default::default())?
    };
    Ok(Embedding::new(embeddings, out_size))
}

pub fn layer_norm<C: Into<LayerNormConfig>>(
    size: usize,
    config: C,
    vb: ShardedVarBuilder,
) -> Result<LayerNorm> {
    let config = config.into();
    let weight = vb.get(size, "weight")?;
    if config.affine {
        let bias = vb.get(size, "bias")?;
        Ok(LayerNorm::new(weight, bias, config.eps))
    } else {
        Ok(LayerNorm::new_no_bias(weight, config.eps))
    }
}

pub fn batch_norm<C: Into<BatchNormConfig>>(
    num_features: usize,
    config: C,
    vb: ShardedVarBuilder,
) -> Result<BatchNorm> {
    let config = config.into();
    if config.eps < 0. {
        candle_core::bail!("batch-norm eps cannot be negative {}", config.eps)
    }
    let running_mean = vb.get(num_features, "running_mean")?;
    let running_var = vb.get(num_features, "running_var")?;

    if config.affine {
        let weight = vb.get(num_features, "weight")?;
        let bias = vb.get(num_features, "bias")?;
        BatchNorm::new(
            num_features,
            running_mean,
            running_var,
            weight,
            bias,
            config.eps,
        )
    } else {
        BatchNorm::new_no_bias(num_features, running_mean, running_var, config.eps)
    }
}

pub fn group_norm(
    num_groups: usize,
    num_channels: usize,
    eps: f64,
    vb: ShardedVarBuilder,
) -> Result<GroupNorm> {
    let weight = vb.get(num_channels, "weight")?;
    let bias = vb.get(num_channels, "bias")?;
    GroupNorm::new(weight, bias, num_channels, num_groups, eps)
}

pub fn conv2d(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    cfg: Conv2dConfig,
    vb: ShardedVarBuilder,
) -> Result<Conv2d> {
    let ws = vb.get(
        (
            out_channels,
            in_channels / cfg.groups,
            kernel_size,
            kernel_size,
        ),
        "weight",
    )?;
    let bs = vb.get(out_channels, "bias")?;
    Ok(Conv2d::new(ws, Some(bs), cfg))
}

pub fn conv2d_no_bias(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    cfg: Conv2dConfig,
    vb: ShardedVarBuilder,
) -> Result<Conv2d> {
    let ws = vb.get(
        (
            out_channels,
            in_channels / cfg.groups,
            kernel_size,
            kernel_size,
        ),
        "weight",
    )?;
    Ok(Conv2d::new(ws, None, cfg))
}

pub fn conv1d(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    cfg: Conv1dConfig,
    vb: ShardedVarBuilder,
) -> Result<Conv1d> {
    let ws = vb.get(
        (out_channels, in_channels / cfg.groups, kernel_size),
        "weight",
    )?;
    let bs = vb.get(out_channels, "bias")?;
    Ok(Conv1d::new(ws, Some(bs), cfg))
}

pub fn conv1d_no_bias(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    cfg: Conv1dConfig,
    vb: ShardedVarBuilder,
) -> Result<Conv1d> {
    let ws = vb.get(
        (out_channels, in_channels / cfg.groups, kernel_size),
        "weight",
    )?;
    Ok(Conv1d::new(ws, None, cfg))
}

pub fn linear(in_dim: usize, out_dim: usize, vb: ShardedVarBuilder) -> Result<Linear> {
    let ws = vb.get((out_dim, in_dim), "weight")?;
    let bs = vb.get(out_dim, "bias")?;
    Ok(Linear::new(ws, Some(bs)))
}

pub fn linear_no_bias(in_dim: usize, out_dim: usize, vb: ShardedVarBuilder) -> Result<Linear> {
    let ws = vb.get((out_dim, in_dim), "weight")?;
    Ok(Linear::new(ws, None))
}

pub fn linear_b(
    in_dim: usize,
    out_dim: usize,
    bias: bool,
    vb: ShardedVarBuilder,
) -> Result<Linear> {
    if bias {
        linear(in_dim, out_dim, vb)
    } else {
        linear_no_bias(in_dim, out_dim, vb)
    }
}

#[derive(Debug, Clone)]
pub struct RmsNorm {
    eps: f64,
    weight: Tensor,
}

impl RmsNorm {
    pub fn new(size: usize, eps: f64, vb: ShardedVarBuilder) -> Result<Self> {
        let w = vb.get(size, "weight")?;
        Ok(Self { eps, weight: w })
    }

    /// Gemma uses weight + 1.0
    #[deprecated(
        note = "Use GemmaRmsNorm::new() instead, which handles UQFF serialization correctly"
    )]
    pub fn new_gemma(size: usize, eps: f64, vb: ShardedVarBuilder) -> Result<Self> {
        let w = vb.get(size, "weight")?;
        let w = (w + 1.0)?;
        Ok(Self { eps, weight: w })
    }

    /// Gemma 3n uses weight
    pub fn new_gemma_3n(
        size: usize,
        eps: f64,
        with_scale: bool,
        vb: ShardedVarBuilder,
    ) -> Result<Self> {
        let w = if with_scale {
            vb.get(size, "weight")?
        } else {
            Tensor::ones(size, vb.dtype(), vb.device())?
        };
        Ok(Self { eps, weight: w })
    }

    /// Gemma uses weight + 1.0. Undo for UQFF generation.
    #[deprecated(note = "Use GemmaRmsNorm instead, which handles UQFF serialization automatically")]
    pub fn undo_gemma(&self) -> Result<Self> {
        Ok(Self {
            eps: self.eps,
            weight: (&self.weight - 1.0)?,
        })
    }

    pub fn from_w(w: Tensor, eps: f64) -> Result<Self> {
        Ok(Self { eps, weight: w })
    }

    pub fn weight(&self) -> &Tensor {
        &self.weight
    }
}

impl Module for RmsNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        candle_nn::ops::rms_norm(&x.contiguous()?, &self.weight, self.eps as f32)
    }
}

/// Gemma-style RmsNorm that adds +1.0 to the weight during initialization.
///
/// Unlike using `RmsNorm::new_gemma()`, this type stores the original checkpoint
/// weight separately, ensuring that UQFF serialization (via `ToTensors`) always
/// returns the un-offset weight. This prevents the double-addition bug where
/// `new_gemma` would add +1.0 on both write and read.
#[derive(Debug, Clone)]
pub struct GemmaRmsNorm {
    eps: f64,
    original_weight: Tensor,
    weight: Tensor,
}

impl GemmaRmsNorm {
    pub fn new(size: usize, eps: f64, vb: ShardedVarBuilder) -> Result<Self> {
        let original_weight = vb.get(size, "weight")?;
        let weight = (&original_weight + 1.0)?;
        Ok(Self {
            eps,
            original_weight,
            weight,
        })
    }

    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    pub fn original_weight(&self) -> &Tensor {
        &self.original_weight
    }
}

impl Module for GemmaRmsNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        candle_nn::ops::rms_norm(&x.contiguous()?, &self.weight, self.eps as f32)
    }
}

#[derive(Debug, Clone)]
pub struct F32RmsNorm {
    w: Tensor,
    eps: f64,
}

impl F32RmsNorm {
    pub fn new(size: usize, eps: f64, vb: ShardedVarBuilder) -> Result<Self> {
        Ok(Self {
            w: vb.get((size,), "weight")?,
            eps,
        })
    }

    pub fn weight(&self) -> &Tensor {
        &self.w
    }
}

impl Module for F32RmsNorm {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let initial_type = xs.dtype();
        let mut xs = xs.to_dtype(DType::F32)?;
        let var = xs.powf(2.)?.mean_keepdim(D::Minus1)?;
        xs = xs.broadcast_mul(&(&var + self.eps)?.recip()?.sqrt()?)?;
        xs.to_dtype(initial_type)?.broadcast_mul(&self.w)
    }
}

#[derive(Debug, Clone)]
pub struct QRmsNorm {
    eps: f64,
    weight: Tensor,
}

impl QRmsNorm {
    pub fn new(scale: QTensor, eps: f32) -> Result<Self> {
        let scale = scale.dequantize(&scale.device())?;
        Ok(Self {
            eps: eps as f64,
            weight: scale,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        candle_nn::ops::rms_norm(&x.contiguous()?, &self.weight, self.eps as f32)
    }
}

/// The offset shared by every row of a decode cohort, or `None` when the rows
/// genuinely differ.
///
/// # Why this is a *value* test and not a length test
///
/// Every rotary implementation in this file gated its batched path on
/// `seqlen_offsets.len() == 1` -- the **length of the vector**, where the
/// property it needs is the **distinctness of the values**. `len()` is the
/// batch size, so the batched path was unreachable at every batch size above
/// one, while the values were uniform anyway: `crate::scheduler` admits a
/// forward pass only when the participating cache lengths are equal. The
/// per-sequence loop was therefore performing B **bit-identical**
/// recomputations of the same `cos`/`sin` rotation and concatenating them back
/// together.
///
/// Returning `Some` is exact rather than approximate. `candle_nn`'s rope
/// kernels take a 2-D `[T, D/2]` cos/sin whose batch stride is 0, so every
/// batch row already reads the same table row, and each output element is an
/// independent two-multiply-one-add of its own inputs. Batching changes which
/// launch computes an element, never the arithmetic that produces it. That is
/// asserted bit-for-bit, not within a tolerance, in `rope_cohort_tests`.
///
/// `None` keeps the old loop verbatim, so ragged cohorts behave exactly as they
/// do today. This is a dispatch fix, not a new ragged path.
///
/// # The V4 rotary, collapsed onto this predicate
///
/// The V4 rotary (`DeepSeekV2RotaryEmbedding`) got this same correction
/// separately, in #198, as a private associated function `uniform_offset` with
/// a byte-identical body. #198 is now merged and that second spelling is gone:
/// `DeepSeekV2RotaryEmbedding::{forward, forward_inverse_tail}` call this
/// function directly, so the two cannot drift apart.
///
/// Two details #198 established that this doc did not previously carry:
///
/// * The uniformity is not an accident of the current scheduler, it is
///   *enforced* by it. `scheduler::default_scheduler::select_running_bucket`
///   admits a forward pass only when cache lengths are exactly equal
///   ("Sequences may only share a forward pass when their cache lengths are
///   exactly equal"), and the single producer of a ragged dense batch is gated
///   behind `ARC_MTP_PER_SEQ_KV`, which defaults off
///   (`models::deepseek4::ragged_row_q0`).
/// * Measured at B=256 on a real decode profile, the per-sequence loop cost
///   ~100,000 GPU launches per step, 19.9% of step time -- of which the
///   `copy2d` half existed only to undo a split the same function had just
///   made. That number was measured on #198's branch, not on this tree.
#[inline]
pub(crate) fn uniform_seqlen_offset(seqlen_offsets: &[usize]) -> Option<usize> {
    match seqlen_offsets {
        [] => None,
        [first, rest @ ..] => rest.iter().all(|o| o == first).then_some(*first),
    }
}

/// Engagement counters for the uniform-offset cohort path -- see
/// [`uniform_seqlen_offset`].
///
/// House rule: "a green result must prove work happened". A fast path that
/// silently never engages produces a perfectly green no-op, so the batched
/// branch counts itself and the per-sequence branch counts itself, and a
/// harness that sees `cohort == 0` must treat that as an environment failure
/// rather than a result.
pub mod rope_cohort_stats {
    use std::sync::atomic::{AtomicU64, Ordering};

    pub(super) static COHORT: AtomicU64 = AtomicU64::new(0);
    pub(super) static PER_SEQUENCE: AtomicU64 = AtomicU64::new(0);

    /// `(cohort, per_sequence)` -- batched RoPE calls vs per-sequence loops.
    ///
    /// Tests assert on *deltas* around a call rather than absolute values, so
    /// no reset hook is needed and the counters stay monotonic for a serving
    /// harness reading them at the end of a run.
    pub fn counts() -> (u64, u64) {
        (
            COHORT.load(Ordering::Relaxed),
            PER_SEQUENCE.load(Ordering::Relaxed),
        )
    }

    #[inline]
    pub(super) fn record_cohort() {
        COHORT.fetch_add(1, Ordering::Relaxed);
        #[cfg(test)]
        bump_local(1, 0);
    }

    #[inline]
    pub(super) fn record_per_sequence() {
        PER_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        #[cfg(test)]
        bump_local(0, 1);
    }

    // ---- Test-only per-thread mirror of the two counters above. -----------
    //
    // The atomics are process-global and, since #198 collapsed its own private
    // copy of this module onto this one, they are written by EVERY rotary in
    // this file -- `RotaryEmbedding`, `Llama3RotaryEmbedding`,
    // `Phi4MMRotaryEmbedding` and `DeepSeekV2RotaryEmbedding` alike. A test
    // that brackets one call with `counts()` is therefore not measuring its own
    // call: `cargo test` runs the suite on many threads, and any other test
    // that drives any rotary lands between the `before` and the `after`.
    //
    // That is not hypothetical. Two rotary forwards in `deepseek4.rs`
    // (`standard_layer_rope_is_unscaled_and_compressed_is_not`,
    // `mtp_block_takes_the_standard_unscaled_rope_table`) race the cohort tests
    // exactly this way -- caught by merging #198 and #121 together and running
    // the full suite, where `--test-threads=1` passes 707 and the parallel run
    // fails one.
    //
    // A mutex was the first fix and it is the wrong one: it obliges every
    // present and future test that touches any rotary to know about a lock in
    // another module, which nothing enforces and the next such test will
    // forget. Counting per-thread instead makes each test's delta its own by
    // construction, needs no cooperation from anyone else, and leaves the
    // serving instrument -- the global `counts()` -- byte-for-byte unchanged.

    #[cfg(test)]
    thread_local! {
        static LOCAL: std::cell::Cell<(u64, u64)> = const { std::cell::Cell::new((0, 0)) };
    }

    #[cfg(test)]
    #[inline]
    fn bump_local(cohort: u64, per_sequence: u64) {
        LOCAL.with(|c| {
            let (a, b) = c.get();
            c.set((a + cohort, b + per_sequence));
        });
    }

    /// `(cohort, per_sequence)` for THIS thread only — the test-facing
    /// counterpart of [`counts`].
    ///
    /// Same monotonic-delta contract, same meaning, but immune to whatever the
    /// rest of the suite is doing on other threads. Tests assert on deltas
    /// around their own call; nothing resets it.
    #[cfg(test)]
    pub fn local_counts() -> (u64, u64) {
        LOCAL.with(|c| c.get())
    }
}

/// `ARC_ROPE_COHORT=1` routes a **ragged** decode cohort — rows at genuinely
/// different positions — through ONE gathered `rope_i` per projection instead
/// of the per-sequence loop. **Default OFF**; unset and every off-spelling
/// keep the loop byte-for-byte as it is today.
///
/// Uniform cohorts are NOT behind this flag: when every row shares an offset,
/// [`uniform_seqlen_offset`] already dispatches one batched `rope_i` with a
/// 2-D table, which is bit-identical to the loop and needs no gather. This
/// flag governs only the arm the uniform fix could not reach — offsets that
/// genuinely differ, where the loop was the only spelling. On the measured
/// B=256 profile that loop is ~9 ops × 43 layers × B ≈ 99,000 launches per
/// step (19.9% of step time); the gathered path is ~5 launches per layer.
///
/// candle's `rope_i` accepts a rank-3 `[B, T, D/2]` cos/sin when `B` matches
/// the input batch (`rope_check_cs`, `candle-nn/src/rotary_emb.rs`; the CUDA
/// wrapper sets `stride_b`, the CPU path indexes `i + b_i * t * d / 2`), so a
/// per-row table is expressible in one launch — the loop was never structural.
///
/// Why opt-in rather than default-ON: the same doctrine as
/// `cuda::qk_norm_rope::fused_cohort_enabled` — unverified means default-off,
/// and unverified means unmeasured on hardware, not new. The A/B box flips
/// this flag; when the gathered arm has been measured correct and faster, the
/// default flips in the same change that records the number.
///
/// Read by VALUE (`mistralrs_quant::env_flag_is_set`), never by presence:
/// `ARC_ROPE_COHORT=0` means OFF (#212 converted 23 flags because
/// `var_os(..).is_some()` made `=0` mean ON). Latched once per process — the
/// dispatch runs on every layer of every step and `std::env::var` takes a
/// global lock. The polarity table lives with `env_flag_value`'s own tests;
/// the two ragged arms themselves are compared bit-for-bit in
/// `deepseek_ragged_cohort_tests`, which drives them directly so the latch
/// cannot make the parity claim untestable.
pub(crate) fn rope_cohort_enabled() -> bool {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| mistralrs_quant::env_flag_is_set("ARC_ROPE_COHORT"))
}

/// Tile a `[max_seq_len, rot_dim]` rotary table into the `[b * seq_len, rot_dim]`
/// layout `mistralrs_quant::rotary::apply_rotary_inplace` requires.
///
/// The CUDA kernel indexes `cos_cache + token_idx * rot_dim` with one block per
/// token of the *flattened* `[b * seq_len, h, d]` tensor, and
/// `mistralrs_quant::rotary` hard-bails unless the cache is exactly
/// `(num_tokens, rot_dim)`. So -- unlike the `candle_nn` paths -- a uniform
/// cohort cannot simply narrow to `[seq_len, rot_dim]`; the rows must actually
/// be repeated, b-major, to match `q.transpose(1, 2)?.flatten(0, 1)?`.
///
/// Uniform cohorts get one broadcast + one copy instead of `b` narrows and a
/// `b`-way `cat`; ragged cohorts keep the `cat`, which is the only thing that
/// can express them.
fn cohort_rotary_table(table: &Tensor, seqlen_offsets: &[usize], seq_len: usize) -> Result<Tensor> {
    match uniform_seqlen_offset(seqlen_offsets) {
        Some(offset) => {
            let rows = table.narrow(0, offset, seq_len)?;
            if seqlen_offsets.len() == 1 {
                return rows.contiguous();
            }
            let rot_dim = rows.dim(D::Minus1)?;
            rows.unsqueeze(0)?
                .broadcast_as((seqlen_offsets.len(), seq_len, rot_dim))?
                .contiguous()?
                .reshape((seqlen_offsets.len() * seq_len, rot_dim))
        }
        None => {
            let mut rows = Vec::with_capacity(seqlen_offsets.len());
            for offset in seqlen_offsets {
                rows.push(table.narrow(0, *offset, seq_len)?);
            }
            Tensor::cat(&rows, 0)
        }
    }
}

/// RoPE supporting LongRope
#[derive(Debug, Clone)]
pub struct PhiRotaryEmbedding {
    short_sin: Tensor,
    short_cos: Tensor,
    long_cos: Option<Tensor>,
    long_sin: Option<Tensor>,
    original_max_position_embeddings: usize,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ScaledRopeType {
    #[serde(alias = "su")]
    #[serde(alias = "longrope")]
    Su,
    #[serde(alias = "yarn")]
    Yarn,
    #[serde(alias = "dynamic")]
    Dynamic,
    #[serde(alias = "linear")]
    Linear,
}

impl FromStr for ScaledRopeType {
    type Err = candle_core::Error;
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "su" | "longrope" => Ok(Self::Su),
            "yarn" => Ok(Self::Yarn),
            "linear" => Ok(Self::Linear),
            "dynamic" => Ok(Self::Dynamic),
            _ => Err(candle_core::Error::Msg(
                "Expected either `su` or `yarn` scaled RoPE type.".to_string(),
            )),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum PhiRopeScalingConfig {
    Classic {
        short_factor: Vec<f64>,
        long_factor: Vec<f64>,
        #[serde(rename = "type")]
        scaling_type: ScaledRopeType,
    },
    Scaled {
        short_factor: Vec<f64>,
        long_factor: Vec<f64>,
        #[serde(rename = "type")]
        scaling_type: ScaledRopeType,
        long_mscale: f64,
        short_mscale: f64,
    },
}

pub struct PhiRopeConfig {
    pub rope_scaling: Option<PhiRopeScalingConfig>,
    pub max_position_embeddings: usize,
    pub original_max_position_embeddings: usize,
    pub rope_theta: f64,
    pub head_dim: usize,
    pub partial_rotary_factor: Option<f64>,
}

impl PhiRotaryEmbedding {
    fn new_classic_scaled(
        short_factor: &[f64],
        long_factor: &[f64],
        scaling_type: &ScaledRopeType,
        cfg: &PhiRopeConfig,
        dtype: DType,
        dev: &Device,
    ) -> Result<Self> {
        let max_seq_len = cfg.max_position_embeddings;
        let dim = (cfg.head_dim as f64 * cfg.partial_rotary_factor.unwrap_or(1.)) as usize;

        // Calculate scale
        let scale =
            cfg.max_position_embeddings as f64 / cfg.original_max_position_embeddings as f64;
        let scaling_factor = if scale <= 1.0 {
            1.0
        } else {
            match scaling_type {
                ScaledRopeType::Su => {
                    (1.0 + scale.ln() / (cfg.original_max_position_embeddings as f64).ln()).sqrt()
                }
                ScaledRopeType::Yarn => 0.1 * scale.ln() + 1.0,
                _ => candle_core::bail!("Expected either `su` or `yarn` RoPE"),
            }
        };

        // Calculate inv freqs for short, long
        let inv_freq_long = (0..dim)
            .step_by(2)
            .enumerate()
            .map(|(k, i)| {
                (1f64 / (long_factor[k] * cfg.rope_theta.powf(i as f64 / dim as f64))) as f32
            })
            .collect::<Vec<_>>();
        let inv_freq_short = (0..dim)
            .step_by(2)
            .enumerate()
            .map(|(k, i)| {
                (1f64 / (short_factor[k] * cfg.rope_theta.powf(i as f64 / dim as f64))) as f32
            })
            .collect::<Vec<_>>();
        let inv_freq_len = inv_freq_long.len();

        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;

        // Calculate sin,cos for long
        let inv_freq_long = Tensor::from_vec(inv_freq_long, (1, inv_freq_len), dev)?;
        let freqs_long = t.matmul(&inv_freq_long)?;
        let long_sin = freqs_long.sin()?.mul(scaling_factor)?.to_dtype(dtype)?;
        let long_cos = freqs_long.cos()?.mul(scaling_factor)?.to_dtype(dtype)?;

        // Calculate sin,cos for short
        let inv_freq_short =
            Tensor::from_vec(inv_freq_short, (1, inv_freq_len), dev)?.to_dtype(DType::F32)?;
        let freqs_short = t.matmul(&inv_freq_short)?;
        let short_sin = freqs_short.sin()?.mul(scaling_factor)?.to_dtype(dtype)?;
        let short_cos = freqs_short.cos()?.mul(scaling_factor)?.to_dtype(dtype)?;

        Ok(Self {
            short_cos,
            short_sin,
            long_cos: Some(long_cos),
            long_sin: Some(long_sin),
            original_max_position_embeddings: cfg.original_max_position_embeddings,
        })
    }

    fn new_unscaled(cfg: &PhiRopeConfig, dtype: DType, dev: &Device) -> Result<Self> {
        let max_seq_len = cfg.max_position_embeddings;
        let dim = (cfg.head_dim as f64 * cfg.partial_rotary_factor.unwrap_or(1.)) as usize;

        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_theta.powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        let sin = freqs.sin()?.to_dtype(dtype)?;
        let cos = freqs.cos()?.to_dtype(dtype)?;
        Ok(Self {
            short_cos: cos,
            short_sin: sin,
            long_cos: None,
            long_sin: None,
            original_max_position_embeddings: cfg.original_max_position_embeddings,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn new_scaled(
        short_factor: &[f64],
        long_factor: &[f64],
        scaling_type: &ScaledRopeType,
        long_mscale: f64,
        short_mscale: f64,
        cfg: &PhiRopeConfig,
        dtype: DType,
        dev: &Device,
    ) -> Result<Self> {
        let max_seq_len = cfg.max_position_embeddings;
        let dim = (cfg.head_dim as f64 * cfg.partial_rotary_factor.unwrap_or(1.)) as usize;

        if !matches!(scaling_type, ScaledRopeType::Su) {
            candle_core::bail!("Scaled Phi3 RoPE (non-classic scaled, with mscales) must have type `su`/`longrope`.");
        }

        if short_factor.len() != dim / 2 {
            candle_core::bail!(
                "Misaligned length {}, expected {} for `su`/`longrope` short rescale factors",
                short_factor.len(),
                dim / 2
            );
        }
        if long_factor.len() != dim / 2 {
            candle_core::bail!(
                "Misaligned length {}, expected {} for `su`/`longrope` long rescale factors",
                long_factor.len(),
                dim / 2
            );
        }

        // Short cos/sin
        let inv_freq_short: Vec<_> = (0..dim)
            .step_by(2)
            .enumerate()
            .map(|(k, i)| {
                1f32 / (short_factor[k] * cfg.rope_theta.powf(i as f64 / dim as f64)) as f32
            })
            .collect();
        let inv_freq_len_short = inv_freq_short.len();
        let inv_freq_short = Tensor::from_vec(inv_freq_short, (1, inv_freq_len_short), dev)?;
        let t_short = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;
        let freqs_short = t_short.matmul(&inv_freq_short)?;
        let sin_short = (freqs_short.sin()?.to_dtype(dtype)? * short_mscale)?;
        let cos_short = (freqs_short.cos()?.to_dtype(dtype)? * short_mscale)?;

        // Long cos/sin
        let inv_freq_long: Vec<_> = (0..dim)
            .step_by(2)
            .enumerate()
            .map(|(k, i)| {
                1f32 / (long_factor[k] * cfg.rope_theta.powf(i as f64 / dim as f64)) as f32
            })
            .collect();
        let inv_freq_len_long = inv_freq_long.len();
        let inv_freq_long = Tensor::from_vec(inv_freq_long, (1, inv_freq_len_long), dev)?;
        let t_long = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;
        let freqs_long = t_long.matmul(&inv_freq_long)?;
        let sin_long = (freqs_long.sin()?.to_dtype(dtype)? * long_mscale)?;
        let cos_long = (freqs_long.cos()?.to_dtype(dtype)? * long_mscale)?;
        Ok(Self {
            short_cos: cos_short,
            short_sin: sin_short,
            long_cos: Some(cos_long),
            long_sin: Some(sin_long),
            original_max_position_embeddings: cfg.original_max_position_embeddings,
        })
    }

    pub fn new(dtype: DType, cfg: impl Into<PhiRopeConfig>, dev: &Device) -> Result<Self> {
        let cfg: PhiRopeConfig = cfg.into();

        match &cfg.rope_scaling {
            Some(PhiRopeScalingConfig::Classic {
                short_factor,
                long_factor,
                scaling_type,
            }) => {
                Self::new_classic_scaled(short_factor, long_factor, scaling_type, &cfg, dtype, dev)
            }

            Some(PhiRopeScalingConfig::Scaled {
                short_factor,
                long_factor,
                scaling_type,
                long_mscale,
                short_mscale,
            }) => Self::new_scaled(
                short_factor,
                long_factor,
                scaling_type,
                *long_mscale,
                *short_mscale,
                &cfg,
                dtype,
                dev,
            ),

            None => Self::new_unscaled(&cfg, dtype, dev),
        }
    }

    /// Returns (sin, cos) taking into account LongRope
    fn get_long_or_short_sin_cos(&self, position_ids: &[usize]) -> (&Tensor, &Tensor) {
        if self.long_cos.is_none() {
            return (&self.short_sin, &self.short_cos);
        }
        let seq_len = position_ids.iter().max().unwrap() + 1;
        if seq_len > self.original_max_position_embeddings {
            (
                self.long_sin.as_ref().unwrap(),
                self.long_cos.as_ref().unwrap(),
            )
        } else {
            (&self.short_sin, &self.short_cos)
        }
    }

    pub fn forward(
        &self,
        q: &Tensor,
        k: &Tensor,
        seqlen_offsets: &[usize],
        position_ids: &[usize],
    ) -> Result<(Tensor, Tensor)> {
        let (sin, cos) = self.get_long_or_short_sin_cos(position_ids);
        let (_b_sz, _h, seq_len, _n_embd) = q.dims4()?;

        let rot_dim = cos.dim(D::Minus1)? * 2;

        // Case for Phi 3 / Phi 4 mini
        if rot_dim != q.dim(D::Minus1)? {
            let rot_dim = cos.dim(D::Minus1)? * 2;
            let q_rot = q.narrow(D::Minus1, 0, rot_dim)?;
            let q_pass = q.narrow(D::Minus1, rot_dim, q.dim(D::Minus1)? - rot_dim)?;
            let k_rot = k.narrow(D::Minus1, 0, rot_dim)?;
            let k_pass = k.narrow(D::Minus1, rot_dim, k.dim(D::Minus1)? - rot_dim)?;

            // Batched whenever every row shares an offset -- see
            // `uniform_seqlen_offset`. This used to require `len() == 1`.
            let (q_rot, k_rot) = if let Some(offset) = uniform_seqlen_offset(seqlen_offsets) {
                rope_cohort_stats::record_cohort();
                let cos = cos.narrow(0, offset, seq_len)?;
                let sin = sin.narrow(0, offset, seq_len)?;
                let q_embed = candle_nn::rotary_emb::rope(&q_rot.contiguous()?, &cos, &sin)?;
                let k_embed = candle_nn::rotary_emb::rope(&k_rot.contiguous()?, &cos, &sin)?;
                (q_embed, k_embed)
            } else {
                rope_cohort_stats::record_per_sequence();
                let mut q_embeds = Vec::new();
                let mut k_embeds = Vec::new();
                for (i, offset) in seqlen_offsets.iter().enumerate() {
                    let cos = cos.narrow(0, *offset, seq_len)?;
                    let sin = sin.narrow(0, *offset, seq_len)?;
                    let q_embed = candle_nn::rotary_emb::rope(
                        &q_rot.i(i)?.unsqueeze(0)?.contiguous()?,
                        &cos,
                        &sin,
                    )?;
                    let k_embed = candle_nn::rotary_emb::rope(
                        &k_rot.i(i)?.unsqueeze(0)?.contiguous()?,
                        &cos,
                        &sin,
                    )?;
                    q_embeds.push(q_embed);
                    k_embeds.push(k_embed);
                }
                let q_rot = Tensor::cat(&q_embeds, 0)?;
                let k_rot = Tensor::cat(&k_embeds, 0)?;
                (q_rot, k_rot)
            };

            Ok((
                Tensor::cat(&[q_rot, q_pass], D::Minus1)?.contiguous()?,
                Tensor::cat(&[k_rot, k_pass], D::Minus1)?.contiguous()?,
            ))
        } else if let Some(offset) = uniform_seqlen_offset(seqlen_offsets) {
            rope_cohort_stats::record_cohort();
            let cos = cos.narrow(0, offset, seq_len)?;
            let sin = sin.narrow(0, offset, seq_len)?;
            let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
            let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
            Ok((q_embed, k_embed))
        } else {
            rope_cohort_stats::record_per_sequence();
            let mut q_embeds = Vec::new();
            let mut k_embeds = Vec::new();
            for (i, offset) in seqlen_offsets.iter().enumerate() {
                let cos = cos.narrow(0, *offset, seq_len)?;
                let sin = sin.narrow(0, *offset, seq_len)?;
                let q_embed =
                    candle_nn::rotary_emb::rope(&q.i(i)?.unsqueeze(0)?.contiguous()?, &cos, &sin)?;
                let k_embed =
                    candle_nn::rotary_emb::rope(&k.i(i)?.unsqueeze(0)?.contiguous()?, &cos, &sin)?;
                q_embeds.push(q_embed);
                k_embeds.push(k_embed);
            }
            Ok((Tensor::cat(&q_embeds, 0)?, Tensor::cat(&k_embeds, 0)?))
        }
    }
}

/// RoPE for Llama3
#[derive(Debug, Clone)]
pub struct Llama3RotaryEmbedding(RotaryEmbedding);

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub enum Llama3RopeType {
    #[serde(rename = "llama3")]
    Llama3,
    #[serde(rename = "linear")]
    Linear,
    #[default]
    #[serde(rename = "default")]
    Default,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct Llama3RopeConfig {
    pub factor: f32,
    pub low_freq_factor: Option<f32>,
    pub high_freq_factor: Option<f32>,
    pub original_max_position_embeddings: Option<usize>,
    pub rope_type: Llama3RopeType,
}

fn calculate_default_inv_freq(cfg: &llama::Config) -> Vec<f32> {
    let head_dim = cfg.hidden_size / cfg.num_attention_heads;
    (0..head_dim)
        .step_by(2)
        .map(|i| 1f32 / cfg.rope_theta.powf(i as f32 / head_dim as f32))
        .collect()
}

fn calculate_default_inv_freq_llama4(cfg: &llama4::TextConfig) -> Vec<f32> {
    let head_dim = cfg.hidden_size / cfg.num_attention_heads;
    (0..head_dim)
        .step_by(2)
        .map(|i| 1f32 / cfg.rope_theta.powf(i as f32 / head_dim as f32))
        .collect()
}

// https://github.com/huggingface/transformers/blob/1392a6867f40a55dfabaf306745c67627598b1af/src/transformers/modeling_rope_utils.py#L298
impl Llama3RotaryEmbedding {
    pub fn new_llama3(
        dtype: DType,
        cfg: &llama::Config,
        dev: &Device,
        is_gpt_neox: bool,
    ) -> Result<Self> {
        match &cfg.rope_scaling {
            None
            | Some(Llama3RopeConfig {
                rope_type: Llama3RopeType::Default,
                ..
            }) => Ok(Self(RotaryEmbedding::new(
                cfg.rope_theta,
                cfg.hidden_size / cfg.num_attention_heads,
                cfg.max_position_embeddings,
                dev,
                is_gpt_neox,
                dtype,
            )?)),
            Some(Llama3RopeConfig {
                rope_type: Llama3RopeType::Llama3,
                factor,
                low_freq_factor,
                high_freq_factor,
                original_max_position_embeddings,
            }) => {
                let low_freq_factor = low_freq_factor.context("low_freq_factor is required")?;
                let high_freq_factor = high_freq_factor.context("high_freq_factor is required")?;
                let original_max_position_embeddings = original_max_position_embeddings
                    .context("original_max_position_embeddings is required")?;

                let low_freq_wavelen = original_max_position_embeddings as f32 / low_freq_factor;
                let high_freq_wavelen = original_max_position_embeddings as f32 / high_freq_factor;

                let inv_freq = calculate_default_inv_freq(cfg)
                    .into_iter()
                    .map(|freq| {
                        let wavelen = 2. * PI / freq;
                        if wavelen < high_freq_wavelen {
                            freq
                        } else if wavelen > low_freq_wavelen {
                            freq / *factor
                        } else {
                            let smooth = (original_max_position_embeddings as f32 / wavelen
                                - low_freq_factor)
                                / (high_freq_factor - low_freq_factor);
                            (1. - smooth) * freq / *factor + smooth * freq
                        }
                    })
                    .collect::<Vec<_>>();
                let inv_freq_len = inv_freq.len();
                let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?;
                let t = Tensor::arange(0u32, cfg.max_position_embeddings as u32, dev)?
                    .to_dtype(DType::F32)?
                    .reshape((cfg.max_position_embeddings, 1))?;
                let freqs = t.matmul(&inv_freq)?;
                let sin = freqs.sin()?.to_dtype(dtype)?;
                let cos = freqs.cos()?.to_dtype(dtype)?;
                Ok(Self(RotaryEmbedding {
                    sin,
                    cos,
                    is_gpt_neox,
                }))
            }
            Some(Llama3RopeConfig {
                rope_type: Llama3RopeType::Linear,
                factor,
                ..
            }) => {
                let inv_freq_vec = calculate_default_inv_freq(cfg)
                    .into_iter()
                    .map(|freq| freq / *factor)
                    .collect::<Vec<_>>();
                let inv_freq_len = inv_freq_vec.len();
                let inv_freq = Tensor::from_vec(inv_freq_vec, (1, inv_freq_len), dev)?;
                let t = Tensor::arange(0u32, cfg.max_position_embeddings as u32, dev)?
                    .to_dtype(DType::F32)?
                    .reshape((cfg.max_position_embeddings, 1))?;
                let freqs = t.matmul(&inv_freq)?;
                let sin = freqs.sin()?.to_dtype(dtype)?;
                let cos = freqs.cos()?.to_dtype(dtype)?;
                Ok(Self(RotaryEmbedding {
                    sin,
                    cos,
                    is_gpt_neox,
                }))
            }
        }
    }

    pub fn new_llama4(
        dtype: DType,
        cfg: &llama4::TextConfig,
        dev: &Device,
        is_gpt_neox: bool,
    ) -> Result<Self> {
        match &cfg.rope_scaling {
            None
            | Some(Llama3RopeConfig {
                rope_type: Llama3RopeType::Default,
                ..
            }) => Ok(Self(RotaryEmbedding::new(
                cfg.rope_theta,
                cfg.hidden_size / cfg.num_attention_heads,
                cfg.max_position_embeddings,
                dev,
                is_gpt_neox,
                dtype,
            )?)),
            Some(Llama3RopeConfig {
                rope_type: Llama3RopeType::Llama3,
                factor,
                low_freq_factor,
                high_freq_factor,
                original_max_position_embeddings,
            }) => {
                let low_freq_factor = low_freq_factor.context("low_freq_factor is required")?;
                let high_freq_factor = high_freq_factor.context("high_freq_factor is required")?;
                let original_max_position_embeddings = original_max_position_embeddings
                    .context("original_max_position_embeddings is required")?;

                let low_freq_wavelen = original_max_position_embeddings as f32 / low_freq_factor;
                let high_freq_wavelen = original_max_position_embeddings as f32 / high_freq_factor;

                let inv_freq = calculate_default_inv_freq_llama4(cfg)
                    .into_iter()
                    .map(|freq| {
                        let wavelen = 2. * PI / freq;
                        if wavelen < high_freq_wavelen {
                            freq
                        } else if wavelen > low_freq_wavelen {
                            freq / *factor
                        } else {
                            let smooth = (original_max_position_embeddings as f32 / wavelen
                                - low_freq_factor)
                                / (high_freq_factor - low_freq_factor);
                            (1. - smooth) * freq / *factor + smooth * freq
                        }
                    })
                    .collect::<Vec<_>>();
                let inv_freq_len = inv_freq.len();
                let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?;
                let t = Tensor::arange(0u32, cfg.max_position_embeddings as u32, dev)?
                    .to_dtype(DType::F32)?
                    .reshape((cfg.max_position_embeddings, 1))?;
                let freqs = t.matmul(&inv_freq)?;
                let sin = freqs.sin()?.to_dtype(dtype)?;
                let cos = freqs.cos()?.to_dtype(dtype)?;
                Ok(Self(RotaryEmbedding {
                    sin,
                    cos,
                    is_gpt_neox,
                }))
            }
            Some(Llama3RopeConfig {
                rope_type: Llama3RopeType::Linear,
                factor,
                ..
            }) => {
                let inv_freq_vec = calculate_default_inv_freq_llama4(cfg)
                    .into_iter()
                    .map(|freq| freq / *factor)
                    .collect::<Vec<_>>();
                let inv_freq_len = inv_freq_vec.len();
                let inv_freq = Tensor::from_vec(inv_freq_vec, (1, inv_freq_len), dev)?;
                let t = Tensor::arange(0u32, cfg.max_position_embeddings as u32, dev)?
                    .to_dtype(DType::F32)?
                    .reshape((cfg.max_position_embeddings, 1))?;
                let freqs = t.matmul(&inv_freq)?;
                let sin = freqs.sin()?.to_dtype(dtype)?;
                let cos = freqs.cos()?.to_dtype(dtype)?;
                Ok(Self(RotaryEmbedding {
                    sin,
                    cos,
                    is_gpt_neox,
                }))
            }
        }
    }

    pub fn new_mllama3(
        dtype: DType,
        cfg: &MLlamaTextConfig,
        dev: &Device,
        is_gpt_neox: bool,
    ) -> Result<Self> {
        match &cfg.rope_scaling {
            None
            | Some(MLlamaRopeScaling {
                rope_type: MLlamaRopeType::Default,
                ..
            }) => Ok(Self(RotaryEmbedding::new(
                cfg.rope_theta,
                cfg.hidden_size / cfg.num_attention_heads,
                cfg.max_position_embeddings,
                dev,
                is_gpt_neox,
                dtype,
            )?)),
            Some(MLlamaRopeScaling {
                rope_type: MLlamaRopeType::Llama3,
                original_max_position_embeddings,
                factor,
                attention_factor: _,
                beta_fast: _,
                beta_slow: _,
                short_factor: _,
                long_factor: _,
                low_freq_factor,
                high_freq_factor,
            }) => {
                let factor = factor.context("MLlama Llama3 RoPE needs `factor` parameter.")?;
                let low_freq_factor = low_freq_factor
                    .context("MLlama Llama3 RoPE needs `low_freq_factor` parameter.")?;
                let high_freq_factor = high_freq_factor
                    .context("MLlama Llama3 RoPE needs `high_freq_factor` parameter.")?;

                let low_freq_wavelen = *original_max_position_embeddings as f32 / low_freq_factor;
                let high_freq_wavelen = *original_max_position_embeddings as f32 / high_freq_factor;

                let head_dim = cfg.hidden_size / cfg.num_attention_heads;

                let inv_freq = (0..head_dim)
                    .step_by(2)
                    .map(|i| 1f32 / cfg.rope_theta.powf(i as f32 / head_dim as f32))
                    .map(|freq| {
                        let wavelen = 2. * PI / freq;
                        if wavelen < high_freq_wavelen {
                            freq
                        } else if wavelen > low_freq_wavelen {
                            freq / factor
                        } else {
                            let smooth = (*original_max_position_embeddings as f32 / wavelen
                                - low_freq_factor)
                                / (high_freq_factor - low_freq_factor);
                            (1. - smooth) * freq / factor + smooth * freq
                        }
                    })
                    .collect::<Vec<_>>();
                let inv_freq_len = inv_freq.len();
                let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?;

                let t = Tensor::arange(0u32, cfg.max_position_embeddings as u32, dev)?
                    .to_dtype(DType::F32)?
                    .reshape((cfg.max_position_embeddings, 1))?;
                let freqs = t.matmul(&inv_freq)?;
                let sin = freqs.sin()?.to_dtype(dtype)?;
                let cos = freqs.cos()?.to_dtype(dtype)?;
                Ok(Self(RotaryEmbedding {
                    sin,
                    cos,
                    is_gpt_neox,
                }))
            }
            Some(MLlamaRopeScaling {
                rope_type: other, ..
            }) => {
                candle_core::bail!(
                    "MLlama doesn't support any other RoPE type than `llama3`, got {other:?}"
                )
            }
        }
    }

    pub fn forward(
        &self,
        q: &Tensor,
        k: &Tensor,
        seqlen_offsets: &[usize],
    ) -> Result<(Tensor, Tensor)> {
        self.0.forward(q, k, seqlen_offsets)
    }
}

/// RoPE for SmolLm3
#[derive(Debug, Clone)]
pub struct SmolLm3RotaryEmbedding(RotaryEmbedding);

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub enum SmolLm3RopeType {
    #[serde(rename = "llama3")]
    Llama3,
    #[serde(rename = "linear")]
    Linear,
    #[default]
    #[serde(rename = "default")]
    Default,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct SmolLm3RopeConfig {
    pub factor: f32,
    pub low_freq_factor: Option<f32>,
    pub high_freq_factor: Option<f32>,
    pub original_max_position_embeddings: Option<usize>,
    pub rope_type: SmolLm3RopeType,
}

fn calculate_default_inv_freq_smollm3(cfg: &smollm3::Config) -> Vec<f32> {
    let head_dim = cfg.hidden_size / cfg.num_attention_heads;
    (0..head_dim)
        .step_by(2)
        .map(|i| 1f32 / cfg.rope_theta.powf(i as f32 / head_dim as f32))
        .collect()
}

impl SmolLm3RotaryEmbedding {
    pub fn new_llama3(
        dtype: DType,
        cfg: &smollm3::Config,
        dev: &Device,
        is_gpt_neox: bool,
    ) -> Result<Self> {
        match &cfg.rope_scaling {
            None
            | Some(SmolLm3RopeConfig {
                rope_type: SmolLm3RopeType::Default,
                ..
            }) => Ok(Self(RotaryEmbedding::new(
                cfg.rope_theta,
                cfg.hidden_size / cfg.num_attention_heads,
                cfg.max_position_embeddings,
                dev,
                is_gpt_neox,
                dtype,
            )?)),
            Some(SmolLm3RopeConfig {
                rope_type: SmolLm3RopeType::Llama3,
                factor,
                low_freq_factor,
                high_freq_factor,
                original_max_position_embeddings,
            }) => {
                let low_freq_factor = low_freq_factor.context("low_freq_factor is required")?;
                let high_freq_factor = high_freq_factor.context("high_freq_factor is required")?;
                let original_max_position_embeddings = original_max_position_embeddings
                    .context("original_max_position_embeddings is required")?;

                let low_freq_wavelen = original_max_position_embeddings as f32 / low_freq_factor;
                let high_freq_wavelen = original_max_position_embeddings as f32 / high_freq_factor;

                let inv_freq = calculate_default_inv_freq_smollm3(cfg)
                    .into_iter()
                    .map(|freq| {
                        let wavelen = 2. * PI / freq;
                        if wavelen < high_freq_wavelen {
                            freq
                        } else if wavelen > low_freq_wavelen {
                            freq / *factor
                        } else {
                            let smooth = (original_max_position_embeddings as f32 / wavelen
                                - low_freq_factor)
                                / (high_freq_factor - low_freq_factor);
                            (1. - smooth) * freq / *factor + smooth * freq
                        }
                    })
                    .collect::<Vec<_>>();
                let inv_freq_len = inv_freq.len();
                let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?;
                let t = Tensor::arange(0u32, cfg.max_position_embeddings as u32, dev)?
                    .to_dtype(DType::F32)?
                    .reshape((cfg.max_position_embeddings, 1))?;
                let freqs = t.matmul(&inv_freq)?;
                let sin = freqs.sin()?.to_dtype(dtype)?;
                let cos = freqs.cos()?.to_dtype(dtype)?;
                Ok(Self(RotaryEmbedding {
                    sin,
                    cos,
                    is_gpt_neox,
                }))
            }
            Some(SmolLm3RopeConfig {
                rope_type: SmolLm3RopeType::Linear,
                factor,
                ..
            }) => {
                let inv_freq_vec = calculate_default_inv_freq_smollm3(cfg)
                    .into_iter()
                    .map(|freq| freq / *factor)
                    .collect::<Vec<_>>();
                let inv_freq_len = inv_freq_vec.len();
                let inv_freq = Tensor::from_vec(inv_freq_vec, (1, inv_freq_len), dev)?;
                let t = Tensor::arange(0u32, cfg.max_position_embeddings as u32, dev)?
                    .to_dtype(DType::F32)?
                    .reshape((cfg.max_position_embeddings, 1))?;
                let freqs = t.matmul(&inv_freq)?;
                let sin = freqs.sin()?.to_dtype(dtype)?;
                let cos = freqs.cos()?.to_dtype(dtype)?;
                Ok(Self(RotaryEmbedding {
                    sin,
                    cos,
                    is_gpt_neox,
                }))
            }
        }
    }

    pub fn forward(
        &self,
        q: &Tensor,
        k: &Tensor,
        seqlen_offsets: &[usize],
    ) -> Result<(Tensor, Tensor)> {
        self.0.forward(q, k, seqlen_offsets)
    }
}

// https://github.com/huggingface/transformers/blob/f2c388e3f946862f657acc1e21b272ec946fc66c/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py#L107
#[derive(Debug, Clone)]
pub struct Qwen2VLRotaryEmbedding {
    inv_freq: Tensor,
    mrope_section: Vec<usize>,
}

impl Qwen2VLRotaryEmbedding {
    pub fn new(
        base: f32,
        head_dim: usize,
        device: &Device,
        mrope_section: Vec<usize>,
    ) -> Result<Self> {
        let inv_freq: Vec<_> = (0..head_dim)
            .step_by(2)
            .map(|i| 1f32 / base.powf(i as f32 / head_dim as f32))
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (inv_freq_len,), device)?.to_dtype(DType::F32)?;
        Ok(Self {
            inv_freq,
            mrope_section,
        })
    }

    /// (cos, sin)
    pub fn compute_cos_sin(&self, position_ids: &Tensor, dtype: DType) -> Result<(Tensor, Tensor)> {
        let inv_freq_expanded =
            self.inv_freq
                .reshape((1, 1, (), 1))?
                .repeat((3, position_ids.dim(1)?, 1, 1))?;
        let position_ids_expanded = position_ids.unsqueeze(2)?;
        let freqs = inv_freq_expanded
            .matmul(&position_ids_expanded.to_dtype(inv_freq_expanded.dtype())?)?
            .transpose(2, 3)?;
        let cos = freqs.cos()?;
        let sin = freqs.sin()?;

        let cos = Tensor::cat(
            &cos.split(&self.mrope_section, D::Minus1)?
                .into_iter()
                .enumerate()
                .map(|(i, m)| m.i(i % 3))
                .collect::<Result<Vec<_>>>()?,
            D::Minus1,
        )?
        .squeeze(0)?
        .to_dtype(dtype)?
        .contiguous()?;
        let sin = Tensor::cat(
            &sin.split(&self.mrope_section, D::Minus1)?
                .into_iter()
                .enumerate()
                .map(|(i, m)| m.i(i % 3))
                .collect::<Result<Vec<_>>>()?,
            D::Minus1,
        )?
        .squeeze(0)?
        .to_dtype(dtype)?
        .contiguous()?;

        Ok((cos, sin))
    }

    // https://github.com/huggingface/transformers/blob/f2c388e3f946862f657acc1e21b272ec946fc66c/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py#L203
    pub fn forward(
        &self,
        (cos, sin): &(Tensor, Tensor),
        q: &mut Tensor,
        k: &mut Tensor,
    ) -> Result<()> {
        *q = candle_nn::rotary_emb::rope(&q.contiguous()?, cos, sin)?;
        *k = candle_nn::rotary_emb::rope(&k.contiguous()?, cos, sin)?;
        Ok(())
    }
}

/// Qwen3 VL uses **interleaved** MRoPE (not chunked like Qwen2 VL).
/// Frequencies are arranged as THW THW THW... TTTT pattern.
/// See `apply_interleaved_mrope` in modeling_qwen3_vl.py.
#[derive(Debug, Clone)]
pub struct Qwen3VLRotaryEmbedding {
    inv_freq: Tensor,
    /// Precomputed interleave indices for H (dim=1, offset=1) and W (dim=2, offset=2).
    /// Stored as (indices_1d, dim_idx) pairs. Created once at init to avoid CPU->GPU sync per step.
    interleave_indices: Vec<(Tensor, usize)>,
}

impl Qwen3VLRotaryEmbedding {
    pub fn new(
        base: f32,
        head_dim: usize,
        device: &Device,
        mrope_section: Vec<usize>,
    ) -> Result<Self> {
        let inv_freq: Vec<_> = (0..head_dim)
            .step_by(2)
            .map(|i| 1f32 / base.powf(i as f32 / head_dim as f32))
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (inv_freq_len,), device)?.to_dtype(DType::F32)?;

        // Precompute interleave index tensors for H (dim=1, offset=1) and W (dim=2, offset=2)
        // to avoid CPU->GPU sync from Tensor::from_vec on every decode step.
        let half_dim = head_dim / 2;
        let mut interleave_indices = Vec::new();
        for (dim_idx, offset) in [(1usize, 1usize), (2usize, 2usize)] {
            let indices: Vec<u32> = (offset..)
                .step_by(3)
                .take(mrope_section[dim_idx])
                .filter(|&i| i < half_dim)
                .map(|i| i as u32)
                .collect();
            if !indices.is_empty() {
                let num = indices.len();
                let idx_tensor = Tensor::from_vec(indices, (num,), device)?;
                interleave_indices.push((idx_tensor, dim_idx));
            }
        }

        Ok(Self {
            inv_freq,
            interleave_indices,
        })
    }

    /// Compute (cos, sin) from 3D position_ids of shape (3, batch, seq_len).
    /// Applies interleaved MRoPE: starts with temporal freqs, then overwrites
    /// H positions (slice 1::3) and W positions (slice 2::3) within their sections.
    pub fn compute_cos_sin(&self, position_ids: &Tensor, dtype: DType) -> Result<(Tensor, Tensor)> {
        // inv_freq: (head_dim/2,) -> (1, 1, head_dim/2, 1) -> expand to (3, batch, head_dim/2, 1)
        let inv_freq_expanded =
            self.inv_freq
                .reshape((1, 1, (), 1))?
                .repeat((3, position_ids.dim(1)?, 1, 1))?;
        // position_ids: (3, batch, seq_len) -> (3, batch, 1, seq_len)
        let position_ids_expanded = position_ids.unsqueeze(2)?;
        // freqs: (3, batch, head_dim/2, 1) @ (3, batch, 1, seq_len) -> (3, batch, head_dim/2, seq_len)
        // -> transpose -> (3, batch, seq_len, head_dim/2)
        let freqs = inv_freq_expanded
            .matmul(&position_ids_expanded.to_dtype(inv_freq_expanded.dtype())?)?
            .transpose(2, 3)?;

        // Apply interleaved MRoPE: start with temporal, overwrite H and W at interleaved positions
        // freqs_t = freqs[0] as base (all temporal)
        let mut freqs_t = freqs.i(0)?.contiguous()?;
        let (batch, seq_len, _) = freqs_t.dims3()?;

        // For H (dim=1) and W (dim=2), overwrite interleaved positions using precomputed indices
        for (idx_tensor, dim_idx) in &self.interleave_indices {
            let freqs_dim = freqs.i(*dim_idx)?.contiguous()?;
            let num_indices = idx_tensor.dim(0)?;
            let idx_expanded = idx_tensor
                .reshape((1, 1, num_indices))?
                .repeat((batch, seq_len, 1))?;
            let src_vals = freqs_dim.gather(&idx_expanded, D::Minus1)?;
            freqs_t = freqs_t.scatter(&idx_expanded, &src_vals, D::Minus1)?;
        }

        // cos/sin from freqs_t -> (batch, seq_len, head_dim/2)
        // candle's rope() expects half-dim cos/sin and handles both halves internally
        let cos = freqs_t.cos()?.to_dtype(dtype)?.contiguous()?;
        let sin = freqs_t.sin()?.to_dtype(dtype)?.contiguous()?;
        Ok((cos, sin))
    }

    pub fn forward(
        &self,
        (cos, sin): &(Tensor, Tensor),
        q: &mut Tensor,
        k: &mut Tensor,
    ) -> Result<()> {
        *q = candle_nn::rotary_emb::rope(&q.contiguous()?, cos, sin)?;
        *k = candle_nn::rotary_emb::rope(&k.contiguous()?, cos, sin)?;
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct Qwen2_5VLRotaryEmbedding {
    inv_freq: Tensor,
    mrope_section: Vec<usize>,
}

impl Qwen2_5VLRotaryEmbedding {
    pub fn new(
        base: f32,
        head_dim: usize,
        device: &Device,
        mrope_section: Vec<usize>,
    ) -> Result<Self> {
        let inv_freq: Vec<_> = (0..head_dim)
            .step_by(2)
            .map(|i| 1f32 / base.powf(i as f32 / head_dim as f32))
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (inv_freq_len,), device)?.to_dtype(DType::F32)?;
        Ok(Self {
            inv_freq,
            mrope_section,
        })
    }

    /// (cos, sin)
    pub fn compute_cos_sin(&self, position_ids: &Tensor, dtype: DType) -> Result<(Tensor, Tensor)> {
        let inv_freq_expanded =
            self.inv_freq
                .reshape((1, 1, (), 1))?
                .repeat((3, position_ids.dim(1)?, 1, 1))?;
        let position_ids_expanded = position_ids.unsqueeze(2)?;
        let freqs = inv_freq_expanded
            .matmul(&position_ids_expanded.to_dtype(inv_freq_expanded.dtype())?)?
            .transpose(2, 3)?;
        let cos = freqs.cos()?;
        let sin = freqs.sin()?;

        let cos = Tensor::cat(
            &cos.split(&self.mrope_section, D::Minus1)?
                .into_iter()
                .enumerate()
                .map(|(i, m)| m.i(i % 3))
                .collect::<Result<Vec<_>>>()?,
            D::Minus1,
        )?
        .squeeze(0)?
        .to_dtype(dtype)?
        .contiguous()?;
        let sin = Tensor::cat(
            &sin.split(&self.mrope_section, D::Minus1)?
                .into_iter()
                .enumerate()
                .map(|(i, m)| m.i(i % 3))
                .collect::<Result<Vec<_>>>()?,
            D::Minus1,
        )?
        .squeeze(0)?
        .to_dtype(dtype)?
        .contiguous()?;

        Ok((cos, sin))
    }

    pub fn forward(
        &self,
        (cos, sin): &(Tensor, Tensor),
        q: &mut Tensor,
        k: &mut Tensor,
    ) -> Result<()> {
        *q = candle_nn::rotary_emb::rope(&q.contiguous()?, cos, sin)?;
        *k = candle_nn::rotary_emb::rope(&k.contiguous()?, cos, sin)?;
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct DeepSeekV2RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
    /// `-sin`, materialised once at construction.
    ///
    /// [`Self::forward_inverse_tail`] rotates by the conjugate `(cos, -sin)`.
    /// It used to spell that as `self.sin.narrow(..)?.neg()?` *inside* the
    /// per-sequence loop, which at B=256 × 43 layers launched `uneg_bf16`
    /// **11,008 times per decode step** to negate a `[1, 32]` slice of a
    /// **constant** table — 0.8% of a measured B=256 step spent re-deriving a
    /// value that never changes.
    ///
    /// Narrowing a pre-negated table is bit-identical to negating a narrowed
    /// one: negation is a sign-bit flip, exact in every float format and
    /// elementwise, so it commutes with a view. See `neg_sin_commutes_with_narrow`.
    neg_sin: Tensor,
    /// Memoised per-row table gather for the ragged-cohort path
    /// ([`rope_cohort_enabled`]). Behind an `Arc` so the per-device sharing in
    /// `deepseek4.rs` (one rotary per device per table kind, cloned into every
    /// layer on that device) shares ONE cache: the offsets are host data, so
    /// one small H2D per step per device is the floor, and this cache holds it
    /// there instead of paying it again on all 43 layers × 3 rope calls
    /// (CLAUDE.md pitfall #5 — no `Tensor::from_vec` per layer on the hot
    /// path). `None` until the gated path first runs; the flag-off paths never
    /// touch it.
    cohort_gather: Arc<std::sync::Mutex<Option<CohortGather>>>,
}

/// One step's gathered `[B, T, rope_dim / 2]` tables for a ragged cohort,
/// keyed by the exact `(seqlen_offsets, seq_len)` that built them. All three
/// tables are gathered together: `forward` reads (cos, sin),
/// `forward_inverse_tail` reads (cos, neg_sin), and they share the step.
#[derive(Debug)]
struct CohortGather {
    offsets: Vec<usize>,
    seq_len: usize,
    /// `[B, T, rope_dim / 2]` each, contiguous, on the tables' device.
    cos: Tensor,
    sin: Tensor,
    neg_sin: Tensor,
}

// The engagement counters for this path are the ones declared beside
// `uniform_seqlen_offset` (`rope_cohort_stats`, above). #198 carried a second
// module of the same name, with the same two atomics and a `counts()` of its
// own; two `pub mod rope_cohort_stats` in one file is a hard name collision,
// and two counters for one claim would let a harness read `cohort > 0` off the
// Phi path while this one never engaged. Collapsed onto the shared module, as
// `uniform_seqlen_offset`'s doc comment asks.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum DeepSeekV2RopeScaling {
    Yarn {
        original_max_position_embeddings: usize,
        beta_fast: f32,
        beta_slow: f32,
        // `mscale` / `mscale_all_dim` are optional. DeepSeek V4 Flash ships with
        // `type: "yarn"`, `original_max_position_embeddings`, `beta_fast`,
        // `beta_slow`, `factor` only — without the two mscale fields, which
        // means the HF reference defaults apply (`mscale = 1.0`,
        // `mscale_all_dim = 1.0` → `yarn_get_mscale` ratio of 1.0, neutral).
        // Serde `default = ...` keeps backward-compat with V2/V3 configs that
        // *do* set these fields explicitly.
        #[serde(default = "default_mscale")]
        mscale: f32,
        #[serde(default = "default_mscale")]
        mscale_all_dim: f32,
        factor: f32,
        #[serde(rename = "type")]
        scaling_type: ScaledRopeType,
    },
    LinearOrDynamic {
        #[serde(rename = "type")]
        scaling_type: ScaledRopeType,
        factor: f64,
    },
}

fn default_mscale() -> f32 {
    1.0
}

pub struct DeepSeekV2RopeConfig {
    pub rope_scaling: Option<DeepSeekV2RopeScaling>,
    pub max_position_embeddings: usize,
    pub rope_theta: f32,
    pub qk_rope_head_dim: usize,
}

impl DeepSeekV2RotaryEmbedding {
    /// The single place the tables are stored, so `neg_sin` cannot be derived
    /// in one constructor and forgotten in the other.
    fn from_tables(sin: Tensor, cos: Tensor) -> Result<Self> {
        let neg_sin = sin.neg()?;
        Ok(Self {
            sin,
            cos,
            neg_sin,
            cohort_gather: Arc::new(std::sync::Mutex::new(None)),
        })
    }

    /// The step's `(cos, sin, neg_sin)` — each `[B, T, rope_dim / 2]` — for a
    /// ragged cohort, one table row per `(b, t)` at absolute position
    /// `seqlen_offsets[b] + t`.
    ///
    /// The row indices are host data (`seqlen_offsets` is a host slice), so
    /// one H2D copy per step is the floor; everything after it — the three
    /// `index_select`s included — happens once per step per device and is
    /// served from [`Self::cohort_gather`] to the other rope calls of the same
    /// step (all layers on this device share one instance of `self`; see the
    /// field's doc). `index_select` output is contiguous and `reshape` of a
    /// contiguous tensor is a view, so the returned tables satisfy `rope_i`'s
    /// contiguity requirement with no extra copy.
    fn cohort_tables(
        &self,
        seqlen_offsets: &[usize],
        seq_len: usize,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let mut cache = self
            .cohort_gather
            .lock()
            .expect("cohort gather cache poisoned");
        if let Some(entry) = cache.as_ref() {
            if entry.seq_len == seq_len && entry.offsets == seqlen_offsets {
                return Ok((entry.cos.clone(), entry.sin.clone(), entry.neg_sin.clone()));
            }
        }
        let b = seqlen_offsets.len();
        let mut rows: Vec<u32> = Vec::with_capacity(b * seq_len);
        for &off in seqlen_offsets {
            for t in 0..seq_len {
                rows.push((off + t) as u32);
            }
        }
        let idx = Tensor::from_vec(rows, b * seq_len, self.cos.device())?;
        let half = self.cos.dim(D::Minus1)?;
        let cos = self
            .cos
            .index_select(&idx, 0)?
            .reshape((b, seq_len, half))?;
        let sin = self
            .sin
            .index_select(&idx, 0)?
            .reshape((b, seq_len, half))?;
        let neg_sin = self
            .neg_sin
            .index_select(&idx, 0)?
            .reshape((b, seq_len, half))?;
        *cache = Some(CohortGather {
            offsets: seqlen_offsets.to_vec(),
            seq_len,
            cos: cos.clone(),
            sin: sin.clone(),
            neg_sin: neg_sin.clone(),
        });
        Ok((cos, sin, neg_sin))
    }

    /// The gated ragged arm of [`Self::forward`]: ONE `rope_i` per projection
    /// over the whole cohort, with the per-row rotation expressed through a
    /// rank-3 `[B, T, D/2]` cos/sin (see [`rope_cohort_enabled`] for why that
    /// is expressible at all). Bit-identical to [`Self::forward_ragged_loop`]:
    /// the gather copies table rows verbatim, and each output element is the
    /// same two-multiply-one-add of the same inputs — batching changes which
    /// launch computes an element, never the arithmetic. Asserted bit-for-bit
    /// in `deepseek_ragged_cohort_tests`.
    fn forward_ragged_cohort(
        &self,
        q: &Tensor,
        k: &Tensor,
        seqlen_offsets: &[usize],
        seq_len: usize,
    ) -> Result<(Tensor, Tensor)> {
        let (cos, sin, _neg_sin) = self.cohort_tables(seqlen_offsets, seq_len)?;
        let q_embed = candle_nn::rotary_emb::rope_i(&q.contiguous()?, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope_i(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }

    /// The per-sequence loop, extracted verbatim so both [`Self::forward`]
    /// arms name the thing they dispatch between and the parity tests can
    /// drive each directly whatever `ARC_ROPE_COHORT` latched to.
    fn forward_ragged_loop(
        &self,
        q: &Tensor,
        k: &Tensor,
        seqlen_offsets: &[usize],
        seq_len: usize,
    ) -> Result<(Tensor, Tensor)> {
        let mut q_embeds = Vec::new();
        let mut k_embeds = Vec::new();
        for (i, offset) in seqlen_offsets.iter().enumerate() {
            let cos = self.cos.narrow(0, *offset, seq_len)?;
            let sin = self.sin.narrow(0, *offset, seq_len)?;
            let q_embed =
                candle_nn::rotary_emb::rope_i(&q.i(i)?.unsqueeze(0)?.contiguous()?, &cos, &sin)?;
            let k_embed =
                candle_nn::rotary_emb::rope_i(&k.i(i)?.unsqueeze(0)?.contiguous()?, &cos, &sin)?;
            q_embeds.push(q_embed);
            k_embeds.push(k_embed);
        }
        Ok((Tensor::cat(&q_embeds, 0)?, Tensor::cat(&k_embeds, 0)?))
    }

    /// Gated ragged arm of [`Self::forward_inverse_tail`]: one `rope_i` over
    /// the whole cohort, conjugate rotation via the pre-negated table gathered
    /// per row. Returns only the rotated tail; the caller re-attaches the NoPE
    /// dims (shared with the loop arm, so the `cat` is spelled once).
    fn inverse_tail_ragged_cohort(
        &self,
        x: &Tensor,
        rope_dim: usize,
        nope: usize,
        seqlen_offsets: &[usize],
        seq_len: usize,
    ) -> Result<Tensor> {
        let (cos, _sin, neg_sin) = self.cohort_tables(seqlen_offsets, seq_len)?;
        let x_pe = x.narrow(3, nope, rope_dim)?.contiguous()?;
        candle_nn::rotary_emb::rope_i(&x_pe, &cos, &neg_sin)
    }

    /// Per-sequence loop arm of [`Self::forward_inverse_tail`], extracted
    /// verbatim (minus the trailing NoPE `cat`, which the caller owns).
    fn inverse_tail_ragged_loop(
        &self,
        x: &Tensor,
        rope_dim: usize,
        nope: usize,
        seqlen_offsets: &[usize],
        seq_len: usize,
    ) -> Result<Tensor> {
        let mut outs = Vec::new();
        for (i, offset) in seqlen_offsets.iter().enumerate() {
            let cos = self.cos.narrow(0, *offset, seq_len)?;
            let sin = self.neg_sin.narrow(0, *offset, seq_len)?;
            let x_pe = x
                .i(i)?
                .unsqueeze(0)?
                .narrow(3, nope, rope_dim)?
                .contiguous()?;
            outs.push(candle_nn::rotary_emb::rope_i(&x_pe, &cos, &sin)?);
        }
        Tensor::cat(&outs, 0)
    }

    // The cohort predicate for this rotary is the free function
    // `uniform_seqlen_offset`, not a method here. #198 introduced
    // `Self::uniform_offset` with a byte-identical body; per that function's
    // doc comment ("there must not be two spellings of the same predicate")
    // the duplicate is collapsed away, and the reasoning it carried has moved
    // onto the survivor.
    fn new_unscaled(cfg: &DeepSeekV2RopeConfig, dtype: DType, dev: &Device) -> Result<Self> {
        let max_seq_len = cfg.max_position_embeddings;
        let dim = cfg.qk_rope_head_dim;

        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_theta.powf(i as f32 / dim as f32))
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;

        let sin = freqs.sin()?.to_dtype(dtype)?;
        let cos = freqs.cos()?.to_dtype(dtype)?;

        Self::from_tables(sin, cos)
    }

    fn yarn_find_correction_dim(
        num_rot: f32,
        dim: usize,
        base: f32,
        max_position_embeddings: usize,
    ) -> f32 {
        (dim as f32 * (max_position_embeddings as f32 / (num_rot * 2. * PI)).ln())
            / (2. * base.ln())
    }

    fn yarn_find_correction_range(
        low_rot: f32,
        high_rot: f32,
        dim: usize,
        base: f32,
        max_position_embeddings: usize,
    ) -> (f32, f32) {
        let low =
            Self::yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings).floor();
        let high =
            Self::yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings).ceil();
        (low.max(0.), high.min(dim as f32 - 1.))
    }

    fn yarn_linear_ramp_mask(min: f32, mut max: f32, dim: usize, dev: &Device) -> Result<Tensor> {
        if min == max {
            // https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite/blob/604d5664dddd88a0433dbae533b7fe9472482de0/modeling_deepseek.py#L255
            max += 0.001;
        }
        let linear_func =
            ((Tensor::arange(0f32, dim as f32, dev)? - min as f64)? / (max as f64 - min as f64))?;
        linear_func.clamp(0., 1)
    }

    pub(crate) fn yarn_get_mscale(scale: f32, mscale: f32) -> f32 {
        if scale <= 1. {
            return 1.;
        }
        0.1 * mscale * scale.ln() + 1.
    }

    #[allow(clippy::too_many_arguments)]
    fn new_yarn(
        cfg: &DeepSeekV2RopeConfig,
        dtype: DType,
        dev: &Device,
        original_max_position_embeddings: usize,
        beta_fast: f32,
        beta_slow: f32,
        factor: f32,
        mscale: f32,
        mscale_all_dim: f32,
    ) -> Result<Self> {
        let freq_extra: Vec<_> = (0..cfg.qk_rope_head_dim)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_theta.powf(i as f32 / cfg.qk_rope_head_dim as f32))
            .collect();
        let freq_extra_len = freq_extra.len();
        let freq_extra = Tensor::from_vec(freq_extra, freq_extra_len, dev)?;
        let freq_inter: Vec<_> = (0..cfg.qk_rope_head_dim)
            .step_by(2)
            .map(|i| 1f32 / (factor * cfg.rope_theta.powf(i as f32 / cfg.qk_rope_head_dim as f32)))
            .collect();
        let freq_inter_len = freq_inter.len();
        let freq_inter = Tensor::from_vec(freq_inter, (1, freq_inter_len), dev)?;

        let (low, high) = Self::yarn_find_correction_range(
            beta_fast,
            beta_slow,
            cfg.qk_rope_head_dim,
            cfg.rope_theta,
            original_max_position_embeddings,
        );
        let inv_freq_mask =
            (1. - Self::yarn_linear_ramp_mask(low, high, cfg.qk_rope_head_dim / 2, dev)?)?;
        let inv_freq = freq_inter
            .broadcast_mul(&(1. - &inv_freq_mask)?)?
            .broadcast_add(&freq_extra.broadcast_mul(&inv_freq_mask)?)?;

        let t = Tensor::arange(0u32, cfg.max_position_embeddings as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((cfg.max_position_embeddings, 1))?;
        let freqs = t.matmul(&inv_freq)?;

        let mscale =
            Self::yarn_get_mscale(factor, mscale) / Self::yarn_get_mscale(factor, mscale_all_dim);
        let sin = (freqs.sin()? * mscale as f64)?.to_dtype(dtype)?;
        let cos = (freqs.cos()? * mscale as f64)?.to_dtype(dtype)?;

        Self::from_tables(sin, cos)
    }

    pub fn new(cfg: &DeepSeekV2RopeConfig, dtype: DType, dev: &Device) -> Result<Self> {
        match &cfg.rope_scaling {
            Some(DeepSeekV2RopeScaling::LinearOrDynamic {
                scaling_type: _,
                factor: _,
            }) => candle_core::bail!("linear and dynamic rope are not implemented yet!"),
            Some(DeepSeekV2RopeScaling::Yarn {
                original_max_position_embeddings,
                beta_fast,
                beta_slow,
                factor,
                mscale,
                mscale_all_dim,
                scaling_type: _,
            }) => Self::new_yarn(
                cfg,
                dtype,
                dev,
                *original_max_position_embeddings,
                *beta_fast,
                *beta_slow,
                *factor,
                *mscale,
                *mscale_all_dim,
            ),
            None => Self::new_unscaled(cfg, dtype, dev),
        }
    }

    /// The full `[max_position_embeddings, rope_dim / 2]` cosine table.
    ///
    /// Exposed for `cuda::qk_norm_rope`, which addresses the table with a base
    /// offset instead of materialising the `narrow` view `forward` takes.
    pub fn cos(&self) -> &Tensor {
        &self.cos
    }

    /// The full `[max_position_embeddings, rope_dim / 2]` sine table. See [`Self::cos`].
    pub fn sin(&self) -> &Tensor {
        &self.sin
    }

    pub fn forward(
        &self,
        q: &Tensor,
        k: &Tensor,
        seqlen_offsets: &[usize],
    ) -> Result<(Tensor, Tensor)> {
        let (_b_sz, _h, seq_len, _n_embd) = q.dims4()?;

        // One `rope_i` over the whole cohort whenever every row rotates at the
        // same position — which is every request today. See
        // [`uniform_seqlen_offset`] for why this used to read `len() == 1`, and
        // what that cost.
        //
        // Bit-identical to the loop below, not merely close: with a 2-D
        // `[T, D/2]` cos/sin the kernel's `stride_b` is 0
        // (`candle-nn/src/rotary_emb.rs`), so every batch row reads the same
        // cos/sin row, and each output element is an independent
        // two-multiply-one-add of its own inputs. Batching changes which
        // launch computes an element, never the arithmetic that produces it.
        if let Some(offset) = uniform_seqlen_offset(seqlen_offsets) {
            rope_cohort_stats::record_cohort();
            let cos = self.cos.narrow(0, offset, seq_len)?;
            let sin = self.sin.narrow(0, offset, seq_len)?;
            let q_embed = candle_nn::rotary_emb::rope_i(&q.contiguous()?, &cos, &sin)?;
            let k_embed = candle_nn::rotary_emb::rope_i(&k.contiguous()?, &cos, &sin)?;
            Ok((q_embed, k_embed))
        } else if rope_cohort_enabled() {
            // Ragged rows, one gathered rope_i per projection — opt-in, see
            // [`rope_cohort_enabled`].
            rope_cohort_stats::record_cohort();
            self.forward_ragged_cohort(q, k, seqlen_offsets, seq_len)
        } else {
            rope_cohort_stats::record_per_sequence();
            self.forward_ragged_loop(q, k, seqlen_offsets, seq_len)
        }
    }

    /// Apply the INVERSE rotation (conjugate: cos, -sin) to the last `rope_dim`
    /// dims of `x`, shaped `[B, H, T, head_dim]`. Used to de-rotate the V4 MLA
    /// attention output, whose value dims carry the key's RoPE (reference
    /// inference/model.py:534 `apply_rotary_emb(o[..., -rd:], freqs_cis, True)`).
    /// RUN-161.
    pub fn forward_inverse_tail(
        &self,
        x: &Tensor,
        rope_dim: usize,
        seqlen_offsets: &[usize],
    ) -> Result<Tensor> {
        let (_b, _h, seq_len, head_dim) = x.dims4()?;
        let nope = head_dim - rope_dim;
        let x_nope = x.narrow(3, 0, nope)?;
        // Same cohort dispatch as `forward`, and the conjugate `sin` now comes
        // from the pre-negated table (see the `neg_sin` field) rather than a
        // `neg` launch per sequence per layer.
        let rotated = if let Some(offset) = uniform_seqlen_offset(seqlen_offsets) {
            rope_cohort_stats::record_cohort();
            let cos = self.cos.narrow(0, offset, seq_len)?;
            let sin = self.neg_sin.narrow(0, offset, seq_len)?;
            let x_pe = x.narrow(3, nope, rope_dim)?.contiguous()?;
            candle_nn::rotary_emb::rope_i(&x_pe, &cos, &sin)?
        } else if rope_cohort_enabled() {
            // Ragged rows, one gathered rope_i — opt-in, see
            // [`rope_cohort_enabled`].
            rope_cohort_stats::record_cohort();
            self.inverse_tail_ragged_cohort(x, rope_dim, nope, seqlen_offsets, seq_len)?
        } else {
            rope_cohort_stats::record_per_sequence();
            self.inverse_tail_ragged_loop(x, rope_dim, nope, seqlen_offsets, seq_len)?
        };
        Tensor::cat(&[&x_nope, &rotated], 3)?.contiguous()
    }

    /// Apply RoPE (adjacent-pair / `rope_i`) to the last `rope_dim` dims of `x`
    /// at the given absolute `positions`, rather than a contiguous range.
    ///
    /// `x`: `[B, H, T, head_dim]`; `positions`: `[T]` (`u32`), one absolute
    /// position per time step. Only the last `rope_dim` dims are rotated; the
    /// leading `head_dim - rope_dim` (NoPE) dims are returned unchanged.
    ///
    /// Used by the V4 compressor: a compressed entry `j` sits at the strided
    /// absolute position `j * ratio`, so the standard contiguous `forward` /
    /// `forward_inverse_tail` (which `narrow`s a window of the cos/sin table)
    /// cannot express it. Reference `inference/model.py` Compressor.forward:
    /// `apply_rotary_emb(kv[..., -rd:], self.freqs_cis[:cutoff:ratio])`.
    pub fn forward_at_positions(
        &self,
        x: &Tensor,
        rope_dim: usize,
        positions: &Tensor,
    ) -> Result<Tensor> {
        let (_b, _h, seq_len, head_dim) = x.dims4()?;
        debug_assert_eq!(positions.dim(0)?, seq_len);
        let nope = head_dim - rope_dim;
        let x_nope = x.narrow(D::Minus1, 0, nope)?;
        let x_pe = x.narrow(D::Minus1, nope, rope_dim)?.contiguous()?;
        // Gather the cos/sin rows for the requested positions: [T, rope_dim/2].
        let positions = positions.to_dtype(DType::U32)?;
        let cos = self.cos.index_select(&positions, 0)?;
        let sin = self.sin.index_select(&positions, 0)?;
        let rotated = candle_nn::rotary_emb::rope_i(&x_pe, &cos, &sin)?;
        Tensor::cat(&[&x_nope, &rotated], D::Minus1)?.contiguous()
    }
}

#[derive(Debug, Clone)]
pub struct Phi4MMRotaryEmbedding {
    short_sin: Tensor,
    short_cos: Tensor,
    long_cos: Option<Tensor>,
    long_sin: Option<Tensor>,
    original_max_position_embeddings: usize,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Phi4MMScaledRopeType {
    #[serde(alias = "longrope")]
    LongRope,
    #[default]
    Default,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Phi4MMRopeScalingConfig {
    short_factor: Option<Vec<f64>>,
    long_factor: Option<Vec<f64>>,
    #[serde(rename = "type")]
    scaling_type: Phi4MMScaledRopeType,
}

impl Phi4MMRotaryEmbedding {
    fn new_unscaled(cfg: &Phi4MMConfig, dtype: DType, dev: &Device) -> Result<Self> {
        let max_seq_len = cfg.max_position_embeddings;
        let dim = (cfg.head_dim() as f64 * cfg.partial_rotary_factor) as usize;

        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_theta.powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        let sin = freqs.sin()?.to_dtype(dtype)?;
        let cos = freqs.cos()?.to_dtype(dtype)?;
        Ok(Self {
            short_cos: cos,
            short_sin: sin,
            long_cos: None,
            long_sin: None,
            original_max_position_embeddings: cfg.original_max_position_embeddings,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn new_longrope(
        short_factor: &[f64],
        long_factor: &[f64],
        cfg: &Phi4MMConfig,
        dtype: DType,
        dev: &Device,
    ) -> Result<Self> {
        let max_seq_len = cfg.max_position_embeddings;
        let dim = (cfg.head_dim() as f64 * cfg.partial_rotary_factor) as usize;

        // Calculate scale
        let scale =
            cfg.max_position_embeddings as f64 / cfg.original_max_position_embeddings as f64;
        let scaling_factor = if scale <= 1.0 {
            1.0
        } else {
            (1.0 + scale.ln() / (cfg.original_max_position_embeddings as f64).ln()).sqrt()
        };

        // Short cos/sin
        let inv_freq_short: Vec<_> = (0..dim)
            .step_by(2)
            .enumerate()
            .map(|(k, i)| {
                1f32 / (short_factor[k] * cfg.rope_theta.powf(i as f64 / dim as f64)) as f32
            })
            .collect();
        let inv_freq_len_short = inv_freq_short.len();
        let inv_freq_short = Tensor::from_vec(inv_freq_short, (1, inv_freq_len_short), dev)?;
        let t_short = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;
        let freqs_short = t_short.matmul(&inv_freq_short)?;
        let sin_short = (freqs_short.sin()?.to_dtype(dtype)? * scaling_factor)?;
        let cos_short = (freqs_short.cos()?.to_dtype(dtype)? * scaling_factor)?;

        // Long cos/sin
        let inv_freq_long: Vec<_> = (0..dim)
            .step_by(2)
            .enumerate()
            .map(|(k, i)| {
                1f32 / (long_factor[k] * cfg.rope_theta.powf(i as f64 / dim as f64)) as f32
            })
            .collect();
        let inv_freq_len_long = inv_freq_long.len();
        let inv_freq_long = Tensor::from_vec(inv_freq_long, (1, inv_freq_len_long), dev)?;
        let t_long = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;
        let freqs_long = t_long.matmul(&inv_freq_long)?;
        let sin_long = (freqs_long.sin()?.to_dtype(dtype)? * scaling_factor)?;
        let cos_long = (freqs_long.cos()?.to_dtype(dtype)? * scaling_factor)?;

        Ok(Self {
            short_cos: cos_short,
            short_sin: sin_short,
            long_cos: Some(cos_long),
            long_sin: Some(sin_long),
            original_max_position_embeddings: cfg.original_max_position_embeddings,
        })
    }

    pub fn new(dtype: DType, cfg: &Phi4MMConfig, dev: &Device) -> Result<Self> {
        match &cfg.rope_scaling {
            Some(Phi4MMRopeScalingConfig {
                scaling_type: Phi4MMScaledRopeType::LongRope,
                short_factor: Some(short_factor),
                long_factor: Some(long_factor),
            }) => Self::new_longrope(short_factor, long_factor, cfg, dtype, dev),

            _ => Self::new_unscaled(cfg, dtype, dev),
        }
    }

    /// Returns (sin, cos) taking into account LongRope
    fn get_long_or_short_sin_cos(&self, position_ids: &[usize]) -> (&Tensor, &Tensor) {
        if self.long_cos.is_none() {
            return (&self.short_sin, &self.short_cos);
        }
        let seq_len = position_ids.iter().max().unwrap() + 1;
        if seq_len > self.original_max_position_embeddings {
            (
                self.long_sin.as_ref().unwrap(),
                self.long_cos.as_ref().unwrap(),
            )
        } else {
            (&self.short_sin, &self.short_cos)
        }
    }

    pub fn forward(
        &self,
        q: &Tensor,
        k: &Tensor,
        seqlen_offsets: &[usize],
        position_ids: &[usize],
    ) -> Result<(Tensor, Tensor)> {
        let (_b_sz, _h, seq_len, _n_embd) = q.dims4()?;
        let (sin, cos) = self.get_long_or_short_sin_cos(position_ids);

        let rot_dim = cos.dim(D::Minus1)? * 2;
        let q_rot = q.narrow(D::Minus1, 0, rot_dim)?;
        let q_pass = q.narrow(D::Minus1, rot_dim, q.dim(D::Minus1)? - rot_dim)?;
        let k_rot = k.narrow(D::Minus1, 0, rot_dim)?;
        let k_pass = k.narrow(D::Minus1, rot_dim, k.dim(D::Minus1)? - rot_dim)?;

        // Batched whenever every row shares an offset -- see
        // `uniform_seqlen_offset`. This used to require `len() == 1`.
        let (q_rot, k_rot) = if let Some(offset) = uniform_seqlen_offset(seqlen_offsets) {
            rope_cohort_stats::record_cohort();
            let cos = cos.narrow(0, offset, seq_len)?;
            let sin = sin.narrow(0, offset, seq_len)?;
            let q_embed = candle_nn::rotary_emb::rope(&q_rot.contiguous()?, &cos, &sin)?;
            let k_embed = candle_nn::rotary_emb::rope(&k_rot.contiguous()?, &cos, &sin)?;
            (q_embed, k_embed)
        } else {
            rope_cohort_stats::record_per_sequence();
            let mut q_embeds = Vec::new();
            let mut k_embeds = Vec::new();
            for (i, offset) in seqlen_offsets.iter().enumerate() {
                let cos = cos.narrow(0, *offset, seq_len)?;
                let sin = sin.narrow(0, *offset, seq_len)?;
                let q_embed = candle_nn::rotary_emb::rope(
                    &q_rot.i(i)?.unsqueeze(0)?.contiguous()?,
                    &cos,
                    &sin,
                )?;
                let k_embed = candle_nn::rotary_emb::rope(
                    &k_rot.i(i)?.unsqueeze(0)?.contiguous()?,
                    &cos,
                    &sin,
                )?;
                q_embeds.push(q_embed);
                k_embeds.push(k_embed);
            }
            let q_rot = Tensor::cat(&q_embeds, 0)?;
            let k_rot = Tensor::cat(&k_embeds, 0)?;
            (q_rot, k_rot)
        };

        Ok((
            Tensor::cat(&[q_rot, q_pass], D::Minus1)?.contiguous()?,
            Tensor::cat(&[k_rot, k_pass], D::Minus1)?.contiguous()?,
        ))
    }
}

#[derive(Debug, Clone)]
pub struct Gemma3nRotaryEmbedding(RotaryEmbedding);

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Gemma3nScaledRopeType {
    #[serde(alias = "linear")]
    Linear,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Gemma3nRopeScalingConfig {
    factor: f64,
    rope_type: Gemma3nScaledRopeType,
}

impl Gemma3nRotaryEmbedding {
    fn new_linear(
        cfg: &Gemma3nTextConfig,
        factor: f64,
        is_gpt_neox: bool,
        dtype: DType,
        dev: &Device,
    ) -> Result<Self> {
        let max_seq_len = cfg.max_position_embeddings;
        let dim = cfg.head_dim;

        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_theta.powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?;
        let inv_freq = (inv_freq / factor)?;

        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        let sin = freqs.sin()?.to_dtype(dtype)?;
        let cos = freqs.cos()?.to_dtype(dtype)?;
        Ok(Self(RotaryEmbedding {
            cos,
            sin,
            is_gpt_neox,
        }))
    }

    pub fn new(
        is_gpt_neox: bool,
        dtype: DType,
        cfg: &Gemma3nTextConfig,
        dev: &Device,
    ) -> Result<Self> {
        match &cfg.rope_scaling {
            Some(Gemma3RopeScalingConfig {
                rope_type: Gemma3ScaledRopeType::Linear,
                factor,
            }) => Self::new_linear(cfg, *factor, is_gpt_neox, dtype, dev),

            _ => Self::new_linear(cfg, 1.0, is_gpt_neox, dtype, dev),
        }
    }

    pub fn get_cos_sin(&self) -> Result<(Tensor, Tensor)> {
        self.0.get_cos_sin()
    }

    pub fn forward(
        &self,
        q: &Tensor,
        k: &Tensor,
        seqlen_offsets: &[usize],
    ) -> Result<(Tensor, Tensor)> {
        self.0.forward(q, k, seqlen_offsets)
    }
}

#[derive(Debug, Clone)]
pub struct Gemma3RotaryEmbedding(RotaryEmbedding);

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Gemma3ScaledRopeType {
    #[serde(alias = "linear")]
    Linear,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Gemma3RopeScalingConfig {
    factor: f64,
    rope_type: Gemma3ScaledRopeType,
}

impl Gemma3RotaryEmbedding {
    fn new_linear(
        cfg: &Gemma3TextConfig,
        factor: f64,
        is_gpt_neox: bool,
        dtype: DType,
        dev: &Device,
    ) -> Result<Self> {
        let max_seq_len = cfg.max_position_embeddings;
        let dim = cfg.head_dim;

        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_theta.powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?;
        let inv_freq = (inv_freq / factor)?;

        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        let sin = freqs.sin()?.to_dtype(dtype)?;
        let cos = freqs.cos()?.to_dtype(dtype)?;
        Ok(Self(RotaryEmbedding {
            cos,
            sin,
            is_gpt_neox,
        }))
    }

    pub fn new(
        is_gpt_neox: bool,
        dtype: DType,
        cfg: &Gemma3TextConfig,
        dev: &Device,
    ) -> Result<Self> {
        match &cfg.rope_scaling {
            Some(Gemma3RopeScalingConfig {
                rope_type: Gemma3ScaledRopeType::Linear,
                factor,
            }) => Self::new_linear(cfg, *factor, is_gpt_neox, dtype, dev),

            _ => Self::new_linear(cfg, 1.0, is_gpt_neox, dtype, dev),
        }
    }

    fn new_linear_embedding_gemma(
        cfg: &EmbeddingGemmaConfig,
        factor: f64,
        is_gpt_neox: bool,
        dtype: DType,
        dev: &Device,
    ) -> Result<Self> {
        let max_seq_len = cfg.max_position_embeddings;
        let dim = cfg.head_dim;

        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_theta.powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?;
        let inv_freq = (inv_freq / factor)?;

        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        let sin = freqs.sin()?.to_dtype(dtype)?;
        let cos = freqs.cos()?.to_dtype(dtype)?;
        Ok(Self(RotaryEmbedding {
            cos,
            sin,
            is_gpt_neox,
        }))
    }

    pub fn new_embedding_gemma(
        is_gpt_neox: bool,
        dtype: DType,
        cfg: &EmbeddingGemmaConfig,
        dev: &Device,
    ) -> Result<Self> {
        match &cfg.rope_scaling {
            Some(Gemma3RopeScalingConfig {
                rope_type: Gemma3ScaledRopeType::Linear,
                factor,
            }) => Self::new_linear_embedding_gemma(cfg, *factor, is_gpt_neox, dtype, dev),

            _ => Self::new_linear_embedding_gemma(cfg, 1.0, is_gpt_neox, dtype, dev),
        }
    }

    pub fn forward(
        &self,
        q: &Tensor,
        k: &Tensor,
        seqlen_offsets: &[usize],
    ) -> Result<(Tensor, Tensor)> {
        self.0.forward(q, k, seqlen_offsets)
    }
}

pub struct DiaRotaryEmbedding {
    timescale: Tensor,
    dtype: DType,
}

impl DiaRotaryEmbedding {
    pub fn new(
        min_timescale: f32,
        max_timescale: f32,
        head_dim: usize,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        assert_eq!(head_dim % 2, 0);
        let half_embedding_dim = head_dim / 2;

        let fraction = (0..half_embedding_dim).map(|i| 2f32 * i as f32 / head_dim as f32);
        let timescale = fraction
            .into_iter()
            .map(|x| min_timescale * (max_timescale / min_timescale).powf(x))
            .collect::<Vec<_>>();

        let timescale_len = timescale.len();
        let timescale = Tensor::from_vec(timescale, timescale_len, device)?;

        Ok(Self { timescale, dtype })
    }

    pub fn forward(&self, xs: &Tensor, positions: &Tensor) -> Result<Tensor> {
        let freqs = positions
            .unsqueeze(D::Minus1)?
            .unsqueeze(D::Minus1)?
            .broadcast_div(&self.timescale)?;

        let sin = freqs.sin()?.to_dtype(self.dtype)?;
        let cos = freqs.cos()?.to_dtype(self.dtype)?;

        let split = xs.chunk(2, D::Minus1)?;
        let first_half = &split[0];
        let second_half = &split[1];

        let first_part = (first_half.broadcast_mul(&cos)? - second_half.broadcast_mul(&sin)?)?;
        let second_part = (second_half.broadcast_mul(&cos)? + first_half.broadcast_mul(&sin)?)?;

        Tensor::cat(&[first_part, second_part], D::Minus1)
    }
}
#[derive(Debug, Clone)]
pub struct QLinear {
    inner: QMatMul,
    bias: Option<Tensor>,
    dtype: DType,
}

impl QLinear {
    pub fn new<R: std::io::Read + std::io::Seek>(
        ct: &mut Content<'_, R>,
        name: &str,
        device: &Device,
    ) -> Result<Self> {
        let w = ct.tensor(&format!("{name}.weight"), device)?;
        let b = ct.tensor(&format!("{name}.bias"), device)?;
        let inner = QMatMul::from_qtensor(w)?;
        let bias = b.dequantize(device)?;
        Ok(Self {
            inner,
            bias: Some(bias),
            dtype: DType::F32,
        })
    }

    pub fn from_linear(linear: Linear) -> Self {
        Self {
            inner: QMatMul::Tensor(linear.weight().clone()),
            bias: linear.bias().cloned(),
            dtype: linear.weight().dtype(),
        }
    }

    pub fn from_parts(w: Tensor, b: Option<Tensor>) -> Self {
        let dtype = w.dtype();
        Self {
            inner: QMatMul::Tensor(w),
            bias: b,
            dtype,
        }
    }

    pub fn from_qparts(w: QTensor, b: Option<Tensor>) -> Self {
        if let Some(ref b) = b {
            assert_eq!(b.dtype(), DType::F32);
        }
        Self {
            inner: QMatMul::QTensor(Arc::new(w)),
            bias: b,
            dtype: DType::F32,
        }
    }

    pub fn from_old_and_qmatmul(inner: QMatMul, old: &Self) -> Self {
        Self {
            inner,
            bias: old.bias.clone(),
            dtype: old.dtype,
        }
    }

    pub fn inner(&mut self) -> &mut QMatMul {
        &mut self.inner
    }

    pub fn inner_ref(&self) -> &QMatMul {
        &self.inner
    }

    pub fn is_quant(&self) -> bool {
        matches!(self.inner, QMatMul::QTensor(_))
    }

    pub fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }

    pub fn bias_mut(&mut self) -> Option<&mut Tensor> {
        self.bias.as_mut()
    }
}

impl Module for QLinear {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = if self.is_quant() {
            xs.to_dtype(DType::F32)?
        } else {
            xs.clone()
        };
        if let Some(bias) = &self.bias {
            self.inner
                .forward(&xs)?
                .broadcast_add(bias)?
                .to_dtype(self.dtype)
        } else {
            self.inner.forward(&xs)?.to_dtype(self.dtype)
        }
    }
}

// Cached strided-position tables for the V4 compressor, one per
// `(ratio, device)`. Keyed that way because V4 mixes compression ratios across
// layers (CSA and HCA); a single-slot cache would thrash and reintroduce a host
// copy on every layer. See `compress_positions` for the rationale.
std::thread_local! {
    static COMPRESS_POSITIONS: std::cell::RefCell<Vec<(usize, Tensor)>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// Round a needed length up so the table is (re)built rarely — and therefore
/// during warmup rather than during capture.
const COMPRESS_POS_CHUNK: usize = 1024;

/// `[t_c]` U32 tensor holding `[0, ratio, 2*ratio, …, (t_c-1)*ratio]`, served
/// as a zero-copy view into a cached device table.
///
/// # Why this is not a `Tensor::arange`: that is a graph-capture HAZARD
///
/// `Tensor::arange` (and `from_vec`/`new`/`from_slice`) builds its data as a
/// transient host `Vec` and uploads it through `CudaDevice::clone_htod`, which
/// issues an **async** `cuMemcpyHtoDAsync` and returns. Outside capture that is
/// merely a host round trip. **Inside `cuStreamBeginCapture` it is a
/// correctness bug**: the copy is not executed, it is *recorded* — the graph
/// stores the HOST POINTER and re-reads it on the first launch and on every
/// replay. The `Vec` is freed as soon as the expression returns, so the graph
/// copies freed host memory into the tensor.
///
/// That is exactly how V4 decode capture died: `compressed_kv_from_rows` built
/// its strided positions with `arange`, and the first `cuGraphLaunch` filled
/// them with garbage, so the very next op —
/// `self.cos.index_select(&positions, 0)` in
/// `DeepSeekV2RotaryEmbedding::forward_at_positions` — tripped candle's
/// device-side bounds assert (`ids[id_i] < src_dim_size`, `T = __nv_bfloat16`
/// for the bf16 cos table, `I = unsigned int` for the U32 positions) and the
/// process took SIGSEGV. candle already documents this failure mode for kernel
/// dims/strides and works around it in `CudaDevice::htod_info` by leaking the
/// host source while capturing; nothing protects `clone_htod`, which is what
/// `arange` uses.
///
/// The table is a pure function of `(ratio, capacity)` and its values never
/// change, so it is built **once, outside capture** (warmup steps run long
/// before `begin_capture`) and every decode step then takes a `narrow` view of
/// it. No host memory is touched inside the captured region, and the base
/// device address is stable across replays.
pub fn compress_positions(t_c: usize, ratio: usize, device: &Device) -> Result<Tensor> {
    if ratio == 0 {
        candle_core::bail!("compress_positions: ratio must be non-zero");
    }
    COMPRESS_POSITIONS.with(|c| {
        let mut cache = c.borrow_mut();
        let hit = cache.iter().position(|(r, t)| {
            *r == ratio && t.device().same_device(device) && t.dim(0).unwrap_or(0) >= t_c
        });
        let idx = match hit {
            Some(i) => i,
            None => {
                // Drop any smaller/stale table for this (ratio, device) so the
                // cache cannot grow without bound.
                cache.retain(|(r, t)| !(*r == ratio && t.device().same_device(device)));
                let cap = t_c.div_ceil(COMPRESS_POS_CHUNK) * COMPRESS_POS_CHUNK;
                // The one `arange` — deliberately here and not in the forward.
                let table = Tensor::arange_step(0u32, (cap * ratio) as u32, ratio as u32, device)?;
                cache.push((ratio, table));
                cache.len() - 1
            }
        };
        cache[idx].1.narrow(0, 0, t_c)
    })
}

/// Thread-local GPU positions tensor for CUDA graph mode.
/// When set, `RotaryEmbedding::forward()` uses GPU-side gather instead of
/// CPU-side `narrow()`, making the forward pass graph-capture compatible.
///
/// Set this before calling the model's forward pass during graph capture.
/// Clear it after capture completes.
#[cfg(feature = "cuda")]
std::thread_local! {
    static GRAPH_MODE_POSITIONS: std::cell::RefCell<Option<Tensor>> = const { std::cell::RefCell::new(None) };
}

/// Set GPU positions for graph-mode RoPE. Call before graph-captured forward pass.
///
/// ⚠️ Handing a FRESH tensor here is only safe before capture. See
/// [`set_graph_mode_positions_in_place`] for the replay-safe form and why the
/// difference matters.
#[cfg(feature = "cuda")]
pub fn set_graph_mode_positions(positions: Option<Tensor>) {
    GRAPH_MODE_POSITIONS.with(|p| *p.borrow_mut() = positions);
}

/// RUN-161 step 2b — write the decode positions into a buffer at a **stable
/// device address**, allocating it only on first use, and return it.
///
/// # Why a fixed address is the whole point
///
/// `cuStreamBeginCapture` records the device pointers the kernels are launched
/// with. Whatever address the positions tensor had *at capture time* is baked
/// into the graph. The previous call pattern —
/// `set_graph_mode_positions(Some(Tensor::from_vec(..)))` — allocated a **new**
/// tensor every decode step, so on replay the graph read the address of a
/// tensor that had already been dropped: at best stale positions, at worst a
/// freed buffer. That is exactly why `normal.rs` discarded the replayed logits
/// and ran an eager forward for the real ones, and therefore why no token Arc
/// has ever emitted came from a graph replay.
///
/// Writing in place keeps the address the graph baked in valid and current, so
/// a replay reads *this* step's positions. `Tensor::slice_set` writes into the
/// destination's existing storage (`candle-core/src/tensor_cat.rs:246`), which
/// is what makes this an update rather than a reallocation.
///
/// The staging `from_vec` is still a fresh allocation, but a constant-size one:
/// the alloc cache serves it from the warm pool, and it is the *source*, so it
/// is never an address the graph depends on.
#[cfg(feature = "cuda")]
pub fn set_graph_mode_positions_in_place(
    values: &[u32],
    device: &candle_core::Device,
) -> candle_core::Result<Tensor> {
    GRAPH_MODE_POSITIONS.with(|p| {
        let mut slot = p.borrow_mut();
        // Reallocate only when the shape or device actually changes — a
        // reallocation invalidates any graph already captured against it, so it
        // must be rare and deliberate, not per-step.
        let reuse = match slot.as_ref() {
            Some(t) => {
                t.dims() == [values.len()]
                    && t.dtype() == candle_core::DType::U32
                    && t.device().same_device(device)
            }
            None => false,
        };
        if !reuse {
            *slot = Some(Tensor::zeros(
                values.len(),
                candle_core::DType::U32,
                device,
            )?);
        }
        let buf = slot
            .as_ref()
            .expect("graph-mode positions buffer was just ensured");
        let src = Tensor::from_vec(values.to_vec(), values.len(), device)?;
        buf.slice_set(&src, 0, 0)?;
        Ok(buf.clone())
    })
}

/// Non-cuda stub.
#[cfg(not(feature = "cuda"))]
pub fn set_graph_mode_positions_in_place(
    _values: &[u32],
    _device: &candle_core::Device,
) -> candle_core::Result<Tensor> {
    candle_core::bail!("graph-mode positions require the cuda feature")
}

/// RUN-161 step 2b — the token IDs half of the static input set.
#[cfg(feature = "cuda")]
std::thread_local! {
    static GRAPH_MODE_INPUT_IDS: std::cell::RefCell<Option<Tensor>> = const { std::cell::RefCell::new(None) };
}

/// Copy `input_ids` into a buffer at a **stable device address** and return it.
///
/// Same reasoning as [`set_graph_mode_positions_in_place`]: the captured graph
/// bakes the pointer it was launched with, so every decode step must present
/// the token IDs at the *same* address or a replay reads a dropped tensor.
/// Positions alone are not enough — a replay that reads this step's positions
/// and last step's token IDs is still wrong, just less obviously.
///
/// The copy is device-to-device (`input_ids` is already on the GPU), so this
/// adds no host sync — unlike the `Tensor::from_vec` in the positions path,
/// which has to cross from the host.
#[cfg(feature = "cuda")]
pub fn set_graph_mode_input_ids_in_place(input_ids: &Tensor) -> candle_core::Result<Tensor> {
    let src = input_ids.contiguous()?;
    GRAPH_MODE_INPUT_IDS.with(|p| {
        let mut slot = p.borrow_mut();
        let reuse = match slot.as_ref() {
            Some(t) => {
                t.dims() == src.dims()
                    && t.dtype() == src.dtype()
                    && t.device().same_device(src.device())
            }
            None => false,
        };
        if !reuse {
            *slot = Some(Tensor::zeros(src.dims(), src.dtype(), src.device())?);
        }
        let buf = slot
            .as_ref()
            .expect("graph-mode input-ids buffer was just ensured");
        buf.slice_set(&src, 0, 0)?;
        Ok(buf.clone())
    })
}

/// Non-cuda stub.
#[cfg(not(feature = "cuda"))]
pub fn set_graph_mode_input_ids_in_place(_input_ids: &Tensor) -> candle_core::Result<Tensor> {
    candle_core::bail!("graph-mode input ids require the cuda feature")
}

/// Thread-local additive length mask for graph-mode fixed-capacity attention:
/// `[B, 1, 1, capacity]`, 0 for valid (`slot <= position`) slots and `-inf` for
/// the unwritten tail. Built by [`graph_mode_length_mask`] from the device-held
/// position, so a CUDA-graph replay that mutates the position buffer in place
/// gets the right mask without recapture. (RUN-161)
///
/// 🔴 An unset mask is **not** a benign default. The fixed-capacity decode arm
/// reads a constant `sliding_window`-wide window of which only `position + 1`
/// slots have been written; the rest are the `Tensor::zeros` the buffer was
/// allocated with. **A zero K row is not a masked row** — it scores logit 0 and
/// takes `exp(0)/Z` of the softmax weight, so it dilutes every real key and
/// contributes its zero V to the output. The arm therefore now *refuses* to run
/// with the mask unset (`models/deepseek4.rs`) rather than returning a finite,
/// wrong answer.
#[cfg(feature = "cuda")]
std::thread_local! {
    static GRAPH_MODE_MASK: std::cell::RefCell<Option<Tensor>> = const { std::cell::RefCell::new(None) };
}

#[cfg(feature = "cuda")]
pub fn set_graph_mode_mask(mask: Option<Tensor>) {
    GRAPH_MODE_MASK.with(|m| *m.borrow_mut() = mask);
}

// Per-layer output handles, for bisecting a CUDA-graph replay against eager.
//
// The same-step probe proved the recorded graph is wired correctly (max|Δ| =
// 0 on the capture step), so a divergence one step later is a per-step input
// that must advance and does not. This finds WHICH one by finding the first
// layer whose output differs.
//
// The mechanism relies on a property of capture: the tensors a captured
// forward produces are ordinary warm-pool buffers, and the recorded kernels
// write to **those addresses** on every replay. Holding a clone of each layer
// output therefore gives a live window onto the graph's own intermediates —
// after a replay, these handles contain the REPLAY's values. Holding them
// also keeps them out of the allocator's free list, so the eager forward that
// follows cannot land on top of them.
//
// Handles only: `push` clones a `Tensor` (an `Arc` bump), so tracing costs no
// device work and is safe to run inside the capture region.
thread_local! {
    static ARC_LAYER_TRACE: std::cell::RefCell<Option<Vec<Tensor>>> =
        const { std::cell::RefCell::new(None) };
}

/// Start recording per-layer outputs, discarding any previous recording.
pub fn arc_layer_trace_begin() {
    ARC_LAYER_TRACE.with(|t| *t.borrow_mut() = Some(Vec::new()));
}

/// Stop recording and take what was recorded.
pub fn arc_layer_trace_take() -> Option<Vec<Tensor>> {
    ARC_LAYER_TRACE.with(|t| t.borrow_mut().take())
}

/// Record one layer's output. No-op unless a trace is open, so the call sites
/// in the model cost a thread-local read on the hot path and nothing else.
pub fn arc_layer_trace_push(xs: &Tensor) {
    ARC_LAYER_TRACE.with(|t| {
        if let Some(v) = t.borrow_mut().as_mut() {
            v.push(xs.clone());
        }
    });
}

#[cfg(feature = "cuda")]
pub fn graph_mode_mask() -> Option<Tensor> {
    GRAPH_MODE_MASK.with(|m| m.borrow().clone())
}

#[cfg(not(feature = "cuda"))]
pub fn set_graph_mode_mask(_mask: Option<Tensor>) {}
#[cfg(not(feature = "cuda"))]
pub fn graph_mode_mask() -> Option<Tensor> {
    None
}

// Cached `[1, capacity]` F32 slot-index vector for `graph_mode_length_mask`.
//
// `Tensor::arange` on a GPU device is a host-to-device copy — banned in a hot
// loop by `CLAUDE.md`, and worse than merely slow here: an H2D copy from
// pageable host memory **cannot be recorded into a CUDA graph**, so building
// the arange inside the capture region would abort the capture. Building it
// once, on the warmup steps that precede capture, keeps the per-step work to
// device-side compare/select kernels.
//
// Keyed by capacity *and* device: a rebuild is only needed when either moves,
// which for one loaded model is never after the first step.
#[allow(clippy::type_complexity)]
std::thread_local! {
    static GRAPH_MODE_SLOT_IDS: std::cell::RefCell<Option<(usize, Tensor)>> =
        const { std::cell::RefCell::new(None) };
}

/// The additive length mask the fixed-capacity graph-decode window needs.
///
/// * `positions` — the device-held `[B]` absolute position of each row's new
///   token, i.e. the slot `write_kv_inplace` just wrote it to.
/// * `capacity` — the constant window width the arm reads back
///   (`sliding_window`), which is what makes the step shape-invariant.
///
/// Returns `[B, 1, 1, capacity]` in `dtype`: `0` on slots `0..=position`
/// (written, real K/V) and `-inf` on `position+1..capacity` (never written).
/// `dsv4_attention` folds it into its own union mask through
/// `compose_caller_mask`, whose raw half is exactly `capacity` columns wide.
///
/// Every operation is device-side and derived from `positions`, which is what
/// makes it replay-safe: a captured graph replays the compare against whatever
/// the position buffer holds at replay time. A mask computed on the host before
/// capture would be a constant baked into the graph and would be wrong on the
/// second and every later token.
///
/// Not `#[cfg(feature = "cuda")]`: the arithmetic is device-agnostic, so it is
/// exercised on CPU by the tests below and by
/// `models::dsv4_attention`'s end-to-end guard.
pub fn graph_mode_length_mask(positions: &Tensor, capacity: usize, dtype: DType) -> Result<Tensor> {
    if capacity == 0 {
        candle_core::bail!("graph_mode_length_mask: capacity must be non-zero");
    }
    let b = positions.dims1()?;
    let dev = positions.device();

    let slots = GRAPH_MODE_SLOT_IDS.with(|c| -> Result<Tensor> {
        let mut c = c.borrow_mut();
        if let Some((cap, t)) = c.as_ref() {
            if *cap == capacity && t.device().same_device(dev) {
                return Ok(t.clone());
            }
        }
        let t = Tensor::arange(0u32, capacity as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((1, capacity))?;
        *c = Some((capacity, t.clone()));
        Ok(t)
    })?;

    let pos = positions.to_dtype(DType::F32)?.reshape((b, 1))?;
    // `slot <= position`: the diagonal is always valid, so no row is fully
    // masked and the softmax cannot produce NaN.
    let valid = slots.broadcast_le(&pos)?; // [B, capacity], u8
    let zeros = Tensor::zeros((b, capacity), dtype, dev)?;
    let neg_inf = Tensor::full(f32::NEG_INFINITY, (b, capacity), dev)?.to_dtype(dtype)?;
    valid
        .where_cond(&zeros, &neg_inf)?
        .reshape((b, 1, 1, capacity))
}

/// `positions mod capacity` — the RING slot a graph-mode decode writes its new
/// K/V row into — computed entirely on the device.
///
/// # Why the graph KV buffer has to be a ring (RUN-161)
///
/// The graph decode arm reads a constant `capacity`-wide window (slots
/// `0..capacity`) so the launch geometry never moves, but it used to *write* at
/// the ABSOLUTE position. Those agree only while `position < capacity`: from
/// token `capacity` onward the new row was written past the end of the window
/// that is read, so the freshest key was invisible to attention and a stale row
/// was attended in its place. The arm was therefore correct only below
/// `sliding_window`.
///
/// A ring of exactly `capacity` slots is the natural fix, because `capacity`
/// **is** `sliding_window`: the raw branch attends precisely the last
/// `sliding_window` tokens, so a full ring holds exactly the right key set and
/// never holds a key that should have been evicted.
///
/// ## Why permuting the keys is safe
///
/// A ring stores keys out of order. That is invariant for this attention:
///  * softmax over keys is permutation-invariant, and K and V are written to
///    the SAME slot, so key `i` stays paired with value `i`;
///  * V4 rotates K *before* caching, so every stored row already carries its
///    own absolute RoPE — no consumer re-derives a position from column index;
///  * the validity mask stays [`graph_mode_length_mask`]'s `slot <= position`,
///    which is already exactly right for a ring: while the ring is filling
///    (`position < capacity`) slot `s` is written iff `s <= position`, and once
///    it is full every slot is valid and the predicate is universally true.
///
/// ## Why this is not just `position % capacity` on the host
///
/// The host knows the position, but a host-resolved slot becomes a literal in
/// the recorded kernel's arguments and every replay would then write to the
/// capture step's slot — the same defect that made host-resolved RoPE offsets
/// wrong (see `DeepSeekV2RotaryEmbedding::cos_sin_for`). Only `capacity` is
/// baked, and it is a genuine compile-time-constant of the run.
///
/// F32 is exact for every integer below 2^24, and positions are bounded by
/// `max_position_embeddings` (163 840 for V4), so the float round trip is exact.
pub fn graph_ring_slot(positions: &Tensor, capacity: usize) -> Result<Tensor> {
    if capacity == 0 {
        candle_core::bail!("graph_ring_slot: capacity must be non-zero");
    }
    let p = positions.to_dtype(DType::F32)?;
    // `affine`'s multiplier is a constant kernel argument, which is safe to
    // bake: `capacity` is fixed for the life of the run. Only the position is
    // allowed to vary between replays, and it is read from device memory.
    let blocks = p.affine(1.0 / capacity as f64, 0.0)?.floor()?;
    let rem = (&p - (blocks * capacity as f64)?)?;
    // `1.0 / capacity` is exact in binary only when `capacity` is a power of
    // two (V4's `sliding_window` is 128, so it is — but this must not depend on
    // that). Otherwise `p * (1/capacity)` can round just below an integer at
    // `p` an exact multiple of `capacity`, the `floor` loses a block, and the
    // remainder comes out as `capacity` — one slot PAST the window, i.e. an
    // out-of-bounds KV write. The rounding error is bounded by well under one
    // ulp of the quotient, so the remainder is in `[0, capacity]` and a single
    // correction is exact for every input.
    let over = rem.ge(capacity as f64)?.to_dtype(DType::F32)?;
    (rem - (over * capacity as f64)?)?.to_dtype(DType::U32)
}

/// Check if graph-mode positions are set.
#[cfg(feature = "cuda")]
pub fn has_graph_mode_positions() -> bool {
    GRAPH_MODE_POSITIONS.with(|p| p.borrow().is_some())
}

/// Get the graph-mode positions tensor (device [B]). Used as the RoPE position,
/// the KV-write slot, and the attention length mask source. (RUN-161)
#[cfg(feature = "cuda")]
pub fn graph_mode_positions() -> Option<Tensor> {
    GRAPH_MODE_POSITIONS.with(|p| p.borrow().clone())
}

/// Non-cuda stubs so call sites compile without the cuda feature.
#[cfg(not(feature = "cuda"))]
pub fn has_graph_mode_positions() -> bool {
    false
}
#[cfg(not(feature = "cuda"))]
pub fn graph_mode_positions() -> Option<Tensor> {
    None
}

#[derive(Debug, Clone)]
pub struct RotaryEmbedding {
    cos: Tensor,
    sin: Tensor,
    is_gpt_neox: bool,
}

impl RotaryEmbedding {
    pub fn new(
        base: f32,
        head_dim: usize,
        max_position_embeddings: usize,
        device: &Device,
        is_gpt_neox: bool,
        dtype: DType,
    ) -> Result<Self> {
        let inv_freq: Vec<_> = (0..head_dim)
            .step_by(2)
            .map(|i| 1f32 / base.powf(i as f32 / head_dim as f32))
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), device)?;
        let t = Tensor::arange(0u32, max_position_embeddings as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((max_position_embeddings, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        let sin = freqs.sin()?.to_dtype(dtype)?;
        let cos = freqs.cos()?.to_dtype(dtype)?;

        Ok(Self {
            cos,
            sin,
            is_gpt_neox,
        })
    }

    pub fn get_cos_sin(&self) -> Result<(Tensor, Tensor)> {
        Ok((self.cos.clone(), self.sin.clone()))
    }

    pub fn new_partial(
        base: f32,
        rot_dim: usize,
        max_position_embeddings: usize,
        device: &Device,
        is_gpt_neox: bool,
        dtype: DType,
    ) -> Result<Self> {
        let inv_freq: Vec<_> = (0..rot_dim)
            .step_by(2)
            .map(|i| 1f32 / base.powf(i as f32 / rot_dim as f32))
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), device)?;
        let t = Tensor::arange(0u32, max_position_embeddings as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((max_position_embeddings, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        let sin = freqs.sin()?.to_dtype(dtype)?;
        let cos = freqs.cos()?.to_dtype(dtype)?;

        Ok(Self {
            cos,
            sin,
            is_gpt_neox,
        })
    }

    pub fn forward(
        &self,
        q: &Tensor,
        k: &Tensor,
        seqlen_offsets: &[usize],
    ) -> Result<(Tensor, Tensor)> {
        let (b_sz, qh, seq_len, n_embd) = q.dims4()?;
        let (_b_sz, kh, _seq_len, __n_embd) = k.dims4()?;

        let rope = if self.is_gpt_neox {
            candle_nn::rotary_emb::rope
        } else {
            candle_nn::rotary_emb::rope_i
        };

        // CUDA graph mode: use GPU positions tensor for graph-capture-safe RoPE.
        // The gather_rope kernel reads positions from GPU memory (stable address),
        // enabling the WHILE loop to update positions between iterations.
        #[cfg(feature = "cuda")]
        if seq_len == 1 {
            let use_graph_rope = GRAPH_MODE_POSITIONS.with(|p| p.borrow().is_some());
            if use_graph_rope {
                return GRAPH_MODE_POSITIONS.with(|p| {
                    let p = p.borrow();
                    let positions = p.as_ref().unwrap();
                    // Gather cos/sin using GPU positions: index_select on dim 0
                    // positions: [batch] i64 → index into cos/sin: [max_seq_len, rot_dim]
                    let positions_i64 = positions.to_dtype(DType::U32)?;
                    let cos = self.cos.index_select(&positions_i64, 0)?; // [batch, rot_dim]
                    let sin = self.sin.index_select(&positions_i64, 0)?; // [batch, rot_dim]

                    let q_embed = q.transpose(1, 2)?.flatten(0, 1)?; // [batch, heads, head_dim]
                    let k_embed = k.transpose(1, 2)?.flatten(0, 1)?;
                    mistralrs_quant::rotary::apply_rotary_inplace(
                        &q_embed,
                        &k_embed,
                        &cos,
                        &sin,
                        self.is_gpt_neox,
                    )?;
                    let mut q = q_embed
                        .reshape((b_sz, seq_len, qh, n_embd))?
                        .transpose(1, 2)?;
                    let mut k = k_embed
                        .reshape((b_sz, seq_len, kh, n_embd))?
                        .transpose(1, 2)?;
                    if !(cfg!(feature = "flash-attn") || cfg!(feature = "flash-attn-v3")) {
                        q = q.contiguous()?;
                        k = k.contiguous()?;
                    }
                    Ok((q, k))
                });
            }
        }

        // 🔴 This used to read `cfg!(feature = "cuda") && qh == kh`, which
        // switched the fused kernel off for **every GQA model** -- qwen2, qwen3,
        // mistral, gemma2, phi3 and the rest all have `qh != kh` by definition,
        // so they fell through to the `candle_nn` path below.
        //
        // The head-count equality was never a kernel requirement.
        // `mistralrs_quant::rotary::apply_rotary_inplace` takes `num_heads` and
        // `num_kv_heads` as separate arguments and the kernel walks them as two
        // independent loops (`nq = num_heads * rot_dim`, `nk = num_kv_heads *
        // rot_dim`, `kernels/rotary/rotary.cu`); the only shape it insists on is
        // `(num_tokens, head_size)` matching between q and k, which GQA
        // preserves. UNVERIFIED ON HARDWARE -- read from the kernel source, not
        // measured.
        //
        // The `is_cuda()` term is new as well: the old condition tested only the
        // compile-time feature, so a cuda-enabled binary running a model on CPU
        // or Metal walked into `apply_rotary_inplace` and hit its
        // "expects a cuda tensor" bail. That was reachable before only for MHA
        // models; widening the head-count gate would have widened that too.
        if cfg!(feature = "cuda") && q.device().is_cuda() {
            let cos = cohort_rotary_table(&self.cos, seqlen_offsets, seq_len)?;
            let sin = cohort_rotary_table(&self.sin, seqlen_offsets, seq_len)?;

            let q_embed = q.transpose(1, 2)?.flatten(0, 1)?;
            let k_embed = k.transpose(1, 2)?.flatten(0, 1)?;
            mistralrs_quant::rotary::apply_rotary_inplace(
                &q_embed,
                &k_embed,
                &cos,
                &sin,
                self.is_gpt_neox,
            )?;
            let mut q = q_embed
                .reshape((b_sz, seq_len, qh, n_embd))?
                .transpose(1, 2)?;
            let mut k = k_embed
                .reshape((b_sz, seq_len, kh, n_embd))?
                .transpose(1, 2)?;
            if !(cfg!(feature = "flash-attn") || cfg!(feature = "flash-attn-v3")) {
                q = q.contiguous()?;
                k = k.contiguous()?;
            }
            Ok((q, k))
        } else if let Some(offset) = uniform_seqlen_offset(seqlen_offsets) {
            rope_cohort_stats::record_cohort();
            let cos = self.cos.narrow(0, offset, seq_len)?;
            let sin = self.sin.narrow(0, offset, seq_len)?;
            let q_embed = rope(&q.contiguous()?, &cos, &sin)?;
            let k_embed = rope(&k.contiguous()?, &cos, &sin)?;
            Ok((q_embed, k_embed))
        } else {
            rope_cohort_stats::record_per_sequence();
            let mut q_embeds = Vec::new();
            let mut k_embeds = Vec::new();
            for (i, offset) in seqlen_offsets.iter().enumerate() {
                let cos = self.cos.narrow(0, *offset, seq_len)?;
                let sin = self.sin.narrow(0, *offset, seq_len)?;
                let q_embed = rope(&q.i(i)?.unsqueeze(0)?.contiguous()?, &cos, &sin)?;
                let k_embed = rope(&k.i(i)?.unsqueeze(0)?.contiguous()?, &cos, &sin)?;
                q_embeds.push(q_embed);
                k_embeds.push(k_embed);
            }
            Ok((Tensor::cat(&q_embeds, 0)?, Tensor::cat(&k_embeds, 0)?))
        }
    }
}

/// GPT-OSS style rotary embedding with YARN scaling support.
/// Uses chunked/GPT-NeoX style rotation and applies attention scaling.
#[derive(Debug, Clone)]
pub struct GptOssRotaryEmbedding {
    cos: Tensor,
    sin: Tensor,
    #[allow(dead_code)]
    attention_scale: f32,
}

impl GptOssRotaryEmbedding {
    /// Create a new GPT-OSS rotary embedding with YARN scaling.
    ///
    /// # Arguments
    /// * `base` - Base frequency for RoPE
    /// * `head_dim` - Dimension of each attention head
    /// * `max_position_embeddings` - Maximum sequence length
    /// * `factor` - YARN scaling factor
    /// * `original_max_position_embeddings` - Original max positions before scaling
    /// * `beta_fast` - YARN beta_fast parameter
    /// * `beta_slow` - YARN beta_slow parameter
    /// * `truncate` - Whether to truncate correction dimensions
    /// * `device` - Device to create tensors on
    /// * `dtype` - Data type for the embeddings
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        base: f64,
        head_dim: usize,
        max_position_embeddings: usize,
        factor: f64,
        original_max_position_embeddings: usize,
        beta_fast: f64,
        beta_slow: f64,
        truncate: bool,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let dim = head_dim;

        // Compute attention scale: 0.1 * ln(factor) + 1.0 for YARN
        let attention_scale = (0.1 * factor.ln() + 1.0) as f32;

        // Helper: find correction dimension based on number of rotations
        // HF: (dim * log(max_pos / (num_rotations * 2 * pi))) / (2 * log(base))
        let find_correction_dim = |num_rotations: f64| -> f64 {
            (dim as f64
                * (original_max_position_embeddings as f64
                    / (num_rotations * 2.0 * std::f64::consts::PI))
                    .ln())
                / (2.0 * base.ln())
        };

        // Find correction range based on beta_fast and beta_slow
        let mut low = find_correction_dim(beta_fast);
        let mut high = find_correction_dim(beta_slow);
        if truncate {
            low = low.floor();
            high = high.ceil();
        }
        low = low.max(0.0);
        high = high.min((dim - 1) as f64);

        // Compute base inverse frequencies
        let half_dim = dim / 2;
        let inv_freq_extrapolation: Vec<f64> = (0..dim)
            .step_by(2)
            .map(|i| 1.0 / base.powf(i as f64 / dim as f64))
            .collect();
        let inv_freq_interpolation: Vec<f64> =
            inv_freq_extrapolation.iter().map(|f| f / factor).collect();

        // Linear ramp factor over dimension indices
        let inv_freq: Vec<f64> = (0..half_dim)
            .map(|i| {
                let range = if (high - low).abs() < 0.001 {
                    0.001
                } else {
                    high - low
                };
                let linear = (i as f64 - low) / range;
                let ramp = linear.clamp(0.0, 1.0);
                inv_freq_interpolation[i] * ramp + inv_freq_extrapolation[i] * (1.0 - ramp)
            })
            .collect();

        let inv_freq_len = inv_freq.len();
        let inv_freq_tensor = Tensor::from_vec(
            inv_freq.iter().map(|&x| x as f32).collect::<Vec<_>>(),
            (1, inv_freq_len),
            device,
        )?;

        let t = Tensor::arange(0u32, max_position_embeddings as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((max_position_embeddings, 1))?;

        let freqs = t.matmul(&inv_freq_tensor)?;

        // Apply attention scale to sin/cos (matches HF transformers behavior)
        let sin = (freqs.sin()? * attention_scale as f64)?.to_dtype(dtype)?;
        let cos = (freqs.cos()? * attention_scale as f64)?.to_dtype(dtype)?;

        Ok(Self {
            cos,
            sin,
            attention_scale,
        })
    }

    pub fn forward(
        &self,
        q: &Tensor,
        k: &Tensor,
        seqlen_offsets: &[usize],
    ) -> Result<(Tensor, Tensor)> {
        #[allow(unused_variables)]
        let (b_sz, qh, seq_len, n_embd) = q.dims4()?;
        #[allow(unused_variables)]
        let (_b_sz, kh, _seq_len, _n_embd) = k.dims4()?;

        // Use the CUDA kernel when available. The kernel uses is_neox=true for
        // chunked/GPT-NeoX style rotary.
        //
        // 🔴 This used to also require `qh == k.dim(1)?`, i.e. MHA. GPT-OSS is
        // GQA (64 query heads, 8 KV heads), so that condition is false for the
        // only architecture that reaches this code and the kernel was dead.
        // `apply_rotary_inplace` takes `num_kv_heads` separately -- see the
        // longer note on `RotaryEmbedding::forward`. UNVERIFIED ON HARDWARE.
        #[cfg(feature = "cuda")]
        if q.device().is_cuda() {
            let cos = cohort_rotary_table(&self.cos, seqlen_offsets, seq_len)?;
            let sin = cohort_rotary_table(&self.sin, seqlen_offsets, seq_len)?;

            // Reshape for CUDA kernel: [b, h, seq, dim] -> [b*seq, h, dim]
            let q_embed = q.transpose(1, 2)?.flatten(0, 1)?;
            let k_embed = k.transpose(1, 2)?.flatten(0, 1)?;

            // Apply rotary with is_neox=true for chunked style
            mistralrs_quant::rotary::apply_rotary_inplace(&q_embed, &k_embed, &cos, &sin, true)?;

            // Reshape back: [b*seq, h, dim] -> [b, h, seq, dim]
            let mut q = q_embed
                .reshape((b_sz, seq_len, qh, n_embd))?
                .transpose(1, 2)?;
            let mut k = k_embed
                .reshape((b_sz, seq_len, kh, n_embd))?
                .transpose(1, 2)?;

            if !(cfg!(feature = "flash-attn") || cfg!(feature = "flash-attn-v3")) {
                q = q.contiguous()?;
                k = k.contiguous()?;
            }
            return Ok((q, k));
        }

        // CPU fallback using candle_nn's rope (GPT-NeoX/chunked style).
        // Batched whenever every row shares an offset -- see
        // `uniform_seqlen_offset`. This used to require `len() == 1`.
        if let Some(offset) = uniform_seqlen_offset(seqlen_offsets) {
            rope_cohort_stats::record_cohort();
            let cos = self.cos.narrow(0, offset, seq_len)?;
            let sin = self.sin.narrow(0, offset, seq_len)?;
            let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
            let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
            Ok((q_embed, k_embed))
        } else {
            rope_cohort_stats::record_per_sequence();
            let mut q_embeds = Vec::new();
            let mut k_embeds = Vec::new();
            for (i, offset) in seqlen_offsets.iter().enumerate() {
                let cos = self.cos.narrow(0, *offset, seq_len)?;
                let sin = self.sin.narrow(0, *offset, seq_len)?;
                let q_embed =
                    candle_nn::rotary_emb::rope(&q.i(i)?.unsqueeze(0)?.contiguous()?, &cos, &sin)?;
                let k_embed =
                    candle_nn::rotary_emb::rope(&k.i(i)?.unsqueeze(0)?.contiguous()?, &cos, &sin)?;
                q_embeds.push(q_embed);
                k_embeds.push(k_embed);
            }
            Ok((Tensor::cat(&q_embeds, 0)?, Tensor::cat(&k_embeds, 0)?))
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Deserialize, Serialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Activation {
    #[default]
    #[serde(alias = "gelu")]
    Gelu,
    #[serde(alias = "gelu_new")]
    NewGelu,
    Relu,
    Relu2,
    Relu6,
    Silu,
    Sigmoid,
    HardSigmoid,
    Swiglu,
    Swish,
    HardSwish,
    Elu(f64),
    LeakyRelu(f64),
    #[serde(alias = "gelu_pytorch_tanh")]
    GeluPytorchTanh,
    QuickGelu,
    /// dReLU activation from Turbo Sparse (Song et al., SJTU-IPADS 2024).
    /// `dReLU(x) = ReLU(gate) * ReLU(up)` where the input tensor is `[..., 2H]`
    /// concatenated as `[gate, up]` along the last dim. Drop-in for SwiGLU on
    /// MoE FFN paths to push activation sparsity from ~75% to ~95%, with quality
    /// matching or exceeding SwiGLU after fine-tune.
    /// See: `research/01_weight_compression/moe_lrd_low_rank_decomposition_2024.pdf`
    DRelu,
}

impl Module for Activation {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Gelu => xs.gelu_erf(),
            // https://github.com/huggingface/transformers/blob/12f043eaeaabfef6f6efea411d98e6f6d3c094b7/src/transformers/activations.py#L49-L78
            Self::NewGelu => xs.gelu(),
            Self::Relu => xs.relu(),
            Self::Relu2 => xs.relu()?.sqr(),
            Self::Relu6 => xs.clamp(0f32, 6f32),
            Self::Silu => xs.silu(),
            Self::Sigmoid => candle_nn::ops::sigmoid(xs),
            Self::HardSigmoid => candle_nn::ops::hard_sigmoid(xs),
            Self::Swiglu => candle_nn::ops::swiglu(xs),
            Self::Swish => xs * candle_nn::ops::sigmoid(xs)?,
            Self::HardSwish => xs * candle_nn::ops::hard_sigmoid(xs)?,
            &Self::Elu(alpha) => xs.elu(alpha),
            &Self::LeakyRelu(negative_slope) => candle_nn::ops::leaky_relu(xs, negative_slope),
            Self::GeluPytorchTanh => xs.gelu(),
            Self::QuickGelu => xs * candle_nn::ops::sigmoid(&(xs * 1.702f64)?),
            Self::DRelu => {
                // Split last dim into (gate, up); ReLU each; multiply.
                let last = xs.dim(candle_core::D::Minus1)?;
                if last % 2 != 0 {
                    candle_core::bail!(
                        "DRelu requires last dim to be even (concat of gate||up), got {last}"
                    );
                }
                let half = last / 2;
                let gate = xs.narrow(candle_core::D::Minus1, 0, half)?;
                let up = xs.narrow(candle_core::D::Minus1, half, half)?;
                gate.relu()?.mul(&up.relu()?)
            }
        }
    }
}

impl TryInto<candle_nn::Activation> for Activation {
    type Error = candle_core::Error;

    fn try_into(self) -> Result<candle_nn::Activation> {
        match self {
            Self::Gelu => Ok(candle_nn::Activation::Gelu),
            Self::Relu => Ok(candle_nn::Activation::Relu),
            Self::Silu => Ok(candle_nn::Activation::Silu),
            Self::NewGelu => Ok(candle_nn::Activation::NewGelu),
            Self::Relu2 => Ok(candle_nn::Activation::Relu2),
            Self::Relu6 => Ok(candle_nn::Activation::Relu6),
            Self::Sigmoid => Ok(candle_nn::Activation::Sigmoid),
            Self::HardSigmoid => Ok(candle_nn::Activation::HardSigmoid),
            Self::Swiglu => Ok(candle_nn::Activation::Swiglu),
            Self::Swish => Ok(candle_nn::Activation::Swish),
            Self::HardSwish => Ok(candle_nn::Activation::HardSwish),
            Self::Elu(x) => Ok(candle_nn::Activation::Elu(x)),
            Self::LeakyRelu(x) => Ok(candle_nn::Activation::LeakyRelu(x)),
            Self::GeluPytorchTanh => Ok(candle_nn::Activation::GeluPytorchTanh),
            Self::QuickGelu => candle_core::bail!("No mapping to candle_nn for QuickGelu"),
            Self::DRelu => candle_core::bail!("No mapping to candle_nn for DRelu (Arc-specific)"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Conv3dConfig {
    pub padding: usize,
    pub stride: usize,
    pub dilation: usize,
    pub groups: usize,
}

impl Default for Conv3dConfig {
    fn default() -> Self {
        Self {
            padding: 0,
            stride: 1,
            dilation: 1,
            groups: 1,
        }
    }
}

pub struct Conv3dNoBias {
    conv2d_1: Conv2d,
    conv2d_2: Conv2d,
}

impl Conv3dNoBias {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_sizes: [usize; 3],
        cfg: Conv3dConfig,
        vb: ShardedVarBuilder,
    ) -> Result<Self> {
        let expected_shape = (
            out_channels,
            in_channels / cfg.groups,
            kernel_sizes[0],
            kernel_sizes[1],
            kernel_sizes[2],
        );
        // MLX format has channels-last: (out, temporal, h, w, in)
        // PyTorch format has channels-first: (out, in, temporal, h, w)
        let mlx_shape = (
            out_channels,
            kernel_sizes[0],
            kernel_sizes[1],
            kernel_sizes[2],
            in_channels / cfg.groups,
        );
        let ws = if vb.contains_tensor("weight") {
            // Try to load with expected shape first, if it fails try MLX shape and permute
            match vb.get(expected_shape, "weight") {
                Ok(ws) => ws,
                Err(_) => {
                    // Try MLX format and permute from (out, t, h, w, in) to (out, in, t, h, w)
                    let ws = vb.get(mlx_shape, "weight")?;
                    ws.permute((0, 4, 1, 2, 3))?
                }
            }
        } else {
            vb.get(expected_shape, "weight")?
        };

        // Split on temporal dimension
        // https://github.com/pytorch/pytorch/issues/139066

        let w1 = ws.i((.., .., 0, .., ..))?;
        let w2 = ws.i((.., .., 1, .., ..))?;

        let cfg = Conv2dConfig {
            padding: cfg.padding,
            stride: cfg.stride,
            dilation: cfg.dilation,
            groups: cfg.groups,
            cudnn_fwd_algo: None,
        };

        Ok(Self {
            conv2d_1: Conv2d::new(w1.contiguous()?, None, cfg),
            conv2d_2: Conv2d::new(w2.contiguous()?, None, cfg),
        })
    }

    pub fn weight(&self) -> Result<Tensor> {
        let w1 = self.conv2d_1.weight().clone().unsqueeze(2)?;
        let w2 = self.conv2d_2.weight().clone().unsqueeze(2)?;
        Tensor::cat(&[w1, w2], 2)
    }
}

impl Module for Conv3dNoBias {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs1 = xs.i((.., .., 0, .., ..))?;
        let xs2 = xs.i((.., .., 1, .., ..))?;

        (Convolution.forward_2d(&self.conv2d_1, &xs1)?
            + Convolution.forward_2d(&self.conv2d_2, &xs2)?)?
        .unsqueeze(2)
    }
}

pub trait TensorInfExtend {
    fn is_inf(&self) -> Result<Self>
    where
        Self: Sized;
    fn any(&self) -> Result<bool>;
}

impl TensorInfExtend for Tensor {
    fn is_inf(&self) -> Result<Self> {
        self.broadcast_eq(&Tensor::new(f32::INFINITY, self.device())?.to_dtype(self.dtype())?)
    }

    fn any(&self) -> Result<bool> {
        let sum = self.sum_all()?;
        match self.dtype() {
            DType::U8 => Ok(sum.to_scalar::<u8>()? == 0),
            DType::U32 => Ok(sum.to_scalar::<u32>()? == 0),
            DType::I16 => Ok(sum.to_scalar::<i16>()? == 0),
            DType::I32 => Ok(sum.to_scalar::<i32>()? == 0),
            DType::I64 => Ok(sum.to_scalar::<i64>()? == 0),
            DType::F16 => Ok(sum.to_scalar::<half::f16>()? == half::f16::from_f32_const(0.)),
            DType::BF16 => Ok(sum.to_scalar::<half::bf16>()? == half::bf16::from_f32_const(0.)),
            DType::F32 => Ok(sum.to_scalar::<f32>()? == 0.),
            DType::F64 => Ok(sum.to_scalar::<f64>()? == 0.),
            DType::F8E4M3 => Ok(sum.to_scalar::<F8E4M3>()? == F8E4M3::ZERO),
            DType::F4 | DType::F6E3M2 | DType::F6E2M3 | DType::F8E8M0 => {
                candle_core::bail!("f4/f6e3m2/f6e2m3/f8e8m0 tensors are not supported with .any")
            }
        }
    }
}

pub fn clamp_for_f16(xs: &Tensor) -> Result<Tensor> {
    let mut max = match xs.dtype() {
        DType::U8 => u8::MAX as f32 - 1000.,
        DType::U32 => u32::MAX as f32 - 1000.,
        DType::I16 => i16::MAX as f32 - 1000.,
        DType::I32 => i32::MAX as f32 - 1000.,
        DType::I64 => i64::MAX as f32 - 1000.,
        DType::F16 => half::f16::MAX.to_f32_const() - 1000.,
        DType::BF16 => half::bf16::MAX.to_f32_const() - 1000.,
        DType::F32 => f32::MAX - 1000.,
        DType::F64 => f64::MAX as f32 - 1000.,
        DType::F8E4M3 => F8E4M3::MAX.to_f32() - 1000.,
        DType::F4 | DType::F6E3M2 | DType::F6E2M3 | DType::F8E8M0 => {
            candle_core::bail!("f4/f6e3m2/f6e2m3/f8e8m0 tensors are not supported with .any")
        }
    };
    if xs.is_inf()?.any()? {
        max -= 1000.;
    }
    xs.clamp(-max, max)
}

pub struct FloatInfo {
    /// Minimum representable value.
    pub min: f64,
    /// Maximum representable value.
    pub max: f64,
    /// The difference between 1.0 and the next smallest representable float larger than 1.0.
    pub eps: f64,
    pub dtype: DType,
}

pub trait GetFloatInfo {
    fn finfo(&self) -> Result<FloatInfo>;
}

impl GetFloatInfo for DType {
    fn finfo(&self) -> Result<FloatInfo> {
        let finfo = match self {
            Self::BF16 => FloatInfo {
                min: bf16::MIN.to_f64(),
                max: bf16::MAX.to_f64(),
                eps: bf16::EPSILON.to_f64(),
                dtype: DType::BF16,
            },
            Self::F16 => FloatInfo {
                min: f16::MIN.to_f64(),
                max: f16::MAX.to_f64(),
                eps: f16::EPSILON.to_f64(),
                dtype: DType::F16,
            },
            Self::F32 => FloatInfo {
                min: f32::MIN as f64,
                max: f32::MAX as f64,
                eps: f32::EPSILON as f64,
                dtype: DType::F32,
            },
            Self::F64 => FloatInfo {
                min: f64::MIN,
                max: f64::MAX,
                eps: f64::EPSILON,
                dtype: DType::F64,
            },
            Self::F8E4M3 => FloatInfo {
                min: F8E4M3::MIN.to_f64(),
                max: F8E4M3::MAX.to_f64(),
                eps: F8E4M3::EPSILON.to_f64(),
                dtype: DType::F8E4M3,
            },
            other => {
                candle_core::bail!("Expected a float type for `GetFloatInfo`, got {other:?}");
            }
        };
        Ok(finfo)
    }
}

#[derive(Clone)]
pub struct Mlp {
    pub gate: Arc<dyn QuantMethod>,
    pub up: Arc<dyn QuantMethod>,
    pub down: Arc<dyn QuantMethod>,
    act: Activation,
    params: Vec<usize>,
    /// Pre-activation SwiGLU clamp limit. `None` (the default, and the case
    /// for every model other than DeepSeek V4) leaves the fused GLU path
    /// untouched. See [`Mlp::with_swiglu_limit`].
    swiglu_limit: Option<f32>,
}

impl Mlp {
    /// Enable the DeepSeek V4 pre-activation SwiGLU clamp on this MLP.
    ///
    /// V4's **shared** expert is clamped exactly like its routed experts: the
    /// reference constructs it as `DeepseekV2MLP(..., swiglu_limit=getattr(
    /// config, "swiglu_limit", None))` (`srt/models/deepseek_v2.py:613-622`)
    /// and applies the limit at `:318-323` via `silu_and_mul_clamp`, which is
    /// the same `silu_and_mul<true>` device function the routed experts use.
    /// SGLang gives up an entire fusion optimization specifically to keep this
    /// correct (`deepseek_v4.py:1266-1271`: "DeepSeek V4 requires different
    /// clamping for shared and routed experts. Shared experts fusion
    /// optimization is disabled.").
    ///
    /// Audit finding 5: Arc dropped this on the shared expert unconditionally,
    /// on every device and backend.
    pub fn with_swiglu_limit(mut self, limit: f32) -> Self {
        self.swiglu_limit = Some(limit);
        self
    }

    /// The active pre-activation SwiGLU clamp, if any.
    pub fn swiglu_limit(&self) -> Option<f32> {
        self.swiglu_limit
    }

    /// `act(gate) * up`, with the V4 SwiGLU clamp applied pre-activation when
    /// one is configured. Bit-for-bit the old expression when it is not.
    fn gated_act(&self, gate: &Tensor, up: &Tensor) -> Result<Tensor> {
        let Some(limit) = self.swiglu_limit else {
            return crate::ops::mul_and_act(gate, up, self.act);
        };
        let out_dtype = gate.dtype();
        // The shared-expert half of the 86 clamp calls per decode forward. Only
        // the CLAMP is fused here, not the activation: `mul_and_act` below
        // lands in `mistralrs_quant::fused_glu`, whose translation unit is
        // built with `--use_fast_math`, so a no-fast-math kernel cannot
        // reproduce its bits. See `crate::moe::fused_swiglu_clamp_split`.
        let (gate, up) = match crate::moe::fused_swiglu_clamp_split(gate, up, limit, self.act)? {
            Some(pair) => pair,
            None => crate::moe::swiglu_clamp(gate, up, limit)?,
        };
        crate::ops::mul_and_act(&gate, &up, self.act)?.to_dtype(out_dtype)
    }

    pub fn new(
        vb: ShardedVarBuilder,
        hidden_size: usize,
        intermediate_size: usize,
        quantization_config: &Option<QuantizedConfig>,
        hidden_act: Activation,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<Self> {
        Ok(Self {
            gate: ColumnParallelLayer::new(
                hidden_size,
                intermediate_size,
                quantization_config,
                false,
                comm,
                vb.pp("gate_proj"),
            )?,
            up: ColumnParallelLayer::new(
                hidden_size,
                intermediate_size,
                quantization_config,
                false,
                comm,
                vb.pp("up_proj"),
            )?,
            down: RowParallelLayer::new(
                intermediate_size,
                hidden_size,
                quantization_config,
                false,
                comm,
                vb.pp("down_proj"),
            )?,
            act: hidden_act,
            params: vec![hidden_size, intermediate_size],
            swiglu_limit: None,
        })
    }

    pub fn new_merged(
        vb: ShardedVarBuilder,
        hidden_size: usize,
        intermediate_size: usize,
        chunks: usize,
        quantization_config: &Option<QuantizedConfig>,
        hidden_act: Activation,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<Self> {
        assert!(chunks == 2, "Only gate_up_proj merge is supported!");
        let gate_up_projs = ColumnParallelLayer::new_merged(
            hidden_size,
            intermediate_size * 2,
            2,
            quantization_config,
            false,
            comm,
            vb.pp("gate_up_proj"),
        )?;

        Ok(Self {
            gate: gate_up_projs[0].to_owned(),
            up: gate_up_projs[1].to_owned(),
            down: RowParallelLayer::new(
                intermediate_size,
                hidden_size,
                quantization_config,
                false,
                comm,
                vb.pp("down_proj"),
            )?,
            act: hidden_act,
            params: vec![hidden_size, intermediate_size],
            swiglu_limit: None,
        })
    }

    pub fn replicate(
        params: &[usize],
        vb: ShardedVarBuilder,
        act: Activation,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<Self> {
        Self::new(vb, params[0], params[1], &None, act, comm)
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let original_dtype = xs.dtype();
        let mut xs = xs.clone();
        if let Some(t) = self.gate.quantized_act_type() {
            xs = xs.to_dtype(t)?;
        }
        let lhs = self.gate.forward(&xs)?;
        let rhs = self.up.forward(&xs)?;
        let mut res = self.down.forward(&self.gated_act(&lhs, &rhs)?)?;
        if self.gate.quantized_act_type().is_some() {
            res = res.to_dtype(original_dtype)?;
        }
        Ok(res)
    }
}

impl AnyMoeTrainableLayer for Mlp {}

impl MlpLayer for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let original_dtype = xs.dtype();
        let mut xs = xs.clone();
        if let Some(t) = self.gate.quantized_act_type() {
            xs = xs.to_dtype(t)?;
        }
        let lhs = MatMul.qmethod_matmul(&xs, &*self.gate)?;
        let rhs = MatMul.qmethod_matmul(&xs, &*self.up)?;
        let mut res = MatMul.qmethod_matmul(&self.gated_act(&lhs, &rhs)?, &*self.down)?;
        if self.gate.quantized_act_type().is_some() {
            res = res.to_dtype(original_dtype)?;
        }
        Ok(res)
    }
    fn get_isq_layers(&mut self) -> Vec<&mut Arc<dyn QuantMethod>> {
        vec![&mut self.gate, &mut self.up, &mut self.down]
    }
    fn clone(&self) -> Box<dyn MlpLayer> {
        Box::new(Clone::clone(self))
    }
    fn get_params(&self) -> &[usize] {
        &self.params
    }
    fn hidden_act(&self) -> Activation {
        self.act
    }
    // gate, up, down
    fn new_added_delta(&self, deltas: Vec<Option<Tensor>>) -> Result<Box<dyn MlpLayer>> {
        let gate = if let Some(ref delta) = deltas[0] {
            self.gate.add_delta_w(delta)?
        } else {
            self.gate.clone()
        };
        let up = if let Some(ref delta) = deltas[1] {
            self.up.add_delta_w(delta)?
        } else {
            self.up.clone()
        };
        let down = if let Some(ref delta) = deltas[2] {
            self.down.add_delta_w(delta)?
        } else {
            self.down.clone()
        };

        Ok(Box::new(Self {
            gate,
            up,
            down,
            act: self.act,
            params: self.params.clone(),
            swiglu_limit: self.swiglu_limit,
        }))
    }

    fn dtype_device(&self) -> (DType, Device) {
        self.gate.dtype_and_device()
    }
}

pub struct AvgPool2d {
    kernel_size: usize,
    stride: usize,
}

impl AvgPool2d {
    pub fn new(kernel_size: usize, stride: usize) -> Self {
        Self {
            kernel_size,
            stride,
        }
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.avg_pool2d_with_stride(self.kernel_size, self.stride)
    }
}

/// Applies 2D reflection padding to a tensor of shape (N, C, H, W).
///
/// The `padding` argument is a 4-tuple (pad_left, pad_right, pad_top, pad_bottom).
/// For left padding, it reflects the values from column 1 up to pad_left (in reverse order);
/// for right padding, it reflects from the second-to-last column backwards, and similarly for
/// vertical (height) padding.
pub struct ReflectionPad2d {
    padding: (usize, usize, usize, usize),
}

impl ReflectionPad2d {
    pub fn new(padding: (usize, usize, usize, usize)) -> Self {
        Self { padding }
    }
}

impl Module for ReflectionPad2d {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (pad_left, pad_right, pad_top, pad_bottom) = self.padding;

        let (_n, _c, h, w) = xs.dims4()?;

        // --- Horizontal Padding (along width, axis = 3) ---
        // For left padding, we reflect columns 1..=pad_left (in reverse order).
        let left_pad = if pad_left > 0 {
            // Create indices: [pad_left, pad_left-1, ..., 1]
            let indices: Vec<i64> = (1..=pad_left as i64).rev().collect();
            Some(xs.index_select(&Tensor::new(indices, &Device::Cpu)?, 3)?)
        } else {
            None
        };

        // For right padding, we reflect from the right side (excluding the last column).
        let right_pad = if pad_right > 0 {
            // For pad_right == 2, generate indices: [w-2, w-3, ... , w-1-pad_right]
            let start = w as i64 - 2;
            let indices: Vec<i64> = (0..pad_right as i64).map(|i| start - i).collect();
            Some(xs.index_select(&Tensor::new(indices, &Device::Cpu)?, 3)?)
        } else {
            None
        };

        // Concatenate horizontally (along width, dim=3)
        let x_padded_width = match (left_pad, right_pad) {
            (Some(l), Some(r)) => Tensor::cat(&[l, xs.clone(), r], 3)?,
            (Some(l), None) => Tensor::cat(&[l, xs.clone()], 3)?,
            (None, Some(r)) => Tensor::cat(&[xs.clone(), r], 3)?,
            (None, None) => xs.clone(),
        };

        // --- Vertical Padding (along height, axis = 2) ---
        // For top padding, reflect rows 1..=pad_top (in reverse order)
        let top_pad = if pad_top > 0 {
            let indices: Vec<i64> = (1..=pad_top as i64).rev().collect();
            Some(x_padded_width.index_select(&Tensor::new(indices, &Device::Cpu)?, 2)?)
        } else {
            None
        };

        // For bottom padding, reflect from the bottom (excluding the last row)
        let bottom_pad = if pad_bottom > 0 {
            let start = h as i64 - 2;
            let indices: Vec<i64> = (0..pad_bottom as i64).map(|i| start - i).collect();
            Some(x_padded_width.index_select(&Tensor::new(indices, &Device::Cpu)?, 2)?)
        } else {
            None
        };

        // Concatenate vertically (along height, dim=2)
        let x_padded = match (top_pad, bottom_pad) {
            (Some(t), Some(b)) => Tensor::cat(&[t, x_padded_width, b], 2)?,
            (Some(t), None) => Tensor::cat(&[t, x_padded_width], 2)?,
            (None, Some(b)) => Tensor::cat(&[x_padded_width, b], 2)?,
            (None, None) => x_padded_width,
        };

        Ok(x_padded)
    }
}

pub struct ScaledEmbedding {
    scale: f64,
    pub embedding: Tensor,
}

impl ScaledEmbedding {
    pub fn new(scale: f64, embedding: Embedding) -> Self {
        Self {
            scale,
            embedding: embedding.embeddings().clone(),
        }
    }

    pub fn embeddings(&self) -> &Tensor {
        &self.embedding
    }
}

impl Module for ScaledEmbedding {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let embedding = Embedding::new(self.embedding.clone(), self.embedding.dim(D::Minus1)?);
        xs.apply(&embedding)? * self.scale
    }
}

#[cfg(test)]
mod compress_positions_tests {
    use super::*;

    /// The cache replaced `(arange(0..t_c).to_f32() * ratio).to_u32()`. The
    /// values must be bit-identical to that expression, or every compressed
    /// row silently rotates at the wrong absolute position — a quality bug no
    /// test outside this one would catch.
    // Named "strided_range" rather than naming the candle call: the Typos lane
    // allow-lists `arange` as a whole identifier, so it is fine in the doc
    // comment above and in the body below, but not as one word inside a longer
    // snake_case name. Same reason as 745c871dd / e4eb59dfb.
    #[test]
    fn matches_the_strided_range_expression_it_replaced() -> Result<()> {
        let dev = Device::Cpu;
        for ratio in [1usize, 2, 4, 8, 32] {
            for t_c in [1usize, 3, 17, 64] {
                let want = (Tensor::arange(0u32, t_c as u32, &dev)?.to_dtype(DType::F32)?
                    * (ratio as f64))?
                    .to_dtype(DType::U32)?
                    .to_vec1::<u32>()?;
                let got = compress_positions(t_c, ratio, &dev)?.to_vec1::<u32>()?;
                assert_eq!(got, want, "ratio={ratio} t_c={t_c}");
            }
        }
        Ok(())
    }

    /// Growth must reuse the table, not silently rebuild it every step —
    /// rebuilding is the host copy this cache exists to remove, and doing it
    /// inside capture is the crash it exists to prevent.
    #[test]
    fn a_longer_request_within_capacity_reuses_the_same_storage() -> Result<()> {
        let dev = Device::Cpu;
        let ratio = 4usize;
        let first = compress_positions(1, ratio, &dev)?;
        let grown = compress_positions(COMPRESS_POS_CHUNK, ratio, &dev)?;
        // Same backing table => same length after the chunk round-up.
        assert_eq!(first.dim(0)?, 1);
        assert_eq!(grown.dim(0)?, COMPRESS_POS_CHUNK);
        let entries = COMPRESS_POSITIONS.with(|c| c.borrow().len());
        assert_eq!(entries, 1, "one (ratio, device) entry, not one per call");
        // Distinct ratios must NOT evict each other: V4 interleaves CSA and HCA
        // layers, so a single-slot cache would rebuild on every layer.
        let _ = compress_positions(8, 32, &dev)?;
        let _ = compress_positions(8, ratio, &dev)?;
        let entries = COMPRESS_POSITIONS.with(|c| c.borrow().len());
        assert_eq!(entries, 2, "ratios must coexist");
        Ok(())
    }

    #[test]
    fn zero_ratio_is_refused() {
        assert!(compress_positions(4, 0, &Device::Cpu).is_err());
    }

    /// At `ratio == 1` the table IS `[0, 1, …, t_c-1]`, which is exactly what
    /// `Tensor::arange(0u32, t_c as u32, dev)` produces. `dsv4_attention`'s
    /// compressed-branch mask relies on that identity to serve its block
    /// indices from the cache instead of a per-layer host->device upload, so
    /// pin it: a table that disagreed here would shift which compressed blocks
    /// a query attends, silently and with no error.
    ///
    /// The `reshape` mirrors the call site, which reshapes the view to
    /// `(1, t_c)` immediately — a narrowed view that could not reshape would
    /// fail there and nowhere else.
    #[test]
    fn ratio_one_is_the_plain_index_range_the_mask_uses() -> Result<()> {
        let dev = Device::Cpu;
        for t_c in [1usize, 2, 7, 64, 1025] {
            let want = Tensor::arange(0u32, t_c as u32, &dev)?.to_vec1::<u32>()?;
            let got = compress_positions(t_c, 1, &dev)?;
            assert_eq!(got.to_vec1::<u32>()?, want, "t_c={t_c}");
            assert_eq!(got.reshape((1, t_c))?.dims(), &[1, t_c], "t_c={t_c}");
        }
        Ok(())
    }
}

#[cfg(test)]
mod graph_mode_mask_tests {
    use super::*;

    /// The mask's whole job: separate slots that were written from slots that
    /// were merely allocated. `0` is additive-neutral; `-inf` removes the
    /// column from the softmax entirely. **Not** a small negative — a large
    /// finite penalty still leaves an unwritten slot voting.
    #[test]
    fn valid_slots_are_neutral_and_the_unwritten_tail_is_negative_infinity() -> Result<()> {
        let dev = Device::Cpu;
        let cap = 8usize;
        for position in 0..cap {
            let positions = Tensor::from_vec(vec![position as u32], (1,), &dev)?;
            let mask = graph_mode_length_mask(&positions, cap, DType::F32)?;
            assert_eq!(mask.dims(), &[1, 1, 1, cap]);
            let v = mask.flatten_all()?.to_vec1::<f32>()?;
            for (slot, got) in v.iter().enumerate() {
                if slot <= position {
                    assert_eq!(*got, 0.0, "position {position} slot {slot}");
                } else {
                    assert!(
                        got.is_infinite() && got.is_sign_negative(),
                        "position {position} slot {slot}: got {got}"
                    );
                }
            }
        }
        Ok(())
    }

    /// The diagonal is always valid, so no row is fully masked and the softmax
    /// cannot divide by zero. At `position == 0` exactly one slot survives.
    #[test]
    fn no_row_is_ever_fully_masked() -> Result<()> {
        let dev = Device::Cpu;
        let positions = Tensor::from_vec(vec![0u32, 0u32], (2,), &dev)?;
        let mask = graph_mode_length_mask(&positions, 16, DType::F32)?;
        let v = mask.flatten_all()?.to_vec1::<f32>()?;
        for row in 0..2 {
            let live = v[row * 16..(row + 1) * 16]
                .iter()
                .filter(|x| x.is_finite())
                .count();
            assert_eq!(live, 1, "row {row} must keep exactly its own written slot");
        }
        Ok(())
    }

    /// The cached slot-index vector must not leak across capacities. It is
    /// keyed by `(capacity, device)` precisely so a model reload at a different
    /// window width cannot silently reuse the wrong arange.
    #[test]
    fn the_cached_slot_vector_is_rebuilt_when_the_capacity_changes() -> Result<()> {
        let dev = Device::Cpu;
        let positions = Tensor::from_vec(vec![1u32], (1,), &dev)?;
        assert_eq!(
            graph_mode_length_mask(&positions, 4, DType::F32)?.dims(),
            &[1, 1, 1, 4]
        );
        assert_eq!(
            graph_mode_length_mask(&positions, 9, DType::F32)?.dims(),
            &[1, 1, 1, 9]
        );
        // …and back again, which is the direction a cache keyed only on
        // "is something cached" would get wrong.
        assert_eq!(
            graph_mode_length_mask(&positions, 4, DType::F32)?.dims(),
            &[1, 1, 1, 4]
        );
        Ok(())
    }

    #[test]
    fn a_zero_capacity_window_is_refused_rather_than_producing_an_empty_mask() {
        let dev = Device::Cpu;
        let positions = Tensor::from_vec(vec![0u32], (1,), &dev).unwrap();
        assert!(graph_mode_length_mask(&positions, 0, DType::F32).is_err());
    }
}

/// Bit-exactness and engagement tests for the uniform-offset RoPE cohort path.
///
/// The change under test replaced `seqlen_offsets.len() == 1` -- a test of the
/// *length* of the offset vector -- with [`uniform_seqlen_offset`], a test of
/// the *distinctness of its values*, at every rotary implementation in this
/// file. At B=256 the length is 256, so the batched path was unreachable at
/// every batch size above one, while the values were uniform anyway; the
/// per-sequence loop was therefore performing B bit-identical recomputations
/// and concatenating them back together.
///
/// Each test holds the old loop as a verbatim oracle and asserts the new
/// dispatch reproduces it **bit for bit**, not within a tolerance -- a
/// tolerance-based assertion would be vacuous, since the claim is that no
/// arithmetic changed at all.
#[cfg(test)]
mod rope_cohort_tests {
    use super::*;
    use candle_core::{DType, Device};

    const B: usize = 4;
    const H: usize = 2;
    const T: usize = 3;
    const D_HEAD: usize = 8;
    const MAX_POS: usize = 64;

    fn xs(dims: &[usize], seed: f32) -> Tensor {
        let n: usize = dims.iter().product();
        let v: Vec<f32> = (0..n)
            .map(|i| ((i as f32) * seed).sin() * 1.7 + 0.11)
            .collect();
        Tensor::from_vec(v, dims, &Device::Cpu).unwrap()
    }

    /// Raw IEEE-754 bit patterns. Float equality would accept `-0.0 == 0.0` and
    /// silently pass a sign error; bit patterns do not.
    fn bits(t: &Tensor) -> Vec<u32> {
        t.flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .into_iter()
            .map(f32::to_bits)
            .collect()
    }

    fn rope() -> RotaryEmbedding {
        RotaryEmbedding::new(10000.0, D_HEAD, MAX_POS, &Device::Cpu, true, DType::F32).unwrap()
    }

    /// The pre-fix per-sequence loop, verbatim, as the oracle. Always loops,
    /// whatever the offsets are.
    fn reference(
        re: &RotaryEmbedding,
        q: &Tensor,
        k: &Tensor,
        offsets: &[usize],
    ) -> (Tensor, Tensor) {
        let seq_len = q.dim(2).unwrap();
        let mut q_embeds = Vec::new();
        let mut k_embeds = Vec::new();
        for (i, offset) in offsets.iter().enumerate() {
            let cos = re.cos.narrow(0, *offset, seq_len).unwrap();
            let sin = re.sin.narrow(0, *offset, seq_len).unwrap();
            q_embeds.push(
                candle_nn::rotary_emb::rope(
                    &q.i(i).unwrap().unsqueeze(0).unwrap().contiguous().unwrap(),
                    &cos,
                    &sin,
                )
                .unwrap(),
            );
            k_embeds.push(
                candle_nn::rotary_emb::rope(
                    &k.i(i).unwrap().unsqueeze(0).unwrap().contiguous().unwrap(),
                    &cos,
                    &sin,
                )
                .unwrap(),
            );
        }
        (
            Tensor::cat(&q_embeds, 0).unwrap(),
            Tensor::cat(&k_embeds, 0).unwrap(),
        )
    }

    /// The regression this whole change is about: a B-long vector of one
    /// repeated value IS uniform, and `len() == 1` said it was not.
    #[test]
    fn uniform_offset_tests_values_not_length() {
        assert_eq!(uniform_seqlen_offset(&[7; 256]), Some(7));
        assert_eq!(uniform_seqlen_offset(&[7]), Some(7));
        assert_eq!(uniform_seqlen_offset(&[0, 0, 0]), Some(0));
        // Genuinely ragged rows must still decline, so the loop is preserved.
        assert_eq!(uniform_seqlen_offset(&[7, 7, 8]), None);
        assert_eq!(uniform_seqlen_offset(&[8, 7, 7]), None);
        assert_eq!(uniform_seqlen_offset(&[]), None);
    }

    /// Batching a uniform cohort must be bit-identical to the loop it replaces.
    #[test]
    fn uniform_cohort_is_bit_identical_to_the_loop() {
        let re = rope();
        let q = xs(&[B, H, T, D_HEAD], 0.013);
        let k = xs(&[B, H, T, D_HEAD], 0.019);
        let offsets = vec![5usize; B];

        let before = rope_cohort_stats::local_counts();
        let (qb, kb) = re.forward(&q, &k, &offsets).unwrap();
        let after = rope_cohort_stats::local_counts();

        // A green result must prove work happened: the batched arm has to have
        // been the one that ran.
        assert!(
            after.0 > before.0,
            "cohort path did not engage -- test would pass vacuously"
        );

        let (qref, kref) = reference(&re, &q, &k, &offsets);
        assert_eq!(
            bits(&qb),
            bits(&qref),
            "batched q differs from per-sequence"
        );
        assert_eq!(
            bits(&kb),
            bits(&kref),
            "batched k differs from per-sequence"
        );
    }

    /// Ragged offsets must still take the per-sequence loop and still be right.
    /// A uniform rotation applied to a ragged cohort would be silently wrong.
    #[test]
    fn ragged_cohort_keeps_the_loop_and_stays_per_row() {
        let re = rope();
        let q = xs(&[3, H, T, D_HEAD], 0.013);
        let k = xs(&[3, H, T, D_HEAD], 0.019);
        let offsets = vec![0usize, 4, 9];

        let before = rope_cohort_stats::local_counts();
        let (qb, kb) = re.forward(&q, &k, &offsets).unwrap();
        let after = rope_cohort_stats::local_counts();
        assert!(
            after.1 > before.1,
            "ragged cohort must NOT take the batched path"
        );

        let (qref, kref) = reference(&re, &q, &k, &offsets);
        assert_eq!(bits(&qb), bits(&qref));
        assert_eq!(bits(&kb), bits(&kref));

        // And the rotation must genuinely differ between rows, or the test is
        // vacuous -- it would pass even if every row got offset 0.
        let d01 = (qb.i(0).unwrap() - qb.i(1).unwrap())
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(d01 > 0.0, "rows at different offsets rotated identically");
    }

    /// GPT-J style (`is_gpt_neox == false`, i.e. `rope_i`) takes the same
    /// dispatch, so it gets the same guarantee.
    #[test]
    fn uniform_cohort_is_bit_identical_for_gpt_j_style() {
        let re = RotaryEmbedding::new(10000.0, D_HEAD, MAX_POS, &Device::Cpu, false, DType::F32)
            .unwrap();
        let q = xs(&[B, H, T, D_HEAD], 0.023);
        let k = xs(&[B, H, T, D_HEAD], 0.029);
        let offsets = vec![11usize; B];

        let (qb, kb) = re.forward(&q, &k, &offsets).unwrap();

        let seq_len = T;
        let mut q_embeds = Vec::new();
        let mut k_embeds = Vec::new();
        for (i, offset) in offsets.iter().enumerate() {
            let cos = re.cos.narrow(0, *offset, seq_len).unwrap();
            let sin = re.sin.narrow(0, *offset, seq_len).unwrap();
            q_embeds.push(
                candle_nn::rotary_emb::rope_i(
                    &q.i(i).unwrap().unsqueeze(0).unwrap().contiguous().unwrap(),
                    &cos,
                    &sin,
                )
                .unwrap(),
            );
            k_embeds.push(
                candle_nn::rotary_emb::rope_i(
                    &k.i(i).unwrap().unsqueeze(0).unwrap().contiguous().unwrap(),
                    &cos,
                    &sin,
                )
                .unwrap(),
            );
        }
        assert_eq!(bits(&qb), bits(&Tensor::cat(&q_embeds, 0).unwrap()));
        assert_eq!(bits(&kb), bits(&Tensor::cat(&k_embeds, 0).unwrap()));
    }

    /// `cohort_rotary_table` feeds the CUDA kernel, which indexes
    /// `cos_cache + token_idx * rot_dim` over the b-major flattened token axis
    /// and hard-bails unless the cache is exactly `(b * seq_len, rot_dim)`.
    /// The uniform fast path must therefore *tile*, not narrow -- this is the
    /// assertion that would catch a shape or ordering mistake on a path CPU
    /// tests cannot otherwise reach.
    #[test]
    fn cohort_rotary_table_matches_the_cat_it_replaces() {
        let table = xs(&[MAX_POS, D_HEAD / 2], 0.031);

        for offsets in [
            vec![5usize; B],       // uniform, B > 1 -> must tile
            vec![5usize],          // single row
            vec![0usize, 4, 9],    // ragged -> must keep the cat
            vec![3usize, 3, 3, 7], // trailing mismatch -> ragged
        ] {
            let mut rows = Vec::new();
            for offset in &offsets {
                rows.push(table.narrow(0, *offset, T).unwrap());
            }
            let expected = Tensor::cat(&rows, 0).unwrap();

            let got = cohort_rotary_table(&table, &offsets, T).unwrap();
            assert_eq!(
                got.dims(),
                &[offsets.len() * T, D_HEAD / 2],
                "wrong shape for offsets {offsets:?} -- the kernel bails on this"
            );
            assert_eq!(
                bits(&got),
                bits(&expected),
                "tiled table differs from the per-offset cat for offsets {offsets:?}"
            );
        }
    }
}

/// Bit-exactness and engagement tests for the DeepSeek V2/V4 RoPE cohort path.
///
/// The change under test replaced `seqlen_offsets.len() == 1` — a test of the
/// *length* of the offset vector — with a test of the *distinctness of its
/// values*. At B=256 the length is 256, so the batched path was unreachable at
/// every batch size above one, while the values were uniform anyway; the
/// per-sequence loop was therefore performing B bit-identical recomputations
/// and concatenating them back together (~100,000 GPU launches per decode step
/// on a measured profile, 19.9% of step time).
///
/// Each test below holds the old loop as a verbatim oracle and asserts the new
/// dispatch reproduces it **bit for bit**, not within a tolerance — a
/// tolerance-based assertion here would be vacuous, since the claim is that no
/// arithmetic changed at all.
#[cfg(test)]
mod deepseek_rope_cohort_tests {
    use super::*;

    const MAX_POS: usize = 32;
    const ROPE_DIM: usize = 8;
    const HEAD_DIM: usize = 12;
    const N_HEADS: usize = 3;
    const BATCH: usize = 4;

    // Engagement is asserted through `rope_cohort_stats::local_counts()`, the
    // per-thread mirror of the global counters. #198 originally serialised
    // these tests on a private mutex, which worked only while this module also
    // carried its own private copy of `rope_cohort_stats`; once the two counter
    // modules were collapsed the atomics became shared and a private lock
    // stopped serialising anything that mattered. A shared lock was the next
    // attempt and it is still wrong -- it obliges unrelated tests in other
    // modules to know about it. See `rope_cohort_stats` for why per-thread
    // counting is the fix.

    fn rope() -> DeepSeekV2RotaryEmbedding {
        let cfg = DeepSeekV2RopeConfig {
            rope_scaling: None,
            max_position_embeddings: MAX_POS,
            rope_theta: 10000.0,
            qk_rope_head_dim: ROPE_DIM,
        };
        DeepSeekV2RotaryEmbedding::new(&cfg, DType::F32, &Device::Cpu).unwrap()
    }

    /// Deterministic, non-degenerate input. Both paths consume the same tensor,
    /// so randomness would be sound too, but a fixed pattern makes a failure
    /// reproducible.
    fn xs(b: usize, h: usize, t: usize, d: usize) -> Tensor {
        let n = b * h * t * d;
        let v: Vec<f32> = (0..n)
            .map(|i| ((i as f32) * 0.37).sin() * 1.7 + 0.11)
            .collect();
        Tensor::from_vec(v, (b, h, t, d), &Device::Cpu).unwrap()
    }

    /// Raw IEEE-754 bit patterns. Float equality would accept `-0.0 == 0.0` and
    /// silently pass a sign error in the conjugate `sin`; bit patterns do not.
    fn bits(t: &Tensor) -> Vec<u32> {
        t.flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .into_iter()
            .map(f32::to_bits)
            .collect()
    }

    /// The pre-fix `forward`, verbatim, as the oracle. Always loops, whatever
    /// the offsets are.
    fn reference_forward(
        re: &DeepSeekV2RotaryEmbedding,
        q: &Tensor,
        k: &Tensor,
        offsets: &[usize],
    ) -> Result<(Tensor, Tensor)> {
        let (_b, _h, seq_len, _n) = q.dims4()?;
        let mut q_embeds = Vec::new();
        let mut k_embeds = Vec::new();
        for (i, offset) in offsets.iter().enumerate() {
            let cos = re.cos.narrow(0, *offset, seq_len)?;
            let sin = re.sin.narrow(0, *offset, seq_len)?;
            q_embeds.push(candle_nn::rotary_emb::rope_i(
                &q.i(i)?.unsqueeze(0)?.contiguous()?,
                &cos,
                &sin,
            )?);
            k_embeds.push(candle_nn::rotary_emb::rope_i(
                &k.i(i)?.unsqueeze(0)?.contiguous()?,
                &cos,
                &sin,
            )?);
        }
        Ok((Tensor::cat(&q_embeds, 0)?, Tensor::cat(&k_embeds, 0)?))
    }

    /// The pre-fix `forward_inverse_tail`, verbatim — including the old
    /// `sin.narrow(..).neg()` spelling, so this oracle also pins the `neg_sin`
    /// precomputation.
    fn reference_inverse_tail(
        re: &DeepSeekV2RotaryEmbedding,
        x: &Tensor,
        rope_dim: usize,
        offsets: &[usize],
    ) -> Result<Tensor> {
        let (_b, _h, seq_len, head_dim) = x.dims4()?;
        let nope = head_dim - rope_dim;
        let x_nope = x.narrow(3, 0, nope)?;
        let mut outs = Vec::new();
        for (i, offset) in offsets.iter().enumerate() {
            let cos = re.cos.narrow(0, *offset, seq_len)?;
            let sin = re.sin.narrow(0, *offset, seq_len)?.neg()?;
            let x_pe = x
                .i(i)?
                .unsqueeze(0)?
                .narrow(3, nope, rope_dim)?
                .contiguous()?;
            outs.push(candle_nn::rotary_emb::rope_i(&x_pe, &cos, &sin)?);
        }
        let rotated = Tensor::cat(&outs, 0)?;
        Tensor::cat(&[&x_nope, &rotated], 3)?.contiguous()
    }

    // The predicate itself is tested once, in `rope_cohort_tests`
    // (`uniform_offset_tests_values_not_length`), against the same six cases.
    // #198 carried a byte-identical copy of that test aimed at
    // `DeepSeekV2RotaryEmbedding::uniform_offset`; with the two spellings
    // collapsed onto `uniform_seqlen_offset` there is one predicate, so a
    // second copy would assert nothing the first does not. What is NOT
    // duplicated, and stays below, is the V4-specific behaviour: `neg_sin`,
    // `forward_inverse_tail`, and the bit-exactness of this rotary's own
    // cohort dispatch.
    #[test]
    fn neg_sin_commutes_with_narrow() {
        let re = rope();
        for offset in [0usize, 1, 17] {
            let precomputed = re.neg_sin.narrow(0, offset, 3).unwrap();
            let on_the_fly = re.sin.narrow(0, offset, 3).unwrap().neg().unwrap();
            assert_eq!(
                bits(&precomputed),
                bits(&on_the_fly),
                "pre-negating the table must be bit-identical at offset {offset}"
            );
        }
    }

    #[test]
    fn cohort_forward_is_bit_identical_to_the_per_sequence_loop() {
        let re = rope();
        // seq_len 1 is decode; seq_len 2 covers uniform-offset chunked prefill.
        for seq_len in [1usize, 2] {
            let q = xs(BATCH, N_HEADS, seq_len, ROPE_DIM);
            let k = xs(BATCH, 1, seq_len, ROPE_DIM);
            let offsets = vec![5usize; BATCH];

            let before = rope_cohort_stats::local_counts();
            let (q_new, k_new) = re.forward(&q, &k, &offsets).unwrap();
            let after = rope_cohort_stats::local_counts();
            let (q_ref, k_ref) = reference_forward(&re, &q, &k, &offsets).unwrap();

            assert_eq!(q_new.dims(), q_ref.dims());
            assert_eq!(bits(&q_new), bits(&q_ref), "Q differs at seq_len={seq_len}");
            assert_eq!(bits(&k_new), bits(&k_ref), "K differs at seq_len={seq_len}");
            // Prove the batched branch is the one that ran: a fast path that
            // silently declined would pass the equality above trivially.
            assert_eq!(
                (after.0 - before.0, after.1 - before.1),
                (1, 0),
                "expected exactly one cohort call and no per-sequence loop"
            );
        }
    }

    #[test]
    fn ragged_forward_still_takes_the_loop_and_still_matches() {
        let re = rope();
        let q = xs(BATCH, N_HEADS, 1, ROPE_DIM);
        let k = xs(BATCH, 1, 1, ROPE_DIM);
        let offsets = vec![5usize, 6, 5, 9];

        let before = rope_cohort_stats::local_counts();
        let (q_new, k_new) = re.forward(&q, &k, &offsets).unwrap();
        let after = rope_cohort_stats::local_counts();
        let (q_ref, k_ref) = reference_forward(&re, &q, &k, &offsets).unwrap();

        assert_eq!(bits(&q_new), bits(&q_ref));
        assert_eq!(bits(&k_new), bits(&k_ref));
        assert_eq!(
            (after.0 - before.0, after.1 - before.1),
            (0, 1),
            "ragged rows must still take the per-sequence loop"
        );
    }

    #[test]
    fn cohort_inverse_tail_is_bit_identical_to_the_per_sequence_loop() {
        let re = rope();
        for seq_len in [1usize, 2] {
            let x = xs(BATCH, N_HEADS, seq_len, HEAD_DIM);
            let offsets = vec![5usize; BATCH];

            let before = rope_cohort_stats::local_counts();
            let new = re.forward_inverse_tail(&x, ROPE_DIM, &offsets).unwrap();
            let after = rope_cohort_stats::local_counts();
            let reference = reference_inverse_tail(&re, &x, ROPE_DIM, &offsets).unwrap();

            assert_eq!(new.dims(), reference.dims());
            assert_eq!(
                bits(&new),
                bits(&reference),
                "inverse tail differs at seq_len={seq_len}"
            );
            assert_eq!((after.0 - before.0, after.1 - before.1), (1, 0));
        }
    }

    #[test]
    fn ragged_inverse_tail_still_takes_the_loop_and_still_matches() {
        let re = rope();
        let x = xs(BATCH, N_HEADS, 1, HEAD_DIM);
        let offsets = vec![5usize, 6, 5, 9];

        let before = rope_cohort_stats::local_counts();
        let new = re.forward_inverse_tail(&x, ROPE_DIM, &offsets).unwrap();
        let after = rope_cohort_stats::local_counts();
        let reference = reference_inverse_tail(&re, &x, ROPE_DIM, &offsets).unwrap();

        assert_eq!(bits(&new), bits(&reference));
        assert_eq!((after.0 - before.0, after.1 - before.1), (0, 1));
    }

    /// The comparator must be able to fail. Every bit-equality assertion above
    /// is worthless if `bits` had degenerated into something that always
    /// matches, so poison one element by a single ULP and require it flagged.
    #[test]
    fn the_bit_comparator_is_live() {
        // Drives a forward, so it must serialise with the counter assertions.
        let re = rope();
        let x = xs(BATCH, N_HEADS, 1, HEAD_DIM);
        let offsets = vec![5usize; BATCH];
        let good = re.forward_inverse_tail(&x, ROPE_DIM, &offsets).unwrap();

        let mut poisoned = bits(&good);
        poisoned[0] += 1; // one ULP
        assert_ne!(
            bits(&good),
            poisoned,
            "the comparator accepted a one-ULP difference; every other \
             assertion in this module would be vacuous"
        );
    }
}

/// Parity for the **gated** ragged-cohort arms ([`rope_cohort_enabled`],
/// `ARC_ROPE_COHORT`): one gathered rank-3 `rope_i` against the per-sequence
/// loop it replaces, bit for bit.
///
/// The arms are driven **directly**, not through the public dispatch: the flag
/// latches once per process (`OnceLock`), so a test that reached the gathered
/// arm by mutating the environment would prove only whatever the first reader
/// happened to see — the same reasoning as `env_flag_value`'s own tests. The
/// dispatch's default (flag unset ⇒ ragged rows keep the loop) is pinned by
/// `deepseek_rope_cohort_tests::ragged_forward_still_takes_the_loop_and_still_matches`,
/// which runs the public `forward` in exactly that environment.
#[cfg(test)]
mod deepseek_ragged_cohort_tests {
    use super::*;

    const MAX_POS: usize = 32;
    const ROPE_DIM: usize = 8;
    const HEAD_DIM: usize = 12;
    const N_HEADS: usize = 3;

    fn rope() -> DeepSeekV2RotaryEmbedding {
        let cfg = DeepSeekV2RopeConfig {
            rope_scaling: None,
            max_position_embeddings: MAX_POS,
            rope_theta: 10000.0,
            qk_rope_head_dim: ROPE_DIM,
        };
        DeepSeekV2RotaryEmbedding::new(&cfg, DType::F32, &Device::Cpu).unwrap()
    }

    fn xs(b: usize, h: usize, t: usize, d: usize) -> Tensor {
        let n = b * h * t * d;
        let v: Vec<f32> = (0..n)
            .map(|i| ((i as f32) * 0.29).sin() * 1.3 + 0.07)
            .collect();
        Tensor::from_vec(v, (b, h, t, d), &Device::Cpu).unwrap()
    }

    /// Raw IEEE-754 bit patterns — the claim is that no arithmetic changed at
    /// all, so a tolerance would be vacuous and `-0.0 == 0.0` would slip a
    /// sign error through.
    fn bits(t: &Tensor) -> Vec<u32> {
        t.flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .into_iter()
            .map(f32::to_bits)
            .collect()
    }

    /// Mixed offsets, B ∈ {1, 3}, decode (T=1) and uniform-length chunk (T=2):
    /// the gathered arm must reproduce the loop bit for bit.
    #[test]
    fn ragged_cohort_forward_matches_the_loop_bitwise() {
        let re = rope();
        for (offsets, t) in [
            (vec![0usize, 4, 9], 1usize),
            (vec![0usize, 4, 9], 2),
            (vec![9usize, 4, 0], 1), // descending — order must be per-row
            (vec![7usize], 1),       // B = 1 through the same arms
        ] {
            let b = offsets.len();
            let q = xs(b, N_HEADS, t, ROPE_DIM);
            let k = xs(b, 1, t, ROPE_DIM);

            let (q_c, k_c) = re.forward_ragged_cohort(&q, &k, &offsets, t).unwrap();
            let (q_l, k_l) = re.forward_ragged_loop(&q, &k, &offsets, t).unwrap();

            assert_eq!(q_c.dims(), q_l.dims());
            assert_eq!(
                bits(&q_c),
                bits(&q_l),
                "gathered Q differs from the loop for offsets {offsets:?}, T={t}"
            );
            assert_eq!(
                bits(&k_c),
                bits(&k_l),
                "gathered K differs from the loop for offsets {offsets:?}, T={t}"
            );

            // Non-vacuity: rows at different offsets must actually rotate
            // differently, or the equality above would pass under a
            // broadcast-one-row bug on BOTH sides.
            if b > 1 {
                let d01 = (q_c.i(0).unwrap() - q_c.i(1).unwrap())
                    .unwrap()
                    .abs()
                    .unwrap()
                    .max_all()
                    .unwrap()
                    .to_scalar::<f32>()
                    .unwrap();
                assert!(d01 > 0.0, "rows at different offsets rotated identically");
            }
        }
    }

    /// Same parity for the conjugate (inverse-tail) rotation, and for the
    /// fully assembled `[NoPE | rotated]` output.
    #[test]
    fn ragged_cohort_inverse_tail_matches_the_loop_bitwise() {
        let re = rope();
        let nope = HEAD_DIM - ROPE_DIM;
        for (offsets, t) in [
            (vec![0usize, 4, 9], 1usize),
            (vec![0usize, 4, 9], 2),
            (vec![7usize], 1),
        ] {
            let b = offsets.len();
            let x = xs(b, N_HEADS, t, HEAD_DIM);

            let tail_c = re
                .inverse_tail_ragged_cohort(&x, ROPE_DIM, nope, &offsets, t)
                .unwrap();
            let tail_l = re
                .inverse_tail_ragged_loop(&x, ROPE_DIM, nope, &offsets, t)
                .unwrap();
            assert_eq!(tail_c.dims(), tail_l.dims());
            assert_eq!(
                bits(&tail_c),
                bits(&tail_l),
                "gathered inverse tail differs from the loop for offsets {offsets:?}, T={t}"
            );

            // The assembled output the model actually consumes.
            let x_nope = x.narrow(3, 0, nope).unwrap();
            let full_c = Tensor::cat(&[&x_nope, &tail_c], 3)
                .unwrap()
                .contiguous()
                .unwrap();
            let full_l = Tensor::cat(&[&x_nope, &tail_l], 3)
                .unwrap()
                .contiguous()
                .unwrap();
            assert_eq!(bits(&full_c), bits(&full_l));
        }
    }

    /// The gather memo must serve repeats AND rebuild when the cohort moves —
    /// a stale entry would rotate every layer of the next step at the previous
    /// step's positions, which is exactly the class of wrong-but-plausible
    /// output that never crashes.
    #[test]
    fn cohort_gather_rebuilds_when_the_cohort_moves() {
        let re = rope();
        let t = 1usize;

        for offsets in [vec![0usize, 4, 9], vec![1usize, 5, 10], vec![2usize, 3, 4]] {
            // Twice with the same key: the second call is the memo hit.
            for pass in 0..2 {
                let (cos, sin, neg_sin) = re.cohort_tables(&offsets, t).unwrap();
                assert_eq!(cos.dims(), &[offsets.len(), t, ROPE_DIM / 2]);
                for (b_i, &off) in offsets.iter().enumerate() {
                    let want_cos = re.cos.narrow(0, off, t).unwrap();
                    let want_sin = re.sin.narrow(0, off, t).unwrap();
                    let want_neg = re.neg_sin.narrow(0, off, t).unwrap();
                    assert_eq!(
                        bits(&cos.i(b_i).unwrap()),
                        bits(&want_cos),
                        "cos row {b_i} wrong for offsets {offsets:?} (pass {pass})"
                    );
                    assert_eq!(bits(&sin.i(b_i).unwrap()), bits(&want_sin));
                    assert_eq!(bits(&neg_sin.i(b_i).unwrap()), bits(&want_neg));
                }
            }
        }
    }

    /// `ARC_ROPE_COHORT` is read by VALUE through `env_flag_is_set`, whose
    /// polarity table (`=0`/unset ⇒ OFF, only yes-spellings ⇒ ON) is pinned in
    /// `mistralrs_quant::env_flag`. What this pins HERE is the dispatch
    /// default: with the flag unset — the state of every CI environment this
    /// runs in — ragged rows must take the per-sequence loop, so the gathered
    /// arm cannot go live by accident of a latch.
    #[test]
    fn ragged_dispatch_defaults_to_the_loop() {
        assert!(
            std::env::var_os("ARC_ROPE_COHORT").is_none(),
            "test environment unexpectedly sets ARC_ROPE_COHORT; this test \
             pins the UNSET default and cannot do so with the flag present"
        );
        assert!(!rope_cohort_enabled(), "unset must mean OFF");

        let re = rope();
        let q = xs(3, N_HEADS, 1, ROPE_DIM);
        let k = xs(3, 1, 1, ROPE_DIM);
        let before = rope_cohort_stats::local_counts();
        re.forward(&q, &k, &[0usize, 4, 9]).unwrap();
        let after = rope_cohort_stats::local_counts();
        assert_eq!(
            (after.0 - before.0, after.1 - before.1),
            (0, 1),
            "with ARC_ROPE_COHORT unset a ragged cohort must take the loop"
        );
    }
}
