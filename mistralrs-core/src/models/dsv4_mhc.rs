#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
//! DeepSeek V4 mHC (Manifold-Constrained Hyper-Connections).
//!
//! Pure-Rust port of SGLang's mHC pre/post split-sinkhorn mixing.
//!
//! References:
//! - Decoder layer (`hc_pre`/`hc_post`):
//!   `research/code/06_foundation/sglang/python/sglang/srt/models/deepseek_v4.py`
//!   lines 657-1002 (`DeepseekV4DecoderLayer`).
//! - Split-sinkhorn kernel reference:
//!   `research/code/06_foundation/sglang/python/sglang/srt/layers/mhc.py`
//!   lines 27-115 (`hc_split_sinkhorn_kernel`, `hc_split_sinkhorn`).
//! - Final head (`hc_head`):
//!   `research/code/06_foundation/sglang/python/sglang/srt/models/deepseek_v4_nextn.py`
//!   lines 117-130.
//!
//! ## Real SGLang semantics
//!
//! mHC replaces standard residual `x + branch(norm(x))` connections with a
//! per-token mixing of `hc_mult` parallel residual branches. Each decoder layer
//! has *two* mixing points (attn + ffn). The full round-trip is:
//!
//! ```text
//! residual: [B, T, hc_mult, hidden]
//! (y, post, comb) = hc_pre(residual, fn, scale, base)   # y: [B, T, hidden]
//! y = norm(y)
//! branch_out = attn_or_mlp(y)                            # [B, T, hidden]
//! new_residual = hc_post(branch_out, residual, post, comb)  # [B, T, hc_mult, hidden]
//! ```
//!
//! `hc_pre` produces three tensors derived from a single linear projection
//! `mixes = (flatten(residual) @ fn.T) * rsqrt(...)`:
//! 1. `pre  [N, hc]`        — sigmoid weights, used to fold the `hc_mult`
//!    branches down into the single input fed to attention/MLP.
//! 2. `post [N, hc]`        — `2 * sigmoid(...)` weights for the branch output
//!    that re-expands into the residual stack.
//! 3. `comb [N, hc, hc]`    — sinkhorn-normalized doubly-stochastic matrix that
//!    re-mixes the original `hc_mult` residual branches alongside the new
//!    branch output.
//!
//! `mix_hc = (2 + hc_mult) * hc_mult` slots split as
//! `[0..hc] | [hc..2*hc] | [2*hc..(2+hc)*hc]` for the three blocks above.
//!
//! ## Spec interface
//!
//! The mission spec asks for a simpler 3-branch `mix_attn(residual, attn_out)`
//! API. That API is implemented here as a convenience wrapper that performs the
//! full `hc_pre + hc_post` round-trip on a `[B, T, hc_mult, hidden]` residual
//! buffer, matching the actual SGLang decoder. The pure 3-branch sigmoid mix
//! (no sinkhorn) referenced in the spec is what `V4MHCHead::forward` does for
//! the final LM head, exactly mirroring `deepseek_v4_nextn.py:117-130`.

use candle_core::{DType, Result, Tensor, D};
use mistralrs_quant::ShardedVarBuilder;

use super::deepseek4::DeepSeekV4Config;

/// Defaults the V4 paper / SGLang configs do not always serialize.
pub const DEFAULT_HC_MULT: usize = 4;
pub const DEFAULT_HC_SINKHORN_ITERS: usize = 20;
pub const DEFAULT_HC_EPS: f64 = 1e-6;

/// V4 mHC layer-side hyperparameters that live alongside the trainable tensors.
///
/// These are not currently present on `DeepSeekV4Config` to keep this agent's
/// surface small. Agent 3 (the wiring agent) should pull these values from the
/// real V4 config when integrating the loader.
#[derive(Debug, Clone, Copy)]
pub struct V4MHCRuntime {
    pub hc_mult: usize,
    pub hc_sinkhorn_iters: usize,
    pub hc_eps: f64,
    pub rms_norm_eps: f64,
}

impl V4MHCRuntime {
    pub fn from_cfg(cfg: &DeepSeekV4Config) -> Self {
        // V4 Flash defaults. Agent 3 will replace with cfg.hc_mult / cfg.hc_eps
        // / cfg.hc_sinkhorn_iters once those fields exist on DeepSeekV4Config.
        Self {
            hc_mult: DEFAULT_HC_MULT,
            hc_sinkhorn_iters: DEFAULT_HC_SINKHORN_ITERS,
            hc_eps: DEFAULT_HC_EPS,
            rms_norm_eps: cfg.rms_norm_eps,
        }
    }
}

/// mHC parameters and constants for a single decoder layer.
///
/// Tensor shapes (all stored as F32 to match SGLang):
/// - `hc_attn_fn`, `hc_ffn_fn`:   `[mix_hc, hc_dim]`   where `mix_hc = (2+hc_mult)*hc_mult`,
///                                                          `hc_dim = hc_mult * hidden_size`
/// - `hc_attn_base`, `hc_ffn_base`: `[mix_hc]`
/// - `hc_attn_scale`, `hc_ffn_scale`: `[3]`            — broadcast across the three
///                                                       `(pre, post, comb)` blocks.
pub struct V4MHCLayerParams {
    pub hc_attn_fn: Tensor,
    pub hc_attn_base: Tensor,
    pub hc_attn_scale: Tensor,
    pub hc_ffn_fn: Tensor,
    pub hc_ffn_base: Tensor,
    pub hc_ffn_scale: Tensor,
    pub mix_hc: usize,
    pub hc_mult: usize,
    pub hidden_size: usize,
    pub rt: V4MHCRuntime,
}

impl V4MHCLayerParams {
    /// Try to load mHC params from `vb` (assumed to be at the layer-root,
    /// i.e. `model.layers.<layer_idx>`). Returns `None` if any expected tensor
    /// is absent — this lets the caller fall back to standard residual.
    ///
    /// `layer_idx` is informational only (used in error messages once the
    /// caller upgrades to log loads). The actual prefix selection is the
    /// caller's responsibility.
    pub fn try_load(
        cfg: &DeepSeekV4Config,
        vb: &ShardedVarBuilder,
        _layer_idx: usize,
    ) -> Option<Self> {
        let rt = V4MHCRuntime::from_cfg(cfg);
        let hc_mult = rt.hc_mult;
        let mix_hc = (2 + hc_mult) * hc_mult;
        let hc_dim = hc_mult * cfg.hidden_size;

        // Probe presence; bail to None if anything is missing.
        let needed = [
            "hc_attn_fn",
            "hc_attn_base",
            "hc_attn_scale",
            "hc_ffn_fn",
            "hc_ffn_base",
            "hc_ffn_scale",
        ];
        if !needed.iter().all(|n| vb.contains_tensor(n)) {
            return None;
        }

        let load = |name: &str, shape: &[usize]| -> Result<Tensor> {
            vb.get_with_hints_dtype(shape, name, Default::default(), DType::F32)
        };

        // Any load failure → None (e.g., shape mismatch).
        let hc_attn_fn = load("hc_attn_fn", &[mix_hc, hc_dim]).ok()?;
        let hc_attn_base = load("hc_attn_base", &[mix_hc]).ok()?;
        let hc_attn_scale = load("hc_attn_scale", &[3]).ok()?;
        let hc_ffn_fn = load("hc_ffn_fn", &[mix_hc, hc_dim]).ok()?;
        let hc_ffn_base = load("hc_ffn_base", &[mix_hc]).ok()?;
        let hc_ffn_scale = load("hc_ffn_scale", &[3]).ok()?;

        Some(Self {
            hc_attn_fn,
            hc_attn_base,
            hc_attn_scale,
            hc_ffn_fn,
            hc_ffn_base,
            hc_ffn_scale,
            mix_hc,
            hc_mult,
            hidden_size: cfg.hidden_size,
            rt,
        })
    }

    /// `hc_pre`: split-sinkhorn pre-step.
    ///
    /// Input `residual` may be either `[B, T, hc_mult, hidden]` (4-D) or
    /// `[N, hc_mult, hidden]` (3-D, already flattened over batch/seq). Output
    /// shapes follow the input rank:
    /// - `y`:    `[B, T, hidden]` or `[N, hidden]`
    /// - `post`: `[B, T, hc_mult]` or `[N, hc_mult]`
    /// - `comb`: `[B, T, hc_mult, hc_mult]` or `[N, hc_mult, hc_mult]`
    pub fn hc_pre(
        &self,
        residual: &Tensor,
        hc_fn: &Tensor,
        hc_scale: &Tensor,
        hc_base: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let dims = residual.dims().to_vec();
        let in_dtype = residual.dtype();
        let hc = self.hc_mult;
        let h = self.hidden_size;

        // Normalize to [N, hc, h] for the inner kernel; restore leading dims at the end.
        let (n, leading_dims): (usize, Vec<usize>) = match dims.len() {
            3 => (dims[0], vec![dims[0]]),
            4 => (dims[0] * dims[1], vec![dims[0], dims[1]]),
            _ => candle_core::bail!(
                "V4 mHC hc_pre: expected residual rank 3 or 4, got {:?}",
                dims
            ),
        };
        if dims[dims.len() - 2] != hc || *dims.last().unwrap() != h {
            candle_core::bail!(
                "V4 mHC hc_pre: expected last two dims [hc_mult={hc}, hidden={h}], got {:?}",
                dims
            );
        }

        let x = residual.reshape((n, hc, h))?;
        // Promote to F32, flatten to [N, hc*h].
        let x_flat = x.reshape((n, hc * h))?.to_dtype(DType::F32)?;

        // rsqrt(mean(x^2) + eps)
        let sq_mean = x_flat.sqr()?.mean_keepdim(D::Minus1)?;
        let rsqrt = (sq_mean + self.rt.rms_norm_eps)?.recip()?.sqrt()?; // [N, 1]

        // mixes = (x_flat @ fn^T) * rsqrt  →  [N, mix_hc]
        // Defensively cast weight tensors to F32 — try_load already produces F32,
        // but hand-constructed callers (tests, external integrators) may not.
        let hc_fn_f32 = hc_fn.to_dtype(DType::F32)?;
        let hc_scale_f32 = hc_scale.to_dtype(DType::F32)?;
        let hc_base_f32 = hc_base.to_dtype(DType::F32)?;
        let mixes_raw = x_flat.matmul(&hc_fn_f32.t()?)?;
        let mixes = mixes_raw.broadcast_mul(&rsqrt)?;

        // Slot indices in `mixes`:
        //   pre  : [..,  0 .. hc)
        //   post : [.., hc .. 2*hc)
        //   comb : [.., 2*hc .. (2+hc)*hc)  reshape to [.., hc, hc]
        let pre_block = mixes.narrow(D::Minus1, 0, hc)?;
        let post_block = mixes.narrow(D::Minus1, hc, hc)?;
        let comb_block = mixes
            .narrow(D::Minus1, 2 * hc, hc * hc)?
            .reshape((n, hc, hc))?;

        // hc_scale is [3]; hc_base is [mix_hc] split into three blocks.
        let s_pre = hc_scale_f32.narrow(0, 0, 1)?;
        let s_post = hc_scale_f32.narrow(0, 1, 1)?;
        let s_comb = hc_scale_f32.narrow(0, 2, 1)?;
        let b_pre = hc_base_f32.narrow(0, 0, hc)?;
        let b_post = hc_base_f32.narrow(0, hc, hc)?;
        let b_comb = hc_base_f32.narrow(0, 2 * hc, hc * hc)?.reshape((hc, hc))?;

        // pre  = sigmoid(pre_block  * s_pre  + b_pre) + eps
        let pre = candle_nn::ops::sigmoid(
            &(pre_block.broadcast_mul(&s_pre)?.broadcast_add(&b_pre)?),
        )?;
        let pre = (pre + self.rt.hc_eps)?;

        // post = 2 * sigmoid(post_block * s_post + b_post)
        let post_sig = candle_nn::ops::sigmoid(
            &(post_block.broadcast_mul(&s_post)?.broadcast_add(&b_post)?),
        )?;
        let two = Tensor::new(2f32, post_sig.device())?;
        let post = post_sig.broadcast_mul(&two)?;

        // comb = sinkhorn_normalize(comb_block * s_comb + b_comb)
        let comb_pre = comb_block
            .broadcast_mul(&s_comb)?
            .broadcast_add(&b_comb)?; // [N, hc, hc]
        let comb = sinkhorn_normalize(&comb_pre, self.rt.hc_sinkhorn_iters, self.rt.hc_eps)?;

        // y = sum_i pre[..., i, None] * x[..., i, :]  →  [N, hidden]
        let pre_b = pre.unsqueeze(D::Minus1)?; // [N, hc, 1]
        let y = pre_b.broadcast_mul(&x.to_dtype(DType::F32)?)?.sum(1)?; // [N, hidden]

        // Restore leading dims.
        let mut y_shape = leading_dims.clone();
        y_shape.push(h);
        let mut post_shape = leading_dims.clone();
        post_shape.push(hc);
        let mut comb_shape = leading_dims.clone();
        comb_shape.push(hc);
        comb_shape.push(hc);

        let y_out = y.reshape(y_shape)?.to_dtype(in_dtype)?;
        let post_out = post.reshape(post_shape)?.to_dtype(in_dtype)?;
        let comb_out = comb.reshape(comb_shape)?.to_dtype(in_dtype)?;
        Ok((y_out, post_out, comb_out))
    }

    /// `hc_post`: re-expand a single branch output back into `hc_mult` residual
    /// streams, mixing with the original residual via `post` (per-branch
    /// scalar weights for the new output) and `comb` (doubly-stochastic
    /// matrix mixing the existing branches among themselves).
    ///
    /// - `x`: branch output, `[B, T, hidden]` or `[N, hidden]`
    /// - `residual`: pre-mix residual stack, `[B, T, hc_mult, hidden]` or `[N, hc_mult, hidden]`
    /// - `post`: `[B, T, hc_mult]` or `[N, hc_mult]`
    /// - `comb`: `[B, T, hc_mult, hc_mult]` or `[N, hc_mult, hc_mult]`
    ///
    /// Returns the new residual stack with the same shape as `residual`.
    pub fn hc_post(
        &self,
        x: &Tensor,
        residual: &Tensor,
        post: &Tensor,
        comb: &Tensor,
    ) -> Result<Tensor> {
        let dims = residual.dims().to_vec();
        let hc = self.hc_mult;
        let h = self.hidden_size;
        let in_dtype = x.dtype();

        let (n, leading_dims): (usize, Vec<usize>) = match dims.len() {
            3 => (dims[0], vec![dims[0]]),
            4 => (dims[0] * dims[1], vec![dims[0], dims[1]]),
            _ => candle_core::bail!(
                "V4 mHC hc_post: expected residual rank 3 or 4, got {:?}",
                dims
            ),
        };

        let x_n = x.reshape((n, h))?.to_dtype(DType::F32)?;
        let residual_n = residual
            .reshape((n, hc, h))?
            .to_dtype(DType::F32)?;
        let post_n = post.reshape((n, hc))?.to_dtype(DType::F32)?;
        let comb_n = comb.reshape((n, hc, hc))?.to_dtype(DType::F32)?;

        // term1 = post[..., :, None] * x[..., None, :]   →  [N, hc, h]
        let term1 = post_n
            .unsqueeze(D::Minus1)? // [N, hc, 1]
            .broadcast_mul(&x_n.unsqueeze(1)?)?; // [N, hc, h]

        // term2 = sum_j comb[..., j, k] * residual[..., j, :]
        //         = einsum("njk,njh->nkh", comb, residual)
        // Implementation: for each k, term2[n, k, h] = sum_j comb[n, j, k] * residual[n, j, h]
        // Use matmul: comb.transpose(1,2) @ residual  →  [N, hc, h]
        let term2 = comb_n.transpose(1, 2)?.matmul(&residual_n)?; // [N, hc, h]

        let out_n = (term1 + term2)?;

        // Restore shape.
        let mut out_shape = leading_dims;
        out_shape.push(hc);
        out_shape.push(h);
        out_n.reshape(out_shape)?.to_dtype(in_dtype)
    }

    /// Spec convenience wrapper: full attn-side round-trip for one decoder layer.
    ///
    /// Mirrors the SGLang `DeepseekV4DecoderLayer.forward` attention block: takes
    /// the pre-attn residual stack and the attention output (computed on
    /// `hc_pre`'s `y` after norm), returns the new residual stack.
    ///
    /// `residual`: `[B, T, hc_mult, hidden]` (or 3-D collapsed).
    /// `attn_out`: `[B, T, hidden]` (or 2-D collapsed).
    pub fn mix_attn(&self, residual: &Tensor, attn_out: &Tensor) -> Result<Tensor> {
        let (_y, post, comb) = self.hc_pre(
            residual,
            &self.hc_attn_fn,
            &self.hc_attn_scale,
            &self.hc_attn_base,
        )?;
        self.hc_post(attn_out, residual, &post, &comb)
    }

    /// Spec convenience wrapper: full FFN-side round-trip for one decoder layer.
    pub fn mix_ffn(&self, residual: &Tensor, ffn_out: &Tensor) -> Result<Tensor> {
        let (_y, post, comb) = self.hc_pre(
            residual,
            &self.hc_ffn_fn,
            &self.hc_ffn_scale,
            &self.hc_ffn_base,
        )?;
        self.hc_post(ffn_out, residual, &post, &comb)
    }
}

/// Sinkhorn normalize the last two dims of a `[N, hc, hc]` tensor.
///
/// Mirrors `hc_split_sinkhorn_kernel` in
/// `sglang/srt/layers/mhc.py:27-91`:
///
/// 1. Subtract row max, exponentiate.
/// 2. Divide by row sums (and add eps).
/// 3. Divide by column sums (with eps in denominator).
/// 4. Repeat (row→col) `sinkhorn_iters - 1` more times.
fn sinkhorn_normalize(comb: &Tensor, sinkhorn_iters: usize, eps: f64) -> Result<Tensor> {
    // Stable softmax along last dim (rows).
    let row_max = comb.max_keepdim(D::Minus1)?; // [N, hc, 1]
    let shifted = comb.broadcast_sub(&row_max)?;
    let exped = shifted.exp()?;
    let row_sum = exped.sum_keepdim(D::Minus1)?; // [N, hc, 1]
    let mut x = exped.broadcast_div(&row_sum)?;
    x = (x + eps)?;

    // Initial column normalize (with eps in denominator to match SGLang).
    let col_sum = x.sum_keepdim(1)?; // [N, 1, hc]
    let col_sum_eps = (col_sum + eps)?;
    x = x.broadcast_div(&col_sum_eps)?;

    // Remaining sinkhorn_iters - 1 row→col passes.
    for _ in 0..sinkhorn_iters.saturating_sub(1) {
        let row_sum = x.sum_keepdim(D::Minus1)?;
        let row_sum_eps = (row_sum + eps)?;
        x = x.broadcast_div(&row_sum_eps)?;
        let col_sum = x.sum_keepdim(1)?;
        let col_sum_eps = (col_sum + eps)?;
        x = x.broadcast_div(&col_sum_eps)?;
    }

    Ok(x)
}

/// Final mHC head — runs once at the model output, before `lm_head`.
///
/// Implements `DeepseekV4ModelNextN.hc_head` from
/// `sglang/srt/models/deepseek_v4_nextn.py:117-130`:
///
/// ```text
/// x_flat = x.flatten(1).float()                                    # [N, hc_mult * hidden]
/// rsqrt  = rsqrt(mean(x_flat^2, -1) + rms_norm_eps)
/// mixes  = (x_flat @ hc_head_fn.T) * rsqrt                          # [N, hc_mult]
/// pre    = sigmoid(mixes * hc_head_scale + hc_head_base) + hc_eps   # [N, hc_mult]
/// y      = sum_i pre[..., i, None] * x[..., i, :]                  # [N, hidden]
/// ```
///
/// Tensor shapes:
/// - `hc_head_fn`:    `[hc_mult, hc_dim]` with `hc_dim = hc_mult * hidden_size`
/// - `hc_head_base`:  `[hc_mult]`
/// - `hc_head_scale`: `[1]`
pub struct V4MHCHead {
    pub hc_head_fn: Tensor,
    pub hc_head_base: Tensor,
    pub hc_head_scale: Tensor,
    pub hc_mult: usize,
    pub hidden_size: usize,
    pub rt: V4MHCRuntime,
}

impl V4MHCHead {
    /// Try to load the final mHC head from `vb` (assumed at model root for
    /// V4 NextN, i.e. `model.`). Returns `None` if any tensor is missing.
    pub fn try_load(cfg: &DeepSeekV4Config, vb: &ShardedVarBuilder) -> Option<Self> {
        let rt = V4MHCRuntime::from_cfg(cfg);
        let hc_mult = rt.hc_mult;
        let hc_dim = hc_mult * cfg.hidden_size;

        let needed = ["hc_head_fn", "hc_head_base", "hc_head_scale"];
        if !needed.iter().all(|n| vb.contains_tensor(n)) {
            return None;
        }

        let load = |name: &str, shape: &[usize]| -> Result<Tensor> {
            vb.get_with_hints_dtype(shape, name, Default::default(), DType::F32)
        };

        let hc_head_fn = load("hc_head_fn", &[hc_mult, hc_dim]).ok()?;
        let hc_head_base = load("hc_head_base", &[hc_mult]).ok()?;
        let hc_head_scale = load("hc_head_scale", &[1]).ok()?;

        Some(Self {
            hc_head_fn,
            hc_head_base,
            hc_head_scale,
            hc_mult,
            hidden_size: cfg.hidden_size,
            rt,
        })
    }

    /// Apply final head mixing.
    ///
    /// Input `hidden`: `[B, T, hc_mult, hidden]` (4-D) or `[N, hc_mult, hidden]` (3-D).
    /// Output: `[B, T, hidden]` or `[N, hidden]` — the `hc_mult` branches folded
    /// down to a single hidden state ready for `lm_head`.
    pub fn forward(&self, hidden: &Tensor) -> Result<Tensor> {
        let dims = hidden.dims().to_vec();
        let in_dtype = hidden.dtype();
        let hc = self.hc_mult;
        let h = self.hidden_size;

        let (n, leading_dims): (usize, Vec<usize>) = match dims.len() {
            3 => (dims[0], vec![dims[0]]),
            4 => (dims[0] * dims[1], vec![dims[0], dims[1]]),
            _ => candle_core::bail!(
                "V4 mHC head: expected rank 3 or 4, got {:?}",
                dims
            ),
        };
        if dims[dims.len() - 2] != hc || *dims.last().unwrap() != h {
            candle_core::bail!(
                "V4 mHC head: expected last two dims [hc_mult={hc}, hidden={h}], got {:?}",
                dims
            );
        }

        let x = hidden.reshape((n, hc, h))?.to_dtype(DType::F32)?;
        let x_flat = x.reshape((n, hc * h))?;

        // rsqrt(mean(x^2, -1) + rms_eps)
        let sq_mean = x_flat.sqr()?.mean_keepdim(D::Minus1)?;
        let rsqrt = (sq_mean + self.rt.rms_norm_eps)?.recip()?.sqrt()?;

        // mixes = (x_flat @ hc_head_fn^T) * rsqrt → [N, hc_mult]
        // Defensively cast weight tensors to F32 — try_load already produces F32,
        // but hand-constructed callers (tests, external integrators) may not.
        let head_fn_f32 = self.hc_head_fn.to_dtype(DType::F32)?;
        let head_scale_f32 = self.hc_head_scale.to_dtype(DType::F32)?;
        let head_base_f32 = self.hc_head_base.to_dtype(DType::F32)?;
        let mixes_raw = x_flat.matmul(&head_fn_f32.t()?)?;
        let mixes = mixes_raw.broadcast_mul(&rsqrt)?;

        // weights = sigmoid(mixes * scale + base) + hc_eps   → [N, hc_mult]
        let scaled = mixes
            .broadcast_mul(&head_scale_f32)?
            .broadcast_add(&head_base_f32)?;
        let weights = candle_nn::ops::sigmoid(&scaled)?;
        let weights = (weights + self.rt.hc_eps)?;

        // y = sum_i weights[..., i, None] * x[..., i, :]  →  [N, hidden]
        let y = weights
            .unsqueeze(D::Minus1)?
            .broadcast_mul(&x)?
            .sum(1)?;

        let mut out_shape = leading_dims;
        out_shape.push(h);
        y.reshape(out_shape)?.to_dtype(in_dtype)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use candle_nn::var_builder::SimpleBackend;
    use mistralrs_quant::ShardedSafeTensors;
    use std::collections::HashMap;

    fn dummy_cfg(hidden_size: usize) -> DeepSeekV4Config {
        // Minimal config — only fields used by mHC are meaningful.
        // Other fields kept consistent with the existing tests in deepseek4.rs.
        #[allow(clippy::needless_update)]
        let json = serde_json::json!({
            "vocab_size": 100,
            "hidden_size": hidden_size,
            "intermediate_size": 64,
            "moe_intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "n_shared_experts": null,
            "n_routed_experts": null,
            "num_experts_per_tok": 2,
            "max_position_embeddings": 256,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "rope_scaling": null,
            "attention_bias": false,
            "q_lora_rank": 8,
            "qk_rope_head_dim": 8,
            "kv_lora_rank": 8,
            "v_head_dim": 8,
            "qk_nope_head_dim": 8,
            "quantization_config": null,
            "n_group": 1,
            "topk_group": 1,
            "topk_method": "greedy",
            "scoring_func": "softmax",
            "hidden_act": "silu",
            "compress_ratios": [],
        });
        serde_json::from_value(json).unwrap()
    }

    fn vb_from(tensors: HashMap<String, Tensor>, device: Device) -> mistralrs_quant::ShardedVarBuilder {
        let backend: Box<dyn SimpleBackend + 'static> = Box::new(tensors);
        ShardedSafeTensors::wrap(backend, DType::F32, device)
    }

    /// 1. try_load returns None when tensors absent.
    #[test]
    fn try_load_returns_none_when_absent() -> Result<()> {
        let cfg = dummy_cfg(8);
        let vb = vb_from(HashMap::new(), Device::Cpu);
        assert!(V4MHCLayerParams::try_load(&cfg, &vb, 0).is_none());
        assert!(V4MHCHead::try_load(&cfg, &vb).is_none());
        Ok(())
    }

    /// 1b. try_load returns Some when all tensors present and shapes correct.
    #[test]
    fn try_load_returns_some_when_present() -> Result<()> {
        let hidden = 4;
        let cfg = dummy_cfg(hidden);
        let hc_mult = DEFAULT_HC_MULT; // 4
        let mix_hc = (2 + hc_mult) * hc_mult; // 24
        let hc_dim = hc_mult * hidden; // 16
        let dev = Device::Cpu;

        let mut tensors = HashMap::new();
        tensors.insert(
            "hc_attn_fn".to_string(),
            Tensor::zeros((mix_hc, hc_dim), DType::F32, &dev)?,
        );
        tensors.insert(
            "hc_attn_base".to_string(),
            Tensor::zeros(mix_hc, DType::F32, &dev)?,
        );
        tensors.insert(
            "hc_attn_scale".to_string(),
            Tensor::zeros(3, DType::F32, &dev)?,
        );
        tensors.insert(
            "hc_ffn_fn".to_string(),
            Tensor::zeros((mix_hc, hc_dim), DType::F32, &dev)?,
        );
        tensors.insert(
            "hc_ffn_base".to_string(),
            Tensor::zeros(mix_hc, DType::F32, &dev)?,
        );
        tensors.insert(
            "hc_ffn_scale".to_string(),
            Tensor::zeros(3, DType::F32, &dev)?,
        );
        let vb = vb_from(tensors, dev);
        let params = V4MHCLayerParams::try_load(&cfg, &vb, 0)
            .expect("layer params should load");
        assert_eq!(params.mix_hc, mix_hc);
        assert_eq!(params.hc_mult, hc_mult);
        assert_eq!(params.hidden_size, hidden);
        Ok(())
    }

    fn make_layer_params(
        hidden: usize,
        hc_mult: usize,
        hc_attn_fn: Tensor,
        hc_attn_base: Tensor,
        hc_attn_scale: Tensor,
    ) -> V4MHCLayerParams {
        let mix_hc = (2 + hc_mult) * hc_mult;
        // Reuse attn tensors for ffn; tests don't exercise mix_ffn separately.
        V4MHCLayerParams {
            hc_attn_fn: hc_attn_fn.clone(),
            hc_attn_base: hc_attn_base.clone(),
            hc_attn_scale: hc_attn_scale.clone(),
            hc_ffn_fn: hc_attn_fn,
            hc_ffn_base: hc_attn_base,
            hc_ffn_scale: hc_attn_scale,
            mix_hc,
            hc_mult,
            hidden_size: hidden,
            rt: V4MHCRuntime {
                hc_mult,
                hc_sinkhorn_iters: DEFAULT_HC_SINKHORN_ITERS,
                hc_eps: DEFAULT_HC_EPS,
                rms_norm_eps: 1e-6,
            },
        }
    }

    /// 2. mix_attn produces correct shape.
    #[test]
    fn mix_attn_shape_4d() -> Result<()> {
        let dev = Device::Cpu;
        let hidden = 4;
        let hc_mult = 4;
        let mix_hc = (2 + hc_mult) * hc_mult;
        let hc_dim = hc_mult * hidden;
        let params = make_layer_params(
            hidden,
            hc_mult,
            Tensor::zeros((mix_hc, hc_dim), DType::F32, &dev)?,
            Tensor::zeros(mix_hc, DType::F32, &dev)?,
            Tensor::zeros(3, DType::F32, &dev)?,
        );

        let b = 2;
        let t = 3;
        let residual = Tensor::randn(0f32, 1f32, (b, t, hc_mult, hidden), &dev)?;
        let attn_out = Tensor::randn(0f32, 1f32, (b, t, hidden), &dev)?;

        let out = params.mix_attn(&residual, &attn_out)?;
        assert_eq!(out.dims(), &[b, t, hc_mult, hidden]);

        // Also exercise 3-D path.
        let residual_3 = residual.reshape((b * t, hc_mult, hidden))?;
        let attn_out_3 = attn_out.reshape((b * t, hidden))?;
        let out_3 = params.mix_attn(&residual_3, &attn_out_3)?;
        assert_eq!(out_3.dims(), &[b * t, hc_mult, hidden]);

        // mix_ffn shape mirror.
        let out_ffn = params.mix_ffn(&residual, &attn_out)?;
        assert_eq!(out_ffn.dims(), &[b, t, hc_mult, hidden]);
        Ok(())
    }

    /// 3. Sanity: with all-zero fn/base/scale, mixes=0 → pre=sigmoid(0)+eps=0.5+eps,
    /// post=2*0.5=1.0, comb=sinkhorn(zeros)=uniform(1/hc).
    /// Then hc_post output equals 1.0 * x + sum_j(1/hc) * residual[:, j, :]
    ///                            = x + mean(residual over branches)
    #[test]
    fn mix_attn_zero_params_matches_hand_compute() -> Result<()> {
        let dev = Device::Cpu;
        let hidden = 4;
        let hc_mult = 4;
        let mix_hc = (2 + hc_mult) * hc_mult;
        let hc_dim = hc_mult * hidden;

        let params = make_layer_params(
            hidden,
            hc_mult,
            Tensor::zeros((mix_hc, hc_dim), DType::F32, &dev)?,
            Tensor::zeros(mix_hc, DType::F32, &dev)?,
            Tensor::zeros(3, DType::F32, &dev)?,
        );

        // residual with distinct per-branch values so the mean-over-branches term
        // is recognizable.
        let n = 2;
        let residual = Tensor::from_vec(
            (0..n * hc_mult * hidden)
                .map(|i| i as f32)
                .collect::<Vec<_>>(),
            (n, hc_mult, hidden),
            &dev,
        )?;
        let attn_out = Tensor::ones((n, hidden), DType::F32, &dev)?;

        let out = params.mix_attn(&residual, &attn_out)?;
        assert_eq!(out.dims(), &[n, hc_mult, hidden]);

        // Expected = attn_out[:, None, :].broadcast + mean(residual, dim=1).broadcast
        // Sinkhorn of all-zero starts from row-softmax(0)=uniform(1/hc), then
        // (+ eps) and column-normalize. With uniform input, the result stays
        // very close to uniform but eps perturbs it slightly. Use a generous
        // tolerance.
        //
        // NOTE: attn_out is [N, hidden]; the output is [N, hc_mult, hidden] where
        // each branch k is `attn_out + mean(residual over branches)`. We must
        // broadcast attn_out across the hc_mult axis to get matching shapes.
        let mean_resid = residual.mean(1)?; // [N, hidden]
        let attn_broadcast = attn_out
            .unsqueeze(1)?
            .broadcast_as((n, hc_mult, hidden))?; // [N, hc_mult, hidden]
        let mean_broadcast = mean_resid
            .unsqueeze(1)?
            .broadcast_as((n, hc_mult, hidden))?; // [N, hc_mult, hidden]
        let expected = (attn_broadcast + mean_broadcast)?;

        let out_v: Vec<f32> = out.flatten_all()?.to_vec1()?;
        let exp_v: Vec<f32> = expected.flatten_all()?.to_vec1()?;
        assert_eq!(out_v.len(), exp_v.len());
        let mut max_err = 0f32;
        for (o, e) in out_v.iter().zip(exp_v.iter()) {
            max_err = max_err.max((o - e).abs());
        }
        // Sinkhorn with eps=1e-6 over many iters keeps comb extremely close to
        // 1/hc even with eps; allow ~1% slack on the residual-sum term.
        assert!(
            max_err < 0.05,
            "mix_attn zero-params output diverged from analytical mean+x: max_err={max_err}"
        );
        Ok(())
    }

    /// 4. Reference vs. computed value: hc_head with known params should match a
    /// hand-computed result for a tiny case.
    #[test]
    fn hc_head_matches_reference() -> Result<()> {
        let dev = Device::Cpu;
        let hidden = 2;
        let hc_mult = 2;
        let hc_dim = hc_mult * hidden;

        // hc_head_fn: identity-ish so `mixes` is easy to compute by hand.
        let fn_w = Tensor::from_vec(
            vec![
                1.0, 0.0, 0.0, 0.0, // row 0: pick x[0]
                0.0, 0.0, 1.0, 0.0, // row 1: pick x[2]
            ],
            (hc_mult, hc_dim),
            &dev,
        )?;
        let base = Tensor::zeros(hc_mult, DType::F32, &dev)?;
        let scale = Tensor::ones(1, DType::F32, &dev)?;

        let head = V4MHCHead {
            hc_head_fn: fn_w,
            hc_head_base: base,
            hc_head_scale: scale,
            hc_mult,
            hidden_size: hidden,
            rt: V4MHCRuntime {
                hc_mult,
                hc_sinkhorn_iters: DEFAULT_HC_SINKHORN_ITERS,
                hc_eps: DEFAULT_HC_EPS,
                rms_norm_eps: 1e-6,
            },
        };

        // x shape [N=1, hc_mult=2, hidden=2]
        // x = [[ [1, 2], [3, 4] ]]   so x_flat = [[1, 2, 3, 4]]
        // x_flat^2 = [[1, 4, 9, 16]], mean = 30/4 = 7.5, +eps ≈ 7.5
        // rsqrt = 1/sqrt(7.5) ≈ 0.3651
        // mixes_raw = fn @ x_flat^T = [[ x_flat[0]=1, x_flat[2]=3 ]] = [[1, 3]]
        // mixes = [[1, 3]] * 0.3651 ≈ [[0.3651, 1.0954]]
        // sigmoid(mixes) + eps ≈ [[0.5903, 0.7494]]
        // y[h] = w[0] * x[0, h] + w[1] * x[1, h]
        //   y[0] = 0.5903*1 + 0.7494*3 = 0.5903 + 2.2482 = 2.8385
        //   y[1] = 0.5903*2 + 0.7494*4 = 1.1806 + 2.9976 = 4.1782
        let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (1, hc_mult, hidden), &dev)?;
        let y = head.forward(&x)?;
        assert_eq!(y.dims(), &[1, hidden]);
        let v: Vec<f32> = y.flatten_all()?.to_vec1()?;

        // Re-derive with f64 precision for the reference.
        let mean_sq = (1.0 + 4.0 + 9.0 + 16.0) / 4.0;
        let rsqrt = 1.0 / (mean_sq + 1e-6_f64).sqrt();
        let m0 = 1.0 * rsqrt;
        let m1 = 3.0 * rsqrt;
        let w0 = 1.0 / (1.0 + (-m0).exp()) + 1e-6;
        let w1 = 1.0 / (1.0 + (-m1).exp()) + 1e-6;
        let exp_y0 = (w0 * 1.0 + w1 * 3.0) as f32;
        let exp_y1 = (w0 * 2.0 + w1 * 4.0) as f32;

        assert!(
            (v[0] - exp_y0).abs() < 1e-4,
            "hc_head y[0]={} expected {}",
            v[0],
            exp_y0
        );
        assert!(
            (v[1] - exp_y1).abs() < 1e-4,
            "hc_head y[1]={} expected {}",
            v[1],
            exp_y1
        );
        Ok(())
    }

    /// Sinkhorn of all-zero input yields rows and columns each summing to ~1.0
    /// after enough iters (allowing for eps).
    #[test]
    fn sinkhorn_zero_input_is_doubly_stochastic() -> Result<()> {
        let dev = Device::Cpu;
        let hc = 4;
        let comb = Tensor::zeros((1, hc, hc), DType::F32, &dev)?;
        let normed = sinkhorn_normalize(&comb, 20, 1e-6)?;
        // Each row should sum to ~1.0, each col should sum to ~1.0.
        let row_sums: Vec<f32> = normed.sum(2)?.flatten_all()?.to_vec1()?;
        let col_sums: Vec<f32> = normed.sum(1)?.flatten_all()?.to_vec1()?;
        for s in &row_sums {
            assert!((s - 1.0).abs() < 1e-3, "row sum {s} not ≈ 1.0");
        }
        for s in &col_sums {
            assert!((s - 1.0).abs() < 1e-3, "col sum {s} not ≈ 1.0");
        }
        Ok(())
    }
}
