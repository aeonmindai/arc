#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
// WIP V4 mHC module: a helper (`collapse_4d_to_3d`) and config field are not
// yet wired in, and the port keeps the reference doc-comment layout. Keep these
// dead-code/doc lints quiet here (CI does not gate mistralrs-core; see
// arc-tools/CI_HYGIENE.md).
#![allow(
    dead_code,
    clippy::doc_overindented_list_items,
    clippy::doc_lazy_continuation
)]
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

    /// Lift a 3-D residual `[B, T, hidden]` (the embedding output) into a
    /// 4-D mHC residual `[B, T, hc_mult, hidden]` by replicating across the
    /// `hc_mult` stream axis. This is the model-entry lift used at the top
    /// of the decoder stack in [`DeepSeekV4::forward`].
    ///
    /// Algebraic identity: with zero-init mHC params, the 4-D stack is the
    /// uniform broadcast of the 3-D input, so a `mean(4d, dim=2)` reduces
    /// back to the original 3-D state. The actual learned mHC weights
    /// drive the per-layer mix into asymmetric stream states.
    pub fn lift_3d_to_4d(&self, xs: &Tensor) -> Result<Tensor> {
        let dims = xs.dims();
        if dims.len() != 3 {
            candle_core::bail!(
                "V4 mHC lift_3d_to_4d: expected rank-3 input [B, T, hidden], got {:?}",
                dims
            );
        }
        let (b, t, h) = (dims[0], dims[1], dims[2]);
        xs.unsqueeze(2)?
            .broadcast_as((b, t, self.hc_mult, h))?
            .contiguous()
    }

    /// Collapse a 4-D mHC residual `[B, T, hc_mult, hidden]` back to 3-D
    /// `[B, T, hidden]` by mean-reducing over the `hc_mult` stream axis.
    ///
    /// This is the *non-learned* fallback collapse — used when the global
    /// mHC head (`hc_head_*` tensors) is absent from the checkpoint but
    /// per-layer mHC params are present. The proper V4 collapse routes
    /// through [`V4MHCHead::forward`], which applies a learned sigmoid
    /// mixing across the `hc_mult` streams.
    pub fn collapse_4d_to_3d(&self, xs_4d: &Tensor) -> Result<Tensor> {
        let dims = xs_4d.dims();
        if dims.len() != 4 {
            candle_core::bail!(
                "V4 mHC collapse_4d_to_3d: expected rank-4 input \
                 [B, T, hc_mult, hidden], got {:?}",
                dims
            );
        }
        if dims[2] != self.hc_mult {
            candle_core::bail!(
                "V4 mHC collapse_4d_to_3d: stream axis dim {} != hc_mult={}",
                dims[2],
                self.hc_mult
            );
        }
        xs_4d.mean(2)
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
        real_device: &candle_core::Device,
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
            vb.get_with_hints_dtype(shape, name, Default::default(), DType::F32)?
                .to_device(real_device)
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
    ///
    /// ## dtype contract
    ///
    /// `y` is returned in the *input* dtype (BF16 in production) because it
    /// feeds `input_layernorm` + attention/MLP, which run in the model dtype.
    ///
    /// `post` and `comb` are returned in **F32** — the dtype the sinkhorn /
    /// sigmoid math already computed them in. They are consumed only by
    /// [`hc_post`], which needs them in F32 anyway; casting them down to BF16
    /// here and back up there was a pure round trip that cost four kernel
    /// launches per call and *lost* mantissa bits. Keeping them in F32 removes
    /// the launches and raises precision. `hc_post` still returns the residual
    /// stack in the input dtype, so nothing downstream sees a dtype change.
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

        // Defensively cast weight tensors to F32 — try_load already produces F32,
        // but hand-constructed callers (tests, external integrators) may not.
        let hc_fn_f32 = hc_fn.to_dtype(DType::F32)?;
        let hc_scale_f32 = hc_scale.to_dtype(DType::F32)?;
        let hc_base_f32 = hc_base.to_dtype(DType::F32)?;
        let mixes_raw = x_flat.matmul(&hc_fn_f32.t()?)?;

        // Everything from here to `comb_pre` is ONE fused kernel on CUDA. The
        // eager chain below it is 18 launches — a 7-launch hand-decomposed RMS
        // statistic (`sqr -> fast_sum -> affine -> affine -> recip -> sqrt ->
        // bmul`) plus 11 for the three scoring blocks — all on 24 floats once
        // the reduction is done, twice per layer, 43 layers. At b=1 that is
        // ~1,460 of the decode step's measured 7,494 kernel launches, i.e. pure
        // launch overhead. `cuda/hc_fused.cu` is bit-identical to this chain by
        // construction, not by tolerance; the eager path stays reachable via
        // `ARC_HC_FUSED=0` so the two can be A/B'd from one binary — and they
        // were: flipping it moves 1,803 launches/step and 8.04 ms/token while
        // leaving 6 greedy completions and their 768 logprobs bit-identical.
        let on_cuda = crate::cuda::hc_fused::usable(&x_flat);
        let shapes_ok = x_flat.is_contiguous()
            && mixes_raw.is_contiguous()
            && hc_scale_f32.is_contiguous()
            && hc_base_f32.is_contiguous()
            && hc_base_f32.dims1().map(|v| v == (2 + hc) * hc).unwrap_or(false)
            && hc_scale_f32.dims1().map(|v| v == 3).unwrap_or(false);
        if on_cuda && !shapes_ok {
            // Falling back on CUDA is a silent 18-launch regression that looks
            // exactly like "the fusion didn't help". Say so once rather than
            // letting a layout change quietly undo the optimisation.
            static WARNED: std::sync::Once = std::sync::Once::new();
            WARNED.call_once(|| {
                tracing::warn!(
                    "V4 mHC: fused hc_pre kernel unusable (x_flat contig={}, mixes contig={}, \
                     scale contig={} dims={:?}, base contig={} dims={:?}) — falling back to the \
                     18-launch eager chain.",
                    x_flat.is_contiguous(),
                    mixes_raw.is_contiguous(),
                    hc_scale_f32.is_contiguous(),
                    hc_scale_f32.dims(),
                    hc_base_f32.is_contiguous(),
                    hc_base_f32.dims(),
                );
            });
        }
        let fused = on_cuda && shapes_ok;

        let (pre, post, comb_pre) = if fused {
            crate::cuda::hc_fused::hc_pre_fused_cuda(
                &x_flat,
                &mixes_raw,
                &hc_scale_f32,
                &hc_base_f32,
                hc,
                self.rt.rms_norm_eps,
                self.rt.hc_eps,
            )?
        } else {
            // rsqrt(mean(x^2) + eps)
            let sq_mean = x_flat.sqr()?.mean_keepdim(D::Minus1)?;
            let rsqrt = (sq_mean + self.rt.rms_norm_eps)?.recip()?.sqrt()?; // [N, 1]

            // mixes = (x_flat @ fn^T) * rsqrt  →  [N, mix_hc]
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
            // NOTE: use affine() for the scalar *2 rather than a device-scalar
            // Tensor::new(2f32, device) — the latter is a per-call CPU->GPU sync
            // (CLAUDE.md pitfall #5) that breaks CUDA-graph capture of the decode
            // forward. affine folds the constant into the kernel, no allocation.
            let post_sig = candle_nn::ops::sigmoid(
                &(post_block.broadcast_mul(&s_post)?.broadcast_add(&b_post)?),
            )?;
            let post = post_sig.affine(2.0, 0.0)?;

            let comb_pre = comb_block
                .broadcast_mul(&s_comb)?
                .broadcast_add(&b_comb)?; // [N, hc, hc]
            (pre, post, comb_pre)
        };

        // comb = sinkhorn_normalize(comb_block * s_comb + b_comb)
        let comb = sinkhorn_normalize(&comb_pre, self.rt.hc_sinkhorn_iters, self.rt.hc_eps)?;

        // y = sum_i pre[..., i, None] * x[..., i, :]  →  [N, hidden]
        // `x_flat` is already the F32 promotion of exactly these elements (it
        // is `x` reshaped to [N, hc*h] then cast), so reshape it back rather
        // than issuing a second BF16->F32 cast of the same buffer. Reshape of
        // a contiguous tensor is metadata-only: no kernel, no allocation. The
        // values are bit-identical — BF16->F32 is an exact widening.
        let x_f32 = x_flat.reshape((n, hc, h))?; // [N, hc, h], F32
        let pre_b = pre.unsqueeze(D::Minus1)?; // [N, hc, 1]
        let y = pre_b.broadcast_mul(&x_f32)?.sum(1)?; // [N, hidden]

        // Restore leading dims.
        let mut y_shape = leading_dims.clone();
        y_shape.push(h);
        let mut post_shape = leading_dims.clone();
        post_shape.push(hc);
        let mut comb_shape = leading_dims.clone();
        comb_shape.push(hc);
        comb_shape.push(hc);

        // `y` feeds layernorm + attention/MLP, so it goes back to the model
        // dtype. `post` / `comb` stay F32 — see the dtype contract above.
        let y_out = y.reshape(y_shape)?.to_dtype(in_dtype)?;
        let post_out = post.reshape(post_shape)?;
        let comb_out = comb.reshape(comb_shape)?;
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
    /// `post` / `comb` arrive in F32 from [`hc_pre`]; the `to_dtype` calls
    /// below are retained only for hand-constructed callers and short-circuit
    /// to a clone (no kernel) on the production path.
    ///
    /// Returns the new residual stack with the same shape *and dtype* as `x`
    /// (the branch output), i.e. the model dtype.
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

    /// 4-D attn-side pre-step. Mirrors the SGLang `DeepseekV4DecoderLayer`
    /// attention block's `hc_pre` call (`hc_attn_*` weights). Returns the
    /// 3-D `y` (the learned single-stream collapse fed into `input_layernorm`
    /// + attention) along with `post` and `comb` (held for the matching
    /// `attn_post` call).
    ///
    /// Used by [`DeepSeekV4::forward`] in the 4-D end-to-end residual
    /// threading path (RUN-164).
    ///
    /// `residual_4d`: `[B, T, hc_mult, hidden]`
    /// Returns `(y, post, comb)`:
    /// - `y`:    `[B, T, hidden]`
    /// - `post`: `[B, T, hc_mult]`
    /// - `comb`: `[B, T, hc_mult, hc_mult]`
    pub fn attn_pre(&self, residual_4d: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        self.hc_pre(
            residual_4d,
            &self.hc_attn_fn,
            &self.hc_attn_scale,
            &self.hc_attn_base,
        )
    }

    /// 4-D FFN-side pre-step. Mirrors `attn_pre` but uses `hc_ffn_*` weights.
    pub fn ffn_pre(&self, residual_4d: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        self.hc_pre(
            residual_4d,
            &self.hc_ffn_fn,
            &self.hc_ffn_scale,
            &self.hc_ffn_base,
        )
    }

    /// 4-D post-step alias for [`hc_post`]. Exists for symmetry with
    /// [`attn_pre`] / [`ffn_pre`] in the 4-D end-to-end flow; the same call
    /// is used for both the attention-side and the FFN-side post-mix
    /// because the `(post, comb)` tensors fed in already encode which
    /// weights drove the pre-step.
    pub fn mix_post_4d(
        &self,
        branch_out: &Tensor,
        residual_4d: &Tensor,
        post: &Tensor,
        comb: &Tensor,
    ) -> Result<Tensor> {
        self.hc_post(branch_out, residual_4d, post, comb)
    }

    /// 3-D ↔ 4-D bridge for the attention block — **legacy / partial-V4
    /// fallback only**.
    ///
    /// Originally RUN-169 used this bridge as the active residual-mix path
    /// inside `DecoderLayer::forward` while the model still threaded a 3-D
    /// residual `[B, T, hidden]` and the full 4-D `[B, T, hc_mult, hidden]`
    /// threading (RUN-164) was not yet in place.
    ///
    /// RUN-164 lifts the residual stack to 4-D end-to-end through the entire
    /// decoder stack; the active path now uses [`attn_pre`] + [`mix_post_4d`]
    /// directly without the lift/collapse-per-layer overhead.
    ///
    /// This bridge is retained for partial-V4 checkpoints — i.e. cases where
    /// per-layer `hc_attn_*` / `hc_ffn_*` tensors are present but the global
    /// `hc_head_*` is absent. In that scenario the model-level forward keeps
    /// a 3-D residual and per-layer mixing falls back to the bridge.
    ///
    /// Input/output:
    /// - `residual`: `[B, T, hidden]`
    /// - `attn_out`: `[B, T, hidden]`
    /// - returns: `[B, T, hidden]`
    ///
    /// The bridge expands the 3-D residual into 4-D by broadcasting it across
    /// the `hc_mult` stream axis (so all four parallel residual streams start
    /// equal), calls `mix_attn`, then folds the result back to 3-D by averaging
    /// the `hc_mult` streams. The learned `hc_attn_scale` / `hc_attn_fn` /
    /// `hc_attn_base` weights drive the mixing inside `hc_pre`/`hc_post`.
    ///
    /// Algebraic identity check (audit §0, §2): with all-zero mHC params
    /// (`hc_attn_fn = hc_attn_base = hc_attn_scale = 0`) this degenerates to
    /// the standard residual `attn_out + residual`, which is the previous
    /// fall-through behavior. Any non-zero learned `hc_attn_scale` makes the
    /// output diverge from the standard residual — that is the assertion the
    /// RUN-169 unit test pins.
    pub fn mix_attn_3d_bridge(&self, residual: &Tensor, attn_out: &Tensor) -> Result<Tensor> {
        let dims = residual.dims();
        if dims.len() != 3 {
            candle_core::bail!(
                "V4 mHC mix_attn_3d_bridge: residual must be 3-D [B, T, hidden], got {:?}",
                dims
            );
        }
        let (b, t, h) = (dims[0], dims[1], dims[2]);
        if h != self.hidden_size {
            candle_core::bail!(
                "V4 mHC mix_attn_3d_bridge: residual hidden={h} disagrees with params hidden={}",
                self.hidden_size
            );
        }
        let hc = self.hc_mult;

        // Broadcast 3-D residual → 4-D [B, T, hc_mult, hidden] (replicate
        // the same value across all hc_mult stream slots).
        let residual_4d = residual
            .unsqueeze(2)?
            .broadcast_as((b, t, hc, h))?
            .contiguous()?;

        let mixed_4d = self.mix_attn(&residual_4d, attn_out)?; // [B, T, hc_mult, hidden]

        // Fold the hc_mult streams back to 3-D via mean reduction. Mean (not
        // sum) preserves the magnitude of `attn_out + residual` at the
        // zero-params identity point above.
        mixed_4d.mean(2)
    }

    /// 3-D ↔ 4-D bridge for the FFN block (mirrors `mix_attn_3d_bridge`).
    /// See [`mix_attn_3d_bridge`] for the legacy / partial-V4 fallback
    /// usage note.
    pub fn mix_ffn_3d_bridge(&self, residual: &Tensor, ffn_out: &Tensor) -> Result<Tensor> {
        let dims = residual.dims();
        if dims.len() != 3 {
            candle_core::bail!(
                "V4 mHC mix_ffn_3d_bridge: residual must be 3-D [B, T, hidden], got {:?}",
                dims
            );
        }
        let (b, t, h) = (dims[0], dims[1], dims[2]);
        if h != self.hidden_size {
            candle_core::bail!(
                "V4 mHC mix_ffn_3d_bridge: residual hidden={h} disagrees with params hidden={}",
                self.hidden_size
            );
        }
        let hc = self.hc_mult;

        let residual_4d = residual
            .unsqueeze(2)?
            .broadcast_as((b, t, hc, h))?
            .contiguous()?;
        let mixed_4d = self.mix_ffn(&residual_4d, ffn_out)?;
        mixed_4d.mean(2)
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
pub(crate) fn sinkhorn_normalize(comb: &Tensor, sinkhorn_iters: usize, eps: f64) -> Result<Tensor> {
    // RUN-161 throughput: fused single-launch CUDA kernel replaces the ~123-op
    // candle chain below (the dominant decode cost — ~13k launch-bound
    // micro-kernels/token across 43 layers). Default-ON; opt out with
    // ARC_NO_FUSED_SINKHORN=1. Falls back to the candle path otherwise
    // (non-CUDA, non-F32, opted out, or any kernel error — silent
    // fall-through).
    //
    // Bit-identity status: VALIDATED on H200, 2026-08-13 (GPU session 2).
    // Both halves of the flip-the-default criteria passed:
    //   - run_ppl.sh --sinkhorn-ab: per-chunk ppl bit-identical on vs off
    //     (7.7113123 / 12.602792 / 16.043064 both ways; ppl_sink_verdict.json
    //     bit_identical=true).
    //   - run_sinkhorn_ab.py: greedy 128-token outputs token-identical on
    //     6/6 prompts, 0 mismatches (sinkhorn_verdict.json).
    // History: the first kernel FAILED the earlier H200 A/B (ppl 58.85084 vs
    // 58.88946, 4/6 greedy divergences) — sequential reductions +
    // --use_fast_math compilation did not match this candle chain's rounding.
    // Fixed by construction in cuda/sinkhorn.cu (candle fast_sum/fast_max
    // pairwise-tree order, unfused IEEE _rn arithmetic, dedicated
    // no-fast-math build); see the contract comment there and the bitwise
    // replica tests in cuda/sinkhorn.rs.
    if std::env::var_os("ARC_NO_FUSED_SINKHORN").is_none()
        && matches!(comb.device(), candle_core::Device::Cuda(_))
        && comb.dtype() == DType::F32
    {
        if let Ok(out) = crate::cuda::sinkhorn::sinkhorn_normalize_cuda(comb, sinkhorn_iters, eps) {
            return Ok(out);
        }
        // fall through to candle path on any error
    }

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
    pub fn try_load(
        cfg: &DeepSeekV4Config,
        vb: &ShardedVarBuilder,
        real_device: &candle_core::Device,
    ) -> Option<Self> {
        let rt = V4MHCRuntime::from_cfg(cfg);
        let hc_mult = rt.hc_mult;
        let hc_dim = hc_mult * cfg.hidden_size;

        let needed = ["hc_head_fn", "hc_head_base", "hc_head_scale"];
        if !needed.iter().all(|n| vb.contains_tensor(n)) {
            return None;
        }

        let load = |name: &str, shape: &[usize]| -> Result<Tensor> {
            vb.get_with_hints_dtype(shape, name, Default::default(), DType::F32)?
                .to_device(real_device)
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
        assert!(V4MHCLayerParams::try_load(&cfg, &vb, 0, &Device::Cpu).is_none());
        assert!(V4MHCHead::try_load(&cfg, &vb, &Device::Cpu).is_none());
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
        let vb = vb_from(tensors, dev.clone());
        let params = V4MHCLayerParams::try_load(&cfg, &vb, 0, &dev)
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

    /// RUN-169: with zero mHC params, the 3-D bridge degenerates exactly to
    /// the standard residual `attn_out + residual`. This pins the
    /// backwards-compat identity so older mHC-absent fixtures behave the
    /// same way.
    #[test]
    fn mix_attn_3d_bridge_zero_params_matches_standard_residual() -> Result<()> {
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
        let residual = Tensor::randn(0f32, 1f32, (b, t, hidden), &dev)?;
        let attn_out = Tensor::randn(0f32, 1f32, (b, t, hidden), &dev)?;

        let bridged = params.mix_attn_3d_bridge(&residual, &attn_out)?;
        assert_eq!(bridged.dims(), &[b, t, hidden]);

        // Compare against the standard residual baseline.
        let baseline = (&attn_out + &residual)?;
        let br_v: Vec<f32> = bridged.flatten_all()?.to_vec1()?;
        let bl_v: Vec<f32> = baseline.flatten_all()?.to_vec1()?;
        assert_eq!(br_v.len(), bl_v.len());
        let mut max_err = 0f32;
        for (b_val, bl) in br_v.iter().zip(bl_v.iter()) {
            max_err = max_err.max((b_val - bl).abs());
        }
        // The sinkhorn-on-zeros step + the +hc_eps inside `pre` and `comb`
        // give a tiny perturbation around the analytical identity; allow
        // ~5% slack matching the existing zero-params test above.
        assert!(
            max_err < 0.05,
            "mix_attn_3d_bridge zero-params output diverged from analytical \
             standard residual: max_err={max_err}"
        );
        Ok(())
    }

    /// RUN-169: with a non-trivial `hc_attn_scale` (and non-trivial
    /// `hc_attn_fn`/`hc_attn_base`) the bridge output MUST differ from the
    /// standard residual baseline. This is the core assertion that the
    /// learned blend coefficient actually drives the forward pass — i.e.
    /// the 0.5/0.5 (and the silent `xs + residual`) fall-through has been
    /// replaced.
    #[test]
    fn mix_attn_3d_bridge_uses_hc_attn_scale() -> Result<()> {
        let dev = Device::Cpu;
        let hidden = 4;
        let hc_mult = 4;
        let mix_hc = (2 + hc_mult) * hc_mult;
        let hc_dim = hc_mult * hidden;

        // Non-trivial fn / base / scale so that mixes != 0 and the
        // sigmoid/sinkhorn produce values away from 0.5 / uniform.
        let hc_attn_fn = Tensor::from_vec(
            (0..mix_hc * hc_dim)
                .map(|i| ((i as f32) * 0.041).sin() * 0.5)
                .collect::<Vec<_>>(),
            (mix_hc, hc_dim),
            &dev,
        )?;
        let hc_attn_base = Tensor::from_vec(
            (0..mix_hc)
                .map(|i| ((i as f32) * 0.13).cos() * 0.2)
                .collect::<Vec<_>>(),
            mix_hc,
            &dev,
        )?;
        // hc_attn_scale[3]: distinct non-zero values across the (pre, post,
        // comb) slots to ensure each pathway is exercised.
        let hc_attn_scale = Tensor::from_vec(vec![1.5f32, -1.0, 0.75], 3, &dev)?;

        let params = make_layer_params(hidden, hc_mult, hc_attn_fn, hc_attn_base, hc_attn_scale);

        let b = 2;
        let t = 3;
        // Use a fixed seed-equivalent deterministic input so the test is
        // reproducible (`Tensor::randn` is deterministic given a seed env,
        // but we use `from_vec` here for explicit determinism).
        let residual = Tensor::from_vec(
            (0..b * t * hidden)
                .map(|i| ((i as f32) * 0.071).sin() * 1.3)
                .collect::<Vec<_>>(),
            (b, t, hidden),
            &dev,
        )?;
        let attn_out = Tensor::from_vec(
            (0..b * t * hidden)
                .map(|i| ((i as f32) * 0.059).cos() * 0.7)
                .collect::<Vec<_>>(),
            (b, t, hidden),
            &dev,
        )?;

        let bridged = params.mix_attn_3d_bridge(&residual, &attn_out)?;
        assert_eq!(bridged.dims(), &[b, t, hidden]);

        let baseline = (&attn_out + &residual)?;
        let br_v: Vec<f32> = bridged.flatten_all()?.to_vec1()?;
        let bl_v: Vec<f32> = baseline.flatten_all()?.to_vec1()?;
        assert_eq!(br_v.len(), bl_v.len());

        // The learned blend must move the output OFF the standard residual
        // baseline for at least a non-trivial fraction of elements. We use
        // a per-element diff threshold + a count check, matching the style
        // of the RUN-162 real-weights-differ-from-uniform test.
        let diff_count = br_v
            .iter()
            .zip(bl_v.iter())
            .filter(|(b_val, bl)| (*b_val - *bl).abs() > 1e-3)
            .count();
        assert!(
            diff_count >= br_v.len() / 2,
            "mix_attn_3d_bridge with non-zero hc_attn_scale produced output \
             matching the standard residual baseline for {} / {} elements. \
             The learned blend coefficient is not driving the forward pass.",
            br_v.len() - diff_count,
            br_v.len()
        );

        // Sanity: also assert the bridge is finite (no NaN/Inf from
        // sinkhorn instability under the test parameters).
        for v in &br_v {
            assert!(
                v.is_finite(),
                "mix_attn_3d_bridge produced non-finite value {v}"
            );
        }
        Ok(())
    }

    /// RUN-169: rank validation — the 3-D bridge rejects 4-D input (callers
    /// must use `mix_attn` directly for 4-D, the bridge is the 3-D ↔ 4-D
    /// adapter used by `DecoderLayer::forward`).
    #[test]
    fn mix_attn_3d_bridge_rejects_non_3d() -> Result<()> {
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

        // 4-D residual should fail.
        let residual_4d = Tensor::zeros((1, 2, hc_mult, hidden), DType::F32, &dev)?;
        let attn_out = Tensor::zeros((1, 2, hidden), DType::F32, &dev)?;
        assert!(params.mix_attn_3d_bridge(&residual_4d, &attn_out).is_err());

        // Mismatched hidden dim should fail.
        let residual_bad = Tensor::zeros((1, 2, hidden + 1), DType::F32, &dev)?;
        let attn_out_bad = Tensor::zeros((1, 2, hidden + 1), DType::F32, &dev)?;
        assert!(params
            .mix_attn_3d_bridge(&residual_bad, &attn_out_bad)
            .is_err());
        Ok(())
    }

    /// RUN-164: `lift_3d_to_4d` produces shape `[B, T, hc_mult, hidden]` with
    /// the input replicated across the new `hc_mult` axis.
    #[test]
    fn lift_3d_to_4d_replicates_streams() -> Result<()> {
        let dev = Device::Cpu;
        let hidden = 4;
        let hc_mult = 4;
        let rt = V4MHCRuntime {
            hc_mult,
            hc_sinkhorn_iters: DEFAULT_HC_SINKHORN_ITERS,
            hc_eps: DEFAULT_HC_EPS,
            rms_norm_eps: 1e-6,
        };

        let b = 2;
        let t = 3;
        let xs = Tensor::from_vec(
            (0..b * t * hidden).map(|i| i as f32).collect::<Vec<_>>(),
            (b, t, hidden),
            &dev,
        )?;
        let xs_4d = rt.lift_3d_to_4d(&xs)?;
        assert_eq!(xs_4d.dims(), &[b, t, hc_mult, hidden]);

        // Each stream slot k must equal the original 3-D input.
        for k in 0..hc_mult {
            let stream_k = xs_4d.narrow(2, k, 1)?.squeeze(2)?;
            let xs_v: Vec<f32> = xs.flatten_all()?.to_vec1()?;
            let k_v: Vec<f32> = stream_k.flatten_all()?.to_vec1()?;
            assert_eq!(xs_v.len(), k_v.len());
            for (a, b) in xs_v.iter().zip(k_v.iter()) {
                assert!(
                    (a - b).abs() < 1e-6,
                    "stream {k} differs from input: {a} vs {b}"
                );
            }
        }
        Ok(())
    }

    /// RUN-164: `collapse_4d_to_3d` (mean reduction) is the inverse of
    /// `lift_3d_to_4d` when no per-layer mixing has happened.
    #[test]
    fn collapse_4d_to_3d_inverse_of_lift() -> Result<()> {
        let dev = Device::Cpu;
        let hidden = 4;
        let hc_mult = 4;
        let rt = V4MHCRuntime {
            hc_mult,
            hc_sinkhorn_iters: DEFAULT_HC_SINKHORN_ITERS,
            hc_eps: DEFAULT_HC_EPS,
            rms_norm_eps: 1e-6,
        };

        let b = 1;
        let t = 4;
        let xs = Tensor::from_vec(
            (0..b * t * hidden).map(|i| (i as f32) * 0.7).collect(),
            (b, t, hidden),
            &dev,
        )?;
        let xs_4d = rt.lift_3d_to_4d(&xs)?;
        let xs_back = rt.collapse_4d_to_3d(&xs_4d)?;
        assert_eq!(xs_back.dims(), xs.dims());
        let xs_v: Vec<f32> = xs.flatten_all()?.to_vec1()?;
        let back_v: Vec<f32> = xs_back.flatten_all()?.to_vec1()?;
        for (a, b) in xs_v.iter().zip(back_v.iter()) {
            assert!((a - b).abs() < 1e-5, "collapse(lift(xs)) != xs: {a} vs {b}");
        }
        Ok(())
    }

    /// RUN-164: `lift_3d_to_4d` rejects non-3-D input.
    #[test]
    fn lift_3d_to_4d_rejects_wrong_rank() -> Result<()> {
        let dev = Device::Cpu;
        let rt = V4MHCRuntime {
            hc_mult: 4,
            hc_sinkhorn_iters: DEFAULT_HC_SINKHORN_ITERS,
            hc_eps: DEFAULT_HC_EPS,
            rms_norm_eps: 1e-6,
        };
        // 2-D and 4-D should fail.
        let bad_2d = Tensor::zeros((4, 8), DType::F32, &dev)?;
        assert!(rt.lift_3d_to_4d(&bad_2d).is_err());
        let bad_4d = Tensor::zeros((1, 2, 4, 8), DType::F32, &dev)?;
        assert!(rt.lift_3d_to_4d(&bad_4d).is_err());
        Ok(())
    }

    /// RUN-164: end-to-end 4-D residual threading through a synthetic
    /// 2-layer decoder stack. Asserts:
    /// (a) intermediate residual stays 4-D throughout all layers,
    /// (b) only the final head-collapse step returns to 3-D,
    /// (c) the resulting 3-D state has the right shape for `lm_head`.
    ///
    /// Uses identity-like "attention" and "FFN" callables (just a small
    /// linear pass + residual broadcast) so the test focuses on the mHC
    /// residual plumbing rather than the attention math.
    #[test]
    fn decoder_layer_threads_4d_residual_end_to_end() -> Result<()> {
        let dev = Device::Cpu;
        let hidden = 4;
        let hc_mult = 4;
        let mix_hc = (2 + hc_mult) * hc_mult;
        let hc_dim = hc_mult * hidden;

        // Build 2 layers worth of mHC params with a deterministic, small,
        // non-zero pattern so the learned blend actually drives the path.
        let make_params = |seed: f32| -> V4MHCLayerParams {
            let hc_attn_fn = Tensor::from_vec(
                (0..mix_hc * hc_dim)
                    .map(|i| ((i as f32) * 0.013 + seed).sin() * 0.3)
                    .collect::<Vec<_>>(),
                (mix_hc, hc_dim),
                &dev,
            )
            .unwrap();
            let hc_attn_base = Tensor::from_vec(
                (0..mix_hc)
                    .map(|i| ((i as f32) * 0.21 + seed).cos() * 0.1)
                    .collect::<Vec<_>>(),
                mix_hc,
                &dev,
            )
            .unwrap();
            let hc_attn_scale = Tensor::from_vec(
                vec![1.0f32 + seed * 0.1, -0.5 + seed * 0.05, 0.7 - seed * 0.03],
                3,
                &dev,
            )
            .unwrap();
            make_layer_params(hidden, hc_mult, hc_attn_fn, hc_attn_base, hc_attn_scale)
        };
        let layer0_params = make_params(0.0);
        let layer1_params = make_params(1.0);
        let layers = [&layer0_params, &layer1_params];

        // Head params: identity-ish fn so we can reduce 4-D back to 3-D.
        let head_fn = {
            // [hc_mult, hc_dim] — for each output channel k, take stream k
            // hidden_size-wide slice. So fn[k, k*hidden..(k+1)*hidden] = 1
            // and zero elsewhere; this means `mixes[n, k] = sum(x[n, k, :])`.
            let mut v = vec![0.0f32; hc_mult * hc_dim];
            for k in 0..hc_mult {
                for h in 0..hidden {
                    v[k * hc_dim + k * hidden + h] = 1.0;
                }
            }
            Tensor::from_vec(v, (hc_mult, hc_dim), &dev)?
        };
        let head_base = Tensor::zeros(hc_mult, DType::F32, &dev)?;
        let head_scale = Tensor::ones(1, DType::F32, &dev)?;
        let head = V4MHCHead {
            hc_head_fn: head_fn,
            hc_head_base: head_base,
            hc_head_scale: head_scale,
            hc_mult,
            hidden_size: hidden,
            rt: layer0_params.rt,
        };

        // Synthetic input embedding: [B, T, hidden].
        let b = 2;
        let t = 3;
        let xs_3d = Tensor::from_vec(
            (0..b * t * hidden)
                .map(|i| ((i as f32) * 0.071).sin() * 1.1)
                .collect::<Vec<_>>(),
            (b, t, hidden),
            &dev,
        )?;

        // Model-entry lift.
        let mut xs_4d = layer0_params.rt.lift_3d_to_4d(&xs_3d)?;
        assert_eq!(xs_4d.dims(), &[b, t, hc_mult, hidden]);

        // Per-layer 4-D forward: thread the residual through each layer's
        // attn-side and ffn-side mHC mix.
        for (layer_idx, p) in layers.iter().enumerate() {
            // === ATTN BLOCK ===
            let residual_attn = xs_4d.clone();
            let (y_attn, post_attn, comb_attn) = p.attn_pre(&residual_attn)?;
            assert_eq!(y_attn.dims(), &[b, t, hidden], "layer {layer_idx} attn y");
            assert_eq!(
                post_attn.dims(),
                &[b, t, hc_mult],
                "layer {layer_idx} attn post"
            );
            assert_eq!(
                comb_attn.dims(),
                &[b, t, hc_mult, hc_mult],
                "layer {layer_idx} attn comb"
            );

            // Fake "attention output" = y_attn * 0.1 + small offset. The
            // exact shape (`[B, T, hidden]`) is what real attention returns.
            let two = Tensor::new(0.1f32, &dev)?;
            let attn_out = y_attn.broadcast_mul(&two)?;
            assert_eq!(attn_out.dims(), &[b, t, hidden]);

            xs_4d = p.mix_post_4d(&attn_out, &residual_attn, &post_attn, &comb_attn)?;
            assert_eq!(
                xs_4d.dims(),
                &[b, t, hc_mult, hidden],
                "layer {layer_idx} attn post xs_4d"
            );

            // === FFN BLOCK ===
            let residual_ffn = xs_4d.clone();
            let (y_ffn, post_ffn, comb_ffn) = p.ffn_pre(&residual_ffn)?;
            assert_eq!(y_ffn.dims(), &[b, t, hidden], "layer {layer_idx} ffn y");

            // Fake "FFN output" = y_ffn * (-0.05).
            let two = Tensor::new(-0.05f32, &dev)?;
            let ffn_out = y_ffn.broadcast_mul(&two)?;
            assert_eq!(ffn_out.dims(), &[b, t, hidden]);

            xs_4d = p.mix_post_4d(&ffn_out, &residual_ffn, &post_ffn, &comb_ffn)?;
            assert_eq!(
                xs_4d.dims(),
                &[b, t, hc_mult, hidden],
                "layer {layer_idx} ffn post xs_4d"
            );
        }

        // Model-exit collapse via the global mHC head.
        let xs_out = head.forward(&xs_4d)?;
        assert_eq!(xs_out.dims(), &[b, t, hidden]);

        // All-finite sanity (no NaN/Inf from sinkhorn instability under
        // the test scales).
        let out_v: Vec<f32> = xs_out.flatten_all()?.to_vec1()?;
        for v in &out_v {
            assert!(v.is_finite(), "non-finite value {v} in xs_out");
        }
        Ok(())
    }

    /// RUN-164: same input through the 3-D bridge (legacy path) vs the 4-D
    /// end-to-end path produces DIFFERENT outputs — they are not the same
    /// algorithm, even though both apply the same per-layer mHC params.
    /// The 4-D path keeps streams asymmetric layer-to-layer; the bridge
    /// resets them via the mean-collapse at every layer boundary.
    #[test]
    fn mhc_4d_path_differs_from_3d_bridge() -> Result<()> {
        let dev = Device::Cpu;
        let hidden = 4;
        let hc_mult = 4;
        let mix_hc = (2 + hc_mult) * hc_mult;
        let hc_dim = hc_mult * hidden;

        // Non-zero mHC params — necessary for the 4-D path to actually
        // differ from the bridge. With zero params both paths reduce to the
        // standard residual identity.
        let hc_attn_fn = Tensor::from_vec(
            (0..mix_hc * hc_dim)
                .map(|i| ((i as f32) * 0.041).sin() * 0.4)
                .collect::<Vec<_>>(),
            (mix_hc, hc_dim),
            &dev,
        )?;
        let hc_attn_base = Tensor::from_vec(
            (0..mix_hc)
                .map(|i| ((i as f32) * 0.13).cos() * 0.15)
                .collect::<Vec<_>>(),
            mix_hc,
            &dev,
        )?;
        let hc_attn_scale = Tensor::from_vec(vec![1.5f32, -1.0, 0.75], 3, &dev)?;
        let params = make_layer_params(hidden, hc_mult, hc_attn_fn, hc_attn_base, hc_attn_scale);

        let b = 2;
        let t = 3;
        let xs = Tensor::from_vec(
            (0..b * t * hidden)
                .map(|i| ((i as f32) * 0.071).sin() * 1.3)
                .collect::<Vec<_>>(),
            (b, t, hidden),
            &dev,
        )?;
        let attn_out = Tensor::from_vec(
            (0..b * t * hidden)
                .map(|i| ((i as f32) * 0.059).cos() * 0.7)
                .collect::<Vec<_>>(),
            (b, t, hidden),
            &dev,
        )?;

        // --- 3-D bridge path: a single attn-side mix.
        let bridged = params.mix_attn_3d_bridge(&xs, &attn_out)?;
        assert_eq!(bridged.dims(), &[b, t, hidden]);

        // --- 4-D path: lift, attn_pre, attn_post (no FFN; the bridge above
        // only mixes attn).
        let xs_4d = params.rt.lift_3d_to_4d(&xs)?;
        let (_y, post, comb) = params.attn_pre(&xs_4d)?;
        let mixed_4d = params.mix_post_4d(&attn_out, &xs_4d, &post, &comb)?;
        assert_eq!(mixed_4d.dims(), &[b, t, hc_mult, hidden]);
        // Collapse via mean — *without* the learned head, so we isolate the
        // contribution of the layer's mHC pre/post path.
        let collapsed = params.rt.collapse_4d_to_3d(&mixed_4d)?;
        assert_eq!(collapsed.dims(), &[b, t, hidden]);

        // The bridge mean-collapses inside `mix_attn_3d_bridge`, after the
        // 4-D mix. The 4-D-then-mean path here does the same operation but
        // with the input residual lifted from the same 3-D source — so
        // when only ONE attn mix is applied the two should be IDENTICAL
        // (since lift+attn_pre+attn_post+mean == mix_attn_3d_bridge by
        // construction). Pin this algebraic identity so future refactors
        // don't accidentally diverge them.
        let br_v: Vec<f32> = bridged.flatten_all()?.to_vec1()?;
        let co_v: Vec<f32> = collapsed.flatten_all()?.to_vec1()?;
        assert_eq!(br_v.len(), co_v.len());
        let mut max_err = 0f32;
        for (a, b) in br_v.iter().zip(co_v.iter()) {
            max_err = max_err.max((a - b).abs());
        }
        assert!(
            max_err < 1e-4,
            "1-layer 4D+collapse should match 3-D bridge exactly: max_err={max_err}"
        );

        // Now do a SECOND attn mix on the previous output. The 4-D path
        // threads the 4-D residual without re-broadcasting (streams stay
        // asymmetric); the bridge collapses to 3-D after layer 1 and
        // re-broadcasts before layer 2.
        let attn_out2 = Tensor::from_vec(
            (0..b * t * hidden)
                .map(|i| ((i as f32) * 0.087).sin() * 0.5)
                .collect::<Vec<_>>(),
            (b, t, hidden),
            &dev,
        )?;
        // 4-D path: continue from mixed_4d, do not re-broadcast.
        let (_y2, post2, comb2) = params.attn_pre(&mixed_4d)?;
        let mixed_4d_2 = params.mix_post_4d(&attn_out2, &mixed_4d, &post2, &comb2)?;
        let collapsed_2 = params.rt.collapse_4d_to_3d(&mixed_4d_2)?;

        // Bridge path: bridge from `bridged` (the 3-D residual after the
        // first mix), which lifts again with broadcast.
        let bridged_2 = params.mix_attn_3d_bridge(&bridged, &attn_out2)?;

        // After 2 layers the paths MUST diverge.
        let br2_v: Vec<f32> = bridged_2.flatten_all()?.to_vec1()?;
        let co2_v: Vec<f32> = collapsed_2.flatten_all()?.to_vec1()?;
        let diff_count = br2_v
            .iter()
            .zip(co2_v.iter())
            .filter(|(a, b)| (*a - *b).abs() > 1e-3)
            .count();
        assert!(
            diff_count >= br2_v.len() / 2,
            "after 2-layer mix, 4-D path and 3-D bridge should diverge for at least \
             half of elements; only {} / {} differ.",
            diff_count,
            br2_v.len()
        );

        // And both must remain finite.
        for v in br2_v.iter().chain(co2_v.iter()) {
            assert!(v.is_finite(), "non-finite value {v}");
        }
        Ok(())
    }

    /// RUN-164: with zero `hc_attn_*` / `hc_ffn_*` params, the 4-D path
    /// reduces to the identity "mean(residual) + branch_out" pattern at
    /// every stream slot, mirroring the standard residual identity that
    /// the 3-D bridge zero-params test pins. After mean-collapse the
    /// 4-D path produces `attn_out + residual` (within sinkhorn-on-zeros
    /// + hc_eps slack).
    #[test]
    fn mhc_4d_zero_params_identity() -> Result<()> {
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
        let xs = Tensor::from_vec(
            (0..b * t * hidden)
                .map(|i| ((i as f32) * 0.071).sin() * 1.3)
                .collect::<Vec<_>>(),
            (b, t, hidden),
            &dev,
        )?;
        let attn_out = Tensor::from_vec(
            (0..b * t * hidden)
                .map(|i| ((i as f32) * 0.059).cos() * 0.7)
                .collect::<Vec<_>>(),
            (b, t, hidden),
            &dev,
        )?;

        // 4-D path: lift → attn_pre → mix_post_4d → collapse(mean).
        let xs_4d = params.rt.lift_3d_to_4d(&xs)?;
        let (_y, post, comb) = params.attn_pre(&xs_4d)?;
        let mixed_4d = params.mix_post_4d(&attn_out, &xs_4d, &post, &comb)?;
        let collapsed = params.rt.collapse_4d_to_3d(&mixed_4d)?;

        // Expected = attn_out + xs (standard residual).
        let baseline = (&attn_out + &xs)?;
        let co_v: Vec<f32> = collapsed.flatten_all()?.to_vec1()?;
        let bl_v: Vec<f32> = baseline.flatten_all()?.to_vec1()?;
        let mut max_err = 0f32;
        for (a, b) in co_v.iter().zip(bl_v.iter()) {
            max_err = max_err.max((a - b).abs());
        }
        // sinkhorn-on-zeros + hc_eps slack, same tolerance as bridge.
        assert!(
            max_err < 0.05,
            "4-D zero-params path diverged from standard residual: max_err={max_err}"
        );
        Ok(())
    }

    /// Pins the `hc_pre` -> `hc_post` dtype contract that removes the F32
    /// round trip (430 kernel launches/token at 43 layers).
    ///
    /// With a BF16 residual — the production dtype — `hc_pre` must hand back
    /// `post` / `comb` in **F32**, not BF16. Casting them down here and back
    /// up inside `hc_post` was four launches per call whose only effect was
    /// to discard mantissa bits. `y` and the `hc_post` output must still come
    /// back in the model dtype, because they feed layernorm / attention and
    /// the residual stack respectively.
    #[test]
    fn hc_pre_hands_post_and_comb_to_hc_post_in_f32() -> Result<()> {
        let dev = Device::Cpu;
        let hidden = 4;
        let hc_mult = 4;
        let mix_hc = (2 + hc_mult) * hc_mult;
        let hc_dim = hc_mult * hidden;

        // Non-trivial params so post/comb are not degenerate.
        let params = make_layer_params(
            hidden,
            hc_mult,
            Tensor::from_vec(
                (0..mix_hc * hc_dim)
                    .map(|i| ((i as f32) * 0.031).sin() * 0.4)
                    .collect::<Vec<_>>(),
                (mix_hc, hc_dim),
                &dev,
            )?,
            Tensor::from_vec(
                (0..mix_hc)
                    .map(|i| ((i as f32) * 0.13).cos() * 0.2)
                    .collect::<Vec<_>>(),
                mix_hc,
                &dev,
            )?,
            Tensor::from_vec(vec![0.9f32, 1.1, 0.7], 3, &dev)?,
        );

        let (b, t) = (2, 3);
        let residual_f32 = Tensor::from_vec(
            (0..b * t * hc_mult * hidden)
                .map(|i| ((i as f32) * 0.047).sin() * 1.1)
                .collect::<Vec<_>>(),
            (b, t, hc_mult, hidden),
            &dev,
        )?;
        let branch_f32 = Tensor::from_vec(
            (0..b * t * hidden)
                .map(|i| ((i as f32) * 0.067).cos() * 0.6)
                .collect::<Vec<_>>(),
            (b, t, hidden),
            &dev,
        )?;

        let residual_bf16 = residual_f32.to_dtype(DType::BF16)?;
        let branch_bf16 = branch_f32.to_dtype(DType::BF16)?;

        let (y, post, comb) = params.attn_pre(&residual_bf16)?;

        // The contract: post/comb stay in the dtype the sinkhorn/sigmoid math
        // produced them in. If either of these flips back to BF16 the round
        // trip has been reintroduced.
        assert_eq!(
            post.dtype(),
            DType::F32,
            "hc_pre must return `post` in F32, not round-trip it through the input dtype"
        );
        assert_eq!(
            comb.dtype(),
            DType::F32,
            "hc_pre must return `comb` in F32, not round-trip it through the input dtype"
        );
        // `y` still feeds layernorm + attention, so it keeps the model dtype.
        assert_eq!(
            y.dtype(),
            DType::BF16,
            "hc_pre must return `y` in the input dtype"
        );
        assert_eq!(y.dims(), &[b, t, hidden]);
        assert_eq!(post.dims(), &[b, t, hc_mult]);
        assert_eq!(comb.dims(), &[b, t, hc_mult, hc_mult]);

        // The residual stack that leaves hc_post is still the model dtype.
        let out = params.mix_post_4d(&branch_bf16, &residual_bf16, &post, &comb)?;
        assert_eq!(
            out.dtype(),
            DType::BF16,
            "hc_post must return the model dtype"
        );
        assert_eq!(out.dims(), &[b, t, hc_mult, hidden]);

        // Sanity: keeping post/comb in F32 tracks the all-F32 computation at
        // least as closely as the old BF16 round trip did. Compare the BF16
        // round trip's output against the pure-F32 reference; the only
        // remaining error is the BF16 residual/branch inputs themselves.
        let (_y_ref, post_ref, comb_ref) = params.attn_pre(&residual_f32)?;
        let out_ref = params.mix_post_4d(&branch_f32, &residual_f32, &post_ref, &comb_ref)?;
        let got: Vec<f32> = out.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
        let want: Vec<f32> = out_ref.flatten_all()?.to_vec1()?;
        let mut max_err = 0f32;
        for (g, w) in got.iter().zip(want.iter()) {
            max_err = max_err.max((g - w).abs());
        }
        // BF16 carries ~8 mantissa bits; inputs of magnitude ~1 quantize to
        // ~4e-3, and hc_post sums hc_mult=4 of them.
        assert!(
            max_err < 0.05,
            "BF16 mHC round trip diverged from the F32 reference: max_err={max_err}"
        );
        Ok(())
    }
}
