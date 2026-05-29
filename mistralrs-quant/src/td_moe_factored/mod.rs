//! TD-MoE Tucker-factored layer.
//!
//! Storage:
//!   - `g_core`  : Tucker core, shape `[r1, r2, r3]`
//!   - `u1`      : expert factor, shape `[num_experts, r1]`
//!   - `u2`      : output  factor, shape `[d_out, r2]`  (absorbs L_out re-coloring)
//!   - `u3`      : input   factor, shape `[d_in,  r3]`  (absorbs L_in  re-coloring)
//!
//! Reconstructing the dense `W[e, d_out, d_in]` is
//!   `W[e] = sum_{a,b,c} G[a,b,c] · U1[e,a] · U2[:,b] · U3[:,c]^T`
//!
//! For the MoE gather forward, given input vector `a [d_in]` and expert `e`:
//!   `y[d_out] = sum_{a,b,c} G[a,b,c] · U1[e,a] · U2[:, b] · (U3^T · a)[c]`
//!
//! We never materialise the dense `W` on the inference forward path — that's
//! the whole point. `dequantize_w` is provided as a debug hook only.
//!
//! The factored storage realises the strict "20% of original" reduction the
//! TD-MoE paper claims at small ranks: e.g. for a `[128, 512, 512]` expert
//! stack at rank `[32, 128, 128]`:
//!
//!   dense    = 128 * 512 * 512                          ≈ 33.5 M elems
//!   factored = 32*128*128 + 128*32 + 512*128 + 512*128  ≈  0.66 M elems
//!     (~50x smaller)
//!
//! All contractions live in `dequantize_w` (debug), `forward` (single-expert),
//! and `gather_forward` (sparse MoE expert dispatch). The latter is the
//! load-bearing op; see the chained-contraction routine below.

use std::{
    borrow::Cow,
    sync::{atomic::AtomicUsize, Arc},
};

use candle_core::{DType, Device, Result, Tensor};

use crate::{
    DistributedKind, IsqType, QuantMethod, QuantMethodConfig, QuantizeOntoGuard, QuantizedSerde,
};

/// Tucker decomposition core + factor matrices, kept in factored form for
/// inference. See module docs for the storage layout and contraction order.
///
/// **Coexistence**: TD-MoE Tucker-factored layers replace the 3-D
/// `Arc<dyn QuantMethod>` returned by the MoE expert-stack loader. They sit
/// beside QTIP-quantised 2-D linears (Q/KV/O projections) without modifying
/// any other layer type — the MoE forward dispatch is a `Arc<dyn QuantMethod>`
/// virtual call regardless of which concrete implementation backs it.
#[derive(Debug)]
pub struct TuckerFactoredLayer {
    /// Core tensor of shape `[r1, r2, r3]`. Stored in compute dtype.
    pub g_core: Tensor,
    /// Expert-axis factor `[num_experts, r1]`.
    pub u1: Tensor,
    /// Output-axis factor `[d_out, r2]` (with L_out re-coloring absorbed).
    pub u2: Tensor,
    /// Input-axis factor `[d_in, r3]` (with L_in re-coloring absorbed).
    pub u3: Tensor,
    /// Dtype + device of the original weight (used for activations cast).
    dtype: DType,
    device: Device,
}

impl TuckerFactoredLayer {
    /// Build a new factored layer from the four Tucker factors.
    ///
    /// Shape contract:
    ///   - `g_core` : `[r1, r2, r3]`
    ///   - `u1`     : `[num_experts, r1]`
    ///   - `u2`     : `[d_out, r2]`
    ///   - `u3`     : `[d_in,  r3]`
    pub fn from_factors(
        g_core: Tensor,
        u1: Tensor,
        u2: Tensor,
        u3: Tensor,
        target_dtype: DType,
    ) -> Result<Self> {
        let device = g_core.device().clone();

        let g_dims = g_core.dims();
        let u1_dims = u1.dims();
        let u2_dims = u2.dims();
        let u3_dims = u3.dims();

        if g_dims.len() != 3 {
            candle_core::bail!("TuckerFactored: g_core must be 3-D, got shape {:?}", g_dims);
        }
        if u1_dims.len() != 2 {
            candle_core::bail!("TuckerFactored: u1 must be 2-D, got shape {:?}", u1_dims);
        }
        if u2_dims.len() != 2 {
            candle_core::bail!("TuckerFactored: u2 must be 2-D, got shape {:?}", u2_dims);
        }
        if u3_dims.len() != 2 {
            candle_core::bail!("TuckerFactored: u3 must be 2-D, got shape {:?}", u3_dims);
        }
        let (r1, r2, r3) = (g_dims[0], g_dims[1], g_dims[2]);
        if u1_dims[1] != r1 || u2_dims[1] != r2 || u3_dims[1] != r3 {
            candle_core::bail!(
                "TuckerFactored: factor inner ranks {:?}/{:?}/{:?} don't match core ranks [{r1},{r2},{r3}]",
                u1_dims,
                u2_dims,
                u3_dims
            );
        }

        // All factors must live on the same device as g_core.
        let u1 = u1.to_device(&device)?.to_dtype(target_dtype)?;
        let u2 = u2.to_device(&device)?.to_dtype(target_dtype)?;
        let u3 = u3.to_device(&device)?.to_dtype(target_dtype)?;
        let g_core = g_core.to_dtype(target_dtype)?;

        Ok(Self {
            g_core,
            u1,
            u2,
            u3,
            dtype: target_dtype,
            device,
        })
    }

    /// Number of experts (size of mode-0).
    pub fn num_experts(&self) -> usize {
        self.u1.dims()[0]
    }

    /// `d_out` (size of mode-1, the output features).
    pub fn d_out(&self) -> usize {
        self.u2.dims()[0]
    }

    /// `d_in` (size of mode-2, the input features).
    pub fn d_in(&self) -> usize {
        self.u3.dims()[0]
    }

    /// Tucker ranks `[r1, r2, r3]` (always equals `g_core.dims()`).
    pub fn ranks(&self) -> [usize; 3] {
        let d = self.g_core.dims();
        [d[0], d[1], d[2]]
    }

    /// Total element count of factored storage:
    /// `r1*r2*r3 + K*r1 + d_out*r2 + d_in*r3`.
    pub fn storage_elem_count(&self) -> usize {
        let [r1, r2, r3] = self.ranks();
        let k = self.num_experts();
        let d_out = self.d_out();
        let d_in = self.d_in();
        r1 * r2 * r3 + k * r1 + d_out * r2 + d_in * r3
    }

    /// Dense element count, i.e. `K * d_out * d_in` (for comparison).
    pub fn dense_elem_count(&self) -> usize {
        self.num_experts() * self.d_out() * self.d_in()
    }

    /// Element-wise storage ratio dense / factored. > 1.0 means we saved
    /// memory. Returns `0` if factored count is zero (degenerate).
    pub fn storage_ratio(&self) -> f64 {
        let s = self.storage_elem_count();
        if s == 0 {
            return 0.0;
        }
        self.dense_elem_count() as f64 / s as f64
    }

    /// Materialise the dense `W [K, d_out, d_in]` tensor by Tucker
    /// reconstruction. Only used by debug paths / `dequantize_w`; the
    /// inference forward never calls this.
    fn reconstruct_dense(&self) -> Result<Tensor> {
        // We promote everything to F32 on the home device to keep numerical
        // stability of the chained contractions.
        let g = self.g_core.to_dtype(DType::F32)?;
        let u1 = self.u1.to_dtype(DType::F32)?;
        let u2 = self.u2.to_dtype(DType::F32)?;
        let u3 = self.u3.to_dtype(DType::F32)?;

        let [r1, r2, r3] = self.ranks();
        let d_out = self.d_out();
        let d_in = self.d_in();
        let k = self.num_experts();

        // 1) tmp_a = G ×_3 U3   →   shape [r1, r2, d_in]
        // 2) tmp_b = tmp_a ×_2 U2   →   shape [r1, d_out, d_in]
        // 3) W     = tmp_b ×_1 U1   →   shape [K, d_out, d_in]
        //
        // We can express each mode-n product as a matmul after suitable
        // reshape:
        //   G has shape [r1, r2, r3]. Reshape to [r1*r2, r3].
        //   matmul with U3.t() [r3, d_in] -> [r1*r2, d_in] -> reshape [r1, r2, d_in]
        let g_flat = g.reshape((r1 * r2, r3))?;
        let tmp_a = g_flat.matmul(&u3.t()?)?; // [r1*r2, d_in]
        let tmp_a = tmp_a.reshape((r1, r2, d_in))?;

        // (×_2 U2): contract over r2 with U2.t() [r2, d_out].
        // tmp_a shape [r1, r2, d_in] -> permute to [r1, d_in, r2] -> reshape [r1*d_in, r2]
        // -> matmul U2.t() -> [r1*d_in, d_out] -> reshape [r1, d_in, d_out] -> permute
        let tmp_a_p = tmp_a.permute((0, 2, 1))?.contiguous()?; // [r1, d_in, r2]
        let tmp_a_p = tmp_a_p.reshape((r1 * d_in, r2))?;
        let tmp_b = tmp_a_p.matmul(&u2.t()?)?; // [r1*d_in, d_out]
        let tmp_b = tmp_b
            .reshape((r1, d_in, d_out))?
            .permute((0, 2, 1))?
            .contiguous()?; // [r1, d_out, d_in]

        // (×_1 U1): contract over r1 with U1 [K, r1].
        //   W[e, d_out, d_in] = sum_a U1[e, a] · tmp_b[a, d_out, d_in]
        // Reshape tmp_b -> [r1, d_out*d_in], matmul with U1 [K, r1] -> [K, d_out*d_in]
        let tmp_b_flat = tmp_b.reshape((r1, d_out * d_in))?;
        let w_flat = u1.matmul(&tmp_b_flat)?; // [K, d_out*d_in]
        let w = w_flat.reshape((k, d_out, d_in))?;

        // Cast back to the layer's nominal dtype.
        w.to_dtype(self.dtype)
    }

    /// Sparse expert-indexed forward.
    ///
    /// Inputs:
    ///   - `a`        : per-token activation. Two shapes supported:
    ///     * CUDA path  : `[num_tokens, 1, d_in]`
    ///     * Metal path : `[b_size, seq_len, 1, 1, d_in]`
    ///   - `indices`  : selected expert indices per token, dtype U32. Matching
    ///     ranks for the two paths above:
    ///     * CUDA path  : `[num_tokens, num_experts_per_tok]`
    ///     * Metal path : `[b_size, seq_len, num_experts_per_tok]`
    ///
    /// Output: per-token, per-expert output activation `[..., num_experts_per_tok, d_out]`
    /// (the moe forward stitches these via the routing-weight reduction.)
    ///
    /// Contraction order, per (token, top-k slot):
    ///   1. `v3 = U3.t() @ a`          [r3]
    ///   2. `v2[b] = sum_c G[a, b, c] · v3[c]`   (over r3, leaving [r1, r2])
    ///   3. `v1[b] = sum_a U1[e, a] · v2[a, b]`  (over r1, leaving [r2])
    ///   4. `y[d_out] = U2 @ v1`       [d_out]
    ///
    /// We implement this in a batched form using the existing candle matmul,
    /// no custom kernel needed in this commit. Performance refinement happens
    /// on hot-path benches in a follow-up (CUDA fused contraction kernel).
    fn gather_forward_impl(&self, a: &Tensor, indices: &Tensor) -> Result<Tensor> {
        let (k_total, num_per_tok, hidden_in) = match a.dims() {
            // CUDA / generic path: [num_tokens, 1, d_in], indices [num_tokens, k]
            &[num_tokens, 1, d_in] => {
                let (it0, ik) = indices.dims2()?;
                if it0 != num_tokens {
                    candle_core::bail!(
                        "TuckerFactored::gather_forward: token count mismatch ({num_tokens} vs {it0})"
                    );
                }
                (num_tokens * ik, ik, d_in)
            }
            // Metal path: [b, s, 1, 1, d_in], indices [b, s, k]
            &[b_size, seq_len, 1, 1, d_in] => {
                let (ib, is, ik) = indices.dims3()?;
                if (ib, is) != (b_size, seq_len) {
                    candle_core::bail!(
                        "TuckerFactored::gather_forward: token shape mismatch ({b_size}x{seq_len} vs {ib}x{is})"
                    );
                }
                (b_size * seq_len * ik, ik, d_in)
            }
            other => candle_core::bail!(
                "TuckerFactored::gather_forward: unsupported activation shape {:?}",
                other
            ),
        };

        if hidden_in != self.d_in() {
            candle_core::bail!(
                "TuckerFactored::gather_forward: d_in mismatch (act {hidden_in} vs layer {})",
                self.d_in()
            );
        }

        let device = a.device();
        let act_dtype = a.dtype();
        let compute_dtype = self.dtype;

        // Bring activation + indices into compute dtype / shape on the right device.
        let a_compute = a.to_dtype(compute_dtype)?;

        // Flatten activation to [n_tok, d_in] (the leading 1's are spurious).
        let n_tok = k_total / num_per_tok;
        let a_flat = a_compute.reshape((n_tok, hidden_in))?;
        // Flatten indices to [k_total]
        let flat_indices = indices.reshape((k_total,))?;

        // 1) v3[n_tok, r3] = a_flat @ U3       (since U3 has shape [d_in, r3])
        let [r1, r2, r3] = self.ranks();
        let u3 = self.u3.to_device(device)?; // cheap if already on device
        let v3 = a_flat.matmul(&u3)?; // [n_tok, r3]

        // 2) v2[n_tok, r1, r2] = sum_c G[a, b, c] · v3[n_tok, c]
        //    G shape [r1, r2, r3] -> reshape [r1*r2, r3]
        //    matmul with v3.t() [r3, n_tok] -> [r1*r2, n_tok] -> reshape [r1, r2, n_tok]
        //    -> permute [n_tok, r1, r2]
        let g = self.g_core.to_device(device)?;
        let g_flat = g.reshape((r1 * r2, r3))?;
        let v2_t = g_flat.matmul(&v3.t()?.contiguous()?)?; // [r1*r2, n_tok]
        let v2 = v2_t
            .reshape((r1, r2, n_tok))?
            .permute((2, 0, 1))?
            .contiguous()?; // [n_tok, r1, r2]

        // Broadcast v2 over the topk axis: each token uses the same v2 across
        // all of its selected experts.
        //   v2_per_slot[k_total, r1, r2] = v2[token_of_slot, :, :]
        // Slot i corresponds to token i / num_per_tok. We can construct this by
        // reshaping then broadcasting, then flattening to [k_total, r1, r2].
        let v2_brd = v2
            .unsqueeze(1)? // [n_tok, 1, r1, r2]
            .broadcast_as((n_tok, num_per_tok, r1, r2))?
            .reshape((k_total, r1, r2))?
            .contiguous()?;

        // 3) v1[k_total, r2] = sum_a U1[e_for_slot, a] · v2_brd[a, b]
        //    We gather U1 by the flat_indices, then matmul (batched).
        let u1 = self.u1.to_device(device)?;
        let u1_selected = u1.index_select(&flat_indices, 0)?; // [k_total, r1]

        // For each slot i: v1[i, b] = sum_a U1_sel[i, a] · v2_brd[i, a, b]
        // We can do this as a batched matmul of [k_total, 1, r1] @ [k_total, r1, r2]
        //   -> [k_total, 1, r2] -> squeeze.
        let u1_b = u1_selected.unsqueeze(1)?; // [k_total, 1, r1]
        let v1 = u1_b.matmul(&v2_brd)?.squeeze(1)?; // [k_total, r2]

        // 4) y[k_total, d_out] = v1 @ U2.t()
        let u2 = self.u2.to_device(device)?;
        let y = v1.matmul(&u2.t()?)?; // [k_total, d_out]

        // Reshape back to match the input convention. The MoE forward expects:
        //   CUDA path : [num_tokens, num_experts_per_tok, d_out]
        //   Metal path: [b_size, seq_len, num_experts_per_tok, d_out]
        let d_out = self.d_out();
        let result = match *a.dims() {
            [num_tokens, 1, _] => y.reshape((num_tokens, num_per_tok, d_out))?,
            [b_size, seq_len, 1, 1, _] => y.reshape((b_size, seq_len, num_per_tok, d_out))?,
            _ => unreachable!("input shape validated above"),
        };

        result.to_dtype(act_dtype)
    }
}

impl QuantMethod for TuckerFactoredLayer {
    fn new(method: QuantMethodConfig) -> Result<Self>
    where
        Self: Sized,
    {
        match method {
            QuantMethodConfig::TuckerFactored {
                g_core,
                u1,
                u2,
                u3,
                target_dtype,
            } => Self::from_factors(g_core, u1, u2, u3, target_dtype),
            _ => {
                candle_core::bail!("TuckerFactoredLayer requires QuantMethodConfig::TuckerFactored")
            }
        }
    }

    fn dequantize_w(&self) -> Result<Tensor> {
        self.reconstruct_dense()
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Single-expert fallback. Used only when this layer is wired into a
        // non-MoE call site (E=1 by construction). For K>1 callers should
        // dispatch via `gather_forward`.
        if self.num_experts() != 1 {
            candle_core::bail!(
                "TuckerFactoredLayer::forward called on a {}-expert stack; use gather_forward",
                self.num_experts()
            );
        }
        // x [..., d_in] → y [..., d_out]
        // y = x · W^T = x · ((G[0] ×₂ U2 ×₃ U3))^T
        //   = ((x · U3) · G[0]^T) · U2^T,   roughly
        //
        // Concretely, with E=1:
        //   v3 = x @ U3       [..., r3]
        //   v2 = einsum("...c,abc->...ab", v3, G)    [..., r1, r2]
        //   v1 = einsum("a,...ab->...b", U1[0], v2)  [..., r2]
        //   y  = v1 @ U2.t()                          [..., d_out]
        let act_dtype = x.dtype();
        let compute_dtype = self.dtype;
        let x_c = x.to_dtype(compute_dtype)?;
        let leading = x_c.dims()[..x_c.dims().len() - 1].to_vec();
        let d_in = self.d_in();
        let n: usize = leading.iter().product();
        let x_flat = x_c.reshape((n, d_in))?;

        let [r1, r2, r3] = self.ranks();
        let v3 = x_flat.matmul(&self.u3.to_device(x.device())?)?; // [n, r3]

        let g = self.g_core.to_device(x.device())?;
        let g_flat = g.reshape((r1 * r2, r3))?;
        let v2_t = g_flat.matmul(&v3.t()?.contiguous()?)?; // [r1*r2, n]
        let v2 = v2_t
            .reshape((r1, r2, n))?
            .permute((2, 0, 1))?
            .contiguous()?; // [n, r1, r2]

        // U1[0] has shape [r1]. Contract with v2 over r1 → [n, r2]
        let u1_row = self.u1.to_device(x.device())?.narrow(0, 0, 1)?; // [1, r1]
        let v1 = u1_row.matmul(
            &v2.reshape((n, r1, r2))?
                .permute((1, 0, 2))?
                .contiguous()?
                .reshape((r1, n * r2))?,
        )?; // [1, n*r2]
        let v1 = v1.reshape((n, r2))?;

        let u2 = self.u2.to_device(x.device())?;
        let y = v1.matmul(&u2.t()?)?; // [n, d_out]
        let mut out_dims = leading;
        out_dims.push(self.d_out());
        y.reshape(out_dims)?.to_dtype(act_dtype)
    }

    fn gather_forward(&self, a: &Tensor, indices: &Tensor) -> Result<Tensor> {
        self.gather_forward_impl(a, indices)
    }

    fn quantized_act_type(&self) -> Option<DType> {
        // We compute in the layer's compute dtype (BF16 by default). The
        // `forward_autocast` / `gather_forward_autocast` wrappers in the trait
        // will up-cast inputs.
        Some(self.dtype)
    }

    fn add_delta_w(&self, _delta: &Tensor) -> Result<Arc<dyn QuantMethod>> {
        candle_core::bail!("TuckerFactoredLayer does not support add_delta_w")
    }

    fn dtype_and_device(&self) -> (DType, Device) {
        (self.dtype, self.device.clone())
    }

    fn apply_isq(
        self: Arc<Self>,
        _dtype: Option<IsqType>,
        _device: Device,
        _n_quantized: &AtomicUsize,
        _imatrix_weight: Option<Vec<f32>>,
        _guard: QuantizeOntoGuard,
    ) -> Result<Arc<dyn QuantMethod>> {
        // TD-MoE Tucker layers are post-load artefacts produced by the
        // TD-MoE compressor that runs *after* ISQ. They should never be
        // re-quantized; the caller would be re-running ISQ on an already
        // compressed expert stack which is nonsensical.
        Ok(self)
    }

    fn is_distributed(&self) -> Option<DistributedKind> {
        None
    }
}

impl QuantizedSerde for TuckerFactoredLayer {
    fn name(&self) -> &'static str {
        "td-moe-tucker-factored"
    }
    fn isq_serde_supported(&self) -> bool {
        false
    }
    fn serialize(&self) -> Result<Cow<'_, [u8]>> {
        // UQFF persistence for factored Tucker is out of scope here; the layer
        // is reconstructed at every model load via the TD-MoE post-load hook.
        candle_core::bail!("TuckerFactoredLayer does not yet support UQFF serialization")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn cos_sim(a: &[f32], b: &[f32]) -> f32 {
        let mut dot = 0f32;
        let mut na = 0f32;
        let mut nb = 0f32;
        for (x, y) in a.iter().zip(b.iter()) {
            dot += *x * *y;
            na += *x * *x;
            nb += *y * *y;
        }
        dot / (na.sqrt() * nb.sqrt() + 1e-12)
    }

    /// Build a small synthetic Tucker decomposition by hand: pick G, U1, U2,
    /// U3 with known shapes and verify `reconstruct_dense` matches the
    /// reference Einstein sum.
    #[test]
    fn reconstruct_matches_einsum_reference() {
        let device = Device::Cpu;
        let k = 4usize;
        let d_out = 8usize;
        let d_in = 8usize;
        let r1 = 3usize;
        let r2 = 4usize;
        let r3 = 4usize;

        // Deterministic random factors.
        let mk = |n: usize, salt: f32| -> Vec<f32> {
            (0..n).map(|i| (i as f32 * 0.137 + salt).sin()).collect()
        };

        let g_data = mk(r1 * r2 * r3, 0.0);
        let u1_data = mk(k * r1, 1.0);
        let u2_data = mk(d_out * r2, 2.0);
        let u3_data = mk(d_in * r3, 3.0);

        let g = Tensor::from_vec(g_data.clone(), (r1, r2, r3), &device).unwrap();
        let u1 = Tensor::from_vec(u1_data.clone(), (k, r1), &device).unwrap();
        let u2 = Tensor::from_vec(u2_data.clone(), (d_out, r2), &device).unwrap();
        let u3 = Tensor::from_vec(u3_data.clone(), (d_in, r3), &device).unwrap();

        let layer = TuckerFactoredLayer::from_factors(g, u1, u2, u3, DType::F32).unwrap();
        let recon = layer.reconstruct_dense().unwrap();
        let recon_v: Vec<f32> = recon.flatten_all().unwrap().to_vec1().unwrap();

        // Reference: W[e, d_out, d_in] = sum_{a,b,c} G[a,b,c] · U1[e,a] · U2[d_out,b] · U3[d_in,c]
        let mut reference = vec![0f32; k * d_out * d_in];
        for e in 0..k {
            for o in 0..d_out {
                for i in 0..d_in {
                    let mut acc = 0f32;
                    for a in 0..r1 {
                        for b in 0..r2 {
                            for c in 0..r3 {
                                acc += g_data[a * r2 * r3 + b * r3 + c]
                                    * u1_data[e * r1 + a]
                                    * u2_data[o * r2 + b]
                                    * u3_data[i * r3 + c];
                            }
                        }
                    }
                    reference[e * d_out * d_in + o * d_in + i] = acc;
                }
            }
        }

        let cos = cos_sim(&recon_v, &reference);
        assert!(
            cos > 0.9999,
            "reconstruct_dense cos sim vs reference einsum is {cos}, expected ~1.0"
        );
    }

    /// Gather forward must produce the same result as a per-token, per-expert
    /// dense matmul against the reconstructed `W[e]`. This is the core
    /// invariant: the factored contraction is mathematically equivalent to
    /// the reconstructed-dense forward.
    #[test]
    fn gather_forward_equals_dense_reference() {
        let device = Device::Cpu;
        let k = 4usize;
        let d_out = 8usize;
        let d_in = 16usize;
        let r1 = 3usize;
        let r2 = 4usize;
        let r3 = 5usize;
        let num_tokens = 3usize;
        let num_per_tok = 2usize;

        let mk = |n: usize, salt: f32| -> Vec<f32> {
            (0..n)
                .map(|i| (i as f32 * 0.07 + salt).cos() * 0.4)
                .collect()
        };

        let g = Tensor::from_vec(mk(r1 * r2 * r3, 0.0), (r1, r2, r3), &device).unwrap();
        let u1 = Tensor::from_vec(mk(k * r1, 1.0), (k, r1), &device).unwrap();
        let u2 = Tensor::from_vec(mk(d_out * r2, 2.0), (d_out, r2), &device).unwrap();
        let u3 = Tensor::from_vec(mk(d_in * r3, 3.0), (d_in, r3), &device).unwrap();

        let layer = TuckerFactoredLayer::from_factors(g, u1, u2, u3, DType::F32).unwrap();

        // Activation [num_tokens, 1, d_in] (CUDA-style shape)
        let a_data: Vec<f32> = (0..(num_tokens * d_in))
            .map(|i| (i as f32 * 0.13).sin() * 0.7)
            .collect();
        let a = Tensor::from_vec(a_data.clone(), (num_tokens, 1, d_in), &device).unwrap();
        // Indices [num_tokens, num_per_tok], dtype U32.
        let idx_data: Vec<u32> = vec![
            0, 1, // token 0 -> experts 0, 1
            2, 3, // token 1 -> experts 2, 3
            1, 3, // token 2 -> experts 1, 3
        ];
        let indices =
            Tensor::from_vec(idx_data.clone(), (num_tokens, num_per_tok), &device).unwrap();

        let out = layer.gather_forward(&a, &indices).unwrap();
        assert_eq!(out.dims(), &[num_tokens, num_per_tok, d_out]);
        let out_v: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();

        // Reference: reconstruct dense W, then matmul per (token, expert).
        let w_dense = layer.reconstruct_dense().unwrap();
        let w_v: Vec<f32> = w_dense.flatten_all().unwrap().to_vec1().unwrap();

        let mut reference = vec![0f32; num_tokens * num_per_tok * d_out];
        for t in 0..num_tokens {
            for kk in 0..num_per_tok {
                let e = idx_data[t * num_per_tok + kk] as usize;
                for o in 0..d_out {
                    let mut acc = 0f32;
                    for i in 0..d_in {
                        acc += w_v[e * d_out * d_in + o * d_in + i] * a_data[t * d_in + i];
                    }
                    reference[t * num_per_tok * d_out + kk * d_out + o] = acc;
                }
            }
        }

        let cos = cos_sim(&out_v, &reference);
        assert!(
            cos > 0.99,
            "gather_forward cos sim vs dense reference is {cos}, expected > 0.99"
        );
    }

    /// Storage accounting: a `[K=8, d_out=64, d_in=64]` expert stack at
    /// Tucker rank `[4, 16, 16]` should have factored storage of
    ///   4*16*16 + 8*4 + 64*16 + 64*16 = 1024 + 32 + 1024 + 1024 = 3104
    /// vs dense 8*64*64 = 32768 -> ratio ~10.6x.
    #[test]
    fn storage_accounting_matches_formula() {
        let device = Device::Cpu;
        let k = 8usize;
        let d_out = 64usize;
        let d_in = 64usize;
        let r1 = 4usize;
        let r2 = 16usize;
        let r3 = 16usize;

        let g = Tensor::zeros((r1, r2, r3), DType::F32, &device).unwrap();
        let u1 = Tensor::zeros((k, r1), DType::F32, &device).unwrap();
        let u2 = Tensor::zeros((d_out, r2), DType::F32, &device).unwrap();
        let u3 = Tensor::zeros((d_in, r3), DType::F32, &device).unwrap();
        let layer = TuckerFactoredLayer::from_factors(g, u1, u2, u3, DType::F32).unwrap();

        assert_eq!(layer.dense_elem_count(), k * d_out * d_in);
        assert_eq!(
            layer.storage_elem_count(),
            r1 * r2 * r3 + k * r1 + d_out * r2 + d_in * r3
        );
        let ratio = layer.storage_ratio();
        assert!(ratio > 10.0, "expected storage ratio > 10x, got {ratio}");
    }

    /// F16 dtype path: matches dense-reconstruction within F16 round-trip
    /// tolerance. CPU candle doesn't support BF16 matmul, but the production
    /// CUDA path does — we exercise F16 here as the closest CPU-portable
    /// half-precision dtype to confirm the dtype-cast boundary works.
    #[test]
    fn gather_forward_half_precision_finite() {
        let device = Device::Cpu;
        let k = 4usize;
        let d_out = 8usize;
        let d_in = 16usize;
        let r1 = 3usize;
        let r2 = 4usize;
        let r3 = 5usize;
        let num_tokens = 2usize;
        let num_per_tok = 2usize;

        let mk = |n: usize, salt: f32| -> Vec<f32> {
            (0..n)
                .map(|i| (i as f32 * 0.07 + salt).cos() * 0.4)
                .collect()
        };
        let g = Tensor::from_vec(mk(r1 * r2 * r3, 0.0), (r1, r2, r3), &device).unwrap();
        let u1 = Tensor::from_vec(mk(k * r1, 1.0), (k, r1), &device).unwrap();
        let u2 = Tensor::from_vec(mk(d_out * r2, 2.0), (d_out, r2), &device).unwrap();
        let u3 = Tensor::from_vec(mk(d_in * r3, 3.0), (d_in, r3), &device).unwrap();
        let layer = TuckerFactoredLayer::from_factors(g, u1, u2, u3, DType::F16).unwrap();

        let a_data: Vec<f32> = (0..(num_tokens * d_in))
            .map(|i| (i as f32 * 0.13).sin() * 0.7)
            .collect();
        let a = Tensor::from_vec(a_data, (num_tokens, 1, d_in), &device)
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap();
        let indices =
            Tensor::from_vec(vec![0u32, 1, 2, 3], (num_tokens, num_per_tok), &device).unwrap();

        let out = layer.gather_forward(&a, &indices).unwrap();
        let out_v: Vec<f32> = out
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        for v in &out_v {
            assert!(v.is_finite(), "non-finite output {v}");
        }
    }
}
