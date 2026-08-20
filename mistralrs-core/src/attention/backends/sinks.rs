use candle_core::{Result, Tensor};

use crate::{attention::SdpaParams, pipeline::text_models_inputs_processor::FlashParams};

/// Fused attention with per-head sinks.
///
/// Dispatches to:
///   CUDA  -> flash_attn_sinks / flash_attn_sinks_varlen
///   Metal -> flash_attn_sinks_metal / flash_attn_sinks_varlen_metal
///   CPU   -> unfused matmul + softmax_with_sinks
///
/// Varlen is used when flash_params contains cu_seqlens_k for this device AND
/// q has batch > 1 AND the caller supplied no explicit mask.
///
/// The varlen backends derive causality and windowing *internally* from
/// `cu_seqlens` + `window_size` (bottom-right aligned; see
/// `flash_attn_sinks.cu`), and `sinks_attn_varlen` therefore takes no mask
/// argument at all. Routing a caller that DID supply an additive mask through
/// it silently discards that mask — padding columns in a ragged batch, a
/// model's own block-causality, a fixed-width graph-decode length mask. That
/// is a correctness bug, not a fast path, so an explicit mask pins the call to
/// the regular path, which honors it.
#[allow(unused_variables, clippy::too_many_arguments)]
pub(crate) fn sinks_attn(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    sinks: &Tensor,
    mask: Option<&Tensor>,
    flash_params: Option<&FlashParams>,
    sdpa_params: &SdpaParams,
) -> Result<Tensor> {
    let (b_sz, _n_heads, _q_len, _head_dim) = q.dims4()?;
    let window_size = sdpa_params.sliding_window.unwrap_or(0);

    // Detect varlen: no caller mask to honor AND flash_params has cu_seqlens_k
    // AND batch > 1.
    let is_varlen = mask.is_none()
        && b_sz > 1
        && flash_params
            .is_some_and(|fp| fp.cumulative_seqlens_k.contains_key(&q.device().location()));

    if is_varlen {
        return sinks_attn_varlen(
            q,
            k,
            v,
            sinks,
            flash_params.unwrap(),
            sdpa_params,
            window_size,
        );
    }

    // Non-varlen path
    sinks_attn_regular(q, k, v, sinks, mask, sdpa_params, window_size)
}

/// Is the fused head_dim=512 sinks path enabled? `ARC_FLASH_512=0|false|off|no`
/// falls back to the unfused matmul + softmax_with_sinks path V4 ran before.
///
/// This exists so a fused-vs-unfused comparison is **one binary, one variable**.
/// Comparing two binaries is how a pin A/B ended up measuring a different
/// allocation population than the hypothesis was about — the arms differed in
/// more ways than the one under test.
///
/// Named once per process at info, so an A/B can ASSERT it actually got two
/// different behaviours rather than two identical arms reporting no difference.
/// A flag whose name is wrong is indistinguishable from a flag that costs
/// nothing.
fn flash_512_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = !matches!(
            std::env::var("ARC_FLASH_512").as_deref(),
            Ok("0") | Ok("false") | Ok("off") | Ok("no")
        );
        tracing::info!(
            target: "arcflash",
            "fused head_dim=512 sinks path is {} (ARC_FLASH_512)",
            if on { "FUSED" } else { "UNFUSED" }
        );
        on
    })
}

/// Non-varlen sinks attention: Q [B, H, q_len, D], K/V [B, kv_H, kv_len, D]
#[allow(unused_variables)]
fn sinks_attn_regular(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    sinks: &Tensor,
    mask: Option<&Tensor>,
    sdpa_params: &SdpaParams,
    window_size: usize,
) -> Result<Tensor> {
    // The CUDA flash-sinks kernel now also instantiates head_dim=512 (V4:
    // symmetric 512/512, MQA). METAL DOES NOT — its kernel still stops at 256
    // (`metal/backend/paged_attention.rs:121`), so the Metal branch below is
    // guarded separately and a 512 head on Metal still takes the unfused
    // matmul + softmax_with_sinks path (GPU-capable via
    // SoftmaxWithSinks::cuda_fwd, same math as the old "cpu" fallback).
    let hd = q.dim(candle_core::D::Minus1)?;
    // ⚠️ TWO BACKENDS, TWO ENVELOPES — this was ONE shared flag, and widening it
    // for CUDA silently widened it for Metal too. Metal's sinks kernel stops at
    // 256: `metal_kernels/mod.rs:3025` errors on any other head_dim, and
    // `sdpa_with_sinks.metal` instantiates only {64,80,96,128,256}. Routing a
    // 512 head there turns a working unfused fallback into a hard
    // CompilationError. Each backend now advertises its own set.
    let flash_sinks_ok_cuda =
        matches!(hd, 64 | 80 | 96 | 112 | 128 | 192 | 256) || (hd == 512 && flash_512_enabled());
    let flash_sinks_ok_metal = matches!(hd, 64 | 80 | 96 | 112 | 128 | 192 | 256);
    #[cfg(not(feature = "cuda"))]
    let _ = flash_sinks_ok_cuda;
    #[cfg(not(feature = "metal"))]
    let _ = flash_sinks_ok_metal;

    // Name the backend, once per process.
    //
    // WHY THIS LINE EXISTS. Expert parallelism was priced on the belief that
    // building with NCCL silently disables flash attention for V4, via
    // `attention/mod.rs`'s `use_nccl() => naive_sdpa` gate — which would make
    // any EP=2-vs-EP=1 comparison a comparison of two different attention
    // kernels, and the measurement worthless. Reading the dispatch says
    // otherwise for V4: `sdpa_params.sinks` is `Some` (V4 loads `attn_sink`),
    // so `Sdpa::run_attention` diverts here on its FIRST line and the
    // `use_nccl()` gate — which lives in `run_attention_noflash` — is never
    // reached. Which of the two arms below a 512 head takes is itself a
    // per-backend question (`flash_sinks_ok_cuda` admits 512 only under
    // `flash_512_enabled()`; Metal's kernel stops at 256), which is one more
    // reason to log the answer rather than argue it.
    //
    // That is a code read. Before anyone spends $9.22/hr on a 2xH100 pair to
    // measure EP, this turns it into an observation: the log says which path
    // the model on the box actually took. Emitted once, off the hot path.
    {
        static NAMED: std::sync::OnceLock<()> = std::sync::OnceLock::new();
        NAMED.get_or_init(|| {
            let fused = (q.device().is_cuda() && flash_sinks_ok_cuda)
                || (q.device().is_metal() && flash_sinks_ok_metal);
            tracing::info!(
                target: "arc_attention_backend",
                "ARC_ATTN_BACKEND: sinks_attn head_dim={hd} \
                 flash_sinks_ok_cuda={flash_sinks_ok_cuda} \
                 flash_sinks_ok_metal={flash_sinks_ok_metal} \
                 device={:?} => {} (the `use_nccl()` gate in \
                 `run_attention_noflash` is NOT on this path)",
                q.device().location(),
                if fused {
                    "fused flash-sinks kernel"
                } else {
                    "ArcFlash/Tile — unfused matmul + softmax_with_sinks"
                },
            );
        });
    }

    // 🔴 THE FUSED KERNEL CANNOT HONOUR AN EXPLICIT MASK, SO IT MUST NOT TAKE
    // WORK THAT HAS ONE.
    //
    // This arm used to engage on head_dim alone and call the kernel without
    // `mask`, so V4's dense `[t_q, n_keys]` mask over
    // `[raw sliding-window KV ++ compressed KV]` (built in
    // `models/dsv4_attention.rs`) was dropped on the floor and the kernel ran a
    // plain causal scan over an axis where relative distance is meaningless --
    // compressed entry `j` stands for absolute position `j * ratio`.
    //
    // ✅ MEASURED, H200, head_dim=512, output magnitude 0.1875:
    //     max|fused - reference(causal only)| = 0.0129
    //     max|fused - reference(masked)|      = 0.7749
    //     max|ref(causal) - ref(masked)|      = 0.7720   <- the mask's whole effect
    // The fused result sat 60.3x closer to the UNMASKED reference, and its error
    // against the masked one was the same size as the mask's entire effect. The
    // mask was discarded, not approximated. End to end this served
    // `'orem, etc. etc. etc.'` where the unfused path served coherent text.
    //
    // `dsv4_attention.rs` predicted this exactly: "a landmine that arms itself
    // the moment a 512-wide flash-sinks kernel lands, and silently, because the
    // fused path takes no mask to disagree with." The kernel landed; the mask
    // is now a parameter of `flash_attn_sinks` so it cannot be forgotten again,
    // and this gate routes masked work to the path that applies it.
    #[cfg(feature = "cuda")]
    if q.device().is_cuda() && flash_sinks_ok_cuda && mask.is_none() {
        crate::attention::arcflash::note(crate::attention::arcflash::Path::VendorSinks);
        return mistralrs_paged_attn::flash_attn_sinks(
            q,
            k,
            v,
            Some(sinks),
            sdpa_params.softmax_scale,
            window_size,
            None,
        );
    }
    #[cfg(feature = "cuda")]
    if q.device().is_cuda() && flash_sinks_ok_cuda && mask.is_some() {
        // Loud once: this is a real performance loss (the fused kernel is 60%
        // of prefill) taken deliberately to keep the answer correct.
        static WARNED: std::sync::OnceLock<()> = std::sync::OnceLock::new();
        WARNED.get_or_init(|| {
            tracing::warn!(
                target: "arcflash",
                "fused head_dim={hd} sinks kernel DECLINED: an explicit attention mask is \
                 present and the kernel cannot honour one. Falling back to the unfused \
                 masked path. This is correct-and-slower on purpose; giving the kernel a \
                 custom-mask mode is what makes it fast again."
            );
        });
    }

    #[cfg(feature = "metal")]
    if q.device().is_metal() && flash_sinks_ok_metal {
        crate::attention::arcflash::note(crate::attention::arcflash::Path::VendorSinks);
        return mistralrs_quant::flash_attn_sinks_metal(
            q,
            k,
            v,
            Some(sinks),
            sdpa_params.softmax_scale,
            window_size,
        );
    }

    // ArcFlash/Tile — the head_dim the vendor sinks kernel does not take.
    // DeepSeek-V4's 512 is the whole reason this arm exists.
    sinks_attn_cpu(q, k, v, sinks, mask, sdpa_params)
}

/// Varlen sinks attention: Q [B, H, max_q, D], K/V packed [total_kv, kv_H, D]
/// or K/V [1, kv_H, total_kv, D] (squeezed+transposed automatically).
#[allow(unused_variables)]
fn sinks_attn_varlen(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    sinks: &Tensor,
    flash_params: &FlashParams,
    sdpa_params: &SdpaParams,
    window_size: usize,
) -> Result<Tensor> {
    let device = q.device();

    // Handle K/V shape: 4D [1, kv_H, total, D] -> 3D [total, kv_H, D], or 3D as-is
    let (k_packed, v_packed) = if k.dims().len() == 4 {
        (
            k.squeeze(0)?.transpose(0, 1)?,
            v.squeeze(0)?.transpose(0, 1)?,
        )
    } else {
        (k.clone(), v.clone())
    };

    // Get cu_seqlens from flash_params (already on GPU as U32, no conversion needed)
    let cu_seqlens_q = &flash_params.cumulative_seqlens_q[&device.location()];
    let cu_seqlens_k = &flash_params.cumulative_seqlens_k[&device.location()];

    // R1: head_dim guard (see sinks_attn_regular).
    let hd = q.dim(candle_core::D::Minus1)?;
    // ⚠️ TWO BACKENDS, TWO ENVELOPES — this was ONE shared flag, and widening it
    // for CUDA silently widened it for Metal too. Metal's sinks kernel stops at
    // 256: `metal_kernels/mod.rs:3025` errors on any other head_dim, and
    // `sdpa_with_sinks.metal` instantiates only {64,80,96,128,256}. Routing a
    // 512 head there turns a working unfused fallback into a hard
    // CompilationError. Each backend now advertises its own set.
    let flash_sinks_ok_cuda =
        matches!(hd, 64 | 80 | 96 | 112 | 128 | 192 | 256) || (hd == 512 && flash_512_enabled());
    let flash_sinks_ok_metal = matches!(hd, 64 | 80 | 96 | 112 | 128 | 192 | 256);
    #[cfg(not(feature = "cuda"))]
    let _ = flash_sinks_ok_cuda;
    #[cfg(not(feature = "metal"))]
    let _ = flash_sinks_ok_metal;

    #[cfg(feature = "cuda")]
    if device.is_cuda() && flash_sinks_ok_cuda {
        return mistralrs_paged_attn::flash_attn_sinks_varlen(
            q,
            &k_packed,
            &v_packed,
            Some(sinks),
            cu_seqlens_q,
            cu_seqlens_k,
            sdpa_params.softmax_scale,
            window_size,
        );
    }

    #[cfg(feature = "metal")]
    if device.is_metal() && flash_sinks_ok_metal {
        return mistralrs_quant::flash_attn_sinks_varlen_metal(
            q,
            &k_packed,
            &v_packed,
            Some(sinks),
            cu_seqlens_q,
            cu_seqlens_k,
            sdpa_params.softmax_scale,
            window_size,
        );
    }

    // CPU fallback: per-sequence loop (to_vec1 is fine on CPU path)
    let cu_q_vec: Vec<u32> = cu_seqlens_q.to_vec1()?;
    let cu_k_vec: Vec<u32> = cu_seqlens_k.to_vec1()?;
    sinks_attn_cpu_varlen(
        q,
        &k_packed,
        &v_packed,
        sinks,
        sdpa_params,
        &cu_q_vec,
        &cu_k_vec,
        window_size,
    )
}

/// Additive `[1, 1, q_len, kv_len]` mask reproducing the varlen kernels' own
/// masking rule, so the CPU fallback computes the same thing the CUDA/Metal
/// varlen kernels do instead of an unmasked (non-causal) attention.
///
/// Rule, transcribed from `mistralrs-paged-attn/src/cuda/flash_attn_sinks.cu`
/// (`flash_attn_sinks_varlen_kernel`): with `kv_offset = kv_len - q_len`
/// (bottom-right alignment, so the last query row sees the whole cache),
///   * causal — `kv_pos < my_kv_end   = q_row + kv_offset + 1`
///   * window — `kv_pos >= my_kv_start = max(0, q_row + kv_offset - window + 1)`
///     when `window_size > 0`, else `0`.
fn varlen_causal_mask(
    q_len: usize,
    kv_len: usize,
    window_size: usize,
    dtype: candle_core::DType,
    device: &candle_core::Device,
) -> Result<Tensor> {
    let kv_offset = kv_len as i64 - q_len as i64;
    let mut data = vec![0f32; q_len * kv_len];
    for (r, row) in data.chunks_mut(kv_len).enumerate() {
        let end = r as i64 + kv_offset + 1;
        let start = if window_size > 0 {
            (r as i64 + kv_offset - window_size as i64 + 1).max(0)
        } else {
            0
        };
        for (j, cell) in row.iter_mut().enumerate() {
            let j = j as i64;
            if j >= end || j < start {
                *cell = f32::NEG_INFINITY;
            }
        }
    }
    Tensor::from_vec(data, (1, 1, q_len, kv_len), device)?.to_dtype(dtype)
}

/// Sinks attention on **ArcFlash/Tile** — the path every DeepSeek-V4 layer
/// takes, because head_dim 512 is outside the fused sinks kernel's
/// `{64,80,96,112,128,192,256}`.
///
/// This used to `repeat_kv` K and V to all `n_kv_groups` heads and materialize
/// the full `[B, H, Tq, Tk]` score matrix twice. At V4's shape (MQA,
/// `n_kv_groups = 64`, head_dim 512) that is ~856 MB for one layer at
/// `Tk ≈ 1400` — which is why long prompts died rather than merely slowed.
/// `union_attention` folds the expansion into a reshape and bounds the scores
/// to one query tile; the numbers are unchanged (see `arcflash.rs`, and
/// `fold_matches_repeat_kv_across_gqa_ratios` which pins it against exactly the
/// implementation this replaced).
///
/// `sinks` arrives shaped `[1, n_heads, 1, 1]` from V4 (the layout the fused
/// kernel wants); `union_attention` flattens and casts it to the logits dtype.
fn sinks_attn_cpu(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    sinks: &Tensor,
    mask: Option<&Tensor>,
    sdpa_params: &SdpaParams,
) -> Result<Tensor> {
    crate::attention::arcflash::union_attention(q, k, v, mask, Some(sinks), sdpa_params)
}

/// CPU fallback for varlen: per-sequence unfused loop.
///
/// Each sequence is masked with `varlen_causal_mask`, matching the fused
/// varlen kernels. Passing `None` here (the pre-fix behavior) made every
/// `b_sz > 1` prefill on this path fully bidirectional — tokens attended to
/// their own future — so batched prefill was silently non-causal.
#[allow(clippy::too_many_arguments)]
fn sinks_attn_cpu_varlen(
    q: &Tensor,
    k_packed: &Tensor,
    v_packed: &Tensor,
    sinks: &Tensor,
    sdpa_params: &SdpaParams,
    cu_q: &[u32],
    cu_k: &[u32],
    window_size: usize,
) -> Result<Tensor> {
    let (b_sz, num_heads, max_q, head_dim) = q.dims4()?;
    let device = q.device();
    let mut outputs = Vec::with_capacity(b_sz);

    for i in 0..b_sz {
        let q_len = (cu_q[i + 1] - cu_q[i]) as usize;
        let kv_start = cu_k[i] as usize;
        let kv_len = (cu_k[i + 1] - cu_k[i]) as usize;

        // Extract this sequence's Q [1, H, q_len, D]
        let qi = q.narrow(0, i, 1)?;
        let qi = qi.narrow(2, 0, q_len)?;

        // Extract this sequence's K/V from packed [total_kv, kv_H, D]
        let ki = k_packed
            .narrow(0, kv_start, kv_len)?
            .transpose(0, 1)?
            .unsqueeze(0)?;
        let vi = v_packed
            .narrow(0, kv_start, kv_len)?
            .transpose(0, 1)?
            .unsqueeze(0)?;

        let mask = varlen_causal_mask(q_len, kv_len, window_size, qi.dtype(), device)?;
        let oi = sinks_attn_cpu(&qi, &ki, &vi, sinks, Some(&mask), sdpa_params)?;

        // Pad back to max_q
        if q_len < max_q {
            let pad = Tensor::zeros((1, num_heads, max_q - q_len, head_dim), qi.dtype(), device)?;
            outputs.push(Tensor::cat(&[&oi, &pad], 2)?);
        } else {
            outputs.push(oi);
        }
    }

    Tensor::cat(&outputs, 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use std::collections::HashMap;

    fn mk(dims: &[usize], seed: f32, dev: &Device) -> Result<Tensor> {
        let n: usize = dims.iter().product();
        let data: Vec<f32> = (0..n).map(|i| ((i as f32) * seed).sin()).collect();
        Tensor::from_vec(data, dims, dev)
    }

    fn params(head_dim: usize, n_kv_groups: usize, window: Option<usize>) -> SdpaParams {
        SdpaParams {
            n_kv_groups,
            softcap: None,
            softmax_scale: 1.0 / (head_dim as f32).sqrt(),
            sliding_window: window,
            sinks: None,
        }
    }

    fn varlen_params(seqlens: &[u32], dev: &Device) -> FlashParams {
        let mut acc = 0u32;
        let mut cu = vec![0u32];
        for l in seqlens {
            acc += l;
            cu.push(acc);
        }
        let cu = Tensor::from_vec(cu, seqlens.len() + 1, dev).unwrap();
        let mut q_map = HashMap::new();
        let mut k_map = HashMap::new();
        q_map.insert(dev.location(), cu.clone());
        k_map.insert(dev.location(), cu);
        FlashParams {
            max_q: seqlens.iter().copied().max().unwrap_or(0),
            max_k: seqlens.iter().copied().max().unwrap_or(0),
            cumulative_seqlens_q: q_map,
            cumulative_seqlens_k: k_map,
            causal: true,
        }
    }

    fn flat(t: &Tensor) -> Vec<f32> {
        t.to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap()
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max)
    }

    /// The CPU varlen fallback must apply the same causal (+ sliding-window)
    /// masking the fused varlen kernels apply internally. It used to pass
    /// `mask = None`, i.e. every token in a `b_sz > 1` prefill attended to its
    /// own future.
    #[test]
    fn cpu_varlen_fallback_is_causal() -> Result<()> {
        let dev = Device::Cpu;
        let (h, d, seq) = (2usize, 16usize, 6usize);
        let b = 2usize;
        let sinks = Tensor::from_vec(vec![0.3f32, 0.7], (1, h, 1, 1), &dev)?;

        let q = mk(&[b, h, seq, d], 0.07, &dev)?;
        // Packed K/V, the layout the varlen path documents: [total_kv, kv_H, D].
        let k_packed = mk(&[b * seq, h, d], 0.11, &dev)?;
        let v_packed = mk(&[b * seq, h, d], 0.17, &dev)?;

        for window in [0usize, 3] {
            let sdpa = SdpaParams {
                sinks: Some(sinks.clone()),
                ..params(d, 1, if window == 0 { None } else { Some(window) })
            };
            let flash = varlen_params(&[seq as u32; 2], &dev);
            let got = sinks_attn(&q, &k_packed, &v_packed, &sinks, None, Some(&flash), &sdpa)?;

            // Reference: per sequence, an explicit causal (+ window) mask.
            for i in 0..b {
                let qi = q.narrow(0, i, 1)?;
                let ki = k_packed
                    .narrow(0, i * seq, seq)?
                    .transpose(0, 1)?
                    .unsqueeze(0)?;
                let vi = v_packed
                    .narrow(0, i * seq, seq)?
                    .transpose(0, 1)?
                    .unsqueeze(0)?;
                let mask = varlen_causal_mask(seq, seq, window, qi.dtype(), &dev)?;
                let want = sinks_attn_cpu(&qi, &ki, &vi, &sinks, Some(&mask), &sdpa)?;
                let diff = max_abs_diff(&flat(&got.narrow(0, i, 1)?), &flat(&want));
                assert_eq!(
                    diff, 0.0,
                    "window={window} seq {i}: varlen fallback disagrees with the masked \
                     per-sequence reference (max abs diff {diff})"
                );

                // Teeth: the unmasked (pre-fix) computation is a different
                // answer, by a wide margin.
                let unmasked = sinks_attn_cpu(&qi, &ki, &vi, &sinks, None, &sdpa)?;
                let signal = max_abs_diff(&flat(&want), &flat(&unmasked));
                assert!(
                    signal > 1e-2,
                    "window={window} seq {i}: masked and unmasked attention are \
                     indistinguishable (diff {signal}); this test has no teeth"
                );
            }
        }
        Ok(())
    }

    /// A caller-supplied mask must pin the call to the regular path. The
    /// varlen backend has no mask parameter, so routing there would drop it
    /// (and, for the `[B, kv_H, T, D]` K/V every in-tree caller actually
    /// passes, `squeeze(0)` is a no-op at `B > 1` and the narrow then fails
    /// outright).
    #[test]
    fn explicit_mask_keeps_the_call_off_the_varlen_path() -> Result<()> {
        let dev = Device::Cpu;
        let (b, h, seq, d) = (2usize, 2usize, 5usize, 16usize);
        let sinks = Tensor::from_vec(vec![0.25f32, 0.5], (1, h, 1, 1), &dev)?;
        let sdpa = SdpaParams {
            sinks: Some(sinks.clone()),
            ..params(d, h, None)
        };

        let q = mk(&[b, h, seq, d], 0.05, &dev)?;
        let k = mk(&[b, 1, seq, d], 0.13, &dev)?;
        let v = mk(&[b, 1, seq, d], 0.23, &dev)?;
        let mask = varlen_causal_mask(seq, seq, 0, q.dtype(), &dev)?;
        let flash = varlen_params(&[seq as u32; 2], &dev);

        let got = sinks_attn(&q, &k, &v, &sinks, Some(&mask), Some(&flash), &sdpa)?;
        let want = sinks_attn_cpu(&q, &k, &v, &sinks, Some(&mask), &sdpa)?;
        assert_eq!(
            max_abs_diff(&flat(&got), &flat(&want)),
            0.0,
            "a masked call did not take the regular (mask-honoring) path"
        );
        Ok(())
    }
}

/// The fused `head_dim=512` flash-sinks path takes no mask argument, so it
/// cannot honour one. These tests turn that code-read into a measurement.
///
///   cargo test -p mistralrs-core --release --features "cuda flash-attn" fused_sinks -- --nocapture
#[cfg(all(test, feature = "cuda"))]
mod fused_sinks_mask_tests {
    use super::*;
    use candle_core::{DType, Device};

    fn mk(dims: &[usize], seed: f32, dev: &Device) -> Result<Tensor> {
        let n: usize = dims.iter().product();
        let data: Vec<f32> = (0..n).map(|i| ((i as f32) * seed).sin()).collect();
        Tensor::from_vec(data, dims, dev)?.to_dtype(DType::BF16)
    }

    fn to_f32(t: &Tensor) -> Vec<f32> {
        t.to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
    }

    fn max_abs(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max)
    }

    /// Additive 0 / -inf mask [1, 1, q, k]: causal, and additionally only the
    /// first `keep` keys are admissible. Two `keep` values give two genuinely
    /// different admissible sets.
    fn band_mask(q_len: usize, kv_len: usize, keep: usize, dev: &Device) -> Result<Tensor> {
        let mut m = vec![0f32; q_len * kv_len];
        for i in 0..q_len {
            for j in 0..kv_len {
                let causal_ok = j <= i + (kv_len - q_len);
                if !causal_ok || j >= keep {
                    m[i * kv_len + j] = f32::NEG_INFINITY;
                }
            }
        }
        Tensor::from_vec(m, (1, 1, q_len, kv_len), dev)?.to_dtype(DType::BF16)
    }

    const B: usize = 1;
    const H: usize = 8;
    const KVH: usize = 1; // V4 is MQA
    const QL: usize = 16;
    const KL: usize = 48;
    const D: usize = 512; // the head dim every V4 layer uses

    /// 🔴 THE DEFECT. `sinks_attn_regular`'s CUDA arm calls
    /// `flash_attn_sinks(q, k, v, Some(sinks), scale, window_size)` — the mask
    /// is not an argument, so the fused output cannot depend on it.
    ///
    /// Compared by MAGNITUDE, not bit-equality: two implementations of the same
    /// math never agree bit-for-bit (summation order moves the low bits), so
    /// bit-inequality would prove nothing. A dropped mask is an O(1) error — it
    /// mixes in values that should have carried zero weight.
    #[test]
    fn fused_sinks_512_ignores_the_mask() -> Result<()> {
        // Environment failure is not a pass.
        let dev = Device::new_cuda(0).expect("this test requires a CUDA device");

        let q = mk(&[B, H, QL, D], 0.017, &dev)?;
        let k = mk(&[B, KVH, KL, D], 0.023, &dev)?;
        let v = mk(&[B, KVH, KL, D], 0.031, &dev)?;
        let sinks = mk(&[H], 0.041, &dev)?;
        let p = SdpaParams {
            n_kv_groups: H / KVH,
            softcap: None,
            softmax_scale: 1.0 / (D as f32).sqrt(),
            sliding_window: None,
            sinks: None,
        };

        let mask_causal = band_mask(QL, KL, KL, &dev)?; // causal only
        let mask_narrow = band_mask(QL, KL, 8, &dev)?; // causal AND only 8 keys

        let ref_causal = to_f32(&crate::attention::arcflash::union_attention(
            &q,
            &k,
            &v,
            Some(&mask_causal),
            Some(&sinks),
            &p,
        )?);
        let ref_narrow = to_f32(&crate::attention::arcflash::union_attention(
            &q,
            &k,
            &v,
            Some(&mask_narrow),
            Some(&sinks),
            &p,
        )?);

        // The fused kernel, called exactly as the dispatch calls it.
        let fused = to_f32(&mistralrs_paged_attn::flash_attn_sinks(
            &q,
            &k,
            &v,
            Some(&sinks),
            p.softmax_scale,
            0,
            None, // deliberately probing what the kernel does with no mask
        )?);

        // NEGATIVE CONTROL: perturb the OUTPUT by one ULP at a known index.
        // Perturbing a shared INPUT would move both paths together and could
        // never detect a difference.
        {
            let mut probe = fused.clone();
            assert_eq!(max_abs(&probe, &fused), 0.0, "self-comparison must be zero");
            probe[7] = f32::from_bits(probe[7].to_bits().wrapping_add(1));
            assert!(
                max_abs(&probe, &fused) > 0.0,
                "comparator is inert: a 1-ULP OUTPUT perturbation did not register"
            );
        }

        let d_causal = max_abs(&fused, &ref_causal);
        let d_narrow = max_abs(&fused, &ref_narrow);
        let d_masks = max_abs(&ref_causal, &ref_narrow);
        let scale = ref_causal.iter().fold(0f32, |m, x| m.max(x.abs()));

        eprintln!(
            "output magnitude ~{scale:.4}\n  \
             max|fused - reference(causal only)| = {d_causal:.6}\n  \
             max|fused - reference(masked)|      = {d_narrow:.6}\n  \
             max|ref(causal) - ref(masked)|      = {d_masks:.6}   <- the mask's own effect"
        );

        assert!(
            d_masks > 1e-3,
            "the two masks did not change the reference ({d_masks}); test would be vacuous"
        );

        eprintln!(
            "VERDICT: fused sits {:.1}x closer to the CAUSAL-ONLY reference than to the MASKED one",
            d_narrow / d_causal.max(1e-9)
        );

        // This documents what the KERNEL does with no mask -- it is the
        // baseline the fix is measured against, not a failure.
        assert!(
            d_narrow > d_causal * 10.0,
            "expected the maskless kernel to track the causal-only reference"
        );
        Ok(())
    }

    /// THE FIX, PART 1 — the interface refuses.
    ///
    /// A kernel that cannot honour a mask must not silently accept work that
    /// has one. `flash_attn_sinks` now takes `mask` precisely so it can reject
    /// it, rather than relying on every caller to remember.
    #[test]
    fn fused_kernel_refuses_a_mask_instead_of_dropping_it() -> Result<()> {
        let dev = Device::new_cuda(0).expect("this test requires a CUDA device");
        let q = mk(&[B, H, QL, D], 0.017, &dev)?;
        let k = mk(&[B, KVH, KL, D], 0.023, &dev)?;
        let v = mk(&[B, KVH, KL, D], 0.031, &dev)?;
        let sinks = mk(&[H], 0.041, &dev)?;
        let mask = band_mask(QL, KL, 8, &dev)?;
        let scale = 1.0 / (D as f32).sqrt();

        let err =
            mistralrs_paged_attn::flash_attn_sinks(&q, &k, &v, Some(&sinks), scale, 0, Some(&mask));
        assert!(
            err.is_err(),
            "the kernel accepted a mask it cannot honour -- the silent-drop defect is back"
        );

        // ...and still works when there is genuinely no mask to honour.
        assert!(
            mistralrs_paged_attn::flash_attn_sinks(&q, &k, &v, Some(&sinks), scale, 0, None)
                .is_ok(),
            "refusing a mask must not break the maskless fast path"
        );
        Ok(())
    }

    /// THE FIX, PART 2 — the dispatch routes masked work to a path that
    /// applies the mask, and the answer is right.
    ///
    /// This is the regression gate: it fails if anyone re-widens the fused arm
    /// to swallow masked work.
    #[test]
    fn masked_sinks_attention_matches_the_masked_reference() -> Result<()> {
        let dev = Device::new_cuda(0).expect("this test requires a CUDA device");
        let q = mk(&[B, H, QL, D], 0.017, &dev)?;
        let k = mk(&[B, KVH, KL, D], 0.023, &dev)?;
        let v = mk(&[B, KVH, KL, D], 0.031, &dev)?;
        let sinks = mk(&[H], 0.041, &dev)?;
        let p = SdpaParams {
            n_kv_groups: H / KVH,
            softcap: None,
            softmax_scale: 1.0 / (D as f32).sqrt(),
            sliding_window: None,
            sinks: None,
        };
        let mask = band_mask(QL, KL, 8, &dev)?;

        let got = to_f32(&sinks_attn(&q, &k, &v, &sinks, Some(&mask), None, &p)?);
        let want = to_f32(&crate::attention::arcflash::union_attention(
            &q,
            &k,
            &v,
            Some(&mask),
            Some(&sinks),
            &p,
        )?);

        // Negative control on the OUTPUT, not a shared input: a shared-input
        // perturbation moves both sides together and can never fire.
        {
            let mut probe = got.clone();
            assert_eq!(max_abs(&probe, &got), 0.0);
            probe[7] = f32::from_bits(probe[7].to_bits().wrapping_add(1));
            assert!(max_abs(&probe, &got) > 0.0, "comparator is inert");
        }

        let d = max_abs(&got, &want);
        let scale = want.iter().fold(0f32, |m, x| m.max(x.abs()));
        eprintln!(
            "masked dispatch vs masked reference: max abs diff {d:.8} (magnitude {scale:.4})"
        );
        assert!(
            d < 1e-3,
            "masked sinks attention does not match the masked reference \
             (max abs diff {d}, magnitude {scale}) -- the mask is being dropped again"
        );
        Ok(())
    }
}
