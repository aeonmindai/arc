// Step 1 correctness + host-path verification. (RUN-161)
//
// Three independent things proved here, per config:
//   host_vs_deq  : host path (gather_forward_cuda) vs a DENSE reference built
//                  from the layer's own dequantized weights via plain candle
//                  ops (index_select + bmm — no QTIP gather code). ~1.0 proves
//                  the HOST PATH itself is correct (computes gather(a, dequant W)).
//   ondev_vs_deq : my new on-device kernel vs the same dense reference. ~1.0
//                  proves the on-device path is correct INDEPENDENTLY (not just
//                  "matches host").
//   ondev_vs_host: the two code paths agree. ~1.0.
//   host_vs_full : host path vs the ORIGINAL full-precision weights. This is the
//                  QUANTIZATION QUALITY (expect ~0.88 Greedy / ~0.96 Viterbi),
//                  a sanity tie-back to the qtip_predict numbers.
use candle_core::{DType, Device, Tensor};
use mistralrs_quant::{QtipLayer, QtipMode};

fn cos(a: &Tensor, b: &Tensor) -> f64 {
    let a = a.flatten_all().unwrap().to_dtype(DType::F32).unwrap();
    let b = b.flatten_all().unwrap().to_dtype(DType::F32).unwrap();
    let dot = (&a * &b)
        .unwrap()
        .sum_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap() as f64;
    let na = (&a * &a)
        .unwrap()
        .sum_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap()
        .sqrt() as f64;
    let nb = (&b * &b)
        .unwrap()
        .sum_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap()
        .sqrt() as f64;
    dot / (na * nb + 1e-12)
}

// Dense gather-matmul reference using plain candle ops:
// y[p, :] = a[p, :] @ W[expert_p, :, :]^T   for each (token,slot) pair p.
fn dense_ref(a: &Tensor, w: &Tensor, indices: &Tensor) -> candle_core::Result<Tensor> {
    let (n_tokens, topk, k) = a.dims3()?;
    let n = w.dim(1)?; // W: [E, N, K]
    let n_pairs = n_tokens * topk;
    let idx = indices.flatten_all()?.to_dtype(DType::U32)?;
    let w_sel = w.index_select(&idx, 0)?; // [n_pairs, N, K]
    let a_flat = a.reshape((n_pairs, 1, k))?.to_dtype(DType::F32)?;
    let y = a_flat.matmul(&w_sel.to_dtype(DType::F32)?.transpose(1, 2)?)?; // [n_pairs,1,N]
    y.reshape((n_tokens, topk, n))
}

fn run(
    mode: QtipMode,
    e: usize,
    n: usize,
    k: usize,
    n_tokens: usize,
    topk: usize,
) -> candle_core::Result<()> {
    let dev = Device::new_cuda(0)?;
    let w_full = Tensor::randn(0f32, 0.02f32, (e, n, k), &dev)?; // original full precision
    let w_bf16 = w_full.to_dtype(DType::BF16)?;
    let use_rotation = matches!(mode, QtipMode::Viterbi);
    let layer = QtipLayer::quantize_with_options(&w_bf16, None, &dev, mode, use_rotation)?;

    let a = Tensor::randn(0f32, 1f32, (n_tokens, topk, k), &dev)?.to_dtype(DType::BF16)?;
    let idx: Vec<u32> = (0..n_tokens * topk).map(|i| (i % e) as u32).collect();
    let indices = Tensor::from_vec(idx, (n_tokens, topk), &dev)?;

    // Dense references (plain candle, no QTIP gather code):
    let w_deq = layer.dequantize_w()?.to_device(&dev)?; // quantized->dense, unrotated
    let ref_deq = dense_ref(&a, &w_deq, &indices)?; // a @ dequant(W)^T
    let ref_full = dense_ref(&a, &w_full, &indices)?; // a @ W_original^T

    std::env::remove_var("ARC_NO_QTIP_ONDEVICE_MOE");
    let y_ondevice = layer.gather_forward(&a, &indices)?;
    std::env::set_var("ARC_NO_QTIP_ONDEVICE_MOE", "1");
    let y_host = layer.gather_forward(&a, &indices)?;
    std::env::remove_var("ARC_NO_QTIP_ONDEVICE_MOE");

    let host_vs_deq = cos(&y_host, &ref_deq);
    let ondev_vs_deq = cos(&y_ondevice, &ref_deq);
    let ondev_vs_host = cos(&y_ondevice, &y_host);
    let host_vs_full = cos(&y_host, &ref_full);

    let modestr = if use_rotation {
        "Viterbi(rot)"
    } else {
        "Greedy(norot)"
    };
    println!(
        "{modestr:14} E={e:<3} N={n:<5} K={k:<5} tok={n_tokens} topk={topk} | host_vs_deq={host_vs_deq:.6} ondev_vs_deq={ondev_vs_deq:.6} ondev_vs_host={ondev_vs_host:.6} | host_vs_full={host_vs_full:.4}"
    );
    Ok(())
}

fn main() -> candle_core::Result<()> {
    println!("host_vs_deq / ondev_vs_deq ~1.0 => both paths CORRECT (vs dense dequant ref).");
    println!("host_vs_full ~0.88 Greedy / ~0.96 Viterbi => quantization quality (tie-back).\n");
    run(QtipMode::Greedy, 8, 256, 512, 1, 6)?;
    run(QtipMode::Greedy, 16, 512, 1024, 1, 6)?;
    run(QtipMode::Viterbi, 8, 256, 512, 1, 6)?;
    run(QtipMode::Viterbi, 32, 2048, 4096, 1, 6)?; // V4-Flash expert shape
    run(QtipMode::Greedy, 8, 256, 512, 4, 6)?; // small-batch (MTP draft)
    Ok(())
}
