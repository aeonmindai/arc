// CUDA-graph capture reproducer (RUN-161 debug).
// Exercises V4's custom kernels under capture: QTIP on-device gather (CLEARED),
// plus the fixed-capacity KV write (kvwrite) + narrow + SDPA-like matmul.
//
//   cargo run --release -p arc-cuda-graph --features cuda --example capture_probe
#[cfg(feature = "cuda")]
fn main() -> candle_core::Result<()> {
    use arc_cuda_graph::CudaGraphRunner;
    use candle_core::{DType, Device, Tensor};
    use mistralrs_quant::kvwrite::write_kv_inplace;
    use mistralrs_quant::{QtipLayer, QtipMode};

    let dev = Device::new_cuda_with_stream(0)?;
    if let Device::Cuda(cd) = &dev {
        cd.set_alloc_cache_enabled(true);
    }
    println!("device created (with stream, alloc cache ON)");

    // --- QTIP gather (V4 MoE) ---
    let (e, k, ntok, topk) = (8usize, 512usize, 1usize, 6usize);
    let w = Tensor::randn(0f32, 0.02f32, (e, 512usize, k), &dev)?.to_dtype(DType::BF16)?;
    let layer = QtipLayer::quantize_with_options(&w, None, &dev, QtipMode::Greedy, false)?;
    let idx: Vec<u32> = (0..ntok * topk).map(|i| (i % e) as u32).collect();
    let indices = Tensor::from_vec(idx, (ntok, topk), &dev)?;
    let x = Tensor::randn(0f32, 1f32, (ntok, topk, k), &dev)?.to_dtype(DType::BF16)?;

    // --- Fixed-capacity KV (V4 attention) ---
    let cap = 128usize;
    let kvbuf = Tensor::zeros((1, 1, cap, 512), DType::BF16, &dev)?; // all_data (held, stable)
    let newk = Tensor::randn(0f32, 1f32, (1, 1, 1, 512), &dev)?.to_dtype(DType::BF16)?;
    let pos = Tensor::from_vec(vec![5u32], (1,), &dev)?; // device slot
    let q = Tensor::randn(0f32, 1f32, (1, 1, 1, 512), &dev)?.to_dtype(DType::BF16)?;

    let run = |a: &Tensor| -> candle_core::Result<Tensor> {
        // MoE gather
        let moe = layer.gather_forward(a, &indices)?; // [1,6,512]
        // Fixed-cap KV write at device slot + constant narrow (V4 2c-read pattern)
        write_kv_inplace(&kvbuf, &newk, &pos)?;
        let kfull = kvbuf.narrow(2, 0, cap)?.contiguous()?; // [1,1,128,512]
        // SDPA-like: q @ kfull^T -> [1,1,1,128]
        let scores = q.matmul(&kfull.transpose(2, 3)?.contiguous()?)?; // [1,1,1,128]
        let ctx = scores.matmul(&kfull)?; // [1,1,1,512]
        let moe_sum = moe.sum(1)?.reshape((1, 1, 1, 512))?;
        ctx.broadcast_add(&moe_sum)
    };

    for _ in 0..5 {
        let _ = run(&x)?;
    }
    dev.synchronize()?;
    println!("cache warmed (5 eager forwards)");

    let eager = run(&x)?;
    dev.synchronize()?;
    let eager_v = eager.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    println!("eager forward ok, out[0..3]={:?}", &eager_v[0..3]);

    let mut runner = CudaGraphRunner::new(&dev, 0)?;
    if !runner.is_enabled() {
        panic!("capture not enabled");
    }
    println!("begin_capture...");
    let (gp, op) = runner.begin_capture(1)?;
    let out = run(&x)?;
    println!("recorded; end_capture_and_cache (instantiate + launch)...");
    let captured = runner.end_capture_and_cache(1, out, gp, op)?;
    dev.synchronize()?;
    let cap_v = captured.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    let maxdiff = eager_v
        .iter()
        .zip(cap_v.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    println!("CAPTURE+LAUNCH OK. max|captured-eager| = {maxdiff:.3e}");
    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn main() {
    println!("requires --features cuda");
}
