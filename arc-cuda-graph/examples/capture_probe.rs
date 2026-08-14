// CUDA-graph capture reproducer (RUN-161) — comprehensive forward covering all
// V4 op types: QTIP on-device MoE gather + elementwise + reduction + cuBLAS,
// under the deferred-free capture protocol.
//   ARC_NO_DEFERRED_FREE=1 -> set_capture_mode is a no-op (tests recycling).
//
//   cargo run --release -p arc-cuda-graph --features cuda --example capture_probe
#[cfg(feature = "cuda")]
fn main() -> candle_core::Result<()> {
    use arc_cuda_graph::CudaGraphRunner;
    use candle_core::{cuda::CudaDevice, DType, Device, Tensor};
    use mistralrs_quant::{QtipLayer, QtipMode};

    let dev = Device::new_cuda_with_stream(0)?;
    let cd: &CudaDevice = match &dev {
        Device::Cuda(cd) => cd,
        _ => unreachable!(),
    };
    cd.set_alloc_cache_enabled(true);
    let deferred = std::env::var_os("ARC_NO_DEFERRED_FREE").is_none();
    println!("device created (stream, cache ON, deferred_free={deferred})");

    let (e, k, ntok, topk) = (8usize, 512usize, 1usize, 6usize);
    let w = Tensor::randn(0f32, 0.02f32, (e, 512usize, k), &dev)?.to_dtype(DType::BF16)?;
    // Greedy is banned outside this crate's own tests (DOCTRINE D4), and the
    // probe runs on CUDA where the trellis quantize is a kernel — the bake is
    // not the thing being measured here anyway.
    let layer = QtipLayer::quantize_with_mode(&w, None, &dev, QtipMode::default_expert_mode())?;
    let idx: Vec<u32> = (0..ntok * topk).map(|i| (i % e) as u32).collect();
    let indices = Tensor::from_vec(idx, (ntok, topk), &dev)?;
    let x = Tensor::randn(0f32, 1f32, (ntok, topk, k), &dev)?.to_dtype(DType::BF16)?;

    let run = |a: &Tensor| -> candle_core::Result<Tensor> {
        let moe = layer.gather_forward(a, &indices)?; // [1,6,512]
        let y = moe.affine(2.0, 1.0)?;
        let y = y.gelu()?;
        let y = (y.exp()? + 1.0)?;
        let s = y.sum(1)?; // reduction -> [1,512]
        let m = y.reshape((topk, k))?; // [6,512]
        let mm = m.matmul(&m.t()?.contiguous()?)?; // cuBLAS [6,6]
        let mm = mm.flatten_all()?.sum(0)?; // [1]
        s.reshape((1, 1, 1, k))?
            .broadcast_add(&mm.reshape((1, 1, 1, 1))?)
    };

    // 1) eager warmups (warm cuBLAS algos + kernels, populate cache peak-live)
    for _ in 0..5 {
        let _ = run(&x)?;
    }
    dev.synchronize()?;
    // 2) hold the eager reference BEFORE the deferred pass (so the held output is
    //    accounted for and the deferred pass grows free to the full count)
    let eager = run(&x)?;
    dev.synchronize()?;
    let eager_v = eager
        .flatten_all()?
        .to_dtype(DType::F32)?
        .to_vec1::<f32>()?;
    println!("eager out[0..3]={:?}", &eager_v[0..3.min(eager_v.len())]);
    // 3) deferred-free pass (no-op when ARC_NO_DEFERRED_FREE)
    cd.set_capture_mode(true);
    let _ = run(&x)?;
    dev.synchronize()?;
    cd.set_capture_mode(false);
    println!("warmed");

    let mut runner = CudaGraphRunner::new(&dev, 0)?;
    if !runner.is_enabled() {
        panic!("capture not enabled");
    }
    // 4) capture with frees deferred (or recycled if ARC_NO_DEFERRED_FREE)
    cd.set_capture_mode(true);
    println!("begin_capture...");
    let (gp, op) = runner.begin_capture(1)?;
    let out = run(&x)?;
    println!("recorded; instantiate+launch...");
    let captured = runner.end_capture_and_cache(1, out, gp, op)?;
    cd.set_capture_mode(false);
    dev.synchronize()?;
    let cap_v = captured
        .flatten_all()?
        .to_dtype(DType::F32)?
        .to_vec1::<f32>()?;
    let maxdiff = eager_v
        .iter()
        .zip(cap_v.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    println!("CAPTURE+LAUNCH OK. max|captured-eager| = {maxdiff:.3e}");

    // 5) replay (2nd launch)
    match runner.replay(1) {
        Ok(r) => {
            dev.synchronize()?;
            let rv = r.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
            let d = eager_v
                .iter()
                .zip(rv.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            println!("REPLAY OK. max|replay-eager| = {d:.3e}");
        }
        Err(e) => println!("REPLAY ERR: {e}"),
    }
    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn main() {
    println!("requires --features cuda");
}
