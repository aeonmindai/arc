// Step 2c-write correctness check. (RUN-161)
//
// Validates the device-indexed in-place KV write kernel `write_kv_inplace`
// against the eager oracle `Tensor::slice_set(src, dim=2, pos)` that the
// non-graph SingleCache path uses today. We simulate a decode run: write N
// successive tokens into a pre-grown [B,H,C,D] cache at positions 0..N, each
// time driving the slot from a DEVICE u32 positions tensor (the replay-safe
// mechanism). After the run the kernel-written buffer must be BITWISE-identical
// to the slice_set oracle.
//
// Run on a CUDA box:
//   cargo run --release -p mistralrs-quant --features cuda --example kvwrite_check
use candle_core::{DType, Device, Tensor};
use mistralrs_quant::kvwrite::write_kv_inplace;

fn max_abs_diff(a: &Tensor, b: &Tensor) -> f64 {
    let a = a.flatten_all().unwrap().to_dtype(DType::F32).unwrap();
    let b = b.flatten_all().unwrap().to_dtype(DType::F32).unwrap();
    (&a - &b)
        .unwrap()
        .abs()
        .unwrap()
        .max(0)
        .unwrap()
        .to_scalar::<f32>()
        .unwrap() as f64
}

fn run(
    dtype: DType,
    b: usize,
    h: usize,
    c: usize,
    d: usize,
    n_steps: usize,
) -> candle_core::Result<()> {
    let dev = Device::new_cuda(0)?;

    // Pre-grown caches: kernel-written vs slice_set oracle. Same zeros init.
    let kernel_cache = Tensor::zeros((b, h, c, d), dtype, &dev)?;
    let oracle_cache = Tensor::zeros((b, h, c, d), dtype, &dev)?;

    for step in 0..n_steps {
        // New token K/V for this step: [B, H, 1, D].
        let src = Tensor::randn(0f32, 1f32, (b, h, 1, d), &dev)?.to_dtype(dtype)?;

        // Device-held slot (same for all batch rows here): [B] u32.
        let pos: Vec<u32> = vec![step as u32; b];
        let positions = Tensor::from_vec(pos, (b,), &dev)?;

        // Kernel path (in-place, device-indexed).
        write_kv_inplace(&kernel_cache, &src, &positions)?;

        // Oracle path (eager host-offset slice_set).
        oracle_cache.slice_set(&src, 2, step)?;
    }
    dev.synchronize()?;

    let diff = max_abs_diff(&kernel_cache, &oracle_cache);
    println!(
        "[{dtype:?}] B={b} H={h} C={c} D={d} steps={n_steps}: max_abs_diff = {diff:.3e}  {}",
        if diff == 0.0 {
            "EXACT ✓"
        } else if diff < 1e-3 {
            "close ✓"
        } else {
            "MISMATCH ✗"
        }
    );
    Ok(())
}

fn main() -> candle_core::Result<()> {
    // V4 decode shape: MQA -> H=1, head_dim=512.
    run(DType::BF16, 1, 1, 256, 512, 200)?;
    run(DType::F16, 1, 1, 256, 512, 200)?;
    run(DType::F32, 1, 1, 256, 512, 200)?;
    // A couple of off-nominal shapes to exercise the index math.
    run(DType::BF16, 2, 4, 128, 64, 100)?;
    run(DType::F32, 1, 8, 64, 80, 64)?;
    Ok(())
}
