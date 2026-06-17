// RUN-161: Viterbi 3-D MoE bake speedup validation.
//
// Times `quantize_with_options_3d` at ARC_QTIP_EXPERT_BATCH = 1 (the old
// per-expert serial path: one CPU<->GPU round-trip per expert) vs 16/32 (the
// new batched path), and PROVES the dequantized output is bit-identical across
// batch sizes -> the speedup is quality-neutral. Extrapolates to a full
// 43-layer V4-Flash bake.
//
// Run on a CUDA box:
//   ARC_QTIP_E=256 ARC_QTIP_N=2048 ARC_QTIP_K=7168 \
//     cargo run --release --example viterbi_bake_bench --features cuda
//
// V4-Flash projections (256 experts, hidden=7168, moe_inter=2048):
//   gate/up : E=256 N=2048 K=7168   (long Viterbi -> dominant cost)
//   down    : E=256 N=7168 K=2048
use candle_core::{DType, Device, Tensor};
use mistralrs_quant::{QtipLayer, QtipMode};
use std::time::Instant;

fn env_usize(k: &str, d: usize) -> usize {
    std::env::var(k).ok().and_then(|s| s.parse().ok()).unwrap_or(d)
}

// Bake one [E,N,K] stack at a given expert-batch; return (seconds, dequant CPU f32).
fn bake(weight_cpu: &Tensor, batch: usize) -> candle_core::Result<(f64, Tensor)> {
    std::env::set_var("ARC_QTIP_EXPERT_BATCH", batch.to_string());
    let t0 = Instant::now();
    let layer =
        QtipLayer::quantize_with_options_3d(weight_cpu, &Device::Cpu, QtipMode::Viterbi, true)?;
    // dequantize_w forces the full pipeline to complete and gives us the bytes
    // to compare. Pull to CPU f32 so the comparison is exact.
    let w = layer
        .dequantize_w()?
        .to_device(&Device::Cpu)?
        .to_dtype(DType::F32)?
        .flatten_all()?;
    let secs = t0.elapsed().as_secs_f64();
    Ok((secs, w))
}

fn sum_abs_diff(a: &Tensor, b: &Tensor) -> candle_core::Result<f64> {
    Ok((a - b)?.abs()?.sum_all()?.to_scalar::<f32>()? as f64)
}

fn main() -> candle_core::Result<()> {
    let e = env_usize("ARC_QTIP_E", 256);
    let n = env_usize("ARC_QTIP_N", 2048);
    let k = env_usize("ARC_QTIP_K", 7168);
    println!("V4 Viterbi bake bench: E={e} N={n} K={k} (weight on CPU, GPU per-batch quant)\n");

    let gen0 = Instant::now();
    let w_cpu = Tensor::randn(0f32, 0.02f32, (e, n, k), &Device::Cpu)?.to_dtype(DType::BF16)?;
    println!("  built [{e},{n},{k}] bf16 weight in {:.1}s", gen0.elapsed().as_secs_f64());

    // Warm GPU + kernels so the first real timing isn't skewed by lazy init.
    let warm = Tensor::randn(0f32, 0.02f32, (2, 64, k), &Device::Cpu)?.to_dtype(DType::BF16)?;
    let _ = QtipLayer::quantize_with_options_3d(&warm, &Device::Cpu, QtipMode::Viterbi, true)?;
    println!("  (warmup done)\n");

    let (t_b1, w_b1) = bake(&w_cpu, 1)?;
    println!("  batch= 1 (old serial): {t_b1:8.2}s   <- baseline");
    let (t_b16, w_b16) = bake(&w_cpu, 16)?;
    println!("  batch=16 (new):        {t_b16:8.2}s   speedup {:.2}x", t_b1 / t_b16);
    let (t_b32, w_b32) = bake(&w_cpu, 32)?;
    println!("  batch=32 (new):        {t_b32:8.2}s   speedup {:.2}x", t_b1 / t_b32);

    let d16 = sum_abs_diff(&w_b1, &w_b16)?;
    let d32 = sum_abs_diff(&w_b1, &w_b32)?;
    println!(
        "\n  bit-identical check (sum|dequant_b1 - dequant_bN|): b16={d16:.3e}  b32={d32:.3e}  (0 => identical)"
    );

    // Extrapolate to a full bake. 43 layers; this projection's time stands in
    // for gate+up (2x) when run with gate dims, or down (1x) with down dims.
    let best = t_b16.min(t_b32);
    println!(
        "\n  this projection @ best batch = {best:.1}s -> x43 layers x(this projection count)\n  e.g. if these are gate dims: 2*43*{best:.1}s = {:.1} min for gate+up alone",
        2.0 * 43.0 * best / 60.0
    );
    Ok(())
}
