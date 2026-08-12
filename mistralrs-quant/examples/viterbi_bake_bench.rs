// RUN-161: Viterbi 3-D MoE bake timing harness.
//
// Times one quantize_with_options_3d (Viterbi) call and prints seconds + a
// dequant checksum. Bit-identical kernel changes must NOT move the checksum.
// All knobs via env so we can iterate without recompiling:
//   ARC_QTIP_E / ARC_QTIP_N / ARC_QTIP_K   -> expert-stack dims (default gate)
//   ARC_QTIP_EXPERT_BATCH                   -> experts per quantize call
//   ARC_VITERBI_SCRATCH_GB                  -> rows-in-flight scratch budget
//
//   cargo run --release --example viterbi_bake_bench --features cuda
//
// V4-Flash projections (256 experts, hidden=7168, moe_inter=2048):
//   gate/up : E=256 N=2048 K=7168   (long Viterbi -> dominant cost)
//   down    : E=256 N=7168 K=2048
// Iterate at reduced N (e.g. N=256): rows = E*N >> rows_in_flight so throughput
// is representative; multiply the time by (2048/N) to extrapolate to a full
// gate/up projection.
use candle_core::{DType, Device, Tensor};
use mistralrs_quant::{QtipLayer, QtipMode};
use std::time::Instant;

fn env_usize(k: &str, d: usize) -> usize {
    std::env::var(k).ok().and_then(|s| s.parse().ok()).unwrap_or(d)
}

fn main() -> candle_core::Result<()> {
    let e = env_usize("ARC_QTIP_E", 256);
    let n = env_usize("ARC_QTIP_N", 2048);
    let k = env_usize("ARC_QTIP_K", 7168);
    let batch = env_usize("ARC_QTIP_EXPERT_BATCH", 16);
    let scratch = std::env::var("ARC_VITERBI_SCRATCH_GB").unwrap_or_else(|_| "default(6)".into());
    println!(
        "Viterbi timing: E={e} N={n} K={k} | expert_batch={batch} scratch_gb={scratch} | rows={}",
        e * n
    );

    let gen0 = Instant::now();
    let w_cpu = Tensor::randn(0f32, 0.02f32, (e, n, k), &Device::Cpu)?.to_dtype(DType::BF16)?;
    println!("  weight built in {:.1}s", gen0.elapsed().as_secs_f64());

    // Warm GPU + kernels (small) so the timed run excludes lazy init.
    let warm = Tensor::randn(0f32, 0.02f32, (2, 64, k), &Device::Cpu)?.to_dtype(DType::BF16)?;
    let _ = QtipLayer::quantize_with_options_3d(&warm, &Device::Cpu, QtipMode::Viterbi, true)?;

    // BAKE TIME = quantize only (a real bake writes packed weights; it never
    // dequantizes). Time it in isolation; the dequant below is only for the
    // quality check and is NOT part of bake cost.
    let t0 = Instant::now();
    let layer = QtipLayer::quantize_with_options_3d(&w_cpu, &Device::Cpu, QtipMode::Viterbi, true)?;
    let secs = t0.elapsed().as_secs_f64();

    let td = Instant::now();
    let deq = layer.dequantize_w()?.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
    let dequant_s = td.elapsed().as_secs_f64();

    // Quality vs original FP weights: cosine similarity of the full tensors.
    // This is the metric that moves with L (exact L=16 ~= 0.96; lower L worse).
    let a = deq.flatten_all()?;
    let b = w_cpu.to_dtype(DType::F32)?.flatten_all()?;
    let dot = (&a * &b)?.sum_all()?.to_scalar::<f32>()? as f64;
    let na = (&a * &a)?.sum_all()?.to_scalar::<f32>()?.sqrt() as f64;
    let nb = (&b * &b)?.sum_all()?.to_scalar::<f32>()?.sqrt() as f64;
    let cos = dot / (na * nb + 1e-12);
    let checksum = a.abs()?.sum_all()?.to_scalar::<f32>()? as f64;

    let full_gate = secs * (2048.0 / n as f64); // extrapolate this projection to N=2048
    // down [256,7168,2048]: 3.5x gate rows but 0.29x per-row (num_symbols
    // 1024 vs 3584) -> ~1x a gate projection. So layer ~= gate+up+down ~= 3x.
    let full_layer = 3.0 * full_gate;
    println!("  QUANTIZE(bake): {secs:.2}s   [dequant-only(not bake): {dequant_s:.2}s]   QUALITY cos(deq,fp)={cos:.5}   checksum={checksum:.6e}");
    println!(
        "  -> full gate/up projection (N=2048) ~= {full_gate:.1}s   | full layer ~= {full_layer:.1}s   | 43-layer bake ~= {:.1} min",
        43.0 * full_layer / 60.0
    );
    Ok(())
}
