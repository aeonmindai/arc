// RUN-161: effective-bandwidth microbench for the 2-bit QTIP gather-GEMV (decode).
//
// Question: at b=1 decode, does our 2-bit gather-GEMV realize enough of H200 HBM
// (4.8 TB/s) that the ~4x byte advantage over FP8 turns into speed? Or is it
// occupancy/overhead bound (low effective BW), eating the advantage?
//
// The on-device decode kernel is gated to n_tokens <= 8 (see gather_forward);
// above that it falls to a host dequantize path. So we sweep b = 1..8 with the
// REAL decode shape a=(n_tokens=b, topk=6, k), indices=(b,6).
//
// Two things separated:
//   - total us/call         : what the engine actually pays per MoE gather (incl.
//                             Rust-side overhead: env reads, contiguous, alloc).
//   - marginal kernel BW    : slope d(time)/d(pairs) -> the overhead-free kernel
//                             bandwidth. intercept -> per-call fixed overhead.
//
// L2 defeat: H200 L2 ~60MB; one b=1 read (~6 experts, ~12MB) would stay resident
// across iters and report L2 BW. We use E=64 (>100MB packed) and rotate which
// experts each iter touches so the working set is evicted before reuse.
//
//   cargo run --release -p mistralrs-quant --example qtip_gemv_bw --features cuda

use candle_core::{DType, Device, Tensor};
use mistralrs_quant::{QtipLayer, QtipMode};
use std::time::Instant;

const H200_PEAK: f64 = 4.8e12; // bytes/s
const TOPK: usize = 6; // V4-Flash active experts/token

struct Shape {
    name: &'static str,
    n: usize, // out rows
    k: usize, // in features
}

// bytes read per touched expert: 2-bit codes (n*k/4, no padding waste) + F32 scales
fn bytes_per_expert(s: &Shape) -> f64 {
    s.n as f64 * s.k as f64 * 0.25 + s.n as f64 * 4.0
}

fn time_call(dev: &Device, layer: &std::sync::Arc<dyn mistralrs_quant::QuantMethod>, a: &Tensor, idx_sets: &[Tensor], iters: usize) -> candle_core::Result<f64> {
    for r in 0..8 {
        let _ = layer.gather_forward(a, &idx_sets[r % idx_sets.len()])?;
    }
    dev.synchronize()?;
    let t = Instant::now();
    for i in 0..iters {
        let _ = layer.gather_forward(a, &idx_sets[i % idx_sets.len()])?;
    }
    dev.synchronize()?;
    Ok(t.elapsed().as_secs_f64() / iters as f64)
}

fn run_shape(dev: &Device, s: &Shape, e: usize, iters: usize) -> candle_core::Result<()> {
    let w = Tensor::randn(0f32, 0.02f32, (e, s.n, s.k), dev)?.to_dtype(DType::BF16)?;
    let layer = QtipLayer::quantize_with_options(&w, None, dev, QtipMode::Greedy, false)?;
    std::env::remove_var("ARC_NO_QTIP_ONDEVICE_MOE");

    let bpe = bytes_per_expert(s);
    println!("[{}] N={} K={}  (bytes/expert={:.2} MB)", s.name, s.n, s.k, bpe / 1e6);

    let batches = [1usize, 2, 4, 8];
    let mut pts: Vec<(f64, f64)> = vec![]; // (pairs, secs)
    for &b in &batches {
        let n_pairs = b * TOPK;
        // rotating index sets: each iter touches a different window of experts
        let n_rot = 16usize;
        let mut idx_sets = Vec::with_capacity(n_rot);
        for r in 0..n_rot {
            let idx: Vec<u32> = (0..n_pairs).map(|i| (((r * n_pairs) + i) % e) as u32).collect();
            idx_sets.push(Tensor::from_vec(idx, (b, TOPK), dev)?);
        }
        let a = Tensor::randn(0f32, 1f32, (b, TOPK, s.k), dev)?.to_dtype(DType::BF16)?;
        let secs = time_call(dev, &layer, &a, &idx_sets, iters)?;
        let bytes = bpe * n_pairs as f64;
        let bw = bytes / secs;
        println!(
            "  b={:<2} pairs={:<3} | {:>8.1} us/call | {:>7.0} GB/s | {:>5.1}% peak",
            b, n_pairs, secs * 1e6, bw / 1e9, bw / H200_PEAK * 100.0
        );
        pts.push((n_pairs as f64, secs));
    }

    // Least-squares fit secs = intercept + slope*pairs  -> overhead + marginal BW
    let n = pts.len() as f64;
    let sx: f64 = pts.iter().map(|p| p.0).sum();
    let sy: f64 = pts.iter().map(|p| p.1).sum();
    let sxx: f64 = pts.iter().map(|p| p.0 * p.0).sum();
    let sxy: f64 = pts.iter().map(|p| p.0 * p.1).sum();
    let slope = (n * sxy - sx * sy) / (n * sxx - sx * sx);
    let intercept = (sy - slope * sx) / n;
    let marginal_bw = bpe / slope; // bytes per pair / secs per pair
    println!(
        "  -> fixed overhead ~{:.1} us/call | marginal kernel BW {:.0} GB/s ({:.1}% peak)\n",
        intercept * 1e6,
        marginal_bw / 1e9,
        marginal_bw / H200_PEAK * 100.0
    );
    Ok(())
}

fn main() -> candle_core::Result<()> {
    let dev = Device::new_cuda(0)?;
    println!("=== QTIP 2-bit gather-GEMV decode bandwidth (H200 peak {:.1} TB/s, topk={}) ===\n", H200_PEAK / 1e12, TOPK);
    let shapes = [
        Shape { name: "gate", n: 2048, k: 4096 },
        Shape { name: "down", n: 4096, k: 2048 },
    ];
    for s in &shapes {
        run_shape(&dev, s, 64, 300)?;
    }
    Ok(())
}
