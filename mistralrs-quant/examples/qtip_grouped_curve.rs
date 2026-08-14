// Session-4: trellis grouped-GEMM batch curve (Arc Stage 4 bring-up).
//
// Question: as decode batch B grows 1 -> 64, how does the qtip2b MoE
// gather cost per step scale on the grouped-GEMM path vs the b<=8 on-device
// gather-GEMV path? This is the "aggregate tok/s curve vs the expert-read
// floor" number the fleet math (docs/FLEET.md section 2) is waiting on.
//
// Two modes per batch size, driven through the PUBLIC dispatch
// (`QuantMethod::gather_forward` on a 3-D expert stack):
//   - grouped : `ARC_QTIP_ONDEVICE_MOE_MAX_TOKENS=0` forces every B onto the
//               trellis grouped GEMM (tokens sorted by expert on-device,
//               persistent tensor-core tile loop).
//   - gemv    : default gate (on-device gather-GEMV) — only engages at
//               B <= 8, giving the crossover reference.
//
// Traffic model: with rotating index windows each call touches
// min(B*topk, E) distinct experts; "GB/s (floor)" divides that unique-expert
// byte count by the call time — the number to compare against the
// saturated-batch expert-read floor, NOT a kernel-internal bandwidth.
//
// L2 defeat: E=64 (>130MB packed) + rotating which experts each iteration
// touches, same recipe as qtip_gemv_bw.
//
// On the GPU box:
//   cargo run --release -p mistralrs-quant --example qtip_grouped_curve --features cuda
// Without `cuda` (or without a GPU) it runs a tiny CPU-reference smoke so the
// harness itself is verifiable anywhere.

use candle_core::{DType, Device, Tensor};
use mistralrs_quant::{Qtip2bLayer, QtipMode, QuantMethod};
use std::sync::Arc;
use std::time::Instant;

const H200_PEAK: f64 = 4.8e12; // bytes/s
const TOPK: usize = 6; // V4-Flash active experts/token
/// MoE layer count used ONLY for the labeled per-step extrapolation lines
/// (~40 of V4 Flash's 43 layers are MoE; parity/traffic numbers don't use it).
const MOE_LAYERS: f64 = 40.0;

struct Shape {
    name: &'static str,
    n: usize,
    k: usize,
    calls_per_layer: f64, // gate+up share a shape (2 calls), down is 1
}

// bytes per touched expert: 2-bit codes (n*k/4) + F32 row scales
fn bytes_per_expert(s: &Shape) -> f64 {
    s.n as f64 * s.k as f64 * 0.25 + s.n as f64 * 4.0
}

#[allow(clippy::too_many_arguments)]
fn time_curve(
    dev: &Device,
    layer: &Arc<dyn QuantMethod>,
    s: &Shape,
    e: usize,
    batches: &[usize],
    iters: usize,
    act_dtype: DType,
    label: &str,
) -> candle_core::Result<Vec<(usize, f64)>> {
    let bpe = bytes_per_expert(s);
    let mut out = Vec::new();
    for &b in batches {
        let n_pairs = b * TOPK;
        let n_rot = 16usize;
        let mut idx_sets = Vec::with_capacity(n_rot);
        let mut distinct = 0usize;
        for r in 0..n_rot {
            let idx: Vec<u32> = (0..n_pairs)
                .map(|i| (((r * n_pairs) + i) % e) as u32)
                .collect();
            if r == 0 {
                let mut seen = vec![false; e];
                for &v in &idx {
                    seen[v as usize] = true;
                }
                distinct = seen.iter().filter(|x| **x).count();
            }
            idx_sets.push(Tensor::from_vec(idx, (b, TOPK), dev)?);
        }
        let a = Tensor::randn(0f32, 1f32, (b, TOPK, s.k), dev)?.to_dtype(act_dtype)?;
        // warmup
        for r in 0..4usize {
            let _ = layer.gather_forward(&a, &idx_sets[r % idx_sets.len()])?;
        }
        dev.synchronize()?;
        let t = Instant::now();
        for i in 0..iters {
            let _ = layer.gather_forward(&a, &idx_sets[i % idx_sets.len()])?;
        }
        dev.synchronize()?;
        let secs = t.elapsed().as_secs_f64() / iters as f64;
        let floor_bytes = bpe * distinct as f64;
        println!(
            "  [{label}] B={:<3} pairs={:<4} experts={:<3} | {:>9.1} us/call | {:>7.0} GB/s (floor) | {:>5.1}% peak",
            b,
            n_pairs,
            distinct,
            secs * 1e6,
            floor_bytes / secs / 1e9,
            floor_bytes / secs / H200_PEAK * 100.0
        );
        out.push((b, secs));
    }
    Ok(out)
}

fn run(dev: &Device, e: usize, batches: &[usize], iters: usize, act_dtype: DType, big: bool) -> candle_core::Result<()> {
    let shapes = if big {
        vec![
            Shape { name: "gate/up", n: 2048, k: 4096, calls_per_layer: 2.0 },
            Shape { name: "down", n: 4096, k: 2048, calls_per_layer: 1.0 },
        ]
    } else {
        vec![
            Shape { name: "gate/up", n: 64, k: 128, calls_per_layer: 2.0 },
            Shape { name: "down", n: 128, k: 64, calls_per_layer: 1.0 },
        ]
    };

    // per-B per-shape grouped timings, for the extrapolation table
    let mut grouped: Vec<Vec<(usize, f64)>> = Vec::new();

    for s in &shapes {
        let w = Tensor::randn(0f32, 0.02f32, (e, s.n, s.k), dev)?.to_dtype(DType::BF16)?;
        // Greedy encode: timing-only bench (search quality is irrelevant here).
        let layer = Qtip2bLayer::quantize_with_options(&w, None, dev, QtipMode::Greedy, false)?;
        println!(
            "[{}] N={} K={} E={e} (bytes/expert={:.2} MB)",
            s.name,
            s.n,
            s.k,
            bytes_per_expert(s) / 1e6
        );

        // Mode 1: grouped GEMM for every B.
        std::env::set_var("ARC_QTIP_ONDEVICE_MOE_MAX_TOKENS", "0");
        std::env::remove_var("ARC_NO_QTIP_GROUPED_MOE");
        let g = time_curve(dev, &layer, s, e, batches, iters, act_dtype, "grouped")?;
        grouped.push(g);

        // Mode 2: default gate -> on-device gather-GEMV, B <= 8 only
        // (crossover reference).
        std::env::remove_var("ARC_QTIP_ONDEVICE_MOE_MAX_TOKENS");
        let small: Vec<usize> = batches.iter().copied().filter(|b| *b <= 8).collect();
        if matches!(dev, Device::Cuda(_)) && !small.is_empty() {
            time_curve(dev, &layer, s, e, &small, iters, act_dtype, "gemv   ")?;
        }
        println!();
    }

    // Extrapolated MoE-GEMM-only step budget (LABELED EXTRAPOLATION: ignores
    // attention, sampling, routing, and inter-layer overhead entirely).
    println!("-- extrapolated MoE-GEMM floor, {MOE_LAYERS} MoE layers (grouped path) --");
    println!("   B | est step ms | est steps/s | est aggregate tok/s");
    for i in 0..batches.len() {
        let b = batches[i];
        let mut step_secs = 0.0;
        for (si, s) in shapes.iter().enumerate() {
            step_secs += grouped[si][i].1 * s.calls_per_layer * MOE_LAYERS;
        }
        println!(
            "  {:>3} | {:>10.2} | {:>10.1} | {:>10.0}",
            b,
            step_secs * 1e3,
            1.0 / step_secs,
            b as f64 / step_secs
        );
    }
    Ok(())
}

fn main() -> candle_core::Result<()> {
    match Device::new_cuda(0) {
        Ok(dev) => {
            println!("=== qtip2b grouped-GEMM batch curve (H200 peak {:.1} TB/s, topk={TOPK}) ===\n", H200_PEAK / 1e12);
            run(&dev, 64, &[1, 2, 4, 8, 16, 32, 64], 200, DType::BF16, true)
        }
        Err(_) => {
            println!("=== no CUDA device: CPU-reference smoke (harness verification only) ===\n");
            run(&Device::Cpu, 4, &[1, 2], 2, DType::F32, false)
        }
    }
}
