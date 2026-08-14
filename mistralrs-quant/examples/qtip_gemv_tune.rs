// Autotuner sweep for the qtip2b bitshift-trellis fused decode+GEMV.
//
// Runs EVERY compiled kernel variant across the decode shapes that matter for
// V4-Flash (gate/up N=2048 K=4096, down N=4096 K=2048; topk 6 and 8;
// b = 1..8), plus the legacy fixed-config kernel as the baseline row. Two
// generations are compiled in and swept together:
//
//   gen 1 (kernels/qtip/qtip_bitshift_tune.cu, ids 0..43): warps/block x
//     rows/warp x ILP streams x vector load width x launch_bounds min-blocks
//     x smem staging x prefetch.
//   gen 2 (kernels/qtip/qtip_bitshift_tune2.cu, ids 44..): cp.async
//     double/triple-buffered staged pipeline x rows/warp x ILP streams out of
//     smem x staged load width x split-K x producer-warp specialization.
//
// The sweep also:
//
//   * prints per-variant GB/s + %peak and a best-per-shape table,
//   * writes tune_results.json — its `winners` array is directly consumable
//     by the production dispatch via ARC_QTIP_TUNE_TABLE=/path/to/it
//     (measure -> apply with NO recompile), and
//   * writes qtip_tune_baked.rs — a ready-to-paste replacement for
//     `QTIP2B_GEMV_BAKED_TABLE` in src/qtip/tune.rs.
//
// Methodology (same as the RUN-161 bandwidth microbench this replaces
// as the tuning driver, examples/qtip_gemv_bw.rs):
//   * E=64 experts (>100 MB of packed codes per shape) with rotating index
//     windows so the working set is evicted from L2 (~60 MB on H200) before
//     reuse — measured numbers are HBM, not cache.
//   * Per point we report total us/call (incl. ~19 us Rust-side overhead);
//     the least-squares slope over pairs isolates the overhead-free marginal
//     kernel bandwidth. Variants are ranked by summed time across all points
//     (fixed overhead is identical across variants, so the ranking is pure
//     kernel).
//   * The gather path is driven through the real `gather_forward` entry
//     (on-device MoE dispatch, n_tokens <= 8), with the variant forced via
//     `set_forced_gemv_variant` — exactly the code path serving uses.
//
// Also prints a batch-64 grouped-GEMM reference row per shape (n_tokens=64
// routes to the trellis grouped GEMM, a different kernel with its own tile
// config — not part of this variant grid, just context for session 4).
//
// Knobs: ARC_TUNE_ITERS (default 150), ARC_TUNE_EXPERTS (default 64),
//        ARC_PEAK_BW_GBPS (default 4800 = H200 HBM3e),
//        ARC_TUNE_VARIANTS — restrict the sweep to a subset, so a rental
//        session can re-measure one generation without paying for the whole
//        grid. Accepts `gen1`, `gen2`, or a comma list of ids and inclusive
//        ranges, e.g. `44-97`, `0,6,21,44-60`. The legacy baseline row is
//        always measured.
//
//   cargo run --release -p mistralrs-quant --example qtip_gemv_tune --features cuda

use candle_core::{DType, Device, Result, Tensor};
use mistralrs_quant::{
    gemv_variant_applicable, qtip2b_gemv_num_variants, qtip2b_gemv_variant_desc,
    set_forced_gemv_variant, Qtip2bLayer, QtipMode, QuantMethod, QTIP2B_GEMV_VARIANT_LEGACY,
};
use std::sync::Arc;
use std::time::Instant;

struct Shape {
    name: &'static str,
    n: usize, // out rows per expert
    k: usize, // in features
}

/// (topk, batch) points swept per variant. pairs = topk * b; the kernel only
/// sees pairs, so topk=6 b=1..8 plus topk=8 b=8 covers both V4 routing
/// configs without doubling the sweep.
const POINTS: &[(usize, usize)] = &[(6, 1), (6, 2), (6, 4), (6, 8), (8, 8)];

// bytes read per touched expert: 2-bit codes (n*k/4, no padding waste) + F32 scales
fn bytes_per_expert(s: &Shape) -> f64 {
    s.n as f64 * s.k as f64 * 0.25 + s.n as f64 * 4.0
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

struct PointResult {
    topk: usize,
    b: usize,
    pairs: usize,
    us: f64,
    gbps: f64,
    pct_peak: f64,
}

struct VariantResult {
    variant: u32, // QTIP2B_GEMV_VARIANT_LEGACY for the baseline row
    label: String,
    points: Vec<PointResult>,
    total_us: f64,
    slope_gbps: f64,
}

fn time_call(
    dev: &Device,
    layer: &Arc<dyn QuantMethod>,
    a: &Tensor,
    idx_sets: &[Tensor],
    iters: usize,
) -> Result<f64> {
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

/// Least-squares fit secs = intercept + slope*pairs -> marginal kernel GB/s.
fn slope_gbps(points: &[PointResult], bpe: f64) -> f64 {
    let n = points.len() as f64;
    let sx: f64 = points.iter().map(|p| p.pairs as f64).sum();
    let sy: f64 = points.iter().map(|p| p.us * 1e-6).sum();
    let sxx: f64 = points.iter().map(|p| (p.pairs as f64).powi(2)).sum();
    let sxy: f64 = points.iter().map(|p| p.pairs as f64 * p.us * 1e-6).sum();
    let denom = n * sxx - sx * sx;
    if denom.abs() < f64::EPSILON {
        return 0.0;
    }
    let slope = (n * sxy - sx * sy) / denom;
    if slope <= 0.0 {
        return 0.0;
    }
    bpe / slope / 1e9
}

#[allow(clippy::too_many_arguments)]
fn run_variant(
    dev: &Device,
    layer: &Arc<dyn QuantMethod>,
    s: &Shape,
    e: usize,
    iters: usize,
    peak: f64,
    variant: u32,
    label: String,
) -> Result<VariantResult> {
    let bpe = bytes_per_expert(s);
    set_forced_gemv_variant(Some(variant));
    let mut points = Vec::with_capacity(POINTS.len());
    for &(topk, b) in POINTS {
        let n_pairs = b * topk;
        // Rotating index sets: each iter touches a different expert window
        // so L2 never serves the packed codes twice.
        let n_rot = 16usize;
        let mut idx_sets = Vec::with_capacity(n_rot);
        for r in 0..n_rot {
            let idx: Vec<u32> = (0..n_pairs)
                .map(|i| (((r * n_pairs) + i) % e) as u32)
                .collect();
            idx_sets.push(Tensor::from_vec(idx, (b, topk), dev)?);
        }
        let a = Tensor::randn(0f32, 1f32, (b, topk, s.k), dev)?.to_dtype(DType::BF16)?;
        let secs = time_call(dev, layer, &a, &idx_sets, iters)?;
        let bytes = bpe * n_pairs as f64;
        let bw = bytes / secs;
        points.push(PointResult {
            topk,
            b,
            pairs: n_pairs,
            us: secs * 1e6,
            gbps: bw / 1e9,
            pct_peak: bw / peak * 100.0,
        });
    }
    let total_us: f64 = points.iter().map(|p| p.us).sum();
    let slope = slope_gbps(&points, bpe);
    Ok(VariantResult {
        variant,
        label,
        points,
        total_us,
        slope_gbps: slope,
    })
}

fn variant_name(v: u32) -> String {
    if v == QTIP2B_GEMV_VARIANT_LEGACY {
        "legacy".to_string()
    } else {
        format!("{v}")
    }
}

/// Variant ids to sweep, honoring `ARC_TUNE_VARIANTS` (see the header).
/// Unparseable entries are ignored; an empty selection falls back to the
/// full grid so a typo cannot silently produce a one-row sweep.
fn selected_variants(n_variants: usize) -> Vec<u32> {
    let all: Vec<u32> = (0..n_variants as u32).collect();
    let Ok(spec) = std::env::var("ARC_TUNE_VARIANTS") else {
        return all;
    };
    let spec = spec.trim();
    let by_gen = |g: u8| -> Vec<u32> {
        all.iter()
            .copied()
            .filter(|v| qtip2b_gemv_variant_desc(*v as usize).is_some_and(|d| d.generation == g))
            .collect()
    };
    let picked: Vec<u32> = match spec.to_ascii_lowercase().as_str() {
        "gen1" => by_gen(1),
        "gen2" => by_gen(2),
        _ => {
            let mut ids = vec![];
            for part in spec.split(',') {
                let part = part.trim();
                match part.split_once('-') {
                    Some((a, b)) => {
                        if let (Ok(a), Ok(b)) = (a.trim().parse::<u32>(), b.trim().parse::<u32>()) {
                            ids.extend((a..=b).filter(|v| (*v as usize) < n_variants));
                        }
                    }
                    None => {
                        if let Ok(v) = part.parse::<u32>() {
                            if (v as usize) < n_variants {
                                ids.push(v);
                            }
                        }
                    }
                }
            }
            ids.sort_unstable();
            ids.dedup();
            ids
        }
    };
    if picked.is_empty() {
        println!("ARC_TUNE_VARIANTS={spec} selected nothing; sweeping the full grid");
        return all;
    }
    println!(
        "ARC_TUNE_VARIANTS={spec} -> {} of {n_variants} variants\n",
        picked.len()
    );
    picked
}

fn main() -> Result<()> {
    // The decode-regime on-device MoE path is what we're tuning; make sure a
    // stale kill-switch doesn't silently reroute the sweep to fallbacks.
    std::env::remove_var("ARC_NO_QTIP_ONDEVICE_MOE");
    let dev = Device::new_cuda(0)?;
    let peak = env_usize("ARC_PEAK_BW_GBPS", 4800) as f64 * 1e9;
    let iters = env_usize("ARC_TUNE_ITERS", 150);
    let e = env_usize("ARC_TUNE_EXPERTS", 64);
    let n_variants = qtip2b_gemv_num_variants();

    if n_variants == 0 {
        println!(
            "no tuned variants compiled in (needs --features cuda on SM80+); nothing to sweep"
        );
        return Ok(());
    }
    let n_gen2 = (0..n_variants)
        .filter(|v| qtip2b_gemv_variant_desc(*v).is_some_and(|d| d.generation >= 2))
        .count();
    println!(
        "=== qtip2b GEMV autotune: {} variants ({} gen1 + {} gen2) + legacy | E={e} iters={iters} peak {:.1} TB/s ===\n",
        n_variants,
        n_variants - n_gen2,
        n_gen2,
        peak / 1e12
    );
    let sweep_ids = selected_variants(n_variants);

    let shapes = [
        Shape {
            name: "gate",
            n: 2048,
            k: 4096,
        },
        Shape {
            name: "down",
            n: 4096,
            k: 2048,
        },
    ];

    let mut winners: Vec<serde_json::Value> = vec![];
    let mut shape_reports: Vec<serde_json::Value> = vec![];

    for s in &shapes {
        let bpe = bytes_per_expert(s);
        println!(
            "[{}] N={} K={}  (bytes/expert={:.2} MB)",
            s.name,
            s.n,
            s.k,
            bpe / 1e6
        );
        let w = Tensor::randn(0f32, 0.02f32, (e, s.n, s.k), &dev)?.to_dtype(DType::BF16)?;
        let layer = Qtip2bLayer::quantize_with_options_3d(&w, &dev, QtipMode::Greedy, false)?;
        drop(w);

        // Baseline row first, then every applicable variant.
        let mut results: Vec<VariantResult> = vec![run_variant(
            &dev,
            &layer,
            s,
            e,
            iters,
            peak,
            QTIP2B_GEMV_VARIANT_LEGACY,
            "legacy(w8_r2_i1_v2_replay)".to_string(),
        )?];
        let mut skipped: Vec<u32> = vec![];
        for &v in &sweep_ids {
            if !gemv_variant_applicable(v, s.n, s.k) {
                skipped.push(v);
                continue;
            }
            let label = qtip2b_gemv_variant_desc(v as usize)
                .map(|d| d.label())
                .unwrap_or_else(|| format!("v{v}"));
            results.push(run_variant(&dev, &layer, s, e, iters, peak, v, label)?);
        }
        set_forced_gemv_variant(None);

        for r in &results {
            let b1 = &r.points[0];
            let bmax = r.points.iter().max_by_key(|p| p.pairs).unwrap();
            println!(
                "  v{:<7} {:<28} | b{} {:>7.1}us | b{}t{} {:>7.1}us {:>6.0} GB/s {:>4.1}% | slope {:>6.0} GB/s {:>4.1}% | total {:>8.1}us",
                variant_name(r.variant),
                r.label,
                b1.b,
                b1.us,
                bmax.b,
                bmax.topk,
                bmax.us,
                bmax.gbps,
                bmax.pct_peak,
                r.slope_gbps,
                r.slope_gbps * 1e9 / peak * 100.0,
                r.total_us,
            );
        }
        if !skipped.is_empty() {
            println!("  (skipped inapplicable variants: {skipped:?})");
        }

        let best = results
            .iter()
            .min_by(|a, b| a.total_us.total_cmp(&b.total_us))
            .unwrap();
        let legacy_total = results[0].total_us;
        println!(
            "  -> WINNER [{}]: v{} {} | total {:.1}us vs legacy {:.1}us ({:+.1}%) | slope {:.0} GB/s ({:.1}% peak)",
            s.name,
            variant_name(best.variant),
            best.label,
            best.total_us,
            legacy_total,
            (best.total_us / legacy_total - 1.0) * 100.0,
            best.slope_gbps,
            best.slope_gbps * 1e9 / peak * 100.0,
        );

        // Batch-64 grouped-GEMM reference (different kernel; context only).
        {
            let (topk, b) = (6usize, 64usize);
            let n_pairs = b * topk;
            let idx: Vec<u32> = (0..n_pairs).map(|i| (i % e) as u32).collect();
            let idx_sets = vec![Tensor::from_vec(idx, (b, topk), &dev)?];
            let a = Tensor::randn(0f32, 1f32, (b, topk, s.k), &dev)?.to_dtype(DType::BF16)?;
            let secs = time_call(&dev, &layer, &a, &idx_sets, iters.min(50))?;
            let bw = bpe * n_pairs.min(e) as f64 / secs; // >=E pairs touch every expert once
            println!(
                "  (grouped-GEMM reference b={b} topk={topk}: {:.1}us/call, ~{:.0} GB/s weight-side)\n",
                secs * 1e6,
                bw / 1e9
            );
        }

        winners.push(serde_json::json!({
            "n": s.n,
            "k": s.k,
            "variant": best.variant,
        }));
        shape_reports.push(serde_json::json!({
            "name": s.name,
            "n": s.n,
            "k": s.k,
            "bytes_per_expert": bpe,
            "skipped_variants": skipped,
            "results": results.iter().map(|r| serde_json::json!({
                "variant": r.variant,
                "label": r.label,
                "total_us": r.total_us,
                "slope_gbps": r.slope_gbps,
                "points": r.points.iter().map(|p| serde_json::json!({
                    "topk": p.topk,
                    "b": p.b,
                    "pairs": p.pairs,
                    "us": p.us,
                    "gbps": p.gbps,
                    "pct_peak": p.pct_peak,
                })).collect::<Vec<_>>(),
            })).collect::<Vec<_>>(),
        }));
    }

    // ---- Artifacts -------------------------------------------------------
    // tune_results.json: `winners` is the machine-consumable part — point
    // ARC_QTIP_TUNE_TABLE at this file and the dispatch applies it, no
    // recompile. `shapes` carries the full sweep for analysis.
    let report = serde_json::json!({
        "device": "cuda:0",
        "peak_gbps": peak / 1e9,
        "iters": iters,
        "experts": e,
        "num_variants": n_variants,
        // Which ids this run actually measured — a filtered sweep's winners
        // are only best-of-subset, so record the subset next to them.
        "swept_variants": sweep_ids,
        "winners": winners,
        "shapes": shape_reports,
    });
    std::fs::write(
        "tune_results.json",
        serde_json::to_string_pretty(&report).map_err(candle_core::Error::wrap)?,
    )
    .map_err(candle_core::Error::wrap)?;

    // qtip_tune_baked.rs: paste over QTIP2B_GEMV_BAKED_TABLE in
    // src/qtip/tune.rs to bake the winners into the default dispatch.
    let mut baked = String::from(
        "// Generated by examples/qtip_gemv_tune.rs — winners per decode shape.\n\
         pub const QTIP2B_GEMV_BAKED_TABLE: &[GemvTuneEntry] = &[\n",
    );
    for w in &winners {
        baked.push_str(&format!(
            "    GemvTuneEntry {{ n: {}, k: {}, variant: {} }},\n",
            w["n"], w["k"], w["variant"]
        ));
    }
    baked.push_str("];\n");
    std::fs::write("qtip_tune_baked.rs", baked).map_err(candle_core::Error::wrap)?;

    println!("wrote tune_results.json (ARC_QTIP_TUNE_TABLE consumable) + qtip_tune_baked.rs");
    Ok(())
}
