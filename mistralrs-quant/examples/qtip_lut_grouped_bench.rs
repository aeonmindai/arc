// LUT-rung (QtipLayer) MoE prefill: the three paths, head to head, at
// V4-Flash's real shapes.
//
// WHY THIS FIXTURE NEEDS NO VITERBI (and why that is not a cheat)
// --------------------------------------------------------------
// Every path here decodes the SAME packed bytes with the SAME trellis, and
// trellis decode cost is content-independent: the number of ALU ops and the
// number of bytes moved per weight do not depend on which symbols the search
// picked. A Viterbi bake changes WHICH symbols are stored, never how many
// bytes they occupy nor how many ops decoding them costs. So random packed
// bytes give the identical performance measurement at ~0 setup cost, and the
// correctness comparison below is likewise unaffected — both arms read the
// same bytes and must agree regardless of what those bytes mean.
// (The sibling `qtip_grouped_curve` example bakes for real and spends hours
// single-threaded at E=256; that cost buys nothing a timing needs.)
//
// THE THREE PATHS
//   * `gemv`    — fused gather-GEMV, one (token, slot) pair per grid.y, NO
//                 dedup: 6*N pair-wise weight reads against E(N) distinct
//                 experts.
//   * `dequant` — the >=683-token path: host-synced, per-distinct-expert
//                 dequantize-materialize to BF16 in HBM, then a matmul per
//                 expert.
//   * `grouped` — this change: tokens sorted by expert on-device, then the
//                 trellis grouped GEMM (`qtip_grouped_gemm_lut.cu`), so each
//                 woken expert's packed bytes are staged once per m-tile.
//
// D12 / engagement: a green must prove work happened. This harness refuses to
// print a table unless (a) the three arms are measurably distinct code paths,
// and (b) `grouped` reproduces `dequant`'s output. Environment failures exit
// 2, never 1.

use candle_core::{DType, Device, Result, Tensor};
use mistralrs_quant::{
    QtipCodebook, QtipGeometry, QtipLayer, QtipSearchDetail, QtipSearchStamp, QuantMethod,
    QTIP_GROUPED_TILE_K,
};
use std::time::Instant;

const LUT_SIZE: usize = 1 << 16;
const V: usize = 2;
/// V4-Flash routes each token to 6 of 256 experts.
const TOPK: usize = 6;
const EXPERTS: usize = 256;
/// ~40 of V4-Flash's 43 layers are MoE; used only for the labeled
/// per-prefill-token extrapolation, never for the measured columns.
const MOE_LAYERS: f64 = 40.0;

struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn f32_unit(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

/// A unit-sigma-ish table in the `[2^16, 2]` layout the rung uses. Its VALUES
/// are irrelevant to timing; they are made finite and O(1) so the correctness
/// comparison is numerically meaningful rather than dominated by junk.
fn make_lut(rng: &mut Rng) -> Vec<f32> {
    let mut v = Vec::with_capacity(LUT_SIZE * V);
    for _ in 0..LUT_SIZE * V {
        // Sum of 4 uniforms -> approximately Gaussian, mean 0, modest spread.
        let s: f32 = (0..4).map(|_| rng.f32_unit()).sum::<f32>() - 2.0;
        v.push(s);
    }
    v
}

/// Each token draws `TOPK` DISTINCT experts uniformly — real top-k semantics,
/// so the number of woken experts follows E(N) = E*(1-(1-k/E)^N).
fn routing(rng: &mut Rng, n_tokens: usize) -> Vec<u32> {
    let mut out = Vec::with_capacity(n_tokens * TOPK);
    let mut chosen: Vec<u32> = Vec::with_capacity(TOPK);
    for _ in 0..n_tokens {
        chosen.clear();
        while chosen.len() < TOPK {
            let c = rng.below(EXPERTS) as u32;
            if !chosen.contains(&c) {
                chosen.push(c);
            }
        }
        out.extend_from_slice(&chosen);
    }
    out
}

fn distinct_experts(idx: &[u32]) -> usize {
    let mut seen = vec![false; EXPERTS];
    let mut n = 0;
    for &e in idx {
        if !seen[e as usize] {
            seen[e as usize] = true;
            n += 1;
        }
    }
    n
}

fn build_layer(
    dev: &Device,
    in_features: usize,
    n_rows: usize,
    rng: &mut Rng,
) -> Result<QtipLayer> {
    let packed_per_row = in_features / 4; // 2 bits/weight
    let nblocks = EXPERTS * n_rows * packed_per_row;
    let mut blocks = Vec::with_capacity(nblocks);
    for _ in 0..nblocks {
        blocks.push((rng.next_u64() & 0xFF) as u8);
    }
    let blocks = Tensor::from_vec(blocks, (EXPERTS, n_rows, packed_per_row), dev)?;

    let mut scales = Vec::with_capacity(EXPERTS * n_rows);
    for _ in 0..EXPERTS * n_rows {
        scales.push(0.01 + rng.f32_unit() * 0.02);
    }
    let row_scales = Tensor::from_vec(scales, (EXPERTS, n_rows), dev)?;

    let lut = Tensor::from_vec(make_lut(rng), (LUT_SIZE, V), &Device::Cpu)?.to_device(dev)?;

    QtipLayer::from_stacked_parts(
        blocks,
        row_scales,
        lut,
        None,
        in_features,
        None, // no rotation: both arms would apply the identical pass, so it
        0,    // cancels in the comparison and only adds setup noise.
        QtipSearchStamp::Unstamped,
        QtipSearchDetail::Unknown,
        QtipCodebook::Gaussian,
        // This bench synthesises a `[65536, 2]` F32 table above; that is the
        // K=4/V=2/L=16 geometry by construction.
        QtipGeometry::K4V2L16,
    )
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Arm {
    Gemv,
    Dequant,
    Grouped,
}

impl Arm {
    fn name(self) -> &'static str {
        match self {
            Arm::Gemv => "gemv",
            Arm::Dequant => "dequant",
            Arm::Grouped => "grouped",
        }
    }
    /// Path selection uses only the PER-CALL env vars. The token-cap var is a
    /// `LazyLock` read once per process, so it is set at startup and never
    /// touched again (moving it mid-process is a silent no-op).
    fn apply(self) {
        match self {
            Arm::Gemv => {
                std::env::remove_var("ARC_NO_QTIP_ONDEVICE_MOE");
                std::env::remove_var("ARC_NO_QTIP_GROUPED_MOE");
            }
            Arm::Dequant => {
                std::env::set_var("ARC_NO_QTIP_ONDEVICE_MOE", "1");
                std::env::set_var("ARC_NO_QTIP_GROUPED_MOE", "1");
            }
            Arm::Grouped => {
                std::env::set_var("ARC_NO_QTIP_ONDEVICE_MOE", "1");
                std::env::remove_var("ARC_NO_QTIP_GROUPED_MOE");
            }
        }
    }
}

fn time_arm(
    layer: &QtipLayer,
    a: &Tensor,
    idx: &Tensor,
    arm: Arm,
    iters: usize,
    dev: &Device,
) -> Result<(f64, Tensor)> {
    arm.apply();
    // Warmup + the output we will compare.
    let out = layer.gather_forward(a, idx)?;
    dev.synchronize()?;
    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = layer.gather_forward(a, idx)?;
    }
    dev.synchronize()?;
    let ms = t0.elapsed().as_secs_f64() * 1e3 / iters as f64;
    Ok((ms, out))
}

/// max |x-y| / (|y| + eps) and cosine similarity, over the whole tensor.
fn compare(x: &Tensor, y: &Tensor) -> Result<(f32, f32)> {
    let x = x.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    let y = y.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    if x.len() != y.len() {
        candle_core::bail!("compare: length mismatch {} vs {}", x.len(), y.len());
    }
    let mut max_rel = 0f32;
    let (mut dot, mut nx, mut ny) = (0f64, 0f64, 0f64);
    for i in 0..x.len() {
        let rel = (x[i] - y[i]).abs() / (y[i].abs() + 1e-3);
        if rel > max_rel {
            max_rel = rel;
        }
        dot += x[i] as f64 * y[i] as f64;
        nx += x[i] as f64 * x[i] as f64;
        ny += y[i] as f64 * y[i] as f64;
    }
    let cos = if nx > 0.0 && ny > 0.0 {
        (dot / (nx.sqrt() * ny.sqrt())) as f32
    } else {
        0.0
    };
    Ok((max_rel, cos))
}

fn main() -> Result<()> {
    // Read ONCE per process, before any gather_forward.
    std::env::set_var("ARC_QTIP_ONDEVICE_MOE_MAX_TOKENS", "1000000");

    let dev = match Device::new_cuda(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("FATAL_ENV: no CUDA device: {e}");
            std::process::exit(2);
        }
    };

    let ns: Vec<usize> = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "128,512,1024,2048".to_string())
        .split(',')
        .map(|s| s.trim().parse::<usize>().unwrap())
        .collect();
    let iters: usize = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "5".to_string())
        .parse()
        .unwrap();

    // Utilization mode: pin ONE arm and loop it for `secs` seconds so an
    // external `nvidia-smi dmon` sample is attributable to a single kernel.
    // Without this the three arms interleave and any SM/MEMCTRL number is a
    // blend of all of them.
    if let Ok(arm_name) = std::env::var("ARC_BENCH_PIN_ARM") {
        let arm = match arm_name.as_str() {
            "gemv" => Arm::Gemv,
            "dequant" => Arm::Dequant,
            "grouped" => Arm::Grouped,
            other => {
                eprintln!("FATAL_ENV: ARC_BENCH_PIN_ARM={other:?} (gemv|dequant|grouped)");
                std::process::exit(2);
            }
        };
        let secs: f64 = std::env::var("ARC_BENCH_SECS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(25.0);
        let n = ns[0];
        let (in_features, n_rows) = (4096usize, 2048usize);
        let mut rng = Rng(0xA5C_1234_5678_9ABC);
        let layer = build_layer(&dev, in_features, n_rows, &mut rng)?;
        let idx_v = routing(&mut rng, n);
        let idx = Tensor::from_vec(idx_v, (n, TOPK), &dev)?;
        let mut av = Vec::with_capacity(n * TOPK * in_features);
        for _ in 0..n * TOPK * in_features {
            av.push(rng.f32_unit() - 0.5);
        }
        let a = Tensor::from_vec(av, (n, TOPK, in_features), &dev)?.to_dtype(DType::BF16)?;
        arm.apply();
        let _ = layer.gather_forward(&a, &idx)?;
        dev.synchronize()?;
        println!("PIN_ARM={} N={n} shape=({in_features},{n_rows})", arm.name());
        println!("LOOP_START");
        let t0 = Instant::now();
        let mut reps = 0usize;
        while t0.elapsed().as_secs_f64() < secs {
            let _ = layer.gather_forward(&a, &idx)?;
            reps += 1;
        }
        dev.synchronize()?;
        // Engagement counter goes to a FILE, never only to stdout.
        let ms = t0.elapsed().as_secs_f64() * 1e3 / reps as f64;
        std::fs::write(
            format!("/root/arc-wt/lutgemm/pin_{}.count", arm.name()),
            format!("arm={} N={n} reps={reps} ms_per_call={ms:.4}\n", arm.name()),
        )
        .ok();
        println!("LOOP_END reps={reps} ms_per_call={ms:.4}");
        return Ok(());
    }

    // V4-Flash's two MoE expert shapes.
    let shapes: [(usize, usize); 2] = [(4096, 2048), (2048, 4096)];
    for (in_features, _) in shapes {
        if !in_features.is_multiple_of(QTIP_GROUPED_TILE_K) {
            eprintln!("FATAL_ENV: in_features {in_features} not a multiple of grouped tile K");
            std::process::exit(2);
        }
    }

    println!("=== QTIP LUT-rung MoE prefill: gemv vs dequant-materialize vs grouped GEMM ===");
    println!("E={EXPERTS} topk={TOPK} iters={iters} shapes={shapes:?}  (random packed bytes: decode cost is content-independent)");
    println!();

    let mut rng = Rng(0xA5C_1234_5678_9ABC);
    let mut engaged = 0usize;

    for (in_features, n_rows) in shapes {
        println!("--- shape in={in_features} rows={n_rows} ---");
        let layer = build_layer(&dev, in_features, n_rows, &mut rng)?;
        println!(
            "{:>6}  {:>7}  {:>11}  {:>11}  {:>11}   {:>9}  {:>8}",
            "N", "experts", "gemv ms", "dequant ms", "grouped ms", "speedup", "cos"
        );
        for &n in &ns {
            let idx_v = routing(&mut rng, n);
            let ndist = distinct_experts(&idx_v);
            let idx = Tensor::from_vec(idx_v, (n, TOPK), &dev)?;

            let mut av = Vec::with_capacity(n * TOPK * in_features);
            for _ in 0..n * TOPK * in_features {
                av.push(rng.f32_unit() - 0.5);
            }
            let a = Tensor::from_vec(av, (n, TOPK, in_features), &dev)?.to_dtype(DType::BF16)?;

            let (t_gemv, _o_gemv) = time_arm(&layer, &a, &idx, Arm::Gemv, iters, &dev)?;
            let (t_deq, o_deq) = time_arm(&layer, &a, &idx, Arm::Dequant, iters, &dev)?;
            let (t_grp, o_grp) = time_arm(&layer, &a, &idx, Arm::Grouped, iters, &dev)?;

            let (max_rel, cos) = compare(&o_grp, &o_deq)?;

            // D12: the arms must be DISTINCT code paths. If grouped and
            // dequant are within 2%, the switch did not take and this row is
            // one kernel measured twice.
            if (t_grp - t_deq).abs() / t_deq < 0.02 {
                eprintln!(
                    "FATAL_ENV: grouped ({t_grp:.3} ms) indistinguishable from dequant \
                     ({t_deq:.3} ms) at N={n} — path switch did not take"
                );
                std::process::exit(2);
            }
            // Correctness: grouped must reproduce the materialize path.
            // NaN must FAIL, not pass — hence the explicit `is_nan` arm rather
            // than a bare `cos <= 0.99`, which NaN silently satisfies as false.
            if cos.is_nan() || cos <= 0.99 {
                eprintln!(
                    "FATAL: grouped output does not match dequant at N={n} \
                     (cos={cos:.6}, max_rel={max_rel:.4}) — kernel is WRONG, not fast"
                );
                std::process::exit(1);
            }
            engaged += 1;

            println!(
                "{n:>6}  {ndist:>7}  {t_gemv:>11.3}  {t_deq:>11.3}  {t_grp:>11.3}   {:>8.2}x  {cos:>8.5}",
                t_deq.min(t_gemv) / t_grp
            );
        }
        println!();
    }

    if engaged == 0 {
        eprintln!("FATAL_ENV: no rows measured — 'no failures' is not 'no results'");
        std::process::exit(2);
    }
    println!("ENGAGED_ROWS={engaged}");
    println!(
        "NOTE: per-prefill-token model cost = (sum of the two shapes' ms) * {MOE_LAYERS} / N."
    );
    Ok(())
}
