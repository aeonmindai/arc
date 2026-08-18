// Trellis grouped-GEMM batch curve — the harness behind the "~1,006 aggregate
// tok/s" headline, repaired so it can settle the qtip2b re-bake decision.
//
// ## The question (wave29-BD, Gate 2)
//
// The group-by-expert kernel (`kernels/qtip/qtip_grouped_gemm.cu`) amortizes
// weight reads across tokens: each woken expert's bytes are staged once per
// m-tile instead of once per (token, slot) pair. But it starts from a MEASURED
// per-byte deficit against the fused gather-GEMV it replaces (s4, H200, E=64:
// 403.8 vs 228.9 us at the same 48 distinct experts — 1.76x). So there is a
// crossover batch below which grouped LOSES, and the whole re-bake decision
// turns on where it is.
//
// BD's predictions, stated in advance so this run can falsify them:
//   * crossover at B ~= 52 +/- 15;
//   * at B=64 grouped ~= 1.1x the GEMV; at B=128, ~= 1.8x;
//   * REFUTED if grouped never overtakes the GEMV below B=68 (the
//     memory-feasible batch ceiling until the `xs` compressor cache shrinks).
//
// ## What changed vs the version that produced the s4 numbers
//
// 1. **It runs at all.** The fixture called `QtipMode::Greedy`, which every
//    quantizer door refuses outside this crate's own `cfg(test)` binaries
//    (DOCTRINE D4). An example is not one, so this died at the first fixture.
//    It now goes through the PRODUCTION door, `quantize_with_mode(Viterbi)` —
//    Viterbi + hadamard rotation, exactly what a served qtip2b artifact
//    carries. The trellis search decides WHICH symbols get packed, never how
//    many bytes they occupy nor how many ops the decode costs, so the measured
//    bandwidth is unchanged; only bake time moves. Rotation is now ON because
//    BOTH paths apply the identical `rotate_x_cuda` pass to the same (pairs, K)
//    activation matrix (`bitshift.rs` gather_forward_cuda_ondevice vs
//    gather_forward_batched), so it cancels in the comparison — and BD flagged
//    the s4 rows' missing rotation as a possible bias.
//
// 2. **E = 64 -> 256.** V4-Flash has 256 experts. At E=64 expert coverage
//    saturates by B~=11, so the "flat 532 us from B=16 to B=64" that the
//    1,006 tok/s headline rests on is the fixture running out of experts, not
//    the amortization curve V4 would actually ride. `assert_coverage_headroom`
//    now refuses the old fixture outright.
//
// 3. **The GEMV arm is measured at EVERY B, not only B<=8.** BD had to
//    *extrapolate* the GEMV column from a single B=8 point because the old
//    harness let the default 8-token cap silence it. The cap is raised once at
//    startup so both paths are measured on the same grid, and the crossover is
//    read off directly instead of projected.
//
// 4. **Routing is real top-6 sampling, not a sliding window.** The old index
//    generator woke `min(B*topk, E)` distinct experts — the maximally-spread
//    worst case, which is not what `E(B) = 256*(1-(1-6/256)^B)` describes and
//    which saturates at B=43 instead of ~B=256. Each token now draws 6
//    DISTINCT experts uniformly, and `assert_routing_matches_model` fails the
//    run if the realized coverage departs from E(B).
//
// 5. **It refuses to report a number it did not measure** — see the D12 guards
//    below. Chief among them: if the two paths' timings are indistinguishable,
//    the mode switch did not take and the whole table is one kernel measured
//    twice.
//
// ## The mode-switch trap (read before editing the env plumbing)
//
// `ARC_QTIP_ONDEVICE_MOE_MAX_TOKENS` is read exactly ONCE per process, behind
// a `LazyLock` in `qtip/gather_policy.rs`. The pre-existing habit of
// `set_var`/`remove_var`-ing it between modes became a silent no-op when that
// memoization landed: every later "gemv" row would have re-measured the
// grouped kernel. So the cap is set once, before the first `gather_forward`,
// and never touched again; path selection uses `ARC_NO_QTIP_ONDEVICE_MOE`,
// which `gather_forward` reads per call.
//
// On the GPU box:
//   cargo run --release -p mistralrs-quant --example qtip_grouped_curve \
//       --features cuda -- --experts 256 --batches 1,8,16,32,52,64,128
// Flags: --experts N | --iters N | --batches a,b,c | --small (tiny shapes).
// Without CUDA it runs the full guard set plus a CPU-reference timing, so the
// harness itself is verifiable anywhere; no kernel is measured there.

use candle_core::{DType, Device, Result, Tensor};
use mistralrs_quant::{
    grouped_launch_counts, qtip_expected_distinct_experts, set_grouped_variant, Qtip2bLayer,
    QtipMode, QuantMethod, QTIP_GATHER_GEMV_MAX_PAIRS, QTIP_GROUPED_TILE_K, QTIP_GROUPED_TILE_M,
    QTIP_GROUPED_VARIANT_BASELINE, QTIP_GROUPED_VARIANT_TUNED, QTIP_ONDEVICE_MOE_MAX_TOKENS_ENV,
};
use std::sync::Arc;
use std::time::Instant;

/// H200 HBM3e peak, bytes/s. Only ever used to render a "% of peak" column —
/// never as a ceiling claim (CEILINGS.json: a % of peak is a verdict on our
/// kernel, not on the format).
const H200_PEAK: f64 = 4.8e12;
/// H200 L2. The GB/s columns are only HBM numbers if the packed working set
/// cannot sit in L2 between iterations.
const H200_L2_BYTES: f64 = 60e6;
/// V4-Flash routes each token to **6** of 256 experts (`v4_audit.md:263-264`).
/// The widely-quoted E(B) amortization table is top-8 and overstates coverage
/// at every batch; every number here uses 6.
const TOPK: usize = 6;
/// V4-Flash expert count.
const DEFAULT_EXPERTS: usize = 256;
/// MoE layer count used ONLY for the labeled per-step extrapolation lines
/// (~40 of V4-Flash's 43 layers are MoE; parity/traffic numbers don't use it).
const MOE_LAYERS: f64 = 40.0;
/// Independent routing draws cycled across timing iterations, so a hot expert
/// window cannot sit in L2 across the measurement.
const ROUTING_WINDOWS: usize = 32;
/// Fixed so two runs on the same box are comparable.
const ROUTING_SEED: u64 = 0x5EED_1006_C0DE_2B00;

// --- BD's predictions, pinned here so the verdict is checkable ------------
const BD_CROSSOVER_LO: usize = 37; // 52 - 15
const BD_CROSSOVER_HI: usize = 67; // 52 + 15
/// Grouped must overtake the GEMV at or below this batch, or BD's amortization
/// claim is REFUTED: above it we cannot schedule anyway (FACTS: memory caps us
/// near B=68 at 2048-token context until the `xs` cache shrinks).
const BD_REFUTED_ABOVE: usize = 68;

// ---------------------------------------------------------------------------
// Deterministic routing that actually reproduces E(B)
// ---------------------------------------------------------------------------

/// SplitMix64. Inlined so the routing model is reproducible without pulling a
/// RNG crate into an example's dependency surface.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

/// One routing draw: `b` tokens, each picking `top_k` **distinct** experts
/// uniformly out of `e`. That is real top-k semantics, and it is what makes
/// `expected_distinct_experts` exact rather than approximate.
///
/// `top_k` is a parameter, not `TOPK`, for one reason: the negative control in
/// [`assert_guards_have_teeth`] draws a deliberately WRONG top-k and requires
/// the routing guard to reject it.
fn routing_window(rng: &mut Rng, b: usize, e: usize, top_k: usize) -> Vec<u32> {
    let mut out = Vec::with_capacity(b * top_k);
    let mut chosen: Vec<u32> = Vec::with_capacity(top_k);
    for _ in 0..b {
        chosen.clear();
        while chosen.len() < top_k {
            let c = rng.below(e) as u32;
            if !chosen.contains(&c) {
                chosen.push(c);
            }
        }
        out.extend_from_slice(&chosen);
    }
    out
}

/// Mean distinct experts woken by `windows` independent draws at `top_k`.
fn mean_distinct(b: usize, e: usize, top_k: usize, windows: usize, seed: u64) -> f64 {
    let mut rng = Rng(seed);
    let mut sum = 0.0f64;
    for _ in 0..windows {
        let idx = routing_window(&mut rng, b, e, top_k);
        sum += expert_counts(&idx, e).iter().filter(|c| **c > 0).count() as f64;
    }
    sum / windows as f64
}

/// Per-expert pair counts for one draw.
fn expert_counts(idx: &[u32], e: usize) -> Vec<usize> {
    let mut counts = vec![0usize; e];
    for &v in idx {
        counts[v as usize] += 1;
    }
    counts
}

// ---------------------------------------------------------------------------
// D12 guards: refuse to report a number this harness did not measure
// ---------------------------------------------------------------------------

/// **Negative control for the guards themselves.** Runs before anything else,
/// on every platform, every invocation.
///
/// DOCTRINE D12 is not "add an assertion", it is "prove the assertion is
/// reachable" — this repo has found 7+ tests that passed while their assertion
/// could not fail, and one whose fixture dimensions made a wrong implementation
/// arithmetically indistinguishable from a right one. Guards written into an
/// `examples/` binary are the worst case for that failure mode: no `cargo test`
/// ever executes them, so a refactor that neuters one is invisible forever.
///
/// So each guard is handed an input it MUST reject and an input it MUST accept.
/// If a guard has gone blind, this fails immediately and loudly instead of
/// letting the run print a number nobody can trust.
fn assert_guards_have_teeth() -> Result<()> {
    // 1. Coverage: the E=64 fixture that produced the s4 "~1,006 tok/s" curve
    //    must be rejected at the top of this sweep, and E=256 accepted.
    let sweep = [1usize, 8, 16, 32, 52, 64, 128];
    if assert_coverage_headroom(64, &sweep).is_ok() {
        candle_core::bail!(
            "GUARD IS BLIND: assert_coverage_headroom accepts E=64 over B={sweep:?}, the exact \
             saturated fixture it exists to reject."
        );
    }
    assert_coverage_headroom(DEFAULT_EXPERTS, &sweep)?;

    // 2. Sweep: a grid that stops below the refutation boundary, and a
    //    non-monotonic one, must both be rejected.
    if assert_sweep_can_decide(&[1, 8, 16, 32]).is_ok() {
        candle_core::bail!("GUARD IS BLIND: assert_sweep_can_decide accepts a grid topping at 32.");
    }
    if assert_sweep_can_decide(&[128, 1, 8, 64]).is_ok() {
        candle_core::bail!("GUARD IS BLIND: assert_sweep_can_decide accepts an unsorted grid.");
    }
    assert_sweep_can_decide(&sweep)?;

    // 3. Routing: THE one the expert-coverage arithmetic rests on. Draw a real
    //    top-8 routing — V3's shape, and the number most of the published E(B)
    //    tables were computed at — and require the top-6 model to reject it. If
    //    this passes, `assert_routing_matches_model` cannot tell V4's routing
    //    from V3's and every byte column in the table is unfalsifiable.
    const PROBE_B: usize = 16;
    const WRONG_TOPK: usize = 8;
    let wrong = mean_distinct(
        PROBE_B,
        DEFAULT_EXPERTS,
        WRONG_TOPK,
        8,
        ROUTING_SEED ^ 0xA5A5,
    );
    if assert_routing_matches_model(PROBE_B, DEFAULT_EXPERTS, wrong).is_ok() {
        candle_core::bail!(
            "GUARD IS BLIND: a top-{WRONG_TOPK} routing at B={PROBE_B} wakes {wrong:.1} distinct \
             experts and assert_routing_matches_model ACCEPTED it against the top-{TOPK} model \
             ({:.1}). The harness cannot tell V4-Flash's top-{TOPK} routing from V3's top-\
             {WRONG_TOPK}, so its traffic model proves nothing.",
            qtip_expected_distinct_experts(PROBE_B, TOPK, DEFAULT_EXPERTS)
        );
    }
    // ...and the routing this harness actually generates must be accepted, or
    // every run would die on a false positive.
    let right = mean_distinct(PROBE_B, DEFAULT_EXPERTS, TOPK, 8, ROUTING_SEED ^ 0x5A5A);
    assert_routing_matches_model(PROBE_B, DEFAULT_EXPERTS, right)?;

    println!(
        "guard self-check OK: top-{WRONG_TOPK} routing ({wrong:.1} experts at B={PROBE_B}) is \
         REJECTED against the top-{TOPK} model ({right:.1} accepted); E=64 rejected, E={DEFAULT_EXPERTS} accepted."
    );
    Ok(())
}

/// The fixture must not run out of experts inside the swept range. At E=64 and
/// top-6, `E(B)` is 99.99% saturated by B=43 — every batch above it moves
/// identical weight traffic, so a flat time curve is the fixture ending, not
/// amortization. This is the exact defect that made the s4 curve unusable for
/// the crossover question.
fn assert_coverage_headroom(e: usize, batches: &[usize]) -> Result<()> {
    let max_b = *batches.iter().max().expect("non-empty batches");
    let cover = qtip_expected_distinct_experts(max_b, TOPK, e);
    let frac = cover / e as f64;
    if frac > 0.98 {
        candle_core::bail!(
            "fixture too small: E={e} with top-{TOPK} is {:.2}% expert-saturated at the top of \
             the sweep (B={max_b}, E(B)={cover:.1}/{e}). Above ~98% every batch moves the same \
             weight bytes, so a flat curve measures the FIXTURE running out of experts, not the \
             grouped kernel amortizing. Raise --experts (V4-Flash is {DEFAULT_EXPERTS}) or lower \
             the top of --batches.",
            frac * 100.0
        );
    }
    if TOPK >= e {
        candle_core::bail!(
            "degenerate routing: top-{TOPK} of E={e} wakes every expert at B=1, so there is no \
             amortization left to measure."
        );
    }
    Ok(())
}

/// The verdict this harness exists to produce is a crossover location. It is
/// only meaningful if the sweep brackets BD's refutation boundary — otherwise
/// "no crossover observed" is a statement about the grid, not about the kernel.
fn assert_sweep_can_decide(batches: &[usize]) -> Result<()> {
    if batches.len() < 4 {
        candle_core::bail!("--batches needs >= 4 points to locate a crossover, got {batches:?}");
    }
    if batches.windows(2).any(|w| w[0] >= w[1]) {
        candle_core::bail!("--batches must be strictly increasing, got {batches:?}");
    }
    let max_b = *batches.iter().max().expect("non-empty batches");
    if max_b < BD_REFUTED_ABOVE {
        candle_core::bail!(
            "--batches tops out at {max_b}, below the B={BD_REFUTED_ABOVE} refutation boundary. \
             A run that stops here cannot distinguish 'grouped never overtakes' from 'we did not \
             sweep far enough' — the exact ambiguity this measurement exists to remove."
        );
    }
    Ok(())
}

/// Config-level reachability, checked on every platform because it decides
/// whether a GPU run could answer the question at all.
fn assert_config_is_measurable(e: usize, batches: &[usize]) -> Result<()> {
    let max_b = *batches.iter().max().expect("non-empty batches");
    let max_pairs = max_b * TOPK;
    if max_pairs > QTIP_GATHER_GEMV_MAX_PAIRS {
        candle_core::bail!(
            "{max_pairs} pairs exceeds the fused GEMV's grid.y limit {QTIP_GATHER_GEMV_MAX_PAIRS}; \
             the GEMV arm cannot be measured at the top of this sweep."
        );
    }
    // The grouped kernel stages each expert once per m-tile, so "each woken
    // expert read once per step" — the model behind the floor column — only
    // holds while the average pairs per woken expert stays within one tile.
    let per_expert = max_pairs as f64 / qtip_expected_distinct_experts(max_b, TOPK, e);
    if per_expert > QTIP_GROUPED_TILE_M as f64 {
        candle_core::bail!(
            "at B={max_b} each woken expert takes {per_expert:.1} pairs, past GROUPED_TILE_M=\
             {QTIP_GROUPED_TILE_M}: the grouped kernel re-stages weights per m-tile beyond that, \
             so the unique-expert floor column would understate its real traffic."
        );
    }
    Ok(())
}

/// CUDA-only: both kernels must be structurally reachable for this fixture, or
/// `gather_forward` silently routes somewhere else and the row is mislabeled.
fn assert_cuda_paths_reachable(k: usize, dtype: DType) -> Result<()> {
    if !k.is_multiple_of(QTIP_GROUPED_TILE_K) {
        candle_core::bail!(
            "K={k} is not a multiple of GROUPED_TILE_K={QTIP_GROUPED_TILE_K}: gather_forward would \
             skip the grouped kernel and fall to the CPU reference, and the 'grouped' rows would \
             be timing candle on the host."
        );
    }
    if !matches!(dtype, DType::BF16 | DType::F16) {
        candle_core::bail!(
            "activation dtype {dtype:?}: the grouped mma.sync pipeline only engages for BF16/F16; \
             anything else falls to the CPU reference."
        );
    }
    Ok(())
}

/// The GB/s columns are HBM numbers only if the packed stack cannot live in L2
/// across iterations. CUDA only — there is no such claim on the CPU path.
fn assert_defeats_l2(e: usize, bytes_per_expert: f64) -> Result<()> {
    let working_set = e as f64 * bytes_per_expert;
    if working_set <= H200_L2_BYTES {
        candle_core::bail!(
            "packed working set {:.1} MB fits in L2 ({:.0} MB): the GB/s columns would report L2 \
             bandwidth, not HBM. Raise --experts or the shape.",
            working_set / 1e6,
            H200_L2_BYTES / 1e6
        );
    }
    Ok(())
}

/// The realized routing must reproduce `E(B) = E*(1-(1-k/E)^B)`. If it does
/// not, every byte count in the table — and the amortization arithmetic the
/// re-bake decision rests on — describes a traffic pattern V4 never generates.
/// Catches a top-k mismatch and the old sliding-window generator alike.
fn assert_routing_matches_model(b: usize, e: usize, measured_distinct: f64) -> Result<()> {
    let expected = qtip_expected_distinct_experts(b, TOPK, e);
    let tol = (0.10 * expected).max(3.0);
    if (measured_distinct - expected).abs() > tol {
        candle_core::bail!(
            "routing does not match the model at B={b}: realized {measured_distinct:.1} distinct \
             experts, E(B)=E*(1-(1-{TOPK}/{e})^{b})={expected:.1} (tolerance {tol:.1}). Either the \
             index generator is not top-{TOPK} sampling, or top-k and E disagree with the \
             constants. Every byte column below would be wrong."
        );
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Path selection
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
enum Path {
    /// Trellis grouped GEMM: tokens sorted by expert on-device, persistent
    /// tensor-core tile loop. Reached by disabling the fused GEMV.
    Grouped,
    /// Fused on-device gather-GEMV: one GEMV per (token, slot) pair, no dedup.
    /// Reached by leaving the GEMV enabled with the token cap raised.
    Gemv,
    /// No CUDA: `gather_forward_cpu`. Neither kernel exists; timings here are
    /// harness validation only.
    CpuRef,
}

impl Path {
    fn label(self) -> &'static str {
        match self {
            Path::Grouped => "grouped",
            Path::Gemv => "gemv   ",
            Path::CpuRef => "cpu-ref",
        }
    }

    /// Per-call env reads only. Deliberately does NOT touch
    /// `ARC_QTIP_ONDEVICE_MOE_MAX_TOKENS` — that one is memoized for the
    /// process lifetime (see the header) and is set once in `main`.
    fn select(self) {
        match self {
            Path::Grouped => std::env::set_var("ARC_NO_QTIP_ONDEVICE_MOE", "1"),
            Path::Gemv | Path::CpuRef => std::env::remove_var("ARC_NO_QTIP_ONDEVICE_MOE"),
        }
        std::env::remove_var("ARC_NO_QTIP_GROUPED_MOE");
    }
}

/// One measured (path, batch) point.
#[derive(Clone, Copy)]
struct Point {
    b: usize,
    /// Mean distinct experts woken across the routing draws.
    distinct: f64,
    /// Mean expert-tiles the grouped kernel stages, `sum_e ceil(count_e/16)`;
    /// equals `pairs` for the GEMV, which has no dedup.
    issued_units: f64,
    secs: f64,
}

// ---------------------------------------------------------------------------
// Measurement
// ---------------------------------------------------------------------------

struct Shape {
    name: &'static str,
    n: usize,
    k: usize,
    /// gate+up share a shape (2 calls per layer), down is 1.
    calls_per_layer: f64,
}

/// Packed bytes per expert: 2-bit codes (n*k/4) + F32 row scales.
fn bytes_per_expert(s: &Shape) -> f64 {
    s.n as f64 * s.k as f64 * 0.25 + s.n as f64 * 4.0
}

/// Build the routing draws once per batch size and share them across paths, so
/// grouped and GEMV are compared on byte-identical traffic.
fn build_routing(dev: &Device, b: usize, e: usize) -> Result<(Vec<Tensor>, f64, f64, f64)> {
    let mut rng = Rng(ROUTING_SEED ^ (b as u64).wrapping_mul(0x9E37_79B9));
    let mut sets = Vec::with_capacity(ROUTING_WINDOWS);
    let (mut sum_distinct, mut sum_tiles) = (0.0f64, 0.0f64);
    for _ in 0..ROUTING_WINDOWS {
        let idx = routing_window(&mut rng, b, e, TOPK);
        let counts = expert_counts(&idx, e);
        sum_distinct += counts.iter().filter(|c| **c > 0).count() as f64;
        sum_tiles += counts
            .iter()
            .map(|c| c.div_ceil(QTIP_GROUPED_TILE_M) as f64)
            .sum::<f64>();
        sets.push(Tensor::from_vec(idx, (b, TOPK), dev)?);
    }
    let n = ROUTING_WINDOWS as f64;
    Ok((sets, sum_distinct / n, sum_tiles / n, (b * TOPK) as f64))
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
    path: Path,
    cap_at_startup: &str,
) -> Result<Vec<Point>> {
    let bpe = bytes_per_expert(s);
    let mut out = Vec::new();
    path.select();
    for &b in batches {
        assert_cap_unchanged(cap_at_startup)?;
        let (idx_sets, mean_distinct, mean_tiles, pairs) = build_routing(dev, b, e)?;
        assert_routing_matches_model(b, e, mean_distinct)?;
        let issued_units = if path == Path::Gemv {
            pairs
        } else {
            mean_tiles
        };

        let a = Tensor::randn(0f32, 1f32, (b, TOPK, s.k), dev)?.to_dtype(act_dtype)?;
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

        let floor_bytes = bpe * mean_distinct;
        let issued_bytes = bpe * issued_units;
        println!(
            "  [{}] B={:<4} pairs={:<5} E(B)={:<6.1} | {:>9.1} us/call | floor {:>6.0} GB/s \
             ({:>4.1}% peak) | issued {:>6.0} GB/s",
            path.label(),
            b,
            pairs as usize,
            mean_distinct,
            secs * 1e6,
            floor_bytes / secs / 1e9,
            floor_bytes / secs / H200_PEAK * 100.0,
            issued_bytes / secs / 1e9,
        );
        out.push(Point {
            b,
            distinct: mean_distinct,
            issued_units,
            secs,
        });
    }
    Ok(out)
}

/// The cap is memoized on first use; if anything moved it mid-run the later
/// rows would silently be the other kernel. Cheap, and it closes the exact
/// hole that made the old mode switch a no-op.
fn assert_cap_unchanged(expected: &str) -> Result<()> {
    let now = std::env::var(QTIP_ONDEVICE_MOE_MAX_TOKENS_ENV).unwrap_or_default();
    if now != expected {
        candle_core::bail!(
            "{QTIP_ONDEVICE_MOE_MAX_TOKENS_ENV} changed mid-run ('{expected}' -> '{now}'). It is \
             read once per process behind a LazyLock, so the change did NOT take effect in the \
             dispatch — every row after this point would be mislabeled."
        );
    }
    Ok(())
}

/// If the two paths are indistinguishable at every batch, the mode switch did
/// not take and the table is one kernel measured twice. That is strictly worse
/// than not running (DOCTRINE D12), so it is an error, not a warning.
fn assert_paths_are_distinguishable(grouped: &[Point], gemv: &[Point]) -> Result<()> {
    let indistinguishable = grouped
        .iter()
        .zip(gemv.iter())
        .all(|(g, v)| (g.secs - v.secs).abs() / g.secs < 0.03);
    if indistinguishable && !grouped.is_empty() {
        candle_core::bail!(
            "grouped and gemv timings agree within 3% at EVERY batch: the path switch did not \
             take, and both columns are the same kernel. Check that ARC_NO_QTIP_ONDEVICE_MOE is \
             honored and that {QTIP_ONDEVICE_MOE_MAX_TOKENS_ENV} was set before the first \
             gather_forward."
        );
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Verdict
// ---------------------------------------------------------------------------

/// The per-byte handicap amortization has to pay off first. BD measured 1.76x
/// at E=64/B=8 on H200 (403.8 vs 228.9 us at the same 48 distinct experts) and
/// flagged it as the single number the re-bake thesis is most sensitive to —
/// it is a tuning deficit (`qtip_grouped_gemm.cu:63-79` lists the unswept
/// axes), not a structural one, so it is worth tracking per batch.
fn report_shape_efficiency(s: &Shape, grouped: &[Point], gemv: &[Point]) {
    println!(
        "  -- per-unit cost ({}): grouped stages one expert per m-tile, gemv reads one per pair --",
        s.name
    );
    println!("     B | E(B)   | grouped us/unit | gemv us/unit | handicap");
    for (g, v) in grouped.iter().zip(gemv.iter()) {
        println!(
            "   {:>3} | {:>6.1} | {:>15.2} | {:>12.2} | {:>7.2}x",
            g.b,
            g.distinct,
            g.secs / g.issued_units * 1e6,
            v.secs / v.issued_units * 1e6,
            (g.secs / g.issued_units) / (v.secs / v.issued_units)
        );
    }
}

/// Per-step MoE-GEMM cost summed over the shapes, in seconds.
fn step_secs(per_shape: &[Vec<Point>], shapes: &[Shape], i: usize) -> f64 {
    per_shape
        .iter()
        .zip(shapes.iter())
        .map(|(pts, s)| pts[i].secs * s.calls_per_layer * MOE_LAYERS)
        .sum()
}

fn report_verdict(
    shapes: &[Shape],
    grouped: &[Vec<Point>],
    gemv: &[Vec<Point>],
    batches: &[usize],
) {
    println!(
        "\n-- grouped vs gemv, MoE-GEMM floor only, {MOE_LAYERS} MoE layers x \
         {:.0} calls (LABELED EXTRAPOLATION: no attention, KV, sampling or launch overhead) --",
        shapes.iter().map(|s| s.calls_per_layer).sum::<f64>()
    );
    println!(
        "    B | grouped step ms | gemv step ms | grouped tok/s | gemv tok/s | grouped speedup"
    );
    let mut crossover: Option<usize> = None;
    let mut speedups: Vec<(usize, f64)> = Vec::new();
    for (i, &b) in batches.iter().enumerate() {
        let gstep = step_secs(grouped, shapes, i);
        let vstep = step_secs(gemv, shapes, i);
        let speedup = vstep / gstep;
        speedups.push((b, speedup));
        if speedup > 1.0 && crossover.is_none() {
            crossover = Some(b);
        }
        println!(
            "  {:>3} | {:>15.2} | {:>12.2} | {:>13.0} | {:>10.0} | {:>14.2}x",
            b,
            gstep * 1e3,
            vstep * 1e3,
            b as f64 / gstep,
            b as f64 / vstep,
            speedup
        );
    }

    println!("\n-- wave29-BD Gate 2 verdict --");
    let at = |want: usize| speedups.iter().find(|(b, _)| *b == want).map(|(_, s)| *s);
    match crossover {
        Some(b) if b <= BD_REFUTED_ABOVE => {
            let bracket = if (BD_CROSSOVER_LO..=BD_CROSSOVER_HI).contains(&b) {
                "INSIDE BD's predicted 52 +/- 15"
            } else {
                "OUTSIDE BD's predicted 52 +/- 15"
            };
            println!("  CONFIRMED: grouped overtakes the GEMV at B={b} ({bracket}).");
        }
        Some(b) => println!(
            "  REFUTED: grouped first overtakes the GEMV only at B={b}, above the \
             B={BD_REFUTED_ABOVE} we can schedule. BD's fallback recommendation applies — do not \
             re-bake; spend the GPU hour on qtip_grouped_gemm.cu's unswept tiling axes."
        ),
        None => println!(
            "  REFUTED: grouped never overtakes the GEMV anywhere in {batches:?}. Same \
             recommendation: do not re-bake, tune the grouped kernel instead."
        ),
    }
    for (b, predicted) in [(64usize, 1.1f64), (128, 1.8)] {
        if let Some(measured) = at(b) {
            println!(
                "  B={b:<4} predicted {predicted:.1}x, measured {measured:.2}x  ({})",
                if (measured - predicted).abs() / predicted <= 0.25 {
                    "within 25%"
                } else {
                    "MISSED by >25%"
                }
            );
        }
    }
}

// ---------------------------------------------------------------------------
// `--ab`: grouped kernel variant A/B with an INTERLEAVED null control
// ---------------------------------------------------------------------------
//
// ONE BINARY, ONE VARIABLE. Both kernel variants are compiled into this
// binary; `set_grouped_variant` picks per launch. The arms are separated by a
// RUNTIME fact — `grouped_launch_counts()` — not by which build ran.
//
// A null control gives the floor for THAT RUN, not "the floor": measured
// across four runs on this fleet it moved ~4x (0.67 / 2.45 / 2.54 / 1.30 %).
// So the sequence is `U F U N` repeated, every arm is differenced against its
// TEMPORAL NEIGHBOUR, and the treatment must beat the null's spread by >=2x.
// `N` is the baseline kernel run in the treatment's slot: any difference it
// shows is pure instrument.

/// Bar for calling the tuned variant real: the mean paired effect must exceed
/// the largest null effect this same run produced, by this factor.
const AB_MARGIN_OVER_NULL: f64 = 2.0;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Arm {
    /// Baseline kernel in the baseline slot.
    U,
    /// Tuned kernel in the treatment slot.
    F,
    /// Baseline kernel in the TREATMENT slot — the run's own floor.
    N,
}

impl Arm {
    fn variant(self) -> i32 {
        match self {
            Arm::F => QTIP_GROUPED_VARIANT_TUNED,
            Arm::U | Arm::N => QTIP_GROUPED_VARIANT_BASELINE,
        }
    }
    fn tag(self) -> &'static str {
        match self {
            Arm::U => "U",
            Arm::F => "F",
            Arm::N => "N",
        }
    }
}

/// Time one arm, and PROVE from the runtime counters that the kernel we asked
/// for is the kernel that ran.
///
/// Three states, not two: `Ok(secs)` measured, `Err` for a real failure, and
/// an explicit bail when the counters cannot answer (no launches at all means
/// `gather_forward` went somewhere else entirely and the timing is of some
/// other code path).
#[allow(clippy::too_many_arguments)]
fn time_arm(
    dev: &Device,
    layer: &Arc<dyn QuantMethod>,
    arm: Arm,
    a: &Tensor,
    idx_sets: &[Tensor],
    iters: usize,
) -> Result<f64> {
    set_grouped_variant(arm.variant());
    let want = arm.variant() as usize;
    let other = 1 - want;

    for r in 0..4usize {
        let _ = layer.gather_forward(a, &idx_sets[r % idx_sets.len()])?;
    }
    dev.synchronize()?;

    let before = grouped_launch_counts();
    let t = Instant::now();
    for i in 0..iters {
        let _ = layer.gather_forward(a, &idx_sets[i % idx_sets.len()])?;
    }
    dev.synchronize()?;
    let secs = t.elapsed().as_secs_f64() / iters as f64;
    let after = grouped_launch_counts();

    let ran = after[want] - before[want];
    let stray = after[other] - before[other];
    if ran != iters as u64 {
        candle_core::bail!(
            "arm {}: asked for grouped variant {want} and the runtime counter advanced by {ran}, \
             not {iters}. The variant switch did not take (or gather_forward left the grouped \
             kernel), so this timing is NOT the kernel it is labeled with.",
            arm.tag()
        );
    }
    if stray != 0 {
        candle_core::bail!(
            "arm {}: {stray} launches of grouped variant {other} happened inside this arm's \
             timing window. The two arms are not separated.",
            arm.tag()
        );
    }
    Ok(secs)
}

/// The tuned variant is a pure re-routing of the same decode — same weights,
/// same k-order, same f32 accumulators — so its output must be BIT-IDENTICAL,
/// not merely close. Checked before any timing.
///
/// D12: the check is handed a case it must reject (a different routing draw),
/// so "identical" cannot be an artifact of comparing a tensor with itself.
fn assert_variants_bit_identical(
    dev: &Device,
    layer: &Arc<dyn QuantMethod>,
    a: &Tensor,
    idx_sets: &[Tensor],
) -> Result<()> {
    let bytes = |t: &Tensor| -> Result<Vec<u8>> {
        let f = t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        Ok(f.iter().flat_map(|v| v.to_bits().to_le_bytes()).collect())
    };

    set_grouped_variant(QTIP_GROUPED_VARIANT_BASELINE);
    let base = bytes(&layer.gather_forward(a, &idx_sets[0])?)?;
    set_grouped_variant(QTIP_GROUPED_VARIANT_TUNED);
    let tuned = bytes(&layer.gather_forward(a, &idx_sets[0])?)?;
    dev.synchronize()?;

    // Negative control: a different routing draw MUST produce different bytes,
    // or this comparison could not have detected a wrong kernel either.
    set_grouped_variant(QTIP_GROUPED_VARIANT_BASELINE);
    let elsewhere = bytes(&layer.gather_forward(a, &idx_sets[1 % idx_sets.len()])?)?;
    if elsewhere == base {
        candle_core::bail!(
            "PARITY CHECK IS BLIND: two DIFFERENT routing draws produced byte-identical output, \
             so byte-equality cannot distinguish the tuned kernel from a wrong one."
        );
    }

    let diff = base
        .iter()
        .zip(tuned.iter())
        .filter(|(x, y)| x != y)
        .count();
    if diff != 0 {
        candle_core::bail!(
            "tuned grouped variant is NOT bit-identical to the baseline: {diff}/{} bytes differ. \
             The pair-reversal is a pure bit permutation, so ANY difference is a bug, not a \
             tolerance question.",
            base.len()
        );
    }
    println!(
        "parity OK: tuned == baseline byte-for-byte over {} output bytes; a different routing \
         draw differs (control has teeth).",
        base.len()
    );
    Ok(())
}

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

fn run_ab(dev: &Device, cfg: &Cfg, act_dtype: DType, b: usize, rounds: usize) -> Result<()> {
    let shapes = ab_shapes(cfg.small);
    assert_coverage_headroom(cfg.experts, &[b])?;

    for s in &shapes {
        assert_cuda_paths_reachable(s.k, act_dtype)?;
        assert_defeats_l2(cfg.experts, bytes_per_expert(s))?;

        let w =
            Tensor::randn(0f32, 0.02f32, (cfg.experts, s.n, s.k), dev)?.to_dtype(DType::BF16)?;
        let layer = Qtip2bLayer::quantize_with_mode(&w, None, dev, QtipMode::Viterbi)?;
        drop(w);

        let (idx_sets, mean_distinct, mean_tiles, pairs) = build_routing(dev, b, cfg.experts)?;
        assert_routing_matches_model(b, cfg.experts, mean_distinct)?;
        let a = Tensor::randn(0f32, 1f32, (b, TOPK, s.k), dev)?.to_dtype(act_dtype)?;

        println!(
            "\n[{}] N={} K={} E={} B={b} pairs={} E(B)={mean_distinct:.1} m-tiles={mean_tiles:.1}",
            s.name, s.n, s.k, cfg.experts, pairs as usize
        );
        assert_variants_bit_identical(dev, &layer, &a, &idx_sets)?;

        // U F U N, repeated. Every treatment slot is differenced against the
        // U that immediately precedes it.
        let seq = [Arm::U, Arm::F, Arm::U, Arm::N];
        let mut treat: Vec<f64> = Vec::new();
        let mut null: Vec<f64> = Vec::new();
        let mut last_u: Option<f64> = None;

        for round in 0..rounds {
            for arm in seq {
                let secs = time_arm(dev, &layer, arm, &a, &idx_sets, cfg.iters)?;
                let unit = secs / mean_tiles * 1e6;
                let rel = last_u.map(|u| (u - secs) / u * 100.0);
                println!(
                    "  r{round} [{}] {:>9.1} us/call | {:>6.2} us/m-tile{}",
                    arm.tag(),
                    secs * 1e6,
                    unit,
                    match rel {
                        Some(p) if arm != Arm::U => format!(" | vs prev U: {p:+.2}%"),
                        _ => String::new(),
                    }
                );
                match arm {
                    Arm::U => last_u = Some(secs),
                    Arm::F => {
                        if let Some(u) = last_u {
                            treat.push((u - secs) / u * 100.0);
                        }
                    }
                    Arm::N => {
                        if let Some(u) = last_u {
                            null.push((u - secs) / u * 100.0);
                        }
                    }
                }
            }
        }

        if treat.is_empty() || null.is_empty() {
            candle_core::bail!("--ab produced no paired differences; raise --rounds");
        }
        let t_mean = mean(&treat);
        let t_spread = treat.iter().cloned().fold(f64::MIN, f64::max)
            - treat.iter().cloned().fold(f64::MAX, f64::min);
        let n_worst = null.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
        println!(
            "  -- {} paired diffs: TREATMENT mean {:+.2}% (spread {:.2}pp), NULL worst |{:.2}|% --",
            s.name, t_mean, t_spread, n_worst
        );
        if t_mean > n_worst * AB_MARGIN_OVER_NULL {
            println!(
                "  VERDICT [{}]: tuned is FASTER by {:+.2}% — clears this run's own floor \
                 ({:.2}%) by {:.1}x.",
                s.name,
                t_mean,
                n_worst,
                t_mean / n_worst.max(1e-9)
            );
        } else {
            println!(
                "  VERDICT [{}]: {:+.2}% does NOT clear this run's null floor ({:.2}%) by the \
                 required {AB_MARGIN_OVER_NULL}x. Report it as a scoping result with the next \
                 lever named — never as a verdict on the kernel.",
                s.name, t_mean, n_worst
            );
        }
    }
    set_grouped_variant(QTIP_GROUPED_VARIANT_BASELINE);
    Ok(())
}

// ---------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------

struct Cfg {
    experts: usize,
    iters: usize,
    batches: Vec<usize>,
    small: bool,
    /// `Some(B)` runs the kernel-variant A/B at batch B instead of the curve.
    ab: Option<usize>,
    /// `U F U N` repetitions in `--ab`.
    rounds: usize,
}

fn parse_args() -> Result<Cfg> {
    let mut cfg = Cfg {
        experts: DEFAULT_EXPERTS,
        iters: 200,
        batches: vec![1, 8, 16, 32, 52, 64, 128],
        small: false,
        ab: None,
        rounds: 3,
    };
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        let need = |i: usize| -> Result<String> {
            args.get(i + 1)
                .cloned()
                .ok_or_else(|| candle_core::Error::msg(format!("{} needs a value", args[i])))
        };
        match args[i].as_str() {
            "--experts" => {
                cfg.experts = need(i)?.parse().map_err(candle_core::Error::wrap)?;
                i += 2;
            }
            "--iters" => {
                cfg.iters = need(i)?.parse().map_err(candle_core::Error::wrap)?;
                i += 2;
            }
            "--batches" => {
                cfg.batches = need(i)?
                    .split(',')
                    .map(|t| t.trim().parse::<usize>().map_err(candle_core::Error::wrap))
                    .collect::<Result<Vec<_>>>()?;
                i += 2;
            }
            "--small" => {
                cfg.small = true;
                i += 1;
            }
            "--ab" => {
                cfg.ab = Some(need(i)?.parse().map_err(candle_core::Error::wrap)?);
                i += 2;
            }
            "--rounds" => {
                cfg.rounds = need(i)?.parse().map_err(candle_core::Error::wrap)?;
                i += 2;
            }
            other => candle_core::bail!(
                "unknown flag {other}; accepted: --experts N --iters N --batches a,b,c --small \
                 --ab B --rounds N"
            ),
        }
    }
    Ok(cfg)
}

/// V4-Flash's two MoE GEMM shapes (or tiny stand-ins for the CPU guard run).
fn ab_shapes(small: bool) -> Vec<Shape> {
    if small {
        vec![
            Shape {
                name: "gate/up",
                n: 32,
                k: 64,
                calls_per_layer: 2.0,
            },
            Shape {
                name: "down",
                n: 64,
                k: 64,
                calls_per_layer: 1.0,
            },
        ]
    } else {
        vec![
            Shape {
                name: "gate/up",
                n: 2048,
                k: 4096,
                calls_per_layer: 2.0,
            },
            Shape {
                name: "down",
                n: 4096,
                k: 2048,
                calls_per_layer: 1.0,
            },
        ]
    }
}

fn run(dev: &Device, cfg: &Cfg, act_dtype: DType, cap: &str) -> Result<()> {
    let cuda = matches!(dev, Device::Cuda(_));
    let shapes = ab_shapes(cfg.small);

    // Prove the guards can still fail BEFORE trusting any of them (D12), then
    // run them against this configuration — all before any GPU time is spent
    // baking a fixture that could not have answered the question.
    assert_guards_have_teeth()?;
    assert_sweep_can_decide(&cfg.batches)?;
    assert_coverage_headroom(cfg.experts, &cfg.batches)?;
    assert_config_is_measurable(cfg.experts, &cfg.batches)?;
    for s in &shapes {
        if cuda {
            assert_cuda_paths_reachable(s.k, act_dtype)?;
            assert_defeats_l2(cfg.experts, bytes_per_expert(s))?;
        }
    }
    println!(
        "guards OK: E={} top-{TOPK}, E(B) spans {:.1} -> {:.1} across B={:?}\n",
        cfg.experts,
        qtip_expected_distinct_experts(cfg.batches[0], TOPK, cfg.experts),
        qtip_expected_distinct_experts(*cfg.batches.last().expect("non-empty"), TOPK, cfg.experts),
        cfg.batches
    );

    let mut grouped: Vec<Vec<Point>> = Vec::new();
    let mut gemv: Vec<Vec<Point>> = Vec::new();

    for s in &shapes {
        let w =
            Tensor::randn(0f32, 0.02f32, (cfg.experts, s.n, s.k), dev)?.to_dtype(DType::BF16)?;
        // The PRODUCTION door: Viterbi + the rotation policy a served qtip2b
        // artifact carries. `Greedy` is refused here in every build (D4), and
        // it is not merely a rule violation — a Greedy layer skips rotation,
        // making the forward path cheaper and the speed number inflated.
        let layer = Qtip2bLayer::quantize_with_mode(&w, None, dev, QtipMode::Viterbi)?;
        drop(w);
        println!(
            "[{}] N={} K={} E={} (bytes/expert={:.2} MB, packed stack {:.0} MB)",
            s.name,
            s.n,
            s.k,
            cfg.experts,
            bytes_per_expert(s) / 1e6,
            cfg.experts as f64 * bytes_per_expert(s) / 1e6
        );

        if cuda {
            let g = time_curve(
                dev,
                &layer,
                s,
                cfg.experts,
                &cfg.batches,
                cfg.iters,
                act_dtype,
                Path::Grouped,
                cap,
            )?;
            let v = time_curve(
                dev,
                &layer,
                s,
                cfg.experts,
                &cfg.batches,
                cfg.iters,
                act_dtype,
                Path::Gemv,
                cap,
            )?;
            assert_paths_are_distinguishable(&g, &v)?;
            report_shape_efficiency(s, &g, &v);
            grouped.push(g);
            gemv.push(v);
        } else {
            time_curve(
                dev,
                &layer,
                s,
                cfg.experts,
                &cfg.batches,
                cfg.iters,
                act_dtype,
                Path::CpuRef,
                cap,
            )?;
        }
        println!();
    }

    if cuda {
        report_verdict(&shapes, &grouped, &gemv, &cfg.batches);
    } else {
        println!(
            "CPU run: guards + routing model verified, `gather_forward_cpu` timed. Neither the \
             grouped GEMM nor the fused GEMV exists here, so NO kernel comparison is reported and \
             no crossover verdict is possible. Run with --features cuda on a GPU box for that."
        );
    }
    Ok(())
}

fn main() -> Result<()> {
    let cfg = parse_args()?;
    // Set the memoized cap ONCE, before the first `gather_forward` in this
    // process, so the GEMV arm is measurable at every swept batch. Path
    // selection afterwards uses the per-call env vars only.
    let max_b = *cfg.batches.iter().max().unwrap_or(&1);
    let cap = max_b.to_string();
    if let Ok(pre) = std::env::var(QTIP_ONDEVICE_MOE_MAX_TOKENS_ENV) {
        if pre != cap {
            candle_core::bail!(
                "{QTIP_ONDEVICE_MOE_MAX_TOKENS_ENV} is already set to '{pre}' in the environment. \
                 It is memoized on first read, so this harness cannot raise it to {cap} and the \
                 gemv arm would silently stop at {pre} tokens. Unset it and re-run."
            );
        }
    }
    std::env::set_var(QTIP_ONDEVICE_MOE_MAX_TOKENS_ENV, &cap);

    match Device::new_cuda(0) {
        Ok(dev) => {
            if let Some(b) = cfg.ab {
                println!(
                    "=== qtip2b grouped-GEMM KERNEL VARIANT A/B at B={b} \
                     (U F U N x {}, {} iters/arm, top-{TOPK}, Viterbi+rotation) ===\n\
                     Instrument: wall clock across {} synchronized gather_forward calls — the \
                     SAME instrument that produced the wave38 crossover table, so the numbers are \
                     comparable to it and NOT to any device-timer span.\n",
                    cfg.rounds, cfg.iters, cfg.iters
                );
                // The guards still have to prove they can fail before any
                // number is printed, even in --ab.
                assert_guards_have_teeth()?;
                return run_ab(&dev, &cfg, DType::BF16, b, cfg.rounds);
            }
            println!(
                "=== qtip2b grouped-GEMM vs fused gather-GEMV batch curve \
                 (H200 peak {:.1} TB/s, top-{TOPK}, Viterbi+rotation) ===\n",
                H200_PEAK / 1e12
            );
            run(&dev, &cfg, DType::BF16, &cap)
        }
        Err(_) => {
            println!("=== no CUDA device: guard + routing-model validation on CPU ===\n");
            let cfg = Cfg {
                small: true,
                iters: 2,
                ..cfg
            };
            run(&Device::Cpu, &cfg, DType::F32, &cap)
        }
    }
}
