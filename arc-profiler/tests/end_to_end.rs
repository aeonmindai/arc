//! End-to-end: drive a realistically-shaped token path, write both artifacts,
//! read them back, and check them the way a reader would.
//!
//! This runs on a laptop with no GPU, which is the point — the JSON/HTML
//! pipeline must be provable without renting anything. What it cannot prove
//! here is device timing (no CUDA backend ⇒ device spans are *unmeasured*, and
//! the test asserts that the report says so rather than reporting zeros).
//!
//! Deliberately one test in one file: `write_outputs` reads `ARC_PROFILE_OUT`
//! from the process environment, and a second concurrent test would race it.

use std::path::PathBuf;

/// Layer count of DeepSeek-V4-Flash, so the tree this produces has the same
/// shape (and roughly the same node count) as a real profile.
const LAYERS: usize = 43;
const STEPS: usize = 6;
const WARMUP: u64 = 2;
const BATCH: usize = 64;

fn simulate_layer() {
    let _layer = arc_profiler::span("layer");
    {
        let _a = arc_profiler::device_span("mhc_attn_pre");
    }
    {
        let _a = arc_profiler::device_span("mla_attn");
        for name in [
            "q_proj",
            "q_rmsnorm",
            "kv_proj",
            "rope",
            "kv_cache_append",
            "sdpa",
            "inv_rope",
            "o_proj",
        ] {
            let _s = arc_profiler::device_span(name);
            std::hint::black_box(name);
        }
    }
    {
        let _m = arc_profiler::device_span("moe");
        {
            let _g = arc_profiler::device_span("moe.gate");
        }
        {
            let _e = arc_profiler::device_span("moe.experts");
            for name in ["experts.gate_proj", "experts.swiglu", "experts.down_proj"] {
                let _s = arc_profiler::device_span(name);
                std::hint::black_box(name);
            }
        }
        {
            let _sh = arc_profiler::device_span("moe.shared_expert");
        }
    }
}

fn simulate_step() {
    let _step = arc_profiler::step_scope("step");
    arc_profiler::set_geometry(BATCH, 1);
    {
        let _s = arc_profiler::span("scheduler.schedule");
    }
    let _decode = arc_profiler::span("decode");
    let _pstep = arc_profiler::span("pipeline.step");
    {
        let _s = arc_profiler::span("input_prep");
        for _ in 0..BATCH {
            let _h = arc_profiler::sync_span("input_prep.h2d_per_seq");
        }
    }
    {
        let _f = arc_profiler::span("forward");
        let _m = arc_profiler::span("model");
        {
            let _e = arc_profiler::device_span("embed");
        }
        {
            let _l = arc_profiler::span("layers");
            for _ in 0..LAYERS {
                simulate_layer();
            }
        }
        {
            let _h = arc_profiler::device_span("lm_head");
        }
        arc_profiler::mark_unreachable(
            "cuda_graph.capture_probe",
            "ARC_V4_CAPTURE_PROBE is unset",
            "normal.rs:1554",
        );
    }
    {
        let _d = arc_profiler::sync_span("logits_d2h");
        std::thread::sleep(std::time::Duration::from_micros(200));
    }
    {
        let _s = arc_profiler::span("cache.post_op");
        let _c = arc_profiler::span("clone_out_cache");
    }
    {
        let _s = arc_profiler::span("sample_and_dispatch");
        {
            let _j = arc_profiler::span("sample.join_all");
        }
        let _fin = arc_profiler::span("finish_or_add_toks");
        for _ in 0..BATCH {
            let _t = arc_profiler::span("detokenize");
        }
    }
}

#[test]
fn writes_a_readable_json_and_a_self_contained_html() {
    let dir: PathBuf = std::env::temp_dir().join(format!("arc-profile-e2e-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    // SAFETY: single-threaded at this point; this file holds exactly one test
    // precisely so nothing else can observe the change.
    unsafe {
        std::env::set_var("ARC_PROFILE_OUT", &dir);
        std::env::set_var("ARC_PROFILE_LABEL", "B=64");
    }

    arc_profiler::__force_enabled(true);
    arc_profiler::__set_warmup(WARMUP);
    arc_profiler::__set_depth(16);
    arc_profiler::reset();
    arc_profiler::set_meta(|h| {
        h.label = "B=64".into();
        h.model = "DeepSeek-V4-Flash".into();
        h.artifact = "aeonmind/DeepSeek-V4-Flash-UQFF-qtip2".into();
        h.commit = "test".into();
        h.requested_batch = BATCH as u32;
    });

    for _ in 0..STEPS {
        simulate_step();
    }

    let (json_path, html_path) = arc_profiler::write_outputs().expect("write");
    eprintln!("wrote {} and {}", json_path.display(), html_path.display());

    // ---- JSON ----
    let raw = std::fs::read_to_string(&json_path).expect("read json");
    let p: arc_profiler::Profile = serde_json::from_str(&raw).expect("parse json");
    assert_eq!(p.schema, arc_profiler::SCHEMA);
    assert_eq!(
        p.run.steps,
        STEPS as u64 - WARMUP,
        "warmup must be excluded"
    );
    assert_eq!(p.totals.tokens, (STEPS as u64 - WARMUP) * BATCH as u64);
    assert_eq!(p.run.model, "DeepSeek-V4-Flash");

    // The tree really is a tree, all the way down.
    let layer = p
        .node("step.decode.pipeline.step.forward.model.layers.layer")
        .unwrap();
    assert_eq!(
        layer.calls,
        (STEPS as u64 - WARMUP) * LAYERS as u64,
        "one `layer` node aggregating every layer of every recorded step"
    );
    let sdpa = p
        .node("step.decode.pipeline.step.forward.model.layers.layer.mla_attn.sdpa")
        .expect("the deepest instrumented node must survive to the report");
    assert_eq!(sdpa.depth, 8);
    assert_eq!(sdpa.kind, arc_profiler::NodeKind::Device);

    // Waiting is not computing.
    let d2h = p.node("step.decode.pipeline.step.logits_d2h").unwrap();
    assert!(d2h.sync_ns > 0);
    assert_eq!(d2h.busy_self_ns, 0);
    assert!(p.totals.sync_ns >= d2h.sync_ns);

    // Geometry survived.
    assert_eq!(p.node("step").unwrap().geom.b, BATCH as u32);

    // Unreachable is labelled, not zeroed.
    assert!(p
        .unreachable
        .iter()
        .any(|u| u.path.ends_with("cuda_graph.capture_probe")));

    // No CUDA backend here, so device spans must read as unmeasured.
    assert_eq!(p.totals.device_ns, 0);
    assert!(p.reconciliation.unresolved_device_spans > 0);
    assert!(p.run.notes.iter().any(|n| n.contains("null")));

    // And the whole thing reconciles.
    assert!(
        p.reconciliation.violations.is_empty(),
        "violations: {:?}",
        p.reconciliation.violations
    );
    assert_eq!(p.reconciliation.misnested_spans, 0);
    assert!(p.recheck(arc_profiler::RECONCILE_TOLERANCE_PCT).is_empty());

    // ---- HTML ----
    let html = std::fs::read_to_string(&html_path).expect("read html");
    assert!(html.contains("arc-profile/1"), "profile must be embedded");
    assert!(
        !html.contains("/*__ARC_PROFILE_DATA__*/[]"),
        "the data marker must have been consumed"
    );
    for forbidden in ["http://", "https://", "//cdn.", "@import url("] {
        assert!(!html.contains(forbidden), "not self-contained: {forbidden}");
    }
    assert!(
        html.len() > 50_000,
        "the page should carry a real UI, not a stub"
    );

    let _ = std::fs::remove_dir_all(&dir);
}
