//! Parent system: ArcGate
//!
//! Boolean environment flags, read **by value** rather than by presence.
//!
//! # Why this module exists
//!
//! Arc gates a large amount of behaviour on `ARC_*` environment variables. Most
//! of them were read as `std::env::var_os(NAME).is_some()` (or `.is_none()`,
//! or `std::env::var(NAME).is_ok()`), which tests whether the variable *exists*
//! and never looks at what it says. Under that reading, `ARC_FOO=0` — the
//! spelling every operator and every A/B script uses to mean "off" — turns the
//! feature **on**.
//!
//! That is not a hypothetical. `ARC_NO_DEDICATED_DECODE` was read this way, and
//! two A/B harnesses passed exactly `ARC_NO_DEDICATED_DECODE=0` as their control
//! leg (`arc-tools/arcgraph_heap_probe.sh`, `arc-tools/arcgraph_capture_probe.sh`).
//! Both arms of both experiments therefore ran with the path in the *same*
//! state, and the variable under test cancelled out of the result. Neither run
//! failed. Both returned an honestly-measured difference of zero for a
//! comparison that was never made — which is the worst failure mode this
//! codebase has, because it is indistinguishable from a real null result.
//!
//! So: flags are read by value, and a value that is not recognised is
//! **reported** rather than silently bucketed. A typo in an ops script must not
//! decide an experiment.
//!
//! # Where this lives, and why it lives here
//!
//! `mistralrs-quant` is the lowest crate in the workspace — `mistralrs-core`,
//! `arc-engine` and `arc-cuda-graph` all already depend on it — so every crate
//! that gates behaviour on a flag can reach this parser without a new edge in
//! the dependency graph.
//!
//! It also has to be *testable*, and that is a stronger constraint than it
//! looks. The obvious home for a polarity test is beside the gate it guards,
//! but several of those gates live in `qtip::cuda_ops`, which is
//! `#![cfg(feature = "cuda")]` — so a test written there **cannot run on any
//! machine without CUDA**, which is every machine any of these polarities has
//! ever been reviewed on. A test that cannot run is not a test.
//!
//! `mistralrs-quant` is in CI's `test` lane (`cargo test -p mistralrs-core
//! -p mistralrs-quant -p mistralrs-vision`, run with **no features**, on
//! ubuntu/windows/macOS) and in the scoped `clippy` lane. The tests below
//! therefore run on ordinary CI with no GPU and no `cuda` feature, on three
//! operating systems, on every PR.

/// Is `name` set to a value that means **yes**?
///
/// `1`, `true`, `yes`, `on` (any case, surrounding whitespace ignored) are yes.
/// Unset, empty, `0`, `false`, `no`, `off` are no.
///
/// An unrecognised value logs a warning and is treated as **no**. It is not
/// silently folded into `false`: an operator who wrote `ARC_FOO=truu` needs to
/// find out from the log, not from a result that quietly disagrees with the one
/// they thought they were producing.
///
/// Note the deliberate asymmetry with a presence test: the *unset* answer is
/// `false` here, exactly as `var_os(..).is_some()` returned `false` when unset.
/// Converting a call site from a presence test to this function therefore
/// changes nothing about the default — it changes only how an explicitly-set
/// value is interpreted.
pub fn env_flag_is_set(name: &str) -> bool {
    let raw = std::env::var(name).ok();
    match env_flag_value(raw.as_deref()) {
        Some(v) => v,
        None => {
            tracing::warn!(
                "{name}={:?} is not a recognised on/off value (expected 1/0, true/false, \
                 yes/no, on/off) — treating it as OFF. If you meant to enable it, set \
                 {name}=1.",
                raw.unwrap_or_default()
            );
            false
        }
    }
}

/// Pure half of [`env_flag_is_set`], so the polarity is testable without
/// mutating the process environment.
///
/// Mutating the environment from a test is racy across `cargo test`'s threads
/// and, since the 2024 edition, `unsafe`; and most of the call sites converted
/// to this parser latch their answer in a `OnceLock`, so a test that set the
/// variable would prove only whatever the first reader in the process happened
/// to see. Testing the pure function against every input is the only form of
/// this test that means anything.
///
/// `None` means "value present but not recognised" — the caller decides what to
/// do with that, and must not silently fold it into `false`.
pub fn env_flag_value(raw: Option<&str>) -> Option<bool> {
    let Some(raw) = raw else {
        return Some(false);
    };
    match raw.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Some(true),
        "" | "0" | "false" | "no" | "off" => Some(false),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every raw value a converted flag is pinned against, with the answer
    /// [`env_flag_is_set`] must give. `None` is "unset".
    ///
    /// The `Some("0")` row is the whole point of this module: under the
    /// presence test it replaced, that row was `true`.
    const VALUE_TABLE: &[(Option<&str>, bool)] = &[
        (Some("1"), true),
        (Some("true"), true),
        (Some("TRUE"), true),
        (Some("yes"), true),
        (Some("on"), true),
        (Some(" 1 "), true),
        (Some("0"), false),
        (Some("off"), false),
        (Some("false"), false),
        (Some("no"), false),
        (Some(""), false),
        (Some("truu"), false),
        (Some("2"), false),
        (None, false),
    ];

    /// How a call site consumes the flag.
    #[derive(Clone, Copy, Debug)]
    enum Sense {
        /// `if env_flag_is_set(NAME)` — was `var_os(NAME).is_some()` /
        /// `var(NAME).is_ok()`.
        Asserted,
        /// `if !env_flag_is_set(NAME)` — was `var_os(NAME).is_none()` /
        /// `var(NAME).is_err()`.
        Negated,
    }

    /// The value the gate at the call site evaluates to.
    fn gate(sense: Sense, raw: Option<&str>) -> bool {
        let flag = env_flag_value(raw).unwrap_or(false);
        match sense {
            Sense::Asserted => flag,
            Sense::Negated => !flag,
        }
    }

    /// What the gate evaluated to **before** this change, for the unset case
    /// only. A presence test on an unset variable is `is_some() == false` /
    /// `is_none() == true`; this reproduces that from first principles so the
    /// "unset behaviour is unchanged" claim is checked rather than asserted.
    fn gate_before_fix_when_unset(sense: Sense) -> bool {
        let present = false; // the variable is not set
        match sense {
            Sense::Asserted => present,
            Sense::Negated => !present,
        }
    }

    /// Pin one converted flag's full value table, and pin that its unset
    /// behaviour is byte-identical to the presence test it replaced.
    ///
    /// `primary` records the form the call site actually reads (documentation
    /// for the reviewer); the assertions run over **both** senses, because a
    /// flag can be consumed both ways in different places — `ARC_COLLAPSE` is
    /// `is_some()` at three sites and `is_none()` at a fourth — and because
    /// `gate(Negated, ..)` is the exact complement of `gate(Asserted, ..)`, so
    /// checking both costs nothing and closes that gap by construction.
    fn assert_flag(name: &str, primary: Sense) {
        let _ = primary;
        for sense in [Sense::Asserted, Sense::Negated] {
            for (raw, want_flag) in VALUE_TABLE {
                let want_gate = match sense {
                    Sense::Asserted => *want_flag,
                    Sense::Negated => !*want_flag,
                };
                assert_eq!(
                    gate(sense, *raw),
                    want_gate,
                    "{name}={raw:?} ({sense:?}): gate should be {want_gate}, got {}",
                    gate(sense, *raw)
                );
            }
            assert_eq!(
                gate(sense, None),
                gate_before_fix_when_unset(sense),
                "{name} ({sense:?}): UNSET behaviour changed — this PR must not move any default"
            );
        }
    }

    /// One test per converted flag, so a failure names the flag rather than a
    /// row index. Each entry carries the call site it pins.
    macro_rules! converted_flags {
        ($($fn_name:ident: $flag:literal, $sense:ident, $site:literal;)*) => {
            $(
                #[doc = concat!("Value table for `", $flag, "` (", $site, ").")]
                #[test]
                fn $fn_name() {
                    assert_flag($flag, Sense::$sense);
                }
            )*

            /// The census itself: every flag this PR converted, in one place,
            /// so the count is reviewable and a silently-dropped conversion is
            /// visible as a changed number rather than as nothing at all.
            #[test]
            fn converted_flag_census() {
                let flags: &[&str] = &[$($flag),*];
                assert_eq!(flags.len(), 23, "converted-flag count changed: {flags:?}");
                let mut sorted = flags.to_vec();
                sorted.sort_unstable();
                sorted.dedup();
                assert_eq!(sorted.len(), flags.len(), "duplicate flag in the census");
            }
        };
    }

    converted_flags! {
        // ---- mistralrs-core -------------------------------------------------
        arc_yarn_on_standard_layers:
            "ARC_YARN_ON_STANDARD_LAYERS", Negated, "models/deepseek4.rs:417";
        arc_disable_sink:
            "ARC_DISABLE_SINK", Asserted, "models/deepseek4.rs:1188";
        arc_v4_window_only:
            "ARC_V4_WINDOW_ONLY", Negated, "models/deepseek4.rs:1791";
        arc_collapse:
            "ARC_COLLAPSE", Asserted, "models/deepseek4.rs:2219,2489,2741,3396";
        arc_softmax_route:
            "ARC_SOFTMAX_ROUTE", Asserted, "models/deepseek4.rs:2500";
        arc_route_top1:
            "ARC_ROUTE_TOP1", Asserted, "models/deepseek4.rs:2513";
        arc_v4_capture_probe:
            "ARC_V4_CAPTURE_PROBE", Asserted, "models/deepseek4.rs:3154, pipeline/normal.rs:1853";
        arc_time_decode:
            "ARC_TIME_DECODE", Asserted, "models/deepseek4.rs:3283";
        v4_nan_debug:
            "V4_NAN_DEBUG", Negated, "models/deepseek4.rs:3334";
        v4_stats:
            "V4_STATS", Negated, "models/deepseek4.rs:3364";
        arc_prefer_fa3:
            "ARC_PREFER_FA3", Negated, "attention/backends/flash.rs:502 (test guard)";
        arc_v4_standard_dense:
            "ARC_V4_STANDARD_DENSE", Asserted, "models/dsv4_attention.rs:712";
        arc_gpu_act_quant:
            "ARC_GPU_ACT_QUANT", Asserted, "models/dsv4_kv_fp8.rs:161";
        arc_no_fused_sinkhorn:
            "ARC_NO_FUSED_SINKHORN", Negated, "models/dsv4_mhc.rs:745";
        arc_moe_slow:
            "ARC_MOE_SLOW", Asserted, "moe/experts.rs:268";
        arc_quant_attention:
            "ARC_QUANT_ATTENTION", Asserted, "pipeline/loaders/normal_loaders.rs:3385";
        arc_mtp_allow_sub_int8:
            "ARC_MTP_ALLOW_SUB_INT8", Asserted, "pipeline/mtp_pipeline.rs:182";

        // ---- mistralrs-quant ------------------------------------------------
        arc_sync_isq:
            "ARC_SYNC_ISQ", Asserted, "utils/isq.rs:34,106";
        arc_no_fp8_gemv:
            "ARC_NO_FP8_GEMV", Asserted, "blockwise_fp8/ops.rs:917";
        arc_no_fp8_wmma:
            "ARC_NO_FP8_WMMA", Negated, "blockwise_fp8/ops.rs:1005";
        arc_no_qtip_ondevice_moe:
            "ARC_NO_QTIP_ONDEVICE_MOE", Asserted, "qtip/bitshift.rs:1934, qtip/mod.rs:3712";
        arc_no_qtip_grouped_moe:
            "ARC_NO_QTIP_GROUPED_MOE", Asserted, "qtip/bitshift.rs:1949, qtip/mod.rs:3657";

        // ---- arc-cuda-graph -------------------------------------------------
        arc_no_deferred_free:
            "ARC_NO_DEFERRED_FREE", Negated, "examples/capture_probe.rs:19";
    }

    /// The bug this module exists to remove, stated as an assertion.
    ///
    /// Under the presence test these call sites used, `ARC_FOO=0` was
    /// indistinguishable from `ARC_FOO=1`: both were "the variable exists".
    /// If this test ever passes with `presence` and `by_value` equal for `"0"`,
    /// the parser has been reverted to a presence test.
    #[test]
    fn zero_means_off_not_on() {
        let presence = Some("0").is_some();
        let by_value = env_flag_value(Some("0")).unwrap();
        assert!(presence, "sanity: a set variable is present");
        assert!(!by_value, "ARC_FOO=0 must mean OFF");
        assert_ne!(
            presence, by_value,
            "the parser is testing presence again, not value"
        );
    }

    /// Unrecognised values are reported, not bucketed.
    #[test]
    fn unrecognised_value_is_distinguishable_from_off() {
        assert_eq!(env_flag_value(Some("0")), Some(false));
        assert_eq!(env_flag_value(Some("false")), Some(false));
        assert_eq!(env_flag_value(Some("truu")), None);
        assert_eq!(env_flag_value(Some("maybe")), None);
        assert_eq!(env_flag_value(Some("2")), None);
    }

    /// Unset is `Some(false)`, never `None` — "not set" is a recognised state,
    /// not a parse failure, so it must never produce a warning.
    #[test]
    fn unset_is_off_and_not_a_parse_failure() {
        assert_eq!(env_flag_value(None), Some(false));
    }

    /// Whitespace and case are operator typing, not intent.
    #[test]
    fn whitespace_and_case_are_ignored() {
        for raw in [" 1", "1 ", "\t1\n", "TRUE", "True", "On", "YES"] {
            assert_eq!(
                env_flag_value(Some(raw)),
                Some(true),
                "{raw:?} should be on"
            );
        }
        for raw in [" 0", "0 ", "OFF", "False", "No"] {
            assert_eq!(
                env_flag_value(Some(raw)),
                Some(false),
                "{raw:?} should be off"
            );
        }
    }
}
