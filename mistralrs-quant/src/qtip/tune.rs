//! Autotune dispatch for the qtip2b bitshift-trellis fused decode+GEMV.
//!
//! `kernels/qtip/qtip_bitshift_tune.cu` compile-time instantiates the GEMV
//! over the tuning axes (warps/block, rows/warp, ILP streams/thread, vector
//! load width, `__launch_bounds__` min-blocks, smem staging, prefetch depth).
//! This module decides WHICH variant a production launch uses, in priority
//! order:
//!
//! 1. A forced variant — `set_forced_gemv_variant` (used by the sweep binary
//!    `examples/qtip_gemv_tune.rs` and the all-variants parity test) or the
//!    `ARC_QTIP_GEMV_VARIANT` env var (`"legacy"` or a variant id), read once.
//! 2. A measured winner table — `ARC_QTIP_TUNE_TABLE=/path/to/tune_results.json`
//!    (the sweep binary's output; only its `winners` array is consumed), read
//!    once. This is the session-4 measure→apply loop: rerun the sweep on the
//!    rental GPU, point the env var at the fresh JSON, no recompile.
//! 3. The baked-in default table below (current best guesses per V4 shape).
//! 4. No entry → the legacy fixed-config kernel (`launch_qtip2b_gemv_*`).
//!
//! Whatever this module picks, `cuda_ops` treats the tuned launcher's non-zero
//! return (variant inapplicable to the shape: row alignment vs the vector
//! load width, or smem capacity) as "fall back to the legacy kernel", so a
//! stale or wrong table can never break correctness — every variant passes
//! the same parity contract as the legacy kernel.

use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::OnceLock;

use serde::{Deserialize, Serialize};

/// Sentinel variant id meaning "use the legacy fixed-config kernel".
/// Accepted anywhere a variant id is: forced env (`ARC_QTIP_GEMV_VARIANT=legacy`),
/// tune JSON winners, and the baked table.
pub const QTIP2B_GEMV_VARIANT_LEGACY: u32 = u32::MAX;

/// One (shape → winning variant) row. `n` is the output rows per expert,
/// `k` the in-features; both in elements (not bytes).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct GemvTuneEntry {
    pub n: usize,
    pub k: usize,
    pub variant: u32,
}

/// Baked-in default winners (current best guesses). Variant 0 is the tuned
/// replica of the production baseline config (warps=8, rows/warp=2, ILP=1,
/// 2-byte loads, smem staging) — behavior-identical to the legacy kernel
/// modulo the O(1) warm-up. Replace these ids with sweep winners from
/// `qtip_gemv_tune` (the binary prints a ready-to-paste table).
pub const QTIP2B_GEMV_BAKED_TABLE: &[GemvTuneEntry] = &[
    // V4-Flash decode shapes (per expert): gate/up and down projections.
    GemvTuneEntry {
        n: 2048,
        k: 4096,
        variant: 0,
    },
    GemvTuneEntry {
        n: 4096,
        k: 2048,
        variant: 0,
    },
];

// Forced-variant state: FORCED_UNINIT until first read (then latched from the
// env), FORCED_NONE when unforced, otherwise the variant id (the legacy
// sentinel fits in i64).
const FORCED_UNINIT: i64 = -2;
const FORCED_NONE: i64 = -1;
static FORCED: AtomicI64 = AtomicI64::new(FORCED_UNINIT);

/// Force every subsequent qtip2b GEMV launch onto one variant
/// (`Some(QTIP2B_GEMV_VARIANT_LEGACY)` forces the legacy kernel; `None`
/// restores normal table-driven dispatch). Process-global — meant for the
/// sweep binary and the parity tests, not for serving.
pub fn set_forced_gemv_variant(v: Option<u32>) {
    FORCED.store(v.map(|x| x as i64).unwrap_or(FORCED_NONE), Ordering::SeqCst);
}

fn parse_forced(s: &str) -> Option<i64> {
    let s = s.trim();
    if s.eq_ignore_ascii_case("legacy") {
        return Some(QTIP2B_GEMV_VARIANT_LEGACY as i64);
    }
    s.parse::<u32>().ok().map(|v| v as i64)
}

/// The currently forced variant, if any (env-latched on first call).
pub fn forced_gemv_variant() -> Option<u32> {
    let mut cur = FORCED.load(Ordering::SeqCst);
    if cur == FORCED_UNINIT {
        let init = std::env::var("ARC_QTIP_GEMV_VARIANT")
            .ok()
            .and_then(|s| parse_forced(&s))
            .unwrap_or(FORCED_NONE);
        // Keep whichever init won the race (or an explicit set that beat us).
        let _ = FORCED.compare_exchange(FORCED_UNINIT, init, Ordering::SeqCst, Ordering::SeqCst);
        cur = FORCED.load(Ordering::SeqCst);
    }
    (cur >= 0).then_some(cur as u32)
}

#[derive(Deserialize)]
struct TuneFile {
    winners: Vec<GemvTuneEntry>,
}

/// Parse the `winners` array out of a tune_results.json payload.
pub(crate) fn parse_winners_json(payload: &str) -> Option<Vec<GemvTuneEntry>> {
    serde_json::from_str::<TuneFile>(payload)
        .ok()
        .map(|f| f.winners)
}

fn override_table() -> Option<&'static [GemvTuneEntry]> {
    static TABLE: OnceLock<Option<Vec<GemvTuneEntry>>> = OnceLock::new();
    TABLE
        .get_or_init(|| {
            let path = std::env::var("ARC_QTIP_TUNE_TABLE").ok()?;
            match std::fs::read_to_string(&path)
                .ok()
                .and_then(|s| parse_winners_json(&s))
            {
                Some(w) => {
                    tracing::info!(
                        "qtip2b GEMV tune table loaded from {path}: {} winner(s)",
                        w.len()
                    );
                    Some(w)
                }
                None => {
                    tracing::warn!(
                        "ARC_QTIP_TUNE_TABLE={path} unreadable or missing a `winners` array; \
                         falling back to the baked table"
                    );
                    None
                }
            }
        })
        .as_deref()
}

fn lookup(table: &[GemvTuneEntry], n: usize, k: usize) -> Option<u32> {
    table
        .iter()
        .find(|e| e.n == n && e.k == k)
        .map(|e| e.variant)
}

/// The variant a qtip2b GEMV launch for `[n_rows, k_in]` (per expert) should
/// try first. `None` and `Some(QTIP2B_GEMV_VARIANT_LEGACY)` both mean the
/// legacy kernel; any other id may still be rejected by the tuned launcher
/// for this shape, in which case the caller falls back to legacy.
pub fn gemv_variant_for_shape(n_rows: usize, k_in: usize) -> Option<u32> {
    if let Some(v) = forced_gemv_variant() {
        return Some(v);
    }
    if let Some(t) = override_table() {
        if let Some(v) = lookup(t, n_rows, k_in) {
            return Some(v);
        }
    }
    lookup(QTIP2B_GEMV_BAKED_TABLE, n_rows, k_in)
}

/// The tuning-axis values of one compiled variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GemvVariantDesc {
    pub warps: usize,
    pub rows_per_warp: usize,
    pub ilp: usize,
    pub wbytes: usize,
    pub min_blocks: usize,
    pub stage: bool,
    pub prefetch: bool,
}

impl GemvVariantDesc {
    /// Compact label, e.g. `w8_r2_i1_v2_mb1_s1_p0`.
    pub fn label(&self) -> String {
        format!(
            "w{}_r{}_i{}_v{}_mb{}_s{}_p{}",
            self.warps,
            self.rows_per_warp,
            self.ilp,
            self.wbytes,
            self.min_blocks,
            self.stage as u8,
            self.prefetch as u8
        )
    }
}

/// Number of compiled GEMV variants (0 when built without CUDA).
#[cfg(feature = "cuda")]
pub fn gemv_num_variants() -> usize {
    if !super::ffi::HAVE_QTIP_KERNELS {
        return 0;
    }
    (unsafe { super::ffi::qtip2b_gemv_num_variants() }).max(0) as usize
}

/// Number of compiled GEMV variants (0 when built without CUDA).
#[cfg(not(feature = "cuda"))]
pub fn gemv_num_variants() -> usize {
    0
}

/// Axis values of variant `idx`, when compiled in.
#[cfg(feature = "cuda")]
pub fn gemv_variant_desc(idx: usize) -> Option<GemvVariantDesc> {
    if !super::ffi::HAVE_QTIP_KERNELS {
        return None;
    }
    let (mut w, mut r, mut i, mut v, mut mb, mut s, mut p) = (0i32, 0, 0, 0, 0, 0, 0);
    let rc = unsafe {
        super::ffi::qtip2b_gemv_variant_info(
            idx as i32, &mut w, &mut r, &mut i, &mut v, &mut mb, &mut s, &mut p,
        )
    };
    (rc == 0).then(|| GemvVariantDesc {
        warps: w as usize,
        rows_per_warp: r as usize,
        ilp: i as usize,
        wbytes: v as usize,
        min_blocks: mb as usize,
        stage: s != 0,
        prefetch: p != 0,
    })
}

/// Axis values of variant `idx`, when compiled in.
#[cfg(not(feature = "cuda"))]
pub fn gemv_variant_desc(_idx: usize) -> Option<GemvVariantDesc> {
    None
}

/// Whether `variant` can launch for a `[n_rows, k_in]` shape — the Rust
/// mirror of the tuned launcher's applicability checks, so the sweep binary
/// can tell "measured the variant" apart from "silently fell back to legacy".
pub fn gemv_variant_applicable(variant: u32, _n_rows: usize, k_in: usize) -> bool {
    if variant == QTIP2B_GEMV_VARIANT_LEGACY {
        return true;
    }
    let Some(d) = gemv_variant_desc(variant as usize) else {
        return false;
    };
    if !k_in.is_multiple_of(4) {
        return false;
    }
    let packed_per_row = k_in / 4;
    if !packed_per_row.is_multiple_of(d.wbytes) {
        return false;
    }
    if d.stage && d.warps * d.rows_per_warp * packed_per_row > 48 * 1024 {
        return false;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn baked_table_covers_v4_decode_shapes() {
        assert_eq!(lookup(QTIP2B_GEMV_BAKED_TABLE, 2048, 4096), Some(0));
        assert_eq!(lookup(QTIP2B_GEMV_BAKED_TABLE, 4096, 2048), Some(0));
        assert_eq!(lookup(QTIP2B_GEMV_BAKED_TABLE, 1234, 5678), None);
    }

    #[test]
    fn winners_json_roundtrip() {
        let entries = vec![
            GemvTuneEntry {
                n: 2048,
                k: 4096,
                variant: 12,
            },
            GemvTuneEntry {
                n: 4096,
                k: 2048,
                variant: QTIP2B_GEMV_VARIANT_LEGACY,
            },
        ];
        // The sweep binary writes winners alongside other keys; the loader
        // must tolerate (ignore) them.
        let payload = format!(
            "{{\"device\":\"cuda:0\",\"winners\":{},\"shapes\":[]}}",
            serde_json::to_string(&entries).unwrap()
        );
        let parsed = parse_winners_json(&payload).unwrap();
        assert_eq!(parsed, entries);
        assert_eq!(lookup(&parsed, 2048, 4096), Some(12));
        assert_eq!(
            lookup(&parsed, 4096, 2048),
            Some(QTIP2B_GEMV_VARIANT_LEGACY)
        );

        assert!(parse_winners_json("{}").is_none());
        assert!(parse_winners_json("not json").is_none());
    }

    #[test]
    fn forced_variant_set_and_parse() {
        // parse_forced: ids, legacy keyword, junk.
        assert_eq!(parse_forced("7"), Some(7));
        assert_eq!(
            parse_forced("legacy"),
            Some(QTIP2B_GEMV_VARIANT_LEGACY as i64)
        );
        assert_eq!(
            parse_forced(" LEGACY "),
            Some(QTIP2B_GEMV_VARIANT_LEGACY as i64)
        );
        assert_eq!(parse_forced("-3"), None);
        assert_eq!(parse_forced("fast"), None);

        // Global forced state: set → read → clear. (Other tests don't touch
        // this global; the env latch only fires when the state is UNINIT, so
        // an explicit set always wins.)
        set_forced_gemv_variant(Some(3));
        assert_eq!(forced_gemv_variant(), Some(3));
        assert_eq!(gemv_variant_for_shape(2048, 4096), Some(3));
        set_forced_gemv_variant(Some(QTIP2B_GEMV_VARIANT_LEGACY));
        assert_eq!(forced_gemv_variant(), Some(QTIP2B_GEMV_VARIANT_LEGACY));
        set_forced_gemv_variant(None);
        assert_eq!(forced_gemv_variant(), None);
        // Back to table-driven dispatch.
        assert_eq!(gemv_variant_for_shape(2048, 4096), Some(0));
    }

    #[test]
    fn variant_desc_label_format() {
        let d = GemvVariantDesc {
            warps: 8,
            rows_per_warp: 2,
            ilp: 4,
            wbytes: 4,
            min_blocks: 2,
            stage: true,
            prefetch: false,
        };
        assert_eq!(d.label(), "w8_r2_i4_v4_mb2_s1_p0");
    }

    #[test]
    fn legacy_sentinel_always_applicable() {
        assert!(gemv_variant_applicable(
            QTIP2B_GEMV_VARIANT_LEGACY,
            2048,
            4096
        ));
    }
}
