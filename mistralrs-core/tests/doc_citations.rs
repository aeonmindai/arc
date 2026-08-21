//! Parent system: ArcGate
//!
//! # The claim that nothing checks
//!
//! Every lane Arc runs compiles code or runs tests. **None of them verifies
//! that a sentence describing the code is true.** A doc can say a feature is
//! reachable only through an env var when it is also a serde config field, and
//! nothing goes red — the compiler has no opinion about prose.
//!
//! Prose cannot be checked mechanically. But the load-bearing part of Arc's
//! docs is not prose, it is **evidence**: file-and-line citations that
//! a reader is expected to follow. Those *are* mechanically checkable, and when
//! they rot the surrounding claim is unfalsifiable — the reader lands on
//! unrelated code and cannot tell whether the doc was ever right.
//!
//! This file checks the citations, not the claims. That is a deliberately
//! narrow contract, stated here so nobody mistakes a green run for "the docs
//! are true".
//!
//! ## What it checks
//!
//! For every `file.ext:N` / `file.ext:N-M` in [`SCAN_ROOTS`]:
//!
//! 1. **The path resolves** to a file in this repo (by exact component-suffix
//!    match, so `pipeline/mod.rs` finds `mistralrs-core/src/pipeline/mod.rs`).
//! 2. **The line is in range** for that file.
//! 3. **The cited symbol is still there.** Where the citation is preceded by a
//!    backticked identifier — `` `build_expert_parallel_plan`
//!    (`deepseek4.rs:2211`) ``, by far the dominant shape in these docs — the
//!    identifier must appear near the cited line.
//!
//! ## Where it looks: `.md` **and Rust comments**
//!
//! For its first life this gate read `.md` files only. That was a hole big
//! enough to drive the whole point through: Arc's densest citation corpus is
//! not the mission record, it is the **module and item docs in the crates
//! themselves** — several hundred `file.rs:N` pointers written by the same
//! agents, into the same moving code, and *none* of them were gated. A doc
//! comment that says "the mask is dropped at `sinks.rs:214`" rots exactly like
//! a `.md` one, and rots faster, because the file it cites is the file it
//! lives next to.
//!
//! So [`SCAN_ROOTS`] now names the first-party crate directories too, and
//! [`analyze`] reads `.rs` alongside `.md`.
//!
//! **In a `.rs` file only comment text is scanned** — see [`rust_comment`].
//! Code is not prose: a `.rs` line may hold a path-and-number inside a string
//! literal (this very file's [`BASELINE`] is a table of them, and its fixtures
//! build citations that are *deliberately* wrong). Scanning those would
//! manufacture [`Kind::SymbolRot`] findings, which have no waiver path, and
//! the gate would be unlandable against its own source. The extractor is
//! deliberately conservative: when it cannot tell whether a `//` is a comment
//! or string content, it treats the line as code and skips it.
//!
//! ## Drift versus rot
//!
//! These are not the same defect and are not treated the same way.
//!
//! * A citation off by a few lines is **churn**. Measured across
//!   `memory/`, roughly half of all identifier-bearing citations have drifted,
//!   some by hundreds of lines, purely because the files grew. Failing on that
//!   would make the gate un-landable and would teach people to delete
//!   citations rather than write them.
//! * A citation into a **deleted file**, a line **past the end** of a file, or
//!   a symbol that exists **nowhere in the repo** is rot. The evidence is gone.
//!
//! So [`Kind::Drift`] is reported and never fails, and [`Kind::SymbolRot`] —
//! the strongest signal, and the one with an empty baseline — hard-fails.
//!
//! ## Mode: ratchet, not hard gate (except for rot)
//!
//! The corpus was already stale when this landed. Rather than rewrite ~800
//! citations (other agents own those docs), the pre-existing violations are
//! pinned in [`BASELINE`] with a reason each. A **new** violation fails.
//! [`baseline_has_no_stale_entries`] fails when a pinned entry starts passing,
//! so the waiver list cannot rot in the other direction either.
//!
//! The overwhelming majority of pinned entries are citations into **external
//! reference trees** — sglang, vLLM, DeepSpeed, TensorRT-LLM, candle, cudarc —
//! which are not vendored here and therefore can never resolve. They are
//! individually listed rather than prefix-allowlisted because many are cited
//! by bare filename (`server_args.py`), which no prefix rule can capture.
//!
//! ## Why not a hard gate on everything
//!
//! Because a guard that cannot be satisfied gets deleted or `#[ignore]`d, and
//! Arc has already shipped three guards that passed on unfixed code. The
//! defence against *this* one becoming decorative is
//! [`fixtures_prove_each_violation_class_fails`]: it builds a synthetic repo
//! with a deliberately wrong line, a deleted file and a renamed symbol, and
//! asserts the checker reports each one. If the checker ever stops being able
//! to fail, that test goes red.
//!
//! [`citation_census_is_not_empty`] is the same idea for scope: a scan that
//! silently matched nothing would otherwise be indistinguishable from a pass.
//!
//! ## Cost
//!
//! No GPU, no network, no new dependency — `std` only, matching
//! `capability_reachability.rs` next door. It indexes ~1.6k repo paths and
//! reads file bodies lazily; the whole-repo symbol search runs only for a
//! citation whose symbol is missing from its own cited file.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs;
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Directories scanned for files carrying citations — `.md` anywhere, and
/// `.rs` **comments** in the first-party crates.
///
/// `docs/` and `research/` are deliberately **excluded for now**: 212 of their
/// 336 citations point into external reference checkouts (the v4 reference
/// audit walks DeepSeek's Python line by line), so gating them would add ~90
/// waiver rows and almost no verification. `memory/` is the mission record and
/// `arc-tools/` the runbooks — that is where a false citation does damage.
/// Adding a root here is a one-line change plus whatever waivers it brings.
///
/// The crate roots below carry the ~310 citations that live in Rust comments.
/// They are listed individually rather than as a bare `"."` for two reasons:
/// `.` would sweep in `docs/` and `research/` by the back door, and it would
/// index the vendored/generated trees that [`SKIP_DIRS`] only partly covers.
/// Crates with no citations *today* are still named, so the first one written
/// tomorrow is gated on arrival rather than on someone remembering this list.
const SCAN_ROOTS: &[&str] = &[
    "memory",
    "arc-tools",
    "mistralrs-core",
    "mistralrs-quant",
    "arc-engine",
    "arc-cli",
    "arc-cuda-graph",
    "arc-bench",
    "arc-profiler",
    "arc-turbo",
];

/// Extensions read as citation-bearing documents by [`analyze`].
///
/// Distinct from [`CITED_EXTS`] (what a citation may *point at*) and from
/// [`CODE_EXTS`] (where a symbol may *live*). A `.rs` entry here means "read
/// this file's comments", never its code — see [`rust_comment`].
const SCANNED_DOC_EXTS: &[&str] = &["md", "rs"];

/// Extensions a citation may point at. Anything else is not a citation.
const CITED_EXTS: &[&str] = &[
    "rs", "cu", "cuh", "h", "hpp", "toml", "py", "sh", "yaml", "yml", "json", "metal", "md",
];

/// Files searched when asking "does this symbol exist anywhere?" — the question
/// behind [`Kind::SymbolRot`], the one verdict with no waiver path.
///
/// Because a *false* rot report cannot be waived, this list errs wide: it names
/// every textual extension tracked in the repo, not just the ones a Rust symbol
/// is likely to live in. Adding a new text format and forgetting it here would
/// manifest as an unfixable red lane.
///
/// It is an allowlist rather than "everything that is not `.md`" because the
/// repo tracks 166 PDFs, several of them 20-30 MB; reading those to look for an
/// identifier would dominate the runtime for no possible hit.
///
/// `.md` is the one text format deliberately left out: a symbol surviving only
/// in prose is precisely the rot this is looking for.
const CODE_EXTS: &[&str] = &[
    "rs", "cu", "cuh", "h", "hpp", "c", "cc", "cpp", "toml", "py", "sh", "bash", "yaml", "yml",
    "json", "jsonl", "metal", "js", "ts", "jinja", "patch", "html", "css", "txt", "cfg", "ini",
];

/// Files larger than this are skipped by the whole-repo symbol search, so a
/// large generated or vendored blob cannot dominate the runtime.
const MAX_SEARCHED_BYTES: u64 = 2 * 1024 * 1024;

/// Directories never indexed.
const SKIP_DIRS: &[&str] = &[".git", "target", "node_modules", ".venv", "__pycache__"];

/// How far a cited identifier may have drifted before it is reported.
///
/// Docs legitimately go stale by a few lines when a file above the citation
/// grows. Fifteen is generous enough that ordinary churn is silent and tight
/// enough that a citation landing in an unrelated function is noticed.
const DRIFT_WINDOW: i64 = 15;

/// Floor on citations found, so an empty or mis-rooted scan fails loudly
/// instead of reading as a pass.
///
/// Measured at 798 when this landed, 1,244 once `.rs` comments were added —
/// the crates carry roughly a quarter of Arc's citation corpus. The floor is
/// set above the `.md`-only total on purpose: reverting [`SCANNED_DOC_EXTS`]
/// to `["md"]`, or dropping the crate [`SCAN_ROOTS`], drops the census back to
/// ~970 and this fails. A scope guard has to notice scope *shrinking*, which
/// is the exact way the Rust corpus went ungated for as long as it did.
const CITATION_FLOOR: usize = 1_100;

// ---------------------------------------------------------------------------
// Findings
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
enum Kind {
    /// The path matches no file in this repo. Either an external reference or
    /// a file that has been deleted or moved.
    Unresolved,
    /// The path resolves but the line is past the end of every candidate.
    OutOfRange,
    /// The cited identifier is not in the cited file, but does exist elsewhere
    /// in the repo — the code moved and the citation did not follow.
    SymbolMoved,
    /// The cited identifier exists nowhere in the repo. **Hard fail.**
    SymbolRot,
    /// The identifier is in the cited file but further than [`DRIFT_WINDOW`]
    /// from the cited line. Reported, never fatal.
    Drift,
}

impl Kind {
    fn label(self) -> &'static str {
        match self {
            Kind::Unresolved => "UNRESOLVED-PATH",
            Kind::OutOfRange => "LINE-OUT-OF-RANGE",
            Kind::SymbolMoved => "SYMBOL-MOVED",
            Kind::SymbolRot => "SYMBOL-ROT",
            Kind::Drift => "DRIFT",
        }
    }

    /// Whether a finding of this kind can be waived by [`BASELINE`].
    ///
    /// [`Kind::SymbolRot`] cannot: its baseline is empty and must stay empty.
    /// A symbol that exists nowhere is never explainable as churn.
    fn is_waivable(self) -> bool {
        !matches!(self, Kind::SymbolRot)
    }
}

/// One violation, carrying everything needed to find and fix it: the doc, the
/// line **in the doc**, and the citation text itself. A report that says
/// "3 citations are stale" without naming them is close to useless.
#[derive(Clone, Debug)]
struct Finding {
    doc: String,
    doc_line: usize,
    citation: String,
    cited_path: String,
    kind: Kind,
    detail: String,
}

impl Finding {
    fn render(&self) -> String {
        format!(
            "  {}:{}  [{}] `{}` — {}",
            self.doc,
            self.doc_line,
            self.kind.label(),
            self.citation,
            self.detail
        )
    }
}

// ---------------------------------------------------------------------------
// The baseline
// ---------------------------------------------------------------------------

/// A pinned, pre-existing violation.
///
/// `cite` matching is deliberately two-mode, so the waiver is no broader than
/// the debt it records:
///
/// * contains `:` — matches that **exact citation**, line numbers and all.
///   Used where the line number is the thing that is wrong, so a second bad
///   citation into the same file is not silently covered. (No example is
///   spelled out here: since this gate reads `.rs` comments, a citation
///   written in prose *about* the matcher would be scanned as a real one.)
/// * no `:` — matches **any citation of that path** in that doc. Used for
///   external references, where every line of `server_args.py` is equally
///   unresolvable and pinning each would be noise.
struct Waiver {
    doc: &'static str,
    cite: &'static str,
    kind: Kind,
    why: &'static str,
}

impl Waiver {
    fn covers(&self, f: &Finding) -> bool {
        if self.kind != f.kind || self.doc != f.doc {
            return false;
        }
        if self.cite.contains(':') {
            self.cite == f.citation
        } else {
            self.cite == f.cited_path
        }
    }
}

/// Violations present when this gate landed. Adding a row is a deliberate act
/// that needs a reason; the ratchet only ever tightens.
static BASELINE: &[Waiver] = &[
    // -- Real rot, pinned for the doc's owner to fix. Named, not hidden. ----
    Waiver {
        doc: "memory/mission/wave64-CP-arcgraph-capture-defects.md",
        cite: "device.rs:754-908",
        kind: Kind::OutOfRange,
        why: "means candle-core's device.rs; the only device.rs here is \
              arc-profiler/src/device.rs (268 lines), so a reader following this \
              lands on an unrelated file entirely",
    },
    Waiver {
        doc: "memory/mission/wave33-BK-attention-kv.md",
        cite: "deepseek4.rs:2168-2210",
        kind: Kind::SymbolMoved,
        why: "act_quant_kv_nope moved to dsv4_kv_fp8.rs (now \
              reference_act_quant_kv_nope); deepseek4.rs:2168 is EpPlacementMode",
    },
    // -- Citations into trees that are not vendored here, so they can never
    //    resolve. Listed individually rather than prefix-allowlisted because
    //    many are cited by bare filename (`server_args.py`), which no prefix
    //    rule can capture.
    Waiver {
        doc: "arc-tools/prereg/PR112_body_snapshot.md",
        cite: "cudarc/result.rs",
        kind: Kind::Unresolved,
        why: "cudarc is a registry dependency",
    },
    Waiver {
        doc: "arc-tools/quality/GPU_SESSION_RUNBOOK_8.md",
        cite: "test/manual/dsv4/test_dsv4_pro_mtp.py",
        kind: Kind::Unresolved,
        why: "external reference source, not vendored here",
    },
    Waiver {
        doc: "memory/mission/wave18-AO-bake-oom.md",
        cite: "candle-core/src/tensor.rs",
        kind: Kind::Unresolved,
        why: "candle is an out-of-tree path dependency",
    },
    Waiver {
        doc: "memory/mission/wave27-AY-decode-serialization.md",
        cite: "candle-core/src/cuda_backend/mod.rs",
        kind: Kind::Unresolved,
        why: "candle is an out-of-tree path dependency",
    },
    Waiver {
        doc: "memory/mission/wave27-AY2-tensor-ptr-fourth-copy.md",
        cite: "candle-core/src/cuda_backend/mod.rs",
        kind: Kind::Unresolved,
        why: "candle is an out-of-tree path dependency",
    },
    Waiver {
        doc: "memory/mission/wave29-BD-rung-decision.md",
        cite: "STATUS.md",
        kind: Kind::Unresolved,
        why: "agent memory file, lives outside the repo",
    },
    Waiver {
        doc: "memory/mission/wave29-BD-rung-decision.md",
        cite: "wave13-AF-cuda-beam.md",
        kind: Kind::Unresolved,
        why: "external project's documentation",
    },
    Waiver {
        doc: "memory/mission/wave29-BD-rung-decision.md",
        cite: "wave9-V-docs.md",
        kind: Kind::Unresolved,
        why: "external project's documentation",
    },
    Waiver {
        doc: "memory/mission/wave33-BK-attention-kv.md",
        cite: "research/code/06_foundation/sglang/python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py",
        kind: Kind::Unresolved,
        why: "sglang reference source, not vendored here",
    },
    Waiver {
        doc: "memory/mission/wave33-BK-attention-kv.md",
        cite: "layers/attention/dsv4/index_buf_accessor.py",
        kind: Kind::Unresolved,
        why: "external reference source, not vendored here",
    },
    Waiver {
        doc: "memory/mission/wave33-BK-attention-kv.md",
        cite: "model_executor/pool_configurator.py",
        kind: Kind::Unresolved,
        why: "external reference source, not vendored here",
    },
    Waiver {
        doc: "memory/mission/wave33-BK-attention-kv.md",
        cite: "model_executor/model_runner_kv_cache_mixin.py",
        kind: Kind::Unresolved,
        why: "external reference source, not vendored here",
    },
    Waiver {
        doc: "memory/mission/wave33-BK-attention-kv.md",
        cite: "pool_configurator.py",
        kind: Kind::Unresolved,
        why: "external reference source, not vendored here",
    },
    Waiver {
        doc: "memory/mission/wave36-BN-host-decode-loop.md",
        cite: "candle-core/src/tensor.rs",
        kind: Kind::Unresolved,
        why: "candle is an out-of-tree path dependency",
    },
    Waiver {
        doc: "memory/mission/wave36-BN-host-decode-loop.md",
        cite: "tensor.rs",
        kind: Kind::Unresolved,
        why: "candle is an out-of-tree path dependency",
    },
    Waiver {
        doc: "memory/mission/wave36-BN-host-decode-loop.md",
        cite: "cuda_backend/mod.rs",
        kind: Kind::Unresolved,
        why: "candle is an out-of-tree path dependency",
    },
    Waiver {
        doc: "memory/mission/wave43-BU-fp8-kv-bytes.md",
        cite: "cuda_backend/mod.rs",
        kind: Kind::Unresolved,
        why: "candle is an out-of-tree path dependency",
    },
    Waiver {
        doc: "memory/mission/wave44-BV-expert-parallelism.md",
        cite: "candle-core/src/cuda_backend/mod.rs",
        kind: Kind::Unresolved,
        why: "candle is an out-of-tree path dependency",
    },
    Waiver {
        doc: "memory/mission/wave44-BV-expert-parallelism.md",
        cite: "cudarc-0.19.4/src/driver/safe/core.rs",
        kind: Kind::Unresolved,
        why: "cudarc is a registry dependency",
    },
    Waiver {
        doc: "memory/mission/wave44-BV-expert-parallelism.md",
        cite: "sglang/python/sglang/srt/eplb/eplb_algorithms/deepseek.py",
        kind: Kind::Unresolved,
        why: "sglang reference source, not vendored here",
    },
    Waiver {
        doc: "memory/mission/wave44-BV-expert-parallelism.md",
        cite: "server_args.py",
        kind: Kind::Unresolved,
        why: "external reference source, not vendored here",
    },
    Waiver {
        doc: "memory/mission/wave44-BV-expert-parallelism.md",
        cite: "eplb_manager.py",
        kind: Kind::Unresolved,
        why: "external reference source, not vendored here",
    },
    Waiver {
        doc: "memory/mission/wave44-BV-expert-parallelism.md",
        cite: "expert_location_updater.py",
        kind: Kind::Unresolved,
        why: "external reference source, not vendored here",
    },
    Waiver {
        doc: "memory/mission/wave44-BV-expert-parallelism.md",
        cite: "vllm/distributed/eplb/policy/default.py",
        kind: Kind::Unresolved,
        why: "vLLM reference source, not vendored here",
    },
    Waiver {
        doc: "memory/mission/wave44-BV-expert-parallelism.md",
        cite: "config/parallel.py",
        kind: Kind::Unresolved,
        why: "external reference source, not vendored here",
    },
    Waiver {
        doc: "memory/mission/wave44-BV-expert-parallelism.md",
        cite: "docs/serving/expert_parallel_deployment.md",
        kind: Kind::Unresolved,
        why: "external project's documentation",
    },
    Waiver {
        doc: "memory/mission/wave44-BV-expert-parallelism.md",
        cite: "tensorrt_llm/examples/wide_ep/ep_load_balancer/README.md",
        kind: Kind::Unresolved,
        why: "TensorRT-LLM reference source, not vendored here",
    },
    Waiver {
        doc: "memory/mission/wave44-BV-expert-parallelism.md",
        cite: "deepspeed/moe/sharded_moe.py",
        kind: Kind::Unresolved,
        why: "DeepSpeed reference source, not vendored here",
    },
    Waiver {
        doc: "memory/mission/wave44-BV-expert-parallelism.md",
        cite: "sglang/python/sglang/srt/layers/moe/deepep_waterfill.py",
        kind: Kind::Unresolved,
        why: "sglang reference source, not vendored here",
    },
    Waiver {
        doc: "memory/mission/wave44-BV-expert-parallelism.md",
        cite: "expert_distribution.py",
        kind: Kind::Unresolved,
        why: "external reference source, not vendored here",
    },
    Waiver {
        doc: "memory/mission/wave44-BV-expert-parallelism.md",
        cite: "sglang/docs/advanced_features/dp_dpa_smg_guide.md",
        kind: Kind::Unresolved,
        why: "sglang reference source, not vendored here",
    },
    Waiver {
        doc: "memory/mission/wave44-BV-expert-parallelism.md",
        cite: "vllm/docs/serving/expert_parallel_deployment.md",
        kind: Kind::Unresolved,
        why: "vLLM reference source, not vendored here",
    },
    Waiver {
        doc: "memory/mission/wave44-BV-expert-parallelism.md",
        cite: "sglang/docs/basic_usage/deepseek_v32.md",
        kind: Kind::Unresolved,
        why: "sglang reference source, not vendored here",
    },
    Waiver {
        doc: "memory/mission/wave44-BV-expert-parallelism.md",
        cite: "sglang/python/sglang/srt/server_args.py",
        kind: Kind::Unresolved,
        why: "sglang reference source, not vendored here",
    },
    Waiver {
        doc: "memory/mission/wave44-BV-expert-parallelism.md",
        cite: "sglang/docs/advanced_features/expert_parallelism.md",
        kind: Kind::Unresolved,
        why: "sglang reference source, not vendored here",
    },
    Waiver {
        doc: "memory/mission/wave44-BV-expert-parallelism.md",
        cite: "layers/moe/utils.py",
        kind: Kind::Unresolved,
        why: "external reference source, not vendored here",
    },
    Waiver {
        doc: "memory/mission/wave44-BV-expert-parallelism.md",
        cite: "token_dispatcher/deepep.py",
        kind: Kind::Unresolved,
        why: "external reference source, not vendored here",
    },
    Waiver {
        doc: "memory/mission/wave44-BV-expert-parallelism.md",
        cite: "deepep.py",
        kind: Kind::Unresolved,
        why: "external reference source, not vendored here",
    },
    Waiver {
        doc: "memory/mission/wave44-BV-expert-parallelism.md",
        cite: "expert_parallelism.md",
        kind: Kind::Unresolved,
        why: "external project's documentation",
    },
    Waiver {
        doc: "memory/mission/wave49-BZ-fp8-kv-optin.md",
        cite: "candle-core/src/tensor_cat.rs",
        kind: Kind::Unresolved,
        why: "candle is an out-of-tree path dependency",
    },
    Waiver {
        doc: "memory/mission/wave60-CK-expert-parallel.md",
        cite: "driver/sys/mod.rs",
        kind: Kind::Unresolved,
        why: "cudarc is a registry dependency",
    },
    Waiver {
        doc: "memory/mission/wave62-CN-mtp-per-sequence-kv.md",
        cite: "v1/core/sched/scheduler.py",
        kind: Kind::Unresolved,
        why: "external reference source, not vendored here",
    },
    Waiver {
        doc: "memory/mission/wave62-CN-mtp-per-sequence-kv.md",
        cite: "speculative/eagle_worker_common.py",
        kind: Kind::Unresolved,
        why: "external reference source, not vendored here",
    },
    Waiver {
        doc: "memory/mission/wave62-CN-mtp-per-sequence-kv.md",
        cite: "batch_result_processor.py",
        kind: Kind::Unresolved,
        why: "external reference source, not vendored here",
    },
    Waiver {
        doc: "memory/mission/wave62-CN-mtp-per-sequence-kv.md",
        cite: "pyexecutor/resource_manager.py",
        kind: Kind::Unresolved,
        why: "TensorRT-LLM reference source, not vendored here",
    },
    Waiver {
        doc: "memory/mission/wave62-CN-mtp-per-sequence-kv.md",
        cite: "eagle_worker_common.py",
        kind: Kind::Unresolved,
        why: "external reference source, not vendored here",
    },
    Waiver {
        doc: "memory/mission/wave62-CN-mtp-per-sequence-kv.md",
        cite: "kv_cache_manager.py",
        kind: Kind::Unresolved,
        why: "external reference source, not vendored here",
    },
    // -- #161 mirrored four mission docs into the repo. Their citations into
    // UPSTREAM trees (SGLang, vLLM, candle, toktrie, CUTLASS/DeepGEMM, the CUDA
    // toolkit) cannot resolve here because those trees are not vendored. Bare
    // paths, so one row covers every line of a given upstream file rather than
    // pinning line numbers we do not control.
    Waiver {
        doc: "memory/mission/00_RESUME_HERE.md",
        cite: "cuda.h",
        kind: Kind::Unresolved,
        why: "CUDA driver API header, ships with the toolkit, not vendored here",
    },
    Waiver {
        doc: "memory/mission/COMPETITIVE_TEARDOWN.md",
        cite: "/fp4/mxfp4_blockwise_moe_kernel.cu",
        kind: Kind::Unresolved,
        why: "vLLM upstream, read for the competitive teardown, not vendored here",
    },
    Waiver {
        doc: "memory/mission/COMPETITIVE_TEARDOWN.md",
        cite: "/marlin_moe_wna16/dequant.h",
        kind: Kind::Unresolved,
        why: "vLLM upstream, read for the competitive teardown, not vendored here",
    },
    Waiver {
        doc: "memory/mission/COMPETITIVE_TEARDOWN.md",
        cite: "arg_groups/overrides.py",
        kind: Kind::Unresolved,
        why: "SGLang upstream, read for the competitive teardown, not vendored here",
    },
    Waiver {
        doc: "memory/mission/COMPETITIVE_TEARDOWN.md",
        cite: "arg_groups/speculative_hook.py",
        kind: Kind::Unresolved,
        why: "SGLang upstream, read for the competitive teardown, not vendored here",
    },
    Waiver {
        doc: "memory/mission/COMPETITIVE_TEARDOWN.md",
        cite: "batch_result_processor.py",
        kind: Kind::Unresolved,
        why: "SGLang upstream, read for the competitive teardown, not vendored here",
    },
    Waiver {
        doc: "memory/mission/COMPETITIVE_TEARDOWN.md",
        cite: "config/scheduler.py",
        kind: Kind::Unresolved,
        why: "SGLang upstream, read for the competitive teardown, not vendored here",
    },
    Waiver {
        doc: "memory/mission/COMPETITIVE_TEARDOWN.md",
        cite: "cuda_graph_buffer_registry.py",
        kind: Kind::Unresolved,
        why: "SGLang upstream, read for the competitive teardown, not vendored here",
    },
    Waiver {
        doc: "memory/mission/COMPETITIVE_TEARDOWN.md",
        cite: "deepseek_v4_backend.py",
        kind: Kind::Unresolved,
        why: "SGLang upstream, read for the competitive teardown, not vendored here",
    },
    Waiver {
        doc: "memory/mission/COMPETITIVE_TEARDOWN.md",
        cite: "dequant.h",
        kind: Kind::Unresolved,
        why: "vLLM upstream, read for the competitive teardown, not vendored here",
    },
    Waiver {
        doc: "memory/mission/COMPETITIVE_TEARDOWN.md",
        cite: "fused_moe/experts/deep_gemm_moe.py",
        kind: Kind::Unresolved,
        why: "vLLM upstream, read for the competitive teardown, not vendored here",
    },
    Waiver {
        doc: "memory/mission/COMPETITIVE_TEARDOWN.md",
        cite: "gpu_model_runner.py",
        kind: Kind::Unresolved,
        why: "vLLM upstream, read for the competitive teardown, not vendored here",
    },
    Waiver {
        doc: "memory/mission/COMPETITIVE_TEARDOWN.md",
        cite: "kernels/ops/attention/metadata.py",
        kind: Kind::Unresolved,
        why: "SGLang upstream, read for the competitive teardown, not vendored here",
    },
    Waiver {
        doc: "memory/mission/COMPETITIVE_TEARDOWN.md",
        cite: "managers/overlap_utils.py",
        kind: Kind::Unresolved,
        why: "SGLang upstream, read for the competitive teardown, not vendored here",
    },
    Waiver {
        doc: "memory/mission/COMPETITIVE_TEARDOWN.md",
        cite: "managers/schedule_policy.py",
        kind: Kind::Unresolved,
        why: "SGLang upstream, read for the competitive teardown, not vendored here",
    },
    Waiver {
        doc: "memory/mission/COMPETITIVE_TEARDOWN.md",
        cite: "managers/scheduler.py",
        kind: Kind::Unresolved,
        why: "SGLang upstream, read for the competitive teardown, not vendored here",
    },
    Waiver {
        doc: "memory/mission/COMPETITIVE_TEARDOWN.md",
        cite: "marlin_template.h",
        kind: Kind::Unresolved,
        why: "vLLM upstream, read for the competitive teardown, not vendored here",
    },
    Waiver {
        doc: "memory/mission/COMPETITIVE_TEARDOWN.md",
        cite: "metadata.py",
        kind: Kind::Unresolved,
        why: "SGLang upstream, read for the competitive teardown, not vendored here",
    },
    Waiver {
        doc: "memory/mission/COMPETITIVE_TEARDOWN.md",
        cite: "model_executor/runner_backend/full_cuda_graph_backend.py",
        kind: Kind::Unresolved,
        why: "SGLang upstream, read for the competitive teardown, not vendored here",
    },
    Waiver {
        doc: "memory/mission/COMPETITIVE_TEARDOWN.md",
        cite: "oracle/mxfp4.py",
        kind: Kind::Unresolved,
        why: "vLLM upstream, read for the competitive teardown, not vendored here",
    },
    Waiver {
        doc: "memory/mission/COMPETITIVE_TEARDOWN.md",
        cite: "quantization/turboquant/config.py",
        kind: Kind::Unresolved,
        why: "SGLang upstream, read for the competitive teardown, not vendored here",
    },
    Waiver {
        doc: "memory/mission/COMPETITIVE_TEARDOWN.md",
        cite: "runner/decode_cuda_graph_runner.py",
        kind: Kind::Unresolved,
        why: "SGLang upstream, read for the competitive teardown, not vendored here",
    },
    Waiver {
        doc: "memory/mission/COMPETITIVE_TEARDOWN.md",
        cite: "runner_utils/pool.py",
        kind: Kind::Unresolved,
        why: "SGLang upstream, read for the competitive teardown, not vendored here",
    },
    Waiver {
        doc: "memory/mission/COMPETITIVE_TEARDOWN.md",
        cite: "scheduler.py",
        kind: Kind::Unresolved,
        why: "SGLang upstream, read for the competitive teardown, not vendored here",
    },
    Waiver {
        doc: "memory/mission/COMPETITIVE_TEARDOWN.md",
        cite: "server_args.py",
        kind: Kind::Unresolved,
        why: "SGLang upstream, read for the competitive teardown, not vendored here",
    },
    Waiver {
        doc: "memory/mission/COMPETITIVE_TEARDOWN.md",
        cite: "speculative/adaptive_spec_params.py",
        kind: Kind::Unresolved,
        why: "SGLang upstream, read for the competitive teardown, not vendored here",
    },
    Waiver {
        doc: "memory/mission/COMPETITIVE_TEARDOWN.md",
        cite: "speculative/eagle_info.py",
        kind: Kind::Unresolved,
        why: "SGLang upstream, read for the competitive teardown, not vendored here",
    },
    Waiver {
        doc: "memory/mission/COMPETITIVE_TEARDOWN.md",
        cite: "v1/core/sched/scheduler.py",
        kind: Kind::Unresolved,
        why: "vLLM upstream, read for the competitive teardown, not vendored here",
    },
    Waiver {
        doc: "memory/mission/COMPETITIVE_TEARDOWN.md",
        cite: "vllm/config/cache.py",
        kind: Kind::Unresolved,
        why: "vLLM upstream, read for the competitive teardown, not vendored here",
    },
    Waiver {
        doc: "memory/mission/COMPETITIVE_TEARDOWN.md",
        cite: "vllm/v1/attention/backends/flash_attn.py",
        kind: Kind::Unresolved,
        why: "vLLM upstream, read for the competitive teardown, not vendored here",
    },
    Waiver {
        doc: "memory/mission/FACTS.md",
        cite: ".../dsv4/compressor.py",
        kind: Kind::Unresolved,
        why: "DeepSeek reference implementation, elided path, read outside the repo",
    },
    Waiver {
        doc: "memory/mission/FACTS.md",
        cite: ".../models/deepseek_v4.py",
        kind: Kind::Unresolved,
        why: "DeepSeek reference implementation, elided path, read outside the repo",
    },
    Waiver {
        doc: "memory/mission/FACTS.md",
        cite: "candle-core/src/cuda_backend/mod.rs",
        kind: Kind::Unresolved,
        why: "candle upstream, a registry dependency, not vendored here",
    },
    Waiver {
        doc: "memory/mission/FACTS.md",
        cite: "candle-kernels/build.rs",
        kind: Kind::Unresolved,
        why: "candle upstream, a registry dependency, not vendored here",
    },
    Waiver {
        doc: "memory/mission/FACTS.md",
        cite: "cuda_backend/mod.rs",
        kind: Kind::Unresolved,
        why: "candle upstream, a registry dependency, not vendored here",
    },
    Waiver {
        doc: "memory/mission/FACTS.md",
        cite: "silu_and_mul_masked_post_quant.cuh",
        kind: Kind::Unresolved,
        why: "DeepGEMM/CUTLASS reference kernels, read upstream, not vendored here",
    },
    Waiver {
        doc: "memory/mission/FACTS.md",
        cite: "tensor.rs",
        kind: Kind::Unresolved,
        why: "candle upstream, a registry dependency, not vendored here",
    },
    Waiver {
        doc: "memory/mission/FACTS.md",
        cite: "toktree.rs",
        kind: Kind::Unresolved,
        why: "toktrie crate, a registry dependency, not vendored here",
    },
    Waiver {
        doc: "memory/mission/FACTS.md",
        cite: "toktrie_hf_tokenizers-1.7.0/src/lib.rs",
        kind: Kind::Unresolved,
        why: "toktrie crate, a registry dependency, not vendored here",
    },
    Waiver {
        doc: "memory/mission/KERNEL_RULES.md",
        cite: "candle-core/src/cuda_backend/mod.rs",
        kind: Kind::Unresolved,
        why: "candle upstream, a registry dependency, not vendored here",
    },
    Waiver {
        doc: "memory/mission/KERNEL_RULES.md",
        cite: "gemm/group_gemm.cuh",
        kind: Kind::Unresolved,
        why: "DeepGEMM/CUTLASS reference kernels, read upstream, not vendored here",
    },
    Waiver {
        doc: "memory/mission/KERNEL_RULES.md",
        cite: "group_gemm_sm90.cuh",
        kind: Kind::Unresolved,
        why: "DeepGEMM/CUTLASS reference kernels, read upstream, not vendored here",
    },
    Waiver {
        doc: "memory/mission/KERNEL_RULES.md",
        cite: "shape.rs",
        kind: Kind::Unresolved,
        why: "candle upstream, a registry dependency, not vendored here",
    },
    // -- Rust-comment citations, added when SCAN_ROOTS grew to cover the
    //    crates (see the module docs). Every row below is a pointer into an
    //    upstream reference checkout — sglang, vLLM, TensorRT-LLM, DeepSeek's
    //    own `inference/`, the QTIP reference, candle, cudarc, toktrie — that
    //    Arc reads beside the code but does not vendor, so none of them can
    //    ever resolve here. **No `SymbolRot` accompanied this widening**: the
    //    ~310 citations in Rust comments all name symbols that still exist.
    Waiver {
        doc: "arc-engine/src/weight_schema.rs",
        cite: "deepseek_v4.py",
        kind: Kind::Unresolved,
        why: "sglang is an upstream reference checkout, not vendored here",
    },
    Waiver {
        doc: "mistralrs-core/src/attention/backends/flash.rs",
        cite: "candle-flash-attn-v3/src/lib.rs",
        kind: Kind::Unresolved,
        why: "candle is an upstream reference checkout, not vendored here",
    },
    Waiver {
        doc: "mistralrs-core/src/attention/backends/flash.rs",
        cite: "hkernel/flash_fwd_launch_template.h",
        kind: Kind::Unresolved,
        why: "flash-attention is an upstream reference checkout, not \
              vendored here",
    },
    Waiver {
        doc: "mistralrs-core/src/kv_cache/single_cache.rs",
        cite: "candle-core/src/tensor_cat.rs",
        kind: Kind::Unresolved,
        why: "candle is an upstream reference checkout, not vendored here",
    },
    Waiver {
        doc: "mistralrs-core/src/kv_sharing/evict.rs",
        cite: "evict_policy.py",
        kind: Kind::Unresolved,
        why: "sglang is an upstream reference checkout, not vendored here",
    },
    Waiver {
        doc: "mistralrs-core/src/kv_sharing/evict.rs",
        cite: "python/sglang/srt/mem_cache/evict_policy.py",
        kind: Kind::Unresolved,
        why: "sglang is an upstream reference checkout, not vendored here",
    },
    Waiver {
        doc: "mistralrs-core/src/kv_sharing/radix.rs",
        cite: "radix_cache.py",
        kind: Kind::Unresolved,
        why: "sglang is an upstream reference checkout, not vendored here",
    },
    Waiver {
        doc: "mistralrs-core/src/layers.rs",
        cite: "candle-core/src/tensor_cat.rs",
        kind: Kind::Unresolved,
        why: "candle is an upstream reference checkout, not vendored here",
    },
    Waiver {
        doc: "mistralrs-core/src/layers.rs",
        cite: "deepseek_v4.py",
        kind: Kind::Unresolved,
        why: "sglang is an upstream reference checkout, not vendored here",
    },
    Waiver {
        doc: "mistralrs-core/src/layers.rs",
        cite: "inference/model.py",
        kind: Kind::Unresolved,
        why: "DeepSeek's own inference/ reference is an upstream reference \
              checkout, not vendored here",
    },
    Waiver {
        doc: "mistralrs-core/src/layers.rs",
        cite: "srt/models/deepseek_v2.py",
        kind: Kind::Unresolved,
        why: "sglang is an upstream reference checkout, not vendored here",
    },
    Waiver {
        doc: "mistralrs-core/src/models/deepseek4.rs",
        cite: "candle-core/src/tensor_cat.rs",
        kind: Kind::Unresolved,
        why: "candle is an upstream reference checkout, not vendored here",
    },
    Waiver {
        doc: "mistralrs-core/src/models/deepseek4.rs",
        cite: "deepseek_v2.py",
        kind: Kind::Unresolved,
        why: "sglang is an upstream reference checkout, not vendored here",
    },
    Waiver {
        doc: "mistralrs-core/src/models/deepseek4.rs",
        cite: "deepseek_v4.py",
        kind: Kind::Unresolved,
        why: "sglang is an upstream reference checkout, not vendored here",
    },
    Waiver {
        doc: "mistralrs-core/src/models/deepseek4.rs",
        cite: "deepseek_v4_memory_pool.py",
        kind: Kind::Unresolved,
        why: "sglang is an upstream reference checkout, not vendored here",
    },
    Waiver {
        doc: "mistralrs-core/src/models/deepseek4.rs",
        cite: "deepseek_v4_nextn.py",
        kind: Kind::Unresolved,
        why: "sglang is an upstream reference checkout, not vendored here",
    },
    Waiver {
        doc: "mistralrs-core/src/models/deepseek4.rs",
        cite: "deepseek_v4_rope.py",
        kind: Kind::Unresolved,
        why: "sglang is an upstream reference checkout, not vendored here",
    },
    Waiver {
        doc: "mistralrs-core/src/models/deepseek4.rs",
        cite: "eagle_utils.py",
        kind: Kind::Unresolved,
        why: "sglang is an upstream reference checkout, not vendored here",
    },
    Waiver {
        doc: "mistralrs-core/src/models/deepseek4.rs",
        cite: "eagle_worker.py",
        kind: Kind::Unresolved,
        why: "sglang is an upstream reference checkout, not vendored here",
    },
    Waiver {
        doc: "mistralrs-core/src/models/deepseek4.rs",
        cite: "inference/model.py",
        kind: Kind::Unresolved,
        why: "DeepSeek's own inference/ reference is an upstream reference \
              checkout, not vendored here",
    },
    Waiver {
        doc: "mistralrs-core/src/models/deepseek4.rs",
        cite: "mem_cache/deepseek_v4_memory_pool.py",
        kind: Kind::Unresolved,
        why: "sglang is an upstream reference checkout, not vendored here",
    },
    Waiver {
        doc: "mistralrs-core/src/models/deepseek4.rs",
        cite: "model.py",
        kind: Kind::Unresolved,
        why: "DeepSeek's own inference/ reference is an upstream reference \
              checkout, not vendored here",
    },
    Waiver {
        doc: "mistralrs-core/src/models/deepseek4.rs",
        cite: "router/gate_linear.py",
        kind: Kind::Unresolved,
        why: "sglang is an upstream reference checkout, not vendored here",
    },
    Waiver {
        doc: "mistralrs-core/src/models/deepseek4.rs",
        cite: "srt/layers/deepseek_v4_rope.py",
        kind: Kind::Unresolved,
        why: "sglang is an upstream reference checkout, not vendored here",
    },
    Waiver {
        doc: "mistralrs-core/src/models/deepseek4.rs",
        cite: "srt/models/deepseek_v2.py",
        kind: Kind::Unresolved,
        why: "sglang is an upstream reference checkout, not vendored here",
    },
    Waiver {
        doc: "mistralrs-core/src/models/deepseek4.rs",
        cite: "srt/models/deepseek_v4.py",
        kind: Kind::Unresolved,
        why: "sglang is an upstream reference checkout, not vendored here",
    },
    Waiver {
        doc: "mistralrs-core/src/models/deepseek4.rs",
        cite: "topk.py",
        kind: Kind::Unresolved,
        why: "sglang is an upstream reference checkout, not vendored here",
    },
    Waiver {
        doc: "mistralrs-core/src/models/dsv4_attention.rs",
        cite: "model_executor/pool_configurator.py",
        kind: Kind::Unresolved,
        why: "TensorRT-LLM is an upstream reference checkout, not vendored \
              here",
    },
    Waiver {
        doc: "mistralrs-core/src/models/dsv4_indexer.rs",
        cite: ".../sglang/srt/models/deepseek_v4.py",
        kind: Kind::Unresolved,
        why: "sglang is an upstream reference checkout, not vendored here",
    },
    Waiver {
        doc: "mistralrs-core/src/models/dsv4_indexer.rs",
        cite: "attention/dsv4/compressor.py",
        kind: Kind::Unresolved,
        why: "sglang is an upstream reference checkout, not vendored here",
    },
    Waiver {
        doc: "mistralrs-core/src/models/dsv4_indexer.rs",
        cite: "compressor.py",
        kind: Kind::Unresolved,
        why: "sglang is an upstream reference checkout, not vendored here",
    },
    Waiver {
        doc: "mistralrs-core/src/models/dsv4_indexer.rs",
        cite: "deepseek_v4.py",
        kind: Kind::Unresolved,
        why: "sglang is an upstream reference checkout, not vendored here",
    },
    Waiver {
        doc: "mistralrs-core/src/models/dsv4_indexer.rs",
        cite: "indexer.py",
        kind: Kind::Unresolved,
        why: "sglang is an upstream reference checkout, not vendored here",
    },
    Waiver {
        doc: "mistralrs-core/src/models/dsv4_kv_fp8.rs",
        cite: "inference/model.py",
        kind: Kind::Unresolved,
        why: "DeepSeek's own inference/ reference is an upstream reference \
              checkout, not vendored here",
    },
    Waiver {
        doc: "mistralrs-core/src/models/dsv4_mhc.rs",
        cite: "deepseek_v4_nextn.py",
        kind: Kind::Unresolved,
        why: "sglang is an upstream reference checkout, not vendored here",
    },
    Waiver {
        doc: "mistralrs-core/src/models/dsv4_mhc.rs",
        cite: "sglang/srt/layers/mhc.py",
        kind: Kind::Unresolved,
        why: "sglang is an upstream reference checkout, not vendored here",
    },
    Waiver {
        doc: "mistralrs-core/src/models/dsv4_mhc.rs",
        cite: "sglang/srt/models/deepseek_v4_nextn.py",
        kind: Kind::Unresolved,
        why: "sglang is an upstream reference checkout, not vendored here",
    },
    Waiver {
        doc: "mistralrs-core/src/moe/expert_parallel.rs",
        cite: "sglang/python/sglang/srt/eplb/eplb_algorithms/deepseek.py",
        kind: Kind::Unresolved,
        why: "sglang is an upstream reference checkout, not vendored here",
    },
    Waiver {
        doc: "mistralrs-core/src/moe/expert_parallel.rs",
        cite: "tensorrt_llm/examples/wide_ep/ep_load_balancer/README.md",
        kind: Kind::Unresolved,
        why: "TensorRT-LLM is an upstream reference checkout, not vendored \
              here",
    },
    Waiver {
        doc: "mistralrs-core/src/moe/experts.rs",
        cite: "inference/model.py",
        kind: Kind::Unresolved,
        why: "DeepSeek's own inference/ reference is an upstream reference \
              checkout, not vendored here",
    },
    Waiver {
        doc: "mistralrs-core/src/moe/experts.rs",
        cite: "sglang/jit_kernel/csrc/deepseek_v4/silu_and_mul_masked_post_quant.cuh",
        kind: Kind::Unresolved,
        why: "sglang is an upstream reference checkout, not vendored here",
    },
    Waiver {
        doc: "mistralrs-core/src/moe/experts.rs",
        cite: "silu_and_mul_masked_post_quant.cuh",
        kind: Kind::Unresolved,
        why: "sglang is an upstream reference checkout, not vendored here",
    },
    Waiver {
        doc: "mistralrs-core/src/moe/experts.rs",
        cite: "sm100_fp8_fp4_mega_moe.cuh",
        kind: Kind::Unresolved,
        why: "sglang is an upstream reference checkout, not vendored here",
    },
    Waiver {
        doc: "mistralrs-core/src/moe/experts.rs",
        cite: "srt/models/deepseek_v2.py",
        kind: Kind::Unresolved,
        why: "sglang is an upstream reference checkout, not vendored here",
    },
    Waiver {
        doc: "mistralrs-core/src/pipeline/llg.rs",
        cite: "toktrie-1.7.0/src/toktree.rs",
        kind: Kind::Unresolved,
        why: "toktrie is an upstream reference checkout, not vendored here",
    },
    Waiver {
        doc: "mistralrs-core/src/pipeline/llg.rs",
        cite: "toktrie_hf_tokenizers-1.7.0/src/lib.rs",
        kind: Kind::Unresolved,
        why: "candle is an upstream reference checkout, not vendored here",
    },
    Waiver {
        doc: "mistralrs-core/src/pipeline/loaders/normal_loaders.rs",
        cite: "deepseek_v4.py",
        kind: Kind::Unresolved,
        why: "sglang is an upstream reference checkout, not vendored here",
    },
    Waiver {
        doc: "mistralrs-core/src/pipeline/mod.rs",
        cite: "cuda_backend/mod.rs",
        kind: Kind::Unresolved,
        why: "candle is an upstream reference checkout, not vendored here",
    },
    Waiver {
        doc: "mistralrs-core/src/pipeline/mod.rs",
        cite: "tensor.rs",
        kind: Kind::Unresolved,
        why: "candle is an upstream reference checkout, not vendored here",
    },
    Waiver {
        doc: "mistralrs-core/src/pipeline/mtp_pipeline.rs",
        cite: "config/speculative.py",
        kind: Kind::Unresolved,
        why: "sglang is an upstream reference checkout, not vendored here",
    },
    Waiver {
        doc: "mistralrs-core/src/pipeline/mtp_pipeline.rs",
        cite: "deepseek_v4_nextn.py",
        kind: Kind::Unresolved,
        why: "sglang is an upstream reference checkout, not vendored here",
    },
    Waiver {
        doc: "mistralrs-core/src/pipeline/mtp_pipeline.rs",
        cite: "eagle_info.py",
        kind: Kind::Unresolved,
        why: "sglang is an upstream reference checkout, not vendored here",
    },
    Waiver {
        doc: "mistralrs-core/src/pipeline/mtp_pipeline.rs",
        cite: "eagle_utils.py",
        kind: Kind::Unresolved,
        why: "sglang is an upstream reference checkout, not vendored here",
    },
    Waiver {
        doc: "mistralrs-core/src/pipeline/mtp_pipeline.rs",
        cite: "eagle_worker.py",
        kind: Kind::Unresolved,
        why: "sglang is an upstream reference checkout, not vendored here",
    },
    Waiver {
        doc: "mistralrs-core/src/pipeline/mtp_pipeline.rs",
        cite: "eagle_worker_common.py",
        kind: Kind::Unresolved,
        why: "sglang is an upstream reference checkout, not vendored here",
    },
    Waiver {
        doc: "mistralrs-core/src/pipeline/mtp_pipeline.rs",
        cite: "logits_processor.py",
        kind: Kind::Unresolved,
        why: "sglang is an upstream reference checkout, not vendored here",
    },
    Waiver {
        doc: "mistralrs-core/src/pipeline/mtp_pipeline.rs",
        cite: "pyexecutor/resource_manager.py",
        kind: Kind::Unresolved,
        why: "TensorRT-LLM is an upstream reference checkout, not vendored \
              here",
    },
    Waiver {
        doc: "mistralrs-core/src/pipeline/mtp_pipeline.rs",
        cite: "scheduler_components/batch_result_processor.py",
        kind: Kind::Unresolved,
        why: "sglang is an upstream reference checkout, not vendored here",
    },
    Waiver {
        doc: "mistralrs-core/src/pipeline/mtp_pipeline.rs",
        cite: "server_args.py",
        kind: Kind::Unresolved,
        why: "sglang is an upstream reference checkout, not vendored here",
    },
    Waiver {
        doc: "mistralrs-core/src/pipeline/mtp_pipeline.rs",
        cite: "spec_utils.py",
        kind: Kind::Unresolved,
        why: "sglang is an upstream reference checkout, not vendored here",
    },
    Waiver {
        doc: "mistralrs-core/src/pipeline/mtp_pipeline.rs",
        cite: "speculative/eagle_worker_common.py",
        kind: Kind::Unresolved,
        why: "sglang is an upstream reference checkout, not vendored here",
    },
    Waiver {
        doc: "mistralrs-core/src/pipeline/mtp_pipeline.rs",
        cite: "v1/core/sched/scheduler.py",
        kind: Kind::Unresolved,
        why: "vLLM is an upstream reference checkout, not vendored here",
    },
    Waiver {
        doc: "mistralrs-core/src/pipeline/mtp_pipeline.rs",
        cite: "v1/spec_decode/metrics.py",
        kind: Kind::Unresolved,
        why: "vLLM is an upstream reference checkout, not vendored here",
    },
    Waiver {
        doc: "mistralrs-core/src/pipeline/normal.rs",
        cite: "candle-core/src/cuda_backend/device.rs",
        kind: Kind::Unresolved,
        why: "candle is an upstream reference checkout, not vendored here",
    },
    Waiver {
        doc: "mistralrs-core/src/prefix_cacher.rs",
        cite: "block_pool.py",
        kind: Kind::Unresolved,
        why: "sglang is an upstream reference checkout, not vendored here",
    },
    Waiver {
        doc: "mistralrs-core/src/prefix_cacher.rs",
        cite: "radix_cache.py",
        kind: Kind::Unresolved,
        why: "sglang is an upstream reference checkout, not vendored here",
    },
    Waiver {
        doc: "mistralrs-core/src/sampler.rs",
        cite: "candle-kernels/src/reduce.cu",
        kind: Kind::Unresolved,
        why: "candle is an upstream reference checkout, not vendored here",
    },
    Waiver {
        doc: "mistralrs-core/tests/synthetic_load_smoke.rs",
        cite: "deepseek_v4_nextn.py",
        kind: Kind::Unresolved,
        why: "sglang is an upstream reference checkout, not vendored here",
    },
    Waiver {
        doc: "mistralrs-core/tests/synthetic_load_smoke.rs",
        cite: "eagle_worker.py",
        kind: Kind::Unresolved,
        why: "sglang is an upstream reference checkout, not vendored here",
    },
    Waiver {
        doc: "mistralrs-quant/src/cuda_peer.rs",
        cite: "src/driver/result.rs",
        kind: Kind::Unresolved,
        why: "cudarc is an upstream reference checkout, not vendored here",
    },
    Waiver {
        doc: "mistralrs-quant/src/cuda_peer.rs",
        cite: "src/driver/safe/core.rs",
        kind: Kind::Unresolved,
        why: "cudarc is an upstream reference checkout, not vendored here",
    },
    Waiver {
        doc: "mistralrs-quant/src/cuda_peer.rs",
        cite: "src/driver/sys/mod.rs",
        kind: Kind::Unresolved,
        why: "cudarc is an upstream reference checkout, not vendored here",
    },
    Waiver {
        doc: "mistralrs-quant/src/qtip/viterbi.rs",
        cite: "lib/algo/ldlq.py",
        kind: Kind::Unresolved,
        why: "the QTIP reference is an upstream reference checkout, not \
              vendored here",
    },
    Waiver {
        doc: "mistralrs-quant/src/qtip/viterbi.rs",
        cite: "lib/codebook/bitshift.py",
        kind: Kind::Unresolved,
        why: "the QTIP reference is an upstream reference checkout, not \
              vendored here",
    },
    Waiver {
        doc: "mistralrs-quant/src/qtip/viterbi.rs",
        cite: "lib/utils/math_utils.py",
        kind: Kind::Unresolved,
        why: "the QTIP reference is an upstream reference checkout, not \
              vendored here",
    },
    Waiver {
        doc: "mistralrs-quant/src/qtip/viterbi.rs",
        cite: "math_utils.py",
        kind: Kind::Unresolved,
        why: "the QTIP reference is an upstream reference checkout, not \
              vendored here",
    },
    Waiver {
        doc: "mistralrs-quant/src/qtip/viterbi.rs",
        cite: "quantize_llama/input_hessian_llama.py",
        kind: Kind::Unresolved,
        why: "the QTIP reference is an upstream reference checkout, not \
              vendored here",
    },
    Waiver {
        doc: "mistralrs-quant/src/qtip/viterbi.rs",
        cite: "research/code/01_weight_compression/qtip/lib/utils/data_utils.py",
        kind: Kind::Unresolved,
        why: "the QTIP reference is an upstream reference checkout, not \
              vendored here",
    },
    Waiver {
        doc: "mistralrs-quant/src/qtip/viterbi.rs",
        cite: "research/code/01_weight_compression/qtip/lib/utils/math_utils.py",
        kind: Kind::Unresolved,
        why: "the QTIP reference is an upstream reference checkout, not \
              vendored here",
    },
];

// ---------------------------------------------------------------------------
// Repo index
// ---------------------------------------------------------------------------

/// Repo-relative paths (always `/`-separated, including on Windows) plus a
/// component-suffix index, so `pipeline/mod.rs` resolves without the docs
/// having to spell out full paths — which, measured, they usually do not.
struct Repo {
    root: PathBuf,
    files: Vec<String>,
    by_suffix: BTreeMap<String, Vec<usize>>,
    bodies: HashMap<usize, Option<Vec<String>>>,
    code_bodies: HashMap<usize, Option<Vec<String>>>,
}

impl Repo {
    fn index(root: &Path) -> Repo {
        let mut files = Vec::new();
        walk(root, root, &mut files);
        files.sort();
        let mut by_suffix: BTreeMap<String, Vec<usize>> = BTreeMap::new();
        for (i, path) in files.iter().enumerate() {
            let parts: Vec<&str> = path.split('/').collect();
            for start in 0..parts.len() {
                by_suffix
                    .entry(parts[start..].join("/"))
                    .or_default()
                    .push(i);
            }
        }
        Repo {
            root: root.to_path_buf(),
            files,
            by_suffix,
            bodies: HashMap::new(),
            code_bodies: HashMap::new(),
        }
    }

    /// Candidate files for a cited path.
    ///
    /// Exact component-suffix match only — **no progressive stripping**. That
    /// restraint is load-bearing: stripping `candle-core/src/cuda_backend/` off
    /// `mod.rs` would resolve an external citation against dozens of unrelated
    /// in-repo `mod.rs` files, one of which is bound to be long enough to make
    /// the line check pass. That would launder rot into a green run, which is
    /// the failure mode this whole file exists to prevent.
    fn candidates(&self, cited: &str) -> &[usize] {
        self.by_suffix
            .get(cited)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    fn lines(&mut self, idx: usize) -> Option<&Vec<String>> {
        if !self.bodies.contains_key(&idx) {
            let full = self.root.join(&self.files[idx]);
            let body = fs::read_to_string(&full)
                .ok()
                .map(|s| s.lines().map(|l| l.to_string()).collect::<Vec<_>>());
            self.bodies.insert(idx, body);
        }
        self.bodies.get(&idx).and_then(|o| o.as_ref())
    }

    fn line_count(&mut self, idx: usize) -> usize {
        self.lines(idx).map(|l| l.len()).unwrap_or(0)
    }

    /// 1-based line numbers where `sym` appears as a whole word.
    fn symbol_lines(&mut self, idx: usize, sym: &str) -> Vec<usize> {
        match self.lines(idx) {
            None => Vec::new(),
            Some(body) => body
                .iter()
                .enumerate()
                .filter(|(_, l)| contains_word(l, sym))
                .map(|(i, _)| i + 1)
                .collect(),
        }
    }

    /// The file's lines with Rust comments blanked out, so a symbol that
    /// survives only in prose does not read as a symbol that still exists.
    ///
    /// Non-`.rs` files are returned unchanged: `.md` never reaches here (it is
    /// absent from [`CODE_EXTS`]) and the rest — `.py`, `.cu`, `.toml` — have
    /// their own comment syntaxes that this gate has no reason to model.
    fn code_lines(&mut self, idx: usize) -> Option<&Vec<String>> {
        if !self.code_bodies.contains_key(&idx) {
            let path = self.files[idx].clone();
            let stripped = self.lines(idx).map(|body| {
                if path.ends_with(".rs") {
                    split_rust(&body.join("\n")).code
                } else {
                    body.clone()
                }
            });
            self.code_bodies.insert(idx, stripped);
        }
        self.code_bodies.get(&idx).and_then(|o| o.as_ref())
    }

    /// Does `sym` appear as a whole word in the **code** of any code file?
    ///
    /// Only reached when a cited symbol is missing from its own cited file, so
    /// the whole-repo read stays rare.
    ///
    /// Comments are excluded — see [`RustSplit`]. Without that, a citation
    /// written in a Rust doc comment is its own witness: the symbol it names
    /// is present in the comment naming it, every [`Kind::SymbolRot`] softens
    /// to a waivable [`Kind::SymbolMoved`], and the one verdict with no waiver
    /// path stops being reachable for the entire crate corpus.
    fn symbol_exists_anywhere(&mut self, sym: &str) -> bool {
        let code: Vec<usize> = (0..self.files.len())
            .filter(|&i| has_ext(&self.files[i], CODE_EXTS))
            .filter(|&i| {
                fs::metadata(self.root.join(&self.files[i]))
                    .map(|m| m.len() <= MAX_SEARCHED_BYTES)
                    .unwrap_or(false)
            })
            .collect();
        for i in code {
            let hit = self
                .code_lines(i)
                .map(|body| body.iter().any(|l| contains_word(l, sym)))
                .unwrap_or(false);
            if hit {
                return true;
            }
        }
        false
    }
}

fn walk(root: &Path, dir: &Path, out: &mut Vec<String>) {
    let entries = match fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().to_string();
        let is_dir = entry.file_type().map(|t| t.is_dir()).unwrap_or(false);
        if is_dir {
            if SKIP_DIRS.contains(&name.as_str()) {
                continue;
            }
            walk(root, &path, out);
        } else if let Ok(rel) = path.strip_prefix(root) {
            // Normalised so Windows checkouts index the same keys as Linux.
            out.push(rel.to_string_lossy().replace('\\', "/"));
        }
    }
}

fn has_ext(path: &str, exts: &[&str]) -> bool {
    match path.rsplit_once('.') {
        Some((stem, ext)) => !stem.is_empty() && exts.contains(&ext),
        None => false,
    }
}

/// Whole-word containment: `act_quant_kv_nope` must not match inside
/// `reference_act_quant_kv_nope`.
fn contains_word(haystack: &str, needle: &str) -> bool {
    if needle.is_empty() {
        return false;
    }
    let h: Vec<char> = haystack.chars().collect();
    let n: Vec<char> = needle.chars().collect();
    if n.len() > h.len() {
        return false;
    }
    let wordish = |c: char| c.is_alphanumeric() || c == '_';
    for start in 0..=(h.len() - n.len()) {
        if h[start..start + n.len()] != n[..] {
            continue;
        }
        let before_ok = start == 0 || !wordish(h[start - 1]);
        let after_ok = start + n.len() == h.len() || !wordish(h[start + n.len()]);
        if before_ok && after_ok {
            return true;
        }
    }
    false
}

// ---------------------------------------------------------------------------
// Citation extraction
// ---------------------------------------------------------------------------

/// One `path.ext:N[-M]` occurrence, with any identifier that introduced it.
#[derive(Debug, Clone)]
struct Citation {
    text: String,
    path: String,
    start: usize,
    end: usize,
    symbol: Option<String>,
}

fn is_path_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | '/' | '+' | '-')
}

/// Rust keywords, primitives and literals that appear backticked in prose and
/// are never useful as a "does this symbol still exist" probe.
const NOT_IDENTIFIERS: &[&str] = &[
    "false", "true", "None", "Some", "Ok", "Err", "self", "Self", "fn", "let", "mut", "return",
    "if", "else", "match", "impl", "struct", "enum", "type", "use", "pub", "const", "static",
    "unsafe", "async", "await", "dyn", "ref", "as", "in", "for", "while", "loop", "break",
    "continue", "move", "where", "trait", "mod", "crate", "super", "box", "Vec", "String",
    "Option", "Result", "usize", "u32", "u64", "i32", "i64", "f32", "f64", "bool", "str",
];

fn looks_like_git_sha(s: &str) -> bool {
    (7..=40).contains(&s.len())
        && s.chars()
            .all(|c| c.is_ascii_digit() || matches!(c, 'a'..='f'))
}

fn valid_identifier(s: &str) -> bool {
    if s.is_empty() {
        return false;
    }
    s.split("::").all(|seg| {
        let mut cs = seg.chars();
        match cs.next() {
            Some(c) if c.is_ascii_alphabetic() || c == '_' => {
                cs.all(|c| c.is_ascii_alphanumeric() || c == '_')
            }
            _ => false,
        }
    })
}

/// Pull the backticked identifier that introduces a citation.
///
/// Targets the shape these docs overwhelmingly use:
/// `` `MoEExperts::new_expert_parallel` (`deepseek4.rs:2292`) ``, tolerating
/// the bold/list punctuation that surrounds it. Returns the **last `::`
/// segment**, since `Moe::new` is written `fn new` at the definition site.
fn identifier_before(chars: &[char], cite_start: usize) -> Option<String> {
    let mut i = cite_start;
    // The citation's own opening backtick, if it has one.
    if i > 0 && chars[i - 1] == '`' {
        i -= 1;
    }
    // Separator punctuation: " (", "** (", ", (", "—" and friends.
    let mut seps = 0;
    while i > 0 && seps < 6 && matches!(chars[i - 1], ' ' | '(' | '[' | ',' | '*' | '—' | '-' | ':')
    {
        i -= 1;
        seps += 1;
    }
    if i == 0 || chars[i - 1] != '`' {
        return None;
    }
    let close = i - 1;
    let open = chars[..close].iter().rposition(|&c| c == '`')?;
    let full: String = chars[open + 1..close].iter().collect();
    if !valid_identifier(&full) || looks_like_git_sha(&full) {
        return None;
    }
    let last = full.rsplit("::").next().unwrap_or(&full).to_string();
    if last.len() < 2 || NOT_IDENTIFIERS.contains(&last.as_str()) {
        return None;
    }
    Some(last)
}

/// A Rust file cut into the two halves this gate treats differently.
///
/// Both halves matter and they matter in *opposite* directions, which is why
/// one pass produces both:
///
/// * [`RustSplit::comments`] is where citations are allowed to live. Scanning
///   code instead would manufacture [`Kind::SymbolRot`] out of string
///   literals — this file's own [`BASELINE`] is a table of citation-shaped
///   strings, and rot has no waiver path, so the gate would be unlandable
///   against its own source.
/// * [`RustSplit::code`] is where a symbol is allowed to *exist*. Once `.rs`
///   files became documents, `symbol_exists_anywhere` started finding symbols
///   in the very comment that cited them — every rot downgraded to the
///   waivable [`Kind::SymbolMoved`] and the one unwaivable verdict quietly
///   stopped being reachable for the whole new corpus. Blanking comments
///   restores it, and it is the same rule `.md` already gets from
///   [`CODE_EXTS`]: a symbol surviving only in prose is exactly the rot this
///   is looking for.
struct RustSplit {
    /// `(1-based line, comment text)`, one entry per comment span.
    comments: Vec<(usize, String)>,
    /// The body line for line, with every comment span blanked to spaces.
    code: Vec<String>,
}

/// Split Rust source into comment spans and comment-free code.
///
/// A line-at-a-time scanner cannot do this: a multi-line raw string
/// (`r#"…"#`), a `\`-continued string literal and a `/* */` block all carry
/// state across newlines, and a `//!` sitting inside any of them is not a
/// comment. Getting that wrong is not cosmetic — the first fixture written for
/// this feature was a multi-line raw string full of deliberately-broken
/// citations, and a per-line scanner reported every one of them as real.
///
/// Handled: `//` line comments (so `///` and `//!` too), nested `/* */`
/// blocks, `"…"` with `\` escapes, `r"…"` / `r#…#"…"#…` raw strings including
/// the `b`-prefixed forms, and `'x'` char literals kept distinct from `'a`
/// lifetimes.
fn split_rust(body: &str) -> RustSplit {
    #[derive(Clone, Copy)]
    enum St {
        Code,
        Line,
        Block(usize),
        Str,
        Raw(usize),
    }

    let chars: Vec<char> = body.chars().collect();
    let mut st = St::Code;
    let mut comments: Vec<(usize, String)> = Vec::new();
    let mut code: Vec<String> = Vec::new();
    let mut line_no = 1usize;
    let mut cur_code = String::new();
    let mut cur_comment = String::new();
    let mut i = 0usize;

    let flush_comment = |buf: &mut String, line: usize, out: &mut Vec<(usize, String)>| {
        if !buf.trim().is_empty() {
            out.push((line, std::mem::take(buf)));
        } else {
            buf.clear();
        }
    };

    while i < chars.len() {
        let c = chars[i];
        if c == '\n' {
            if matches!(st, St::Line) {
                st = St::Code;
            }
            flush_comment(&mut cur_comment, line_no, &mut comments);
            code.push(std::mem::take(&mut cur_code));
            line_no += 1;
            i += 1;
            continue;
        }
        match st {
            St::Line => {
                cur_comment.push(c);
                cur_code.push(' ');
                i += 1;
            }
            St::Block(depth) => {
                if c == '*' && chars.get(i + 1) == Some(&'/') {
                    st = if depth == 1 {
                        St::Code
                    } else {
                        St::Block(depth - 1)
                    };
                    cur_comment.push_str("*/");
                    cur_code.push_str("  ");
                    i += 2;
                } else if c == '/' && chars.get(i + 1) == Some(&'*') {
                    st = St::Block(depth + 1);
                    cur_comment.push_str("/*");
                    cur_code.push_str("  ");
                    i += 2;
                } else {
                    cur_comment.push(c);
                    cur_code.push(' ');
                    i += 1;
                }
            }
            St::Str => {
                cur_code.push(c);
                if c == '\\' {
                    if let Some(&n) = chars.get(i + 1) {
                        // A `\` at end of line continues the literal; the
                        // newline arm above must still see the `\n`.
                        if n != '\n' {
                            cur_code.push(n);
                            i += 2;
                            continue;
                        }
                    }
                    i += 1;
                    continue;
                }
                if c == '"' {
                    st = St::Code;
                }
                i += 1;
            }
            St::Raw(hashes) => {
                cur_code.push(c);
                if c == '"' {
                    let closed = (1..=hashes).all(|k| chars.get(i + k) == Some(&'#'));
                    if closed {
                        for _ in 0..hashes {
                            cur_code.push('#');
                        }
                        st = St::Code;
                        i += 1 + hashes;
                        continue;
                    }
                }
                i += 1;
            }
            St::Code => {
                if c == '/' && chars.get(i + 1) == Some(&'/') {
                    st = St::Line;
                    cur_comment.push_str("//");
                    cur_code.push_str("  ");
                    i += 2;
                } else if c == '/' && chars.get(i + 1) == Some(&'*') {
                    st = St::Block(1);
                    cur_comment.push_str("/*");
                    cur_code.push_str("  ");
                    i += 2;
                } else if c == '"' {
                    st = St::Str;
                    cur_code.push(c);
                    i += 1;
                } else if c == 'r' && raw_open(&chars, i).is_some() {
                    let (hashes, span) = raw_open(&chars, i).expect("checked");
                    st = St::Raw(hashes);
                    cur_code.extend(chars[i..i + span].iter());
                    i += span;
                } else if c == '\'' {
                    // `'x'` / `'\n'` are literals; `'a` in `&'a str` is not.
                    let escaped =
                        chars.get(i + 1) == Some(&'\\') && chars.get(i + 3) == Some(&'\'');
                    let plain = chars.get(i + 1) != Some(&'\\') && chars.get(i + 2) == Some(&'\'');
                    let span = if escaped {
                        4
                    } else if plain {
                        3
                    } else {
                        1
                    };
                    cur_code.extend(chars[i..(i + span).min(chars.len())].iter());
                    i += span;
                } else {
                    cur_code.push(c);
                    i += 1;
                }
            }
        }
    }
    flush_comment(&mut cur_comment, line_no, &mut comments);
    if !cur_code.is_empty() {
        code.push(cur_code);
    }
    RustSplit { comments, code }
}

/// If a raw-string literal opens at `i` (the `r`), its hash count and the
/// width of the opening token. `br"…"` reaches here with `i` on the `r`.
fn raw_open(chars: &[char], i: usize) -> Option<(usize, usize)> {
    let mut k = i + 1;
    let mut hashes = 0usize;
    while chars.get(k) == Some(&'#') {
        hashes += 1;
        k += 1;
    }
    if chars.get(k) == Some(&'"') {
        Some((hashes, k + 1 - i))
    } else {
        None
    }
}

/// Every `(line, text)` in a document that a citation may live in.
///
/// `.md` is prose end to end. `.rs` is prose only inside comments.
fn prose_spans(doc: &str, body: &str) -> Vec<(usize, String)> {
    if doc.ends_with(".rs") {
        split_rust(body).comments
    } else {
        body.lines()
            .enumerate()
            .map(|(n, l)| (n + 1, l.to_string()))
            .collect()
    }
}

/// Extract every citation on one line of a doc.
fn citations_in_line(line: &str) -> Vec<Citation> {
    let chars: Vec<char> = line.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        if chars[i] != ':' || i + 1 >= chars.len() || !chars[i + 1].is_ascii_digit() {
            i += 1;
            continue;
        }
        // Walk left for the path.
        let mut j = i;
        while j > 0 && is_path_char(chars[j - 1]) {
            j -= 1;
        }
        let path: String = chars[j..i].iter().collect();
        if !has_ext(&path, CITED_EXTS) {
            i += 1;
            continue;
        }
        // Start line.
        let mut k = i + 1;
        let mut start = 0usize;
        while k < chars.len() && chars[k].is_ascii_digit() {
            start = start
                .saturating_mul(10)
                .saturating_add(chars[k] as usize - '0' as usize);
            k += 1;
        }
        // Optional end of range.
        let mut end = start;
        if k + 1 < chars.len() && chars[k] == '-' && chars[k + 1].is_ascii_digit() {
            let mut k2 = k + 1;
            let mut e = 0usize;
            while k2 < chars.len() && chars[k2].is_ascii_digit() {
                e = e
                    .saturating_mul(10)
                    .saturating_add(chars[k2] as usize - '0' as usize);
                k2 += 1;
            }
            end = e;
            k = k2;
        }
        let text: String = chars[j..k].iter().collect();
        out.push(Citation {
            text,
            path,
            start,
            end,
            symbol: identifier_before(&chars, j),
        });
        i = k;
    }
    out
}

// ---------------------------------------------------------------------------
// The check
// ---------------------------------------------------------------------------

struct Census {
    total: usize,
    ambiguous: usize,
    with_symbol: usize,
    findings: Vec<Finding>,
}

fn analyze(root: &Path, scan_roots: &[&str]) -> Census {
    let mut repo = Repo::index(root);
    let mut docs: Vec<String> = Vec::new();
    for scan in scan_roots {
        let base = root.join(scan);
        let mut found = Vec::new();
        walk(root, &base, &mut found);
        docs.extend(found.into_iter().filter(|p| has_ext(p, SCANNED_DOC_EXTS)));
    }
    docs.sort();
    // So a nested scan root ("memory" plus "memory/mission") counts each doc
    // once instead of inflating the census and duplicating every finding.
    docs.dedup();

    let mut census = Census {
        total: 0,
        ambiguous: 0,
        with_symbol: 0,
        findings: Vec::new(),
    };

    for doc in &docs {
        let body = match fs::read_to_string(root.join(doc)) {
            Ok(b) => b,
            Err(_) => continue,
        };
        for (doc_line, prose) in prose_spans(doc, &body) {
            for c in citations_in_line(&prose) {
                census.total += 1;
                let mk = |kind: Kind, detail: String| Finding {
                    doc: doc.clone(),
                    doc_line,
                    citation: c.text.clone(),
                    cited_path: c.path.clone(),
                    kind,
                    detail,
                };

                let cands: Vec<usize> = repo.candidates(&c.path).to_vec();
                if cands.is_empty() {
                    census.findings.push(mk(
                        Kind::Unresolved,
                        "no file in this repo matches that path".to_string(),
                    ));
                    continue;
                }
                if cands.len() > 1 {
                    census.ambiguous += 1;
                }

                // In range for *any* candidate is enough: a bare `normal.rs`
                // legitimately matches two files and the doc means whichever
                // one the line fits.
                let hi = c.end.max(c.start);
                let in_range: Vec<usize> = cands
                    .iter()
                    .copied()
                    .filter(|&i| c.start >= 1 && hi <= repo.line_count(i))
                    .collect();
                if in_range.is_empty() {
                    let detail = cands
                        .iter()
                        .map(|&i| {
                            let n = repo.line_count(i);
                            format!("{} has {} lines", repo.files[i], n)
                        })
                        .collect::<Vec<_>>()
                        .join("; ");
                    census.findings.push(mk(Kind::OutOfRange, detail));
                    continue;
                }

                let sym = match &c.symbol {
                    Some(s) => s.clone(),
                    None => continue,
                };
                census.with_symbol += 1;

                let mut nearest: Option<(i64, String, usize)> = None;
                for &i in &in_range {
                    for hit in repo.symbol_lines(i, &sym) {
                        let d = if hit >= c.start && hit <= hi {
                            0
                        } else if hit < c.start {
                            (c.start - hit) as i64
                        } else {
                            (hit - hi) as i64
                        };
                        if nearest.as_ref().is_none_or(|(bd, _, _)| d < *bd) {
                            nearest = Some((d, repo.files[i].clone(), hit));
                        }
                    }
                }

                match nearest {
                    Some((d, _, _)) if d <= DRIFT_WINDOW => {}
                    Some((d, file, hit)) => census.findings.push(mk(
                        Kind::Drift,
                        format!("`{sym}` is at {file}:{hit}, {d} lines from the citation"),
                    )),
                    None => {
                        let finding = if repo.symbol_exists_anywhere(&sym) {
                            mk(
                                Kind::SymbolMoved,
                                format!("`{sym}` is not in the cited file; it exists elsewhere"),
                            )
                        } else {
                            mk(
                                Kind::SymbolRot,
                                format!("`{sym}` exists nowhere in this repo"),
                            )
                        };
                        census.findings.push(finding);
                    }
                }
            }
        }
    }
    census
}

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("mistralrs-core has a parent directory")
        .to_path_buf()
}

fn census() -> Census {
    analyze(&repo_root(), SCAN_ROOTS)
}

fn unwaived(findings: &[Finding]) -> Vec<&Finding> {
    findings
        .iter()
        .filter(|f| f.kind != Kind::Drift)
        .filter(|f| !f.kind.is_waivable() || !BASELINE.iter().any(|w| w.covers(f)))
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// The scan reached the corpus. Without this, deleting `memory/` or renaming a
/// scan root would turn every other test here green by matching nothing —
/// the "guard that cannot fail" shape Arc has already shipped three times.
#[test]
fn citation_census_is_not_empty() {
    let c = census();
    for scan in SCAN_ROOTS {
        assert!(
            repo_root().join(scan).is_dir(),
            "SCAN_ROOTS names `{scan}`, which is not a directory. Either restore it \
             or remove it from SCAN_ROOTS deliberately."
        );
    }
    let drift = c.findings.iter().filter(|f| f.kind == Kind::Drift).count();
    let gating = c.findings.len() - drift;
    let live = unwaived(&c.findings).len();
    println!(
        "doc citations: {} scanned across {:?}, {} resolved to more than one file, \
         {} carry a checkable identifier | gating findings: {} ({} pinned in BASELINE, \
         {} live) | drift (reported, never fatal): {}",
        c.total,
        SCAN_ROOTS,
        c.ambiguous,
        c.with_symbol,
        gating,
        gating - live,
        live,
        drift,
    );
    assert!(
        c.total >= CITATION_FLOOR,
        "only {} citations found across {:?}, expected at least {}. \
         The scanner or the scan roots are broken — a citation gate that matches \
         nothing passes trivially.",
        c.total,
        SCAN_ROOTS,
        CITATION_FLOOR
    );
}

/// **The ratchet.** Every citation must resolve to a real file at a real line,
/// and cited symbols must still be in the file they are cited from. Pre-existing
/// violations are pinned in [`BASELINE`]; a new one fails here.
#[test]
fn doc_citations_do_not_rot() {
    let c = census();
    let bad = unwaived(&c.findings);
    if bad.is_empty() {
        return;
    }
    let mut by_kind: BTreeMap<Kind, Vec<&Finding>> = BTreeMap::new();
    for f in &bad {
        by_kind.entry(f.kind).or_default().push(f);
    }
    let mut msg = format!("\n{} doc citation(s) do not check out:\n", bad.len());
    for (kind, list) in &by_kind {
        msg.push_str(&format!("\n{} ({}):\n", kind.label(), list.len()));
        for f in list {
            msg.push_str(&f.render());
            msg.push('\n');
        }
    }
    msg.push_str(
        "\nFix the citation (preferred), or — if it points into an external \
         reference tree that is not vendored here — add a Waiver to BASELINE \
         with a reason.\n",
    );
    panic!("{msg}");
}

/// A cited symbol that exists **nowhere** in the repo is rot, not drift, and
/// has no waiver path: [`Kind::SymbolRot`] is not waivable. The baseline for
/// this class is empty and must stay empty.
#[test]
fn cited_symbols_still_exist_somewhere() {
    let c = census();
    let rot: Vec<&Finding> = c
        .findings
        .iter()
        .filter(|f| f.kind == Kind::SymbolRot)
        .collect();
    assert!(
        rot.is_empty(),
        "\n{} citation(s) name a symbol that exists nowhere in this repo. \
         The evidence behind the surrounding claim is gone:\n{}\n",
        rot.len(),
        rot.iter()
            .map(|f| f.render())
            .collect::<Vec<_>>()
            .join("\n")
    );
}

/// The waiver list cannot rot in the other direction: an entry whose violation
/// has been fixed must be removed, or the baseline silently re-opens the hole
/// it was pinning.
#[test]
fn baseline_has_no_stale_entries() {
    let c = census();
    let stale: Vec<String> = BASELINE
        .iter()
        .filter(|w| !c.findings.iter().any(|f| w.covers(f)))
        .map(|w| format!("  {} — `{}` [{}]", w.doc, w.cite, w.kind.label()))
        .collect();
    assert!(
        stale.is_empty(),
        "\nBASELINE entries no longer match any violation — the citation was \
         fixed or the doc moved. Delete them:\n{}\n",
        stale.join("\n")
    );
}

/// Duplicate waivers would make the ratchet ambiguous and hide a second debt
/// behind the first.
#[test]
fn baseline_is_a_set() {
    let mut seen = BTreeSet::new();
    let dupes: Vec<String> = BASELINE
        .iter()
        .filter(|w| !seen.insert((w.doc, w.cite, w.kind)))
        .map(|w| format!("{} :: {}", w.doc, w.cite))
        .collect();
    assert!(dupes.is_empty(), "duplicate BASELINE entries: {dupes:?}");
}

/// Drift is reported, never fatal — see the module docs. This test exists so
/// the census is printed with `--nocapture` and so the drift set stays visible
/// rather than being silently dropped.
#[test]
fn drift_is_reported_but_not_fatal() {
    let c = census();
    let drift: Vec<&Finding> = c
        .findings
        .iter()
        .filter(|f| f.kind == Kind::Drift)
        .collect();
    println!(
        "{} citation(s) have drifted beyond ±{DRIFT_WINDOW} lines:",
        drift.len()
    );
    for f in &drift {
        println!("{}", f.render());
    }
}

// ---------------------------------------------------------------------------
// Proof that the checker can fail
// ---------------------------------------------------------------------------

struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Scratch {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "arc_doc_citations_{}_{}_{:?}",
            tag,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        let _ = fs::remove_dir_all(&p);
        fs::create_dir_all(p.join("src")).expect("scratch src");
        fs::create_dir_all(p.join("memory")).expect("scratch memory");
        Scratch(p)
    }
    fn write(&self, rel: &str, body: &str) {
        let path = self.0.join(rel);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).expect("scratch parent");
        }
        fs::write(path, body).expect("scratch write");
    }
    fn run(&self) -> Census {
        analyze(&self.0, &["memory"])
    }
    fn run_roots(&self, roots: &[&str]) -> Census {
        analyze(&self.0, roots)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// A 40-line source file with `alpha_beta` on line 10 and nothing else notable.
fn fixture_source() -> String {
    let mut s = String::new();
    for i in 1..=40 {
        if i == 10 {
            s.push_str("pub fn alpha_beta() -> usize { 7 }\n");
        } else {
            s.push_str(&format!("// filler line {i}\n"));
        }
    }
    s
}

/// **The demonstration that this gate is not decorative.**
///
/// Three ways a citation rots — a wrong line, a deleted file, a renamed symbol
/// — each asserted to produce a finding of the right kind, against a synthetic
/// repo built in a temp dir. If the checker ever loses the ability to fail,
/// this test is what goes red.
#[test]
fn fixtures_prove_each_violation_class_fails() {
    let s = Scratch::new("classes");
    s.write("src/foo.rs", &fixture_source());
    s.write(
        "memory/claims.md",
        "\
1. wrong line: `alpha_beta` (`src/foo.rs:999`) is past the end.
2. deleted file: `alpha_beta` (`src/deleted.rs:12`) no longer exists.
3. renamed symbol: `gamma_delta_gone` (`src/foo.rs:10`) was renamed away.
",
    );
    let c = s.run();
    assert_eq!(c.total, 3, "fixture should yield exactly 3 citations");

    let kinds: BTreeSet<Kind> = c.findings.iter().map(|f| f.kind).collect();
    let rendered = c
        .findings
        .iter()
        .map(|f| f.render())
        .collect::<Vec<_>>()
        .join("\n");

    assert!(
        kinds.contains(&Kind::OutOfRange),
        "a line past the end of the file must be caught:\n{rendered}"
    );
    assert!(
        kinds.contains(&Kind::Unresolved),
        "a citation into a deleted file must be caught:\n{rendered}"
    );
    assert!(
        kinds.contains(&Kind::SymbolRot),
        "a symbol that exists nowhere must be caught:\n{rendered}"
    );

    // And each is attributed precisely: doc, line in the doc, citation text.
    let oor = c
        .findings
        .iter()
        .find(|f| f.kind == Kind::OutOfRange)
        .expect("out-of-range finding");
    assert_eq!(oor.doc, "memory/claims.md");
    assert_eq!(oor.doc_line, 1);
    assert_eq!(oor.citation, "src/foo.rs:999");
    assert!(
        oor.detail.contains("40 lines"),
        "detail should state the real length, got: {}",
        oor.detail
    );

    let missing = c
        .findings
        .iter()
        .find(|f| f.kind == Kind::Unresolved)
        .expect("unresolved finding");
    assert_eq!(missing.doc_line, 2);
    assert_eq!(missing.citation, "src/deleted.rs:12");

    let rot = c
        .findings
        .iter()
        .find(|f| f.kind == Kind::SymbolRot)
        .expect("symbol-rot finding");
    assert_eq!(rot.doc_line, 3);
    assert!(rot.detail.contains("gamma_delta_gone"));

    // Rot is not waivable — no BASELINE row could ever silence it.
    assert!(!Kind::SymbolRot.is_waivable());
}

/// **Proof that reading `.rs` is not decorative, and that it reads only
/// comments.**
///
/// Two failures are possible when a citation gate grows a second file type,
/// and both are silent: it can scan `.rs` and find nothing (the corpus stays
/// ungated while the census looks bigger), or it can scan `.rs` code and
/// manufacture unwaivable [`Kind::SymbolRot`] out of string literals. The
/// fixture below contains one of each shape, plus the three lines that
/// [`rust_comment`] has to get right — an escaped quote, a raw string holding
/// `//`, and a lifetime — and asserts exactly which citations come back.
#[test]
fn rust_comments_are_scanned_and_code_is_not() {
    let s = Scratch::new("rustcomments");
    s.write("src/foo.rs", &fixture_source());
    s.write(
        "src/claims.rs",
        r####"
//! Module doc: `alpha_beta` (`src/foo.rs:10`) is the entry point.
/// Item doc citing a range: `alpha_beta` (`src/foo.rs:5-15`).
const CITE: &str = "`ghost_symbol` (src/foo.rs:10)"; // a literal, not a claim
let msg = "he said \"see src/foo.rs:10\""; // escaped quotes stay in the string
let url = r#"https://x/y//src/foo.rs:10"#; // a raw string holding a comment marker
fn f<'a>(x: &'a str) {} // lifetime, then a real one: `alpha_beta` (`src/foo.rs:10`)
"####,
    );
    let c = s.run_roots(&["src"]);

    // Every citation the extractor found, in order.
    let body = fs::read_to_string(s.0.join("src/claims.rs")).expect("fixture readable");
    let found: Vec<String> = prose_spans("src/claims.rs", &body)
        .into_iter()
        .flat_map(|(_, p)| citations_in_line(&p).into_iter().map(|c| c.text))
        .collect();
    assert_eq!(
        found,
        vec![
            "src/foo.rs:10",   // module doc
            "src/foo.rs:5-15", // item doc
            "src/foo.rs:10",   // trailing comment on the lifetime line
        ],
        "comment text must be scanned and code text must not"
    );

    // And end to end: nothing in that file is a violation. If string literals
    // leaked in, `ghost_symbol` would be an unwaivable SymbolRot.
    assert!(
        c.findings.is_empty(),
        "a .rs fixture whose comments all check out must be clean:\n{}",
        c.findings
            .iter()
            .map(|f| f.render())
            .collect::<Vec<_>>()
            .join("\n")
    );
    assert_eq!(c.total, 3, "3 citations live in comments in that fixture");
}

/// A `.rs` file whose **comment** carries a rotten citation is caught, so the
/// widening is a gate and not just a bigger census.
#[test]
fn rust_comment_citations_are_gated() {
    let s = Scratch::new("rustgate");
    s.write("src/foo.rs", &fixture_source());
    s.write(
        "src/claims.rs",
        "//! `gamma_delta_gone` (`src/foo.rs:10`) does the work.\n\
         //! Also `alpha_beta` (`src/foo.rs:999`).\n",
    );
    let c = s.run_roots(&["src"]);
    let kinds: BTreeSet<Kind> = c.findings.iter().map(|f| f.kind).collect();
    assert!(
        kinds.contains(&Kind::SymbolRot),
        "a dead symbol cited from a Rust comment must be rot: {:?}",
        c.findings.iter().map(|f| f.render()).collect::<Vec<_>>()
    );
    assert!(
        kinds.contains(&Kind::OutOfRange),
        "a past-the-end line cited from a Rust comment must be out of range: {:?}",
        c.findings.iter().map(|f| f.render()).collect::<Vec<_>>()
    );
}

/// The other half of "prove it red": the same checker must stay **green** on
/// citations that are merely churned. A gate that fires on everything is as
/// useless as one that fires on nothing.
#[test]
fn fixtures_tolerate_ordinary_churn() {
    let s = Scratch::new("churn");
    s.write("src/foo.rs", &fixture_source());
    s.write(
        "memory/claims.md",
        "\
exact hit: `alpha_beta` (`src/foo.rs:10`)
within the window: `alpha_beta` (`src/foo.rs:20`)
range spanning it: `alpha_beta` (`src/foo.rs:5-15`)
no identifier at all: (`src/foo.rs:40`)
bold and listed: - **`alpha_beta`** (`src/foo.rs:12`)
",
    );
    let c = s.run();
    assert_eq!(c.total, 5, "fixture should yield exactly 5 citations");
    assert!(
        c.findings.is_empty(),
        "ordinary churn must not fail:\n{}",
        c.findings
            .iter()
            .map(|f| f.render())
            .collect::<Vec<_>>()
            .join("\n")
    );
    assert_eq!(
        c.with_symbol, 4,
        "four of the five citations carry a backticked identifier"
    );
}

/// Drift past the window is reported as [`Kind::Drift`] and not as rot — the
/// distinction the whole design rests on.
#[test]
fn fixtures_separate_drift_from_rot() {
    let s = Scratch::new("drift");
    s.write("src/foo.rs", &fixture_source());
    s.write("src/other.rs", "pub fn moved_away() {}\n");
    s.write(
        "memory/claims.md",
        "\
drifted: `alpha_beta` (`src/foo.rs:38`)
moved: `moved_away` (`src/foo.rs:10`)
",
    );
    let c = s.run();
    let kinds: Vec<Kind> = c.findings.iter().map(|f| f.kind).collect();
    assert!(
        kinds.contains(&Kind::Drift),
        "28 lines away is drift, got {kinds:?}"
    );
    assert!(
        kinds.contains(&Kind::SymbolMoved),
        "a symbol living in another file is SymbolMoved, not SymbolRot: {kinds:?}"
    );
    assert!(
        !kinds.contains(&Kind::SymbolRot),
        "must not report rot for a symbol that still exists: {kinds:?}"
    );
    // Drift never reaches the ratchet.
    assert!(unwaived(&c.findings).iter().all(|f| f.kind != Kind::Drift));
}

/// The extractor's edges, pinned so a future tweak cannot quietly shrink what
/// counts as a citation.
#[test]
fn extractor_edges() {
    // URLs and version strings are not citations.
    assert!(citations_in_line("see http://example.com:8080/x").is_empty());
    assert!(citations_in_line("version 0.7.1-alpha.1:12").is_empty());
    assert!(citations_in_line("ratio 3.5:1 held").is_empty());

    // Ranges, dotted paths and leading-dot paths all parse.
    let c = citations_in_line("(`mistralrs-core/src/pipeline/mod.rs:1088-1099`)");
    assert_eq!(c.len(), 1);
    assert_eq!(c[0].path, "mistralrs-core/src/pipeline/mod.rs");
    assert_eq!((c[0].start, c[0].end), (1088, 1099));

    let c = citations_in_line("in `.github/workflows/ci.yml:42`");
    assert_eq!(c.len(), 1);
    assert_eq!(c[0].path, ".github/workflows/ci.yml");

    // Only the first of a comma list is a citation; the rest are bare numbers.
    let c = citations_in_line("(`deepseek4.rs:1432,1451,1484`)");
    assert_eq!(c.len(), 1);
    assert_eq!(c[0].start, 1432);

    // Identifier capture, including the `Type::method` -> `method` reduction.
    let c = citations_in_line("`MoEExperts::new_expert_parallel` (`deepseek4.rs:2292`)");
    assert_eq!(c[0].symbol.as_deref(), Some("new_expert_parallel"));

    // Git SHAs and keywords are not identifiers.
    assert_eq!(
        citations_in_line("`f76b6af0a` (`bitshift.rs:569`)")[0].symbol,
        None
    );
    assert_eq!(
        citations_in_line("returns `false` (`normal_loaders.rs:3231`)")[0].symbol,
        None
    );

    // Whole-word matching, the thing that keeps SymbolMoved honest.
    assert!(contains_word("fn act_quant_kv_nope(", "act_quant_kv_nope"));
    assert!(!contains_word(
        "fn reference_act_quant_kv_nope(",
        "act_quant_kv_nope"
    ));
}
