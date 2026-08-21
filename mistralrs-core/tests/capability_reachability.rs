//! Parent system: ArcGate
//!
//! # The switched-off guard
//!
//! Arc's recurring bug class is not "unbuilt" — it is **built, wired, and
//! switched off by something adjacent**. A feature lands, compiles, passes
//! review, and then no default run ever reaches it. Nobody notices, because
//! *nothing fails*: the symbol exists, the tests that exercise it pass, and the
//! only trace is a `dead_code` warning in a crate whose warnings are not gated.
//!
//! `mistralrs-core` is deliberately outside the `-D warnings` clippy lane (the
//! workspace-wide form floods with pre-existing upstream findings), so the one
//! signal the compiler *does* emit for this class is discarded. That is the
//! hole this file closes.
//!
//! ## What it checks
//!
//! Two independent checks, because they fail in opposite directions:
//!
//! 1. **[`REGISTRY`] — named capabilities.** Each entry names a feature and
//!    asserts something *outside tests* reaches it. Registered features cannot
//!    go dark without this test going red. Entries marked
//!    [`Status::Tracked`] are known-dark on purpose and carry a reason; if one
//!    of those *becomes* reachable the test **also** goes red, telling you to
//!    promote it. The list cannot rot in either direction.
//!
//! 2. **[`DEAD_SYMBOL_BASELINE`] — the ratchet.** A scan of Arc-owned source
//!    for definitions with no production reference, compared against a
//!    checked-in baseline. Pre-existing debt is *recorded* rather than
//!    forgotten; a **newly** unreachable symbol fails. This is what catches a
//!    feature nobody thought to register — the case where the author did not
//!    know they were adding a dark room.
//!
//! ## Why it is static, not runtime
//!
//! Runtime engagement markers (`arc_profiler::mark_unreachable`, the profiler's
//! engagement asserts) are strictly better evidence — they observe the real
//! machine — but they only speak about the configuration that was actually run,
//! and they need a GPU and a model. This check runs on every CI host, on every
//! PR, with no hardware, and it speaks about *all* configurations at once. The
//! two are complements: this one proves a call path is written, the runtime
//! markers prove it was taken.
//!
//! ## Known limitation, stated on purpose
//!
//! Reachability here is "a production reference exists", not "a production
//! reference is itself reachable". A cluster of dead functions calling only
//! each other satisfies this check. That is a real gap, and it is the reason
//! check 2 exists at file granularity as well: a whole dead cluster shows up as
//! several new baseline entries at once.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// The registry
// ---------------------------------------------------------------------------

/// How a capability proves it is switched on.
enum Check {
    /// A symbol that production code must reference. `defined_in` is workspace
    /// relative; its definition site is excluded from the count, so defining a
    /// function is never mistaken for using it.
    Symbol {
        symbol: &'static str,
        defined_in: &'static str,
    },
    /// **The `logit_bias` class.** A request parameter accepted at the API
    /// boundary that must be *read* by the runtime that is supposed to honour
    /// it. Accepting a parameter and silently ignoring it is worse than
    /// rejecting it: the caller believes it took effect.
    ///
    /// `accepted_in` are the files that populate the field (the promise);
    /// `honoured_in` are the files that must read it (the delivery). A field
    /// written but never read is a broken promise, not merely dead code.
    ApiPromise {
        field: &'static str,
        accepted_in: &'static [&'static str],
        honoured_in: &'static [&'static str],
    },
}

enum Status {
    /// Must be reachable from production code. Red if it is not.
    Live,
    /// Known dark, deliberately, with a reason. Red if it *becomes* reachable
    /// — that is the signal to promote it to [`Status::Live`].
    Tracked { reason: &'static str },
}

struct Capability {
    /// Taxonomy name. Every capability has an absolute parent system —
    /// see `memory/mission/TAXONOMY.md`.
    name: &'static str,
    parent: &'static str,
    check: Check,
    status: Status,
}

/// The named capabilities. Add an entry when you add a feature that a default
/// run is supposed to reach.
static REGISTRY: &[Capability] = &[
    Capability {
        name: "ragged decode: a batch may hold rows of differing length",
        parent: "ArcInfer / ArcKV",
        check: Check::Symbol {
            symbol: "batch_can_be_ragged",
            defined_in: "mistralrs-core/src/kv_cache/mod.rs",
        },
        status: Status::Live,
    },
    Capability {
        name: "ragged decode: per-sequence xs history for V4",
        parent: "ArcInfer / ArcKV",
        check: Check::Symbol {
            symbol: "xs_per_sequence_enabled",
            defined_in: "mistralrs-core/src/kv_cache/xs_rolling.rs",
        },
        status: Status::Live,
    },
    Capability {
        name: "ragged decode: per-row lengths on the xs cache",
        parent: "ArcInfer / ArcKV",
        check: Check::Symbol {
            symbol: "set_row_lens",
            defined_in: "mistralrs-core/src/kv_cache/xs_rolling.rs",
        },
        status: Status::Live,
    },
    Capability {
        name: "ragged decode: the dead prefix attention must mask",
        parent: "ArcInfer / ArcAttention",
        check: Check::Symbol {
            symbol: "ragged_lead_pad",
            defined_in: "mistralrs-core/src/kv_cache/mod.rs",
        },
        status: Status::Live,
    },
    Capability {
        // The mask the entry above forces into existence, built on the DEVICE:
        // the host triple-loop build (kept as
        // `make_left_padded_causal_mask_host`, the test oracle) was 4.2 MB of
        // host work + H2D per token at B=256 ctx-4096, i.e. the thing that
        // would make ragged decode read as a regression. If production stops
        // reaching this builder, ragged batches are serving unmasked dead
        // prefixes — silently wrong, not slow.
        name: "ragged decode: the left-padded mask is built device-side",
        parent: "ArcInfer / ArcAttention",
        check: Check::Symbol {
            symbol: "make_left_padded_causal_mask",
            defined_in: "mistralrs-core/src/layers_masker.rs",
        },
        status: Status::Live,
    },
    Capability {
        // The channel the SCHEDULER reads to stop exact-length bucketing.
        // `batch_can_be_ragged` (above) is the producer; this is the consumer
        // side, and either going dark re-imposes the one-bucket-per-step
        // ceiling with no error anywhere.
        name: "ragged decode: the scheduler reads the published capability",
        parent: "ArcInfer / ArcSched",
        check: Check::Symbol {
            symbol: "ragged_decode_supported",
            defined_in: "mistralrs-core/src/kv_cache/mod.rs",
        },
        status: Status::Live,
    },
    Capability {
        // The supported ON-switch for the V4 ragged pair
        // (`--v4-ragged-decode` / `v4_ragged_decode` config key, with
        // `ARC_V4_XS_PER_SEQ` as the env fallback). A capability reachable
        // only through an undocumented env var is indistinguishable from one
        // that does not exist — which is how the ragged-batching result stayed
        // off the shipped configuration once already.
        name: "ragged decode: the CLI/config surface latches the V4 xs flag",
        parent: "ArcInfer / ArcKV",
        check: Check::Symbol {
            symbol: "request_xs_per_sequence",
            defined_in: "mistralrs-core/src/kv_cache/xs_rolling.rs",
        },
        status: Status::Live,
    },
    Capability {
        name: "prefill admission cap (--prefill-max-seqs)",
        parent: "ArcInfer / ArcSched",
        check: Check::Symbol {
            symbol: "prefill_admission_cap",
            defined_in: "mistralrs-core/src/scheduler/default_scheduler.rs",
        },
        status: Status::Live,
    },
    Capability {
        // Default ON since arcsched/ragged-and-floor: unset resolves to a
        // floor of 4 (`DEFAULT_PREFILL_FLOOR_STEPS`);
        // `ARC_PREFILL_FLOOR_STEPS=0` is the kill-switch that restores the
        // pre-floor selection key for key.
        name: "prompt-starvation floor (ARC_PREFILL_FLOOR_STEPS, default 4)",
        parent: "ArcInfer / ArcSched",
        check: Check::Symbol {
            symbol: "prefill_starvation_floor",
            defined_in: "mistralrs-core/src/scheduler/default_scheduler.rs",
        },
        status: Status::Live,
    },
    Capability {
        name: "unreachability reporting: features declare when they are dark",
        parent: "ArcLab",
        check: Check::Symbol {
            symbol: "mark_unreachable",
            defined_in: "arc-profiler/src/lib.rs",
        },
        status: Status::Live,
    },
    // -- the ApiPromise class ------------------------------------------------
    Capability {
        name: "sampling: logit_bias reaches the sampler",
        parent: "ArcInfer / ArcSample",
        check: Check::ApiPromise {
            field: "logits_bias",
            accepted_in: &[
                "mistralrs-server-core/src/completions.rs",
                "mistralrs-server-core/src/chat_completion.rs",
            ],
            honoured_in: &["mistralrs-core/src/sampler.rs"],
        },
        // On `cca5e5c6e` the field occurred in `sampler.rs` exactly twice — the
        // declaration and its `None` default — and was never read. The OpenAI
        // endpoints accepted `logit_bias` and silently dropped it.
        //
        // PROMOTED by this PR, which is the transition the entry was written to
        // demand. `logits_bias` is now read at `sampler.rs:401`
        // (`has_logits_bias`) and `:502` (`apply_logits_bias`), and `sample`
        // applies it to the raw logits before any filtering. The gate flagged
        // this itself — it went red on the rebase asking to be promoted, which
        // is exactly the behaviour its own reason string predicted.
        status: Status::Live,
    },
    Capability {
        name: "sampling: top_k survives onto the GPU-autonomous decode path",
        parent: "ArcInfer / ArcGraph",
        check: Check::ApiPromise {
            field: "top_k",
            accepted_in: &["mistralrs-core/src/sampler.rs"],
            honoured_in: &["arc-cuda-graph/src/autonomous.rs"],
        },
        // `AutonomousDecodeConfig` (autonomous.rs:70-82) carries temperature,
        // top_p, both penalties and `greedy` — and neither top_k nor min_p. A
        // request setting them would have them silently dropped the moment this
        // path is reached. It is not reached today (`mark_unreachable(
        // "cuda_graph.autonomous_decode", ...)`), which is the only reason this
        // is not already a user-visible defect.
        status: Status::Tracked {
            reason: "AutonomousDecodeConfig has no top_k field at all; the path is itself \
                     unreachable today, so this is latent rather than live. Fix before \
                     GPU-autonomous decode is switched on.",
        },
    },
    Capability {
        name: "sampling: min_p survives onto the GPU-autonomous decode path",
        parent: "ArcInfer / ArcGraph",
        check: Check::ApiPromise {
            field: "min_p",
            accepted_in: &["mistralrs-core/src/sampler.rs"],
            honoured_in: &["arc-cuda-graph/src/autonomous.rs"],
        },
        status: Status::Tracked {
            reason: "AutonomousDecodeConfig has no min_p field at all; same shape and same \
                     blocking condition as the top_k entry above.",
        },
    },
];

/// Symbols that are defined in Arc-owned source and have **no production
/// reference today**. This is debt, recorded so it is tracked rather than
/// forgotten.
///
/// **Adding a name here is a deliberate act.** If this test tells you to add
/// one, the honest options in order of preference are: wire the feature up,
/// delete it, or add it here *with the reason in the PR description*. Silently
/// appending to this list is how the bug class comes back.
///
/// Removing a name (because you wired it up) is always safe and the test will
/// tell you when a listed symbol is no longer dead.
/// Cross-checked against `cargo check -p mistralrs-core --message-format=json`
/// on 2026-08-19 at `cca5e5c6e`: rustc's own `dead_code` analysis independently
/// reports `assign_row_lens`, `extend_draft_kv` and `run_target_forward`. Two
/// instruments agreeing is why these are recorded as debt rather than treated
/// as scanner noise.
static DEAD_SYMBOL_BASELINE: &[(&str, &str)] = &[
    // (workspace-relative file, symbol)
    //
    // 🔴 THE CHUNKED-PREFILL SAMPLING HOLE — read this before enabling
    // `ARC_PREFILL_CHUNK`. This entry is option 3, taken deliberately, and it
    // records an INCOMPLETE FEATURE rather than a harmless spare part.
    //
    // The flag is written and never read. `engine/mod.rs:539` computes whether
    // this prefill step is the last chunk of its cohort and, when it is not,
    // installs `mark_prefill_intermediate()`. Nothing anywhere calls
    // `prefill_chunk_is_intermediate()` to consult it — its own doc comment
    // says it is "read by `Pipeline::step`'s sampling stage", and that reader
    // does not exist.
    //
    // So with chunking on, every intermediate chunk still reaches
    // `sample_causal_gen` and emits a token mid-prompt. A 2048-token prompt at
    // C=512 would produce four tokens nobody asked for before the prompt is
    // even finished.
    //
    // Not wired here on purpose. Suppressing the sample means deciding what the
    // sequence does instead — it must not advance, must not emit, and must stay
    // in `RunningPrompt` — across four `sample_causal_gen` call sites
    // (`mod.rs:1118`, `mod.rs:1346`, `amoe.rs:279`, `mtp_pipeline.rs:3202`).
    // That is generation behaviour, and it is not trustworthy until it has run
    // on a card. This session had no GPU.
    //
    // Why it is safe to sit here meanwhile: the hole is unreachable by default.
    // `prefill_chunk_size()` returns `None` unless `ARC_PREFILL_CHUNK` is set,
    // and it must stay unset regardless — chunking is measured NEGATIVE until
    // the QTIP expert gather is fixed (71.3% of an N=128 prefill step, billed
    // per step, so chunking pays it ceil(N/C) times).
    //
    // ⇒ Whoever turns chunking on owns finishing this first. Delete this entry
    //   in the same PR that adds the reader.
    (
        "mistralrs-core/src/pipeline/mod.rs",
        "prefill_chunk_is_intermediate",
    ),
    //
    // A fused SiLU·mul·down GEMV kernel that is compiled into
    // `libarccudagraph.a` on every CUDA build and launched by nothing. Declared
    // twice (`gemv_ffi.rs:32` and here) and called from neither.
    (
        "arc-cuda-graph/src/decode_forward.rs",
        "arc_launch_gemv_bf16_silu_mul_down",
    ),
    // The per-row length setter for the xs cache. Its only callers are tests
    // (`kv_cache/mod.rs:3127` sits inside a `#[cfg(test)]` region) — the ragged
    // path reaches the same state through `set_row_lens` instead.
    (
        "mistralrs-core/src/kv_cache/xs_rolling.rs",
        "assign_row_lens",
    ),
    // A `Pipeline` trait method implemented by all eight pipelines and called
    // by nobody: the accessor for the CUDA graph runner. Consistent with the
    // runner itself only ever being constructed on the PagedAttention arm.
    (
        "mistralrs-core/src/pipeline/mod.rs",
        "cuda_graph_runner_mut",
    ),
    // MTP draft-KV extension, superseded by the batched path.
    (
        "mistralrs-core/src/pipeline/mtp_pipeline.rs",
        "extend_draft_kv",
    ),
    // Per-sequence MTP verify forward; only `run_target_forward_batch` is live.
    (
        "mistralrs-core/src/pipeline/mtp_pipeline.rs",
        "run_target_forward",
    ),
];

/// Crate source roots scanned for *references*. Production reachability can
/// come from anywhere in the workspace, so this list is wide.
static SCAN_ROOTS: &[&str] = &[
    "mistralrs-core/src",
    "mistralrs-quant/src",
    "mistralrs-server-core/src",
    "mistralrs-server/src",
    "mistralrs-cli/src",
    "mistralrs-vision/src",
    "mistralrs-paged-attn/src",
    "mistralrs-audio/src",
    "mistralrs-mcp/src",
    "mistralrs/src",
    "mistralrs-pyo3/src",
    "arc-engine/src",
    "arc-cuda-graph/src",
    "arc-profiler/src",
    "arc-bench/src",
    "arc-cli/src",
    "arc-turbo/src",
];

// ---------------------------------------------------------------------------
// Source scanning
// ---------------------------------------------------------------------------

fn workspace_root() -> PathBuf {
    // `mistralrs-core/` -> workspace root.
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("mistralrs-core has a parent directory")
        .to_path_buf()
}

/// Replace the bytes of comments and string/char literals with spaces, keeping
/// every newline so line numbers survive.
///
/// This matters more than it looks: Arc's source carries long explanatory
/// comments that name the very symbols under test. Matching raw text would
/// count a doc-comment mention of `xs_per_sequence_enabled` as a call site and
/// report a dark feature as live — the exact failure mode this file exists to
/// prevent.
fn scrub(src: &str) -> String {
    #[derive(PartialEq)]
    enum S {
        Code,
        Line,
        Block,
        Str,
        Char,
        RawStr(usize),
    }
    let b = src.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut st = S::Code;
    let mut i = 0usize;
    let mut block_depth = 0usize;
    while i < b.len() {
        let c = b[i];
        let n = if i + 1 < b.len() { b[i + 1] } else { 0 };
        match st {
            S::Code => {
                // raw string: r"..." / r#"..."#
                if c == b'r' && (n == b'"' || n == b'#') {
                    let mut j = i + 1;
                    let mut hashes = 0usize;
                    while j < b.len() && b[j] == b'#' {
                        hashes += 1;
                        j += 1;
                    }
                    if j < b.len() && b[j] == b'"' {
                        for _ in i..=j {
                            out.push(b' ');
                        }
                        i = j + 1;
                        st = S::RawStr(hashes);
                        continue;
                    }
                }
                if c == b'/' && n == b'/' {
                    st = S::Line;
                    out.push(b' ');
                    out.push(b' ');
                    i += 2;
                    continue;
                }
                if c == b'/' && n == b'*' {
                    st = S::Block;
                    block_depth = 1;
                    out.push(b' ');
                    out.push(b' ');
                    i += 2;
                    continue;
                }
                if c == b'"' {
                    st = S::Str;
                    out.push(b' ');
                    i += 1;
                    continue;
                }
                // Lifetimes (`'a`) look like char literals; only treat `'x'`
                // and `'\n'` shapes as literals.
                if c == b'\'' {
                    let is_lit = (i + 2 < b.len() && b[i + 2] == b'\'')
                        || (i + 1 < b.len() && b[i + 1] == b'\\');
                    if is_lit {
                        st = S::Char;
                        out.push(b' ');
                        i += 1;
                        continue;
                    }
                }
                out.push(c);
                i += 1;
            }
            S::Line => {
                if c == b'\n' {
                    st = S::Code;
                    out.push(b'\n');
                } else {
                    out.push(b' ');
                }
                i += 1;
            }
            S::Block => {
                if c == b'/' && n == b'*' {
                    block_depth += 1;
                    out.push(b' ');
                    out.push(b' ');
                    i += 2;
                    continue;
                }
                if c == b'*' && n == b'/' {
                    block_depth -= 1;
                    out.push(b' ');
                    out.push(b' ');
                    i += 2;
                    if block_depth == 0 {
                        st = S::Code;
                    }
                    continue;
                }
                out.push(if c == b'\n' { b'\n' } else { b' ' });
                i += 1;
            }
            S::Str => {
                if c == b'\\' {
                    out.push(b' ');
                    if i + 1 < b.len() {
                        out.push(if n == b'\n' { b'\n' } else { b' ' });
                    }
                    i += 2;
                    continue;
                }
                if c == b'"' {
                    st = S::Code;
                    out.push(b' ');
                    i += 1;
                    continue;
                }
                out.push(if c == b'\n' { b'\n' } else { b' ' });
                i += 1;
            }
            S::Char => {
                if c == b'\\' {
                    out.push(b' ');
                    out.push(b' ');
                    i += 2;
                    continue;
                }
                if c == b'\'' {
                    st = S::Code;
                }
                out.push(if c == b'\n' { b'\n' } else { b' ' });
                i += 1;
            }
            S::RawStr(hashes) => {
                if c == b'"' {
                    let mut j = i + 1;
                    let mut seen = 0usize;
                    while j < b.len() && b[j] == b'#' && seen < hashes {
                        seen += 1;
                        j += 1;
                    }
                    if seen == hashes {
                        for _ in i..j {
                            out.push(b' ');
                        }
                        i = j;
                        st = S::Code;
                        continue;
                    }
                }
                out.push(if c == b'\n' { b'\n' } else { b' ' });
                i += 1;
            }
        }
    }
    let scrubbed = String::from_utf8(out).expect("scrubbing only replaces bytes with ASCII spaces");

    // Attributes are the one place a string literal names a real symbol:
    // `#[serde(deserialize_with = "parse_thing")]` is a genuine call site, and
    // blanking it reports `parse_thing` as dead when serde calls it on every
    // deserialize. Restore attribute lines verbatim.
    let orig: Vec<&str> = src.lines().collect();
    scrubbed
        .lines()
        .enumerate()
        .map(|(i, line)| {
            let o = orig.get(i).copied().unwrap_or(line);
            let t = o.trim_start();
            if (t.starts_with("#[") || t.starts_with("#![")) && o.contains('"') {
                o
            } else {
                line
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Line ranges of `impl <Trait> for <Type> { … }` blocks.
///
/// Methods inside a trait impl are invoked *through the trait*, never by name
/// at the impl site, so "no textual reference" says nothing about them —
/// rustc's own `dead_code` lint exempts them for the same reason. Counting them
/// would bury the real findings under serde `visit_str`/`expecting` noise.
fn trait_impl_ranges(scrubbed: &str) -> Vec<(usize, usize)> {
    let lines: Vec<&str> = scrubbed.lines().collect();
    let mut ranges = Vec::new();
    for (i, line) in lines.iter().enumerate() {
        let t = line.trim_start();
        if !(t.starts_with("impl") || t.starts_with("unsafe impl")) {
            continue;
        }
        // `impl Trait for Type` — an inherent `impl Type` has no ` for `.
        let head: String = lines[i..]
            .iter()
            .take(4)
            .copied()
            .collect::<Vec<_>>()
            .join(" ");
        let Some(brace_rel) = head.find('{') else {
            continue;
        };
        if !head[..brace_rel].contains(" for ") {
            continue;
        }
        let mut depth = 0i64;
        let mut opened = false;
        let mut end = i;
        'outer: for (k, l) in lines.iter().enumerate().skip(i) {
            for ch in l.chars() {
                if ch == '{' {
                    depth += 1;
                    opened = true;
                } else if ch == '}' {
                    depth -= 1;
                    if opened && depth == 0 {
                        end = k;
                        break 'outer;
                    }
                }
            }
        }
        ranges.push((i + 1, end + 1));
    }
    ranges
}

/// Line ranges (1-based, inclusive) that are compiled only under `cfg(test)`.
///
/// Covers `#[cfg(test)] mod tests { … }`, `#[cfg(test)]` on any other braced
/// item, and `#[test]`/`#[tokio::test]`-annotated functions wherever they sit.
///
/// Getting this map wrong is not cosmetic: a test region mistaken for
/// production would mark a dead feature "reachable" because its unit tests call
/// it — which is the precise illusion this file exists to dispel. Two real bugs
/// were found here by running it, both of which under-reported test regions:
/// a fixed lookahead window missed `#[cfg(test)]` separated from its `mod` by a
/// comment block and a second attribute, and whole-file test modules were not
/// detected at all.
fn test_mod_ranges(scrubbed: &str) -> Vec<(usize, usize)> {
    let lines: Vec<&str> = scrubbed.lines().collect();
    let mut ranges = Vec::new();

    // Close a braced item starting at `start`, returning its last line.
    let close_braced = |start: usize| -> usize {
        let mut depth = 0i64;
        let mut opened = false;
        for (k, line) in lines.iter().enumerate().skip(start) {
            for ch in line.chars() {
                if ch == '{' {
                    depth += 1;
                    opened = true;
                } else if ch == '}' {
                    depth -= 1;
                    if opened && depth == 0 {
                        return k;
                    }
                }
            }
        }
        lines.len().saturating_sub(1)
    };

    for (i, line) in lines.iter().enumerate() {
        let is_cfg_test = line.contains("#[cfg(test)]");
        let is_test_attr =
            line.contains("#[test]") || line.contains("#[tokio::test]") || line.contains("::test]");
        if !is_cfg_test && !is_test_attr {
            continue;
        }
        // Walk forward past blank lines (scrubbed comments) and further
        // attributes to the item this applies to. No fixed window: attribute
        // stacks in this codebase run to a dozen lines with prose between them.
        let mut j = i + 1;
        while j < lines.len() {
            let t = lines[j].trim();
            if t.is_empty() || t.starts_with("#[") || t.starts_with("#!") {
                j += 1;
                continue;
            }
            break;
        }
        if j >= lines.len() {
            continue;
        }
        let item = lines[j];
        if is_cfg_test && item.contains("mod ") && !item.contains('{') && item.contains(';') {
            // `#[cfg(test)] mod foo;` — the whole of foo.rs is test-only, which
            // is handled at file granularity by `Corpus::load`.
            continue;
        }
        if item.contains('{') || lines[j..].iter().take(8).any(|l| l.contains('{')) {
            let end = close_braced(j);
            ranges.push((i + 1, end + 1));
        } else {
            ranges.push((i + 1, j + 1));
        }
    }

    // Merge overlaps so a `#[test]` inside a `#[cfg(test)] mod` is not double
    // counted and the ranges stay cheap to query.
    ranges.sort_unstable();
    let mut merged: Vec<(usize, usize)> = Vec::new();
    for (a, b) in ranges {
        match merged.last_mut() {
            Some(last) if a <= last.1 + 1 => last.1 = last.1.max(b),
            _ => merged.push((a, b)),
        }
    }
    merged
}

/// Modules declared `#[cfg(test)] mod NAME;` — the whole of `NAME.rs` (or
/// `NAME/mod.rs`) is test-only, however it is written inside.
fn test_only_module_names(scrubbed: &str) -> Vec<String> {
    let lines: Vec<&str> = scrubbed.lines().collect();
    let mut out = Vec::new();
    for (i, line) in lines.iter().enumerate() {
        if !line.contains("#[cfg(test)]") {
            continue;
        }
        let mut j = i + 1;
        while j < lines.len() {
            let t = lines[j].trim();
            if t.is_empty() || t.starts_with("#[") {
                j += 1;
                continue;
            }
            break;
        }
        if j >= lines.len() {
            continue;
        }
        let t = lines[j].trim();
        if let Some(rest) = t
            .strip_prefix("mod ")
            .or_else(|| t.strip_prefix("pub mod "))
        {
            let name: String = rest
                .chars()
                .take_while(|c| c.is_alphanumeric() || *c == '_')
                .collect();
            if !name.is_empty() && rest.contains(';') && !rest.contains('{') {
                out.push(name);
            }
        }
    }
    out
}

/// Does `line` reference `symbol` as a whole word?
fn references(line: &str, symbol: &str) -> bool {
    let bytes = line.as_bytes();
    let sb = symbol.as_bytes();
    let mut from = 0usize;
    while let Some(pos) = line[from..].find(symbol) {
        let start = from + pos;
        let end = start + sb.len();
        let before_ok = start == 0 || !is_ident_byte(bytes[start - 1]);
        let after_ok = end >= bytes.len() || !is_ident_byte(bytes[end]);
        if before_ok && after_ok {
            return true;
        }
        from = start + 1;
        if from >= line.len() {
            break;
        }
    }
    false
}

fn is_ident_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

/// Does `line` **read** the field `field`, as opposed to declaring or writing it?
///
/// This distinction is the whole value of the [`Check::ApiPromise`] class, and
/// getting it wrong made this guard pass vacuously on its first run: in
/// `sampler.rs` the field `logits_bias` occurs exactly twice — once as the
/// declaration `pub logits_bias: Option<HashMap<u32, f32>>` and once as the
/// default `logits_bias: None` — and *neither is a read*. Counting them made a
/// famously dead wire report as honoured.
///
/// The discriminator: in a declaration and in a struct-literal initializer the
/// field name is immediately followed by `:`. A read is `self.logits_bias`,
/// `params.logits_bias`, or a bare use in an expression.
fn reads_field(line: &str, field: &str) -> bool {
    let bytes = line.as_bytes();
    let mut from = 0usize;
    while let Some(pos) = line[from..].find(field) {
        let start = from + pos;
        let end = start + field.len();
        let before_ok = start == 0 || !is_ident_byte(bytes[start - 1]);
        let after_ok = end >= bytes.len() || !is_ident_byte(bytes[end]);
        if before_ok && after_ok {
            // Skip whitespace, then decide.
            let mut k = end;
            while k < bytes.len() && bytes[k] == b' ' {
                k += 1;
            }
            let is_decl_or_write = k < bytes.len() && bytes[k] == b':';
            if !is_decl_or_write {
                return true;
            }
        }
        from = start + 1;
        if from >= line.len() {
            break;
        }
    }
    false
}

/// Is this line the *definition* of `symbol` rather than a use of it?
fn is_definition(line: &str, symbol: &str) -> bool {
    for kw in [
        "fn ", "struct ", "enum ", "const ", "static ", "trait ", "type ", "union ", "mod ",
    ] {
        if let Some(p) = line.find(kw) {
            let rest = line[p + kw.len()..].trim_start();
            if rest.starts_with(symbol) {
                let tail = &rest[symbol.len()..];
                if tail
                    .chars()
                    .next()
                    .is_none_or(|c| !c.is_alphanumeric() && c != '_')
                {
                    return true;
                }
            }
        }
    }
    false
}

struct Reference {
    file: String,
    line: usize,
    in_test: bool,
}

fn rust_files(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = fs::read_dir(&dir) else {
            continue;
        };
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.extension().is_some_and(|x| x == "rs") {
                out.push(p);
            }
        }
    }
    out.sort();
    out
}

/// Cached scrub + test-range computation for the whole scan set.
struct Corpus {
    /// workspace-relative path -> (scrubbed source, test line ranges)
    files: BTreeMap<String, (String, Vec<(usize, usize)>)>,
}

impl Corpus {
    fn load() -> Self {
        let root = workspace_root();
        let mut files = BTreeMap::new();
        // Pass 1: read and scrub, and collect `#[cfg(test)] mod NAME;`
        // declarations so the files they name can be marked test-only whole.
        let mut raw: Vec<(String, PathBuf, String)> = Vec::new();
        let mut test_only_files: BTreeSet<String> = BTreeSet::new();
        for r in SCAN_ROOTS {
            let dir = root.join(r);
            if !dir.exists() {
                continue;
            }
            for f in rust_files(&dir) {
                let Ok(src) = fs::read_to_string(&f) else {
                    continue;
                };
                let rel = f
                    .strip_prefix(&root)
                    .unwrap_or(&f)
                    .to_string_lossy()
                    .replace('\\', "/");
                let scrubbed = scrub(&src);
                for name in test_only_module_names(&scrubbed) {
                    let dir_of = f.parent().unwrap_or(&root);
                    for cand in [
                        dir_of.join(format!("{name}.rs")),
                        dir_of.join(&name).join("mod.rs"),
                    ] {
                        if let Ok(p) = cand.strip_prefix(&root) {
                            test_only_files.insert(p.to_string_lossy().replace('\\', "/"));
                        }
                    }
                }
                raw.push((rel, f, scrubbed));
            }
        }
        // Pass 2: compute test ranges, widening to the whole file for modules
        // that are only compiled under `cfg(test)`.
        for (rel, _f, scrubbed) in raw {
            let ranges = if test_only_files.contains(&rel) {
                vec![(1usize, scrubbed.lines().count().max(1))]
            } else {
                test_mod_ranges(&scrubbed)
            };
            files.insert(rel, (scrubbed, ranges));
        }
        assert!(
            files.len() > 100,
            "corpus looks wrong: found only {} files under {:?}. The scan roots are probably \
             misconfigured, and an empty corpus would make every capability look dark.",
            files.len(),
            root
        );
        Self { files }
    }

    /// Is there a production reference to `symbol` that is not itself a
    /// definition of it, anywhere in the workspace?
    ///
    /// Definitions are excluded in **every** file, not just the declaring one:
    /// a trait method implemented by eight pipelines has eight `fn` lines, and
    /// counting those as callers would report a method nobody invokes as live.
    /// (Observed: `cuda_graph_runner_mut` did exactly that before this existed.)
    fn has_production_reference(&self, symbol: &str) -> bool {
        self.find(symbol, None).iter().any(|r| {
            if r.in_test {
                return false;
            }
            let Some((src, _)) = self.files.get(&r.file) else {
                return false;
            };
            let Some(line) = src.lines().nth(r.line - 1) else {
                return false;
            };
            !is_definition(line, symbol)
        })
    }

    fn find(&self, symbol: &str, defined_in: Option<&str>) -> Vec<Reference> {
        let mut out = Vec::new();
        for (rel, (src, ranges)) in &self.files {
            for (idx, line) in src.lines().enumerate() {
                let lineno = idx + 1;
                if !references(line, symbol) {
                    continue;
                }
                // The definition is not a use of itself.
                if defined_in == Some(rel.as_str()) && is_definition(line, symbol) {
                    continue;
                }
                let in_test = ranges.iter().any(|(a, b)| lineno >= *a && lineno <= *b);
                out.push(Reference {
                    file: rel.clone(),
                    line: lineno,
                    in_test,
                });
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Check 1 — the registry
// ---------------------------------------------------------------------------

#[test]
fn every_registered_capability_has_a_production_call_path() {
    let corpus = Corpus::load();
    let mut failures: Vec<String> = Vec::new();

    for cap in REGISTRY {
        match &cap.check {
            Check::Symbol { symbol, defined_in } => {
                assert!(
                    corpus.files.contains_key(*defined_in),
                    "capability `{}` names `defined_in = {}`, which is not in the scan set. \
                     Fix the path or add its crate to SCAN_ROOTS — a missing file would \
                     otherwise read as `no callers` for the wrong reason.",
                    cap.name,
                    defined_in
                );
                let refs = corpus.find(symbol, Some(defined_in));
                let prod: Vec<&Reference> = refs.iter().filter(|r| !r.in_test).collect();
                match cap.status {
                    Status::Live => {
                        if prod.is_empty() {
                            let test_only = refs.len();
                            failures.push(format!(
                                "SWITCHED OFF: `{}` ({})\n    symbol `{}` defined at {} has \
                                 NO production reference ({} test-only reference(s)).\n    \
                                 Either wire it into a default path, or change its Status to \
                                 Tracked with a reason.",
                                cap.name, cap.parent, symbol, defined_in, test_only
                            ));
                        }
                    }
                    Status::Tracked { reason } => {
                        if !prod.is_empty() {
                            failures.push(format!(
                                "PROMOTE ME: `{}` ({})\n    symbol `{}` is registered as \
                                 Tracked (\"{}\") but now HAS production references at {}.\n    \
                                 Change its Status to Live so it is guarded from going dark \
                                 again.",
                                cap.name,
                                cap.parent,
                                symbol,
                                reason,
                                prod.iter()
                                    .take(3)
                                    .map(|r| format!("{}:{}", r.file, r.line))
                                    .collect::<Vec<_>>()
                                    .join(", ")
                            ));
                        }
                    }
                }
            }
            Check::ApiPromise {
                field,
                accepted_in,
                honoured_in,
            } => {
                let accepted: Vec<&Reference> = Vec::new();
                let mut accepted = accepted;
                let all = corpus.find(field, None);
                for r in &all {
                    if accepted_in.contains(&r.file.as_str()) && !r.in_test {
                        accepted.push(r);
                    }
                }
                // A read, not a declaration and not a struct-literal write.
                let honoured: Vec<&Reference> = all
                    .iter()
                    .filter(|r| {
                        if !honoured_in.contains(&r.file.as_str()) || r.in_test {
                            return false;
                        }
                        corpus
                            .files
                            .get(&r.file)
                            .and_then(|(src, _)| src.lines().nth(r.line - 1))
                            .is_some_and(|line| reads_field(line, field))
                    })
                    .collect();

                let live = matches!(cap.status, Status::Live);
                if accepted.is_empty() && live {
                    failures.push(format!(
                        "BROKEN REGISTRY ENTRY: `{}` ({})\n    field `{}` is not populated in \
                         any of {:?}. The entry is stale — fix the paths.",
                        cap.name, cap.parent, field, accepted_in
                    ));
                } else if honoured.is_empty() && live {
                    failures.push(format!(
                        "DEAD WIRE: `{}` ({})\n    field `{}` IS accepted at the API boundary \
                         ({}) but is never read in {:?}.\n    The endpoint accepts the \
                         parameter and silently ignores it. Either honour it or reject it — \
                         accepting and dropping is the worst of the three.",
                        cap.name,
                        cap.parent,
                        field,
                        accepted
                            .iter()
                            .take(2)
                            .map(|r| format!("{}:{}", r.file, r.line))
                            .collect::<Vec<_>>()
                            .join(", "),
                        honoured_in
                    ));
                } else if !honoured.is_empty() {
                    if let Status::Tracked { reason } = cap.status {
                        failures.push(format!(
                            "PROMOTE ME: `{}` ({}) is Tracked (\"{}\") but field `{}` is now \
                             read at {}. Change its Status to Live.",
                            cap.name,
                            cap.parent,
                            reason,
                            field,
                            honoured
                                .iter()
                                .take(2)
                                .map(|r| format!("{}:{}", r.file, r.line))
                                .collect::<Vec<_>>()
                                .join(", ")
                        ));
                    }
                }
            }
        }
    }

    if !failures.is_empty() {
        panic!(
            "\n\n{} registered capability/capabilities are switched off:\n\n{}\n\n\
             This test exists because Arc's recurring fault is code that is built, wired, \
             and then unreachable. See mistralrs-core/tests/capability_reachability.rs.\n",
            failures.len(),
            failures.join("\n\n")
        );
    }
}

// ---------------------------------------------------------------------------
// Check 2 — the ratchet
// ---------------------------------------------------------------------------

/// Directories the ratchet scans for *definitions*. Narrower than
/// [`SCAN_ROOTS`], which is where references may live: a reference from
/// anywhere in the workspace keeps a symbol alive, but only Arc-owned code is
/// held to the no-new-dark-rooms rule. Upstream mistral.rs files carry their
/// own dead code and are not ours to churn (fork policy).
static RATCHET_ROOTS: &[&str] = &[
    "mistralrs-core/src/kv_cache",
    "mistralrs-core/src/kv_sharing",
    "mistralrs-core/src/moe",
    "mistralrs-core/src/scheduler",
    "mistralrs-core/src/pipeline",
    "mistralrs-core/src/engine",
    "arc-cuda-graph/src",
    "arc-profiler/src",
];

/// A definition the ratchet considers. Only items rustc's own `dead_code` lint
/// would consider — i.e. **not** crate-public API, which is legitimately
/// unused inside its own crate.
fn definition_symbol(line: &str) -> Option<String> {
    let t = line.trim_start();
    // Crate-public API is exported; "no internal caller" is not a defect.
    if t.starts_with("pub ") && !t.starts_with("pub(crate)") && !t.starts_with("pub(super)") {
        return None;
    }
    let after_vis = t
        .strip_prefix("pub(crate)")
        .or_else(|| t.strip_prefix("pub(super)"))
        .unwrap_or(t)
        .trim_start();
    // Skip declarations that are not definitions of a callable/nameable item.
    for kw in ["fn ", "struct ", "enum ", "trait ", "const ", "static "] {
        let body = after_vis
            .strip_prefix("async ")
            .unwrap_or(after_vis)
            .trim_start();
        let body = body.strip_prefix("unsafe ").unwrap_or(body).trim_start();
        let body = body
            .strip_prefix("extern \"C\" ")
            .unwrap_or(body)
            .trim_start();
        if let Some(rest) = body.strip_prefix(kw) {
            let name: String = rest
                .chars()
                .take_while(|c| c.is_alphanumeric() || *c == '_')
                .collect();
            // Short names (`new`, `len`, `run`) collide across the workspace and
            // would read as reachable regardless; they carry no signal here.
            if name.len() >= 6 && !name.starts_with("__") {
                return Some(name);
            }
            return None;
        }
    }
    None
}

/// **The ratchet.** Any Arc-owned definition with no production reference
/// anywhere in the workspace is a dark room. Pre-existing ones are recorded in
/// [`DEAD_SYMBOL_BASELINE`]; a new one fails.
///
/// This is the check that catches a feature nobody thought to register — the
/// author did not know they were adding a switched-off feature, so they would
/// never have added a REGISTRY entry for it.
#[test]
fn no_new_unreachable_symbols_in_arc_owned_code() {
    let corpus = Corpus::load();
    let root = workspace_root();

    // Every definition line of every name, so a definition is never counted as
    // a use of itself — in any file, not just the declaring one (a trait
    // declaration and its impls all define the same name).
    let mut definitions: BTreeMap<String, Vec<(String, usize)>> = BTreeMap::new();
    for dir in RATCHET_ROOTS {
        let full = root.join(dir);
        if !full.exists() {
            continue;
        }
        for f in rust_files(&full) {
            let rel = f
                .strip_prefix(&root)
                .unwrap_or(&f)
                .to_string_lossy()
                .replace('\\', "/");
            let Some((src, ranges)) = corpus.files.get(&rel) else {
                continue;
            };
            let impls = trait_impl_ranges(src);
            for (idx, line) in src.lines().enumerate() {
                let lineno = idx + 1;
                if ranges.iter().any(|(a, b)| lineno >= *a && lineno <= *b) {
                    continue; // a definition inside #[cfg(test)] is test scaffolding
                }
                if impls.iter().any(|(a, b)| lineno >= *a && lineno <= *b) {
                    continue; // trait impl methods are called through the trait
                }
                if let Some(sym) = definition_symbol(line) {
                    definitions
                        .entry(sym)
                        .or_default()
                        .push((rel.clone(), lineno));
                }
            }
        }
    }

    let baseline: BTreeSet<(&str, &str)> = DEAD_SYMBOL_BASELINE.iter().copied().collect();
    let mut new_dark: Vec<String> = Vec::new();

    for (sym, sites) in &definitions {
        if corpus.has_production_reference(sym) {
            continue;
        }
        let (file, lineno) = &sites[0];
        if baseline.contains(&(file.as_str(), sym.as_str())) {
            continue;
        }
        new_dark.push(format!(
            "  {file}:{lineno}  `{sym}` — defined, never referenced by production code"
        ));
    }

    new_dark.sort();
    assert!(
        new_dark.is_empty(),
        "\n\n{} NEW switched-off symbol(s) in Arc-owned code:\n\n{}\n\n\
         Something was built and left unreachable. In order of preference:\n  \
         1. wire it into a default path (and add a REGISTRY entry so it cannot go dark again);\n  \
         2. delete it;\n  \
         3. add it to DEAD_SYMBOL_BASELINE, and say why in the PR description.\n\n\
         Option 3 is a deliberate act, not a formality — it is how this bug class comes back.\n",
        new_dark.len(),
        new_dark.join("\n")
    );
}

/// The baseline must not carry names that are no longer dead — otherwise it
/// silently grants permission that is no longer needed, and the next genuinely
/// dead symbol can hide behind a stale entry.
#[test]
fn dead_symbol_baseline_has_no_stale_entries() {
    let corpus = Corpus::load();
    let mut stale = Vec::new();
    for (file, symbol) in DEAD_SYMBOL_BASELINE {
        if !corpus.files.contains_key(*file) {
            stale.push(format!(
                "  {file} :: {symbol} — file no longer in the scan set (moved or deleted?)"
            ));
            continue;
        }
        if corpus.has_production_reference(symbol) {
            stale.push(format!(
                "  {file} :: {symbol} — now HAS a production reference; remove it from \
                 DEAD_SYMBOL_BASELINE (and consider registering it in REGISTRY so it cannot \
                 go dark again)"
            ));
        }
    }
    assert!(
        stale.is_empty(),
        "\n\nDEAD_SYMBOL_BASELINE is stale:\n{}\n",
        stale.join("\n")
    );
}

/// Every baseline entry is unique — a duplicated name would make the ratchet
/// permit two symbols where the author reviewed one.
#[test]
fn dead_symbol_baseline_is_a_set() {
    let mut seen = BTreeSet::new();
    let mut dupes = Vec::new();
    for entry in DEAD_SYMBOL_BASELINE {
        if !seen.insert(*entry) {
            dupes.push(format!("{}::{}", entry.0, entry.1));
        }
    }
    assert!(dupes.is_empty(), "duplicate baseline entries: {dupes:?}");
}

/// The registry itself must be well formed: no duplicate names, every entry
/// carries an absolute parent system (TAXONOMY.md — "no subsystem may be left
/// without an absolute parent system name").
#[test]
fn registry_is_well_formed() {
    let mut seen = BTreeSet::new();
    for cap in REGISTRY {
        assert!(
            seen.insert(cap.name),
            "duplicate capability name: {}",
            cap.name
        );
        assert!(
            cap.parent.starts_with("Arc"),
            "capability `{}` has parent `{}`, which is not an Arc system. Every subsystem \
             needs an absolute parent — see memory/mission/TAXONOMY.md.",
            cap.name,
            cap.parent
        );
        if let Status::Tracked { reason } = cap.status {
            assert!(
                reason.len() > 20,
                "capability `{}` is Tracked with reason `{}` — too short to be a real reason. \
                 Say what blocks it and who owns it.",
                cap.name,
                reason
            );
        }
    }
}

// ---------------------------------------------------------------------------
// The guard's own self-tests — D18: a green must prove work happened
// ---------------------------------------------------------------------------

/// **Proves the scanner can say no.**
///
/// A guard that only ever passes is worthless, and this project has shipped two
/// of those in a single session. So: a symbol that certainly does not exist
/// must produce zero production references. If this test fails, the scanner is
/// matching things it should not and every green above is meaningless.
#[test]
fn scanner_reports_absent_symbols_as_unreachable() {
    let corpus = Corpus::load();
    let refs = corpus.find("arc_guard_symbol_that_does_not_exist_anywhere", None);
    assert!(
        refs.is_empty(),
        "scanner found phantom references: {:?}",
        refs.iter()
            .map(|r| format!("{}:{}", r.file, r.line))
            .collect::<Vec<_>>()
    );
}

/// **Proves the scanner does not count comments as call sites.**
///
/// This is the failure mode that would make the whole file lie: Arc's source
/// names its own symbols constantly in prose. `xs_per_sequence_enabled` appears
/// in ~20 comments and a handful of real calls; if comments counted, a fully
/// dark feature would read as live.
#[test]
fn scrubber_removes_comments_and_strings() {
    let src = r##"
// call foo_bar_baz() here
/// doc mentions foo_bar_baz
/* block foo_bar_baz */
let s = "foo_bar_baz";
let r = r#"foo_bar_baz"#;
foo_bar_baz();
"##;
    let scrubbed = scrub(src);
    let hits: Vec<usize> = scrubbed
        .lines()
        .enumerate()
        .filter(|(_, l)| references(l, "foo_bar_baz"))
        .map(|(i, _)| i + 1)
        .collect();
    assert_eq!(
        hits.len(),
        1,
        "expected exactly one surviving reference (the real call), got lines {hits:?} in:\n{scrubbed}"
    );
    // Line numbers must survive scrubbing, or every report points at the wrong place.
    assert_eq!(scrubbed.lines().count(), src.lines().count());
}

/// **Proves test-only references do not count as production.**
#[test]
fn scanner_separates_test_references_from_production() {
    let src = r#"
fn widget_alpha() {}
fn caller() { widget_alpha(); }

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn t() { widget_alpha(); }
}
"#;
    let scrubbed = scrub(src);
    let ranges = test_mod_ranges(&scrubbed);
    assert_eq!(
        ranges.len(),
        1,
        "expected one #[cfg(test)] mod, got {ranges:?}"
    );
    let (a, b) = ranges[0];
    let mut prod = 0;
    let mut test = 0;
    for (i, line) in scrubbed.lines().enumerate() {
        let n = i + 1;
        if !references(line, "widget_alpha") || is_definition(line, "widget_alpha") {
            continue;
        }
        if n >= a && n <= b {
            test += 1;
        } else {
            prod += 1;
        }
    }
    assert_eq!(prod, 1, "expected 1 production reference");
    assert_eq!(test, 1, "expected 1 test reference");
}

/// **Proves a definition is not mistaken for a use.**
///
/// Without this, a function that nothing calls would look reachable because its
/// own `fn` line references its own name — which is precisely how a dark
/// feature would slip past.
#[test]
fn a_definition_is_not_a_caller() {
    assert!(is_definition(
        "pub(crate) fn prefill_chunk_is_intermediate() -> bool {",
        "prefill_chunk_is_intermediate"
    ));
    assert!(is_definition(
        "static REGISTRY: &[Capability] = &[",
        "REGISTRY"
    ));
    assert!(is_definition(
        "pub struct XsRollingCache {",
        "XsRollingCache"
    ));
    assert!(!is_definition(
        "    let x = prefill_chunk_is_intermediate();",
        "prefill_chunk_is_intermediate"
    ));
    assert!(!is_definition(
        "    if xs_per_sequence_enabled() {",
        "xs_per_sequence_enabled"
    ));
}

/// **Proves a declared-but-never-read field is not counted as honoured.**
///
/// This test exists because the guard shipped a false green on its first run:
/// the `logit_bias` dead wire was reported as live, because the only two
/// occurrences of the field in `sampler.rs` are its declaration and its `None`
/// default. Both were counted as reads. This pins the fix.
#[test]
fn a_declared_or_written_field_is_not_a_read() {
    // Declarations — not reads.
    assert!(!reads_field(
        "    pub logits_bias: Option<HashMap<u32, f32>>,",
        "logits_bias"
    ));
    assert!(!reads_field("    logits_bias: None,", "logits_bias"));
    // Struct-literal writes — not reads.
    assert!(!reads_field(
        "                logits_bias: oairequest.logit_bias,",
        "logits_bias"
    ));
    // Genuine reads.
    assert!(reads_field(
        "        if let Some(bias) = &self.logits_bias {",
        "logits_bias"
    ));
    assert!(reads_field(
        "        for (tok, b) in params.logits_bias.iter().flatten() {",
        "logits_bias"
    ));
}

/// **Proves whole-word matching.** `set_row_lens` must not be satisfied by
/// `set_row_lens_unchecked`, or renaming a function to a superstring would
/// leave the guard green while the feature went dark.
#[test]
fn matching_is_whole_word() {
    assert!(references("foo(bar_baz);", "bar_baz"));
    assert!(!references("foo(bar_baz_qux);", "bar_baz"));
    assert!(!references("foo(x_bar_baz);", "bar_baz"));
    assert!(references("a.bar_baz();", "bar_baz"));
}
