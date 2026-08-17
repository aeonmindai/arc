// CLI help/usage doc comments intentionally indent command examples under list
// items; keep that layout rather than reflow it to satisfy the doc-list lints.
#![allow(clippy::doc_overindented_list_items, clippy::doc_lazy_continuation)]
//! Arc inference engine CLI.
//!
//! **Parent system: ArcServe/CLI** (see `memory/mission/TAXONOMY.md`).
//!
//! This is the `arc` binary — a wrapper around the mistral.rs CLI that adds Arc
//! branding, the `validate` pre-flight command, and the AgentPerf bench suite.
//!
//! ⚠️ **TurboQuant is the *nominal* PagedAttention default only.**
//! `PagedCacheType::resolve_for_model` falls back to the unquantized `auto`
//! cache — with a warning — for any model TurboQuant cannot support, which
//! means every MLA model and every head_dim other than 128. Requesting it
//! explicitly turns that fallback into a hard error instead. The eager
//! (non-paged) KV path is separately opt-in via `ARC_TURBOQUANT_KV=1`.
//! In practice almost no model runs TurboQuant today, and none has been
//! measured with it.
//!
//! Usage:
//!   arc serve -m <model_id>                          # Start serving
//!   arc run -m <model_id>                            # Interactive chat
//!   arc bench -m <model_id>                          # Run benchmarks
//!   arc validate --index <path> --arch <arch>        # Pre-flight weight schema validation
//!   arc validate --target-hbm 60 -m <model_id>       # HBM footprint check (RUN-191)
//!   arc serve --pa-cache-type auto                   # Disable TurboQuant, use upstream defaults

mod bench;
mod validate;

use clap::{Parser, Subcommand};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Duration;

/// Arc — A high-performance LLM inference engine with TurboQuant compression.
///
/// Built on mistral.rs. Defaults to TurboQuant 3.5-bit KV cache (lossless).
#[derive(Parser)]
#[command(name = "arc", version, about, long_about = None)]
#[command(
    after_help = "Arc inference engine by Aeonmind, LLC\nhttps://runcrate.ai/arc\nPowered by mistral.rs + TurboQuant (ICLR 2026)"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the Arc inference server (OpenAI-compatible API)
    Serve {
        /// All arguments are forwarded to the mistral.rs serve command
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },
    /// Interactive chat with a model
    Run {
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },
    /// Benchmark a model.
    ///
    /// Two suites:
    ///
    /// 1. AA-AgentPerf (default) — TUI dashboard + exponential-ramp +
    ///    binary-search to the largest concurrent-user K that passes the
    ///    SLO tier. Mirrors Artificial Analysis's AgentPerf methodology.
    ///    Add `--mock` to exercise the full code path without a GPU.
    ///
    /// 2. Legacy — pass `--suite simple` and any trailing args; they are
    ///    forwarded verbatim to the upstream `mistralrs bench` subprocess.
    Bench {
        /// Suite to run: agentperf (default) or simple.
        #[arg(long, default_value = "agentperf")]
        suite: String,
        /// HuggingFace model id.
        #[arg(short = 'm', long, default_value = "deepseek-ai/DeepSeek-V4-Flash")]
        model: String,
        /// SLO tier (1=strictest, 4=loosest). See `bench::slo` for defaults.
        #[arg(long, default_value_t = 2)]
        slo_tier: u8,
        /// Use the offline mock vendor — no GPU required.
        ///
        /// Hidden: emits a fully-formed results artifact containing synthetic
        /// numbers. It exists for harness tests; surfacing it in `--help`
        /// invites publishing mock throughput as if it were measured.
        #[arg(long, hide = true)]
        mock: bool,
        /// Cap on concurrent users explored by the scheduler.
        #[arg(long, default_value_t = 256)]
        max_users: u32,
        /// Output path (extension `.json` / `.md` selects which artefact;
        /// no extension writes both).
        #[arg(long, default_value = "tests/results/agentperf.json")]
        output: PathBuf,
        /// Warmup seconds per phase (the steady-state window uses the same
        /// length). Override for fast integration tests.
        #[arg(long, default_value_t = 30.0)]
        warmup_seconds: f64,
        /// Steady-state seconds per phase. Defaults to `warmup_seconds`.
        #[arg(long)]
        steady_state_seconds: Option<f64>,
        /// Skip the TUI even on a TTY. Streams phase events to stderr instead.
        #[arg(long)]
        headless: bool,
        /// (legacy `--suite simple`) Pass-through args forwarded to
        /// `mistralrs bench`. Ignored by agentperf.
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        legacy_args: Vec<String>,
    },
    /// Pre-flight validation: check a model against Arc's expected schema
    /// (tensor names *or* HBM footprint).
    ///
    /// Two modes — selected by which flags are passed:
    ///
    /// 1. Schema validation (the original mode):
    ///       arc validate --index <model.safetensors.index.json> --arch deepseekv4
    ///    Catches missing/mismatched tensors before any GPU load.
    ///    Supported --arch values: deepseekv4, kimi_k2, glm5moedsa.
    ///
    /// 2. HBM footprint validation (RUN-191):
    ///       arc validate --target-hbm 60 --model deepseek-ai/DeepSeek-V4-Flash \
    ///                    --compression-stack qtip2+td-moe
    ///    Loads the model with the requested compression stack and verifies
    ///    that the post-load HBM residency fits the target ceiling. Add
    ///    `--mock` to compute the estimate analytically (CI / off-GPU).
    Validate {
        // --- schema-mode flags ---
        /// (schema mode) Path to model.safetensors.index.json
        #[arg(long)]
        index: Option<PathBuf>,
        /// (schema mode) Target architecture: deepseekv4 | kimi_k2 | glm5moedsa
        #[arg(long)]
        arch: Option<String>,
        /// (schema mode) For V4: assume LoRA o_proj layout instead of the default Either fallback.
        ///
        /// Hidden: a weight-schema assumption used to A/B checkpoint layouts.
        /// The `either` default accepts both, so overriding it can only make
        /// validation stricter than the loader actually is.
        #[arg(long, default_value = "either", hide = true)]
        o_proj: String,

        // --- HBM-mode flags (RUN-191) ---
        /// (HBM mode) Maximum HBM residency in GB. Passing this flag selects HBM mode.
        #[arg(long)]
        target_hbm: Option<f64>,
        /// (HBM mode) HuggingFace model id.
        #[arg(short = 'm', long, default_value = "deepseek-ai/DeepSeek-V4-Flash")]
        model: String,
        /// (HBM mode) Compression stack: bf16, fp8-only, nvfp4, qtip2-only, qtip2+td-moe.
        #[arg(long, default_value = "qtip2+td-moe")]
        compression_stack: String,
        /// (HBM mode) Output JSON path for the validation report.
        #[arg(long, default_value = "tests/results/v4_flash_h100_footprint.json")]
        output: PathBuf,
        /// (HBM mode) Skip real GPU query and use the analytic estimate. CI-safe.
        #[arg(long)]
        mock: bool,
    },
}

fn main() {
    // Print Arc banner.
    //
    // D18: the banner states identity only. It must NOT name a subsystem as
    // active — this process has not loaded a model yet, so it cannot know
    // which cache type, attention backend, or MoE path will be resolved. The
    // previous banner claimed "TurboQuant 3.5-bit KV cache compression
    // (lossless, default)" unconditionally; that line was false for every MLA
    // model and every head_dim != 128 (i.e. almost every model), and
    // "lossless" was never measured at all. The engine now reports what it
    // actually resolved, after load, via the ArcServe startup summary.
    eprintln!("Arc inference engine v{}", env!("CARGO_PKG_VERSION"));
    eprintln!("Aeonmind, LLC | https://runcrate.ai/arc");
    eprintln!();

    // Delegate to the mistralrs binary with the same args.
    // The TurboQuant default is already set in mistralrs-core's PagedCacheType.
    //
    // In production, this will be a proper clap integration that reuses
    // the mistralrs-cli command definitions. For now, exec the upstream binary.
    let cli = Cli::parse();

    let (subcmd, args) = match cli.command {
        Commands::Serve { args } => ("serve", args),
        Commands::Run { args } => ("run", args),
        Commands::Bench {
            suite,
            model,
            slo_tier,
            mock,
            max_users,
            output,
            warmup_seconds,
            steady_state_seconds,
            headless,
            legacy_args,
        } => {
            let parsed = match bench::Suite::parse(&suite) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("ERROR: {e}");
                    std::process::exit(2);
                }
            };
            // For `--suite simple`, fall through to the old `mistralrs bench` shell-out.
            if parsed == bench::Suite::Simple {
                ("bench", legacy_args)
            } else {
                let warmup = Duration::from_secs_f64(warmup_seconds.max(0.001));
                let steady = Duration::from_secs_f64(
                    steady_state_seconds.unwrap_or(warmup_seconds).max(0.001),
                );
                let opts = bench::BenchOptions {
                    suite: parsed,
                    model,
                    slo_tier,
                    mock,
                    max_users,
                    output,
                    warmup,
                    steady_state: steady,
                    headless,
                };
                match bench::run(opts) {
                    Ok(code) => std::process::exit(code),
                    Err(e) => {
                        eprintln!("ERROR: {e:?}");
                        std::process::exit(2);
                    }
                }
            }
        }
        Commands::Validate {
            index,
            arch,
            o_proj,
            target_hbm,
            model,
            compression_stack,
            output,
            mock,
        } => {
            // Dispatch: --target-hbm picks HBM mode, otherwise fall through to schema mode.
            if let Some(target_hbm_gb) = target_hbm {
                let stack = match validate::CompressionStack::parse(&compression_stack) {
                    Ok(s) => s,
                    Err(e) => {
                        eprintln!("ERROR: {e}");
                        std::process::exit(2);
                    }
                };
                let opts = validate::HbmValidateOptions {
                    model_id: model,
                    compression_stack: stack,
                    target_hbm_gb,
                    output_path: output,
                    mock,
                };
                match validate::run(opts) {
                    Ok(code) => std::process::exit(code),
                    Err(e) => {
                        eprintln!("ERROR: {e:?}");
                        std::process::exit(2);
                    }
                }
            }

            // Schema mode requires --index and --arch.
            let index = match index {
                Some(p) => p,
                None => {
                    eprintln!(
                        "ERROR: schema-mode `arc validate` requires --index <path> --arch <arch>.\n\
                         For HBM-footprint mode, pass --target-hbm <gb>."
                    );
                    std::process::exit(2);
                }
            };
            let arch = match arch {
                Some(a) => a,
                None => {
                    eprintln!(
                        "ERROR: schema-mode `arc validate` requires --arch (deepseekv4 | kimi_k2 | glm5moedsa)."
                    );
                    std::process::exit(2);
                }
            };
            std::process::exit(run_validate(&index, &arch, &o_proj));
        }
    };

    // Extract Arc-specific flags from `args` before forwarding to mistralrs.
    // The subprocess reads these via env vars (ARC_TD_MOE_RANK, etc.) since
    // mistralrs-cli doesn't natively know about them.
    let (mut extracted_env, forwarded_args) = extract_arc_flags(args);

    // Build the command for the mistralrs binary
    let mut command = std::process::Command::new("mistralrs");
    command.arg(subcmd).args(&forwarded_args);
    for (k, v) in extracted_env.drain() {
        command.env(k, v);
    }
    let status = command.status();

    match status {
        Ok(s) => std::process::exit(s.code().unwrap_or(1)),
        Err(e) => {
            eprintln!("Failed to execute mistralrs: {e}");
            eprintln!("Make sure mistralrs-cli is installed: cargo install --path mistralrs-cli");
            std::process::exit(1);
        }
    }
}

/// Strip Arc-only flags (`--td-moe-rank N`, `--td-moe-calibration N`,
/// `--calib <path>`) from `args` and translate them into environment variables
/// that the spawned `mistralrs` subprocess will pick up via
/// `arc_engine::td_moe_loader::register_td_moe_hook`.
///
/// `--calib <path>` points at a `.arccalib` artifact produced by
/// `mistralrs calibrate`; supplying it switches TD-MoE whitening from the
/// identity to the measured input activation covariance. Omitting it keeps the
/// historical identity behaviour.
///
/// Returns `(env_vars_to_set, remaining_args_to_forward)`.
fn extract_arc_flags(args: Vec<String>) -> (HashMap<String, String>, Vec<String>) {
    let mut env_vars: HashMap<String, String> = HashMap::new();
    let mut out: Vec<String> = Vec::with_capacity(args.len());
    let mut i = 0;
    while i < args.len() {
        let arg = &args[i];
        match arg.as_str() {
            "--td-moe-rank" => {
                if let Some(val) = args.get(i + 1) {
                    env_vars.insert("ARC_TD_MOE_RANK".into(), val.clone());
                    i += 2;
                    continue;
                } else {
                    eprintln!("ERROR: --td-moe-rank requires a value");
                    std::process::exit(2);
                }
            }
            // `--td-moe-calibration` is retired: it set ARC_TD_MOE_CALIBRATION,
            // which arc-engine parsed and then bound to an unread parameter. It
            // never influenced an output. Accept-and-warn for one release so
            // existing invocations keep running instead of failing on an
            // unknown flag; use `--calib <path>` for real calibration.
            "--td-moe-calibration" => {
                if args.get(i + 1).is_some() {
                    eprintln!(
                        "WARNING: --td-moe-calibration is deprecated and has no effect \
                         (it never had one); use --calib <path.arccalib>"
                    );
                    i += 2;
                    continue;
                } else {
                    eprintln!("ERROR: --td-moe-calibration requires a value");
                    std::process::exit(2);
                }
            }
            "--calib" => {
                if let Some(val) = args.get(i + 1) {
                    env_vars.insert("ARC_TD_MOE_CALIB_PATH".into(), val.clone());
                    i += 2;
                    continue;
                } else {
                    eprintln!("ERROR: --calib requires a path to a .arccalib artifact");
                    std::process::exit(2);
                }
            }
            _ => {
                // Also support --flag=value form for the same Arc flags.
                if let Some(rest) = arg.strip_prefix("--td-moe-rank=") {
                    env_vars.insert("ARC_TD_MOE_RANK".into(), rest.to_string());
                    i += 1;
                    continue;
                }
                if arg.strip_prefix("--td-moe-calibration=").is_some() {
                    eprintln!(
                        "WARNING: --td-moe-calibration is deprecated and has no effect \
                         (it never had one); use --calib <path.arccalib>"
                    );
                    i += 1;
                    continue;
                }
                if let Some(rest) = arg.strip_prefix("--calib=") {
                    env_vars.insert("ARC_TD_MOE_CALIB_PATH".into(), rest.to_string());
                    i += 1;
                    continue;
                }
                out.push(arg.clone());
                i += 1;
            }
        }
    }
    (env_vars, out)
}

/// Read a safetensors index, validate against the target architecture's schema,
/// print a structured report, return exit code (0 = OK, 1 = missing tensors).
fn run_validate(index_path: &PathBuf, arch: &str, o_proj_layout: &str) -> i32 {
    use arc_engine::weight_schema::{
        validate_glm5_dsa_weights, validate_kimi_k2_weights, validate_v4_weights_with_layout,
        OProjLayout,
    };

    // Read safetensors index JSON
    let content = match std::fs::read_to_string(index_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("ERROR: failed to read index {}: {e}", index_path.display());
            return 1;
        }
    };

    // The index format: { "metadata": {...}, "weight_map": { "tensor_name": "filename", ... } }
    let parsed: serde_json::Value = match serde_json::from_str(&content) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("ERROR: failed to parse index as JSON: {e}");
            return 1;
        }
    };

    let weight_map = match parsed.get("weight_map").and_then(|v| v.as_object()) {
        Some(m) => m,
        None => {
            eprintln!("ERROR: index has no `weight_map` field");
            return 1;
        }
    };

    let keys: Vec<&str> = weight_map.keys().map(String::as_str).collect();

    eprintln!(
        "Arc preflight: validating {} tensors against `{}` schema",
        keys.len(),
        arch
    );
    eprintln!();

    let validation = match arch {
        "deepseekv4" | "v4" => {
            let layout = match o_proj_layout {
                "single" => OProjLayout::Single,
                "lora" => OProjLayout::Lora,
                _ => OProjLayout::Either,
            };
            validate_v4_weights_with_layout(keys.iter().copied(), layout)
        }
        "kimi_k2" | "k2" => validate_kimi_k2_weights(keys.iter().copied()),
        "glm5moedsa" | "glm5" => validate_glm5_dsa_weights(keys.iter().copied()),
        other => {
            eprintln!(
                "ERROR: unknown --arch `{other}`. Supported: deepseekv4, kimi_k2, glm5moedsa"
            );
            return 1;
        }
    };

    println!("{}", validation.report());

    if validation.is_valid() {
        // Group extras by category for informational output
        let mut extra_categories: HashMap<&str, usize> = HashMap::new();
        for ex in &validation.extra {
            let cat = if ex.contains("compressor") {
                "compressor (V4 CSA/HCA)"
            } else if ex.contains("mtp.layers") {
                "MTP heads"
            } else if ex.contains("vision") || ex.contains("moonvit") {
                "vision tower"
            } else if ex.contains("language_model.") {
                "K2 language_model prefix"
            } else {
                "other"
            };
            *extra_categories.entry(cat).or_insert(0) += 1;
        }
        if !extra_categories.is_empty() {
            eprintln!("Informational — extra tensors present (not required, but interesting):");
            for (cat, count) in &extra_categories {
                eprintln!("  {cat}: {count} tensors");
            }
        }
        eprintln!();
        eprintln!("✓ PRE-FLIGHT OK — model should load through Arc's dispatcher");
        0
    } else {
        eprintln!();
        eprintln!("✗ PRE-FLIGHT FAILED — model will NOT load through Arc as-is");
        eprintln!("  Fix: update the loader to handle the missing tensor layout, OR");
        eprintln!("        check that this checkpoint is for the right architecture.");
        1
    }
}
