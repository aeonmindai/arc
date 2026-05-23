//! Arc inference engine CLI.
//!
//! This is the `arc` binary — a thin wrapper around the mistral.rs CLI
//! that adds Arc branding and defaults to TurboQuant KV cache compression.
//!
//! Usage:
//!   arc serve -m <model_id>                          # Start serving with TurboQuant (default)
//!   arc run -m <model_id>                            # Interactive chat
//!   arc bench -m <model_id>                          # Run benchmarks
//!   arc validate --index <path> --arch <arch>        # Pre-flight weight schema validation
//!   arc validate --target-hbm 60 -m <model_id>       # HBM footprint check (RUN-191)
//!   arc serve --pa-cache-type auto                   # Disable TurboQuant, use upstream defaults

mod validate;

use clap::{Parser, Subcommand};
use std::collections::HashMap;
use std::path::PathBuf;

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
    /// Benchmark a model
    Bench {
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
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
        #[arg(long, default_value = "either")]
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
    // Print Arc banner
    eprintln!("Arc inference engine v{}", env!("CARGO_PKG_VERSION"));
    eprintln!("TurboQuant 3.5-bit KV cache compression (lossless, default)");
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
        Commands::Bench { args } => ("bench", args),
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

    // Build the command for the mistralrs binary
    let status = std::process::Command::new("mistralrs")
        .arg(subcmd)
        .args(&args)
        .status();

    match status {
        Ok(s) => std::process::exit(s.code().unwrap_or(1)),
        Err(e) => {
            eprintln!("Failed to execute mistralrs: {e}");
            eprintln!("Make sure mistralrs-cli is installed: cargo install --path mistralrs-cli");
            std::process::exit(1);
        }
    }
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

    eprintln!("Arc preflight: validating {} tensors against `{}` schema", keys.len(), arch);
    eprintln!();

    let validation = match arch {
        "deepseekv4" | "v4" => {
            let layout = match o_proj_layout {
                "single" => OProjLayout::Single,
                "lora" => OProjLayout::Lora,
                "either" | _ => OProjLayout::Either,
            };
            validate_v4_weights_with_layout(keys.iter().copied(), layout)
        }
        "kimi_k2" | "k2" => validate_kimi_k2_weights(keys.iter().copied()),
        "glm5moedsa" | "glm5" => validate_glm5_dsa_weights(keys.iter().copied()),
        other => {
            eprintln!("ERROR: unknown --arch `{other}`. Supported: deepseekv4, kimi_k2, glm5moedsa");
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
