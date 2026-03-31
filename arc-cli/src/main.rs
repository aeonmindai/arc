//! Arc inference engine CLI.
//!
//! This is the `arc` binary — a thin wrapper around the mistral.rs CLI
//! that adds Arc branding and defaults to TurboQuant KV cache compression.
//!
//! Usage:
//!   arc serve -m <model_id>          # Start serving with TurboQuant (default)
//!   arc run -m <model_id>            # Interactive chat
//!   arc bench -m <model_id>          # Run benchmarks
//!   arc serve --pa-cache-type auto   # Disable TurboQuant, use upstream defaults

use clap::{Parser, Subcommand};

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
