//! mistralrs-cli - Clean command-line interface for mistral.rs
//!
//! A new CLI design with:
//! - Orthogonal flags (format, adapter, modality are independent)
//! - Unified PagedAttention configuration
//! - Logical argument grouping
//! - Config-file-first support

mod args;
mod commands;
mod config;
mod ui;

use anyhow::Result;
use clap::{Command as ClapCommand, CommandFactory, FromArgMatches};
use clap_complete::generate;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

use args::{
    resolve_calibrate_model_type, resolve_model_type, resolve_quantize_model_type, CacheCommand,
    Cli, Command,
};
use commands::{
    run_bench, run_cache_delete, run_cache_list, run_calibrate, run_doctor, run_from_config,
    run_interactive, run_login, run_quantize, run_server, run_tune,
};

/// Recursively un-hide every argument so `--help-all` can render the complete
/// surface.
///
/// Hiding a flag must never make it undiscoverable — the short `--help` answers
/// "what do I need to type", and this answers "what else exists". Every hidden
/// flag keeps working exactly as before; it only leaves the default menu.
fn unhide_all(cmd: ClapCommand) -> ClapCommand {
    cmd.mut_args(|a| a.hide(false)).mut_subcommands(unhide_all)
}

/// Advertise `--help-all` on every subcommand, not just the root — a user who
/// types `arc serve --help` is exactly the person who needs to know the short
/// list is deliberate and that the rest is one flag away.
const HELP_ALL_HINT: &str = "Arc chooses sensible defaults for everything not listed here.\n\
     Run `--help-all` (works on any subcommand) for the complete set,\n\
     including quantisation internals, device-map planning and debug switches.\n\
     On startup Arc logs an `ArcServe:` line reporting what it actually resolved.";

fn with_help_all_hint(cmd: ClapCommand) -> ClapCommand {
    cmd.after_help(HELP_ALL_HINT)
        .mut_subcommands(with_help_all_hint)
}

/// Handle `--help-all` before clap parses, since it is not a real argument.
///
/// Returns `true` if help was printed and the process should exit.
fn print_help_all_if_requested() -> bool {
    if !std::env::args().any(|a| a == "--help-all") {
        return false;
    }
    // Render the full help for the deepest subcommand named on the command
    // line, so `arc serve --help-all` shows serve's flags rather than the top
    // level. Anything that is not a known subcommand (values, flags) stops the
    // descent.
    let mut cmd = unhide_all(Cli::command());
    for token in std::env::args().skip(1) {
        if token.starts_with('-') {
            continue;
        }
        let matched = cmd
            .get_subcommands()
            .find(|s| s.get_name() == token || s.get_all_aliases().any(|a| a == token))
            .cloned();
        match matched {
            Some(sub) => cmd = sub,
            None => break,
        }
    }
    cmd.print_long_help().ok();
    println!();
    true
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing (can be customized via RUST_LOG env var)
    tracing_subscriber::registry()
        .with(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")))
        .with(tracing_subscriber::fmt::layer())
        .init();

    // Arc: register the TD-MoE post-load hook. This is a no-op unless the
    // ARC_TD_MOE_RANK env var is set (which arc-cli sets when the user
    // passes --td-moe-rank N). When active, the hook compresses MoE expert
    // weights via whitened Tucker decomposition after model load.
    arc_engine::td_moe_loader::register_td_moe_hook();

    if print_help_all_if_requested() {
        return Ok(());
    }

    let matches = with_help_all_hint(Cli::command()).get_matches();
    let cli = match Cli::from_arg_matches(&matches) {
        Ok(cli) => cli,
        Err(e) => e.exit(),
    };

    match cli.command {
        Command::Serve {
            model_type,
            default_model,
            server,
            runtime,
        } => {
            let model_type = resolve_model_type(model_type, default_model)?;
            run_server(model_type, server, runtime, cli.global).await?;
        }

        Command::Run {
            model_type,
            default_model,
            runtime,
            enable_thinking,
        } => {
            let model_type = resolve_model_type(model_type, default_model)?;
            run_interactive(model_type, runtime, cli.global, enable_thinking).await?;
        }

        Command::Completions { shell } => {
            let mut cmd = Cli::command();
            let name = cmd.get_name().to_string();
            generate(shell, &mut cmd, name, &mut std::io::stdout());
        }

        Command::Quantize {
            model_type,
            default_quantize,
        } => {
            let model_type = resolve_quantize_model_type(model_type, default_quantize)?;
            run_quantize(model_type, cli.global).await?;
        }

        Command::Calibrate {
            model_type,
            default_calibrate,
        } => {
            let model_type = resolve_calibrate_model_type(model_type, default_calibrate)?;
            run_calibrate(model_type, cli.global).await?;
        }

        Command::FromConfig { file } => {
            run_from_config(file).await?;
        }

        Command::Doctor { json } => {
            run_doctor(json)?;
        }

        Command::Tune {
            model_type,
            default_model,
            profile,
            json,
            emit_config,
        } => {
            let model_type = resolve_model_type(model_type, default_model)?;
            run_tune(model_type, cli.global, profile, json, emit_config).await?;
        }

        Command::Login { token } => {
            run_login(token)?;
        }

        Command::Cache { cmd } => match cmd {
            CacheCommand::List => run_cache_list()?,
            CacheCommand::Delete { model_id } => run_cache_delete(&model_id)?,
        },

        Command::Bench {
            model_type,
            default_model,
            runtime,
            prompt_len,
            gen_len,
            iterations,
            warmup,
        } => {
            let model_type = resolve_model_type(model_type, default_model)?;
            run_bench(
                model_type, runtime, cli.global, prompt_len, gen_len, iterations, warmup,
            )
            .await?;
        }
    }

    Ok(())
}
