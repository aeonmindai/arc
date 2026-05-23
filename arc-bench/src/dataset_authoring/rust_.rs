//! Rust trajectories (5).
//!
//! Public-repo provenance is recorded in each `source` field.

use super::{assemble, assistant_call, assistant_text, src, tool, user};
use crate::trajectory::Trajectory;
use serde_json::json;

pub fn all() -> Vec<Trajectory> {
    vec![
        rust_001_substring(),
        rust_002_tokio_oneshot(),
        rust_004_clap_subcommand(),
        rust_005_axum_router_extract(),
        super::long_trajectories::rust_long(),
    ]
}

/// Refactor a string-search helper to use `str::find` (inspired by
/// the `ripgrep` repo's early `searcher` helpers).
fn rust_001_substring() -> Trajectory {
    let mut turns = Vec::new();
    turns.push(user(
        "Refactor the find_substring function in src/searcher.rs to use the standard library's str::find instead of the hand-rolled byte loop. Keep the existing return type and add a doc comment.",
    ));
    turns.push(assistant_call(
        "I'll start by reading the current implementation to see exactly what we are replacing.",
        "call_001",
        "read_file",
        json!({"path":"src/searcher.rs"}),
        &turns,
    ));
    turns.push(tool(
        "call_001",
        "pub fn find_substring(haystack: &str, needle: &str) -> Option<usize> {\n    if needle.is_empty() {\n        return Some(0);\n    }\n    let hb = haystack.as_bytes();\n    let nb = needle.as_bytes();\n    if nb.len() > hb.len() {\n        return None;\n    }\n    for i in 0..=hb.len() - nb.len() {\n        if &hb[i..i + nb.len()] == nb {\n            return Some(i);\n        }\n    }\n    None\n}\n\n#[cfg(test)]\nmod tests {\n    use super::*;\n    #[test]\n    fn empty_needle_returns_zero() {\n        assert_eq!(find_substring(\"abc\", \"\"), Some(0));\n    }\n    #[test]\n    fn finds_first_match() {\n        assert_eq!(find_substring(\"hello world\", \"world\"), Some(6));\n    }\n}\n",
    ));
    turns.push(assistant_call(
        "The existing helper does a manual byte walk. `str::find` already returns Option<usize> for the leftmost match and is well-optimised. I'll rewrite the body and add documentation.",
        "call_002",
        "write_file",
        json!({
            "path":"src/searcher.rs",
            "contents":"/// Return the byte offset of the first occurrence of `needle` in\n/// `haystack`, or `None` if it is not present.\n///\n/// Delegates to [`str::find`], which is implemented with the SIMD\n/// two-way matcher when available.\npub fn find_substring(haystack: &str, needle: &str) -> Option<usize> {\n    haystack.find(needle)\n}\n\n#[cfg(test)]\nmod tests {\n    use super::*;\n    #[test]\n    fn empty_needle_returns_zero() {\n        assert_eq!(find_substring(\"abc\", \"\"), Some(0));\n    }\n    #[test]\n    fn finds_first_match() {\n        assert_eq!(find_substring(\"hello world\", \"world\"), Some(6));\n    }\n}\n"
        }),
        &turns,
    ));
    turns.push(tool("call_002", "wrote 532 bytes to src/searcher.rs"));
    turns.push(assistant_text(
        "Done. `find_substring` now delegates to `str::find`, which preserves the existing return semantics (including `Some(0)` for an empty needle) and removes the bounds-checked manual loop.",
        &turns,
    ));

    assemble(
        "agentperf_tuning_001",
        "rust",
        src("github.com/BurntSushi/ripgrep", "MIT", "2c1d2ea"),
        turns,
    )
}

/// Add a `tokio::sync::oneshot` shutdown signal to a small TCP echo
/// server, modeled on the patterns in `tokio-rs/tokio` examples.
fn rust_002_tokio_oneshot() -> Trajectory {
    let mut turns = Vec::new();
    turns.push(user(
        "Add a graceful shutdown signal to the echo server in examples/echo.rs. Use a tokio::sync::oneshot channel — when the receiver fires, the accept loop should exit and any active connection should drain.",
    ));
    turns.push(assistant_call(
        "Let me see the current server first.",
        "call_001",
        "read_file",
        json!({"path":"examples/echo.rs"}),
        &turns,
    ));
    turns.push(tool(
        "call_001",
        "use tokio::io::{AsyncReadExt, AsyncWriteExt};\nuse tokio::net::TcpListener;\n\n#[tokio::main]\nasync fn main() -> std::io::Result<()> {\n    let listener = TcpListener::bind(\"127.0.0.1:9000\").await?;\n    loop {\n        let (mut socket, _) = listener.accept().await?;\n        tokio::spawn(async move {\n            let mut buf = vec![0u8; 1024];\n            loop {\n                let n = match socket.read(&mut buf).await {\n                    Ok(0) => return,\n                    Ok(n) => n,\n                    Err(_) => return,\n                };\n                if socket.write_all(&buf[..n]).await.is_err() {\n                    return;\n                }\n            }\n        });\n    }\n}\n",
    ));
    turns.push(assistant_call(
        "I will wrap the accept loop in a `select!` that races against the oneshot receiver. The active connections continue until the inner read returns 0, so they drain naturally.",
        "call_002",
        "write_file",
        json!({
            "path":"examples/echo.rs",
            "contents":"use tokio::io::{AsyncReadExt, AsyncWriteExt};\nuse tokio::net::TcpListener;\nuse tokio::sync::oneshot;\n\npub async fn run_with_shutdown(\n    addr: &str,\n    mut shutdown: oneshot::Receiver<()>,\n) -> std::io::Result<()> {\n    let listener = TcpListener::bind(addr).await?;\n    loop {\n        tokio::select! {\n            res = listener.accept() => {\n                let (mut socket, _) = res?;\n                tokio::spawn(async move {\n                    let mut buf = vec![0u8; 1024];\n                    loop {\n                        let n = match socket.read(&mut buf).await {\n                            Ok(0) => return,\n                            Ok(n) => n,\n                            Err(_) => return,\n                        };\n                        if socket.write_all(&buf[..n]).await.is_err() {\n                            return;\n                        }\n                    }\n                });\n            }\n            _ = &mut shutdown => {\n                eprintln!(\"shutdown signal received\");\n                return Ok(());\n            }\n        }\n    }\n}\n\n#[tokio::main]\nasync fn main() -> std::io::Result<()> {\n    let (tx, rx) = oneshot::channel::<()>();\n    tokio::spawn(async move {\n        tokio::signal::ctrl_c().await.ok();\n        let _ = tx.send(());\n    });\n    run_with_shutdown(\"127.0.0.1:9000\", rx).await\n}\n"
        }),
        &turns,
    ));
    turns.push(tool("call_002", "wrote 1224 bytes to examples/echo.rs"));
    turns.push(assistant_call(
        "Now let me confirm it still compiles.",
        "call_003",
        "run_shell",
        json!({"cmd":"cargo check --example echo"}),
        &turns,
    ));
    turns.push(tool(
        "call_003",
        "    Checking demo v0.1.0\n    Finished `dev` profile [optimized + debuginfo] target(s) in 1.42s\n",
    ));
    turns.push(assistant_text(
        "Shutdown wired. The `run_with_shutdown` helper is now testable in isolation, and `main` reproduces the previous behavior by listening for Ctrl-C and firing the oneshot.",
        &turns,
    ));

    assemble(
        "agentperf_tuning_002",
        "rust",
        src("github.com/tokio-rs/tokio", "MIT", "9f88e60"),
        turns,
    )
}

/// Add a Serde derive to a config struct (modeled on the `serde-rs`
/// examples crate). Currently parked — kept for future use when we
/// expand the dataset beyond 50 trajectories.
#[allow(dead_code)]
fn rust_003_serde_derive() -> Trajectory {
    let mut turns = Vec::new();
    turns.push(user(
        "src/config.rs defines a Config struct. Add Deserialize so we can load it from JSON, and add a `from_file` helper that takes &Path.",
    ));
    turns.push(assistant_call(
        "Reading the file.",
        "call_001",
        "read_file",
        json!({"path":"src/config.rs"}),
        &turns,
    ));
    turns.push(tool(
        "call_001",
        "pub struct Config {\n    pub bind: String,\n    pub port: u16,\n    pub workers: usize,\n    pub log_level: String,\n}\n\nimpl Default for Config {\n    fn default() -> Self {\n        Self { bind: \"0.0.0.0\".into(), port: 8080, workers: 4, log_level: \"info\".into() }\n    }\n}\n",
    ));
    turns.push(assistant_call(
        "I'll add `#[derive(Deserialize)]` and a `from_file` that does a `fs::read_to_string` followed by `serde_json::from_str`.",
        "call_002",
        "write_file",
        json!({
            "path":"src/config.rs",
            "contents":"use serde::Deserialize;\nuse std::fs;\nuse std::path::Path;\n\n#[derive(Debug, Clone, Deserialize)]\npub struct Config {\n    pub bind: String,\n    pub port: u16,\n    pub workers: usize,\n    pub log_level: String,\n}\n\nimpl Default for Config {\n    fn default() -> Self {\n        Self { bind: \"0.0.0.0\".into(), port: 8080, workers: 4, log_level: \"info\".into() }\n    }\n}\n\nimpl Config {\n    pub fn from_file(path: &Path) -> anyhow::Result<Self> {\n        let raw = fs::read_to_string(path)?;\n        let cfg: Config = serde_json::from_str(&raw)?;\n        Ok(cfg)\n    }\n}\n"
        }),
        &turns,
    ));
    turns.push(tool("call_002", "wrote 608 bytes to src/config.rs"));
    turns.push(assistant_text(
        "Config now deserialises from JSON via `Config::from_file`.",
        &turns,
    ));

    assemble(
        "agentperf_tuning_003",
        "rust",
        src("github.com/serde-rs/serde", "MIT", "9bb8de7"),
        turns,
    )
}

/// Wire a clap subcommand (modeled on the `clap-rs/clap` derive
/// pattern).
fn rust_004_clap_subcommand() -> Trajectory {
    let mut turns = Vec::new();
    turns.push(user(
        "Add a `migrate` subcommand to the CLI. It should take `--from <db-url>` and `--to <db-url>` (both required). Use clap's derive API and route from main.",
    ));
    turns.push(assistant_call(
        "Reading the existing CLI definition.",
        "call_001",
        "read_file",
        json!({"path":"src/main.rs"}),
        &turns,
    ));
    turns.push(tool(
        "call_001",
        "use clap::{Parser, Subcommand};\n\n#[derive(Parser)]\nstruct Cli {\n    #[command(subcommand)]\n    command: Cmd,\n}\n\n#[derive(Subcommand)]\nenum Cmd {\n    Serve { #[arg(long)] port: u16 },\n}\n\nfn main() {\n    let cli = Cli::parse();\n    match cli.command {\n        Cmd::Serve { port } => println!(\"serving on {port}\"),\n    }\n}\n",
    ));
    turns.push(assistant_call(
        "I'll add a `Migrate` variant with the two required flags and dispatch it.",
        "call_002",
        "write_file",
        json!({
            "path":"src/main.rs",
            "contents":"use clap::{Parser, Subcommand};\n\n#[derive(Parser)]\nstruct Cli {\n    #[command(subcommand)]\n    command: Cmd,\n}\n\n#[derive(Subcommand)]\nenum Cmd {\n    Serve {\n        #[arg(long)] port: u16,\n    },\n    Migrate {\n        #[arg(long)] from: String,\n        #[arg(long)] to: String,\n    },\n}\n\nfn main() {\n    let cli = Cli::parse();\n    match cli.command {\n        Cmd::Serve { port } => println!(\"serving on {port}\"),\n        Cmd::Migrate { from, to } => println!(\"migrating {from} -> {to}\"),\n    }\n}\n"
        }),
        &turns,
    ));
    turns.push(tool("call_002", "wrote 540 bytes to src/main.rs"));
    turns.push(assistant_call(
        "Verify it parses.",
        "call_003",
        "run_shell",
        json!({"cmd":"cargo run -- migrate --from sqlite://a.db --to postgres://b"}),
        &turns,
    ));
    turns.push(tool(
        "call_003",
        "    Finished `dev` profile [optimized + debuginfo] target(s) in 1.21s\n     Running `target/debug/cli migrate --from sqlite://a.db --to postgres://b`\nmigrating sqlite://a.db -> postgres://b\n",
    ));
    turns.push(assistant_text("Subcommand wired and verified.", &turns));

    assemble(
        "agentperf_tuning_004",
        "rust",
        src("github.com/clap-rs/clap", "MIT", "f3d7d3e"),
        turns,
    )
}

/// Add an axum route extractor (medium-length trajectory).
fn rust_005_axum_router_extract() -> Trajectory {
    let mut turns = Vec::new();
    turns.push(user(
        "Add a POST /users handler to the axum router in src/app.rs. It should deserialize a CreateUser JSON body (name, email) and return 201 Created with the new user as JSON. Use tokio_util's request id middleware that's already wired.",
    ));
    turns.push(assistant_call(
        "Reading the router setup.",
        "call_001",
        "read_file",
        json!({"path":"src/app.rs"}),
        &turns,
    ));
    turns.push(tool(
        "call_001",
        "use axum::{Router, routing::get};\n\npub fn app() -> Router {\n    Router::new()\n        .route(\"/health\", get(|| async { \"ok\" }))\n}\n",
    ));
    turns.push(assistant_call(
        "Now reading the existing models file to understand the User shape.",
        "call_002",
        "read_file",
        json!({"path":"src/models.rs"}),
        &turns,
    ));
    turns.push(tool(
        "call_002",
        "use serde::{Deserialize, Serialize};\n\n#[derive(Debug, Clone, Serialize, Deserialize)]\npub struct User {\n    pub id: u64,\n    pub name: String,\n    pub email: String,\n}\n",
    ));
    turns.push(assistant_call(
        "I'll add a `CreateUser` DTO and a handler that builds a User with a hashed id.",
        "call_003",
        "write_file",
        json!({
            "path":"src/app.rs",
            "contents":"use axum::{Json, Router, http::StatusCode, response::IntoResponse, routing::{get, post}};\nuse serde::Deserialize;\nuse crate::models::User;\n\n#[derive(Deserialize)]\npub struct CreateUser {\n    pub name: String,\n    pub email: String,\n}\n\nasync fn create_user(Json(body): Json<CreateUser>) -> impl IntoResponse {\n    // In production we would persist; here we synthesise a hash id.\n    let mut hasher = std::collections::hash_map::DefaultHasher::new();\n    std::hash::Hash::hash(&body.email, &mut hasher);\n    let id = std::hash::Hasher::finish(&hasher);\n    let user = User { id, name: body.name, email: body.email };\n    (StatusCode::CREATED, Json(user))\n}\n\npub fn app() -> Router {\n    Router::new()\n        .route(\"/health\", get(|| async { \"ok\" }))\n        .route(\"/users\", post(create_user))\n}\n"
        }),
        &turns,
    ));
    turns.push(tool("call_003", "wrote 968 bytes to src/app.rs"));
    turns.push(assistant_call(
        "Check it builds.",
        "call_004",
        "run_shell",
        json!({"cmd":"cargo check"}),
        &turns,
    ));
    turns.push(tool(
        "call_004",
        "    Checking demo-axum v0.1.0\n    Finished `dev` profile [optimized + debuginfo] target(s) in 2.04s\n",
    ));
    turns.push(assistant_text(
        "POST /users now decodes a CreateUser body, returns 201 with the new user as JSON, and the route is registered on the existing Router.",
        &turns,
    ));

    assemble(
        "agentperf_tuning_005",
        "rust",
        src("github.com/tokio-rs/axum", "MIT", "8a7a36f"),
        turns,
    )
}
