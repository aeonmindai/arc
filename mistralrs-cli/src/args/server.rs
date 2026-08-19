//! Server configuration options

use clap::Args;
use serde::Deserialize;
use std::path::PathBuf;

/// HTTP server configuration
#[derive(Args, Clone, Deserialize)]
pub struct ServerOptions {
    /// HTTP server port
    #[arg(short = 'p', long, default_value_t = 1234, help_heading = "Serving")]
    #[serde(default = "default_port")]
    pub port: u16,

    /// Bind address. Defaults to all interfaces.
    #[arg(long, default_value = "0.0.0.0", help_heading = "Serving")]
    #[serde(default = "default_host")]
    pub host: String,

    /// MCP protocol server port (enables MCP if set)
    ///
    /// Hidden: not implemented in this binary — see the error it raises.
    #[arg(long, hide = true)]
    #[serde(default)]
    pub mcp_port: Option<u16>,

    /// MCP client configuration file path
    ///
    /// Hidden: not implemented in this binary — see the error it raises.
    #[arg(long, hide = true)]
    #[serde(default)]
    pub mcp_config: Option<PathBuf>,

    /// Serve the built-in web UI at /ui
    #[arg(long, help_heading = "Serving")]
    #[serde(default)]
    pub ui: bool,
}

impl Default for ServerOptions {
    fn default() -> Self {
        Self {
            port: 1234,
            host: "0.0.0.0".to_string(),
            mcp_port: None,
            mcp_config: None,
            ui: false,
        }
    }
}

fn default_port() -> u16 {
    1234
}

fn default_host() -> String {
    "0.0.0.0".to_string()
}
