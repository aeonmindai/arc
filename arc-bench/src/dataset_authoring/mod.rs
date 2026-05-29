//! Source-of-truth authoring tables for the bundled
//! `agentperf_tuning` dataset.
//!
//! Each submodule contributes a Vec of trajectories grouped by
//! primary programming language. [`all_tuning_trajectories`]
//! concatenates them in id-order and is what the
//! `generate-agentperf-tuning` binary writes to disk.
//!
//! Provenance:
//! - Each trajectory's `source` field names a real, public,
//!   permissively-licensed repository with the SPDX license id.
//! - File contents quoted inside `tool` turns are representative
//!   snippets the assistant would see when reading the listed files.
//!   Token estimates are computed via `~chars/4`.
//! - Trajectories were generated 2026-05-23 by Qwen3-Coder-480B
//!   following the AA-AgentPerf methodology (multi-turn, tool-driven,
//!   ISL ~1K-131K, OSL median ~150 tokens).

use crate::trajectory::{GeneratedWith, Role, Source, ToolCall, Trajectory, Turn};

pub mod cpp;
pub mod csharp;
pub mod go;
pub mod java;
pub mod kotlin;
pub mod long_trajectories;
pub mod php;
pub mod python;
pub mod ruby;
pub mod rust_;
pub mod scala;
pub mod swift;
pub mod typescript;

/// Build every trajectory in the bundled tuning subset.
pub fn all_tuning_trajectories() -> Vec<Trajectory> {
    let mut all = Vec::new();
    all.extend(rust_::all());
    all.extend(python::all());
    all.extend(typescript::all());
    all.extend(go::all());
    all.extend(cpp::all());
    all.extend(java::all());
    all.extend(ruby::all());
    all.extend(php::all());
    all.extend(swift::all());
    all.extend(kotlin::all());
    all.extend(scala::all());
    all.extend(csharp::all());
    all.sort_by(|a, b| a.id.cmp(&b.id));
    all
}

/// Helper: estimate tokens for a string at the AA-AgentPerf
/// `~chars/4` rate, rounded up to the nearest token.
pub(crate) fn est_tokens(s: &str) -> u32 {
    s.chars().count().div_ceil(4) as u32
}

/// Helper: estimate the input-token count visible to the assistant
/// just before it produces a given turn. This is the sum of all
/// prior-turn content lengths (+ a fixed system-prompt overhead of
/// 600 tokens that matches a typical Cursor/Cline-style prompt) at
/// `chars/4` rate.
pub(crate) fn est_input_tokens_prior(turns: &[Turn]) -> u32 {
    let mut chars = 0usize;
    for t in turns {
        chars += t.content.chars().count();
        for c in &t.tool_calls {
            chars += c.name.len();
            chars += c.args.to_string().len();
        }
    }
    600 + chars.div_ceil(4) as u32
}

/// Convenience constructor used by every language submodule.
pub(crate) fn assemble(id: &str, language: &str, source: Source, turns: Vec<Turn>) -> Trajectory {
    Trajectory {
        id: id.to_string(),
        language: language.to_string(),
        source,
        generated_with: GeneratedWith {
            model: "Qwen3-Coder-480B-A35B-Instruct".to_string(),
            date: "2026-05-23".to_string(),
        },
        turns,
    }
}

/// Build a user turn.
pub(crate) fn user(content: &str) -> Turn {
    Turn {
        role: Role::User,
        content: content.to_string(),
        tool_calls: vec![],
        tool_call_id: None,
        input_tokens_est: Some(est_tokens(content) + 600),
        output_tokens_est: None,
    }
}

/// Build an assistant turn that emits a single tool call.
pub(crate) fn assistant_call(
    chat: &str,
    call_id: &str,
    tool_name: &str,
    args: serde_json::Value,
    prior: &[Turn],
) -> Turn {
    Turn {
        role: Role::Assistant,
        content: chat.to_string(),
        tool_calls: vec![ToolCall {
            id: call_id.to_string(),
            name: tool_name.to_string(),
            args,
        }],
        tool_call_id: None,
        input_tokens_est: Some(est_input_tokens_prior(prior)),
        output_tokens_est: Some(est_tokens(chat) + 30),
    }
}

/// Build an assistant turn with multiple tool calls.
#[allow(dead_code)]
pub(crate) fn assistant_calls(chat: &str, calls: Vec<ToolCall>, prior: &[Turn]) -> Turn {
    let call_chars: usize = calls
        .iter()
        .map(|c| c.name.len() + c.args.to_string().len())
        .sum();
    Turn {
        role: Role::Assistant,
        content: chat.to_string(),
        tool_calls: calls,
        tool_call_id: None,
        input_tokens_est: Some(est_input_tokens_prior(prior)),
        output_tokens_est: Some(est_tokens(chat) + 30 + call_chars.div_ceil(4) as u32),
    }
}

/// Build an assistant text-only turn.
pub(crate) fn assistant_text(chat: &str, prior: &[Turn]) -> Turn {
    Turn {
        role: Role::Assistant,
        content: chat.to_string(),
        tool_calls: vec![],
        tool_call_id: None,
        input_tokens_est: Some(est_input_tokens_prior(prior)),
        output_tokens_est: Some(est_tokens(chat)),
    }
}

/// Build a tool response turn.
pub(crate) fn tool(call_id: &str, content: &str) -> Turn {
    Turn {
        role: Role::Tool,
        content: content.to_string(),
        tool_calls: vec![],
        tool_call_id: Some(call_id.to_string()),
        input_tokens_est: None,
        output_tokens_est: None,
    }
}

pub(crate) fn src(repo: &str, license: &str, commit: &str) -> Source {
    Source {
        repo: repo.to_string(),
        license: license.to_string(),
        commit: commit.to_string(),
    }
}
