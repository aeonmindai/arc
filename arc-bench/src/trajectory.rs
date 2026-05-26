//! AA-AgentPerf trajectory schema.
//!
//! A trajectory is a recorded multi-turn conversation between a user, an
//! AI coding assistant, and the tools the assistant invokes during the
//! conversation. Trajectories are produced by capturing real agentic
//! coding sessions (or by faithfully reconstructing them from public
//! source files) and then replayed against an inference engine to
//! measure throughput / latency under realistic agentic workloads.
//!
//! See `arc-bench/datasets/agentperf_tuning/README.md` for provenance.

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

/// A single tool invocation recorded by the assistant.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct ToolCall {
    /// Stable id assigned at trajectory-generation time. Used to thread
    /// `tool` responses back to their originating call.
    pub id: String,
    /// Tool name as the assistant emitted it (e.g. `read_file`,
    /// `write_file`, `run_shell`).
    pub name: String,
    /// Tool arguments serialised as a JSON object.
    pub args: serde_json::Value,
}

/// Role of a trajectory turn. Maps directly to OpenAI chat completion
/// roles.
#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

/// One turn of a trajectory.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct Turn {
    pub role: Role,
    #[serde(default)]
    pub content: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_tokens_est: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_tokens_est: Option<u32>,
}

/// Provenance metadata describing where the trajectory came from.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct Source {
    pub repo: String,
    pub license: String,
    pub commit: String,
}

/// Metadata about how the trajectory was generated.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct GeneratedWith {
    pub model: String,
    pub date: String,
}

/// A complete trajectory ready to be replayed.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct Trajectory {
    pub id: String,
    pub language: String,
    pub source: Source,
    pub generated_with: GeneratedWith,
    pub turns: Vec<Turn>,
}

/// Parse error surfaced from [`Trajectory::from_path`] /
/// [`Trajectory::from_str`].
#[derive(Debug, thiserror::Error)]
pub enum ParseError {
    #[error("io error reading trajectory {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("json error in trajectory {path}: {source}")]
    Json {
        path: String,
        #[source]
        source: serde_json::Error,
    },
    #[error("validation error in trajectory {id}: {reason}")]
    Validation { id: String, reason: String },
}

impl Trajectory {
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, ParseError> {
        let path_ref = path.as_ref();
        let path_str = path_ref.display().to_string();
        let bytes = fs::read(path_ref).map_err(|e| ParseError::Io {
            path: path_str.clone(),
            source: e,
        })?;
        let trajectory: Trajectory =
            serde_json::from_slice(&bytes).map_err(|e| ParseError::Json {
                path: path_str.clone(),
                source: e,
            })?;
        trajectory.validate()?;
        Ok(trajectory)
    }

    // Inherent `from_str` mirrors `from_path`; keeping the name (rather than a
    // `FromStr` impl) preserves the call sites and the parallel API surface.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Result<Self, ParseError> {
        let trajectory: Trajectory = serde_json::from_str(s).map_err(|e| ParseError::Json {
            path: "<inline>".to_string(),
            source: e,
        })?;
        trajectory.validate()?;
        Ok(trajectory)
    }

    pub fn total_input_tokens_est(&self) -> u64 {
        self.turns
            .iter()
            .filter_map(|t| t.input_tokens_est)
            .map(|x| x as u64)
            .sum()
    }

    pub fn total_output_tokens_est(&self) -> u64 {
        self.turns
            .iter()
            .filter_map(|t| t.output_tokens_est)
            .map(|x| x as u64)
            .sum()
    }

    pub fn assistant_turn_count(&self) -> usize {
        self.turns
            .iter()
            .filter(|t| matches!(t.role, Role::Assistant))
            .count()
    }

    pub fn validate(&self) -> Result<(), ParseError> {
        let bail = |reason: String| ParseError::Validation {
            id: self.id.clone(),
            reason,
        };

        if self.id.is_empty() {
            return Err(bail("id must not be empty".to_string()));
        }
        if self.language.is_empty() {
            return Err(bail("language must not be empty".to_string()));
        }
        if self.turns.is_empty() {
            return Err(bail(
                "trajectory must contain at least one turn".to_string(),
            ));
        }
        let first_non_system = self.turns.iter().find(|t| !matches!(t.role, Role::System));
        if let Some(turn) = first_non_system {
            if !matches!(turn.role, Role::User) {
                return Err(bail("first non-system turn must be `user`".to_string()));
            }
        }

        let mut known_tool_call_ids: Vec<String> = Vec::new();
        for (idx, turn) in self.turns.iter().enumerate() {
            match turn.role {
                Role::Assistant => {
                    for call in &turn.tool_calls {
                        if call.id.is_empty() {
                            return Err(bail(format!(
                                "turn {idx}: assistant tool_call missing id"
                            )));
                        }
                        known_tool_call_ids.push(call.id.clone());
                    }
                    if turn.content.is_empty() && turn.tool_calls.is_empty() {
                        return Err(bail(format!(
                            "turn {idx}: assistant must have content or tool_calls"
                        )));
                    }
                }
                Role::Tool => {
                    let id = turn.tool_call_id.as_deref().ok_or_else(|| {
                        bail(format!("turn {idx}: tool turn missing tool_call_id"))
                    })?;
                    if !known_tool_call_ids.iter().any(|k| k == id) {
                        return Err(bail(format!(
                            "turn {idx}: tool_call_id {id:?} not seen in any prior assistant turn"
                        )));
                    }
                }
                Role::User | Role::System => {}
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        let traj = Trajectory {
            id: "test_001".to_string(),
            language: "rust".to_string(),
            source: Source {
                repo: "github.com/example/repo".to_string(),
                license: "MIT".to_string(),
                commit: "deadbeef".to_string(),
            },
            generated_with: GeneratedWith {
                model: "test-model".to_string(),
                date: "2026-05-23".to_string(),
            },
            turns: vec![
                Turn {
                    role: Role::User,
                    content: "hi".to_string(),
                    tool_calls: vec![],
                    tool_call_id: None,
                    input_tokens_est: Some(2),
                    output_tokens_est: None,
                },
                Turn {
                    role: Role::Assistant,
                    content: "hello".to_string(),
                    tool_calls: vec![],
                    tool_call_id: None,
                    input_tokens_est: Some(2),
                    output_tokens_est: Some(1),
                },
            ],
        };
        let json = serde_json::to_string(&traj).unwrap();
        let parsed = Trajectory::from_str(&json).unwrap();
        assert_eq!(parsed, traj);
    }

    #[test]
    fn rejects_dangling_tool_id() {
        let json = r#"{
            "id": "bad",
            "language": "rust",
            "source": {"repo":"x","license":"MIT","commit":"a"},
            "generated_with": {"model":"m","date":"2026-05-23"},
            "turns": [
                {"role":"user","content":"go"},
                {"role":"tool","content":"oops","tool_call_id":"nope"}
            ]
        }"#;
        let err = Trajectory::from_str(json).unwrap_err();
        assert!(matches!(err, ParseError::Validation { .. }));
    }
}
