//! Multi-turn replay engine.
//!
//! The replayer walks a [`Trajectory`] turn-by-turn, sending the user
//! and recorded tool messages to a `Vendor`, asking the vendor to
//! generate the next assistant response, and recording wall-clock /
//! TTFT / token metrics. Tool execution is *not* performed at replay
//! time: when an assistant emits a tool call, the recorded `tool`
//! response from the trajectory is injected directly into the
//! conversation. This keeps the replay deterministic across runs of a
//! stochastic model.

use crate::trajectory::{Role, ToolCall, Trajectory, Turn};
use async_trait::async_trait;
use std::time::{Duration, Instant};

/// A streaming response from an inference vendor.
#[derive(Debug, Clone)]
pub struct VendorResponse {
    pub time_to_first_token: Duration,
    pub total_duration: Duration,
    pub output_tokens: u32,
    pub text: String,
    pub tool_calls: Vec<ToolCall>,
}

/// A pluggable inference backend.
#[async_trait]
pub trait Vendor: Send + Sync {
    async fn complete(&self, messages: &[ChatMessage]) -> anyhow::Result<VendorResponse>;
}

/// Chat message in OpenAI-compatible shape.
#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: Role,
    pub content: String,
    pub tool_calls: Vec<ToolCall>,
    pub tool_call_id: Option<String>,
}

impl ChatMessage {
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: content.into(),
            tool_calls: vec![],
            tool_call_id: None,
        }
    }

    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: content.into(),
            tool_calls: vec![],
            tool_call_id: None,
        }
    }

    pub fn assistant_text(content: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: content.into(),
            tool_calls: vec![],
            tool_call_id: None,
        }
    }

    pub fn assistant_with_tools(
        content: impl Into<String>,
        tool_calls: Vec<ToolCall>,
    ) -> Self {
        Self {
            role: Role::Assistant,
            content: content.into(),
            tool_calls,
            tool_call_id: None,
        }
    }

    pub fn tool(call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: Role::Tool,
            content: content.into(),
            tool_calls: vec![],
            tool_call_id: Some(call_id.into()),
        }
    }
}

/// Per-turn metrics collected during replay.
#[derive(Debug, Clone)]
pub struct TurnResult {
    pub turn_index: usize,
    pub ttft: Duration,
    pub total_duration: Duration,
    pub prompt_chars: usize,
    pub output_tokens: u32,
    pub recorded: Turn,
    pub vendor_response: VendorResponse,
    pub diagnostics: Vec<ReplayDiagnostic>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReplayDiagnostic {
    UnexpectedToolCall { name: String, call_id: String },
    MissingToolCall { expected_id: String },
}

/// Result of replaying a full trajectory.
#[derive(Debug, Clone)]
pub struct ReplayResult {
    pub trajectory_id: String,
    pub turns: Vec<TurnResult>,
    pub total_wall: Duration,
}

impl ReplayResult {
    pub fn mean_ttft(&self) -> Option<Duration> {
        if self.turns.is_empty() {
            return None;
        }
        let total: Duration = self.turns.iter().map(|t| t.ttft).sum();
        Some(total / self.turns.len() as u32)
    }

    pub fn total_output_tokens(&self) -> u64 {
        self.turns
            .iter()
            .map(|t| t.output_tokens as u64)
            .sum()
    }
}

/// The replay state machine.
pub struct TrajectoryReplayer<V: Vendor> {
    trajectory: Trajectory,
    vendor: V,
}

impl<V: Vendor> TrajectoryReplayer<V> {
    pub fn new(trajectory: Trajectory, vendor: V) -> Self {
        Self {
            trajectory,
            vendor,
        }
    }

    pub fn trajectory(&self) -> &Trajectory {
        &self.trajectory
    }

    /// Replay the trajectory end-to-end against the configured vendor.
    pub async fn run(&self) -> anyhow::Result<ReplayResult> {
        let start = Instant::now();
        let mut history: Vec<ChatMessage> = Vec::new();
        let mut results: Vec<TurnResult> = Vec::new();

        let turns = &self.trajectory.turns;
        let mut i = 0usize;
        while i < turns.len() {
            let turn = &turns[i];
            match turn.role {
                Role::System => {
                    history.push(ChatMessage::system(turn.content.clone()));
                    i += 1;
                }
                Role::User => {
                    history.push(ChatMessage::user(turn.content.clone()));
                    i += 1;
                }
                Role::Assistant => {
                    let mut diagnostics = Vec::new();
                    let prompt_chars: usize =
                        history.iter().map(|m| m.content.len()).sum();
                    let vendor_response = self.vendor.complete(&history).await?;

                    if !turn.tool_calls.is_empty() && vendor_response.tool_calls.is_empty() {
                        for expected_call in &turn.tool_calls {
                            diagnostics.push(ReplayDiagnostic::MissingToolCall {
                                expected_id: expected_call.id.clone(),
                            });
                        }
                    }
                    for emitted_call in &vendor_response.tool_calls {
                        let was_expected = turn
                            .tool_calls
                            .iter()
                            .any(|recorded| recorded.id == emitted_call.id);
                        if !was_expected {
                            diagnostics.push(ReplayDiagnostic::UnexpectedToolCall {
                                name: emitted_call.name.clone(),
                                call_id: emitted_call.id.clone(),
                            });
                        }
                    }

                    history.push(ChatMessage::assistant_with_tools(
                        turn.content.clone(),
                        turn.tool_calls.clone(),
                    ));

                    results.push(TurnResult {
                        turn_index: i,
                        ttft: vendor_response.time_to_first_token,
                        total_duration: vendor_response.total_duration,
                        prompt_chars,
                        output_tokens: vendor_response.output_tokens,
                        recorded: turn.clone(),
                        vendor_response,
                        diagnostics,
                    });
                    i += 1;

                    while i < turns.len() && matches!(turns[i].role, Role::Tool) {
                        let tool_turn = &turns[i];
                        let call_id = tool_turn
                            .tool_call_id
                            .clone()
                            .unwrap_or_default();
                        history.push(ChatMessage::tool(call_id, tool_turn.content.clone()));
                        i += 1;
                    }
                }
                Role::Tool => {
                    i += 1;
                }
            }
        }

        Ok(ReplayResult {
            trajectory_id: self.trajectory.id.clone(),
            turns: results,
            total_wall: start.elapsed(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trajectory::{GeneratedWith, Source};
    use std::sync::Arc;
    use std::sync::Mutex;

    struct MockVendor {
        responses: Mutex<std::collections::VecDeque<VendorResponse>>,
        observed_histories: Arc<Mutex<Vec<Vec<ChatMessage>>>>,
    }

    impl MockVendor {
        fn new(responses: Vec<VendorResponse>) -> Self {
            Self {
                responses: Mutex::new(responses.into_iter().collect()),
                observed_histories: Arc::new(Mutex::new(Vec::new())),
            }
        }
    }

    #[async_trait]
    impl Vendor for MockVendor {
        async fn complete(
            &self,
            messages: &[ChatMessage],
        ) -> anyhow::Result<VendorResponse> {
            self.observed_histories
                .lock()
                .unwrap()
                .push(messages.to_vec());
            let response = self
                .responses
                .lock()
                .unwrap()
                .pop_front()
                .ok_or_else(|| anyhow::anyhow!("mock vendor exhausted"))?;
            Ok(response)
        }
    }

    fn dummy_response(text: &str, tool_calls: Vec<ToolCall>) -> VendorResponse {
        VendorResponse {
            time_to_first_token: Duration::from_millis(40),
            total_duration: Duration::from_millis(200),
            output_tokens: 30,
            text: text.to_string(),
            tool_calls,
        }
    }

    fn simple_trajectory() -> Trajectory {
        Trajectory {
            id: "t".to_string(),
            language: "rust".to_string(),
            source: Source {
                repo: "x".to_string(),
                license: "MIT".to_string(),
                commit: "a".to_string(),
            },
            generated_with: GeneratedWith {
                model: "m".to_string(),
                date: "2026-05-23".to_string(),
            },
            turns: vec![
                Turn {
                    role: Role::User,
                    content: "Refactor".to_string(),
                    tool_calls: vec![],
                    tool_call_id: None,
                    input_tokens_est: Some(10),
                    output_tokens_est: None,
                },
                Turn {
                    role: Role::Assistant,
                    content: "I'll read first.".to_string(),
                    tool_calls: vec![ToolCall {
                        id: "c1".to_string(),
                        name: "read_file".to_string(),
                        args: serde_json::json!({"path":"src/lib.rs"}),
                    }],
                    tool_call_id: None,
                    input_tokens_est: Some(10),
                    output_tokens_est: Some(30),
                },
                Turn {
                    role: Role::Tool,
                    content: "fn main() {}".to_string(),
                    tool_calls: vec![],
                    tool_call_id: Some("c1".to_string()),
                    input_tokens_est: None,
                    output_tokens_est: None,
                },
                Turn {
                    role: Role::Assistant,
                    content: "Done.".to_string(),
                    tool_calls: vec![],
                    tool_call_id: None,
                    input_tokens_est: Some(40),
                    output_tokens_est: Some(20),
                },
            ],
        }
    }

    #[tokio::test]
    async fn replay_threads_history_and_injects_tools() {
        let vendor = MockVendor::new(vec![
            dummy_response(
                "I'll read first.",
                vec![ToolCall {
                    id: "c1".to_string(),
                    name: "read_file".to_string(),
                    args: serde_json::json!({"path":"src/lib.rs"}),
                }],
            ),
            dummy_response("Done.", vec![]),
        ]);
        let histories = vendor.observed_histories.clone();
        let replayer = TrajectoryReplayer::new(simple_trajectory(), vendor);
        let result = replayer.run().await.unwrap();
        assert_eq!(result.turns.len(), 2);
        let snapshot = histories.lock().unwrap().clone();
        let h0 = &snapshot[0];
        assert_eq!(h0.len(), 1);
        assert_eq!(h0[0].role, Role::User);
        let h1 = &snapshot[1];
        assert_eq!(h1.len(), 3);
        assert_eq!(h1[0].role, Role::User);
        assert_eq!(h1[1].role, Role::Assistant);
        assert_eq!(h1[1].tool_calls.len(), 1);
        assert_eq!(h1[2].role, Role::Tool);
        assert_eq!(h1[2].content, "fn main() {}");
    }

    #[tokio::test]
    async fn replay_records_missing_tool_call_diag() {
        let vendor = MockVendor::new(vec![
            dummy_response("Oops, no tool.", vec![]),
            dummy_response("Done.", vec![]),
        ]);
        let replayer = TrajectoryReplayer::new(simple_trajectory(), vendor);
        let result = replayer.run().await.unwrap();
        let first = &result.turns[0];
        assert!(first
            .diagnostics
            .iter()
            .any(|d| matches!(d, ReplayDiagnostic::MissingToolCall { .. })));
    }

    #[tokio::test]
    async fn replay_records_unexpected_tool_call_diag() {
        let vendor = MockVendor::new(vec![
            dummy_response(
                "I'll read first.",
                vec![
                    ToolCall {
                        id: "c1".to_string(),
                        name: "read_file".to_string(),
                        args: serde_json::json!({}),
                    },
                    ToolCall {
                        id: "c99".to_string(),
                        name: "list_dir".to_string(),
                        args: serde_json::json!({}),
                    },
                ],
            ),
            dummy_response("Done.", vec![]),
        ]);
        let replayer = TrajectoryReplayer::new(simple_trajectory(), vendor);
        let result = replayer.run().await.unwrap();
        assert!(result.turns[0]
            .diagnostics
            .iter()
            .any(|d| matches!(
                d,
                ReplayDiagnostic::UnexpectedToolCall { call_id, .. } if call_id == "c99"
            )));
    }

    #[tokio::test]
    async fn replay_metrics_aggregate() {
        let vendor = MockVendor::new(vec![
            dummy_response(
                "tool",
                vec![ToolCall {
                    id: "c1".to_string(),
                    name: "read_file".to_string(),
                    args: serde_json::json!({}),
                }],
            ),
            dummy_response("done", vec![]),
        ]);
        let replayer = TrajectoryReplayer::new(simple_trajectory(), vendor);
        let result = replayer.run().await.unwrap();
        assert_eq!(result.total_output_tokens(), 60);
        let mean = result.mean_ttft().unwrap();
        assert_eq!(mean, Duration::from_millis(40));
    }

    // Suppress the dead-code warning for `assistant_text`, which is
    // part of the public API but not exercised in unit tests directly
    // because we use `assistant_with_tools` to be more explicit.
    #[test]
    fn assistant_text_helper_compiles() {
        let _ = ChatMessage::assistant_text("hi");
    }
}
