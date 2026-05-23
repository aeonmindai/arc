//! Integration tests for the replay state machine using a mock
//! vendor.

use arc_bench::{
    ChatMessage, ReplayDiagnostic, Role, ToolCall, Trajectory, TrajectoryReplayer, Vendor,
    VendorResponse,
};
use async_trait::async_trait;
use std::sync::{Arc, Mutex};
use std::time::Duration;

struct ScriptedVendor {
    responses: Mutex<std::collections::VecDeque<VendorResponse>>,
    observed: Arc<Mutex<Vec<Vec<ChatMessage>>>>,
}

impl ScriptedVendor {
    fn new(responses: Vec<VendorResponse>) -> Self {
        Self {
            responses: Mutex::new(responses.into_iter().collect()),
            observed: Arc::new(Mutex::new(Vec::new())),
        }
    }
}

#[async_trait]
impl Vendor for ScriptedVendor {
    async fn complete(
        &self,
        messages: &[ChatMessage],
    ) -> anyhow::Result<VendorResponse> {
        self.observed.lock().unwrap().push(messages.to_vec());
        Ok(self
            .responses
            .lock()
            .unwrap()
            .pop_front()
            .expect("vendor exhausted"))
    }
}

fn canned(text: &str, tool_calls: Vec<ToolCall>) -> VendorResponse {
    VendorResponse {
        time_to_first_token: Duration::from_millis(40),
        total_duration: Duration::from_millis(200),
        output_tokens: 30,
        text: text.to_string(),
        tool_calls,
    }
}

fn first_committed_trajectory() -> Trajectory {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("datasets/agentperf_tuning/trajectories")
        .join("agentperf_tuning_001.json");
    Trajectory::from_path(&path).expect("agentperf_tuning_001 must exist")
}

#[tokio::test]
async fn end_to_end_replay_on_committed_trajectory() {
    let traj = first_committed_trajectory();
    // Build canned responses matching each assistant turn.
    let mut canned_responses = Vec::new();
    for turn in &traj.turns {
        if matches!(turn.role, Role::Assistant) {
            let mut calls = turn.tool_calls.clone();
            // Tweak ids on calls so we keep them in lockstep; the
            // recorded turn already has them so just clone.
            // (Mock just echoes back the recorded calls.)
            for c in &mut calls { /* no-op */
                let _ = c;
            }
            canned_responses.push(canned(&turn.content, calls));
        }
    }
    let vendor = ScriptedVendor::new(canned_responses);
    let observed = vendor.observed.clone();
    let replayer = TrajectoryReplayer::new(traj.clone(), vendor);
    let result = replayer.run().await.expect("replay should succeed");

    let assistant_turns = traj
        .turns
        .iter()
        .filter(|t| matches!(t.role, Role::Assistant))
        .count();
    assert_eq!(result.turns.len(), assistant_turns);
    // The vendor was called once per assistant turn.
    assert_eq!(observed.lock().unwrap().len(), assistant_turns);
    // Every recorded turn should carry zero diagnostics because the
    // mock perfectly echoes the recorded output.
    for tr in &result.turns {
        assert!(tr.diagnostics.is_empty(), "unexpected diagnostics: {:?}", tr.diagnostics);
    }
    // Sanity: aggregate metrics non-zero.
    let total = result.total_output_tokens();
    assert!(total > 0, "no output tokens recorded");
    let mean_ttft = result.mean_ttft().expect("mean ttft");
    assert!(mean_ttft.as_millis() > 0);
}

#[tokio::test]
async fn vendor_called_with_increasing_context() {
    let traj = first_committed_trajectory();
    let mut canned_responses = Vec::new();
    for turn in &traj.turns {
        if matches!(turn.role, Role::Assistant) {
            canned_responses.push(canned(&turn.content, turn.tool_calls.clone()));
        }
    }
    let vendor = ScriptedVendor::new(canned_responses);
    let observed = vendor.observed.clone();
    let replayer = TrajectoryReplayer::new(traj.clone(), vendor);
    let _ = replayer.run().await.expect("replay should succeed");

    // History strictly grows assistant turn over assistant turn.
    let histories = observed.lock().unwrap();
    if histories.len() >= 2 {
        let lens: Vec<usize> = histories.iter().map(|h| h.len()).collect();
        for w in lens.windows(2) {
            assert!(w[1] > w[0], "history must grow: {:?}", lens);
        }
    }
}

#[tokio::test]
async fn missing_tool_call_diagnostic_is_emitted() {
    // Take the committed trajectory but configure the vendor to emit
    // NO tool calls on the first assistant turn. The replayer should
    // surface a `MissingToolCall` diagnostic.
    let traj = first_committed_trajectory();
    let mut canned_responses: Vec<VendorResponse> = Vec::new();
    let mut first_assistant_seen = false;
    for turn in &traj.turns {
        if matches!(turn.role, Role::Assistant) {
            let tool_calls = if first_assistant_seen {
                turn.tool_calls.clone()
            } else {
                first_assistant_seen = true;
                vec![]
            };
            canned_responses.push(canned(&turn.content, tool_calls));
        }
    }
    let vendor = ScriptedVendor::new(canned_responses);
    let replayer = TrajectoryReplayer::new(traj.clone(), vendor);
    let result = replayer.run().await.expect("replay should succeed");
    let first_assistant_recorded = traj
        .turns
        .iter()
        .find(|t| matches!(t.role, Role::Assistant))
        .unwrap();
    if !first_assistant_recorded.tool_calls.is_empty() {
        let first = &result.turns[0];
        let has_missing = first
            .diagnostics
            .iter()
            .any(|d| matches!(d, ReplayDiagnostic::MissingToolCall { .. }));
        assert!(has_missing, "expected MissingToolCall diagnostic");
    }
}
