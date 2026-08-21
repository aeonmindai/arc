//! SSE streaming utilities.

use std::env;

use mistralrs_core::Response;
use tokio::sync::mpsc::Receiver;

use crate::types::SharedMistralRsState;

/// Default keep-alive interval for Server-Sent Events (SSE) streams in milliseconds.
pub const DEFAULT_KEEP_ALIVE_INTERVAL_MS: u64 = 10_000;

/// Represents the current state of a streaming response.
pub enum DoneState {
    /// The stream is actively processing and sending response chunks
    Running,
    /// The stream has finished processing and is about to send the `[DONE]` message
    SendingDone,
    /// The stream has completed entirely
    Done,
}

/// A streaming response handler.
///
/// It processes incoming response chunks from a model and converts them
/// into Server-Sent Events (SSE) format for real-time streaming to clients.
pub struct BaseStreamer<R, C, D> {
    /// Channel receiver for incoming model responses
    pub rx: Receiver<Response>,
    /// Current state of the streaming operation
    pub done_state: DoneState,
    /// Underlying mistral.rs instance
    pub state: SharedMistralRsState,
    /// Whether to store chunks for the completion callback
    pub store_chunks: bool,
    /// All chunks received during streaming (if `store_chunks` is true)
    pub chunks: Vec<R>,
    /// Optional callback to process each chunk before sending
    pub on_chunk: Option<C>,
    /// Optional callback to execute when streaming completes
    pub on_done: Option<D>,
}

/// Generic function to create a SSE streamer with optional callbacks.
pub(crate) fn base_create_streamer<R, C, D>(
    rx: Receiver<Response>,
    state: SharedMistralRsState,
    on_chunk: Option<C>,
    on_done: Option<D>,
) -> BaseStreamer<R, C, D> {
    let store_chunks = on_done.is_some();

    BaseStreamer {
        rx,
        done_state: DoneState::Running,
        store_chunks,
        state,
        chunks: Vec::new(),
        on_chunk,
        on_done,
    }
}

/// Gets the keep-alive interval for SSE streams from environment or default.
pub fn get_keep_alive_interval() -> u64 {
    env::var("KEEP_ALIVE_INTERVAL")
        .map(|val| {
            val.parse::<u64>().unwrap_or_else(|e| {
                tracing::warn!("Failed to parse KEEP_ALIVE_INTERVAL: {}. Using default.", e);
                DEFAULT_KEEP_ALIVE_INTERVAL_MS
            })
        })
        .unwrap_or(DEFAULT_KEEP_ALIVE_INTERVAL_MS)
}

// ---------------------------------------------------------------------------
// Stream-terminating errors must be VISIBLE to an OpenAI-compatible client.
// ---------------------------------------------------------------------------

/// The OpenAI error envelope, as it appears in a Server-Sent Events stream.
///
/// Why this exists: a streaming error used to be emitted as `Event::default()
/// .data(message)` — a BARE STRING in the `data:` field. Every OpenAI-compatible
/// client parses `data:` as JSON, so a bare string is either a parse error it
/// discards or a frame it skips. The observable result was an HTTP **200** with
/// an immediate `[DONE]` and zero tokens: indistinguishable from "the model had
/// nothing to say".
///
/// The error was always being sent. It was being sent in a shape nothing could
/// read, which is the same failure as not sending it. Once the SSE stream has
/// started the status line is already 200 and cannot be retracted, so the body
/// is the only channel left — and it has to be in the shape clients decode.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SseErrorEnvelope {
    pub error: SseErrorBody,
}

/// The inner object of an OpenAI error envelope.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SseErrorBody {
    /// Human-readable description. This must carry the actual cause.
    pub message: String,
    /// OpenAI error class, e.g. `invalid_request_error` / `internal_error`.
    #[serde(rename = "type")]
    pub kind: String,
    pub param: Option<String>,
    pub code: Option<String>,
}

/// OpenAI error classes used by the streaming paths.
pub mod sse_error_kind {
    /// The request itself cannot be served (missing chat template, bad args).
    pub const INVALID_REQUEST: &str = "invalid_request_error";
    /// The server failed while serving an otherwise valid request.
    pub const INTERNAL: &str = "internal_error";
    /// The model errored mid-generation.
    pub const MODEL: &str = "model_error";
}

/// Build a stream-terminating error frame that a client can actually decode.
///
/// Serialization cannot fail for this type (all fields are plain strings), but
/// if it somehow did we must still say something legible rather than emit an
/// empty frame — an empty frame is the very bug this function exists to remove.
pub fn sse_error_payload(kind: &str, message: impl Into<String>) -> SseErrorEnvelope {
    SseErrorEnvelope {
        error: SseErrorBody {
            message: message.into(),
            kind: kind.to_string(),
            param: None,
            code: None,
        },
    }
}

/// What the client is told when the engine drops a request's responder without
/// ever sending a terminal chunk.
///
/// This is a SECOND way to produce an HTTP 200 with zero tokens, distinct from
/// the bare-string frame above and not fixed by it. Several engine paths make a
/// sequence terminal without constructing any `Response` at all — a
/// `FinishedIgnored` sequence whose KV allocation failed even after preemption
/// (`paged_attention/scheduler.rs`), and a `Canceled` sequence dropped by
/// `TERMINATE_ALL_NEXT_STEP`. The sender is then dropped, the receiver yields
/// `None`, and a streamer that maps `None` to "stream over" ends the response
/// cleanly: status 200, no error, and — because these paths fire before any
/// chunk — frequently no content either.
///
/// The non-streaming handlers already turn a closed channel into a 500. Only
/// the streaming paths were silent, and only because the status line is long
/// gone by then, so the body is the sole remaining channel.
pub const STREAM_TRUNCATED_MESSAGE: &str =
    "the engine closed this request's response channel without completing it: no \
     finish_reason was ever sent. The request did NOT finish normally — any content \
     received before this frame is partial. This commonly means the sequence was \
     dropped by the scheduler (KV cache exhaustion) or cancelled mid-flight.";

/// Build the stream-terminating frame for an abnormally closed channel.
///
/// Callers must reach this only from [`DoneState::Running`]: once a terminal
/// chunk has been seen the state machine has already moved to `SendingDone`,
/// so a closed channel there is the normal end of a healthy stream.
pub fn sse_stream_truncated_event() -> axum::response::sse::Event {
    sse_error_event(sse_error_kind::INTERNAL, STREAM_TRUNCATED_MESSAGE)
}

pub fn sse_error_event(kind: &str, message: impl Into<String>) -> axum::response::sse::Event {
    let message = message.into();
    let envelope = sse_error_payload(kind, message.clone());
    match axum::response::sse::Event::default().json_data(&envelope) {
        Ok(event) => event,
        Err(e) => axum::response::sse::Event::default().data(format!(
            "{{\"error\":{{\"message\":\"error serialization failed: {e}; original: {}\",\
             \"type\":\"{kind}\",\"param\":null,\"code\":null}}}}",
            message.replace('"', "'")
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The property that was missing: a streaming error frame must PARSE as
    /// JSON in the OpenAI error shape. It always carried the right words; it
    /// carried them in a shape no client decodes, which is the same as silence.
    #[test]
    fn a_stream_error_frame_is_parseable_openai_json() {
        let cause = "Received messages for a model which does not have a chat template.";
        let payload = sse_error_payload(sse_error_kind::INVALID_REQUEST, cause);
        let encoded = serde_json::to_string(&payload).expect("payload must serialize");

        // A client does exactly this to every `data:` frame.
        let parsed: serde_json::Value =
            serde_json::from_str(&encoded).expect("stream error frame must be valid JSON");

        assert_eq!(
            parsed["error"]["message"], cause,
            "the cause must survive into error.message: {encoded}"
        );
        assert_eq!(
            parsed["error"]["type"],
            sse_error_kind::INVALID_REQUEST,
            "error.type must name the OpenAI error class: {encoded}"
        );
        // Indexing a `Value` yields Null for a MISSING key too, so ask the
        // object directly -- otherwise this assertion cannot tell "present and
        // null" from "absent" and would silently pass on either.
        let error_obj = parsed["error"]
            .as_object()
            .expect("error must be a JSON object");
        for field in ["param", "code"] {
            assert!(
                error_obj.contains_key(field),
                "error.{field} must be PRESENT (and null), not omitted: {encoded}"
            );
            assert!(
                error_obj[field].is_null(),
                "error.{field} must be null here: {encoded}"
            );
        }
    }

    /// The truncated-stream frame must be decodable and must say the request
    /// did not finish. A client that cannot tell this from a normal completion
    /// is back to the original bug with extra steps.
    #[test]
    fn a_truncated_stream_frame_says_the_request_did_not_finish() {
        let payload = sse_error_payload(sse_error_kind::INTERNAL, STREAM_TRUNCATED_MESSAGE);
        let encoded = serde_json::to_string(&payload).expect("payload must serialize");
        let parsed: serde_json::Value =
            serde_json::from_str(&encoded).expect("truncation frame must be valid JSON");
        assert_eq!(parsed["error"]["type"], sse_error_kind::INTERNAL);

        let message = parsed["error"]["message"]
            .as_str()
            .expect("message must be a string");
        // The two facts a client needs: it did not finish, and what it has is partial.
        assert!(
            message.contains("did NOT finish"),
            "the frame must state the request did not finish: {message}"
        );
        assert!(
            message.contains("partial"),
            "the frame must warn that received content is partial: {message}"
        );
    }

    /// Non-vacuity control: an EMPTY message — what a client effectively got
    /// when the stream simply ended — carries neither fact, so the assertions
    /// above are load-bearing rather than trivially true.
    #[test]
    fn an_empty_message_would_fail_the_truncation_check() {
        let silent = "";
        assert!(
            !silent.contains("did NOT finish") && !silent.contains("partial"),
            "control broken: the silent frame must carry neither fact"
        );
    }

    /// Non-vacuity control: the OLD frame — a bare string in `data:` — fails
    /// the very check above. This is what produced an HTTP 200 with an
    /// immediate `[DONE]` and zero tokens.
    #[test]
    fn the_old_bare_string_frame_would_fail_that_check() {
        let old_frame = "Received messages for a model which does not have a chat template.";
        assert!(
            serde_json::from_str::<serde_json::Value>(old_frame).is_err(),
            "the legacy bare-string frame parses as JSON; this control is broken"
        );
    }

    /// Every error class the streaming paths emit must produce a decodable
    /// frame — not just the one that prompted the fix.
    #[test]
    fn every_stream_error_kind_produces_a_decodable_frame() {
        for kind in [
            sse_error_kind::INVALID_REQUEST,
            sse_error_kind::INTERNAL,
            sse_error_kind::MODEL,
        ] {
            let encoded = serde_json::to_string(&sse_error_payload(kind, "boom"))
                .expect("payload must serialize");
            let parsed: serde_json::Value = serde_json::from_str(&encoded)
                .unwrap_or_else(|e| panic!("{kind} frame is not valid JSON: {e}"));
            assert_eq!(parsed["error"]["type"], kind);
            assert_eq!(parsed["error"]["message"], "boom");
        }
    }
}
