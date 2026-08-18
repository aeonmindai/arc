//! Core functionality for handlers.

use anyhow::{Context, Result};
use axum::{extract::Json, http::StatusCode, response::IntoResponse};
use mistralrs_core::{Request, Response};
use serde::Serialize;
use tokio::sync::mpsc::{channel, Receiver, Sender};

use crate::types::SharedMistralRsState;

/// Default buffer size for the response channel used in streaming operations.
///
/// This constant defines the maximum number of response messages that can be buffered
/// in the channel before backpressure is applied. A larger buffer reduces the likelihood
/// of blocking but uses more memory.
pub const DEFAULT_CHANNEL_BUFFER_SIZE: usize = 10_000;

/// Trait for converting errors to HTTP responses with appropriate status codes.
pub(crate) trait ErrorToResponse: Serialize {
    /// Converts the error to an HTTP response with the specified status code.
    fn to_response(&self, code: StatusCode) -> axum::response::Response {
        let mut response = Json(self).into_response();
        *response.status_mut() = code;
        response
    }
}

/// Standard JSON error response structure.
#[derive(Serialize, Debug)]
pub(crate) struct JsonError {
    pub(crate) message: String,
}

impl JsonError {
    /// Creates a new JSON error with the specified message.
    pub(crate) fn new(message: String) -> Self {
        Self { message }
    }
}

impl std::fmt::Display for JsonError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for JsonError {}
impl ErrorToResponse for JsonError {}

/// Internal error type for model-related errors with a descriptive message.
///
/// This struct wraps error messages from the underlying model and implements
/// the standard error traits for proper error handling and display.
#[derive(Debug)]
pub(crate) struct ModelErrorMessage(pub(crate) String);

impl std::fmt::Display for ModelErrorMessage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for ModelErrorMessage {}

/// Generic JSON error response structure
#[derive(Serialize, Debug)]
pub(crate) struct BaseJsonModelError<T> {
    pub(crate) message: String,
    pub(crate) partial_response: T,
}

impl<T> BaseJsonModelError<T> {
    pub(crate) fn new(message: String, partial_response: T) -> Self {
        Self {
            message,
            partial_response,
        }
    }
}

/// Creates a channel for response communication.
pub fn create_response_channel(
    buffer_size: Option<usize>,
) -> (Sender<Response>, Receiver<Response>) {
    let channel_buffer_size = buffer_size.unwrap_or(DEFAULT_CHANNEL_BUFFER_SIZE);
    channel(channel_buffer_size)
}

/// Sends a request to the model processing pipeline.
pub async fn send_request(state: &SharedMistralRsState, request: Request) -> Result<()> {
    send_request_with_model(state, request, None).await
}

pub async fn send_request_with_model(
    state: &SharedMistralRsState,
    request: Request,
    model_id: Option<&str>,
) -> Result<()> {
    let sender = state
        .get_sender(model_id)
        .context("mistral.rs sender not available.")?;

    sender
        .send(request)
        .await
        .context("Failed to send request to model pipeline")
}

/// Generic function to process non-streaming responses.
pub(crate) async fn base_process_non_streaming_response<R, M, E>(
    rx: &mut Receiver<Response>,
    state: SharedMistralRsState,
    match_fn: M,
    error_handler: E,
) -> R
where
    M: FnOnce(SharedMistralRsState, Response) -> R,
    E: FnOnce(SharedMistralRsState, Box<dyn std::error::Error + Send + Sync + 'static>) -> R,
{
    match rx.recv().await {
        Some(response) => match_fn(state, response),
        None => {
            let error = anyhow::Error::msg("No response received from the model.");
            error_handler(state, error.into())
        }
    }
}

#[cfg(test)]
mod client_liveness_tests {
    use super::create_response_channel;

    /// The one link in the phantom-sequence fix that no scheduler unit test can
    /// reach: does the engine-side `Sender` actually observe a departed client?
    ///
    /// `Sequence::client_is_gone` is `responder.is_closed()`, and the whole
    /// reap in `DefaultScheduler::schedule` rests on that being `true` once the
    /// HTTP handler is gone. A handler's future is aborted on client disconnect,
    /// which drops every local it owns — including the `Receiver` this creates.
    /// This pins that channel's semantics rather than assuming them.
    ///
    /// Mutation check (run 2026-08-18): replace the `drop(rx)` with
    /// `std::mem::forget(rx)` and the second assertion fails — which is exactly
    /// the shape of the bug, a client that is gone but whose channel still looks
    /// open.
    #[test]
    fn dropping_the_handlers_receiver_closes_the_engines_sender() {
        let (tx, rx) = create_response_channel(None);
        assert!(
            !tx.is_closed(),
            "while the handler holds its Receiver the client is present"
        );

        // What axum does to a handler future when the client goes away.
        drop(rx);

        assert!(
            tx.is_closed(),
            "once the handler's Receiver is dropped the engine MUST be able to \
             see it — otherwise the scheduler's reap can never fire and \
             abandoned sequences run to max_tokens"
        );
    }

    /// A buffered-but-unread channel is NOT a departed client. The engine must
    /// not reap a slow reader — backpressure is not abandonment.
    #[tokio::test]
    async fn a_slow_client_that_is_not_reading_is_not_treated_as_gone() {
        let (tx, _rx) = create_response_channel(Some(1));
        assert!(!tx.is_closed());
        // Fill the single buffer slot; nobody is calling recv().
        tx.try_send(mistralrs_core::Response::InternalError(Box::new(
            std::io::Error::other("filler"),
        )))
        .expect("first send fits in the buffer");
        assert!(
            !tx.is_closed(),
            "a full buffer means a slow reader, not a departed one"
        );
    }
}
