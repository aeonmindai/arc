pub(crate) mod debug;
pub(crate) mod gguf_metadata;
pub(crate) mod memory_usage;
pub(crate) mod model_config;
pub(crate) mod normal;
pub(crate) mod progress;
pub(crate) mod tiktoken;
pub(crate) mod tokenizer;
pub(crate) mod tokens;
pub(crate) mod unvarbuilder;
pub(crate) mod varbuilder_utils;

/// Hand a response to its client, without suspending the caller when the
/// client's channel has room.
///
/// The engine dispatches responses one sequence at a time from inside the decode
/// loop, holding the pipeline mutex. A bare `Sender::send(..).await` makes every
/// one of those an await point on the engine task, so any single slow client sits
/// on the critical path of every other client's *compute*. `try_send` is a plain
/// push and covers the whole steady state (the server's per-request channel is
/// 10,000 deep); only a genuinely full channel falls back to the awaiting send,
/// which is exactly where backpressure belongs.
///
/// Ordering is unchanged. A sequence produces at most one response per engine
/// step and steps are sequential, so the per-client sequence of responses is the
/// same as before — this only changes whether the engine parks while pushing it.
pub async fn send_fast(
    responder: &tokio::sync::mpsc::Sender<crate::response::Response>,
    response: crate::response::Response,
) -> Result<(), tokio::sync::mpsc::error::SendError<crate::response::Response>> {
    use tokio::sync::mpsc::error::{SendError, TrySendError};
    match responder.try_send(response) {
        Ok(()) => Ok(()),
        Err(TrySendError::Full(response)) => responder.send(response).await,
        Err(TrySendError::Closed(response)) => Err(SendError(response)),
    }
}

#[doc(hidden)]
/// Deliver a terminal error response to one sequence's client, tolerating a
/// client that has already gone away.
///
/// 🔑 wave51-CB section 4.1 / wave56-CG. This was
/// `seq.responder().send(..).await.unwrap()` inline in
/// [`handle_pipeline_forward_error!`], and it is the
/// `SendError { .. }` panic reported at `engine/mod.rs:428:25` — the macro's
/// expansion site — that killed the engine 1 h 13 m into the 1,319-problem
/// GSM8K run and cost 16 unrelated in-flight requests.
///
/// **Cause or symptom: symptom, and independently a defect.** It can only fire
/// after some *other* error has already put the batch on the failure path, so
/// it is never the first thing to go wrong. But the receiver here is the HTTP
/// handler's channel, and any client that has already given up — a timeout, an
/// abort, a dropped connection — has closed it. Reporting a model error to a
/// departed client therefore escalated one failed request into a dead engine
/// and orphaned every other sequence in flight. Error *reporting* is the one
/// path that must not be able to fail loudly.
pub async fn send_response_or_log(
    responder: &tokio::sync::mpsc::Sender<crate::response::Response>,
    response: crate::response::Response,
    seq_id: usize,
) {
    if responder.send(response).await.is_err() {
        tracing::warn!(
            "Receiver for sequence {seq_id} disconnected before its error response could be \
             delivered; dropping the response. The engine is unaffected."
        );
    }
}

#[cfg(test)]
mod send_fast_tests {
    use super::send_fast;
    use crate::response::Response;
    use futures::FutureExt;

    /// The whole point of `send_fast`: on the steady-state path it must complete
    /// without ever yielding, so the engine loop never parks while dispatching a
    /// response to a client that has room.
    ///
    /// `now_or_never` polls exactly once. If the future is not ready after that
    /// single poll it returns `None` — which is precisely "this awaited".
    ///
    /// Mutation check: replace the `try_send` fast path with a plain
    /// `responder.send(response).await` and this assertion still passes (tokio's
    /// `send` is also ready when there is room), so the *discriminating* case is
    /// the full-channel test below, which is where the two shapes differ.
    #[test]
    fn a_client_with_room_is_served_without_suspending() {
        let (tx, mut rx) = tokio::sync::mpsc::channel::<Response>(4);

        let done = send_fast(&tx, Response::ValidationError("chunk".into())).now_or_never();

        assert!(
            done.is_some(),
            "send_fast suspended on a channel with room; the engine would park here"
        );
        assert!(done.unwrap().is_ok());
        assert!(
            rx.try_recv().is_ok(),
            "the response must actually have been delivered, not merely not-awaited"
        );
    }

    /// Backpressure is preserved: a genuinely full channel still blocks the
    /// sender, and the message survives the fallback rather than being dropped.
    #[tokio::test]
    async fn a_full_channel_still_applies_backpressure_and_keeps_the_message() {
        let (tx, mut rx) = tokio::sync::mpsc::channel::<Response>(1);
        tx.send(Response::ValidationError("first".into()))
            .await
            .unwrap();

        // Fixture discrimination (D12): the channel must really be full, or this
        // test would pass for a fast path that was never exercised.
        assert!(
            send_fast(&tx, Response::ValidationError("second".into()))
                .now_or_never()
                .is_none(),
            "fixture cannot discriminate: the channel must be full here"
        );

        let drainer = tokio::spawn(async move {
            let first = rx.recv().await;
            let second = rx.recv().await;
            (first, second)
        });
        send_fast(&tx, Response::ValidationError("second".into()))
            .await
            .expect("the fallback send must succeed once the receiver drains");
        drop(tx);

        let (first, second) = drainer.await.unwrap();
        assert!(
            matches!(first, Some(Response::ValidationError(ref s)) if s.to_string() == "first")
        );
        assert!(
            matches!(second, Some(Response::ValidationError(ref s)) if s.to_string() == "second")
        );
    }

    /// A departed client is an error, not a hang — and the undelivered response
    /// comes back with it, matching `Sender::send`'s contract.
    #[test]
    fn a_departed_client_returns_the_message() {
        let (tx, rx) = tokio::sync::mpsc::channel::<Response>(4);
        drop(rx);

        let err = send_fast(&tx, Response::ValidationError("orphan".into()))
            .now_or_never()
            .expect("a closed channel must not suspend")
            .expect_err("a closed channel must be an error");
        assert!(matches!(err.0, Response::ValidationError(ref s) if s.to_string() == "orphan"));
    }
}

#[cfg(test)]
mod send_response_or_log_tests {
    use super::send_response_or_log;
    use crate::response::Response;

    /// wave51-CB section 4.1 — `engine/mod.rs:428:25`, `SendError { .. }`.
    ///
    /// Reproduces the exact condition: a sequence whose client has gone away
    /// (receiver dropped) is handed a terminal error response. The old code
    /// `.unwrap()`ed this send, so it panicked on the engine task.
    ///
    /// Mutation check: change `send_response_or_log`'s body to
    /// `responder.send(response).await.unwrap();` and this test fails with
    /// `called \`Result::unwrap()\` on an \`Err\` value: SendError { .. }` —
    /// the message from the field.
    #[tokio::test]
    async fn a_departed_client_does_not_take_the_engine_down() {
        let (tx, rx) = tokio::sync::mpsc::channel::<Response>(1);

        // Fixture discrimination (D12): the channel must actually be closed,
        // or this test passes for a send that was never going to fail.
        drop(rx);
        assert!(
            tx.send(Response::ValidationError("probe".into()))
                .await
                .is_err(),
            "fixture cannot discriminate: the receiver must really be gone, \
             or there is no SendError to survive"
        );

        // The real call, on the real path's helper. Returning at all is the
        // assertion: a panic here is the engine death.
        send_response_or_log(&tx, Response::ValidationError("boom".into()), 42).await;
    }

    /// The happy path still delivers, so the fix did not turn error reporting
    /// into a silent drop.
    #[tokio::test]
    async fn a_live_client_still_receives_its_error() {
        let (tx, mut rx) = tokio::sync::mpsc::channel::<Response>(1);
        send_response_or_log(&tx, Response::ValidationError("boom".into()), 7).await;
        match rx.try_recv() {
            Ok(Response::ValidationError(e)) => assert_eq!(e.to_string(), "boom"),
            Ok(_) => panic!("expected a ValidationError to be delivered"),
            Err(e) => panic!("expected the error to be delivered, got {e:?}"),
        }
    }
}

#[doc(hidden)]
#[macro_export]
macro_rules! get_mut_arcmutex {
    ($thing:expr) => {
        loop {
            if let Ok(inner) = $thing.try_lock() {
                break inner;
            }
            // Yield to allow other threads to make progress and release the lock.
            // This prevents deadlock when a spawned async task busy-loops while
            // another task holds the lock across an await point.
            std::thread::yield_now();
        }
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! handle_seq_error {
    ($fallible:expr, $response:expr) => {
        match $fallible {
            Ok(v) => v,
            Err(e) => {
                use $crate::response::Response;
                if let Err(_) = $response.send(Response::InternalError(e.into())).await {
                    tracing::warn!("Receiver disconnected");
                }
                return;
            }
        }
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! handle_seq_error_ok {
    ($fallible:expr, $response:expr) => {
        match $fallible {
            Ok(v) => v,
            Err(e) => {
                use $crate::response::Response;
                if let Err(_) = $response.send(Response::InternalError(e.into())).await {
                    tracing::warn!("Receiver disconnected");
                }
                return Ok(());
            }
        }
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! handle_seq_error_stateaware_ok {
    ($fallible:expr, $seq:expr) => {
        match $fallible {
            Ok(v) => v,
            Err(e) => {
                use $crate::response::Response;
                use $crate::sequence::SequenceState;
                if let Err(_) = $seq
                    .responder()
                    .send(Response::InternalError(e.into()))
                    .await
                {
                    tracing::warn!("Receiver disconnected");
                }
                $seq.set_state(SequenceState::Error);
                return Ok(());
            }
        }
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! handle_pipeline_forward_error {
    ($stage: tt, $fallible:expr, $seq_slice:expr, $pipeline:expr, $label:tt, $prefix_cacher:expr) => {
        match $fallible {
            Ok(v) => v,
            Err(e) => {
                // Auto-retry on iOS Metal background GPU error: when the iOS app
                // goes to background, Metal rejects command buffers. We detect this,
                // reset cache, sleep, and let the engine loop retry. Sequences stay
                // in the scheduler (still in Running state) and are re-scheduled.
                #[cfg(feature = "metal")]
                {
                    let err_str = e.to_string();
                    if err_str.contains("Insufficient Permission")
                        || err_str.contains("BackgroundExecutionNotPermitted")
                    {
                        tracing::warn!(
                            "Metal GPU background error detected (iOS app likely in background). \
                             Pausing 1s before retry..."
                        );
                        {
                            let p = get_mut_arcmutex!($pipeline);
                            p.set_none_cache($seq_slice, true, true, false);
                        }
                        if let Err(e) = get_mut_arcmutex!($prefix_cacher).evict_all_caches() {
                            tracing::warn!("Failed to evict prefix caches: {e}");
                        }
                        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                        continue $label;
                    }
                }

                let (tokenizer, pipeline_name) = {
                    let pipeline = get_mut_arcmutex!($pipeline);
                    let pipeline_name = pipeline.name();
                    let tokenizer = pipeline.tokenizer();
                    (tokenizer, pipeline_name)
                };
                use $crate::response::Response;
                use $crate::sequence::SequenceState;
                use $crate::response::SYSTEM_FINGERPRINT;
                use tracing::error;
                error!("{} - Model failed with error: {:?}", $stage, &e);
                for seq in $seq_slice.iter_mut() {
                    // Step 1: Add all choices to groups
                    let start = seq.prompt_tokens().min(seq.get_toks().len());
                    let res = match &tokenizer {
                        Some(tok) => match tok.decode(&seq.get_toks()[start..], false) {
                            Ok(t) => t,
                            Err(_) => "".to_string(),
                        },
                        None => "".to_string(),
                    };

                    if seq.get_mut_group().is_chat {
                        let choice = Choice {
                            finish_reason: "error".to_string(),
                            index: seq.get_response_index(),
                            message: ResponseMessage {
                                content: Some(res),
                                role: "assistant".to_string(),
                                tool_calls: None,
                                reasoning_content: None,
                            },
                            logprobs: None,
                            confidence: None,
                            lowest_group_confidence: None,
                        };
                        seq.add_choice_to_group(choice);
                    } else {
                        let choice = CompletionChoice {
                            finish_reason: "error".to_string(),
                            index: seq.get_response_index(),
                            text: res,
                            logprobs: None,
                        };
                        seq.add_completion_choice_to_group(choice);
                    }
                }
                for seq in $seq_slice.iter_mut() {
                    // Step 2: Respond with all groups. The group lock is taken
                    // to BUILD the payload and dropped before the send await —
                    // the guard is `!Send`, so the old shape does not compile.
                    let response = {
                        let group = seq.get_mut_group();
                        if group.is_chat {
                            Response::ModelError(
                                e.to_string(),
                                ChatCompletionResponse {
                                    id: seq.id().to_string(),
                                    choices: group.get_choices().to_vec(),
                                    created: seq.creation_time(),
                                    model: pipeline_name.clone(),
                                    system_fingerprint: SYSTEM_FINGERPRINT.to_string(),
                                    object: "chat.completion".to_string(),
                                    usage: group.get_usage(),
                                    vote: None,
                                },
                            )
                        } else {
                            Response::CompletionModelError(
                                e.to_string(),
                                CompletionResponse {
                                    id: seq.id().to_string(),
                                    choices: group.get_completion_choices().to_vec(),
                                    created: seq.creation_time(),
                                    model: pipeline_name.clone(),
                                    system_fingerprint: SYSTEM_FINGERPRINT.to_string(),
                                    object: "text_completion".to_string(),
                                    usage: group.get_usage(),
                                },
                            )
                        }
                    };
                    $crate::utils::send_response_or_log(
                        &seq.responder(),
                        response,
                        *seq.id(),
                    )
                    .await;
                }
                for seq in $seq_slice.iter_mut() {
                    // Step 3: Set state - This cannot be done in Step 2 as `group` is locking the refcell
                    seq.set_state(SequenceState::Error);
                }

                let p = get_mut_arcmutex!($pipeline);
                // Also reset non granular state because:
                // - The sequence is gone
                // - We should reset the state then, including draft.
                p.set_none_cache($seq_slice, true, true, false);
                // The last step of the recovery path must not itself be able
                // to kill the engine: everything above has already been done,
                // and an eviction failure is at worst a stale prefix entry.
                if let Err(e) = get_mut_arcmutex!($prefix_cacher).evict_all_caches() {
                    tracing::warn!("Failed to evict prefix caches after a forward error: {e}");
                }

                continue $label;
            }
        }
    };
}

/// Acquire the sequence's [`crate::sequence::SequenceGroup`].
///
/// This used to be a `try_lock` in a bare `loop` with `std::thread::yield_now()`
/// — a spin. Two things were wrong with it. It burns a core whenever the lock is
/// actually contended, and `std::thread::yield_now()` yields the *OS thread*,
/// not the tokio task, so a spinner could not let a suspended lock holder on the
/// same worker run at all. The group is now a `std::sync::Mutex`, so this is a
/// real blocking acquire that parks instead of spinning.
///
/// The guard is `!Send`, which is deliberate: it makes "hold the group lock
/// across an `.await`" a compile error rather than a latent hazard.
///
/// Poisoning is recovered rather than propagated: a `SequenceGroup` is plain
/// accumulator data with no invariant a panic could break, and turning a panic
/// in one request into a permanent hang for the engine loop is strictly worse.
#[doc(hidden)]
#[macro_export]
macro_rules! get_mut_group {
    ($this:expr) => {
        match $this.group.lock() {
            Ok(inner) => inner,
            Err(poisoned) => poisoned.into_inner(),
        }
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! serde_default_fn {
    ($t:ty, $name:ident, $v:expr) => {
        fn $name() -> $t {
            $v
        }
    };
}

/// `true` if built with CUDA (requires Unix) /Metal
#[cfg(any(all(feature = "cuda", target_family = "unix"), feature = "metal"))]
pub const fn paged_attn_supported() -> bool {
    true
}

/// `true` if built with CUDA (requires Unix) /Metal
#[cfg(not(any(all(feature = "cuda", target_family = "unix"), feature = "metal")))]
pub const fn paged_attn_supported() -> bool {
    false
}

/// `true` if built with the `flash-attn` or `flash-attn-v3` features, false otherwise.
#[cfg(not(any(feature = "flash-attn", feature = "flash-attn-v3")))]
pub const fn using_flash_attn() -> bool {
    false
}

/// `true` if built with the `flash-attn` or `flash-attn-v3` features, false otherwise.
#[cfg(any(feature = "flash-attn", feature = "flash-attn-v3"))]
pub const fn using_flash_attn() -> bool {
    true
}
