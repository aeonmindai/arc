use std::sync::Arc;

use candle_core::{DType, Result, Tensor};
use rand_isaac::Isaac64Rng;

use crate::{
    prefix_cacher::PrefixCacheManagerV2,
    sampler::Logprobs,
    sequence::{Sequence, SequenceRecognizer, SequenceState, StopReason},
    tools::{parse_text_tools, ToolCallResponse, ToolCallType},
};
use mistralrs_mcp::CalledFunction;

use super::Pipeline;

macro_rules! fixup_sentencepiece {
    ($txt:expr) => {
        $txt.to_string().replace("▁", " ")
    };
    (Option $txt:expr) => {
        match &$txt {
            Some(txt) => Some(fixup_sentencepiece!(txt)),
            None => None,
        }
    };
}

pub(crate) async fn finish_or_add_toks_to_seq(
    this: &dyn Pipeline,
    prefix_cacher: &mut PrefixCacheManagerV2,
    seq: &mut Sequence,
    logprobs: Logprobs,
    eos_tok: Option<&[u32]>,
    use_prefix_cacher: bool,
) -> Result<()> {
    // Stop-token / max-length / stop-string scanning. The stop-string arm walks
    // the accumulated completion bytes, so its cost is not constant per token.
    let mut is_done = {
        let _s = arc_profiler::span("stop_check");
        seq.is_done(logprobs.token, eos_tok, this.get_metadata().max_seq_len)
    };
    let metadata = this.get_metadata();
    let tok_env = metadata.tok_env().ok_or(candle_core::Error::Msg(
        "`finish_or_add_toks_to_seq` requires the pipeline to have a token trie".to_string(),
    ))?;
    // Include special tokens when tool calling is active (so tool parsers can see
    // delimiters like <tool_call>, [TOOL_CALLS], <|python_tag|>) or when think tag
    // mode is enabled (so <think>/<\/think> delimiters are visible in the output).
    let include_special = seq.tools.is_some() || seq.is_think_tag_mode();
    let completion_bytes = {
        let _s = arc_profiler::span("detokenize");
        tok_env
            .tok_trie()
            .decode_ext(&[logprobs.token], include_special)
    };
    {
        let _s = arc_profiler::span("seq.add_token");
        seq.add_token(logprobs.clone(), completion_bytes, &is_done);
    }

    // If we can have a tool and we got a tool, stop the sequence early.
    // Doesn't conflict with the logic below because it does the same thing anyway.
    if let Some(ref t) = seq.tools {
        if let Ok(Some(ref d)) = seq.peek_delta() {
            let (_tool_use_still_possible, tool_use_is_done) =
                t.prefix_could_be_tool(this, d.as_str())?;

            if tool_use_is_done
                && matches!(
                    parse_text_tools(this, d, seq.tools.clone()),
                    Ok((None, _tools))
                )
            {
                seq.set_state(SequenceState::Done(StopReason::Eos));
                is_done = Some(StopReason::Eos);
            }
        }
    };

    // Arc Boost: DeepConf-low early termination of low-confidence vote chains.
    if is_done.is_none() {
        if let Some(reason) = seq.check_confidence_early_stop() {
            is_done = Some(reason);
        }
    }

    // Handle streaming requests.
    //
    // `get_mut_group()` is now a blocking `std::sync::Mutex` acquire, not the
    // `try_lock` + `std::thread::yield_now()` spin it used to be. Both flags are
    // read under one acquisition rather than two separate ones a few lines
    // apart; the group's mode cannot change mid-request, so this is also the
    // more honest read.
    let (group_is_streaming, group_is_chat) = {
        let _s = arc_profiler::span("group_lock.mode");
        let group = seq.get_mut_group();
        (group.is_streaming, group.is_chat)
    };
    if group_is_streaming {
        let mut tool_use_still_possible = false;
        let mut tool_use_is_done = false;
        if let Some(ref t) = seq.tools {
            if let Ok(Some(ref d)) = seq.peek_delta() {
                (tool_use_still_possible, tool_use_is_done) =
                    t.prefix_could_be_tool(this, d.as_str())?;
            }
        };

        // let send = seq.get_toks().len() % 2 == 0 || is_done.is_some();
        let send = true;
        // Send chunks when:
        // 1. Tool call is not possible (!tool_use_still_possible) - normal streaming
        // 2. Tool call is complete (tool_use_is_done) - send the tool call
        // 3. Sequence is done (is_done.is_some()) - send buffered output as text since it wasn't a valid tool call
        if !tool_use_still_possible || tool_use_is_done || is_done.is_some() {
            if send {
                let delta_result = seq.get_delta();
                if let Some(delta) = crate::handle_seq_error_ok!(delta_result, seq.responder()) {
                    if group_is_chat {
                        // Check if we're in Harmony mode or think tag mode and use parsed content
                        let (content_delta, reasoning_delta) = if seq.is_harmony_mode() {
                            // In Harmony mode, use the parsed final content and reasoning
                            let final_delta = seq.get_harmony_final_delta();
                            let reasoning = seq.get_harmony_reasoning_delta();
                            (final_delta, reasoning)
                        } else if seq.is_think_tag_mode() {
                            // In think tag mode, use the parsed content and reasoning
                            let content = seq.get_think_tag_content_delta();
                            let reasoning = seq.get_think_tag_reasoning_delta();
                            (content, reasoning)
                        } else {
                            // Not in Harmony or think tag mode, use raw delta
                            let (text_new, _) =
                                parse_text_tools(this, delta.as_str(), seq.tools.clone())
                                    .map_err(candle_core::Error::msg)?;
                            (text_new.map(ToString::to_string), None)
                        };

                        // Detect tool calls
                        let tool_calls = if seq.is_harmony_mode() {
                            // In Harmony mode, only finalize tool calls when the sequence is done
                            // (EOS token or stop string), not when we first detect a tool call.
                            // This ensures tool call arguments are fully generated.
                            if is_done.is_some() && seq.has_harmony_tool_calls() {
                                // Sequence is done and has tool calls - finalize and send them
                                is_done = Some(StopReason::ToolCalls);
                                let harmony_tool_calls = seq.get_harmony_tool_calls();
                                harmony_tool_calls
                                    .into_iter()
                                    .enumerate()
                                    .map(|(i, tc)| ToolCallResponse {
                                        index: i,
                                        id: tc.id,
                                        tp: ToolCallType::Function,
                                        function: CalledFunction {
                                            name: tc.name,
                                            arguments: tc.arguments,
                                        },
                                    })
                                    .collect()
                            } else {
                                vec![]
                            }
                        } else {
                            // Not in Harmony mode - parse text for tool calls
                            let (_, tool_calls) =
                                parse_text_tools(this, delta.as_str(), seq.tools.clone())
                                    .map_err(candle_core::Error::msg)?;
                            if !tool_calls.is_empty() {
                                is_done = Some(StopReason::ToolCalls);
                            }
                            tool_calls
                        };

                        seq.add_streaming_chunk_choice_to_group(crate::ChunkChoice {
                            delta: crate::Delta {
                                content: fixup_sentencepiece!(Option content_delta),
                                role: "assistant".to_string(),
                                tool_calls: Some(tool_calls).filter(|v| !v.is_empty()),
                                reasoning_content: reasoning_delta,
                            },
                            index: seq.get_response_index(),
                            finish_reason: is_done.map(|x| x.to_string()),
                            logprobs: if seq.return_logprobs() {
                                Some(crate::ResponseLogprob {
                                    token: delta,
                                    bytes: logprobs.bytes.clone().map(|b| b.into_bytes()),
                                    logprob: logprobs.logprob,
                                    top_logprobs: logprobs.top_logprobs.unwrap().clone(),
                                })
                            } else {
                                None
                            },
                        });
                    } else {
                        seq.add_streaming_completion_chunk_choice_to_group(
                            crate::CompletionChunkChoice {
                                text: fixup_sentencepiece!(delta),
                                index: seq.get_response_index(),
                                finish_reason: is_done.map(|x| x.to_string()),
                                logprobs: if seq.return_logprobs() {
                                    Some(crate::ResponseLogprob {
                                        token: delta,
                                        bytes: logprobs.bytes.clone().map(|b| b.into_bytes()),
                                        logprob: logprobs.logprob,
                                        top_logprobs: logprobs.top_logprobs.unwrap().clone(),
                                    })
                                } else {
                                    None
                                },
                            },
                        );
                    }
                }
            }

            // Build the chunk and release the group lock BEFORE dispatching it.
            // Usage on the final chunk is read under the same acquisition that
            // resets the counters and takes the buffered chunks, instead of the
            // four separate ones this used to be.
            let response = {
                let _s = arc_profiler::span("group_lock.take_response");
                let mut group = seq.get_mut_group();
                let usage_opt = if is_done.is_some() {
                    let usage = group.get_usage();
                    group.total_prompt_toks = 0;
                    group.total_toks = 0;
                    Some(usage)
                } else {
                    None
                };
                group.take_streaming_response(seq, this.name().clone(), usage_opt)
            };

            // Dispatch with the group lock dropped. `send_fast` pushes without
            // awaiting whenever the client's channel has room, so one slow
            // client no longer sits on the engine's critical path — and the
            // pipeline mutex is no longer held across a suspension point here.
            if let Some(response) = response {
                let _s = arc_profiler::span("response.send");
                if crate::utils::send_fast(&seq.responder(), response)
                    .await
                    .is_err()
                {
                    // If we can't send the response, cancel the sequence
                    seq.set_state(crate::sequence::SequenceState::Done(
                        crate::sequence::StopReason::Canceled,
                    ));
                    this.reset_non_granular_state();
                }
            }
        }

        // Handle Done state regardless of tool detection - must be outside the tool_use check
        // to ensure sequence completes even when tool detection thinks output might be a tool call
        if let Some(reason) = is_done {
            if use_prefix_cacher {
                let recurrent_snapshots = if this.cache().is_hybrid() {
                    seq.recurrent_state_idx()
                        .and_then(|idx| this.cache().hybrid().snapshot_recurrent_state(idx).ok())
                } else {
                    None
                };
                prefix_cacher.add_sequence(seq, recurrent_snapshots);
                prefix_cacher.evict_caches()?;
            }
            seq.set_state(crate::sequence::SequenceState::Done(reason));
            this.reset_non_granular_state();
        }
    } else if let Some(mut reason) = is_done {
        /*
        ***********************
        Finish the sequence now
        ***********************
        */
        {
            seq.set_state(crate::sequence::SequenceState::Done(reason));
            let (tokenizer, pipeline_name) = {
                let pipeline_name = this.name();
                let tokenizer = this.tokenizer();
                (tokenizer, pipeline_name)
            };

            let logprobs = if seq.return_logprobs() {
                let mut logprobs = Vec::new();
                for logprob in seq.logprobs() {
                    let resp_logprob = crate::ResponseLogprob {
                        token: crate::handle_seq_error_ok!(
                        tokenizer
                        .as_ref()
                        .ok_or(candle_core::Error::Msg(
                            "`finish_or_add_toks_to_seq` requires the pipeline to have a tokenizer"
                                .to_string(),
                        ))?.decode(&[logprob.token], false),
                        seq.responder()
                    ),
                        bytes: logprob.bytes.clone().map(|b| b.into_bytes()),
                        logprob: logprob.logprob,
                        top_logprobs: logprob.top_logprobs.clone().unwrap(),
                    };
                    logprobs.push(resp_logprob);
                }
                Some(logprobs)
            } else {
                None
            };

            // Signal EOS to Harmony parser if in Harmony mode
            seq.harmony_process_eos();

            // Finalize think tag parser if in think tag mode
            seq.think_tag_finalize();

            let text = match reason {
                crate::sequence::StopReason::Length(_)
                | crate::sequence::StopReason::ModelLength(_)
                | crate::sequence::StopReason::Eos
                | crate::sequence::StopReason::StopTok(_)
                | crate::sequence::StopReason::Canceled
                | crate::sequence::StopReason::LowConfidence
                | crate::sequence::StopReason::ToolCalls => {
                    String::from_utf8_lossy(seq.completion_bytes())
                        .trim_start()
                        .to_string()
                }
                crate::sequence::StopReason::StopString {
                    completion_bytes_pos,
                    ..
                } => {
                    let txt = String::from_utf8_lossy(seq.completion_bytes());
                    txt[..completion_bytes_pos].trim_start().to_string()
                }
                crate::sequence::StopReason::GeneratedImage
                | crate::sequence::StopReason::GeneratedSpeech => {
                    candle_core::bail!("Stop reason was `GeneratedImage`.")
                }
            };

            if group_is_chat {
                // In Harmony or think tag mode, use parsed content and tool calls
                let (text_new, tool_calls, reasoning_content) = if seq.is_harmony_mode() {
                    let final_content = seq.get_harmony_final_content();
                    let reasoning = seq.get_harmony_reasoning_content();

                    // Get Harmony tool calls
                    let harmony_tool_calls = seq.get_harmony_tool_calls();
                    let tool_calls: Vec<ToolCallResponse> = harmony_tool_calls
                        .into_iter()
                        .enumerate()
                        .map(|(i, tc)| ToolCallResponse {
                            index: i,
                            id: tc.id,
                            tp: ToolCallType::Function,
                            function: CalledFunction {
                                name: tc.name,
                                arguments: tc.arguments,
                            },
                        })
                        .collect();

                    (final_content, tool_calls, reasoning)
                } else if seq.is_think_tag_mode() {
                    // In think tag mode - finalize and get parsed content
                    seq.think_tag_finalize();
                    let final_content = seq.get_think_tag_content();
                    let reasoning = seq.get_think_tag_reasoning_content();

                    // Parse for tool calls in final content
                    let (text_new, tool_calls) = if let Some(ref content) = final_content {
                        parse_text_tools(this, content.as_str(), seq.tools.clone())
                            .map_err(candle_core::Error::msg)?
                    } else {
                        (None, vec![])
                    };
                    (
                        text_new.map(ToString::to_string).or(final_content),
                        tool_calls,
                        reasoning,
                    )
                } else {
                    // Not in Harmony or think tag mode - parse text for tool calls
                    let (text_new, tool_calls) =
                        parse_text_tools(this, text.as_str(), seq.tools.clone())
                            .map_err(candle_core::Error::msg)?;
                    (text_new.map(ToString::to_string), tool_calls, None)
                };

                if !tool_calls.is_empty() {
                    reason = StopReason::ToolCalls;
                }

                let choice = crate::Choice {
                    finish_reason: fixup_sentencepiece!(reason),
                    index: seq.get_response_index(),
                    message: crate::ResponseMessage {
                        content: text_new,
                        role: "assistant".to_string(),
                        tool_calls: Some(tool_calls).filter(|v| !v.is_empty()),
                        reasoning_content,
                    },
                    logprobs: logprobs.map(|l| crate::Logprobs { content: Some(l) }),
                    confidence: seq.confidence().mean(),
                    lowest_group_confidence: seq.confidence().lowest_group(),
                };
                seq.add_choice_to_group(choice);
            } else {
                let choice = crate::CompletionChoice {
                    finish_reason: fixup_sentencepiece!(reason),
                    index: seq.get_response_index(),
                    text,
                    logprobs: logprobs.map(|l| crate::Logprobs { content: Some(l) }),
                };
                seq.add_completion_choice_to_group(choice);
            }

            if use_prefix_cacher {
                let recurrent_snapshots = if this.cache().is_hybrid() {
                    seq.recurrent_state_idx()
                        .and_then(|idx| this.cache().hybrid().snapshot_recurrent_state(idx).ok())
                } else {
                    None
                };
                prefix_cacher.add_sequence(seq, recurrent_snapshots);
                prefix_cacher.evict_caches()?;
            }

            // Ensure timing info is synced to group before sending response
            seq.update_time_info();

            // Build under the lock, dispatch after dropping it.
            let response = {
                let group = seq.get_mut_group();
                if group.is_chat {
                    group.chat_done_response(crate::ChatCompletionResponse {
                        id: seq.id().to_string(),
                        choices: group.get_choices().to_vec(),
                        created: seq.creation_time(),
                        model: pipeline_name,
                        system_fingerprint: crate::SYSTEM_FINGERPRINT.to_string(),
                        object: "chat.completion".to_string(),
                        usage: group.get_usage(),
                        vote: None,
                    })
                } else {
                    group.completion_done_response(crate::CompletionResponse {
                        id: seq.id().to_string(),
                        choices: group.get_completion_choices().to_vec(),
                        created: seq.creation_time(),
                        model: pipeline_name,
                        system_fingerprint: crate::SYSTEM_FINGERPRINT.to_string(),
                        object: "text_completion".to_string(),
                        usage: group.get_usage(),
                    })
                }
            };
            if let Some(response) = response {
                crate::utils::send_fast(&seq.responder(), response)
                    .await
                    .map_err(candle_core::Error::msg)?;
            }
        }
        this.reset_non_granular_state();
    }

    Ok(())
}

pub async fn sample_and_add_toks(
    this: &dyn Pipeline,
    seqs: &mut [&mut Sequence],
    logits_seq: Vec<Tensor>,
    prefix_cacher: &mut PrefixCacheManagerV2,
    disable_eos_stop: bool,
    rng: Arc<std::sync::Mutex<Isaac64Rng>>,
) -> Result<()> {
    let seqs_len = seqs.len();
    debug_assert_eq!(logits_seq.len(), seqs_len);

    let use_async_pool = seqs_len > 1;

    let sampling_futures: Vec<_> = std::iter::zip(logits_seq, seqs.iter_mut())
        .map(|(logits_per_seq, seq)| {
            let return_logprobs = seq.return_logprobs();
            sample_sequence(
                logits_per_seq,
                seq,
                return_logprobs,
                rng.clone(),
                use_async_pool,
                false,
                use_async_pool,
            )
        })
        .collect();
    let sampled_vec = {
        // Wall time to sample the whole batch. When `use_async_pool` is true the
        // per-sequence `Sampler::sample` runs on the rayon pool, off this
        // thread, so it is deliberately NOT broken out below — see the note.
        let _s = arc_profiler::span("sample.join_all");
        arc_profiler::note(
            "sample.join_all is not decomposed per sequence: for B>1 the sampler runs on the \
             rayon pool via tokio_rayon::spawn, and opening a span inside a concurrently-polled \
             future would interleave with its siblings and corrupt the tree shape. Its wall time \
             is the batch's sampling cost; the per-sequence host prologue IS broken out as \
             sample.logits_cast and sample.ctx_clone.",
        );
        futures::future::join_all(sampling_futures).await
    };

    let _s = arc_profiler::span("finish_or_add_toks");
    for (sampled, seq) in std::iter::zip(sampled_vec, seqs.iter_mut()) {
        let next_token = crate::handle_seq_error_stateaware_ok!(sampled, seq);
        // Arc Boost budget policy: graceful end-think injection on budget hit.
        // Applies to the normal decode path only (not speculative decoding).
        let next_token = seq.apply_reasoning_budget(next_token);

        let metadata = this.get_metadata();
        let eos_tok = if disable_eos_stop {
            None
        } else {
            Some(&metadata.eos_tok[..])
        };

        finish_or_add_toks_to_seq(this, prefix_cacher, seq, next_token, eos_tok, true).await?;
    }

    Ok(())
}

/// Is this sequence one the ArcGraph device decode loop may drive, and does it
/// already have a token waiting?
///
/// Two jobs, deliberately in one place. It **publishes** eligibility, because
/// `Pipeline::forward_inputs` decides whether to run a burst but is handed only
/// `inputs` and `return_raw_logits` — it never sees a `Sequence`, so it cannot
/// ask whether sampling is greedy. And it **takes** the token a burst already
/// produced.
///
/// Greedy only. The device sampler runs Splitmix64 per row against the host's
/// Isaac64, so under any stochastic configuration the two draw different —
/// individually valid — tokens, which would silently break seeded
/// reproducibility. Under greedy they agree exactly, which is also what makes
/// the path verifiable against the eager one.
///
/// The returned `Logprobs` is not a lossy stand-in: `logprob: 0.0` is exactly
/// what the greedy host path returns (`sampler.rs:1490`), and `bytes` /
/// `top_logprobs` are read only when `return_logprobs` is set, which this
/// refuses.
// `&mut` only because `Sequence::sampler()` takes `&mut self` (`sequence.rs:885`)
// to hand back a clone of an `Arc`. Nothing here mutates the sequence.
//
// Deliberately NOT `cfg(feature = "cuda")`: only the CUDA plumbing can *park*
// tokens, but the take/stand-down half is plain host code, and gating it made
// the cross-request leak untestable on every machine that reviewed it. On a
// non-CUDA build nothing ever parks, `take_pending_token` is always `None`,
// and this is one thread-local read per sample.
fn device_loop_pre_sampled_token(
    seq: &mut Sequence,
    return_logprobs: bool,
    sample_speculative: bool,
) -> Result<Option<Logprobs>> {
    let seq_id = *seq.id();
    let eligible = !return_logprobs
        && !sample_speculative
        && seq.sampler().is_greedy_trivial()
        && matches!(seq.recognizer, SequenceRecognizer::None);
    arc_cuda_graph::set_device_loop_eligible(seq_id, eligible);
    // Did the forward serve this step from the pending queue WITHOUT
    // launching? Then the logits this sample holds alias graph-owned storage
    // from an earlier step — sound only if a parked token is taken below.
    // Consumed here, once, whatever the outcome.
    let aliased_logits = arc_cuda_graph::take_aliased_logits_marker();
    if !eligible {
        // Stand down rather than merely decline: a token parked by an earlier,
        // eligible sequence must never be handed to this one.
        arc_cuda_graph::stand_down();
        if aliased_logits {
            candle_core::bail!(
                "ArcGraph: this step's logits were served from the device loop's pending queue \
                 for another sequence and alias graph-owned storage; this sequence (id {seq_id}) \
                 is not device-loop eligible and cannot consume them. Failing the step rather \
                 than sampling stale logits."
            );
        }
        return Ok(None);
    }
    match arc_cuda_graph::take_pending_token(seq_id) {
        arc_cuda_graph::PendingTake::Taken(token) => Ok(Some(Logprobs {
            token,
            logprob: 0.0,
            bytes: None,
            top_logprobs: None,
        })),
        // Nothing parked: the forward really ran for this step (the pending
        // short-circuit only fires with a non-empty queue), so the logits are
        // real and the host samples them. `aliased_logits` cannot be set here
        // in a consistent engine: it implies the queue was non-empty when the
        // forward ran and empty now, on the same thread with no take between.
        arc_cuda_graph::PendingTake::Empty => {
            if aliased_logits {
                candle_core::bail!(
                    "ArcGraph: the forward served this step from the pending queue but the queue \
                     is empty at sampling time — the aliased logits cannot be sampled. Failing \
                     the step."
                );
            }
            Ok(None)
        }
        // The queue belonged to another sequence and was dropped whole (and
        // logged: "ArcGraph: dropped N foreign parked tokens"). If the
        // forward ALSO short-circuited on that foreign queue, these logits
        // are a stale alias and sampling them would hand this sequence a
        // plausible token from the other sequence's distribution — fail the
        // step loudly instead. Without the marker the forward genuinely ran
        // (e.g. a batched eager step), the logits are this sequence's own,
        // and sampling proceeds normally.
        arc_cuda_graph::PendingTake::Foreign { dropped } => {
            if aliased_logits {
                candle_core::bail!(
                    "ArcGraph: this step's logits alias graph storage served against {dropped} \
                     parked token(s) belonging to another sequence; sampling them would leak \
                     another user's distribution into sequence {seq_id}. Failing the step."
                );
            }
            Ok(None)
        }
    }
}

/// Async sample optionally adding to trie.
#[allow(clippy::too_many_arguments)]
pub async fn sample_sequence(
    logits: Tensor,
    seq: &mut Sequence,
    return_logprobs: bool,
    rng: Arc<std::sync::Mutex<Isaac64Rng>>,
    use_async_pool: bool,
    sample_speculative: bool,
    multiple_sequences: bool,
) -> Result<Logprobs> {
    // ── ArcInfer/ArcGraph device decode loop ────────────────────────────────
    // If a burst already sampled this token ON DEVICE, take it and skip the
    // host argmax entirely. That D2H (`sampler.rs:1479`) is the last
    // synchronization in the decode step, so this is the half of the change
    // that makes removing `replay()`'s `cudaStreamSynchronize` worth anything.
    if let Some(pre_sampled) =
        device_loop_pre_sampled_token(seq, return_logprobs, sample_speculative)?
    {
        return Ok(pre_sampled);
    }

    // Both of these open and close inside a single poll, before the first
    // `.await`, so they nest correctly under `sample.join_all` even when B
    // futures are being polled in turn on this thread.
    let logits = {
        let _s = arc_profiler::span("sample.logits_cast");
        logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?
    };

    let sampler = seq.sampler();
    // Named suspect: a full copy of the sequence's token history, per sequence,
    // per step — O(context) host work that grows as generation proceeds.
    let ctx_clone = {
        let _s = arc_profiler::span("sample.ctx_clone");
        seq.get_toks().to_vec()
    };
    let rng_clone = rng.clone();
    let logits_clone = logits.clone();
    let first_lobprobs_response = if use_async_pool {
        tokio_rayon::spawn(move || {
            sampler.sample(
                logits_clone,
                &ctx_clone,
                return_logprobs,
                rng_clone,
                sample_speculative,
                multiple_sequences,
            )
        })
        .await?
    } else {
        sampler.sample(
            logits_clone,
            &ctx_clone,
            return_logprobs,
            rng_clone,
            sample_speculative,
            multiple_sequences,
        )?
    };

    let bias_if_not_allowed = match &mut seq.recognizer {
        SequenceRecognizer::Llguidance(ref mut llg) => {
            if !llg.is_stopped()
                && llg
                    .validate_tokens(&[first_lobprobs_response.token])
                    .unwrap_or(0)
                    == 1
            {
                None
            } else {
                let mask = llg.compute_mask_or_eos().map_err(candle_core::Error::msg)?;
                if mask.is_allowed(first_lobprobs_response.token) {
                    // shouldn't really happen, except for EOS
                    None
                } else {
                    let mut acc = vec![-f32::INFINITY; logits.shape().dims1().unwrap()];
                    mask.iter_set_entries(|idx| {
                        if idx < acc.len() {
                            acc[idx] = 0.0;
                        }
                    });

                    Some(acc)
                }
            }
        }
        SequenceRecognizer::None => None,
    };
    let second_logprobs_response = match bias_if_not_allowed {
        Some(acc) => {
            let new_logits = (&logits + Tensor::from_slice(&acc, acc.len(), logits.device())?)?;

            let ctx_clone = seq.get_toks().to_vec();
            let rng_clone = rng.clone();
            let sampler = seq.sampler();
            if use_async_pool {
                tokio_rayon::spawn(move || {
                    sampler.sample(
                        new_logits,
                        &ctx_clone,
                        return_logprobs,
                        rng_clone,
                        sample_speculative,
                        multiple_sequences,
                    )
                })
                .await?
            } else {
                sampler.sample(
                    new_logits,
                    &ctx_clone,
                    return_logprobs,
                    rng_clone,
                    sample_speculative,
                    multiple_sequences,
                )?
            }
        }
        None => first_lobprobs_response,
    };

    match seq.recognizer {
        SequenceRecognizer::Llguidance(ref mut llg) => {
            if !llg.is_stopped() {
                llg.consume_token(second_logprobs_response.token)
                    .map_err(candle_core::Error::msg)?;
            }
        }
        SequenceRecognizer::None => {}
    }

    Ok(second_logprobs_response)
}

#[derive(Clone)]
pub struct SpeculativeSample {
    pub sample: Logprobs,
}

/// Async sample without modifying sequence (except for the constraint).
///
/// # Verification is greedy-only
///
/// A draft token is accepted iff it is *equal* to the token the target model
/// produces at that slot. That test is lossless exactly when both models are
/// decoding greedily: argmax of the target alone equals argmax via
/// draft-then-verify.
///
/// It is **not** correct for `temperature > 0`. The distribution-preserving
/// test is rejection sampling — accept the draft with probability
/// `min(1, p(x)/q(x))` and, on rejection, redraw from the normalised residual
/// `max(0, p(x) - q(x))` — and neither the accept test nor the residual draw
/// exists here. Accepting on token equality instead would emit whatever
/// `Sampler::sample(.., sample_speculative = true, ..)` returns, which is an
/// argmax at any temperature (see `Sampler::sample_speculative_top_kp_min_p`),
/// i.e. greedy output for a request that asked for sampling.
///
/// So when the sequence is not greedy we refuse to speculate: draw one token
/// from the target the ordinary (stochastic) way and reject every draft. The
/// caller already handles a short accept list — it narrows the caches by
/// `gamma - accepted.len()` — so this degrades to plain non-speculative
/// decode, with correct output and no speedup.
pub async fn sample_target_sequence_speculative(
    logits: Tensor,
    seq: &mut Sequence,
    return_logprobs: bool,
    rng: Arc<std::sync::Mutex<Isaac64Rng>>,
    draft_samples: &[SpeculativeSample],
) -> Result<Vec<SpeculativeSample>> {
    let n_toks = draft_samples.len();

    // first, rollback the llg
    match &mut seq.recognizer {
        SequenceRecognizer::Llguidance(ref mut llg) => {
            llg.rollback(n_toks).map_err(candle_core::Error::msg)?;
        }
        SequenceRecognizer::None => {}
    }

    if !seq.sampler().is_greedy() {
        let Some(first) = logits.chunk(n_toks, 1)?.into_iter().next() else {
            return Ok(Vec::new());
        };
        let sample = sample_sequence(
            first,
            seq,
            return_logprobs,
            rng,
            true,
            false, // NOT speculative: a real draw from the target distribution
            false,
        )
        .await?;
        return Ok(vec![SpeculativeSample { sample }]);
    }

    let mut sampled = Vec::new();
    for (chunk, draft) in logits
        .chunk(n_toks, 1)?
        .into_iter()
        .zip(draft_samples.iter())
    {
        let sample = sample_sequence(
            chunk,
            seq,
            return_logprobs,
            rng.clone(),
            true, // TODO(EricLBuehler): does this hurt perf?
            true,
            false,
        )
        .await?;
        let sampled_token = sample.token;
        sampled.push(SpeculativeSample { sample });
        if sampled_token != draft.sample.token {
            break;
        }
    }
    Ok(sampled)
}

#[cfg(test)]
mod speculative_verification_tests {
    use super::*;
    use crate::sampler::Sampler;
    use crate::sequence::{SeqStepType, SequenceGroup};
    use candle_core::Device;
    use rand::SeedableRng;
    use std::cell::RefCell;
    use std::collections::HashSet;

    const VOCAB: usize = 32;
    const GAMMA: usize = 4;
    /// The token every slot's logits peak at, so greedy verification accepts
    /// the whole draft.
    const PEAK: u32 = 7;

    thread_local! {
        /// Keeps each fixture sequence's `Receiver` alive; a dropped receiver
        /// models a disconnected client and would silently change behaviour.
        static LIVE_CLIENTS: RefCell<Vec<tokio::sync::mpsc::Receiver<crate::response::Response>>> =
            const { RefCell::new(Vec::new()) };
    }

    /// Minimal sequence carrying `sampler`. `sample_target_sequence_speculative`
    /// only reads the sampler, the token history and the (absent) recognizer.
    fn seq_with(sampler: Sampler) -> Sequence {
        let (dummy_sender, rx) = tokio::sync::mpsc::channel(1);
        LIVE_CLIENTS.with(|k| k.borrow_mut().push(rx));
        let group = Arc::new(std::sync::Mutex::new(SequenceGroup::new(
            1, false, false, None,
        )));
        Sequence::new_waiting(
            vec![1u32; 4],
            String::new(),
            0,
            0,
            1,
            dummy_sender,
            sampler,
            vec![],
            vec![],
            None,
            false,
            false,
            group,
            0,
            0,
            SequenceRecognizer::None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            SeqStepType::PromptAndDecode,
            None,
            None,
            None,
            false,
            vec![],
        )
    }

    fn sampler_with(temperature: Option<f64>) -> Sampler {
        Sampler::new(
            temperature,
            0,
            None,
            None,
            None,
            None,
            None,
            -1,  // top_k disabled
            1.0, // top_p disabled
            0.0, // min_p disabled
            None,
            vec![],
            None,
        )
        .unwrap()
    }

    /// `(1, GAMMA, VOCAB)` logits. `peak = true` puts a dominant spike on
    /// `PEAK` in every slot; `peak = false` is flat (uniform).
    fn logits(peak: bool) -> Tensor {
        let mut raw = vec![0f32; GAMMA * VOCAB];
        if peak {
            for slot in 0..GAMMA {
                raw[slot * VOCAB + PEAK as usize] = 20.0;
            }
        }
        Tensor::from_vec(raw, (1, GAMMA, VOCAB), &Device::Cpu).unwrap()
    }

    fn drafts(token: u32) -> Vec<SpeculativeSample> {
        (0..GAMMA)
            .map(|_| SpeculativeSample {
                sample: Logprobs {
                    token,
                    logprob: 0.0,
                    top_logprobs: None,
                    bytes: None,
                },
            })
            .collect()
    }

    /// Greedy speculation is lossless, so a draft that matches the target's
    /// argmax in every slot must still be accepted in full. This is the
    /// control: it pins that the temperature refusal below does not disable
    /// speculative decoding outright.
    #[tokio::test]
    async fn greedy_speculation_accepts_a_matching_draft() {
        let mut seq = seq_with(sampler_with(None));
        let rng = Arc::new(std::sync::Mutex::new(Isaac64Rng::seed_from_u64(0)));
        let accepted =
            sample_target_sequence_speculative(logits(true), &mut seq, false, rng, &drafts(PEAK))
                .await
                .unwrap();

        assert_eq!(
            accepted.len(),
            GAMMA,
            "greedy verification must accept a draft that matches the target argmax"
        );
        assert!(accepted.iter().all(|s| s.sample.token == PEAK));
    }

    /// With `temperature > 0` the accept-on-equality test is not rejection
    /// sampling, so speculation must be refused: exactly one token, drawn from
    /// the target the ordinary way, and every draft rejected.
    ///
    /// Discriminator: the logits peak hard on `PEAK`, and the speculative
    /// sampling branch is an argmax at any temperature, so without the refusal
    /// all GAMMA drafts are accepted deterministically.
    #[tokio::test]
    async fn temperature_speculation_is_refused() {
        let mut seq = seq_with(sampler_with(Some(1.0)));
        let rng = Arc::new(std::sync::Mutex::new(Isaac64Rng::seed_from_u64(0)));
        let accepted =
            sample_target_sequence_speculative(logits(true), &mut seq, false, rng, &drafts(PEAK))
                .await
                .unwrap();

        assert_eq!(
            accepted.len(),
            1,
            "temperature > 0 must fall back to non-speculative decode (1 token, all drafts \
             rejected); accepting {} means tokens were verified by equality rather than by \
             rejection sampling",
            accepted.len()
        );
    }

    /// The one token the refusal path returns must be a real draw from the
    /// target distribution, not the argmax the speculative branch produces.
    ///
    /// Discriminator: a flat distribution over 32 tokens. A genuine draw
    /// spreads over the support; the speculative branch's `argmax_f32` returns
    /// token 0 every single time.
    #[tokio::test]
    async fn temperature_fallback_token_is_sampled_not_argmaxed() {
        let mut seen = HashSet::new();
        let rng = Arc::new(std::sync::Mutex::new(Isaac64Rng::seed_from_u64(0xC0FFEE)));
        for _ in 0..200 {
            let mut seq = seq_with(sampler_with(Some(1.0)));
            let accepted = sample_target_sequence_speculative(
                logits(false),
                &mut seq,
                false,
                rng.clone(),
                &drafts(0),
            )
            .await
            .unwrap();
            assert_eq!(accepted.len(), 1);
            seen.insert(accepted[0].sample.token);
        }
        assert!(
            seen.len() > VOCAB / 2,
            "the fallback covered only {} of {VOCAB} tokens over 200 draws from a flat \
             distribution — it is returning an argmax, not a sample",
            seen.len()
        );
    }
}

/// ArcInfer/ArcGraph: a device-loop burst that outlives its sequence must
/// never reach the next sequence.
///
/// The burst runs with `eos_token_id = -1` (`device_decode_burst` passes -1,
/// so no on-device EOS truncation ever fires) and parks its full length; the
/// engine drains one token per step. A sequence that stops mid-burst — stop
/// string, `max_tokens`, cancel, error — leaves the tail parked, and
/// `device_loop_pre_sampled_token` hands parked tokens to whichever eligible
/// sequence samples next, **including the next request's first sample after
/// its prefill**. That is one user's tokens inside another user's response.
///
/// The fix under test is the funnel: `Sequence::set_state` stands the device
/// loop down on every transition out of the running set. These tests drive the
/// REAL `sample_sequence` path, so they fail loudly if that call is removed —
/// unlike the arc-cuda-graph unit tests, which only ever exercised
/// `clear_pending_tokens` itself, not that anything calls it.
#[cfg(test)]
mod device_loop_cross_sequence_leak_tests {
    use super::*;
    use crate::sampler::Sampler;
    use crate::sequence::{SeqStepType, SequenceGroup, SequenceState, StopReason};
    use candle_core::Device;
    use rand::SeedableRng;
    use std::cell::RefCell;

    const VOCAB: usize = 32;
    /// Argmax of every logits tensor handed to sequence A.
    const PEAK_A: u32 = 7;
    /// Argmax of every logits tensor handed to sequence B. Distinct from
    /// `PEAK_A` and from the burst, so the origin of B's token is unambiguous.
    const PEAK_B: u32 = 9;
    /// The burst sequence A's device loop parked. A drains the first token,
    /// then stops mid-burst, leaving `[22, 23, 24]`.
    const BURST: [i32; 4] = [21, 22, 23, 24];

    thread_local! {
        /// Keeps each fixture sequence's `Receiver` alive; a dropped receiver
        /// models a disconnected client and would silently change behaviour.
        static LIVE_CLIENTS: RefCell<Vec<tokio::sync::mpsc::Receiver<crate::response::Response>>> =
            const { RefCell::new(Vec::new()) };
    }

    /// A greedy-trivial sequence: no temperature, penalties, bias, processors
    /// or recognizer — exactly the shape `device_loop_pre_sampled_token`
    /// accepts, so the pending queue is live for it. `id` matters: the
    /// pending queue is owner-tagged by sequence id, so every test sequence
    /// must carry a distinct one, as real requests do.
    fn greedy_seq(id: usize) -> Sequence {
        let (dummy_sender, rx) = tokio::sync::mpsc::channel(1);
        LIVE_CLIENTS.with(|k| k.borrow_mut().push(rx));
        let group = Arc::new(std::sync::Mutex::new(SequenceGroup::new(
            1, false, false, None,
        )));
        let sampler = Sampler::new(
            None,
            0,
            None,
            None,
            None,
            None,
            None,
            -1,
            1.0,
            0.0,
            None,
            vec![],
            None,
        )
        .unwrap();
        Sequence::new_waiting(
            vec![1u32; 4],
            String::new(),
            id,
            0,
            1,
            dummy_sender,
            sampler,
            vec![],
            vec![],
            None,
            false,
            false,
            group,
            0,
            0,
            SequenceRecognizer::None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            SeqStepType::PromptAndDecode,
            None,
            None,
            None,
            false,
            vec![],
        )
    }

    /// `(1, 1, VOCAB)` logits peaking hard at `peak` — the shape
    /// `sample_sequence` receives for one decode step of one sequence.
    fn peaked_logits(peak: u32) -> Tensor {
        let mut raw = vec![0f32; VOCAB];
        raw[peak as usize] = 20.0;
        Tensor::from_vec(raw, (1, 1, VOCAB), &Device::Cpu).unwrap()
    }

    async fn sample(seq: &mut Sequence, peak: u32, rng: &Arc<std::sync::Mutex<Isaac64Rng>>) -> u32 {
        sample_sequence(
            peaked_logits(peak),
            seq,
            false,
            rng.clone(),
            false,
            false,
            false,
        )
        .await
        .unwrap()
        .token
    }

    /// THE leak test: A parks a burst, stops mid-burst on a non-fallback path
    /// (`max_tokens` → `StopReason::Length`, the transition
    /// `finish_or_add_toks_to_seq` performs at `sampling.rs:267`/`:277`), and
    /// B's first sample must consume nothing of A's.
    ///
    /// Mutation check (run both ways before shipping): with the
    /// `stand_down` arm removed from `Sequence::set_state`, this fails at the
    /// queue assertion with 3 parked tokens, and B's first token comes back
    /// `22` — A's first undrained parked id — instead of `PEAK_B`.
    #[tokio::test]
    async fn a_sequence_stopping_mid_burst_leaks_nothing_into_the_next_sequence() {
        // This thread's queue, not shared with other tests (thread-local).
        arc_cuda_graph::stand_down();
        let rng = Arc::new(std::sync::Mutex::new(Isaac64Rng::seed_from_u64(0)));

        // ── Sequence A, decode step N: queue empty, host argmax, and the
        // pre-hook publishes A's eligibility — the same order the engine
        // runs (sample N publishes, forward N+1 bursts).
        let mut seq_a = greedy_seq(1);
        assert_eq!(sample(&mut seq_a, PEAK_A, &rng).await, PEAK_A);

        // ── Forward N+1: the burst parks its FULL length (eos -1).
        arc_cuda_graph::push_pending_tokens(&BURST);

        // ── Step N+1's sample drains one parked token. This is the control
        // that proves the pending-token path is LIVE in this build: if the
        // pre-hook were compiled out or inert, this would return `PEAK_A`
        // and the leak below could never manifest — a test that cannot fail.
        assert_eq!(
            sample(&mut seq_a, PEAK_A, &rng).await,
            BURST[0] as u32,
            "the device-loop take path is not live in this build; the leak assertions below \
             would be vacuous"
        );

        // ── A stops MID-BURST via a non-fallback path: max_tokens.
        seq_a.set_state(SequenceState::Done(StopReason::Length(2)));

        // ── The claim, part 1: nothing of A's survives its completion.
        assert_eq!(
            arc_cuda_graph::pending_token_count(),
            0,
            "sequence A finished mid-burst but its parked tokens survived; the next \
             sequence's first sample will consume them"
        );

        // ── Sequence B's FIRST sample (the one right after its prefill).
        let mut seq_b = greedy_seq(2);
        let b_first = sample(&mut seq_b, PEAK_B, &rng).await;

        // ── The claim, part 2: B's token comes from B's own logits and is
        // none of A's parked ids.
        assert!(
            !BURST.contains(&(b_first as i32)),
            "cross-request token leak: sequence B's first sampled token is {b_first}, one of \
             sequence A's parked burst tokens"
        );
        assert_eq!(
            b_first, PEAK_B,
            "sequence B's first token must be the argmax of B's own logits"
        );
    }

    /// Every transition out of the running set clears the queue — not just
    /// `Done(Length)`. This pins the whole funnel map: client cancel,
    /// forward-error/panic recovery (`Error`), PagedAttention eviction
    /// (`FinishedAborted`/`FinishedIgnored`), and preemption
    /// (`Waiting`/`Swapped`), where consuming the stale burst after
    /// re-prefill would duplicate output.
    #[tokio::test]
    async fn every_transition_out_of_the_running_set_clears_parked_tokens() {
        for state in [
            SequenceState::Done(StopReason::Canceled),
            SequenceState::Done(StopReason::StopString {
                stop_string_idx: 0,
                completion_bytes_pos: 0,
            }),
            SequenceState::Error,
            SequenceState::FinishedAborted,
            SequenceState::FinishedIgnored,
            SequenceState::Waiting,
            SequenceState::Swapped,
        ] {
            arc_cuda_graph::stand_down();
            arc_cuda_graph::push_pending_tokens(&BURST);
            assert_eq!(arc_cuda_graph::pending_token_count(), BURST.len());
            let seq = greedy_seq(3);
            seq.set_state(state);
            assert_eq!(
                arc_cuda_graph::pending_token_count(),
                0,
                "transition to {state:?} left parked tokens for the next sequence"
            );
        }
    }

    /// The interleave the completion funnel CANNOT see: the DefaultScheduler
    /// bucketing waitlist moves a running sequence aside **without a state
    /// modification** (`default_scheduler.rs`, `bucket_and_waitlist_seqs_waiting`
    /// says so in its doc), so no `set_state` ever fires for A. The owner tag
    /// is the structural defence: B's first sample must consume nothing of
    /// A's — the foreign queue is dropped whole (logged as
    /// "ArcGraph: dropped N foreign parked tokens") rather than served, to A
    /// included, whose device-side burst state is stale by then.
    ///
    /// Mutation check (run both ways before shipping): with the owner
    /// comparison in `take_pending_token` forced to `true`, B's first token
    /// comes back `22` — A's parked id — instead of `PEAK_B`.
    #[tokio::test]
    async fn a_waitlisted_sequence_with_no_state_change_cannot_leak_into_the_next() {
        arc_cuda_graph::stand_down();
        let rng = Arc::new(std::sync::Mutex::new(Isaac64Rng::seed_from_u64(0)));

        // ── A (id 31) decodes at batch 1; its burst parks; it drains one
        // (the control proving the take path is live in this build).
        let mut seq_a = greedy_seq(31);
        assert_eq!(sample(&mut seq_a, PEAK_A, &rng).await, PEAK_A);
        arc_cuda_graph::push_pending_tokens(&BURST);
        assert_eq!(sample(&mut seq_a, PEAK_A, &rng).await, BURST[0] as u32);

        // ── A is WAITLISTED: no set_state, no funnel, nothing runs. Its
        // undrained tail [22, 23, 24] is still parked when B is scheduled.

        // ── B (id 32) runs; its forward really ran, so its logits are real.
        let mut seq_b = greedy_seq(32);
        let b_first = sample(&mut seq_b, PEAK_B, &rng).await;
        assert!(
            !BURST.contains(&(b_first as i32)),
            "cross-request token leak through the waitlist interleave: B's first sampled token \
             is {b_first}, one of A's parked burst tokens"
        );
        assert_eq!(b_first, PEAK_B, "B must sample its own logits");

        // ── A's foreign-tagged tail was dropped whole, not left to resurface.
        assert_eq!(
            arc_cuda_graph::pending_token_count(),
            0,
            "the foreign queue must be dropped at B's take, not survive it"
        );

        // ── A resumes: nothing stale to consume, so it host-samples its own
        // logits. (Its dropped tokens were computed against device state that
        // no longer matches; replaying them would corrupt A's output.)
        assert_eq!(sample(&mut seq_a, PEAK_A, &rng).await, PEAK_A);
    }

    /// The double fault: the forward served a step from the pending queue
    /// (returning the ALIASED stale logits tensor, launching nothing) and the
    /// scheduler then swapped sequences before the sample ran. The taker is
    /// foreign, so there is no parked token to return — and the logits in
    /// hand are another sequence's stale graph output. Sampling them would
    /// leak A's distribution into B; the step must fail loudly instead.
    #[tokio::test]
    async fn an_aliased_step_served_for_another_sequence_fails_instead_of_sampling_stale_logits() {
        arc_cuda_graph::stand_down();
        let rng = Arc::new(std::sync::Mutex::new(Isaac64Rng::seed_from_u64(0)));

        // A (id 41) parks a burst; the forward short-circuits on it.
        let mut seq_a = greedy_seq(41);
        assert_eq!(sample(&mut seq_a, PEAK_A, &rng).await, PEAK_A);
        arc_cuda_graph::push_pending_tokens(&BURST);
        arc_cuda_graph::note_aliased_logits_served();

        // B (id 42) is what actually samples that step.
        let mut seq_b = greedy_seq(42);
        let res = sample_sequence(
            peaked_logits(PEAK_B),
            &mut seq_b,
            false,
            rng.clone(),
            false,
            false,
            false,
        )
        .await;
        assert!(
            res.is_err(),
            "a stale aliased-logits step served against a foreign queue must fail the step, \
             not return a plausible token"
        );

        // Control: the marker is harmless when the owner itself samples — the
        // parked token is taken and the aliased tensor is never read.
        let mut seq_c = greedy_seq(43);
        assert_eq!(sample(&mut seq_c, PEAK_A, &rng).await, PEAK_A);
        arc_cuda_graph::push_pending_tokens(&[25]);
        arc_cuda_graph::note_aliased_logits_served();
        assert_eq!(
            sample(&mut seq_c, PEAK_A, &rng).await,
            25,
            "the owner's own aliased-served step must take its parked token normally"
        );
    }
}
