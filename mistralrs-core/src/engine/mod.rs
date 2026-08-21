use crate::{
    distributed,
    paged_attention::block_hash::compute_block_hashes,
    pipeline::{
        llg::{constraint_from_llg_grammar, llg_grammar_from_constraint},
        text_models_inputs_processor::PagedAttentionMeta,
        CacheBackendMetadata, CacheInstruction,
    },
    prefix_cacher::PrefixCacheManagerV2,
    response::CompletionChoice,
    scheduler::{Scheduler, SchedulerOutput},
    search::{self, rag::SearchPipeline},
    sequence::{SeqStepType, StopReason},
    tools, CompletionResponse, SchedulerConfig, DEBUG,
};
use futures::FutureExt;
use interprocess::local_socket::{traits::Listener, ListenerOptions};
use llguidance::ParserFactory;
pub use logger::IntervalLogger;
use mistralrs_quant::RingConfig;
use rand::SeedableRng;
use rand_isaac::Isaac64Rng;
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    fmt,
    io::{BufWriter, Write},
    net::TcpListener,
    ops::Deref,
    str::FromStr,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, LazyLock,
    },
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use tokio::{
    select,
    sync::{
        mpsc::{error::TryRecvError, Receiver, Sender},
        Mutex, Notify,
    },
    task::JoinHandle,
};

use crate::{
    get_mut_arcmutex, handle_pipeline_forward_error,
    pipeline::{ModelCategory, Pipeline},
    request::Request,
    response::{ChatCompletionResponse, Choice, ResponseMessage},
    sequence::{SequenceRecognizer, SequenceState},
    Constraint,
};

mod add_request;
mod logger;
mod search_request;

pub enum EngineInstruction {
    Terminate,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
/// Embedding model used for ranking web search results internally.
pub enum SearchEmbeddingModel {
    #[default]
    #[serde(rename = "embedding_gemma")]
    EmbeddingGemma300M,
}

impl SearchEmbeddingModel {
    pub fn hf_model_id(&self) -> &'static str {
        match self {
            Self::EmbeddingGemma300M => "google/embeddinggemma-300m",
        }
    }

    pub fn variants() -> &'static [&'static str] {
        &["embedding_gemma"]
    }
}

impl fmt::Display for SearchEmbeddingModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmbeddingGemma300M => f.write_str("embedding_gemma"),
        }
    }
}

impl FromStr for SearchEmbeddingModel {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "embedding_gemma" => Ok(Self::EmbeddingGemma300M),
            other => Err(format!(
                "Unknown search embedding model `{other}`. Supported values: {}",
                Self::variants().join(", ")
            )),
        }
    }
}

const SEED: u64 = 0;
/// Terminate all sequences on the next scheduling step. Be sure to reset this.
/// This is a global flag for terminating all engines at once (e.g., Ctrl+C).
pub static TERMINATE_ALL_NEXT_STEP: AtomicBool = AtomicBool::new(false);

/// Engine-specific termination flags, per Engine thread ID.
static ENGINE_TERMINATE_FLAGS: LazyLock<
    std::sync::Mutex<HashMap<std::thread::ThreadId, Arc<AtomicBool>>>,
> = LazyLock::new(|| std::sync::Mutex::new(HashMap::new()));

/// Get or create a termination flag for the current engine thread.
pub fn get_engine_terminate_flag() -> Arc<AtomicBool> {
    let thread_id = std::thread::current().id();
    let mut flags = ENGINE_TERMINATE_FLAGS.lock().unwrap();
    flags
        .entry(thread_id)
        .or_insert_with(|| Arc::new(AtomicBool::new(false)))
        .clone()
}

/// Check if the current engine should terminate sequences.
pub fn should_terminate_engine_sequences() -> bool {
    // Check global flag first
    if TERMINATE_ALL_NEXT_STEP.load(Ordering::SeqCst) {
        return true;
    }
    // Then check engine-specific flag
    let thread_id = std::thread::current().id();
    if let Ok(flags) = ENGINE_TERMINATE_FLAGS.lock() {
        if let Some(flag) = flags.get(&thread_id) {
            return flag.load(Ordering::SeqCst);
        }
    }
    false
}

/// Reset termination flags for the current engine.
pub fn reset_engine_terminate_flag() {
    let thread_id = std::thread::current().id();
    if let Ok(flags) = ENGINE_TERMINATE_FLAGS.lock() {
        if let Some(flag) = flags.get(&thread_id) {
            flag.store(false, Ordering::SeqCst);
        }
    }
}

/// Engine instructions, per Engine (MistralRs) ID.
pub static ENGINE_INSTRUCTIONS: LazyLock<
    std::sync::Mutex<HashMap<usize, Option<EngineInstruction>>>,
> = LazyLock::new(|| std::sync::Mutex::new(HashMap::new()));

/// Turn a **panic** inside a pipeline step into an ordinary `Err`.
///
/// wave51-CB §3.2 / §4.1: three separate engine deaths in one session, all of
/// them a `.unwrap()` inside the forward reached by one unlucky batch. A panic
/// on the engine task takes the task down, and with it every *other* in-flight
/// request — the GSM8K run lost 32 requests that had nothing wrong with them.
/// The engine reboots lazily, on the next `get_sender`, and on the MTP pipeline
/// it did not recover at all.
///
/// Catching here converts "the engine died" into "this batch failed", which
/// `handle_pipeline_forward_error!` already knows how to report and recover
/// from (respond to each sequence, mark it `Error`, reset the cache, continue).
/// Known-bad states are still fixed at their source — this is the backstop for
/// the ones nobody has found yet, and it is what makes a single bad request
/// cost a single request.
///
/// The default panic hook still runs, so the backtrace is not lost.
async fn step_catching_panics<F>(stage: &str, fut: F) -> candle_core::Result<Duration>
where
    F: std::future::Future<Output = candle_core::Result<Duration>>,
{
    match std::panic::AssertUnwindSafe(fut).catch_unwind().await {
        Ok(res) => res,
        Err(panic) => {
            let msg = panic
                .downcast_ref::<&str>()
                .map(|s| (*s).to_string())
                .or_else(|| panic.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "<non-string panic payload>".to_string());
            Err(candle_core::Error::msg(format!(
                "panic in {stage}: {msg} (contained: the batch fails, the engine keeps running)"
            )))
        }
    }
}

pub struct Engine {
    tx: Sender<Request>,
    rx: Arc<Mutex<Receiver<Request>>>,
    pipeline: Arc<Mutex<dyn Pipeline>>,
    search_pipeline: Arc<Mutex<Option<SearchPipeline>>>,
    search_callback: Option<Arc<search::SearchCallback>>,
    tool_callbacks: tools::ToolCallbacks,
    tool_callbacks_with_tools: tools::ToolCallbacksWithTools,
    scheduler: Arc<Mutex<dyn Scheduler>>,
    id: Arc<Mutex<usize>>,
    no_kv_cache: bool,
    prefix_cacher: Arc<Mutex<PrefixCacheManagerV2>>,
    is_debug: bool,
    disable_eos_stop: bool,
    throughput_logging_enabled: bool,
    logger: Arc<IntervalLogger>,
    handles: Arc<Mutex<Vec<JoinHandle<()>>>>,
    pending_notify: Arc<Notify>,
}

impl Drop for Engine {
    fn drop(&mut self) {
        for handle in &*get_mut_arcmutex!(self.handles) {
            handle.abort();
        }
    }
}

impl Engine {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        tx: Sender<Request>,
        rx: Receiver<Request>,
        pipeline: Arc<Mutex<dyn Pipeline>>,
        config: SchedulerConfig,
        mut no_kv_cache: bool,
        mut no_prefix_cache: bool,
        prefix_cache_n: usize,
        disable_eos_stop: bool,
        throughput_logging_enabled: bool,
        search_embedding_model: Option<SearchEmbeddingModel>,
        search_callback: Option<Arc<search::SearchCallback>>,
        tool_callbacks: tools::ToolCallbacks,
        tool_callbacks_with_tools: tools::ToolCallbacksWithTools,
        logger: Arc<IntervalLogger>,
    ) -> anyhow::Result<Self> {
        no_kv_cache |= get_mut_arcmutex!(pipeline).get_metadata().no_kv_cache;

        no_prefix_cache = no_prefix_cache
            || no_kv_cache
            || get_mut_arcmutex!(pipeline).get_metadata().no_prefix_cache
            || prefix_cache_n == 0;

        // 🔑 TurboQuant and prefix caching are mutually exclusive today, and
        // this is where one of them is switched off on the user's behalf.
        //
        // Say so in full. The previous message named neither flag, so a run
        // that lost prefix caching gave the operator nothing to act on — and
        // this is not an exotic configuration: `--pa-cache-type` unset means
        // TurboQuant (`mistralrs-cli/src/args/paged_attn.rs:44-53`),
        // PagedAttention is the CUDA default, and `--prefix-cache-n` defaults
        // to 16, so the DEFAULT command line lands here.
        //
        // `prefix_cache_conflict` returns `None` when `prefix_cache_n == 0`,
        // i.e. when nothing was actually taken away.
        if !no_prefix_cache {
            let conflict = get_mut_arcmutex!(pipeline)
                .get_metadata()
                .cache_config
                .as_ref()
                .and_then(|c| c.cache_type.prefix_cache_conflict(prefix_cache_n));
            if let Some(conflict) = conflict {
                tracing::warn!("{conflict}");
                no_prefix_cache = true;
            }
        }

        let search_pipeline = match search_embedding_model {
            Some(search_embedding_model) => Some(SearchPipeline::new(
                search_embedding_model,
                &get_mut_arcmutex!(pipeline).device(),
            )?),
            None => None,
        };

        let scheduler = config.into_scheduler();

        // Configure prefix caching on the scheduler based on the global no_prefix_cache flag
        // This ensures PagedAttention prefix caching respects the same setting
        get_mut_arcmutex!(scheduler).set_prefix_caching_enabled(!no_prefix_cache);

        let has_paged_attention = get_mut_arcmutex!(scheduler).kv_cache_manager().is_some();

        Ok(Self {
            tx,
            rx: Arc::new(Mutex::new(rx)),
            pipeline,
            search_pipeline: Arc::new(Mutex::new(search_pipeline)),
            search_callback,
            tool_callbacks,
            tool_callbacks_with_tools,
            scheduler: scheduler.clone(),
            id: Arc::new(Mutex::new(0)),
            no_kv_cache,
            prefix_cacher: Arc::new(Mutex::new(PrefixCacheManagerV2::new(
                prefix_cache_n,
                no_prefix_cache,
                has_paged_attention,
            ))),
            is_debug: DEBUG.load(Ordering::Relaxed),
            disable_eos_stop,
            throughput_logging_enabled,
            logger,
            handles: Arc::new(Mutex::new(Vec::new())),
            pending_notify: Arc::new(Notify::new()),
        })
    }

    /// Returns the maximum supported sequence length for the underlying model, if applicable.
    #[allow(dead_code)]
    pub fn max_sequence_length(&self) -> Option<usize> {
        let pipeline = get_mut_arcmutex!(self.pipeline);
        let category = pipeline.category();

        if matches!(category, ModelCategory::Diffusion | ModelCategory::Speech) {
            None
        } else {
            Some(pipeline.get_metadata().max_seq_len)
        }
    }

    pub async fn run(self: Arc<Self>) {
        if self.throughput_logging_enabled {
            self.logger.enable_logging();
        }

        let rng = Arc::new(std::sync::Mutex::new(Isaac64Rng::seed_from_u64(SEED)));
        let mut last_completion_ids: Vec<usize> = vec![];
        'lp: loop {
            let should_terminate = || {
                matches!(
                    ENGINE_INSTRUCTIONS
                        .lock()
                        .expect("`ENGINE_INSTRUCTIONS` was poisoned")
                        .get(get_mut_arcmutex!(self.id).deref()),
                    Some(Some(EngineInstruction::Terminate))
                )
            };

            if should_terminate() {
                self.replicate_request_to_daemons(&Request::Terminate);
                break 'lp;
            }

            let mut channel_disconnected = false;
            loop {
                let next_request = {
                    let mut rx = self.rx.lock().await;
                    rx.try_recv()
                };

                match next_request {
                    Ok(request) => {
                        self.replicate_request_to_daemons(&request);
                        if matches!(request, Request::Terminate) {
                            break 'lp;
                        }
                        self.clone().handle_request(request).await;
                    }
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => {
                        channel_disconnected = true;
                        break;
                    }
                }
            }

            if channel_disconnected {
                break 'lp;
            }

            let (waiting_len, running_len) = {
                let scheduler = get_mut_arcmutex!(self.scheduler);
                (scheduler.waiting_len(), scheduler.running_len())
            };
            let scheduler_idle = waiting_len == 0 && running_len == 0;

            if scheduler_idle {
                if should_terminate() {
                    self.replicate_request_to_daemons(&Request::Terminate);
                    break 'lp;
                }
                enum WaitEvent {
                    Request(Option<Request>),
                    Wake,
                }
                let wait_for_request = async {
                    let mut rx = self.rx.lock().await;
                    rx.recv().await
                };
                tokio::pin!(wait_for_request);
                let wait_for_wake = self.pending_notify.notified();
                tokio::pin!(wait_for_wake);

                let event = select! {
                    res = &mut wait_for_request => WaitEvent::Request(res),
                    _ = &mut wait_for_wake => WaitEvent::Wake,
                };

                match event {
                    WaitEvent::Request(Some(request)) => {
                        self.replicate_request_to_daemons(&request);
                        if matches!(request, Request::Terminate) {
                            break 'lp;
                        }
                        self.clone().handle_request(request).await;
                        continue;
                    }
                    WaitEvent::Request(None) => break 'lp,
                    WaitEvent::Wake => {
                        continue;
                    }
                }
            }

            if TERMINATE_ALL_NEXT_STEP.load(Ordering::SeqCst) {
                self.replicate_request_to_daemons(&Request::TerminateAllSeqsNextStep);
            }

            let run_start = Instant::now();
            // Root of the profile tree. Opened here rather than at the top of
            // the loop because everything above this point is the idle path,
            // and averaging idle iterations into a step time would understate
            // every number below it.
            let _prof_step = arc_profiler::step_scope("step");
            let mut scheduler = {
                let _s = arc_profiler::span("scheduler.lock");
                get_mut_arcmutex!(self.scheduler)
            };
            let scheduled = {
                let _s = arc_profiler::span("scheduler.schedule");
                scheduler.schedule(&self.logger)
            };

            match scheduled {
                SchedulerOutput::DefaultScheduler {
                    output: mut scheduled,
                } => {
                    if !scheduled.completion.is_empty() {
                        // Stamped while `step` is the innermost span so the root
                        // carries the batch this profile describes. A profile
                        // that cannot say which B produced it is unattributable
                        // after the fact.
                        arc_profiler::set_geometry(scheduled.completion.len(), 1);
                        // Decode and prefill are separated *above* the pipeline
                        // so their sub-trees never merge. A prefill step is
                        // orders of magnitude larger; averaging the two under
                        // one `pipeline.step` node would hide the decode cost
                        // this profiler exists to find.
                        let _pdecode = arc_profiler::span("decode");
                        let current_completion_ids: Vec<usize> =
                            scheduled.completion.iter().map(|seq| *seq.id()).collect();
                        let res = {
                            // The pipeline mutex is held across the whole step,
                            // including the serial `responder.send().await` loop
                            // at the end of it, so the time spent waiting to
                            // acquire it is a scheduling cost worth naming.
                            let mut pipeline = {
                                let _s = arc_profiler::span("pipeline.lock");
                                get_mut_arcmutex!(self.pipeline)
                            };
                            let _pstep = arc_profiler::span("pipeline.step");
                            let pre_op = if !self.no_kv_cache
                                && last_completion_ids != current_completion_ids
                            {
                                CacheInstruction::In
                            } else {
                                CacheInstruction::Nothing
                            };
                            let post_op = if !self.no_kv_cache {
                                CacheInstruction::Out
                            } else {
                                CacheInstruction::Reset {
                                    load_preallocated_cache: false,
                                    reset_non_granular: false,
                                }
                            };

                            let return_raw_logits = scheduled.completion[0].return_raw_logits;
                            assert!(
                                scheduled
                                    .completion
                                    .iter()
                                    .all(|seq| seq.return_raw_logits == return_raw_logits),
                                "All sequences must either return raw logits, or not."
                            );

                            step_catching_panics(
                                "completion step",
                                pipeline.step(
                                    &mut scheduled.completion,
                                    false,
                                    return_raw_logits,
                                    &mut *get_mut_arcmutex!(self.prefix_cacher),
                                    self.disable_eos_stop,
                                    rng.clone(),
                                    CacheBackendMetadata::DefaultInstructions { pre_op, post_op },
                                ),
                            )
                            .await
                        };

                        handle_pipeline_forward_error!(
                            "completion step",
                            res,
                            &mut scheduled.completion,
                            self.pipeline,
                            'lp,
                            self.prefix_cacher
                        );

                        self.logger.add_tokens_processed(scheduled.completion.len());

                        last_completion_ids = current_completion_ids;
                    }

                    if !scheduled.prompt.is_empty() {
                        arc_profiler::set_geometry(
                            scheduled.prompt.len(),
                            scheduled.prompt.iter().map(|s| s.len()).max().unwrap_or(1),
                        );
                        let _pprompt = arc_profiler::span("prompt");
                        // Is this the LAST chunk of every prompt in the cohort?
                        // They share a cursor (the bucket key includes
                        // `token_offset`), so one answer covers the batch.
                        let chunk =
                            crate::pipeline::text_models_inputs_processor::prefill_chunk_size();
                        let cursor = scheduled.prompt[0].token_offset();
                        let final_chunk = match chunk {
                            Some(c) => scheduled
                                .prompt
                                .iter()
                                .all(|s| cursor + c >= s.get_toks().len()),
                            None => true,
                        };
                        let _chunk_guard =
                            (!final_chunk).then(crate::pipeline::mark_prefill_intermediate);

                        // Record prompt timing BEFORE step() so it is available if the
                        // response is sent from inside step().
                        //
                        // 🔴 This is why `pp N` reported `0.000±0.000` for every prompt
                        // benchmark. A prefill-only request (`max_len = 1`) finishes
                        // *during* this prompt step: `pipeline/sampling.rs` calls
                        // `seq.update_time_info()` and then `group.get_usage()` and
                        // dispatches `Response::Done` from inside `step()`. But on this
                        // (default, non-paged) path `prompt_timestamp` was only stamped
                        // AFTER `step()` returned, ~100 lines below. So at the moment the
                        // usage was built, `prompt_timestamp` was still `None`,
                        // `update_time_info` skipped `group.total_prompt_time`
                        // (`sequence.rs`, `if let Some(ts) = self.prompt_timestamp`), and
                        // `get_usage` took the `total_prompt_time == 0` branch and
                        // returned `avg_prompt_tok_per_sec: 0.0`. The harness then printed
                        // that zero as if it were a measurement.
                        //
                        // The PagedAttention arm below already does exactly this, with
                        // this same comment — the fix was simply never applied to the arm
                        // that actually serves (PagedAttention is banned here: it shadows
                        // the graph arm and measures zero tokens).
                        //
                        // `set_step_start_instant` gives `update_time_info` its in-flight
                        // fallback (`start.elapsed()`) while the prompt step is running;
                        // the post-step block below still overwrites `total_prompt_time`
                        // with the precise `prompt_exec_time` for sequences that go on to
                        // decode, so steady-state accounting is unchanged.
                        {
                            let now = SystemTime::now()
                                .duration_since(UNIX_EPOCH)
                                .expect("Time travel has occurred!")
                                .as_millis();
                            for seq in scheduled.prompt.iter_mut() {
                                seq.prompt_timestamp = Some(now);
                                seq.set_step_start_instant();
                            }
                        }

                        let prompt_exec_time = {
                            let mut pipeline = {
                                let _s = arc_profiler::span("pipeline.lock");
                                get_mut_arcmutex!(self.pipeline)
                            };
                            let _pstep = arc_profiler::span("pipeline.step");

                            // Run the prompt seqs
                            let post_op = if !self.no_kv_cache {
                                CacheInstruction::Out
                            } else {
                                CacheInstruction::Reset {
                                    load_preallocated_cache: false,
                                    reset_non_granular: false,
                                }
                            };

                            let return_raw_logits = scheduled.prompt[0].return_raw_logits;
                            assert!(
                                scheduled
                                    .prompt
                                    .iter()
                                    .all(|seq| seq.return_raw_logits == return_raw_logits),
                                "All sequences must either return raw logits, or not."
                            );

                            // This comes from prefix caching
                            // The invariant where all token offsets are the same is handled by the scheduler
                            let pre_op = if scheduled.prompt[0].token_offset() != 0 {
                                CacheInstruction::In
                            } else {
                                CacheInstruction::Reset {
                                    load_preallocated_cache: true,
                                    reset_non_granular: false,
                                }
                            };

                            step_catching_panics(
                                "prompt step",
                                pipeline.step(
                                    &mut scheduled.prompt,
                                    true,
                                    return_raw_logits,
                                    &mut *get_mut_arcmutex!(self.prefix_cacher),
                                    self.disable_eos_stop,
                                    rng.clone(),
                                    CacheBackendMetadata::DefaultInstructions { pre_op, post_op },
                                ),
                            )
                            .await
                        };

                        let prompt_exec_time = handle_pipeline_forward_error!(
                            "prompt step",
                            prompt_exec_time,
                            &mut scheduled.prompt,
                            self.pipeline,
                            'lp,
                            self.prefix_cacher
                        );

                        let total_processed_tokens: usize = scheduled
                            .prompt
                            .iter()
                            .map(|seq| seq.get_toks().len())
                            .sum();
                        self.logger.add_tokens_processed(total_processed_tokens);

                        if !final_chunk {
                            // Not done prefilling: advance the cursor and leave
                            // every row in `RunningPrompt`. The scheduler's
                            // bucket key includes `token_offset`, so the cohort
                            // stays together across chunks, and the loop returns
                            // to the top — which is the whole point, because
                            // that is where decode gets its turn.
                            let c = chunk.unwrap_or(0);
                            for seq in scheduled.prompt.iter_mut() {
                                seq.set_token_offset(cursor + c);
                            }
                            self.logger.add_tokens_processed(c * scheduled.prompt.len());
                            continue 'lp;
                        }
                        for seq in scheduled.prompt.iter_mut() {
                            // Prefill is finished, so the cursor must go back to
                            // zero: it is part of the bucket key, and a decode
                            // cohort carrying stale per-row offsets would shatter
                            // into one bucket per offset — the exact pathology
                            // `feat/dense-ragged-decode` removes.
                            seq.set_token_offset(0);
                            match seq.sequence_stepping_type() {
                                SeqStepType::OneShot => {
                                    seq.set_state(SequenceState::Done(StopReason::GeneratedImage))
                                }
                                SeqStepType::PromptAndDecode => {
                                    seq.set_state(SequenceState::RunningCompletion)
                                }
                            }
                            let now = SystemTime::now()
                                .duration_since(UNIX_EPOCH)
                                .expect("Time travel has occurred!")
                                .as_millis();
                            #[allow(clippy::cast_precision_loss)]
                            let prompt_tok_per_sec =
                                seq.len() as f32 / prompt_exec_time.as_secs_f32();
                            seq.prompt_tok_per_sec = prompt_tok_per_sec;
                            seq.prompt_timestamp = Some(now);
                            seq.total_prompt_time = Some(prompt_exec_time.as_millis());
                            seq.step_start_instant = None;
                        }

                        // 🔑 Publish each prompt's KV NOW, while its request is
                        // still generating.
                        //
                        // Until this ran, `prefix_cacher.add_sequence` had two
                        // callers and both sat inside `if let Some(reason) =
                        // is_done` (`pipeline/sampling.rs:264`, `:423`), so a
                        // 2,048-token system prompt stayed invisible to every
                        // other request for the whole generation that followed
                        // it — tens of seconds during which each arrival paid
                        // the full prefill again.
                        //
                        // Here rather than in `sampling.rs` because this is the
                        // one place that knows a *prompt* step just completed,
                        // and it has already stamped `total_prompt_time`, which
                        // is the measured cost the eviction scorer wants. A
                        // `OneShot` sequence is skipped: it was just set to
                        // `Done`, has no continuation, and the finish path
                        // stores it anyway.
                        {
                            let mut prefix_cacher = get_mut_arcmutex!(self.prefix_cacher);
                            let pipeline = get_mut_arcmutex!(self.pipeline);
                            let is_hybrid = !self.no_kv_cache && pipeline.cache().is_hybrid();
                            for seq in scheduled.prompt.iter_mut() {
                                if !matches!(
                                    seq.sequence_stepping_type(),
                                    SeqStepType::PromptAndDecode
                                ) {
                                    continue;
                                }
                                let recurrent_snapshots = if is_hybrid {
                                    seq.recurrent_state_idx().and_then(|idx| {
                                        pipeline.cache().hybrid().snapshot_recurrent_state(idx).ok()
                                    })
                                } else {
                                    None
                                };
                                prefix_cacher.add_prefilled_sequence(seq, recurrent_snapshots);
                            }
                            if let Err(e) = prefix_cacher.evict_caches() {
                                tracing::warn!(
                                    "prefix cache: eviction after prefill publication failed: {e}"
                                );
                            }
                        }
                        last_completion_ids = vec![];
                    }

                    if self.is_debug {
                        let ms_from_last_run = run_start.elapsed().as_secs_f64();
                        let total_len = scheduled.prompt.len() + scheduled.completion.len();
                        if total_len > 0 {
                            let prompt_lengths = scheduled
                                .prompt
                                .iter()
                                .map(|seq| seq.len().to_string())
                                .collect::<Vec<_>>()
                                .join(", ");

                            let completion_lengths = scheduled
                                .completion
                                .iter()
                                .map(|seq| seq.len().to_string())
                                .collect::<Vec<_>>()
                                .join(", ");

                            tracing::info!(
                                "Prompt[{}] Completion[{}] - {}ms",
                                prompt_lengths,
                                completion_lengths,
                                ms_from_last_run * 1000.,
                            );
                        }
                    }
                }
                SchedulerOutput::PagedAttention { mut output } => {
                    if !output.scheduled.is_empty() {
                        let is_prompt = get_mut_arcmutex!(output.scheduled[0]).is_prompt();

                        // Record prompt timing BEFORE step() so it's available if response is sent inside step()
                        if is_prompt {
                            let now = SystemTime::now()
                                .duration_since(UNIX_EPOCH)
                                .expect("Time travel has occurred!")
                                .as_millis();
                            for seq in output.scheduled.iter() {
                                let mut seq_guard = get_mut_arcmutex!(seq);
                                seq_guard.prompt_timestamp = Some(now);
                                // Start the timer using Instant for accurate duration measurement
                                seq_guard.set_step_start_instant();
                            }
                        }

                        let mut guards = output
                            .scheduled
                            .iter_mut()
                            .map(|seq| seq.lock().unwrap())
                            .collect::<Vec<_>>();

                        let mut guards_mut =
                            guards.iter_mut().map(|seq| &mut **seq).collect::<Vec<_>>();

                        let res = {
                            let mut pipeline = get_mut_arcmutex!(self.pipeline);

                            let block_size = scheduler.block_size().unwrap();

                            // For hybrid models under paged attention, restore recurrent state
                            // from block-hash keyed prefix snapshots before prompt prefill.
                            if is_prompt && pipeline.cache().is_hybrid() {
                                let mut hybrid_cache = pipeline.cache().hybrid();
                                let mut prefix_cacher = get_mut_arcmutex!(self.prefix_cacher);
                                let kv_cache_manager = scheduler.kv_cache_manager().unwrap();

                                for seq in guards_mut.iter_mut() {
                                    let cached_prefix_len = seq.prefix_cache_len();
                                    if cached_prefix_len == 0 {
                                        continue;
                                    }

                                    let mut fallback_to_full_prompt = false;

                                    let slot_idx = match seq.recurrent_state_idx() {
                                        Some(idx) => idx,
                                        None => {
                                            tracing::warn!("Sequence {} has paged prefix hit but no recurrent_state_idx; recomputing full prompt.", seq.id());
                                            fallback_to_full_prompt = true;
                                            // Dummy value, unused in fallback path.
                                            0usize
                                        }
                                    };

                                    if !fallback_to_full_prompt {
                                        if cached_prefix_len % block_size != 0 {
                                            tracing::warn!(
                                                "Sequence {} has non-aligned paged prefix len {}; recomputing full prompt.",
                                                seq.id(),
                                                cached_prefix_len
                                            );
                                            fallback_to_full_prompt = true;
                                        } else {
                                            let num_prefix_blocks = cached_prefix_len / block_size;
                                            let block_hashes = compute_block_hashes(
                                                seq.get_toks(),
                                                block_size,
                                                seq.mm_features(),
                                                &[],
                                            );
                                            if block_hashes.len() < num_prefix_blocks {
                                                fallback_to_full_prompt = true;
                                            } else if let Some(snapshots) = prefix_cacher
                                                .get_paged_recurrent_prefix(
                                                    &block_hashes[..num_prefix_blocks],
                                                )
                                            {
                                                if let Err(e) = hybrid_cache
                                                    .restore_recurrent_state(slot_idx, &snapshots)
                                                {
                                                    tracing::warn!(
                                                        "Failed restoring paged recurrent prefix state for sequence {}: {e}",
                                                        seq.id()
                                                    );
                                                    fallback_to_full_prompt = true;
                                                }
                                            } else {
                                                tracing::warn!(
                                                    "No recurrent prefix snapshot for sequence {} at cached prefix length {}; recomputing full prompt.",
                                                    seq.id(),
                                                    cached_prefix_len
                                                );
                                                fallback_to_full_prompt = true;
                                            }
                                        }
                                    }

                                    if fallback_to_full_prompt {
                                        let seq_id = *seq.id();
                                        let num_tokens = seq.get_toks().len();
                                        let mut kv_mgr = get_mut_arcmutex!(kv_cache_manager);
                                        kv_mgr.free(seq_id);
                                        let realloc_ok = kv_mgr
                                            .allocate_slots(seq_id, num_tokens, &[])
                                            .is_some();
                                        drop(kv_mgr);

                                        if !realloc_ok {
                                            tracing::warn!(
                                                "Failed to reallocate fresh paged KV blocks for sequence {} after recurrent-prefix fallback.",
                                                seq_id
                                            );
                                            seq.set_state(SequenceState::FinishedIgnored);
                                        }
                                        seq.set_prefix_cache_len(0);
                                    }
                                }

                                // Drop sequences that were canceled due fallback allocation failures.
                                guards_mut.retain(|seq| !seq.is_finished_paged_attn());
                            }

                            if guards_mut.is_empty() {
                                Ok(Duration::ZERO)
                            } else {
                                let metadata = PagedAttentionMeta {
                                    block_size,
                                    sliding_window: pipeline.get_metadata().sliding_window,
                                    kv_cache_manager: scheduler.kv_cache_manager().unwrap(),
                                };

                                let return_raw_logits = guards_mut[0].return_raw_logits;
                                assert!(
                                    guards_mut
                                        .iter()
                                        .all(|seq| seq.return_raw_logits == return_raw_logits),
                                    "All sequences must either return raw logits, or not."
                                );

                                // Autonomous decode fast path: when the
                                // pipeline has an autonomous runner with a
                                // captured graph, the full decode loop
                                // (forward → sample → step → check_done)
                                // runs entirely on the GPU. We only attempt
                                // it for decode batches (not prompts) and
                                // when raw logits are not requested.
                                //
                                // The trait method returns `Ok(None)` if the
                                // runner is unavailable or not yet captured,
                                // in which case we fall through to the
                                // standard step-by-step decode path.
                                #[cfg(feature = "cuda")]
                                let autonomous_handled = if !is_prompt && !return_raw_logits {
                                    let __auto_t0 = Instant::now();
                                    // Build AutonomousDecodeContext: per-seq
                                    // last token, position, block_table, slot
                                    // mapping. The runner uses these to
                                    // prime its input buffers before launching
                                    // the captured graph. Heap-allocations here
                                    // are O(batch * max_blocks) — acceptable
                                    // because we only run this when
                                    // autonomous decode is otherwise viable.
                                    let bs = guards_mut.len();
                                    let max_blocks_per_seq = (pipeline.get_metadata().max_seq_len
                                        + block_size - 1) / block_size;
                                    let mut next_token_ids: Vec<i32> = Vec::with_capacity(bs);
                                    let mut positions: Vec<i32> = Vec::with_capacity(bs);
                                    let mut context_lens: Vec<i32> = Vec::with_capacity(bs);
                                    let mut slot_mappings: Vec<i64> = Vec::with_capacity(bs);
                                    let mut block_tables_flat: Vec<i32> =
                                        Vec::with_capacity(bs * max_blocks_per_seq);
                                    let mut ctx_build_ok = true;
                                    {
                                        let kv_mgr_arc = scheduler.kv_cache_manager().unwrap();
                                        let kv_mgr = get_mut_arcmutex!(kv_mgr_arc);
                                        for seq in guards_mut.iter() {
                                            let toks = seq.get_toks();
                                            if toks.is_empty() {
                                                ctx_build_ok = false;
                                                break;
                                            }
                                            let last_tok = *toks.last().unwrap() as i32;
                                            let pos = (toks.len() - 1) as i32;
                                            next_token_ids.push(last_tok);
                                            positions.push(pos);
                                            context_lens.push(toks.len() as i32);
                                            let seq_id = *seq.id();
                                            let bt = match kv_mgr.get_block_table(seq_id, max_blocks_per_seq) {
                                                Some(v) => v,
                                                None => { ctx_build_ok = false; break; }
                                            };
                                            block_tables_flat.extend_from_slice(&bt);
                                            let slots = match kv_mgr.get_slot_mapping(seq_id, toks.len() - 1, 1) {
                                                Some(v) => v,
                                                None => { ctx_build_ok = false; break; }
                                            };
                                            if let Some(&s) = slots.first() {
                                                slot_mappings.push(s);
                                            } else {
                                                ctx_build_ok = false;
                                                break;
                                            }
                                        }
                                    }
                                    if !ctx_build_ok {
                                        tracing::debug!(
                                            "autonomous_decode: failed to build per-seq context; falling back"
                                        );
                                        false
                                    } else {
                                    let ctx = crate::pipeline::AutonomousDecodeContext {
                                        next_token_ids: &next_token_ids,
                                        positions: &positions,
                                        block_tables_flat: &block_tables_flat,
                                        context_lens: &context_lens,
                                        slot_mappings: &slot_mappings,
                                        block_size,
                                        max_blocks_per_seq,
                                    };
                                    match pipeline.autonomous_decode(&mut guards_mut, &ctx) {
                                        Ok(Some(tokens_per_seq)) => {
                                            // The autonomous runner generated
                                            // tokens entirely on-GPU. Now we
                                            // need to thread them through the
                                            // normal add_token / stop-string /
                                            // streaming machinery so callers
                                            // see the output. We reuse
                                            // `finish_or_add_toks_to_seq`
                                            // (the same helper the sampling
                                            // path uses post-CPU-sample) per
                                            // token. Logprobs are zeroed —
                                            // sampling happened on-GPU so we
                                            // don't have the post-softmax
                                            // distribution to reconstruct.
                                            let metadata = pipeline.get_metadata();
                                            let eos_tok_vec = if self.disable_eos_stop {
                                                None
                                            } else {
                                                Some(metadata.eos_tok.clone())
                                            };
                                            let pipeline_ref: &dyn Pipeline = &*pipeline;
                                            let mut prefix_cacher = get_mut_arcmutex!(self.prefix_cacher);
                                            let mut synth_err: Option<candle_core::Error> = None;

                                            // tokens_per_seq is indexed by the
                                            // RUNNER's padded batch slot, NOT
                                            // by guards_mut index. The
                                            // pipeline-level
                                            // `autonomous_decode` impl is
                                            // responsible for returning one
                                            // Vec per active sequence in the
                                            // order matching guards_mut.
                                            for (seq_idx, token_ids) in tokens_per_seq.iter().enumerate() {
                                                if seq_idx >= guards_mut.len() { break; }
                                                let seq = &mut guards_mut[seq_idx];
                                                for &tok_id in token_ids {
                                                    // Skip negative / sentinel ids the kernel may
                                                    // have written for "no token this step" rows.
                                                    if tok_id < 0 { continue; }
                                                    let logprobs = crate::sampler::Logprobs {
                                                        token: tok_id as u32,
                                                        logprob: 0.0,
                                                        bytes: None,
                                                        top_logprobs: None,
                                                    };
                                                    if let Err(e) = crate::pipeline::sampling::finish_or_add_toks_to_seq(
                                                        pipeline_ref,
                                                        &mut prefix_cacher,
                                                        seq,
                                                        logprobs,
                                                        eos_tok_vec.as_deref(),
                                                        true,
                                                    )
                                                    .await
                                                    {
                                                        synth_err = Some(e);
                                                        break;
                                                    }
                                                    // Stop early on EOS / stop-string / length —
                                                    // finish_or_add_toks_to_seq sets seq state.
                                                    if matches!(
                                                        seq.getstate(),
                                                        crate::sequence::SequenceState::Done(_)
                                                    ) {
                                                        break;
                                                    }
                                                }
                                                if synth_err.is_some() { break; }
                                            }
                                            drop(prefix_cacher);

                                            match synth_err {
                                                Some(e) => {
                                                    tracing::warn!(
                                                        "autonomous_decode response synthesis failed after {:?}: {e}; falling back to step()",
                                                        __auto_t0.elapsed()
                                                    );
                                                    false
                                                }
                                                None => {
                                                    tracing::debug!(
                                                        "autonomous_decode handled batch in {:?} ({} sequences)",
                                                        __auto_t0.elapsed(),
                                                        tokens_per_seq.len(),
                                                    );
                                                    true
                                                }
                                            }
                                        }
                                        Ok(None) => false,
                                        Err(e) => {
                                            tracing::warn!(
                                                "autonomous_decode error, falling back to step(): {e}"
                                            );
                                            false
                                        }
                                    }
                                    } // close else of !ctx_build_ok
                                } else {
                                    false
                                };

                                #[cfg(not(feature = "cuda"))]
                                let autonomous_handled = false;

                                if autonomous_handled {
                                    // Autonomous path already advanced
                                    // sequences and sent responses; skip the
                                    // step-by-step path for this scheduled
                                    // batch. Return zero duration since the
                                    // GPU-only path's runtime is tracked
                                    // separately via the runner's profiling.
                                    Ok(Duration::ZERO)
                                } else {
                                    step_catching_panics(
                                        "step",
                                        pipeline.step(
                                            &mut guards_mut,
                                            is_prompt,
                                            return_raw_logits,
                                            &mut *get_mut_arcmutex!(self.prefix_cacher),
                                            self.disable_eos_stop,
                                            rng.clone(),
                                            CacheBackendMetadata::PagedAttention { metadata },
                                        ),
                                    )
                                    .await
                                }
                            }
                        };

                        handle_pipeline_forward_error!(
                            "step",
                            res,
                            &mut guards_mut,
                            self.pipeline,
                            'lp,
                            self.prefix_cacher
                        );

                        let total_processed_tokens: usize = guards_mut
                            .iter()
                            .map(|seq| {
                                if seq.is_prompt() {
                                    seq.get_toks().len()
                                } else {
                                    1
                                }
                            })
                            .sum();
                        self.logger.add_tokens_processed(total_processed_tokens);

                        // Capture recurrent states at full-block boundaries so hybrid models can
                        // reuse recurrent prefix state when paged prefix caching hits.
                        {
                            let pipeline = get_mut_arcmutex!(self.pipeline);
                            if pipeline.cache().is_hybrid() {
                                let block_size = scheduler.block_size().unwrap();
                                let hybrid_cache = pipeline.cache().hybrid();
                                let mut prefix_cacher = get_mut_arcmutex!(self.prefix_cacher);

                                for seq in guards_mut.iter() {
                                    let seq_len = seq.get_toks().len();
                                    if seq_len == 0 || seq_len % block_size != 0 {
                                        continue;
                                    }

                                    let Some(slot_idx) = seq.recurrent_state_idx() else {
                                        continue;
                                    };

                                    let snapshots = match hybrid_cache
                                        .snapshot_recurrent_state(slot_idx)
                                    {
                                        Ok(snapshots) => snapshots,
                                        Err(e) => {
                                            tracing::warn!(
                                                    "Failed snapshotting recurrent state for sequence {}: {e}",
                                                    seq.id()
                                                );
                                            continue;
                                        }
                                    };
                                    if snapshots.is_empty() {
                                        continue;
                                    }

                                    let num_blocks = seq_len / block_size;
                                    let block_hashes = compute_block_hashes(
                                        seq.get_toks(),
                                        block_size,
                                        seq.mm_features(),
                                        &[],
                                    );
                                    if block_hashes.len() < num_blocks {
                                        continue;
                                    }
                                    prefix_cacher.add_paged_recurrent_prefix(
                                        block_hashes[..num_blocks].to_vec(),
                                        snapshots,
                                    );
                                }
                            }
                        }

                        if self.is_debug {
                            let ms_from_last_run = run_start.elapsed().as_secs_f64();
                            let total_len = guards.len();
                            if total_len > 0 {
                                let lengths = guards
                                    .iter()
                                    .map(|seq| seq.len().to_string())
                                    .collect::<Vec<_>>()
                                    .join(", ");

                                let (prompt_lengths, completion_lengths) = if is_prompt {
                                    (lengths, "".to_string())
                                } else {
                                    ("".to_string(), lengths)
                                };

                                tracing::info!(
                                    "Prompt[{}] Completion[{}] - {}ms",
                                    prompt_lengths,
                                    completion_lengths,
                                    ms_from_last_run * 1000.,
                                );
                            }
                        }

                        if is_prompt {
                            #[allow(clippy::cast_precision_loss)]
                            for mut seq in guards {
                                // Use Instant duration for accurate prompt timing
                                if let Some(start) = seq.step_start_instant {
                                    let duration = start.elapsed();
                                    seq.prompt_tok_per_sec =
                                        seq.len() as f32 / duration.as_secs_f32();
                                    seq.total_prompt_time = Some(duration.as_millis());
                                    seq.step_start_instant = None;
                                }
                                let now = SystemTime::now()
                                    .duration_since(UNIX_EPOCH)
                                    .expect("Time travel has occurred!")
                                    .as_millis();
                                seq.prompt_timestamp = Some(now);
                            }
                        }
                    }
                }
            }

            // Free recurrent state pool slots for finished sequences (hybrid models)
            {
                let pipeline = get_mut_arcmutex!(self.pipeline);
                if !pipeline.get_metadata().no_kv_cache && pipeline.cache().is_hybrid() {
                    let recurrent_indices = scheduler.get_finished_recurrent_indices();
                    if !recurrent_indices.is_empty() {
                        let mut hybrid_cache = pipeline.cache().hybrid();
                        for idx in recurrent_indices {
                            hybrid_cache.free_seq(idx);
                        }
                    }
                }
            }
            scheduler.free_finished_sequence_groups();
        }
    }

    fn build_sequence_recognizer(
        factory: &Option<Arc<ParserFactory>>,
        constraint: &Constraint,
    ) -> anyhow::Result<SequenceRecognizer> {
        if let Some(grm) = llg_grammar_from_constraint(constraint)? {
            let factory = factory
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("No token environment (llg_factory) found."))?;
            let llg = constraint_from_llg_grammar(factory, grm)?;
            Ok(SequenceRecognizer::Llguidance(Box::new(llg)))
        } else {
            Ok(SequenceRecognizer::None)
        }
    }

    fn replicate_request_to_daemons(&self, request: &Request) {
        if !distributed::is_daemon() && mistralrs_quant::distributed::use_nccl() {
            let name = distributed::ipc_name().unwrap();
            let num_workers =
                mistralrs_quant::distributed::get_global_tp_size_from_devices().unwrap() - 1;
            let listener = ListenerOptions::new().name(name).create_sync().unwrap();

            for _ in 0..num_workers {
                let stream = listener.accept().unwrap();
                let mut writer = BufWriter::new(stream);
                let req = format!("{}\n", serde_json::to_string(&request).unwrap());
                writer.write_all(req.as_bytes()).unwrap();
            }
        } else if !distributed::is_daemon() && cfg!(feature = "ring") {
            let num_workers =
                mistralrs_quant::distributed::get_global_tp_size_from_devices().unwrap() - 1;
            let master_port = RingConfig::load().master_port;
            let listener =
                TcpListener::bind(format!("0.0.0.0:{master_port}")).expect("bind replicator");

            for _ in 0..num_workers {
                let (stream, _) = listener.accept().unwrap();
                let mut writer = BufWriter::new(stream);
                let req = format!("{}\n", serde_json::to_string(&request).unwrap());
                writer.write_all(req.as_bytes()).unwrap();
            }
        }
    }
}

#[cfg(test)]
mod step_panic_containment_tests {
    use super::step_catching_panics;
    use std::time::Duration;

    /// wave51-CB: three engine deaths in one session, all of them a panic on
    /// the engine task reached by one batch. The engine reboots only lazily
    /// (on the next `get_sender`), and on the MTP pipeline it did not recover
    /// at all — subsequent requests hung with the GPU at 0%.
    ///
    /// The specific panics are fixed at their source; this is the backstop that
    /// makes "one unlucky sequence" cost one batch instead of the process.
    ///
    /// Mutation check: replace `step_catching_panics`' body with
    /// `fut.await` and this test aborts the harness with the panic instead of
    /// returning an `Err`.
    #[tokio::test]
    async fn a_panicking_step_becomes_an_error_not_an_engine_death() {
        // Keep the default hook's noise out of the test log while still
        // proving the panic really happened.
        let prev = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let res = step_catching_panics("completion step", async {
            panic!("shape mismatch on dim 1, 18 <> 22");
        })
        .await;
        std::panic::set_hook(prev);

        let err = res
            .expect_err("a panicking step must surface as an Err, not unwind the engine task")
            .to_string();
        assert!(
            err.contains("panic in completion step"),
            "the error must name the stage, got: {err}"
        );
        assert!(
            err.contains("shape mismatch on dim 1, 18 <> 22"),
            "the original panic message must survive so the operator can still \
             diagnose it, got: {err}"
        );
    }

    /// Containment must not change the happy path.
    #[tokio::test]
    async fn a_healthy_step_is_passed_through_unchanged() {
        let res = step_catching_panics("prompt step", async { Ok(Duration::from_millis(7)) })
            .await
            .expect("a successful step must pass through");
        assert_eq!(res, Duration::from_millis(7));
    }
}
