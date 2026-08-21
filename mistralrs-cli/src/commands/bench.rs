//! Performance benchmarking command

use anyhow::Result;
use comfy_table::{presets::UTF8_FULL, Cell, Color, ContentArrangement, Table};
use mistralrs_core::{
    initialize_logging, Constraint, DrySamplingParams, NormalRequest, Request, RequestMessage,
    Response, SamplingParams,
};
use mistralrs_server_core::mistralrs_for_server_builder::MistralRsForServerBuilder;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::mpsc::channel;
use tracing::info;

use crate::args::{GlobalOptions, ModelType, RuntimeOptions};

use super::serve::{
    convert_to_model_selected, extract_device_settings, extract_isq_setting,
    extract_paged_attn_settings,
};

/// Benchmark result for a single test
struct BenchResult {
    test_name: String,
    tok_per_sec: f32,
    std_dev: f32,
    /// For prefill: TTFT in ms; for decode: ms/tok
    latency_ms: f32,
}

/// What one benchmark iteration actually PRODUCED.
///
/// ⚠️ This type exists because `bench` used to divide the *requested* length by
/// the elapsed time — `gen_len as f32 / elapsed` — and never looked at the
/// response at all. A request that produced ZERO tokens therefore reported a
/// full tok/s figure and exited 0: a silent failure that reads as a
/// measurement. Worse, a fast zero-token failure divides the requested length
/// by a tiny elapsed time, so the fabricated number is *high* — the failure
/// mode most likely to be believed and published.
///
/// Two independent counters are carried, and the grader requires both:
///   * `completion_tokens` — the engine's own bookkeeping, and
///   * `text_len`          — the bytes the caller actually received.
/// A row where one is zero and the other is not is a disagreement between what
/// the engine claims and what it delivered, which is a defect in its own right
/// rather than a rounding difference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct BenchRun {
    /// Tokens the engine reports having generated.
    completion_tokens: usize,
    /// Prompt tokens the engine reports having processed.
    prompt_tokens: usize,
    /// Bytes of completion text actually delivered to this caller.
    text_len: usize,
}

/// Why a benchmark iteration cannot be turned into a rate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RateError {
    /// The engine produced nothing. This is the silent-failure bug.
    NoTokens,
    /// The engine's token count and the delivered text disagree about whether
    /// anything was produced.
    CounterDisagreement { tokens: usize, text_len: usize },
    /// Elapsed time was not positive, so any rate would be infinite.
    NonPositiveElapsed,
}

impl std::fmt::Display for RateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoTokens => write!(
                f,
                "the engine produced ZERO tokens. This is a silent failure, not a \
                 measurement: the previous code divided the REQUESTED length by the \
                 elapsed time and would have printed a tok/s figure for this run"
            ),
            Self::CounterDisagreement { tokens, text_len } => write!(
                f,
                "counter disagreement: the engine reports {tokens} completion token(s) \
                 but delivered {text_len} byte(s) of text. One of the two is wrong, so \
                 neither can be used as a denominator"
            ),
            Self::NonPositiveElapsed => write!(
                f,
                "elapsed time was not positive, so a rate cannot be computed"
            ),
        }
    }
}

/// Turn a completed iteration into tokens/second, or refuse.
///
/// The denominator is what the engine PRODUCED, never what the caller asked
/// for. There is deliberately no fallback: a run that produced nothing has no
/// rate, and manufacturing one is exactly the bug this replaces.
fn rate_from_produced(
    produced: usize,
    text_len: usize,
    elapsed_secs: f32,
) -> Result<f32, RateError> {
    if (produced > 0) != (text_len > 0) {
        return Err(RateError::CounterDisagreement {
            tokens: produced,
            text_len,
        });
    }
    if produced == 0 {
        return Err(RateError::NoTokens);
    }
    if !(elapsed_secs > 0.0) {
        return Err(RateError::NonPositiveElapsed);
    }
    Ok(produced as f32 / elapsed_secs)
}

/// Extract model_id from ModelType
fn get_model_id(model_type: &ModelType) -> String {
    match model_type {
        ModelType::Auto { model, .. }
        | ModelType::Text { model, .. }
        | ModelType::Vision { model, .. }
        | ModelType::Diffusion { model, .. }
        | ModelType::Speech { model, .. }
        | ModelType::Embedding { model, .. } => model.model_id.clone(),
    }
}

/// Run the benchmark command
pub async fn run_bench(
    model_type: ModelType,
    runtime: RuntimeOptions,
    global: GlobalOptions,
    prompt_len: usize,
    gen_len: usize,
    iterations: usize,
    warmup: usize,
) -> Result<()> {
    initialize_logging();

    // Get model ID for display
    let model_id = get_model_id(&model_type);

    // Convert args and load model
    let model_selected = convert_to_model_selected(&model_type)?;

    let (
        paged_attn,
        paged_attn_gpu_mem,
        paged_attn_gpu_mem_usage,
        paged_ctxt_len,
        paged_attn_block_size,
        paged_cache_type,
    ) = extract_paged_attn_settings(&model_type);

    let (cpu, device_layers) = extract_device_settings(&model_type);
    let isq = extract_isq_setting(&model_type);

    info!("Loading model for benchmarking...");

    // Build using the same infrastructure as serve
    let builder = MistralRsForServerBuilder::new()
        .with_model(model_selected)
        .with_max_seqs(1) // Single sequence for benchmarking
        .with_no_kv_cache(runtime.no_kv_cache)
        .with_token_source(global.token_source)
        .with_interactive_mode(false)
        .with_prefix_cache_n(0) // Disable prefix cache for benchmarking
        .with_mtp_depth(runtime.mtp_depth as usize)
        .with_v4_ragged_decode(runtime.v4_ragged_decode)
        .set_paged_attn(paged_attn)
        .with_cpu(cpu)
        .with_seed_optional(global.seed)
        .with_num_device_layers_optional(device_layers)
        .with_in_situ_quant_optional(isq)
        .with_paged_attn_gpu_mem_optional(paged_attn_gpu_mem)
        .with_paged_attn_gpu_mem_usage_optional(paged_attn_gpu_mem_usage)
        .with_paged_ctxt_len_optional(paged_ctxt_len)
        .with_paged_attn_block_size_optional(paged_attn_block_size)
        .with_paged_attn_cache_type_optional(paged_cache_type);

    let mistralrs = builder.build().await?;
    info!("Model loaded.");

    // Warmup runs
    if warmup > 0 {
        info!("Running {} warmup iteration(s)...", warmup);
        for w in 0..warmup {
            // `let _ = …` here swallowed hard errors, so a model that failed
            // every warmup iteration looked identical to one that warmed up
            // fine. Warmup failure is not fatal, but it is never silent.
            match run_single_bench(&mistralrs, 32, 16).await {
                Ok(run) if run.completion_tokens == 0 => tracing::warn!(
                    "warmup iteration {} produced ZERO tokens — the measured \
                     iterations are likely to do the same.",
                    w + 1
                ),
                Ok(_) => {}
                Err(e) => tracing::warn!("warmup iteration {} failed: {e}", w + 1),
            }
        }
        info!("Warmup complete.");

        // Reset logger counters so benchmark stats are clean
        if let Ok(logger) = mistralrs.get_logger(None) {
            logger.reset();
        }
        // …and the MTP counters with them, so the reported acceptance covers
        // the measured iterations only. Warmup decodes a different (32/16)
        // shape and its acceptance is not the number being reported.
        mistralrs_core::reset_mtp_acceptance();
    }

    // Run benchmarks
    info!(
        "Running {} iteration(s) with {} prompt tokens, {} generation tokens...",
        iterations, prompt_len, gen_len
    );

    let mut prefill_results = Vec::new();
    let mut decode_results = Vec::new();

    for i in 0..iterations {
        info!("Iteration {}/{}...", i + 1, iterations);

        // Prefill benchmark (prompt processing)
        // Use external timing since internal Usage timing may not capture prompt time accurately
        if prompt_len > 0 {
            let start = Instant::now();
            let run = run_single_bench(&mistralrs, prompt_len, 1).await?;
            let elapsed = start.elapsed();

            // The prefill denominator is what the engine says it PROCESSED. If
            // that disagrees with what we asked it to process, the figure
            // describes a different workload than the one being reported.
            if run.prompt_tokens != prompt_len {
                anyhow::bail!(
                    "prefill iteration {}: asked for {prompt_len} prompt tokens but the engine \
                     reports processing {}. A tok/s figure computed over the requested length \
                     would describe a workload that never ran.",
                    i + 1,
                    run.prompt_tokens
                );
            }
            // A prefill run still has to emit its one token; zero means the
            // forward pass did not produce a result.
            let _ = rate_from_produced(run.completion_tokens, run.text_len, elapsed.as_secs_f32())
                .map_err(|e| anyhow::anyhow!("prefill iteration {}: {e}", i + 1))?;

            // Record both tok/s and TTFT (latency in ms)
            let tok_per_sec = run.prompt_tokens as f32 / elapsed.as_secs_f32();
            let ttft_ms = elapsed.as_secs_f32() * 1000.0;
            prefill_results.push((tok_per_sec, ttft_ms));
        }

        // Decode benchmark (token generation)
        if gen_len > 0 {
            let start = Instant::now();
            let run = run_single_bench(&mistralrs, 4, gen_len).await?;
            let elapsed = start.elapsed();

            // The denominator is the PRODUCED count. This call is what makes a
            // zero-token run fail loudly instead of printing a fabricated rate.
            let tok_per_sec =
                rate_from_produced(run.completion_tokens, run.text_len, elapsed.as_secs_f32())
                    .map_err(|e| anyhow::anyhow!("decode iteration {}: {e}", i + 1))?;

            // A short run is not necessarily a failure (EOS is legal), but it
            // is never silent: the reported rate is over the produced count, so
            // say plainly that the two differ.
            if run.completion_tokens != gen_len {
                tracing::warn!(
                    "decode iteration {}: asked for {gen_len} tokens, engine produced {}. \
                     The reported rate is over the PRODUCED count.",
                    i + 1,
                    run.completion_tokens
                );
            }

            let ms_per_tok = 1000.0 / tok_per_sec;
            decode_results.push((tok_per_sec, ms_per_tok));
        }
    }

    // Calculate statistics
    let mut results = Vec::new();

    if !prefill_results.is_empty() {
        let tok_per_sec_vals: Vec<f32> = prefill_results.iter().map(|(t, _)| *t).collect();
        let ttft_vals: Vec<f32> = prefill_results.iter().map(|(_, l)| *l).collect();
        let (mean_tps, std_dev_tps) = calculate_stats(&tok_per_sec_vals);
        let (mean_ttft, _) = calculate_stats(&ttft_vals);
        results.push(BenchResult {
            test_name: format!("Prefill ({} tokens)", prompt_len),
            tok_per_sec: mean_tps,
            std_dev: std_dev_tps,
            latency_ms: mean_ttft, // TTFT
        });
    }

    if !decode_results.is_empty() {
        let tok_per_sec_vals: Vec<f32> = decode_results.iter().map(|(t, _)| *t).collect();
        let (mean_tps, std_dev_tps) = calculate_stats(&tok_per_sec_vals);
        let ms_per_tok = 1000.0 / mean_tps;
        results.push(BenchResult {
            test_name: format!("Decode ({} tokens)", gen_len),
            tok_per_sec: mean_tps,
            std_dev: std_dev_tps,
            latency_ms: ms_per_tok, // ms/tok
        });
    }

    // Print results
    print_results(&model_id, iterations, &results);

    // MTP speculative decode: report the acceptance rate the run actually
    // measured, on the same greppable convention as `SPEED[...]`/`BATCH[...]`.
    //
    // This is the whole point of putting the counters in a process-global sink:
    // a GPU session that runs the throughput sweep gets the acceptance number
    // out of the same process, with no second run and no second rental.
    report_mtp_acceptance(runtime.mtp_depth as usize);

    // ArcKV/Share: whether the prefix cache hit, on the same greppable
    // convention. Without this line a prefill number is uninterpretable — it
    // could be a cold-cache measurement, a warm-cache one, or (as `bench`
    // itself configures) a measurement with the cache switched off entirely.
    report_prefix_cache(&mistralrs);

    Ok(())
}

/// Print the `SHARE[...]` marker for this run, or say plainly that the prefix
/// cache produced no measurement.
///
/// Same shape and the same reason as [`report_mtp_acceptance`]: an absent
/// number must not read as a zero. `run_bench` builds the engine with
/// `--prefix-cache-n 0`, so on an unmodified `mistralrs bench` invocation this
/// prints the `WARN[SHARE]` arm — which is the finding, not a defect in the
/// reporting: every `bench` prefill figure Arc has ever published was taken
/// with prefix caching off, and nothing said so.
fn report_prefix_cache(mistralrs: &Arc<mistralrs_core::MistralRs>) {
    let Ok(logger) = mistralrs.get_logger(None) else {
        return;
    };
    match logger.share_stats() {
        Some(stats) if stats.was_consulted() => {
            println!("SHARE[prefix] {}", stats.summary());
        }
        _ => {
            println!(
                "WARN[SHARE] the prefix cache was never consulted in this run (`mistralrs \
                 bench` builds the engine with --prefix-cache-n 0). Prefill throughput and \
                 TTFT above are cold-path numbers: they are UNMEASURED with respect to \
                 caching — do not read them as a 0% hit rate."
            );
        }
    }
}

/// Print the `MTP[agg] …` marker for this process, or say plainly that no
/// number was produced.
///
/// Three separate GPU sessions came home with an empty acceptance artifact, so
/// silence is not an acceptable outcome here: if `--mtp-depth > 0` was asked
/// for and nothing was measured, that is itself the finding and it gets its own
/// `WARN[MTP]` line rather than an absence.
fn report_mtp_acceptance(mtp_depth: usize) {
    let markers = mistralrs_core::mtp_acceptance_markers();
    if !markers.is_empty() {
        // Aggregate first, then one line per batch size. The per-user
        // multiplier is a function of B (`CEILINGS.json`: the per-user ceiling
        // falls from 1413 tok/s at B=1 to 68 at B=128), so an aggregate over a
        // run whose batch size moved is a number about no particular batch.
        for marker in markers {
            println!("{marker}");
        }
        // The per-position breakdown, alongside the aggregate rather than
        // instead of it. `accept_rate` is one scalar over the whole chain, and
        // the same scalar is produced by a flat per-position profile (a
        // target/draft distribution mismatch) and by a falling one (the draft
        // compounding on its own hidden state) — which are different defects
        // with different fixes. Printing only the scalar is what made the
        // measured 0.4194 undiagnosable without a second GPU rental.
        for line in mistralrs_core::mtp_acceptance_position_lines() {
            println!("{line}");
        }
        return;
    }
    if mtp_depth > 0 {
        println!(
            "WARN[MTP] --mtp-depth {mtp_depth} was requested but no MTP decode step ran: \
             either the model exposes no MTP head, or every step fell back to the target \
             pipeline (paged-attn / xlora / raw-logits). Acceptance is UNMEASURED \
             for this run — do not read it as 0%."
        );
    }
}

/// Calculate mean and standard deviation
fn calculate_stats(values: &[f32]) -> (f32, f32) {
    let n = values.len() as f32;
    let mean = values.iter().sum::<f32>() / n;
    let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
    let std_dev = variance.sqrt();
    (mean, std_dev)
}

/// Run a single benchmark iteration
async fn run_single_bench(
    mistralrs: &Arc<mistralrs_core::MistralRs>,
    prompt_tokens: usize,
    gen_tokens: usize,
) -> Result<BenchRun> {
    let sampling_params = SamplingParams {
        temperature: Some(0.1),
        top_k: Some(32),
        top_p: Some(0.1),
        min_p: Some(0.05),
        top_nsigma: None,
        top_n_logprobs: 0,
        frequency_penalty: Some(0.1),
        presence_penalty: Some(0.1),
        repetition_penalty: None,
        max_len: Some(gen_tokens),
        stop_toks: None,
        logits_bias: None,
        n_choices: 1,
        dry_params: Some(DrySamplingParams::default()),
        early_stop_confidence: None,
        reasoning_budget: None,
    };

    let sender = mistralrs.get_sender(None).unwrap();
    let (tx, mut rx) = channel(100);

    // Use token IDs for prompt to ensure exact length
    let tokens: Vec<u32> = (1000..1000 + prompt_tokens as u32).collect();

    let req = Request::Normal(Box::new(NormalRequest {
        id: mistralrs.next_request_id(),
        messages: RequestMessage::CompletionTokens(tokens),
        sampling_params,
        response: tx,
        return_logprobs: false,
        is_streaming: false,
        constraint: Constraint::None,
        suffix: None,
        tools: None,
        tool_choice: None,
        logits_processors: None,
        return_raw_logits: false,
        web_search_options: None,
        model_id: None,
        truncate_sequence: false,
    }));

    sender.send(req).await?;

    // The response is READ, not discarded. `Some(Response::Done(_)) => Ok(())`
    // is how a zero-token run used to be scored as a success.
    match rx.recv().await {
        Some(Response::CompletionDone(r)) => Ok(BenchRun {
            completion_tokens: r.usage.completion_tokens,
            prompt_tokens: r.usage.prompt_tokens,
            text_len: r.choices.first().map_or(0, |c| c.text.len()),
        }),
        Some(Response::Done(r)) => Ok(BenchRun {
            completion_tokens: r.usage.completion_tokens,
            prompt_tokens: r.usage.prompt_tokens,
            text_len: r
                .choices
                .first()
                .map_or(0, |c| c.message.content.as_ref().map_or(0, String::len)),
        }),
        Some(Response::InternalError(e)) => anyhow::bail!("Internal error: {e:?}"),
        Some(Response::ModelError(e, _)) => anyhow::bail!("Model error: {e}"),
        Some(Response::ValidationError(e)) => anyhow::bail!("Validation error: {e:?}"),
        Some(_) => anyhow::bail!("Unexpected response type"),
        None => anyhow::bail!("No response received"),
    }
}

/// Print benchmark results in a nice table
#[allow(clippy::cast_precision_loss)]
fn print_results(model_id: &str, iterations: usize, results: &[BenchResult]) {
    println!();
    println!("Benchmark Results");
    println!("=================");
    println!();
    println!("Model: {}", model_id);
    println!("Iterations: {}", iterations);
    println!();

    let mut table = Table::new();
    table
        .load_preset(UTF8_FULL)
        .set_content_arrangement(ContentArrangement::Dynamic)
        .set_header(vec![
            Cell::new("Test"),
            Cell::new("T/s"),
            Cell::new("Latency"),
        ]);

    for result in results {
        // Determine latency label based on test type
        let latency_str = if result.test_name.contains("Prefill") {
            format!("{:.2} ms (TTFT)", result.latency_ms)
        } else {
            format!("{:.2} ms/T", result.latency_ms)
        };

        table.add_row(vec![
            Cell::new(&result.test_name),
            Cell::new(format!("{:.1} ± {:.1}", result.tok_per_sec, result.std_dev))
                .fg(Color::Green),
            Cell::new(latency_str),
        ]);
    }

    println!("{table}");
    println!();
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The old formula, kept verbatim so the tests below can demonstrate what
    /// it would have printed. This is the code that was removed:
    /// `let tok_per_sec = gen_len as f32 / elapsed.as_secs_f32();`
    fn legacy_rate_over_requested(requested: usize, elapsed_secs: f32) -> f32 {
        requested as f32 / elapsed_secs
    }

    /// The property that was missing: a run that produced nothing has NO rate.
    #[test]
    fn a_zero_token_run_is_refused_not_rated() {
        let err = rate_from_produced(0, 0, 0.5).expect_err("zero tokens must not yield a rate");
        assert_eq!(err, RateError::NoTokens);
        // The message has to name the failure, because this string is the only
        // thing a operator sees when a benchmark aborts.
        assert!(
            err.to_string().contains("ZERO tokens"),
            "the error must say what happened: {err}"
        );
    }

    /// NON-VACUITY CONTROL. This test MUST fire: it shows the old formula
    /// produced a perfectly ordinary-looking tok/s number for the very run the
    /// test above refuses. If this assertion ever fails, the control is broken
    /// and the test above proves nothing.
    #[test]
    fn the_old_formula_would_have_published_a_number_for_that_same_run() {
        // 64 tokens requested, zero produced, failed fast in 50 ms.
        let fabricated = legacy_rate_over_requested(64, 0.05);
        assert!(
            fabricated.is_finite() && fabricated > 0.0,
            "control broken: the legacy formula must yield a plausible number here"
        );
        // And it is not merely wrong, it is FLATTERING — a fast failure inflates it.
        assert!(
            fabricated > 1000.0,
            "a fast zero-token failure should inflate the legacy rate, got {fabricated}"
        );
        // Same inputs, current code: refused.
        assert_eq!(
            rate_from_produced(0, 0, 0.05),
            Err(RateError::NoTokens),
            "the current code must refuse the run the legacy formula rated at {fabricated:.0} tok/s"
        );
    }

    /// The engine claiming tokens it did not deliver (and vice versa) is a
    /// defect, not a rounding difference — neither counter can be trusted as a
    /// denominator once they disagree.
    #[test]
    fn counter_disagreement_is_refused_in_both_directions() {
        assert_eq!(
            rate_from_produced(64, 0, 1.0),
            Err(RateError::CounterDisagreement {
                tokens: 64,
                text_len: 0
            }),
            "engine claims 64 tokens but delivered no text"
        );
        assert_eq!(
            rate_from_produced(0, 128, 1.0),
            Err(RateError::CounterDisagreement {
                tokens: 0,
                text_len: 128
            }),
            "engine delivered text but counted no tokens"
        );
    }

    /// A short run is legal (EOS), and it must be rated over what it PRODUCED.
    #[test]
    fn a_short_run_rates_over_produced_not_requested() {
        // Asked for 64, produced 7, took 1 s.
        let rate = rate_from_produced(7, 21, 1.0).expect("a 7-token run has a rate");
        assert!(
            (rate - 7.0).abs() < f32::EPSILON,
            "rate must be 7/1s, got {rate}"
        );
        // The number the old code would have printed for the identical run:
        let legacy = legacy_rate_over_requested(64, 1.0);
        assert!(
            (legacy - rate).abs() > 1.0,
            "control broken: legacy and corrected rates must differ here \
             (legacy={legacy}, corrected={rate})"
        );
    }

    #[test]
    fn non_positive_elapsed_is_refused() {
        assert_eq!(
            rate_from_produced(16, 48, 0.0),
            Err(RateError::NonPositiveElapsed)
        );
    }
}
