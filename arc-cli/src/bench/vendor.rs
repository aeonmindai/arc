//! Vendor abstraction for the bench harness.
//!
//! A "vendor" is anything that can answer "run this request, give me back
//! a TTFT + output token speed measurement". For the offline / CI path we
//! provide [`MockVendor`] which generates synthetic responses with a
//! tunable degradation curve. For the on-GPU path (`--features cuda`) we
//! provide [`InProcessVendor`] which drives `mistralrs-core`'s `MistralRs`
//! engine *in process* (NOT a subshell — that's the post-RUN-191 pattern,
//! same as `arc validate`).
//!
//! Both implementations expose the same async trait so the scheduler can
//! be developed and tested without a GPU.

use anyhow::Result;
use async_trait::async_trait;
use std::sync::Arc;
use std::time::Duration;

/// One request's measurement. The scheduler collects these from the
/// **steady-state** window of each phase, computes P25 / P95, and feeds
/// the result to the SLO comparator.
#[derive(Debug, Clone, Copy)]
pub struct RequestResult {
    /// Wall-clock time from request submission to first token. Always > 0.
    pub ttft: Duration,
    /// End-to-end wall-clock duration including TTFT.
    pub total: Duration,
    /// Output tokens generated (does NOT include prompt tokens).
    pub output_tokens: u32,
    /// True if the request finished cleanly. Failures are still recorded
    /// so the scheduler can compute an error rate; we treat them as
    /// non-passing for SLO purposes by reporting infinite TTFT.
    pub ok: bool,
}

impl RequestResult {
    /// Output token speed in tokens/second. Returns 0 for zero-token / failed
    /// responses to avoid `NaN` polluting the percentile calculation.
    pub fn output_speed_tok_per_s(&self) -> f64 {
        if !self.ok || self.output_tokens == 0 {
            return 0.0;
        }
        let body = self.total.saturating_sub(self.ttft);
        let secs = body.as_secs_f64();
        if secs <= 0.0 {
            return 0.0;
        }
        self.output_tokens as f64 / secs
    }

    /// TTFT in seconds, with failed requests reported as +∞ so they always
    /// blow the P95 cap. This is the convention AA-AgentPerf uses.
    pub fn ttft_seconds_for_slo(&self) -> f64 {
        if !self.ok {
            return f64::INFINITY;
        }
        self.ttft.as_secs_f64()
    }
}

/// The contract a vendor implementation must satisfy. The harness drives
/// many concurrent in-flight requests through this trait and only requires
/// the per-request result.
#[async_trait]
pub trait Vendor: Send + Sync {
    /// Display name for the TUI ("arc (this build)", "mock", etc).
    fn name(&self) -> &str;
    /// Run one request with the given prompt and the configured
    /// `max_tokens` budget. The vendor is responsible for whatever fan-out
    /// it needs internally; the harness uses `tokio::spawn` + `join_all`
    /// across the trait to parallelise.
    async fn run_request(&self, prompt: &str, max_tokens: u32) -> Result<RequestResult>;
    /// Hook fired by the scheduler at the start of each phase so vendors
    /// with synthetic degradation curves know the current load. Default
    /// impl is a no-op; the real in-process vendor doesn't need this.
    fn set_concurrent_users(&self, _k: u32) {}
}

// --------------------------------------------------------------------------
// MockVendor — the test path. Synthesises results without touching a GPU.
// --------------------------------------------------------------------------

/// Knobs for the synthetic degradation curve.
///
/// The TTFT is sampled uniformly in [ttft_min, ttft_max] **scaled** by a
/// `concurrent_users / users_for_2x_ttft` factor so the latency curve
/// degrades when the harness ramps up. The output token speed follows
/// `base_speed / (1 + concurrent_users / users_for_2x_slowdown)` which is
/// a classic queueing-flat-region → 1/n degradation shape.
#[derive(Debug, Clone, Copy)]
pub struct MockVendorConfig {
    pub base_output_speed_tok_per_s: f64,
    pub ttft_min_ms: u64,
    pub ttft_max_ms: u64,
    pub users_for_2x_ttft: f64,
    pub users_for_2x_slowdown: f64,
}

impl Default for MockVendorConfig {
    fn default() -> Self {
        Self {
            // 120 tok/s at K=0 lets tier 1 (≥100 tok/s) pass for small K,
            // tier 2 (≥60 tok/s) fail somewhere around K≈100.
            base_output_speed_tok_per_s: 120.0,
            ttft_min_ms: 200,
            ttft_max_ms: 1200,
            users_for_2x_ttft: 64.0,
            users_for_2x_slowdown: 100.0,
        }
    }
}

/// Output-speed degradation function exposed so unit tests can probe it
/// without spinning a runtime.
pub fn mock_output_speed(cfg: &MockVendorConfig, concurrent_users: u32) -> f64 {
    let k = concurrent_users as f64;
    cfg.base_output_speed_tok_per_s / (1.0 + k / cfg.users_for_2x_slowdown)
}

/// TTFT-degradation function (returns the *scaled* mean of the uniform
/// interval, not a sample). Exposed for unit tests.
#[allow(dead_code)]
pub fn mock_ttft_mean_seconds(cfg: &MockVendorConfig, concurrent_users: u32) -> f64 {
    let k = concurrent_users as f64;
    let scale = 1.0 + k / cfg.users_for_2x_ttft;
    let mid_ms = (cfg.ttft_min_ms + cfg.ttft_max_ms) as f64 / 2.0;
    (mid_ms * scale) / 1000.0
}

pub struct MockVendor {
    cfg: MockVendorConfig,
    current_users: Arc<std::sync::atomic::AtomicU32>,
    seed: Arc<std::sync::Mutex<u64>>,
}

impl MockVendor {
    pub fn new(cfg: MockVendorConfig) -> Self {
        Self {
            cfg,
            current_users: Arc::new(std::sync::atomic::AtomicU32::new(0)),
            seed: Arc::new(std::sync::Mutex::new(0xDEAD_BEEFu64)),
        }
    }

    /// Tiny xorshift PRNG. We don't need cryptographic randomness, only
    /// determinism + good-enough mixing for unit tests.
    fn next_u32(&self) -> u32 {
        let mut s = self.seed.lock().unwrap();
        let mut x = *s;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *s = x;
        x as u32
    }
}

#[async_trait]
impl Vendor for MockVendor {
    fn name(&self) -> &str {
        "mock"
    }

    fn set_concurrent_users(&self, k: u32) {
        self.current_users
            .store(k, std::sync::atomic::Ordering::SeqCst);
    }

    async fn run_request(&self, _prompt: &str, max_tokens: u32) -> Result<RequestResult> {
        let k = self
            .current_users
            .load(std::sync::atomic::Ordering::SeqCst)
            .max(1);

        // Sample TTFT uniformly in the configured range, scaled.
        let span_ms = self
            .cfg
            .ttft_max_ms
            .saturating_sub(self.cfg.ttft_min_ms)
            .max(1);
        let raw_ms = self.cfg.ttft_min_ms + (self.next_u32() as u64 % span_ms);
        let scale = 1.0 + k as f64 / self.cfg.users_for_2x_ttft;
        let ttft_ms = (raw_ms as f64 * scale) as u64;
        let ttft = Duration::from_millis(ttft_ms.max(1));

        // Output speed degrades with K; total body time = tokens / speed.
        let speed = mock_output_speed(&self.cfg, k);
        let body_secs = max_tokens as f64 / speed.max(1.0);
        let total = ttft + Duration::from_secs_f64(body_secs);

        // Don't actually sleep — that would multiply wall-clock time by
        // the number of requests per phase. The mock path is meant to
        // exercise the *scheduling* code path, not real latency.
        // Phase wall-clock pacing is handled by the scheduler's tokio sleeps.
        tokio::task::yield_now().await;

        Ok(RequestResult {
            ttft,
            total,
            output_tokens: max_tokens,
            ok: true,
        })
    }
}

// --------------------------------------------------------------------------
// InProcessVendor — real-GPU path. Drives `mistralrs-core` directly.
// --------------------------------------------------------------------------

/// Real-vendor on-GPU implementation. Only compiles with `--features cuda`
/// because it pulls in `mistralrs-core` types that themselves require a
/// CUDA build (CUDA / cuDNN / flash-attn — same gating as `arc validate`'s
/// `run_real_gpu`).
#[cfg(feature = "cuda")]
pub mod in_process {
    use super::*;
    use anyhow::Context;
    use std::sync::Arc;
    use std::time::Instant;

    /// In-process vendor that submits OpenAI-style chat requests through
    /// `MistralRs` and times them from the result-channel perspective.
    ///
    /// Construction is async because `MistralRsForServerBuilder::build()` is
    /// async and blocks on the model load. We deliberately keep the
    /// `Arc<MistralRs>` alive for the lifetime of the vendor so the engine
    /// stays warm across phases.
    pub struct InProcessVendor {
        runner: Arc<mistralrs_core::MistralRs>,
        name: String,
    }

    impl InProcessVendor {
        pub async fn new(
            model_id: &str,
            isq: Option<String>,
        ) -> Result<Self> {
            use candle_core::Device;
            use mistralrs_core::{AutoDeviceMapParams, ModelDType, ModelSelected, TokenSource};
            use mistralrs_server_core::mistralrs_for_server_builder::MistralRsForServerBuilder;

            let device = Device::new_cuda(0).context("failed to initialise CUDA device 0")?;

            let model_selected = ModelSelected::Plain {
                model_id: model_id.to_string(),
                tokenizer_json: None,
                arch: None,
                dtype: ModelDType::Auto,
                topology: None,
                organization: None,
                write_uqff: None,
                from_uqff: None,
                imatrix: None,
                calibration_file: None,
                max_seq_len: AutoDeviceMapParams::DEFAULT_MAX_SEQ_LEN,
                max_batch_size: AutoDeviceMapParams::DEFAULT_MAX_BATCH_SIZE,
                hf_cache_path: None,
                matformer_config_path: None,
                matformer_slice_name: None,
            };

            let runner = MistralRsForServerBuilder::new()
                .with_model(model_selected)
                .with_device(device)
                .with_interactive_mode(false)
                .with_token_source(TokenSource::CacheToken)
                .with_in_situ_quant_optional(isq)
                .build()
                .await
                .context("failed to load model for bench")?;

            Ok(Self {
                runner,
                name: format!("arc:{}", model_id),
            })
        }
    }

    #[async_trait]
    impl Vendor for InProcessVendor {
        fn name(&self) -> &str {
            &self.name
        }

        async fn run_request(&self, prompt: &str, max_tokens: u32) -> Result<RequestResult> {
            use either::Either;
            use indexmap::IndexMap;
            use mistralrs_core::{
                Constraint, MessageContent, NormalRequest, Request, RequestMessage, Response,
                SamplingParams,
            };

            // Build a single-turn user message. MessageContent is
            // `Either<String, Vec<IndexMap<String, Value>>>` so we use
            // `Either::Left(prompt.to_string())` for the plain-text variant.
            let mut msg: IndexMap<String, MessageContent> = IndexMap::new();
            msg.insert("role".to_string(), Either::Left("user".to_string()));
            msg.insert("content".to_string(), Either::Left(prompt.to_string()));
            let messages = vec![msg];

            // Sampling tuned for determinism so each phase's measurement
            // isn't dominated by random reroll latency. We use top_k=1
            // (greedy) and no penalties; the SLO is about throughput, not
            // generation quality.
            let sampling_params = SamplingParams {
                temperature: Some(0.0),
                top_k: Some(1),
                top_p: None,
                min_p: None,
                top_n_logprobs: 0,
                frequency_penalty: None,
                presence_penalty: None,
                repetition_penalty: None,
                stop_toks: None,
                max_len: Some(max_tokens as usize),
                logits_bias: None,
                n_choices: 1,
                dry_params: None,
            };

            let (tx, mut rx) = tokio::sync::mpsc::channel::<Response>(32);
            let req = Request::Normal(Box::new(NormalRequest {
                messages: RequestMessage::Chat {
                    messages,
                    enable_thinking: None,
                    reasoning_effort: None,
                },
                sampling_params,
                response: tx,
                return_logprobs: false,
                is_streaming: true,
                id: 0,
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

            let started = Instant::now();
            self.runner
                .get_sender(None)
                .map_err(|e| anyhow::anyhow!("no sender: {e}"))?
                .send(req)
                .await
                .context("failed to send request to runner")?;

            let mut ttft: Option<Duration> = None;
            let mut output_tokens: u32 = 0;
            let mut ok = true;

            while let Some(resp) = rx.recv().await {
                match resp {
                    Response::Chunk(chunk) => {
                        if ttft.is_none() {
                            ttft = Some(started.elapsed());
                        }
                        // Each chunk carries 1+ tokens worth of delta text;
                        // we count chunks-with-non-empty-content as a proxy
                        // because the streaming API doesn't expose a per-
                        // token boundary directly. The runner's `usage` on
                        // the final `Done` packet (handled below) is the
                        // authoritative count; this chunk-count is a
                        // fallback for runners that don't fill `usage`.
                        output_tokens += chunk
                            .choices
                            .iter()
                            .map(|c| {
                                !c.delta.content.as_deref().unwrap_or("").is_empty() as u32
                            })
                            .sum::<u32>();
                    }
                    Response::Done(done) => {
                        if ttft.is_none() {
                            ttft = Some(started.elapsed());
                        }
                        if done.usage.completion_tokens > 0 {
                            output_tokens = done.usage.completion_tokens as u32;
                        }
                        break;
                    }
                    Response::ModelError(_, _)
                    | Response::ValidationError(_)
                    | Response::InternalError(_) => {
                        ok = false;
                        break;
                    }
                    // Non-chat variants (image / speech / embeddings / raw)
                    // shouldn't appear for a chat request; ignore if they do.
                    _ => {}
                }
            }

            let total = started.elapsed();
            let ttft = ttft.unwrap_or(total);

            Ok(RequestResult {
                ttft,
                total,
                output_tokens,
                ok,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mock_output_speed_monotone_decreasing() {
        let cfg = MockVendorConfig::default();
        let s0 = mock_output_speed(&cfg, 1);
        let s50 = mock_output_speed(&cfg, 50);
        let s200 = mock_output_speed(&cfg, 200);
        assert!(s0 > s50, "{s0} should be > {s50}");
        assert!(s50 > s200, "{s50} should be > {s200}");
    }

    #[test]
    fn mock_output_speed_2x_users_2x_slowdown() {
        let cfg = MockVendorConfig {
            users_for_2x_slowdown: 100.0,
            ..MockVendorConfig::default()
        };
        // At K = users_for_2x_slowdown, speed should be ≈ base / 2.
        let base = cfg.base_output_speed_tok_per_s;
        let observed = mock_output_speed(&cfg, 100);
        let half = base / 2.0;
        assert!(
            (observed - half).abs() / half < 0.01,
            "expected ~{half}, got {observed}"
        );
    }

    #[test]
    fn mock_ttft_mean_increases_with_users() {
        let cfg = MockVendorConfig::default();
        let t0 = mock_ttft_mean_seconds(&cfg, 1);
        let t256 = mock_ttft_mean_seconds(&cfg, 256);
        assert!(t256 > t0);
    }

    #[test]
    fn request_result_output_speed_failed_returns_zero() {
        let r = RequestResult {
            ttft: Duration::from_millis(100),
            total: Duration::from_secs(1),
            output_tokens: 50,
            ok: false,
        };
        assert_eq!(r.output_speed_tok_per_s(), 0.0);
        assert!(r.ttft_seconds_for_slo().is_infinite());
    }

    #[test]
    fn request_result_output_speed_normal_case() {
        // 100 output tokens in (1000 - 100) ms = 900ms = 0.9s → 111.11 tok/s.
        let r = RequestResult {
            ttft: Duration::from_millis(100),
            total: Duration::from_millis(1000),
            output_tokens: 100,
            ok: true,
        };
        let s = r.output_speed_tok_per_s();
        assert!((s - 111.111).abs() < 0.1, "expected ~111.11, got {s}");
    }

    #[tokio::test]
    async fn mock_vendor_returns_plausible_result() {
        let v = MockVendor::new(MockVendorConfig::default());
        v.set_concurrent_users(10);
        let r = v.run_request("hello", 64).await.unwrap();
        assert!(r.ok);
        assert_eq!(r.output_tokens, 64);
        assert!(r.ttft.as_millis() > 0);
        assert!(r.total >= r.ttft);
        let speed = r.output_speed_tok_per_s();
        // At K=10, with defaults, speed should still be roughly within
        // a factor of 2 of base (120). We bound loosely to allow PRNG.
        assert!(
            speed > 30.0 && speed < 200.0,
            "expected 30-200 tok/s at K=10, got {speed}"
        );
    }
}
