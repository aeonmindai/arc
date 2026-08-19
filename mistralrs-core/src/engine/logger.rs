#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use tracing::info;

use crate::kv_sharing::ShareStats;

/// The `ArcKV/Share` counters, as last published by the engine.
///
/// `None` means the prefix cache has not been consulted **once** in this
/// process — which is what `--prefix-cache-n 0` and every `mistralrs bench`
/// run before this change looked like from the outside, and is not the same
/// statement as a 0% hit rate. Consumers must render the two differently; see
/// [`ShareStats::summary`].
type SharedShareStats = Arc<Mutex<Option<ShareStats>>>;

pub struct IntervalLogger {
    enable_logging: Arc<AtomicBool>,
    prefix_cache_hits: Arc<AtomicUsize>,
    tokens_processed: Arc<AtomicUsize>,
    total_new_seqs: Arc<AtomicUsize>,
    num_running: Arc<AtomicUsize>,
    num_waiting: Arc<AtomicUsize>,
    encoder_cache_hits: Option<Arc<AtomicUsize>>,
    encoder_cache_misses: Option<Arc<AtomicUsize>>,
    /// 🔑 The whole point of this field: `PrefixCacheManagerV2` has computed a
    /// token-weighted hit rate, tokens-not-recomputed, bytes saved and a
    /// cross-prefix reuse meter since it was written, and **nothing outside
    /// `prefix_cacher.rs` ever read them**. Every Arc prefill measurement was
    /// therefore uninterpretable: it could not say whether the cache hit.
    share_stats: SharedShareStats,
}

impl IntervalLogger {
    /// Starts an interval logger. Call `begin_logging` to begin the logging process.
    pub fn new(
        interval: Duration,
        encoder_cache_counters: Option<(Arc<AtomicUsize>, Arc<AtomicUsize>)>,
    ) -> Self {
        let prefix_cache_hits = Arc::new(AtomicUsize::new(0));
        let tokens_processed = Arc::new(AtomicUsize::new(0));
        let total_new_seqs = Arc::new(AtomicUsize::new(0));
        let enable_logging = Arc::new(AtomicBool::new(false));
        let num_running = Arc::new(AtomicUsize::new(0));
        let num_waiting = Arc::new(AtomicUsize::new(0));

        let t_prefix_cache_hits = prefix_cache_hits.clone();
        let t_tokens_processed = tokens_processed.clone();
        let t_total_new_seqs = total_new_seqs.clone();
        let t_enable_logging = enable_logging.clone();
        let t_num_running = num_running.clone();
        let t_num_waiting = num_waiting.clone();
        let (encoder_cache_hits, encoder_cache_misses) = match encoder_cache_counters {
            Some((h, m)) => (Some(h), Some(m)),
            None => (None, None),
        };
        let t_enc_hits = encoder_cache_hits.clone();
        let t_enc_misses = encoder_cache_misses.clone();
        let share_stats: SharedShareStats = Arc::new(Mutex::new(None));
        let t_share_stats = share_stats.clone();
        thread::spawn(move || {
            // Cumulative GPU-sampler counters as of the previous tick, so we
            // can report this window's delta rather than a since-boot total.
            let mut prev_sampler = (0u64, 0u64, 0u64);
            // Start the actual logging
            loop {
                thread::sleep(interval);
                if !t_enable_logging.load(Ordering::Relaxed) {
                    continue;
                }

                let total_new_seqs = t_total_new_seqs.load(Ordering::Relaxed);
                let prefix_cache_hits = t_prefix_cache_hits.load(Ordering::Relaxed);
                let tokens_processed = t_tokens_processed.swap(0, Ordering::Relaxed);
                let num_running = t_num_running.load(Ordering::Relaxed);
                let num_waiting = t_num_waiting.load(Ordering::Relaxed);

                // GPU sampler health for this window. A CPU fallback costs a
                // full-logits D2H + full-vocab sort per sequence per step, so
                // it belongs next to the throughput number it explains —
                // previously it was only a per-token WARN and was missed.
                let sampler_now = crate::sampler::gpu_sampling_health::stats();
                let sampler_info = {
                    let d_ok = sampler_now.0.saturating_sub(prev_sampler.0);
                    let d_declined = sampler_now.1.saturating_sub(prev_sampler.1);
                    let d_failed = sampler_now.2.saturating_sub(prev_sampler.2);
                    prev_sampler = sampler_now;
                    if d_failed > 0 {
                        format!(
                            ", SAMPLER ON CPU SLOW PATH: {d_failed} GPU top-k failures this \
                             interval (vs {d_ok} on GPU)"
                        )
                    } else if d_declined > 0 && d_ok == 0 {
                        format!(", sampler CPU path: {d_declined} GPU top-k declined (by config)")
                    } else {
                        String::new()
                    }
                };

                if total_new_seqs != 0 && tokens_processed != 0 {
                    let enc_cache_info =
                        if let (Some(ref hits), Some(ref misses)) = (&t_enc_hits, &t_enc_misses) {
                            let h = hits.load(Ordering::Relaxed);
                            let m = misses.load(Ordering::Relaxed);
                            let total = h + m;
                            if total > 0 {
                                format!(
                                    ", Encoder cache hitrate {:.2}%",
                                    100. * h as f64 / total as f64
                                )
                            } else {
                                String::new()
                            }
                        } else {
                            String::new()
                        };

                    // Throughput = tokens processed during this interval / interval duration.
                    // Combines both prefill and decode tokens. The counter is atomically
                    // swapped to 0 each interval, so the metric reflects only the current
                    // window and is not cumulative.
                    info!(
                        "Throughput (T/s) {:.2}, Prefix cache hitrate {:.2}%{enc_cache_info}, {num_running} running, {num_waiting} waiting{sampler_info}",
                        tokens_processed as f64 / interval.as_secs_f64(),
                        100. * prefix_cache_hits as f64 / total_new_seqs as f64,
                    );

                    // The request-level rate above answers "did some prefix
                    // match"; this answers "how much prefill did that avoid",
                    // which is the number a prefill measurement needs beside it.
                    let share = t_share_stats.lock().ok().and_then(|s| *s);
                    match share {
                        Some(s) => info!("ArcKV/Share: {}", s.summary()),
                        None => info!(
                            "ArcKV/Share: prefix cache not consulted this process — prefill \
                             numbers from this run say nothing about caching"
                        ),
                    }
                }
            }
        });

        Self {
            prefix_cache_hits,
            tokens_processed,
            total_new_seqs,
            enable_logging,
            num_running,
            num_waiting,
            encoder_cache_hits,
            encoder_cache_misses,
            share_stats,
        }
    }

    pub fn enable_logging(&self) {
        self.enable_logging.store(true, Ordering::Relaxed);
    }

    /// Reset all counters to zero. Call after warmup/dummy runs to get clean stats.
    ///
    /// Deliberately does **not** clear the `ArcKV/Share` mirror: it is a view of
    /// the radix tree's own process-lifetime counters, which this type does not
    /// own and cannot zero. Blanking the mirror would report "never consulted"
    /// about a cache that had been, which is the one confusion the mirror
    /// exists to remove.
    pub fn reset(&self) {
        self.prefix_cache_hits.store(0, Ordering::Relaxed);
        self.tokens_processed.store(0, Ordering::Relaxed);
        self.total_new_seqs.store(0, Ordering::Relaxed);
        self.num_running.store(0, Ordering::Relaxed);
        self.num_waiting.store(0, Ordering::Relaxed);
        if let Some(ref hits) = self.encoder_cache_hits {
            hits.store(0, Ordering::Relaxed);
        }
        if let Some(ref misses) = self.encoder_cache_misses {
            misses.store(0, Ordering::Relaxed);
        }
        crate::sampler::gpu_sampling_health::reset();
    }

    pub fn add_tokens_processed(&self, num_tokens: usize) {
        self.tokens_processed
            .fetch_add(num_tokens, Ordering::Relaxed);
    }

    pub fn add_new_sequence(&self) {
        self.total_new_seqs.fetch_add(1, Ordering::Relaxed);
    }

    pub fn add_prefix_cache_hit(&self) {
        self.prefix_cache_hits.fetch_add(1, Ordering::Relaxed);
    }

    pub fn set_num_running(&self, running: usize) {
        self.num_running.store(running, Ordering::Relaxed);
    }

    pub fn set_num_waiting(&self, waiting: usize) {
        self.num_waiting.store(waiting, Ordering::Relaxed);
    }

    /// Return cumulative prefix cache (hits, total_sequences).
    pub fn prefix_cache_stats(&self) -> (usize, usize) {
        (
            self.prefix_cache_hits.load(Ordering::Relaxed),
            self.total_new_seqs.load(Ordering::Relaxed),
        )
    }

    /// Publish the `ArcKV/Share` counters. Called by the engine wherever it
    /// already holds the prefix cache, so the snapshot is consistent with the
    /// lookup that produced it.
    pub fn set_share_stats(&self, stats: ShareStats) {
        if let Ok(mut slot) = self.share_stats.lock() {
            *slot = Some(stats);
        }
    }

    /// The last published `ArcKV/Share` counters, or `None` if the prefix cache
    /// has not been consulted at all in this process.
    pub fn share_stats(&self) -> Option<ShareStats> {
        self.share_stats.lock().ok().and_then(|s| *s)
    }

    /// The line `bench`, `serve` and interactive all print, so they cannot
    /// disagree about what "the cache hit" means.
    ///
    /// The `None` case is deliberately not rendered as 0%: a run that never
    /// consulted the cache (`--prefix-cache-n 0`, which is what `mistralrs
    /// bench` sets) has made no measurement of it at all.
    pub fn share_stats_line(&self) -> String {
        match self.share_stats() {
            Some(s) => s.summary(),
            None => "prefix cache not consulted (disabled, or no request reached it) — no \
                     prefill number from this run measures caching"
                .to_string(),
        }
    }

    /// Return cumulative encoder cache (hits, misses), or `None` if no encoder cache exists.
    pub fn encoder_cache_stats(&self) -> Option<(usize, usize)> {
        match (&self.encoder_cache_hits, &self.encoder_cache_misses) {
            (Some(h), Some(m)) => Some((h.load(Ordering::Relaxed), m.load(Ordering::Relaxed))),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_sharing::{KvBlockLayout, SharedPrefixCache};

    /// A long interval so the background thread never ticks during the test.
    fn quiet_logger() -> IntervalLogger {
        IntervalLogger::new(Duration::from_secs(3600), None)
    }

    /// The failure this whole change is about: before it, every consumer read
    /// `prefix_cache_stats()` — a request counter — and there was no way to ask
    /// whether the cache had been consulted at all. A fresh logger must say
    /// "not consulted", not "0%".
    #[test]
    fn a_logger_that_was_never_published_to_reports_not_consulted() {
        let logger = quiet_logger();
        assert!(logger.share_stats().is_none());
        let line = logger.share_stats_line();
        assert!(
            line.contains("not consulted"),
            "an unpublished logger must not imply a measured miss; got {line:?}"
        );
    }

    /// End to end through the real tree: counters move, the engine publishes,
    /// the CLI-facing line carries the measured token-weighted rate.
    #[test]
    fn a_published_snapshot_carries_the_measured_token_hit_rate() {
        let mut cache: SharedPrefixCache<()> = SharedPrefixCache::new();
        let prefix: Vec<u32> = (0..40).collect();
        cache.insert(&prefix, (), KvBlockLayout::default());
        let mut query = prefix.clone();
        query.extend(40..50);
        assert!(cache.lookup(&query, |_| true).is_some());

        let logger = quiet_logger();
        logger.set_share_stats(cache.stats());

        let stats = logger.share_stats().expect("published");
        assert_eq!(stats.tokens_not_recomputed(), 40);
        let line = logger.share_stats_line();
        assert!(
            line.contains("80.00% of prompt tokens served from cache (40/50)"),
            "the line must carry the tree's own measured rate; got {line:?}"
        );
        assert!(!line.contains("not consulted"), "got {line:?}");
    }

    /// `reset()` clears the request counters it owns and must NOT blank the
    /// share mirror, whose counters live in the tree and are not resettable
    /// from here — blanking it would report "never consulted" about a cache
    /// that had been.
    #[test]
    fn reset_clears_request_counters_but_not_the_share_mirror() {
        let mut cache: SharedPrefixCache<()> = SharedPrefixCache::new();
        cache.insert(&[1, 2, 3], (), KvBlockLayout::default());
        assert!(cache.lookup(&[1, 2, 3, 4], |_| true).is_some());

        let logger = quiet_logger();
        logger.add_new_sequence();
        logger.add_prefix_cache_hit();
        logger.set_share_stats(cache.stats());

        logger.reset();
        assert_eq!(logger.prefix_cache_stats(), (0, 0));
        assert!(
            logger.share_stats().is_some(),
            "the share mirror must survive a counter reset"
        );
    }
}
