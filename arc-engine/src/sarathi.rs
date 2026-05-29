//! Sarathi-Serve chunked prefill scheduler.
//!
//! Paper: `research/14_serving_systems/sarathi_serve_chunked_prefill.pdf`
//! Repo:  `research/code/04_aggregate_throughput/sarathi_serve/sarathi/core/scheduler/`
//!
//! ## What
//!
//! Standard serving runs an entire prefill in one kernel call. For a 1M-token
//! prompt this blocks all decoding for hundreds of milliseconds — terrible
//! tail latency. Sarathi-Serve chunks prefills into small pieces (default 512
//! tokens) and interleaves them with ongoing decode steps.
//!
//! Two-Batch Overlap (TBO): launch the prefill chunk kernel and the decode
//! kernel on separate CUDA streams so they fill different SMs concurrently.
//!
//! ## Tier A scope
//!
//! - `ChunkedPrefillScheduler`: maintains queues of prefill chunks + decode steps
//! - `next_batch()`: picks the next mix of prefill chunks + decode steps,
//!   respecting a token budget per step
//! - Tests: split correctness on long prompts, interleaving rule, token budget
//!
//! ## Tier B (deferred)
//!
//! - CUDA-stream-level TBO orchestration
//! - Wire into `mistralrs-core/src/scheduler/`
//! - End-to-end mixed-workload throughput measurements

use std::collections::VecDeque;

#[derive(Debug, Clone)]
pub struct PrefillRequest {
    pub user_id: u64,
    /// Number of prompt tokens still to process.
    pub remaining_tokens: usize,
    /// How many tokens have already been processed in earlier chunks.
    pub tokens_processed: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefillChunk {
    pub user_id: u64,
    pub start: usize,
    pub end: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct ChunkConfig {
    /// Maximum prefill tokens per chunk. Default 512.
    pub chunk_size: usize,
    /// Maximum total tokens per scheduling step (prefill chunks + decode steps).
    /// Default 4096.
    pub token_budget: usize,
}

impl Default for ChunkConfig {
    fn default() -> Self {
        Self {
            chunk_size: 512,
            token_budget: 4096,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ChunkedPrefillScheduler {
    prefill_queue: VecDeque<PrefillRequest>,
    decode_users: Vec<u64>,
    cfg: ChunkConfig,
}

impl ChunkedPrefillScheduler {
    pub fn new(cfg: ChunkConfig) -> Self {
        Self {
            prefill_queue: VecDeque::new(),
            decode_users: Vec::new(),
            cfg,
        }
    }

    pub fn enqueue_prefill(&mut self, user_id: u64, prompt_tokens: usize) {
        self.prefill_queue.push_back(PrefillRequest {
            user_id,
            remaining_tokens: prompt_tokens,
            tokens_processed: 0,
        });
    }

    pub fn register_decode_user(&mut self, user_id: u64) {
        if !self.decode_users.contains(&user_id) {
            self.decode_users.push(user_id);
        }
    }

    pub fn unregister_decode_user(&mut self, user_id: u64) {
        self.decode_users.retain(|&u| u != user_id);
    }

    /// Number of pending prefill requests.
    pub fn pending_prefills(&self) -> usize {
        self.prefill_queue.len()
    }

    /// Active decode users.
    pub fn active_decoders(&self) -> usize {
        self.decode_users.len()
    }

    /// Build the next scheduling batch. Returns:
    /// - decode users to process this step (1 token each)
    /// - prefill chunks to process this step
    ///
    /// Decode steps are scheduled first (they're latency-sensitive). The remaining
    /// token budget goes to prefill chunks, oldest first.
    pub fn next_batch(&mut self) -> (Vec<u64>, Vec<PrefillChunk>) {
        let mut chunks = Vec::new();
        let decode_users = self.decode_users.clone();
        let decode_tokens = decode_users.len(); // 1 token per decoder
        let mut budget = self.cfg.token_budget.saturating_sub(decode_tokens);

        while budget >= 1 && !self.prefill_queue.is_empty() {
            let mut head = self.prefill_queue.pop_front().unwrap();
            let take = head.remaining_tokens.min(self.cfg.chunk_size).min(budget);
            if take == 0 {
                break;
            }
            chunks.push(PrefillChunk {
                user_id: head.user_id,
                start: head.tokens_processed,
                end: head.tokens_processed + take,
            });
            head.tokens_processed += take;
            head.remaining_tokens -= take;
            budget -= take;
            if head.remaining_tokens > 0 {
                // Push back to the FRONT so the next iteration continues this prefill
                // for the same user. Deterministic per-user FIFO.
                self.prefill_queue.push_front(head);
            }
            // If remaining_tokens == 0, drop the head — prefill is fully done for this user.
        }

        (decode_users, chunks)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_prompt_emits_single_chunk() {
        let mut sched = ChunkedPrefillScheduler::new(ChunkConfig::default());
        sched.enqueue_prefill(42, 100);
        let (decoders, chunks) = sched.next_batch();
        assert!(decoders.is_empty());
        assert_eq!(chunks.len(), 1);
        assert_eq!(
            chunks[0],
            PrefillChunk {
                user_id: 42,
                start: 0,
                end: 100
            }
        );
        assert_eq!(sched.pending_prefills(), 0);
    }

    #[test]
    fn long_prompt_splits_into_chunks() {
        let mut sched = ChunkedPrefillScheduler::new(ChunkConfig {
            chunk_size: 512,
            token_budget: 4096,
        });
        sched.enqueue_prefill(7, 1500); // 3 chunks: 512, 512, 476

        let (_, chunks) = sched.next_batch();
        // Single user, budget allows all 3 chunks in one batch
        assert_eq!(chunks.len(), 3);
        assert_eq!(
            chunks,
            vec![
                PrefillChunk {
                    user_id: 7,
                    start: 0,
                    end: 512
                },
                PrefillChunk {
                    user_id: 7,
                    start: 512,
                    end: 1024
                },
                PrefillChunk {
                    user_id: 7,
                    start: 1024,
                    end: 1500
                },
            ]
        );
    }

    #[test]
    fn budget_limits_chunks_per_step() {
        let mut sched = ChunkedPrefillScheduler::new(ChunkConfig {
            chunk_size: 512,
            token_budget: 600,
        });
        sched.enqueue_prefill(1, 2000);
        let (_, chunks) = sched.next_batch();
        // budget=600 → first chunk of 512, then remaining 88 budget → second partial chunk
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].end - chunks[0].start, 512);
        assert_eq!(chunks[1].end - chunks[1].start, 88);
        assert_eq!(sched.pending_prefills(), 1); // 2000 - 600 = 1400 remaining
    }

    #[test]
    fn decode_users_consume_budget_first() {
        let mut sched = ChunkedPrefillScheduler::new(ChunkConfig {
            chunk_size: 512,
            token_budget: 100,
        });
        // Register 80 decode users → 80 decode tokens this step → 20 left for prefill
        for u in 0..80 {
            sched.register_decode_user(u);
        }
        sched.enqueue_prefill(999, 1000);
        let (decoders, chunks) = sched.next_batch();
        assert_eq!(decoders.len(), 80);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].end - chunks[0].start, 20);
    }

    #[test]
    fn empty_scheduler_returns_empty_batch() {
        let mut sched = ChunkedPrefillScheduler::new(ChunkConfig::default());
        let (decoders, chunks) = sched.next_batch();
        assert!(decoders.is_empty());
        assert!(chunks.is_empty());
    }

    #[test]
    fn multiple_users_interleave_fifo() {
        let mut sched = ChunkedPrefillScheduler::new(ChunkConfig {
            chunk_size: 256,
            token_budget: 4096,
        });
        sched.enqueue_prefill(1, 1000);
        sched.enqueue_prefill(2, 1000);
        let (_, chunks) = sched.next_batch();
        // User 1 fully processed first (FIFO), then user 2.
        assert_eq!(chunks.len(), 8);
        let users: Vec<u64> = chunks.iter().map(|c| c.user_id).collect();
        assert_eq!(users, vec![1, 1, 1, 1, 2, 2, 2, 2]);
    }
}
