//! MagicDec long-context speculative decoding hybrid.
//!
//! Paper: `research/15_speculative_extensions/magicdec_speculative_long_context.pdf`
//! Repo:  `research/code/03_per_token_speed/magicdec/`
//!
//! ## Why
//!
//! Standard speculative decoding (EAGLE-2/3) helps at short context but loses
//! effectiveness above ~64K tokens because the draft model's KV cache becomes
//! the bottleneck. MagicDec runs the draft on a **fixed-size sliding window of
//! the recent context**, while the target still verifies on the full context.
//! Layered on EAGLE-3 or MTP, adds ~1.5× decode at 1M context.
//!
//! ## Tier A scope
//!
//! - `MagicDecConfig`: window size, activation threshold (context length that
//!   triggers MagicDec mode)
//! - `should_activate()`: decide whether to use MagicDec for a given context
//! - `compute_draft_kv_window()`: given the full KV positions and the current
//!   query position, return the sliding window the draft should attend to
//! - Tests: activation threshold, window correctness on edges
//!
//! ## Tier B (deferred)
//!
//! - Plumb into `mistralrs-core/src/pipeline/speculative.rs`
//! - Real EAGLE-3-with-sliding-window dispatch
//! - 1M-context benchmarks on B200

#[derive(Debug, Clone, Copy)]
pub struct MagicDecConfig {
    /// Sliding window length the draft attends to. Default 4096 per paper.
    pub window: usize,
    /// Context length above which MagicDec auto-activates. Default 65536 (64K).
    pub trigger: usize,
}

impl Default for MagicDecConfig {
    fn default() -> Self {
        Self {
            window: 4096,
            trigger: 65_536,
        }
    }
}

impl MagicDecConfig {
    /// Should MagicDec mode be active at the given current context length?
    pub fn should_activate(&self, current_context_len: usize) -> bool {
        current_context_len >= self.trigger
    }

    /// Compute the draft's sliding window: `[start, end)` of token positions
    /// it should attend to. Always includes the current query position; extends
    /// backwards `window-1` tokens.
    ///
    /// At short context (below trigger), returns the full range — MagicDec is
    /// disabled and the draft uses everything.
    pub fn compute_draft_kv_window(
        &self,
        full_context_len: usize,
        current_query_pos: usize,
    ) -> (usize, usize) {
        if !self.should_activate(full_context_len) {
            return (0, full_context_len);
        }
        let end = (current_query_pos + 1).min(full_context_len);
        let start = end.saturating_sub(self.window);
        (start, end)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_trigger_at_64k() {
        let cfg = MagicDecConfig::default();
        assert!(!cfg.should_activate(32_768));
        assert!(cfg.should_activate(65_536));
        assert!(cfg.should_activate(1_000_000));
    }

    #[test]
    fn below_trigger_returns_full_window() {
        let cfg = MagicDecConfig::default();
        // 32K context (below trigger of 64K) → draft sees full
        let (s, e) = cfg.compute_draft_kv_window(32_768, 16_000);
        assert_eq!((s, e), (0, 32_768));
    }

    #[test]
    fn above_trigger_uses_sliding_window() {
        let cfg = MagicDecConfig::default();
        // 1M context, query at position 999_000 → window [994904, 999001]
        let (s, e) = cfg.compute_draft_kv_window(1_000_000, 999_000);
        assert_eq!(e, 999_001);
        assert_eq!(e - s, 4096);
    }

    #[test]
    fn early_query_in_long_context_starts_at_zero() {
        let cfg = MagicDecConfig::default();
        // Query at position 100 in a 1M context → window [0, 101]
        let (s, e) = cfg.compute_draft_kv_window(1_000_000, 100);
        assert_eq!(s, 0);
        assert_eq!(e, 101);
    }

    #[test]
    fn custom_window_size() {
        let cfg = MagicDecConfig {
            window: 1024,
            trigger: 10_000,
        };
        assert!(cfg.should_activate(10_000));
        let (s, e) = cfg.compute_draft_kv_window(50_000, 30_000);
        assert_eq!(e - s, 1024);
    }
}
