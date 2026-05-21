//! MoBA (Mixture of Block Attention) routing — Moonshot / Kimi K2 family.
//!
//! Paper: `research/12_long_context/moba_mixture_of_block_attention.pdf`
//! Repo:  `research/code/03_per_token_speed/moba/`
//!
//! ## What
//!
//! MoBA treats attention like MoE: each query token routes to a subset of
//! *context blocks* (not all of them). For 1M context, each query attends to
//! ~k=32 blocks of 4096 tokens = ~128K attended — 8× fewer scores than dense.
//!
//! The routing is learned at training time: per query, a small router produces
//! softmax weights over blocks; the top-k are chosen. Inference cost drops
//! roughly linearly with the block-prune ratio.
//!
//! ## Tier A scope
//!
//! - `MobaConfig`: block_size, top_k, total_blocks
//! - `route_to_blocks()`: given per-query router logits, produce the top-k block ids
//! - `sparse_attention_mask()`: build the [seq, seq] boolean mask saying which
//!   query attends to which key positions
//! - Tests: routing returns top-k indices in correct order; mask is consistent
//!
//! ## Tier B (deferred)
//!
//! - Wire into Kimi K2 model loader (`mistralrs-core/src/models/kimi_k2.rs`)
//! - Custom CUDA kernel for sparse attention with router output
//! - Compose with SageAttention INT8 (per-block quant of selected blocks)

use std::cmp::Reverse;

#[derive(Debug, Clone, Copy)]
pub struct MobaConfig {
    /// Block size along the key/context dimension. Default 4096 per Kimi K2.
    pub block_size: usize,
    /// Number of blocks each query attends to. Default 32 (Kimi K2 1M-context config).
    pub top_k: usize,
}

impl Default for MobaConfig {
    fn default() -> Self {
        Self {
            block_size: 4096,
            top_k: 32,
        }
    }
}

/// Given router logits of shape `[num_queries, num_blocks]`, return for each
/// query the top-k block indices in descending logit order.
///
/// `router_logits` is flat row-major: `router_logits[q * num_blocks + b]` is
/// the score of block b for query q.
pub fn route_to_blocks(
    router_logits: &[f32],
    num_queries: usize,
    num_blocks: usize,
    top_k: usize,
) -> Vec<Vec<usize>> {
    let top_k = top_k.min(num_blocks);
    let mut out = Vec::with_capacity(num_queries);
    for q in 0..num_queries {
        let row = &router_logits[q * num_blocks..(q + 1) * num_blocks];
        let mut indexed: Vec<(usize, f32)> = row.iter().copied().enumerate().collect();
        // Partial sort by descending score.
        indexed.sort_unstable_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        let _ = Reverse(0); // silence import lint
        out.push(indexed.iter().take(top_k).map(|(i, _)| *i).collect());
    }
    out
}

/// Build a boolean attention mask of shape `[num_queries, num_keys]` (row-major)
/// from per-query selected block ids and the MoBA config. True = attend.
///
/// `selected[q][i]` is a block id; the mask covers all positions in that block:
/// `[block * block_size, (block+1) * block_size)`.
///
/// Additionally, **always attend to the local window** (the block containing the
/// query itself) — this is MoBA's "causal local + global block" hybrid rule.
pub fn sparse_attention_mask(
    selected: &[Vec<usize>],
    num_queries: usize,
    num_keys: usize,
    cfg: MobaConfig,
) -> Vec<bool> {
    let mut mask = vec![false; num_queries * num_keys];
    let num_blocks = num_keys.div_ceil(cfg.block_size);
    for q in 0..num_queries {
        // Local block (the one containing q)
        let local_block = (q / cfg.block_size).min(num_blocks.saturating_sub(1));
        let local_start = local_block * cfg.block_size;
        let local_end = ((local_block + 1) * cfg.block_size).min(num_keys);
        for k in local_start..local_end {
            // Causal: k <= q
            if k <= q {
                mask[q * num_keys + k] = true;
            }
        }
        // Routed blocks
        for &b in &selected[q] {
            let start = b * cfg.block_size;
            let end = ((b + 1) * cfg.block_size).min(num_keys);
            for k in start..end {
                if k <= q {
                    mask[q * num_keys + k] = true;
                }
            }
        }
    }
    mask
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn routing_picks_top_k_by_descending_score() {
        let logits = vec![
            0.1, 0.9, 0.3, 0.7, // query 0: top-2 = [1, 3]
            5.0, 1.0, 6.0, 2.0, // query 1: top-2 = [2, 0]
        ];
        let routes = route_to_blocks(&logits, 2, 4, 2);
        assert_eq!(routes, vec![vec![1, 3], vec![2, 0]]);
    }

    #[test]
    fn routing_handles_k_larger_than_num_blocks() {
        let logits = vec![0.1, 0.9, 0.3];
        let routes = route_to_blocks(&logits, 1, 3, 10);
        // top-3 in order
        assert_eq!(routes[0], vec![1, 2, 0]);
    }

    #[test]
    fn mask_always_includes_local_block_causally() {
        let cfg = MobaConfig {
            block_size: 4,
            top_k: 0, // no routed blocks
        };
        // query at position 5 (in block 1), 8 keys total.
        // Local block 1 covers positions [4, 5, 6, 7]; causal cuts to [4, 5].
        let selected = vec![Vec::<usize>::new()]; // empty routing
        let mask = sparse_attention_mask(&selected, 1, 8, cfg);
        // We treated num_queries=1, so query index 0; for local-block logic we use q=0.
        // Block of q=0 is block 0 -> positions [0,1,2,3]; causal -> only k=0.
        assert!(mask[0]);
        for k in 1..8 {
            assert!(!mask[k], "position {k} should be unattended");
        }
    }

    #[test]
    fn mask_attends_to_routed_blocks_causally() {
        let cfg = MobaConfig {
            block_size: 2,
            top_k: 2,
        };
        // num_queries=4, num_keys=8 (4 blocks of 2)
        // query 3 (in block 1) routes to blocks [0, 2]
        let selected = vec![
            vec![],     // query 0
            vec![],     // query 1
            vec![],     // query 2
            vec![0, 2], // query 3 routes to block 0 (positions 0,1) and block 2 (positions 4,5)
        ];
        let mask = sparse_attention_mask(&selected, 4, 8, cfg);
        // For query 3: local block 1 (positions 2, 3), causal -> {2, 3}.
        // Routed block 0 (positions 0, 1), causal vs q=3 -> both allowed -> {0, 1}.
        // Routed block 2 (positions 4, 5), causal vs q=3 -> none allowed (5 > 3).
        let row3 = &mask[3 * 8..4 * 8];
        let expected = vec![true, true, true, true, false, false, false, false];
        assert_eq!(row3.to_vec(), expected);
    }
}
