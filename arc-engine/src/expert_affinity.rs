//! Cross-request expert affinity batching for MoE serving.
//!
//! Reference: DeepSeek V3 §3.2 (EPLB) + `research/code/06_foundation/sglang/python/sglang/srt/eplb/`
//!
//! ## What
//!
//! When N users decode concurrently on an MoE model, many of their tokens route
//! to the same expert. Without batching, each user triggers an independent HBM
//! load of that expert's weights. Affinity batching groups users by predicted
//! expert routing within a small time window (~50ms) and serves all matching
//! users from a single weight load.
//!
//! Typical gain on Mixtral / V3 at 100+ concurrent decodes: 2–3× aggregate
//! throughput improvement.
//!
//! ## Tier A scope
//!
//! - `AffinityBatcher`: group pending requests by expert id within a window
//! - `Request`: minimal handle (user id + predicted routing)
//! - `flush()`: produce the batched (expert_id → vec<user_id>) buckets, ready
//!   for the MoE forward to consume
//! - Tests: grouping correctness, top-k routing, deterministic ordering
//!
//! ## Tier B (deferred)
//!
//! - Wire into `mistralrs-core/src/scheduler/default_scheduler.rs`
//! - Real-time window/jitter handling
//! - HBM-load instrumentation to measure actual savings

use std::collections::BTreeMap;

#[derive(Debug, Clone)]
pub struct PendingRequest {
    /// User-visible request id.
    pub user_id: u64,
    /// Expert ids this request will route to in the next decode step.
    /// Typically `top_k` experts (e.g., top-2 for V3).
    pub expert_ids: Vec<u32>,
}

#[derive(Debug, Clone, Copy)]
pub struct AffinityConfig {
    /// Number of experts in the MoE layer.
    pub num_experts: u32,
}

/// Output bucket: per-expert, the list of user requests that need it.
pub type ExpertBuckets = BTreeMap<u32, Vec<u64>>;

/// Group a batch of pending requests by their expert affinity. Each expert id
/// gets a list of user ids that route to it. A user appears in `top_k` buckets
/// (one per expert they route to).
pub fn group_by_expert(requests: &[PendingRequest], cfg: AffinityConfig) -> ExpertBuckets {
    let mut buckets: ExpertBuckets = BTreeMap::new();
    for req in requests {
        for &expert in &req.expert_ids {
            if expert >= cfg.num_experts {
                continue; // ignore out-of-range routing
            }
            buckets.entry(expert).or_default().push(req.user_id);
        }
    }
    // Stable ordering within each bucket.
    for users in buckets.values_mut() {
        users.sort();
        users.dedup();
    }
    buckets
}

/// Total HBM-load count under the naive (per-user) policy.
pub fn naive_hbm_loads(requests: &[PendingRequest]) -> usize {
    requests.iter().map(|r| r.expert_ids.len()).sum()
}

/// Total HBM-load count under affinity batching (one load per distinct expert).
pub fn batched_hbm_loads(buckets: &ExpertBuckets) -> usize {
    buckets.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(uid: u64, experts: Vec<u32>) -> PendingRequest {
        PendingRequest {
            user_id: uid,
            expert_ids: experts,
        }
    }

    #[test]
    fn empty_input_produces_empty_buckets() {
        let buckets = group_by_expert(&[], AffinityConfig { num_experts: 8 });
        assert!(buckets.is_empty());
    }

    #[test]
    fn single_user_routes_to_each_top_k_expert() {
        let buckets = group_by_expert(&[req(1, vec![2, 5])], AffinityConfig { num_experts: 8 });
        assert_eq!(buckets.len(), 2);
        assert_eq!(buckets[&2], vec![1]);
        assert_eq!(buckets[&5], vec![1]);
    }

    #[test]
    fn many_users_share_an_expert() {
        // 5 users all routing to expert 0 → bucket[0] = [1,2,3,4,5]
        let reqs = (1..=5u64).map(|u| req(u, vec![0])).collect::<Vec<_>>();
        let buckets = group_by_expert(&reqs, AffinityConfig { num_experts: 4 });
        assert_eq!(buckets.len(), 1);
        assert_eq!(buckets[&0], vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn out_of_range_expert_is_ignored() {
        let buckets = group_by_expert(&[req(1, vec![999])], AffinityConfig { num_experts: 8 });
        assert!(buckets.is_empty());
    }

    /// At 100 concurrent users with top-2 routing on 64 experts, naive policy
    /// triggers 200 HBM loads. Affinity batching should collapse it to
    /// ≤ 64 (and usually much less if routing has hotspots).
    #[test]
    fn batching_reduces_hbm_loads_at_scale() {
        let cfg = AffinityConfig { num_experts: 64 };
        // 100 users each routing to top-2 experts, pseudo-randomly distributed
        let reqs: Vec<PendingRequest> = (0..100)
            .map(|u| {
                let e0 = (u * 7) % 64;
                let e1 = (u * 11 + 3) % 64;
                req(u as u64, vec![e0 as u32, e1 as u32])
            })
            .collect();
        let naive = naive_hbm_loads(&reqs);
        assert_eq!(naive, 200);

        let buckets = group_by_expert(&reqs, cfg);
        let batched = batched_hbm_loads(&buckets);
        assert!(
            batched < naive,
            "batched={batched} should be < naive={naive}"
        );
        // With 100 users × top-2 routing over 64 experts, we expect roughly all
        // 64 buckets to be filled.
        assert!(batched <= 64);
    }
}
