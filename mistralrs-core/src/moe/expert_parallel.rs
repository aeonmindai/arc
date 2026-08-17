//! Expert parallelism (EP) for MoE layers.
//!
//! # What this is
//!
//! Under expert parallelism the `E` routed experts of a MoE layer are **split
//! across ranks** instead of replicated. Each rank holds `E / ep_size` experts
//! and reads only those weights per decode step. Attention and the dense terms
//! stay replicated (data-parallel attention), because V4's K/V is a single
//! fused MQA head and tensor-parallelising it would duplicate the KV cache on
//! every card — and KV memory, not bandwidth, is what caps batch size.
//!
//! The design, the arithmetic and the reference-implementation survey live in
//! `memory/mission/wave44-BV-expert-parallelism.md`. The two numbers that
//! matter here:
//!
//! * the saturated per-user floor at B=128 moves **65 tok/s (N=1) → 126 (N=2)**,
//!   so two cards are the first rung that clears the 100 tok/s/user target;
//! * weight-read imbalance is **hard-bounded at +5.05% at B=128**, because a
//!   GPU cannot read more experts than it owns. That is why stage 1 ships
//!   without EPLB and only instruments for it. (The note gives +4.2…4.8% and a
//!   per-N spread; the spread does not follow from its own definition and the
//!   real bound is a single slightly larger number — see
//!   [`weight_read_imbalance_bound`], which carries the correction.)
//!
//! # The contract
//!
//! For a token `t` and routing slot `j`, EP=1 computes
//!
//! ```text
//! y[t] = Σ_j w[t,j] · Expert_{g[t,j]}(x[t])
//! ```
//!
//! Under EP=N, rank `r` computes the **partial**
//!
//! ```text
//! y_r[t] = Σ_j [owner(g[t,j]) == r] · w[t,j] · Expert_{g[t,j]}(x[t])
//! ```
//!
//! and the combine is `y = Σ_r y_r`. Every term appears in exactly one rank's
//! partial, so the sum is the EP=1 result up to floating-point re-association.
//! [`ExpertParallelPlan::localize`] produces the masked weights and the
//! rank-local expert ids that make `y_r` computable with the ordinary,
//! unmodified expert kernels.
//!
//! Slots this rank does not own are **not removed** — they are pinned to local
//! expert 0 with weight exactly `0.0`. That costs a little compute but no extra
//! weight bytes (local expert 0 is essentially always already resident at the
//! batch sizes where EP is worth doing), and it keeps every tensor shape
//! identical across ranks, which is what lets the combine be a plain sum.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use candle_core::{DType, Device, Result, Tensor};
use mistralrs_quant::Shard;

/// Which rank owns each expert, and where that expert sits in its owner's
/// local expert list.
///
/// Every rank must own the **same number** of experts: the expert-stacked
/// weight tensors have `num_local_experts` as their leading dimension, so an
/// uneven split would give the ranks different kernel shapes and break the
/// "combine is a plain sum of identically-shaped partials" contract.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertPlacement {
    /// `expert_to_rank[g]` — the rank that owns global expert `g`.
    expert_to_rank: Vec<usize>,
    /// `expert_to_local[g]` — the index of `g` inside its owner's local list.
    expert_to_local: Vec<usize>,
    /// `rank_experts[r]` — the global ids owned by rank `r`, ascending.
    rank_experts: Vec<Vec<usize>>,
}

impl ExpertPlacement {
    /// Contiguous block sharding: rank `r` owns
    /// `[r * E/N, (r+1) * E/N)`. This is the placement whose load can be
    /// expressed as a plain leading-dimension [`Shard`], i.e. the only one that
    /// slices the checkpoint at load time instead of after it.
    pub fn contiguous(num_experts: usize, ep_size: usize) -> Result<Self> {
        if ep_size == 0 {
            candle_core::bail!("expert parallelism: ep_size must be >= 1");
        }
        if num_experts % ep_size != 0 {
            candle_core::bail!(
                "expert parallelism: num_experts ({num_experts}) is not divisible by ep_size ({ep_size})"
            );
        }
        let per_rank = num_experts / ep_size;
        let expert_to_rank = (0..num_experts).map(|g| g / per_rank).collect::<Vec<_>>();
        Self::from_expert_to_rank(expert_to_rank, ep_size)
    }

    /// Build a placement from an explicit expert → rank assignment.
    ///
    /// Rejects an assignment that does not give every rank exactly
    /// `num_experts / ep_size` experts. That rejection is load-bearing: an
    /// uneven placement would silently produce differently-shaped expert
    /// tensors per rank.
    pub fn from_expert_to_rank(expert_to_rank: Vec<usize>, ep_size: usize) -> Result<Self> {
        if ep_size == 0 {
            candle_core::bail!("expert parallelism: ep_size must be >= 1");
        }
        let num_experts = expert_to_rank.len();
        if num_experts == 0 {
            candle_core::bail!("expert parallelism: placement needs at least one expert");
        }
        if num_experts % ep_size != 0 {
            candle_core::bail!(
                "expert parallelism: num_experts ({num_experts}) is not divisible by ep_size ({ep_size})"
            );
        }
        let per_rank = num_experts / ep_size;

        let mut rank_experts = vec![Vec::with_capacity(per_rank); ep_size];
        for (g, &r) in expert_to_rank.iter().enumerate() {
            if r >= ep_size {
                candle_core::bail!(
                    "expert parallelism: expert {g} assigned to rank {r}, but ep_size is {ep_size}"
                );
            }
            rank_experts[r].push(g);
        }
        for (r, experts) in rank_experts.iter().enumerate() {
            if experts.len() != per_rank {
                candle_core::bail!(
                    "expert parallelism: rank {r} owns {} experts, expected exactly {per_rank}. \
                     An uneven placement would give the ranks different expert-tensor shapes.",
                    experts.len()
                );
            }
        }

        let mut expert_to_local = vec![0usize; num_experts];
        for experts in &rank_experts {
            for (local, &g) in experts.iter().enumerate() {
                expert_to_local[g] = local;
            }
        }

        Ok(Self {
            expert_to_rank,
            expert_to_local,
            rank_experts,
        })
    }

    /// Balanced placement from per-expert loads, by longest-processing-time
    /// bin packing under an exact-cardinality constraint: sort experts by load
    /// descending, give each to the least-loaded rank that still has room.
    ///
    /// This is the same greedy `balanced_packing` shape DeepSeek's EPLB uses
    /// (`sglang/python/sglang/srt/eplb/eplb_algorithms/deepseek.py:7-52`),
    /// minus the redundant-expert replication, which stage 1 deliberately does
    /// not ship. Ties break on the lower rank index so the result is
    /// deterministic and reproducible offline.
    pub fn balanced(loads: &[f64], ep_size: usize) -> Result<Self> {
        if ep_size == 0 {
            candle_core::bail!("expert parallelism: ep_size must be >= 1");
        }
        if loads.is_empty() {
            candle_core::bail!("expert parallelism: placement needs at least one expert");
        }
        if loads.len() % ep_size != 0 {
            candle_core::bail!(
                "expert parallelism: num_experts ({}) is not divisible by ep_size ({ep_size})",
                loads.len()
            );
        }
        let per_rank = loads.len() / ep_size;

        let mut order: Vec<usize> = (0..loads.len()).collect();
        // Descending load, then ascending expert id — total order, no ties.
        order.sort_by(|&a, &b| {
            loads[b]
                .partial_cmp(&loads[a])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        });

        let mut rank_load = vec![0.0f64; ep_size];
        let mut rank_count = vec![0usize; ep_size];
        let mut expert_to_rank = vec![0usize; loads.len()];
        for &g in &order {
            let mut best: Option<usize> = None;
            for r in 0..ep_size {
                if rank_count[r] >= per_rank {
                    continue;
                }
                match best {
                    None => best = Some(r),
                    Some(b) if rank_load[r] < rank_load[b] => best = Some(r),
                    _ => {}
                }
            }
            // `best` is always Some: total capacity equals the expert count.
            let r = best.expect("bin packing ran out of capacity before experts");
            expert_to_rank[g] = r;
            rank_load[r] += loads[g];
            rank_count[r] += 1;
        }

        Self::from_expert_to_rank(expert_to_rank, ep_size)
    }

    pub fn ep_size(&self) -> usize {
        self.rank_experts.len()
    }

    pub fn num_experts(&self) -> usize {
        self.expert_to_rank.len()
    }

    pub fn num_local_experts(&self) -> usize {
        self.num_experts() / self.ep_size()
    }

    pub fn owner(&self, global_expert: usize) -> usize {
        self.expert_to_rank[global_expert]
    }

    /// Index of `global_expert` inside its owner's local expert list.
    pub fn local_of(&self, global_expert: usize) -> usize {
        self.expert_to_local[global_expert]
    }

    /// Global expert ids owned by `rank`, ascending.
    pub fn experts_of(&self, rank: usize) -> &[usize] {
        &self.rank_experts[rank]
    }

    /// `true` when every rank owns one contiguous block `[r*E/N, (r+1)*E/N)`.
    /// Only then can the weight load be expressed as a leading-dimension
    /// [`Shard`] and slice the checkpoint instead of the loaded tensor.
    pub fn is_contiguous(&self) -> bool {
        let per_rank = self.num_local_experts();
        self.expert_to_rank
            .iter()
            .enumerate()
            .all(|(g, &r)| r == g / per_rank)
    }

    /// Per-rank sums of a per-expert quantity (token counts, byte counts, …).
    pub fn per_rank_totals(&self, per_expert: &[f64]) -> Result<Vec<f64>> {
        if per_expert.len() != self.num_experts() {
            candle_core::bail!(
                "expert parallelism: expected {} per-expert values, got {}",
                self.num_experts(),
                per_expert.len()
            );
        }
        let mut totals = vec![0.0f64; self.ep_size()];
        for (g, &v) in per_expert.iter().enumerate() {
            totals[self.owner(g)] += v;
        }
        Ok(totals)
    }
}

/// One rank's view of an [`ExpertPlacement`], plus the device-resident lookup
/// tables that [`Self::localize`] needs.
///
/// The tables are built **once, at layer construction**, never inside the
/// forward — `Tensor::from_vec` against a GPU device forces a host→device sync
/// (CLAUDE.md pitfall 5), and a sync inside the routing path is incompatible
/// with CUDA-graph capture and with overlapping the all-to-all.
#[derive(Debug, Clone)]
pub struct ExpertParallelPlan {
    placement: std::sync::Arc<ExpertPlacement>,
    rank: usize,
    /// `[num_experts]` U32 — global id → local id, or 0 for experts this rank
    /// does not own.
    local_idx_table: Option<Tensor>,
    /// `[num_experts]` F32 — 1.0 for owned experts, 0.0 otherwise.
    owned_mask: Option<Tensor>,
}

impl ExpertParallelPlan {
    /// The disabled plan: one rank owning everything. Bit-for-bit the
    /// pre-EP behaviour — [`Self::localize`] is the identity and
    /// [`Self::is_enabled`] is `false`, so no call site changes.
    pub fn single(num_experts: usize) -> Self {
        let placement = ExpertPlacement::contiguous(num_experts, 1)
            .expect("contiguous placement with ep_size 1 is always valid");
        Self {
            placement: std::sync::Arc::new(placement),
            rank: 0,
            local_idx_table: None,
            owned_mask: None,
        }
    }

    /// Build the plan for `rank` and materialise the routing tables on
    /// `device`.
    pub fn new(
        placement: std::sync::Arc<ExpertPlacement>,
        rank: usize,
        device: &Device,
    ) -> Result<Self> {
        if rank >= placement.ep_size() {
            candle_core::bail!(
                "expert parallelism: rank {rank} is out of range for ep_size {}",
                placement.ep_size()
            );
        }
        if placement.ep_size() == 1 {
            return Ok(Self {
                placement,
                rank,
                local_idx_table: None,
                owned_mask: None,
            });
        }

        let n = placement.num_experts();
        let mut local_idx = vec![0u32; n];
        let mut mask = vec![0f32; n];
        for g in 0..n {
            if placement.owner(g) == rank {
                local_idx[g] = placement.local_of(g) as u32;
                mask[g] = 1.0;
            }
        }
        let local_idx_table = Tensor::from_vec(local_idx, (n,), device)?;
        let owned_mask = Tensor::from_vec(mask, (n,), device)?;

        Ok(Self {
            placement,
            rank,
            local_idx_table: Some(local_idx_table),
            owned_mask: Some(owned_mask),
        })
    }

    pub fn is_enabled(&self) -> bool {
        self.placement.ep_size() > 1
    }

    pub fn ep_size(&self) -> usize {
        self.placement.ep_size()
    }

    pub fn rank(&self) -> usize {
        self.rank
    }

    pub fn placement(&self) -> &ExpertPlacement {
        &self.placement
    }

    pub fn num_local_experts(&self) -> usize {
        self.placement.num_local_experts()
    }

    /// Global expert ids this rank owns, ascending.
    pub fn owned_experts(&self) -> &[usize] {
        self.placement.experts_of(self.rank)
    }

    /// A leading-dimension shard hint for the expert-stacked weight tensors,
    /// when (and only when) the placement is contiguous. `None` means the
    /// caller must select `owned_experts()` explicitly after loading.
    pub fn expert_shard(&self) -> Option<Shard> {
        if !self.is_enabled() {
            return None;
        }
        self.placement.is_contiguous().then_some(Shard::Simple {
            dim: 0,
            rank: self.rank,
            world_size: self.placement.ep_size(),
        })
    }

    /// Map global routing ids to this rank's local expert ids and zero the
    /// weights of slots this rank does not own.
    ///
    /// `topk_ids` is `[tokens, top_k]` (U32); `topk_weights` is
    /// `[tokens, top_k]`. Returns `(local_ids, masked_weights)` with the same
    /// shapes. With EP disabled this is the identity and does no work.
    pub fn localize(&self, topk_ids: &Tensor, topk_weights: &Tensor) -> Result<(Tensor, Tensor)> {
        let (Some(local_idx_table), Some(owned_mask)) =
            (self.local_idx_table.as_ref(), self.owned_mask.as_ref())
        else {
            return Ok((topk_ids.clone(), topk_weights.clone()));
        };

        let shape = topk_ids.shape().clone();
        let flat = topk_ids.flatten_all()?.to_dtype(DType::U32)?;
        let local = local_idx_table.index_select(&flat, 0)?.reshape(&shape)?;
        let mask = owned_mask
            .index_select(&flat, 0)?
            .reshape(&shape)?
            .to_dtype(topk_weights.dtype())?;
        let weights = (topk_weights * mask)?;
        Ok((local.to_dtype(topk_ids.dtype())?, weights))
    }
}

/// Mean / max / ratio of a per-rank quantity. `ratio` is the number the
/// reference systems publish as the "imbalance ratio": the hottest rank's
/// share divided by the mean. TensorRT-LLM measures **1.564** for DeepSeek-R1
/// at EP=32 on token counts
/// (`tensorrt_llm/examples/wide_ep/ep_load_balancer/README.md:49-70`).
#[derive(Debug, Clone, PartialEq)]
pub struct Balancedness {
    pub per_rank: Vec<f64>,
    pub mean: f64,
    pub max: f64,
    pub ratio: f64,
}

/// Mean/max/ratio for a per-rank vector. An all-zero vector has ratio 1.0
/// (perfectly balanced by definition) rather than NaN.
pub fn balancedness(per_rank: &[f64]) -> Balancedness {
    let n = per_rank.len().max(1) as f64;
    let sum: f64 = per_rank.iter().sum();
    let mean = sum / n;
    let max = per_rank.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let ratio = if mean > 0.0 { max / mean } else { 1.0 };
    Balancedness {
        per_rank: per_rank.to_vec(),
        mean,
        max,
        ratio: if ratio.is_finite() { ratio } else { 1.0 },
    }
}

/// The **hard bound** on weight-read imbalance, which is what makes EPLB
/// skippable in stage 1.
///
/// Arc's dominant per-step cost is expert *weight bytes*, and a rank cannot
/// read more experts than it owns. So however skewed the routing is, the
/// hottest rank reads at most `min(experts_owned, distinct_experts_woken)`
/// experts against a balanced `distinct / ep_size`. Returns the worst-case
/// ratio (1.0 = no possible imbalance).
///
/// # Correction to wave44-BV §3.2
///
/// The design note tabulates a per-N spread (+4.8% / +4.6% / +4.2% at
/// N=2/4/8, B=128). **That spread does not follow from the definition it
/// states.** Once coverage exceeds `E/N` — i.e. once every rank's whole slice
/// can be woken — the hottest rank reads `E/N` experts against a balanced
/// `distinct/N`, and the `N` cancels:
///
/// ```text
/// (E/N) / (distinct/N) = E / distinct
/// ```
///
/// so the bound is **independent of N** in that regime. Recomputed for V4's
/// 256 experts, top-6:
///
/// | B | coverage | bound (every N) | the note's figure |
/// |---|---|---|---|
/// | 64 | 78.1% | **+28.1%** | +23…27% |
/// | 128 | 95.2% | **+5.05%** | +4.2…4.8% |
/// | 256 | 99.8% | **+0.23%** | +0.2% |
///
/// The corrected bound is slightly **worse** than the note claimed, and the
/// note's per-N variation is an artefact. The conclusion is unchanged and is
/// the part that matters: at the batch sizes where EP is worth doing, routing
/// skew cannot cost more than ~5% of the weight-read term, so stage 1 ships
/// without EPLB and instruments for it instead.
///
/// Below `E/N` coverage — B ≲ 64 for V4 at N=2 — the bound is genuinely loose
/// and N-dependent, and skew really can bite; but there a single card is
/// already nearer the per-user target.
pub fn weight_read_imbalance_bound(num_experts: usize, ep_size: usize, distinct: f64) -> f64 {
    if ep_size <= 1 || num_experts == 0 {
        return 1.0;
    }
    let owned = (num_experts / ep_size) as f64;
    let distinct = distinct.clamp(0.0, num_experts as f64);
    if distinct <= 0.0 {
        return 1.0;
    }
    let balanced = distinct / ep_size as f64;
    (owned.min(distinct) / balanced).max(1.0)
}

/// Expected number of **distinct** experts woken by `tokens` tokens each
/// picking `top_k` of `num_experts` uniformly: `E · (1 − (1 − k/E)^T)`.
///
/// This is the coverage curve wave44-BV §3.2 tabulates (31.6% at B=16 …
/// 99.8% at B=256 for V4's 256/6). It assumes uniform routing; real routing is
/// more concentrated, which *lowers* coverage and therefore *widens* the bound
/// above — so this is a floor on coverage, not a forecast.
pub fn expected_distinct_experts(num_experts: usize, top_k: usize, tokens: usize) -> f64 {
    if num_experts == 0 {
        return 0.0;
    }
    let e = num_experts as f64;
    let p_miss = 1.0 - (top_k as f64 / e);
    e * (1.0 - p_miss.powi(tokens as i32))
}

/// Per-rank token counters, so imbalance is **measured** rather than assumed.
///
/// wave44-BV's conclusion is that EPLB's value for Arc is a function of kernel
/// efficiency, not of routing: while the step is bandwidth-bound the bound in
/// [`weight_read_imbalance_bound`] caps the cost at a few percent, and only if
/// the step goes compute-bound does skew start costing 40-50%. That is a
/// claim about the running system, so it needs a counter.
///
/// Recording reads `topk_ids` back to the host, which is a device sync. It is
/// therefore **off unless `ARC_EP_BALANCE=1`** and must never be enabled on a
/// measurement run.
#[derive(Debug)]
pub struct BalancednessCounter {
    /// Tokens routed to each rank (a token counts once per routing slot).
    tokens_per_rank: Vec<AtomicU64>,
    /// Distinct (layer, expert) pairs seen, per rank — the *byte* term.
    distinct_per_rank: std::sync::Mutex<Vec<HashMap<(usize, usize), ()>>>,
    steps: AtomicU64,
}

impl BalancednessCounter {
    pub fn new(ep_size: usize) -> Self {
        Self {
            tokens_per_rank: (0..ep_size).map(|_| AtomicU64::new(0)).collect(),
            distinct_per_rank: std::sync::Mutex::new(vec![HashMap::new(); ep_size]),
            steps: AtomicU64::new(0),
        }
    }

    /// `true` when `ARC_EP_BALANCE=1`. Checked by the caller before doing the
    /// device→host read.
    pub fn enabled() -> bool {
        std::env::var("ARC_EP_BALANCE").is_ok_and(|v| v == "1")
    }

    /// Record one MoE layer's routing. `topk_ids` is `[tokens, top_k]` of
    /// **global** expert ids.
    pub fn record(
        &self,
        layer_idx: usize,
        topk_ids: &Tensor,
        placement: &ExpertPlacement,
    ) -> Result<()> {
        let ids = topk_ids
            .flatten_all()?
            .to_dtype(DType::U32)?
            .to_vec1::<u32>()?;
        let mut distinct = self
            .distinct_per_rank
            .lock()
            .map_err(|e| candle_core::Error::msg(format!("balancedness mutex poisoned: {e:?}")))?;
        for id in ids {
            let g = id as usize;
            if g >= placement.num_experts() {
                continue;
            }
            let r = placement.owner(g);
            self.tokens_per_rank[r].fetch_add(1, Ordering::Relaxed);
            distinct[r].insert((layer_idx, g), ());
        }
        self.steps.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Token-count balancedness — the *compute* term.
    pub fn tokens(&self) -> Balancedness {
        let per_rank = self
            .tokens_per_rank
            .iter()
            .map(|c| c.load(Ordering::Relaxed) as f64)
            .collect::<Vec<_>>();
        balancedness(&per_rank)
    }

    /// Distinct-expert balancedness — the *weight byte* term, the one
    /// [`weight_read_imbalance_bound`] bounds.
    pub fn distinct_experts(&self) -> Balancedness {
        let guard = match self.distinct_per_rank.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        let per_rank = guard.iter().map(|m| m.len() as f64).collect::<Vec<_>>();
        balancedness(&per_rank)
    }

    pub fn steps(&self) -> u64 {
        self.steps.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contiguous_placement_splits_into_blocks() {
        let p = ExpertPlacement::contiguous(8, 2).unwrap();
        assert_eq!(p.experts_of(0), &[0, 1, 2, 3]);
        assert_eq!(p.experts_of(1), &[4, 5, 6, 7]);
        assert_eq!(p.local_of(5), 1);
        assert_eq!(p.owner(5), 1);
        assert!(p.is_contiguous());
    }

    #[test]
    fn placement_rejects_an_indivisible_expert_count() {
        assert!(ExpertPlacement::contiguous(7, 2).is_err());
        assert!(ExpertPlacement::contiguous(256, 2).is_ok());
    }

    /// The uneven-placement rejection is the guard that keeps the combine a
    /// plain sum of identically-shaped partials.
    #[test]
    fn placement_rejects_an_uneven_assignment() {
        // 4 experts, ep_size 2, but rank 0 gets three of them.
        let err = ExpertPlacement::from_expert_to_rank(vec![0, 0, 0, 1], 2).unwrap_err();
        assert!(
            format!("{err}").contains("owns 3 experts"),
            "unexpected error: {err}"
        );
        // The balanced version of the same shape is accepted.
        assert!(ExpertPlacement::from_expert_to_rank(vec![0, 0, 1, 1], 2).is_ok());
    }

    #[test]
    fn placement_rejects_an_out_of_range_rank() {
        let err = ExpertPlacement::from_expert_to_rank(vec![0, 2, 1, 1], 2).unwrap_err();
        assert!(format!("{err}").contains("ep_size is 2"), "{err}");
    }

    /// A permuted placement is legal but is NOT expressible as a leading-dim
    /// shard — `expert_shard()` must return `None` so the loader takes the
    /// index-select path instead of silently loading the wrong block.
    #[test]
    fn permuted_placement_is_not_contiguous_and_has_no_shard_hint() {
        let p = ExpertPlacement::from_expert_to_rank(vec![1, 0, 1, 0], 2).unwrap();
        assert!(!p.is_contiguous());
        let plan = ExpertParallelPlan::new(std::sync::Arc::new(p), 0, &Device::Cpu).unwrap();
        assert!(plan.expert_shard().is_none());
        assert_eq!(plan.owned_experts(), &[1, 3]);

        let contiguous = ExpertPlacement::contiguous(4, 2).unwrap();
        let plan =
            ExpertParallelPlan::new(std::sync::Arc::new(contiguous), 1, &Device::Cpu).unwrap();
        assert_eq!(
            plan.expert_shard(),
            Some(Shard::Simple {
                dim: 0,
                rank: 1,
                world_size: 2
            })
        );
    }

    /// Balanced packing must beat contiguous on a skewed load, and must still
    /// hand every rank the same expert count.
    #[test]
    fn balanced_packing_evens_out_a_skewed_load() {
        // Experts 0..3 are hot, 4..7 are cold. Contiguous puts all the heat on
        // rank 0; balanced must split it.
        let loads = vec![100.0, 90.0, 80.0, 70.0, 1.0, 1.0, 1.0, 1.0];

        let contiguous = ExpertPlacement::contiguous(8, 2).unwrap();
        let c = balancedness(&contiguous.per_rank_totals(&loads).unwrap());
        assert!(c.ratio > 1.9, "contiguous should be badly skewed: {c:?}");

        let packed = ExpertPlacement::balanced(&loads, 2).unwrap();
        let b = balancedness(&packed.per_rank_totals(&loads).unwrap());
        assert!(
            b.ratio < 1.05,
            "balanced packing left {:.3}x imbalance: {b:?}",
            b.ratio
        );
        assert_eq!(packed.experts_of(0).len(), 4);
        assert_eq!(packed.experts_of(1).len(), 4);
    }

    #[test]
    fn balanced_packing_is_deterministic() {
        let loads: Vec<f64> = (0..16).map(|i| ((i * 7) % 5) as f64).collect();
        let a = ExpertPlacement::balanced(&loads, 4).unwrap();
        let b = ExpertPlacement::balanced(&loads, 4).unwrap();
        assert_eq!(a, b);
    }

    /// The whole point of `localize`: owned slots keep their weight and get a
    /// local id; unowned slots are pinned to local expert 0 with weight
    /// exactly 0.0.
    #[test]
    fn localize_masks_unowned_slots_and_remaps_owned_ones() {
        let placement = std::sync::Arc::new(ExpertPlacement::contiguous(8, 2).unwrap());
        let dev = Device::Cpu;
        // One token, four slots: experts 0 and 3 belong to rank 0; 4 and 7 to rank 1.
        let ids = Tensor::from_vec(vec![0u32, 3, 4, 7], (1, 4), &dev).unwrap();
        let w = Tensor::from_vec(vec![0.1f32, 0.2, 0.3, 0.4], (1, 4), &dev).unwrap();

        let r0 = ExpertParallelPlan::new(placement.clone(), 0, &dev).unwrap();
        let (ids0, w0) = r0.localize(&ids, &w).unwrap();
        assert_eq!(
            ids0.flatten_all().unwrap().to_vec1::<u32>().unwrap(),
            vec![0, 3, 0, 0]
        );
        assert_eq!(
            w0.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![0.1, 0.2, 0.0, 0.0]
        );

        let r1 = ExpertParallelPlan::new(placement, 1, &dev).unwrap();
        let (ids1, w1) = r1.localize(&ids, &w).unwrap();
        assert_eq!(
            ids1.flatten_all().unwrap().to_vec1::<u32>().unwrap(),
            vec![0, 0, 0, 3]
        );
        assert_eq!(
            w1.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![0.0, 0.0, 0.3, 0.4]
        );
    }

    /// Every routing slot must be owned by exactly one rank — this is the
    /// invariant that makes `Σ_r partial_r` equal the EP=1 result. Asserted on
    /// the masks themselves, so a placement bug that double-counts or drops an
    /// expert fails here rather than as a quality regression.
    #[test]
    fn masks_partition_the_routing_slots_across_ranks() {
        let placement = std::sync::Arc::new(ExpertPlacement::contiguous(8, 4).unwrap());
        let dev = Device::Cpu;
        let ids = Tensor::from_vec(vec![0u32, 1, 2, 3, 4, 5, 6, 7], (2, 4), &dev).unwrap();
        let w = Tensor::from_vec(vec![1.0f32; 8], (2, 4), &dev).unwrap();

        let mut total = vec![0.0f32; 8];
        for rank in 0..4 {
            let plan = ExpertParallelPlan::new(placement.clone(), rank, &dev).unwrap();
            let (_, masked) = plan.localize(&ids, &w).unwrap();
            for (i, v) in masked
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
                .into_iter()
                .enumerate()
            {
                total[i] += v;
            }
        }
        assert_eq!(
            total,
            vec![1.0f32; 8],
            "slots were dropped or double-counted"
        );
    }

    #[test]
    fn a_disabled_plan_is_the_identity() {
        let dev = Device::Cpu;
        let plan = ExpertParallelPlan::single(8);
        assert!(!plan.is_enabled());
        assert!(plan.expert_shard().is_none());
        let ids = Tensor::from_vec(vec![0u32, 3, 4, 7], (1, 4), &dev).unwrap();
        let w = Tensor::from_vec(vec![0.1f32, 0.2, 0.3, 0.4], (1, 4), &dev).unwrap();
        let (ids2, w2) = plan.localize(&ids, &w).unwrap();
        assert_eq!(
            ids2.flatten_all().unwrap().to_vec1::<u32>().unwrap(),
            vec![0, 3, 4, 7]
        );
        assert_eq!(
            w2.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![0.1, 0.2, 0.3, 0.4]
        );
    }

    /// wave44-BV §3.2's coverage table, reproduced from the formula.
    #[test]
    fn expected_coverage_matches_the_design_note() {
        let cov = |b: usize| expected_distinct_experts(256, 6, b) / 256.0;
        assert!((cov(16) - 0.316).abs() < 0.002, "B=16: {}", cov(16));
        assert!((cov(64) - 0.781).abs() < 0.002, "B=64: {}", cov(64));
        assert!((cov(128) - 0.952).abs() < 0.002, "B=128: {}", cov(128));
        assert!((cov(256) - 0.998).abs() < 0.002, "B=256: {}", cov(256));
    }

    /// The bound that justifies skipping EPLB in stage 1 — **recomputed**, and
    /// deliberately asserting the corrected values rather than wave44-BV's.
    ///
    /// The note's per-N spread (+4.8/+4.6/+4.2% at B=128) does not follow from
    /// its own definition: above `E/N` coverage the `N` cancels. See
    /// [`weight_read_imbalance_bound`]. This test pins the correction, so a
    /// future edit that "restores" the note's numbers fails here.
    #[test]
    fn weight_read_imbalance_bound_is_n_independent_above_full_slice_coverage() {
        let d128 = expected_distinct_experts(256, 6, 128);
        let bounds: Vec<f64> = [2, 4, 8]
            .into_iter()
            .map(|n| weight_read_imbalance_bound(256, n, d128))
            .collect();
        for b in &bounds {
            assert!((b - 1.0505).abs() < 0.001, "B=128 bound: {b}");
        }
        assert!(
            (bounds[0] - bounds[2]).abs() < 1e-12,
            "the bound must not vary with N once coverage exceeds E/N: {bounds:?}"
        );

        let d256 = expected_distinct_experts(256, 6, 256);
        let b256 = weight_read_imbalance_bound(256, 2, d256);
        assert!((b256 - 1.0023).abs() < 0.001, "B=256 bound: {b256}");

        // Small batches are where skew genuinely can bite, and the bound must
        // NOT claim safety there — at B=16 one rank could own every woken
        // expert. This is the half that makes the test discriminating: a bound
        // that always returned ~1.05 would pass every assertion above.
        let d16 = expected_distinct_experts(256, 6, 16);
        let b16_n2 = weight_read_imbalance_bound(256, 2, d16);
        let b16_n8 = weight_read_imbalance_bound(256, 8, d16);
        assert!((b16_n2 - 2.0).abs() < 0.01, "B=16 N=2 bound: {b16_n2}");
        assert!(b16_n8 > 3.0, "B=16 N=8 bound: {b16_n8}");

        // EP=1 can never be imbalanced.
        assert_eq!(weight_read_imbalance_bound(256, 1, d128), 1.0);
    }

    #[test]
    fn balancedness_reports_mean_max_and_ratio() {
        let b = balancedness(&[10.0, 30.0]);
        assert_eq!(b.mean, 20.0);
        assert_eq!(b.max, 30.0);
        assert!((b.ratio - 1.5).abs() < 1e-12);
        // All-zero must not be NaN.
        assert_eq!(balancedness(&[0.0, 0.0]).ratio, 1.0);
    }

    /// The counter must attribute tokens by OWNER, not by expert id — a
    /// counter keyed on the raw id would report the same numbers for every
    /// placement and could never detect a skew.
    #[test]
    fn counter_attributes_tokens_and_distinct_experts_to_owners() {
        let placement = ExpertPlacement::from_expert_to_rank(vec![0, 1, 0, 1], 2).unwrap();
        let counter = BalancednessCounter::new(2);
        let dev = Device::Cpu;
        // Six slots: 0,0,2 → rank 0 (2 distinct experts); 1,1,1 → rank 1 (1 distinct).
        let ids = Tensor::from_vec(vec![0u32, 0, 2, 1, 1, 1], (2, 3), &dev).unwrap();
        counter.record(7, &ids, &placement).unwrap();

        assert_eq!(counter.tokens().per_rank, vec![3.0, 3.0]);
        assert_eq!(counter.distinct_experts().per_rank, vec![2.0, 1.0]);
        assert!((counter.distinct_experts().ratio - 4.0 / 3.0).abs() < 1e-12);

        // A second layer's identical routing adds new (layer, expert) pairs.
        counter.record(8, &ids, &placement).unwrap();
        assert_eq!(counter.distinct_experts().per_rank, vec![4.0, 2.0]);
        assert_eq!(counter.steps(), 2);
    }
}
