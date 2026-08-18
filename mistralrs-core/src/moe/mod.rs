mod expert_parallel;
mod experts;

use mistralrs_quant::Shard;

pub use expert_parallel::{
    balancedness, expected_distinct_experts, weight_read_imbalance_bound, Balancedness,
    BalancednessCounter, ExpertParallelPlan, ExpertPlacement,
};
pub(crate) use experts::swiglu_clamp;
pub use experts::{MoEExperts, MoEExpertsConfig};

pub fn shard(dim: usize, rank: usize, world_size: usize) -> Shard {
    Shard::Simple {
        dim,
        rank,
        world_size,
    }
}
