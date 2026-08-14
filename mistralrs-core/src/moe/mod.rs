mod experts;

use mistralrs_quant::Shard;

pub use experts::{MoEExperts, MoEExpertsConfig};
pub(crate) use experts::swiglu_clamp;

pub fn shard(dim: usize, rank: usize, world_size: usize) -> Shard {
    Shard::Simple {
        dim,
        rank,
        world_size,
    }
}
