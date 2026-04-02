/// Content-addressable block hashing for prefix caching (vLLM v1 approach).
pub mod block_hash;
/// Flat block pool with LRU free list for KV cache block management (vLLM v1 approach).
pub mod block_pool;
/// This is the lower-level manager of the cache. It manages swapping and copying the blocks and
/// actually allocates the KV cache for the CPU and GPU. It is used by the LLMEngine to execute
/// operations issued by the scheduler.
mod cache_engine;
mod config;
/// Encoder output cache for multimodal models (vision/audio encoder outputs).
pub mod encoder_cache;
/// KV Cache Manager: high-level block allocation, prefix cache lookups, per-request tracking.
pub mod kv_cache_manager;
pub(crate) mod layers;
mod scheduler;
pub const _PAD_SLOT_ID: i64 = -1;

pub use cache_engine::{CacheConfig, CacheEngine, PagedCacheType};
use candle_core::{DType, Device, Tensor};
use std::sync::Mutex;

// Global TurboQuant norms registry
static TURBO_NORMS_GLOBAL: Mutex<Vec<(Tensor, Tensor)>> = Mutex::new(Vec::new());

pub fn set_global_turbo_norms(norms: Vec<(Tensor, Tensor)>) {
    *TURBO_NORMS_GLOBAL.lock().unwrap() = norms;
}

pub fn clear_global_turbo_norms() {
    TURBO_NORMS_GLOBAL.lock().unwrap().clear();
}

pub(crate) fn get_global_turbo_norms(layer_idx: usize) -> Option<(Tensor, Tensor)> {
    let norms = TURBO_NORMS_GLOBAL.lock().unwrap();
    if layer_idx < norms.len() {
        Some((norms[layer_idx].0.clone(), norms[layer_idx].1.clone()))
    } else {
        None
    }
}

pub(crate) fn has_global_turbo_norms() -> bool {
    !TURBO_NORMS_GLOBAL.lock().unwrap().is_empty()
}
pub use config::{KvCacheLayout, ModelConfigLike, ModelConfigMetadata};
pub use kv_cache_manager::KVCacheManager;
pub use layers::PagedAttention;
pub use scheduler::{
    PagedAttentionScheduler, PagedAttentionSchedulerConfig, PagedAttentionSchedulerOutput,
};

use crate::MemoryUsage;
use tracing::info;

pub const DEFAULT_PAGED_ATTENTION_BLOCK_SIZE: usize = 32;

/// All memory counts in MB. Default for block size is 32.
#[derive(Clone, Copy)]
pub struct PagedAttentionConfig {
    pub(crate) block_size: Option<usize>,
    pub(crate) mem_gpu: MemoryGpuConfig,
    pub(crate) cache_type: PagedCacheType,
}

impl PagedAttentionConfig {
    pub fn new(
        block_size: Option<usize>,
        mem_gpu: MemoryGpuConfig,
        cache_type: PagedCacheType,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            block_size,
            mem_gpu,
            cache_type,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AttentionImplementation {
    Eager,
    PagedAttention,
}

#[derive(Clone, Copy)]
#[cfg_attr(feature = "pyo3_macros", pyo3::pyclass)]
pub enum MemoryGpuConfig {
    MbAmount(usize),
    Utilization(f32),
    ContextSize(usize),
}

// See `pagedattention.cu` CALL_V1_LAUNCHER_BLOCK_SIZE
const SUPPORTED_BLOCK_SIZE: &[usize] = &[8, 16, 32];

const SIZE_IN_MB: usize = 1024 * 1024;

macro_rules! mb_to_blocks {
    ($mb_size:expr, $dtype_size:expr, $block_size:expr, $config:expr) => {
        $mb_size
            / $dtype_size
            / $block_size
            / $config.num_layers()
            / $config.kv_cache_elements_per_token()
    };
}

macro_rules! ctxt_to_blocks {
    ($context_len:expr, $dtype_size:expr, $block_size:expr, $config:expr) => {
        $context_len * $dtype_size * $config.num_layers() * $config.kv_cache_elements_per_token()
    };
}

/// Memory values are in MBs or a percentage in [0,1]. Specify block size or the default is 32.
///
/// `model_weight_size_in_bytes`: total model weight footprint. When provided, the per-device
/// share (divided by number of devices for tensor parallelism) is subtracted from the KV cache
/// memory budget. Pass `Some(total_model_size_in_bytes)` when calling **before** model loading
/// (e.g. during device mapping) so the KV cache estimate reflects memory that will actually
/// remain after the weights are loaded. Post-loading callers should pass `None` since
/// `get_memory_available()` already reflects the loaded model.
///
/// `max_num_tokens`: on Metal (unified memory), caps the KV cache to this many tokens.
/// Unlike CUDA with dedicated VRAM where unused memory is wasted, Metal's wired buffers
/// compete with the OS and CPU for the same physical RAM. On CUDA this is ignored.
/// If `None` on Metal, falls back to `config.max_seq_len()`.
#[allow(clippy::too_many_arguments)]
pub fn calculate_cache_config(
    mem_gpu: MemoryGpuConfig,
    block_size: Option<usize>,
    dtype: DType,
    cache_type: PagedCacheType,
    config: &dyn ModelConfigLike,
    device: &Device,
    layer_devices: &[Option<Device>],
    silent: bool,
    model_weight_size_in_bytes: Option<usize>,
    max_num_tokens: Option<usize>,
) -> anyhow::Result<CacheConfig> {
    let block_size = block_size.unwrap_or(DEFAULT_PAGED_ATTENTION_BLOCK_SIZE);
    if !SUPPORTED_BLOCK_SIZE.contains(&block_size) {
        anyhow::bail!("Block size must be in {SUPPORTED_BLOCK_SIZE:?}, got {block_size}");
    }
    let dtype = cache_type.to_dtype(dtype);
    let dtype_size = dtype.size_in_bytes();

    // For TurboQuant, override the effective bytes-per-element to account for
    // sub-byte packing. The block shapes are already packed (K: 64 bytes/head,
    // V: 52 bytes/head for d=128), but mb_to_blocks uses dtype_size * elements
    // which overcounts. We compute an effective dtype_size that gives the correct
    // total bytes when multiplied by kv_cache_elements_per_token.
    //
    // Actual packed bytes per token = num_kv_heads * (k_packed + v_packed + 4 norms)
    // kv_cache_elements_per_token = 2 * num_kv_heads * head_dim
    // effective_dtype_size = actual_bytes / elements_per_token
    //
    // For head_dim=128: packed = 8*(64+52+4) = 960, elements = 2*8*128 = 2048
    // effective = 960/2048 ≈ 0.47, but dtype_size is usize so we can't use fractions.
    //
    // Instead, directly compute num_gpu_blocks for TurboQuant.
    let turboquant_bytes_per_token_per_layer = if cache_type.is_turboquant() {
        let kv_heads = config.num_kv_heads();
        let k_packed = config.k_head_dim() / 2; // 4-bit: 64 bytes
        let v_packed = (config.v_head_dim().div_ceil(10)) * 4; // 3-bit: 52 bytes
        let norms = 2 * 2; // 2 F16 norms = 4 bytes
        Some(kv_heads * (k_packed + v_packed + norms))
    } else {
        None
    };

    // For tensor parallelism, each device holds a fraction of the model weights. Approximate it like this.
    let num_devices = layer_devices.len().max(1);
    let model_weight_per_device_mb =
        model_weight_size_in_bytes.unwrap_or(0) / num_devices / SIZE_IN_MB;

    let mut min_mem_gpu = usize::MAX;
    for dev in layer_devices {
        let device = dev.as_ref().unwrap_or(device);

        #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
        let mem_gpu = match mem_gpu {
            MemoryGpuConfig::MbAmount(v) => v,
            MemoryGpuConfig::Utilization(f) => {
                let total = MemoryUsage.get_total_memory(device)? as f32 / SIZE_IN_MB as f32;
                if model_weight_size_in_bytes.is_some() {
                    // Pre-loading: compute budget from total memory and known model size.
                    (total * f - model_weight_per_device_mb as f32).max(0.0) as usize
                } else {
                    let free = MemoryUsage.get_memory_available(device)? as f32 / SIZE_IN_MB as f32;
                    #[allow(unused_mut)]
                    let mut used = total - free;
                    // On Metal, get_total_memory (wired limit) and get_memory_available
                    // (recommendedMaxWorkingSetSize - allocated) have different bases,
                    // so `total - free` is incorrect. Use the device's tracked
                    // allocation size directly.
                    #[cfg(feature = "metal")]
                    if let Device::Metal(dev) = device {
                        used = dev.current_allocated_size() as f32 / SIZE_IN_MB as f32;
                    }
                    (total * f - used).max(0.0) as usize
                }
            }
            MemoryGpuConfig::ContextSize(toks) => {
                // ContextSize is demand-driven (bytes needed for N tokens), not a memory budget, so model weight does not apply here.
                ctxt_to_blocks!(toks, dtype_size, block_size, config) / SIZE_IN_MB
            }
        };
        min_mem_gpu = min_mem_gpu.min(mem_gpu);
    }

    // On Metal (unified memory), cap KV cache to what the model can actually use.
    // Unlike CUDA with dedicated VRAM where unused memory is wasted, Metal's wired
    // buffers compete with the OS and CPU for the same physical RAM.
    // On CUDA, all available memory is used for maximum request concurrency (vLLM approach).
    #[allow(unused_mut, unused_variables)]
    let mut mem_gpu = min_mem_gpu;
    if device.is_metal() {
        let max_tokens = max_num_tokens.unwrap_or(config.max_seq_len());
        let mem_for_tokens =
            ctxt_to_blocks!(max_tokens, dtype_size, block_size, config) / SIZE_IN_MB;
        if mem_for_tokens < mem_gpu {
            if !silent {
                info!(
                    "Metal: capping KV cache from {} MB to {} MB ({} tokens).",
                    mem_gpu, mem_for_tokens, max_tokens
                );
            }
            mem_gpu = mem_for_tokens;
        }
    }

    let num_gpu_blocks = if let Some(bytes_per_tok_per_layer) = turboquant_bytes_per_token_per_layer {
        // TurboQuant: compute blocks from actual packed byte sizes
        // total_bytes = num_blocks * block_size * num_layers * bytes_per_token_per_layer
        // num_blocks = total_bytes / (block_size * num_layers * bytes_per_token_per_layer)
        (mem_gpu * SIZE_IN_MB) / (block_size * config.num_layers() * bytes_per_tok_per_layer)
    } else {
        mb_to_blocks!(mem_gpu * SIZE_IN_MB, dtype_size, block_size, config)
    };
    if num_gpu_blocks == 0 {
        anyhow::bail!("Num GPU blocks is 0. This means there is not enough memory. Either reduce the memory amount/utilization/context size or disable PagedAttention.");
    }

    if !silent {
        info!("Allocating {mem_gpu} MB for PagedAttention KV cache per GPU");
        info!("PagedAttention KV cache type is {dtype:?}");
        info!("Using PagedAttention with block size {block_size} and {num_gpu_blocks} GPU blocks: available context length is {} tokens", num_gpu_blocks*block_size);
    }
    Ok(CacheConfig {
        block_size,
        num_gpu_blocks,
        cache_type,
    })
}
