use std::{
    str::FromStr,
    sync::{Arc, Mutex, MutexGuard},
};

use candle_core::{DType, Device, Result, Tensor};
use serde::{Deserialize, Serialize};

use super::config::{KvCacheLayout, ModelConfigLike};

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Default)]
#[cfg_attr(feature = "pyo3_macros", pyo3::pyclass(eq, eq_int))]
pub enum PagedCacheType {
    Auto,
    F8E4M3,
    /// TurboQuant default: 4-bit keys, 3-bit values (3.5 bits avg). Lossless quality.
    #[default]
    TurboQuant,
    /// TurboQuant balanced: 3-bit keys, 3-bit values (3.0 bits avg).
    TurboQuant3,
    /// TurboQuant aggressive: 3-bit keys, 2-bit values (2.5 bits avg).
    TurboQuantAggressive,
}

impl PagedCacheType {
    pub fn to_dtype(&self, act_dtype: DType) -> DType {
        match self {
            PagedCacheType::F8E4M3 => DType::F8E4M3,
            // TurboQuant bypasses paged attention and uses the normal cache path
            // with TurboQuantCache. This dtype is only used if paged attn is somehow
            // still active, in which case fall through to act_dtype.
            PagedCacheType::TurboQuant
            | PagedCacheType::TurboQuant3
            | PagedCacheType::TurboQuantAggressive => act_dtype,
            PagedCacheType::Auto => act_dtype,
        }
    }

    /// Whether this cache type uses TurboQuant compression.
    pub fn is_turboquant(&self) -> bool {
        matches!(
            self,
            PagedCacheType::TurboQuant
                | PagedCacheType::TurboQuant3
                | PagedCacheType::TurboQuantAggressive
        )
    }

    /// Get the TurboQuant preset for this cache type, if applicable.
    pub fn turboquant_preset(&self) -> Option<mistralrs_quant::turboquant::TurboQuantPreset> {
        match self {
            PagedCacheType::TurboQuant => {
                Some(mistralrs_quant::turboquant::TurboQuantPreset::Default)
            }
            PagedCacheType::TurboQuant3 => {
                Some(mistralrs_quant::turboquant::TurboQuantPreset::Balanced)
            }
            PagedCacheType::TurboQuantAggressive => {
                Some(mistralrs_quant::turboquant::TurboQuantPreset::Aggressive)
            }
            _ => None,
        }
    }
}

impl FromStr for PagedCacheType {
    type Err = String;
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "auto" => Ok(Self::Auto),
            "f8e4m3" => Ok(Self::F8E4M3),
            "turboquant" => Ok(Self::TurboQuant),
            "turboquant-3" => Ok(Self::TurboQuant3),
            "turboquant-aggressive" => Ok(Self::TurboQuantAggressive),
            other => Err(format!(
                "Unexpected `PagedCacheType`, got `{other}` but expected one of: \
                 `auto`, `f8e4m3`, `turboquant`, `turboquant-3`, `turboquant-aggressive`."
            )),
        }
    }
}

#[derive(Clone, Debug)]
pub struct CacheConfig {
    pub block_size: usize,
    pub num_gpu_blocks: usize,
    pub cache_type: PagedCacheType,
}

/// KV cache storage. Standard is (key_cache, value_cache).
/// TurboQuant adds norm tensors for the L2 norms stored alongside packed indices.
pub type KVCache = (Tensor, Tensor);

/// Optional TurboQuant norm tensors, stored per-layer alongside KVCache.
pub type TurboQuantNorms = (Tensor, Tensor); // (k_norms, v_norms)

pub struct CacheEngine {
    gpu_cache: Arc<Mutex<Vec<KVCache>>>,
    /// Per-layer norm tensors for TurboQuant. None if not using TurboQuant.
    turbo_norms: Option<Arc<Mutex<Vec<TurboQuantNorms>>>>,
}

impl CacheEngine {
    pub fn new(
        model_config: &dyn ModelConfigLike,
        cache_config: &CacheConfig,
        dtype: DType,
        device: &Device,
        layer_devices: Vec<Option<Device>>,
    ) -> Result<Self> {
        let is_turbo = cache_config.cache_type.is_turboquant();
        let cache_dtype = cache_config.cache_type.to_dtype(dtype);

        let turbo_norms = if is_turbo {
            // Allocate F16 norm tensors for each layer
            let mut norms = Vec::new();
            for dev in layer_devices
                .iter()
                .take(model_config.num_layers())
                .map(|x| x.as_ref().unwrap_or(device))
            {
                let norm_shape = (
                    cache_config.num_gpu_blocks,
                    model_config.num_kv_heads(),
                    cache_config.block_size,
                );
                let k_norms = Tensor::zeros(norm_shape, DType::F16, dev)?;
                let v_norms = Tensor::zeros(norm_shape, DType::F16, dev)?;
                norms.push((k_norms, v_norms));
            }
            Some(Arc::new(Mutex::new(norms)))
        } else {
            None
        };

        Ok(Self {
            gpu_cache: Arc::new(Mutex::new(Self::allocate_gpu_cache(
                model_config,
                cache_config,
                cache_dtype,
                device,
                layer_devices,
            )?)),
            turbo_norms,
        })
    }

    pub fn get_kv_cache(&self) -> MutexGuard<'_, Vec<KVCache>> {
        self.gpu_cache.lock().expect("KV cache mutex was poisoned")
    }

    pub fn get_turbo_norms(&self) -> Option<MutexGuard<'_, Vec<TurboQuantNorms>>> {
        self.turbo_norms
            .as_ref()
            .map(|n| n.lock().expect("TurboQuant norms mutex was poisoned"))
    }

    fn allocate_gpu_cache(
        model_config: &dyn ModelConfigLike,
        cache_config: &CacheConfig,
        dtype: DType,
        device: &Device,
        layer_devices: Vec<Option<Device>>,
    ) -> Result<Vec<KVCache>> {
        let kv_cache_layout = model_config.kv_cache_layout();
        let mut gpu_cache = Vec::new();

        for device in layer_devices
            .iter()
            .take(model_config.num_layers())
            .map(|x| x.as_ref().unwrap_or(device))
        {
            let (key_blocks, value_blocks) = match kv_cache_layout {
                KvCacheLayout::Standard => {
                    let key_block_shape = Self::calculate_key_block_shape(
                        model_config,
                        dtype,
                        cache_config.block_size,
                    );
                    let value_block_shape =
                        Self::calculate_value_block_shape(model_config, cache_config.block_size);
                    #[allow(unused)]
                    let key_blocks = if let Device::Metal(dev) = &device {
                        #[cfg(feature = "metal")]
                        {
                            use candle_core::{MetalStorage, Shape, Storage};

                            let elem_count = cache_config.num_gpu_blocks
                                * key_block_shape.0
                                * key_block_shape.1
                                * key_block_shape.2
                                * key_block_shape.3;
                            let buffer = dev.new_private_buffer(elem_count, dtype, "k_cache")?;
                            let storage = Storage::Metal(MetalStorage::new(
                                buffer,
                                dev.clone(),
                                elem_count,
                                dtype,
                            ));
                            Tensor::from((
                                storage,
                                Shape::from_dims(&[
                                    cache_config.num_gpu_blocks,
                                    key_block_shape.0,
                                    key_block_shape.1,
                                    key_block_shape.2,
                                    key_block_shape.3,
                                ]),
                            ))
                        }

                        #[cfg(not(feature = "metal"))]
                        {
                            unreachable!()
                        }
                    } else {
                        unsafe {
                            Tensor::empty(
                                (
                                    cache_config.num_gpu_blocks,
                                    key_block_shape.0,
                                    key_block_shape.1,
                                    key_block_shape.2,
                                    key_block_shape.3,
                                ),
                                dtype,
                                device,
                            )?
                        }
                    };
                    #[allow(unused)]
                    let value_blocks = if let Device::Metal(dev) = &device {
                        #[cfg(feature = "metal")]
                        {
                            use candle_core::{MetalStorage, Shape, Storage};

                            let elem_count = cache_config.num_gpu_blocks
                                * value_block_shape.0
                                * value_block_shape.1
                                * value_block_shape.2;
                            let buffer = dev.new_private_buffer(elem_count, dtype, "v_cache")?;
                            let storage = Storage::Metal(MetalStorage::new(
                                buffer,
                                dev.clone(),
                                elem_count,
                                dtype,
                            ));
                            Tensor::from((
                                storage,
                                Shape::from_dims(&[
                                    cache_config.num_gpu_blocks,
                                    value_block_shape.0,
                                    value_block_shape.1,
                                    value_block_shape.2,
                                ]),
                            ))
                        }

                        #[cfg(not(feature = "metal"))]
                        {
                            unreachable!()
                        }
                    } else {
                        unsafe {
                            Tensor::empty(
                                (
                                    cache_config.num_gpu_blocks,
                                    value_block_shape.0,
                                    value_block_shape.1,
                                    value_block_shape.2,
                                ),
                                dtype,
                                device,
                            )?
                        }
                    };
                    (key_blocks, value_blocks)
                }
                KvCacheLayout::Mla {
                    kv_lora_rank,
                    kpe_head_dim,
                } => {
                    #[allow(unused)]
                    let key_blocks = if let Device::Metal(dev) = &device {
                        #[cfg(feature = "metal")]
                        {
                            use candle_core::{MetalStorage, Shape, Storage};

                            let elem_count = cache_config.num_gpu_blocks
                                * cache_config.block_size
                                * kv_lora_rank;
                            let buffer = dev.new_private_buffer(elem_count, dtype, "k_cache")?;
                            let storage = Storage::Metal(MetalStorage::new(
                                buffer,
                                dev.clone(),
                                elem_count,
                                dtype,
                            ));
                            Tensor::from((
                                storage,
                                Shape::from_dims(&[
                                    cache_config.num_gpu_blocks,
                                    cache_config.block_size,
                                    kv_lora_rank,
                                ]),
                            ))
                        }

                        #[cfg(not(feature = "metal"))]
                        {
                            unreachable!()
                        }
                    } else {
                        unsafe {
                            Tensor::empty(
                                (
                                    cache_config.num_gpu_blocks,
                                    cache_config.block_size,
                                    kv_lora_rank,
                                ),
                                dtype,
                                device,
                            )?
                        }
                    };
                    #[allow(unused)]
                    let value_blocks = if let Device::Metal(dev) = &device {
                        #[cfg(feature = "metal")]
                        {
                            use candle_core::{MetalStorage, Shape, Storage};

                            let elem_count = cache_config.num_gpu_blocks
                                * cache_config.block_size
                                * kpe_head_dim;
                            let buffer = dev.new_private_buffer(elem_count, dtype, "v_cache")?;
                            let storage = Storage::Metal(MetalStorage::new(
                                buffer,
                                dev.clone(),
                                elem_count,
                                dtype,
                            ));
                            Tensor::from((
                                storage,
                                Shape::from_dims(&[
                                    cache_config.num_gpu_blocks,
                                    cache_config.block_size,
                                    kpe_head_dim,
                                ]),
                            ))
                        }

                        #[cfg(not(feature = "metal"))]
                        {
                            unreachable!()
                        }
                    } else {
                        unsafe {
                            Tensor::empty(
                                (
                                    cache_config.num_gpu_blocks,
                                    cache_config.block_size,
                                    kpe_head_dim,
                                ),
                                dtype,
                                device,
                            )?
                        }
                    };
                    (key_blocks, value_blocks)
                }
            };
            gpu_cache.push((key_blocks, value_blocks));
        }
        Ok(gpu_cache)
    }

    fn calculate_key_block_shape(
        model_config: &dyn ModelConfigLike,
        dtype: DType,
        block_size: usize,
    ) -> (usize, usize, usize, usize) {
        let element_size = dtype.size_in_bytes();
        let x = 16 / element_size;
        (
            model_config.num_kv_heads(),
            model_config.k_head_dim() / x,
            block_size,
            x,
        )
    }

    fn calculate_value_block_shape(
        model_config: &dyn ModelConfigLike,
        block_size: usize,
    ) -> (usize, usize, usize) {
        (
            model_config.num_kv_heads(),
            model_config.v_head_dim(),
            block_size,
        )
    }
}
