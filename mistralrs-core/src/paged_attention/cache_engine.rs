use std::{
    str::FromStr,
    sync::{Arc, Mutex, MutexGuard, Once},
};

use candle_core::{DType, Device, Result, Tensor};
use serde::{Deserialize, Serialize};

use super::config::{KvCacheLayout, ModelConfigLike};

/// Head dimensions the TurboQuant CUDA kernels are instantiated for.
///
/// Re-exported from `mistralrs_quant::turboquant::cuda_tables`, which is also
/// where the unit test lives that pins this list to the `case` arms of the
/// kernel's own dispatch. Keeping one list on both sides of the FFI is what
/// makes an unsupported width a refusal instead of an untouched output buffer.
pub use mistralrs_quant::turboquant::TURBOQUANT_CUDA_HEAD_DIMS;

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Default)]
#[cfg_attr(feature = "pyo3_macros", pyo3::pyclass(eq, eq_int))]
pub enum PagedCacheType {
    Auto,
    F8E4M3,
    /// TurboQuant default: 4-bit keys, 3-bit values (3.5 bits avg).
    ///
    /// This is the `#[default]`, and it is the default the server actually
    /// resolves to (`defaults::PAGED_CACHE_TYPE`) when `--pa-cache-type` is
    /// left unset — so a standard-layout head_dim-128 model on CUDA gets it
    /// without asking. It has one end-to-end serving run behind it (Qwen3-32B
    /// on a B200, 55 tok/s, correct output, 2026-04-06); it has **no quality
    /// evaluation** at any width. Earlier revisions of this comment read
    /// "Lossless quality" — that was never measured and has been removed.
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
            // TurboQuant stores quantized indices as U8 in paged cache blocks.
            PagedCacheType::TurboQuant
            | PagedCacheType::TurboQuant3
            | PagedCacheType::TurboQuantAggressive => DType::U8,
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

    /// Whether prefix caching can be used with this cache type. TurboQuant
    /// blocks are packed 4-bit K / 3-bit V codebook indices with the
    /// per-token norms held in separate tensors
    /// (`turbo_paged_attention.cuh:15-19`); `gather_kv_cache` has no dequantize
    /// path and no norms argument, so a prefix-cache hit cannot be served.
    pub fn supports_prefix_cache(&self) -> bool {
        !self.is_turboquant()
    }

    /// The user-facing account of the TurboQuant / prefix-cache conflict, or
    /// `None` when there is no conflict.
    ///
    /// 🔑 Two things Arc sells cannot both be switched on, and until this
    /// existed the loser was chosen for the user in one line of `tracing::warn`
    /// that named **neither flag**. That matters more here than it would
    /// elsewhere, because **TurboQuant is what an untouched command line
    /// resolves to**: `--pa-cache-type` unset means TurboQuant, not `auto`
    /// (`mistralrs-cli/src/args/paged_attn.rs:44-53`), PagedAttention is the
    /// CUDA default, and `--prefix-cache-n` defaults to 16
    /// (`mistralrs-cli/src/args/mod.rs:425`). So the default configuration is
    /// the conflicting one, and the message has to say which flag produced it
    /// and which flag it silently overrode.
    ///
    /// `prefix_cache_n` is carried so the message can distinguish "you asked
    /// for prefix caching and lost it" from the degenerate case where prefix
    /// caching was off anyway — there is nothing to warn about in the latter.
    pub fn prefix_cache_conflict(&self, prefix_cache_n: usize) -> Option<String> {
        if self.supports_prefix_cache() || prefix_cache_n == 0 {
            return None;
        }
        Some(format!(
            "PagedAttention KV cache type is `{self}` (TurboQuant) and prefix caching was \
             requested with `--prefix-cache-n {prefix_cache_n}`. These are mutually exclusive \
             today: TurboQuant blocks are packed codebook indices with their norms in separate \
             tensors, and `gather_kv_cache` has no dequantize-on-gather path, so a prefix-cache \
             hit cannot be served. PREFIX CACHING IS NOW DISABLED for this run; the compressed \
             KV cache is kept. Note that `--pa-cache-type` unset means TurboQuant, NOT `auto` — \
             if you would rather keep prefix caching, pass `--pa-cache-type auto` (or \
             `f8e4m3`); if you would rather keep the compressed cache and silence this, pass \
             `--prefix-cache-n 0`."
        ))
    }

    /// Head dims the **ambient default** is allowed to route onto TurboQuant.
    ///
    /// Deliberately narrower than [`TURBOQUANT_CUDA_HEAD_DIMS`]. Compiling a
    /// kernel instantiation proves it *builds*; it does not prove it produces
    /// correct KV on hardware, and none of 64/256/512 has ever executed. Since
    /// [`PagedCacheType::TurboQuant`] is `#[default]`, widening the gate alone
    /// would have silently moved every standard-layout model at those widths
    /// onto unproven kernels — and silently dropped prefix caching with it
    /// ([`Self::supports_prefix_cache`] is false for every TurboQuant variant).
    /// Two regressions, neither visible at the call site.
    ///
    /// This repo has paid for that exact shape once already: FP8 KV shipped
    /// default-on and unmeasured (wave43-BU) and every V4 request died.
    ///
    /// So: **compiled** widens what you may ask for; **measured** widens what
    /// you get without asking. Move a width here when its hardware gate has
    /// passed, not when its kernel compiles.
    const TURBOQUANT_DEFAULT_HEAD_DIMS: [usize; 1] = [128];

    /// Whether the TurboQuant kernels support this model's KV geometry.
    ///
    /// K and V are compressed independently, so they only have to be widths the
    /// kernels are instantiated for — they no longer have to be *the same*
    /// width, and no longer have to be 128.
    ///
    /// `forced` selects which set applies: an explicit request may use any
    /// instantiated width, the ambient default only a measured one.
    fn turboquant_supports_model(config: &dyn ModelConfigLike, forced: bool) -> bool {
        let accepts = |head_dim: usize| {
            if forced {
                mistralrs_quant::turboquant::cuda_supports_head_dim(head_dim)
            } else {
                Self::TURBOQUANT_DEFAULT_HEAD_DIMS.contains(&head_dim)
            }
        };
        matches!(config.kv_cache_layout(), KvCacheLayout::Standard)
            && accepts(config.k_head_dim())
            && accepts(config.v_head_dim())
    }

    /// Resolve this cache type against the target model's KV geometry.
    ///
    /// The TurboQuant kernels support the standard KV layout at any head
    /// dimension in [`TURBOQUANT_CUDA_HEAD_DIMS`]:
    /// * a head size with no instantiation falls through the kernel's dispatch
    ///   switch, leaving the (uninitialized) output buffer untouched — silent
    ///   garbage, which is why this gate runs before any allocation;
    /// * MLA-layout models (DeepSeek V2/V3, GLM4-MoE-lite) write through
    ///   `concat_and_cache_mla`, which bails on a packed U8 cache at runtime.
    ///
    /// For the ambient default (`TurboQuant` the user never asked for), fall
    /// back to [`PagedCacheType::Auto`] with a single warning. If the user
    /// explicitly chose a TurboQuant type (`explicitly_requested`, or one of
    /// the non-default `TurboQuant3`/`TurboQuantAggressive` presets, which are
    /// never a default anywhere), error instead so the mismatch cannot be
    /// missed.
    pub fn resolve_for_model(
        self,
        config: &dyn ModelConfigLike,
        explicitly_requested: bool,
    ) -> anyhow::Result<Self> {
        // `TurboQuant3`/`TurboQuantAggressive` are never a default anywhere, so
        // selecting one is always an explicit act. Computed before the support
        // check because it decides *which* set of head dims applies.
        let forced = explicitly_requested || !matches!(self, PagedCacheType::TurboQuant);
        if !self.is_turboquant() || Self::turboquant_supports_model(config, forced) {
            return Ok(self);
        }
        let reason = match config.kv_cache_layout() {
            KvCacheLayout::Mla { .. } => {
                "the model uses an MLA KV cache layout, which TurboQuant does not support"
                    .to_string()
            }
            // Split deliberately: "no kernel exists" and "a kernel exists but has
            // never run on hardware, so the default will not choose it for you"
            // are different facts, and collapsing them would tell a user to stop
            // trying when the real answer is "opt in explicitly".
            KvCacheLayout::Standard
                if mistralrs_quant::turboquant::cuda_supports_head_dim(config.k_head_dim())
                    && mistralrs_quant::turboquant::cuda_supports_head_dim(config.v_head_dim()) =>
            {
                format!(
                    "the model has head_dim k={}/v={}: the TurboQuant kernels are instantiated \
                     for it, but only head_dim in {:?} has been measured on hardware, so the \
                     default will not route onto it. Pass `--pa-cache-type turboquant` to opt in \
                     deliberately",
                    config.k_head_dim(),
                    config.v_head_dim(),
                    Self::TURBOQUANT_DEFAULT_HEAD_DIMS,
                )
            }
            KvCacheLayout::Standard => format!(
                "the model has head_dim k={}/v={}, but the TurboQuant kernels are instantiated \
                 for head_dim in {TURBOQUANT_CUDA_HEAD_DIMS:?}",
                config.k_head_dim(),
                config.v_head_dim(),
            ),
        };
        if forced {
            anyhow::bail!(
                "PagedAttention cache type {self:?} was explicitly requested, but {reason}. \
                 Use `--pa-cache-type auto` or `--pa-cache-type f8e4m3` for this model."
            );
        }
        static FALLBACK_WARNING: Once = Once::new();
        FALLBACK_WARNING.call_once(|| {
            tracing::warn!(
                "Default PagedAttention cache type {self:?} is unsupported here: {reason}. \
                 Falling back to the unquantized KV cache (`auto`)."
            );
        });
        Ok(PagedCacheType::Auto)
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

/// Renders as the exact `--pa-cache-type` spelling, so a diagnostic that names
/// the active cache type names a value the user can actually pass back.
/// Round-tripped against [`FromStr`] by
/// `every_cache_type_renders_as_the_flag_value_that_parses_back`.
impl std::fmt::Display for PagedCacheType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Auto => "auto",
            Self::F8E4M3 => "f8e4m3",
            Self::TurboQuant => "turboquant",
            Self::TurboQuant3 => "turboquant-3",
            Self::TurboQuantAggressive => "turboquant-aggressive",
        })
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

/// KV cache storage per layer: (key_cache, value_cache, optional_k_norms, optional_v_norms).
/// Standard paged attention: k_norms and v_norms are None.
/// TurboQuant: k_norms and v_norms hold per-block per-head per-slot F16 norms.
pub type KVCache = (Tensor, Tensor, Option<Tensor>, Option<Tensor>);

pub struct CacheEngine {
    gpu_cache: Arc<Mutex<Vec<KVCache>>>,
}

impl CacheEngine {
    pub fn new(
        model_config: &dyn ModelConfigLike,
        cache_config: &CacheConfig,
        dtype: DType,
        device: &Device,
        layer_devices: Vec<Option<Device>>,
    ) -> Result<Self> {
        let cache_dtype = cache_config.cache_type.to_dtype(dtype);
        let is_turbo = cache_config.cache_type.is_turboquant();

        let mut gpu_cache = Self::allocate_gpu_cache(
            model_config,
            cache_config,
            cache_dtype,
            device,
            layer_devices.clone(),
        )?;

        // For TurboQuant, add norm tensors to each layer's cache entry
        if is_turbo {
            for (i, dev) in layer_devices
                .iter()
                .take(model_config.num_layers())
                .map(|x| x.as_ref().unwrap_or(device))
                .enumerate()
            {
                let norm_shape = (
                    cache_config.num_gpu_blocks,
                    model_config.num_kv_heads(),
                    cache_config.block_size,
                );
                let k_norms = Tensor::zeros(norm_shape, DType::F16, dev)?;
                let v_norms = Tensor::zeros(norm_shape, DType::F16, dev)?;
                gpu_cache[i].2 = Some(k_norms);
                gpu_cache[i].3 = Some(v_norms);
            }
        }

        Ok(Self {
            gpu_cache: Arc::new(Mutex::new(gpu_cache)),
        })
    }

    pub fn get_kv_cache(&self) -> MutexGuard<'_, Vec<KVCache>> {
        self.gpu_cache.lock().expect("KV cache mutex was poisoned")
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
            let (key_blocks, value_blocks, k_norms, v_norms) = match kv_cache_layout {
                KvCacheLayout::Standard => {
                    let key_block_shape = Self::calculate_key_block_shape(
                        model_config,
                        dtype,
                        cache_config.block_size,
                        &cache_config.cache_type,
                    );
                    let value_block_shape = Self::calculate_value_block_shape(
                        model_config,
                        cache_config.block_size,
                        &cache_config.cache_type,
                    );
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
                    (key_blocks, value_blocks, None, None)
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
                    (key_blocks, value_blocks, None, None)
                }
            };
            gpu_cache.push((key_blocks, value_blocks, k_norms, v_norms));
        }
        Ok(gpu_cache)
    }

    fn calculate_key_block_shape(
        model_config: &dyn ModelConfigLike,
        dtype: DType,
        block_size: usize,
        cache_type: &PagedCacheType,
    ) -> (usize, usize, usize, usize) {
        if cache_type.is_turboquant() {
            // TurboQuant 4-bit packed: 2 values per byte = head_dim/2 bytes per head
            // Store as (num_kv_heads, packed_bytes/16, block_size, 16) to keep 5D
            let packed_bytes = model_config.k_head_dim() / 2; // 4-bit: 64 bytes for d=128
            let x = 16usize;
            (
                model_config.num_kv_heads(),
                packed_bytes / x, // 64/16 = 4
                block_size,
                x,
            )
        } else {
            let element_size = dtype.size_in_bytes();
            let x = 16 / element_size;
            (
                model_config.num_kv_heads(),
                model_config.k_head_dim() / x,
                block_size,
                x,
            )
        }
    }

    fn calculate_value_block_shape(
        model_config: &dyn ModelConfigLike,
        block_size: usize,
        cache_type: &PagedCacheType,
    ) -> (usize, usize, usize) {
        if cache_type.is_turboquant() {
            // TurboQuant 3-bit packed: 10 values per 4 bytes = ceil(head_dim/10)*4 bytes
            let packed_bytes = (model_config.v_head_dim().div_ceil(10)) * 4; // 52 for d=128
            (model_config.num_kv_heads(), packed_bytes, block_size)
        } else {
            (
                model_config.num_kv_heads(),
                model_config.v_head_dim(),
                block_size,
            )
        }
    }
}

#[cfg(test)]
mod cache_type_tests {
    use super::super::config::ModelConfigMetadata;
    use super::*;

    fn meta(k_head_dim: usize, v_head_dim: usize, layout: KvCacheLayout) -> ModelConfigMetadata {
        ModelConfigMetadata {
            max_seq_len: 4096,
            num_layers: 2,
            hidden_size: 512,
            num_kv_heads: 2,
            num_attn_heads: 4,
            sliding_window: None,
            k_head_dim,
            v_head_dim,
            kv_cache_layout: layout,
        }
    }

    const TURBO_TYPES: [PagedCacheType; 3] = [
        PagedCacheType::TurboQuant,
        PagedCacheType::TurboQuant3,
        PagedCacheType::TurboQuantAggressive,
    ];

    /// Supported geometry, **explicitly requested**: every TurboQuant preset
    /// resolves to itself at any head dim the kernels are instantiated for.
    ///
    /// Driven off `TURBOQUANT_CUDA_HEAD_DIMS` rather than a literal, so adding
    /// a kernel instantiation extends this test automatically instead of
    /// leaving the new width silently uncovered.
    #[test]
    fn explicit_turboquant_stays_on_any_instantiated_geometry() {
        for head_dim in TURBOQUANT_CUDA_HEAD_DIMS {
            let cfg = meta(head_dim, head_dim, KvCacheLayout::Standard);
            for t in TURBO_TYPES {
                assert_eq!(
                    t.resolve_for_model(&cfg, true).unwrap(),
                    t,
                    "head_dim={head_dim} is instantiated and an explicit request must be kept"
                );
            }
        }
    }

    /// **The regression this PR would otherwise have shipped.**
    ///
    /// `PagedCacheType::TurboQuant` is `#[default]`. Widening the acceptance
    /// gate from `head_dim == 128` to every instantiated width would have moved
    /// standard-layout models at 64/256/512 onto kernels that have never
    /// executed, and silently dropped prefix caching with them — without anyone
    /// asking for TurboQuant at all.
    ///
    /// The ambient default must therefore keep falling back to `Auto` at every
    /// instantiated-but-unmeasured width, while the same width stays available
    /// on explicit request (covered above).
    #[test]
    fn default_turboquant_does_not_route_onto_unmeasured_head_dims() {
        // Non-vacuity guard. This loop skips every width that IS a default, so
        // widening `TURBOQUANT_DEFAULT_HEAD_DIMS` to the full instantiated set
        // would empty the loop and make the test pass while asserting nothing —
        // which is precisely the silent-success shape this test exists to catch.
        // Caught by mutating the const during review; keep this assertion.
        let unmeasured: Vec<usize> = TURBOQUANT_CUDA_HEAD_DIMS
            .into_iter()
            .filter(|hd| !PagedCacheType::TURBOQUANT_DEFAULT_HEAD_DIMS.contains(hd))
            .collect();
        assert!(
            !unmeasured.is_empty(),
            "every instantiated head dim is now a default width, so this test asserts \
             nothing. If a hardware gate genuinely measured them all, delete this test \
             deliberately — do not let it pass vacuously."
        );

        for head_dim in unmeasured {
            let cfg = meta(head_dim, head_dim, KvCacheLayout::Standard);
            assert_eq!(
                PagedCacheType::TurboQuant
                    .resolve_for_model(&cfg, false)
                    .unwrap(),
                PagedCacheType::Auto,
                "head_dim={head_dim} is instantiated but unmeasured; the DEFAULT must not \
                 route onto it"
            );
        }
    }

    /// Every measured width must still be taken by the ambient default —
    /// the narrowing must not become "TurboQuant is off by accident".
    #[test]
    fn default_turboquant_still_taken_on_measured_head_dims() {
        for head_dim in PagedCacheType::TURBOQUANT_DEFAULT_HEAD_DIMS {
            assert!(
                TURBOQUANT_CUDA_HEAD_DIMS.contains(&head_dim),
                "head_dim={head_dim} is a default width but has no kernel instantiation"
            );
            let cfg = meta(head_dim, head_dim, KvCacheLayout::Standard);
            assert_eq!(
                PagedCacheType::TurboQuant
                    .resolve_for_model(&cfg, false)
                    .unwrap(),
                PagedCacheType::TurboQuant,
                "head_dim={head_dim} is measured and must still be the default"
            );
        }
    }

    /// DeepSeek-V4's width specifically. It is the reason the 128-only kernel
    /// limit was lifted, so it gets its own named assertion — but on the
    /// *explicit* path, because 512 has not been measured.
    #[test]
    fn turboquant_accepts_v4_head_dim_512_when_asked_for() {
        assert!(
            TURBOQUANT_CUDA_HEAD_DIMS.contains(&512),
            "512 must stay in the instantiated set"
        );
        let cfg = meta(512, 512, KvCacheLayout::Standard);
        assert_eq!(
            PagedCacheType::TurboQuant
                .resolve_for_model(&cfg, true)
                .unwrap(),
            PagedCacheType::TurboQuant
        );
        // ...and is NOT taken by the ambient default.
        assert_eq!(
            PagedCacheType::TurboQuant
                .resolve_for_model(&cfg, false)
                .unwrap(),
            PagedCacheType::Auto
        );
    }

    /// K and V are compressed independently, so mixed widths are fine as long
    /// as each is instantiated — on the explicit path.
    #[test]
    fn turboquant_accepts_mixed_but_instantiated_head_dims() {
        let cfg = meta(512, 128, KvCacheLayout::Standard);
        assert_eq!(
            PagedCacheType::TurboQuant
                .resolve_for_model(&cfg, true)
                .unwrap(),
            PagedCacheType::TurboQuant
        );
    }

    /// The default TurboQuant type falls back to Auto (instead of falling
    /// through the kernel's dispatch switch and leaving the output buffer
    /// untouched) for head dims with no instantiation.
    #[test]
    fn default_turboquant_falls_back_on_unsupported_head_dim() {
        for head_dim in [96, 192, 320, 1024] {
            let cfg = meta(head_dim, head_dim, KvCacheLayout::Standard);
            assert_eq!(
                PagedCacheType::TurboQuant
                    .resolve_for_model(&cfg, false)
                    .unwrap(),
                PagedCacheType::Auto,
                "head_dim={head_dim} must fall back"
            );
        }
        // A mixed pair still falls back when either half is uninstantiated.
        let cfg = meta(128, 192, KvCacheLayout::Standard);
        assert_eq!(
            PagedCacheType::TurboQuant
                .resolve_for_model(&cfg, false)
                .unwrap(),
            PagedCacheType::Auto
        );
    }

    /// MLA-layout models (deepseek2/3, glm4_moe_lite) get a U8 MLA cache that
    /// `concat_and_cache_mla` rejects at runtime; the default falls back.
    #[test]
    fn default_turboquant_falls_back_on_mla_layout() {
        let cfg = meta(
            192,
            192,
            KvCacheLayout::Mla {
                kv_lora_rank: 512,
                kpe_head_dim: 64,
            },
        );
        assert_eq!(
            PagedCacheType::TurboQuant
                .resolve_for_model(&cfg, false)
                .unwrap(),
            PagedCacheType::Auto
        );
    }

    /// An explicitly requested TurboQuant type must hard-error on unsupported
    /// models rather than silently switching caches.
    #[test]
    fn explicit_turboquant_errors_on_unsupported_model() {
        let cfg = meta(192, 192, KvCacheLayout::Standard);
        for t in TURBO_TYPES {
            let err = t.resolve_for_model(&cfg, true).unwrap_err();
            assert!(
                err.to_string().contains("explicitly requested"),
                "unexpected error: {err}"
            );
        }
    }

    /// TurboQuant3/TurboQuantAggressive are never a default anywhere, so they
    /// count as explicit even when the caller could not plumb the flag.
    #[test]
    fn non_default_turbo_presets_are_treated_as_explicit() {
        let cfg = meta(
            192,
            192,
            KvCacheLayout::Mla {
                kv_lora_rank: 512,
                kpe_head_dim: 64,
            },
        );
        for t in [
            PagedCacheType::TurboQuant3,
            PagedCacheType::TurboQuantAggressive,
        ] {
            assert!(t.resolve_for_model(&cfg, false).is_err());
        }
    }

    /// Non-TurboQuant types are never touched, whatever the geometry.
    #[test]
    fn non_turbo_types_pass_through() {
        let mla = meta(
            192,
            192,
            KvCacheLayout::Mla {
                kv_lora_rank: 512,
                kpe_head_dim: 64,
            },
        );
        let small = meta(64, 64, KvCacheLayout::Standard);
        for t in [PagedCacheType::Auto, PagedCacheType::F8E4M3] {
            for cfg in [&mla, &small] {
                for explicit in [false, true] {
                    assert_eq!(t.resolve_for_model(cfg, explicit).unwrap(), t);
                }
            }
        }
    }

    /// Prefix caching is unsupported on packed TurboQuant caches (a prefix
    /// hit would fail in `gather_kv_cache`), and supported everywhere else.
    #[test]
    fn prefix_cache_support_matches_cache_type() {
        for t in TURBO_TYPES {
            assert!(!t.supports_prefix_cache());
        }
        assert!(PagedCacheType::Auto.supports_prefix_cache());
        assert!(PagedCacheType::F8E4M3.supports_prefix_cache());
    }

    /// 🔴 The conflict must be LOUD, and loud means actionable: it has to name
    /// the flag that produced the compressed cache, the flag whose feature was
    /// taken away, and which of the two lost — otherwise an operator whose
    /// prefix cache silently vanished has nothing to act on.
    ///
    /// This is the DEFAULT configuration, not an exotic one: `--pa-cache-type`
    /// unset resolves to TurboQuant, PagedAttention is the CUDA default, and
    /// `--prefix-cache-n` defaults to 16.
    #[test]
    fn the_conflict_names_both_flags_and_says_which_one_lost() {
        for t in TURBO_TYPES {
            let msg = t
                .prefix_cache_conflict(16)
                .expect("a TurboQuant cache with prefix caching on IS a conflict");
            assert!(
                msg.contains("--pa-cache-type"),
                "must name the flag that selected the compressed cache; got {msg:?}"
            );
            assert!(
                msg.contains("--prefix-cache-n"),
                "must name the flag whose feature was overridden; got {msg:?}"
            );
            assert!(
                msg.contains("PREFIX CACHING IS NOW DISABLED"),
                "must say which of the two lost; got {msg:?}"
            );
            assert!(
                msg.contains(&t.to_string()),
                "must name the active cache type by its own flag value; got {msg:?}"
            );
        }
    }

    /// No conflict where there is none. A cache type that supports prefix
    /// caching is silent, and so is `--prefix-cache-n 0` — nothing was taken
    /// away, so warning would be noise that trains operators to ignore it.
    #[test]
    fn no_conflict_is_reported_when_nothing_was_actually_disabled() {
        for t in [PagedCacheType::Auto, PagedCacheType::F8E4M3] {
            assert!(t.prefix_cache_conflict(16).is_none());
        }
        for t in TURBO_TYPES {
            assert!(
                t.prefix_cache_conflict(0).is_none(),
                "prefix caching was already off; there is nothing to warn about"
            );
        }
    }

    /// The rendered name must be a value the user can hand back to
    /// `--pa-cache-type`, or the diagnostic's advice is unusable.
    #[test]
    fn every_cache_type_renders_as_the_flag_value_that_parses_back() {
        for t in [
            PagedCacheType::Auto,
            PagedCacheType::F8E4M3,
            PagedCacheType::TurboQuant,
            PagedCacheType::TurboQuant3,
            PagedCacheType::TurboQuantAggressive,
        ] {
            assert_eq!(t.to_string().parse::<PagedCacheType>().unwrap(), t);
        }
    }
}
