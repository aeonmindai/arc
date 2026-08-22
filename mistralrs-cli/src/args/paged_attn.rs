//! PagedAttention configuration options
//!
//! Unified design replacing the confusing 5-flag system with clear semantics.

use clap::{Args, ValueEnum};
use mistralrs_core::PagedCacheType;
use serde::Deserialize;

/// Cache and attention configuration
#[derive(Args, Clone, Deserialize, Default)]
pub struct CacheOptions {
    #[command(flatten)]
    pub paged_attn: PagedAttentionOptions,
}

/// PagedAttention configuration
#[derive(Args, Clone, Deserialize)]
pub struct PagedAttentionOptions {
    /// PagedAttention mode
    /// - auto: enabled on CUDA, disabled on Metal/CPU (default)
    /// - on: force enable (fails if unsupported)
    /// - off: force disable
    ///
    /// Hidden: `auto` already enables it everywhere it works. Forcing it off
    /// costs throughput; forcing it on fails on unsupported devices.
    #[arg(long = "paged-attn", default_value = "auto", value_enum, hide = true)]
    #[serde(default)]
    pub mode: PagedAttnMode,

    /// How long conversations may get: sizes the KV cache for this many
    /// tokens. Defaults to filling ~90% of free VRAM.
    #[arg(long = "pa-context-len", help_heading = "Serving")]
    pub context_len: Option<usize>,

    /// GPU memory to allocate in MBs (alternative to context-len)
    ///
    /// Hidden: a second way to say --pa-context-len, in units that do not
    /// describe what the user actually wants.
    #[arg(long = "pa-memory-mb", conflicts_with = "context_len", hide = true)]
    pub memory_mb: Option<usize>,

    /// GPU memory utilization fraction 0.0-1.0 (alternative to context-len/memory-mb)
    ///
    /// Hidden: third spelling of the same decision.
    #[arg(long = "pa-memory-fraction", conflicts_with_all = ["context_len", "memory_mb"], hide = true)]
    pub memory_fraction: Option<f32>,

    /// Tokens per block (default: 32 on CUDA)
    ///
    /// Hidden: a paging granularity tradeoff with no user-visible meaning;
    /// wrong values cost throughput silently.
    #[arg(long = "pa-block-size", hide = true)]
    pub block_size: Option<usize>,

    /// KV cache quantization type: turboquant (K4/V3, 3.5-bit), turboquant-3
    /// (K3/V3), turboquant-aggressive (K3/V2), auto, f8e4m3.
    /// UNSET MEANS TURBOQUANT, NOT `auto`. On a standard-layout model with
    /// head_dim 128 (with PagedAttention on, which is the CUDA default) leaving
    /// this flag off gives you TurboQuant KV and, with it, no prefix caching.
    /// Pass `auto` to opt out. Every other geometry — MLA layouts and every
    /// other head_dim — auto-falls back to `auto` with a warning, so those
    /// models are unaffected. Setting the flag explicitly widens the accepted
    /// set to every instantiated head_dim (64/128/256/512) and turns that
    /// fallback into a hard error instead.
    /// Provenance: TurboQuant K4/V3 has one end-to-end serving run behind it —
    /// Qwen3-32B on a B200, 55 tok/s with correct output (2026-04-06). That run
    /// was b=1, head_dim 128, on Blackwell, and it did not isolate TurboQuant
    /// from the rest of the decode path. There is no A/B against an
    /// unquantized cache and no quality evaluation at any preset, so treat
    /// quality as unestablished; 64/256/512 are compiled but have never
    /// executed on hardware, which is why the default will not choose them for
    /// you, and `turboquant-3` and `turboquant-aggressive` have never run at
    /// all.
    // Hidden: overriding Arc's resolved KV cache type is the clearest way to
    // silently make a deployment worse — every TurboQuant variant currently
    // disables prefix caching, and an explicit choice converts the safe
    // auto-fallback into a hard error. Still fully supported via `--help-all`.
    // (Attribute only — the doc text above is owned by the TurboQuant chain.)
    #[arg(long = "pa-cache-type", value_parser = parse_cache_type, hide = true)]
    #[serde(default)]
    pub cache_type: Option<PagedCacheType>,
}

impl Default for PagedAttentionOptions {
    fn default() -> Self {
        Self {
            mode: PagedAttnMode::Auto,
            context_len: None,
            memory_mb: None,
            memory_fraction: None,
            block_size: None,
            cache_type: None,
        }
    }
}

/// PagedAttention operation mode
#[derive(Clone, Copy, ValueEnum, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum PagedAttnMode {
    /// Automatic: enabled on CUDA, disabled on Metal/CPU
    #[default]
    Auto,
    /// Force enable (error if device doesn't support it)
    On,
    /// Force disable
    Off,
}

impl PagedAttentionOptions {
    /// Convert to the flags expected by MistralRsForServerBuilder
    pub fn into_builder_flags(self) -> PagedAttnBuilderFlags {
        let enable = match self.mode {
            PagedAttnMode::Auto => None,
            PagedAttnMode::On => Some(true),
            PagedAttnMode::Off => Some(false),
        };

        (
            enable,
            self.memory_mb,
            self.memory_fraction,
            self.context_len,
            self.block_size,
            self.cache_type,
        )
    }
}

fn parse_cache_type(s: &str) -> Result<PagedCacheType, String> {
    s.parse()
}

/// PagedAttention builder flags type alias
pub type PagedAttnBuilderFlags = (
    Option<bool>,           // paged_attn enable flag
    Option<usize>,          // gpu_mem (MBs)
    Option<f32>,            // gpu_mem_usage (fraction)
    Option<usize>,          // context_len
    Option<usize>,          // block_size
    Option<PagedCacheType>, // cache_type (None = TurboQuant default with auto-fallback)
);
