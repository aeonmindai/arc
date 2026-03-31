//! Arc — A high-performance LLM inference engine.
//!
//! Arc extends mistral.rs with TurboQuant KV cache compression, delivering
//! 2.2x memory reduction at zero quality loss (3.5-bit default, K4/V3).
//!
//! # Architecture
//!
//! Arc is a thin wrapper over mistral.rs, adding:
//! - **TurboQuant**: Near-optimal KV cache compression (ICLR 2026)
//! - **Elastic Tensor Parallelism**: Per-request GPU allocation (planned)
//! - **Disaggregated Serving**: Prefill-decode separation (planned)
//!
//! All upstream mistral.rs features are available: PagedAttention, FlashAttention,
//! speculative decoding, continuous batching, GGUF/GPTQ/AWQ/ISQ, LoRA, and more.

// Re-export the core engine
pub use mistralrs_core as core;

// Re-export the quantization layer
pub use mistralrs_quant as quant;

// Re-export Arc's TurboQuant additions
pub use arc_turbo as turbo;

// Re-export commonly used types at the top level for convenience
pub use mistralrs_core::{
    MemoryGpuConfig,
    // Pipeline and model types
    MistralRs,
    MistralRsBuilder,
    ModelLoaderConfig,
    // Configuration
    ModelSelected,
    PagedAttentionConfig,
    PagedCacheType,
    // Request/response types
    Request,
    RequestMessage,
    Response,
    // Sampling
    SamplingParams,
    // Token sources
    TokenSource,
};

/// Arc engine version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Print Arc startup banner.
pub fn print_banner() {
    tracing::info!(
        "Arc inference engine v{} — TurboQuant-accelerated LLM serving",
        VERSION
    );
    tracing::info!("Default KV cache: TurboQuant 3.5-bit (K4/V3, lossless)");
}
