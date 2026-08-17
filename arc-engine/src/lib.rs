//! Arc — A high-performance LLM inference engine.
//!
//! **Parent system: ArcServe/SDK** (see `memory/mission/TAXONOMY.md`). This crate
//! is the Rust façade — it re-exports `mistralrs_core`, `mistralrs_quant` and
//! `arc_turbo`, and hosts the Tier-A research modules listed below.
//!
//! Arc extends mistral.rs with QTIP 2-bit trellis weight quantization (the
//! `qtip2b` rung is what Arc serves), the V4 architecture, a radix KV sharing
//! tree, and a GPU-autonomous decode path.
//!
//! ⚠️ **TurboQuant KV compression is experimental and off by default.** The
//! eager path is opt-in via `ARC_TURBOQUANT_KV=1`; the paged kernel exists at
//! head_dim 128 only; there is no kernel at head_dim 512, so DeepSeek V4 cannot
//! use it. No TurboQuant serving measurement exists — any compression ratio
//! quoted for it is format arithmetic, not a measured forward pass.
//!
//! ⚠️ Several modules here are **Tier A**: standalone, tested, and NOT wired
//! into the engine loop (`sarathi`, `expert_affinity`, `magicdec`, `yoco`).
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

// Arc-specific modules
pub mod deepseek_v4;
pub mod dsv4;
pub mod eagle3;
pub mod expert_affinity;
pub mod glm_moe;
pub mod kimi_k2;
pub mod magicdec;
pub mod moba;
pub mod mtp;
pub mod sage;
pub mod sarathi;
pub mod td_moe;
pub mod td_moe_loader;
pub mod turbo_sparse;
pub mod weight_schema;
pub mod yoco;

#[cfg(test)]
mod v2_stack_smoke;

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
