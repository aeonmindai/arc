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
//! ⚠️ **TurboQuant KV is the paged default, not an opt-in.** With
//! PagedAttention on (the CUDA default), a standard-layout head_dim-128 model
//! gets TurboQuant K4/V3 KV without any flag, and loses prefix caching with it;
//! `--pa-cache-type auto` opts out. Other geometries fall back to `auto` with a
//! warning — there is no kernel at head_dim 512, so DeepSeek V4 does not take
//! it. The separate **eager** path is genuinely opt-in (`ARC_TURBOQUANT_KV=1`).
//!
//! ⚠️ **Measured vs not, precisely.** The paged path has one end-to-end run:
//! Qwen3-32B on a B200 at **55 tok/s with correct output** (2026-04-06,
//! `4eba13905`), plus eight CUDA correctness fixes found on that hardware
//! 2026-04-02. It was b=1, one card, one model, head_dim 128, and it did not
//! isolate TurboQuant from the rest of the decode path — so no speed delta is
//! attributable to compression alone, and **no quality evaluation exists**.
//! Compression *ratios* (the retracted "4.27×" among them) remain format
//! arithmetic, not measured forward passes.
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
    // "lossless" was never measured — no quality evaluation has been run under
    // any TurboQuant preset. State the default and how to leave it instead.
    tracing::info!(
        "Default paged KV cache: TurboQuant 3.5-bit (K4/V3) where supported; \
         `--pa-cache-type auto` for an unquantized cache"
    );
}
