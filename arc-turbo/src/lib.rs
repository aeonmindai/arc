//! TurboQuant — Near-optimal KV cache compression for LLM inference.
//!
//! **Parent system: ArcQuant/TurboQuant** (see `memory/mission/TAXONOMY.md`).
//!
//! ⚠️ **Experimental, off by default, never measured.** Eager KV is opt-in via
//! `ARC_TURBOQUANT_KV=1`; the paged kernel covers head_dim 128 only and there
//! is none at head_dim 512, so DeepSeek V4 cannot use it. The prefix cache
//! auto-disables under TurboQuant because packed U8 blocks cannot be gathered.
//!
//! Implements the TurboQuant algorithm (ICLR 2026, arXiv:2504.19874) for
//! compressing LLM key-value caches to 2-4 bits per coordinate with
//! mathematically bounded quality degradation.
//!
//! # Presets
//!
//! | Preset | Keys | Values | Avg bits | Compression | Quality |
//! |--------|------|--------|----------|-------------|---------|
//! | Default | 4-bit | 3-bit | 3.5 | 2.2x | Lossless |
//! | Balanced | 3-bit | 3-bit | 3.0 | 2.56x | ~0.1% loss |
//! | Aggressive | 3-bit | 2-bit | 2.5 | 4.1x | ~1.2% loss |

// Re-export algorithm primitives from mistralrs-quant
pub use mistralrs_quant::turboquant::*;

// Arc's TurboQuant cache (Apache-2.0 licensed)
pub mod cache;
pub use cache::{TurboQuantCache, TurboQuantSingleCache};
