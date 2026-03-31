//! TurboQuant — Near-optimal KV cache compression for LLM inference.
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

// Arc's TurboQuant cache (BSL licensed)
pub mod cache;
pub use cache::{TurboQuantCache, TurboQuantSingleCache};
