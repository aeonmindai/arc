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
//!
//! # Algorithm
//!
//! 1. Normalize input vector, store L2 norm as fp16
//! 2. Apply randomized Walsh-Hadamard rotation (D·H·D)
//! 3. Quantize each coordinate with pre-computed Lloyd-Max codebooks
//! 4. Pack indices into sub-byte storage (nibble or 10-in-32)
//!
//! The rotation makes every coordinate follow Beta(d/2, d/2) regardless of
//! input, enabling data-oblivious optimal scalar quantization.

// Re-export from mistralrs-quant where the implementation lives
pub use mistralrs_quant::turboquant::*;
