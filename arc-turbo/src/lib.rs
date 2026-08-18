//! TurboQuant — Near-optimal KV cache compression for LLM inference.
//!
//! **Parent system: ArcQuant/TurboQuant** (see `memory/mission/TAXONOMY.md`).
//!
//! **Two paths, opposite defaults — do not conflate them.**
//!
//! * **Paged** (`--pa-cache-type`): TurboQuant K4/V3 is the **default**. It is
//!   what `defaults::PAGED_CACHE_TYPE` resolves to, so a standard-layout
//!   head_dim-128 model on CUDA gets TurboQuant KV with no flag set. Every
//!   other geometry falls back to `auto` with a warning. Prefix caching
//!   auto-disables under TurboQuant because packed U8 blocks cannot be
//!   gathered.
//! * **Eager** (`NormalCache`): **opt-in**, `ARC_TURBOQUANT_KV=1`, because that
//!   path has no fused kernel and round-trips through the host per step. See
//!   `mistralrs_core::kv_cache::resolve_eager_turboquant`.
//!
//! # What has and has not been measured
//!
//! **Measured** — the paged K4/V3 path served Qwen3-32B end-to-end on a B200
//! at **55 tok/s with correct output** (2026-04-06, commit `4eba13905`; the
//! harness is `deploy/modal_b200.py`). Eight CUDA correctness defects were
//! found and fixed against that hardware on 2026-04-02, including a V-cache
//! stride mismatch (`143b5ab20`) and a Q·K warp-reduction deadlock
//! (`fd0074792`). Statements that "no TurboQuant forward pass has ever been
//! benchmarked" are false and have been corrected.
//!
//! **Not measured** — that run was b=1, one model, one card, head_dim 128,
//! `Default` preset only, and it did **not** isolate TurboQuant from the rest
//! of the decode path, so no speed delta is attributable to compression alone.
//! There is no quality evaluation at any preset, and no run at any other head
//! dim. The "4.27× KV compression" figure once published is format arithmetic
//! and stays retracted.
//!
//! Implements the TurboQuant algorithm (ICLR 2026, arXiv:2504.19874) for
//! compressing LLM key-value caches to 2-4 bits per coordinate with
//! mathematically bounded quality degradation.
//!
//! # Presets
//!
//! Compression is format arithmetic. **The quality column is the paper's
//! claim, not our measurement** — Arc has run no quality evaluation under any
//! preset, and `Balanced`/`Aggressive` have never executed on hardware at all.
//!
//! | Preset | Keys | Values | Avg bits | Compression \[arithmetic] | Quality \[paper, unreproduced] |
//! |--------|------|--------|----------|-------------|---------|
//! | Default | 4-bit | 3-bit | 3.5 | 2.2x | lossless |
//! | Balanced | 3-bit | 3-bit | 3.0 | 2.56x | ~0.1% loss |
//! | Aggressive | 3-bit | 2-bit | 2.5 | 4.1x | ~1.2% loss |

// Re-export algorithm primitives from mistralrs-quant
pub use mistralrs_quant::turboquant::*;

// Arc's TurboQuant cache (Apache-2.0 licensed)
pub mod cache;
pub use cache::{TurboQuantCache, TurboQuantSingleCache};
