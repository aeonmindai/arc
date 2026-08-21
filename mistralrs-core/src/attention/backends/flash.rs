//! FlashAttention CUDA backend dispatch (FA2 / FA3).
//!
//! # Why this file has a capability layer
//!
//! Arc can be built with `flash-attn` (FlashAttention-2, via `candle-flash-attn`),
//! with `flash-attn-v3` (FlashAttention-3, via `candle-flash-attn-v3`), or with
//! **both**. The two backends do *not* accept the same inputs, and the
//! difference is not cosmetic — it is the difference between a correct answer, a
//! hard runtime error, and silently wrong numbers:
//!
//! | | FA2 (`candle-flash-attn`) | FA3 (`candle-flash-attn-v3`) |
//! |---|---|---|
//! | head_dim | any multiple of 8, `<= 256` | **only 64, 128, 256** |
//! | softcap (Gemma-2 / Grok style) | yes | **not implemented** |
//! | sliding window | yes | yes |
//! | dtype | f16 / bf16 | f16 / bf16 |
//!
//! Those envelopes are read directly off the backends' own validation, not
//! assumed:
//!
//! * FA2 — `candle-flash-attn/src/lib.rs`: `if head_size_og > 256 { bail!("only
//!   supports head dimension at most 256") }` and `if head_size_og % 8 != 0 {
//!   bail!("only supports head sizes that are a multiple of 8") }`.
//! * FA3 — `candle-flash-attn-v3/src/lib.rs`, in **both** the dense and the
//!   varlen op: `if !(head_size_og == 256 || head_size_og == 128 ||
//!   head_size_og == 64) { bail!("only supports head dimension 64, 128 and 256")
//!   }`. There is no `*_softcap` entry point in the FA3 crate at all.
//!
//! # The three bugs this layer fixes
//!
//! 1. **`flash-attn` + `flash-attn-v3` together did not compile.** Both features
//!    defined `pub(crate) fn flash_attn` in this module with only a `#[cfg]`
//!    apiece, so enabling both was an `E0428` duplicate definition. The natural
//!    way to try FA3 (append the feature to the existing build line) therefore
//!    failed with an error pointing at this file rather than at the flag.
//! 2. **FA3 silently dropped the sliding window** on the dense path: it called
//!    `candle_flash_attn_v3::flash_attn(..)`, which hard-codes
//!    `window_size_left = None`, and never read `sdpa_params.sliding_window`.
//!    Sliding-window models (Mistral, Phi-3, the windowed layers of hybrid
//!    models) attended over the *whole* cache instead of their window. FA3 does
//!    expose `flash_attn_windowed`; it just was not being called.
//! 3. **FA3 silently dropped softcap.** `sdpa_params.softcap` was never read on
//!    the FA3 path, so a softcap model built with `flash-attn-v3` produced
//!    uncapped logits — wrong numbers, no error. Softcap now routes to FA2 when
//!    FA2 is available and refuses the flash path entirely when it is not.
//!
//! # Dispatch order — FA2 first, and why FA3 is NOT preferred
//!
//! FA3 would be the natural Hopper default. It is not the default here, because
//! `candle-flash-attn-v3` at the revision this workspace pins
//! (`aeonmindai/candle` @ `d2d1d07`, `Cargo.toml:48-49`) is **broken for causal
//! attention on both of its paths**. Verified in the pinned source, and
//! described upstream in <https://github.com/huggingface/candle/pull/3606>
//! (open since 2026-06-11):
//!
//! * **Dense causal → illegal memory access.** The causal/local non-varlen path
//!   selects `DynamicPersistentTileScheduler`, which does
//!   `atomicAdd(params.tile_count_semaphore, 1)`
//!   (`hkernel/flash_fwd_launch_template.h:120` threads
//!   `params.tile_count_semaphore` into the scheduler args). The Rust binding
//!   zeroes `Flash_fwd_params` and **never allocates that counter** — `grep -r
//!   tile_count_semaphore candle-flash-attn-v3/src/` returns **0 hits** while
//!   the headers reference it throughout. The pointer stays NULL and every
//!   dense causal call performs an illegal global atomic. Upstream measured
//!   9/9 causal tests failing on an H200.
//! * **Varlen causal → silently non-causal.** The varlen path (and only it)
//!   clamps the window up before `is_causal` is derived
//!   (`candle-flash-attn-v3/src/lib.rs:591`: `if window_size_right <
//!   self.max_seqlen_k as i32 { window_size_right = self.max_seqlen_k as i32 }`).
//!   `causal = true` encodes as `window_size_right = Some(0)`, so the clamp
//!   rewrites it to full attention. No error, no warning — just wrong numbers.
//!   Upstream measured 9/9 causal varlen tests returning the non-causal result.
//!
//! Causal attention is essentially all autoregressive LLM attention, so until
//! the pinned candle carries those fixes, **preferring FA3 would be a crash or a
//! correctness regression, not a speedup.** FA2 therefore serves every shape it
//! can, and FA3 is reached only when explicitly opted into with
//! `ARC_PREFER_FA3=1` (for A/B testing once the fork is patched) or when it is
//! the only backend compiled in.
//!
//! When the fixes land in `aeonmindai/candle`, flipping the default is a
//! one-line change to [`fa3_preferred`] plus a benchmark.
//!
//! [`cuda_flash_attn_supported`] is what `Sdpa::run_attention` consults so that
//! an out-of-envelope shape never enters the flash path at all — previously an
//! unsupported `head_dim` reached the backend and turned into a hard `bail!`
//! instead of degrading to the naive kernel.
//!
//! # Not applicable to DeepSeek-V4
//!
//! V4 sets `SdpaParams::sinks` on every layer, and `Sdpa::run_attention`
//! dispatches to `sinks_attn` *before* the flash branch, so no V4 layer reaches
//! this file. V4's `head_dim = 512` is outside every FA generation's envelope
//! anyway (FA2/FA3 cap at 256). Neither the feature flag nor anything in this
//! module changes a kernel V4 executes.

use candle_core::{Result, Tensor};

use crate::attention::SdpaParams;

/// Head dims accepted by FlashAttention-3 (`candle-flash-attn-v3`).
///
/// Mirrors the crate's own check, which is applied identically in the dense and
/// varlen ops: `head_size_og == 256 || head_size_og == 128 || head_size_og == 64`.
const FA3_HEAD_DIMS: [usize; 3] = [64, 128, 256];

/// Maximum head dim accepted by FlashAttention-2 (`candle-flash-attn`):
/// `if head_size_og > 256 { bail!(..) }`.
const FA2_MAX_HEAD_DIM: usize = 256;

/// Opt-in switch for preferring FlashAttention-3 over FlashAttention-2 when
/// both are compiled in.
///
/// Defaults to **false**: the pinned `candle-flash-attn-v3` crashes on dense
/// causal attention (NULL `tile_count_semaphore`) and silently computes
/// non-causal attention on the varlen causal path. See the module docs and
/// <https://github.com/huggingface/candle/pull/3606>. Flip this default only
/// once the fork carries those fixes *and* a benchmark says FA3 is faster here.
/// Always compiled (not `#[cfg]`-gated) so the default stays unit-testable on a
/// host with no CUDA toolchain; only the both-backends dispatcher calls it.
#[cfg_attr(
    not(all(feature = "flash-attn", feature = "flash-attn-v3")),
    allow(dead_code)
)]
pub(crate) fn fa3_preferred() -> bool {
    use std::sync::OnceLock;
    static PREFERRED: OnceLock<bool> = OnceLock::new();
    *PREFERRED.get_or_init(|| {
        matches!(
            std::env::var("ARC_PREFER_FA3").as_deref(),
            Ok("1") | Ok("true")
        )
    })
}

/// Would FlashAttention-2 accept this shape?
///
/// Pure predicate over the documented envelope — always compiled (regardless of
/// feature flags) so it can be unit-tested on any host.
pub(crate) fn fa2_accepts(head_dim: usize, softcap: Option<f32>) -> bool {
    let _ = softcap; // FA2 implements softcap natively.
    head_dim <= FA2_MAX_HEAD_DIM && head_dim % 8 == 0
}

/// Would FlashAttention-3 accept this shape?
///
/// Note the softcap term: FA3 has no softcap entry point, so a softcapped call
/// must not be routed here — doing so would drop the cap silently.
pub(crate) fn fa3_accepts(head_dim: usize, softcap: Option<f32>) -> bool {
    // A softcap of 1.0 is the identity (`tanh(x/1)*1` is only applied when the
    // caller asked for capping); the rest of the codebase already treats
    // `Some(1.0)` as "no softcap" (see the Metal path in `attention/mod.rs`).
    let softcap_ok = matches!(softcap, None) || matches!(softcap, Some(c) if c == 1.0);
    FA3_HEAD_DIMS.contains(&head_dim) && softcap_ok
}

/// Can *this build* serve this shape on a CUDA flash kernel?
///
/// Consulted by `Sdpa::run_attention` before it commits to the flash path. If
/// this returns `false` the caller degrades to non-flash SDPA, which has no
/// head-dim restriction.
///
/// `cfg!` (not `#[cfg]`) is deliberate: the function body compiles under every
/// feature combination, which keeps the envelope unit-testable on a host with no
/// CUDA toolchain.
pub(crate) fn cuda_flash_attn_supported(
    head_dim: usize,
    k_head_dim: usize,
    v_head_dim: usize,
    softcap: Option<f32>,
) -> bool {
    // Every FA kernel requires a single shared head dim across q/k/v. Models
    // with an asymmetric value head (e.g. non-absorbed MLA, where
    // `v_head_dim != qk_head_dim`) must not take this path.
    if head_dim != k_head_dim || k_head_dim != v_head_dim {
        return false;
    }

    (cfg!(feature = "flash-attn-v3") && fa3_accepts(head_dim, softcap))
        || (cfg!(feature = "flash-attn") && fa2_accepts(head_dim, softcap))
}

// ---------------------------------------------------------------------------
// FlashAttention-2
// ---------------------------------------------------------------------------

#[cfg(feature = "flash-attn")]
fn flash_attn_v2(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    flash_params: Option<&crate::pipeline::text_models_inputs_processor::FlashParams>,
    sdpa_params: &SdpaParams,
) -> Result<Tensor> {
    let (_b_sz, _n_attn_heads, seq_len, _head_dim) = q.dims4()?;
    let window_size_left = sdpa_params.sliding_window;
    let default_causal = seq_len > 1;

    // If flash_params provides cumulative_seqlens for this device, use the varlen path.
    if let Some(params) = flash_params {
        if let Some(cumulative_seqlens_q) = params.cumulative_seqlens_q.get(&q.device().location())
        {
            let cumulative_seqlens_k = &params.cumulative_seqlens_k[&q.device().location()];
            let window_size_right = if params.causal { Some(0) } else { None };
            let qshape = q.shape();
            let q = q.flatten_to(1)?;
            let k = k.flatten_to(1)?;
            let v = v.flatten_to(1)?;

            if let Some(softcap) = sdpa_params.softcap {
                return candle_flash_attn::flash_attn_varlen_alibi_windowed_softcap(
                    &q,
                    &k,
                    &v,
                    None,
                    cumulative_seqlens_q,
                    cumulative_seqlens_k,
                    params.max_q as usize,
                    params.max_k as usize,
                    sdpa_params.softmax_scale,
                    window_size_left,
                    window_size_right,
                    softcap,
                )?
                .reshape(qshape);
            } else {
                return candle_flash_attn::flash_attn_varlen_windowed(
                    &q,
                    &k,
                    &v,
                    cumulative_seqlens_q,
                    cumulative_seqlens_k,
                    params.max_q as usize,
                    params.max_k as usize,
                    sdpa_params.softmax_scale,
                    window_size_left,
                    window_size_right,
                )?
                .reshape(qshape);
            }
        }
    }

    // Non-varlen path: use flash_params.causal if provided, otherwise default (seq_len > 1).
    let causal = flash_params.map_or(default_causal, |p| p.causal);
    let window_size_right = if causal { Some(0) } else { None };
    if let Some(softcap) = sdpa_params.softcap {
        candle_flash_attn::flash_attn_alibi_windowed_softcap(
            q,
            k,
            v,
            None,
            sdpa_params.softmax_scale,
            window_size_left,
            window_size_right,
            softcap,
        )
    } else {
        candle_flash_attn::flash_attn_windowed(
            q,
            k,
            v,
            sdpa_params.softmax_scale,
            window_size_left,
            window_size_right,
        )
    }
}

// ---------------------------------------------------------------------------
// FlashAttention-3
// ---------------------------------------------------------------------------

#[cfg(feature = "flash-attn-v3")]
fn flash_attn_v3(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    flash_params: Option<&crate::pipeline::text_models_inputs_processor::FlashParams>,
    sdpa_params: &SdpaParams,
) -> Result<Tensor> {
    let (_b_sz, _n_attn_heads, seq_len, _head_dim) = q.dims4()?;
    let default_causal = seq_len > 1;
    let window_size_left = sdpa_params.sliding_window;

    // FA3 has no softcap kernel. `cuda_flash_attn_supported` / the dispatcher
    // keep softcapped calls off this path; this is the belt-and-braces check so
    // a future caller cannot reintroduce the silent-drop bug.
    if sdpa_params.softcap.is_some_and(|c| c != 1.0) {
        candle_core::bail!(
            "flash-attn-v3 has no softcap kernel (requested softcap {:?}). \
             Build with `--features flash-attn` as well so softcapped models can \
             fall back to FlashAttention-2.",
            sdpa_params.softcap
        );
    }

    // If flash_params provides cumulative_seqlens for this device, use the varlen path.
    if let Some(params) = flash_params {
        if let Some(cumulative_seqlens_q) = params.cumulative_seqlens_q.get(&q.device().location())
        {
            let cumulative_seqlens_k = &params.cumulative_seqlens_k[&q.device().location()];
            let qshape = q.shape();
            let q = q.flatten_to(1)?;
            let k = k.flatten_to(1)?;
            let v = v.flatten_to(1)?;

            let window_size_right = if params.causal { Some(0) } else { None };

            return candle_flash_attn_v3::flash_attn_varlen_windowed(
                &q,
                &k,
                &v,
                cumulative_seqlens_q,
                cumulative_seqlens_k,
                params.max_q as usize,
                params.max_k as usize,
                sdpa_params.softmax_scale,
                window_size_left,
                window_size_right,
                true,
            )?
            .reshape(qshape);
        }
    }

    // Non-varlen path: use flash_params.causal if provided, otherwise default (seq_len > 1).
    //
    // NOTE: `flash_attn_windowed`, not `flash_attn`. The latter hard-codes
    // `window_size_left = None`, which dropped `sdpa_params.sliding_window` on
    // every dense FA3 call.
    let causal = flash_params.map_or(default_causal, |p| p.causal);
    let window_size_right = if causal { Some(0) } else { None };
    candle_flash_attn_v3::flash_attn_windowed(
        q,
        k,
        v,
        sdpa_params.softmax_scale,
        window_size_left,
        window_size_right,
        true,
    )
}

// ---------------------------------------------------------------------------
// Dispatcher — one definition per feature combination
// ---------------------------------------------------------------------------

/// Both backends available.
///
/// FA2 serves everything unless FA3 is explicitly opted into — see the module
/// docs for why FA3 is not the default despite Hopper being its target.
#[cfg(all(feature = "flash-attn", feature = "flash-attn-v3"))]
pub(crate) fn flash_attn(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    flash_params: Option<&crate::pipeline::text_models_inputs_processor::FlashParams>,
    sdpa_params: &SdpaParams,
) -> Result<Tensor> {
    let head_dim = q.dim(candle_core::D::Minus1)?;
    if fa3_preferred() && fa3_accepts(head_dim, sdpa_params.softcap) {
        flash_attn_v3(q, k, v, flash_params, sdpa_params)
    } else {
        flash_attn_v2(q, k, v, flash_params, sdpa_params)
    }
}

/// FA2 only.
#[cfg(all(feature = "flash-attn", not(feature = "flash-attn-v3")))]
pub(crate) fn flash_attn(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    flash_params: Option<&crate::pipeline::text_models_inputs_processor::FlashParams>,
    sdpa_params: &SdpaParams,
) -> Result<Tensor> {
    flash_attn_v2(q, k, v, flash_params, sdpa_params)
}

/// FA3 only. Shapes outside FA3's envelope are rejected with an actionable
/// message rather than silently mis-computed; `cuda_flash_attn_supported` should
/// have diverted them to non-flash SDPA before reaching here.
#[cfg(all(feature = "flash-attn-v3", not(feature = "flash-attn")))]
pub(crate) fn flash_attn(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    flash_params: Option<&crate::pipeline::text_models_inputs_processor::FlashParams>,
    sdpa_params: &SdpaParams,
) -> Result<Tensor> {
    // FA3 is the only backend here, so there is nothing to fall back to. Warn
    // once about the known causal defects rather than degrading silently — a
    // build that enables `flash-attn-v3` alone has asked for exactly this.
    {
        use std::sync::Once;
        static WARNED: Once = Once::new();
        WARNED.call_once(|| {
            tracing::warn!(
                "Built with `flash-attn-v3` only. The pinned candle-flash-attn-v3 \
                 crashes on dense causal attention (unallocated tile_count_semaphore) \
                 and silently computes NON-causal attention on the varlen causal path \
                 (huggingface/candle#3606). Add `--features flash-attn` so causal \
                 shapes fall back to FlashAttention-2."
            );
        });
    }

    let head_dim = q.dim(candle_core::D::Minus1)?;
    if !fa3_accepts(head_dim, sdpa_params.softcap) {
        candle_core::bail!(
            "flash-attn-v3 supports head_dim 64, 128 or 256 without softcap \
             (got head_dim {head_dim}, softcap {:?}). Add `--features flash-attn` \
             so this shape can fall back to FlashAttention-2.",
            sdpa_params.softcap
        );
    }
    flash_attn_v3(q, k, v, flash_params, sdpa_params)
}

/// Neither backend compiled in.
#[cfg(not(any(feature = "flash-attn", feature = "flash-attn-v3")))]
pub(crate) fn flash_attn(
    _: &Tensor,
    _: &Tensor,
    _: &Tensor,
    _: Option<&crate::pipeline::text_models_inputs_processor::FlashParams>,
    _: &SdpaParams,
) -> Result<Tensor> {
    unimplemented!("Compile with `--features flash-attn` or `--features flash-attn-v3`.")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// FA2's envelope, per `candle-flash-attn`'s own `bail!`s: `<= 256` and a
    /// multiple of 8.
    #[test]
    fn fa2_envelope_matches_candle_flash_attn() {
        for hd in [8, 32, 64, 72, 80, 96, 112, 120, 128, 192, 224, 256] {
            assert!(fa2_accepts(hd, None), "FA2 should accept head_dim {hd}");
        }
        // Not a multiple of 8.
        for hd in [1, 63, 100, 130] {
            assert!(!fa2_accepts(hd, None), "FA2 should reject head_dim {hd}");
        }
        // Above the cap — DeepSeek-V4's 512 among them.
        for hd in [264, 512, 576] {
            assert!(!fa2_accepts(hd, None), "FA2 should reject head_dim {hd}");
        }
        // FA2 implements softcap natively, so a cap does not narrow its envelope.
        assert!(fa2_accepts(128, Some(30.0)));
    }

    /// FA3's envelope, per `candle-flash-attn-v3`'s `bail!` in *both* the dense
    /// and varlen ops: exactly {64, 128, 256}.
    #[test]
    fn fa3_envelope_matches_candle_flash_attn_v3() {
        for hd in [64, 128, 256] {
            assert!(fa3_accepts(hd, None), "FA3 should accept head_dim {hd}");
        }
        // Multiples of 8 that FA2 takes but FA3 does not — the shapes that would
        // have hard-errored had the default been flipped naively.
        for hd in [72, 80, 96, 112, 120, 192, 224] {
            assert!(!fa3_accepts(hd, None), "FA3 should reject head_dim {hd}");
        }
        for hd in [512, 576] {
            assert!(!fa3_accepts(hd, None), "FA3 should reject head_dim {hd}");
        }
    }

    /// FA3 has no softcap kernel; a softcapped call must never route to it.
    #[test]
    fn fa3_refuses_softcap_but_accepts_identity_cap() {
        assert!(!fa3_accepts(128, Some(30.0)));
        assert!(!fa3_accepts(128, Some(50.0)));
        // `Some(1.0)` is the identity and is treated as "no softcap" elsewhere.
        assert!(fa3_accepts(128, Some(1.0)));
        assert!(fa3_accepts(128, None));
    }

    /// Asymmetric value heads (non-absorbed MLA) must not take the flash path
    /// under any feature combination.
    #[test]
    fn mismatched_head_dims_are_never_flash_eligible() {
        assert!(!cuda_flash_attn_supported(192, 192, 128, None));
        assert!(!cuda_flash_attn_supported(128, 64, 64, None));
        // DeepSeek-V4's shape, for the record: symmetric, but far past the cap.
        assert!(!cuda_flash_attn_supported(512, 512, 512, None));
    }

    /// FA3 must not be preferred by default: the pinned `candle-flash-attn-v3`
    /// crashes on dense causal attention and silently drops causality on the
    /// varlen path (huggingface/candle#3606). This test is the tripwire — if it
    /// starts failing, someone flipped the default, and that is only correct
    /// once the candle fork carries the fixes.
    #[test]
    fn fa3_is_not_preferred_by_default() {
        // Guard against a stray env var in the test environment making this
        // vacuous: only assert the default when the opt-in is not ASKED FOR.
        // Read by value, so a box that exports `ARC_PREFER_FA3=0` still runs
        // the assertion — under the presence test it silently skipped it,
        // which is a tripwire that quietly stopped being a tripwire.
        if !mistralrs_quant::env_flag_is_set("ARC_PREFER_FA3") {
            assert!(
                !fa3_preferred(),
                "FA3 must stay opt-in while candle#3606 is unfixed in the pinned fork"
            );
        }
    }

    /// The build-level predicate must agree with whichever backends are actually
    /// compiled in. With no flash feature enabled it must always be `false`, so
    /// `Sdpa::run_attention` never enters a path that would `unimplemented!()`.
    #[test]
    fn build_predicate_agrees_with_enabled_features() {
        let any_flash = cfg!(feature = "flash-attn") || cfg!(feature = "flash-attn-v3");
        assert_eq!(
            cuda_flash_attn_supported(128, 128, 128, None),
            any_flash,
            "head_dim 128 is in both envelopes, so support must track the features"
        );
        // head_dim 80 is FA2-only.
        assert_eq!(
            cuda_flash_attn_supported(80, 80, 80, None),
            cfg!(feature = "flash-attn"),
        );
        // Softcap is FA2-only.
        assert_eq!(
            cuda_flash_attn_supported(128, 128, 128, Some(30.0)),
            cfg!(feature = "flash-attn"),
        );
    }
}
