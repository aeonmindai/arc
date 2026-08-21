//! `arc validate --target-hbm` — measure compressed memory footprint.
//!
//! This module implements the **HBM residency** mode of `arc validate`.
//! Schema-name validation (the older mode, `--index` + `--arch`) lives in
//! `main.rs` and is unchanged.
//!
//! Two execution paths:
//!
//! 1. **Real-GPU path** (default when CUDA is available): snapshot free HBM
//!    *before* model load, load the model with the requested compression
//!    stack, snapshot free HBM *after* load, compute the delta. Requires the
//!    `cuda` feature; without it, returns a clear error.
//!
//! 2. **Mock path** (`--mock`): compute the *expected* footprint analytically
//!    from the model's config + compression-stack assumptions. Always works
//!    off-GPU; used by CI to verify the script's JSON schema, exit codes,
//!    and pass/fail logic without renting hardware.
//!
//! The output JSON is the contract that the rental verification step (and
//! downstream CI) reads. Schema is intentionally narrow and stable.
//!
//! # Exit codes
//!
//! | code | meaning | who acts |
//! |---|---|---|
//! | `0` | measured, and it fits | nobody |
//! | `1` | measured, and it does not fit | the model owner |
//! | `2` | **not measured** — the run proves nothing | the operator |
//!
//! The third row is the whole point of [`hbm_verdict`]. A zero HBM delta —
//! model loaded on the CPU, snapshots straddling a driver reset, free memory
//! not going down — used to satisfy `0.0 <= target` and exit `0`, so this gate
//! would certify "V4 fits in 60 GB on this H100" off a run where V4 never
//! reached the H100. **A gate that cannot distinguish "it fits" from "we did
//! not look" is worse than no gate**, because it manufactures the exact
//! number a rental decision is made on. `2` is reserved for the environment,
//! never `1`, so that a red lane always says whether to look at the model or
//! at the box.

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Bytes per parameter for each compression stack. The number is the *effective*
/// average — TurboQuant has overhead from scales/codebooks; FP8 has no overhead
/// per param but a small workspace; NVFP4 is a per-block format.
///
/// These numbers are conservative upper bounds calibrated against published
/// papers + the V3-class memory footprints we have measured on rentals so far.
/// The real `--target-hbm` run on hardware is the source of truth; this table
/// is used by the `--mock` path so that CI can exercise the JSON contract
/// without a GPU.
const BYTES_PER_PARAM_BF16: f64 = 2.0;
const BYTES_PER_PARAM_FP8: f64 = 1.0;
const BYTES_PER_PARAM_NVFP4: f64 = 0.5 + 0.0625; // 4-bit + ~1-byte/16-elem scale
const BYTES_PER_PARAM_QTIP2: f64 = 0.4375; // 3.5-bit (TurboQuant headline)
const BYTES_PER_PARAM_TD_MOE_EXPERT: f64 = 0.4; // Expert-only QTIP2 + sparsity gain

/// Compression-stack identifier. We accept a small set of well-known names
/// for now — adding a new stack means adding a row to this enum and
/// the corresponding bytes-per-param entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionStack {
    /// No compression — model weights stored in BF16.
    Bf16Only,
    /// FP8 (W8A8) for all linear layers — the simplest H100-friendly stack.
    Fp8Only,
    /// NVFP4 (W4A4) for all linear layers — Blackwell-native, falls back on H100.
    Nvfp4,
    /// QTIP2 (TurboQuant 3.5-bit) for attention/dense + TD-MoE for experts.
    /// This is Arc's headline default for V4 Flash on H100.
    Qtip2PlusTdMoe,
    /// QTIP2 everywhere (no TD-MoE differentiation).
    Qtip2Only,
}

impl CompressionStack {
    pub fn parse(s: &str) -> Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "bf16" | "bf16-only" => Ok(Self::Bf16Only),
            "fp8" | "fp8-only" => Ok(Self::Fp8Only),
            "nvfp4" => Ok(Self::Nvfp4),
            "qtip2" | "qtip2-only" => Ok(Self::Qtip2Only),
            "qtip2+td-moe" | "qtip2-tdmoe" | "qtip2+tdmoe" => Ok(Self::Qtip2PlusTdMoe),
            other => Err(anyhow!(
                "unknown compression stack `{other}`. Supported: bf16, fp8-only, nvfp4, qtip2-only, qtip2+td-moe"
            )),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Bf16Only => "bf16",
            Self::Fp8Only => "fp8-only",
            Self::Nvfp4 => "nvfp4",
            Self::Qtip2Only => "qtip2-only",
            Self::Qtip2PlusTdMoe => "qtip2+td-moe",
        }
    }
}

/// JSON output schema. The rental verification step reads this file to decide
/// whether the deployment passed or failed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HbmReport {
    /// HuggingFace model id (e.g., "deepseek-ai/DeepSeek-V4-Flash").
    pub model: String,
    /// Compression stack identifier (e.g., "qtip2+td-moe").
    pub compression_stack: String,
    /// Target HBM ceiling in GB (passes if `measured.total_gb <= target_gb`).
    pub target_gb: f64,
    /// Measured (or mocked) memory breakdown.
    pub measured: MeasuredFootprint,
    /// True when the run produced a usable measurement **and**
    /// `measured.total_gb <= target_gb`.
    ///
    /// Never true when [`HbmReport::unmeasured`] is set: a run that measured
    /// nothing has not shown that the model fits, and `0.0 <= target` is not
    /// evidence of anything.
    pub pass: bool,
    /// The run did not produce a usable measurement — the model never became
    /// resident on the device.
    ///
    /// This is the difference between "V4 does not fit in 60 GB" and "we do
    /// not know whether V4 fits in 60 GB", and it is the difference between
    /// exit 1 and exit 2. `#[serde(default)]` so reports written before this
    /// field existed still parse, as `false`.
    #[serde(default)]
    pub unmeasured: bool,
    /// "mock" or "gpu".
    pub mode: String,
    /// GPU descriptor (None for mock path).
    pub gpu: Option<GpuInfo>,
    /// Free-form notes — used for follow-up actions like "next: rent + re-run".
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasuredFootprint {
    pub total_gb: f64,
    pub weight_gb: f64,
    pub workspace_gb: f64,
    pub kv_estimate_gb: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuInfo {
    pub name: String,
    pub total_hbm_gb: f64,
}

/// Parsed CLI options for `arc validate --target-hbm`.
#[derive(Debug, Clone)]
pub struct HbmValidateOptions {
    pub model_id: String,
    pub compression_stack: CompressionStack,
    pub target_hbm_gb: f64,
    pub output_path: PathBuf,
    pub mock: bool,
}

/// Estimate model parameters from the model id. This uses a small hard-coded
/// table of known models because:
///   1. `--mock` mode runs offline (no HF download)
///   2. The numbers come from the published model cards / configs
///   3. New models get added here as we onboard them
///
/// Returns (total_param_count, active_param_count_per_token, has_moe).
///
/// The V4 figures were reconciled in May 2026 (this function used to be the lone
/// in-repo outlier at 140B Flash / 680B Pro — that dispute is RESOLVED):
///
/// - **V4 Flash = 284B logical / 13B active.** HF-verified four ways: model-card
///   prose, the official release announcement
///   (`api-docs.deepseek.com/news/news260424`), config geometry (43 layers,
///   256+1 experts), and the HF safetensors breakdown (~148.7 GiB on disk).
///   Note: the HF API's `safetensors.total` ≈ 158B is a *packed-element* count
///   (experts ship ~4-bit packed), NOT the logical parameter count.
/// - **V4 Pro = 1.6T / 49B active** per the official announcement. API-tier —
///   no open checkpoint, so the vendor figure is not independently verifiable.
///
/// The `--mock` estimate derived from this table is deliberately conservative:
/// it assumes a 20/80 attention/expert split, but V4 Flash's expert share is
/// closer to ~95%, so the mock overestimates the qtip2+td-moe footprint
/// (~116 GiB mocked vs the measured 68 GB UQFF artifact, which loads and serves
/// on a single H100/H200 — empirically established June 2026 on RUN-161 and
/// re-validated on an H200 in Aug 2026; see docs/BENCHMARKS.md). The real
/// `--target-hbm` run on hardware is always the source of truth.
pub fn known_model_params(model_id: &str) -> Option<(u64, u64, bool)> {
    let normalised = model_id.to_ascii_lowercase();
    if normalised.contains("deepseek-v4-flash") || normalised.contains("deepseekv4-flash") {
        // V4 Flash: 43 layers, 256+1 experts. 284B logical / 13B active.
        // HF-verified — see fn doc-comment above for the provenance chain.
        Some((284_000_000_000, 13_000_000_000, true))
    } else if normalised.contains("deepseek-v4-pro") || normalised.contains("deepseekv4-pro") {
        // V4 Pro: 1.6T / 49B active per the official announcement (API-tier,
        // no open checkpoint to verify against).
        Some((1_600_000_000_000, 49_000_000_000, true))
    } else if normalised.contains("kimi-k2") {
        // K2-class: 61 layers, 384 experts. ~1.1T total, ~30B active.
        Some((1_100_000_000_000, 30_000_000_000, true))
    } else if normalised.contains("glm-5") || normalised.contains("glm5") {
        // GLM-5.1-class: smaller MoE, ~400B total, ~30B active.
        Some((400_000_000_000, 30_000_000_000, true))
    } else if normalised.contains("llama-3.1-8b") || normalised.contains("llama3.1-8b") {
        // Llama 3.1 8B: dense.
        Some((8_000_000_000, 8_000_000_000, false))
    } else {
        None
    }
}

/// Compute expected total weight bytes for a model under a compression stack.
/// For MoE models this counts *all* experts (resident footprint), not just active.
pub fn estimate_weight_bytes(total_params: u64, has_moe: bool, stack: CompressionStack) -> u64 {
    // Split params: ~20% attention/dense, ~80% experts for an MoE model.
    // For dense models, everything counts as "non-expert".
    let (attn_share, expert_share) = if has_moe { (0.20, 0.80) } else { (1.0, 0.0) };
    let attn_params = (total_params as f64 * attn_share) as u64;
    let expert_params = (total_params as f64 * expert_share) as u64;

    let bytes_per_attn = match stack {
        CompressionStack::Bf16Only => BYTES_PER_PARAM_BF16,
        CompressionStack::Fp8Only => BYTES_PER_PARAM_FP8,
        CompressionStack::Nvfp4 => BYTES_PER_PARAM_NVFP4,
        CompressionStack::Qtip2Only | CompressionStack::Qtip2PlusTdMoe => BYTES_PER_PARAM_QTIP2,
    };

    let bytes_per_expert = match stack {
        CompressionStack::Bf16Only => BYTES_PER_PARAM_BF16,
        CompressionStack::Fp8Only => BYTES_PER_PARAM_FP8,
        CompressionStack::Nvfp4 => BYTES_PER_PARAM_NVFP4,
        CompressionStack::Qtip2Only => BYTES_PER_PARAM_QTIP2,
        CompressionStack::Qtip2PlusTdMoe => BYTES_PER_PARAM_TD_MOE_EXPERT,
    };

    let attn_bytes = (attn_params as f64 * bytes_per_attn) as u64;
    let expert_bytes = (expert_params as f64 * bytes_per_expert) as u64;
    attn_bytes + expert_bytes
}

/// Rough workspace estimate: 8% of weights, capped at 8 GB.
/// This covers activation buffers, PagedAttention block table, and
/// scratch space for the dispatcher. Real runs may be ±2 GB off; that's
/// why `--target-hbm` has a small head-room.
fn estimate_workspace_bytes(weight_bytes: u64) -> u64 {
    let frac = (weight_bytes as f64 * 0.08) as u64;
    let cap = 8u64 * 1024 * 1024 * 1024;
    frac.min(cap)
}

/// Rough KV cache estimate at default settings (32 concurrent seqs, 32k context).
/// V4 MLA KV is small per-token; we use 0.6 GB as a placeholder for the mock.
///
/// Note: the real-GPU path does *not* use this — it loads with paged-attn off
/// + max_seqs=1, so the KV portion of the measurement is essentially zero
/// and folds into "workspace." This estimate is mock-mode only.
fn estimate_kv_bytes() -> u64 {
    (0.6 * 1024.0 * 1024.0 * 1024.0) as u64
}

const GB: f64 = 1024.0 * 1024.0 * 1024.0;

/// Convert bytes to gigabytes (binary GB).
/// 🔴 Decide `(pass, unmeasured)` from one post-load HBM delta.
///
/// **A zero delta is an environment failure, not a pass.**
///
/// The real-GPU path computes `used_bytes = free_before.saturating_sub(
/// free_after)`. If the model loaded on the CPU, or the two snapshots straddled
/// a CUDA driver state reset, or free memory simply did not go down, that
/// expression is `0` — and the verdict used to be `total_gb <= target`, i.e.
/// `0.0 <= 60.0`, i.e. **true**. The gate printed `PASS` and exited `0`. That
/// is this tool certifying "V4 fits in 60 GB on this H100" from a run in which
/// V4 never touched the H100, and it is the worst failure a rental gate can
/// have: the number it invents is the number a buying decision rests on. The
/// zero case *was* detected — and pushed a free-form `note`. A note changes no
/// exit code and nothing in CI reads one.
///
/// This lives outside `#[cfg(feature = "cuda")]` **on purpose**. The defect sat
/// in CUDA-gated code that no CPU CI job ever compiled, let alone ran, so a
/// test next to it would have been as unreachable as the bug. Keeping the
/// decision in always-compiled code is what makes
/// [`tests::zero_hbm_delta_is_unmeasured_not_a_pass`] able to fail on a laptop.
pub fn hbm_verdict(used_bytes: u64, total_gb: f64, target_hbm_gb: f64) -> (bool, bool) {
    let unmeasured = used_bytes == 0;
    (!unmeasured && total_gb <= target_hbm_gb, unmeasured)
}

/// Process exit code for a finished report: `0` fits, `1` measured and does not
/// fit, `2` we do not know.
///
/// Kept separate from [`run`] so the mapping is testable without a GPU, a
/// model download, or stdout capture.
pub fn exit_code(report: &HbmReport) -> i32 {
    if report.unmeasured {
        2
    } else if report.pass {
        0
    } else {
        1
    }
}

fn to_gb(bytes: u64) -> f64 {
    bytes as f64 / GB
}

/// Run the `--mock` path: synthesize a footprint report from the config table
/// without touching any GPU. Always succeeds; the pass/fail is decided purely
/// by `measured.total_gb <= target_gb`.
pub fn run_mock(opts: &HbmValidateOptions) -> Result<HbmReport> {
    let (total_params, _active_params, has_moe) = known_model_params(&opts.model_id)
        .ok_or_else(|| {
            anyhow!(
                "unknown model `{}` in --mock mode. Add an entry to known_model_params() in arc-cli/src/validate.rs, \
                or run without --mock on a CUDA host.",
                opts.model_id
            )
        })?;

    let weight_bytes = estimate_weight_bytes(total_params, has_moe, opts.compression_stack);
    let workspace_bytes = estimate_workspace_bytes(weight_bytes);
    let kv_bytes = estimate_kv_bytes();
    let total_bytes = weight_bytes + workspace_bytes + kv_bytes;

    let total_gb = to_gb(total_bytes);
    let pass = total_gb <= opts.target_hbm_gb;

    let mut notes = vec![
        "mock mode — numbers come from known_model_params() + bytes-per-param table, not real GPU"
            .to_string(),
        "for ground truth, run without --mock on a CUDA-enabled host".to_string(),
    ];
    if !pass {
        notes.push(format!(
            "FAIL: measured {:.2} GB exceeds target {:.2} GB by {:.2} GB",
            total_gb,
            opts.target_hbm_gb,
            total_gb - opts.target_hbm_gb
        ));
    }

    Ok(HbmReport {
        model: opts.model_id.clone(),
        compression_stack: opts.compression_stack.as_str().to_string(),
        target_gb: opts.target_hbm_gb,
        measured: MeasuredFootprint {
            total_gb,
            weight_gb: to_gb(weight_bytes),
            workspace_gb: to_gb(workspace_bytes),
            kv_estimate_gb: to_gb(kv_bytes),
        },
        pass,
        // The mock path computes rather than measures, so it always produces a
        // number. It can be wrong; it cannot be absent.
        unmeasured: false,
        mode: "mock".to_string(),
        gpu: None,
        notes,
    })
}

/// Map a compression stack to the ISQ string the mistralrs loader understands.
///
/// `None` means "do not apply in-situ quantization" — the loader honors whatever
/// `quantization_config` is in the model's `config.json`. That covers both the
/// QTIP2 case (V4-style models ship with `quant_method: "qtip"` baked in) and
/// the NVFP4 case (model is pre-quantized at rest). The ISQ-string overrides
/// here are for stacks that *force* on-the-fly conversion from a BF16 base.
#[cfg(feature = "cuda")]
fn isq_string_for_stack(stack: CompressionStack) -> Option<String> {
    match stack {
        // No ISQ: pass-through whatever the model's config.json specifies. For
        // BF16-only this means "leave weights in BF16"; for QTIP2/NVFP4 this
        // means "trust the on-disk quantization." This matches what `arc bench`
        // does today.
        CompressionStack::Bf16Only
        | CompressionStack::Nvfp4
        | CompressionStack::Qtip2Only
        | CompressionStack::Qtip2PlusTdMoe => None,
        // FP8-only is the one stack we'd typically force from a BF16 base when
        // benchmarking on H100. mistralrs's `parse_isq_value` accepts "fp8".
        CompressionStack::Fp8Only => Some("fp8".to_string()),
    }
}

/// Build a `ModelSelected::Plain` for a given HuggingFace model id with all
/// the defaults that match the `arc bench` loader path. We use `Plain` (not
/// `Run` / auto-loader) because we want the loader to use the model's own
/// `config.json` to determine architecture — V4 ships as a Plain text model.
#[cfg(feature = "cuda")]
fn build_plain_model_selected(model_id: &str) -> mistralrs_core::ModelSelected {
    use mistralrs_core::{AutoDeviceMapParams, ModelDType, ModelSelected};
    ModelSelected::Plain {
        model_id: model_id.to_string(),
        tokenizer_json: None,
        arch: None,
        dtype: ModelDType::Auto,
        topology: None,
        organization: None,
        write_uqff: None,
        from_uqff: None,
        imatrix: None,
        calibration_file: None,
        max_seq_len: AutoDeviceMapParams::DEFAULT_MAX_SEQ_LEN,
        max_batch_size: AutoDeviceMapParams::DEFAULT_MAX_BATCH_SIZE,
        hf_cache_path: None,
        matformer_config_path: None,
        matformer_slice_name: None,
    }
}

/// Parse the `ARC_VALIDATE_DEVICE_INDEX` env var (or default to 0). This lets
/// us point the validate run at a non-default GPU on multi-GPU rentals without
/// adding a new CLI flag for what is intrinsically a single-device measurement.
#[cfg(feature = "cuda")]
fn cuda_device_index() -> usize {
    std::env::var("ARC_VALIDATE_DEVICE_INDEX")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(0)
}

/// Run the real-GPU path. Without the `cuda` feature, this returns a clear
/// error rather than silently mocking — the operator should know they need
/// to pass `--mock` on a CPU-only host.
///
/// **What this actually does:**
///
/// 1. Bind to CUDA device `ARC_VALIDATE_DEVICE_INDEX` (default 0).
/// 2. Snapshot free HBM via `cudaMemGetInfo`.
/// 3. Build a `MistralRsForServerBuilder` with the user's model id, pinned to
///    that device, with PagedAttention disabled and `max_seqs=1` so the load
///    grabs roughly only the model weights (plus a small workspace), not a
///    full KV-cache reservation. The ISQ string (if any) is derived from the
///    compression-stack flag via [`isq_string_for_stack`].
/// 4. Synchronously drive that builder's async `build()` to completion through
///    a fresh Tokio current-thread runtime — this matches what `arc bench`
///    does (which goes through `tokio::main`) but lets us stay inside a sync
///    function.
/// 5. Snapshot free HBM again.
/// 6. Attribute the delta into weight / workspace / KV with the same accountant
///    used by the mock path.
///
/// We deliberately keep a strong reference to the `MistralRs` instance until
/// after the second HBM snapshot — dropping it earlier would release the
/// allocations and skew the measurement to zero.
#[cfg(feature = "cuda")]
pub fn run_real_gpu(opts: &HbmValidateOptions) -> Result<HbmReport> {
    use candle_core::Device;
    use mistralrs_core::{MemoryUsage, TokenSource};
    use mistralrs_server_core::mistralrs_for_server_builder::MistralRsForServerBuilder;

    let device_index = cuda_device_index();
    let device = Device::new_cuda(device_index)
        .with_context(|| format!("failed to initialise CUDA device {device_index}"))?;

    let mem = MemoryUsage;
    let total_hbm_bytes = mem
        .get_total_memory(&device)
        .context("failed to query total HBM")? as u64;
    let free_before = mem
        .get_memory_available(&device)
        .context("failed to query free HBM before load")? as u64;

    // Build the model selector + ISQ string the same way `arc bench` /
    // `arc serve` would, just with the minimum knobs needed for a memory-only
    // probe (single-seq, no paged-attention so we don't pre-reserve KV).
    let model_selected = build_plain_model_selected(&opts.model_id);
    let isq = isq_string_for_stack(opts.compression_stack);

    // The builder is async; run it on a fresh single-threaded Tokio runtime so
    // we don't need to make `run_real_gpu` itself async. This mirrors the
    // pattern used by `arc-cli`'s schema-mode `run_validate` (sync entry point
    // that's invoked from a sync `main`).
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("failed to build Tokio runtime for model load")?;

    // Keep the loaded `MistralRs` alive across the post-load snapshot so the
    // GPU allocations remain resident when we measure. We hold it in `_keepalive`
    // until after the second `mem.get_memory_available` call.
    let _keepalive = runtime.block_on(async {
        MistralRsForServerBuilder::new()
            .with_model(model_selected)
            .with_device(device.clone())
            .with_max_seqs(1)
            .with_interactive_mode(false)
            // Disable prefix caching so we don't carry an extra HBM allocation
            // we'd then have to subtract back out.
            .with_prefix_cache_n(0)
            .with_token_source(TokenSource::CacheToken)
            // Disable paged attention so the measurement reflects model
            // residency, not a 90%-of-HBM KV-cache reservation.
            .set_paged_attn(Some(false))
            .with_in_situ_quant_optional(isq)
            .build()
            .await
            .context("failed to load model for HBM measurement")
    })?;

    let free_after = mem
        .get_memory_available(&device)
        .context("failed to query free HBM after load")? as u64;

    let used_bytes = free_before.saturating_sub(free_after);
    let workspace_bytes = estimate_workspace_bytes(used_bytes);
    // With paged-attn disabled and max_seqs=1, the KV reservation is the
    // minimum the runtime needs to satisfy `Pipeline::get_metadata` — call it
    // zero for accounting purposes and let it fall into "workspace" if it's
    // non-trivial. This keeps the gpu-mode breakdown comparable to the mock
    // path's structure even though the totals come from a different source.
    let kv_bytes: u64 = 0;
    let weight_bytes = used_bytes.saturating_sub(workspace_bytes + kv_bytes);

    let total_gb = to_gb(used_bytes);
    let (pass, unmeasured) = hbm_verdict(used_bytes, total_gb, opts.target_hbm_gb);

    let mut notes = vec![
        format!(
            "gpu mode — total HBM={:.1} GB, free before load={:.1} GB, free after={:.1} GB",
            to_gb(total_hbm_bytes),
            to_gb(free_before),
            to_gb(free_after)
        ),
        format!(
            "loader: MistralRsForServerBuilder (paged-attn=off, max_seqs=1, isq={})",
            isq_string_for_stack(opts.compression_stack)
                .as_deref()
                .unwrap_or("none")
        ),
    ];
    if unmeasured {
        notes.push(
            "UNMEASURED: post-load HBM delta is 0 — either the model loaded entirely on CPU \
             (no device map?) or the snapshots straddle a CUDA driver state reset. \
             This run proves nothing about residency; exit code is 2, not 0 or 1. \
             Inspect logs."
                .to_string(),
        );
    }
    if !pass {
        notes.push(format!(
            "FAIL: measured {:.2} GB exceeds target {:.2} GB by {:.2} GB",
            total_gb,
            opts.target_hbm_gb,
            total_gb - opts.target_hbm_gb
        ));
    }

    let report = HbmReport {
        model: opts.model_id.clone(),
        compression_stack: opts.compression_stack.as_str().to_string(),
        target_gb: opts.target_hbm_gb,
        measured: MeasuredFootprint {
            total_gb,
            weight_gb: to_gb(weight_bytes),
            workspace_gb: to_gb(workspace_bytes),
            kv_estimate_gb: to_gb(kv_bytes),
        },
        pass,
        unmeasured,
        mode: "gpu".to_string(),
        gpu: Some(GpuInfo {
            name: detect_gpu_name(),
            total_hbm_gb: to_gb(total_hbm_bytes),
        }),
        notes,
    };

    // Now that we've captured the post-load snapshot, the runtime + builder
    // can drop, freeing HBM. We do this explicitly so the order is obvious to
    // future readers (rather than relying on end-of-scope drop ordering).
    drop(_keepalive);
    drop(runtime);

    Ok(report)
}

/// CPU-only build path: emit a clear error so the operator knows to pass
/// `--mock` or rebuild with `--features cuda`.
#[cfg(not(feature = "cuda"))]
pub fn run_real_gpu(_opts: &HbmValidateOptions) -> Result<HbmReport> {
    Err(anyhow!(
        "real-GPU memory query requires the `cuda` feature.\n\
         Rebuild: cargo build --release -p arc-cli --features cuda\n\
         Or pass --mock to use the offline estimate (CI-only)."
    ))
}

#[cfg(feature = "cuda")]
fn detect_gpu_name() -> String {
    // Try `nvidia-smi --query-gpu=name --format=csv,noheader` for a friendly
    // name. Fall back to "cuda:0" if the call fails.
    std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=name", "--format=csv,noheader"])
        .output()
        .ok()
        .and_then(|o| {
            if o.status.success() {
                Some(String::from_utf8_lossy(&o.stdout).trim().to_string())
            } else {
                None
            }
        })
        .unwrap_or_else(|| "cuda:0".to_string())
}

/// Write the JSON report to `path`, creating parent directories as needed.
pub fn write_report(report: &HbmReport, path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create parent directory for {}", path.display()))?;
    }
    let json =
        serde_json::to_string_pretty(report).context("failed to serialise HBM report to JSON")?;
    std::fs::write(path, json)
        .with_context(|| format!("failed to write HBM report to {}", path.display()))?;
    Ok(())
}

/// Top-level entry point used by `main.rs`. Returns the exit code for the
/// process (0 if the report passes, 1 if it fails).
pub fn run(opts: HbmValidateOptions) -> Result<i32> {
    let report = if opts.mock {
        run_mock(&opts)?
    } else {
        run_real_gpu(&opts)?
    };

    write_report(&report, &opts.output_path)?;

    println!("Arc HBM footprint validation");
    println!("============================");
    println!("Model:             {}", report.model);
    println!("Compression stack: {}", report.compression_stack);
    println!("Mode:              {}", report.mode);
    if let Some(gpu) = &report.gpu {
        println!(
            "GPU:               {} ({:.1} GB total)",
            gpu.name, gpu.total_hbm_gb
        );
    }
    println!();
    println!("  weight:    {:>7.2} GB", report.measured.weight_gb);
    println!("  workspace: {:>7.2} GB", report.measured.workspace_gb);
    println!("  kv (est.): {:>7.2} GB", report.measured.kv_estimate_gb);
    println!("  ---------- -------");
    println!("  total:     {:>7.2} GB", report.measured.total_gb);
    println!("  target:    {:>7.2} GB", report.target_gb);
    println!();
    if report.unmeasured {
        println!("  result: UNMEASURED (no HBM was consumed — this run proves nothing)");
    } else if report.pass {
        println!("  result: PASS (measured <= target)");
    } else {
        println!(
            "  result: FAIL (measured exceeds target by {:.2} GB)",
            report.measured.total_gb - report.target_gb
        );
    }
    if !report.notes.is_empty() {
        println!();
        println!("Notes:");
        for n in &report.notes {
            println!("  - {n}");
        }
    }
    println!();
    println!("Report written to: {}", opts.output_path.display());

    // 0 = fits. 1 = measured, and does not fit. 2 = we do not know.
    //
    // The three must stay distinct. `1` is a verdict about the model and CI is
    // entitled to act on it; `2` is a verdict about the run and means the
    // operator has to look at the box. Collapsing "unmeasured" into `1` would
    // read as "V4 does not fit" — the opposite error to the one this code used
    // to make, and just as wrong. `2` also matches what `main.rs` already exits
    // with for a bad `--compression-stack` or a load error, so callers need no
    // new rule: **non-zero-and-not-1 means the environment, not the model.**
    Ok(exit_code(&report))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    /// Mint a unique path inside the OS temp dir for a test. We avoid a
    /// `tempfile` dependency to keep arc-cli's dep list minimal.
    fn unique_tmp_path(suffix: &str) -> PathBuf {
        static N: AtomicU64 = AtomicU64::new(0);
        let n = N.fetch_add(1, Ordering::SeqCst);
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        std::env::temp_dir()
            .join(format!("arc-cli-test-{}-{}-{}", std::process::id(), ts, n))
            .join(suffix)
    }

    /// RAII guard that deletes a directory tree when dropped. Replaces
    /// `tempfile::TempDir` so we can keep arc-cli dep-free.
    struct TmpDir(PathBuf);
    impl TmpDir {
        fn new() -> Self {
            let p = unique_tmp_path("dir");
            std::fs::create_dir_all(&p).unwrap();
            Self(p)
        }
        fn path(&self) -> &Path {
            &self.0
        }
    }
    impl Drop for TmpDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn parse_compression_stack_known_values() {
        assert_eq!(
            CompressionStack::parse("bf16").unwrap(),
            CompressionStack::Bf16Only
        );
        assert_eq!(
            CompressionStack::parse("fp8-only").unwrap(),
            CompressionStack::Fp8Only
        );
        assert_eq!(
            CompressionStack::parse("nvfp4").unwrap(),
            CompressionStack::Nvfp4
        );
        assert_eq!(
            CompressionStack::parse("qtip2+td-moe").unwrap(),
            CompressionStack::Qtip2PlusTdMoe
        );
        assert_eq!(
            CompressionStack::parse("QTIP2+TD-MOE").unwrap(),
            CompressionStack::Qtip2PlusTdMoe
        );
        assert!(CompressionStack::parse("garbage").is_err());
    }

    #[test]
    fn weight_bytes_shrink_with_more_aggressive_compression() {
        // For an MoE model, more compression = fewer bytes.
        let bf16 = estimate_weight_bytes(140_000_000_000, true, CompressionStack::Bf16Only);
        let fp8 = estimate_weight_bytes(140_000_000_000, true, CompressionStack::Fp8Only);
        let qtip2 = estimate_weight_bytes(140_000_000_000, true, CompressionStack::Qtip2PlusTdMoe);
        assert!(bf16 > fp8, "bf16 should be heavier than fp8");
        assert!(fp8 > qtip2, "fp8 should be heavier than qtip2+tdmoe");
        // Sanity: 140B params under BF16 is ~280 GB.
        let bf16_gb = bf16 as f64 / GB;
        assert!(
            bf16_gb > 240.0 && bf16_gb < 280.0,
            "bf16 weight footprint should be ~260 GB, got {bf16_gb:.1}"
        );
    }

    #[test]
    fn known_model_params_v4_flash() {
        // HF-verified: 284B logical / 13B active (see known_model_params docs).
        let (total, active, has_moe) = known_model_params("deepseek-ai/DeepSeek-V4-Flash").unwrap();
        assert_eq!(total, 284_000_000_000);
        assert_eq!(active, 13_000_000_000);
        assert!(has_moe);
    }

    #[test]
    fn known_model_params_v4_pro() {
        // Official announcement figure: 1.6T / 49B active (API-tier model).
        let (total, active, has_moe) = known_model_params("deepseek-ai/DeepSeek-V4-Pro").unwrap();
        assert_eq!(total, 1_600_000_000_000);
        assert_eq!(active, 49_000_000_000);
        assert!(has_moe);
    }

    #[test]
    fn known_model_params_unknown_model_returns_none() {
        assert!(known_model_params("foo/bar").is_none());
    }

    #[test]
    fn mock_v4_flash_qtip2_fails_60gb_but_passes_h200_target() {
        // At the corrected 284B logical params, the deliberately-conservative
        // 20/80 mock split lands at ~116 GiB — over an 80 GB H100 ceiling but
        // under a 141 GB H200. (The *measured* artifact is 68 GB and serves on
        // one H100/H200 — the mock overestimates by design; see fn docs.)
        let mk = |target_hbm_gb: f64| HbmValidateOptions {
            model_id: "deepseek-ai/DeepSeek-V4-Flash".to_string(),
            compression_stack: CompressionStack::Qtip2PlusTdMoe,
            target_hbm_gb,
            output_path: PathBuf::from("/tmp/unused.json"),
            mock: true,
        };

        let tight = run_mock(&mk(60.0)).unwrap();
        assert!(
            !tight.pass,
            "mock V4 Flash @ qtip2+td-moe should exceed 60 GB (conservative split); got {:.2} GB",
            tight.measured.total_gb
        );

        let h200 = run_mock(&mk(141.0)).unwrap();
        assert!(
            h200.pass,
            "mock V4 Flash @ qtip2+td-moe should fit an H200-class 141 GB; got {:.2} GB",
            h200.measured.total_gb
        );
        assert_eq!(h200.mode, "mock");
        assert_eq!(h200.compression_stack, "qtip2+td-moe");
        assert!(h200.measured.weight_gb > 0.0);
        assert!(h200.measured.workspace_gb > 0.0);
    }

    #[test]
    fn mock_v4_flash_bf16_fails_60gb_target() {
        let opts = HbmValidateOptions {
            model_id: "deepseek-ai/DeepSeek-V4-Flash".to_string(),
            compression_stack: CompressionStack::Bf16Only,
            target_hbm_gb: 60.0,
            output_path: PathBuf::from("/tmp/unused.json"),
            mock: true,
        };
        let report = run_mock(&opts).unwrap();
        assert!(
            !report.pass,
            "V4 Flash @ bf16 should NOT fit in 60 GB; got {:.2} GB",
            report.measured.total_gb
        );
        assert!(report.notes.iter().any(|n| n.contains("FAIL")));
    }

    #[test]
    fn mock_unknown_model_errors() {
        let opts = HbmValidateOptions {
            model_id: "this-model-does-not-exist/foo".to_string(),
            compression_stack: CompressionStack::Qtip2PlusTdMoe,
            target_hbm_gb: 60.0,
            output_path: PathBuf::from("/tmp/unused.json"),
            mock: true,
        };
        let err = run_mock(&opts).unwrap_err();
        assert!(err.to_string().contains("unknown model"));
    }

    #[test]
    fn write_report_creates_parent_directories() {
        let tmp = TmpDir::new();
        let nested = tmp.path().join("a/b/c/report.json");
        let report = HbmReport {
            model: "test".into(),
            compression_stack: "qtip2+td-moe".into(),
            target_gb: 60.0,
            measured: MeasuredFootprint {
                total_gb: 50.0,
                weight_gb: 40.0,
                workspace_gb: 5.0,
                kv_estimate_gb: 5.0,
            },
            pass: true,
            unmeasured: false,
            mode: "mock".into(),
            gpu: None,
            notes: vec![],
        };
        write_report(&report, &nested).unwrap();
        assert!(nested.exists());

        // Round-trip the JSON to be sure the schema is parseable.
        let raw = std::fs::read_to_string(&nested).unwrap();
        let parsed: HbmReport = serde_json::from_str(&raw).unwrap();
        assert_eq!(parsed.model, "test");
        assert_eq!(parsed.compression_stack, "qtip2+td-moe");
        assert!(parsed.pass);
    }

    #[test]
    fn run_end_to_end_writes_json_and_returns_zero() {
        let tmp = TmpDir::new();
        let out = tmp.path().join("v4_flash_h200_footprint.json");
        let opts = HbmValidateOptions {
            model_id: "deepseek-ai/DeepSeek-V4-Flash".to_string(),
            compression_stack: CompressionStack::Qtip2PlusTdMoe,
            // H200-class ceiling: the conservative mock lands at ~116 GiB for
            // the corrected 284B params (measured artifact is 68 GB).
            target_hbm_gb: 141.0,
            output_path: out.clone(),
            mock: true,
        };
        let exit = run(opts).unwrap();
        assert_eq!(
            exit, 0,
            "qtip2+td-moe on V4 Flash should pass a 141 GB target"
        );
        assert!(out.exists());
        let parsed: HbmReport =
            serde_json::from_str(&std::fs::read_to_string(&out).unwrap()).unwrap();
        assert!(parsed.pass);
        assert_eq!(parsed.mode, "mock");
    }

    /// 🔴 **The regression this file exists to never repeat.**
    ///
    /// `used_bytes = free_before.saturating_sub(free_after)` is `0` whenever
    /// the model did not become resident. The old verdict was
    /// `pass = total_gb <= target` — `0.0 <= 60.0` — so the gate printed PASS
    /// and exited 0 off a run in which nothing loaded onto the GPU.
    ///
    /// This test is in always-compiled code deliberately: the defect lived
    /// inside `#[cfg(feature = "cuda")]`, which no CPU CI job compiles, so a
    /// test next to the bug would have been as unreachable as the bug.
    #[test]
    fn zero_hbm_delta_is_unmeasured_not_a_pass() {
        let (pass, unmeasured) = hbm_verdict(0, 0.0, 60.0);
        assert!(
            !pass,
            "a zero HBM delta must never pass — 0.0 <= 60.0 is not evidence"
        );
        assert!(
            unmeasured,
            "a zero HBM delta must be reported as unmeasured"
        );
    }

    /// The zero check must not swallow real results in either direction.
    #[test]
    fn nonzero_hbm_delta_is_judged_against_the_target() {
        // 40 GiB measured against a 60 GB target: fits.
        let used = 40 * 1024 * 1024 * 1024;
        let (pass, unmeasured) = hbm_verdict(used, to_gb(used), 60.0);
        assert!(pass && !unmeasured, "40 GiB must pass a 60 GB target");

        // 80 GiB measured against the same target: a real failure, and it must
        // stay distinguishable from "we did not look".
        let used = 80 * 1024 * 1024 * 1024;
        let (pass, unmeasured) = hbm_verdict(used, to_gb(used), 60.0);
        assert!(!pass, "80 GiB must fail a 60 GB target");
        assert!(
            !unmeasured,
            "80 GiB was measured — it is a model verdict, not an env one"
        );

        // Exactly on the line still passes; the contract is `<=`.
        let (pass, _) = hbm_verdict(1, 60.0, 60.0);
        assert!(pass, "the target is inclusive");
    }

    /// Exit codes must stay three-valued: 0 fits, 1 does not fit, **2 unknown**.
    ///
    /// Collapsing unknown into 1 would read as "V4 does not fit" — the opposite
    /// error to the one this code used to make, and just as wrong.
    #[test]
    fn exit_code_separates_the_model_verdict_from_the_environment() {
        let mk = |pass: bool, unmeasured: bool| HbmReport {
            model: "m".into(),
            compression_stack: "bf16".into(),
            target_gb: 60.0,
            measured: MeasuredFootprint {
                total_gb: 0.0,
                weight_gb: 0.0,
                workspace_gb: 0.0,
                kv_estimate_gb: 0.0,
            },
            pass,
            unmeasured,
            mode: "gpu".into(),
            gpu: None,
            notes: vec![],
        };
        assert_eq!(exit_code(&mk(true, false)), 0, "fits");
        assert_eq!(exit_code(&mk(false, false)), 1, "measured, does not fit");
        assert_eq!(exit_code(&mk(false, true)), 2, "unmeasured");
        // Belt and braces: even a report that somehow claims both must not
        // report success, because `unmeasured` is the stronger statement.
        assert_eq!(exit_code(&mk(true, true)), 2, "unmeasured wins over pass");
    }

    /// An older report on disk, written before `unmeasured` existed, must still
    /// parse — the JSON file is a cross-version contract with the rental step.
    #[test]
    fn reports_without_the_unmeasured_field_still_parse() {
        let legacy = r#"{
            "model": "deepseek-ai/DeepSeek-V4-Flash",
            "compression_stack": "qtip2+td-moe",
            "target_gb": 60.0,
            "measured": {"total_gb": 55.0, "weight_gb": 50.0, "workspace_gb": 4.0, "kv_estimate_gb": 1.0},
            "pass": true,
            "mode": "gpu",
            "gpu": null,
            "notes": []
        }"#;
        let parsed: HbmReport = serde_json::from_str(legacy).expect("legacy report parses");
        assert!(parsed.pass);
        assert!(!parsed.unmeasured, "absent means false, not unknown");
    }

    #[test]
    fn run_end_to_end_returns_one_when_target_too_tight() {
        let tmp = TmpDir::new();
        let out = tmp.path().join("tight.json");
        let opts = HbmValidateOptions {
            model_id: "deepseek-ai/DeepSeek-V4-Flash".to_string(),
            compression_stack: CompressionStack::Bf16Only,
            target_hbm_gb: 60.0,
            output_path: out.clone(),
            mock: true,
        };
        let exit = run(opts).unwrap();
        assert_eq!(exit, 1, "bf16 should not fit in 60 GB");
        assert!(out.exists());
    }

    // -------- ISQ-string-for-stack mapping (compiles regardless of `cuda`
    // feature, but only meaningful on CUDA builds; we re-implement the same
    // logic on non-CUDA so the test still proves the mapping is stable). ----

    #[cfg(feature = "cuda")]
    #[test]
    fn isq_string_for_stack_maps_fp8_only_to_fp8() {
        assert_eq!(
            isq_string_for_stack(CompressionStack::Fp8Only).as_deref(),
            Some("fp8")
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn isq_string_for_stack_returns_none_for_config_driven_stacks() {
        // BF16: nothing to do. QTIP2 / NVFP4: model's config.json drives it.
        assert!(isq_string_for_stack(CompressionStack::Bf16Only).is_none());
        assert!(isq_string_for_stack(CompressionStack::Nvfp4).is_none());
        assert!(isq_string_for_stack(CompressionStack::Qtip2Only).is_none());
        assert!(isq_string_for_stack(CompressionStack::Qtip2PlusTdMoe).is_none());
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn build_plain_model_selected_round_trip() {
        use mistralrs_core::ModelSelected;
        let s = build_plain_model_selected("deepseek-ai/DeepSeek-V4-Flash");
        match s {
            ModelSelected::Plain { model_id, arch, .. } => {
                assert_eq!(model_id, "deepseek-ai/DeepSeek-V4-Flash");
                // We always leave `arch = None` so the loader auto-detects.
                assert!(arch.is_none());
            }
            _ => panic!("expected ModelSelected::Plain"),
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_device_index_defaults_to_zero() {
        // Don't override the env var here — just check the default codepath.
        // (We can't safely mutate process env in parallel tests, so we only
        // assert behavior when the var is absent or empty.)
        if std::env::var("ARC_VALIDATE_DEVICE_INDEX").is_err() {
            assert_eq!(cuda_device_index(), 0);
        }
    }

    /// Smoke test for the real-GPU path. This is gated behind both `cfg(test)`
    /// and `feature = "cuda"`, so it compiles only on CUDA-enabled builds.
    /// At runtime, we additionally check whether the harness host actually has
    /// a CUDA device — if not, we skip cleanly. That way the test can ship in
    /// the binary and be a no-op on non-GPU CI runners while exercising the
    /// real loader on rental hardware.
    ///
    /// The model id is deliberately a tiny dense Llama (3.1 8B Instruct) so
    /// the load fits comfortably on any modern GPU and finishes in seconds.
    /// We do NOT test against V4 here because:
    ///   1. V4 weights are 100+ GB and would force an HBM-class GPU just for
    ///      the unit-test loop;
    ///   2. the integration-level "does V4 fit in 60 GB?" question is what
    ///      the binary itself is for — running it on rental hardware is the
    ///      validation step, not a unit test.
    #[cfg(all(test, feature = "cuda"))]
    #[test]
    fn real_gpu_load_small_model_smoke() {
        use candle_core::Device;

        // Skip if there's no usable CUDA device — this is normal on Mac CI.
        if Device::new_cuda(0).is_err() {
            eprintln!("skipping real_gpu_load_small_model_smoke: no CUDA device");
            return;
        }

        let opts = HbmValidateOptions {
            // Small dense model — every Arc dev box has it cached locally
            // after a single `arc bench` run.
            model_id: "meta-llama/Llama-3.1-8B-Instruct".to_string(),
            compression_stack: CompressionStack::Bf16Only,
            // Generous target so the test never spuriously fails on a 16-GB
            // dev GPU even if the workspace overhead is larger than expected.
            target_hbm_gb: 80.0,
            // run_real_gpu returns the report directly without writing — we
            // don't need a real file path here. `run()` is the wrapper that
            // serialises to disk.
            output_path: PathBuf::from("/tmp/unused-validate-smoke.json"),
            mock: false,
        };
        let report = run_real_gpu(&opts).expect("run_real_gpu succeeded");
        assert_eq!(report.mode, "gpu");
        assert_eq!(report.model, "meta-llama/Llama-3.1-8B-Instruct");
        assert!(
            report.measured.total_gb > 0.0,
            "expected non-zero HBM delta after real model load"
        );
        // 8B in BF16 ≈ 16 GB; the loader's workspace pushes that a bit higher.
        // Bound generously to avoid env-dependent flakiness — the key check is
        // "we actually consumed HBM," not "we consumed exactly N bytes."
        assert!(
            report.measured.total_gb < 60.0,
            "8B BF16 should fit in <60 GB on any test GPU; got {:.2}",
            report.measured.total_gb
        );
    }
}
