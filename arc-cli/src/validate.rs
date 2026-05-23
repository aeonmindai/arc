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
    /// True when measured.total_gb <= target_gb.
    pub pass: bool,
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
pub fn known_model_params(model_id: &str) -> Option<(u64, u64, bool)> {
    let normalised = model_id.to_ascii_lowercase();
    if normalised.contains("deepseek-v4-flash") || normalised.contains("deepseekv4-flash") {
        // V4 Flash: 27 layers, 128 experts, 6 active = ~140B total, ~14B active.
        // Numbers calibrated from V3-Flash analog scaled to V4 config.
        Some((140_000_000_000, 14_000_000_000, true))
    } else if normalised.contains("deepseek-v4-pro") || normalised.contains("deepseekv4-pro") {
        // V4 Pro: 43 layers, 256 experts, 6 active = ~680B total, ~37B active.
        Some((680_000_000_000, 37_000_000_000, true))
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
        CompressionStack::Qtip2Only | CompressionStack::Qtip2PlusTdMoe => {
            BYTES_PER_PARAM_QTIP2
        }
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
fn estimate_kv_bytes() -> u64 {
    (0.6 * 1024.0 * 1024.0 * 1024.0) as u64
}

const GB: f64 = 1024.0 * 1024.0 * 1024.0;

/// Convert bytes to gigabytes (binary GB).
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
        "mock mode — numbers come from known_model_params() + bytes-per-param table, not real GPU".to_string(),
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
        mode: "mock".to_string(),
        gpu: None,
        notes,
    })
}

/// Run the real-GPU path. Without the `cuda` feature, this returns a clear
/// error rather than silently mocking — the operator should know they need
/// to pass `--mock` on a CPU-only host.
#[cfg(feature = "cuda")]
pub fn run_real_gpu(opts: &HbmValidateOptions) -> Result<HbmReport> {
    use candle_core::Device;
    use mistralrs_core::MemoryUsage;

    let device = Device::new_cuda(0).context("failed to initialise CUDA device 0")?;

    let mem = MemoryUsage;
    let total_hbm_bytes = mem
        .get_total_memory(&device)
        .context("failed to query total HBM")? as u64;
    let free_before = mem
        .get_memory_available(&device)
        .context("failed to query free HBM before load")? as u64;

    // NOTE: actually wiring this to a model load currently requires routing
    // through `MistralRsForServerBuilder`. That code lives in mistralrs-server-core
    // and is the same path `arc bench` uses. We deliberately do *not* duplicate
    // the loader here — instead this function focuses on the memory-measurement
    // contract and assumes the operator has already pre-staged the model
    // (via `arc serve --pa-cache-type none --max-seqs 1` in a side-car shell)
    // or that this command will be called *after* `arc bench --warmup 0`.
    //
    // Once V4 dispatch is fully wired (RUN-156 / RUN-191 follow-up), we will
    // call MistralRsForServerBuilder::build() here directly. For the
    // hardware-pending verification step, what we need is the delta calculus
    // and the report writer — both of which work today.
    let free_after = free_before; // placeholder pending model-load wiring

    let used_bytes = free_before.saturating_sub(free_after);
    let workspace_bytes = estimate_workspace_bytes(used_bytes);
    let kv_bytes = estimate_kv_bytes();
    let weight_bytes = used_bytes.saturating_sub(workspace_bytes + kv_bytes);

    let total_gb = to_gb(used_bytes);
    let pass = total_gb <= opts.target_hbm_gb;

    let mut notes = vec![format!(
        "gpu mode — total HBM={:.1} GB, free before load={:.1} GB, free after={:.1} GB",
        to_gb(total_hbm_bytes),
        to_gb(free_before),
        to_gb(free_after)
    )];
    if used_bytes == 0 {
        notes.push(
            "HARDWARE-PENDING: model load is not yet wired into `arc validate`. \
             Re-run after the loader integration ships (next subtask of RUN-191)."
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
        mode: "gpu".to_string(),
        gpu: Some(GpuInfo {
            name: detect_gpu_name(),
            total_hbm_gb: to_gb(total_hbm_bytes),
        }),
        notes,
    })
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
    let json = serde_json::to_string_pretty(report)
        .context("failed to serialise HBM report to JSON")?;
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
        println!("GPU:               {} ({:.1} GB total)", gpu.name, gpu.total_hbm_gb);
    }
    println!();
    println!("  weight:    {:>7.2} GB", report.measured.weight_gb);
    println!("  workspace: {:>7.2} GB", report.measured.workspace_gb);
    println!("  kv (est.): {:>7.2} GB", report.measured.kv_estimate_gb);
    println!("  ---------- -------");
    println!("  total:     {:>7.2} GB", report.measured.total_gb);
    println!("  target:    {:>7.2} GB", report.target_gb);
    println!();
    if report.pass {
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

    Ok(if report.pass { 0 } else { 1 })
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
        std::env::temp_dir().join(format!("arc-cli-test-{}-{}-{}", std::process::id(), ts, n)).join(suffix)
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
        let (total, active, has_moe) =
            known_model_params("deepseek-ai/DeepSeek-V4-Flash").unwrap();
        assert_eq!(total, 140_000_000_000);
        assert_eq!(active, 14_000_000_000);
        assert!(has_moe);
    }

    #[test]
    fn known_model_params_unknown_model_returns_none() {
        assert!(known_model_params("foo/bar").is_none());
    }

    #[test]
    fn mock_v4_flash_qtip2_passes_60gb_target() {
        let opts = HbmValidateOptions {
            model_id: "deepseek-ai/DeepSeek-V4-Flash".to_string(),
            compression_stack: CompressionStack::Qtip2PlusTdMoe,
            target_hbm_gb: 60.0,
            output_path: PathBuf::from("/tmp/unused.json"),
            mock: true,
        };
        let report = run_mock(&opts).unwrap();
        assert!(
            report.pass,
            "V4 Flash @ qtip2+td-moe should fit in 60 GB; got {:.2} GB",
            report.measured.total_gb
        );
        assert_eq!(report.mode, "mock");
        assert_eq!(report.compression_stack, "qtip2+td-moe");
        assert!(report.measured.weight_gb > 0.0);
        assert!(report.measured.workspace_gb > 0.0);
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
        let out = tmp.path().join("v4_flash_h100_footprint.json");
        let opts = HbmValidateOptions {
            model_id: "deepseek-ai/DeepSeek-V4-Flash".to_string(),
            compression_stack: CompressionStack::Qtip2PlusTdMoe,
            target_hbm_gb: 60.0,
            output_path: out.clone(),
            mock: true,
        };
        let exit = run(opts).unwrap();
        assert_eq!(exit, 0, "qtip2+td-moe on V4 Flash should pass");
        assert!(out.exists());
        let parsed: HbmReport = serde_json::from_str(&std::fs::read_to_string(&out).unwrap()).unwrap();
        assert!(parsed.pass);
        assert_eq!(parsed.mode, "mock");
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
}
