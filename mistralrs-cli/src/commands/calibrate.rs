//! `calibrate` command — collect per-linear activation statistics.
//!
//! Loads the model **unquantized**, sweeps a text corpus through it
//! forward-only, and writes a `.arccalib` artifact keyed by ISQ layer index.
//! No quantization happens: `--isq` is deliberately not accepted, and no UQFF
//! is written.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use tracing::{info, warn};

use mistralrs_core::{
    initialize_logging, set_calibration_request, CalibrationRequest, IsqOrganization, ModelSelected,
};
use mistralrs_quant::{CalibOptions, CalibrationArtifact, GramMode, CALIB_EXTENSION};
use mistralrs_server_core::mistralrs_for_server_builder::{defaults, MistralRsForServerBuilder};

use crate::args::{
    CalibrateCollectionOptions, CalibrateDeviceOptions, CalibrateModelType, GlobalOptions,
    GramModeArg,
};

/// Corpus shipped with the repository, used when `--corpus` is omitted.
const DEFAULT_CORPUS: &str = "calibration_data/calibration_datav3_small.txt";
/// Fallback if only the full corpus is present.
const DEFAULT_CORPUS_FULL: &str = "calibration_data/calibration_datav3.txt";

fn parts(
    model_type: &CalibrateModelType,
) -> (
    &CalibrateCollectionOptions,
    &CalibrateDeviceOptions,
    &PathBuf,
    &str,
) {
    match model_type {
        CalibrateModelType::Auto {
            model,
            collection,
            device,
            output,
        } => (collection, device, &output.out, &model.model_id),
        CalibrateModelType::Text {
            model,
            collection,
            device,
            output,
            ..
        } => (collection, device, &output.out, &model.model_id),
    }
}

/// Resolve the calibration corpus, defaulting to the bundled one.
fn resolve_corpus(explicit: Option<&PathBuf>) -> Result<PathBuf> {
    if let Some(p) = explicit {
        if !p.exists() {
            anyhow::bail!("Calibration corpus `{}` does not exist", p.display());
        }
        return Ok(p.clone());
    }
    for candidate in [DEFAULT_CORPUS, DEFAULT_CORPUS_FULL] {
        let p = Path::new(candidate);
        if p.exists() {
            info!("Using bundled calibration corpus `{candidate}`");
            return Ok(p.to_path_buf());
        }
    }
    anyhow::bail!(
        "No calibration corpus given and neither `{DEFAULT_CORPUS}` nor `{DEFAULT_CORPUS_FULL}` \
         was found in the working directory. Pass --corpus <path> with a plain-text corpus \
         (a WikiText slice works well)."
    )
}

fn build_options(c: &CalibrateCollectionOptions) -> Result<CalibOptions> {
    let gram = match c.gram {
        GramModeArg::None => GramMode::DiagOnly,
        GramModeArg::Block => {
            if c.gram_block == 0 {
                anyhow::bail!("--gram-block must be non-zero");
            }
            GramMode::Blockwise {
                block: c.gram_block,
            }
        }
        GramModeArg::Full => GramMode::Full {
            max_dim: c.gram_max_dim,
        },
    };
    Ok(CalibOptions {
        gram,
        per_expert: c.per_expert,
        min_expert_tokens: c.min_expert_tokens,
    })
}

/// Run a calibration sweep and write the artifact.
pub async fn run_calibrate(model_type: CalibrateModelType, global: GlobalOptions) -> Result<()> {
    initialize_logging();

    let (collection, device, out, model_id) = parts(&model_type);
    let out = out.clone();
    let model_id = model_id.to_string();

    if out.extension().is_none_or(|e| e != CALIB_EXTENSION) {
        warn!(
            "Output `{}` does not use the conventional `.{CALIB_EXTENSION}` extension.",
            out.display()
        );
    }

    let corpus = resolve_corpus(collection.corpus.as_ref())?;
    let opts = build_options(collection)?;
    let arch = match &model_type {
        CalibrateModelType::Text { arch, .. } => arch.as_ref().map(|a| a.to_string()),
        CalibrateModelType::Auto { .. } => None,
    };

    info!(
        "Calibrating `{model_id}` from `{}` (samples: {}, gram: {:?}, per-expert: {}) -> `{}`",
        corpus.display(),
        collection
            .samples
            .map(|n| n.to_string())
            .unwrap_or_else(|| "all".to_string()),
        opts.gram,
        opts.per_expert,
        out.display()
    );

    set_calibration_request(CalibrationRequest {
        out: out.clone(),
        opts,
        max_chunks: collection.samples,
        model_id: model_id.clone(),
        arch,
    });

    let model_selected = convert_to_model_selected(&model_type, &corpus)?;

    // Building the pipeline performs the load + the forward-only sweep; the
    // loader writes the artifact at the end of the sweep.
    let _mistralrs = MistralRsForServerBuilder::new()
        .with_model(model_selected)
        .with_max_seqs(1)
        .with_no_kv_cache(defaults::NO_KV_CACHE)
        .with_token_source(global.token_source.clone())
        .with_interactive_mode(defaults::INTERACTIVE_MODE)
        .with_prefix_cache_n(0)
        .with_cpu(device.cpu)
        .with_num_device_layers_optional(device.device_layers.clone())
        .build()
        .await
        .context("model load / calibration sweep failed")?;

    let artifact = CalibrationArtifact::load(&out)
        .with_context(|| format!("calibration artifact `{}` was not written", out.display()))?;
    info!("Calibration complete:\n{}", artifact.summary());
    Ok(())
}

/// Build the `ModelSelected` for a calibration run: `calibration_file` set,
/// no ISQ, no UQFF output. That combination makes the loader run the sweep and
/// skip quantization entirely.
fn convert_to_model_selected(
    model_type: &CalibrateModelType,
    corpus: &Path,
) -> Result<ModelSelected> {
    let calibration_file = Some(corpus.to_path_buf());
    Ok(match model_type {
        CalibrateModelType::Auto {
            model,
            collection,
            device,
            ..
        } => ModelSelected::Run {
            model_id: model.model_id.clone(),
            tokenizer_json: model
                .tokenizer
                .as_ref()
                .map(|p| p.to_string_lossy().to_string()),
            dtype: model.dtype,
            topology: device
                .topology
                .as_ref()
                .map(|p| p.to_string_lossy().to_string()),
            organization: collection
                .isq_organization
                .or(Some(IsqOrganization::Default)),
            write_uqff: None,
            from_uqff: None,
            imatrix: None,
            calibration_file,
            max_edge: None,
            max_seq_len: device.max_seq_len,
            max_batch_size: device.max_batch_size,
            max_num_images: None,
            max_image_length: None,
            hf_cache_path: device.hf_cache.clone(),
            matformer_config_path: None,
            matformer_slice_name: None,
        },
        CalibrateModelType::Text {
            model,
            arch,
            collection,
            device,
            ..
        } => ModelSelected::Plain {
            model_id: model.model_id.clone(),
            tokenizer_json: model
                .tokenizer
                .as_ref()
                .map(|p| p.to_string_lossy().to_string()),
            arch: arch.clone(),
            dtype: model.dtype,
            topology: device
                .topology
                .as_ref()
                .map(|p| p.to_string_lossy().to_string()),
            organization: collection
                .isq_organization
                .or(Some(IsqOrganization::Default)),
            write_uqff: None,
            from_uqff: None,
            imatrix: None,
            calibration_file,
            max_seq_len: device.max_seq_len,
            max_batch_size: device.max_batch_size,
            hf_cache_path: device.hf_cache.clone(),
            matformer_config_path: None,
            matformer_slice_name: None,
        },
    })
}
