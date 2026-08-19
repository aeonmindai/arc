//! `calibrate` command arguments — collect per-linear activation statistics.

use clap::{Args, Subcommand, ValueEnum};
use mistralrs_core::{AutoDeviceMapParams, IsqOrganization, ModelDType, NormalLoaderType};
use std::path::PathBuf;

/// Calibration model type selection. Only text-generation models are supported:
/// the sweep drives the model with tokenized text, so vision/audio encoders
/// would never be exercised.
#[derive(Subcommand, Clone)]
pub enum CalibrateModelType {
    /// Auto-detect model type (recommended)
    Auto {
        #[command(flatten)]
        model: CalibrateModelSourceOptions,

        #[command(flatten)]
        collection: CalibrateCollectionOptions,

        #[command(flatten)]
        device: CalibrateDeviceOptions,

        #[command(flatten)]
        output: CalibrateOutputOptions,
    },

    /// Text generation model with explicit architecture
    Text {
        #[command(flatten)]
        model: CalibrateModelSourceOptions,

        /// Model architecture (auto-detected if not specified)
        #[arg(short = 'a', long, value_parser = parse_arch)]
        arch: Option<NormalLoaderType>,

        #[command(flatten)]
        collection: CalibrateCollectionOptions,

        #[command(flatten)]
        device: CalibrateDeviceOptions,

        #[command(flatten)]
        output: CalibrateOutputOptions,
    },
}

/// Model source options for calibration.
#[derive(Args, Clone)]
pub struct CalibrateModelSourceOptions {
    /// Model ID to load (HuggingFace repo or local path)
    #[arg(short = 'm', long)]
    pub model_id: String,

    /// Path to local tokenizer.json file
    #[arg(short = 't', long)]
    pub tokenizer: Option<PathBuf>,

    /// Model data type
    #[arg(long, default_value = "auto", value_parser = parse_dtype)]
    pub dtype: ModelDType,
}

/// Off-diagonal `XᵀX` collection policy.
#[derive(Clone, Copy, Debug, ValueEnum, Default, PartialEq, Eq)]
pub enum GramModeArg {
    /// `diag(XᵀX)` only — the default. ~`in_features` f64 per layer.
    #[default]
    None,
    /// Block-diagonal `XᵀX` with `--gram-block` wide blocks. Storage grows
    /// linearly in the layer width, so this is the practical choice for real
    /// models.
    Block,
    /// Full `XᵀX` for layers no wider than `--gram-max-dim`. Quadratic in the
    /// layer width — only sane for small models or narrow layers.
    Full,
}

/// What the sweep accumulates.
#[derive(Args, Clone)]
pub struct CalibrateCollectionOptions {
    /// Calibration corpus (plain text). Defaults to the repo's
    /// `calibration_data/calibration_datav3_small.txt` when it is present in
    /// the working directory.
    #[arg(long)]
    pub corpus: Option<PathBuf>,

    /// Number of calibration samples, where one sample is a 1024-token chunk of
    /// the corpus. Defaults to the whole corpus (29 samples for the bundled
    /// small corpus, 62 for the full one).
    #[arg(long)]
    pub samples: Option<usize>,

    /// Off-diagonal `XᵀX` collection policy.
    #[arg(long, value_enum, default_value_t = GramModeArg::None)]
    pub gram: GramModeArg,

    /// Block width for `--gram block`.
    #[arg(long, default_value_t = 128, hide = true)]
    pub gram_block: usize,

    /// Maximum `in_features` for `--gram full`; wider layers fall back to
    /// diagonal-only (and are still emitted).
    #[arg(long, default_value_t = 4096, hide = true)]
    pub gram_max_dim: usize,

    /// Also collect per-expert statistics for MoE expert stacks. This
    /// multiplies the per-layer artifact size by the expert count, so it is off
    /// by default.
    #[arg(long)]
    pub per_expert: bool,

    /// Experts routed fewer than this many rows are flagged `insufficient` in
    /// the artifact so consumers can fall back to layer-global statistics.
    #[arg(long, default_value_t = 32, hide = true)]
    pub min_expert_tokens: u64,

    /// ISQ organization determining which layers are calibrated: `default` (all
    /// ISQ-eligible linears) or `moqe` (MoE experts only). Must match the
    /// organization the eventual quantization run uses, since the artifact is
    /// keyed by position in that layer list.
    #[arg(long)]
    pub isq_organization: Option<IsqOrganization>,
}

/// Device options for calibration.
#[derive(Args, Clone)]
pub struct CalibrateDeviceOptions {
    /// Force CPU-only execution
    #[arg(long)]
    pub cpu: bool,

    /// Device layer mapping (format: ORD:NUM;... e.g., "0:10;1:20")
    #[arg(short = 'n', long, value_delimiter = ';')]
    pub device_layers: Option<Vec<String>>,

    /// Topology YAML file for device mapping
    #[arg(long, hide = true)]
    pub topology: Option<PathBuf>,

    /// Custom HuggingFace cache directory
    #[arg(long, hide = true)]
    pub hf_cache: Option<PathBuf>,

    /// Max sequence length for automatic device mapping
    #[arg(long, default_value_t = AutoDeviceMapParams::DEFAULT_MAX_SEQ_LEN, hide = true)]
    pub max_seq_len: usize,

    /// Max batch size for automatic device mapping
    #[arg(long, default_value_t = AutoDeviceMapParams::DEFAULT_MAX_BATCH_SIZE, hide = true)]
    pub max_batch_size: usize,
}

/// Where the artifact goes.
#[derive(Args, Clone)]
pub struct CalibrateOutputOptions {
    /// Output path for the `.arccalib` artifact.
    #[arg(short = 'o', long = "out", required = true)]
    pub out: PathBuf,
}

/// Default options for `calibrate` when no model-type subcommand is given.
#[derive(Args, Clone)]
pub struct CalibrateDefaultOptions {
    /// HuggingFace model ID or local path to model directory
    #[arg(short = 'm', long)]
    pub model_id: Option<String>,

    /// Path to local tokenizer.json file
    #[arg(short = 't', long)]
    pub tokenizer: Option<PathBuf>,

    /// Model data type
    #[arg(long, default_value = "auto", value_parser = parse_dtype)]
    pub dtype: ModelDType,

    #[command(flatten)]
    pub collection: CalibrateCollectionOptions,

    #[command(flatten)]
    pub device: CalibrateDeviceOptions,

    /// Output path for the `.arccalib` artifact.
    #[arg(short = 'o', long = "out")]
    pub out: Option<PathBuf>,
}

impl CalibrateDefaultOptions {
    /// Convert into the `Auto` variant, erroring on missing required fields.
    pub fn into_calibrate_model_type(self) -> anyhow::Result<CalibrateModelType> {
        let model_id = self
            .model_id
            .ok_or_else(|| anyhow::anyhow!("--model-id (-m) is required"))?;
        let out = self
            .out
            .ok_or_else(|| anyhow::anyhow!("--out (-o) is required"))?;
        Ok(CalibrateModelType::Auto {
            model: CalibrateModelSourceOptions {
                model_id,
                tokenizer: self.tokenizer,
                dtype: self.dtype,
            },
            collection: self.collection,
            device: self.device,
            output: CalibrateOutputOptions { out },
        })
    }
}

/// Resolve the effective [`CalibrateModelType`], falling back to the default
/// options when no subcommand was given.
pub fn resolve_calibrate_model_type(
    model_type: Option<CalibrateModelType>,
    default_options: CalibrateDefaultOptions,
) -> anyhow::Result<CalibrateModelType> {
    match model_type {
        Some(mt) => Ok(mt),
        None => default_options.into_calibrate_model_type(),
    }
}

fn parse_arch(s: &str) -> Result<NormalLoaderType, String> {
    s.parse()
}

fn parse_dtype(s: &str) -> Result<ModelDType, String> {
    s.parse()
}
