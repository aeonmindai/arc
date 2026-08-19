//! Model-related argument structs

use clap::{Args, ValueEnum};
use mistralrs_core::{AutoDeviceMapParams, IsqOrganization, ModelDType, NormalLoaderType};
use serde::Deserialize;
use std::path::PathBuf;

/// Model source options
#[derive(Args, Clone, Deserialize)]
pub struct ModelSourceOptions {
    /// HuggingFace model ID or local path to model directory
    #[arg(short = 'm', long, help_heading = "Model")]
    pub model_id: String,

    /// Path to local tokenizer.json file
    ///
    /// Hidden: resolved from the model repo. Only needed when a local
    /// directory is missing its tokenizer.
    #[arg(short = 't', long, hide = true)]
    pub tokenizer: Option<PathBuf>,

    /// Model architecture (auto-detected if not specified)
    ///
    /// Hidden: detection from `config.json` is authoritative, and this is
    /// silently dropped on the `auto`, `vision`, `embedding`, `diffusion` and
    /// `speech` paths — so setting it mostly does nothing, and where it does
    /// work an incorrect value mis-loads the weights.
    #[arg(short = 'a', long, value_parser = parse_arch, hide = true)]
    pub arch: Option<NormalLoaderType>,

    /// Model data type
    ///
    /// Hidden: `auto` reads the dtype the checkpoint was trained in. Forcing
    /// f16 on a bf16 model degrades quality with no error.
    #[arg(long, default_value = "auto", value_parser = parse_dtype, hide = true)]
    #[serde(default)]
    pub dtype: ModelDType,
}

/// Format options for model loading
#[derive(Args, Clone, Default, Deserialize)]
pub struct FormatOptions {
    /// Model format: plain (safetensors), gguf, or ggml.
    /// Defaults to `plain`; `-f/--quantized-file` is ignored unless this is
    /// set to `gguf` or `ggml`.
    //
    // This previously claimed "Auto-detected if not specified". There is no
    // detection: every consumer does `format.unwrap_or(ModelFormat::Plain)`,
    // so a user passing only `-f model.gguf` silently loaded the plain path.
    //
    // Hidden: Arc's own path is safetensors + UQFF/ISQ. The GGUF/GGML loaders
    // are inherited from upstream and stay fully supported via `--help-all`.
    #[arg(long, value_enum, hide = true)]
    pub format: Option<ModelFormat>,

    /// Quantized model filename(s) for GGUF/GGML (semicolon-separated for multiple)
    ///
    /// Hidden: inert unless `--format gguf|ggml` is also passed.
    #[arg(short = 'f', long, hide = true)]
    pub quantized_file: Option<String>,

    /// Model ID for tokenizer when using quantized format
    #[arg(long, hide = true)]
    pub tok_model_id: Option<String>,

    /// GQA value for GGML models
    ///
    /// Hidden: GGML-only, and a wrong value produces garbage tokens rather
    /// than an error.
    #[arg(long, default_value_t = 1, hide = true)]
    #[serde(default = "default_gqa")]
    pub gqa: usize,
}

/// Model format type
#[derive(Clone, Copy, ValueEnum, Default, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ModelFormat {
    /// Plain model (safetensors)
    #[default]
    Plain,
    /// GGUF quantized model
    Gguf,
    /// GGML quantized model
    Ggml,
}

/// Adapter options (LoRA/X-LoRA)
#[derive(Args, Clone, Default, Deserialize)]
pub struct AdapterOptions {
    /// LoRA adapter model ID(s), semicolon-separated for multiple
    ///
    /// Hidden: fully supported, but a minority serving shape. See `--help-all`.
    #[arg(long, hide = true)]
    pub lora: Option<String>,

    /// X-LoRA adapter model ID
    #[arg(long, conflicts_with = "lora", hide = true)]
    pub xlora: Option<String>,

    /// X-LoRA ordering JSON file
    #[arg(long, requires = "xlora", hide = true)]
    pub xlora_order: Option<PathBuf>,

    /// Target non-granular index for X-LoRA.
    ///
    /// Hidden: an X-LoRA scaling-granularity cutoff that also silently forces
    /// `max_seqs = 1`. Developer/ablation knob, not a serving control.
    #[arg(long, requires = "xlora", hide = true)]
    pub tgt_non_granular_index: Option<usize>,
}

/// Quantization options
#[derive(Args, Clone, Default, Deserialize)]
pub struct QuantizationOptions {
    /// Quantize the weights on load (e.g. "4", "8", "q4k"). Omit to serve the
    /// checkpoint as-is, or use --from-uqff for a pre-baked Arc artifact.
    #[arg(long = "isq", help_heading = "Model")]
    pub in_situ_quant: Option<String>,

    /// Load a pre-baked Arc UQFF artifact. Shards are auto-discovered:
    /// naming the first (q4k-0.uqff) picks up q4k-1.uqff and so on.
    #[arg(long, help_heading = "Model")]
    pub from_uqff: Option<String>,

    /// ISQ organization strategy: default or moqe
    ///
    /// Hidden: `moqe` quantizes only expert layers. It must agree with how the
    /// artifact was baked, and a mismatch is not detected.
    #[arg(long, hide = true)]
    pub isq_organization: Option<IsqOrganization>,

    /// imatrix file for enhanced quantization
    #[arg(long, hide = true)]
    pub imatrix: Option<PathBuf>,

    /// Calibration file for imatrix generation
    #[arg(long, conflicts_with = "imatrix", hide = true)]
    pub calibration_file: Option<PathBuf>,
}

/// Device and compute options
#[derive(Args, Clone, Default, Deserialize)]
pub struct DeviceOptions {
    /// Force CPU-only execution
    #[arg(long, help_heading = "Hardware")]
    #[serde(default)]
    pub cpu: bool,

    /// Pin layers to specific GPUs (ORD:NUM;... e.g. "0:10;1:20").
    /// Omit to let Arc plan the device map.
    #[arg(short = 'n', long, value_delimiter = ';', help_heading = "Hardware")]
    pub device_layers: Option<Vec<String>>,

    /// Topology YAML file for device mapping
    ///
    /// Hidden: a per-layer device/dtype plan. Superseded by automatic mapping
    /// for every supported deployment, and silently ignored for diffusion and
    /// speech models.
    #[arg(long, hide = true)]
    pub topology: Option<PathBuf>,

    /// Custom HuggingFace cache directory
    ///
    /// Hidden: prefer the standard HF_HOME environment variable.
    #[arg(long, hide = true)]
    pub hf_cache: Option<PathBuf>,

    /// Max sequence length for automatic device mapping
    ///
    /// Hidden — and this is the single most misread flag in the CLI. It is a
    /// *device-map planning hint*, NOT the runtime context limit; raising it
    /// does not let conversations grow, it only changes how layers are split
    /// across GPUs. Use --pa-context-len to size the KV cache.
    #[arg(long, default_value_t = AutoDeviceMapParams::DEFAULT_MAX_SEQ_LEN, hide = true)]
    #[serde(default = "default_max_seq_len")]
    pub max_seq_len: usize,

    /// Max batch size for automatic device mapping
    ///
    /// Hidden: another planning hint, defaulting to 1 while --max-seqs
    /// defaults to 32. The two are unrelated and the mismatch invites
    /// "tuning" that does nothing.
    #[arg(long, default_value_t = AutoDeviceMapParams::DEFAULT_MAX_BATCH_SIZE, hide = true)]
    #[serde(default = "default_max_batch_size")]
    pub max_batch_size: usize,
}

/// Vision model specific options
#[derive(Args, Clone, Default, Deserialize)]
pub struct VisionOptions {
    /// Maximum edge length for image resizing (aspect ratio preserved)
    #[arg(long, hide = true)]
    pub max_edge: Option<u32>,

    /// Maximum number of images per request
    #[arg(long, hide = true)]
    pub max_num_images: Option<usize>,

    /// Maximum image dimension for device mapping
    #[arg(long, hide = true)]
    pub max_image_length: Option<usize>,
}

fn parse_arch(s: &str) -> Result<NormalLoaderType, String> {
    s.parse()
}

fn parse_dtype(s: &str) -> Result<ModelDType, String> {
    s.parse()
}

fn default_gqa() -> usize {
    1
}

fn default_max_seq_len() -> usize {
    AutoDeviceMapParams::DEFAULT_MAX_SEQ_LEN
}

fn default_max_batch_size() -> usize {
    AutoDeviceMapParams::DEFAULT_MAX_BATCH_SIZE
}
