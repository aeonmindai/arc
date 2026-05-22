use std::{
    collections::HashMap,
    fmt::{Debug, Display},
    str::FromStr,
    sync::Arc,
};

use crate::{attention::ATTENTION_CHUNK_SIZE, matformer::MatformerSliceConfig};

use crate::{
    amoe::AnyMoeBaseModelMixin,
    device_map::DeviceMapper,
    lora::{LoraConfig, Ordering},
    paged_attention::{AttentionImplementation, ModelConfigLike, ModelConfigMetadata},
    pipeline::{
        isq::IsqModelLoader,
        text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata},
        EitherCache, IsqModel,
    },
    utils::varbuilder_utils::DeviceForLoadTensor,
    xlora_models::NonGranularState,
};
use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use mistralrs_quant::log::once_log_info;

use indicatif::MultiProgress;
use mistralrs_quant::ShardedVarBuilder;
#[cfg(feature = "pyo3_macros")]
use pyo3::pyclass;

use regex::Regex;
use serde::Deserialize;

use crate::{
    models,
    xlora_models::{self, XLoraConfig},
};

use super::{AutoDeviceMapParams, DeviceMappedModelLoader};

pub trait NormalModel: IsqModel + AnyMoeBaseModelMixin {
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offsets: &[usize],
        context_lens: Vec<(usize, usize)>,
        position_ids: Vec<usize>,
        metadata: Option<(Vec<(Tensor, Tensor, Option<Tensor>, Option<Tensor>)>, &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> candle_core::Result<Tensor>;
    #[allow(clippy::too_many_arguments)]
    fn xlora_forward(
        &self,
        input_ids: &Tensor,
        input_ids_full: &Tensor,
        seqlen_offsets: &[usize],
        seqlen_offsets_full: &[usize],
        no_kv_cache: bool,
        non_granular_state: &Option<NonGranularState>,
        context_lens: Vec<(usize, usize)>,
        position_ids: Vec<usize>,
        flash_params: &FlashParams,
        flash_params_full: &FlashParams,
    ) -> candle_core::Result<Tensor>;
    fn is_xlora(&self) -> bool;
    fn device(&self) -> &Device;
    fn cache(&self) -> &EitherCache;
    fn cache_mut(&mut self) -> &mut EitherCache;
    fn max_seq_len(&self) -> usize;
    fn config(&self) -> &ModelConfigMetadata;

    /// Return the components needed to run MTP speculative drafting.
    ///
    /// Default: `None` — model does not have an MTP head. Models that do
    /// (currently only DeepSeekV4 with `mtp.layers.0.*` tensors) override
    /// this to expose their `embed_tokens`, `lm_head`, and `h_proj`/`e_proj`
    /// projections so a [`crate::pipeline::MtpSpeculativePipeline`] can be
    /// constructed.
    fn mtp_decode_kit(
        &self,
    ) -> Option<crate::pipeline::mtp_pipeline::MtpDecodeKit> {
        None
    }
}

/// Metadata for loading a model with ISQ or device mapping.
pub struct NormalLoadingMetadata {
    // Device mapping metadata which can be used to construct a concrete device mapper
    pub mapper: Box<dyn DeviceMapper + Send + Sync>,
    // Flag to check if loading in ISQ
    pub loading_isq: bool,
    // Device mapping target device (the one that is not the cpu)
    pub real_device: Device,
    // MultiProgress support for parallelized loading
    pub multi_progress: Arc<MultiProgress>,
    // Optional Matryoshka Transformer slicing configuration
    pub matformer_slicing_config: Option<MatformerSliceConfig>,
}

pub trait NormalModelLoader: IsqModelLoader + Send + Sync + DeviceMappedModelLoader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn NormalModel + Send + Sync>>;
    #[allow(clippy::too_many_arguments)]
    fn load_xlora(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        lora_config: &[((String, String), LoraConfig)],
        xlora_config: Option<XLoraConfig>,
        xlora_ordering: Ordering,
        normal_loading_metadata: NormalLoadingMetadata,
        preload_adapters: &Option<HashMap<String, (ShardedVarBuilder, LoraConfig)>>,
    ) -> Result<Box<dyn NormalModel + Send + Sync>>;
    fn is_gptx(&self, config: &str) -> Result<bool>;
    fn supports_paged_attention(&self, _config: &str) -> Result<bool> {
        Ok(true)
    }
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>>;
    fn get_device_for_tensor(
        &self,
        config: &str,
        _mapper: &dyn DeviceMapper,
        loading_isq: bool,
    ) -> Result<Arc<dyn Fn(String) -> DeviceForLoadTensor + Send + Sync + 'static>> {
        if loading_isq {
            Ok(Arc::new(|_| DeviceForLoadTensor::Base))
        } else {
            let re = Regex::new(r"\.layers\.(\d+)\.").unwrap();
            let num_layers = self.model_config(config)?.num_layers();
            let closure = move |name: String| {
                if let Some(captures) = re.captures(&name) {
                    captures
                        .get(1)
                        .and_then(|m| m.as_str().parse::<usize>().ok())
                        .map(|l| l.min(num_layers))
                        .map(DeviceForLoadTensor::Idx)
                        .unwrap_or(DeviceForLoadTensor::Base)
                } else {
                    DeviceForLoadTensor::Base
                }
            };

            Ok(Arc::new(closure))
        }
    }
}

#[cfg_attr(feature = "pyo3_macros", pyclass(eq, eq_int))]
#[derive(Clone, Debug, Deserialize, serde::Serialize, PartialEq)]
/// The architecture to load the normal model as.
pub enum NormalLoaderType {
    #[serde(rename = "mistral")]
    Mistral,
    #[serde(rename = "gemma")]
    Gemma,
    #[serde(rename = "mixtral")]
    Mixtral,
    #[serde(rename = "llama")]
    Llama,
    #[serde(rename = "phi2")]
    Phi2,
    #[serde(rename = "phi3")]
    Phi3,
    #[serde(rename = "qwen2")]
    Qwen2,
    #[serde(rename = "gemma2")]
    Gemma2,
    #[serde(rename = "starcoder2")]
    Starcoder2,
    #[serde(rename = "phi3.5moe")]
    Phi3_5MoE,
    #[serde(rename = "deepseekv2")]
    DeepSeekV2,
    #[serde(rename = "deepseekv3")]
    DeepSeekV3,
    /// DeepSeek V4 (Pro + Flash). Architecturally a superset of V3:
    /// inherits MLA + MoE, adds per-layer CSA/HCA hybrid attention.
    /// At Tier A, dispatched to the V3 loader (compresses to dense MLA);
    /// the V4-specific CSA/HCA attention path lives in `arc-engine::dsv4`
    /// and is wired in via `Sdpa::Dsv4` when compress_ratios is non-empty.
    #[serde(rename = "deepseekv4")]
    DeepSeekV4,
    /// Kimi K2.5 / K2.6. Text-side architecture is DeepSeek V3 (per Moonshot's
    /// implementation in SGLang: `kimi_k25.py` extends `DeepseekV3ForCausalLM`).
    #[serde(rename = "kimi_k2")]
    KimiK2,
    /// GLM-5.0 / 5.1 with DSA (V3-NSA-style) attention. SGLang serves this via
    /// `GlmMoeDsaForCausalLM` which extends `DeepseekV2ForCausalLM`.
    #[serde(rename = "glm5moedsa")]
    GLM5MoeDsa,
    #[serde(rename = "qwen3")]
    Qwen3,
    #[serde(rename = "glm4")]
    GLM4,
    #[serde(rename = "glm4moelite")]
    GLM4MoeLite,
    #[serde(rename = "glm4moe")]
    GLM4Moe,
    #[serde(rename = "qwen3moe")]
    Qwen3Moe,
    #[serde(rename = "smollm3")]
    SmolLm3,
    #[serde(rename = "granitemoehybrid")]
    GraniteMoeHybrid,
    #[serde(rename = "gpt_oss")]
    GptOss,
    #[serde(rename = "qwen3next")]
    Qwen3Next,
}

// https://github.com/huggingface/transformers/blob/cff06aac6fad28019930be03f5d467055bf62177/src/transformers/models/auto/modeling_auto.py#L448
impl NormalLoaderType {
    pub fn from_causal_lm_name(name: &str) -> Result<Self> {
        match name {
            "MistralForCausalLM" => Ok(Self::Mistral),
            "MixtralForCausalLM" => Ok(Self::Mixtral),
            "GemmaForCausalLM" => Ok(Self::Gemma),
            "Gemma2ForCausalLM" => Ok(Self::Gemma2),
            "PhiForCausalLM" => Ok(Self::Phi2),
            "Phi3ForCausalLM" => Ok(Self::Phi3),
            "LlamaForCausalLM" => Ok(Self::Llama),
            "Qwen2ForCausalLM" => Ok(Self::Qwen2),
            "Starcoder2ForCausalLM" => Ok(Self::Starcoder2),
            "PhiMoEForCausalLM" => Ok(Self::Phi3_5MoE),
            "DeepseekV2ForCausalLM" => Ok(Self::DeepSeekV2),
            "DeepseekV3ForCausalLM" => Ok(Self::DeepSeekV3),
            // V4 Pro and V4 Flash both use this architecture name
            "DeepseekV4ForCausalLM" => Ok(Self::DeepSeekV4),
            // Kimi K2.5 / K2.6 (text + multimodal both register the same generation class)
            "KimiK25ForConditionalGeneration" => Ok(Self::KimiK2),
            "KimiK2ForCausalLM" => Ok(Self::KimiK2),
            "KimiK26ForCausalLM" => Ok(Self::KimiK2),
            "KimiK26VLForConditionalGeneration" => Ok(Self::KimiK2),
            // GLM-5: SGLang's GlmMoeDsaForCausalLM
            "GlmMoeDsaForCausalLM" => Ok(Self::GLM5MoeDsa),
            "Qwen3ForCausalLM" => Ok(Self::Qwen3),
            "Glm4ForCausalLM" => Ok(Self::GLM4),
            "Glm4MoeLiteForCausalLM" => Ok(Self::GLM4MoeLite),
            "Glm4MoeForCausalLM" => Ok(Self::GLM4Moe),
            "Qwen3MoeForCausalLM" => Ok(Self::Qwen3Moe),
            "SmolLM3ForCausalLM" => Ok(Self::SmolLm3),
            "GraniteMoeHybridForCausalLM" => Ok(Self::GraniteMoeHybrid),
            "GptOssForCausalLM" => Ok(Self::GptOss),
            "Qwen3NextForCausalLM" => Ok(Self::Qwen3Next),
            other => anyhow::bail!(
                "Unsupported Hugging Face Transformers -CausalLM model class `{other}`. Please raise an issue."
            ),
        }
    }
}

impl FromStr for NormalLoaderType {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "mistral" => Ok(Self::Mistral),
            "gemma" => Ok(Self::Gemma),
            "mixtral" => Ok(Self::Mixtral),
            "llama" => Ok(Self::Llama),
            "phi2" => Ok(Self::Phi2),
            "phi3" => Ok(Self::Phi3),
            "qwen2" => Ok(Self::Qwen2),
            "gemma2" => Ok(Self::Gemma2),
            "starcoder2" => Ok(Self::Starcoder2),
            "phi3.5moe" => Ok(Self::Phi3_5MoE),
            "deepseekv2" => Ok(Self::DeepSeekV2),
            "deepseekv3" => Ok(Self::DeepSeekV3),
            "deepseekv4" => Ok(Self::DeepSeekV4),
            "kimi_k2" | "kimi_k25" | "kimi_k26" => Ok(Self::KimiK2),
            "glm5moedsa" | "glm5" => Ok(Self::GLM5MoeDsa),
            "qwen3" => Ok(Self::Qwen3),
            "glm4" => Ok(Self::GLM4),
            "glm4moelite" => Ok(Self::GLM4MoeLite),
            "glm4moe" => Ok(Self::GLM4Moe),
            "qwen3moe" => Ok(Self::Qwen3Moe),
            "smollm3" => Ok(Self::SmolLm3),
            "granitemoehybrid" => Ok(Self::GraniteMoeHybrid),
            "gpt_oss" => Ok(Self::GptOss),
            "qwen3next" => Ok(Self::Qwen3Next),
            a => Err(format!("Unknown architecture `{a}`. Possible architectures: `mistral`, `gemma`, `mixtral`, `llama`, `phi2`, `phi3`, `qwen2`, `gemma2`, `starcoder2`, `phi3.5moe`, `deepseekv2`, `deepseekv3`, `deepseekv4`, `kimi_k2`, `glm5moedsa`, `qwen3`, `glm4`, `glm4moelite`, `glm4moe`, `qwen3moe`, `smollm3`, `granitemoehybrid`, `gpt_oss`, `qwen3next`.")),
        }
    }
}

impl Display for NormalLoaderType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Gemma => write!(f, "gemma"),
            Self::Gemma2 => write!(f, "gemma2"),
            Self::Llama => write!(f, "llama"),
            Self::Mistral => write!(f, "mistral"),
            Self::Mixtral => write!(f, "mixtral"),
            Self::Phi2 => write!(f, "phi2"),
            Self::Phi3 => write!(f, "phi3"),
            Self::Phi3_5MoE => write!(f, "phi3.5moe"),
            Self::Qwen2 => write!(f, "qwen2"),
            Self::Starcoder2 => write!(f, "starcoder2"),
            Self::DeepSeekV2 => write!(f, "deepseekv2"),
            Self::DeepSeekV3 => write!(f, "deepseekv3"),
            Self::DeepSeekV4 => write!(f, "deepseekv4"),
            Self::KimiK2 => write!(f, "kimi_k2"),
            Self::GLM5MoeDsa => write!(f, "glm5moedsa"),
            Self::Qwen3 => write!(f, "qwen3"),
            Self::GLM4 => write!(f, "glm4"),
            Self::GLM4MoeLite => write!(f, "glm4moelite"),
            Self::GLM4Moe => write!(f, "glm4moe"),
            Self::Qwen3Moe => write!(f, "qwen3moe"),
            Self::SmolLm3 => write!(f, "smollm3"),
            Self::GraniteMoeHybrid => write!(f, "granitemoehybrid"),
            Self::GptOss => write!(f, "gpt_oss"),
            Self::Qwen3Next => write!(f, "qwen3next"),
        }
    }
}

macro_rules! bias_if {
    ($cond:expr, $size:expr) => {
        if $cond {
            $size
        } else {
            0
        }
    };
}

/// Load a model based on the Hugging Face Transformers -CausalLM model class
pub struct AutoNormalLoader;

#[derive(Deserialize)]
struct AutoNormalLoaderConfig {
    architectures: Vec<String>,
}

impl AutoNormalLoader {
    fn get_loader(config: &str) -> Result<Box<dyn NormalModelLoader>> {
        let auto_cfg: AutoNormalLoaderConfig = serde_json::from_str(config)?;
        if auto_cfg.architectures.len() != 1 {
            anyhow::bail!("Expected to have one name for `architectures` config field.")
        }

        let name = &auto_cfg.architectures[0];

        let tp = NormalLoaderType::from_causal_lm_name(name)?;

        once_log_info(format!("Automatic loader type determined to be `{tp}`"));

        match tp {
            NormalLoaderType::Mistral => Ok(Box::new(MistralLoader)),
            NormalLoaderType::Gemma => Ok(Box::new(GemmaLoader)),
            NormalLoaderType::Llama => Ok(Box::new(LlamaLoader)),
            NormalLoaderType::Mixtral => Ok(Box::new(MixtralLoader)),
            NormalLoaderType::Phi2 => Ok(Box::new(Phi2Loader)),
            NormalLoaderType::Phi3 => Ok(Box::new(Phi3Loader)),
            NormalLoaderType::Qwen2 => Ok(Box::new(Qwen2Loader)),
            NormalLoaderType::Gemma2 => Ok(Box::new(Gemma2Loader)),
            NormalLoaderType::Starcoder2 => Ok(Box::new(Starcoder2Loader)),
            NormalLoaderType::Phi3_5MoE => Ok(Box::new(Phi3_5MoELoader)),
            NormalLoaderType::DeepSeekV2 => Ok(Box::new(DeepSeekV2Loader)),
            NormalLoaderType::DeepSeekV3 => Ok(Box::new(DeepSeekV3Loader)),
            // V4 routes to the dedicated DeepSeekV4Loader (mistralrs-core/src/models/deepseek4.rs).
            // At Tier A, the V4 loader's forward path delegates to V3-class dense MLA
            // (still loads V4-shaped weights but doesn't dispatch CSA/HCA/MTP).
            // CSA/HCA + MTP wiring is RUN-155/156 (rental cycle).
            NormalLoaderType::DeepSeekV4 => Ok(Box::new(DeepSeekV4Loader)),
            // Kimi K2.5/K2.6 text architecture IS DeepSeek V3 (per SGLang's
            // kimi_k25.py line 39: `from sglang.srt.models.deepseek_v2 import
            // DeepseekV3ForCausalLM`). Direct alias.
            NormalLoaderType::KimiK2 => Ok(Box::new(DeepSeekV3Loader)),
            // GLM-5 (DSA variant) extends DeepSeek V2/V3-style MLA + DSA attention.
            // SGLang: `class GlmMoeDsaForCausalLM(DeepseekV2ForCausalLM)`.
            // Tier A: dispatch to GLM4Moe loader (uses standard attention).
            // Tier B: V3 path with DSA awareness.
            NormalLoaderType::GLM5MoeDsa => Ok(Box::new(GLM4MoeLoader)),
            NormalLoaderType::Qwen3 => Ok(Box::new(Qwen3Loader)),
            NormalLoaderType::GLM4 => Ok(Box::new(GLM4Loader)),
            NormalLoaderType::GLM4MoeLite => Ok(Box::new(GLM4MoeLiteLoader)),
            NormalLoaderType::GLM4Moe => Ok(Box::new(GLM4MoeLoader)),
            NormalLoaderType::Qwen3Moe => Ok(Box::new(Qwen3MoELoader)),
            NormalLoaderType::SmolLm3 => Ok(Box::new(SmolLm3Loader)),
            NormalLoaderType::GraniteMoeHybrid => Ok(Box::new(GraniteMoeHybridLoader)),
            NormalLoaderType::GptOss => Ok(Box::new(GptOssLoader)),
            NormalLoaderType::Qwen3Next => Ok(Box::new(Qwen3NextLoader)),
        }
    }
}

impl NormalModelLoader for AutoNormalLoader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        Self::get_loader(config)?.load(config, vb, normal_loading_metadata, attention_mechanism)
    }
    fn load_xlora(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        lora_config: &[((String, String), LoraConfig)],
        xlora_config: Option<XLoraConfig>,
        xlora_ordering: Ordering,
        normal_loading_metadata: NormalLoadingMetadata,
        preload_adapters: &Option<HashMap<String, (ShardedVarBuilder, LoraConfig)>>,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        Self::get_loader(config)?.load_xlora(
            config,
            vb,
            lora_config,
            xlora_config,
            xlora_ordering,
            normal_loading_metadata,
            preload_adapters,
        )
    }
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        Self::get_loader(config)?.get_config_repr(config)
    }
    fn supports_paged_attention(&self, config: &str) -> Result<bool> {
        Self::get_loader(config)?.supports_paged_attention(config)
    }
    fn is_gptx(&self, config: &str) -> Result<bool> {
        Self::get_loader(config)?.is_gptx(config)
    }
}

impl IsqModelLoader for AutoNormalLoader {
    fn immediate_isq_predicates(&self, config: &str) -> Result<Vec<Regex>> {
        Self::get_loader(config)?.immediate_isq_predicates(config)
    }
    fn immediate_isq_predicates_moqe(&self, config: &str) -> Result<Vec<Regex>> {
        Self::get_loader(config)?.immediate_isq_predicates_moqe(config)
    }
    fn isq_layer_regexes(&self, config: &str) -> Result<Vec<Regex>> {
        Self::get_loader(config)?.isq_layer_regexes(config)
    }
    fn isq_layer_regexes_moqe(&self, config: &str) -> Result<Vec<Regex>> {
        Self::get_loader(config)?.isq_layer_regexes_moqe(config)
    }
}

impl DeviceMappedModelLoader for AutoNormalLoader {
    fn non_mapped_size_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        Self::get_loader(config)?.non_mapped_size_in_bytes(
            config,
            dtype,
            weight_pack_factor,
            _matformer_config,
        )
    }
    fn num_layers(&self, config: &str) -> Result<usize> {
        Self::get_loader(config)?.num_layers(config)
    }
    fn layer_sizes_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<Vec<usize>> {
        Self::get_loader(config)?.layer_sizes_in_bytes(
            config,
            dtype,
            weight_pack_factor,
            _matformer_config,
        )
    }
    fn mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &super::AutoDeviceMapParams,
    ) -> Result<usize> {
        Self::get_loader(config)?.mapped_max_act_size_elems(config, params)
    }
    fn non_mapped_max_act_size_elems(
        &self,
        _config: &str,
        _params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        Ok(0)
    }
    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        Self::get_loader(config)?.model_config(config)
    }
}

// ======================== Mistral loader

pub struct MistralLoader;

impl NormalModelLoader for MistralLoader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        let cfg: crate::models::mistral::Config = serde_json::from_str(config)?;
        Ok(Box::new(models::mistral::Model::new(
            &cfg,
            vb,
            self.is_gptx(config)?,
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }
    fn load_xlora(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        lora_config: &[((String, String), LoraConfig)],
        xlora_config: Option<XLoraConfig>,
        xlora_ordering: Ordering,
        normal_loading_metadata: NormalLoadingMetadata,
        preload_adapters: &Option<HashMap<String, (ShardedVarBuilder, LoraConfig)>>,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        let cfg: crate::models::mistral::Config = serde_json::from_str(config)?;
        Ok(Box::new(xlora_models::XLoraMistral::new(
            &cfg,
            vb,
            lora_config,
            xlora_config,
            xlora_ordering,
            self.is_gptx(config)?,
            normal_loading_metadata,
            preload_adapters,
        )?))
    }
    fn is_gptx(&self, _: &str) -> Result<bool> {
        Ok(true)
    }
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        let cfg: crate::models::mistral::Config = serde_json::from_str(config)?;
        Ok(Box::new(cfg))
    }
}

impl IsqModelLoader for MistralLoader {
    fn isq_layer_regexes(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(r"lm_head\.(weight|bias)$")?,
            // Attention
            Regex::new(r"layers\.(\d+)\.self_attn\.q_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.k_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.v_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.o_proj\.(weight|bias)$")?,
            // MLP
            Regex::new(r"layers\.(\d+)\.mlp\.gate_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.mlp\.up_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.mlp\.down_proj\.(weight|bias)$")?,
        ])
    }
    fn immediate_isq_predicates(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes(config)
    }
}

impl DeviceMappedModelLoader for MistralLoader {
    fn mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        let AutoDeviceMapParams::Text {
            max_seq_len,
            max_batch_size,
        } = params
        else {
            anyhow::bail!("Expected text AutoDeviceMapParams for this model!")
        };

        let cfg: crate::models::mistral::Config = serde_json::from_str(config)?;

        Ok(
            max_batch_size
                * cfg.num_attention_heads
                * max_seq_len.min(&ATTENTION_CHUNK_SIZE).pow(2),
        )
    }
    fn non_mapped_max_act_size_elems(
        &self,
        _config: &str,
        _params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        Ok(0)
    }

    fn non_mapped_size_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        let cfg: crate::models::mistral::Config = serde_json::from_str(config)?;

        let elems = {
            let embed_tokens = cfg.hidden_size * cfg.vocab_size / weight_pack_factor;
            // If embeddings are tied and no packing, reuse weights -> no separate lm_head needed
            let lm_head = if !cfg.tie_word_embeddings || weight_pack_factor != 1 {
                cfg.hidden_size * cfg.vocab_size / weight_pack_factor
            } else {
                0
            };
            let norm = cfg.hidden_size;
            embed_tokens + lm_head + norm
        };
        Ok(elems * dtype.size_in_bytes())
    }

    fn layer_sizes_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<Vec<usize>> {
        let cfg: crate::models::mistral::Config = serde_json::from_str(config)?;

        let per_layer_elems = {
            let input_layernorm = cfg.hidden_size;
            let post_attention_layernorm = cfg.hidden_size;

            let size_in = cfg.hidden_size;
            let size_q = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_attention_heads;
            let size_kv = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_key_value_heads;
            let q_proj = size_in * size_q / weight_pack_factor;
            let k_proj = size_in * size_kv / weight_pack_factor;
            let v_proj = size_in * size_kv / weight_pack_factor;
            let o_proj = size_q * size_in / weight_pack_factor;

            let h_size = cfg.hidden_size;
            let i_size = cfg.intermediate_size;
            let gate_proj = h_size * i_size / weight_pack_factor;
            let up_proj = h_size * i_size / weight_pack_factor;
            let down_proj = i_size * h_size / weight_pack_factor;

            input_layernorm
                + post_attention_layernorm
                + q_proj
                + k_proj
                + v_proj
                + o_proj
                + gate_proj
                + up_proj
                + down_proj
        };
        Ok(vec![
            per_layer_elems * dtype.size_in_bytes();
            cfg.num_hidden_layers
        ])
    }

    fn num_layers(&self, config: &str) -> Result<usize> {
        let cfg: crate::models::mistral::Config = serde_json::from_str(config)?;
        Ok(cfg.num_hidden_layers)
    }

    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg: crate::models::mistral::Config = serde_json::from_str(config)?;

        let cfg = ModelConfigMetadata {
            max_seq_len: cfg.max_position_embeddings,
            num_layers: cfg.num_hidden_layers,
            hidden_size: cfg.hidden_size,
            num_kv_heads: cfg.num_key_value_heads,
            num_attn_heads: cfg.num_attention_heads,
            sliding_window: cfg.sliding_window,
            k_head_dim: cfg.head_dim(),
            v_head_dim: cfg.head_dim(),
            kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
        };

        Ok(Box::new(cfg))
    }
}

// ======================== Gemma loader

/// [`NormalLoader`] for a Gemma model.
///
/// [`NormalLoader`]: https://docs.rs/mistralrs/latest/mistralrs/struct.NormalLoader.html
pub struct GemmaLoader;

impl NormalModelLoader for GemmaLoader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        let cfg: crate::models::gemma::Config = serde_json::from_str(config)?;

        Ok(Box::new(models::gemma::Model::new(
            &cfg,
            vb,
            self.is_gptx(config)?,
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }
    fn load_xlora(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        lora_config: &[((String, String), LoraConfig)],
        xlora_config: Option<XLoraConfig>,
        xlora_ordering: Ordering,
        normal_loading_metadata: NormalLoadingMetadata,
        preload_adapters: &Option<HashMap<String, (ShardedVarBuilder, LoraConfig)>>,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        let cfg: crate::models::gemma::Config = serde_json::from_str(config)?;

        Ok(Box::new(xlora_models::XLoraGemma::new(
            &cfg,
            vb,
            lora_config,
            xlora_config,
            xlora_ordering,
            self.is_gptx(config)?,
            normal_loading_metadata,
            preload_adapters,
        )?))
    }
    fn is_gptx(&self, _: &str) -> Result<bool> {
        Ok(true)
    }
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        let cfg: crate::models::gemma::Config = serde_json::from_str(config)?;
        Ok(Box::new(cfg))
    }
}

impl IsqModelLoader for GemmaLoader {
    fn isq_layer_regexes(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(r"lm_head\.(weight|bias)$")?,
            // Attention
            Regex::new(r"layers\.(\d+)\.self_attn\.q_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.k_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.v_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.o_proj\.(weight|bias)$")?,
            // MLP
            Regex::new(r"layers\.(\d+)\.mlp\.gate_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.mlp\.up_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.mlp\.down_proj\.(weight|bias)$")?,
        ])
    }
    fn immediate_isq_predicates(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes(config)
    }
}

impl DeviceMappedModelLoader for GemmaLoader {
    fn mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        let AutoDeviceMapParams::Text {
            max_seq_len,
            max_batch_size,
        } = params
        else {
            anyhow::bail!("Expected text AutoDeviceMapParams for this model!")
        };

        let cfg: crate::models::gemma::Config = serde_json::from_str(config)?;

        Ok(
            max_batch_size
                * cfg.num_attention_heads
                * max_seq_len.min(&ATTENTION_CHUNK_SIZE).pow(2),
        )
    }
    fn non_mapped_max_act_size_elems(
        &self,
        _config: &str,
        _params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        Ok(0)
    }

    fn non_mapped_size_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        let cfg: crate::models::gemma::Config = serde_json::from_str(config)?;

        let elems = {
            let embed_tokens = cfg.hidden_size * cfg.vocab_size / weight_pack_factor;
            // If embeddings are tied and no packing, reuse weights -> no separate lm_head needed
            let lm_head = if !cfg.tie_word_embeddings || weight_pack_factor != 1 {
                cfg.hidden_size * cfg.vocab_size / weight_pack_factor
            } else {
                0
            };
            let norm = cfg.hidden_size;
            embed_tokens + lm_head + norm
        };
        Ok(elems * dtype.size_in_bytes())
    }

    fn layer_sizes_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<Vec<usize>> {
        let cfg: crate::models::gemma::Config = serde_json::from_str(config)?;

        let per_layer_elems = {
            let input_layernorm = cfg.hidden_size;
            let post_attention_layernorm = cfg.hidden_size;

            let size_in = cfg.hidden_size;
            let size_q = cfg.head_dim * cfg.num_attention_heads;
            let size_kv = cfg.head_dim * cfg.num_key_value_heads;
            let q_proj =
                size_in * size_q / weight_pack_factor + bias_if!(cfg.attention_bias, size_q);
            let k_proj =
                size_in * size_kv / weight_pack_factor + bias_if!(cfg.attention_bias, size_kv);
            let v_proj =
                size_in * size_kv / weight_pack_factor + bias_if!(cfg.attention_bias, size_kv);
            let o_proj =
                size_q * size_in / weight_pack_factor + bias_if!(cfg.attention_bias, size_in);

            let h_size = cfg.hidden_size;
            let i_size = cfg.intermediate_size;
            let gate_proj = h_size * i_size / weight_pack_factor;
            let up_proj = h_size * i_size / weight_pack_factor;
            let down_proj = i_size * h_size / weight_pack_factor;

            input_layernorm
                + post_attention_layernorm
                + q_proj
                + k_proj
                + v_proj
                + o_proj
                + gate_proj
                + up_proj
                + down_proj
        };
        Ok(vec![
            per_layer_elems * dtype.size_in_bytes();
            cfg.num_hidden_layers
        ])
    }

    fn num_layers(&self, config: &str) -> Result<usize> {
        let cfg: crate::models::gemma::Config = serde_json::from_str(config)?;
        Ok(cfg.num_hidden_layers)
    }

    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg: crate::models::gemma::Config = serde_json::from_str(config)?;

        let cfg = ModelConfigMetadata {
            max_seq_len: cfg.max_position_embeddings,
            num_layers: cfg.num_hidden_layers,
            hidden_size: cfg.hidden_size,
            num_kv_heads: cfg.num_key_value_heads,
            num_attn_heads: cfg.num_attention_heads,
            sliding_window: None,
            k_head_dim: cfg.head_dim,
            v_head_dim: cfg.head_dim,
            kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
        };

        Ok(Box::new(cfg))
    }
}

// ======================== Llama loader

/// [`NormalLoader`] for a Llama model.
///
/// [`NormalLoader`]: https://docs.rs/mistralrs/latest/mistralrs/struct.NormalLoader.html
pub struct LlamaLoader;

impl NormalModelLoader for LlamaLoader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        let cfg: crate::models::llama::Config = serde_json::from_str(config)?;

        Ok(Box::new(models::llama::Llama::new(
            &cfg,
            vb,
            self.is_gptx(config)?,
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }
    fn load_xlora(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        lora_config: &[((String, String), LoraConfig)],
        xlora_config: Option<XLoraConfig>,
        xlora_ordering: Ordering,
        normal_loading_metadata: NormalLoadingMetadata,
        preload_adapters: &Option<HashMap<String, (ShardedVarBuilder, LoraConfig)>>,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        let cfg: crate::models::llama::Config = serde_json::from_str(config)?;

        Ok(Box::new(xlora_models::XLoraLlama::new(
            &cfg,
            vb,
            lora_config,
            xlora_config,
            xlora_ordering,
            self.is_gptx(config)?,
            normal_loading_metadata,
            preload_adapters,
        )?))
    }
    fn is_gptx(&self, _: &str) -> Result<bool> {
        Ok(true)
    }
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        let cfg: crate::models::llama::Config = serde_json::from_str(config)?;
        Ok(Box::new(cfg))
    }
}

impl IsqModelLoader for LlamaLoader {
    fn isq_layer_regexes(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(r"lm_head\.(weight|bias)$")?,
            // Attention
            Regex::new(r"layers\.(\d+)\.self_attn\.q_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.k_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.v_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.o_proj\.(weight|bias)$")?,
            // MLP
            Regex::new(r"layers\.(\d+)\.mlp\.gate_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.mlp\.up_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.mlp\.down_proj\.(weight|bias)$")?,
        ])
    }
    fn immediate_isq_predicates(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes(config)
    }
}

impl DeviceMappedModelLoader for LlamaLoader {
    fn mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        let AutoDeviceMapParams::Text {
            max_seq_len,
            max_batch_size,
        } = params
        else {
            anyhow::bail!("Expected text AutoDeviceMapParams for this model!")
        };

        let cfg: crate::models::llama::Config = serde_json::from_str(config)?;

        Ok(
            max_batch_size
                * cfg.num_attention_heads
                * max_seq_len.min(&ATTENTION_CHUNK_SIZE).pow(2),
        )
    }
    fn non_mapped_max_act_size_elems(
        &self,
        _config: &str,
        _params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        Ok(0)
    }

    fn non_mapped_size_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        let cfg: crate::models::llama::Config = serde_json::from_str(config)?;

        let elems = {
            let embed_tokens = cfg.hidden_size * cfg.vocab_size / weight_pack_factor;
            // If embeddings are tied and no packing, reuse weights -> no separate lm_head needed
            let lm_head = if !cfg.tie_word_embeddings || weight_pack_factor != 1 {
                cfg.hidden_size * cfg.vocab_size / weight_pack_factor
            } else {
                0
            };
            let norm = cfg.hidden_size;
            embed_tokens + lm_head + norm
        };
        Ok(elems * dtype.size_in_bytes())
    }

    fn layer_sizes_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<Vec<usize>> {
        let cfg: crate::models::llama::Config = serde_json::from_str(config)?;

        let per_layer_elems = {
            let input_layernorm = cfg.hidden_size;
            let post_attention_layernorm = cfg.hidden_size;

            let size_in = cfg.hidden_size;
            let size_q = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_attention_heads;
            let size_kv = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_key_value_heads;
            let q_proj = size_in * size_q / weight_pack_factor;
            let k_proj = size_in * size_kv / weight_pack_factor;
            let v_proj = size_in * size_kv / weight_pack_factor;
            let o_proj = size_q * size_in / weight_pack_factor;

            let h_size = cfg.hidden_size;
            let i_size = cfg.intermediate_size;
            let gate_proj = h_size * i_size / weight_pack_factor;
            let up_proj = h_size * i_size / weight_pack_factor;
            let down_proj = i_size * h_size / weight_pack_factor;

            input_layernorm
                + post_attention_layernorm
                + q_proj
                + k_proj
                + v_proj
                + o_proj
                + gate_proj
                + up_proj
                + down_proj
        };
        Ok(vec![
            per_layer_elems * dtype.size_in_bytes();
            cfg.num_hidden_layers
        ])
    }

    fn num_layers(&self, config: &str) -> Result<usize> {
        let cfg: crate::models::llama::Config = serde_json::from_str(config)?;

        Ok(cfg.num_hidden_layers)
    }
    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg: crate::models::llama::Config = serde_json::from_str(config)?;

        let cfg = ModelConfigMetadata {
            max_seq_len: cfg.max_position_embeddings,
            num_layers: cfg.num_hidden_layers,
            hidden_size: cfg.hidden_size,
            num_kv_heads: cfg.num_key_value_heads,
            num_attn_heads: cfg.num_attention_heads,
            sliding_window: None,
            k_head_dim: cfg.hidden_size / cfg.num_attention_heads,
            v_head_dim: cfg.hidden_size / cfg.num_attention_heads,
            kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
        };

        Ok(Box::new(cfg))
    }
}

// ======================== Mixtral loader

pub struct MixtralLoader;

impl NormalModelLoader for MixtralLoader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        let cfg: crate::models::mixtral::Config = serde_json::from_str(config)?;

        Ok(Box::new(models::mixtral::Model::new(
            &cfg,
            vb,
            self.is_gptx(config)?,
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }
    fn load_xlora(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        lora_config: &[((String, String), LoraConfig)],
        xlora_config: Option<XLoraConfig>,
        xlora_ordering: Ordering,
        normal_loading_metadata: NormalLoadingMetadata,
        preload_adapters: &Option<HashMap<String, (ShardedVarBuilder, LoraConfig)>>,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        let cfg: crate::models::mixtral::Config = serde_json::from_str(config)?;

        Ok(Box::new(xlora_models::XLoraMixtral::new(
            &cfg,
            vb,
            lora_config,
            xlora_config,
            xlora_ordering,
            self.is_gptx(config)?,
            normal_loading_metadata,
            preload_adapters,
        )?))
    }
    fn is_gptx(&self, _: &str) -> Result<bool> {
        Ok(true)
    }
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        let cfg: crate::models::mixtral::Config = serde_json::from_str(config)?;

        Ok(Box::new(cfg))
    }
}

impl IsqModelLoader for MixtralLoader {
    fn isq_layer_regexes(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(r"lm_head\.(weight|bias)$")?,
            // Attention
            Regex::new(r"layers\.(\d+)\.self_attn\.q_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.k_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.v_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.o_proj\.(weight|bias)$")?,
            // Experts
            Regex::new(r"layers\.(\d+)\.block_sparse_moe\.gate\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.w1\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.w2\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.w3\.(weight|bias)$")?,
        ])
    }
    fn immediate_isq_predicates(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes(config)
    }
}

impl DeviceMappedModelLoader for MixtralLoader {
    fn mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        let AutoDeviceMapParams::Text {
            max_seq_len,
            max_batch_size,
        } = params
        else {
            anyhow::bail!("Expected text AutoDeviceMapParams for this model!")
        };

        let cfg: crate::models::mixtral::Config = serde_json::from_str(config)?;

        Ok(
            max_batch_size
                * cfg.num_attention_heads
                * max_seq_len.min(&ATTENTION_CHUNK_SIZE).pow(2),
        )
    }
    fn non_mapped_max_act_size_elems(
        &self,
        _config: &str,
        _params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        Ok(0)
    }

    fn non_mapped_size_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        let cfg: crate::models::mixtral::Config = serde_json::from_str(config)?;

        let elems = {
            let embed_tokens = cfg.hidden_size * cfg.vocab_size / weight_pack_factor;
            // If embeddings are tied and no packing, reuse weights -> no separate lm_head needed
            let lm_head = if !cfg.tie_word_embeddings || weight_pack_factor != 1 {
                cfg.hidden_size * cfg.vocab_size / weight_pack_factor
            } else {
                0
            };
            let norm = cfg.hidden_size;
            embed_tokens + lm_head + norm
        };
        Ok(elems * dtype.size_in_bytes())
    }

    fn layer_sizes_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<Vec<usize>> {
        let cfg: crate::models::mixtral::Config = serde_json::from_str(config)?;

        let per_layer_elems = {
            let input_layernorm = cfg.hidden_size;
            let post_attention_layernorm = cfg.hidden_size;

            let size_in = cfg.hidden_size;
            let size_q = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_attention_heads;
            let size_kv = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_key_value_heads;
            let q_proj = size_in * size_q / weight_pack_factor;
            let k_proj = size_in * size_kv / weight_pack_factor;
            let v_proj = size_in * size_kv / weight_pack_factor;
            let o_proj = size_q * size_in / weight_pack_factor;

            let moe_block = {
                let gate = cfg.hidden_size * cfg.num_local_experts;
                // Assume quantizing weight pack factor
                let w1 = cfg.hidden_size * cfg.intermediate_size / weight_pack_factor;
                let w2 = cfg.hidden_size * cfg.intermediate_size / weight_pack_factor;
                let w3 = cfg.hidden_size * cfg.intermediate_size / weight_pack_factor;
                gate + cfg.num_local_experts * w1
                    + cfg.num_local_experts * w2
                    + cfg.num_local_experts * w3
            };

            input_layernorm
                + post_attention_layernorm
                + q_proj
                + k_proj
                + v_proj
                + o_proj
                + moe_block
        };
        Ok(vec![
            per_layer_elems * dtype.size_in_bytes();
            cfg.num_hidden_layers
        ])
    }

    fn num_layers(&self, config: &str) -> Result<usize> {
        let cfg: crate::models::mixtral::Config = serde_json::from_str(config)?;

        Ok(cfg.num_hidden_layers)
    }

    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg: crate::models::mixtral::Config = serde_json::from_str(config)?;

        let cfg = ModelConfigMetadata {
            max_seq_len: cfg.max_position_embeddings,
            num_layers: cfg.num_hidden_layers,
            hidden_size: cfg.hidden_size,
            num_kv_heads: cfg.num_key_value_heads,
            num_attn_heads: cfg.num_attention_heads,
            sliding_window: cfg.sliding_window,
            k_head_dim: cfg.hidden_size / cfg.num_attention_heads,
            v_head_dim: cfg.hidden_size / cfg.num_attention_heads,
            kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
        };

        Ok(Box::new(cfg))
    }
}

// ======================== Phi2 loader

/// [`NormalLoader`] for a Phi 2 model.
///
/// [`NormalLoader`]: https://docs.rs/mistralrs/latest/mistralrs/struct.NormalLoader.html
pub struct Phi2Loader;

impl NormalModelLoader for Phi2Loader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        let cfg: crate::models::phi2::Config = serde_json::from_str(config)?;

        Ok(Box::new(models::phi2::Model::new(
            &cfg,
            vb,
            self.is_gptx(config)?,
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }
    fn load_xlora(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        lora_config: &[((String, String), LoraConfig)],
        xlora_config: Option<XLoraConfig>,
        xlora_ordering: Ordering,
        normal_loading_metadata: NormalLoadingMetadata,
        preload_adapters: &Option<HashMap<String, (ShardedVarBuilder, LoraConfig)>>,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        let cfg: crate::models::phi2::Config = serde_json::from_str(config)?;

        Ok(Box::new(xlora_models::XLoraPhi2::new(
            &cfg,
            vb,
            lora_config,
            xlora_config,
            xlora_ordering,
            self.is_gptx(config)?,
            normal_loading_metadata,
            preload_adapters,
        )?))
    }
    fn is_gptx(&self, _: &str) -> Result<bool> {
        Ok(true)
    }
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        let cfg: crate::models::phi2::Config = serde_json::from_str(config)?;

        Ok(Box::new(cfg))
    }
}

impl IsqModelLoader for Phi2Loader {
    fn isq_layer_regexes(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(r"lm_head\.(weight|bias)$")?,
            // Attention
            Regex::new(r"layers\.(\d+)\.self_attn\.q_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.k_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.v_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.dense\.(weight|bias)$")?,
            // MLP
            Regex::new(r"layers\.(\d+)\.mlp\.fc1\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.mlp\.fc2\.(weight|bias)$")?,
        ])
    }
    fn immediate_isq_predicates(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes(config)
    }
}

impl DeviceMappedModelLoader for Phi2Loader {
    fn mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        let AutoDeviceMapParams::Text {
            max_seq_len,
            max_batch_size,
        } = params
        else {
            anyhow::bail!("Expected text AutoDeviceMapParams for this model!")
        };

        let cfg: crate::models::phi2::Config = serde_json::from_str(config)?;

        Ok(
            max_batch_size
                * cfg.num_attention_heads
                * max_seq_len.min(&ATTENTION_CHUNK_SIZE).pow(2),
        )
    }
    fn non_mapped_max_act_size_elems(
        &self,
        _config: &str,
        _params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        Ok(0)
    }

    fn non_mapped_size_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        let cfg: crate::models::phi2::Config = serde_json::from_str(config)?;

        let elems = {
            let embed_tokens = cfg.hidden_size * cfg.vocab_size / weight_pack_factor;
            // If embeddings are tied and no packing, reuse weights -> no separate lm_head needed
            let lm_head = if !cfg.tie_word_embeddings || weight_pack_factor != 1 {
                cfg.hidden_size * cfg.vocab_size / weight_pack_factor
            } else {
                0
            };
            let norm = cfg.hidden_size;
            embed_tokens + lm_head + norm
        };
        Ok(elems * dtype.size_in_bytes())
    }

    fn layer_sizes_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<Vec<usize>> {
        let cfg: crate::models::phi2::Config = serde_json::from_str(config)?;

        let per_layer_elems = {
            let input_layernorm = cfg.hidden_size + cfg.hidden_size;

            let size_in = cfg.hidden_size;
            let size_q = cfg.head_dim() * cfg.num_attention_heads;
            let size_kv = cfg.head_dim() * cfg.num_key_value_heads();
            let q_proj = size_in * size_q / weight_pack_factor + size_q;
            let k_proj = size_in * size_kv / weight_pack_factor + size_kv;
            let v_proj = size_in * size_kv / weight_pack_factor + size_kv;
            let o_proj = size_q * size_in / weight_pack_factor + size_in;
            let (q_norm, k_norm) = if cfg.qk_layernorm {
                (cfg.head_dim(), cfg.head_dim())
            } else {
                (0, 0)
            };

            let h_size = cfg.hidden_size;
            let i_size = cfg.intermediate_size;
            let fc1 = h_size * i_size / weight_pack_factor;
            let fc2 = h_size * i_size / weight_pack_factor;

            input_layernorm + q_proj + k_proj + v_proj + o_proj + q_norm + k_norm + fc1 + fc2
        };
        Ok(vec![
            per_layer_elems * dtype.size_in_bytes();
            cfg.num_hidden_layers
        ])
    }

    fn num_layers(&self, config: &str) -> Result<usize> {
        let cfg: crate::models::phi2::Config = serde_json::from_str(config)?;

        Ok(cfg.num_hidden_layers)
    }

    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg: crate::models::phi2::Config = serde_json::from_str(config)?;

        let cfg = ModelConfigMetadata {
            max_seq_len: cfg.max_position_embeddings,
            num_layers: cfg.num_hidden_layers,
            hidden_size: cfg.hidden_size,
            num_kv_heads: cfg.num_key_value_heads(),
            num_attn_heads: cfg.num_attention_heads,
            sliding_window: None,
            k_head_dim: cfg.head_dim(),
            v_head_dim: cfg.head_dim(),
            kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
        };

        Ok(Box::new(cfg))
    }
}

// ======================== Phi3 loader

/// [`NormalLoader`] for a Phi 3 model.
///
/// [`NormalLoader`]: https://docs.rs/mistralrs/latest/mistralrs/struct.NormalLoader.html
pub struct Phi3Loader;

impl NormalModelLoader for Phi3Loader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        let cfg: crate::models::phi3::Config = serde_json::from_str(config)?;

        Ok(Box::new(models::phi3::Model::new(
            &cfg,
            vb,
            self.is_gptx(config)?,
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }
    fn load_xlora(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        lora_config: &[((String, String), LoraConfig)],
        xlora_config: Option<XLoraConfig>,
        xlora_ordering: Ordering,
        normal_loading_metadata: NormalLoadingMetadata,
        preload_adapters: &Option<HashMap<String, (ShardedVarBuilder, LoraConfig)>>,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        let cfg: crate::models::phi3::Config = serde_json::from_str(config)?;

        Ok(Box::new(xlora_models::XLoraPhi3::new(
            &cfg,
            vb,
            lora_config,
            xlora_config,
            xlora_ordering,
            self.is_gptx(config)?,
            normal_loading_metadata,
            preload_adapters,
        )?))
    }
    fn is_gptx(&self, _: &str) -> Result<bool> {
        Ok(true)
    }
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        let cfg: crate::models::phi3::Config = serde_json::from_str(config)?;

        Ok(Box::new(cfg))
    }
}

impl IsqModelLoader for Phi3Loader {
    fn isq_layer_regexes(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(r"lm_head\.(weight|bias)$")?,
            // Attention
            Regex::new(r"layers\.(\d+)\.self_attn\.qkv_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.o_proj\.(weight|bias)$")?,
            // MLP
            Regex::new(r"layers\.(\d+)\.mlp\.gate_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.mlp\.up_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.mlp\.down_proj\.(weight|bias)$")?,
        ])
    }
    fn immediate_isq_predicates(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes(config)
    }
}

impl DeviceMappedModelLoader for Phi3Loader {
    fn mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        let AutoDeviceMapParams::Text {
            max_seq_len,
            max_batch_size,
        } = params
        else {
            anyhow::bail!("Expected text AutoDeviceMapParams for this model!")
        };

        let cfg: crate::models::phi3::Config = serde_json::from_str(config)?;

        Ok(
            max_batch_size
                * cfg.num_attention_heads
                * max_seq_len.min(&ATTENTION_CHUNK_SIZE).pow(2),
        )
    }
    fn non_mapped_max_act_size_elems(
        &self,
        _config: &str,
        _params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        Ok(0)
    }

    fn non_mapped_size_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        let cfg: crate::models::phi3::Config = serde_json::from_str(config)?;

        let elems = {
            let embed_tokens = cfg.hidden_size * cfg.vocab_size / weight_pack_factor;
            // If embeddings are tied and no packing, reuse weights -> no separate lm_head needed
            let lm_head = if !cfg.tie_word_embeddings || weight_pack_factor != 1 {
                cfg.hidden_size * cfg.vocab_size / weight_pack_factor
            } else {
                0
            };
            let norm = cfg.hidden_size;
            embed_tokens + lm_head + norm
        };
        Ok(elems * dtype.size_in_bytes())
    }

    fn layer_sizes_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<Vec<usize>> {
        let cfg: crate::models::phi3::Config = serde_json::from_str(config)?;

        let per_layer_elems = {
            let input_layernorm = cfg.hidden_size;
            let post_attention_layernorm = cfg.hidden_size;

            let size_in = cfg.hidden_size;
            let head_dim = cfg.head_dim();
            let op_size =
                cfg.num_attention_heads * head_dim + 2 * cfg.num_key_value_heads * head_dim;
            let qkv_proj = size_in * op_size / weight_pack_factor;
            let o_proj =
                (cfg.num_attention_heads * head_dim) * size_in / weight_pack_factor + size_in;

            let h_size = cfg.hidden_size;
            let i_size = cfg.intermediate_size;
            let gate_up_proj = h_size * (2 * i_size) / weight_pack_factor;
            let down_proj = h_size * i_size / weight_pack_factor;

            input_layernorm
                + post_attention_layernorm
                + qkv_proj
                + o_proj
                + gate_up_proj
                + down_proj
        };
        Ok(vec![
            per_layer_elems * dtype.size_in_bytes();
            cfg.num_hidden_layers
        ])
    }

    fn num_layers(&self, config: &str) -> Result<usize> {
        let cfg: crate::models::phi3::Config = serde_json::from_str(config)?;

        Ok(cfg.num_hidden_layers)
    }

    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg: crate::models::phi3::Config = serde_json::from_str(config)?;

        let cfg = ModelConfigMetadata {
            max_seq_len: cfg.max_position_embeddings,
            num_layers: cfg.num_hidden_layers,
            hidden_size: cfg.hidden_size,
            num_kv_heads: cfg.num_key_value_heads,
            num_attn_heads: cfg.num_attention_heads,
            sliding_window: cfg.sliding_window,
            k_head_dim: cfg.head_dim(),
            v_head_dim: cfg.head_dim(),
            kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
        };

        Ok(Box::new(cfg))
    }
}

// ======================== Qwen2 loader

/// [`NormalLoader`] for a Qwen 2 model.
///
/// [`NormalLoader`]: https://docs.rs/mistralrs/latest/mistralrs/struct.NormalLoader.html
pub struct Qwen2Loader;

impl NormalModelLoader for Qwen2Loader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        let cfg: crate::models::qwen2::Config = serde_json::from_str(config)?;

        Ok(Box::new(models::qwen2::Model::new(
            &cfg,
            vb,
            self.is_gptx(config)?,
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }
    fn load_xlora(
        &self,
        _config: &str,
        _vb: ShardedVarBuilder,
        _lora_config: &[((String, String), LoraConfig)],
        _xlora_config: Option<XLoraConfig>,
        _xlora_ordering: Ordering,
        _normal_loading_metadata: NormalLoadingMetadata,
        _preload_adapters: &Option<HashMap<String, (ShardedVarBuilder, LoraConfig)>>,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        todo!()
    }
    fn is_gptx(&self, _: &str) -> Result<bool> {
        Ok(true)
    }
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        let cfg: crate::models::qwen2::Config = serde_json::from_str(config)?;

        Ok(Box::new(cfg))
    }
}

impl IsqModelLoader for Qwen2Loader {
    fn isq_layer_regexes(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(r"lm_head\.(weight|bias)$")?,
            // Attention
            Regex::new(r"layers\.(\d+)\.self_attn\.q_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.k_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.v_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.o_proj\.(weight|bias)$")?,
            // MLP
            Regex::new(r"layers\.(\d+)\.mlp\.gate_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.mlp\.up_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.mlp\.down_proj\.(weight|bias)$")?,
            // MLP MoE
            Regex::new(r"layers\.(\d+)\.mlp\.experts\.(\d+)\.gate_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.mlp\.experts\.(\d+)\.up_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.mlp\.experts\.(\d+)\.down_proj\.(weight|bias)$")?,
        ])
    }
    fn immediate_isq_predicates(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes(config)
    }
}

impl DeviceMappedModelLoader for Qwen2Loader {
    fn mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        let AutoDeviceMapParams::Text {
            max_seq_len,
            max_batch_size,
        } = params
        else {
            anyhow::bail!("Expected text AutoDeviceMapParams for this model!")
        };

        let cfg: crate::models::qwen2::Config = serde_json::from_str(config)?;

        Ok(
            max_batch_size
                * cfg.num_attention_heads
                * max_seq_len.min(&ATTENTION_CHUNK_SIZE).pow(2),
        )
    }
    fn non_mapped_max_act_size_elems(
        &self,
        _config: &str,
        _params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        Ok(0)
    }

    fn non_mapped_size_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        let cfg: crate::models::qwen2::Config = serde_json::from_str(config)?;

        let elems = {
            let embed_tokens = cfg.hidden_size * cfg.vocab_size / weight_pack_factor;
            // If embeddings are tied and no packing, reuse weights -> no separate lm_head needed
            let lm_head = if !cfg.tie_word_embeddings || weight_pack_factor != 1 {
                cfg.hidden_size * cfg.vocab_size / weight_pack_factor
            } else {
                0
            };
            let norm = cfg.hidden_size;
            embed_tokens + lm_head + norm
        };
        Ok(elems * dtype.size_in_bytes())
    }

    fn layer_sizes_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<Vec<usize>> {
        let cfg: crate::models::qwen2::Config = serde_json::from_str(config)?;

        let per_layer_elems = {
            let input_layernorm = cfg.hidden_size;
            let post_attention_layernorm = cfg.hidden_size;

            let size_in = cfg.hidden_size;
            let size_q = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_attention_heads;
            let size_kv = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_key_value_heads;
            let q_proj = size_in * size_q / weight_pack_factor + size_q;
            let k_proj = size_in * size_kv / weight_pack_factor + size_kv;
            let v_proj = size_in * size_kv / weight_pack_factor + size_kv;
            let o_proj = size_q * size_in / weight_pack_factor;

            let h_size = cfg.hidden_size;
            let i_size = cfg.intermediate_size;
            let gate_proj = h_size * i_size / weight_pack_factor;
            let up_proj = h_size * i_size / weight_pack_factor;
            let down_proj = i_size * h_size / weight_pack_factor;

            input_layernorm
                + post_attention_layernorm
                + q_proj
                + k_proj
                + v_proj
                + o_proj
                + gate_proj
                + up_proj
                + down_proj
        };
        Ok(vec![
            per_layer_elems * dtype.size_in_bytes();
            cfg.num_hidden_layers
        ])
    }

    fn num_layers(&self, config: &str) -> Result<usize> {
        let cfg: crate::models::qwen2::Config = serde_json::from_str(config)?;

        Ok(cfg.num_hidden_layers)
    }

    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg: crate::models::qwen2::Config = serde_json::from_str(config)?;

        let cfg = ModelConfigMetadata {
            max_seq_len: cfg.max_position_embeddings,
            num_layers: cfg.num_hidden_layers,
            hidden_size: cfg.hidden_size,
            num_kv_heads: cfg.num_key_value_heads,
            num_attn_heads: cfg.num_attention_heads,
            sliding_window: cfg.sliding_window,
            k_head_dim: cfg.hidden_size / cfg.num_attention_heads,
            v_head_dim: cfg.hidden_size / cfg.num_attention_heads,
            kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
        };

        Ok(Box::new(cfg))
    }
}

// ======================== Gemma2 loader

/// [`NormalLoader`] for a Gemma2 model.
///
/// [`NormalLoader`]: https://docs.rs/mistralrs/latest/mistralrs/struct.NormalLoader.html
pub struct Gemma2Loader;

impl NormalModelLoader for Gemma2Loader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        let cfg: crate::models::gemma2::Config = serde_json::from_str(config)?;

        Ok(Box::new(models::gemma2::Model::new(
            &cfg,
            vb,
            self.is_gptx(config)?,
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }
    fn load_xlora(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        lora_config: &[((String, String), LoraConfig)],
        xlora_config: Option<XLoraConfig>,
        xlora_ordering: Ordering,
        normal_loading_metadata: NormalLoadingMetadata,
        preload_adapters: &Option<HashMap<String, (ShardedVarBuilder, LoraConfig)>>,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        let cfg: crate::models::gemma2::Config = serde_json::from_str(config)?;

        Ok(Box::new(xlora_models::XLoraGemma2::new(
            &cfg,
            vb,
            lora_config,
            xlora_config,
            xlora_ordering,
            self.is_gptx(config)?,
            normal_loading_metadata,
            preload_adapters,
        )?))
    }
    fn is_gptx(&self, _: &str) -> Result<bool> {
        Ok(true)
    }
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        let cfg: crate::models::gemma2::Config = serde_json::from_str(config)?;

        Ok(Box::new(cfg))
    }
}

impl IsqModelLoader for Gemma2Loader {
    fn isq_layer_regexes(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(r"lm_head\.(weight|bias)$")?,
            // Attention
            Regex::new(r"layers\.(\d+)\.self_attn\.q_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.k_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.v_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.o_proj\.(weight|bias)$")?,
            // MLP
            Regex::new(r"layers\.(\d+)\.mlp\.gate_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.mlp\.up_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.mlp\.down_proj\.(weight|bias)$")?,
        ])
    }
    fn immediate_isq_predicates(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes(config)
    }
}

impl DeviceMappedModelLoader for Gemma2Loader {
    fn mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        let AutoDeviceMapParams::Text {
            max_seq_len,
            max_batch_size,
        } = params
        else {
            anyhow::bail!("Expected text AutoDeviceMapParams for this model!")
        };

        let cfg: crate::models::gemma2::Config = serde_json::from_str(config)?;

        Ok(
            max_batch_size
                * cfg.num_attention_heads
                * max_seq_len.min(&ATTENTION_CHUNK_SIZE).pow(2),
        )
    }
    fn non_mapped_max_act_size_elems(
        &self,
        _config: &str,
        _params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        Ok(0)
    }

    fn non_mapped_size_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        let cfg: crate::models::gemma2::Config = serde_json::from_str(config)?;

        let elems = {
            let embed_tokens = cfg.hidden_size * cfg.vocab_size / weight_pack_factor;
            // If embeddings are tied and no packing, reuse weights -> no separate lm_head needed
            let lm_head = if !cfg.tie_word_embeddings || weight_pack_factor != 1 {
                cfg.hidden_size * cfg.vocab_size / weight_pack_factor
            } else {
                0
            };
            let norm = cfg.hidden_size;
            embed_tokens + lm_head + norm
        };
        Ok(elems * dtype.size_in_bytes())
    }

    fn layer_sizes_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<Vec<usize>> {
        let cfg: crate::models::gemma2::Config = serde_json::from_str(config)?;

        let per_layer_elems = {
            let input_layernorm = cfg.hidden_size;
            let post_attention_layernorm = cfg.hidden_size;

            let size_in = cfg.hidden_size;
            let size_q = cfg.head_dim * cfg.num_attention_heads;
            let size_kv = cfg.head_dim * cfg.num_key_value_heads;
            let q_proj =
                size_in * size_q / weight_pack_factor + bias_if!(cfg.attention_bias, size_q);
            let k_proj =
                size_in * size_kv / weight_pack_factor + bias_if!(cfg.attention_bias, size_kv);
            let v_proj =
                size_in * size_kv / weight_pack_factor + bias_if!(cfg.attention_bias, size_kv);
            let o_proj =
                size_q * size_in / weight_pack_factor + bias_if!(cfg.attention_bias, size_in);

            let h_size = cfg.hidden_size;
            let i_size = cfg.intermediate_size;
            let gate_proj = h_size * i_size / weight_pack_factor;
            let up_proj = h_size * i_size / weight_pack_factor;
            let down_proj = i_size * h_size / weight_pack_factor;

            input_layernorm
                + post_attention_layernorm
                + q_proj
                + k_proj
                + v_proj
                + o_proj
                + gate_proj
                + up_proj
                + down_proj
        };
        Ok(vec![
            per_layer_elems * dtype.size_in_bytes();
            cfg.num_hidden_layers
        ])
    }

    fn num_layers(&self, config: &str) -> Result<usize> {
        let cfg: crate::models::gemma2::Config = serde_json::from_str(config)?;

        Ok(cfg.num_hidden_layers)
    }
    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg: crate::models::gemma2::Config = serde_json::from_str(config)?;

        let cfg = ModelConfigMetadata {
            max_seq_len: cfg.max_position_embeddings,
            num_layers: cfg.num_hidden_layers,
            hidden_size: cfg.hidden_size,
            num_kv_heads: cfg.num_key_value_heads,
            num_attn_heads: cfg.num_attention_heads,
            sliding_window: None, // None to be more forgiving, some do not
            k_head_dim: cfg.hidden_size / cfg.num_attention_heads,
            v_head_dim: cfg.hidden_size / cfg.num_attention_heads,
            kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
        };

        Ok(Box::new(cfg))
    }
}

// ======================== Starcoder2 loader

/// [`NormalLoader`] for a Starcoder2 model.
///
/// [`NormalLoader`]: https://docs.rs/mistralrs/latest/mistralrs/struct.NormalLoader.html
pub struct Starcoder2Loader;

impl NormalModelLoader for Starcoder2Loader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        let cfg: crate::models::starcoder2::Config = serde_json::from_str(config)?;

        Ok(Box::new(models::starcoder2::Model::new(
            &cfg,
            vb,
            self.is_gptx(config)?,
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }
    fn load_xlora(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        lora_config: &[((String, String), LoraConfig)],
        xlora_config: Option<XLoraConfig>,
        xlora_ordering: Ordering,
        normal_loading_metadata: NormalLoadingMetadata,
        preload_adapters: &Option<HashMap<String, (ShardedVarBuilder, LoraConfig)>>,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        let cfg: crate::models::starcoder2::Config = serde_json::from_str(config)?;

        Ok(Box::new(xlora_models::XLoraStarcoder2::new(
            &cfg,
            vb,
            lora_config,
            xlora_config,
            xlora_ordering,
            self.is_gptx(config)?,
            normal_loading_metadata,
            preload_adapters,
        )?))
    }
    fn is_gptx(&self, _: &str) -> Result<bool> {
        Ok(true)
    }
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        let cfg: crate::models::starcoder2::Config = serde_json::from_str(config)?;

        Ok(Box::new(cfg))
    }
}

impl IsqModelLoader for Starcoder2Loader {
    fn isq_layer_regexes(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(r"lm_head\.(weight|bias)$")?,
            // Attention
            Regex::new(r"layers\.(\d+)\.self_attn\.q_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.k_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.v_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.o_proj\.(weight|bias)$")?,
            // MLP
            Regex::new(r"layers\.(\d+)\.mlp\.fc1\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.mlp\.c_proj\.(weight|bias)$")?,
        ])
    }
    fn immediate_isq_predicates(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes(config)
    }
}

impl DeviceMappedModelLoader for Starcoder2Loader {
    fn mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        let AutoDeviceMapParams::Text {
            max_seq_len,
            max_batch_size,
        } = params
        else {
            anyhow::bail!("Expected text AutoDeviceMapParams for this model!")
        };

        let cfg: crate::models::starcoder2::Config = serde_json::from_str(config)?;

        Ok(
            max_batch_size
                * cfg.num_attention_heads
                * max_seq_len.min(&ATTENTION_CHUNK_SIZE).pow(2),
        )
    }
    fn non_mapped_max_act_size_elems(
        &self,
        _config: &str,
        _params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        Ok(0)
    }

    fn non_mapped_size_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        let cfg: crate::models::starcoder2::Config = serde_json::from_str(config)?;

        let elems = {
            let embed_tokens = cfg.hidden_size * cfg.vocab_size / weight_pack_factor;
            // If embeddings are tied and no packing, reuse weights -> no separate lm_head needed
            let lm_head = if !cfg.tie_word_embeddings || weight_pack_factor != 1 {
                cfg.hidden_size * cfg.vocab_size / weight_pack_factor
            } else {
                0
            };
            let norm = cfg.hidden_size + cfg.hidden_size;
            embed_tokens + lm_head + norm
        };
        Ok(elems * dtype.size_in_bytes())
    }

    fn layer_sizes_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<Vec<usize>> {
        let cfg: crate::models::starcoder2::Config = serde_json::from_str(config)?;

        let per_layer_elems = {
            let input_layernorm = cfg.hidden_size + cfg.hidden_size;
            let post_attention_layernorm = cfg.hidden_size + cfg.hidden_size;

            let size_in = cfg.hidden_size;
            let size_q = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_attention_heads;
            let size_kv = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_key_value_heads;
            let q_proj = size_in * size_q / weight_pack_factor + bias_if!(cfg.use_bias, size_q);
            let k_proj = size_in * size_kv / weight_pack_factor + bias_if!(cfg.use_bias, size_kv);
            let v_proj = size_in * size_kv / weight_pack_factor + bias_if!(cfg.use_bias, size_kv);
            let o_proj = size_q * size_in / weight_pack_factor + bias_if!(cfg.use_bias, size_in);

            let h_size = cfg.hidden_size;
            let i_size = cfg.intermediate_size;
            let fc1 = h_size * i_size / weight_pack_factor + bias_if!(cfg.use_bias, i_size);
            let fc2 = h_size * i_size / weight_pack_factor + bias_if!(cfg.use_bias, h_size);

            input_layernorm
                + post_attention_layernorm
                + q_proj
                + k_proj
                + v_proj
                + o_proj
                + fc1
                + fc2
        };
        Ok(vec![
            per_layer_elems * dtype.size_in_bytes();
            cfg.num_hidden_layers
        ])
    }

    fn num_layers(&self, config: &str) -> Result<usize> {
        let cfg: crate::models::starcoder2::Config = serde_json::from_str(config)?;

        Ok(cfg.num_hidden_layers)
    }

    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg: crate::models::starcoder2::Config = serde_json::from_str(config)?;

        let cfg = ModelConfigMetadata {
            max_seq_len: cfg.max_position_embeddings,
            num_layers: cfg.num_hidden_layers,
            hidden_size: cfg.hidden_size,
            num_kv_heads: cfg.num_key_value_heads,
            num_attn_heads: cfg.num_attention_heads,
            sliding_window: cfg.sliding_window,
            k_head_dim: cfg.hidden_size / cfg.num_attention_heads,
            v_head_dim: cfg.hidden_size / cfg.num_attention_heads,
            kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
        };

        Ok(Box::new(cfg))
    }
}

// ======================== Phi3 loader

/// [`NormalLoader`] for a Phi 3.5 MoE model.
///
/// [`NormalLoader`]: https://docs.rs/mistralrs/latest/mistralrs/struct.NormalLoader.html
pub struct Phi3_5MoELoader;

impl NormalModelLoader for Phi3_5MoELoader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        let cfg: crate::models::phi3_5_moe::Config = serde_json::from_str(config)?;

        Ok(Box::new(models::phi3_5_moe::Model::new(
            &cfg,
            vb,
            self.is_gptx(config)?,
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }
    fn load_xlora(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        lora_config: &[((String, String), LoraConfig)],
        xlora_config: Option<XLoraConfig>,
        xlora_ordering: Ordering,
        normal_loading_metadata: NormalLoadingMetadata,
        preload_adapters: &Option<HashMap<String, (ShardedVarBuilder, LoraConfig)>>,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        let cfg: crate::models::phi3::Config = serde_json::from_str(config)?;

        Ok(Box::new(xlora_models::XLoraPhi3::new(
            &cfg,
            vb,
            lora_config,
            xlora_config,
            xlora_ordering,
            self.is_gptx(config)?,
            normal_loading_metadata,
            preload_adapters,
        )?))
    }
    fn is_gptx(&self, _: &str) -> Result<bool> {
        Ok(true)
    }
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        let cfg: crate::models::phi3_5_moe::Config = serde_json::from_str(config)?;

        Ok(Box::new(cfg))
    }
}

impl IsqModelLoader for Phi3_5MoELoader {
    fn isq_layer_regexes(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(r"lm_head\.(weight|bias)$")?,
            // Attention
            Regex::new(r"layers\.(\d+)\.self_attn\.q_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.k_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.v_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.o_proj\.(weight|bias)$")?,
            // MLP
            Regex::new(r"layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.w1\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.w2\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.w3\.(weight|bias)$")?,
        ])
    }
    fn immediate_isq_predicates(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes(config)
    }

    fn isq_layer_regexes_moqe(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(r"lm_head\.(weight|bias)$")?,
            // MLP
            Regex::new(r"layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.w1\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.w2\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.w3\.(weight|bias)$")?,
        ])
    }
    fn immediate_isq_predicates_moqe(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes_moqe(config)
    }
}

impl DeviceMappedModelLoader for Phi3_5MoELoader {
    fn mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        let AutoDeviceMapParams::Text {
            max_seq_len,
            max_batch_size,
        } = params
        else {
            anyhow::bail!("Expected text AutoDeviceMapParams for this model!")
        };

        let cfg: crate::models::phi3_5_moe::Config = serde_json::from_str(config)?;

        Ok(
            max_batch_size
                * cfg.num_attention_heads
                * max_seq_len.min(&ATTENTION_CHUNK_SIZE).pow(2),
        )
    }
    fn non_mapped_max_act_size_elems(
        &self,
        _config: &str,
        _params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        Ok(0)
    }

    fn non_mapped_size_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        let cfg: crate::models::phi3_5_moe::Config = serde_json::from_str(config)?;

        let elems = {
            let embed_tokens = cfg.hidden_size * cfg.vocab_size / weight_pack_factor;
            // If embeddings are tied and no packing, reuse weights -> no separate lm_head needed
            let lm_head = if !cfg.tie_word_embeddings || weight_pack_factor != 1 {
                cfg.hidden_size * cfg.vocab_size / weight_pack_factor
            } else {
                0
            };
            let norm = cfg.hidden_size;
            embed_tokens + lm_head + norm
        };
        Ok(elems * dtype.size_in_bytes())
    }

    fn layer_sizes_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<Vec<usize>> {
        let cfg: crate::models::phi3_5_moe::Config = serde_json::from_str(config)?;

        let per_layer_elems = {
            let input_layernorm = cfg.hidden_size;
            let post_attention_layernorm = cfg.hidden_size;

            let size_in = cfg.hidden_size;
            let size_q = cfg.head_dim() * cfg.num_attention_heads;
            let size_kv = cfg.head_dim() * cfg.num_key_value_heads;
            let q_proj =
                size_in * size_q / weight_pack_factor + bias_if!(cfg.attention_bias, size_q);
            let k_proj =
                size_in * size_kv / weight_pack_factor + bias_if!(cfg.attention_bias, size_kv);
            let v_proj =
                size_in * size_kv / weight_pack_factor + bias_if!(cfg.attention_bias, size_kv);
            let o_proj =
                size_q * size_in / weight_pack_factor + bias_if!(cfg.attention_bias, size_in);

            let moe_block = {
                let gate = cfg.hidden_size * cfg.num_local_experts;
                // Assume quantizing weight pack factor
                let w1 = cfg.hidden_size * cfg.intermediate_size / weight_pack_factor;
                let w2 = cfg.hidden_size * cfg.intermediate_size / weight_pack_factor;
                let w3 = cfg.hidden_size * cfg.intermediate_size / weight_pack_factor;
                gate + cfg.num_local_experts * w1
                    + cfg.num_local_experts * w2
                    + cfg.num_local_experts * w3
            };

            input_layernorm
                + post_attention_layernorm
                + q_proj
                + k_proj
                + v_proj
                + o_proj
                + moe_block
        };
        Ok(vec![
            per_layer_elems * dtype.size_in_bytes();
            cfg.num_hidden_layers
        ])
    }

    fn num_layers(&self, config: &str) -> Result<usize> {
        let cfg: crate::models::phi3_5_moe::Config = serde_json::from_str(config)?;

        Ok(cfg.num_hidden_layers)
    }

    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg: crate::models::phi3_5_moe::Config = serde_json::from_str(config)?;

        let cfg = ModelConfigMetadata {
            max_seq_len: cfg.max_position_embeddings,
            num_layers: cfg.num_hidden_layers,
            hidden_size: cfg.hidden_size,
            num_kv_heads: cfg.num_key_value_heads,
            num_attn_heads: cfg.num_attention_heads,
            sliding_window: cfg.sliding_window,
            k_head_dim: cfg.head_dim(),
            v_head_dim: cfg.head_dim(),
            kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
        };

        Ok(Box::new(cfg))
    }
}

/// [`NormalLoader`] for a DeepSeekV2 model.
///
/// [`NormalLoader`]: https://docs.rs/mistralrs/latest/mistralrs/struct.NormalLoader.html
pub struct DeepSeekV2Loader;

impl NormalModelLoader for DeepSeekV2Loader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        let cfg: crate::models::deepseek2::DeepSeekV2Config = serde_json::from_str(config)?;

        Ok(Box::new(models::deepseek2::DeepSeekV2::new(
            &cfg,
            vb,
            self.is_gptx(config)?,
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }
    fn load_xlora(
        &self,
        _config: &str,
        _vb: ShardedVarBuilder,
        _lora_config: &[((String, String), LoraConfig)],
        _xlora_config: Option<XLoraConfig>,
        _xlora_ordering: Ordering,
        _normal_loading_metadata: NormalLoadingMetadata,
        _preload_adapters: &Option<HashMap<String, (ShardedVarBuilder, LoraConfig)>>,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        todo!()
    }
    fn is_gptx(&self, _: &str) -> Result<bool> {
        Ok(true)
    }
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        let cfg: crate::models::deepseek2::DeepSeekV2Config = serde_json::from_str(config)?;
        Ok(Box::new(cfg))
    }
}

impl IsqModelLoader for DeepSeekV2Loader {
    fn isq_layer_regexes(&self, config: &str) -> Result<Vec<Regex>> {
        let mut data = vec![
            Regex::new(r"lm_head\.(weight|bias)$")?,
            // Attention
            Regex::new(r"layers\.(\d+)\.self_attn\.kv_a_proj_with_mqa\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.kv_b_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.o_proj\.(weight|bias)$")?,
        ];
        let cfg: crate::models::deepseek2::DeepSeekV2Config = serde_json::from_str(config)?;
        if cfg.q_lora_rank.is_some() {
            data.extend(vec![
                Regex::new(r"layers\.(\d+)\.self_attn\.q_a_proj\.(weight|bias)$")?,
                Regex::new(r"layers\.(\d+)\.self_attn\.q_b_proj\.(weight|bias)$")?,
            ]);
        } else {
            data.push(Regex::new(
                r"layers\.(\d+)\.self_attn\.q_proj\.(weight|bias)$",
            )?);
        }
        for layer_idx in 0..cfg.num_hidden_layers {
            if let Some(n_routed_experts) = cfg.n_routed_experts.filter(|_| {
                layer_idx >= cfg.first_k_dense_replace && layer_idx % cfg.moe_layer_freq == 0
            }) {
                for i in 0..n_routed_experts {
                    data.extend(vec![
                        Regex::new(&format!(
                            r"layers\.{layer_idx}\.mlp\.experts\.{i}\.gate_proj\.(weight|bias)$"
                        ))?,
                        Regex::new(&format!(
                            r"layers\.{layer_idx}\.mlp\.experts\.{i}\.up_proj\.(weight|bias)$"
                        ))?,
                        Regex::new(&format!(
                            r"layers\.{layer_idx}\.mlp\.experts\.{i}\.down_proj\.(weight|bias)$"
                        ))?,
                    ]);
                }
                if cfg.n_shared_experts.is_some() {
                    data.extend(vec![
                        Regex::new(&format!(
                            r"layers\.{layer_idx}\.mlp\.shared_experts\.gate_proj\.(weight|bias)$"
                        ))?,
                        Regex::new(&format!(
                            r"layers\.{layer_idx}\.mlp\.shared_experts\.up_proj\.(weight|bias)$"
                        ))?,
                        Regex::new(&format!(
                            r"layers\.{layer_idx}\.mlp\.shared_experts\.down_proj\.(weight|bias)$"
                        ))?,
                    ]);
                }
            } else {
                data.extend(vec![
                    Regex::new(&format!(
                        r"layers\.{layer_idx}\.mlp\.gate_proj\.(weight|bias)$"
                    ))?,
                    Regex::new(&format!(r"layers.{layer_idx}.mlp\.up_proj\.(weight|bias)$"))?,
                    Regex::new(&format!(
                        r"layers\.{layer_idx}\.mlp\.down_proj\.(weight|bias)$"
                    ))?,
                ]);
            };
        }
        Ok(data)
    }
    fn immediate_isq_predicates(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes(config)
    }

    fn isq_layer_regexes_moqe(&self, config: &str) -> Result<Vec<Regex>> {
        let mut data = vec![Regex::new(r"lm_head\.(weight|bias)$")?];
        let cfg: crate::models::deepseek2::DeepSeekV2Config = serde_json::from_str(config)?;
        for layer_idx in 0..cfg.num_hidden_layers {
            if let Some(n_routed_experts) = cfg.n_routed_experts.filter(|_| {
                layer_idx >= cfg.first_k_dense_replace && layer_idx % cfg.moe_layer_freq == 0
            }) {
                for i in 0..n_routed_experts {
                    data.extend(vec![
                        Regex::new(&format!(
                            r"layers\.{layer_idx}\.mlp\.experts\.{i}\.gate_proj\.(weight|bias)$"
                        ))?,
                        Regex::new(&format!(
                            r"layers\.{layer_idx}\.mlp\.experts\.{i}\.up_proj\.(weight|bias)$"
                        ))?,
                        Regex::new(&format!(
                            r"layers\.{layer_idx}\.mlp\.experts\.{i}\.down_proj\.(weight|bias)$"
                        ))?,
                    ]);
                }
                if cfg.n_shared_experts.is_some() {
                    data.extend(vec![
                        Regex::new(&format!(
                            r"layers\.{layer_idx}\.mlp\.shared_experts\.gate_proj\.(weight|bias)$"
                        ))?,
                        Regex::new(&format!(
                            r"layers\.{layer_idx}\.mlp\.shared_experts\.up_proj\.(weight|bias)$"
                        ))?,
                        Regex::new(&format!(
                            r"layers\.{layer_idx}\.mlp\.shared_experts\.down_proj\.(weight|bias)$"
                        ))?,
                    ]);
                }
            } else {
                data.extend(vec![
                    Regex::new(&format!(
                        r"layers\.{layer_idx}\.mlp\.gate_proj\.(weight|bias)$"
                    ))?,
                    Regex::new(&format!(r"layers.{layer_idx}.mlp\.up_proj\.(weight|bias)$"))?,
                    Regex::new(&format!(
                        r"layers\.{layer_idx}\.mlp\.down_proj\.(weight|bias)$"
                    ))?,
                ]);
            };
        }
        Ok(data)
    }
    fn immediate_isq_predicates_moqe(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes_moqe(config)
    }
}

impl DeviceMappedModelLoader for DeepSeekV2Loader {
    fn mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        let AutoDeviceMapParams::Text {
            max_seq_len,
            max_batch_size,
        } = params
        else {
            anyhow::bail!("Expected text AutoDeviceMapParams for this model!")
        };

        let cfg: crate::models::deepseek2::DeepSeekV2Config = serde_json::from_str(config)?;

        Ok(
            max_batch_size
                * cfg.num_attention_heads
                * max_seq_len.min(&ATTENTION_CHUNK_SIZE).pow(2),
        )
    }
    fn non_mapped_max_act_size_elems(
        &self,
        _config: &str,
        _params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        Ok(0)
    }

    fn non_mapped_size_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        let cfg: crate::models::deepseek2::DeepSeekV2Config = serde_json::from_str(config)?;
        let elems = {
            let embed_tokens = cfg.hidden_size * cfg.vocab_size / weight_pack_factor;
            // If embeddings are tied and no packing, reuse weights -> no separate lm_head needed
            let lm_head = if !cfg.tie_word_embeddings || weight_pack_factor != 1 {
                cfg.hidden_size * cfg.vocab_size / weight_pack_factor
            } else {
                0
            };
            let norm = cfg.hidden_size;
            embed_tokens + lm_head + norm
        };
        Ok(elems * dtype.size_in_bytes())
    }

    fn layer_sizes_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<Vec<usize>> {
        let cfg: crate::models::deepseek2::DeepSeekV2Config = serde_json::from_str(config)?;
        let mut per_layer_elems = Vec::new();

        for layer_idx in 0..cfg.num_hidden_layers {
            let input_layernorm = cfg.hidden_size;
            let post_attention_layernorm = cfg.hidden_size;

            let q_proj = match cfg.q_lora_rank {
                Some(lora_rank) => {
                    let a = cfg.hidden_size * lora_rank;
                    let norm = lora_rank;
                    let b = (cfg.num_attention_heads * cfg.q_head_dim()) * lora_rank;
                    a + norm + b
                }
                None => (cfg.num_attention_heads * cfg.q_head_dim()) * cfg.hidden_size,
            };
            let kv_a_proj_with_mqa = cfg.hidden_size * (cfg.kv_lora_rank + cfg.qk_rope_head_dim)
                / weight_pack_factor
                + bias_if!(cfg.attention_bias, cfg.kv_lora_rank + cfg.qk_rope_head_dim);
            let kv_a_layernorm = cfg.kv_lora_rank;
            let kv_b_proj = cfg.kv_lora_rank
                * cfg.num_attention_heads
                * (cfg.q_head_dim() - cfg.qk_rope_head_dim + cfg.v_head_dim)
                / weight_pack_factor;
            let o_proj = cfg.num_attention_heads * cfg.v_head_dim * cfg.hidden_size
                / weight_pack_factor
                + bias_if!(cfg.attention_bias, cfg.hidden_size);

            let moe_block = {
                let mut sum = 0;
                if let Some(n_routed_experts) = cfg.n_routed_experts.filter(|_| {
                    layer_idx >= cfg.first_k_dense_replace && layer_idx % cfg.moe_layer_freq == 0
                }) {
                    let h_size = cfg.hidden_size;
                    let gate_proj =
                        h_size * cfg.moe_intermediate_size / weight_pack_factor * n_routed_experts;
                    let up_proj =
                        h_size * cfg.moe_intermediate_size / weight_pack_factor * n_routed_experts;
                    let down_proj =
                        cfg.moe_intermediate_size * h_size / weight_pack_factor * n_routed_experts;
                    let shared_experts = if let Some(n_shared_experts) = cfg.n_shared_experts {
                        let gate_proj = h_size * (cfg.intermediate_size * n_shared_experts)
                            / weight_pack_factor;
                        let up_proj = h_size * (cfg.intermediate_size * n_shared_experts)
                            / weight_pack_factor;
                        let down_proj = (cfg.intermediate_size * n_shared_experts) * h_size
                            / weight_pack_factor;
                        gate_proj + up_proj + down_proj
                    } else {
                        0
                    };
                    let gate_weight = n_routed_experts * cfg.hidden_size;
                    sum += gate_proj + up_proj + down_proj + shared_experts + gate_weight;
                } else {
                    let h_size = cfg.hidden_size;
                    let i_size = cfg.intermediate_size;
                    let gate_proj = h_size * i_size / weight_pack_factor;
                    let up_proj = h_size * i_size / weight_pack_factor;
                    let down_proj = i_size * h_size / weight_pack_factor;
                    sum += gate_proj + up_proj + down_proj;
                }
                sum
            };

            per_layer_elems.push(
                input_layernorm
                    + post_attention_layernorm
                    + q_proj
                    + kv_a_layernorm
                    + kv_a_proj_with_mqa
                    + kv_b_proj
                    + o_proj
                    + moe_block,
            );
        }

        Ok(per_layer_elems
            .into_iter()
            .map(|x| x * dtype.size_in_bytes())
            .collect())
    }

    fn num_layers(&self, config: &str) -> Result<usize> {
        let cfg: crate::models::deepseek2::DeepSeekV2Config = serde_json::from_str(config)?;
        Ok(cfg.num_hidden_layers)
    }

    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg: crate::models::deepseek2::DeepSeekV2Config = serde_json::from_str(config)?;

        let cfg = ModelConfigMetadata {
            max_seq_len: cfg.max_position_embeddings,
            num_layers: cfg.num_hidden_layers,
            hidden_size: cfg.hidden_size,
            num_kv_heads: cfg.num_attention_heads,
            num_attn_heads: cfg.num_attention_heads,
            sliding_window: None,
            k_head_dim: cfg.qk_rope_head_dim + cfg.qk_nope_head_dim,
            v_head_dim: cfg.v_head_dim,
            kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
        };

        Ok(Box::new(cfg))
    }
}

/// [`NormalLoader`] for a DeepSeekV3 model.
///
/// [`NormalLoader`]: https://docs.rs/mistralrs/latest/mistralrs/struct.NormalLoader.html
pub struct DeepSeekV3Loader;

impl NormalModelLoader for DeepSeekV3Loader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        let cfg: crate::models::deepseek3::DeepSeekV3Config = serde_json::from_str(config)?;
        Ok(Box::new(models::deepseek3::DeepSeekV3::new(
            &cfg,
            vb,
            self.is_gptx(config)?,
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }
    fn load_xlora(
        &self,
        _config: &str,
        _vb: ShardedVarBuilder,
        _lora_config: &[((String, String), LoraConfig)],
        _xlora_config: Option<XLoraConfig>,
        _xlora_ordering: Ordering,
        _normal_loading_metadata: NormalLoadingMetadata,
        _preload_adapters: &Option<HashMap<String, (ShardedVarBuilder, LoraConfig)>>,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        todo!()
    }
    fn is_gptx(&self, _: &str) -> Result<bool> {
        Ok(true)
    }
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        let cfg: crate::models::deepseek3::DeepSeekV3Config = serde_json::from_str(config)?;
        Ok(Box::new(cfg))
    }
}

impl IsqModelLoader for DeepSeekV3Loader {
    fn isq_layer_regexes(&self, config: &str) -> Result<Vec<Regex>> {
        let mut data = vec![
            Regex::new(r"lm_head\.(weight|bias)$")?,
            // Attention
            Regex::new(r"layers\.(\d+)\.self_attn\.kv_a_proj_with_mqa\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.kv_b_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.o_proj\.(weight|bias)$")?,
        ];
        let cfg: crate::models::deepseek3::DeepSeekV3Config = serde_json::from_str(config)?;
        if cfg.q_lora_rank.is_some() {
            data.extend(vec![
                Regex::new(r"layers\.(\d+)\.self_attn\.q_a_proj\.(weight|bias)$")?,
                Regex::new(r"layers\.(\d+)\.self_attn\.q_b_proj\.(weight|bias)$")?,
            ]);
        } else {
            data.push(Regex::new(
                r"layers\.(\d+)\.self_attn\.q_proj\.(weight|bias)$",
            )?);
        }
        for layer_idx in 0..cfg.num_hidden_layers {
            if let Some(n_routed_experts) = cfg.n_routed_experts.filter(|_| {
                layer_idx >= cfg.first_k_dense_replace && layer_idx % cfg.moe_layer_freq == 0
            }) {
                for i in 0..n_routed_experts {
                    data.extend(vec![
                        Regex::new(&format!(
                            r"layers\.{layer_idx}\.mlp\.experts\.{i}\.gate_proj\.(weight|bias)$"
                        ))?,
                        Regex::new(&format!(
                            r"layers\.{layer_idx}\.mlp\.experts\.{i}\.up_proj\.(weight|bias)$"
                        ))?,
                        Regex::new(&format!(
                            r"layers\.{layer_idx}\.mlp\.experts\.{i}\.down_proj\.(weight|bias)$"
                        ))?,
                    ]);
                }
                if cfg.n_shared_experts.is_some() {
                    data.extend(vec![
                        Regex::new(&format!(
                            r"layers\.{layer_idx}\.mlp\.shared_experts\.gate_proj\.(weight|bias)$"
                        ))?,
                        Regex::new(&format!(
                            r"layers\.{layer_idx}\.mlp\.shared_experts\.up_proj\.(weight|bias)$"
                        ))?,
                        Regex::new(&format!(
                            r"layers\.{layer_idx}\.mlp\.shared_experts\.down_proj\.(weight|bias)$"
                        ))?,
                    ]);
                }
            } else {
                data.extend(vec![
                    Regex::new(&format!(
                        r"layers\.{layer_idx}\.mlp\.gate_proj\.(weight|bias)$"
                    ))?,
                    Regex::new(&format!(r"layers.{layer_idx}.mlp\.up_proj\.(weight|bias)$"))?,
                    Regex::new(&format!(
                        r"layers\.{layer_idx}\.mlp\.down_proj\.(weight|bias)$"
                    ))?,
                ]);
            };
        }
        Ok(data)
    }
    fn immediate_isq_predicates(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes(config)
    }

    fn isq_layer_regexes_moqe(&self, config: &str) -> Result<Vec<Regex>> {
        let mut data = vec![Regex::new(r"lm_head\.(weight|bias)$")?];
        let cfg: crate::models::deepseek3::DeepSeekV3Config = serde_json::from_str(config)?;
        for layer_idx in 0..cfg.num_hidden_layers {
            if let Some(n_routed_experts) = cfg.n_routed_experts.filter(|_| {
                layer_idx >= cfg.first_k_dense_replace && layer_idx % cfg.moe_layer_freq == 0
            }) {
                for i in 0..n_routed_experts {
                    data.extend(vec![
                        Regex::new(&format!(
                            r"layers\.{layer_idx}\.mlp\.experts\.{i}\.gate_proj\.(weight|bias)$"
                        ))?,
                        Regex::new(&format!(
                            r"layers\.{layer_idx}\.mlp\.experts\.{i}\.up_proj\.(weight|bias)$"
                        ))?,
                        Regex::new(&format!(
                            r"layers\.{layer_idx}\.mlp\.experts\.{i}\.down_proj\.(weight|bias)$"
                        ))?,
                    ]);
                }
                if cfg.n_shared_experts.is_some() {
                    data.extend(vec![
                        Regex::new(&format!(
                            r"layers\.{layer_idx}\.mlp\.shared_experts\.gate_proj\.(weight|bias)$"
                        ))?,
                        Regex::new(&format!(
                            r"layers\.{layer_idx}\.mlp\.shared_experts\.up_proj\.(weight|bias)$"
                        ))?,
                        Regex::new(&format!(
                            r"layers\.{layer_idx}\.mlp\.shared_experts\.down_proj\.(weight|bias)$"
                        ))?,
                    ]);
                }
            } else {
                data.extend(vec![
                    Regex::new(&format!(
                        r"layers\.{layer_idx}\.mlp\.gate_proj\.(weight|bias)$"
                    ))?,
                    Regex::new(&format!(r"layers.{layer_idx}.mlp\.up_proj\.(weight|bias)$"))?,
                    Regex::new(&format!(
                        r"layers\.{layer_idx}\.mlp\.down_proj\.(weight|bias)$"
                    ))?,
                ]);
            };
        }
        Ok(data)
    }
    fn immediate_isq_predicates_moqe(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes_moqe(config)
    }
}

impl DeviceMappedModelLoader for DeepSeekV3Loader {
    fn mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        let AutoDeviceMapParams::Text {
            max_seq_len,
            max_batch_size,
        } = params
        else {
            anyhow::bail!("Expected text AutoDeviceMapParams for this model!")
        };

        let cfg: crate::models::deepseek3::DeepSeekV3Config = serde_json::from_str(config)?;

        Ok(
            max_batch_size
                * cfg.num_attention_heads
                * max_seq_len.min(&ATTENTION_CHUNK_SIZE).pow(2),
        )
    }
    fn non_mapped_max_act_size_elems(
        &self,
        _config: &str,
        _params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        Ok(0)
    }

    fn non_mapped_size_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        let cfg: crate::models::deepseek3::DeepSeekV3Config = serde_json::from_str(config)?;
        let elems = {
            let embed_tokens = cfg.hidden_size * cfg.vocab_size / weight_pack_factor;
            // If embeddings are tied and no packing, reuse weights -> no separate lm_head needed
            let lm_head = if !cfg.tie_word_embeddings || weight_pack_factor != 1 {
                cfg.hidden_size * cfg.vocab_size / weight_pack_factor
            } else {
                0
            };
            let norm = cfg.hidden_size;
            embed_tokens + lm_head + norm
        };
        Ok(elems * dtype.size_in_bytes())
    }

    fn layer_sizes_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<Vec<usize>> {
        let cfg: crate::models::deepseek3::DeepSeekV3Config = serde_json::from_str(config)?;
        let mut per_layer_elems = Vec::new();

        for layer_idx in 0..cfg.num_hidden_layers {
            let input_layernorm = cfg.hidden_size;
            let post_attention_layernorm = cfg.hidden_size;

            let q_proj = match cfg.q_lora_rank {
                Some(lora_rank) => {
                    let a = cfg.hidden_size * lora_rank;
                    let norm = lora_rank;
                    let b = (cfg.num_attention_heads * cfg.q_head_dim()) * lora_rank;
                    a + norm + b
                }
                None => (cfg.num_attention_heads * cfg.q_head_dim()) * cfg.hidden_size,
            };
            let kv_a_proj_with_mqa = cfg.hidden_size * (cfg.kv_lora_rank + cfg.qk_rope_head_dim)
                / weight_pack_factor
                + bias_if!(cfg.attention_bias, cfg.kv_lora_rank + cfg.qk_rope_head_dim);
            let kv_a_layernorm = cfg.kv_lora_rank;
            let kv_b_proj = cfg.kv_lora_rank
                * cfg.num_attention_heads
                * (cfg.q_head_dim() - cfg.qk_rope_head_dim + cfg.v_head_dim)
                / weight_pack_factor;
            let o_proj = cfg.num_attention_heads * cfg.v_head_dim * cfg.hidden_size
                / weight_pack_factor
                + bias_if!(cfg.attention_bias, cfg.hidden_size);

            let moe_block = {
                let mut sum = 0;
                if let Some(n_routed_experts) = cfg.n_routed_experts.filter(|_| {
                    layer_idx >= cfg.first_k_dense_replace && layer_idx % cfg.moe_layer_freq == 0
                }) {
                    let h_size = cfg.hidden_size;
                    let gate_proj =
                        h_size * cfg.moe_intermediate_size / weight_pack_factor * n_routed_experts;
                    let up_proj =
                        h_size * cfg.moe_intermediate_size / weight_pack_factor * n_routed_experts;
                    let down_proj =
                        cfg.moe_intermediate_size * h_size / weight_pack_factor * n_routed_experts;
                    let shared_experts = if let Some(n_shared_experts) = cfg.n_shared_experts {
                        let gate_proj = h_size * (cfg.intermediate_size * n_shared_experts)
                            / weight_pack_factor;
                        let up_proj = h_size * (cfg.intermediate_size * n_shared_experts)
                            / weight_pack_factor;
                        let down_proj = (cfg.intermediate_size * n_shared_experts) * h_size
                            / weight_pack_factor;
                        gate_proj + up_proj + down_proj
                    } else {
                        0
                    };
                    let gate_weight = n_routed_experts * cfg.hidden_size;
                    sum += gate_proj + up_proj + down_proj + shared_experts + gate_weight;
                } else {
                    let h_size = cfg.hidden_size;
                    let i_size = cfg.intermediate_size;
                    let gate_proj = h_size * i_size / weight_pack_factor;
                    let up_proj = h_size * i_size / weight_pack_factor;
                    let down_proj = i_size * h_size / weight_pack_factor;
                    sum += gate_proj + up_proj + down_proj;
                }
                sum
            };

            per_layer_elems.push(
                input_layernorm
                    + post_attention_layernorm
                    + q_proj
                    + kv_a_layernorm
                    + kv_a_proj_with_mqa
                    + kv_b_proj
                    + o_proj
                    + moe_block,
            );
        }

        Ok(per_layer_elems
            .into_iter()
            .map(|x| x * dtype.size_in_bytes())
            .collect())
    }

    fn num_layers(&self, config: &str) -> Result<usize> {
        let cfg: crate::models::deepseek3::DeepSeekV3Config = serde_json::from_str(config)?;
        Ok(cfg.num_hidden_layers)
    }

    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg: crate::models::deepseek3::DeepSeekV3Config = serde_json::from_str(config)?;

        let cfg = ModelConfigMetadata {
            max_seq_len: cfg.max_position_embeddings,
            num_layers: cfg.num_hidden_layers,
            hidden_size: cfg.hidden_size,
            num_kv_heads: cfg.num_attention_heads,
            num_attn_heads: cfg.num_attention_heads,
            sliding_window: None,
            k_head_dim: cfg.qk_rope_head_dim + cfg.qk_nope_head_dim,
            v_head_dim: cfg.v_head_dim,
            kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
        };

        Ok(Box::new(cfg))
    }
}

/// [`NormalLoader`] for a DeepSeek V4 model (Pro or Flash).
///
/// V4 has a distinct tensor layout from V3 (per
/// `research/code/06_foundation/sglang/python/sglang/srt/models/deepseek_v4.py`).
/// This loader reuses V3's MLA forward path at Tier A (V3-quality dense MLA);
/// real V4 perf wins from CSA/HCA/MTP/mHC require CUDA TileLang kernels
/// and are scheduled for the B200 rental cycle (RUN-155, RUN-156).
///
/// [`NormalLoader`]: https://docs.rs/mistralrs/latest/mistralrs/struct.NormalLoader.html
pub struct DeepSeekV4Loader;

impl NormalModelLoader for DeepSeekV4Loader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        let cfg: crate::models::deepseek4::DeepSeekV4Config = serde_json::from_str(config)?;
        // V4 publishes FP8 block scales under `<weight>.scale`; Arc's
        // BlockwiseFP8Linear expects HF-convention `<weight>.weight_scale_inv`.
        // Attach a transparent rename layer so the loader can use the HF
        // path regardless of which naming the on-disk checkpoint uses.
        // Audit §0 P0 #5 + SGLang `remap_weight_name_to_dpsk_hf_format`
        // (deepseek_v4.py:1413).
        //
        // The wrapper is no-op for HF-style checkpoints (the original name
        // exists directly), so this is safe to apply unconditionally.
        let vb = mistralrs_quant::attach_rename_rules(
            vb,
            mistralrs_quant::v4_scale_rename_rules(),
        );
        Ok(Box::new(models::deepseek4::DeepSeekV4::new(
            &cfg,
            vb,
            self.is_gptx(config)?,
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }
    fn load_xlora(
        &self,
        _config: &str,
        _vb: ShardedVarBuilder,
        _lora_config: &[((String, String), LoraConfig)],
        _xlora_config: Option<XLoraConfig>,
        _xlora_ordering: Ordering,
        _normal_loading_metadata: NormalLoadingMetadata,
        _preload_adapters: &Option<HashMap<String, (ShardedVarBuilder, LoraConfig)>>,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        anyhow::bail!("V4 xlora is not supported")
    }
    fn is_gptx(&self, _: &str) -> Result<bool> {
        Ok(true)
    }
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        let cfg: crate::models::deepseek4::DeepSeekV4Config = serde_json::from_str(config)?;
        Ok(Box::new(cfg))
    }
}

impl IsqModelLoader for DeepSeekV4Loader {
    fn isq_layer_regexes(&self, config: &str) -> Result<Vec<Regex>> {
        // V4 tensor regexes match the HF-mapped names (same paths as V3 since we
        // delegate to V3's MLA structure at the loader level).
        let mut data = vec![
            Regex::new(r"lm_head\.(weight|bias)$")?,
            // V4 attention: q_a/q_b (LoRA), kv_a/kv_b (still MLA-LoRA at this scaffolding tier), o_a/o_b (V4 LoRA)
            Regex::new(r"layers\.(\d+)\.self_attn\.q_a_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.q_b_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.kv_a_proj_with_mqa\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.kv_b_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.o_a_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.o_b_proj\.(weight|bias)$")?,
        ];
        let cfg: crate::models::deepseek4::DeepSeekV4Config = serde_json::from_str(config)?;
        for layer_idx in 0..cfg.num_hidden_layers {
            if let Some(n_routed_experts) = cfg.n_routed_experts.filter(|_| {
                layer_idx >= cfg.first_k_dense_replace && layer_idx % cfg.moe_layer_freq == 0
            }) {
                for i in 0..n_routed_experts {
                    data.extend(vec![
                        Regex::new(&format!(
                            r"layers\.{layer_idx}\.mlp\.experts\.{i}\.gate_proj\.(weight|bias)$"
                        ))?,
                        Regex::new(&format!(
                            r"layers\.{layer_idx}\.mlp\.experts\.{i}\.up_proj\.(weight|bias)$"
                        ))?,
                        Regex::new(&format!(
                            r"layers\.{layer_idx}\.mlp\.experts\.{i}\.down_proj\.(weight|bias)$"
                        ))?,
                    ]);
                }
            }
        }
        Ok(data)
    }
    fn immediate_isq_predicates(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes(config)
    }
    fn isq_layer_regexes_moqe(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes(config)
    }
    fn immediate_isq_predicates_moqe(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes(config)
    }
}

impl DeviceMappedModelLoader for DeepSeekV4Loader {
    fn mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        let AutoDeviceMapParams::Text {
            max_seq_len,
            max_batch_size,
        } = params
        else {
            anyhow::bail!("Expected text AutoDeviceMapParams for V4")
        };
        let cfg: crate::models::deepseek4::DeepSeekV4Config = serde_json::from_str(config)?;
        Ok(max_batch_size * cfg.num_attention_heads * max_seq_len.min(&ATTENTION_CHUNK_SIZE).pow(2))
    }
    fn non_mapped_max_act_size_elems(
        &self,
        _config: &str,
        _params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        Ok(0)
    }
    fn non_mapped_size_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        let cfg: crate::models::deepseek4::DeepSeekV4Config = serde_json::from_str(config)?;
        let elems = {
            let embed = cfg.hidden_size * cfg.vocab_size / weight_pack_factor;
            let lm_head = if !cfg.tie_word_embeddings || weight_pack_factor != 1 {
                cfg.hidden_size * cfg.vocab_size / weight_pack_factor
            } else {
                0
            };
            let norm = cfg.hidden_size;
            embed + lm_head + norm
        };
        Ok(elems * dtype.size_in_bytes())
    }
    fn layer_sizes_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<Vec<usize>> {
        let cfg: crate::models::deepseek4::DeepSeekV4Config = serde_json::from_str(config)?;
        let mut per_layer_elems = Vec::new();
        let head_dim = cfg.head_dim;
        for layer_idx in 0..cfg.num_hidden_layers {
            let input_layernorm = cfg.hidden_size;
            let post_attention_layernorm = cfg.hidden_size;
            // V4 Q: LoRA (wq_a + q_norm + wq_b). Audit §0 + §2.
            let q_proj = match cfg.q_lora_rank {
                Some(lora_rank) => {
                    let a = cfg.hidden_size * lora_rank;
                    let norm = lora_rank;
                    let b = (cfg.num_attention_heads * head_dim) * lora_rank;
                    a + norm + b
                }
                None => (cfg.num_attention_heads * head_dim) * cfg.hidden_size,
            };
            // V4 K/V: SINGLE fused wkv (hidden → head_dim). No LoRA-A/B split.
            // Audit §0 + §5 lines 477-482. No `kv_b_proj` at all.
            let wkv = cfg.hidden_size * head_dim / weight_pack_factor;
            let kv_norm = head_dim;
            // V4 grouped o_proj: wo_a + wo_b. Audit §0 + §2.
            let o_lora_rank = cfg.o_lora_rank.unwrap_or(cfg.hidden_size);
            let o_groups = cfg.o_groups.unwrap_or(1).max(1);
            let o_inner = o_groups * o_lora_rank;
            let wo_a_in = cfg.num_attention_heads * head_dim / o_groups;
            let wo_a = wo_a_in * o_inner / weight_pack_factor;
            let wo_b = o_inner * cfg.hidden_size / weight_pack_factor;

            let moe_block = {
                let mut sum: usize = 0;
                if let Some(n_routed_experts) = cfg.n_routed_experts.filter(|_| {
                    layer_idx >= cfg.first_k_dense_replace && layer_idx % cfg.moe_layer_freq == 0
                }) {
                    let h = cfg.hidden_size;
                    let i = cfg.moe_intermediate_size;
                    let gate = h * i / weight_pack_factor * n_routed_experts;
                    let up = h * i / weight_pack_factor * n_routed_experts;
                    let down = i * h / weight_pack_factor * n_routed_experts;
                    let shared = if let Some(ns) = cfg.n_shared_experts {
                        // V4 uses moe_intermediate_size * n_shared_experts
                        // for the shared expert (audit §0 + V4 Flash config).
                        let inter = cfg.moe_intermediate_size * ns;
                        let sg = h * inter / weight_pack_factor;
                        let su = h * inter / weight_pack_factor;
                        let sd = inter * h / weight_pack_factor;
                        sg + su + sd
                    } else {
                        0
                    };
                    sum += gate + up + down + shared + n_routed_experts * cfg.hidden_size;
                } else if let Some(i) = cfg.intermediate_size {
                    // Dense MLP fallback (V3-style fixtures only;
                    // unreachable for V4 Flash since first_k_dense_replace=0).
                    let h = cfg.hidden_size;
                    sum += 3 * h * i / weight_pack_factor;
                }
                sum
            };
            per_layer_elems.push(
                input_layernorm + post_attention_layernorm
                    + q_proj + kv_norm + wkv
                    + wo_a + wo_b + moe_block,
            );
        }
        Ok(per_layer_elems.into_iter().map(|x| x * dtype.size_in_bytes()).collect())
    }
    fn num_layers(&self, config: &str) -> Result<usize> {
        let cfg: crate::models::deepseek4::DeepSeekV4Config = serde_json::from_str(config)?;
        Ok(cfg.num_hidden_layers)
    }
    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg: crate::models::deepseek4::DeepSeekV4Config = serde_json::from_str(config)?;
        let cfg = ModelConfigMetadata {
            max_seq_len: cfg.max_position_embeddings,
            num_layers: cfg.num_hidden_layers,
            hidden_size: cfg.hidden_size,
            // V4 MQA: 1 KV head. Audit §0 + §3.
            num_kv_heads: cfg.num_key_value_heads.max(1),
            num_attn_heads: cfg.num_attention_heads,
            sliding_window: Some(cfg.sliding_window),
            // V4: single head_dim for both K and V (same fused wkv).
            // Audit §3.
            k_head_dim: cfg.head_dim,
            v_head_dim: cfg.head_dim,
            kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
        };
        Ok(Box::new(cfg))
    }
}

/// [`NormalLoader`] for a Qwen 3 model.
///
/// [`NormalLoader`]: https://docs.rs/mistralrs/latest/mistralrs/struct.NormalLoader.html
pub struct Qwen3Loader;

impl NormalModelLoader for Qwen3Loader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        let cfg: crate::models::qwen3::Config = serde_json::from_str(config)?;

        Ok(Box::new(models::qwen3::Model::new(
            &cfg,
            vb,
            self.is_gptx(config)?,
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }
    fn load_xlora(
        &self,
        _config: &str,
        _vb: ShardedVarBuilder,
        _lora_config: &[((String, String), LoraConfig)],
        _xlora_config: Option<XLoraConfig>,
        _xlora_ordering: Ordering,
        _normal_loading_metadata: NormalLoadingMetadata,
        _preload_adapters: &Option<HashMap<String, (ShardedVarBuilder, LoraConfig)>>,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        todo!()
    }
    fn is_gptx(&self, _: &str) -> Result<bool> {
        Ok(true)
    }
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        let cfg: crate::models::qwen3::Config = serde_json::from_str(config)?;

        Ok(Box::new(cfg))
    }
}

impl IsqModelLoader for Qwen3Loader {
    fn isq_layer_regexes(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(r"lm_head\.(weight|bias)$")?,
            // Attention
            Regex::new(r"layers\.(\d+)\.self_attn\.q_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.k_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.v_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.o_proj\.(weight|bias)$")?,
            // MLP
            Regex::new(r"layers\.(\d+)\.mlp\.gate_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.mlp\.up_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.mlp\.down_proj\.(weight|bias)$")?,
        ])
    }
    fn immediate_isq_predicates(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes(config)
    }
}

impl DeviceMappedModelLoader for Qwen3Loader {
    fn mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        let AutoDeviceMapParams::Text {
            max_seq_len,
            max_batch_size,
        } = params
        else {
            anyhow::bail!("Expected text AutoDeviceMapParams for this model!")
        };

        let cfg: models::qwen3::Config = serde_json::from_str(config)?;

        Ok(
            max_batch_size
                * cfg.num_attention_heads
                * max_seq_len.min(&ATTENTION_CHUNK_SIZE).pow(2),
        )
    }
    fn non_mapped_max_act_size_elems(
        &self,
        _config: &str,
        _params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        Ok(0)
    }

    fn non_mapped_size_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        let cfg: models::qwen3::Config = serde_json::from_str(config)?;
        let elems = {
            let embed_tokens = cfg.hidden_size * cfg.vocab_size / weight_pack_factor;
            // If embeddings are tied and no packing, reuse weights -> no separate lm_head needed
            let lm_head = if !cfg.tie_word_embeddings || weight_pack_factor != 1 {
                cfg.hidden_size * cfg.vocab_size / weight_pack_factor
            } else {
                0
            };
            let norm = cfg.hidden_size;
            embed_tokens + lm_head + norm
        };
        Ok(elems * dtype.size_in_bytes())
    }

    fn layer_sizes_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<Vec<usize>> {
        let cfg: models::qwen3::Config = serde_json::from_str(config)?;
        let per_layer_elems = {
            let input_layernorm = cfg.hidden_size;
            let post_attention_layernorm = cfg.hidden_size;

            let size_in = cfg.hidden_size;
            let size_q = cfg.head_dim() * cfg.num_attention_heads;
            let size_kv = cfg.head_dim() * cfg.num_key_value_heads;
            let q_proj = size_in * size_q / weight_pack_factor + size_q;
            let k_proj = size_in * size_kv / weight_pack_factor + size_kv;
            let v_proj = size_in * size_kv / weight_pack_factor + size_kv;
            let o_proj = size_q * size_in / weight_pack_factor;

            let h_size = cfg.hidden_size;
            let i_size = cfg.intermediate_size;
            let gate_proj = h_size * i_size / weight_pack_factor;
            let up_proj = h_size * i_size / weight_pack_factor;
            let down_proj = i_size * h_size / weight_pack_factor;

            let q_norm = cfg.head_dim();
            let k_norm = cfg.head_dim();

            input_layernorm
                + post_attention_layernorm
                + q_proj
                + k_proj
                + v_proj
                + o_proj
                + gate_proj
                + up_proj
                + down_proj
                + q_norm
                + k_norm
        };
        Ok(vec![
            per_layer_elems * dtype.size_in_bytes();
            cfg.num_hidden_layers
        ])
    }

    fn num_layers(&self, config: &str) -> Result<usize> {
        let cfg: models::qwen3::Config = serde_json::from_str(config)?;
        Ok(cfg.num_hidden_layers)
    }

    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg: models::qwen3::Config = serde_json::from_str(config)?;

        let cfg = ModelConfigMetadata {
            max_seq_len: cfg.max_position_embeddings,
            num_layers: cfg.num_hidden_layers,
            hidden_size: cfg.hidden_size,
            num_kv_heads: cfg.num_key_value_heads,
            num_attn_heads: cfg.num_attention_heads,
            sliding_window: cfg.sliding_window,
            k_head_dim: cfg.hidden_size / cfg.num_attention_heads,
            v_head_dim: cfg.hidden_size / cfg.num_attention_heads,
            kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
        };

        Ok(Box::new(cfg))
    }
}

/// [`NormalLoader`] for a GLM 4 model.
///
/// [`NormalLoader`]: https://docs.rs/mistralrs/latest/mistralrs/struct.NormalLoader.html
pub struct GLM4Loader;

impl NormalModelLoader for GLM4Loader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        let cfg: crate::models::glm4::Config = serde_json::from_str(config)?;

        Ok(Box::new(models::glm4::Model::new(
            &cfg,
            vb,
            self.is_gptx(config)?,
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }
    fn load_xlora(
        &self,
        _config: &str,
        _vb: ShardedVarBuilder,
        _lora_config: &[((String, String), LoraConfig)],
        _xlora_config: Option<XLoraConfig>,
        _xlora_ordering: Ordering,
        _normal_loading_metadata: NormalLoadingMetadata,
        _preload_adapters: &Option<HashMap<String, (ShardedVarBuilder, LoraConfig)>>,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        todo!()
    }
    fn is_gptx(&self, _: &str) -> Result<bool> {
        Ok(true)
    }
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        let cfg: crate::models::glm4::Config = serde_json::from_str(config)?;

        Ok(Box::new(cfg))
    }
}

impl IsqModelLoader for GLM4Loader {
    fn isq_layer_regexes(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(r"lm_head\.(weight|bias)$")?,
            // Attention
            Regex::new(r"layers\.(\d+)\.self_attn\.q_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.k_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.v_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.o_proj\.(weight|bias)$")?,
            // MLP
            Regex::new(r"layers\.(\d+)\.mlp\.gate_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.mlp\.up_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.mlp\.down_proj\.(weight|bias)$")?,
        ])
    }
    fn immediate_isq_predicates(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes(config)
    }
}

impl DeviceMappedModelLoader for GLM4Loader {
    fn mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        let AutoDeviceMapParams::Text {
            max_seq_len,
            max_batch_size,
        } = params
        else {
            anyhow::bail!("Expected text AutoDeviceMapParams for this model!")
        };

        let cfg: models::glm4::Config = serde_json::from_str(config)?;

        Ok(
            max_batch_size
                * cfg.num_attention_heads
                * max_seq_len.min(&ATTENTION_CHUNK_SIZE).pow(2),
        )
    }
    fn non_mapped_max_act_size_elems(
        &self,
        _config: &str,
        _params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        Ok(0)
    }

    fn non_mapped_size_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        let cfg: models::glm4::Config = serde_json::from_str(config)?;
        let elems = {
            let embed_tokens = cfg.hidden_size * cfg.vocab_size / weight_pack_factor;
            // If embeddings are tied and no packing, reuse weights -> no separate lm_head needed
            let lm_head = if !cfg.tie_word_embeddings || weight_pack_factor != 1 {
                cfg.hidden_size * cfg.vocab_size / weight_pack_factor
            } else {
                0
            };
            let norm = cfg.hidden_size;
            embed_tokens + lm_head + norm
        };
        Ok(elems * dtype.size_in_bytes())
    }

    fn layer_sizes_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<Vec<usize>> {
        let cfg: models::glm4::Config = serde_json::from_str(config)?;
        let per_layer_elems = {
            let input_layernorm = cfg.hidden_size;
            let post_attention_layernorm = cfg.hidden_size * 3; //+post_self_attn_layernorm and post_mlp_layernorm

            let size_in = cfg.hidden_size;
            let size_q = cfg.head_dim() * cfg.num_attention_heads;
            let size_kv = cfg.head_dim() * cfg.num_key_value_heads;
            let q_proj = size_in * size_q / weight_pack_factor + size_q;
            let k_proj = size_in * size_kv / weight_pack_factor + size_kv;
            let v_proj = size_in * size_kv / weight_pack_factor + size_kv;
            let o_proj = size_q * size_in / weight_pack_factor;

            let h_size = cfg.hidden_size;
            let i_size = cfg.intermediate_size;
            let gate_proj = h_size * i_size / weight_pack_factor;
            let up_proj = h_size * i_size / weight_pack_factor;
            let down_proj = i_size * h_size / weight_pack_factor;

            input_layernorm
                + post_attention_layernorm
                + q_proj
                + k_proj
                + v_proj
                + o_proj
                + gate_proj
                + up_proj
                + down_proj
        };
        Ok(vec![
            per_layer_elems * dtype.size_in_bytes();
            cfg.num_hidden_layers
        ])
    }

    fn num_layers(&self, config: &str) -> Result<usize> {
        let cfg: models::glm4::Config = serde_json::from_str(config)?;
        Ok(cfg.num_hidden_layers)
    }

    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg: models::glm4::Config = serde_json::from_str(config)?;

        let cfg = ModelConfigMetadata {
            max_seq_len: cfg.max_position_embeddings,
            num_layers: cfg.num_hidden_layers,
            hidden_size: cfg.hidden_size,
            num_kv_heads: cfg.num_key_value_heads,
            num_attn_heads: cfg.num_attention_heads,
            sliding_window: cfg.sliding_window,
            k_head_dim: cfg.hidden_size / cfg.num_attention_heads,
            v_head_dim: cfg.hidden_size / cfg.num_attention_heads,
            kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
        };

        Ok(Box::new(cfg))
    }
}

/// [`NormalLoader`] for a GLM 4 MoE Lite model (GLM-4.7-Flash).
///
/// [`NormalLoader`]: https://docs.rs/mistralrs/latest/mistralrs/struct.NormalLoader.html
pub struct GLM4MoeLiteLoader;

impl NormalModelLoader for GLM4MoeLiteLoader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        let cfg: crate::models::glm4_moe_lite::Glm4MoeLiteConfig = serde_json::from_str(config)?;
        Ok(Box::new(models::glm4_moe_lite::Glm4MoeLite::new(
            &cfg,
            vb,
            self.is_gptx(config)?,
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }
    fn load_xlora(
        &self,
        _config: &str,
        _vb: ShardedVarBuilder,
        _lora_config: &[((String, String), LoraConfig)],
        _xlora_config: Option<XLoraConfig>,
        _xlora_ordering: Ordering,
        _normal_loading_metadata: NormalLoadingMetadata,
        _preload_adapters: &Option<HashMap<String, (ShardedVarBuilder, LoraConfig)>>,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        todo!()
    }
    fn is_gptx(&self, _: &str) -> Result<bool> {
        Ok(true)
    }
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        let cfg: crate::models::glm4_moe_lite::Glm4MoeLiteConfig = serde_json::from_str(config)?;
        Ok(Box::new(cfg))
    }
}

impl IsqModelLoader for GLM4MoeLiteLoader {
    fn isq_layer_regexes(&self, config: &str) -> Result<Vec<Regex>> {
        let mut data = vec![
            Regex::new(r"lm_head\.(weight|bias)$")?,
            // Attention (MLA)
            Regex::new(r"layers\.(\d+)\.self_attn\.kv_a_proj_with_mqa\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.kv_b_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.o_proj\.(weight|bias)$")?,
            // Q LoRA projections
            Regex::new(r"layers\.(\d+)\.self_attn\.q_a_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.q_b_proj\.(weight|bias)$")?,
        ];
        let cfg: crate::models::glm4_moe_lite::Glm4MoeLiteConfig = serde_json::from_str(config)?;
        for layer_idx in 0..cfg.num_hidden_layers {
            if layer_idx >= cfg.first_k_dense_replace && layer_idx % cfg.moe_layer_freq == 0 {
                // MoE layer
                for i in 0..cfg.n_routed_experts {
                    data.extend(vec![
                        Regex::new(&format!(
                            r"layers\.{layer_idx}\.mlp\.experts\.{i}\.gate_proj\.(weight|bias)$"
                        ))?,
                        Regex::new(&format!(
                            r"layers\.{layer_idx}\.mlp\.experts\.{i}\.up_proj\.(weight|bias)$"
                        ))?,
                        Regex::new(&format!(
                            r"layers\.{layer_idx}\.mlp\.experts\.{i}\.down_proj\.(weight|bias)$"
                        ))?,
                    ]);
                }
                if cfg.n_shared_experts > 0 {
                    data.extend(vec![
                        Regex::new(&format!(
                            r"layers\.{layer_idx}\.mlp\.shared_experts\.gate_proj\.(weight|bias)$"
                        ))?,
                        Regex::new(&format!(
                            r"layers\.{layer_idx}\.mlp\.shared_experts\.up_proj\.(weight|bias)$"
                        ))?,
                        Regex::new(&format!(
                            r"layers\.{layer_idx}\.mlp\.shared_experts\.down_proj\.(weight|bias)$"
                        ))?,
                    ]);
                }
            } else {
                // Dense MLP layer
                data.extend(vec![
                    Regex::new(&format!(
                        r"layers\.{layer_idx}\.mlp\.gate_proj\.(weight|bias)$"
                    ))?,
                    Regex::new(&format!(
                        r"layers\.{layer_idx}\.mlp\.up_proj\.(weight|bias)$"
                    ))?,
                    Regex::new(&format!(
                        r"layers\.{layer_idx}\.mlp\.down_proj\.(weight|bias)$"
                    ))?,
                ]);
            };
        }
        Ok(data)
    }
    fn immediate_isq_predicates(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes(config)
    }

    fn isq_layer_regexes_moqe(&self, config: &str) -> Result<Vec<Regex>> {
        let mut data = vec![Regex::new(r"lm_head\.(weight|bias)$")?];
        let cfg: crate::models::glm4_moe_lite::Glm4MoeLiteConfig = serde_json::from_str(config)?;
        for layer_idx in 0..cfg.num_hidden_layers {
            if layer_idx >= cfg.first_k_dense_replace && layer_idx % cfg.moe_layer_freq == 0 {
                // MoE layer
                for i in 0..cfg.n_routed_experts {
                    data.extend(vec![
                        Regex::new(&format!(
                            r"layers\.{layer_idx}\.mlp\.experts\.{i}\.gate_proj\.(weight|bias)$"
                        ))?,
                        Regex::new(&format!(
                            r"layers\.{layer_idx}\.mlp\.experts\.{i}\.up_proj\.(weight|bias)$"
                        ))?,
                        Regex::new(&format!(
                            r"layers\.{layer_idx}\.mlp\.experts\.{i}\.down_proj\.(weight|bias)$"
                        ))?,
                    ]);
                }
                if cfg.n_shared_experts > 0 {
                    data.extend(vec![
                        Regex::new(&format!(
                            r"layers\.{layer_idx}\.mlp\.shared_experts\.gate_proj\.(weight|bias)$"
                        ))?,
                        Regex::new(&format!(
                            r"layers\.{layer_idx}\.mlp\.shared_experts\.up_proj\.(weight|bias)$"
                        ))?,
                        Regex::new(&format!(
                            r"layers\.{layer_idx}\.mlp\.shared_experts\.down_proj\.(weight|bias)$"
                        ))?,
                    ]);
                }
            } else {
                // Dense MLP layer
                data.extend(vec![
                    Regex::new(&format!(
                        r"layers\.{layer_idx}\.mlp\.gate_proj\.(weight|bias)$"
                    ))?,
                    Regex::new(&format!(
                        r"layers\.{layer_idx}\.mlp\.up_proj\.(weight|bias)$"
                    ))?,
                    Regex::new(&format!(
                        r"layers\.{layer_idx}\.mlp\.down_proj\.(weight|bias)$"
                    ))?,
                ]);
            };
        }
        Ok(data)
    }
    fn immediate_isq_predicates_moqe(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes_moqe(config)
    }
}

impl DeviceMappedModelLoader for GLM4MoeLiteLoader {
    fn mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        let AutoDeviceMapParams::Text {
            max_seq_len,
            max_batch_size,
        } = params
        else {
            anyhow::bail!("Expected text AutoDeviceMapParams for this model!")
        };

        let cfg: crate::models::glm4_moe_lite::Glm4MoeLiteConfig = serde_json::from_str(config)?;

        Ok(
            max_batch_size
                * cfg.num_attention_heads
                * max_seq_len.min(&ATTENTION_CHUNK_SIZE).pow(2),
        )
    }
    fn non_mapped_max_act_size_elems(
        &self,
        _config: &str,
        _params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        Ok(0)
    }

    fn non_mapped_size_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        let cfg: crate::models::glm4_moe_lite::Glm4MoeLiteConfig = serde_json::from_str(config)?;
        let elems = {
            let embed_tokens = cfg.hidden_size * cfg.vocab_size / weight_pack_factor;
            // If embeddings are tied and no packing, reuse weights -> no separate lm_head needed
            let lm_head = if !cfg.tie_word_embeddings || weight_pack_factor != 1 {
                cfg.hidden_size * cfg.vocab_size / weight_pack_factor
            } else {
                0
            };
            let norm = cfg.hidden_size;
            embed_tokens + lm_head + norm
        };
        Ok(elems * dtype.size_in_bytes())
    }

    fn layer_sizes_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<Vec<usize>> {
        let cfg: crate::models::glm4_moe_lite::Glm4MoeLiteConfig = serde_json::from_str(config)?;
        let mut per_layer_elems = Vec::new();

        for layer_idx in 0..cfg.num_hidden_layers {
            let input_layernorm = cfg.hidden_size;
            let post_attention_layernorm = cfg.hidden_size;

            // Q LoRA projection
            let q_proj = {
                let a = cfg.hidden_size * cfg.q_lora_rank / weight_pack_factor;
                let norm = cfg.q_lora_rank;
                let b = (cfg.num_attention_heads * cfg.q_head_dim()) * cfg.q_lora_rank
                    / weight_pack_factor;
                a + norm + b
            };
            let kv_a_proj_with_mqa =
                cfg.hidden_size * (cfg.kv_lora_rank + cfg.qk_rope_head_dim) / weight_pack_factor;
            let kv_a_layernorm = cfg.kv_lora_rank;
            let kv_b_proj = cfg.kv_lora_rank
                * cfg.num_attention_heads
                * (cfg.q_head_dim() - cfg.qk_rope_head_dim + cfg.v_head_dim)
                / weight_pack_factor;
            let o_proj =
                cfg.num_attention_heads * cfg.v_head_dim * cfg.hidden_size / weight_pack_factor;

            let moe_block = {
                let mut sum = 0;
                if layer_idx >= cfg.first_k_dense_replace && layer_idx % cfg.moe_layer_freq == 0 {
                    // MoE layer
                    let h_size = cfg.hidden_size;
                    let gate_proj = h_size * cfg.moe_intermediate_size / weight_pack_factor
                        * cfg.n_routed_experts;
                    let up_proj = h_size * cfg.moe_intermediate_size / weight_pack_factor
                        * cfg.n_routed_experts;
                    let down_proj = cfg.moe_intermediate_size * h_size / weight_pack_factor
                        * cfg.n_routed_experts;
                    let shared_experts = if cfg.n_shared_experts > 0 {
                        let gate_proj = h_size * cfg.moe_intermediate_size / weight_pack_factor;
                        let up_proj = h_size * cfg.moe_intermediate_size / weight_pack_factor;
                        let down_proj = cfg.moe_intermediate_size * h_size / weight_pack_factor;
                        gate_proj + up_proj + down_proj
                    } else {
                        0
                    };
                    let gate_weight = cfg.n_routed_experts * cfg.hidden_size;
                    let e_score_correction_bias = cfg.n_routed_experts;
                    sum += gate_proj
                        + up_proj
                        + down_proj
                        + shared_experts
                        + gate_weight
                        + e_score_correction_bias;
                } else {
                    // Dense MLP layer
                    let h_size = cfg.hidden_size;
                    let i_size = cfg.intermediate_size;
                    let gate_proj = h_size * i_size / weight_pack_factor;
                    let up_proj = h_size * i_size / weight_pack_factor;
                    let down_proj = i_size * h_size / weight_pack_factor;
                    sum += gate_proj + up_proj + down_proj;
                }
                sum
            };

            per_layer_elems.push(
                input_layernorm
                    + post_attention_layernorm
                    + q_proj
                    + kv_a_layernorm
                    + kv_a_proj_with_mqa
                    + kv_b_proj
                    + o_proj
                    + moe_block,
            );
        }

        Ok(per_layer_elems
            .into_iter()
            .map(|x| x * dtype.size_in_bytes())
            .collect())
    }

    fn num_layers(&self, config: &str) -> Result<usize> {
        let cfg: crate::models::glm4_moe_lite::Glm4MoeLiteConfig = serde_json::from_str(config)?;
        Ok(cfg.num_hidden_layers)
    }

    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg: crate::models::glm4_moe_lite::Glm4MoeLiteConfig = serde_json::from_str(config)?;

        let cfg = ModelConfigMetadata {
            max_seq_len: cfg.max_position_embeddings,
            num_layers: cfg.num_hidden_layers,
            hidden_size: cfg.hidden_size,
            num_kv_heads: cfg.num_attention_heads,
            num_attn_heads: cfg.num_attention_heads,
            sliding_window: None,
            k_head_dim: cfg.qk_rope_head_dim + cfg.qk_nope_head_dim,
            v_head_dim: cfg.v_head_dim,
            kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
        };

        Ok(Box::new(cfg))
    }
}

/// [`NormalLoader`] for a GLM 4 MoE model (GLM-4.5).
///
/// [`NormalLoader`]: https://docs.rs/mistralrs/latest/mistralrs/struct.NormalLoader.html
pub struct GLM4MoeLoader;

impl NormalModelLoader for GLM4MoeLoader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        let cfg: crate::models::glm4_moe::Glm4MoeConfig = serde_json::from_str(config)?;
        Ok(Box::new(models::glm4_moe::Glm4Moe::new(
            &cfg,
            vb,
            self.is_gptx(config)?,
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }
    fn load_xlora(
        &self,
        _config: &str,
        _vb: ShardedVarBuilder,
        _lora_config: &[((String, String), LoraConfig)],
        _xlora_config: Option<XLoraConfig>,
        _xlora_ordering: Ordering,
        _normal_loading_metadata: NormalLoadingMetadata,
        _preload_adapters: &Option<HashMap<String, (ShardedVarBuilder, LoraConfig)>>,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        todo!()
    }
    fn is_gptx(&self, _: &str) -> Result<bool> {
        Ok(true)
    }
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        let cfg: crate::models::glm4_moe::Glm4MoeConfig = serde_json::from_str(config)?;
        Ok(Box::new(cfg))
    }
}

impl IsqModelLoader for GLM4MoeLoader {
    fn isq_layer_regexes(&self, config: &str) -> Result<Vec<Regex>> {
        let mut data = vec![
            Regex::new(r"lm_head\.(weight|bias)$")?,
            // Attention (standard GQA)
            Regex::new(r"layers\.(\d+)\.self_attn\.q_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.k_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.v_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.o_proj\.(weight|bias)$")?,
        ];
        let cfg: crate::models::glm4_moe::Glm4MoeConfig = serde_json::from_str(config)?;
        for layer_idx in 0..cfg.num_hidden_layers {
            if layer_idx >= cfg.first_k_dense_replace {
                // MoE layer
                for i in 0..cfg.n_routed_experts {
                    data.extend(vec![
                        Regex::new(&format!(
                            r"layers\.{layer_idx}\.mlp\.experts\.{i}\.gate_proj\.(weight|bias)$"
                        ))?,
                        Regex::new(&format!(
                            r"layers\.{layer_idx}\.mlp\.experts\.{i}\.up_proj\.(weight|bias)$"
                        ))?,
                        Regex::new(&format!(
                            r"layers\.{layer_idx}\.mlp\.experts\.{i}\.down_proj\.(weight|bias)$"
                        ))?,
                    ]);
                }
                if cfg.n_shared_experts > 0 {
                    data.extend(vec![
                        Regex::new(&format!(
                            r"layers\.{layer_idx}\.mlp\.shared_experts\.gate_proj\.(weight|bias)$"
                        ))?,
                        Regex::new(&format!(
                            r"layers\.{layer_idx}\.mlp\.shared_experts\.up_proj\.(weight|bias)$"
                        ))?,
                        Regex::new(&format!(
                            r"layers\.{layer_idx}\.mlp\.shared_experts\.down_proj\.(weight|bias)$"
                        ))?,
                    ]);
                }
            } else {
                // Dense MLP layer
                data.extend(vec![
                    Regex::new(&format!(
                        r"layers\.{layer_idx}\.mlp\.gate_proj\.(weight|bias)$"
                    ))?,
                    Regex::new(&format!(
                        r"layers\.{layer_idx}\.mlp\.up_proj\.(weight|bias)$"
                    ))?,
                    Regex::new(&format!(
                        r"layers\.{layer_idx}\.mlp\.down_proj\.(weight|bias)$"
                    ))?,
                ]);
            };
        }
        Ok(data)
    }
    fn immediate_isq_predicates(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes(config)
    }

    fn isq_layer_regexes_moqe(&self, config: &str) -> Result<Vec<Regex>> {
        let mut data = vec![Regex::new(r"lm_head\.(weight|bias)$")?];
        let cfg: crate::models::glm4_moe::Glm4MoeConfig = serde_json::from_str(config)?;
        for layer_idx in 0..cfg.num_hidden_layers {
            if layer_idx >= cfg.first_k_dense_replace {
                // MoE layer
                for i in 0..cfg.n_routed_experts {
                    data.extend(vec![
                        Regex::new(&format!(
                            r"layers\.{layer_idx}\.mlp\.experts\.{i}\.gate_proj\.(weight|bias)$"
                        ))?,
                        Regex::new(&format!(
                            r"layers\.{layer_idx}\.mlp\.experts\.{i}\.up_proj\.(weight|bias)$"
                        ))?,
                        Regex::new(&format!(
                            r"layers\.{layer_idx}\.mlp\.experts\.{i}\.down_proj\.(weight|bias)$"
                        ))?,
                    ]);
                }
                if cfg.n_shared_experts > 0 {
                    data.extend(vec![
                        Regex::new(&format!(
                            r"layers\.{layer_idx}\.mlp\.shared_experts\.gate_proj\.(weight|bias)$"
                        ))?,
                        Regex::new(&format!(
                            r"layers\.{layer_idx}\.mlp\.shared_experts\.up_proj\.(weight|bias)$"
                        ))?,
                        Regex::new(&format!(
                            r"layers\.{layer_idx}\.mlp\.shared_experts\.down_proj\.(weight|bias)$"
                        ))?,
                    ]);
                }
            } else {
                // Dense MLP layer
                data.extend(vec![
                    Regex::new(&format!(
                        r"layers\.{layer_idx}\.mlp\.gate_proj\.(weight|bias)$"
                    ))?,
                    Regex::new(&format!(
                        r"layers\.{layer_idx}\.mlp\.up_proj\.(weight|bias)$"
                    ))?,
                    Regex::new(&format!(
                        r"layers\.{layer_idx}\.mlp\.down_proj\.(weight|bias)$"
                    ))?,
                ]);
            };
        }
        Ok(data)
    }
    fn immediate_isq_predicates_moqe(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes_moqe(config)
    }
}

impl DeviceMappedModelLoader for GLM4MoeLoader {
    fn mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        let AutoDeviceMapParams::Text {
            max_seq_len,
            max_batch_size,
        } = params
        else {
            anyhow::bail!("Expected text AutoDeviceMapParams for this model!")
        };

        let cfg: crate::models::glm4_moe::Glm4MoeConfig = serde_json::from_str(config)?;

        Ok(
            max_batch_size
                * cfg.num_attention_heads
                * max_seq_len.min(&ATTENTION_CHUNK_SIZE).pow(2),
        )
    }
    fn non_mapped_max_act_size_elems(
        &self,
        _config: &str,
        _params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        Ok(0)
    }

    fn non_mapped_size_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        let cfg: crate::models::glm4_moe::Glm4MoeConfig = serde_json::from_str(config)?;
        let elems = {
            let embed_tokens = cfg.hidden_size * cfg.vocab_size / weight_pack_factor;
            let lm_head = if !cfg.tie_word_embeddings || weight_pack_factor != 1 {
                cfg.hidden_size * cfg.vocab_size / weight_pack_factor
            } else {
                0
            };
            let norm = cfg.hidden_size;
            embed_tokens + lm_head + norm
        };
        Ok(elems * dtype.size_in_bytes())
    }

    fn layer_sizes_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<Vec<usize>> {
        let cfg: crate::models::glm4_moe::Glm4MoeConfig = serde_json::from_str(config)?;
        let mut per_layer_elems = Vec::new();

        let head_dim = cfg.head_dim();
        for layer_idx in 0..cfg.num_hidden_layers {
            let input_layernorm = cfg.hidden_size;
            let post_attention_layernorm = cfg.hidden_size;

            // Standard GQA attention
            let q_proj = cfg.hidden_size * cfg.num_attention_heads * head_dim / weight_pack_factor
                + bias_if!(cfg.attention_bias, cfg.num_attention_heads * head_dim);
            let k_proj = cfg.hidden_size * cfg.num_key_value_heads * head_dim / weight_pack_factor
                + bias_if!(cfg.attention_bias, cfg.num_key_value_heads * head_dim);
            let v_proj = cfg.hidden_size * cfg.num_key_value_heads * head_dim / weight_pack_factor
                + bias_if!(cfg.attention_bias, cfg.num_key_value_heads * head_dim);
            let o_proj = cfg.num_attention_heads * head_dim * cfg.hidden_size / weight_pack_factor;

            // QK norm if enabled
            let qk_norm = if cfg.use_qk_norm {
                head_dim * 2 // q_norm + k_norm
            } else {
                0
            };

            let moe_block = {
                let mut sum = 0;
                if layer_idx >= cfg.first_k_dense_replace {
                    // MoE layer
                    let h_size = cfg.hidden_size;
                    let gate_proj = h_size * cfg.moe_intermediate_size / weight_pack_factor
                        * cfg.n_routed_experts;
                    let up_proj = h_size * cfg.moe_intermediate_size / weight_pack_factor
                        * cfg.n_routed_experts;
                    let down_proj = cfg.moe_intermediate_size * h_size / weight_pack_factor
                        * cfg.n_routed_experts;
                    let shared_experts = if cfg.n_shared_experts > 0 {
                        let gate_proj = h_size * cfg.moe_intermediate_size / weight_pack_factor;
                        let up_proj = h_size * cfg.moe_intermediate_size / weight_pack_factor;
                        let down_proj = cfg.moe_intermediate_size * h_size / weight_pack_factor;
                        gate_proj + up_proj + down_proj
                    } else {
                        0
                    };
                    let gate_weight = cfg.n_routed_experts * cfg.hidden_size;
                    let e_score_correction_bias = cfg.n_routed_experts;
                    sum += gate_proj
                        + up_proj
                        + down_proj
                        + shared_experts
                        + gate_weight
                        + e_score_correction_bias;
                } else {
                    // Dense MLP layer
                    let h_size = cfg.hidden_size;
                    let i_size = cfg.intermediate_size;
                    let gate_proj = h_size * i_size / weight_pack_factor;
                    let up_proj = h_size * i_size / weight_pack_factor;
                    let down_proj = i_size * h_size / weight_pack_factor;
                    sum += gate_proj + up_proj + down_proj;
                }
                sum
            };

            per_layer_elems.push(
                input_layernorm
                    + post_attention_layernorm
                    + q_proj
                    + k_proj
                    + v_proj
                    + o_proj
                    + qk_norm
                    + moe_block,
            );
        }

        Ok(per_layer_elems
            .into_iter()
            .map(|x| x * dtype.size_in_bytes())
            .collect())
    }

    fn num_layers(&self, config: &str) -> Result<usize> {
        let cfg: crate::models::glm4_moe::Glm4MoeConfig = serde_json::from_str(config)?;
        Ok(cfg.num_hidden_layers)
    }

    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg: crate::models::glm4_moe::Glm4MoeConfig = serde_json::from_str(config)?;

        let head_dim = cfg.head_dim();
        let cfg = ModelConfigMetadata {
            max_seq_len: cfg.max_position_embeddings,
            num_layers: cfg.num_hidden_layers,
            hidden_size: cfg.hidden_size,
            num_kv_heads: cfg.num_key_value_heads,
            num_attn_heads: cfg.num_attention_heads,
            sliding_window: None,
            k_head_dim: head_dim,
            v_head_dim: head_dim,
            kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
        };

        Ok(Box::new(cfg))
    }
}

/// [`NormalLoader`] for a Qwen 3 MoE model.
///
/// [`NormalLoader`]: https://docs.rs/mistralrs/latest/mistralrs/struct.NormalLoader.html
pub struct Qwen3MoELoader;

impl NormalModelLoader for Qwen3MoELoader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        let cfg: crate::models::qwen3_moe::Config = serde_json::from_str(config)?;

        Ok(Box::new(models::qwen3_moe::Model::new(
            &cfg,
            vb,
            self.is_gptx(config)?,
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }
    fn load_xlora(
        &self,
        _config: &str,
        _vb: ShardedVarBuilder,
        _lora_config: &[((String, String), LoraConfig)],
        _xlora_config: Option<XLoraConfig>,
        _xlora_ordering: Ordering,
        _normal_loading_metadata: NormalLoadingMetadata,
        _preload_adapters: &Option<HashMap<String, (ShardedVarBuilder, LoraConfig)>>,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        todo!()
    }
    fn is_gptx(&self, _: &str) -> Result<bool> {
        Ok(true)
    }
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        let cfg: crate::models::qwen3_moe::Config = serde_json::from_str(config)?;

        Ok(Box::new(cfg))
    }
}

impl IsqModelLoader for Qwen3MoELoader {
    fn isq_layer_regexes(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(r"lm_head\.(weight|bias)$")?,
            // Attention
            Regex::new(r"layers\.(\d+)\.self_attn\.q_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.k_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.v_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.o_proj\.(weight|bias)$")?,
            // MLP
            Regex::new(r"layers\.(\d+)\.mlp\.gate_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.mlp\.up_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.mlp\.down_proj\.(weight|bias)$")?,
            // MLP MoE
            Regex::new(r"layers\.(\d+)\.mlp\.experts\.(\d+)\.gate_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.mlp\.experts\.(\d+)\.up_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.mlp\.experts\.(\d+)\.down_proj\.(weight|bias)$")?,
        ])
    }
    fn immediate_isq_predicates(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes(config)
    }
    fn immediate_isq_predicates_moqe(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes_moqe(config)
    }
}

impl DeviceMappedModelLoader for Qwen3MoELoader {
    fn mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        let AutoDeviceMapParams::Text {
            max_seq_len,
            max_batch_size,
        } = params
        else {
            anyhow::bail!("Expected text AutoDeviceMapParams for this model!")
        };

        let cfg: models::qwen3_moe::Config = serde_json::from_str(config)?;

        Ok(
            max_batch_size
                * cfg.num_attention_heads
                * max_seq_len.min(&ATTENTION_CHUNK_SIZE).pow(2),
        )
    }
    fn non_mapped_max_act_size_elems(
        &self,
        _config: &str,
        _params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        Ok(0)
    }

    fn non_mapped_size_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        let cfg: models::qwen3_moe::Config = serde_json::from_str(config)?;
        let elems = {
            let embed_tokens = cfg.hidden_size * cfg.vocab_size;
            // If embeddings are tied and no packing, reuse weights -> no separate lm_head needed
            let lm_head = if !cfg.tie_word_embeddings || weight_pack_factor != 1 {
                cfg.hidden_size * cfg.vocab_size / weight_pack_factor
            } else {
                0
            };
            let norm = cfg.hidden_size;
            embed_tokens + lm_head + norm
        };
        Ok(elems * dtype.size_in_bytes())
    }

    fn layer_sizes_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<Vec<usize>> {
        let cfg: models::qwen3_moe::Config = serde_json::from_str(config)?;

        let mut layer_sizes_in_bytes = Vec::new();
        for layer_idx in 0..cfg.num_hidden_layers {
            let input_layernorm = cfg.hidden_size;
            let post_attention_layernorm = cfg.hidden_size;

            let size_in = cfg.hidden_size;
            let size_q = cfg.head_dim() * cfg.num_attention_heads;
            let size_kv = cfg.head_dim() * cfg.num_key_value_heads;
            let q_proj = size_in * size_q / weight_pack_factor;
            let k_proj = size_in * size_kv / weight_pack_factor;
            let v_proj = size_in * size_kv / weight_pack_factor;
            let o_proj = size_q * size_in / weight_pack_factor;

            let mlp_size = if !cfg.mlp_only_layers.contains(&layer_idx)
                && (cfg.num_experts > 0 && (layer_idx + 1) % cfg.decoder_sparse_step == 0)
            {
                let gate_size = cfg.hidden_size * cfg.num_experts;
                let expert_size = {
                    let h_size = cfg.hidden_size;
                    let i_size = cfg.moe_intermediate_size;
                    let gate_proj = h_size * i_size / weight_pack_factor;
                    let up_proj = h_size * i_size / weight_pack_factor;
                    let down_proj = i_size * h_size / weight_pack_factor;
                    gate_proj + up_proj + down_proj
                };
                expert_size * cfg.num_experts + gate_size
            } else {
                let h_size = cfg.hidden_size;
                let i_size = cfg.intermediate_size;
                let gate_proj = h_size * i_size / weight_pack_factor;
                let up_proj = h_size * i_size / weight_pack_factor;
                let down_proj = i_size * h_size / weight_pack_factor;
                gate_proj + up_proj + down_proj
            };

            let q_norm = cfg.head_dim();
            let k_norm = cfg.head_dim();

            let size_elems = input_layernorm
                + post_attention_layernorm
                + q_proj
                + k_proj
                + v_proj
                + o_proj
                + mlp_size
                + q_norm
                + k_norm;

            let size_in_bytes = size_elems * dtype.size_in_bytes();
            layer_sizes_in_bytes.push(size_in_bytes);
        }

        Ok(layer_sizes_in_bytes)
    }

    fn num_layers(&self, config: &str) -> Result<usize> {
        let cfg: models::qwen3_moe::Config = serde_json::from_str(config)?;
        Ok(cfg.num_hidden_layers)
    }

    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg: models::qwen3_moe::Config = serde_json::from_str(config)?;

        let cfg = ModelConfigMetadata {
            max_seq_len: cfg.max_position_embeddings,
            num_layers: cfg.num_hidden_layers,
            hidden_size: cfg.hidden_size,
            num_kv_heads: cfg.num_key_value_heads,
            num_attn_heads: cfg.num_attention_heads,
            sliding_window: cfg.sliding_window,
            k_head_dim: cfg.hidden_size / cfg.num_attention_heads,
            v_head_dim: cfg.hidden_size / cfg.num_attention_heads,
            kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
        };

        Ok(Box::new(cfg))
    }
}

// ======================== SmolLm3 loader

/// [`NormalLoader`] for a SmolLm3 model.
///
/// [`NormalLoader`]: https://docs.rs/mistralrs/latest/mistralrs/struct.NormalLoader.html
pub struct SmolLm3Loader;

impl NormalModelLoader for SmolLm3Loader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        let cfg: crate::models::smollm3::Config = serde_json::from_str(config)?;

        Ok(Box::new(models::smollm3::SmolLm3::new(
            &cfg,
            vb,
            self.is_gptx(config)?,
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }
    fn load_xlora(
        &self,
        _config: &str,
        _vb: ShardedVarBuilder,
        _lora_config: &[((String, String), LoraConfig)],
        _xlora_config: Option<XLoraConfig>,
        _xlora_ordering: Ordering,
        _normal_loading_metadata: NormalLoadingMetadata,
        _preload_adapters: &Option<HashMap<String, (ShardedVarBuilder, LoraConfig)>>,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        todo!()
    }
    fn is_gptx(&self, _: &str) -> Result<bool> {
        Ok(true)
    }
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        let cfg: crate::models::smollm3::Config = serde_json::from_str(config)?;
        Ok(Box::new(cfg))
    }
}

impl IsqModelLoader for SmolLm3Loader {
    fn isq_layer_regexes(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(r"lm_head\.(weight|bias)$")?,
            // Attention
            Regex::new(r"layers\.(\d+)\.self_attn\.q_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.k_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.v_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.o_proj\.(weight|bias)$")?,
            // MLP
            Regex::new(r"layers\.(\d+)\.mlp\.gate_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.mlp\.up_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.mlp\.down_proj\.(weight|bias)$")?,
        ])
    }
    fn immediate_isq_predicates(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes(config)
    }
}

impl DeviceMappedModelLoader for SmolLm3Loader {
    fn mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        let AutoDeviceMapParams::Text {
            max_seq_len,
            max_batch_size,
        } = params
        else {
            anyhow::bail!("Expected text AutoDeviceMapParams for this model!")
        };

        let cfg: crate::models::smollm3::Config = serde_json::from_str(config)?;

        Ok(
            max_batch_size
                * cfg.num_attention_heads
                * max_seq_len.min(&ATTENTION_CHUNK_SIZE).pow(2),
        )
    }
    fn non_mapped_max_act_size_elems(
        &self,
        _config: &str,
        _params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        Ok(0)
    }

    fn non_mapped_size_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        let cfg: crate::models::smollm3::Config = serde_json::from_str(config)?;

        let elems = {
            let embed_tokens = cfg.hidden_size * cfg.vocab_size / weight_pack_factor;
            // If embeddings are tied and no packing, reuse weights -> no separate lm_head needed
            let lm_head = if !cfg.tie_word_embeddings || weight_pack_factor != 1 {
                cfg.hidden_size * cfg.vocab_size / weight_pack_factor
            } else {
                0
            };
            let norm = cfg.hidden_size;
            embed_tokens + lm_head + norm
        };
        Ok(elems * dtype.size_in_bytes())
    }

    fn layer_sizes_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<Vec<usize>> {
        let cfg: crate::models::smollm3::Config = serde_json::from_str(config)?;

        let per_layer_elems = {
            let input_layernorm = cfg.hidden_size;
            let post_attention_layernorm = cfg.hidden_size;

            let size_in = cfg.hidden_size;
            let size_q = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_attention_heads;
            let size_kv = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_key_value_heads;
            let q_proj = size_in * size_q / weight_pack_factor;
            let k_proj = size_in * size_kv / weight_pack_factor;
            let v_proj = size_in * size_kv / weight_pack_factor;
            let o_proj = size_q * size_in / weight_pack_factor;

            let h_size = cfg.hidden_size;
            let i_size = cfg.intermediate_size;
            let gate_proj = h_size * i_size / weight_pack_factor;
            let up_proj = h_size * i_size / weight_pack_factor;
            let down_proj = i_size * h_size / weight_pack_factor;

            input_layernorm
                + post_attention_layernorm
                + q_proj
                + k_proj
                + v_proj
                + o_proj
                + gate_proj
                + up_proj
                + down_proj
        };
        Ok(vec![
            per_layer_elems * dtype.size_in_bytes();
            cfg.num_hidden_layers
        ])
    }

    fn num_layers(&self, config: &str) -> Result<usize> {
        let cfg: crate::models::smollm3::Config = serde_json::from_str(config)?;

        Ok(cfg.num_hidden_layers)
    }
    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg: crate::models::smollm3::Config = serde_json::from_str(config)?;

        let cfg = ModelConfigMetadata {
            max_seq_len: cfg.max_position_embeddings,
            num_layers: cfg.num_hidden_layers,
            hidden_size: cfg.hidden_size,
            num_kv_heads: cfg.num_key_value_heads,
            num_attn_heads: cfg.num_attention_heads,
            sliding_window: None,
            k_head_dim: cfg.hidden_size / cfg.num_attention_heads,
            v_head_dim: cfg.hidden_size / cfg.num_attention_heads,
            kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
        };

        Ok(Box::new(cfg))
    }
}

// ======================== GraniteMoeHybrid loader

/// [`NormalLoader`] for a GraniteMoeHybrid model (IBM Granite 4.0).
///
/// [`NormalLoader`]: https://docs.rs/mistralrs/latest/mistralrs/struct.NormalLoader.html
pub struct GraniteMoeHybridLoader;

impl NormalModelLoader for GraniteMoeHybridLoader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        let cfg: crate::models::granite::Config = serde_json::from_str(config)?;

        Ok(Box::new(models::granite::GraniteMoeHybrid::new(
            &cfg,
            vb,
            self.is_gptx(config)?,
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }
    fn load_xlora(
        &self,
        _config: &str,
        _vb: ShardedVarBuilder,
        _lora_config: &[((String, String), LoraConfig)],
        _xlora_config: Option<XLoraConfig>,
        _xlora_ordering: Ordering,
        _normal_loading_metadata: NormalLoadingMetadata,
        _preload_adapters: &Option<HashMap<String, (ShardedVarBuilder, LoraConfig)>>,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        anyhow::bail!("GraniteMoeHybrid does not support X-LoRA")
    }
    fn is_gptx(&self, _: &str) -> Result<bool> {
        Ok(true)
    }
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        let cfg: crate::models::granite::Config = serde_json::from_str(config)?;
        Ok(Box::new(cfg))
    }
    fn supports_paged_attention(&self, _config: &str) -> Result<bool> {
        Ok(true)
    }
}

impl IsqModelLoader for GraniteMoeHybridLoader {
    fn isq_layer_regexes(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(r"lm_head\.(weight|bias)$")?,
            // Attention
            Regex::new(r"layers\.(\d+)\.self_attn\.q_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.k_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.v_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.o_proj\.(weight|bias)$")?,
            // MLP (GraniteMLP uses shared_mlp.input_linear and shared_mlp.output_linear)
            Regex::new(r"layers\.(\d+)\.shared_mlp\.input_linear\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.shared_mlp\.output_linear\.(weight|bias)$")?,
        ])
    }
    fn immediate_isq_predicates(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes(config)
    }
}

impl DeviceMappedModelLoader for GraniteMoeHybridLoader {
    fn mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        let AutoDeviceMapParams::Text {
            max_seq_len,
            max_batch_size,
        } = params
        else {
            anyhow::bail!("Expected text AutoDeviceMapParams for this model!")
        };

        let cfg: crate::models::granite::Config = serde_json::from_str(config)?;

        Ok(
            max_batch_size
                * cfg.num_attention_heads
                * max_seq_len.min(&ATTENTION_CHUNK_SIZE).pow(2),
        )
    }
    fn non_mapped_max_act_size_elems(
        &self,
        _config: &str,
        _params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        Ok(0)
    }

    fn non_mapped_size_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        let cfg: crate::models::granite::Config = serde_json::from_str(config)?;

        let elems = {
            let embed_tokens = cfg.hidden_size * cfg.vocab_size / weight_pack_factor;
            // If embeddings are tied and no packing, reuse weights -> no separate lm_head needed
            let lm_head = if !cfg.tie_word_embeddings || weight_pack_factor != 1 {
                cfg.hidden_size * cfg.vocab_size / weight_pack_factor
            } else {
                0
            };
            let norm = cfg.hidden_size;
            embed_tokens + lm_head + norm
        };
        Ok(elems * dtype.size_in_bytes())
    }

    fn layer_sizes_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<Vec<usize>> {
        let cfg: crate::models::granite::Config = serde_json::from_str(config)?;

        let per_layer_elems = {
            let input_layernorm = cfg.hidden_size;
            let post_attention_layernorm = cfg.hidden_size;

            let size_in = cfg.hidden_size;
            let size_q = cfg.head_dim() * cfg.num_attention_heads;
            let size_kv = cfg.head_dim() * cfg.num_key_value_heads();
            let q_proj = size_in * size_q / weight_pack_factor;
            let k_proj = size_in * size_kv / weight_pack_factor;
            let v_proj = size_in * size_kv / weight_pack_factor;
            let o_proj = size_q * size_in / weight_pack_factor;

            let h_size = cfg.hidden_size;
            let shared_i_size = cfg.shared_intermediate_size();
            // GraniteMLP: input_linear (h_size -> shared_i_size * 2), output_linear (shared_i_size -> h_size)
            let input_linear = h_size * shared_i_size * 2 / weight_pack_factor;
            let output_linear = shared_i_size * h_size / weight_pack_factor;

            input_layernorm
                + post_attention_layernorm
                + q_proj
                + k_proj
                + v_proj
                + o_proj
                + input_linear
                + output_linear
        };
        Ok(vec![
            per_layer_elems * dtype.size_in_bytes();
            cfg.num_hidden_layers
        ])
    }

    fn num_layers(&self, config: &str) -> Result<usize> {
        let cfg: crate::models::granite::Config = serde_json::from_str(config)?;

        Ok(cfg.num_hidden_layers)
    }
    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg: crate::models::granite::Config = serde_json::from_str(config)?;

        let cfg = ModelConfigMetadata {
            max_seq_len: cfg.max_position_embeddings,
            num_layers: cfg.num_hidden_layers,
            hidden_size: cfg.hidden_size,
            num_kv_heads: cfg.num_key_value_heads(),
            num_attn_heads: cfg.num_attention_heads,
            sliding_window: None,
            k_head_dim: cfg.head_dim(),
            v_head_dim: cfg.head_dim(),
            kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
        };

        Ok(Box::new(cfg))
    }
}

// ======================== GPT-OSS loader

/// [`NormalLoader`] for a GPT-OSS model.
///
/// [`NormalLoader`]: https://docs.rs/mistralrs/latest/mistralrs/struct.NormalLoader.html
pub struct GptOssLoader;

impl NormalModelLoader for GptOssLoader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        let cfg: crate::models::gpt_oss::Config = serde_json::from_str(config)?;

        Ok(Box::new(models::gpt_oss::Model::new(
            &cfg,
            vb,
            self.is_gptx(config)?,
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }
    fn load_xlora(
        &self,
        _config: &str,
        _vb: ShardedVarBuilder,
        _lora_config: &[((String, String), LoraConfig)],
        _xlora_config: Option<XLoraConfig>,
        _xlora_ordering: Ordering,
        _normal_loading_metadata: NormalLoadingMetadata,
        _preload_adapters: &Option<HashMap<String, (ShardedVarBuilder, LoraConfig)>>,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        anyhow::bail!("GPT-OSS does not support X-LoRA")
    }
    fn is_gptx(&self, _: &str) -> Result<bool> {
        Ok(true)
    }
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        let cfg: crate::models::gpt_oss::Config = serde_json::from_str(config)?;
        Ok(Box::new(cfg))
    }
}

impl IsqModelLoader for GptOssLoader {
    fn isq_layer_regexes(&self, _config: &str) -> Result<Vec<Regex>> {
        // Only attention layers are ISQ-able - MoE experts are already MXFP4 quantized
        Ok(vec![
            Regex::new(r"lm_head\.(weight|bias)$")?,
            // Attention
            Regex::new(r"layers\.(\d+)\.self_attn\.q_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.k_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.v_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.o_proj\.(weight|bias)$")?,
        ])
    }
    fn immediate_isq_predicates(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes(config)
    }
}

impl DeviceMappedModelLoader for GptOssLoader {
    fn mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        let AutoDeviceMapParams::Text {
            max_seq_len,
            max_batch_size,
        } = params
        else {
            anyhow::bail!("Expected text AutoDeviceMapParams for this model!")
        };

        let cfg: crate::models::gpt_oss::Config = serde_json::from_str(config)?;

        Ok(
            max_batch_size
                * cfg.num_attention_heads
                * max_seq_len.min(&ATTENTION_CHUNK_SIZE).pow(2),
        )
    }
    fn non_mapped_max_act_size_elems(
        &self,
        _config: &str,
        _params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        Ok(0)
    }

    fn non_mapped_size_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        let cfg: crate::models::gpt_oss::Config = serde_json::from_str(config)?;

        let elems = {
            let embed_tokens = cfg.hidden_size * cfg.vocab_size / weight_pack_factor;
            let lm_head = if !cfg.tie_word_embeddings || weight_pack_factor != 1 {
                cfg.hidden_size * cfg.vocab_size / weight_pack_factor
            } else {
                0
            };
            let norm = cfg.hidden_size;
            embed_tokens + lm_head + norm
        };
        Ok(elems * dtype.size_in_bytes())
    }

    fn layer_sizes_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<Vec<usize>> {
        let cfg: crate::models::gpt_oss::Config = serde_json::from_str(config)?;

        let per_layer_elems = {
            let input_layernorm = cfg.hidden_size;
            let post_attention_layernorm = cfg.hidden_size;

            let size_in = cfg.hidden_size;
            let head_dim = cfg.head_dim();
            let size_q = head_dim * cfg.num_attention_heads;
            let size_kv = head_dim * cfg.num_key_value_heads;
            let q_proj =
                size_in * size_q / weight_pack_factor + bias_if!(cfg.attention_bias, size_q);
            let k_proj =
                size_in * size_kv / weight_pack_factor + bias_if!(cfg.attention_bias, size_kv);
            let v_proj =
                size_in * size_kv / weight_pack_factor + bias_if!(cfg.attention_bias, size_kv);
            let o_proj =
                size_q * size_in / weight_pack_factor + bias_if!(cfg.attention_bias, size_in);

            // MoE experts - MXFP4 quantized, so very compact
            // gate_up_proj: [num_experts, intermediate_size * 2, hidden_size/2] packed
            // down_proj: [num_experts, hidden_size, intermediate_size/2] packed
            // At 4 bits per weight, packing factor is 2
            let mxfp4_pack = 2;
            let gate_up_proj_size =
                cfg.num_local_experts * cfg.intermediate_size * 2 * cfg.hidden_size / mxfp4_pack;
            let down_proj_size =
                cfg.num_local_experts * cfg.hidden_size * cfg.intermediate_size / mxfp4_pack;
            // Plus scales at 1 byte per 32 elements
            let gate_up_scales =
                cfg.num_local_experts * cfg.intermediate_size * 2 * cfg.hidden_size / 32;
            let down_scales = cfg.num_local_experts * cfg.hidden_size * cfg.intermediate_size / 32;
            // Plus biases
            let gate_up_bias = cfg.num_local_experts * cfg.intermediate_size * 2;
            let down_bias = cfg.num_local_experts * cfg.hidden_size;
            // Router
            let router = cfg.hidden_size * cfg.num_local_experts;
            // Sinks per head
            let sinks = cfg.num_attention_heads;

            input_layernorm
                + post_attention_layernorm
                + q_proj
                + k_proj
                + v_proj
                + o_proj
                + gate_up_proj_size
                + down_proj_size
                + gate_up_scales
                + down_scales
                + gate_up_bias
                + down_bias
                + router
                + sinks
        };
        Ok(vec![
            per_layer_elems * dtype.size_in_bytes();
            cfg.num_hidden_layers
        ])
    }

    fn num_layers(&self, config: &str) -> Result<usize> {
        let cfg: crate::models::gpt_oss::Config = serde_json::from_str(config)?;

        Ok(cfg.num_hidden_layers)
    }
    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg: crate::models::gpt_oss::Config = serde_json::from_str(config)?;

        let head_dim = cfg.head_dim();
        let cfg = ModelConfigMetadata {
            max_seq_len: cfg.max_position_embeddings,
            num_layers: cfg.num_hidden_layers,
            hidden_size: cfg.hidden_size,
            num_kv_heads: cfg.num_key_value_heads,
            num_attn_heads: cfg.num_attention_heads,
            sliding_window: cfg.sliding_window,
            k_head_dim: head_dim,
            v_head_dim: head_dim,
            kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
        };

        Ok(Box::new(cfg))
    }
}

// ======================== Qwen3Next loader

/// [`NormalLoader`] for a Qwen3Next (Qwen3-Coder-Next) model.
///
/// [`NormalLoader`]: https://docs.rs/mistralrs/latest/mistralrs/struct.NormalLoader.html
pub struct Qwen3NextLoader;

impl NormalModelLoader for Qwen3NextLoader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        let cfg: crate::models::qwen3_next::Config = serde_json::from_str(config)?;

        Ok(Box::new(models::qwen3_next::Model::new(
            &cfg,
            vb,
            self.is_gptx(config)?,
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }
    fn load_xlora(
        &self,
        _config: &str,
        _vb: ShardedVarBuilder,
        _lora_config: &[((String, String), LoraConfig)],
        _xlora_config: Option<XLoraConfig>,
        _xlora_ordering: Ordering,
        _normal_loading_metadata: NormalLoadingMetadata,
        _preload_adapters: &Option<HashMap<String, (ShardedVarBuilder, LoraConfig)>>,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        anyhow::bail!("Qwen3Next does not support X-LoRA")
    }
    fn is_gptx(&self, _: &str) -> Result<bool> {
        Ok(true)
    }
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        let cfg: crate::models::qwen3_next::Config = serde_json::from_str(config)?;
        Ok(Box::new(cfg))
    }
    fn supports_paged_attention(&self, _config: &str) -> Result<bool> {
        Ok(true)
    }
}

impl IsqModelLoader for Qwen3NextLoader {
    fn isq_layer_regexes(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(r"lm_head\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.q_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.k_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.v_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.o_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.linear_attn\.out_proj\.(weight|bias)$")?,
            Regex::new(
                r"layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.(weight|bias)$",
            )?,
            Regex::new(
                r"layers\.(\d+)\.mlp\.shared_expert\.(gate_proj|up_proj|down_proj)\.(weight|bias)$",
            )?,
        ])
    }
    fn immediate_isq_predicates(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes(config)
    }
}

impl DeviceMappedModelLoader for Qwen3NextLoader {
    fn mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        let AutoDeviceMapParams::Text {
            max_seq_len,
            max_batch_size,
        } = params
        else {
            anyhow::bail!("Expected text AutoDeviceMapParams for this model!")
        };

        let cfg: crate::models::qwen3_next::Config = serde_json::from_str(config)?;

        Ok(
            max_batch_size
                * cfg.num_attention_heads
                * max_seq_len.min(&ATTENTION_CHUNK_SIZE).pow(2),
        )
    }
    fn non_mapped_max_act_size_elems(
        &self,
        _config: &str,
        _params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        Ok(0)
    }

    fn non_mapped_size_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        let cfg: crate::models::qwen3_next::Config = serde_json::from_str(config)?;

        let elems = {
            let embed_tokens = cfg.hidden_size * cfg.vocab_size / weight_pack_factor;
            let lm_head = if !cfg.tie_word_embeddings || weight_pack_factor != 1 {
                cfg.hidden_size * cfg.vocab_size / weight_pack_factor
            } else {
                0
            };
            let norm = cfg.hidden_size;
            embed_tokens + lm_head + norm
        };
        Ok(elems * dtype.size_in_bytes())
    }

    fn layer_sizes_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<Vec<usize>> {
        let cfg: crate::models::qwen3_next::Config = serde_json::from_str(config)?;
        let layer_types = cfg.layer_types();
        let mut layer_sizes = Vec::with_capacity(cfg.num_hidden_layers);

        for layer_type in &layer_types {
            let input_layernorm = cfg.hidden_size;
            let post_attention_layernorm = cfg.hidden_size;

            let attn_elems = match layer_type {
                crate::models::qwen3_next::LayerType::FullAttention => {
                    let hidden = cfg.hidden_size;
                    let q_dim = cfg.head_dim * cfg.num_attention_heads;
                    let kv_dim = cfg.head_dim * cfg.num_key_value_heads;
                    let q_proj = hidden * q_dim * 2 / weight_pack_factor;
                    let k_proj = hidden * kv_dim / weight_pack_factor;
                    let v_proj = hidden * kv_dim / weight_pack_factor;
                    let o_proj = q_dim * hidden / weight_pack_factor;
                    let q_norm = cfg.head_dim;
                    let k_norm = cfg.head_dim;
                    q_proj + k_proj + v_proj + o_proj + q_norm + k_norm
                }
                crate::models::qwen3_next::LayerType::LinearAttention => {
                    let hidden = cfg.hidden_size;
                    let key_dim = cfg.linear_key_dim();
                    let value_dim = cfg.linear_value_dim();
                    let conv_dim = cfg.linear_conv_dim();
                    // in_proj_qkvz: (2 * key_dim + 2 * value_dim, hidden)
                    let in_proj_qkvz = hidden * (key_dim * 2 + value_dim * 2);
                    // in_proj_ba: (2 * num_v_heads, hidden)
                    let in_proj_ba = hidden * (cfg.linear_num_value_heads * 2);
                    let out_proj = value_dim * hidden / weight_pack_factor;
                    let conv1d = conv_dim * cfg.linear_conv_kernel_dim;
                    let dt_bias = cfg.linear_num_value_heads;
                    let a_log = cfg.linear_num_value_heads;
                    let norm = cfg.linear_value_head_dim;
                    in_proj_qkvz + in_proj_ba + out_proj + conv1d + dt_bias + a_log + norm
                }
            };

            let moe_gate = cfg.hidden_size * cfg.num_experts;
            let shared_expert =
                3 * cfg.hidden_size * cfg.shared_expert_intermediate_size / weight_pack_factor;
            let routed_experts = cfg.num_experts * 3 * cfg.hidden_size * cfg.moe_intermediate_size
                / weight_pack_factor;

            let per_layer_elems = input_layernorm
                + post_attention_layernorm
                + attn_elems
                + moe_gate
                + shared_expert
                + routed_experts;

            layer_sizes.push(per_layer_elems * dtype.size_in_bytes());
        }

        Ok(layer_sizes)
    }

    fn num_layers(&self, config: &str) -> Result<usize> {
        let cfg: crate::models::qwen3_next::Config = serde_json::from_str(config)?;
        Ok(cfg.num_hidden_layers)
    }
    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg: crate::models::qwen3_next::Config = serde_json::from_str(config)?;

        let cfg = ModelConfigMetadata {
            max_seq_len: cfg.max_position_embeddings,
            num_layers: cfg.num_hidden_layers,
            hidden_size: cfg.hidden_size,
            num_kv_heads: cfg.num_key_value_heads,
            num_attn_heads: cfg.num_attention_heads,
            sliding_window: None,
            k_head_dim: cfg.head_dim,
            v_head_dim: cfg.head_dim,
            kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
        };

        Ok(Box::new(cfg))
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dispatcher_recognizes_v4_architectures() {
        assert!(matches!(
            NormalLoaderType::from_causal_lm_name("DeepseekV4ForCausalLM").unwrap(),
            NormalLoaderType::DeepSeekV4
        ));
    }

    #[test]
    fn dispatcher_recognizes_kimi_k2_family() {
        let variants = [
            "KimiK25ForConditionalGeneration",
            "KimiK2ForCausalLM",
            "KimiK26ForCausalLM",
            "KimiK26VLForConditionalGeneration",
        ];
        for v in &variants {
            let tp = NormalLoaderType::from_causal_lm_name(v).unwrap_or_else(
                |_| panic!("{v} should map to KimiK2"),
            );
            assert!(matches!(tp, NormalLoaderType::KimiK2));
        }
    }

    #[test]
    fn dispatcher_recognizes_glm5_dsa() {
        assert!(matches!(
            NormalLoaderType::from_causal_lm_name("GlmMoeDsaForCausalLM").unwrap(),
            NormalLoaderType::GLM5MoeDsa
        ));
    }

    #[test]
    fn dispatcher_from_str_recognizes_new_arches() {
        assert!(matches!(
            "deepseekv4".parse::<NormalLoaderType>().unwrap(),
            NormalLoaderType::DeepSeekV4
        ));
        for s in &["kimi_k2", "kimi_k25", "kimi_k26"] {
            assert!(matches!(
                s.parse::<NormalLoaderType>().unwrap(),
                NormalLoaderType::KimiK2
            ));
        }
        for s in &["glm5moedsa", "glm5"] {
            assert!(matches!(
                s.parse::<NormalLoaderType>().unwrap(),
                NormalLoaderType::GLM5MoeDsa
            ));
        }
    }

    #[test]
    fn display_round_trips_new_arches() {
        let cases = [
            (NormalLoaderType::DeepSeekV4, "deepseekv4"),
            (NormalLoaderType::KimiK2, "kimi_k2"),
            (NormalLoaderType::GLM5MoeDsa, "glm5moedsa"),
        ];
        for (tp, s) in &cases {
            assert_eq!(&format!("{tp}"), s);
        }
    }

    #[test]
    fn auto_loader_dispatches_v4_config_without_crash() {
        let cfg = r#"{"architectures": ["DeepseekV4ForCausalLM"]}"#;
        // The loader should be constructible without panicking
        AutoNormalLoader::get_loader(cfg).expect("V4 dispatch should succeed");
    }

    #[test]
    fn auto_loader_dispatches_kimi_config_without_crash() {
        let cfg = r#"{"architectures": ["KimiK25ForConditionalGeneration"]}"#;
        AutoNormalLoader::get_loader(cfg).expect("Kimi K2 dispatch should succeed");
    }

    #[test]
    fn auto_loader_dispatches_glm5_config_without_crash() {
        let cfg = r#"{"architectures": ["GlmMoeDsaForCausalLM"]}"#;
        AutoNormalLoader::get_loader(cfg).expect("GLM-5 DSA dispatch should succeed");
    }

    /// Realistic V4 Flash config: parse through V3's config struct AND through
    /// the AutoNormalLoader dispatcher. Verifies V4's qk_nope_head_dim=448,
    /// v_head_dim=512 etc. survive the V3 parser without loss.
    #[test]
    fn realistic_v4_config_parses_through_v3_struct() {
        let cfg = r#"{
            "architectures": ["DeepseekV4ForCausalLM"],
            "vocab_size": 129280,
            "hidden_size": 7168,
            "intermediate_size": 18432,
            "moe_intermediate_size": 2048,
            "num_hidden_layers": 27,
            "num_attention_heads": 64,
            "num_key_value_heads": 1,
            "n_routed_experts": 256,
            "n_shared_experts": 1,
            "num_experts_per_tok": 8,
            "moe_layer_freq": 1,
            "first_k_dense_replace": 3,
            "max_position_embeddings": 65536,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "attention_bias": false,
            "q_lora_rank": 1536,
            "qk_nope_head_dim": 448,
            "qk_rope_head_dim": 64,
            "kv_lora_rank": 512,
            "v_head_dim": 512,
            "n_group": 8,
            "topk_group": 4,
            "topk_method": "noaux_tc",
            "scoring_func": "sigmoid",
            "routed_scaling_factor": 2.5,
            "compress_ratios": [0, 4, 4, 128, 4, 4, 128, 4, 4, 128, 4, 4, 128, 4, 4, 128, 4, 4, 128, 4, 4, 128, 4, 4, 128, 4, 0],
            "window_size": 128,
            "compress_rope_theta": 40000,
            "index_n_heads": 64,
            "index_head_dim": 128,
            "index_topk": 512,
            "o_lora_rank": 1024,
            "o_groups": 8
        }"#;

        // Dispatcher should construct a working loader
        let _loader = AutoNormalLoader::get_loader(cfg)
            .expect("V4 Flash realistic config should dispatch through AutoLoader");

        // V3 parser should swallow V4's superset config without error
        let v3_cfg: crate::models::deepseek3::DeepSeekV3Config =
            serde_json::from_str(cfg).expect("V4 config should parse via V3 struct");

        // Critical V4-specific shapes survive the V3 deserializer
        assert_eq!(v3_cfg.qk_nope_head_dim, 448);
        assert_eq!(v3_cfg.v_head_dim, 512);
        assert_eq!(v3_cfg.kv_lora_rank, 512);
        assert_eq!(v3_cfg.q_lora_rank, Some(1536));
        assert_eq!(v3_cfg.num_hidden_layers, 27);
        assert_eq!(v3_cfg.num_attention_heads, 64);
        assert_eq!(v3_cfg.n_routed_experts, Some(256));
        assert_eq!(v3_cfg.first_k_dense_replace, 3);
    }

    /// V4-specific config parses through the new DeepSeekV4Config (with V4 fields
    /// like compress_ratios, o_lora_rank, sliding_window).
    #[test]
    fn v4_config_parses_through_v4_struct() {
        let cfg = r#"{
            "architectures": ["DeepseekV4ForCausalLM"],
            "vocab_size": 129280,
            "hidden_size": 4096,
            "intermediate_size": 18432,
            "moe_intermediate_size": 2048,
            "num_hidden_layers": 43,
            "num_attention_heads": 64,
            "n_routed_experts": 256,
            "n_shared_experts": 1,
            "num_experts_per_tok": 6,
            "first_k_dense_replace": 3,
            "max_position_embeddings": 1048576,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "q_lora_rank": 1536,
            "qk_nope_head_dim": 448,
            "qk_rope_head_dim": 64,
            "kv_lora_rank": 512,
            "v_head_dim": 512,
            "n_group": 8,
            "topk_group": 4,
            "compress_ratios": [0, 4, 4, 128, 4, 4, 128, 4, 4, 128, 4, 4, 128, 4, 4, 128, 4, 4, 128, 4, 4, 128, 4, 4, 128, 4, 0],
            "sliding_window": 128,
            "compress_rope_theta": 40000.0,
            "o_lora_rank": 1024,
            "o_groups": 8,
            "index_n_heads": 64,
            "index_head_dim": 128,
            "index_topk": 512
        }"#;

        let v4_cfg: crate::models::deepseek4::DeepSeekV4Config =
            serde_json::from_str(cfg).expect("V4 config should parse via V4 struct");

        // V4-specific fields preserved
        assert_eq!(v4_cfg.compress_ratios.len(), 27);
        assert_eq!(v4_cfg.compress_ratios[3], 128); // HCA layer
        assert_eq!(v4_cfg.compress_ratios[4], 4);   // CSA layer
        assert_eq!(v4_cfg.sliding_window, 128);
        assert_eq!(v4_cfg.o_lora_rank, Some(1024));
        assert_eq!(v4_cfg.o_groups, Some(8));
        assert_eq!(v4_cfg.qk_nope_head_dim, Some(448));
        assert_eq!(v4_cfg.v_head_dim, Some(512));

        // Dispatcher constructs V4Loader (not V3Loader)
        let _loader = AutoNormalLoader::get_loader(cfg)
            .expect("V4 dispatch should succeed");
    }

    /// V4 per-layer compress_ratio lookup.
    #[test]
    fn v4_layer_compress_ratio_dispatch() {
        let cfg_str = r#"{
            "architectures": ["DeepseekV4ForCausalLM"],
            "vocab_size": 129280, "hidden_size": 4096, "intermediate_size": 18432,
            "moe_intermediate_size": 2048, "num_hidden_layers": 4, "num_attention_heads": 64,
            "n_routed_experts": 256, "num_experts_per_tok": 6,
            "max_position_embeddings": 65536, "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
            "q_lora_rank": 1536, "qk_nope_head_dim": 448, "qk_rope_head_dim": 64,
            "kv_lora_rank": 512, "v_head_dim": 512, "n_group": 8, "topk_group": 4,
            "compress_ratios": [0, 4, 128, 0]
        }"#;
        let cfg: crate::models::deepseek4::DeepSeekV4Config = serde_json::from_str(cfg_str).unwrap();
        assert_eq!(cfg.layer_compress_ratio(0), 0);   // standard MLA
        assert_eq!(cfg.layer_compress_ratio(1), 4);   // CSA
        assert_eq!(cfg.layer_compress_ratio(2), 128); // HCA
        assert_eq!(cfg.layer_compress_ratio(3), 0);   // standard
        // Out of bounds → standard
        assert_eq!(cfg.layer_compress_ratio(99), 0);
    }

    /// Realistic K2.5 / K2.6 config (text-side): V3 with K2-specific vocab,
    /// expert count, etc. Verifies V3 deserializer accepts K2-shaped JSON.
    #[test]
    fn realistic_k2_config_parses_through_v3_struct() {
        let cfg = r#"{
            "architectures": ["KimiK25ForConditionalGeneration"],
            "vocab_size": 160000,
            "hidden_size": 7168,
            "intermediate_size": 18432,
            "moe_intermediate_size": 2048,
            "num_hidden_layers": 61,
            "num_attention_heads": 64,
            "num_key_value_heads": 1,
            "n_routed_experts": 384,
            "n_shared_experts": 1,
            "num_experts_per_tok": 8,
            "moe_layer_freq": 1,
            "first_k_dense_replace": 1,
            "max_position_embeddings": 262144,
            "rms_norm_eps": 1e-6,
            "rope_theta": 50000.0,
            "attention_bias": false,
            "q_lora_rank": 1536,
            "qk_nope_head_dim": 128,
            "qk_rope_head_dim": 64,
            "kv_lora_rank": 512,
            "v_head_dim": 128,
            "n_group": 8,
            "topk_group": 4,
            "topk_method": "noaux_tc",
            "scoring_func": "sigmoid",
            "routed_scaling_factor": 2.5
        }"#;

        let _loader = AutoNormalLoader::get_loader(cfg)
            .expect("K2 config should dispatch through AutoLoader");

        let v3_cfg: crate::models::deepseek3::DeepSeekV3Config =
            serde_json::from_str(cfg).expect("K2 config should parse via V3 struct");

        assert_eq!(v3_cfg.vocab_size, 160000);
        assert_eq!(v3_cfg.n_routed_experts, Some(384));
        assert_eq!(v3_cfg.num_hidden_layers, 61);
        assert_eq!(v3_cfg.max_position_embeddings, 262144);
    }

    /// Realistic GLM-5 DSA config: V3 with `index_topk` for DSA top-k indexing.
    #[test]
    fn realistic_glm5_config_parses_through_v3_struct() {
        let cfg = r#"{
            "architectures": ["GlmMoeDsaForCausalLM"],
            "vocab_size": 151552,
            "hidden_size": 7168,
            "intermediate_size": 18432,
            "moe_intermediate_size": 2048,
            "num_hidden_layers": 92,
            "num_attention_heads": 64,
            "num_key_value_heads": 1,
            "n_routed_experts": 256,
            "n_shared_experts": 1,
            "num_experts_per_tok": 8,
            "moe_layer_freq": 1,
            "first_k_dense_replace": 3,
            "max_position_embeddings": 131072,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "attention_bias": false,
            "q_lora_rank": 1536,
            "qk_nope_head_dim": 128,
            "qk_rope_head_dim": 64,
            "kv_lora_rank": 512,
            "v_head_dim": 128,
            "n_group": 8,
            "topk_group": 4,
            "topk_method": "noaux_tc",
            "scoring_func": "sigmoid",
            "routed_scaling_factor": 2.5,
            "index_n_heads": 64,
            "index_head_dim": 128,
            "index_topk": 512
        }"#;

        let _loader = AutoNormalLoader::get_loader(cfg)
            .expect("GLM-5 DSA config should dispatch through AutoLoader");

        let v3_cfg: crate::models::deepseek3::DeepSeekV3Config =
            serde_json::from_str(cfg).expect("GLM-5 config should parse via V3 struct");

        assert_eq!(v3_cfg.vocab_size, 151552);
        assert_eq!(v3_cfg.num_hidden_layers, 92);
    }
}

