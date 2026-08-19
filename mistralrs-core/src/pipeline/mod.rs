mod amoe;
mod auto;
pub mod calibration;
pub mod chat_template;
mod diffusion;
mod embedding;
mod ggml;
mod gguf;
pub(crate) mod hf;
mod inputs_processor;
mod isq;
pub(crate) mod llg;
mod loaders;
mod macros;
mod normal;
#[cfg(test)]
mod parallel_bake_tests;
mod paths;
pub mod post_load_hooks;
pub mod mtp_pipeline;
mod processing;
mod response;
pub(crate) mod sampling;
mod speculative;
mod speech;
mod vision;

pub use super::diffusion_models::DiffusionGenerationParams;
use crate::amoe::{AnyMoeConfig, AnyMoeExpertType, AnyMoeTrainingInputs, AnyMoeTrainingResult};
use crate::device_map::DeviceMapper;
use crate::paged_attention::{CacheConfig, CacheEngine, ModelConfigLike};
use crate::prefix_cacher::PrefixCacheManagerV2;
use crate::PagedAttentionConfig;
pub use amoe::{AnyMoeLoader, AnyMoePipeline};
pub use auto::{AutoLoader, AutoLoaderBuilder};
use chat_template::ChatTemplate;
pub use diffusion::{DiffusionLoader, DiffusionLoaderBuilder};
pub use embedding::{EmbeddingLoader, EmbeddingLoaderBuilder, EmbeddingSpecificConfig};
pub use ggml::{GGMLLoader, GGMLLoaderBuilder, GGMLSpecificConfig};
pub use gguf::{GGUFLoader, GGUFLoaderBuilder, GGUFSpecificConfig};
use image::DynamicImage;
pub use inputs_processor::InputProcessorOutput;
pub(crate) use isq::IsqModelLoader;
pub use isq::{
    expand_isq_value, isq_artifact_tensor_name, parse_isq_value, IsqModel, IsqOrganization,
    UqffFullSer, UqffSourceWeights, UQFF_MTP_TENSOR_PREFIX, UQFF_MULTI_FILE_DELIMITER,
};
pub use post_load_hooks::{register_post_load_hook, PostLoadHook};
use llguidance::toktrie::TokEnv;
pub use loaders::{
    AdapterKind, AutoDeviceMapParams, AutoEmbeddingLoader, AutoNormalLoader, AutoVisionLoader,
    DeepSeekV2Loader, DeepSeekV3Loader, DeepSeekV4Loader, DeviceMappedModelLoader, DiffusionLoaderType,
    DiffusionModel, DiffusionModelLoader, EmbeddingGemmaLoader, EmbeddingLoaderType,
    EmbeddingModel, EmbeddingModelLoader, EmbeddingModelPaths, EmbeddingModule,
    EmbeddingModulePaths, EmbeddingModuleType, FluxLoader, GLM4Loader, GLM4MoeLiteLoader,
    GLM4MoeLoader, Gemma2Loader, Gemma3Loader, Gemma3nLoader, GemmaLoader, GptOssLoader,
    GraniteMoeHybridLoader, Idefics2Loader, Idefics3Loader, LLaVALoader, LLaVANextLoader,
    LlamaLoader, Loader, LocalModelPaths, MiniCpmOLoader, Mistral3Loader, MistralLoader,
    MixtralLoader, ModelKind, ModelPaths, NormalLoaderType, NormalLoadingMetadata, NormalModel,
    NormalModelLoader, Phi2Loader, Phi3Loader, Phi3VLoader, Phi3_5MoELoader, Phi4MMLoader,
    PrettyName, QuantizationKind, Qwen2Loader, Qwen2VLLoader, Qwen2_5VLLoader,
    Qwen3EmbeddingLoader, Qwen3Loader, Qwen3MoELoader, Qwen3NextLoader, Qwen3VLLoader,
    Qwen3VLMoELoader, Qwen3_5Loader, Qwen3_5MoeLoader, SmolLm3Loader, Starcoder2Loader,
    TokenSource, VLlama4Loader, VLlamaLoader, VisionLoaderType, VisionModel, VisionModelLoader,
    VoxtralLoader,
};
#[allow(clippy::too_many_arguments)]
pub(crate) fn get_device_layers_for_loader(
    loader: &dyn loaders::DeviceMappedModelLoader,
    config: &str,
    num_layers: usize,
    layer_sizes_in_bytes: Vec<usize>,
    non_mapped_size_in_bytes: usize,
    total_model_size_in_bytes: usize,
    devices: &[Device],
    dtype: DType,
    params: &loaders::AutoDeviceMapParams,
    paged_attn_config: Option<&PagedAttentionConfig>,
) -> Result<crate::device_map::DeviceMapMetadata> {
    loaders::auto_device_map::get_device_layers(
        loader,
        config,
        num_layers,
        layer_sizes_in_bytes,
        non_mapped_size_in_bytes,
        total_model_size_in_bytes,
        devices,
        dtype,
        params,
        paged_attn_config,
    )
}
use mistralrs_quant::IsqType;
pub use normal::{NormalLoader, NormalLoaderBuilder, NormalSpecificConfig};
pub(crate) use paths::{get_chat_template, get_model_paths, get_xlora_paths};
pub use paths::{AdapterPaths, LoraAdapterPaths};
pub(crate) use processing::{
    apply_chat_template, BasicProcessor, MessagesAction, Processor, ProcessorCreator,
};
use rand_isaac::Isaac64Rng;
pub use mtp_pipeline::{
    mtp_acceptance, mtp_acceptance_by_batch, mtp_acceptance_marker, mtp_acceptance_markers,
    mtp_acceptance_position_lines, mtp_load_depth, mtp_uqff_bake, record_mtp_batch_step,
    record_mtp_step, reset_mtp_acceptance, set_mtp_load_depth, set_mtp_uqff_bake,
    synthetic_acceptance, try_wrap_pipeline_with_mtp, verify_proposed, verify_proposed_with,
    CaptureOffsets, MtpAcceptance, MtpDecodeKit, MtpHiddenCapture, MtpSpeculativePipeline,
    SyntheticAcceptance, VerifyResult, MTP_MAX_TRACKED_POSITIONS, SIMULATE_ACC_LEN_ENV,
};
/// The MTP rejection rollback, reachable from the V4 model's own tests so the
/// cache it truncates (K/V **and** `XsRolling` entries) is the real one.
#[cfg(test)]
pub(crate) use mtp_pipeline::{n_cache_positions_to_drop, truncate_cache_by};
pub use speculative::{SpeculativeConfig, SpeculativeLoader, SpeculativePipeline};
pub use speech::{SpeechLoader, SpeechPipeline};
use std::any::Any;
use std::fmt::Debug;
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokenizers::Tokenizer;
pub use vision::{VisionLoader, VisionLoaderBuilder, VisionSpecificConfig};

use anyhow::Result;
use candle_core::{DType, Device, IndexOp, Tensor, Var};

use crate::sequence::Sequence;

pub use self::inputs_processor::{
    text_models_inputs_processor, InputsProcessor, InputsProcessorType,
};
use self::text_models_inputs_processor::PagedAttentionMeta;
pub use crate::kv_cache::{
    Cache, CacheManager, EitherCache, HybridLayerCache, KvCache, LayerCaches, NormalCache,
    NormalCacheType,
};

#[derive(Clone, PartialEq, Eq)]
pub enum SupportedModality {
    Text,
    Audio,
    Vision,
    Embedding,
}

impl Debug for SupportedModality {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Text => write!(f, "📝 Text"),
            Self::Audio => write!(f, "🔊 Audio"),
            Self::Vision => write!(f, "🖼️ Vision"),
            Self::Embedding => write!(f, "🔢 Embedding"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Modalities {
    pub input: Vec<SupportedModality>,
    pub output: Vec<SupportedModality>,
}

pub struct GeneralMetadata {
    pub max_seq_len: usize,
    /// Only None if it doesn't make sense for the model
    pub llg_factory: Option<Arc<llguidance::ParserFactory>>,
    pub no_kv_cache: bool,
    pub no_prefix_cache: bool,
    pub num_hidden_layers: usize,
    pub eos_tok: Vec<u32>,
    pub kind: ModelKind,
    // TODO: Replace is_xlora queries to check via kind instead:
    pub is_xlora: bool,
    pub activation_dtype: DType,
    pub sliding_window: Option<usize>,
    // PagedAttention stuff
    pub cache_config: Option<CacheConfig>,
    pub cache_engine: Option<CacheEngine>,
    pub model_metadata: Option<Arc<dyn ModelConfigLike + Send + Sync>>,
    pub modalities: Modalities,
}

impl GeneralMetadata {
    pub fn tok_env(&self) -> Option<TokEnv> {
        self.llg_factory.as_ref().map(|f| f.tok_env().clone())
    }
}

/// Per-batch context the engine passes to `Pipeline::autonomous_decode`.
/// Carries the paged-attention metadata the runner needs to prime its
/// on-GPU input buffers (`prime_for_step`). The engine reads from its
/// scheduler's `kv_cache_manager` to build these vectors, so pipelines
/// don't have to thread the scheduler through their state.
///
/// CUDA-gated because `AutonomousDecodeRunner` only exists under feature
/// `cuda`. Lifetimes are shorter than the engine's outer step loop.
#[cfg(feature = "cuda")]
pub struct AutonomousDecodeContext<'a> {
    /// One next-token id per sequence in the batch (the runner's
    /// `prime_for_step` pads to `padded_batch_size` if needed).
    pub next_token_ids: &'a [i32],
    /// Per-sequence current decode position (== seq.len() - 1 for the
    /// last token, or seq.len() if the next forward should produce token
    /// at index seq.len() given normal append semantics).
    pub positions: &'a [i32],
    /// Per-sequence block tables flattened row-major.
    /// Layout: `block_tables_flat[b * max_blocks + i]` = block id of i-th
    /// physical block held by sequence b.
    pub block_tables_flat: &'a [i32],
    /// Per-sequence current context length (== seq.len()).
    pub context_lens: &'a [i32],
    /// Per-sequence slot index (= block_id * block_size + offset).
    pub slot_mappings: &'a [i64],
    /// Block size the kv cache manager is using; the runner cross-checks
    /// this against its captured config and refuses if they differ.
    pub block_size: usize,
    /// Maximum number of blocks the runner's `block_tables` row holds.
    /// Must match the runner's `max_blocks_per_seq` for the captured
    /// graph; if it differs the runner refuses (host-side error, fallback).
    pub max_blocks_per_seq: usize,
}

#[derive(Clone, Copy)]
pub enum CacheInstruction {
    In,
    Out,
    /// load_preallocated_cache means to load the preallocated cache, if applicable.
    Reset {
        load_preallocated_cache: bool,
        reset_non_granular: bool,
    },
    Nothing,
}

pub trait PreProcessingMixin: MetadataMixin {
    fn get_processor(&self) -> Arc<dyn Processor> {
        Arc::new(BasicProcessor)
    }
    /// Only None if it doesnt make sense for the model
    fn get_chat_template(&self) -> Option<Arc<ChatTemplate>>;
    fn get_input_processor_config(&self) -> Option<Arc<dyn Any>>;
}

pub trait IsqPipelineMixin {
    fn re_isq_model(&mut self, dtype: IsqType) -> Result<()>;
}

pub trait CacheManagerMixin {
    /// Clone the cache FROM the sequences' cache TO the model cache. Only called for completion seqs.
    /// It is not a guarantee that this will be called for each completion step.
    /// Build the model's batched cache from `seqs`' per-sequence caches.
    ///
    /// Fallible so a batch that cannot be represented as one dense cache
    /// fails the *requests* rather than panicking the engine task — see
    /// [`crate::kv_cache::CacheManager::clone_in_cache`].
    fn clone_in_cache(&self, seqs: &mut [&mut Sequence]) -> candle_core::Result<()>;
    /// Clone the cache FROM the model cache TO the sequences. Called for prompt and completion seqs.
    /// It is not a guarantee that this will be called for each step.
    fn clone_out_cache(&self, seqs: &mut [&mut Sequence]);
    /// Set the model cache to all None. Only called for prompt seqs.
    /// It is not a guarantee that this will be called for each prompt step.
    /// This may also reset the non granular state if applicable.
    fn set_none_cache(
        &self,
        seqs: &mut [&mut Sequence],
        reset_non_granular: bool,
        modify_draft_cache: bool,
        load_preallocated_cache: bool,
    );
    fn cache(&self) -> &EitherCache;
    fn do_preallocated_cache(&self) -> bool {
        matches!(self.cache(), EitherCache::Normal(_))
    }
}

pub trait MetadataMixin {
    fn device(&self) -> Device;
    /// Only None if it doesnt make sense for the model
    fn tokenizer(&self) -> Option<Arc<Tokenizer>>;
    fn name(&self) -> String;
    fn reset_non_granular_state(&self);
    fn get_metadata(&self) -> Arc<GeneralMetadata>;
    fn device_mapper(&self) -> Option<&dyn DeviceMapper>;
}

/// Implemented by the base model of an AnyMoe.
pub trait AnyMoePipelineMixin {
    /// Get vars for each gating layer
    fn amoe_layer_vars(&self) -> Vec<Vec<Var>> {
        unreachable!()
    }
    fn amoe_finish_training(&mut self, _gate_model_id: Option<String>) -> candle_core::Result<()> {
        unreachable!()
    }
    fn amoe_base_model_trainable_params(&self) -> usize {
        unreachable!()
    }
    fn amoe_supported(&self) -> bool {
        false
    }
    /// Per-layer cached outputs.
    fn amoe_take_cached_gating_outputs(&mut self) -> Vec<Tensor> {
        unreachable!()
    }
    /// Inject the MoE layers
    #[allow(clippy::too_many_arguments)]
    fn amoe_create_layers(
        &mut self,
        _model_ids: Vec<String>,
        _token: &TokenSource,
        _revision: Option<String>,
        _match_regex: &str,
        _config: AnyMoeConfig,
        _dtype: DType,
        _dev: &Device,
        (_prefix, _mlp): (String, String),
        _layers: Vec<usize>,
        _expert_type: AnyMoeExpertType,
        _silent: bool,
        _gate_model_id: Option<String>,
    ) -> candle_core::Result<()> {
        unreachable!()
    }
    /// Pre-train the gating layers
    #[allow(clippy::too_many_arguments)]
    fn amoe_pre_train(
        &self,
        _inputs: AnyMoeTrainingInputs,
        (_prefix, _mlp): (String, String),
        _model_ids: Vec<String>,
        _token: TokenSource,
        _revision: Option<String>,
        _layers: Vec<usize>,
        _silent: bool,
    ) -> Result<Option<AnyMoeTrainingResult>, candle_core::Error> {
        unreachable!()
    }
}

/// Category of the model. This can also be used to extract model-category specific tools,
/// such as the vision model prompt prefixer.
#[derive(Clone)]
pub enum ModelCategory {
    Text,
    Vision {
        prefixer: Arc<dyn MultimodalPromptPrefixer>,
    },
    Diffusion,
    Audio,
    Speech,
    Embedding,
}

impl std::fmt::Debug for ModelCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelCategory::Text => write!(f, "ModelCategory::Text"),
            ModelCategory::Vision { .. } => write!(f, "ModelCategory::Vision {{ prefixer: .. }}"),
            ModelCategory::Diffusion => write!(f, "ModelCategory::Diffusion"),
            ModelCategory::Audio => write!(f, "ModelCategory::Audio"),
            ModelCategory::Speech => write!(f, "ModelCategory::Speech"),
            ModelCategory::Embedding => write!(f, "ModelCategory::Embedding"),
        }
    }
}

impl PartialEq for ModelCategory {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Text, Self::Text) => true,
            (Self::Vision { .. }, Self::Vision { .. }) => true,
            (Self::Audio, Self::Audio) => true,
            (Self::Speech, Self::Speech) => true,
            (Self::Diffusion, Self::Diffusion) => true,
            (Self::Embedding, Self::Embedding) => true,
            (
                Self::Text
                | Self::Vision { .. }
                | Self::Diffusion
                | Self::Audio
                | Self::Speech
                | Self::Embedding,
                _,
            ) => false,
        }
    }
}

/// Prepend a vision tag appropriate for the model to the prompt. Image indexing is assumed that start at 0.
pub trait MultimodalPromptPrefixer: Send + Sync {
    /// Prefix for inclusion in messages (may do nothing if the chat template handles it).
    fn prefix_image(&self, _image_indices: Vec<usize>, prompt: &str) -> String {
        prompt.to_string()
    }
    /// Prefix for inclusion in messages (may do nothing if the chat template handles it).
    fn prefix_audio(&self, _audio_indexes: Vec<usize>, prompt: &str) -> String {
        prompt.to_string()
    }
}

thread_local! {
    /// Set by `Engine::run` around a prefill step that is NOT the last chunk of
    /// its prompts, and read by `Pipeline::step`'s sampling stage.
    static PREFILL_INTERMEDIATE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Mark the step about to run as an intermediate prefill chunk. Returns a guard
/// that clears the flag, so an early return or a panic inside the step cannot
/// leave the next decode step believing it must not sample.
pub(crate) fn mark_prefill_intermediate() -> PrefillChunkGuard {
    PREFILL_INTERMEDIATE.with(|f| f.set(true));
    PrefillChunkGuard
}

pub(crate) struct PrefillChunkGuard;

impl Drop for PrefillChunkGuard {
    fn drop(&mut self) {
        PREFILL_INTERMEDIATE.with(|f| f.set(false));
    }
}

pub(crate) fn prefill_chunk_is_intermediate() -> bool {
    PREFILL_INTERMEDIATE.with(std::cell::Cell::get)
}

#[derive(Clone)]
pub enum CacheBackendMetadata {
    DefaultInstructions {
        pre_op: CacheInstruction,
        post_op: CacheInstruction,
    },
    PagedAttention {
        metadata: PagedAttentionMeta,
    },
}

#[derive(Clone, Debug)]
pub enum ForwardInputsResult {
    RawLogits {
        logits: Tensor,
    },
    Embeddings {
        embeddings: Tensor,
    },
    CausalGeneration {
        logits: Tensor,
    },
    Image {
        images: Vec<DynamicImage>,
    },
    Speech {
        pcms: Vec<Arc<Vec<f32>>>,
        rates: Vec<usize>,
        channels: Vec<usize>,
    },
}

impl ForwardInputsResult {
    fn index_bs(&self, bs_idx: usize) -> candle_core::Result<Self> {
        match self {
            Self::CausalGeneration { logits } => Ok(Self::CausalGeneration {
                logits: logits.i(bs_idx)?,
            }),
            Self::Embeddings { embeddings } => Ok(Self::Embeddings {
                embeddings: embeddings.i(bs_idx)?,
            }),
            Self::RawLogits { logits } => Ok(Self::RawLogits {
                logits: logits.i(bs_idx)?,
            }),
            Self::Image { images } => Ok(Self::Image {
                images: vec![images[bs_idx].clone()],
            }),
            Self::Speech {
                pcms,
                rates,
                channels,
            } => Ok(Self::Speech {
                pcms: vec![pcms[bs_idx].clone()],
                rates: vec![rates[bs_idx]],
                channels: vec![channels[bs_idx]],
            }),
        }
    }

    fn to_device(&self, device: &Device) -> candle_core::Result<Self> {
        match self {
            Self::CausalGeneration { logits } => Ok(Self::CausalGeneration {
                logits: logits.to_device(device)?,
            }),
            Self::RawLogits { logits } => Ok(Self::RawLogits {
                logits: logits.to_device(device)?,
            }),
            Self::Embeddings { embeddings } => Ok(Self::Embeddings {
                embeddings: embeddings.to_device(device)?,
            }),
            Self::Image { .. } => Ok(self.clone()),
            Self::Speech { .. } => Ok(self.clone()),
        }
    }
}

/// How many host copies of a **batched** forward result
/// [`host_copy_batched_result`] has issued.
///
/// One per decode step is correct. One per *sequence* per step is the O(B²)
/// regression this counter exists to pin — see the function docs and
/// `pipeline::host_copy_tests`.
pub(crate) static LOGITS_HOST_COPIES: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Move a batched forward result to `host` with a **single** transfer, before
/// it is split into per-sequence views.
///
/// This ordering is load-bearing, not stylistic. `Tensor::i()` /
/// `Tensor::narrow` return a *view* that clones the storage `Arc`, so a
/// per-sequence slice of a `[B, 1, vocab]` logits tensor still refers to the
/// whole `B * vocab` allocation. `Tensor::to_device` then copies the **entire
/// storage** — `Tensor::to_device` → `CudaStorage::to_cpu_storage` →
/// `clone_dtoh(slice)` (candle `d2d1d07`, `tensor.rs:2379`,
/// `cuda_backend/mod.rs:1788`) — and merely carries the view's layout across.
///
/// So splitting first and copying after moves `B * (B * vocab)` elements over
/// PCIe **every decode step**, i.e. O(B²). Copying first and splitting after
/// moves `B * vocab`, i.e. O(B). At B=256 with V4's `vocab_size = 129_280` in
/// BF16 that is 16.9 GB/step versus 66 MB/step.
///
/// A single sequence is left on the device on purpose: `Sampler::sample`'s
/// GPU fast path is gated on `!logits.device().is_cpu()` (`sampler.rs`), so
/// moving a B=1 batch to the host would silently disable it.
pub(crate) fn host_copy_batched_result(
    raw: ForwardInputsResult,
    n_seqs: usize,
    host: &Device,
) -> candle_core::Result<ForwardInputsResult> {
    let needs_copy = match &raw {
        // Always returned to the caller on the host.
        ForwardInputsResult::RawLogits { .. } | ForwardInputsResult::Embeddings { .. } => true,
        ForwardInputsResult::CausalGeneration { .. } => n_seqs > 1,
        ForwardInputsResult::Image { .. } | ForwardInputsResult::Speech { .. } => false,
    };
    if !needs_copy {
        return Ok(raw);
    }
    LOGITS_HOST_COPIES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    raw.to_device(host)
}

#[derive(serde::Serialize, serde::Deserialize)]
pub(crate) struct FileListCache {
    files: Vec<String>,
}

/// Wrap a pre-allocated F32 GPU logits buffer as a Candle Tensor.
/// The buffer is owned by DedicatedDecodePath — we copy to a fresh tensor
/// to avoid aliasing (Candle may hold the tensor across steps).
#[cfg(feature = "cuda")]
fn wrap_f32_logits(
    ptr: u64,
    batch_size: usize,
    vocab_size: usize,
    device: &Device,
) -> candle_core::Result<Tensor> {
    use candle_core::{DType, Shape};

    let elem_count = batch_size * vocab_size;
    let shape = Shape::from_dims(&[batch_size, vocab_size]);

    unsafe {
        extern "C" {
            fn cudaMemcpy(
                dst: *mut std::ffi::c_void, src: *const std::ffi::c_void,
                count: usize, kind: u32,
            ) -> u32;
        }
        // Allocate F32 tensor and D2D copy (no BF16→F32 cast needed)
        let fresh = Tensor::zeros(shape, DType::F32, device)?;
        let fresh_ptr = arc_cuda_graph::tensor_device_ptr(&fresh)?;
        let rc = cudaMemcpy(fresh_ptr as *mut _, ptr as *const _, elem_count * 4, 3);
        // Discarding this return is how a 4 MB over-read of `logits_f32` became
        // someone else's error message two decode steps later: the runtime
        // latches the error and the next `cudaGetLastError()` anywhere in the
        // process reports it against an unrelated kernel. The *fix* for the
        // over-read is that `ensure_buffers` now grows, so `batch_size` never
        // exceeds the allocated row capacity; this check is here so that if it
        // ever does again, it is named here rather than blamed elsewhere.
        if rc != 0 {
            candle_core::bail!(
                "wrap_f32_logits: cudaMemcpy D2D of {elem_count} f32 \
                 (batch_size={batch_size}, vocab_size={vocab_size}) failed with cudaError {rc}"
            );
        }
        Ok(fresh)
    }
}

#[async_trait::async_trait]
pub trait Pipeline:
    Send
    + Sync
    + PreProcessingMixin
    + IsqPipelineMixin
    + CacheManagerMixin
    + MetadataMixin
    + AnyMoePipelineMixin
{
    fn forward_inputs(
        &mut self,
        inputs: Box<dyn Any>,
        return_raw_logits: bool,
    ) -> Result<ForwardInputsResult, candle_core::Error>;

    /// Access the CUDA graph runner for decode acceleration.
    /// Returns None by default. Pipelines on CUDA devices override this.
    #[cfg(feature = "cuda")]
    fn cuda_graph_runner_mut(&mut self) -> Option<&mut arc_cuda_graph::CudaGraphRunner> {
        None
    }

    /// Access the autonomous decode runner for full GPU-autonomous generation.
    #[cfg(feature = "cuda")]
    fn autonomous_runner_mut(&mut self) -> Option<&mut arc_cuda_graph::AutonomousDecodeRunner> {
        None
    }

    /// Access the dedicated decode path (bypasses Candle, runs on non-blocking stream).
    #[cfg(feature = "cuda")]
    fn dedicated_decode_mut(&mut self) -> Option<&mut arc_cuda_graph::DedicatedDecodePath> {
        None
    }

    /// Run a full autonomous decode loop (forward+sampling+step_update per iteration).
    ///
    /// This replaces the engine's step-by-step decode loop for sequences that can
    /// be decoded entirely on GPU. Pre-allocated KV blocks must be set up before calling.
    ///
    /// Returns generated token IDs per sequence in the batch, or `Ok(None)` if
    /// the autonomous runner is unavailable or its graph has not yet been
    /// captured (caller must fall back to step-by-step decode).
    ///
    /// The `ctx` argument carries per-step inputs (block tables, slot
    /// mappings, etc.) that the runner needs to prime its on-GPU input
    /// buffers before each captured-graph launch. The default impl ignores
    /// it; pipelines that wire autonomous decode use it to call
    /// `AutonomousDecodeRunner::prime_for_step`.
    ///
    /// The ring buffer is polled for streaming output.
    #[cfg(feature = "cuda")]
    fn autonomous_decode(
        &mut self,
        _input_seqs: &mut [&mut crate::sequence::Sequence],
        _ctx: &crate::pipeline::AutonomousDecodeContext<'_>,
    ) -> Result<Option<Vec<Vec<i32>>>, candle_core::Error> {
        if let Some(runner) = self.autonomous_runner_mut() {
            // The graph capture is deferred: pipelines that wire autonomous
            // decode must call `runner.capture(&forward_fn)` once the first
            // real decode batch arrives. Until that happens, gracefully
            // fall back to the step-by-step path.
            if !runner.is_captured() {
                return Ok(None);
            }
            let tokens = runner.run_decode_loop()?;
            Ok(Some(tokens))
        } else {
            Ok(None) // autonomous runner not available, fall back to step-by-step
        }
    }

    /// Run forward_inputs, with CUDA graph capture-once + replay for decode steps.
    ///
    /// First decode at a given batch size: capture the forward pass into a graph.
    /// Subsequent decodes at the same batch size: replay the cached graph.
    /// Prompts always run eagerly.
    fn graph_wrapped_forward(
        &mut self,
        inputs: Box<dyn Any>,
        _is_prompt: bool,
        return_raw_logits: bool,
    ) -> Result<ForwardInputsResult, candle_core::Error> {
        // Dedicated decode path: bypasses Candle, runs on non-blocking stream
        #[cfg(feature = "cuda")]
        if !_is_prompt && !return_raw_logits {
            // Build paged attention state from ModelInputs + CacheEngine, then run dedicated path
            let dedicated_result = self.try_dedicated_decode(&inputs);
            if let Some(result) = dedicated_result {
                return result;
            }
        }

        self.forward_inputs(inputs, return_raw_logits)
    }

    /// Attempt to run the dedicated decode path. Returns Some(result) if it ran,
    /// None if we should fall through to Candle.
    #[cfg(feature = "cuda")]
    fn try_dedicated_decode(
        &mut self,
        inputs: &Box<dyn Any>,
    ) -> Option<Result<ForwardInputsResult, candle_core::Error>> {
        use arc_cuda_graph::{PagedAttentionState, LayerKvCache};
        use candle_core::cuda::cudarc::driver::DevicePtr;
        let __dd_t0 = Instant::now();

        // Extract model inputs
        let model_inputs = inputs.downcast_ref::<text_models_inputs_processor::ModelInputs>()?;
        let paged_meta = model_inputs.paged_attn_meta.as_ref()?;

        // Check dedicated path readiness
        let metadata = self.get_metadata();
        let cache_engine = metadata.cache_engine.as_ref()?;
        let cache_config = metadata.cache_config.as_ref()?;
        let device = self.device();

        let is_turbo = cache_config.cache_type.is_turboquant();
        // FP8 cache not yet supported in dedicated path
        if matches!(cache_config.cache_type, crate::paged_attention::PagedCacheType::F8E4M3) {
            return None;
        }

        // Get the dedicated decode path — need two-phase borrow to avoid holding &mut self
        // First check warmup/enabled without running
        {
            let dedicated = self.dedicated_decode_mut()?;
            if dedicated.tick_warmup() {
                return None; // Still warming up
            }
            if !dedicated.is_enabled() {
                return None;
            }
        }

        // Token IDs: prefer the host-side Vec stashed by the input processor.
        // It's the same data the input processor used to build `input_ids` on GPU,
        // so we avoid a D2H sync (~1.7 ms/token) entirely.
        let token_ids: Vec<i32> = if let Some(cpu) = model_inputs.input_ids_cpu.as_ref() {
            cpu.iter().map(|&x| x as i32).collect()
        } else {
            // Fallback: pull from the GPU tensor (slow path).
            let ids = model_inputs.input_ids.flatten_all()
                .and_then(|t| t.to_vec1::<u32>())
                .unwrap_or_default();
            ids.into_iter().map(|x| x as i32).collect()
        };
        let positions: Vec<i32> = model_inputs.seqlen_offsets
            .iter().map(|&x| x as i32).collect();
        let __dd_t1 = Instant::now();

        if token_ids.is_empty() {
            return None;
        }
        let batch_size = token_ids.len();

        // REFUSE batch > 1. The dedicated decode path is structurally batch-1
        // only: every projection in `decode_forward` (qkv, o_proj, gate, up,
        // down, lm_head) is a GEMV whose launcher takes no batch dimension --
        // e.g. `arc_launch_gemv_bf16_f32out(weight, input, output, M, K,
        // stream)` grids over output rows, not over sequences. Only the
        // elementwise and attention kernels ever saw `bs`. So at batch > 1,
        // sequences 1..N-1 read uninitialised rows of every activation buffer
        // and sample from garbage logits.
        //
        // MEASURED on H200 (qwen05, paged, 2026-08-18), same binary, one
        // variable -- the env var below:
        //   path ON : B=8 -> 71/512 tokens, B=32 -> 95/2048
        //   path OFF: B=8 -> 512/512,       B=32 -> 2048/2048
        // 71 = 64 + 7*1 and 95 = 64 + 31*1: sequence 0 emits its full 64
        // tokens, every other sequence emits exactly one and stops. Two batch
        // sizes agreeing on 64+(B-1) is a structural signature -- a bandwidth
        // or scheduling problem cannot produce it; only "one sequence computed,
        // the rest garbage" can.
        //
        // Falling through to Candle is proven-good: that is exactly what the
        // OFF arm above measures. This guard is the correctness fix; batching
        // the GEMVs is the performance work it makes safe to attempt.
        if batch_size > 1 {
            static WARNED: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            if !WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                tracing::warn!(
                    "Dedicated decode path refused for batch_size={batch_size}: its GEMVs are \
                     batch-1 only. Falling back to the Candle paged path (correct, slower). \
                     This refusal is permanent for multi-sequence batches."
                );
            }
            return None;
        }

        // Extract KV cache pointers from CacheEngine
        let kv_cache = cache_engine.get_kv_cache();
        let num_layers = kv_cache.len();

        let mut layer_caches = Vec::with_capacity(num_layers);
        for (key_cache, value_cache, k_norms_opt, v_norms_opt) in kv_cache.iter() {
            let kc_ptr = match arc_cuda_graph::tensor_device_ptr(key_cache) {
                Ok(p) => p, Err(e) => return Some(Err(e)),
            };
            let vc_ptr = match arc_cuda_graph::tensor_device_ptr(value_cache) {
                Ok(p) => p, Err(e) => return Some(Err(e)),
            };
            let kn_ptr = match k_norms_opt.as_ref().map(arc_cuda_graph::tensor_device_ptr) {
                Some(Ok(p)) => p, Some(Err(e)) => return Some(Err(e)), None => 0,
            };
            let vn_ptr = match v_norms_opt.as_ref().map(arc_cuda_graph::tensor_device_ptr) {
                Some(Ok(p)) => p, Some(Err(e)) => return Some(Err(e)), None => 0,
            };
            layer_caches.push(LayerKvCache {
                key_cache: kc_ptr,
                value_cache: vc_ptr,
                k_norms: kn_ptr,
                v_norms: vn_ptr,
            });
        }

        // Extract block_tables, context_lens, slot_mappings pointers
        let dev_loc = candle_core::DeviceLocation::Cuda { gpu_id: 0 };

        let block_tables_tensor = paged_meta.block_tables.as_ref()
            .and_then(|bt| bt.get(&dev_loc));
        let context_lens_tensor = paged_meta.context_lens.as_ref()
            .and_then(|cl| cl.get(&dev_loc));
        let slot_mappings_tensor = paged_meta.slot_mappings.get(&dev_loc);

        let (block_tables_ptr, context_lens_ptr, slot_mappings_ptr) = match (
            block_tables_tensor, context_lens_tensor, slot_mappings_tensor
        ) {
            (Some(bt), Some(cl), Some(sm)) => {
                let bt_ptr = match arc_cuda_graph::tensor_device_ptr(bt) {
                    Ok(p) => p, Err(e) => return Some(Err(e)),
                };
                let cl_ptr = match arc_cuda_graph::tensor_device_ptr(cl) {
                    Ok(p) => p, Err(e) => return Some(Err(e)),
                };
                let sm_ptr = match arc_cuda_graph::tensor_device_ptr(sm) {
                    Ok(p) => p, Err(e) => return Some(Err(e)),
                };
                (bt_ptr, cl_ptr, sm_ptr)
            }
            _ => return None, // No paged attention metadata — fall through
        };

        // Compute cache strides from key_cache shape
        let kc_shape = kv_cache[0].0.dims();
        let block_size = cache_config.block_size as i32;
        let max_blocks = block_tables_tensor.map(|bt| {
            if bt.dims().len() >= 2 { bt.dims()[1] } else { 1 }
        }).unwrap_or(1) as i32;

        let (kv_block_stride, kv_head_stride, x) = if kc_shape.len() == 5 {
            // Standard: [num_blocks, num_kv_heads, head_dim/x, block_size, x]
            let (d2, d3, d4) = (kc_shape[2], kc_shape[3], kc_shape[4]);
            ((kc_shape[1] * d2 * d3 * d4) as i32, (d2 * d3 * d4) as i32, d4 as i32)
        } else if kc_shape.len() == 4 {
            // TurboQuant: [num_blocks, num_kv_heads, packed_bytes, block_size]
            let (d2, d3) = (kc_shape[2], kc_shape[3]);
            ((kc_shape[1] * d2 * d3) as i32, (d2 * d3) as i32, 1)
        } else {
            return None;
        };

        // Norm strides for TurboQuant: norms shape [num_blocks, num_kv_heads, block_size]
        let (norm_block_stride, norm_head_stride) = if is_turbo {
            if let Some(ref kn) = kv_cache[0].2 {
                let ns = kn.dims();
                if ns.len() == 3 {
                    ((ns[1] * ns[2]) as i32, ns[2] as i32)
                } else { (0, 0) }
            } else { (0, 0) }
        } else { (0, 0) };

        let max_context_len = paged_meta.max_context_len.unwrap_or(0) as i32;

        let paged_attn_state = PagedAttentionState {
            layer_caches,
            block_tables: block_tables_ptr,
            context_lens: context_lens_ptr,
            slot_mappings: slot_mappings_ptr,
            block_size,
            max_context_len,
            max_num_blocks_per_seq: max_blocks,
            kv_block_stride,
            kv_head_stride,
            norm_block_stride,
            norm_head_stride,
            x,
            is_turbo,
        };

        // Drop the KV cache lock before running
        drop(kv_cache);
        let __dd_t2 = Instant::now();

        // Run the dedicated decode path with CPU-staged token IDs.
        let dedicated = self.dedicated_decode_mut().unwrap();
        let logits_ptr = match dedicated.run_step(&token_ids, &positions, &paged_attn_state) {
            Ok(ptr) => ptr,
            Err(e) => return Some(Err(e)),
        };
        let __dd_t3 = Instant::now();

        // Wrap BF16 logits as a Candle Tensor
        let vocab_size = dedicated.weights.config.vocab_size;
        let logits_tensor = match wrap_f32_logits(logits_ptr, batch_size, vocab_size, &device) {
            Ok(t) => t,
            Err(e) => return Some(Err(e)),
        };
        let __dd_t4 = Instant::now();

        // Periodic per-phase timing
        {
            use std::sync::atomic::{AtomicU64, Ordering};
            static N: AtomicU64 = AtomicU64::new(0);
            static SUM_INPUT: AtomicU64 = AtomicU64::new(0);
            static SUM_META: AtomicU64 = AtomicU64::new(0);
            static SUM_RUN: AtomicU64 = AtomicU64::new(0);
            static SUM_WRAP: AtomicU64 = AtomicU64::new(0);
            let n = N.fetch_add(1, Ordering::Relaxed) + 1;
            SUM_INPUT.fetch_add(__dd_t1.duration_since(__dd_t0).as_micros() as u64, Ordering::Relaxed);
            SUM_META.fetch_add(__dd_t2.duration_since(__dd_t1).as_micros() as u64, Ordering::Relaxed);
            SUM_RUN.fetch_add(__dd_t3.duration_since(__dd_t2).as_micros() as u64, Ordering::Relaxed);
            SUM_WRAP.fetch_add(__dd_t4.duration_since(__dd_t3).as_micros() as u64, Ordering::Relaxed);
            if n > 4 && (n - 4) % 50 == 0 {
                let nn = n as f64;
                tracing::info!(
                    "DD_PHASES_us avg n={}: input={:.0} meta={:.0} run={:.0} wrap={:.0}",
                    n,
                    SUM_INPUT.load(Ordering::Relaxed) as f64 / nn,
                    SUM_META.load(Ordering::Relaxed) as f64 / nn,
                    SUM_RUN.load(Ordering::Relaxed) as f64 / nn,
                    SUM_WRAP.load(Ordering::Relaxed) as f64 / nn,
                );
            }
        }

        Some(Ok(ForwardInputsResult::CausalGeneration { logits: logits_tensor }))
    }

    /// Returns the total of model execution time.
    #[allow(clippy::too_many_arguments)]
    async fn step(
        &mut self,
        input_seqs: &mut [&mut Sequence],
        is_prompt: bool,
        return_raw_logits: bool,
        prefix_cacher: &mut PrefixCacheManagerV2,
        disable_eos_stop: bool,
        rng: Arc<std::sync::Mutex<Isaac64Rng>>,
        backend_metadata: CacheBackendMetadata,
    ) -> Result<Duration, candle_core::Error> {
        match backend_metadata {
            CacheBackendMetadata::DefaultInstructions { pre_op, post_op } => {
                // This is the arm V4 takes. It is *not* the arm that carried
                // the old `STEP_us TOTAL/fwd/sample/other` line — that lives
                // only on the PagedAttention arm below, which V4 can never
                // reach because `DeepSeekV4Loader::supports_paged_attention`
                // returns false. Hence: no host/forward split existed here at
                // all until this instrumentation.
                arc_profiler::mark_unreachable(
                    "paged_attention",
                    "this pipeline was configured with DefaultInstructions; the PagedAttention \
                     step arm (and with it graph_wrapped_forward / dedicated decode / CUDA graph \
                     replay) is not on this path",
                    "pipeline/mod.rs:1088",
                );
                let inputs_iter = {
                    let _s = arc_profiler::span("input_prep");
                    std::iter::once(self.get_processor().inputs_processor().process_inputs(
                        self.tokenizer(),
                        input_seqs,
                        is_prompt,
                        self.get_metadata().is_xlora,
                        &self.device(),
                        self.get_metadata().no_kv_cache,
                        None,
                        return_raw_logits,
                        self.get_input_processor_config(),
                        None,
                        self.device_mapper(),
                    ))
                };

                let mut logits = vec![None; input_seqs.len()];
                let len_inputs = 1;
                let mut raw_out_logits = vec![vec![None; len_inputs]; input_seqs.len()];
                let mut embedding_logits = vec![None; input_seqs.len()];

                let mut exec_duration = Duration::ZERO;
                for (i, inputs) in inputs_iter.into_iter().enumerate() {
                    let InputProcessorOutput {
                        inputs,
                        seq_indices,
                    } = inputs.map_err(candle_core::Error::msg)?;
                    if i == 0 {
                        let _s = arc_profiler::span("cache.pre_op");
                        match pre_op {
                            CacheInstruction::In => self.clone_in_cache(input_seqs)?,
                            CacheInstruction::Nothing => (),
                            CacheInstruction::Reset {
                                load_preallocated_cache,
                                reset_non_granular,
                            } => self.set_none_cache(
                                input_seqs,
                                reset_non_granular,
                                false,
                                load_preallocated_cache,
                            ),
                            _ => unreachable!("Unreachable PRE cache op."),
                        }
                    }

                    let start = Instant::now();
                    let raw_logits = {
                        let _s = arc_profiler::span("forward");
                        self.forward_inputs(inputs, return_raw_logits)?
                    };
                    let end = Instant::now();
                    exec_duration += end.duration_since(start);

                    // ONE host copy for the whole batch, *before* the
                    // per-sequence split. See `host_copy_batched_result`: the
                    // reverse order costs O(B²) D2H bytes per step.
                    //
                    // A `sync_span`, not a plain span: this is a blocking D2H,
                    // so its cost is the host *waiting* on the device, and it
                    // is also the point where every kernel queued by the
                    // forward above finally has to have finished. Reported as
                    // wait time it is diagnostic; reported as host time it
                    // would look like CPU work that does not exist.
                    let raw_logits = {
                        let _s = arc_profiler::sync_span("logits_d2h");
                        host_copy_batched_result(raw_logits, input_seqs.len(), &Device::Cpu)?
                    };

                    let _s = arc_profiler::span("logits_split");
                    for (logit_idx, seq_idx) in seq_indices.into_iter().enumerate() {
                        if let ForwardInputsResult::RawLogits { logits } = &raw_logits {
                            raw_out_logits[seq_idx][i] = Some(logits.i(logit_idx)?);
                        } else if let ForwardInputsResult::Embeddings { embeddings } = &raw_logits {
                            embedding_logits[seq_idx] = Some(embeddings.i(logit_idx)?);
                        } else {
                            logits[seq_idx] = Some(raw_logits.index_bs(logit_idx)?);
                        }
                    }
                }

                {
                    // `post_op` is `CacheInstruction::Out` unconditionally on
                    // every decode step (`engine/mod.rs:397-404`), so
                    // `clone_out_cache` rebuilds B x (layers + compressor
                    // slots) per-sequence caches every token. This span is what
                    // turns that from a code reading into a number.
                    let _s = arc_profiler::span("cache.post_op");
                    match post_op {
                        CacheInstruction::Out => self.clone_out_cache(input_seqs),
                        CacheInstruction::Nothing => (),
                        CacheInstruction::Reset {
                            load_preallocated_cache,
                            reset_non_granular,
                        } => self.set_none_cache(
                            input_seqs,
                            reset_non_granular,
                            false,
                            load_preallocated_cache,
                        ),
                        _ => unreachable!("Unreachable POST cache op."),
                    }
                }

                if raw_out_logits[0][0].is_some() {
                    let start = Instant::now();
                    response::send_raw_responses(
                        input_seqs,
                        raw_out_logits
                            .into_iter()
                            .map(|raw| raw.into_iter().flatten().collect::<Vec<_>>())
                            .collect(),
                    )
                    .await?;
                    let end = Instant::now();
                    exec_duration += end.duration_since(start);

                    return Ok(exec_duration);
                }
                if embedding_logits[0].is_some() {
                    let start = Instant::now();
                    response::send_embedding_responses(
                        input_seqs,
                        embedding_logits
                            .into_iter()
                            .map(|raw| {
                                raw.unwrap()
                                    .to_dtype(DType::F32)
                                    .unwrap()
                                    .to_vec1::<f32>()
                                    .unwrap()
                            })
                            .collect(),
                    )
                    .await?;
                    let end = Instant::now();
                    exec_duration += end.duration_since(start);

                    return Ok(exec_duration);
                }

                let start = Instant::now();
                // Sampling, detokenisation, stop-string scanning and the serial
                // `responder.send().await` loop all live under here, and the
                // pipeline mutex is held for every one of them.
                let _s = arc_profiler::span("sample_and_dispatch");
                // Already on the host (one batched copy above) when B > 1.
                let logits = logits
                    .into_iter()
                    .map(|l| l.expect("Did not get any inputs. This is shocking."))
                    .collect::<Vec<_>>();

                match &logits[0] {
                    ForwardInputsResult::RawLogits { .. }
                    | ForwardInputsResult::Embeddings { .. } => unreachable!(),
                    ForwardInputsResult::CausalGeneration { .. } => {
                        self.sample_causal_gen(
                            input_seqs,
                            logits
                                .into_iter()
                                .map(|r| {
                                    #[allow(irrefutable_let_patterns)]
                                    let ForwardInputsResult::CausalGeneration { logits } = r
                                    else {
                                        unreachable!(
                                            "All results must have same type, `CausalGeneration`"
                                        )
                                    };
                                    logits
                                })
                                .collect::<Vec<_>>(),
                            prefix_cacher,
                            disable_eos_stop,
                            rng,
                        )
                        .await?;
                    }
                    ForwardInputsResult::Image { .. } => {
                        response::send_image_responses(
                            input_seqs,
                            logits
                                .into_iter()
                                .map(|r| {
                                    #[allow(irrefutable_let_patterns)]
                                    let ForwardInputsResult::Image { images } = r
                                    else {
                                        unreachable!("All results must have same type, `Image`")
                                    };
                                    images
                                        .into_iter()
                                        .next()
                                        .expect("Must have at least 1 element.")
                                })
                                .collect::<Vec<_>>(),
                        )
                        .await?;
                    }
                    ForwardInputsResult::Speech { .. } => {
                        let rates = logits
                            .iter()
                            .map(|r| {
                                #[allow(irrefutable_let_patterns)]
                                let ForwardInputsResult::Speech { rates, .. } = r
                                else {
                                    unreachable!("All results must have same type, `Speech`")
                                };
                                assert_eq!(rates.len(), 1, "Each sequence must have 1 PCM output.");
                                *rates.first().unwrap()
                            })
                            .collect::<Vec<_>>();
                        let channels = logits
                            .iter()
                            .map(|r| {
                                #[allow(irrefutable_let_patterns)]
                                let ForwardInputsResult::Speech { channels, .. } = r
                                else {
                                    unreachable!("All results must have same type, `Speech`")
                                };
                                assert_eq!(
                                    channels.len(),
                                    1,
                                    "Each sequence must have 1 PCM output."
                                );
                                *channels.first().unwrap()
                            })
                            .collect::<Vec<_>>();
                        let pcms = logits
                            .into_iter()
                            .map(|r| {
                                #[allow(irrefutable_let_patterns)]
                                let ForwardInputsResult::Speech { pcms, .. } = r
                                else {
                                    unreachable!("All results must have same type, `Speech`")
                                };
                                assert_eq!(pcms.len(), 1, "Each sequence must have 1 PCM output.");
                                pcms.into_iter().nth(0).unwrap()
                            })
                            .collect::<Vec<_>>();
                        response::send_speech_responses(input_seqs, &pcms, &rates, &channels)
                            .await?;
                    }
                }
                let end = Instant::now();
                exec_duration += end.duration_since(start);

                Ok(exec_duration)
            }
            CacheBackendMetadata::PagedAttention { metadata } => {
                // For hybrid models, build state_indices tensor from sequences'
                // recurrent_state_idx so recurrent layers are active during forward.
                // Paged attention manages KV caches separately, but recurrent state
                // pool access still needs the indices tensor to be set.
                if self.cache().is_hybrid() {
                    let mut hybrid_cache = self.cache().hybrid();
                    let recurrent_device = hybrid_cache.caches.iter().find_map(|c| {
                        if let HybridLayerCache::Recurrent(pool) = c {
                            Some(pool.device().clone())
                        } else {
                            None
                        }
                    });
                    if let Some(device) = recurrent_device {
                        #[allow(clippy::cast_possible_truncation)]
                        let indices: Vec<u32> = input_seqs
                            .iter()
                            .filter_map(|seq| seq.recurrent_state_idx().map(|idx| idx as u32))
                            .collect();
                        if indices.len() == input_seqs.len() {
                            if let Ok(si) = Tensor::from_vec(indices, (input_seqs.len(),), &device)
                            {
                                hybrid_cache.set_state_indices(Some(si));
                            }
                        }
                    }
                }

                let inputs_iter =
                    std::iter::once(self.get_processor().inputs_processor().process_inputs(
                        self.tokenizer(),
                        input_seqs,
                        is_prompt,
                        self.get_metadata().is_xlora,
                        &self.device(),
                        self.get_metadata().no_kv_cache,
                        None,
                        return_raw_logits,
                        self.get_input_processor_config(),
                        Some(metadata),
                        self.device_mapper(),
                    ));

                let mut logits = vec![None; input_seqs.len()];
                let len_inputs = 1;
                let mut raw_out_logits = vec![vec![None; len_inputs]; input_seqs.len()];
                let mut embedding_logits = vec![None; input_seqs.len()];

                let __pa_t_step_start = Instant::now();
                let mut __pa_t_fwd_us: f64 = 0.0;
                let mut exec_duration = Duration::ZERO;
                for (i, inputs) in inputs_iter.into_iter().enumerate() {
                    let InputProcessorOutput {
                        inputs,
                        seq_indices,
                    } = inputs.map_err(candle_core::Error::msg)?;

                    let start = Instant::now();
                    let raw_logits = {
                        let _s = arc_profiler::span("forward");
                        self.graph_wrapped_forward(inputs, is_prompt, return_raw_logits)?
                    };
                    let end = Instant::now();
                    exec_duration += end.duration_since(start);
                    __pa_t_fwd_us += end.duration_since(start).as_secs_f64() * 1e6;

                    // ONE host copy for the whole batch, *before* the
                    // per-sequence split. See `host_copy_batched_result`: the
                    // reverse order costs O(B²) D2H bytes per step. (This moves
                    // the logits D2H out of `STEP_us`' `sample` bucket and into
                    // `other`; it is the same work, attributed where it happens.)
                    let raw_logits = {
                        let _s = arc_profiler::sync_span("logits_d2h");
                        host_copy_batched_result(raw_logits, input_seqs.len(), &Device::Cpu)?
                    };

                    for (logit_idx, seq_idx) in seq_indices.into_iter().enumerate() {
                        if let ForwardInputsResult::RawLogits { logits } = &raw_logits {
                            raw_out_logits[seq_idx][i] = Some(logits.i(logit_idx)?);
                        } else if let ForwardInputsResult::Embeddings { embeddings } = &raw_logits {
                            embedding_logits[seq_idx] = Some(embeddings.i(logit_idx)?);
                        } else {
                            logits[seq_idx] = Some(raw_logits.index_bs(logit_idx)?);
                        }
                    }
                }

                if raw_out_logits[0][0].is_some() {
                    let start = Instant::now();
                    response::send_raw_responses(
                        input_seqs,
                        raw_out_logits
                            .into_iter()
                            .map(|raw| raw.into_iter().flatten().collect::<Vec<_>>())
                            .collect(),
                    )
                    .await?;
                    let end = Instant::now();
                    exec_duration += end.duration_since(start);

                    return Ok(exec_duration);
                }
                if embedding_logits[0].is_some() {
                    let start = Instant::now();
                    response::send_embedding_responses(
                        input_seqs,
                        embedding_logits
                            .into_iter()
                            .map(|raw| {
                                raw.unwrap()
                                    .to_dtype(DType::F32)
                                    .unwrap()
                                    .to_vec1::<f32>()
                                    .unwrap()
                            })
                            .collect(),
                    )
                    .await?;
                    let end = Instant::now();
                    exec_duration += end.duration_since(start);

                    return Ok(exec_duration);
                }

                let start = Instant::now();
                let _s = arc_profiler::span("sample_and_dispatch");
                // Already on the host (one batched copy above) when B > 1.
                let logits = logits
                    .into_iter()
                    .map(|l| l.expect("Did not get any inputs. This is shocking."))
                    .collect::<Vec<_>>();

                match &logits[0] {
                    ForwardInputsResult::RawLogits { .. }
                    | ForwardInputsResult::Embeddings { .. } => unreachable!(),
                    ForwardInputsResult::CausalGeneration { .. } => {
                        self.sample_causal_gen(
                            input_seqs,
                            logits
                                .into_iter()
                                .map(|r| {
                                    #[allow(irrefutable_let_patterns)]
                                    let ForwardInputsResult::CausalGeneration { logits } = r
                                    else {
                                        unreachable!("All results must have same type")
                                    };
                                    logits
                                })
                                .collect::<Vec<_>>(),
                            prefix_cacher,
                            disable_eos_stop,
                            rng,
                        )
                        .await?;
                    }
                    ForwardInputsResult::Image { .. } => {
                        response::send_image_responses(
                            input_seqs,
                            logits
                                .into_iter()
                                .map(|r| {
                                    #[allow(irrefutable_let_patterns)]
                                    let ForwardInputsResult::Image { images } = r
                                    else {
                                        unreachable!("All results must have same type, `Image`")
                                    };
                                    images
                                        .into_iter()
                                        .next()
                                        .expect("Must have at least 1 element.")
                                })
                                .collect::<Vec<_>>(),
                        )
                        .await?;
                    }
                    ForwardInputsResult::Speech { .. } => {
                        let rates = logits
                            .iter()
                            .map(|r| {
                                #[allow(irrefutable_let_patterns)]
                                let ForwardInputsResult::Speech { rates, .. } = r
                                else {
                                    unreachable!("All results must have same type, `Speech`")
                                };
                                assert_eq!(rates.len(), 1, "Each sequence must have 1 PCM output.");
                                *rates.first().unwrap()
                            })
                            .collect::<Vec<_>>();
                        let channels = logits
                            .iter()
                            .map(|r| {
                                #[allow(irrefutable_let_patterns)]
                                let ForwardInputsResult::Speech { channels, .. } = r
                                else {
                                    unreachable!("All results must have same type, `Speech`")
                                };
                                assert_eq!(
                                    channels.len(),
                                    1,
                                    "Each sequence must have 1 PCM output."
                                );
                                *channels.first().unwrap()
                            })
                            .collect::<Vec<_>>();
                        let pcms = logits
                            .into_iter()
                            .map(|r| {
                                #[allow(irrefutable_let_patterns)]
                                let ForwardInputsResult::Speech { pcms, .. } = r
                                else {
                                    unreachable!("All results must have same type, `Speech`")
                                };
                                assert_eq!(pcms.len(), 1, "Each sequence must have 1 PCM output.");
                                pcms.into_iter().nth(0).unwrap()
                            })
                            .collect::<Vec<_>>();
                        response::send_speech_responses(input_seqs, &pcms, &rates, &channels)
                            .await?;
                    }
                }
                let end = Instant::now();
                exec_duration += end.duration_since(start);
                let __pa_t_sample_us = end.duration_since(start).as_secs_f64() * 1e6;
                let __pa_t_total_us = __pa_t_step_start.elapsed().as_secs_f64() * 1e6;
                {
                    use std::sync::atomic::{AtomicU64, Ordering};
                    static N: AtomicU64 = AtomicU64::new(0);
                    static SUM_TOTAL: AtomicU64 = AtomicU64::new(0);
                    static SUM_FWD: AtomicU64 = AtomicU64::new(0);
                    static SUM_SAMPLE: AtomicU64 = AtomicU64::new(0);
                    let n = N.fetch_add(1, Ordering::Relaxed) + 1;
                    SUM_TOTAL.fetch_add(__pa_t_total_us as u64, Ordering::Relaxed);
                    SUM_FWD.fetch_add(__pa_t_fwd_us as u64, Ordering::Relaxed);
                    SUM_SAMPLE.fetch_add(__pa_t_sample_us as u64, Ordering::Relaxed);
                    if n > 4 && (n - 4) % 25 == 0 {
                        let nn = (n - 4) as f64;
                        let total = SUM_TOTAL.load(Ordering::Relaxed) as f64 / nn;
                        let fwd = SUM_FWD.load(Ordering::Relaxed) as f64 / nn;
                        let samp = SUM_SAMPLE.load(Ordering::Relaxed) as f64 / nn;
                        let other = total - fwd - samp;
                        tracing::info!(
                            "STEP_us avg over {} steps: TOTAL={:.0} fwd={:.0} sample={:.0} other={:.0} (=> {:.1} tok/s)",
                            n - 4, total, fwd, samp, other, 1e6 / total
                        );
                    }
                }

                Ok(exec_duration)
            }
        }
    }

    async fn sample_causal_gen(
        &self,
        seqs: &mut [&mut Sequence],
        logits: Vec<Tensor>,
        prefix_cacher: &mut PrefixCacheManagerV2,
        disable_eos_stop: bool,
        rng: Arc<std::sync::Mutex<Isaac64Rng>>,
    ) -> Result<(), candle_core::Error>;

    fn category(&self) -> ModelCategory;

    /// Return encoder cache hit/miss counters (hits, misses) if this pipeline has an encoder cache.
    fn encoder_cache_counters(&self) -> Option<(Arc<AtomicUsize>, Arc<AtomicUsize>)> {
        None
    }

    /// Return the components needed to construct an [`mtp_pipeline::MtpSpeculativePipeline`]
    /// over this pipeline. Default: `None` (no MTP support).
    ///
    /// `NormalPipeline` overrides to delegate to its underlying [`NormalModel`].
    fn mtp_decode_kit(&self) -> Option<mtp_pipeline::MtpDecodeKit> {
        None
    }
}

pub(crate) fn extract_logits(
    logits: &Tensor,
    context_lens: Vec<(usize, usize)>,
) -> candle_core::Result<Tensor> {
    let mut toks = Vec::new();
    for (dim, (start, len)) in logits.chunk(logits.dims()[0], 0)?.iter().zip(context_lens) {
        toks.push(dim.narrow(1, start, len)?);
    }
    Tensor::cat(&toks, 0)
}

#[cfg(test)]
mod tests {
    use crate::MessageContent;
    use either::Either;
    use indexmap::IndexMap;
    use serde_json::Value;

    macro_rules! hashmap {
        (@single $($x:tt)*) => (());
        (@count $($rest:expr),*) => (<[()]>::len(&[$(hashmap!(@single $rest)),*]));

        ($($key:expr => $value:expr,)+) => { hashmap!($($key => $value),+) };
        ($($key:expr => $value:expr),*) => {
            {
                let _cap = hashmap!(@count $($key),*);
                let mut _map = ::indexmap::IndexMap::with_capacity(_cap);
                $(
                    let _ = _map.insert($key, Value::String($value));
                )*
                _map
            }
        };
    }

    #[cfg(test)]
    #[track_caller]
    fn test_with_inputs(
        templates: &[(bool, &str, &str, &str, &str)],
        expected_outputs: &[&str],
        inputs: Vec<IndexMap<String, MessageContent>>,
    ) {
        use crate::pipeline::chat_template::ChatTemplateValue;

        use super::chat_template::apply_chat_template_to;
        let mut failed = Vec::new();
        let n_templates = templates.len();
        for ((has_system, bos, eos, unk, template), expected) in
            templates.iter().zip(expected_outputs)
        {
            let output = match apply_chat_template_to(
                if !has_system {
                    inputs[1..].to_vec()
                } else {
                    inputs.clone()
                },
                true,
                None,
                None, // reasoning_effort
                &ChatTemplateValue(Either::Left(template.to_string())),
                Some(bos.to_string()),
                Some(eos.to_string()),
                Some(unk.to_string()),
                Vec::new(),
            ) {
                Ok(v) => v,
                Err(e) => {
                    failed.push(format!("Failed with {e}."));
                    continue;
                }
            };
            if output != *expected {
                failed.push(format!(
                    "Expected: `{}` \n\nGot:      `{}`",
                    expected.replace('\n', "\\n"),
                    output.replace('\n', "\\n")
                ));
            }
        }
        if !failed.is_empty() {
            for (i, line) in failed.iter().enumerate() {
                println!("------------ Template {i} ------------");
                println!("{line}");
            }
            println!("------------------------");
            panic!("{}/{n_templates} chat templates failed.", failed.len());
        }
    }

    #[test]
    /// Generating these cases:
    /// ```py
    /// >>> t=transformers.AutoTokenizer.from_pretrained(...)
    /// # If non-system prompt model
    /// >>> t.apply_chat_template([{"role":"user","content":"Hello"},{"role":"assistant","content":"Hi there"},{"role":"user","content":"Who are you"},{"role":"assistant","content":"   I am an assistant   "},{"role":"user","content":"Another question"}], add_generation_prompt=True, tokenize=False)
    /// # If system prompt model
    /// >>> t.apply_chat_template([{"role":"system","content":"You are a helpful assistant"},{"role":"user","content":"Hello"},{"role":"assistant","content":"Hi there"},{"role":"user","content":"Who are you"},{"role":"assistant","content":"   I am an assistant   "},{"role":"user","content":"Another question"}], add_generation_prompt=True, tokenize=False)
    /// ```
    fn test_chat_templates() {
        let templates = [
            // ChatML: https://huggingface.co/teknium/OpenHermes-2.5-Mistral-7B
            (true, "<s>", "</s>", "<unk>", "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"),
            // mistralai/Mistral-7B-Instruct-v0.1
            (false, "<s>", "</s>", "<unk>", "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token + ' ' }}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}"),
            // meta-llama/Llama-2-13b-chat-hf
            (true, "<s>", "</s>", "<unk>", "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 and system_message != false %}{% set content = '<<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ bos_token + '[INST] ' + content.strip() + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' '  + content.strip() + ' ' + eos_token }}{% endif %}{% endfor %}"),
            // mistralai/Mixtral-8x7B-Instruct-v0.1
            (false, "<s>", "</s>", "<unk>", "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token}}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}"),
            // google/gemma-7b-it
            (false, "<bos>", "<eos>", "<unk>", "{{ bos_token }}{% if messages[0]['role'] == 'system' %}{{ raise_exception('System role not supported') }}{% endif %}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if (message['role'] == 'assistant') %}{% set role = 'model' %}{% else %}{% set role = message['role'] %}{% endif %}{{ '<start_of_turn>' + role + '\n' + message['content'] | trim + '<end_of_turn>\n' }}{% endfor %}{% if add_generation_prompt %}{{'<start_of_turn>model\n'}}{% endif %}"),
            // HuggingFaceM4/idefics2-8b-chatty
            (true, "<s>", "</s>", "<unk>", "{% for message in messages %}{{message['role'].capitalize()}}{% if message['content'][0]['type'] == 'image' %}{{':'}}{% else %}{{': '}}{% endif %}{% for line in message['content'] %}{% if line['type'] == 'text' %}{{line['text']}}{% elif line['type'] == 'image' %}{{ '<image>' }}{% endif %}{% endfor %}<end_of_utterance>\n{% endfor %}{% if add_generation_prompt %}{{ 'Assistant:' }}{% endif %}"),
        ];
        let expected_outputs = [
            // ChatML: https://huggingface.co/teknium/OpenHermes-2.5-Mistral-7B
            "<|im_start|>system\nYou are a helpful assistant<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\nHi there<|im_end|>\n<|im_start|>user\nWho are you<|im_end|>\n<|im_start|>assistant\n   I am an assistant   <|im_end|>\n<|im_start|>user\nAnother question<|im_end|>\n<|im_start|>assistant\n",
            // mistralai/Mistral-7B-Instruct-v0.1
            "<s>[INST] Hello [/INST]Hi there</s> [INST] Who are you [/INST]   I am an assistant   </s> [INST] Another question [/INST]",
            // meta-llama/Llama-2-13b-chat-hf
            "<s>[INST] <<SYS>>\nYou are a helpful assistant\n<</SYS>>\n\nHello [/INST] Hi there </s><s>[INST] Who are you [/INST] I am an assistant </s><s>[INST] Another question [/INST]",
            // mistralai/Mixtral-8x7B-Instruct-v0.1
            "<s>[INST] Hello [/INST]Hi there</s>[INST] Who are you [/INST]   I am an assistant   </s>[INST] Another question [/INST]",
            // google/gemma-7b-it
            "<bos><start_of_turn>user\nHello<end_of_turn>\n<start_of_turn>model\nHi there<end_of_turn>\n<start_of_turn>user\nWho are you<end_of_turn>\n<start_of_turn>model\nI am an assistant<end_of_turn>\n<start_of_turn>user\nAnother question<end_of_turn>\n<start_of_turn>model\n",
        ];
        let messages = [
            ["system", "You are a helpful assistant"],
            ["user", "Hello"],
            ["assistant", "Hi there"],
            ["user", "Who are you"],
            ["assistant", "   I am an assistant   "],
            ["user", "Another question"],
        ];
        let mut inputs = Vec::new();
        for [role, content] in messages {
            let mut message: IndexMap<String, Either<String, Vec<IndexMap<String, Value>>>> =
                IndexMap::new();
            message.insert("role".to_string(), Either::Left(role.to_string()));
            message.insert("content".to_string(), Either::Left(content.to_string()));
            inputs.push(message);
        }
        test_with_inputs(&templates, &expected_outputs, inputs);
    }

    #[test]
    /// Generating these cases:
    /// ```py
    /// >>> processor=transformers.AutoProcessor.from_pretrained(...)
    /// >>> processor.apply_chat_template([
    ///         {"role":"system","content":[{"type":"text", "text": "You are a helpful assistant"}]},
    ///         {"role":"user","content":[{"type":"image"}, {"type":"text", "text": "Hello, please describe the above."}]},
    ///         {"role":"assistant","content":[{"type":"text", "text": "Hi there"}]},
    ///         {"role":"user","content":[{"type":"text", "text": "Who are you"}]},
    ///         {"role":"assistant","content":[{"type":"text", "text": "   I am an assistant   "}]},
    ///         {"role":"user","content":[{"type":"text", "text": "Another question"}]}
    ///     ], add_generation_prompt=True, tokenize=False)
    /// ```
    fn test_image_chat_templates() {
        let templates = [
            // HuggingFaceM4/idefics2-8b-chatty
            (true, "<s>", "</s>", "<unk>", "{% for message in messages %}{{message['role'].capitalize()}}{% if message['content'][0]['type'] == 'image' %}{{':'}}{% else %}{{': '}}{% endif %}{% for line in message['content'] %}{% if line['type'] == 'text' %}{{line['text']}}{% elif line['type'] == 'image' %}{{ '<image>' }}{% endif %}{% endfor %}<end_of_utterance>\n{% endfor %}{% if add_generation_prompt %}{{ 'Assistant:' }}{% endif %}"),
        ];
        let expected_outputs = [
            // HuggingFaceM4/idefics2-8b-chatty
            "System: You are a helpful assistant<end_of_utterance>\nUser:<image>Hello, please describe the above.<end_of_utterance>\nAssistant: Hi there<end_of_utterance>\nUser:<image>This is me, who are you<end_of_utterance>\nAssistant:    I am an assistant   <end_of_utterance>\nUser:<image>Another question, what is this?<end_of_utterance>\nAssistant:",
        ];

        let mut inputs = Vec::new();

        let mut message: IndexMap<String, Either<String, Vec<IndexMap<String, Value>>>> =
            IndexMap::new();
        message.insert("role".to_string(), Either::Left("system".to_string()));
        message.insert(
            "content".to_string(),
            Either::Right(vec![hashmap! {
                "type".to_string() => "text".to_string(),
                "text".to_string() => "You are a helpful assistant".to_string()
            }]),
        );
        inputs.push(message);

        let mut message: IndexMap<String, Either<String, Vec<IndexMap<String, Value>>>> =
            IndexMap::new();
        message.insert("role".to_string(), Either::Left("user".to_string()));
        message.insert(
            "content".to_string(),
            Either::Right(vec![
                hashmap! {
                    "type".to_string() => "image".to_string()
                },
                hashmap! {
                    "type".to_string() => "text".to_string(),
                    "text".to_string() => "Hello, please describe the above.".to_string()
                },
            ]),
        );
        inputs.push(message);

        let mut message: IndexMap<String, Either<String, Vec<IndexMap<String, Value>>>> =
            IndexMap::new();
        message.insert("role".to_string(), Either::Left("assistant".to_string()));
        message.insert(
            "content".to_string(),
            Either::Right(vec![hashmap! {
                "type".to_string() => "text".to_string(),
                "text".to_string() => "Hi there".to_string()
            }]),
        );
        inputs.push(message);

        let mut message: IndexMap<String, Either<String, Vec<IndexMap<String, Value>>>> =
            IndexMap::new();
        message.insert("role".to_string(), Either::Left("user".to_string()));
        message.insert(
            "content".to_string(),
            Either::Right(vec![
                hashmap! {
                    "type".to_string() => "image".to_string()
                },
                hashmap! {
                    "type".to_string() => "text".to_string(),
                    "text".to_string() => "This is me, who are you".to_string()
                },
            ]),
        );
        inputs.push(message);

        let mut message: IndexMap<String, Either<String, Vec<IndexMap<String, Value>>>> =
            IndexMap::new();
        message.insert("role".to_string(), Either::Left("assistant".to_string()));
        message.insert(
            "content".to_string(),
            Either::Right(vec![hashmap! {
                "type".to_string() => "text".to_string(),
                "text".to_string() => "   I am an assistant   ".to_string()
            }]),
        );
        inputs.push(message);

        let mut message: IndexMap<String, Either<String, Vec<IndexMap<String, Value>>>> =
            IndexMap::new();
        message.insert("role".to_string(), Either::Left("user".to_string()));
        message.insert(
            "content".to_string(),
            Either::Right(vec![
                hashmap! {
                    "type".to_string() => "image".to_string()
                },
                hashmap! {
                    "type".to_string() => "text".to_string(),
                    "text".to_string() => "Another question, what is this?".to_string()
                },
            ]),
        );
        inputs.push(message);

        test_with_inputs(&templates, &expected_outputs, inputs);
    }
}

/// Regression tests for the per-step host copy of the batched logits.
///
/// The decode step used to slice the `[B, 1, vocab]` logits into B
/// per-sequence tensors and *then* move each one to the host. Because
/// `Tensor::i()` returns a view that keeps the whole batch storage alive, and
/// `Tensor::to_device` copies the whole storage, that moved `B * (B * vocab)`
/// elements over PCIe every decode step. These tests pin the fixed order and
/// prove the assertions can fail under the old one.
#[cfg(test)]
mod host_copy_tests {
    use super::*;
    use candle_core::{CpuStorage, Storage};
    use std::sync::atomic::Ordering;

    /// Number of elements candle would move across a device boundary for `t`:
    /// the size of the whole backing storage, NOT `t.elem_count()`.
    /// `Tensor::to_device` hands the *storage* to `to_cpu_storage()` and only
    /// carries the view's layout across (candle `d2d1d07`, `tensor.rs:2379`,
    /// `cuda_backend/mod.rs:1788`).
    fn storage_elems(t: &Tensor) -> usize {
        let (storage, _layout) = t.storage_and_layout();
        match &*storage {
            Storage::Cpu(CpuStorage::F32(v)) => v.len(),
            _ => panic!("fixture must be a CPU F32 tensor"),
        }
    }

    fn logits_of(r: &ForwardInputsResult) -> &Tensor {
        match r {
            ForwardInputsResult::CausalGeneration { logits } => logits,
            _ => panic!("fixture is CausalGeneration"),
        }
    }

    /// `0.0, 1.0, ... (n-1).0` without an integer→float cast.
    fn ramp(n: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(n);
        let mut x = 0f32;
        for _ in 0..n {
            out.push(x);
            x += 1.0;
        }
        out
    }

    /// `[b, 1, v]` holding `0..b*v`, so every row is distinguishable from
    /// every other row and from its own index.
    fn batch(b: usize, v: usize) -> ForwardInputsResult {
        ForwardInputsResult::CausalGeneration {
            logits: Tensor::from_vec(ramp(b * v), (b, 1, v), &Device::Cpu).unwrap(),
        }
    }

    #[test]
    fn batched_host_copy_is_o1_in_batch_while_the_old_order_was_o_b_squared() {
        // 7 is equal to none of the batch sizes used below and coprime with
        // all of them, so no assertion can pass by `b * v` aliasing `v`, `b`,
        // or `v * b` (DOCTRINE D12: the fixture must discriminate).
        const V: usize = 7;
        const BATCHES: [usize; 4] = [2, 5, 8, 64];

        // ── The premise, measured against candle rather than assumed ───────
        // A per-sequence slice is a VIEW: it addresses V elements but still
        // owns the whole B*V storage, so ONE `to_device` on it costs B*V.
        {
            let b = 5;
            assert_ne!(b, V, "fixture must not let B and V alias");
            let raw = batch(b, V);
            let row = raw.index_bs(2).unwrap();
            assert_eq!(
                logits_of(&row).elem_count(),
                V,
                "the view addresses exactly one row"
            );
            assert_eq!(
                storage_elems(logits_of(&row)),
                b * V,
                "...but candle would copy the WHOLE batch storage for it. If \
                 this ever fails, `Tensor::i()` started copying and the O(B^2) \
                 premise behind `host_copy_batched_result` is retired."
            );
        }

        // ── Fixed order: copy the batch once, then split ───────────────────
        for b in BATCHES {
            let before = LOGITS_HOST_COPIES.load(Ordering::Relaxed);
            let host = host_copy_batched_result(batch(b, V), b, &Device::Cpu).unwrap();
            let copies = LOGITS_HOST_COPIES.load(Ordering::Relaxed) - before;

            assert_eq!(
                copies, 1,
                "B={b}: one host copy per STEP, not one per sequence"
            );
            assert_eq!(
                storage_elems(logits_of(&host)),
                b * V,
                "B={b}: exactly one batch of elements crosses the boundary"
            );

            // The split must still hand each sequence its own row.
            let all = ramp(b * V);
            for k in 0..b {
                let row = host.index_bs(k).unwrap();
                assert_eq!(
                    logits_of(&row)
                        .flatten_all()
                        .unwrap()
                        .to_vec1::<f32>()
                        .unwrap(),
                    all[k * V..(k + 1) * V].to_vec(),
                    "B={b}: row {k} is wrong after the hoisted copy"
                );
            }
        }

        // ── MUTATION CONTROL: the pre-fix order, split then copy ───────────
        // Same counter, same helper, only the ORDER differs. This is what
        // proves `copies == 1` above is falsifiable rather than vacuous.
        for b in BATCHES {
            let raw = batch(b, V);
            let before = LOGITS_HOST_COPIES.load(Ordering::Relaxed);
            let mut old_elems = 0usize;
            for k in 0..b {
                let row = raw.index_bs(k).unwrap();
                old_elems += storage_elems(logits_of(&row));
                // `n_seqs = 2` only so the helper takes the copy branch; this
                // is the per-sequence `to_device` the fix removed.
                let _ = host_copy_batched_result(row, 2, &Device::Cpu).unwrap();
            }
            let old_copies = LOGITS_HOST_COPIES.load(Ordering::Relaxed) - before;

            assert_eq!(
                old_copies,
                u64::try_from(b).unwrap(),
                "pre-fix order: one host copy per SEQUENCE"
            );
            assert_ne!(
                old_copies, 1,
                "B={b}: the fixture must distinguish the two orders"
            );
            assert_eq!(old_elems, b * b * V, "pre-fix order moved O(B^2) elements");
            assert_eq!(
                old_elems / (b * V),
                b,
                "B={b}: the fix is worth exactly a factor of B in D2H bytes"
            );
        }

        // ── A single sequence must NOT be pulled to the host ───────────────
        // `Sampler::sample`'s GPU fast path is gated on
        // `!logits.device().is_cpu()`; copying a B=1 batch would disable it.
        let before = LOGITS_HOST_COPIES.load(Ordering::Relaxed);
        let one = host_copy_batched_result(batch(1, V), 1, &Device::Cpu).unwrap();
        assert_eq!(
            LOGITS_HOST_COPIES.load(Ordering::Relaxed) - before,
            0,
            "B=1 CausalGeneration must stay on the device"
        );
        assert_eq!(logits_of(&one).dims(), &[1, 1, V]);

        // ...but raw-logit and embedding requests always come back, at any B.
        for raw in [
            ForwardInputsResult::RawLogits {
                logits: Tensor::zeros((1, 1, V), DType::F32, &Device::Cpu).unwrap(),
            },
            ForwardInputsResult::Embeddings {
                embeddings: Tensor::zeros((1, V), DType::F32, &Device::Cpu).unwrap(),
            },
        ] {
            let before = LOGITS_HOST_COPIES.load(Ordering::Relaxed);
            host_copy_batched_result(raw, 1, &Device::Cpu).unwrap();
            assert_eq!(
                LOGITS_HOST_COPIES.load(Ordering::Relaxed) - before,
                1,
                "raw logits / embeddings always return to the host"
            );
        }
    }
}
