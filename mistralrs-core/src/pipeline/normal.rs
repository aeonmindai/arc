use super::isq::ImatrixDataSource;
use super::llg::build_llg_factory;
use super::{
    get_model_paths, get_xlora_paths, text_models_inputs_processor::ModelInputs, AdapterKind,
    CacheManager, GeneralMetadata, Loader, ModelKind, ModelPaths, NormalModel, NormalModelLoader,
    TokenSource,
};
use super::{
    AnyMoePipelineMixin, CacheManagerMixin, EitherCache, ForwardInputsResult, IsqOrganization,
    IsqPipelineMixin, MetadataMixin, ModelCategory, PreProcessingMixin,
};
use super::{
    AutoNormalLoader, DeepSeekV2Loader, DeepSeekV3Loader, DeepSeekV4Loader, GLM4Loader,
    GLM4MoeLiteLoader, GLM4MoeLoader, Gemma2Loader, GemmaLoader, GptOssLoader,
    GraniteMoeHybridLoader, LlamaLoader, MistralLoader, MixtralLoader, NormalLoaderType,
    Phi2Loader, Phi3Loader, Phi3_5MoELoader, Qwen2Loader, Qwen3Loader, Qwen3MoELoader,
    Qwen3NextLoader, SmolLm3Loader, Starcoder2Loader,
};
use crate::amoe::AnyMoeExpertType;
use crate::attention::ATTENTION_CHUNK_SIZE;
use crate::device_map::{self, DeviceMapper};
use crate::distributed::{self, WorkerTransferData};
use crate::kv_cache::{FullCacheManager, HybridCacheManager, NormalCacheManager};
use crate::lora::Ordering;
use crate::paged_attention::{calculate_cache_config, AttentionImplementation, CacheEngine};
use crate::pipeline::chat_template::{calculate_eos_tokens, GenerationConfig};
use crate::pipeline::isq::{UqffFullSer, UqffSourceWeights};
use crate::pipeline::loaders::auto_device_map;
use crate::pipeline::loaders::QuantizationConfigShim;
use crate::pipeline::sampling::sample_and_add_toks;
use crate::pipeline::text_models_inputs_processor::make_prompt_chunk;
use crate::pipeline::{get_chat_template, Modalities, SupportedModality};
use crate::pipeline::{ChatTemplate, LocalModelPaths};
use crate::prefix_cacher::PrefixCacheManagerV2;
use crate::sequence::Sequence;
use crate::utils::tokenizer::get_tokenizer;
use crate::utils::varbuilder_utils::DeviceForLoadTensor;
use crate::utils::{
    progress::{new_multi_progress, ProgressScopeGuard},
    tokens::get_token,
    varbuilder_utils::from_mmaped_safetensors,
};
use crate::xlora_models::NonGranularState;
use crate::{
    api_dir_list, api_get_file, get_mut_arcmutex, get_paths, get_uqff_paths, lora_model_loader,
    normal_model_loader, normal_model_loader_sharded, xlora_model_loader, DeviceMapSetting,
    PagedAttentionConfig, Pipeline, Topology, TryIntoDType, GLOBAL_HF_CACHE,
};
use anyhow::Result;
use candle_core::{Device, Tensor, Var};
use hf_hub::Cache;
use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};
use mistralrs_quant::log::once_log_info;
use mistralrs_quant::{
    AfqLayer, GgufMatMul, HqqLayer, ImmediateIsqOverride, IsqType, QuantizedSerdeType,
};
use rand_isaac::Isaac64Rng;
use regex_automata::meta::Regex;
use std::any::Any;
use std::borrow::Cow;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::{Arc, RwLock};
use std::time::Instant;
use std::{env, fs};
use tokenizers::Tokenizer;
use tokio::sync::Mutex;
use tracing::{info, warn};

pub struct NormalPipeline {
    model: Box<dyn NormalModel + Send + Sync>,
    tokenizer: Arc<Tokenizer>,
    no_kv_cache: bool,
    chat_template: Arc<ChatTemplate>,
    non_granular_state: Option<NonGranularState>,
    model_id: String,
    metadata: Arc<GeneralMetadata>,
    topology: Option<Topology>,
    silent: bool,
    organization: IsqOrganization,
    // For full UQFF serialization
    template_filename: Option<PathBuf>,
    generation_config: Option<PathBuf>,
    config: String,
    imatrix: Option<PathBuf>,
    mapper: Box<dyn DeviceMapper + Send + Sync>,
    #[cfg(feature = "cuda")]
    cuda_graph_runner: Option<arc_cuda_graph::CudaGraphRunner>,
    #[cfg(feature = "cuda")]
    _capturable_device: Option<Device>,
    /// Dedicated decode path — bypasses Candle, runs on its own non-blocking stream.
    #[cfg(feature = "cuda")]
    dedicated_decode: Option<arc_cuda_graph::DedicatedDecodePath>,
    /// Autonomous decode runner — runs the full decode loop on GPU (forward →
    /// sample → step → check_done) with zero CPU sync per token. Allocated at
    /// load time; graph capture is deferred until first decode call.
    #[cfg(feature = "cuda")]
    autonomous_runner: Option<arc_cuda_graph::AutonomousDecodeRunner>,
}

/// What a forward pass should do with candle's caching allocator before it
/// runs. See [`alloc_cache_action`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AllocCacheAction {
    /// Turn the cache on. Frees start being recycled instead of returned to
    /// the driver.
    Enable,
    /// Return every held buffer to the driver and turn the cache off.
    DrainAndDisable,
    /// Already in the right state; touch nothing.
    Leave,
}

/// Decide whether candle's caching allocator should be on for this forward.
///
/// # Why this is not simply "on"
///
/// The cache (fork `88d86a2`, `candle-core/src/cuda_backend/device.rs:42`) is
/// keyed on **exact byte count** — `free: HashMap<usize, Vec<CUdeviceptr>>`,
/// looked up with `free.get_mut(&bytes)`. No bucketing, no smallest-fit, no
/// splitting. It also has **no capacity bound and no eviction**: the only ways
/// memory goes back to the driver are `set_alloc_cache_enabled(false)` and
/// `drain_alloc_cache_and_free()`.
///
/// Those two facts together decide the policy:
///
/// * **Decode wants it on.** A decode step allocates the same shapes 43 times
///   over (once per layer), and the shapes that do not depend on KV length —
///   hidden states, MLP and expert intermediates — repeat step after step.
///   Exact-size keying is a perfect fit for that traffic.
/// * **Prefill must drain it.** Prefill shapes scale with prompt length, are
///   large, and are hit once. Leaving the cache on across a prefill parks
///   those buffers for the process lifetime under a byte-size key nothing will
///   ever request again. That is the failure mode `ARC_NO_DEDICATED_DECODE`
///   already exists to work around, and it is why this returns
///   [`AllocCacheAction::DrainAndDisable`] rather than `Leave` on `seq_len != 1`.
///
/// # The measurement this is waiting on
///
/// Buffers whose size tracks KV length — the causal mask most obviously —
/// change size every decode step, so each step files one more never-reused
/// entry. That is bounded by `O(context^2)` bytes in the worst case and is
/// *not* bounded by this policy. It is small at short context (a `[1,1,1,kv]`
/// BF16 mask over 4k tokens sums to ~16 MB) and is not small at 128k.
///
/// The fixed-capacity graph-mode path (`deepseek4.rs:4344-4353`, which swaps
/// the growing causal mask for `graph_mode_length_mask` at
/// `cfg_full.sliding_window`) removes that growth entirely, but it is reached
/// only under `ARC_V4_CAPTURE_PROBE`. Until the shape-invariance work lands,
/// `ARC_CANDLE_ALLOC_CACHE=0` is the kill switch, and the long-context
/// high-water mark is the number a GPU run should falsify this with.
///
/// # Contract
///
/// * `seq_len` — the forward's sequence length. `1` is decode.
/// * `enabled` — what candle reports *now* (`alloc_cache_enabled()`), so the
///   action is idempotent and we never drain a cache that is already off.
/// * `killed` — `ARC_CANDLE_ALLOC_CACHE=0` was set.
///
/// Pure, so the policy is testable without a GPU — which matters, because the
/// allocator itself has no tests at all in either repo.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) fn alloc_cache_action(
    seq_len: usize,
    enabled: bool,
    killed: bool,
) -> AllocCacheAction {
    let want = !killed && seq_len == 1;
    match (want, enabled) {
        (true, false) => AllocCacheAction::Enable,
        (false, true) => AllocCacheAction::DrainAndDisable,
        _ => AllocCacheAction::Leave,
    }
}

/// `ARC_CANDLE_ALLOC_CACHE=0` turns the caching allocator off entirely.
///
/// Any other value — and *unset* — leaves the default policy in force. The
/// variable used to be the on-switch, so the value `1` the ops scripts pass
/// (`arc-tools/arcgraph_heap_probe.sh:191`) still means what it always meant.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) fn alloc_cache_killed() -> bool {
    std::env::var("ARC_CANDLE_ALLOC_CACHE").is_ok_and(|v| v == "0")
}

/// Retention cap for the caching allocator, in bytes, or `None` to leave
/// candle's own default (1 GiB) in force.
///
/// `ARC_ALLOC_CACHE_MAX_MB=0` means unbounded — the pre-bounding behaviour, kept
/// reachable so the leak it causes can be A/B'd rather than argued about.
/// Measured unbounded on V4: **+6.04 MiB per decoded token with no plateau**,
/// against +0.057 MiB/token with the cache off entirely.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) fn alloc_cache_capacity_bytes() -> Option<usize> {
    parse_alloc_cache_capacity(std::env::var("ARC_ALLOC_CACHE_MAX_MB").ok().as_deref())
}

/// Pure half of [`alloc_cache_capacity_bytes`], so the parse is testable.
///
/// `None` (unset, or unparsable) leaves candle's own default in force rather
/// than silently picking a different one — a typo in an ops script must not
/// quietly hand back the unbounded allocator this change exists to remove.
pub(crate) fn parse_alloc_cache_capacity(raw: Option<&str>) -> Option<usize> {
    let mb: usize = raw?.trim().parse().ok()?;
    Some(if mb == 0 {
        usize::MAX
    } else {
        mb.saturating_mul(1024 * 1024)
    })
}

/// Is `name` set to a value that means **yes**?
///
/// # This exists because presence-testing an on/off flag is a live bug class
///
/// `ARC_NO_DEDICATED_DECODE` was read as `var_os(..).is_some()`, so
/// `ARC_NO_DEDICATED_DECODE=0` — the spelling any reader would take for "off" —
/// *disabled* the dedicated decode path. Two A/B harnesses passed exactly that
/// as their control leg (`arc-tools/arcgraph_heap_probe.sh`,
/// `arc-tools/arcgraph_capture_probe.sh`), so both arms of both experiments ran
/// with the path off and the variable under test cancelled out of the result.
/// The experiments did not fail; they returned a difference of zero, honestly
/// measured, for a comparison that was never made.
///
/// So flags are read by *value*, never by presence, and an unrecognised value
/// is reported rather than silently bucketed — a typo in an ops script must not
/// decide an experiment.
///
/// `1`, `true`, `yes`, `on` (any case, surrounding whitespace ignored) are yes.
/// Unset, empty, `0`, `false`, `no`, `off` are no.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) fn env_flag_is_set(name: &str) -> bool {
    let raw = std::env::var(name).ok();
    match env_flag_value(raw.as_deref()) {
        Some(v) => v,
        None => {
            tracing::warn!(
                "{name}={:?} is not a recognised on/off value (use 1/0) — treating it as OFF. \
                 Note that {name} is an opt-OUT: 1 disables, 0 and unset leave it enabled.",
                raw.unwrap_or_default()
            );
            false
        }
    }
}

/// Pure half of [`env_flag_is_set`], so the polarity is testable without
/// mutating process environment.
///
/// `None` means "value present but not recognised" — the caller decides what to
/// do with that, and must not silently fold it into `false`.
pub(crate) fn env_flag_value(raw: Option<&str>) -> Option<bool> {
    let Some(raw) = raw else {
        return Some(false);
    };
    match raw.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Some(true),
        "" | "0" | "false" | "no" | "off" => Some(false),
        _ => None,
    }
}

/// Emit the allocator's counters every `ARC_ALLOC_CACHE_STATS` decode steps.
///
/// The counters are what this cache has to be judged on. A green log — the
/// server did not crash, the output looked fine — is not evidence that the
/// cache is working: an earlier arena in this codebase reported "accounting OK"
/// and bit-identical output across 52 steps while silently bypassing itself for
/// every buffer under 128 bytes (`KERNEL_RULES.md:977-984`). What distinguishes
/// a working bounded cache from that is arithmetic, and it is printed here:
///
/// * `alloc/step` — real `cuMemAllocAsync` calls per decode step. Low means the
///   cache is absorbing the ~11 k allocations a step makes.
/// * `free/step` — real `cuMemFreeAsync` calls per decode step. **Non-zero is
///   the point.** The unbounded cache's was exactly zero, forever, which is why
///   it grew without bound.
/// * `held` — bytes retained right now, against the cap.
#[cfg(feature = "cuda")]
fn report_alloc_cache_step(cd: &candle_core::CudaDevice, seq_len: usize) {
    use std::sync::atomic::{AtomicU64, Ordering};
    static EVERY: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    let every = *EVERY.get_or_init(|| {
        std::env::var("ARC_ALLOC_CACHE_STATS")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .unwrap_or(0)
    });
    if every == 0 || seq_len != 1 {
        return;
    }
    static STEP: AtomicU64 = AtomicU64::new(0);
    static LAST_ALLOC: AtomicU64 = AtomicU64::new(0);
    static LAST_FREE: AtomicU64 = AtomicU64::new(0);
    static LAST_STEP: AtomicU64 = AtomicU64::new(0);
    let step = STEP.fetch_add(1, Ordering::Relaxed) + 1;
    if step % every != 0 {
        return;
    }
    let s = cd.alloc_cache_stats();
    let d_step = step - LAST_STEP.swap(step, Ordering::Relaxed);
    let d_alloc = s.misses - LAST_ALLOC.swap(s.misses, Ordering::Relaxed);
    let d_free = s.frees() - LAST_FREE.swap(s.frees(), Ordering::Relaxed);
    let n = d_step.max(1) as f64;
    tracing::info!(
        "[alloc-cache] step {step}  alloc/step {:.1}  free/step {:.1}  \
         hit-rate {:.4}  held {:.1} MiB / cap {}  high-water {:.1} MiB  sizes {}",
        d_alloc as f64 / n,
        d_free as f64 / n,
        s.hits as f64 / (s.hits + s.misses).max(1) as f64,
        s.cached_bytes as f64 / (1024.0 * 1024.0),
        if s.capacity_bytes == usize::MAX {
            "unbounded".to_string()
        } else {
            format!("{} MiB", s.capacity_bytes / (1024 * 1024))
        },
        s.high_water_bytes as f64 / (1024.0 * 1024.0),
        s.size_classes,
    );
}

/// A loader for a "normal" (non-quantized) model.
pub struct NormalLoader {
    inner: Box<dyn NormalModelLoader>,
    model_id: String,
    config: NormalSpecificConfig,
    xlora_model_id: Option<String>,
    lora_adapter_ids: Option<Vec<String>>,
    kind: ModelKind,
    xlora_order: Option<Ordering>,
    no_kv_cache: bool,
    chat_template: Option<String>,
    tokenizer_json: Option<String>,
    tgt_non_granular_index: Option<usize>,
    token_source: RwLock<Option<TokenSource>>,
    revision: RwLock<Option<String>>,
    from_uqff: RwLock<Option<Vec<PathBuf>>>,
    jinja_explicit: Option<String>,
    hf_cache_path: Option<PathBuf>,
}

#[derive(Default)]
/// A builder for a loader for a "normal" (non-quantized) model.
pub struct NormalLoaderBuilder {
    model_id: Option<String>,
    config: NormalSpecificConfig,
    xlora_model_id: Option<String>,
    lora_adapter_ids: Option<Vec<String>>,
    kind: ModelKind,
    xlora_order: Option<Ordering>,
    no_kv_cache: bool,
    chat_template: Option<String>,
    tokenizer_json: Option<String>,
    tgt_non_granular_index: Option<usize>,
    jinja_explicit: Option<String>,
    hf_cache_path: Option<PathBuf>,
}

#[derive(Clone, Default)]
/// Config specific to loading a normal model.
pub struct NormalSpecificConfig {
    pub topology: Option<Topology>,
    pub organization: IsqOrganization,
    pub write_uqff: Option<PathBuf>,
    pub from_uqff: Option<Vec<PathBuf>>,
    pub imatrix: Option<PathBuf>,
    pub calibration_file: Option<PathBuf>,
    pub hf_cache_path: Option<PathBuf>,
    pub matformer_config_path: Option<PathBuf>,
    pub matformer_slice_name: Option<String>,
}

impl NormalLoaderBuilder {
    pub fn new(
        config: NormalSpecificConfig,
        chat_template: Option<String>,
        tokenizer_json: Option<String>,
        model_id: Option<String>,
        no_kv_cache: bool,
        jinja_explicit: Option<String>,
    ) -> Self {
        Self {
            config,
            chat_template,
            tokenizer_json,
            model_id,
            kind: ModelKind::Normal,
            jinja_explicit,
            no_kv_cache,
            ..Default::default()
        }
    }

    fn with_adapter(
        mut self,
        xlora_model_id: String,
        xlora_order: Ordering,
        no_kv_cache: bool,
        tgt_non_granular_index: Option<usize>,
    ) -> Self {
        self.xlora_model_id = Some(xlora_model_id);
        self.xlora_order = Some(xlora_order);
        self.no_kv_cache = no_kv_cache;
        self.tgt_non_granular_index = tgt_non_granular_index;
        self.model_id = if let Some(id) = self.model_id {
            Some(id)
        } else {
            info!(
                "Using adapter base model ID: `{}`",
                self.xlora_order.as_ref().unwrap().base_model_id
            );
            Some(self.xlora_order.as_ref().unwrap().base_model_id.clone())
        };
        self
    }

    pub fn with_xlora(
        mut self,
        xlora_model_id: String,
        xlora_order: Ordering,
        no_kv_cache: bool,
        tgt_non_granular_index: Option<usize>,
    ) -> Self {
        self.kind = ModelKind::Adapter {
            adapter: AdapterKind::XLora,
        };
        self.with_adapter(
            xlora_model_id,
            xlora_order,
            no_kv_cache,
            tgt_non_granular_index,
        )
    }

    pub fn with_lora(mut self, lora_adapter_ids: Vec<String>) -> Self {
        self.kind = ModelKind::Adapter {
            adapter: AdapterKind::Lora,
        };
        self.lora_adapter_ids = Some(lora_adapter_ids);
        self
    }

    pub fn hf_cache_path(mut self, hf_cache_path: PathBuf) -> Self {
        self.hf_cache_path = Some(hf_cache_path);
        self
    }

    /// If the loader type is not specified, loader type is automatically determined from the
    /// `architectures` array in the config.
    pub fn build(self, loader_tp: Option<NormalLoaderType>) -> anyhow::Result<Box<dyn Loader>> {
        let loader: Box<dyn NormalModelLoader> = match loader_tp {
            Some(NormalLoaderType::Mistral) => Box::new(MistralLoader),
            Some(NormalLoaderType::Gemma) => Box::new(GemmaLoader),
            Some(NormalLoaderType::Llama) => Box::new(LlamaLoader),
            Some(NormalLoaderType::Mixtral) => Box::new(MixtralLoader),
            Some(NormalLoaderType::Phi2) => Box::new(Phi2Loader),
            Some(NormalLoaderType::Phi3) => Box::new(Phi3Loader),
            Some(NormalLoaderType::Qwen2) => Box::new(Qwen2Loader),
            Some(NormalLoaderType::Gemma2) => Box::new(Gemma2Loader),
            Some(NormalLoaderType::Starcoder2) => Box::new(Starcoder2Loader),
            Some(NormalLoaderType::Phi3_5MoE) => Box::new(Phi3_5MoELoader),
            Some(NormalLoaderType::DeepSeekV2) => Box::new(DeepSeekV2Loader),
            Some(NormalLoaderType::DeepSeekV3) => Box::new(DeepSeekV3Loader),
            // V4, KimiK2 (text-side = DeepSeek V3), and GLM-5 DSA all dispatch
            // through V3-derived loaders at Tier A. Per-architecture quirks
            // (V4's CSA/HCA, GLM-5's DSA backend) are wired in via the attention
            // dispatcher when the model graph is constructed.
            Some(NormalLoaderType::DeepSeekV4) => Box::new(DeepSeekV4Loader),
            Some(NormalLoaderType::KimiK2) => Box::new(DeepSeekV3Loader),
            Some(NormalLoaderType::GLM5MoeDsa) => Box::new(GLM4MoeLoader),
            Some(NormalLoaderType::Qwen3) => Box::new(Qwen3Loader),
            Some(NormalLoaderType::GLM4) => Box::new(GLM4Loader),
            Some(NormalLoaderType::GLM4MoeLite) => Box::new(GLM4MoeLiteLoader),
            Some(NormalLoaderType::GLM4Moe) => Box::new(GLM4MoeLoader),
            Some(NormalLoaderType::Qwen3Moe) => Box::new(Qwen3MoELoader),
            Some(NormalLoaderType::SmolLm3) => Box::new(SmolLm3Loader),
            Some(NormalLoaderType::GraniteMoeHybrid) => Box::new(GraniteMoeHybridLoader),
            Some(NormalLoaderType::GptOss) => Box::new(GptOssLoader),
            Some(NormalLoaderType::Qwen3Next) => Box::new(Qwen3NextLoader),
            None => Box::new(AutoNormalLoader),
        };
        Ok(Box::new(NormalLoader {
            inner: loader,
            model_id: self.model_id.unwrap(),
            config: self.config,
            xlora_model_id: self.xlora_model_id,
            lora_adapter_ids: self.lora_adapter_ids,
            kind: self.kind,
            xlora_order: self.xlora_order,
            no_kv_cache: self.no_kv_cache,
            chat_template: self.chat_template,
            tokenizer_json: self.tokenizer_json,
            tgt_non_granular_index: self.tgt_non_granular_index,
            jinja_explicit: self.jinja_explicit,
            token_source: RwLock::new(None),
            revision: RwLock::new(None),
            from_uqff: RwLock::new(None),
            hf_cache_path: self.hf_cache_path,
        }))
    }
}

/// Tolerance for accepting a CUDA-graph replay's logits as equal to the eager
/// forward's. The two run the SAME kernels on the SAME inputs in the SAME
/// order, so they should agree bit-for-bit; this is deliberately tight rather
/// than a "close enough" band, because the failure this guards against —
/// a graph reading a stale device address — produces plausible logits, not
/// wildly wrong ones. A loose tolerance would wave exactly that through.
#[cfg(feature = "cuda")]
const GRAPH_REPLAY_TOLERANCE: f32 = 1e-4;

/// Max absolute elementwise difference between two logit tensors.
///
/// Costs one device→host sync of a single scalar, and only runs on the handful
/// of verification steps, never on the trusted path.
#[cfg(feature = "cuda")]
thread_local! {
    /// The EAGER logits of the step the CUDA graph was captured on, kept so a
    /// later replay can be measured against them as well as against its own
    /// step. A replay that reproduces these is a graph frozen at capture — its
    /// static input buffers are not being read. One that matches neither them
    /// nor the current step is partially advancing, which names a subset of
    /// stale inputs rather than the whole set. (RUN-161)
    static ARC_CAPTURE_STEP_EAGER: std::cell::RefCell<Option<Tensor>> =
        const { std::cell::RefCell::new(None) };
    /// How many replays past the capture step we are, for the same probe.
    static ARC_REPLAY_DISTANCE: std::cell::RefCell<u64> = const { std::cell::RefCell::new(0) };
    /// Handles to the per-layer outputs the CAPTURED forward produced. The
    /// recorded kernels write to these same addresses on every replay, so
    /// after a replay these hold the replay's per-layer values and can be
    /// bisected against an eager forward's. See `layers::arc_layer_trace_*`.
    static ARC_GRAPH_LAYER_BUFS: std::cell::RefCell<Option<Vec<Tensor>>> =
        const { std::cell::RefCell::new(None) };
}

/// Is the per-layer graph-vs-eager bisect enabled?
fn arc_layer_bisect() -> bool {
    std::env::var("ARC_GRAPH_LAYER_BISECT").as_deref() == Ok("1")
}

fn max_abs_diff(a: &Tensor, b: &Tensor) -> candle_core::Result<f32> {
    if a.dims() != b.dims() {
        candle_core::bail!(
            "shape mismatch: replay {:?} vs eager {:?}",
            a.dims(),
            b.dims()
        );
    }
    a.to_dtype(candle_core::DType::F32)?
        .sub(&b.to_dtype(candle_core::DType::F32)?)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()
}

impl Loader for NormalLoader {
    #[allow(clippy::type_complexity, clippy::too_many_arguments)]
    fn load_model_from_hf(
        &self,
        revision: Option<String>,
        token_source: TokenSource,
        dtype: &dyn TryIntoDType,
        device: &Device,
        silent: bool,
        mapper: DeviceMapSetting,
        in_situ_quant: Option<IsqType>,
        paged_attn_config: Option<PagedAttentionConfig>,
    ) -> Result<Arc<Mutex<dyn Pipeline + Send + Sync>>> {
        let _progress_guard = ProgressScopeGuard::new(silent);
        let cache = self
            .hf_cache_path
            .clone()
            .map(Cache::new)
            .unwrap_or_default();
        GLOBAL_HF_CACHE.get_or_init(|| cache);

        let paths: anyhow::Result<Box<dyn ModelPaths>> = get_paths!(
            LocalModelPaths,
            &token_source,
            revision.clone(),
            self,
            None,
            None,
            silent,
            self.config.from_uqff.is_some()
        );
        if let Some(from_uqff) = self.config.from_uqff.clone() {
            *self.from_uqff.write().unwrap() = Some(get_uqff_paths!(&from_uqff, self, silent));
        }
        *self
            .token_source
            .write()
            .expect("Failed to write to token source") = Some(token_source);
        *self.revision.write().expect("Failed to write to revision") = revision;
        self.load_model_from_path(
            &paths?,
            dtype,
            device,
            silent,
            mapper,
            in_situ_quant,
            paged_attn_config,
        )
    }

    #[allow(clippy::type_complexity, clippy::too_many_arguments)]
    fn load_model_from_path(
        &self,
        paths: &Box<dyn ModelPaths>,
        dtype: &dyn TryIntoDType,
        device: &Device,
        silent: bool,
        mut mapper: DeviceMapSetting,
        in_situ_quant: Option<IsqType>,
        mut paged_attn_config: Option<PagedAttentionConfig>,
    ) -> Result<Arc<Mutex<dyn Pipeline + Send + Sync>>> {
        let _progress_guard = ProgressScopeGuard::new(silent);
        let config = std::fs::read_to_string(paths.get_config_filename())?;

        if !self.inner.supports_paged_attention(&config)? {
            paged_attn_config = None;
        }

        info!("Prompt chunk size is {ATTENTION_CHUNK_SIZE}.");

        let use_nccl = mistralrs_quant::distributed::use_nccl();

        let available_devices = if let Ok(payload) = env::var(distributed::IS_DAEMON_FLAG) {
            let payload: WorkerTransferData = serde_json::from_str(&payload)?;
            let WorkerTransferData::Init { id: _, worker_rank } = payload;
            vec![candle_core::Device::new_cuda(worker_rank + 1)?]
        } else if use_nccl {
            vec![candle_core::Device::new_cuda(0)?]
        } else {
            device_map::get_all_similar_devices(device)?
        };
        #[cfg(feature = "cuda")]
        for device in &available_devices {
            if let Device::Cuda(dev) = device {
                unsafe { dev.disable_event_tracking() };
            }
        }
        let device = if use_nccl || cfg!(feature = "ring") {
            available_devices[0].clone()
        } else {
            device.clone()
        };

        // If auto, convert to Map if not using nccl
        let mut max_kv_tokens: Option<usize> = None;
        if use_nccl || cfg!(feature = "ring") {
            mapper = DeviceMapSetting::DummyNccl {
                nm_device: available_devices[0].clone(),
            };
        } else if let DeviceMapSetting::Auto(params) = mapper.clone() {
            max_kv_tokens = Some(params.max_seq_len() * params.max_batch_size());
            // Initial dtype
            let dtype = dtype.try_into_dtype(&available_devices.iter().collect::<Vec<_>>())?;

            // ISQ or UQFF: quantized path
            // Match logic below where UQFF has priority
            let (layer_sizes_in_bytes, non_mapped_size_in_bytes, total_model_size_in_bytes) =
                if let Some(serialized) = &*self.from_uqff.read().unwrap() {
                    let weight_pack_factor = {
                        let ser_artifacts = unsafe {
                            candle_core::safetensors::MmapedSafetensors::multi(serialized)?
                        };
                        // Size-weighted average pack factor. A plain count
                        // average is dragged toward 1 by the many tiny
                        // unquantized tensors (norms, router, mHC params,
                        // pack_factor=1), which over-sizes an expert-dominated
                        // model (e.g. V4 Flash) ~1.6x and spuriously offloads
                        // layers to CPU. Weight each tensor's pack_factor by its
                        // quantized byte size so the large QTIP tensors dominate.
                        // (RUN-161)
                        let mut weighted_pack = 0usize;
                        let mut total_len = 0usize;
                        for (_, artifact) in ser_artifacts.tensors() {
                            let artifact = artifact.data();
                            // NOTE(EricLBuehler): isq type is ALWAYS byte 4 (5th) of the tensor.
                            let isq_type = artifact[mistralrs_quant::UQFF_QUANT_TYPE_OFFSET];
                            let pack_factor = match QuantizedSerdeType::try_from(isq_type as usize)?
                            {
                                QuantizedSerdeType::Hqq => {
                                    HqqLayer::get_isq_type_from_uqff(Cow::Borrowed(artifact))?
                                        .pack_factor(dtype)
                                }
                                QuantizedSerdeType::Gguf => {
                                    GgufMatMul::get_isq_type_from_uqff(Cow::Borrowed(artifact))?
                                        .pack_factor(dtype)
                                }
                                QuantizedSerdeType::Fp8 => IsqType::F8E4M3.pack_factor(dtype),
                                QuantizedSerdeType::Unquant => 1,
                                QuantizedSerdeType::Afq => {
                                    AfqLayer::get_isq_type_from_uqff(Cow::Borrowed(artifact))?
                                        .pack_factor(dtype)
                                }
                                QuantizedSerdeType::F8Q8 => IsqType::F8Q8.pack_factor(dtype),
                                QuantizedSerdeType::Mxfp4 => IsqType::MXFP4.pack_factor(dtype),
                                QuantizedSerdeType::Nvfp4 => IsqType::NVFP4.pack_factor(dtype),
                                QuantizedSerdeType::Qtip => {
                                    IsqType::QtipBitshift2.pack_factor(dtype)
                                }
                                QuantizedSerdeType::Qtip2b => IsqType::Qtip2b.pack_factor(dtype),
                                QuantizedSerdeType::TdMoeTucker => 1,
                            };
                            let len = artifact.len();
                            weighted_pack += len * pack_factor;
                            total_len += len;
                        }

                        (weighted_pack / total_len.max(1)).max(1)
                    };

                    let layer_sizes_in_bytes = self.inner.layer_sizes_in_bytes(
                        &config,
                        dtype,
                        weight_pack_factor,
                        None,
                    )?;
                    let non_mapped_size_in_bytes = self.inner.non_mapped_size_in_bytes(
                        &config,
                        dtype,
                        weight_pack_factor,
                        None,
                    )?;
                    let layer_sizes_sum = layer_sizes_in_bytes.iter().sum::<usize>();
                    (
                        layer_sizes_in_bytes,
                        non_mapped_size_in_bytes,
                        layer_sizes_sum + non_mapped_size_in_bytes,
                    )
                } else if let Some(isq) = in_situ_quant {
                    let weight_pack_factor = isq.pack_factor(dtype);
                    let layer_sizes_in_bytes = self.inner.layer_sizes_in_bytes(
                        &config,
                        dtype,
                        weight_pack_factor,
                        None,
                    )?;
                    let non_mapped_size_in_bytes = self.inner.non_mapped_size_in_bytes(
                        &config,
                        dtype,
                        weight_pack_factor,
                        None,
                    )?;
                    let layer_sizes_sum = layer_sizes_in_bytes.iter().sum::<usize>();
                    (
                        layer_sizes_in_bytes,
                        non_mapped_size_in_bytes,
                        layer_sizes_sum + non_mapped_size_in_bytes,
                    )
                } else {
                    // Be sure to get the weight pack factor here; we might be loading a prequantized model.
                    let weight_pack_factor =
                        QuantizationConfigShim::get_quant_config_pack_factor(&config, dtype)?;
                    let layer_sizes_in_bytes = self.inner.layer_sizes_in_bytes(
                        &config,
                        dtype,
                        weight_pack_factor,
                        None,
                    )?;
                    let non_mapped_size_in_bytes = self.inner.non_mapped_size_in_bytes(
                        &config,
                        dtype,
                        weight_pack_factor,
                        None,
                    )?;
                    let layer_sizes_sum = layer_sizes_in_bytes.iter().sum::<usize>();
                    (
                        layer_sizes_in_bytes,
                        non_mapped_size_in_bytes,
                        layer_sizes_sum + non_mapped_size_in_bytes,
                    )
                };

            let new = auto_device_map::get_device_layers(
                &*self.inner,
                &config,
                self.inner.num_layers(&config)?,
                layer_sizes_in_bytes,
                non_mapped_size_in_bytes,
                total_model_size_in_bytes,
                &available_devices,
                dtype,
                &params,
                paged_attn_config.as_ref(),
            )?;
            mapper = DeviceMapSetting::Map(new);
        }

        let pipeline_mapper = mapper.into_mapper(
            self.inner.num_layers(&config)?,
            &device,
            self.config.topology.as_ref(),
            &available_devices,
        )?;
        let mapper = mapper.into_mapper(
            self.inner.num_layers(&config)?,
            &device,
            self.config.topology.as_ref(),
            &available_devices,
        )?;
        let mut layer_devices = Vec::new();
        for layer in 0..self.inner.num_layers(&config)? {
            let device = mapper.device_for(layer, false).cloned();
            layer_devices.push(device);
        }
        let dtype = mapper.get_min_dtype(dtype)?;

        // TODO: PagedAttention is not supported with CPU for now.
        // This check is not really necessary because `get_device_layers` should prevent it.
        let mapping_uses_cpu = mapper.get_unique_devices().iter().any(Device::is_cpu);
        if mapping_uses_cpu && paged_attn_config.is_some() {
            warn!("Device mapping contains a mix of GPU and CPU. There is no CPU support for PagedAttention, disabling PagedAttention.");
            paged_attn_config = None;
        }

        info!("Model config: {:?}", self.inner.get_config_repr(&config)?);
        if crate::using_flash_attn() {
            once_log_info("FlashAttention is enabled.");
        }

        let topology_overrides = self
            .config
            .topology
            .as_ref()
            .map(|topology| {
                topology
                    .pattern_overrides()
                    .into_iter()
                    .map(|(regex, layer)| ImmediateIsqOverride {
                        predicate: regex,
                        ty: layer.isq,
                        device: layer.device.clone(),
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let has_override_isq = topology_overrides
            .iter()
            .any(|override_entry| override_entry.ty.is_some());
        let topology_requires_post_quant = self
            .config
            .topology
            .as_ref()
            .is_some_and(|topology| topology.requires_post_quantization());

        let allow_immediate_cli = self.config.imatrix.is_none()
            && self.config.calibration_file.is_none()
            && in_situ_quant.is_some();

        let mut immediate_ty = None;
        let mut immediate_predicates = Vec::new();
        if allow_immediate_cli {
            immediate_ty = in_situ_quant;
            immediate_predicates =
                if matches!(self.config.organization, IsqOrganization::MoeExpertsOnly) {
                    self.inner.immediate_isq_predicates_moqe(&config)?
                } else {
                    self.inner.immediate_isq_predicates(&config)?
                };
            info!("Applying ISQ to {in_situ_quant:?}");
            if immediate_predicates.is_empty() {
                warn!("No predicates for this model and ISQ setting detected. ISQ will not be applied to any weights!");
            }
        }

        let use_immediate = allow_immediate_cli || has_override_isq;
        if use_immediate {
            let (pool, num_threads) =
                mistralrs_quant::create_isq_thread_pool_for_device(immediate_ty, Some(&device));
            info!("Applying immediate ISQ in parallel on {num_threads} threads.");
            mistralrs_quant::set_immediate_isq_with_pool(
                immediate_ty,
                immediate_predicates.clone(),
                topology_overrides.clone(),
                pool,
            );
        }

        // Logic for ISQ here: if no calibration (i.e imatrix), then allow immediate ISQ. Otherwise, back to normal.
        let mut loading_isq = if use_immediate {
            false
        } else {
            in_situ_quant.is_some()
        };
        if self.config.imatrix.is_some() || self.config.calibration_file.is_some() {
            loading_isq = true;
        }
        loading_isq |= topology_requires_post_quant;
        loading_isq |= self.config.from_uqff.is_some();

        // RUN-161: signal UQFF loading to the per-expert MoE constructor so it
        // builds DummyLayer placeholders (filled by the UQFF deserializer)
        // instead of re-dequantizing+re-quantizing the base experts. Set every
        // load (self-resetting) on the construction thread.
        mistralrs_quant::set_loading_from_uqff(self.config.from_uqff.is_some());

        // UQFF bake: force-load the V4 MTP decoder block (even without
        // `--mtp-depth`) so its tensors are quantized and included in the
        // artifact (~800MB at 2-bit) under `mtp.<j>` names. Without this, a
        // bake made without `--mtp-depth` yields a UQFF that a
        // `--mtp-depth > 0` serve can only satisfy by falling back to the
        // source checkpoint. Set every load (self-resetting).
        crate::pipeline::mtp_pipeline::set_mtp_uqff_bake(
            self.config.write_uqff.is_some() && self.config.from_uqff.is_none(),
        );

        // wave18 — UQFF bake memory policy.
        //
        // A bake constructs the model only to serialize it: no forward pass
        // ever runs. Retaining each quantized MoE expert stack on the GPU
        // therefore buys nothing and costs the whole artifact in device memory
        // (~68 GB for V4 Flash), which is what made a 43-layer bake die at
        // layer 28 on a 140 GB H200 with a 4 KB output directory. With this
        // set, the quantize still runs on the GPU but the packed result is
        // materialized on the host, so device usage is flat across layers.
        //
        // Excluded when a post-load hook is registered (arc-engine's TD-MoE
        // compressor rewrites the quantized layers in place after the load and
        // expects them where the model was mapped), and when loading *from* a
        // UQFF, which is a serve.
        let is_uqff_bake = self.config.write_uqff.is_some()
            && self.config.from_uqff.is_none()
            && !crate::pipeline::post_load_hooks::has_registered_hooks();
        mistralrs_quant::set_bake_isq_to_host(is_uqff_bake);
        if is_uqff_bake {
            info!(
                "UQFF bake: quantized MoE expert stacks will be materialized on the host \
                 (quantize still runs on the accelerator)."
            );
            mistralrs_quant::arm_bake_budget(self.inner.num_layers(&config)?);
        } else {
            mistralrs_quant::disarm_bake_budget();
        }

        if self.config.imatrix.is_some() && self.config.calibration_file.is_some() {
            anyhow::bail!(
                "`imatrix` and `calibration_file` were both specified, this is not allowed."
            );
        }

        // Load onto the regular device if not using isq or if the calibration file is specified.
        // For immediate ISQ on discrete GPUs, load to CPU: the mapper will set the correct target
        // device per-layer, and linear constructors will override to CPU for ISQ-targeted weights.
        // On integrated/unified memory systems (e.g. Grace Blackwell), CPU and GPU share memory,
        // so we load directly to the device.
        let load_device = if !loading_isq || self.config.calibration_file.is_some() {
            loading_isq = false;
            if use_immediate && !crate::utils::normal::is_integrated_gpu(&device) {
                Device::Cpu
            } else {
                device.clone()
            }
        } else if self.config.from_uqff.is_some() {
            // RUN-161: from-UQFF loads the ISQ layers (experts/attention) via
            // deserialize directly onto their mapped (GPU) device, and the base
            // expert weights are NOT loaded as BF16 (DummyLayer placeholders), so
            // there is no OOM risk. Load to the primary device so non-ISQ weights
            // (embeddings, norms, lm_head, compressor) land on GPU instead of
            // being stranded on CPU -> "Expected CUDA storage" at forward time.
            // Per-layer mapper.set_device still overrides for any CPU-mapped layer.
            device.clone()
        } else {
            Device::Cpu
        };

        let is_xlora = self.kind.is_adapted_and(|a| a.is_x_lora());

        let attention_mechanism = if paged_attn_config.is_some() {
            AttentionImplementation::PagedAttention
        } else {
            AttentionImplementation::Eager
        };

        let multi_progress = Arc::new(new_multi_progress());

        // Load matformer slicing config if provided
        let matformer_slicing_config = if let Some(matformer_path) =
            &self.config.matformer_config_path
        {
            use crate::matformer::{MatformerConfig, MatformerSliceConfig};
            info!("Loading Matformer config from {:?}", matformer_path);
            let config = Arc::new(MatformerConfig::from_file(matformer_path)?);

            if let Some(slice_name) = &self.config.matformer_slice_name {
                info!("Using Matformer slice: {}", slice_name);
                Some(MatformerSliceConfig::new(slice_name.clone(), config))
            } else {
                // If no slice name is provided but config exists, we'll need to handle this
                // For now, return None and let the model handle the default slice selection
                warn!("Matformer config loaded but no slice name specified. Models will use their default slice.");
                None
            }
        } else {
            None
        };

        // TurboQuant for the eager KV cache. Must run *before* the model is
        // built: every model constructor calls `NormalCache::new`, which reads
        // this gate. See `kv_cache::resolve_eager_turboquant` for why this is
        // opt-in while the paged `PagedCacheType::TurboQuant` default is not.
        {
            let model_cfg = self.inner.model_config(&config)?;
            let decision = crate::kv_cache::configure_eager_turboquant(
                model_cfg.k_head_dim(),
                model_cfg.v_head_dim(),
                matches!(
                    model_cfg.kv_cache_layout(),
                    crate::paged_attention::KvCacheLayout::Standard
                ),
                paged_attn_config.is_some(),
            );
            match decision {
                crate::kv_cache::EagerTurboQuantDecision::Enabled(k, v) => {
                    let preset = mistralrs_quant::turboquant::TurboQuantPreset::default();
                    info!(
                        "TurboQuant KV cache ON for the eager path: {preset} at \
                         k_head_dim={k}, v_head_dim={v} ({:.2}x vs FP16).",
                        preset.compression_ratio(k)
                    );
                }
                crate::kv_cache::EagerTurboQuantDecision::Disabled(reason) => {
                    tracing::debug!("TurboQuant KV cache off for the eager path: {reason}");
                }
            }
        }

        let mut model = if use_nccl || cfg!(feature = "ring") {
            let (mapper, sharded_vb) = distributed::prepare_distributed_mapper(
                dtype,
                &device,
                &available_devices,
                silent,
                &config,
                loading_isq,
                self.config.from_uqff.is_some(),
                self.config.organization,
                &*self.inner,
                paths.as_ref(),
            )?;

            // Special case for where things can be more optimially loaded.
            match self.kind {
                ModelKind::Normal => normal_model_loader_sharded!(
                    sharded_vb,
                    config,
                    self.inner,
                    mapper,
                    loading_isq,
                    device.clone(),
                    attention_mechanism,
                    multi_progress.clone(),
                    matformer_slicing_config.clone(),
                ),
                ModelKind::Adapter {
                    adapter: AdapterKind::XLora,
                } => xlora_model_loader!(
                    paths,
                    Some(dtype),
                    &load_device,
                    layer_devices.clone(),
                    config,
                    self.inner,
                    silent,
                    mapper,
                    loading_isq,
                    device.clone(),
                    multi_progress.clone(),
                    matformer_slicing_config.clone(),
                ),
                ModelKind::Adapter {
                    adapter: AdapterKind::Lora,
                } => lora_model_loader!(
                    paths,
                    Some(dtype),
                    &load_device,
                    layer_devices.clone(),
                    config,
                    self.inner,
                    silent,
                    mapper,
                    loading_isq,
                    self.config.from_uqff.is_some(),
                    device.clone(),
                    attention_mechanism,
                    matches!(self.config.organization, IsqOrganization::MoeExpertsOnly),
                    multi_progress.clone(),
                    matformer_slicing_config.clone(),
                ),
                _ => unreachable!(),
            }
        } else {
            match self.kind {
                ModelKind::Normal => normal_model_loader!(
                    paths,
                    Some(dtype),
                    &load_device,
                    layer_devices.clone(),
                    config,
                    self.inner,
                    silent,
                    mapper,
                    loading_isq,
                    self.config.from_uqff.is_some(),
                    device.clone(),
                    attention_mechanism,
                    matches!(self.config.organization, IsqOrganization::MoeExpertsOnly),
                    multi_progress.clone(),
                    matformer_slicing_config.clone(),
                ),
                ModelKind::Adapter {
                    adapter: AdapterKind::XLora,
                } => xlora_model_loader!(
                    paths,
                    Some(dtype),
                    &load_device,
                    layer_devices.clone(),
                    config,
                    self.inner,
                    silent,
                    mapper,
                    loading_isq,
                    device.clone(),
                    multi_progress.clone(),
                    matformer_slicing_config.clone(),
                ),
                ModelKind::Adapter {
                    adapter: AdapterKind::Lora,
                } => lora_model_loader!(
                    paths,
                    Some(dtype),
                    &load_device,
                    layer_devices.clone(),
                    config,
                    self.inner,
                    silent,
                    mapper,
                    loading_isq,
                    self.config.from_uqff.is_some(),
                    device.clone(),
                    attention_mechanism,
                    matches!(self.config.organization, IsqOrganization::MoeExpertsOnly),
                    multi_progress.clone(),
                    matformer_slicing_config.clone(),
                ),
                _ => unreachable!(),
            }
        };

        let tokenizer = get_tokenizer(paths.get_tokenizer_filename(), None)?;
        let gen_conf: Option<GenerationConfig> = paths.get_gen_conf_filename().and_then(|f| {
            match serde_json::from_str::<GenerationConfig>(&fs::read_to_string(f).unwrap()) {
                Ok(conf) => Some(conf),
                Err(e) => {
                    warn!("Failed to parse generation_config.json: {}", e);
                    None
                }
            }
        });

        let chat_template_explicit = paths
            .get_chat_template_explicit()
            .as_ref()
            .map(|x| x.to_string_lossy().to_string());
        let chat_template = get_chat_template(
            paths,
            self.jinja_explicit.as_ref(),
            chat_template_explicit.as_ref(),
            self.chat_template.as_ref(),
            None,
        );

        if let Some(calibration_file) = &self.config.calibration_file {
            let calibration_data = std::fs::read_to_string(calibration_file)?;
            // Tokenize, don't add bos yet
            let tokens = tokenizer
                .encode_fast(calibration_data, false)
                .map_err(anyhow::Error::msg)?
                .get_ids()
                .to_vec();
            info!(
                "Collecting imatrix from calibration file `{}` of {} tokens.",
                calibration_file.display(),
                tokens.len()
            );
            let bos_tok_id = chat_template
                .bos_tok()
                .as_deref()
                .and_then(|tok| tokenizer.token_to_id(tok));

            match self.config.organization {
                IsqOrganization::Default => model.begin_track_stats()?,
                IsqOrganization::MoeExpertsOnly => model.begin_track_stats_moe_experts_only()?,
            }

            // Arc: a registered calibration request additionally arms the
            // richer per-layer accumulators (raw diag(XᵀX) + optional gram +
            // per-expert). With no request this is a no-op and the sweep
            // behaves exactly as before.
            let calib_request = crate::pipeline::calibration::take_calibration_request();
            if let Some(req) = &calib_request {
                let report =
                    crate::pipeline::calibration::begin_model_calibration(&mut *model, &req.opts);
                info!(
                    "Arc calibration: armed {}/{} ISQ layers (options {:?})",
                    report.armed, report.total, req.opts
                );
                if report.armed == 0 {
                    warn!(
                        "Arc calibration: no ISQ layer accepted an accumulator. The artifact \
                         will contain a layer inventory but no statistics. This happens when the \
                         checkpoint is already quantized on disk (FP8/GPTQ)."
                    );
                }
            }
            let max_chunks = calib_request.as_ref().and_then(|r| r.max_chunks);

            const CHUNK_SIZE: usize = 1024;
            let n_chunks = match max_chunks {
                Some(max) => tokens.len().div_ceil(CHUNK_SIZE).min(max),
                None => tokens.len().div_ceil(CHUNK_SIZE),
            };
            let mut swept_chunks = 0usize;
            let mut swept_tokens = 0u64;
            let start = Instant::now();
            for (i, chunk) in tokens.chunks(CHUNK_SIZE).enumerate() {
                if max_chunks.is_some_and(|max| i >= max) {
                    break;
                }
                let mut chunk = chunk.to_vec();
                if let Some(bos_tok_id) = bos_tok_id {
                    chunk.insert(0, bos_tok_id);
                }
                let chunk_len = chunk.len();

                let start = Instant::now();
                let inputs = make_prompt_chunk(
                    0,
                    vec![&chunk],
                    &[0],
                    &load_device,
                    None,
                    false,
                    None,
                    Some(pipeline_mapper.as_ref()),
                    None,
                    None,
                )?;

                model.forward(
                    &inputs.input.to_device(model.device())?,
                    &inputs.positions,
                    inputs.context_lens.clone(),
                    inputs.position_ids.clone(),
                    None,
                    &inputs.flash_meta.clone(),
                )?;

                match model.cache_mut() {
                    EitherCache::Full(full) => {
                        for layer in &mut *full.lock() {
                            *layer = None
                        }
                    }
                    EitherCache::Normal(normal) => {
                        for layer in &mut *normal.lock().unwrap().0 {
                            layer.reset();
                        }
                    }
                    EitherCache::Hybrid(hybrid) => {
                        hybrid.lock().unwrap().reset();
                    }
                }

                swept_chunks += 1;
                swept_tokens += chunk_len as u64;

                let end = Instant::now();
                info!(
                    "Processed chunk {}/{n_chunks} ({chunk_len} tokens), {:.2}s",
                    i + 1,
                    end.duration_since(start).as_secs_f32()
                );
            }
            load_device.synchronize()?;
            let end = Instant::now();
            info!(
                "Finished collecting imatrix in {:.2}s",
                end.duration_since(start).as_secs_f32()
            );

            // Arc: harvest the calibration accumulators and write the artifact.
            if let Some(req) = calib_request {
                let run_info = crate::pipeline::calibration::CalibrationRunInfo {
                    model_id: req.model_id.clone(),
                    model_sha: None,
                    arch: req.arch.clone(),
                    isq_organization: match self.config.organization {
                        IsqOrganization::Default => "default".to_string(),
                        IsqOrganization::MoeExpertsOnly => "moqe".to_string(),
                    },
                    calibration_file: Some(calibration_file.display().to_string()),
                    samples: swept_chunks,
                    seq_len: CHUNK_SIZE,
                    total_tokens: swept_tokens,
                    model_dtype: format!("{dtype:?}"),
                    options: req.opts,
                };
                let artifact = crate::pipeline::calibration::extract_calibration_artifact(
                    &mut *model,
                    run_info,
                )?;
                artifact.save(&req.out)?;
                info!(
                    "Arc calibration: wrote `{}` ({} layers, {} with statistics, {} tokens)",
                    req.out.display(),
                    artifact.layers.len(),
                    artifact.supported_layer_count(),
                    swept_tokens
                );
            }
        }

        // Construction (and with it every per-layer quantize) is done, so the
        // bake budget has nothing left to police. Disarm it before the
        // serialize pass so a serve later in this process is not judged against
        // a stale projection. (wave18)
        mistralrs_quant::disarm_bake_budget();

        // Only if loading from UQFF
        let should_serialize = self.config.write_uqff.is_some();
        let should_quantize_pass = loading_isq;

        // When a post-load hook (e.g. TD-MoE Tucker factorization) will modify
        // the quantized layers, defer the UQFF write until *after* the hook so
        // the written file reflects the final (factored) model rather than the
        // intermediate QTIP one. (RUN-161)
        let defer_uqff_for_hooks = self.config.write_uqff.is_some()
            && crate::pipeline::post_load_hooks::has_registered_hooks();

        if (should_quantize_pass || should_serialize) && self.config.from_uqff.is_none() {
            let imatrix_source = if should_quantize_pass {
                match (
                    self.config.imatrix.as_ref(),
                    self.config.calibration_file.is_some(),
                ) {
                    (None, false) => None,
                    (Some(file), false) => Some(ImatrixDataSource::File(file)),
                    (None, true) => Some(ImatrixDataSource::Collected),
                    (Some(_), true) => unreachable!(),
                }
            } else {
                None
            };

            if should_quantize_pass {
                info!("Applying ISQ to all ranks.");
            } else {
                info!("Serializing existing ISQ tensors without additional quantization.");
            }

            let multi_progress = Arc::new(new_multi_progress());

            model.quantize(
                in_situ_quant,
                model.device().clone(),
                self.config.topology.as_ref(),
                silent,
                imatrix_source,
                self.config.organization,
                should_quantize_pass,
                if defer_uqff_for_hooks {
                    None
                } else {
                    self.config.write_uqff.as_ref()
                },
                UqffFullSer {
                    tokenizer: &tokenizer,
                    template_filename: paths.get_template_filename(),
                    generation_config: paths.get_gen_conf_filename(),
                    config: config.clone(),
                    processor_filename: &None,
                    preprocessor_filename: &None,
                    modules: None,
                    module_paths: None,
                },
                multi_progress.clone(),
            )?;
        } else if let Some(from_uqff) = &*self.from_uqff.read().unwrap() {
            model.load_from_artifacts(
                device.clone(),
                self.config.topology.as_ref(),
                silent,
                from_uqff,
                // Source checkpoint fallback: if the artifact was baked
                // without `--mtp-depth`, the V4 MTP decoder block is loaded
                // unquantized from these weights instead of panicking on
                // leftover DummyLayers.
                Some(UqffSourceWeights {
                    weight_files: paths.get_weight_filenames(),
                    dtype,
                }),
            )?;
        }

        // Expert parallelism: a UQFF artifact holds every expert, so the shard
        // could not be applied at construction. Narrow the deserialized expert
        // stacks now, before anything can run a forward. A no-op when EP is
        // off or the experts were already sliced at load.
        let narrowed = model.apply_pending_expert_parallel_slice()?;
        if narrowed > 0 {
            info!("Expert parallelism: narrowed {narrowed} expert-stacked layers to this rank's shard.");
        }

        // Run any registered post-load hooks (e.g. Arc's TD-MoE compressor).
        // Hooks are no-ops when nothing has been registered.
        crate::pipeline::post_load_hooks::run_post_load_hooks(&mut *model)
            .map_err(|e| candle_core::Error::Msg(format!("post-load hook failed: {e}")))?;

        // Deferred UQFF write: the post-load hook (TD-MoE) has now replaced the
        // expert layers with their factored form, so serialize the final model.
        // Serialize-only pass — no further quantization. (RUN-161)
        if defer_uqff_for_hooks {
            info!("Serializing post-hook (TD-MoE) model to UQFF.");
            let multi_progress = Arc::new(new_multi_progress());
            model.quantize(
                None,
                model.device().clone(),
                self.config.topology.as_ref(),
                silent,
                None,
                self.config.organization,
                false,
                self.config.write_uqff.as_ref(),
                UqffFullSer {
                    tokenizer: &tokenizer,
                    template_filename: paths.get_template_filename(),
                    generation_config: paths.get_gen_conf_filename(),
                    config: config.clone(),
                    processor_filename: &None,
                    preprocessor_filename: &None,
                    modules: None,
                    module_paths: None,
                },
                multi_progress,
            )?;
        }

        // After ISQ, the CUDA driver's default memory pool may be holding large
        // amounts of freed memory (e.g. temporary BF16 tensors from INT4 dequant →
        // QTIP quantize). Trim the pool so that `cuMemGetInfo` reports accurate
        // free VRAM for the PagedAttention KV-cache budget below.
        #[cfg(feature = "cuda")]
        crate::trim_cuda_memory_pools();

        let paged_attn_config = if matches!(
            self.kind,
            ModelKind::Adapter {
                adapter: AdapterKind::XLora
            }
        ) {
            warn!(
                "Adapter parallel_models do not currently support PagedAttention, running without"
            );
            None
        } else {
            paged_attn_config
        };

        let (cache_config, cache_engine) = if let Some(paged_attn_config) = paged_attn_config {
            let cache_config = calculate_cache_config(
                paged_attn_config.mem_gpu,
                paged_attn_config.block_size,
                dtype,
                paged_attn_config.cache_type,
                paged_attn_config.cache_type_explicit,
                model.config(),
                &device,
                &pipeline_mapper
                    .get_unique_devices()
                    .into_iter()
                    .map(Some)
                    .collect::<Vec<_>>(),
                silent,
                None,
                max_kv_tokens,
            )?;

            let mut layer_devices = Vec::new();
            for layer in 0..self.inner.num_layers(&config)? {
                let device = model.get_layers().1.device_for(layer, false).cloned();
                layer_devices.push(device);
            }
            let cache_engine = CacheEngine::new(
                model.config(),
                &cache_config,
                dtype,
                model.device(),
                layer_devices.clone(),
            )?;

            // Set TurboQuant norms globally for PagedAttention layers
            {
                let kv_cache = cache_engine.get_kv_cache();
                let norms: Vec<(candle_core::Tensor, candle_core::Tensor)> = kv_cache
                    .iter()
                    .filter_map(|(_, _, kn, vn)| match (kn, vn) {
                        (Some(k), Some(v)) => Some((k.clone(), v.clone())),
                        _ => None,
                    })
                    .collect();
                if !norms.is_empty() {
                    crate::paged_attention::set_global_turbo_norms(norms);
                } else {
                    crate::paged_attention::clear_global_turbo_norms();
                }
            }

            (Some(cache_config), Some(cache_engine))
        } else {
            crate::paged_attention::clear_global_turbo_norms();
            (None, None)
        };

        let max_seq_len = model.max_seq_len();
        let llg_factory = build_llg_factory(tokenizer.clone())?;
        let num_hidden_layers = match model.cache() {
            EitherCache::Full(full) => full.lock().len(),
            EitherCache::Normal(normal) => normal.lock().unwrap().0.len(),
            EitherCache::Hybrid(hybrid) => hybrid.lock().unwrap().num_layers(),
        };
        let eos = calculate_eos_tokens(&chat_template, gen_conf, &tokenizer);
        let sliding_window = model.config().sliding_window;
        let model_metadata = Arc::new(model.config().clone());

        #[cfg(feature = "cuda")]
        let _graph_device = model.device().clone();

        // Extract weight pointers for the dedicated decode path (model-agnostic).
        // This copies/extracts decode weights into separate buffers; for very
        // large models that barely fit VRAM (e.g. V4 236B @ qtip2 ~66GB on an
        // 80GB H100) the extraction OOMs and its failed allocations get cached
        // by the CUDA allocator, fragmenting VRAM and hanging the first forward.
        // It's a decode *speed* optimization only — gate it off via
        // ARC_NO_DEDICATED_DECODE=1 to reclaim that headroom. Default unchanged.
        #[cfg(feature = "cuda")]
        let _decode_weights = if env_flag_is_set("ARC_NO_DEDICATED_DECODE") {
            tracing::info!("Dedicated decode path extraction skipped (ARC_NO_DEDICATED_DECODE=1).");
            None
        } else if {
            // ARCHITECTURE GUARD, deliberately evaluated BEFORE the block below.
            //
            // `extract_model_weights` re-checks this — that is the real contract
            // and must hold for every caller — but checking it only there would
            // mean a model we are about to refuse has already paid for two full
            // BF16 `dequantize_w()` calls (lm_head, and whatever sits at index
            // 5). On a model that barely fits VRAM those are exactly the
            // allocations whose failures the CUDA allocator caches and
            // fragments the pool with — the reason ARC_NO_DEDICATED_DECODE
            // exists at all. A refusal that lands after the damage is not a
            // refusal. `get_layers()` here is cheap: it collects references and
            // dequantizes nothing.
            let num_layers = model.config().num_layers;
            let (layers_mut, _) = model.get_layers();
            let tags: Vec<Option<usize>> = layers_mut.iter().map(|(_, idx)| *idx).collect();
            match arc_cuda_graph::check_dense_layer_inventory(&tags, num_layers) {
                // The layout is rebuilt (not carried) by `extract_model_weights`
                // below: `get_layers()` is re-called there under a fresh borrow,
                // and a layout that describes one call's vector must not be used
                // to index another's. It is cheap to recompute and unsound to
                // reuse, so it is recomputed.
                Ok(_layout) => false,
                Err(e) => {
                    tracing::warn!(
                        "Dedicated decode path declined (by design), before any weight was \
                         dequantized: {e}"
                    );
                    arc_profiler::mark_unreachable(
                        "decode.dedicated",
                        "the dense decode path refused this architecture; decode runs on the \
                         standard Candle path",
                        "normal.rs:1544",
                    );
                    true
                }
            }
        } {
            None
        } else {
            let cfg = model.config().clone();
            // Get residuals first (immutable borrow), then get_layers (mutable borrow)
            let residuals = model.residual_tensors();
            // get_layers requires &mut self — call after residuals is collected
            let (layers_mut, _) = model.get_layers();
            let layers_ref: Vec<_> = layers_mut
                .iter()
                .map(|(l, idx)| (l as &std::sync::Arc<dyn mistralrs_quant::QuantMethod>, *idx))
                .collect();
            // Infer intermediate_size and vocab_size from weight shapes
            let lm_head_w = layers_ref[0].0.dequantize_w().ok();
            let vocab_size = lm_head_w.as_ref().map(|w| w.dims()[0]).unwrap_or(0);

            // gate_proj is at index 1 + 4 (5th projection in first layer: q,k,v,o,gate)
            let gate_idx = if layers_ref.len() > 5 { 5 } else { 0 };
            let gate_w = layers_ref
                .get(gate_idx)
                .and_then(|(l, _)| l.dequantize_w().ok());
            let intermediate_size = gate_w.as_ref().map(|w| w.dims()[0]).unwrap_or(0);

            // Read rms_norm_eps and rope_theta from the raw config JSON (model-agnostic)
            let config_json: serde_json::Value = serde_json::from_str(&config).unwrap_or_default();
            let rms_norm_eps = config_json
                .get("rms_norm_eps")
                .or_else(|| config_json.get("layer_norm_epsilon"))
                .or_else(|| config_json.get("layer_norm_eps"))
                .and_then(|v| v.as_f64())
                .unwrap_or(1e-6) as f32;
            let rope_theta = config_json
                .get("rope_theta")
                .and_then(|v| v.as_f64())
                .unwrap_or(10000.0) as f32;
            let has_qk_norm = config_json
                .get("qk_norm")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);

            let decode_config = arc_cuda_graph::DecodeConfig {
                num_layers: cfg.num_layers,
                hidden_size: cfg.hidden_size,
                num_heads: cfg.num_attn_heads,
                num_kv_heads: cfg.num_kv_heads,
                head_dim: cfg.k_head_dim,
                intermediate_size,
                vocab_size,
                rms_norm_eps,
                rope_theta,
                has_qk_norm,
                max_position_embeddings: max_seq_len,
                is_gpt_neox: true, // GPT-NeoX style (half-split) — standard for all modern models
            };
            match arc_cuda_graph::extract_model_weights(&layers_ref, &residuals, decode_config) {
                Ok(w) => {
                    tracing::info!("Decode path: {} layers extracted", w.layers.len());
                    Some(w)
                }
                Err(e) => {
                    // Two very different things land here and they must not read
                    // the same. A refusal from the architecture guard is the
                    // system working: the dense decode path cannot describe this
                    // model, so it declines rather than computing the wrong one.
                    // Anything else is a genuine fault.
                    let refused = e
                        .to_string()
                        .contains("dedicated decode path does not support this architecture");
                    if refused {
                        tracing::info!("Dedicated decode path declined (by design): {e}");
                        arc_profiler::mark_unreachable(
                            "decode.dedicated",
                            "the dense decode path refused this architecture; decode runs on the \
                             standard Candle path",
                            "normal.rs:1630",
                        );
                    } else {
                        tracing::warn!(
                            "Decode path extraction FAILED (not a refusal — this is a fault): {e}. \
                             Decode falls back to the standard Candle path."
                        );
                    }
                    None
                }
            }
        };

        // Bind the profiler's device timer to the device the model actually
        // runs on, and stamp the provenance a report is worthless without.
        // Both are no-ops unless `ARC_PROFILE=1`.
        arc_profiler::attach_device(&device);
        // `ARC_PROFILE_SELFTEST=1`: prove on this box, before any real
        // measurement, that device time comes from CUDA events and not from
        // launch timing. The verdict goes into the report's notes either way.
        arc_profiler::maybe_selftest(&device);
        arc_profiler::set_meta(|h| {
            h.model = self.model_id.clone();
            h.build_features = vec![
                #[cfg(feature = "cuda")]
                "cuda".to_string(),
                #[cfg(feature = "flash-attn")]
                "flash-attn".to_string(),
                #[cfg(feature = "cudnn")]
                "cudnn".to_string(),
                #[cfg(feature = "metal")]
                "metal".to_string(),
            ];
        });

        Ok(Arc::new(Mutex::new(NormalPipeline {
            model,
            tokenizer: tokenizer.into(),
            no_kv_cache: self.no_kv_cache,
            chat_template: Arc::new(chat_template),
            non_granular_state: self.tgt_non_granular_index.map(|tgt_non_granular_index| {
                NonGranularState {
                    non_granular_index: Arc::new(Mutex::new(0)),
                    tgt_non_granular_index,
                }
            }),
            model_id: self.model_id.clone(),
            metadata: Arc::new(GeneralMetadata {
                max_seq_len,
                llg_factory: Some(llg_factory),
                no_kv_cache: self.no_kv_cache,
                no_prefix_cache: is_xlora,
                num_hidden_layers,
                eos_tok: eos,
                kind: self.kind.clone(),
                is_xlora,
                activation_dtype: dtype,
                sliding_window,
                cache_config,
                cache_engine,
                model_metadata: Some(model_metadata),
                modalities: Modalities {
                    input: vec![SupportedModality::Text],
                    output: vec![SupportedModality::Text],
                },
            }),
            topology: self.config.topology.clone(),
            silent,
            organization: self.config.organization,
            template_filename: paths.get_template_filename().clone(),
            generation_config: paths.get_gen_conf_filename().cloned(),
            config,
            imatrix: self.config.imatrix.clone(),
            mapper: pipeline_mapper,
            #[cfg(feature = "cuda")]
            cuda_graph_runner: arc_cuda_graph::try_init_graph_runner(&_graph_device),
            #[cfg(feature = "cuda")]
            _capturable_device: Some(_graph_device),
            #[cfg(feature = "cuda")]
            dedicated_decode: _decode_weights.and_then(|w| {
                match arc_cuda_graph::DedicatedDecodePath::new(w) {
                    Ok(d) => Some(d),
                    Err(e) => {
                        tracing::warn!("Dedicated decode path init failed: {e}");
                        None
                    }
                }
            }),
            // Autonomous runner is lazily initialized on first decode call. At
            // load time we don't yet know batch_size / max_tokens / sampling
            // params, and the AutonomousDecodeRunner pre-allocates buffers
            // based on those. NormalPipeline::autonomous_decode performs the
            // lazy init (and graph capture) once a real decode batch arrives.
            #[cfg(feature = "cuda")]
            autonomous_runner: None,
        })))
    }

    fn get_id(&self) -> String {
        self.model_id.clone()
    }

    fn get_kind(&self) -> ModelKind {
        self.kind.clone()
    }
}

impl PreProcessingMixin for NormalPipeline {
    fn get_chat_template(&self) -> Option<Arc<ChatTemplate>> {
        Some(self.chat_template.clone())
    }
    fn get_input_processor_config(&self) -> Option<Arc<dyn Any>> {
        None
    }
}

impl IsqPipelineMixin for NormalPipeline {
    fn re_isq_model(&mut self, dtype: IsqType) -> Result<()> {
        let device = self.device().clone();
        let multi_progress = Arc::new(new_multi_progress());
        self.model.quantize(
            Some(dtype),
            device.clone(),
            self.topology.as_ref(),
            self.silent,
            self.imatrix.as_ref().map(ImatrixDataSource::File),
            self.organization,
            true,
            None,
            UqffFullSer {
                tokenizer: &self.tokenizer,
                template_filename: &self.template_filename,
                generation_config: self.generation_config.as_ref(),
                config: self.config.clone(),
                processor_filename: &None,
                preprocessor_filename: &None,
                modules: None,
                module_paths: None,
            },
            multi_progress.clone(),
        )?;
        Ok(())
    }
}

impl CacheManagerMixin for NormalPipeline {
    fn clone_in_cache(&self, seqs: &mut [&mut Sequence]) -> candle_core::Result<()> {
        match self.model.cache() {
            EitherCache::Full(_) => FullCacheManager.clone_in_cache(self, seqs, false),
            EitherCache::Normal(_) => NormalCacheManager.clone_in_cache(self, seqs, false),
            EitherCache::Hybrid(_) => HybridCacheManager.clone_in_cache(self, seqs, false),
        }
    }
    fn clone_out_cache(&self, seqs: &mut [&mut Sequence]) {
        match self.model.cache() {
            EitherCache::Full(_) => FullCacheManager.clone_out_cache(self, seqs, false),
            EitherCache::Normal(_) => NormalCacheManager.clone_out_cache(self, seqs, false),
            EitherCache::Hybrid(_) => HybridCacheManager.clone_out_cache(self, seqs, false),
        }
    }
    fn set_none_cache(
        &self,
        seqs: &mut [&mut Sequence],
        reset_non_granular: bool,
        modify_draft_cache: bool,
        load_preallocated_cache: bool,
    ) {
        match self.model.cache() {
            EitherCache::Full(_) => {
                FullCacheManager.set_none_cache(self, seqs, modify_draft_cache, false)
            }
            EitherCache::Normal(_) => NormalCacheManager.set_none_cache(
                self,
                seqs,
                modify_draft_cache,
                load_preallocated_cache,
            ),
            EitherCache::Hybrid(_) => HybridCacheManager.set_none_cache(
                self,
                seqs,
                modify_draft_cache,
                load_preallocated_cache,
            ),
        }
        if reset_non_granular {
            self.reset_non_granular_state()
        }
    }
    fn cache(&self) -> &EitherCache {
        self.model.cache()
    }
}

impl MetadataMixin for NormalPipeline {
    fn device(&self) -> Device {
        self.model.device().clone()
    }
    fn tokenizer(&self) -> Option<Arc<Tokenizer>> {
        Some(self.tokenizer.clone())
    }
    fn name(&self) -> String {
        self.model_id.clone()
    }
    fn reset_non_granular_state(&self) {
        if let Some(s) = self.non_granular_state.as_ref() {
            *self.cache().full().get_scalings_cache() = None;
            *get_mut_arcmutex!(s.non_granular_index) = 0;
        }
    }
    fn get_metadata(&self) -> Arc<GeneralMetadata> {
        self.metadata.clone()
    }
    fn device_mapper(&self) -> Option<&dyn DeviceMapper> {
        Some(&*self.mapper)
    }
}

#[async_trait::async_trait]
impl Pipeline for NormalPipeline {
    fn forward_inputs(
        &mut self,
        inputs: Box<dyn Any>,
        return_raw_logits: bool,
    ) -> Result<ForwardInputsResult, candle_core::Error> {
        let ModelInputs {
            input_ids,
            input_ids_cpu: _,
            input_ids_full,
            seqlen_offsets,
            seqlen_offsets_full,
            context_lens,
            position_ids,
            paged_attn_meta,
            flash_meta,
            flash_meta_full,
        } = *inputs.downcast().expect("Downcast failed.");
        let metadata = self.get_metadata();
        let paged_attn_meta = match (&metadata.cache_engine, &paged_attn_meta) {
            (Some(cache_engine), Some(meta)) => Some((cache_engine, meta)),
            (Some(_), None) => {
                candle_core::bail!("Forward step expected a PagedAttention input metadata. This was not provided, please ensure that the scheduler config is correctly configured for PagedAttention.")
            }
            (None, Some(_)) => {
                candle_core::bail!("Forward step got a PagedAttention input metadata but there is no cache engine. Please raise an issue.")
            }
            (None, None) => None,
        };

        let logits = match self.model.is_xlora() {
            false => {
                let paged_attn_meta = paged_attn_meta
                    .as_ref()
                    .map(|meta| (meta.0.get_kv_cache().clone(), meta.1.clone()));

                // RUN-161 Step 2a: CUDA-graph capture probe for V4 decode.
                // Gated by ARC_V4_CAPTURE_PROBE. Validates the premise that the
                // whole V4 candle forward can be RECORDED into a CUDA graph on
                // candle's stream (i.e. the stream is capturable and the forward
                // is sync-free with ARC_GPU_ACT_QUANT=1 + on-device MoE).
                //
                // Capture records (does not execute); end_capture_and_cache
                // instantiates + launches once -> a single correct forward, so
                // the captured output is used as this step's logits. REPLAY
                // output correctness needs static input buffers (2b) + a
                // device-indexed KV write/read (2c); until then we only time a
                // replay and discard it, using eager for the real logits.
                #[cfg(feature = "cuda")]
                {
                    let probe = std::env::var_os("ARC_V4_CAPTURE_PROBE").is_some();
                    // Shadowed (cuda-only) so step 2b can swap in the
                    // address-stable buffer without making the outer
                    // destructuring binding `mut`, which would warn on every
                    // non-cuda build. Tensor::clone is an Arc bump.
                    #[allow(unused_mut)]
                    let mut input_ids = input_ids.clone();
                    let (bs, seq_len) = input_ids.dims2().unwrap_or((0, 0));
                    if !probe {
                        // The only CUDA-graph path structurally reachable from
                        // a plain V4 run, and it is gated off by default. Say so
                        // in the report: a graph node reporting 0 ns because it
                        // never executed must not read like a graph node that
                        // executed instantly.
                        arc_profiler::mark_unreachable(
                            "cuda_graph.capture_probe",
                            "ARC_V4_CAPTURE_PROBE is unset, so capture/replay is skipped and the \
                             forward runs eagerly",
                            "normal.rs:1922",
                        );
                    }
                    // Candle's caching allocator, driven by `alloc_cache_action`.
                    //
                    // This used to be `probe && seq_len == 1 && env(...)`, which
                    // made a general-purpose allocator reachable only when the
                    // V4 capture probe was also on. The cache recycles frees for
                    // ANY decode step — it is not capture machinery — and the
                    // three stacked default-off gates meant the ~11k allocations
                    // per token were a disabled feature rather than a missing
                    // one. It is now on for decode by default and drained on
                    // prefill; see `alloc_cache_action` for why prefill must
                    // drain rather than coast.
                    if let candle_core::Device::Cuda(cd) = self.device() {
                        match alloc_cache_action(
                            seq_len,
                            cd.alloc_cache_enabled(),
                            alloc_cache_killed(),
                        ) {
                            AllocCacheAction::Enable => {
                                cd.set_alloc_cache_enabled(true);
                                if let Some(cap) = alloc_cache_capacity_bytes() {
                                    cd.set_alloc_cache_capacity(cap);
                                }
                            }
                            AllocCacheAction::DrainAndDisable => {
                                cd.set_alloc_cache_enabled(false)
                            }
                            AllocCacheAction::Leave => {}
                        }
                        report_alloc_cache_step(&cd, seq_len);
                    }
                    if probe && seq_len == 1 {
                        // RUN-161 step 2b. Set the graph-mode device position:
                        // drives RoPE + the fixed-capacity KV write slot, and
                        // makes warmup forwards take the shape-constant path so
                        // the cache populates with capture-shape buffers.
                        //
                        // This used to allocate a FRESH tensor every step, which
                        // is fine up to and including the single-launch capture
                        // and wrong for every replay after it: capture bakes the
                        // device pointer, so a replay read an address whose
                        // tensor had already been dropped. Writing in place keeps
                        // the baked address valid and current.
                        let pos = seqlen_offsets.first().copied().unwrap_or(0) as u32;
                        let nb = bs.max(1);
                        let dev_for_pos = self.device();
                        // The token IDs must be address-stable for the same
                        // reason. A replay reading this step's positions but
                        // last step's ids is still wrong, just less visibly.
                        match crate::layers::set_graph_mode_input_ids_in_place(&input_ids) {
                            Ok(stable) => input_ids = stable,
                            Err(e) => tracing::warn!(
                                "ARC capture: could not stage input_ids at a stable address: {e}"
                            ),
                        }
                        if let Err(e) = crate::layers::set_graph_mode_positions_in_place(
                            &vec![pos; nb],
                            &dev_for_pos,
                        ) {
                            tracing::warn!(
                                "ARC capture: could not update graph-mode positions in place: {e}; \
                                 replay would read stale positions, so capture is disabled for \
                                 this run"
                            );
                            if let Some(r) = self.cuda_graph_runner.as_mut() {
                                r.disable();
                            }
                        }
                    }
                    let captured: Option<Tensor> = if probe
                        && seq_len == 1
                        && self.cuda_graph_runner.is_some()
                    {
                        // Own the runner locally so `self.model.forward` is
                        // free of the runner's borrow; restore before return.
                        let mut runner = self.cuda_graph_runner.take().unwrap();
                        let result = if runner.tick_warmup() {
                            None
                        } else if runner.is_enabled()
                            && !runner.has_graph(bs)
                            && runner.try_take_deferred_pass()
                        {
                            // RUN-161 deferred-free pass (generic): one eager
                            // forward with the caching allocator in capture
                            // mode so the free pool grows to the FULL
                            // per-forward alloc count (eager warmups only
                            // reach peak-live; capture needs every alloc
                            // distinct). Output is this step's logits (eager).
                            if let candle_core::Device::Cuda(cd) = self.device() {
                                cd.set_capture_mode(true);
                            }
                            let t_eager = std::time::Instant::now();
                            let out = self.model.forward(
                                &input_ids,
                                &seqlen_offsets,
                                context_lens.clone(),
                                position_ids.clone(),
                                paged_attn_meta.as_ref().map(|(a, b)| (a.clone(), b)),
                                &flash_meta,
                            );
                            // Do NOT discard this. `let _ =` here swallowed the
                            // only place an ASYNCHRONOUS device fault in a warm
                            // pass could ever be reported: the forward itself
                            // returns Ok (kernels are launched, not awaited),
                            // and the sync is what collects the fault. A
                            // swallowed fault leaves a sticky context error, and
                            // the next CUDA call to report it is the first
                            // cuGraphLaunch — which is then blamed for a fault
                            // it inherited.
                            if let Err(e) = self.device().synchronize() {
                                tracing::error!(
                                    "ARC capture: warm-pass forward FAULTED at sync ({e}). The \
                                     CUDA context is now sticky-errored and every later call, \
                                     including the graph launch, will fail with it. Capture is \
                                     disabled for this run."
                                );
                                runner.disable();
                            }
                            tracing::info!(
                                "ARC capture: EAGER forward (sync'd) = {:?}",
                                t_eager.elapsed()
                            );
                            if let candle_core::Device::Cuda(cd) = self.device() {
                                cd.set_capture_mode(false);
                            }
                            match out {
                                Ok(o) => {
                                    tracing::info!(
                                            "ARC capture: deferred-free warmup pass done (cache grown to full per-forward count)"
                                        );
                                    Some(o)
                                }
                                Err(e) => {
                                    tracing::warn!(
                                        "ARC capture: deferred pass forward errored: {e}; eager"
                                    );
                                    None
                                }
                            }
                        } else if runner.is_enabled() && !runner.has_graph(bs) {
                            // ── PROBE, NOT A FIX ──────────────────────────
                            // `ARC_GRAPH_PREWARM_SIZES=172032[,...]` seeds the
                            // alloc cache with blocks of exactly these byte
                            // sizes before capture begins, so an allocation
                            // that would otherwise MISS the warm pool during
                            // capture becomes a hit.
                            //
                            // This exists to TEST a hypothesis, not to ship:
                            // if the surviving capture-time miss is what
                            // corrupts the heap, pre-warming its size should
                            // make the crash disappear. The real fix is that
                            // the buffer stops being allocated per decode step
                            // — this only hides the symptom, and hiding it is
                            // the point of the experiment.
                            //
                            // Seeded BEFORE `set_capture_mode(true)` on
                            // purpose: while capturing, a freed buffer parks
                            // in `deferred` and is NOT servable, so seeding
                            // inside capture would warm nothing.
                            if let Some(spec) = std::env::var_os("ARC_GRAPH_PREWARM_SIZES") {
                                let spec = spec.to_string_lossy().to_string();
                                let dev = self.device();
                                let mut seeded = Vec::new();
                                for tok in spec.split(',').filter(|t| !t.trim().is_empty()) {
                                    match tok.trim().parse::<usize>() {
                                        Ok(n) if n > 0 => {
                                            match Tensor::zeros(n, candle_core::DType::U8, &dev) {
                                                // Dropping it immediately is the
                                                // whole mechanism: the buffer
                                                // returns to the cache's `free`
                                                // list at exactly `n` bytes.
                                                Ok(t) => {
                                                    drop(t);
                                                    seeded.push(n);
                                                }
                                                Err(e) => tracing::warn!(
                                                    "ARC prewarm: could not seed {n} bytes: {e}"
                                                ),
                                            }
                                        }
                                        _ => tracing::warn!(
                                            "ARC prewarm: ignoring unparsable size {tok:?}"
                                        ),
                                    }
                                }
                                tracing::info!(
                                    "ARC prewarm: seeded alloc cache with sizes {seeded:?} \
                                         (PROBE — not a fix; the buffer still allocates per step)"
                                );
                            }
                            // CAPTURE: frees are deferred so every allocation
                            // is a stable cache hit (no within-capture
                            // aliasing, no unstable graph memory nodes).
                            if let candle_core::Device::Cuda(cd) = self.device() {
                                cd.set_capture_mode(true);
                            }
                            let cl = context_lens.clone();
                            let pid = position_ids.clone();
                            // Open the per-layer trace BEFORE capture begins so
                            // the recorded forward's own layer outputs are the
                            // handles we keep. Costs 43 Arc bumps, no device
                            // work, so it is safe inside the capture region.
                            if arc_layer_bisect() {
                                crate::layers::arc_layer_trace_begin();
                            }
                            let cap_result = match runner.begin_capture(bs) {
                                Ok((gp, op)) => {
                                    match self.model.forward(
                                        &input_ids,
                                        &seqlen_offsets,
                                        cl,
                                        pid,
                                        paged_attn_meta.as_ref().map(|(a, b)| (a.clone(), b)),
                                        &flash_meta,
                                    ) {
                                        Ok(output) => {
                                            tracing::info!(
                                                    "ARC capture: V4 forward RECORDED (bs={bs}); instantiating + launching"
                                                );
                                            match runner.end_capture_and_cache(bs, output, gp, op) {
                                                Ok(out) => {
                                                    tracing::info!(
                                                        "ARC capture: graph CAPTURED + launched OK"
                                                    );
                                                        if arc_layer_bisect() {
                                                            let bufs =
                                                                crate::layers::arc_layer_trace_take();
                                                            tracing::error!(
                                                                "ARC BISECT: holding {} captured layer buffers",
                                                                bufs.as_ref()
                                                                    .map(|b| b.len())
                                                                    .unwrap_or(0)
                                                            );
                                                            ARC_GRAPH_LAYER_BUFS
                                                                .with(|c| *c.borrow_mut() = bufs);
                                                        }
                                                        // ── SAME-STEP PROBE ─────────────
                                                        // Separates the two ways a graph can
                                                        // be wrong, which the replay-step
                                                        // comparison cannot tell apart.
                                                        //
                                                        // Here the position buffer, the token
                                                        // ids and the KV ring are EXACTLY what
                                                        // capture recorded against — nothing
                                                        // has advanced. So:
                                                        //   Δ ≈ 0  ⇒ the graph is wired right
                                                        //            and the replay-step Δ is
                                                        //            stale per-step state.
                                                        //   Δ large ⇒ a plain wiring fault
                                                        //            (wrong buffer / wrong
                                                        //            node); position state is
                                                        //            innocent.
                                                        // `error!` on purpose: this is the
                                                        // measurement the run exists for and
                                                        // must never be filtered out.
                                                        if std::env::var("ARC_GRAPH_SAMESTEP")
                                                            .as_deref()
                                                            == Ok("1")
                                                        {
                                                            if let candle_core::Device::Cuda(cd) =
                                                                self.device()
                                                            {
                                                                cd.set_capture_mode(false);
                                                            }
                                                            match self.model.forward(
                                                                &input_ids,
                                                                &seqlen_offsets,
                                                                context_lens.clone(),
                                                                position_ids.clone(),
                                                                paged_attn_meta
                                                                    .as_ref()
                                                                    .map(|(a, b)| (a.clone(), b)),
                                                                &flash_meta,
                                                            ) {
                                                                Ok(eager_same) => {
                                                                    match max_abs_diff(
                                                                        &out,
                                                                        &eager_same,
                                                                    ) {
                                                                        Ok(d) => tracing::error!(
                                                                            "ARC SAMESTEP: first-launch vs eager, SAME step: max|Δ|={d:.3e}"
                                                                        ),
                                                                        Err(e) => tracing::error!(
                                                                            "ARC SAMESTEP: first-launch compare failed: {e}"
                                                                        ),
                                                                    }
                                                                    // Keep this step's eager
                                                                    // logits: every later replay
                                                                    // is measured against them
                                                                    // too, which is what tells
                                                                    // "frozen" from "partially
                                                                    // advancing".
                                                                    ARC_CAPTURE_STEP_EAGER.with(
                                                                        |c| {
                                                                            *c.borrow_mut() =
                                                                                Some(
                                                                                    eager_same
                                                                                        .clone(),
                                                                                );
                                                                        },
                                                                    );
                                                                    // Second launch of the same
                                                                    // graph, still same step.
                                                                    // Distinguishes "the graph
                                                                    // is wrong" from "re-launch
                                                                    // specifically is wrong".
                                                                    match runner.replay(bs) {
                                                                        Ok(again) => {
                                                                            match max_abs_diff(
                                                                                &again,
                                                                                &eager_same,
                                                                            ) {
                                                                                Ok(d) => tracing::error!(
                                                                                    "ARC SAMESTEP: re-launch vs eager, SAME step: max|Δ|={d:.3e}"
                                                                                ),
                                                                                Err(e) => tracing::error!(
                                                                                    "ARC SAMESTEP: re-launch compare failed: {e}"
                                                                                ),
                                                                            }
                                                                        }
                                                                        Err(e) => tracing::error!(
                                                                            "ARC SAMESTEP: second launch failed: {e}"
                                                                        ),
                                                                    }
                                                                }
                                                                Err(e) => tracing::error!(
                                                                    "ARC SAMESTEP: eager forward failed: {e}"
                                                                ),
                                                            }
                                                        }
                                                        Some(out)
                                                    }
                                                    Err(e) => {
                                                        tracing::warn!(
                                                            "ARC capture: instantiate/launch failed: {e}; eager"
                                                        );
                                                    None
                                                }
                                            }
                                        }
                                        Err(e) => {
                                            runner.cancel_capture(gp, op);
                                            tracing::warn!(
                                                    "ARC capture: forward errored DURING capture (likely a host sync): {e}; eager"
                                                );
                                            None
                                        }
                                    }
                                }
                                Err(e) => {
                                    tracing::warn!("ARC capture: begin_capture failed: {e}; eager");
                                    None
                                }
                            };
                            // Safe ONLY because every pool-destroying path in
                            // `graph.rs` now drains the alloc cache before
                            // `cuMemPoolDestroy`. This call moves `deferred`
                            // into the reusable `free` list; before the drain
                            // ordering existed, on an error path that moved
                            // pointers whose private pool had ALREADY been
                            // destroyed straight into the list the allocator
                            // serves from. On the success path the pool is
                            // still alive (owned by `CapturedGraph`), so the
                            // pointers are valid and reuse is correct.
                            if let candle_core::Device::Cuda(cd) = self.device() {
                                cd.set_capture_mode(false);
                            }
                            cap_result
                        } else if runner.has_graph(bs) {
                            let t = std::time::Instant::now();
                            let replayed = runner.replay(bs);
                            match replayed {
                                Ok(out) => {
                                    let dt = t.elapsed();
                                    if runner.needs_verification()
                                        || std::env::var("ARC_GRAPH_VERIFY_ALWAYS").as_deref()
                                            == Ok("1")
                                    {
                                        // RUN-161 step 3. Turning replay
                                        // output on is the one change here
                                        // that can corrupt tokens SILENTLY:
                                        // a graph reading a stale address
                                        // returns plausible logits, not an
                                        // error. So prove it against eager
                                        // before trusting it, and use the
                                        // eager result this step either way
                                        // — it is already computed and it is
                                        // the one we know is right.
                                        // `out` above came from the replay, so
                                        // the held capture-time layer handles
                                        // now hold the REPLAY's per-layer
                                        // values. Trace the eager forward and
                                        // compare layer by layer: the first
                                        // index that differs is the first layer
                                        // reading a stale per-step input.
                                        if arc_layer_bisect() {
                                            crate::layers::arc_layer_trace_begin();
                                        }
                                        let eager = self.model.forward(
                                            &input_ids,
                                            &seqlen_offsets,
                                            context_lens.clone(),
                                            position_ids.clone(),
                                            paged_attn_meta.as_ref().map(|(a, b)| (a.clone(), b)),
                                            &flash_meta,
                                        )?;
                                        if arc_layer_bisect() {
                                            let eager_layers =
                                                crate::layers::arc_layer_trace_take();
                                            ARC_GRAPH_LAYER_BUFS.with(|c| {
                                                if let (Some(g), Some(e)) =
                                                    (c.borrow().as_ref(), eager_layers.as_ref())
                                                {
                                                    let n = g.len().min(e.len());
                                                    let mut first = None;
                                                    let mut line = String::new();
                                                    for i in 0..n {
                                                        let d = max_abs_diff(&g[i], &e[i])
                                                            .unwrap_or(f32::NAN);
                                                        if i < 12 || first.map_or(false, |f| i <= f + 2)
                                                        {
                                                            line.push_str(&format!(
                                                                " L{i}={d:.2e}"
                                                            ));
                                                        }
                                                        if first.is_none()
                                                            && !(d <= GRAPH_REPLAY_TOLERANCE)
                                                        {
                                                            first = Some(i);
                                                            line.push_str("<<FIRST");
                                                        }
                                                    }
                                                    // Slot map: 0=graph length
                                                    // mask, 1=embedding,
                                                    // 2=mHC lift, 3+=layer
                                                    // (slot-3).
                                                    tracing::error!(
                                                        "ARC BISECT: slots={n} first_diverging={:?} [0=mask 1=embed 2=lift then per graph-arm layer: k, k_full, attn, layer_out]{line}",
                                                        first
                                                    );
                                                }
                                            });
                                        }
                                        // ── DISTANCE-FROM-CAPTURE PROBE ──
                                        // The same-step probe already proved
                                        // the graph is wired right (Δ=0 at the
                                        // capture step), so whatever diverges
                                        // here is per-step state that must
                                        // advance and does not. This says WHICH
                                        // kind of stale:
                                        //   Δ vs capture-step eager ≈ 0 ⇒ the
                                        //     graph is FROZEN — its static
                                        //     inputs are not being read at all.
                                        //   Δ nonzero against BOTH ⇒ partially
                                        //     advancing: some inputs update
                                        //     (positions) and some are stale.
                                        // Also tracks how Δ moves with distance
                                        // from the capture step, which a frozen
                                        // input and a drifting one separate on.
                                        ARC_CAPTURE_STEP_EAGER.with(|c| {
                                            if let Some(cap_eager) = c.borrow().as_ref() {
                                                let n = ARC_REPLAY_DISTANCE
                                                    .with(|d| {
                                                        let mut d = d.borrow_mut();
                                                        *d += 1;
                                                        *d
                                                    });
                                                let d_cap = max_abs_diff(&out, cap_eager);
                                                let d_cap_eager = max_abs_diff(&eager, cap_eager);
                                                tracing::error!(
                                                    "ARC DISTANCE +{n}: replay-vs-CAPTURE-eager={:?} \
                                                     eager-vs-CAPTURE-eager={:?}",
                                                    d_cap.map(|v| format!("{v:.3e}")),
                                                    d_cap_eager.map(|v| format!("{v:.3e}")),
                                                );
                                            }
                                        });
                                        // ── ORDER PROBE ──────────────────
                                        // `out` above was produced BEFORE this
                                        // step's eager forward ran. Launch the
                                        // same graph again now, AFTER it. The
                                        // two answers separate the only two
                                        // remaining causes:
                                        //   still diverges ⇒ the graph reads an
                                        //     input that is genuinely stale.
                                        //   now matches ⇒ the graph is not
                                        //     stale at all; it depends on
                                        //     per-step work done by HOST code
                                        //     inside the eager forward, which a
                                        //     replay never executes. That is a
                                        //     missing-recorded-work bug, and a
                                        //     completely different fix.
                                        // Gated: it costs one extra launch and
                                        // inflates the replay counter.
                                        //
                                        // 🔴 `replay` hands back a CLONE of the
                                        // captured output tensor, so a second
                                        // launch overwrites `out`'s storage.
                                        // The verdict must therefore be taken
                                        // from `out` BEFORE the extra launch,
                                        // or the gate silently grades the wrong
                                        // tensor.
                                        let verdict = max_abs_diff(&out, &eager);
                                        if std::env::var("ARC_GRAPH_SAMESTEP").as_deref()
                                            == Ok("1")
                                        {
                                            let before =
                                                verdict.as_ref().map(|v| format!("{v:.3e}"));
                                            let after = runner
                                                .replay(bs)
                                                .and_then(|again| max_abs_diff(&again, &eager))
                                                .map(|v| format!("{v:.3e}"));
                                            tracing::error!(
                                                "ARC ORDER: replay-before-eager={before:?} \
                                                 replay-after-eager={after:?}"
                                            );
                                        }
                                        match verdict {
                                                Ok(d) if d <= GRAPH_REPLAY_TOLERANCE => {
                                                    tracing::info!(
                                                        "ARC capture: replay matched eager \
                                                         (max|Δ|={d:.3e}, replay={dt:?})"
                                                    );
                                                    runner.record_verification_pass();
                                                }
                                                Ok(d) => runner.record_verification_failure(
                                                    &format!("max|Δ| = {d:.3e} exceeds {GRAPH_REPLAY_TOLERANCE:.3e}"),
                                                ),
                                                Err(e) => runner.record_verification_failure(
                                                    &format!("could not compare outputs: {e}"),
                                                ),
                                            }
                                        Some(eager)
                                    } else if runner.replay_output_trusted() {
                                        tracing::debug!("ARC capture: replay {dt:?} (output USED)");
                                        Some(out)
                                    } else {
                                        // Verification failed earlier; the
                                        // graph still replays but its output
                                        // is not to be believed.
                                        None
                                    }
                                }
                                Err(e) => {
                                    tracing::warn!("ARC capture: replay failed: {e}");
                                    None
                                }
                            }
                        } else {
                            None
                        };
                        self.cuda_graph_runner = Some(runner);
                        result
                    } else {
                        None
                    };
                    match captured {
                        Some(o) => o,
                        None => {
                            let attempt = self.model.forward(
                                &input_ids,
                                &seqlen_offsets,
                                context_lens.clone(),
                                position_ids.clone(),
                                paged_attn_meta.as_ref().map(|(a, b)| (a.clone(), b)),
                                &flash_meta,
                            );
                            match attempt {
                                Ok(o) => o,
                                // AVAILABILITY. This forward runs with graph-mode
                                // positions SET, so it takes V4's graph KV arm —
                                // which can legitimately refuse (the pinned graph
                                // buffer must not be resized under a captured
                                // graph). Propagating that `?` would KILL THE
                                // REQUEST, and it would do so on real traffic: the
                                // trigger is a sequence longer than whatever the
                                // buffer was pinned at. Trading a user's answer for
                                // a memory saving is a worse bug than the one the
                                // pin fixes, and a quieter one — the OOM it
                                // replaced at least failed loudly at startup.
                                //
                                // So degrade instead: drop graph mode, disable the
                                // runner for the rest of the process, and retry
                                // eagerly. `append` (the eager path) has no pin and
                                // grows normally, so the retry succeeds and the user
                                // still gets an answer — slower, and correct.
                                Err(e) if crate::layers::has_graph_mode_positions() => {
                                    tracing::warn!(
                                        "ARC capture: graph-mode forward failed ({e}); falling back                                          to eager for this and every later step so the request still                                          completes. CUDA graphs are now OFF for this process."
                                    );
                                    crate::layers::set_graph_mode_positions(None);
                                    if let Some(r) = self.cuda_graph_runner.as_mut() {
                                        r.disable();
                                    }
                                    // THE CACHE WAS NEVER DRAINED. `set_alloc_cache_enabled(true)`
                                    // is called once at the start of capture and nothing ever
                                    // called it with `false`, so every buffer it ever held —
                                    // including private-pool allocations from every capture —
                                    // was retained for the process lifetime. Now that graphs are
                                    // off for good, the cache buys nothing and its buffers are
                                    // returned to the driver.
                                    if let candle_core::Device::Cuda(cd) = self.device() {
                                        cd.set_alloc_cache_enabled(false);
                                    }
                                    self.model.forward(
                                        &input_ids,
                                        &seqlen_offsets,
                                        context_lens,
                                        position_ids,
                                        paged_attn_meta.as_ref().map(|(a, b)| (a.clone(), b)),
                                        &flash_meta,
                                    )?
                                }
                                Err(e) => return Err(e),
                            }
                        }
                    }
                }
                #[cfg(not(feature = "cuda"))]
                {
                    self.model.forward(
                        &input_ids,
                        &seqlen_offsets,
                        context_lens,
                        position_ids,
                        paged_attn_meta.as_ref().map(|(a, b)| (a.clone(), b)),
                        &flash_meta,
                    )?
                }
            }
            true => self.model.xlora_forward(
                &input_ids,
                input_ids_full.as_ref().unwrap_or(&input_ids),
                &seqlen_offsets,
                seqlen_offsets_full.as_ref().unwrap_or(&seqlen_offsets),
                self.no_kv_cache,
                &self.non_granular_state,
                context_lens,
                position_ids,
                &flash_meta,
                flash_meta_full.as_ref().unwrap_or(&flash_meta),
            )?,
        };
        if return_raw_logits {
            Ok(ForwardInputsResult::RawLogits { logits })
        } else {
            Ok(ForwardInputsResult::CausalGeneration { logits })
        }
    }
    async fn sample_causal_gen(
        &self,
        seqs: &mut [&mut Sequence],
        logits: Vec<Tensor>,
        prefix_cacher: &mut PrefixCacheManagerV2,
        disable_eos_stop: bool,
        rng: Arc<std::sync::Mutex<Isaac64Rng>>,
    ) -> Result<(), candle_core::Error> {
        sample_and_add_toks(self, seqs, logits, prefix_cacher, disable_eos_stop, rng).await
    }
    fn category(&self) -> ModelCategory {
        ModelCategory::Text
    }
    /// Forward to the underlying NormalModel — only V4 currently returns a
    /// real kit; all other models return `None` and the engine falls back to
    /// non-MTP decoding.
    fn mtp_decode_kit(&self) -> Option<super::mtp_pipeline::MtpDecodeKit> {
        self.model.mtp_decode_kit()
    }
    #[cfg(feature = "cuda")]
    fn cuda_graph_runner_mut(&mut self) -> Option<&mut arc_cuda_graph::CudaGraphRunner> {
        self.cuda_graph_runner.as_mut()
    }
    #[cfg(feature = "cuda")]
    fn dedicated_decode_mut(&mut self) -> Option<&mut arc_cuda_graph::DedicatedDecodePath> {
        self.dedicated_decode.as_mut()
    }
    #[cfg(feature = "cuda")]
    fn autonomous_runner_mut(&mut self) -> Option<&mut arc_cuda_graph::AutonomousDecodeRunner> {
        self.autonomous_runner.as_mut()
    }

    /// Best-effort GPU-autonomous decode override. The first time this is
    /// invoked we attempt to lazily allocate an `AutonomousDecodeRunner` and
    /// capture its CUDA graph using the dedicated decode path's forward
    /// kernels. If anything goes wrong (no dedicated path, no CUDA device,
    /// graph capture fails) we log + return `Ok(None)` so the engine falls
    /// back to the standard step-by-step decode path.
    ///
    /// On a successful capture, subsequent calls replay the graph (one
    /// cuGraphLaunch on CUDA 12.4+, host-driven loop otherwise) and return
    /// the generated token IDs per sequence.
    ///
    /// IMPORTANT: **capture is not wired, so this method never returns
    /// `Some`.** The per-step metadata bridge this comment used to name as the
    /// missing piece now exists — `prime_for_step` is called below, from the
    /// engine's `AutonomousDecodeContext`. What is still missing is the
    /// `paged_state_factory` that `AutonomousDecodeRunner::capture_via_decode_forward`
    /// requires, and the `DedicatedDecodePath::has_cached_kv_info()` accessor
    /// its precondition was meant to test. Neither exists; see the comment at
    /// the unconditional `return Ok(None)` below for exactly what each needs.
    ///
    /// For DeepSeek-V4 none of that is reached in any case: `dedicated_decode`
    /// is `None` (the dense-layer inventory refuses V4 at `normal.rs:1544`), so
    /// the first guard in this method returns and every guard after it is dead
    /// code on that model.
    #[cfg(feature = "cuda")]
    fn autonomous_decode(
        &mut self,
        input_seqs: &mut [&mut Sequence],
        ctx: &crate::pipeline::AutonomousDecodeContext<'_>,
    ) -> std::result::Result<Option<Vec<Vec<i32>>>, candle_core::Error> {
        if input_seqs.is_empty() {
            return Ok(None);
        }

        // Lazy init: build the runner + capture the graph on first call.
        if self.autonomous_runner.is_none() {
            // Decision: require the dedicated decode path to be available.
            // Without it we have nothing to put in the capture closure
            // (Candle forward isn't graph-capturable in our current path).
            //
            // THIS GUARD CLOSES FIRST, and for DeepSeek-V4 it is the *only* one
            // that ever evaluates. `dedicated_decode` is set at exactly one site
            // (`normal.rs:1718`) and V4 is refused there by the dense-layer
            // inventory, so every guard below — including the `cache_config`
            // one, which our own notes blamed — is dead code on V4. That matters
            // because those guards carry `mark_unreachable` calls: their absence
            // from an ARC_PROFILE report is not evidence that they passed, it is
            // evidence that control never reached them. Report from here so the
            // first closing guard is the one that speaks.
            if self.dedicated_decode.is_none() {
                arc_profiler::mark_unreachable(
                    "cuda_graph.autonomous_decode",
                    "the dedicated decode path is unavailable (normal.rs:1718 declined this \
                     architecture, or ARC_NO_DEDICATED_DECODE=1), so there is no forward closure \
                     to capture; every guard below this one is unreachable",
                    "normal.rs:2403",
                );
                tracing::debug!(
                    "autonomous_decode: dedicated decode path unavailable; \
                     skipping autonomous capture (engine will use step-by-step decode)"
                );
                return Ok(None);
            }

            // Probe the device from any anchor. We need a CUDA device to
            // allocate the runner's pinned ring buffer + graph state.
            let device = self.device();
            if !matches!(device, Device::Cuda(_)) {
                return Ok(None);
            }

            // Derive sampling config from the first sequence. All sequences
            // in a single decode batch share the same sampling settings in
            // the current scheduler, so this is safe; if that ever changes
            // we'll have to refuse the batch and fall back.
            let first_sampler = input_seqs[0].sampler();
            if first_sampler.has_custom_logits_processors() {
                tracing::debug!(
                    "autonomous_decode: custom logits processors present; \
                     GPU sampler cannot run them — falling back"
                );
                return Ok(None);
            }
            if first_sampler.top_nsigma().is_some() {
                tracing::debug!(
                    "autonomous_decode: top_nsigma set; \
                     GPU sampler does not implement it — falling back"
                );
                return Ok(None);
            }
            if first_sampler.has_logits_bias() {
                tracing::debug!(
                    "autonomous_decode: logit_bias set; the GPU sampler draws from the \
                     model's raw logits and would silently drop it — falling back"
                );
                return Ok(None);
            }

            // Vocab size comes from the dedicated decode path (it has the
            // model config we extracted at load time).
            let vocab_size = self
                .dedicated_decode
                .as_ref()
                .map(|d| d.weights().config.vocab_size)
                .unwrap_or(0);
            if vocab_size == 0 {
                tracing::debug!("autonomous_decode: vocab_size==0, falling back");
                return Ok(None);
            }

            let block_size = match self.metadata.cache_config.as_ref() {
                Some(c) => c.block_size,
                None => {
                    // NOT the V4 guard, despite what this used to claim. V4
                    // cannot reach this line at all — `dedicated_decode` is
                    // `None` for it and the guard at :2404 returned already. The
                    // reason V4 never had a `cache_config` is real, but it is
                    // not what closes the door on V4, and attributing it here
                    // sent readers to `supports_paged_attention()` instead of to
                    // the dense-layer inventory.
                    arc_profiler::mark_unreachable(
                        "cuda_graph.autonomous_decode.cache_config",
                        "cache_config is None, so no PagedAttention cache exists to capture \
                         against; the runner cannot be sized",
                        "normal.rs:2464",
                    );
                    tracing::debug!("autonomous_decode: no cache_config, falling back");
                    return Ok(None);
                }
            };

            let max_blocks_per_seq = (self.metadata.max_seq_len + block_size - 1) / block_size;
            let eos_token_id = self.metadata.eos_tok.first().copied().unwrap_or(0) as i32;
            let greedy = first_sampler.is_greedy();
            let temperature = first_sampler.temperature().unwrap_or(1.0) as f32;
            let top_p = first_sampler.top_p() as f32;
            let frequency_penalty = first_sampler.frequency_penalty().unwrap_or(0.0);
            let presence_penalty = first_sampler.presence_penalty().unwrap_or(0.0);

            let config = arc_cuda_graph::AutonomousDecodeConfig {
                padded_batch_size: input_seqs.len().max(1),
                // The GPU loop runs up to `max_tokens` iterations. We use the
                // model's max seq len as a safe upper bound — per-sequence
                // max_len enforcement still happens on the engine side after
                // tokens are flushed.
                max_tokens: self.metadata.max_seq_len.max(1),
                max_blocks_per_seq,
                block_size,
                eos_token_id,
                vocab_size,
                temperature,
                top_p,
                frequency_penalty,
                presence_penalty,
                greedy,
            };

            self.autonomous_runner =
                arc_cuda_graph::try_init_autonomous_runner(&device, config.clone());
            if self.autonomous_runner.is_none() {
                tracing::warn!(
                    "autonomous_decode: try_init_autonomous_runner returned None; falling back"
                );
                return Ok(None);
            }

            // Ensure dedicated path buffers exist for the batch size. The
            // captured graph reads activation buffers from the dedicated path
            // (we only override token_ids/positions with runner pointers).
            let bs = config.padded_batch_size;
            {
                let dedicated = self.dedicated_decode.as_mut().unwrap();
                if let Err(e) = dedicated.ensure_and_get_buffers(bs) {
                    tracing::warn!(
                        "autonomous_decode: dedicated.ensure_and_get_buffers failed: {e}; falling back"
                    );
                    self.autonomous_runner = None;
                    return Ok(None);
                }
            }

            // Prime the runner's input buffers with the per-step context the
            // engine just built. The captured graph reads from these buffers
            // on each iteration; `launch_decode_step_update` then advances
            // them in place.
            {
                let runner = self.autonomous_runner.as_mut().unwrap();
                let pad = config.padded_batch_size;
                // Build padded copies of the per-row vectors. The engine sent
                // unpadded slices sized by `input_seqs.len()`.
                let mut padded_tokens: Vec<i32> = ctx.next_token_ids.to_vec();
                let mut padded_positions: Vec<i32> = ctx.positions.to_vec();
                let mut padded_ctx_lens: Vec<i32> = ctx.context_lens.to_vec();
                let mut padded_slots: Vec<i64> = ctx.slot_mappings.to_vec();
                padded_tokens.resize(pad, 0);
                padded_positions.resize(pad, 0);
                padded_ctx_lens.resize(pad, 0);
                padded_slots.resize(pad, -1);
                let row_len = ctx.max_blocks_per_seq;
                let mut padded_bt: Vec<i32> = ctx.block_tables_flat.to_vec();
                padded_bt.resize(pad * row_len, 0);
                if let Err(e) = runner.prime_for_step(
                    &padded_tokens,
                    &padded_positions,
                    &padded_bt,
                    &padded_ctx_lens,
                    &padded_slots,
                ) {
                    tracing::warn!("autonomous_decode: prime_for_step failed: {e}; falling back");
                    self.autonomous_runner = None;
                    return Ok(None);
                }
            }

            // ───────────────────────────────────────────────────────────────
            // CAPTURE IS NOT WIRED. This return is unconditional, and it is
            // guarding work that was described and never written — not a
            // leftover. Removing it would call nothing, because the two things
            // a call to `AutonomousDecodeRunner::capture_via_decode_forward`
            // needs do not exist:
            //
            //   1. `DedicatedDecodePath::has_cached_kv_info()`. The comment this
            //      replaces said it was "added in the same change as the
            //      accessor below". It was not — there is no such method on
            //      `DedicatedDecodePath` anywhere in the workspace. The
            //      precondition it was meant to test (has the dedicated path
            //      seen a real `PagedAttentionState` yet, i.e. is
            //      `cached_layer_caches` populated?) is therefore untested, and
            //      the fall-back is unconditional instead of conditional.
            //
            //   2. The `paged_state_factory: Fn() -> PagedAttentionState` that
            //      `capture_via_decode_forward` takes. `AutonomousDecodeContext`
            //      carries only host-side per-row vectors (tokens, positions,
            //      block tables, slot mappings) — no KV-cache device pointers
            //      and no cache strides. Those are computed from the live
            //      `CacheEngine` tensors at `pipeline/mod.rs:871`, inside
            //      `try_dedicated_decode`, and are reachable here only via
            //      `DedicatedDecodePath::build_paged_attn_state`, which needs
            //      `&mut self` — while `self.autonomous_runner` is already
            //      mutably borrowed for the capture. A `Fn()` closure over the
            //      staged pointers is the missing piece; it is not a rename.
            //
            // Consequence: `capture` and `capture_via_decode_forward` have zero
            // callers, `is_captured()` is permanently false, and the replay path
            // below is dead. `arc-cuda-graph/src/lib.rs:171` already says this
            // in prose; this records where the chain is actually cut.
            //
            // Reported so a profile shows the tier as dark rather than absent.
            arc_profiler::mark_unreachable(
                "cuda_graph.autonomous_decode.capture",
                "capture is never attempted: DedicatedDecodePath::has_cached_kv_info() does not \
                 exist and there is no paged_state_factory to hand to \
                 AutonomousDecodeRunner::capture_via_decode_forward, so is_captured() stays false \
                 and every decode falls back to the step-by-step path",
                "normal.rs:2619",
            );
            tracing::warn!(
                "autonomous_decode: runner allocated (batch={}, max_tokens={}, vocab={}) but \
                 capture is NOT wired — capture_via_decode_forward has no caller and its \
                 paged_state_factory does not exist. This runner will never capture; decode \
                 falls back to the step-by-step path on every batch.",
                bs,
                config.max_tokens,
                vocab_size,
            );
            return Ok(None);
        }

        // Runner already exists. If it has been captured, run the loop.
        // Otherwise fall back (capture didn't complete on first call).
        let runner = match self.autonomous_runner.as_mut() {
            Some(r) => r,
            None => return Ok(None),
        };
        if !runner.is_captured() {
            // Try priming + run if captured later; for now fall back.
            // Suppress unused warning on ctx by referring to it.
            let _ = ctx;
            return Ok(None);
        }
        // Replay path: prime with current step's context, then run.
        let pad = runner.config.padded_batch_size;
        let mut padded_tokens: Vec<i32> = ctx.next_token_ids.to_vec();
        let mut padded_positions: Vec<i32> = ctx.positions.to_vec();
        let mut padded_ctx_lens: Vec<i32> = ctx.context_lens.to_vec();
        let mut padded_slots: Vec<i64> = ctx.slot_mappings.to_vec();
        padded_tokens.resize(pad, 0);
        padded_positions.resize(pad, 0);
        padded_ctx_lens.resize(pad, 0);
        padded_slots.resize(pad, -1);
        let row_len = ctx.max_blocks_per_seq;
        let mut padded_bt: Vec<i32> = ctx.block_tables_flat.to_vec();
        padded_bt.resize(pad * row_len, 0);
        if let Err(e) = runner.prime_for_step(
            &padded_tokens,
            &padded_positions,
            &padded_bt,
            &padded_ctx_lens,
            &padded_slots,
        ) {
            tracing::warn!("autonomous_decode (replay): prime_for_step failed: {e}; falling back");
            return Ok(None);
        }
        match runner.run_decode_loop() {
            Ok(tokens) => Ok(Some(tokens)),
            Err(e) => {
                tracing::warn!(
                    "autonomous_decode (replay): run_decode_loop failed: {e}; falling back"
                );
                Ok(None)
            }
        }
    }
}

impl AnyMoePipelineMixin for NormalPipeline {
    fn amoe_finish_training(&mut self, gate_model_id: Option<String>) -> candle_core::Result<()> {
        self.model.finish_training(gate_model_id)
    }
    fn amoe_layer_vars(&self) -> Vec<Vec<Var>> {
        self.model.get_vars()
    }
    fn amoe_base_model_trainable_params(&self) -> usize {
        self.model.trainable_params()
    }
    fn amoe_take_cached_gating_outputs(&mut self) -> Vec<Tensor> {
        self.model.take_cached_gating_outputs()
    }
    fn amoe_create_layers(
        &mut self,
        model_ids: Vec<String>,
        token: &TokenSource,
        revision: Option<String>,
        match_regex: &str,
        config: crate::amoe::AnyMoeConfig,
        dtype: candle_core::DType,
        dev: &Device,
        (prefix, mlp): (String, String),
        layers: Vec<usize>,
        expert_type: AnyMoeExpertType,
        silent: bool,
        gate_model_id: Option<String>,
    ) -> candle_core::Result<()> {
        let mut vbs = Vec::new();
        // Precompile regex here
        let regex = Regex::new(match_regex).map_err(candle_core::Error::msg)?;
        for model_id in model_ids {
            let model_id_str = &model_id;
            let model_id = Path::new(&model_id);

            let api = {
                let cache = GLOBAL_HF_CACHE.get().cloned().unwrap_or_default();
                let mut api = ApiBuilder::from_cache(cache)
                    .with_progress(!silent)
                    .with_token(get_token(token).map_err(candle_core::Error::msg)?);
                if let Some(cache_dir) = crate::hf_hub_cache_dir() {
                    api = api.with_cache_dir(cache_dir);
                }
                api.build().map_err(candle_core::Error::msg)?
            };
            let revision = revision.clone().unwrap_or("main".to_string());
            let api = api.repo(Repo::with_revision(
                model_id_str.clone(),
                RepoType::Model,
                revision.clone(),
            ));

            let mut filenames = vec![];
            for rfilename in
                api_dir_list!(api, model_id, true).filter(|x| x.ends_with(".safetensors"))
            {
                filenames.push(api_get_file!(api, &rfilename, model_id));
            }

            let regex = regex.clone();
            let match_regex_clone = match_regex.to_string();
            let layers_clone = layers.clone();
            let vb = from_mmaped_safetensors(
                filenames,
                vec![],
                Some(dtype),
                dev,
                vec![None],
                silent,
                None,
                move |key| {
                    if regex.is_match(&key) {
                        // Idx of the last char of the layer id, +1
                        // Assumes N.MLP
                        let last_layer_idx = key.find(&match_regex_clone).unwrap() - 1;
                        let first_layer_idx = key[..last_layer_idx].rfind('.').unwrap();
                        let layer_n = key[first_layer_idx + 1..last_layer_idx]
                            .parse::<usize>()
                            .unwrap();
                        layers_clone.contains(&layer_n) || layers_clone.is_empty()
                    } else {
                        false
                    }
                },
                Arc::new(|_| DeviceForLoadTensor::Base),
            )?;
            vbs.push(vb);
        }

        let gate_vb = if let Some(gate_model_id) = gate_model_id {
            let model_id_str = &gate_model_id;
            let model_id = Path::new(&gate_model_id);

            let api = {
                let cache = GLOBAL_HF_CACHE.get().cloned().unwrap_or_default();
                let mut api = ApiBuilder::from_cache(cache)
                    .with_progress(!silent)
                    .with_token(get_token(token).map_err(candle_core::Error::msg)?);
                if let Some(cache_dir) = crate::hf_hub_cache_dir() {
                    api = api.with_cache_dir(cache_dir);
                }
                api.build().map_err(candle_core::Error::msg)?
            };
            let revision = revision.clone().unwrap_or("main".to_string());
            let api = api.repo(Repo::with_revision(
                model_id_str.clone(),
                RepoType::Model,
                revision.clone(),
            ));

            let mut gate_filenames = vec![];
            for rfilename in
                api_dir_list!(api, model_id, true).filter(|x| x.ends_with(".safetensors"))
            {
                gate_filenames.push(api_get_file!(api, &rfilename, model_id));
            }
            assert_eq!(
                gate_filenames.len(),
                1,
                "Gate model ID must contain only one .safetensors file"
            );

            let vb = from_mmaped_safetensors(
                gate_filenames.clone(),
                vec![],
                Some(dtype),
                dev,
                vec![None],
                silent,
                None,
                |_| true,
                Arc::new(|_| DeviceForLoadTensor::Base),
            )?;
            info!(
                "Loaded gating layers from `{}`",
                gate_filenames[0].display()
            );
            Some(vb)
        } else {
            None
        };

        self.model.create_anymoe_layers(
            vbs.clone(),
            config.clone(),
            (prefix.clone(), mlp.clone()),
            layers.clone(),
            expert_type.clone(),
            gate_vb.clone(),
        )?;

        Ok(())
    }
    fn amoe_supported(&self) -> bool {
        self.model.amoe_supported()
    }
}

/// The caching-allocator policy's contract.
///
/// The allocator itself has **no tests at all** — not in this repo, and not in
/// the candle fork that implements it (`grep -rn "alloc_cache" --include="*.rs"`
/// over `candle-core` finds only `cuda_backend/device.rs` and
/// `cuda_backend/mod.rs`, with no `#[cfg(test)]` in either). Everything that
/// observes it needs a GPU, so nothing observes it in CI.
///
/// The decision of *when* it is on does not need a GPU, so it is tested here.
#[cfg(test)]
mod alloc_cache_policy_tests {
    use super::{alloc_cache_action, AllocCacheAction};

    /// The change this policy exists to make: a decode step turns the cache on
    /// without any capture probe being involved.
    #[test]
    fn decode_enables_the_cache() {
        assert_eq!(
            alloc_cache_action(1, false, false),
            AllocCacheAction::Enable
        );
    }

    /// Prefill must hand its buffers back. They are large, they are keyed by an
    /// exact byte size no later request will ask for, and nothing evicts them.
    #[test]
    fn prefill_drains_the_cache() {
        assert_eq!(
            alloc_cache_action(512, true, false),
            AllocCacheAction::DrainAndDisable
        );
    }

    /// Called once per forward, so it must be a no-op in the steady state
    /// rather than re-enabling (or re-draining) every step.
    #[test]
    fn steady_state_touches_nothing() {
        assert_eq!(alloc_cache_action(1, true, false), AllocCacheAction::Leave);
        assert_eq!(
            alloc_cache_action(512, false, false),
            AllocCacheAction::Leave
        );
    }

    /// The kill switch has to work from either state, including turning off a
    /// cache that a previous step already enabled — otherwise setting it
    /// mid-run would leave the buffers stranded.
    #[test]
    fn kill_switch_disables_and_drains_from_either_state() {
        assert_eq!(
            alloc_cache_action(1, true, true),
            AllocCacheAction::DrainAndDisable
        );
        assert_eq!(alloc_cache_action(1, false, true), AllocCacheAction::Leave);
    }

    /// A prompt that happens to be one token long is a prefill by every other
    /// measure, but it allocates decode-shaped buffers, so the policy keys on
    /// the shape it will actually see. Pinned so the equivalence is deliberate.
    #[test]
    fn one_token_prompt_is_treated_as_decode() {
        assert_eq!(
            alloc_cache_action(1, false, false),
            AllocCacheAction::Enable
        );
    }

    /// A zero-length forward is not decode. Guards against `seq_len == 0`
    /// (which `dims2().unwrap_or((0, 0))` produces on a shape error) quietly
    /// enabling the cache.
    #[test]
    fn degenerate_zero_length_forward_does_not_enable() {
        assert_eq!(
            alloc_cache_action(0, false, false),
            AllocCacheAction::Leave
        );
        assert_eq!(
            alloc_cache_action(0, true, false),
            AllocCacheAction::DrainAndDisable
        );
    }
}

/// The retention cap's parse.
///
/// The cap is the whole of the fix — candle's caching allocator had none, and
/// retained 6.04 MiB per decoded token forever — so the way it is configured has
/// to be unambiguous. In particular a typo must not fall back to "unbounded".
#[cfg(test)]
mod alloc_cache_capacity_tests {
    use super::parse_alloc_cache_capacity;

    #[test]
    fn unset_leaves_candles_default_in_force() {
        assert_eq!(parse_alloc_cache_capacity(None), None);
    }

    /// A typo is not a licence to run unbounded. `None` means "don't touch it",
    /// and candle's default is bounded, so a bad value degrades to bounded.
    #[test]
    fn an_unparsable_value_does_not_become_unbounded() {
        for bad in ["", "  ", "lots", "1GiB", "-1", "1.5"] {
            assert_eq!(parse_alloc_cache_capacity(Some(bad)), None, "{bad:?}");
        }
    }

    /// `0` is the documented escape hatch back to the old unbounded allocator,
    /// kept reachable only so its leak can be re-measured rather than argued
    /// about.
    #[test]
    fn zero_is_the_explicit_unbounded_opt_in() {
        assert_eq!(parse_alloc_cache_capacity(Some("0")), Some(usize::MAX));
    }

    #[test]
    fn megabytes_convert_and_do_not_overflow() {
        assert_eq!(parse_alloc_cache_capacity(Some("1")), Some(1024 * 1024));
        assert_eq!(
            parse_alloc_cache_capacity(Some(" 512 ")),
            Some(512 * 1024 * 1024)
        );
        assert_eq!(
            parse_alloc_cache_capacity(Some(&usize::MAX.to_string())),
            Some(usize::MAX),
            "saturates instead of wrapping to a tiny cap"
        );
    }
}

/// The env-flag polarity contract.
///
/// This is a bug class in this repo, not a hypothetical: `ARC_NO_DEDICATED_DECODE`
/// was read as `std::env::var_os(..).is_some()`, so `=0` — the spelling that
/// reads as "off" — turned the dedicated decode path *off*. Two A/B harnesses
/// used exactly that spelling for their control leg
/// (`arc-tools/arcgraph_heap_probe.sh`, `arc-tools/arcgraph_capture_probe.sh`),
/// so both arms of both experiments ran with the path disabled and the variable
/// under test cancelled out of the result. Nothing crashed and nothing logged a
/// complaint; the experiments measured a comparison that had never been made.
///
/// The parse is pure so the polarity can be asserted without mutating the
/// process environment, which every test in the binary shares.
#[cfg(test)]
mod env_flag_polarity_tests {
    use super::env_flag_value;

    /// THE REGRESSION. Under the old `var_os(..).is_some()` reading every one of
    /// these was "set" and therefore "yes". A presence test cannot tell them
    /// apart from `=1`, which is the whole defect.
    #[test]
    fn zero_and_its_spellings_are_off_not_merely_present() {
        for off in ["0", "false", "no", "off", "OFF", " 0 ", "False", ""] {
            assert_eq!(
                env_flag_value(Some(off)),
                Some(false),
                "{off:?} must read as OFF; a presence test reads it as ON"
            );
        }
    }

    #[test]
    fn one_and_its_spellings_are_on() {
        for on in ["1", "true", "yes", "on", "ON", " 1 ", "True"] {
            assert_eq!(env_flag_value(Some(on)), Some(true), "{on:?}");
        }
    }

    #[test]
    fn unset_is_off() {
        assert_eq!(env_flag_value(None), Some(false));
    }

    /// An unrecognised value is `None`, not `false`, so the caller can say so.
    /// Silently bucketing a typo is how a mis-set flag decides an experiment
    /// without anyone finding out.
    #[test]
    fn an_unrecognised_value_is_reported_rather_than_bucketed() {
        for bad in ["2", "enabled", "-1", "y", "t", "disable"] {
            assert_eq!(
                env_flag_value(Some(bad)),
                None,
                "{bad:?} must be reported, not silently treated as on or off"
            );
        }
    }
}
