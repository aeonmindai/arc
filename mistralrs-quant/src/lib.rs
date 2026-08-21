use std::{
    borrow::Cow,
    fmt::Debug,
    num::NonZeroUsize,
    sync::{atomic::AtomicUsize, Arc, Mutex, MutexGuard},
};

use blockwise_fp8::blockwise_fp8_linear_b;
use candle_core::{
    quantized::{GgmlDType, QMatMul, QTensor},
    DType, Device, Result, Tensor,
};
use pertensor_fp8::pertensor_fp8_linear_b;

#[cfg(feature = "metal")]
mod metal_kernels;

mod afq;
pub mod arc_kvquant;
// Not `#[cfg(feature = "cuda")]`: the allocation helpers inside are, but the
// POLICY TABLE is host logic, and gating the whole module put its table beyond
// reach of the free CPU test lane — the only lane that runs tests at all. See
// the module header for why the default matters enough to pin with a test.
mod arc_outbuf;
mod bitsandbytes;
mod blockwise_fp8;
pub mod calibration;
pub mod cublaslt;
pub mod cuda_peer;
pub mod distributed;
mod dummy;
pub mod env_flag;
pub mod f8q8;
mod fp8;
pub mod gemv;
mod gguf;
mod gptq;
mod hqq;
mod imatrix;
pub mod kvwrite;
mod lora;
mod mxfp4;
pub mod name_remap;
mod nvfp4;
mod pending_layer;
mod pertensor_fp8;
mod qtip;
pub mod rotary;
pub mod safetensors;
mod scalar_fp8;
pub mod td_moe_factored;
#[cfg(test)]
mod test_rng;
pub mod turboquant;
mod unquantized;
mod utils;
mod vector_fp8;

use gptq::gptq_linear;
use lora::merge_lora_weights;
pub use name_remap::{v4_scale_rename_rules, RenameRule, RenamingBackend};
use regex::Regex;
pub use safetensors::{attach_rename_rules, Shard, ShardedSafeTensors, ShardedVarBuilder};

pub use afq::{AfqBits, AfqGroupSize, AfqLayer};
pub use bitsandbytes::{BnbLinear, BnbQuantParams, BnbQuantType};
pub use blockwise_fp8::{
    blockwise_fp8_moe, fp8_blockwise_dequantize, fp8_blockwise_quantize,
    mx_int4_blockwise_dequantize, BlockwiseFP8Linear,
};
pub use calibration::{
    CalibAccumulator, CalibLayerData, CalibOptions, CalibrationArtifact, CalibrationMeta,
    ExpertCalibData, ExpertStatus, GramBlocks, GramLayout, GramMode, LayerCalibStats,
    CALIB_COLLECTOR_VERSION, CALIB_EXTENSION, CALIB_FORMAT_VERSION,
};
pub use cuda_peer::{enable_peer_access, PeerAccessReport, PeerAccessStatus, PeerPair};
pub use env_flag::{env_flag_is_set, env_flag_value};
pub use distributed::{
    layers::{
        compute_kv_shard, compute_n_kv_groups, ColumnParallelLayer, ExpertSubset, FusedExperts,
        PackedExperts, ReplicatedLayer, RowParallelLayer,
    },
    socket::{Client, Server},
    BarrierLike, Comm, Id, RingConfig, SumAllReduce,
};
pub use dummy::DummyLayer;
pub use f8q8::F8Q8Linear;
pub use fp8::FP8Linear;
#[cfg(feature = "cuda")]
pub use gemv::gemv;
pub use gemv::{should_use_gemv, GEMV_CONTROLLER};
pub use gguf::GgufMatMul;
pub use gptq::GptqLayer;
pub use hqq::{HqqAxis, HqqBits, HqqConfig, HqqLayer};
pub use imatrix::{CollectedImatrixData, ImatrixLayerStats};
pub use lora::{
    clear_applied_loras, get_applied_loras, linear_no_bias_static_lora, push_applied_lora,
    LoraAdapter, LoraConfig, StaticLoraConfig, MULTI_LORA_DELIMITER,
};
pub use mxfp4::MXFP4Layer;
pub use nvfp4::NVFP4Layer;
pub use pending_layer::PendingIsqLayer;
pub use pertensor_fp8::PerTensorFP8Linear;
/// TCFRAG-2B — the tensor-core fragment byte order for the K=2/V=1/L=16 rung,
/// and the D22 compute-capability-keyed layout table that selects it.
///
/// ⚠️ The kernel this format feeds is UNVERIFIED ON HARDWARE — never run.
pub use qtip::tcfrag2b;
/// The V=4 / L=12 decode family: a 32,768 B bf16 reproduction table with K as
/// a parameter over symbol extraction. K=8 is the byte-aligned control at
/// 2.00 bpw; K=9 (2.25 bpw) is the quality winner. See the module docs.
pub use qtip::trellis_v4l12;
pub use qtip::tune::{
    gemv_num_variants as qtip2b_gemv_num_variants, gemv_variant_applicable,
    gemv_variant_desc as qtip2b_gemv_variant_desc,
    gemv_variant_for_shape as qtip2b_gemv_variant_for_shape, set_forced_gemv_variant,
    spec_pin_gemm, GemvTuneEntry, GemvVariantDesc, QTIP2B_GEMV_BAKED_TABLE,
    QTIP2B_GEMV_VARIANT_LEGACY,
};
pub use qtip::{
    bake_cache, gpu_quantize_cpu_fallback_count, grouped_launch_counts, grouped_variant,
    hessian_row_weights, qtip_expected_distinct_experts, qtip_expected_pairs_per_distinct_expert,
    qtip_grouped_gemm_tile_fill, set_grouped_variant, viterbi_quantize_row,
    BakeCacheError, BakeKey, ExpertBpwTable, Qtip2bLayer, QtipBakeConfig, QtipCodebook, QtipLayer,
    QtipMode, QtipPackedView, QtipRotation, QtipSearchDetail, QtipSearchStamp, TrellisBpw,
    TrellisSearch, QTIP2B_MCG_MULT, QTIP_GATHER_GEMV_MAX_PAIRS, QTIP_GROUPED_TILE_K,
    QTIP_GROUPED_TILE_M, QTIP_GROUPED_TILE_N, QTIP_GROUPED_VARIANT_BASELINE,
    QTIP_GROUPED_VARIANT_COUNT, QTIP_GROUPED_VARIANT_ENV, QTIP_GROUPED_VARIANT_LDST,
    QTIP_GROUPED_VARIANT_TUNED, QTIP_ONDEVICE_MOE_MAX_TOKENS_ENV,
};
pub use td_moe_factored::TuckerFactoredLayer;
pub use unquantized::UnquantLinear;
pub use utils::bake_budget::{
    arm_bake_budget, bake_budget_armed, disarm_bake_budget, note_bake_layer, project_bake_peak,
    BakeBudgetVerdict, BakeProjection,
};
pub use utils::bake_devices::{
    bake_device_layer_counts, bake_device_ordinals, clear_bake_device_override, parse_bake_devices,
    reset_bake_device_layer_counts, set_bake_device_override, set_worker_cuda_ordinal,
    worker_cuda_ordinal, BAKE_DEVICES_ENV,
};
pub use utils::flash_attn_sinks_metal;
pub use utils::flash_attn_sinks_varlen_metal;
#[cfg(feature = "cuda")]
pub use utils::gptoss_swiglu_fused;
#[cfg(feature = "cuda")]
pub use utils::gptoss_swiglu_interleaved;
pub use utils::isq::apply_immediate_isq;
pub use utils::softmax_with_sinks;
pub use utils::{fused_glu, GluActivationType};
pub use utils::{log, BitWiseOp, CumSumOp, LeftshiftOp, NonZeroOp, SortOp, UQFF_QUANT_TYPE_OFFSET};
pub use vector_fp8::{fp8_vector_dequantize, fp8_vector_quantize};

use candle_nn::{Conv1d, Conv2d, Linear, Module};
use serde::{Deserialize, Deserializer, Serialize};

#[derive(Clone, Debug)]
pub struct ImmediateIsqParams {
    pub guard: QuantizeOntoGuard,
    pub ty: Option<IsqType>,
    pub predicates: Vec<Regex>,
    pub overrides: Vec<ImmediateIsqOverride>,
    /// Thread pool for parallel immediate ISQ on discrete GPUs.
    /// When `Some`, `apply_immediate_isq` will spawn quantization tasks
    /// on this pool and return `PendingIsqLayer` wrappers.
    pub pool: Option<Arc<rayon::ThreadPool>>,
}

#[derive(Clone, Debug)]
pub struct ImmediateIsqOverride {
    pub predicate: Regex,
    pub ty: Option<IsqType>,
    pub device: Option<Device>,
}

#[derive(Clone, Debug)]
pub struct ImmediateIsqMatch {
    pub ty: IsqType,
    pub device: Option<Device>,
}

thread_local! {
    static ENGINE_IMMEDIATE_ISQ: std::cell::RefCell<Option<ImmediateIsqParams>> = const { std::cell::RefCell::new(None) } ;
    static LOADING_FROM_UQFF: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Mark whether the current load is deserializing from a UQFF file. When true,
/// the per-expert MoE constructor builds DummyLayer placeholders instead of
/// re-loading + dequantizing + re-quantizing the base experts (~100s/layer) that
/// the UQFF already holds; the UQFF deserializer then fills the placeholders.
/// Self-resetting: every load sets it to its own `from_uqff` state. (RUN-161)
pub fn set_loading_from_uqff(v: bool) {
    LOADING_FROM_UQFF.with(|c| c.set(v));
}

pub fn loading_from_uqff() -> bool {
    LOADING_FROM_UQFF.with(|c| c.get())
}

/// Process-global: whether the current run is a UQFF *bake* whose quantized
/// output only has to be serialized, never used for a forward pass.
static BAKE_ISQ_TO_HOST: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Mark the current load as a UQFF bake, so quantized MoE expert stacks are
/// materialized on the **host** instead of being retained on the accelerator.
///
/// Why this exists (wave18): a bake used to leave every quantized layer
/// resident on the GPU, so device usage grew by the full artifact size —
/// ~1.6 GiB per layer for V4 Flash, ~68 GB over 43 layers — on top of the
/// per-layer quantize working set, and the allocator had to thread its
/// multi-GiB transients around a permanently growing set of blocks. A 43-layer
/// bake died at layer 28 on a 140 GB H200 with a 4 KB output directory.
///
/// A bake never runs a forward pass: the layers exist only to be handed to
/// `QuantizedSerde::serialize`, which works the same from host memory. The
/// quantize math still runs on the GPU — only the *result* comes back.
///
/// Not thread-local: the immediate-ISQ thread pool may run a layer's quantize
/// off the construction thread, and the answer must not depend on which thread
/// asks. Self-resetting: every load sets it to its own state.
pub fn set_bake_isq_to_host(v: bool) {
    BAKE_ISQ_TO_HOST.store(v, std::sync::atomic::Ordering::Relaxed);
}

/// Whether quantized expert stacks should land on the host — see
/// [`set_bake_isq_to_host`].
pub fn bake_isq_to_host() -> bool {
    BAKE_ISQ_TO_HOST.load(std::sync::atomic::Ordering::Relaxed)
}

pub fn set_immediate_isq(isq: Option<IsqType>, predicates: Vec<Regex>) {
    let (pool, _) = create_isq_thread_pool(isq);
    set_immediate_isq_with_pool(isq, predicates, Vec::new(), pool);
}

pub fn set_immediate_isq_with_pool(
    isq: Option<IsqType>,
    predicates: Vec<Regex>,
    overrides: Vec<ImmediateIsqOverride>,
    pool: rayon::ThreadPool,
) {
    ENGINE_IMMEDIATE_ISQ.with(|cell| {
        *cell.borrow_mut() = Some(ImmediateIsqParams {
            guard: QuantizeOntoGuard::new(),
            ty: isq,
            predicates,
            overrides,
            pool: Some(Arc::new(pool)),
        });
    });
}

/// Resolve the ISQ quantize thread count together with the one-line rationale
/// that justifies it, so a bake log always says *why* it picked N.
///
/// `device` is the device the quantize will target; pass `None` when the call
/// site does not know it and the compiled-in backend should be assumed.
pub fn isq_thread_policy(ty: Option<IsqType>, device: Option<&Device>) -> (usize, &'static str) {
    let backend = match device {
        Some(device) => IsqQuantizeBackend::for_device(device),
        None => IsqQuantizeBackend::assumed(),
    };
    isq_thread_policy_on(
        ty,
        backend,
        std::env::var("MISTRALRS_ISQ_SINGLETHREAD").is_ok(),
    )
}

/// [`isq_thread_policy`] with the backend and the force-single-thread escape
/// hatch supplied explicitly, so the decision is testable without touching
/// process-global environment state.
pub fn isq_thread_policy_on(
    ty: Option<IsqType>,
    backend: IsqQuantizeBackend,
    force_singlethread: bool,
) -> (usize, &'static str) {
    if force_singlethread {
        return (1, "MISTRALRS_ISQ_SINGLETHREAD is set");
    }
    match ty {
        Some(ty) => {
            let (threads, rationale) = ty.isq_cpu_thread_policy(backend);
            (
                threads
                    .map(usize::from)
                    .unwrap_or_else(rayon::current_num_threads),
                rationale,
            )
        }
        None => (
            rayon::current_num_threads(),
            "no ISQ type selected; all rayon workers",
        ),
    }
}

/// [`isq_thread_policy`] for a bake spread across `n_devices` devices
/// (wave22, `--bake-devices`).
///
/// The single-device answer is a **per-device submission width**, not a global
/// one. PR #25 capped QTIP at one host thread because N rayon workers all
/// submitting Viterbi kernels and transfers to *the same* device only contend
/// (session 5 measured 4-9 min/layer at 24 threads vs ~30 s/layer at 1). That
/// reasoning is about contention for one device and says nothing about a second
/// device sitting idle next to it, so with N devices the correct width is
/// `per_device * N`: one submitter each, never two on one.
///
/// The multiplication applies only when the per-device cap exists *because* the
/// quantize math runs on the device (`get_max_isq_cpu_threads_on` returns a
/// bound under [`IsqQuantizeBackend::Gpu`]). Rungs that quantize on host cores
/// — GGML Q*_K, FP8 — already ask for every core and gain nothing from extra
/// devices, so their width is left alone.
///
/// `MISTRALRS_ISQ_SINGLETHREAD` is honored as *one thread per device*, which is
/// what it has always meant: it exists to stop several host threads piling onto
/// a single device.
pub fn isq_thread_policy_for_devices(
    ty: Option<IsqType>,
    device: Option<&Device>,
    n_devices: usize,
) -> (usize, &'static str) {
    let backend = match device {
        Some(device) => IsqQuantizeBackend::for_device(device),
        None => IsqQuantizeBackend::assumed(),
    };
    isq_thread_policy_for_devices_on(
        ty,
        backend,
        std::env::var("MISTRALRS_ISQ_SINGLETHREAD").is_ok(),
        n_devices,
    )
}

/// [`isq_thread_policy_for_devices`] with the backend and the force-single-thread
/// escape hatch supplied explicitly, so the decision is testable without a
/// second GPU or process-global environment state.
pub fn isq_thread_policy_for_devices_on(
    ty: Option<IsqType>,
    backend: IsqQuantizeBackend,
    force_singlethread: bool,
    n_devices: usize,
) -> (usize, &'static str) {
    let (per_device, rationale) = isq_thread_policy_on(ty, backend, force_singlethread);
    let n_devices = n_devices.max(1);
    if n_devices == 1 {
        return (per_device, rationale);
    }
    // The cap only means "one submitter per device" when it exists *because*
    // the quantize math runs on the device. A rung that quantizes on host cores
    // already asked for every core and gains nothing from a second GPU.
    let device_bound = matches!(backend, IsqQuantizeBackend::Gpu)
        && ty.is_some_and(|ty| ty.get_max_isq_cpu_threads_on(backend).is_some());
    if device_bound {
        (
            per_device.saturating_mul(n_devices),
            "one submit thread per bake device — distinct devices do not contend (PR #25 capped submitters per device, not per box)",
        )
    } else {
        (
            per_device,
            "quantize math runs on host cores; extra bake devices add no submit width",
        )
    }
}

/// Create a rayon thread pool for parallel immediate ISQ.
/// Returns `(pool, num_threads)` so callers can log the thread count.
///
/// Assumes the compiled-in backend for the quantize target; use
/// [`create_isq_thread_pool_for_device`] when the device is known.
pub fn create_isq_thread_pool(ty: Option<IsqType>) -> (rayon::ThreadPool, usize) {
    create_isq_thread_pool_for_device(ty, None)
}

/// Create a rayon thread pool for parallel immediate ISQ, targeting a known device.
/// Returns `(pool, num_threads)` so callers can log the thread count.
///
/// Thread count is based on the quantization type *and* on where its quantize
/// math actually runs (see [`IsqQuantizeBackend`]):
/// - GGML types (Q2K-Q8K) and F8E4M3: `rayon::current_num_threads()` (CPU quantization)
/// - HQQ/AFQ: 1 thread (GPU quantization, serialized by `QuantizeOntoGuard`)
/// - QTIP: 1 thread on a GPU-backed quantize, all cores only on a CPU-side Viterbi
pub fn create_isq_thread_pool_for_device(
    ty: Option<IsqType>,
    device: Option<&Device>,
) -> (rayon::ThreadPool, usize) {
    let (num_threads, rationale) = isq_thread_policy(ty, device);
    tracing::info!("ISQ thread policy: {num_threads} thread(s) — {rationale}.");

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build()
        .expect("Failed to create ISQ thread pool");
    (pool, num_threads)
}

/// Environment override for the host-side expert-unpack width.
pub const EXPERT_UNPACK_THREADS_ENV: &str = "ARC_UNPACK_THREADS";

/// Resolve the expert-unpack width from an explicit override and the machine's
/// available parallelism, with no process-global state so it stays testable.
///
/// A malformed or zero override is ignored rather than honored: the failure
/// mode of `ARC_UNPACK_THREADS=oops` must not be a serial bake.
fn expert_unpack_threads_from(override_value: Option<&str>, available: usize) -> usize {
    override_value
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|n| *n >= 1)
        .unwrap_or_else(|| available.max(1))
}

/// How many threads unpack (dequantize) packed MoE experts on the host.
///
/// **This is deliberately NOT [`isq_thread_policy`], and the two must not be
/// collapsed back into one number.** They size different resources:
///
/// * [`isq_thread_policy`] sizes the pool that *submits quantize work to the
///   GPU*. For QTIP on a GPU backend it must stay at 1. Session 5 ran it at 24
///   and measured **4-9 min/layer** with no warning: every rayon worker
///   launched Viterbi kernels and host<->device copies against the one device,
///   so they only contended. PR #25 closed that trap and it stays closed.
/// * this sizes the pool that *unpacks INT4/FP8 expert weights into BF16 on the
///   CPU*. That work is pure host compute over mmap'd bytes, touches no device,
///   and is embarrassingly parallel across experts, so it wants every core.
///
/// Session 6 (H200, beam W=256) is why they are now separate: the beam search
/// made the GPU half of a layer ~64x cheaper, and the bake settled at
/// **240 s/layer** with the process at **100.5% CPU** (one core of 24) and
/// loadavg **1.14** — the CPU unpack, not the GPU search, had become the
/// limiter. Anything that reunites these two numbers reintroduces one of those
/// two failures, depending on which value survives.
///
/// Override with `ARC_UNPACK_THREADS` (see [`EXPERT_UNPACK_THREADS_ENV`]);
/// defaults to the machine's available parallelism. Never reads
/// `MISTRALRS_ISQ_SINGLETHREAD` — that flag is about GPU submission.
pub fn expert_unpack_threads() -> usize {
    let override_value = std::env::var(EXPERT_UNPACK_THREADS_ENV).ok();
    expert_unpack_threads_from(
        override_value.as_deref(),
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1),
    )
}

fn build_expert_unpack_pool(num_threads: usize) -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads.max(1))
        .thread_name(|i| format!("arc-unpack-{i}"))
        .build()
        .expect("Failed to create expert unpack thread pool")
}

/// Process-wide pool for the host-side expert unpack.
///
/// Callers `install` on it rather than using the ambient rayon context, so the
/// unpack keeps its full width *by construction* even when the calling thread
/// is a worker of a narrower pool (e.g. the 1-thread ISQ submission pool). See
/// [`expert_unpack_threads`] for why the two widths differ.
pub fn expert_unpack_pool() -> &'static rayon::ThreadPool {
    static POOL: std::sync::OnceLock<rayon::ThreadPool> = std::sync::OnceLock::new();
    POOL.get_or_init(|| {
        let num_threads = expert_unpack_threads();
        tracing::info!(
            "Expert unpack pool: {num_threads} thread(s) (host-side INT4/FP8 -> BF16 dequant; \
             independent of the ISQ thread policy, override with {EXPERT_UNPACK_THREADS_ENV})."
        );
        build_expert_unpack_pool(num_threads)
    })
}

pub fn get_immediate_isq() -> Option<ImmediateIsqParams> {
    ENGINE_IMMEDIATE_ISQ.with(|cell| cell.borrow().clone())
}

pub fn clear_immediate_isq() {
    ENGINE_IMMEDIATE_ISQ.with(|cell| {
        *cell.borrow_mut() = None;
    });
}

pub fn should_apply_immediate_isq(vb: &ShardedVarBuilder) -> bool {
    immediate_isq_match(vb).is_some()
}

pub fn immediate_isq_match(vb: &ShardedVarBuilder) -> Option<ImmediateIsqMatch> {
    let immediate_isq = get_immediate_isq()?;
    // Add a .weight to match the ISQ regexes!
    let prefix = format!("{}.weight", vb.prefix());
    resolve_immediate_isq(&immediate_isq, &prefix)
}

fn resolve_immediate_isq(params: &ImmediateIsqParams, prefix: &str) -> Option<ImmediateIsqMatch> {
    if let Some(override_hit) = params
        .overrides
        .iter()
        .find(|override_pred| override_pred.predicate.is_match(prefix))
    {
        if let Some(ty) = override_hit.ty.or(params.ty) {
            return Some(ImmediateIsqMatch {
                ty,
                device: override_hit.device.clone(),
            });
        }
        return None;
    }

    if let Some(ty) = params.ty {
        if params
            .predicates
            .iter()
            .any(|predicate| predicate.is_match(prefix))
        {
            return Some(ImmediateIsqMatch { ty, device: None });
        }
    }

    None
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "quant_method", rename_all = "lowercase")]
pub enum QuantizedConfig {
    GptqAwq {
        bits: usize,
        group_size: usize,
        checkpoint_format: Option<String>,
        is_awq: bool,
    },
    Fp8 {
        weight_block_size: Option<Vec<usize>>,
    },
    Bitsandbytes {
        bnb_4bit_quant_type: Option<String>,
    },
    Afq {
        bits: usize,
        group_size: usize,
    },
    MXFP4 {},
    NVFP4 {},
    Qtip {},
}

// Common fields for all variants
#[derive(Deserialize)]
struct RawConfig {
    quant_method: Option<String>,
    bits: Option<usize>,
    group_size: Option<usize>,
    checkpoint_format: Option<String>,
    weight_block_size: Option<Vec<usize>>,
    bnb_4bit_quant_type: Option<String>,
}

// Custom deserializer implementation
impl<'de> Deserialize<'de> for QuantizedConfig {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = RawConfig::deserialize(deserializer)?;

        match &raw.quant_method {
            Some(m) if m == "gptq" || m == "awq" => {
                let bits = raw
                    .bits
                    .ok_or_else(|| serde::de::Error::missing_field("bits"))?;
                let group_size = raw
                    .group_size
                    .ok_or_else(|| serde::de::Error::missing_field("group_size"))?;
                Ok(QuantizedConfig::GptqAwq {
                    bits,
                    group_size,
                    checkpoint_format: raw.checkpoint_format,
                    is_awq: m == "awq",
                })
            }
            Some(m) if m == "fp8" => {
                // weight_block_size is optional - None means per-tensor quantization
                Ok(QuantizedConfig::Fp8 {
                    weight_block_size: raw.weight_block_size,
                })
            }
            Some(m) if m == "bitsandbytes" => Ok(QuantizedConfig::Bitsandbytes {
                bnb_4bit_quant_type: raw.bnb_4bit_quant_type,
            }),
            Some(m) if m == "afq" => {
                let bits = raw
                    .bits
                    .ok_or_else(|| serde::de::Error::missing_field("bits"))?;
                let group_size = raw
                    .group_size
                    .ok_or_else(|| serde::de::Error::missing_field("group_size"))?;
                Ok(QuantizedConfig::Afq { bits, group_size })
            }
            Some(m) if m == "mxfp4" => {
                Ok(QuantizedConfig::MXFP4 {  })
            }
            Some(m) if m == "nvfp4" => {
                Ok(QuantizedConfig::NVFP4 {  })
            }
            Some(m) if m == "qtip" => {
                Ok(QuantizedConfig::Qtip {  })
            }
            None => {
                let bits = raw
                    .bits
                    .ok_or_else(|| serde::de::Error::missing_field("bits"))?;
                let group_size = raw
                    .group_size
                    .ok_or_else(|| serde::de::Error::missing_field("group_size"))?;
                Ok(QuantizedConfig::Afq { bits, group_size })
            }
            Some(unknown_method) => {
                Err(serde::de::Error::custom(format!(
                    "Unknown quantization method: {unknown_method}. Expected one of: gptq, fp8, bitsandbytes, afq, mxfp4, nvfp4, qtip, or not specified"
                )))
            },
        }
    }
}

impl QuantizedConfig {
    pub fn name(&self) -> &'static str {
        match self {
            Self::GptqAwq { .. } => "gptq",
            Self::Fp8 { .. } => "fp8",
            Self::Bitsandbytes { .. } => "bitsandbytes",
            Self::Afq { .. } => "afq",
            Self::MXFP4 { .. } => "mxfp4",
            Self::NVFP4 { .. } => "nvfp4",
            Self::Qtip { .. } => "qtip",
        }
    }

    pub fn get_bits_name(&self, _vb: &ShardedVarBuilder) -> String {
        match self {
            Self::GptqAwq { bits, .. } => format!("{bits} bits"),
            Self::Fp8 { .. } => "8 bits".to_string(),
            Self::Bitsandbytes {
                bnb_4bit_quant_type: Some(_),
            } => "4 bits".to_string(),
            Self::Bitsandbytes {
                bnb_4bit_quant_type: None,
            } => "8 bits".to_string(),
            Self::Afq { bits, .. } => format!("{bits} bits"),
            Self::MXFP4 {} => format!("{} bits", mxfp4::N_BITS),
            Self::NVFP4 {} => format!("{} bits", nvfp4::N_BITS),
            Self::Qtip {} => format!("{} bits", qtip::N_BITS),
        }
    }

    pub fn pack_factor(&self, dtype: DType) -> usize {
        match self {
            Self::GptqAwq { bits, .. } | Self::Afq { bits, .. } => match bits {
                2 => IsqType::Q2K.pack_factor(dtype),
                3 => IsqType::Q3K.pack_factor(dtype),
                4 => IsqType::Q4K.pack_factor(dtype),
                5 => IsqType::Q5K.pack_factor(dtype),
                6 => IsqType::Q6K.pack_factor(dtype),
                8 => IsqType::Q8_0.pack_factor(dtype),
                40 => 4, // mxfp4: 2 FP4 values per byte = factor of 4
                other => panic!("Unexpected bits in `pack_factor` {other}"),
            },
            Self::Fp8 { .. } => IsqType::Q8_0.pack_factor(dtype),
            Self::Bitsandbytes {
                bnb_4bit_quant_type: Some(_),
            }
            | Self::Bitsandbytes {
                bnb_4bit_quant_type: None,
            } => IsqType::Q4K.pack_factor(dtype),
            Self::MXFP4 {} => IsqType::Q4_0.pack_factor(dtype),
            Self::NVFP4 {} => IsqType::Q4_0.pack_factor(dtype),
            Self::Qtip {} => IsqType::Q2K.pack_factor(dtype),
        }
    }
}

#[derive(Debug, Clone)]
pub enum QuantMethodConfig {
    GptqAwq {
        bits: i32,
        use_exllama: bool,
        q_weight: Tensor,
        qzeros: Option<Tensor>,
        scales: Tensor,
        g_idx: Option<Tensor>,
        bias: Option<Tensor>,
        workspace: Option<Tensor>,
        is_marlin: bool,
        is_awq: bool,
    },
    Gguf {
        q_weight: Arc<QTensor>,
        b: Option<Tensor>,
    },
    Unquantized(Linear),
    Hqq {
        tensor: Tensor,
        bits: HqqBits,
        group_size: NonZeroUsize,
        axis: HqqAxis,
        optimization_steps: Option<usize>,
        round_zeros: Option<bool>,
        channel_wise: Option<bool>,
        bias: Option<Tensor>,
    },
    Dummy,
    FP8 {
        lin: Linear,
        dtype: DType,
    },
    Bnb {
        weight: Tensor,
        bias: Option<Tensor>,
        params: BnbQuantParams,
        quant_ty: BnbQuantType,
    },
    BlockwiseFP8 {
        weight: Tensor,
        weight_scale_inv: Tensor,
        bias: Option<Tensor>,
        dequant_dtype: DType,
        weight_block_size: Vec<usize>,
    },
    PerTensorFP8 {
        weight: Tensor,
        weight_scale_inv: Tensor,
        activation_scale: Option<Tensor>,
        bias: Option<Tensor>,
        dequant_dtype: DType,
    },
    Afq {
        weight: Tensor,
        bias: Option<Tensor>,
        bits: AfqBits,
        group_size: AfqGroupSize,
    },
    MXFP4 {
        blocks: Tensor,
        scales: Tensor,
        bias: Option<Tensor>,
    },
    NVFP4 {
        blocks: Tensor,
        scales: Tensor,
        tensor_scale: Tensor,
        bias: Option<Tensor>,
    },
    Qtip {
        blocks: Tensor,
        row_scales: Tensor,
        lut: Tensor,
        bias: Option<Tensor>,
        in_features: usize,
        /// Hadamard incoherence rotation signs [in_features], ±1.0. `None`
        /// when the layer was quantized without rotation (back-compat).
        rotation_signs: Option<Tensor>,
        /// Block size for the block-diagonal Hadamard rotation. 0 disables.
        rotation_block: usize,
    },
    /// QTIP computed-codebook "bitshift trellis" rung (`qtip2b`). Unlike
    /// [`QuantMethodConfig::Qtip`] there is no LUT tensor — the codebook is a
    /// pure function of the trellis state and the MCG multiplier. Blocks may
    /// be rank-2 (`[N, K/4]`) or rank-3 (`[E, N, K/4]`, stacked experts).
    Qtip2b {
        blocks: Tensor,
        row_scales: Tensor,
        bias: Option<Tensor>,
        in_features: usize,
        /// MCG multiplier defining the computed codebook.
        mcg_mult: u32,
        /// Hadamard incoherence rotation signs [in_features], ±1.0.
        rotation_signs: Option<Tensor>,
        /// Block size for the block-diagonal Hadamard rotation. 0 disables.
        rotation_block: usize,
    },
    /// TD-MoE Tucker decomposition kept in factored form (paper §3.3 storage
    /// path). Stores `G ×₁ U₁ ×₂ U₂ ×₃ U₃` for MoE expert stacks of shape
    /// `[K, d_out, d_in]`. See `td_moe_factored::TuckerFactoredLayer`.
    TuckerFactored {
        /// Core `[r1, r2, r3]`.
        g_core: Tensor,
        /// Expert factor `[num_experts, r1]`.
        u1: Tensor,
        /// Output factor `[d_out, r2]` (with re-coloring absorbed).
        u2: Tensor,
        /// Input factor `[d_in, r3]` (with re-coloring absorbed).
        u3: Tensor,
        /// Storage / compute dtype. Inputs are auto-cast in
        /// `forward_autocast` / `gather_forward_autocast`.
        target_dtype: DType,
    },
}

/// Device/configurable intelligent matrix multiplication
/// - Handles limitation of `accelerate` which requires f32
pub struct MatMul;

impl MatMul {
    /// Compute matrix-matrix product.
    pub fn matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "accelerate")]
        {
            let original_dtype = a.dtype();
            a.to_dtype(DType::F32)?
                .matmul(&b.to_dtype(DType::F32)?)?
                .to_dtype(original_dtype)
        }
        #[cfg(not(feature = "accelerate"))]
        {
            if a.device().is_cpu() {
                let original_dtype = a.dtype();
                a.to_dtype(DType::F16)?
                    .matmul(&b.to_dtype(DType::F16)?)?
                    .to_dtype(original_dtype)
            } else {
                a.matmul(b)
            }
        }
    }

    /// Compute matrix-matrix product.
    /// The result will be divided by the `scale` parameter in an affine division.
    pub fn matmul_affine_div(&self, a: &Tensor, b: &Tensor, scale: f64) -> Result<Tensor> {
        // TODO(EricLBuehler): Optimize this by using the gemm parameter?
        self.matmul(a, b)? / scale
    }

    /// Compute matrix-matrix product.
    /// The result will be divided by the `scale` parameter in an affine multiplication.
    pub fn matmul_affine_mul(&self, a: &Tensor, b: &Tensor, scale: f64) -> Result<Tensor> {
        // TODO(EricLBuehler): Optimize this by using the gemm parameter?
        self.matmul(a, b)? * scale
    }

    /// Compute quantized matrix-matrix product.
    pub fn qmatmul(&self, x: &Tensor, matmul: &QMatMul) -> Result<Tensor> {
        matmul.forward(x)
    }

    /// Compute quantized matrix-matrix product.
    pub fn qmethod_matmul(&self, x: &Tensor, matmul: &dyn QuantMethod) -> Result<Tensor> {
        matmul.forward(x)
    }
}

/// Device/configurable intelligent convolution
/// - Handles limitation of cpu which requires f32
pub struct Convolution;

impl Convolution {
    pub fn forward_1d(&self, layer: &Conv1d, x: &Tensor) -> Result<Tensor> {
        if x.device().is_cpu() {
            let original_dtype = x.dtype();
            Conv1d::new(
                layer.weight().to_dtype(DType::F32)?,
                layer.bias().map(|b| b.to_dtype(DType::F32)).transpose()?,
                *layer.config(),
            )
            .forward(&x.to_dtype(DType::F32)?)?
            .to_dtype(original_dtype)
        } else {
            layer.forward(x)
        }
    }

    pub fn forward_2d(&self, layer: &Conv2d, x: &Tensor) -> Result<Tensor> {
        if x.device().is_cpu() {
            let original_dtype = x.dtype();
            Conv2d::new(
                layer.weight().to_dtype(DType::F32)?,
                layer.bias().map(|b| b.to_dtype(DType::F32)).transpose()?,
                *layer.config(),
            )
            .forward(&x.to_dtype(DType::F32)?)?
            .to_dtype(original_dtype)
        } else {
            layer.forward(x)
        }
    }
}

/// In-situ quantization type specifying the format to apply to model weights.
#[derive(Clone, Copy, Debug, PartialEq, Hash, Eq, Serialize, Deserialize)]
pub enum IsqType {
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
    HQQ8,
    HQQ4,
    // HQQ3,
    // HQQ2,
    // HQQ1,
    F8E4M3,
    AFQ8,
    AFQ6,
    AFQ4,
    AFQ3,
    AFQ2,
    F8Q8,
    MXFP4,
    NVFP4,
    QtipBitshift2,
    /// QTIP 2-bit with a **computed** codebook (`qtip2b`): 3-ALU-op MCG
    /// decode in registers instead of the 512 KB LUT gather. Same 2 bits/wt
    /// rate as `QtipBitshift2`; K=2/V=1 trellis. See `qtip::bitshift`.
    Qtip2b,
}

/// Target bit width for automatic ISQ quantization.
///
/// On Metal, these select AFQ variants; on CUDA/CPU, they select Q*K variants.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum IsqBits {
    /// 2-bit quantization (AFQ2 on Metal, Q2K otherwise).
    Two,
    /// 3-bit quantization (AFQ3 on Metal, Q3K otherwise).
    Three,
    /// 4-bit quantization (AFQ4 on Metal, Q4K otherwise).
    Four,
    /// 5-bit quantization (Q5K on all platforms).
    Five,
    /// 6-bit quantization (AFQ6 on Metal, Q6K otherwise).
    Six,
    /// 8-bit quantization (AFQ8 on Metal, Q8_0 otherwise).
    Eight,
}

impl IsqBits {
    /// Resolve to the platform-appropriate `IsqType` for the given device.
    pub fn resolve(self, device: &Device) -> IsqType {
        match (self, device.is_metal()) {
            (Self::Two, true) => IsqType::AFQ2,
            (Self::Two, false) => IsqType::Q2K,
            (Self::Three, true) => IsqType::AFQ3,
            (Self::Three, false) => IsqType::Q3K,
            (Self::Four, true) => IsqType::AFQ4,
            (Self::Four, false) => IsqType::Q4K,
            (Self::Five, _) => IsqType::Q5K,
            (Self::Six, true) => IsqType::AFQ6,
            (Self::Six, false) => IsqType::Q6K,
            (Self::Eight, true) => IsqType::AFQ8,
            (Self::Eight, false) => IsqType::Q8_0,
        }
    }

    /// Return all platform variants (non-Metal first, then Metal if different).
    pub fn expand(self) -> Vec<IsqType> {
        match self {
            Self::Two => vec![IsqType::Q2K, IsqType::AFQ2],
            Self::Three => vec![IsqType::Q3K, IsqType::AFQ3],
            Self::Four => vec![IsqType::Q4K, IsqType::AFQ4],
            Self::Five => vec![IsqType::Q5K],
            Self::Six => vec![IsqType::Q6K, IsqType::AFQ6],
            Self::Eight => vec![IsqType::Q8_0, IsqType::AFQ8],
        }
    }
}

/// Where the quantize math for an ISQ rung actually executes.
///
/// This is *not* where the resulting weights live — it is where the
/// quantizer's inner loop runs, and it is the only input that decides the ISQ
/// thread policy. A GPU-backed quantize must stay near-serial on the host: N
/// rayon workers each submitting kernels to ONE device just contend for it
/// (and on CUDA `QuantizeOntoGuard` is a no-op, so nothing else serializes
/// them). A genuinely CPU-side quantize wants every core.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum IsqQuantizeBackend {
    /// Quantize kernels run on an accelerator.
    Gpu,
    /// Quantize kernels run on host CPU cores.
    Cpu,
}

impl IsqQuantizeBackend {
    /// The backend a quantize onto `device` will use.
    ///
    /// Only CUDA has GPU quantize kernels for the rungs whose policy depends
    /// on this (QTIP); a Metal or CPU device takes the host path.
    pub fn for_device(device: &Device) -> Self {
        if cfg!(feature = "cuda") && device.is_cuda() {
            Self::Gpu
        } else {
            Self::Cpu
        }
    }

    /// The backend to assume when the call site does not know the device.
    /// A CUDA-enabled build is overwhelmingly loading onto CUDA.
    pub fn assumed() -> Self {
        if cfg!(feature = "cuda") {
            Self::Gpu
        } else {
            Self::Cpu
        }
    }
}

impl TryFrom<&str> for IsqBits {
    type Error = ();
    fn try_from(s: &str) -> std::result::Result<Self, ()> {
        match s {
            "2" => Ok(Self::Two),
            "3" => Ok(Self::Three),
            "4" => Ok(Self::Four),
            "5" => Ok(Self::Five),
            "6" => Ok(Self::Six),
            "8" => Ok(Self::Eight),
            _ => Err(()),
        }
    }
}

impl std::fmt::Display for IsqType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Q4_0 => write!(f, "q4_0"),
            Self::Q4_1 => write!(f, "q4_1"),
            Self::Q5_0 => write!(f, "q5_0"),
            Self::Q5_1 => write!(f, "q5_1"),
            Self::Q8_0 => write!(f, "q8_0"),
            Self::Q8_1 => write!(f, "q8_1"),
            Self::Q2K => write!(f, "q2k"),
            Self::Q3K => write!(f, "q3k"),
            Self::Q4K => write!(f, "q4k"),
            Self::Q5K => write!(f, "q5k"),
            Self::Q6K => write!(f, "q6k"),
            Self::Q8K => write!(f, "q8k"),
            Self::HQQ8 => write!(f, "hqq8"),
            Self::HQQ4 => write!(f, "hqq4"),
            Self::F8E4M3 => write!(f, "fp8"),
            Self::AFQ8 => write!(f, "afq8"),
            Self::AFQ6 => write!(f, "afq6"),
            Self::AFQ4 => write!(f, "afq4"),
            Self::AFQ3 => write!(f, "afq3"),
            Self::AFQ2 => write!(f, "afq2"),
            Self::F8Q8 => write!(f, "f8q8"),
            Self::MXFP4 => write!(f, "mxfp4"),
            Self::NVFP4 => write!(f, "nvfp4"),
            Self::QtipBitshift2 => write!(f, "qtip2"),
            Self::Qtip2b => write!(f, "qtip2b"),
        }
    }
}

impl IsqType {
    /// Factor by which the weight size is reduced over the given dtype.
    /// original size / pack factor = quantized size
    pub fn pack_factor(&self, dtype: DType) -> usize {
        match self {
            Self::Q4_0 | Self::AFQ4 => (dtype.size_in_bytes() * GgmlDType::Q4_0.block_size())
                .div_ceil(GgmlDType::Q4_0.type_size()),
            Self::Q4_1 => (dtype.size_in_bytes() * GgmlDType::Q4_1.block_size())
                .div_ceil(GgmlDType::Q4_1.type_size()),
            Self::Q5_0 => (dtype.size_in_bytes() * GgmlDType::Q5_0.block_size())
                .div_ceil(GgmlDType::Q5_0.type_size()),
            Self::Q5_1 => (dtype.size_in_bytes() * GgmlDType::Q5_1.block_size())
                .div_ceil(GgmlDType::Q5_1.type_size()),
            Self::Q8_0 | Self::AFQ8 => (dtype.size_in_bytes() * GgmlDType::Q8_0.block_size())
                .div_ceil(GgmlDType::Q8_0.type_size()),
            Self::Q8_1 => (dtype.size_in_bytes() * GgmlDType::Q8_1.block_size())
                .div_ceil(GgmlDType::Q8_1.type_size()),
            Self::Q2K | Self::AFQ2 => (dtype.size_in_bytes() * GgmlDType::Q2K.block_size())
                .div_ceil(GgmlDType::Q2K.type_size()),
            Self::Q3K | Self::AFQ3 => (dtype.size_in_bytes() * GgmlDType::Q3K.block_size())
                .div_ceil(GgmlDType::Q3K.type_size()),
            Self::Q4K => (dtype.size_in_bytes() * GgmlDType::Q4K.block_size())
                .div_ceil(GgmlDType::Q4K.type_size()),
            Self::Q5K => (dtype.size_in_bytes() * GgmlDType::Q5K.block_size())
                .div_ceil(GgmlDType::Q5K.type_size()),
            Self::Q6K | Self::AFQ6 => (dtype.size_in_bytes() * GgmlDType::Q6K.block_size())
                .div_ceil(GgmlDType::Q6K.type_size()),
            Self::Q8K => (dtype.size_in_bytes() * GgmlDType::Q8K.block_size())
                .div_ceil(GgmlDType::Q8K.type_size()),
            // F8Q8: 33 bytes per 32 values -> similar to Q8_0
            Self::F8Q8 => (dtype.size_in_bytes() * 32).div_ceil(33),
            // Estimates
            Self::HQQ4 => 4,
            Self::HQQ8 => 2,
            Self::F8E4M3 => 2,
            // MXFP4: 4 bits per value + 1 byte scale per 32 values
            // For BF16 (2 bytes): (2*32)/(16+1) ≈ 3.76 → 3
            Self::MXFP4 => 3,
            // NVFP4: 4 bits per value + 1 byte scale per 16 values + tiny per-tensor scale
            // (2 * 16 bytes) / (8 + 1 bytes) ≈ 3.56 → 3
            Self::NVFP4 => 3,
            // QTIP 2-bit: 2 bits per weight + tiny per-row fp32 scale + shared LUT.
            // True factor vs an N-bit dtype = (8*size_in_bytes)/2 = size_in_bytes*4
            // (e.g. BF16 -> 8). The old Q2K proxy (~2.6 bits/wt) returned 7, a ~14%
            // overcount that spuriously offloads V4 Flash layers to CPU at its
            // 95%-of-80GB fit. The per-row scale + shared LUT overhead is
            // negligible vs the 2-bit blocks. (RUN-161)
            Self::QtipBitshift2 => dtype.size_in_bytes() * 4,
            // qtip2b: same 2 bits/weight; per-row fp32 scale is negligible and
            // there is no LUT at all (computed codebook).
            Self::Qtip2b => dtype.size_in_bytes() * 4,
        }
    }

    /// Max ISQ worker threads for this rung, assuming the compiled-in backend.
    /// `None` means "use every rayon worker".
    pub fn get_max_isq_cpu_threads(&self) -> Option<NonZeroUsize> {
        self.get_max_isq_cpu_threads_on(IsqQuantizeBackend::assumed())
    }

    /// Max ISQ worker threads for this rung when the quantize runs on `backend`.
    /// `None` means "use every rayon worker".
    pub fn get_max_isq_cpu_threads_on(&self, backend: IsqQuantizeBackend) -> Option<NonZeroUsize> {
        self.isq_cpu_thread_policy(backend).0
    }

    /// Max ISQ worker threads plus the one-line rationale for the choice, so
    /// bake logs record *why* a thread count was picked.
    pub fn isq_cpu_thread_policy(
        &self,
        backend: IsqQuantizeBackend,
    ) -> (Option<NonZeroUsize>, &'static str) {
        self.isq_cpu_thread_policy_with_qtip_mode(backend, crate::QtipMode::default_expert_mode())
    }

    /// Thread policy with the QTIP bake mode supplied explicitly rather than
    /// read from the environment. Tests use this to pin every rung/mode
    /// combination without racing on process-global env state.
    pub fn isq_cpu_thread_policy_with_qtip_mode(
        &self,
        backend: IsqQuantizeBackend,
        qtip_mode: crate::QtipMode,
    ) -> (Option<NonZeroUsize>, &'static str) {
        let one = || Some(NonZeroUsize::new(1).unwrap());
        match self {
            /*IsqType::HQQ1 | IsqType::HQQ2 | IsqType::HQQ3 | */
            IsqType::HQQ4
            | IsqType::HQQ8
            | IsqType::AFQ2
            | IsqType::AFQ3
            | IsqType::AFQ4
            | IsqType::AFQ6
            | IsqType::AFQ8
            | IsqType::MXFP4
            | IsqType::NVFP4 => {
                // Use 1 because our HQQ quantizes on the GPU
                (one(), "HQQ/AFQ-family quantize runs on the GPU")
            }
            IsqType::QtipBitshift2 | IsqType::Qtip2b => match backend {
                // GPU-backed bake: the trellis search is a CUDA kernel and the
                // whole bake targets ONE device, while `QuantizeOntoGuard` is a
                // no-op under the `cuda` feature. Handing rayon all cores means
                // N host threads racing to submit Viterbi work plus its
                // transfer traffic to that single device. Session 5 logged
                // "Applying immediate ISQ in parallel on 24 threads" and saw
                // 4-9 min/layer; session 3's fast bake logged 1 thread and did
                // ~30 s/layer, so 1 is the empirically-fast configuration.
                IsqQuantizeBackend::Gpu => (
                    one(),
                    "QTIP quantize runs in GPU kernels on one device; extra host threads only contend for it (session-5 bake trap)",
                ),
                // On a CPU-side bake QTIP really does burn host cores (per-row
                // trellis search). Greedy is fast enough single-threaded;
                // Viterbi is ~10x heavier and must use all cores or a full
                // requantize takes hours. The per-tensor build is sequential,
                // so the per-row par_iter is memory-safe at full width.
                // Keyed off the same decision point as the bake itself
                // (wave3-G: Viterbi is now the expert default). (RUN-161)
                IsqQuantizeBackend::Cpu => match qtip_mode {
                    crate::QtipMode::Greedy => (
                        one(),
                        "QTIP greedy trellis search on CPU is fast enough single-threaded",
                    ),
                    crate::QtipMode::Viterbi => (
                        None,
                        "QTIP Viterbi trellis search runs on host CPU; all cores",
                    ),
                },
            },
            IsqType::F8E4M3 | IsqType::F8Q8 => (None, "FP8 quantize runs on host CPU; all cores"),
            IsqType::Q2K
            | IsqType::Q3K
            | IsqType::Q4K
            | IsqType::Q4_0
            | IsqType::Q4_1
            | IsqType::Q5K
            | IsqType::Q5_0
            | IsqType::Q5_1
            | IsqType::Q6K
            | IsqType::Q8K
            | IsqType::Q8_0
            | IsqType::Q8_1 => (None, "GGML quantize runs on host CPU; all cores"),
        }
    }
}

impl TryFrom<IsqType> for GgmlDType {
    type Error = candle_core::Error;

    fn try_from(value: IsqType) -> Result<Self> {
        let tp = match value {
            IsqType::Q2K => Self::Q2K,
            IsqType::Q3K => Self::Q3K,
            IsqType::Q4K => Self::Q4K,
            IsqType::Q4_0 => Self::Q4_0,
            IsqType::Q4_1 => Self::Q4_1,
            IsqType::Q5K => Self::Q5K,
            IsqType::Q5_0 => Self::Q5_0,
            IsqType::Q5_1 => Self::Q5_1,
            IsqType::Q6K => Self::Q6K,
            IsqType::Q8K => Self::Q8K,
            IsqType::Q8_0 => Self::Q8_0,
            IsqType::Q8_1 => Self::Q8_1,
            _ => candle_core::bail!("Expected valid GGML ISQ type."),
        };
        #[cfg(feature = "cuda")]
        {
            if !matches!(
                tp,
                GgmlDType::Q4_0
                    | GgmlDType::Q4_1
                    | GgmlDType::Q5_0
                    | GgmlDType::Q5_1
                    | GgmlDType::Q8_0
                    | GgmlDType::Q2K
                    | GgmlDType::Q3K
                    | GgmlDType::Q4K
                    | GgmlDType::Q5K
                    | GgmlDType::Q6K
            ) {
                candle_core::bail!("GGML ISQ type on CUDA must be one of `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `Q2K`, `Q3K`, `Q4K`, `Q5K`, `Q6K`, `HQQ8`, `HQQ4`")
            }
        }
        Ok(tp)
    }
}

impl TryFrom<GgmlDType> for IsqType {
    type Error = candle_core::Error;

    fn try_from(value: GgmlDType) -> Result<Self> {
        match value {
            GgmlDType::Q2K => Ok(Self::Q2K),
            GgmlDType::Q3K => Ok(Self::Q3K),
            GgmlDType::Q4K => Ok(Self::Q4K),
            GgmlDType::Q5K => Ok(Self::Q5K),
            GgmlDType::Q6K => Ok(Self::Q6K),
            GgmlDType::Q4_0 => Ok(Self::Q4_0),
            GgmlDType::Q4_1 => Ok(Self::Q4_1),
            GgmlDType::Q5_0 => Ok(Self::Q5_0),
            GgmlDType::Q5_1 => Ok(Self::Q5_1),
            GgmlDType::Q8_0 => Ok(Self::Q8_0),
            GgmlDType::Q8_1 => Ok(Self::Q8_1),
            GgmlDType::Q8K => Ok(Self::Q8K),
            GgmlDType::BF16 | GgmlDType::F32 | GgmlDType::F16 => {
                candle_core::bail!("Expected valid GGML ISQ type.")
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum QuantizedSerdeType {
    Gguf = 0,
    Unquant = 1,
    Hqq = 2,
    Fp8 = 3,
    Afq = 4,
    F8Q8 = 5,
    Mxfp4 = 6,
    Nvfp4 = 7,
    Qtip = 8,
    TdMoeTucker = 9,
    /// QTIP computed-codebook bitshift-trellis rung (`qtip2b`).
    Qtip2b = 10,
}

impl TryFrom<usize> for QuantizedSerdeType {
    type Error = candle_core::Error;
    fn try_from(value: usize) -> std::result::Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Gguf),
            1 => Ok(Self::Unquant),
            2 => Ok(Self::Hqq),
            3 => Ok(Self::Fp8),
            4 => Ok(Self::Afq),
            5 => Ok(Self::F8Q8),
            6 => Ok(Self::Mxfp4),
            7 => Ok(Self::Nvfp4),
            8 => Ok(Self::Qtip),
            9 => Ok(Self::TdMoeTucker),
            10 => Ok(Self::Qtip2b),
            other => candle_core::bail!("QuantizedSerdeType {other} is invalid."),
        }
    }
}

pub trait QuantizedSerde {
    fn name(&self) -> &'static str;
    fn isq_serde_supported(&self) -> bool {
        false
    }
    fn serialize(&self) -> Result<Cow<'_, [u8]>> {
        candle_core::bail!("`QuantizedSerde::serialize` is not supported.")
    }
    fn deserialize(
        _data: Cow<[u8]>,
        _device: &Device,
        _comm: &Arc<crate::Comm>,
        _guard: QuantizeOntoGuard,
    ) -> Result<Arc<dyn QuantMethod>>
    where
        Self: Sized,
    {
        candle_core::bail!("`QuantizedSerde::deserialize` is not supported.")
    }
    fn deserialize_ext_bias(
        _data: Cow<[u8]>,
        _device: &Device,
        _guard: QuantizeOntoGuard,
    ) -> Result<(Arc<dyn QuantMethod>, Option<Tensor>)>
    where
        Self: Sized,
    {
        candle_core::bail!("`QuantizedSerde::deserialize_ext_bias` is not supported.")
    }
    /// NOT meant for external calling
    fn serialize_with_bias(&self, _bias: Option<Tensor>) -> Result<Cow<'_, [u8]>> {
        candle_core::bail!("`QuantizedSerde::serialize_with_bias` is not supported.")
    }
}

/// Used to gate access to quantizing onto the host device
#[derive(Clone, Debug)]
#[allow(unused)]
pub struct QuantizeOntoGuard {
    pub inner: Arc<Mutex<()>>,
}

/// Real (for Metal) and Fake (for CUDA)
pub enum QuantizeOntoDropGuard<'a> {
    Real(MutexGuard<'a, ()>),
    Fake,
}

impl Default for QuantizeOntoGuard {
    fn default() -> Self {
        Self::new()
    }
}

impl QuantizeOntoGuard {
    pub fn new() -> Self {
        QuantizeOntoGuard {
            inner: Arc::new(Mutex::new(())),
        }
    }

    /// Acquire the quantize drop guard to protect the critical section.
    ///
    /// On metal, this waits for outstanding work to finish to avoid "A command encoder is already encoding to this command buffer"
    pub fn acquire(&self, device: &Device) -> QuantizeOntoDropGuard<'_> {
        #[cfg(feature = "cuda")]
        {
            let _ = device;
            QuantizeOntoDropGuard::Fake
        }

        #[cfg(not(feature = "cuda"))]
        {
            #[cfg(feature = "metal")]
            if let Device::Metal(dev) = device {
                // This is necessary to avoid the errors of "A command encoder is already encoding to this command buffer"
                dev.wait_until_completed()
                    .expect("Failed to flush command buffer.");
            }
            #[cfg(not(feature = "metal"))]
            let _ = device;

            QuantizeOntoDropGuard::Real(self.inner.lock().expect("QuantizeOntoGuard was poisoned!"))
        }
    }
}

pub enum DistributedKind {
    ColumnParallel,
    RowParallel,
    Replicated,
}

/// Quantized method for a quantized matmul.
pub trait QuantMethod: Send + Sync + Debug + QuantizedSerde {
    fn new(method: QuantMethodConfig) -> Result<Self>
    where
        Self: Sized;

    fn dequantize_w(&self) -> Result<Tensor>;

    /// Compute matmul of `self` and `a`. `self` should contain the weights.
    /// Automatically cast to required quantization activation type and back
    fn forward_autocast(&self, a: &Tensor) -> Result<Tensor> {
        let original_ty = a.dtype();
        let a = if let Some(t) = self.quantized_act_type() {
            a.to_dtype(t)?
        } else {
            a.clone()
        };
        self.forward(&a)?.to_dtype(original_ty)
    }

    /// Compute matmul of `self` and `a`. `self` should contain the weights.
    fn forward(&self, a: &Tensor) -> Result<Tensor>;

    /// Compute matmul of `self` and `a`. `self` should contain the weights.
    /// Automatically cast to required quantization activation type and back.
    ///
    /// If `a` is (n_tokens, n_experts, cols), `self` weights are (n_experts, rows, cols),
    /// then the indices are (n_tokens, n_experts).
    fn gather_forward_autocast(&self, a: &Tensor, indices: &Tensor) -> Result<Tensor> {
        let original_ty = a.dtype();
        let a = if let Some(t) = self.quantized_act_type() {
            a.to_dtype(t)?
        } else {
            a.clone()
        };
        self.gather_forward(&a, indices)?.to_dtype(original_ty)
    }

    /// Compute matmul of `self` and `a`. `self` should contain the weights.
    ///
    /// If `a` is (n_tokens, n_experts, cols), `self` weights are (n_experts, rows, cols),
    /// then the indices are (n_tokens, n_experts).
    fn gather_forward(&self, _a: &Tensor, _indices: &Tensor) -> Result<Tensor> {
        candle_core::bail!(
            "{} does not support `gather_forward`. Please raise an issue.",
            self.name()
        )
    }

    /// Keep only the experts named by `ids` (ascending global indices) out of
    /// an expert-stacked weight, returning a layer that holds `ids.len()`
    /// experts indexed `0..ids.len()`.
    ///
    /// This is the **expert-parallel slice**. It exists as a post-load
    /// operation because a UQFF artifact holds every expert: the shard cannot
    /// be applied while constructing the layer, only after
    /// `load_from_artifacts` has deserialized it.
    ///
    /// The default refuses, loudly and by name. Silently returning `self`
    /// would leave every rank holding the whole expert set while believing it
    /// had been sharded — EP would appear to work and buy nothing.
    fn select_experts(&self, _ids: &[usize]) -> Result<Arc<dyn QuantMethod>> {
        candle_core::bail!(
            "{} does not support the expert-parallel slice (`select_experts`). \
             Run with ep_size = 1, or add the slice for this quantization.",
            self.name()
        )
    }

    /// If a quantized method, return the activation dtype.
    fn quantized_act_type(&self) -> Option<DType>;

    /// Weight dtype and device
    fn dtype_and_device(&self) -> (DType, Device);

    /// Add a delta weight from LoRA to the weights. This should be prescaled with alpha.
    fn add_delta_w(&self, delta: &Tensor) -> Result<Arc<dyn QuantMethod>>;

    /// If the quant is backed by a qmatmul.
    fn apply_isq(
        self: Arc<Self>,
        dtype: Option<IsqType>,
        device: Device,
        n_quantized: &AtomicUsize,
        imatrix_weight: Option<Vec<f32>>,
        guard: QuantizeOntoGuard,
    ) -> Result<Arc<dyn QuantMethod>>;

    fn unquant_weight_bias(&self) -> Option<(Tensor, Option<Tensor>)> {
        None
    }

    /// Borrowed view of QTIP packed trellis weights, for custom decode paths
    /// (e.g. `arc-cuda-graph`) that read the raw 2-bit bytes instead of
    /// dequantizing to a dense BF16 weight. Returns `None` for non-QTIP methods.
    fn qtip_packed(&self) -> Option<crate::qtip::QtipPackedView<'_>> {
        None
    }

    /// Begin tracking stats into an ImatrixLayerStats
    fn begin_track_stats(&mut self) -> Result<()> {
        candle_core::bail!("`{}` does not support tracking stats.", self.name())
    }

    /// End tracking stats into an ImatrixLayerStats. Returns the computed imatrix.
    fn end_track_stats(&self) -> Result<Tensor> {
        candle_core::bail!("`{}` does not support tracking stats.", self.name())
    }

    /// Arm a richer calibration accumulator on this layer for a forward-only
    /// sweep (see [`crate::calibration`]). Independent of the imatrix path:
    /// this collects raw `diag(XᵀX)` plus optional gram blocks and per-expert
    /// statistics, with explicit token counts.
    ///
    /// The default declines; callers must treat that as "layer emits no
    /// statistics", not as a fatal error, because layers already quantized on
    /// disk (FP8, GPTQ, …) can never track activations.
    fn begin_calibration(&mut self, _opts: &crate::calibration::CalibOptions) -> Result<()> {
        candle_core::bail!("`{}` does not support calibration stats.", self.name())
    }

    /// Finish the calibration sweep for this layer and yield its statistics.
    fn end_calibration(&self) -> Result<crate::calibration::CalibLayerData> {
        candle_core::bail!("`{}` does not support calibration stats.", self.name())
    }

    fn is_distributed(&self) -> Option<DistributedKind> {
        None
    }
}

impl Module for dyn QuantMethod {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        Self::forward(self, xs)
    }
}

pub fn linear_no_bias(
    in_dim: usize,
    out_dim: usize,
    config: &Option<QuantizedConfig>,
    vb: ShardedVarBuilder,
) -> Result<Arc<dyn QuantMethod>> {
    let base_vb = vb.clone();
    let vb = if should_apply_immediate_isq(&vb) {
        vb.set_device(Device::Cpu)
    } else {
        vb
    };

    let layer = if let Some(quant_conf) = &config {
        match quant_conf {
            QuantizedConfig::GptqAwq { .. } => gptq_linear(in_dim, out_dim, quant_conf, vb)?,
            QuantizedConfig::Fp8 { weight_block_size } => {
                if weight_block_size.is_some() {
                    blockwise_fp8_linear_b(
                        in_dim,
                        out_dim,
                        quant_conf,
                        false,
                        Default::default(),
                        vb,
                    )?
                } else {
                    pertensor_fp8_linear_b(
                        in_dim,
                        out_dim,
                        quant_conf,
                        false,
                        Default::default(),
                        vb,
                    )?
                }
            }
            QuantizedConfig::Bitsandbytes { .. } => {
                Arc::new(BnbLinear::linear_b(in_dim, out_dim, false, vb)?) as Arc<_>
            }
            QuantizedConfig::Afq { .. } => {
                AfqLayer::afq_linear_b(in_dim, out_dim, quant_conf, false, vb)?
            }
            QuantizedConfig::MXFP4 {} => {
                MXFP4Layer::linear_b(in_dim, out_dim, quant_conf, false, vb)?
            }
            QuantizedConfig::NVFP4 {} => {
                NVFP4Layer::linear_b(in_dim, out_dim, quant_conf, false, vb)?
            }
            QuantizedConfig::Qtip {} => {
                QtipLayer::linear_b(in_dim, out_dim, quant_conf, false, vb)?
            }
        }
    } else {
        // Handle the case where the layer is dummy (no tensors)
        if !vb.contains_tensor("weight") {
            let layer = <DummyLayer as QuantMethod>::new(QuantMethodConfig::Dummy)?;
            Arc::new(layer) as Arc<dyn QuantMethod>
        } else {
            let weight = vb.get_with_hints((out_dim, in_dim), "weight", Default::default())?;
            let weight = merge_lora_weights(&vb, weight, in_dim, out_dim, Default::default())?;

            let layer = <UnquantLinear as QuantMethod>::new(QuantMethodConfig::Unquantized(
                Linear::new(weight, None),
            ))?;
            Arc::new(layer) as Arc<dyn QuantMethod>
        }
    };
    apply_immediate_isq(layer, base_vb)
}

pub fn linear(
    in_dim: usize,
    out_dim: usize,
    config: &Option<QuantizedConfig>,
    vb: ShardedVarBuilder,
) -> Result<Arc<dyn QuantMethod>> {
    let base_vb = vb.clone();
    let vb = if should_apply_immediate_isq(&vb) {
        vb.set_device(Device::Cpu)
    } else {
        vb
    };

    let layer = if let Some(quant_conf) = &config {
        match quant_conf {
            QuantizedConfig::GptqAwq { .. } => gptq_linear(in_dim, out_dim, quant_conf, vb)?,
            QuantizedConfig::Fp8 { weight_block_size } => {
                if weight_block_size.is_some() {
                    blockwise_fp8_linear_b(
                        in_dim,
                        out_dim,
                        quant_conf,
                        true,
                        Default::default(),
                        vb,
                    )?
                } else {
                    pertensor_fp8_linear_b(
                        in_dim,
                        out_dim,
                        quant_conf,
                        true,
                        Default::default(),
                        vb,
                    )?
                }
            }
            QuantizedConfig::Bitsandbytes { .. } => {
                Arc::new(BnbLinear::linear_b(in_dim, out_dim, true, vb)?) as Arc<_>
            }
            QuantizedConfig::Afq { .. } => {
                AfqLayer::afq_linear_b(in_dim, out_dim, quant_conf, true, vb)?
            }
            QuantizedConfig::MXFP4 {} => {
                MXFP4Layer::linear_b(in_dim, out_dim, quant_conf, true, vb)?
            }
            QuantizedConfig::NVFP4 {} => {
                NVFP4Layer::linear_b(in_dim, out_dim, quant_conf, true, vb)?
            }
            QuantizedConfig::Qtip {} => QtipLayer::linear_b(in_dim, out_dim, quant_conf, true, vb)?,
        }
    } else {
        // Handle the case where the layer is dummy (no tensors)
        if !(vb.contains_tensor("weight") && vb.contains_tensor("bias")) {
            let layer = <DummyLayer as QuantMethod>::new(QuantMethodConfig::Dummy)?;
            Arc::new(layer) as Arc<dyn QuantMethod>
        } else {
            let weight = vb.get_with_hints((out_dim, in_dim), "weight", Default::default())?;
            let weight = merge_lora_weights(&vb, weight, in_dim, out_dim, Default::default())?;
            let bias = vb.get_with_hints((out_dim,), "bias", Default::default())?;

            let layer = <UnquantLinear as QuantMethod>::new(QuantMethodConfig::Unquantized(
                Linear::new(weight, Some(bias)),
            ))?;
            Arc::new(layer) as Arc<dyn QuantMethod>
        }
    };
    apply_immediate_isq(layer, base_vb)
}

pub fn linear_b(
    in_dim: usize,
    out_dim: usize,
    bias: bool,
    config: &Option<QuantizedConfig>,
    vb: ShardedVarBuilder,
) -> Result<Arc<dyn QuantMethod>> {
    if bias {
        linear(in_dim, out_dim, config, vb)
    } else {
        linear_no_bias(in_dim, out_dim, config, vb)
    }
}

/// Pins the ISQ thread policy per rung / backend / QTIP mode.
///
/// The regression this guards: PR #20 made Viterbi the QTIP expert default,
/// which flipped `get_max_isq_cpu_threads()` to `None` (= all cores) for the
/// QTIP rungs. On a GPU bake that put N host threads on ONE device — session 5
/// logged "Applying immediate ISQ in parallel on 24 threads" and 4-9 min/layer
/// versus session 3's 1 thread and ~30 s/layer.
#[cfg(test)]
mod isq_thread_policy_tests {
    use super::*;

    fn threads(ty: IsqType, backend: IsqQuantizeBackend, mode: QtipMode) -> Option<usize> {
        ty.isq_cpu_thread_policy_with_qtip_mode(backend, mode)
            .0
            .map(usize::from)
    }

    #[test]
    fn qtip_rungs_are_single_threaded_on_gpu_in_both_modes() {
        for ty in [IsqType::QtipBitshift2, IsqType::Qtip2b] {
            for mode in [QtipMode::Greedy, QtipMode::Viterbi] {
                assert_eq!(
                    threads(ty, IsqQuantizeBackend::Gpu, mode),
                    Some(1),
                    "{ty:?} on GPU with {mode:?} must cap at 1 thread"
                );
            }
        }
    }

    #[test]
    fn qtip_rungs_use_all_cores_only_for_cpu_viterbi() {
        for ty in [IsqType::QtipBitshift2, IsqType::Qtip2b] {
            assert_eq!(
                threads(ty, IsqQuantizeBackend::Cpu, QtipMode::Viterbi),
                None,
                "{ty:?} CPU Viterbi should use all cores"
            );
            assert_eq!(
                threads(ty, IsqQuantizeBackend::Cpu, QtipMode::Greedy),
                Some(1),
                "{ty:?} CPU greedy is fast enough single-threaded"
            );
        }
    }

    #[test]
    fn non_qtip_rungs_are_backend_independent() {
        // GPU-quantizing families stay at 1; CPU-quantizing families stay at all-cores,
        // regardless of which backend the QTIP decision would have picked.
        for backend in [IsqQuantizeBackend::Gpu, IsqQuantizeBackend::Cpu] {
            for ty in [
                IsqType::HQQ4,
                IsqType::HQQ8,
                IsqType::AFQ2,
                IsqType::AFQ3,
                IsqType::AFQ4,
                IsqType::AFQ6,
                IsqType::AFQ8,
                IsqType::MXFP4,
                IsqType::NVFP4,
            ] {
                assert_eq!(threads(ty, backend, QtipMode::Viterbi), Some(1), "{ty:?}");
            }
            for ty in [
                IsqType::Q2K,
                IsqType::Q3K,
                IsqType::Q4K,
                IsqType::Q4_0,
                IsqType::Q4_1,
                IsqType::Q5K,
                IsqType::Q5_0,
                IsqType::Q5_1,
                IsqType::Q6K,
                IsqType::Q8K,
                IsqType::Q8_0,
                IsqType::Q8_1,
                IsqType::F8E4M3,
                IsqType::F8Q8,
            ] {
                assert_eq!(threads(ty, backend, QtipMode::Viterbi), None, "{ty:?}");
            }
        }
    }

    #[test]
    fn every_rung_carries_a_rationale() {
        for backend in [IsqQuantizeBackend::Gpu, IsqQuantizeBackend::Cpu] {
            for mode in [QtipMode::Greedy, QtipMode::Viterbi] {
                for ty in [
                    IsqType::QtipBitshift2,
                    IsqType::Qtip2b,
                    IsqType::HQQ4,
                    IsqType::Q4K,
                    IsqType::F8E4M3,
                ] {
                    let (_, rationale) = ty.isq_cpu_thread_policy_with_qtip_mode(backend, mode);
                    assert!(!rationale.is_empty(), "{ty:?}/{backend:?}/{mode:?}");
                }
            }
        }
    }

    #[test]
    fn singlethread_override_wins_over_every_rung() {
        for ty in [
            Some(IsqType::QtipBitshift2),
            Some(IsqType::Qtip2b),
            Some(IsqType::Q4K),
            Some(IsqType::F8E4M3),
            None,
        ] {
            for backend in [IsqQuantizeBackend::Gpu, IsqQuantizeBackend::Cpu] {
                let (n, rationale) = isq_thread_policy_on(ty, backend, true);
                assert_eq!(n, 1, "{ty:?}/{backend:?}");
                assert!(rationale.contains("MISTRALRS_ISQ_SINGLETHREAD"));
            }
        }
    }

    #[test]
    fn policy_resolves_all_cores_to_a_concrete_count() {
        let (n, _) = isq_thread_policy_on(Some(IsqType::Q4K), IsqQuantizeBackend::Cpu, false);
        assert_eq!(n, rayon::current_num_threads());
        let (n, _) =
            isq_thread_policy_on(Some(IsqType::QtipBitshift2), IsqQuantizeBackend::Gpu, false);
        assert_eq!(n, 1);
        let (n, _) = isq_thread_policy_on(None, IsqQuantizeBackend::Gpu, false);
        assert_eq!(n, rayon::current_num_threads());
    }

    #[test]
    fn a_multi_device_bake_gets_one_submitter_per_device() {
        // PR #25's cap is per DEVICE. Session 5's trap was 24 host threads on
        // ONE device (4-9 min/layer); two threads on two devices is not that
        // trap, and refusing to widen here would leave every device but one
        // idle for the whole bake.
        for ty in [IsqType::QtipBitshift2, IsqType::Qtip2b] {
            let (one, _) =
                isq_thread_policy_for_devices_on(Some(ty), IsqQuantizeBackend::Gpu, false, 1);
            assert_eq!(one, 1, "{ty:?} on a single device stays at 1");
            for n in [2usize, 4, 8] {
                let (width, rationale) =
                    isq_thread_policy_for_devices_on(Some(ty), IsqQuantizeBackend::Gpu, false, n);
                assert_eq!(width, n, "{ty:?} across {n} devices");
                assert!(rationale.contains("per bake device"), "{rationale}");
            }
        }
        // HQQ/AFQ quantize on the device too, and get the same treatment.
        let (width, _) = isq_thread_policy_for_devices_on(
            Some(IsqType::AFQ4),
            IsqQuantizeBackend::Gpu,
            false,
            4,
        );
        assert_eq!(width, 4);
    }

    #[test]
    fn host_quantized_rungs_do_not_widen_with_more_devices() {
        // GGML and FP8 quantize on host cores; a second GPU adds no compute, so
        // multiplying their width would only oversubscribe the CPU.
        for ty in [IsqType::Q4K, IsqType::Q8_0, IsqType::F8E4M3] {
            for backend in [IsqQuantizeBackend::Gpu, IsqQuantizeBackend::Cpu] {
                let (width, _) = isq_thread_policy_for_devices_on(Some(ty), backend, false, 4);
                assert_eq!(
                    width,
                    rayon::current_num_threads(),
                    "{ty:?}/{backend:?} must keep its host width"
                );
            }
        }
    }

    #[test]
    fn one_device_is_exactly_the_single_device_policy() {
        // The default path must not shift for existing users: with one device
        // the new entry point has to agree with the old one everywhere.
        for ty in [
            Some(IsqType::QtipBitshift2),
            Some(IsqType::Qtip2b),
            Some(IsqType::Q4K),
            Some(IsqType::AFQ4),
            Some(IsqType::F8E4M3),
            None,
        ] {
            for backend in [IsqQuantizeBackend::Gpu, IsqQuantizeBackend::Cpu] {
                for forced in [false, true] {
                    assert_eq!(
                        isq_thread_policy_for_devices_on(ty, backend, forced, 1),
                        isq_thread_policy_on(ty, backend, forced),
                        "{ty:?}/{backend:?}/forced={forced}"
                    );
                    // A zero-length list is not a way to get zero threads.
                    assert_eq!(
                        isq_thread_policy_for_devices_on(ty, backend, forced, 0),
                        isq_thread_policy_on(ty, backend, forced),
                    );
                }
            }
        }
    }

    #[test]
    fn singlethread_flag_means_one_thread_per_device() {
        // MISTRALRS_ISQ_SINGLETHREAD exists to stop several host threads piling
        // onto ONE device; across N devices its faithful reading is one each.
        let (width, _) = isq_thread_policy_for_devices_on(
            Some(IsqType::QtipBitshift2),
            IsqQuantizeBackend::Gpu,
            /* forced */ true,
            4,
        );
        assert_eq!(width, 4);
    }

    #[test]
    fn unpack_width_is_not_the_isq_width() {
        // The whole point of the split: layer-level GPU submission stays at 1
        // for QTIP-on-GPU (session-5's 4-9 min/layer trap stays closed) while
        // the host-side expert unpack takes the whole machine (session-6's
        // 240 s/layer at 100.5% CPU on 24 cores).
        for ty in [IsqType::QtipBitshift2, IsqType::Qtip2b] {
            let (isq_threads, _) = isq_thread_policy_on(Some(ty), IsqQuantizeBackend::Gpu, false);
            assert_eq!(
                isq_threads, 1,
                "{ty:?} GPU submission must stay at 1 thread"
            );
        }
        assert_eq!(expert_unpack_threads_from(None, 24), 24);
        // MISTRALRS_ISQ_SINGLETHREAD (the bake exports it) must keep forcing
        // layer-level single-threading and must not reach across and
        // re-serialize the unpack, which reads its own override only.
        for backend in [IsqQuantizeBackend::Gpu, IsqQuantizeBackend::Cpu] {
            let (n, rationale) = isq_thread_policy_on(
                Some(IsqType::QtipBitshift2),
                backend,
                /* forced */ true,
            );
            assert_eq!(n, 1);
            assert!(rationale.contains("MISTRALRS_ISQ_SINGLETHREAD"));
        }
        assert_eq!(expert_unpack_threads_from(None, 24), 24);
        assert_eq!(expert_unpack_threads_from(Some("6"), 24), 6);
    }

    #[test]
    fn unpack_width_defaults_and_override() {
        assert_eq!(expert_unpack_threads_from(None, 1), 1);
        assert_eq!(expert_unpack_threads_from(None, 0), 1, "never zero threads");
        assert_eq!(expert_unpack_threads_from(Some(" 8 "), 24), 8);
        // A malformed or zero override must not silently serialize a bake.
        assert_eq!(expert_unpack_threads_from(Some("oops"), 24), 24);
        assert_eq!(expert_unpack_threads_from(Some("0"), 24), 24);
        assert_eq!(expert_unpack_threads_from(Some(""), 24), 24);
    }

    /// The failure this guards: a bare `into_par_iter()` inherits the pool of
    /// whatever thread calls it, so an expert unpack reached from inside the
    /// 1-thread ISQ pool would run serially with no warning. `install` on the
    /// dedicated pool must restore full width even from inside a 1-thread pool.
    #[test]
    fn unpack_pool_is_concurrent_even_nested_in_a_single_thread_pool() {
        use rayon::prelude::*;
        use std::sync::atomic::{AtomicUsize, Ordering};

        const WORKERS: usize = 4;
        let isq_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        let unpack_pool = build_expert_unpack_pool(WORKERS);

        let in_flight = AtomicUsize::new(0);
        let peak = AtomicUsize::new(0);

        isq_pool.install(|| {
            assert_eq!(
                rayon::current_num_threads(),
                1,
                "precondition: the ISQ pool is single-threaded"
            );
            unpack_pool.install(|| {
                assert_eq!(rayon::current_num_threads(), WORKERS);
                (0..WORKERS).into_par_iter().for_each(|_| {
                    let now = in_flight.fetch_add(1, Ordering::SeqCst) + 1;
                    peak.fetch_max(now, Ordering::SeqCst);
                    // Spin (bounded) until every worker is in flight, so the
                    // assertion below cannot pass on a serial run and cannot
                    // hang if the pool is narrower than expected.
                    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
                    while in_flight.load(Ordering::SeqCst) < WORKERS
                        && std::time::Instant::now() < deadline
                    {
                        std::thread::yield_now();
                    }
                    in_flight.fetch_sub(1, Ordering::SeqCst);
                });
            });
        });

        assert_eq!(
            peak.load(Ordering::SeqCst),
            WORKERS,
            "expert unpack ran with fewer than {WORKERS} concurrent workers: the \
             dedicated pool did not survive being nested in the 1-thread ISQ pool"
        );
    }

    #[test]
    fn cpu_device_never_resolves_to_the_gpu_backend() {
        assert_eq!(
            IsqQuantizeBackend::for_device(&Device::Cpu),
            IsqQuantizeBackend::Cpu
        );
    }
}
