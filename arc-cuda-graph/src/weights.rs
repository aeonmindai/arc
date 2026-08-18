//! Parent system: ArcInfer / ArcGraph.
//!
//! Weight pointer extraction from loaded Candle models, for the **dense**
//! decode path.
//!
//! Uses IsqModel::get_layers() for projection weights and
//! IsqModel::residual_tensors() for norms and embeddings.
//!
//! # This is NOT model-agnostic, despite what it used to claim
//!
//! [`DecodeConfig`] describes exactly one family: a **dense, Llama-shaped**
//! decoder — seven projections per layer (`q, k, v, o, gate, up, down`), one
//! MLP, one KV head group, a fused QKV buffer, plain RoPE. `DedicatedDecodePath`
//! and `decode_forward` consume that description literally.
//!
//! Handed a model that is not that shape, the old code did not notice. It
//! indexed `layers[1 + i*7 .. ]` positionally and read whatever happened to sit
//! there. On DeepSeek-V4 — MLA (`wq_a, wq_b, wkv, wo_a, wo_b`), a 256-expert
//! MoE, mHC 4-D residual — index `1 + i*7 + 4` is not a gate projection and the
//! resulting pointer set describes a model that does not exist.
//!
//! On V4 this never fired, but only because extraction happened to fail first
//! on an unrelated error (a non-CUDA residual tensor, `tensor_device_ptr`'s
//! `_` arm). That is luck, not a contract, and luck is not a safety mechanism:
//! the day that unrelated failure is fixed, V4 decode would silently route into
//! a dense-transformer kernel stack and return garbage with no error at all.
//!
//! [`check_dense_layer_inventory`] converts the luck into a contract. It runs
//! before any pointer is taken and refuses, by name and by per-layer kind, any
//! model whose layer inventory is not the dense shape. On acceptance it returns
//! the [`DenseLayout`] it walked, and `extract_model_weights` indexes that —
//! the validated layout and the performed reads are one object. See its docs
//! for the rule.

#[cfg(feature = "cuda")]
use candle_core::{Storage, Tensor};

/// Projections per layer the dense decode path requires:
/// `q, k, v, o, gate, up, down`.
///
/// This is not a tunable. `extract_model_weights` indexes
/// `layers[1 + i * DENSE_PROJS_PER_LAYER + k]` positionally and
/// `LayerWeights` has exactly these seven slots.
pub const DENSE_PROJS_PER_LAYER: usize = 7;

/// The per-layer *kind* the inventory reports, and the only kind the dense
/// decode path can consume.
///
/// The old check asked one question with one global constant — "did every
/// layer contribute exactly seven?" — which is a count, not a kind. Counting is
/// enough to *refuse*, but it cannot say what a heterogeneous model is made of,
/// and DeepSeek-V4 is heterogeneous by construction: `deepseek4.rs:4931-4942`
/// emits three projections for a `MoeOrMlp::Mlp` layer and
/// `moe.get_isq_layers()` — shared experts plus every routed expert's
/// gate/up/down — for a `MoeOrMlp::Moe` layer, in the same model.
///
/// [`LayerCensus`] therefore classifies every layer independently and reports
/// the distribution, so a refusal can say "0 of 62 layers are dense" instead of
/// naming layer 0 and stopping.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerCensus {
    /// Layers that contributed exactly [`DENSE_PROJS_PER_LAYER`] entries.
    pub dense: usize,
    /// Layers that contributed anything else — MoE experts, MLA projections,
    /// a fused QKV, a two-matrix MLP. All undescribable by [`DecodeConfig`].
    pub non_dense: usize,
}

/// Why a model's layer inventory is not the dense shape the decode path needs.
///
/// Every variant names the concrete counts, because the whole point is that the
/// operator can tell *which* assumption broke without attaching a debugger.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DenseShapeError {
    /// `DecodeConfig::num_layers` was zero — nothing to describe.
    NoLayers,
    /// `get_layers()` did not lead with the untagged (`None`) lm_head.
    HeadNotFirst { first: Option<usize> },
    /// An untagged entry sits at a position `extract_model_weights` reads as a
    /// projection — anywhere in `1 ..= num_layers * DENSE_PROJS_PER_LAYER`.
    ///
    /// This is the only untagged entry that can hurt. An untagged tensor
    /// *inside* the dense block displaces every projection after it, so the
    /// positional read returns a tensor of the wrong role. Untagged tensors
    /// appended *after* the block (V4's `mtp.h_proj`, `mtp.e_proj` and the MTP
    /// block) are never dereferenced and are therefore not this guard's
    /// business — see [`check_dense_layer_inventory`].
    UntaggedInsideDenseBlock { position: usize, last_read: usize },
    /// A layer index at or beyond `num_layers`.
    LayerIndexOutOfRange { layer: usize, num_layers: usize },
    /// A layer did not contribute exactly [`DENSE_PROJS_PER_LAYER`] entries.
    /// This is the variant a MoE model trips: 256 experts x 3 projections plus
    /// the attention projections is not 7.
    ///
    /// `census` carries the whole model's distribution, not just this layer's,
    /// so the message can distinguish "one odd layer" from "this is a MoE".
    LayerProjCountMismatch {
        layer: usize,
        found: usize,
        expected: usize,
        census: LayerCensus,
        num_layers: usize,
    },
    /// Position `position` should hold a projection of layer `expected` and
    /// holds a projection of layer `found` instead (`None` = the inventory ends
    /// before that position).
    ///
    /// Counts alone cannot catch this. An inventory of `[lm_head, 1×7, 0×7]`
    /// gives both layers a count of seven, so the histogram rule accepts it —
    /// and then `extract_model_weights` reads positions `1..=7` as layer 0 and
    /// builds a pointer set in which every layer's weights belong to a
    /// different layer. No bound is violated, nothing faults, and decode
    /// returns fluent garbage. Only walking the blocks *in order* sees it.
    LayerOutOfOrder {
        position: usize,
        expected: usize,
        found: Option<usize>,
    },
}

impl std::fmt::Display for DenseShapeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "model is not the dense shape the decode path requires: ")?;
        match self {
            Self::NoLayers => write!(f, "num_layers is 0"),
            Self::HeadNotFirst { first } => write!(
                f,
                "get_layers() must lead with the untagged lm_head, but entry 0 is tagged {first:?}"
            ),
            Self::UntaggedInsideDenseBlock {
                position,
                last_read,
            } => write!(
                f,
                "an untagged entry sits at position {position}, inside the block the extractor reads \
                 positionally (1..={last_read}) — it would be dereferenced as a projection and it \
                 displaces every projection after it"
            ),
            Self::LayerIndexOutOfRange { layer, num_layers } => write!(
                f,
                "layer index {layer} is >= num_layers ({num_layers})"
            ),
            Self::LayerProjCountMismatch {
                layer,
                found,
                expected,
                census,
                num_layers,
            } => write!(
                f,
                "layer {layer} contributed {found} quantized projections, expected exactly {expected} \
                 (q, k, v, o, gate, up, down) — a MoE or MLA layer is not describable by DecodeConfig. \
                 Across the whole model {}/{num_layers} layers are dense and {}/{num_layers} are not, \
                 so this is the architecture, not one odd layer",
                census.dense, census.non_dense
            ),
            Self::LayerOutOfOrder {
                position,
                expected,
                found,
            } => match found {
                Some(found) => write!(
                    f,
                    "position {position} holds a projection of layer {found} where the extractor \
                     reads layer {expected} — the per-layer blocks are out of order or interleaved. \
                     Every layer contributed the right *count*, so a histogram accepts this and the \
                     extractor then assigns every layer another layer's weights, silently"
                ),
                None => write!(
                    f,
                    "the inventory ends before position {position}, which the extractor reads as a \
                     projection of layer {expected}"
                ),
            },
        }
    }
}

impl std::error::Error for DenseShapeError {}

/// Where each layer's projections actually sit in the inventory.
///
/// This is the check's *output*, not a by-product: `extract_model_weights` used
/// to recompute `1 + i * DENSE_PROJS_PER_LAYER + k` for itself, which meant the
/// guard validated one indexing scheme and the extractor performed another one
/// that merely happened to agree. Returning the offsets the guard walked closes
/// that gap — the extractor can only read what was validated.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DenseLayout {
    /// `starts[i]` is the index in `tags` of layer `i`'s first projection.
    starts: Vec<usize>,
}

impl DenseLayout {
    /// Number of layers described.
    pub fn num_layers(&self) -> usize {
        self.starts.len()
    }

    /// Index of layer `layer`'s first projection (`q_proj`).
    ///
    /// # Panics
    /// If `layer >= num_layers()`. Callers iterate `0..num_layers()`.
    pub fn layer_start(&self, layer: usize) -> usize {
        self.starts[layer]
    }

    /// Index of layer `layer`'s `k`-th projection, in
    /// `q, k, v, o, gate, up, down` order.
    ///
    /// # Panics
    /// If `layer >= num_layers()` or `k >= DENSE_PROJS_PER_LAYER`. Both are
    /// loop bounds at the only call site, so a panic here is a code defect, not
    /// a bad model — a bad model is refused before this type exists.
    pub fn proj(&self, layer: usize, k: usize) -> usize {
        assert!(
            k < DENSE_PROJS_PER_LAYER,
            "projection {k} is out of range for a dense layer ({DENSE_PROJS_PER_LAYER} slots)"
        );
        self.starts[layer] + k
    }
}

/// Refuse any model whose layer inventory is not the dense
/// `q, k, v, o, gate, up, down` shape `DecodeConfig` describes.
///
/// `tags` is the `Option<usize>` layer tag of each entry `IsqModel::get_layers`
/// returned, in order. The rule, all three parts required:
///
/// 1. entry 0 is untagged (`None`) — the lm_head, which `extract_model_weights`
///    reads as `layers[0]`;
/// 2. no *other* untagged entry appears at a position the extractor reads,
///    i.e. within `1 ..= num_layers * DENSE_PROJS_PER_LAYER`;
/// 3. every layer index `0..num_layers` appears exactly
///    [`DENSE_PROJS_PER_LAYER`] times, and no index outside that range appears;
/// 4. those entries are **contiguous and in ascending layer order**, starting at
///    position 1.
///
/// Parts 3 and 4 together are what protects the positional indexing. Part 3
/// alone does not: an inventory of `[lm_head, 1×7, 0×7]` satisfies it and is
/// still mis-read, because the extractor takes positions `1..=7` to be layer 0.
/// Part 4 is the half that was missing, and it is not hypothetical — it is the
/// same failure mode as the count mismatch (a pointer set describing a model
/// that does not exist) with none of the symptoms, since no bound is violated.
///
/// On success the walk's offset table is returned as a [`DenseLayout`] and the
/// extractor indexes *that* rather than recomputing the stride, so the
/// validated layout and the performed reads cannot drift apart.
///
/// # Per-layer kind, not one global constant
///
/// The rule is stated per layer and the refusal reports the whole model's
/// distribution ([`LayerCensus`]). That matters for the models this actually
/// meets: DeepSeek-V4 emits a *different* number of entries for a
/// `MoeOrMlp::Mlp` layer (3) than for a `MoeOrMlp::Moe` layer
/// (`get_isq_layers()`: shared experts + 3 per routed expert) in the same
/// model (`deepseek4.rs:4931-4942`), so "the count for layer 0" is not the
/// model's shape and must not be reported as though it were.
///
/// # Part 2 is scoped to what is actually read
///
/// It used to demand that lm_head be the *only* untagged entry anywhere in the
/// inventory. That is stricter than the extractor needs and it refused models
/// the extractor would have described correctly.
///
/// `extract_model_weights` dereferences exactly `layers[0]` and
/// `layers[1 + i * DENSE_PROJS_PER_LAYER + k]` for `i < num_layers` and
/// `k < DENSE_PROJS_PER_LAYER`. The largest index it can reach is therefore
/// `num_layers * DENSE_PROJS_PER_LAYER`. Entries past that are never touched,
/// so an auxiliary head *appended after* the dense block — which is exactly
/// where `deepseek4.rs:4580-4582` puts `mtp.h_proj`, `mtp.e_proj` and the MTP
/// block — cannot corrupt anything. An untagged entry *inside* the block is a
/// different matter and is still refused: it is read as a projection, and it
/// displaces every projection after it.
///
/// This does not admit V4. V4 is refused by part 3 — its layers emit MLA
/// projections plus 256 experts x 3, not seven — and that is the truthful
/// reason. The old rule reported the MTP head instead, because it happened to
/// be checked first, which named a consequence rather than the cause.
///
/// Nor *should* it admit V4, and no amount of work on this function will: the
/// refusal is downstream of a structural fact. [`LayerWeights`] has seven
/// non-optional projection slots and `decode_forward` computes a fused-QKV
/// attention followed by a `gate/up/silu/down` MLP. There is no MLA slot and no
/// expert slot to put V4's weights in, so accepting V4 here would only move the
/// failure from a named refusal to a wrong-tensor read. V4 needs a *layer kind*
/// in `DecodeConfig`/`decode_forward`, not a looser guard.
///
/// Pure, and deliberately not CUDA-gated, so the contract is unit-testable on
/// any host — including the macOS dev machines where `cargo check` does not
/// type-check CUDA-gated code at all.
///
/// # Honest limits — what this does NOT check
///
/// It sees only the layer *tags*, so it verifies counts and grouping, not
/// **order**. A model contributing exactly seven projections per layer in some
/// order other than `q, k, v, o, gate, up, down` passes here and is still
/// mis-assigned downstream. Nothing available at this boundary distinguishes
/// them; catching that needs per-projection identity, which `get_layers()` does
/// not carry.
///
/// The check is therefore **necessary, not sufficient** — it is a cheap refusal
/// of the architectures that are obviously not describable (MoE, MLA, auxiliary
/// heads, non-Llama MLPs), not a proof that a passing model is correct. It is
/// still a strict improvement: every model it refuses was already being
/// silently mis-indexed by the positional reads below, because seven-per-layer
/// is exactly what those reads assume.
pub fn check_dense_layer_inventory(
    tags: &[Option<usize>],
    num_layers: usize,
) -> Result<DenseLayout, DenseShapeError> {
    if num_layers == 0 {
        return Err(DenseShapeError::NoLayers);
    }

    match tags.first() {
        Some(None) => {}
        Some(&first) => return Err(DenseShapeError::HeadNotFirst { first }),
        None => return Err(DenseShapeError::HeadNotFirst { first: None }),
    }

    // The highest index `extract_model_weights` can reach:
    // `1 + (num_layers - 1) * 7 + 6`. Positions past it are never dereferenced.
    let last_read = num_layers * DENSE_PROJS_PER_LAYER;
    if let Some(position) = tags
        .iter()
        .enumerate()
        .skip(1)
        .take(last_read)
        .find_map(|(i, t)| t.is_none().then_some(i))
    {
        return Err(DenseShapeError::UntaggedInsideDenseBlock {
            position,
            last_read,
        });
    }

    // Counted over the WHOLE inventory, not just the read window. This is what
    // catches a layer that emitted more than seven entries: the excess spills
    // past `last_read`, and a window-scoped count would see a truncated seven
    // and wave a 256-expert MoE layer through.
    let mut per_layer = vec![0usize; num_layers];
    for &tag in tags {
        let Some(idx) = tag else { continue };
        if idx >= num_layers {
            return Err(DenseShapeError::LayerIndexOutOfRange {
                layer: idx,
                num_layers,
            });
        }
        per_layer[idx] += 1;
    }

    // The per-layer KIND, computed for every layer before anything is reported.
    // A refusal that names only the first offending layer cannot distinguish a
    // MoE architecture from one malformed layer in an otherwise dense model,
    // and those want different responses from whoever reads the log.
    let non_dense = per_layer
        .iter()
        .filter(|&&found| found != DENSE_PROJS_PER_LAYER)
        .count();
    let census = LayerCensus {
        dense: num_layers - non_dense,
        non_dense,
    };

    // Walk the block the extractor reads, layer by layer, in order — this is
    // what turns a histogram into an offset table, and what catches blocks that
    // have the right counts in the wrong places.
    let mut starts = Vec::with_capacity(num_layers);
    let mut position = 1usize;
    for (layer, &found) in per_layer.iter().enumerate() {
        if found != DENSE_PROJS_PER_LAYER {
            return Err(DenseShapeError::LayerProjCountMismatch {
                layer,
                found,
                expected: DENSE_PROJS_PER_LAYER,
                census,
                num_layers,
            });
        }
        starts.push(position);
        for k in 0..DENSE_PROJS_PER_LAYER {
            let at = position + k;
            match tags.get(at).copied() {
                // The only accepting arm.
                Some(Some(idx)) if idx == layer => {}
                Some(Some(idx)) => {
                    return Err(DenseShapeError::LayerOutOfOrder {
                        position: at,
                        expected: layer,
                        found: Some(idx),
                    })
                }
                // Unreachable while every count is exactly seven (the untagged
                // sweep above already covered `1..=last_read`), but the walk
                // must be total: an inventory it cannot index is a refusal, not
                // a panic.
                Some(None) => {
                    return Err(DenseShapeError::UntaggedInsideDenseBlock {
                        position: at,
                        last_read,
                    })
                }
                None => {
                    return Err(DenseShapeError::LayerOutOfOrder {
                        position: at,
                        expected: layer,
                        found: None,
                    })
                }
            }
        }
        position += DENSE_PROJS_PER_LAYER;
    }

    Ok(DenseLayout { starts })
}

#[cfg(feature = "cuda")]
#[derive(Clone, Copy, Debug)]
pub struct WeightPtr {
    pub ptr: u64,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(feature = "cuda")]
#[derive(Clone, Debug)]
pub struct LayerWeights {
    pub input_layernorm: u64,
    pub post_attn_layernorm: u64,
    pub q_proj: WeightPtr,
    pub k_proj: WeightPtr,
    pub v_proj: WeightPtr,
    pub o_proj: WeightPtr,
    pub q_norm: Option<u64>,
    pub k_norm: Option<u64>,
    pub gate_proj: WeightPtr,
    pub up_proj: WeightPtr,
    pub down_proj: WeightPtr,
    // Fused QKV: contiguous [q_rows + k_rows + v_rows, hidden] — set by DedicatedDecodePath
    pub qkv_fused: u64,
    pub qkv_rows: usize, // q_rows + k_rows + v_rows
}

/// Per-layer Tensor anchors that own the device storage backing the `u64`
/// pointers in `LayerWeights`. Held inside `ModelWeights::anchors`.
///
/// `qkv` covers q_proj/k_proj/v_proj — these can be dropped after
/// `DedicatedDecodePath::fuse_qkv` since the fused buffer owns its own
/// memory. The remaining four are read at every decode step.
#[cfg(feature = "cuda")]
#[derive(Clone, Debug, Default)]
pub struct LayerAnchors {
    pub qkv: Vec<candle_core::Tensor>, // 3 entries: q, k, v (clearable after fuse)
    pub o: Option<candle_core::Tensor>,
    pub gate: Option<candle_core::Tensor>,
    pub up: Option<candle_core::Tensor>,
    pub down: Option<candle_core::Tensor>,
}

#[cfg(feature = "cuda")]
impl LayerAnchors {
    pub fn count(&self) -> usize {
        self.qkv.len()
            + self.o.is_some() as usize
            + self.gate.is_some() as usize
            + self.up.is_some() as usize
            + self.down.is_some() as usize
    }
}

/// All Tensor anchors needed to keep every `u64` device pointer in
/// `ModelWeights` valid for the lifetime of the struct.
#[cfg(feature = "cuda")]
#[derive(Clone, Debug, Default)]
pub struct WeightAnchors {
    pub lm_head: Option<candle_core::Tensor>,
    pub residuals: Vec<candle_core::Tensor>, // norms, embeds
    pub layers: Vec<LayerAnchors>,
}

#[cfg(feature = "cuda")]
impl WeightAnchors {
    pub fn count(&self) -> usize {
        self.lm_head.is_some() as usize
            + self.residuals.len()
            + self.layers.iter().map(|l| l.count()).sum::<usize>()
    }
}

#[cfg(feature = "cuda")]
#[derive(Clone, Debug)]
pub struct ModelWeights {
    pub embed_tokens: u64,
    pub final_norm: u64,
    pub lm_head: WeightPtr,
    pub layers: Vec<LayerWeights>,
    pub config: DecodeConfig,
    /// Owners of the device buffers that back every `u64` pointer above.
    ///
    /// `QuantMethod::dequantize_w()` returns a fresh `Tensor` whose `CudaSlice`
    /// is freed as soon as the last Tensor clone drops. If we kept only the
    /// raw `u64` device pointer, the next CUDA op (e.g. `fuse_qkv` reading
    /// `q_proj.ptr`) would dereference a freed device address — manifesting
    /// as a SEGV inside `libcuda`'s SSE-aligned memcpy fast path (the page
    /// table walk for the source pointer faults). Holding the Tensors here
    /// keeps the Arc<Storage> alive for the lifetime of `ModelWeights`, so
    /// every device pointer above remains valid until this struct drops.
    ///
    /// The breakdown by role lets `DedicatedDecodePath` drop just the
    /// per-layer Q/K/V anchors after fusion (saving ~3 BF16 projections per
    /// layer for ISQ models) while keeping O/gate/up/down/lm_head/residuals
    /// alive for the decode loop.
    pub anchors: WeightAnchors,
}

#[cfg(feature = "cuda")]
#[derive(Clone, Debug)]
pub struct DecodeConfig {
    pub num_layers: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub has_qk_norm: bool,
    pub max_position_embeddings: usize,
    pub is_gpt_neox: bool,
}

/// Whether [`tensor_device_ptr`] can produce a device pointer for `dt`.
///
/// This is the single source of truth for the dispatch table below and is
/// deliberately *not* CUDA-gated so the coverage test runs on any host.
///
/// The table is exactly candle's `cuda_dtype!` instantiations
/// (`candle-core/src/cuda_backend/mod.rs`): `u8, u32, i16, i32, i64, f16,
/// bf16, f32, f64, float8::F8E4M3`. The MX dummy formats (`F6E2M3`,
/// `F6E3M2`, `F4`, `F8E8M0`) have no `CudaDType` impl, so `as_cuda_slice`
/// can never yield them — and candle reports `size_in_bytes() == 0` for the
/// sub-byte ones, which would make the `start_offset() * size_in_bytes()`
/// byte-offset arithmetic below silently wrong for a view.
///
/// History: `I32` was missing here, which took out **every** consumer that
/// passes an `int32_t` buffer to a kernel — the GPU radix top-k sampler
/// (`seq_lens`), the fused `CudaSampler` (`token_ids`, `keep_idx_scratch`)
/// and the V4 indexer's `score_and_topk_*`. The sampler's failure surfaced
/// only as a per-token `WARN … falling back to CPU`, so it ran the slow path
/// on every decode step in production. Widen this table, never the callers.
pub const fn device_ptr_supports_dtype(dt: candle_core::DType) -> bool {
    use candle_core::DType;
    matches!(
        dt,
        DType::U8
            | DType::U32
            | DType::I16
            | DType::I32
            | DType::I64
            | DType::BF16
            | DType::F16
            | DType::F32
            | DType::F64
            | DType::F8E4M3
    )
}

/// Extract raw u64 device pointer from a Candle tensor (any dtype).
///
/// The dtype dispatch below must stay in lockstep with
/// [`device_ptr_supports_dtype`] — `device_ptr_covers_every_storable_dtype`
/// in this module's tests enforces it.
#[cfg(feature = "cuda")]
pub fn tensor_device_ptr(tensor: &Tensor) -> candle_core::Result<u64> {
    use candle_core::cuda::cudarc::driver::DevicePtr;
    use candle_core::DType;

    let (storage, layout) = tensor.storage_and_layout();
    let offset_bytes = layout.start_offset() * tensor.dtype().size_in_bytes();

    match &*storage {
        Storage::Cuda(cuda_storage) => {
            // Match on dtype to get correctly-typed CudaSlice (cudarc requires type match)
            let base_ptr: u64 = match tensor.dtype() {
                DType::BF16 => {
                    let s = cuda_storage.as_cuda_slice::<half::bf16>()?;
                    let (p, _) = s.device_ptr(s.stream());
                    p
                }
                DType::F16 => {
                    let s = cuda_storage.as_cuda_slice::<half::f16>()?;
                    let (p, _) = s.device_ptr(s.stream());
                    p
                }
                DType::F32 => {
                    let s = cuda_storage.as_cuda_slice::<f32>()?;
                    let (p, _) = s.device_ptr(s.stream());
                    p
                }
                DType::U32 => {
                    let s = cuda_storage.as_cuda_slice::<u32>()?;
                    let (p, _) = s.device_ptr(s.stream());
                    p
                }
                // I32 is what every `int32_t*` kernel argument in this crate is
                // fed with (radix top-k `seq_lens`, `CudaSampler` token ids and
                // keep-list scratch). Candle's `Tensor::from_vec(Vec<i32>, …)`
                // produces it, so leaving it out disabled those paths wholesale.
                //
                // Concretely, from #130: `CudaSampler::sample` REQUIRES
                // `token_ids` to be I32 (`sampling_cuda.rs:347`) and allocates
                // an I32 `keep_idx_scratch` (`:298`). Both resolve through this
                // function, so before I32 was listed both fell to the catch-all
                // `bail!` and `CudaSampler::sample()` returned `unsupported
                // dtype I32` on EVERY call — the replay-safe top-k/top-p
                // sampler could never run on a GPU. Its test suite is a CPU
                // simulator (`gpu_algorithm_simulate`), which by construction
                // cannot observe that.
                DType::I32 => {
                    let s = cuda_storage.as_cuda_slice::<i32>()?;
                    let (p, _) = s.device_ptr(s.stream());
                    p
                }
                DType::I16 => {
                    let s = cuda_storage.as_cuda_slice::<i16>()?;
                    let (p, _) = s.device_ptr(s.stream());
                    p
                }
                DType::I64 => {
                    let s = cuda_storage.as_cuda_slice::<i64>()?;
                    let (p, _) = s.device_ptr(s.stream());
                    p
                }
                DType::F64 => {
                    let s = cuda_storage.as_cuda_slice::<f64>()?;
                    let (p, _) = s.device_ptr(s.stream());
                    p
                }
                DType::U8 => {
                    let s = cuda_storage.as_cuda_slice::<u8>()?;
                    let (p, _) = s.device_ptr(s.stream());
                    p
                }
                // NOT `as_cuda_slice::<u8>()`: candle stores this as
                // `CudaStorageSlice::F8E4M3(CudaSlice<float8::F8E4M3>)`, and
                // `as_cuda_slice::<T>` matches the storage variant by `T`, so
                // asking for `u8` here returned `UnexpectedDType { expected:
                // U8, got: F8E4M3 }` for every FP8 tensor — same class of
                // defect as the missing I32 arm, on the FP8 KV-cache path.
                DType::F8E4M3 => {
                    let s = cuda_storage.as_cuda_slice::<float8::F8E4M3>()?;
                    let (p, _) = s.device_ptr(s.stream());
                    p
                }
                dt => candle_core::bail!(
                    "tensor_device_ptr: unsupported dtype {dt:?} (sub-byte dtypes have \
                     size_in_bytes()==0 so the offset arithmetic is undefined; every other \
                     dtype belongs in this table — see device_ptr_supports_dtype)"
                ),
            };
            Ok(base_ptr + offset_bytes as u64)
        }
        _ => candle_core::bail!("tensor_device_ptr requires CUDA tensor"),
    }
}

/// Extract a WeightPtr from a QuantMethod (uses dequantize_w for BF16 weights).
///
/// Returns both the `WeightPtr` and the backing `Tensor`. The caller MUST
/// keep the Tensor alive for as long as the device pointer in `WeightPtr` is
/// used — otherwise `Drop` on the Tensor frees the CUDA storage and the
/// pointer becomes dangling. For unquantized methods this is a cheap clone
/// of the model's own weight; for ISQ-quantized methods this owns the freshly
/// dequantized BF16/F16 buffer.
#[cfg(feature = "cuda")]
pub fn quant_method_ptr(
    qm: &dyn mistralrs_quant::QuantMethod,
) -> candle_core::Result<(WeightPtr, candle_core::Tensor)> {
    // Try unquant first (zero-copy), fall back to dequantize
    let tensor = if let Some((w, _)) = qm.unquant_weight_bias() {
        w
    } else {
        qm.dequantize_w()?
    };
    let dims = tensor.dims();
    let (rows, cols) = if dims.len() == 2 {
        (dims[0], dims[1])
    } else {
        (dims.iter().product::<usize>(), 1)
    };
    let ptr = tensor_device_ptr(&tensor)?;
    Ok((WeightPtr { ptr, rows, cols }, tensor))
}

/// Build ModelWeights from IsqModel trait methods.
///
/// `get_layers()` returns projection weights in a fixed order per layer:
///   [lm_head, q_0, k_0, v_0, o_0, gate_0, up_0, down_0, q_1, k_1, ...]
///
/// `residual_tensors()` returns named tensors:
///   model.embed_tokens.weight, model.norm.weight,
///   model.layers.N.input_layernorm.weight, model.layers.N.post_attention_layernorm.weight,
///   model.layers.N.self_attn.q_norm.weight (optional), etc.
#[cfg(feature = "cuda")]
pub fn extract_model_weights(
    layers: &[(
        &std::sync::Arc<dyn mistralrs_quant::QuantMethod>,
        Option<usize>,
    )],
    residuals: &[(String, candle_core::Tensor)],
    config: DecodeConfig,
) -> candle_core::Result<ModelWeights> {
    let num_layers = config.num_layers;

    // ARCHITECTURE GUARD — must run before any pointer is taken.
    //
    // Everything below indexes `layers` positionally on the assumption of one
    // untagged lm_head followed by exactly seven projections per layer. On a
    // model that is not that shape those indices address unrelated tensors and
    // the result is a pointer set for a model that does not exist — which the
    // decode kernels then execute without complaint. Refuse by name here rather
    // than produce silent garbage downstream. See the module docs.
    //
    // The guard hands back the offset table it walked. Indexing *that* rather
    // than recomputing `1 + i * 7 + k` is the point: the layout that was
    // validated and the reads that are performed are now the same object, so
    // they cannot drift apart in a later edit.
    let tags: Vec<Option<usize>> = layers.iter().map(|(_, idx)| *idx).collect();
    let layout = match check_dense_layer_inventory(&tags, num_layers) {
        Ok(layout) => layout,
        Err(e) => candle_core::bail!(
            "dedicated decode path does not support this architecture: {e}. \
             The dense decode path (DecodeConfig/decode_forward) is only valid for \
             Llama-shaped models; it cannot describe MLA or MoE layers. This is a \
             refusal, not a failure — decode continues on the standard Candle path."
        ),
    };

    let mut anchors = WeightAnchors::default();
    anchors.residuals.reserve(residuals.len());
    anchors.layers = Vec::with_capacity(num_layers);

    // First element is lm_head (layer_idx = None)
    let (lm_head, lm_head_t) = quant_method_ptr(&**layers[0].0)?;
    anchors.lm_head = Some(lm_head_t);

    // Build layer weights from projections
    let mut layer_weights = Vec::with_capacity(num_layers);
    for i in 0..num_layers {
        let (q_proj, q_t) = quant_method_ptr(&**layers[layout.proj(i, 0)].0)?;
        let (k_proj, k_t) = quant_method_ptr(&**layers[layout.proj(i, 1)].0)?;
        let (v_proj, v_t) = quant_method_ptr(&**layers[layout.proj(i, 2)].0)?;
        let (o_proj, o_t) = quant_method_ptr(&**layers[layout.proj(i, 3)].0)?;
        let (gate_proj, gate_t) = quant_method_ptr(&**layers[layout.proj(i, 4)].0)?;
        let (up_proj, up_t) = quant_method_ptr(&**layers[layout.proj(i, 5)].0)?;
        let (down_proj, down_t) = quant_method_ptr(&**layers[layout.proj(i, 6)].0)?;
        anchors.layers.push(LayerAnchors {
            qkv: vec![q_t, k_t, v_t],
            o: Some(o_t),
            gate: Some(gate_t),
            up: Some(up_t),
            down: Some(down_t),
        });
        layer_weights.push(LayerWeights {
            input_layernorm: 0,     // filled from residuals below
            post_attn_layernorm: 0, // filled from residuals below
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm: None,
            k_norm: None,
            gate_proj,
            up_proj,
            down_proj,
            qkv_fused: 0,
            qkv_rows: 0,
        });
    }

    // Fill in residual tensors (norms, embeddings).
    // These tensors' storage is owned by the model itself, but we *also*
    // anchor a clone here so the pointer stays valid even if the model
    // were ever to be partially reloaded under us.
    let mut embed_tokens: u64 = 0;
    let mut final_norm: u64 = 0;

    for (name, tensor) in residuals {
        let ptr = tensor_device_ptr(tensor)?;
        anchors.residuals.push(tensor.clone());

        if name.ends_with("embed_tokens.weight") {
            embed_tokens = ptr;
        } else if name == "model.norm.weight"
            || name.ends_with(".norm.weight") && !name.contains("layers")
        {
            final_norm = ptr;
        } else if name.contains("input_layernorm.weight") {
            // Extract layer index: model.layers.N.input_layernorm.weight
            if let Some(idx) = extract_layer_idx(name) {
                if idx < num_layers {
                    layer_weights[idx].input_layernorm = ptr;
                }
            }
        } else if name.contains("post_attention_layernorm.weight") {
            if let Some(idx) = extract_layer_idx(name) {
                if idx < num_layers {
                    layer_weights[idx].post_attn_layernorm = ptr;
                }
            }
        } else if name.contains("q_norm.weight") {
            if let Some(idx) = extract_layer_idx(name) {
                if idx < num_layers {
                    layer_weights[idx].q_norm = Some(ptr);
                }
            }
        } else if name.contains("k_norm.weight") {
            if let Some(idx) = extract_layer_idx(name) {
                if idx < num_layers {
                    layer_weights[idx].k_norm = Some(ptr);
                }
            }
        }
    }

    Ok(ModelWeights {
        embed_tokens,
        final_norm,
        lm_head,
        layers: layer_weights,
        config,
        anchors,
    })
}

/// Extract layer index from a tensor name like "model.layers.42.input_layernorm.weight"
#[cfg(feature = "cuda")]
fn extract_layer_idx(name: &str) -> Option<usize> {
    let parts: Vec<&str> = name.split('.').collect();
    for (i, part) in parts.iter().enumerate() {
        if *part == "layers" && i + 1 < parts.len() {
            return parts[i + 1].parse().ok();
        }
    }
    None
}

/// The architecture guard's contract. No CUDA, no GPU, no model — the guard is
/// a pure function precisely so this runs in CI and on a macOS dev box, where
/// `cargo check` does not type-check CUDA-gated code at all.
#[cfg(test)]
mod dense_shape_tests {
    use super::{check_dense_layer_inventory, DenseShapeError, LayerCensus, DENSE_PROJS_PER_LAYER};

    /// Exactly what `llama.rs` emits: the untagged lm_head, then `q, k, v, o`
    /// plus the MLP's `gate, up, down` for each layer.
    fn dense_tags(num_layers: usize) -> Vec<Option<usize>> {
        let mut tags = vec![None];
        for i in 0..num_layers {
            tags.extend(std::iter::repeat_n(Some(i), DENSE_PROJS_PER_LAYER));
        }
        tags
    }

    #[test]
    fn dense_llama_inventory_is_accepted() {
        let layout = check_dense_layer_inventory(&dense_tags(32), 32)
            .expect("a Llama-shaped inventory must be accepted");
        assert_eq!(layout.num_layers(), 32);
        // The path must not require a *particular* depth, only the shape.
        assert!(check_dense_layer_inventory(&dense_tags(1), 1).is_ok());
    }

    /// The layout is the check's product, and the extractor now reads *it*
    /// rather than recomputing the stride. So assert the offsets it hands back
    /// are the positions `extract_model_weights` must dereference — otherwise
    /// the two agree only by coincidence, which is the arrangement this change
    /// exists to end.
    #[test]
    fn accepted_layout_addresses_the_projections_the_extractor_reads() {
        const NUM_LAYERS: usize = 5;
        let tags = dense_tags(NUM_LAYERS);
        let layout = check_dense_layer_inventory(&tags, NUM_LAYERS).expect("dense fixture");

        for layer in 0..NUM_LAYERS {
            assert_eq!(
                layout.layer_start(layer),
                1 + layer * DENSE_PROJS_PER_LAYER,
                "layer {layer} start"
            );
            for k in 0..DENSE_PROJS_PER_LAYER {
                let at = layout.proj(layer, k);
                assert_eq!(at, 1 + layer * DENSE_PROJS_PER_LAYER + k);
                // The decisive property: whatever index the layout produces,
                // the entry there really does belong to that layer.
                assert_eq!(tags[at], Some(layer), "layout.proj({layer}, {k}) -> {at}");
            }
        }
    }

    /// THE ORDERING HOLE. Both layers contribute exactly seven entries, so the
    /// per-layer histogram — the whole of the old rule — accepts this. The
    /// extractor then reads positions `1..=7` as layer 0 and gets layer 1's
    /// projections, and every layer in the model is assigned another layer's
    /// weights. Nothing is out of bounds, nothing faults, and decode returns
    /// fluent nonsense.
    ///
    /// Mutation check: with the ordering walk removed and the histogram left in
    /// place, this input returns `Ok` and the assertion below fails.
    #[test]
    fn blocks_in_the_wrong_order_are_rejected() {
        let mut tags = vec![None];
        tags.extend(std::iter::repeat_n(Some(1), DENSE_PROJS_PER_LAYER));
        tags.extend(std::iter::repeat_n(Some(0), DENSE_PROJS_PER_LAYER));

        // The fixture must actually defeat a histogram, or the test proves nothing.
        let mut per_layer = [0usize; 2];
        for &t in &tags {
            if let Some(i) = t {
                per_layer[i] += 1;
            }
        }
        assert_eq!(
            per_layer, [DENSE_PROJS_PER_LAYER; 2],
            "fixture must pass a per-layer count rule"
        );

        let err =
            check_dense_layer_inventory(&tags, 2).expect_err("out-of-order blocks must be refused");
        assert_eq!(
            err,
            DenseShapeError::LayerOutOfOrder {
                position: 1,
                expected: 0,
                found: Some(1),
            }
        );
        assert!(err.to_string().contains("out of order"), "{err}");
    }

    /// The same hole in its interleaved form: counts are right, placement is
    /// not. `[lm_head, 0, 1, 0, 1, ...]` gives each layer seven entries.
    #[test]
    fn interleaved_layer_blocks_are_rejected() {
        let mut tags = vec![None];
        for _ in 0..DENSE_PROJS_PER_LAYER {
            tags.push(Some(0));
            tags.push(Some(1));
        }
        let err =
            check_dense_layer_inventory(&tags, 2).expect_err("interleaved blocks must be refused");
        assert_eq!(
            err,
            DenseShapeError::LayerOutOfOrder {
                // position 1 is layer 0's q_proj and is correct; position 2
                // should be layer 0's k_proj and holds layer 1's q_proj.
                position: 2,
                expected: 0,
                found: Some(1),
            }
        );
    }

    /// The case this guard exists for. `deepseek4.rs:4541-4600` emits, per
    /// layer, `wq_a, wq_b, wkv, wo_a, wo_b` (+ optional compressor gate) and
    /// then every expert's three projections — and afterwards appends
    /// `mtp.h_proj`, `mtp.e_proj` and the whole MTP block as *untagged*
    /// entries.
    ///
    /// The refusal must name the **layer shape**, not the MTP head. The MTP
    /// entries are appended past every position the extractor reads and are
    /// harmless; what makes V4 undescribable is that a layer emits MLA
    /// projections and 3 x experts, which is not seven. Reporting the MTP head
    /// named a coincidence of check order and sent the reader after the wrong
    /// thing.
    #[test]
    fn v4_shaped_inventory_is_rejected_for_its_layer_shape() {
        let num_layers = 4;
        let experts = 8; // the real model has 256; 8 is enough to be un-dense
        let mut tags = vec![None]; // lm_head
        for i in 0..num_layers {
            // MLA: wq_a, wq_b, wkv, wo_a, wo_b
            tags.extend(std::iter::repeat_n(Some(i), 5));
            // MoE: three projections per expert
            tags.extend(std::iter::repeat_n(Some(i), experts * 3));
        }
        // MTP head + block, all untagged, all past `num_layers * 7`.
        tags.extend([None, None, None]);

        let err = check_dense_layer_inventory(&tags, num_layers)
            .expect_err("a V4-shaped inventory must be refused");
        assert_eq!(
            err,
            DenseShapeError::LayerProjCountMismatch {
                layer: 0,
                found: 5 + experts * 3,
                expected: DENSE_PROJS_PER_LAYER,
                census: LayerCensus {
                    dense: 0,
                    non_dense: num_layers,
                },
                num_layers,
            }
        );
        // The message must name the reason; an operator reads this, not the enum.
        assert!(err.to_string().contains("MoE or MLA"), "{err}");
        // ...and it must say the architecture is not dense, not merely that
        // layer 0 is odd. `0/4 layers are dense` is the difference between
        // "wrong model for this path" and "one layer needs looking at".
        assert!(err.to_string().contains("0/4 layers are dense"), "{err}");
    }

    /// V4 is **heterogeneous**, which is the case a single global constant
    /// cannot describe at all. `deepseek4.rs:4931-4942` emits three entries for
    /// a `MoeOrMlp::Mlp` layer and `moe.get_isq_layers()` for a
    /// `MoeOrMlp::Moe` layer, in the same model — so the count for layer 0 is
    /// not the model's shape.
    ///
    /// Here layer 0 is dense-shaped by accident (7 entries) and the rest are
    /// MoE. The refusal must still name the model, and the census is the only
    /// field that can: it reports 1 dense of 3, so a reader can tell this from
    /// a uniformly-MoE model without re-deriving the inventory.
    #[test]
    fn census_reports_the_whole_model_not_just_the_first_offending_layer() {
        const NUM_LAYERS: usize = 3;
        let mut tags = vec![None];
        // Layer 0: exactly seven — accepted on its own terms.
        tags.extend(std::iter::repeat_n(Some(0), DENSE_PROJS_PER_LAYER));
        // Layers 1, 2: MLA + a small MoE.
        for i in 1..NUM_LAYERS {
            tags.extend(std::iter::repeat_n(Some(i), 5 + 4 * 3));
        }

        let err = check_dense_layer_inventory(&tags, NUM_LAYERS)
            .expect_err("a heterogeneous MoE model must be refused");
        assert_eq!(
            err,
            DenseShapeError::LayerProjCountMismatch {
                // The first layer that is NOT dense — not layer 0.
                layer: 1,
                found: 17,
                expected: DENSE_PROJS_PER_LAYER,
                census: LayerCensus {
                    dense: 1,
                    non_dense: 2,
                },
                num_layers: NUM_LAYERS,
            }
        );
        assert!(err.to_string().contains("1/3 layers are dense"), "{err}");
        assert!(err.to_string().contains("2/3 are not"), "{err}");
    }

    /// The half of the old untagged rule that was load-bearing. An untagged
    /// entry *inside* the read window is dereferenced as a projection and
    /// displaces every projection after it, so it is still refused — and the
    /// refusal names the position, because that is what the reader needs.
    #[test]
    fn untagged_entry_inside_the_dense_block_is_still_rejected() {
        const NUM_LAYERS: usize = 3;
        let mut tags = dense_tags(NUM_LAYERS);
        // Splice an untagged tensor into the middle of layer 1's projections.
        tags.insert(10, None);
        let err = check_dense_layer_inventory(&tags, NUM_LAYERS)
            .expect_err("an untagged entry inside the block must be refused");
        assert_eq!(
            err,
            DenseShapeError::UntaggedInsideDenseBlock {
                position: 10,
                last_read: NUM_LAYERS * DENSE_PROJS_PER_LAYER,
            }
        );
        assert!(err.to_string().contains("displaces"), "{err}");
    }

    /// A MoE model with no auxiliary heads still has the wrong per-layer count.
    /// This is the case the untagged check alone would let through.
    #[test]
    fn moe_layer_is_rejected_on_projection_count() {
        let mut tags = vec![None];
        tags.extend(std::iter::repeat_n(Some(0), 5 + 8 * 3));
        assert_eq!(
            check_dense_layer_inventory(&tags, 1),
            Err(DenseShapeError::LayerProjCountMismatch {
                layer: 0,
                found: 29,
                expected: DENSE_PROJS_PER_LAYER,
                census: LayerCensus {
                    dense: 0,
                    non_dense: 1,
                },
                num_layers: 1,
            })
        );
    }

    /// Mixtral, from the real emission order rather than a synthetic shape.
    /// `mistralrs-core/src/models/mixtral.rs:686-698` pushes, per layer:
    /// `q, k, v, o`, then `block_sparse_moe.gate`, then `w1, w2, w3` for every
    /// expert. Mixtral-8x7B has 8 experts, so 4 + 1 + 24 = 29 — not 7.
    #[test]
    fn mixtral_8x7b_shaped_inventory_is_rejected() {
        const EXPERTS: usize = 8;
        const PER_LAYER: usize = 4 + 1 + 3 * EXPERTS; // q,k,v,o + gate + w1/w2/w3
        assert_eq!(PER_LAYER, 29, "derived from mixtral.rs:686-698");

        let mut tags = vec![None];
        tags.extend(std::iter::repeat_n(Some(0), PER_LAYER));
        tags.extend(std::iter::repeat_n(Some(1), PER_LAYER));
        assert_eq!(
            check_dense_layer_inventory(&tags, 2),
            Err(DenseShapeError::LayerProjCountMismatch {
                layer: 0,
                found: PER_LAYER,
                expected: DENSE_PROJS_PER_LAYER,
                census: LayerCensus {
                    dense: 0,
                    non_dense: 2,
                },
                num_layers: 2,
            })
        );
    }

    /// Phi-2 is the *under*-count case, and it is qualitatively worse than a
    /// mis-assignment. `mistralrs-core/src/models/phi2.rs:621-636` pushes
    /// `q, k, v, dense` plus `Mlp::get_isq_layers()` — which is `[fc1, fc2]`
    /// (`phi2.rs:125-127`) — so **6** entries per layer, not 7.
    ///
    /// With 6 emitted and 7 assumed, `extract_model_weights`'s
    /// `layers[1 + i * 7 + k]` mis-assigns from layer 1 on *and then walks off
    /// the end of the vector entirely*: for a 32-layer Phi-2 the inventory has
    /// `1 + 32*6 = 193` entries (max index 192) while the last read is
    /// `1 + 31*7 + 6 = 224`. The guard turns an index-out-of-bounds panic into
    /// a named refusal.
    #[test]
    fn phi2_shaped_inventory_is_rejected_before_it_runs_off_the_end() {
        const PER_LAYER: usize = 4 + 2; // q,k,v,dense + fc1,fc2
        const NUM_LAYERS: usize = 32; // Phi-2
        let mut tags = vec![None];
        for i in 0..NUM_LAYERS {
            tags.extend(std::iter::repeat_n(Some(i), PER_LAYER));
        }

        // The arithmetic the refusal is protecting, asserted rather than asserted-about.
        let last_read = 1 + (NUM_LAYERS - 1) * DENSE_PROJS_PER_LAYER + (DENSE_PROJS_PER_LAYER - 1);
        assert!(
            last_read >= tags.len(),
            "fixture must actually overrun: last read {last_read} vs len {}",
            tags.len()
        );

        assert_eq!(
            check_dense_layer_inventory(&tags, NUM_LAYERS),
            Err(DenseShapeError::LayerProjCountMismatch {
                layer: 0,
                found: PER_LAYER,
                expected: DENSE_PROJS_PER_LAYER,
                census: LayerCensus {
                    dense: 0,
                    non_dense: NUM_LAYERS,
                },
                num_layers: NUM_LAYERS,
            })
        );
    }

    /// THE MUTATION THAT MATTERS. A guard written as a total-length check —
    /// `tags.len() == 1 + num_layers * 7`, the obvious cheap version — passes
    /// this input, because 8 + 6 == 14 == 2 * 7. The extractor would then read
    /// layer 1's projections starting one slot late and silently mis-assign
    /// every pointer from there on. Only a per-layer histogram catches it.
    #[test]
    fn correct_total_with_wrong_distribution_is_rejected() {
        let mut tags = vec![None];
        tags.extend(std::iter::repeat_n(Some(0), 8));
        tags.extend(std::iter::repeat_n(Some(1), 6));
        assert_eq!(
            tags.len(),
            1 + 2 * DENSE_PROJS_PER_LAYER,
            "fixture must fool a length check"
        );
        assert_eq!(
            check_dense_layer_inventory(&tags, 2),
            Err(DenseShapeError::LayerProjCountMismatch {
                layer: 0,
                found: 8,
                expected: DENSE_PROJS_PER_LAYER,
                census: LayerCensus {
                    dense: 0,
                    non_dense: 2,
                },
                num_layers: 2,
            })
        );
    }

    /// A dense model that also carries a trailing auxiliary head must be
    /// ACCEPTED. `extract_model_weights` reads exactly `layers[0]` and
    /// `layers[1 + i*7 + k]` for `i < num_layers, k < 7` — a maximum index of
    /// `7 * num_layers`. Entries past that are never dereferenced, so refusing
    /// a model because of them refuses a model the extractor would have
    /// described correctly.
    #[test]
    fn dense_model_with_trailing_auxiliary_head_is_accepted() {
        const NUM_LAYERS: usize = 4;
        let mut tags = dense_tags(NUM_LAYERS);
        let last_read = NUM_LAYERS * DENSE_PROJS_PER_LAYER;
        assert_eq!(
            tags.len() - 1,
            last_read,
            "fixture must be exactly the reads the extractor makes"
        );
        // MTP-style: h_proj, e_proj and a block, all untagged, all appended
        // after every position the extractor reads.
        tags.extend([None, None, None]);
        let layout = check_dense_layer_inventory(&tags, NUM_LAYERS)
            .expect("a trailing auxiliary head must not refuse a dense model");
        assert_eq!(layout.num_layers(), NUM_LAYERS);
        // Nothing the layout addresses reaches the appended head.
        assert_eq!(
            layout.proj(NUM_LAYERS - 1, DENSE_PROJS_PER_LAYER - 1),
            last_read
        );
    }

    #[test]
    fn head_must_be_the_first_entry() {
        let mut tags = vec![Some(0)];
        tags.extend(std::iter::repeat_n(Some(0), DENSE_PROJS_PER_LAYER - 1));
        tags.push(None);
        assert_eq!(
            check_dense_layer_inventory(&tags, 1),
            Err(DenseShapeError::HeadNotFirst { first: Some(0) })
        );
    }

    #[test]
    fn empty_inventory_is_rejected() {
        assert_eq!(
            check_dense_layer_inventory(&[], 1),
            Err(DenseShapeError::HeadNotFirst { first: None })
        );
    }

    #[test]
    fn layer_index_beyond_num_layers_is_rejected() {
        let mut tags = vec![None];
        tags.extend(std::iter::repeat_n(Some(0), DENSE_PROJS_PER_LAYER));
        tags.extend(std::iter::repeat_n(Some(9), DENSE_PROJS_PER_LAYER));
        assert_eq!(
            check_dense_layer_inventory(&tags, 2),
            Err(DenseShapeError::LayerIndexOutOfRange {
                layer: 9,
                num_layers: 2,
            })
        );
    }

    #[test]
    fn zero_layers_is_rejected() {
        assert_eq!(
            check_dense_layer_inventory(&dense_tags(0), 0),
            Err(DenseShapeError::NoLayers)
        );
    }

    /// A missing layer reads as a count of zero, not as a silent pass.
    #[test]
    fn absent_layer_is_rejected() {
        let mut tags = vec![None];
        tags.extend(std::iter::repeat_n(Some(0), DENSE_PROJS_PER_LAYER));
        assert_eq!(
            check_dense_layer_inventory(&tags, 2),
            Err(DenseShapeError::LayerProjCountMismatch {
                layer: 1,
                found: 0,
                expected: DENSE_PROJS_PER_LAYER,
                census: LayerCensus {
                    dense: 1,
                    non_dense: 1,
                },
                num_layers: 2,
            })
        );
    }
}

/// Regression test: prove `Tensor::clone` shares storage. This is the
/// invariant the anchor mechanism relies on — cloning a Tensor must
/// keep the underlying storage alive but must NOT allocate new device
/// memory. If candle ever changes Tensor semantics, this test catches
/// it on CPU before we ship a quadratic-memory regression to CUDA.
///
/// This test runs without the cuda feature because it uses CPU tensors.
#[cfg(test)]
mod cpu_anchor_tests {
    use candle_core::{DType, Device, Storage, Tensor};

    fn storage_addr(t: &Tensor) -> usize {
        let (storage, _) = t.storage_and_layout();
        match &*storage {
            Storage::Cpu(cpu) => {
                // CpuStorage variants hold a Vec<T>. We use the discriminant
                // address as a stand-in: clones share the same backing
                // RwLock<Storage>, so the addresses match.
                cpu as *const _ as usize
            }
            _ => panic!("test expects CPU storage"),
        }
    }

    #[test]
    fn tensor_clone_shares_storage() {
        let src = Tensor::zeros((4, 8), DType::F32, &Device::Cpu).unwrap();
        let original_addr = storage_addr(&src);
        let clone = src.clone();
        let clone_addr = storage_addr(&clone);
        assert_eq!(
            original_addr, clone_addr,
            "Tensor::clone must share storage (Arc bump) — \
             anchor mechanism relies on this. If this fails, the \
             dangling-pointer fix in DedicatedDecodePath is broken."
        );
    }

    #[test]
    fn tensor_drop_after_clone_keeps_storage_alive() {
        let original_addr: usize;
        let anchor;
        {
            let src = Tensor::zeros((4, 8), DType::F32, &Device::Cpu).unwrap();
            original_addr = storage_addr(&src);
            anchor = src.clone();
            // `src` drops at end of block — but `anchor` should keep
            // the storage alive.
        }
        assert_eq!(
            original_addr,
            storage_addr(&anchor),
            "Anchored Tensor must keep storage alive after the original drops"
        );
        // Touch the storage to prove it's still readable.
        let v = anchor.to_vec2::<f32>().expect("tensor should be readable");
        assert_eq!(v.len(), 4);
        assert_eq!(v[0].len(), 8);
    }

    /// Every dtype a Candle CUDA tensor can actually hold must be resolvable
    /// to a device pointer.
    ///
    /// `tensor_device_ptr`'s dispatch is a `match` on `DType`; a missing arm
    /// is invisible until a kernel argument happens to use that dtype at
    /// runtime, at which point the caller either hard-errors or — worse —
    /// swallows it into a fallback. That is exactly how the missing `I32` arm
    /// disabled the GPU radix top-k sampler for every decode step in
    /// production while only emitting a per-token WARN.
    ///
    /// The `_dtype_is_exhaustive` match below is the tripwire: adding a
    /// variant to candle's `DType` breaks compilation here, forcing a
    /// decision about `tensor_device_ptr` instead of shipping another gap.
    #[test]
    fn device_ptr_covers_every_storable_dtype() {
        use super::device_ptr_supports_dtype;

        // Exhaustive over candle's DType — compilation fails if upstream adds
        // a variant. `true` == candle has a `cuda_dtype!` impl for it, so
        // `as_cuda_slice` can return it and `tensor_device_ptr` must handle it.
        fn has_cuda_slice_impl(dt: DType) -> bool {
            match dt {
                DType::U8
                | DType::U32
                | DType::I16
                | DType::I32
                | DType::I64
                | DType::BF16
                | DType::F16
                | DType::F32
                | DType::F64
                | DType::F8E4M3 => true,
                // MX dummy formats: raw-byte storage, no `CudaDType` impl.
                DType::F6E2M3 | DType::F6E3M2 | DType::F4 | DType::F8E8M0 => false,
            }
        }

        const ALL: &[DType] = &[
            DType::U8,
            DType::U32,
            DType::I16,
            DType::I32,
            DType::I64,
            DType::BF16,
            DType::F16,
            DType::F32,
            DType::F64,
            DType::F8E4M3,
            DType::F6E2M3,
            DType::F6E3M2,
            DType::F4,
            DType::F8E8M0,
        ];

        for &dt in ALL {
            assert_eq!(
                device_ptr_supports_dtype(dt),
                has_cuda_slice_impl(dt),
                "tensor_device_ptr dtype coverage drifted for {dt:?}: candle \
                 {} a CudaSlice for it",
                if has_cuda_slice_impl(dt) {
                    "can produce"
                } else {
                    "cannot produce"
                }
            );
        }

        // The specific regression: I32 buffers back every `int32_t*` kernel
        // argument in this crate.
        assert!(
            device_ptr_supports_dtype(DType::I32),
            "I32 must be supported — radix top-k `seq_lens`, CudaSampler \
             `token_ids` and `keep_idx_scratch` are all I32"
        );
    }
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    #[test]
    fn extract_layer_idx_basic() {
        assert_eq!(
            super::extract_layer_idx("model.layers.42.input_layernorm.weight"),
            Some(42)
        );
        assert_eq!(
            super::extract_layer_idx("model.layers.0.post_attention_layernorm.weight"),
            Some(0)
        );
        assert_eq!(super::extract_layer_idx("model.embed_tokens.weight"), None);
        assert_eq!(super::extract_layer_idx("lm_head.weight"), None);
    }

    /// Regression test for the libcuda SEGV root cause: `quant_method_ptr`
    /// must return the backing Tensor so the caller can anchor it. Without
    /// the anchor the underlying storage drops at the end of the helper and
    /// the raw `u64` device pointer becomes a use-after-free. This test only
    /// asserts the *type contract* — the runtime behavior requires CUDA and
    /// is exercised by `DedicatedDecodePath` integration tests.
    #[test]
    fn quant_method_ptr_returns_tensor_anchor() {
        // We can't construct a real QuantMethod without CUDA. The compile-time
        // signature check is the regression guard: any future refactor that
        // drops the Tensor from the return type will fail to compile.
        fn _signature_check<F>(_: F)
        where
            F: Fn(
                &dyn mistralrs_quant::QuantMethod,
            ) -> candle_core::Result<(super::WeightPtr, candle_core::Tensor)>,
        {
        }
        _signature_check(super::quant_method_ptr);
    }

    /// Regression test for the libcuda SEGV root cause: `ModelWeights` must
    /// carry an `anchors` field whose lifetime is tied to the struct. Compile-
    /// time check: if `anchors` is ever removed from `ModelWeights`, callers
    /// will fail to build, surfacing the regression immediately.
    #[test]
    fn model_weights_owns_anchors() {
        // Build an empty `WeightAnchors` to prove the field exists and is
        // constructible from `()` (i.e. default).
        let a = super::WeightAnchors::default();
        assert_eq!(a.count(), 0);

        // `LayerAnchors::qkv` MUST be a `Vec` so `DedicatedDecodePath` can
        // `.clear()` it after fusion without touching the still-live
        // o/gate/up/down anchors. Compile-time guard:
        let empty: Vec<candle_core::Tensor> = vec![];
        fn _is_vec<T>(_: &Vec<T>) {}
        _is_vec(a.layers.first().map(|l| &l.qkv).unwrap_or(&empty));
    }
}
