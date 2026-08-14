//! Synthetic-weight load smoke tests for the three Tier-A model families that
//! Arc will rent B200 cycles to validate end-to-end: DeepSeek V4 Flash,
//! Kimi K2.5/K2.6, and GLM-4.5 / GLM-5 (DSA).
//!
//! Each test:
//!   1. Builds a tiny config that matches the real architecture (V4: hash-MoE
//!      + LoRA Q + fused wkv; K2.x: V3 MLA + sigmoid-gated MoE; GLM-5: GLM4Moe
//!      attention + sigmoid-gated MoE).
//!   2. Synthesises every tensor the loader probes for, with shapes derived
//!      from the config — NOT hard-coded.
//!   3. Calls the actual `NormalModelLoader::load(...)` and walks the full
//!      construction path. Loader failure (missing tensor, wrong shape) means
//!      we'd have learned that on a B200 rental instead of here.
//!   4. Runs a single-token forward pass and asserts the output is finite and
//!      has shape `[batch=1, seq=2, vocab]`.
//!
//! Why this matters: tensor-shape drift between the V4/K2/GLM-5 loaders and
//! their on-disk SafeTensors index is the most common B200-rental-burner.
//! These tests catch it for free at `cargo test` time. The synthetic weights
//! are degenerate (ones for norm gains, zeros elsewhere) so the test's value
//! is in the **structural** assertions, not numerical correctness.
//!
//! When a model family's loader can't be exercised because of a deliberate
//! Tier-A scaffold (e.g. GLM-5 DSA delegates to GLM4Moe but the SGLang DSA
//! attention isn't ported yet), the test still exercises the loader pathway
//! that real GLM-5 checkpoints will take — namely the GLM4MoeLoader with a
//! GLM4Moe-shaped config. The DSA-specific MLA tensors are NOT loaded; if/when
//! GLM-5 grows its own loader, this test will be the first to flag the
//! divergence.

use std::collections::HashMap;
use std::sync::Arc;

use candle_core::{DType, Device, IndexOp, Module, Result as CandleResult, Tensor};
use indicatif::MultiProgress;

use mistralrs_core::{
    AttentionImplementation, DeepSeekV3Loader, DeepSeekV4Loader, DeviceMapper, GLM4MoeLoader,
    IsqOrganization, NormalLoaderType, NormalLoadingMetadata, NormalModel, NormalModelLoader,
    TextFlashParams as FlashParams, UqffFullSer, UqffSourceWeights, UQFF_MTP_TENSOR_PREFIX,
};
use mistralrs_quant::{safetensors::ShardedSafeTensors, ShardedVarBuilder};

/// The MTP load-depth / UQFF-bake gates are process-wide atomics
/// (`set_mtp_load_depth` / `set_mtp_uqff_bake`); tests that flip them must
/// not interleave with each other. Poison-tolerant so one panicked test
/// cannot wedge the rest.
static MTP_GATE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn mtp_gate_guard() -> std::sync::MutexGuard<'static, ()> {
    MTP_GATE_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner())
}

// ---------------------------------------------------------------------------
// Shared test plumbing: a CPU-only `DeviceMapper` (the production
// `DummyDeviceMapper` is not re-exported) and the FlashParams empty struct
// that takes the non-varlen / non-flash kernel path.
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct CpuMapper {
    device: Device,
}

impl DeviceMapper for CpuMapper {
    fn map(&self, input: Tensor, _: usize) -> CandleResult<Tensor> {
        Ok(input)
    }
    fn set_device<'a>(
        &self,
        _: usize,
        varbuilder: ShardedVarBuilder,
        loading_isq: bool,
    ) -> ShardedVarBuilder {
        if loading_isq {
            varbuilder.set_device(Device::Cpu)
        } else {
            varbuilder.set_device(self.device.clone())
        }
    }
    fn device_for(&self, _: usize, _loading_isq: bool) -> Option<&Device> {
        Some(&self.device)
    }
    fn get_unique_devices(&self) -> Vec<Device> {
        vec![self.device.clone()]
    }
    fn cast_nm_device(&self, x: &Tensor, loading_isq: bool) -> CandleResult<Tensor> {
        if loading_isq {
            x.to_device(&Device::Cpu)
        } else {
            x.to_device(&self.device)
        }
    }
    fn set_nm_device<'a>(
        &self,
        varbuilder: ShardedVarBuilder,
        loading_isq: bool,
    ) -> ShardedVarBuilder {
        if loading_isq {
            varbuilder.set_device(Device::Cpu)
        } else {
            varbuilder.set_device(self.device.clone())
        }
    }
    fn get_min_dtype(&self, dtype: &dyn mistralrs_core::TryIntoDType) -> CandleResult<DType> {
        dtype
            .try_into_dtype(&[&self.device])
            .map_err(candle_core::Error::msg)
    }
    fn num_device_mapping_layers(&self) -> usize {
        1
    }
    fn get_comm_for(&self, _layer_idx: usize) -> CandleResult<Arc<mistralrs_quant::Comm>> {
        let id = mistralrs_quant::Id::new();
        Ok(Arc::new(mistralrs_quant::Comm::from_device(
            id,
            &self.device,
            0,
            1,
        )?))
    }
}

fn empty_flash_params() -> FlashParams {
    FlashParams {
        max_q: 0,
        max_k: 0,
        cumulative_seqlens_q: HashMap::new(),
        cumulative_seqlens_k: HashMap::new(),
        causal: true,
    }
}

fn make_metadata(device: &Device) -> NormalLoadingMetadata {
    NormalLoadingMetadata {
        mapper: Box::new(CpuMapper {
            device: device.clone(),
        }),
        loading_isq: false,
        real_device: device.clone(),
        multi_progress: Arc::new(MultiProgress::new()),
        matformer_slicing_config: None,
    }
}

fn wrap_as_vb(tensors: HashMap<String, Tensor>, device: &Device) -> ShardedVarBuilder {
    let backend: Box<dyn candle_nn::var_builder::SimpleBackend + 'static> = Box::new(tensors);
    ShardedSafeTensors::wrap(backend, DType::F32, device.clone())
}

fn zeros(shape: &[usize], device: &Device) -> CandleResult<Tensor> {
    Tensor::zeros(shape, DType::F32, device)
}

fn ones(shape: &[usize], device: &Device) -> CandleResult<Tensor> {
    Tensor::ones(shape, DType::F32, device)
}

/// Deterministic non-constant weights: element `n` is
/// `base + n * step` folded into a small band. Used where a test needs the
/// value to actually depend on its input (see `synthetic_v4_weights_with_mtp`).
fn ramp(shape: &[usize], base: f32, step: f32, device: &Device) -> CandleResult<Tensor> {
    let n: usize = shape.iter().product();
    let data: Vec<f32> = (0..n)
        .map(|i| {
            let v = base + (i % 97) as f32 * step;
            if i % 2 == 0 {
                v
            } else {
                -v
            }
        })
        .collect();
    Tensor::from_vec(data, shape, device)
}

/// Run a `[1, seq]` forward pass and return the logits. Single-token seq=1 is
/// the cheapest path and exercises every per-layer probe in the loader.
fn run_forward_smoke(
    model: &(dyn NormalModel + Send + Sync),
    input_ids: &Tensor,
) -> CandleResult<Tensor> {
    let seq_len = input_ids.dim(1)?;
    let context_lens = vec![(0usize, seq_len)];
    let position_ids = (0..seq_len).collect::<Vec<_>>();
    let seqlen_offsets = vec![0usize];
    let flash_params = empty_flash_params();
    model.forward(
        input_ids,
        &seqlen_offsets,
        context_lens,
        position_ids,
        None,
        &flash_params,
    )
}

fn assert_finite(logits: &Tensor, model_name: &str) {
    let flat: Vec<f32> = logits
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    let nan = flat.iter().filter(|v| v.is_nan()).count();
    let inf = flat.iter().filter(|v| v.is_infinite()).count();
    assert_eq!(
        nan, 0,
        "{model_name}: {nan} NaN(s) in logits. Likely softmax over a zero \
         logit row, or RoPE on an unhandled length."
    );
    assert_eq!(
        inf, 0,
        "{model_name}: {inf} Inf(s) in logits. Likely overflow in the softmax \
         scale or a missing norm."
    );
}

// ===========================================================================
// V4 FLASH SYNTHETIC SMOKE
// ===========================================================================
//
// V4 is fully implemented (post-Wave-2) and is the canonical Tier-A target.
// We synthesise the entire HF-format tensor layout, dispatch through the
// public `DeepSeekV4Loader`, and run a real forward pass.

mod v4 {
    use super::*;

    pub const NUM_LAYERS: usize = 2;
    pub const HIDDEN_SIZE: usize = 64;
    pub const HEAD_DIM: usize = 32;
    pub const QK_ROPE_HEAD_DIM: usize = 8;
    pub const NUM_ATTN_HEADS: usize = 2;
    pub const NUM_KV_HEADS: usize = 1;
    pub const Q_LORA_RANK: usize = 16;
    pub const O_LORA_RANK: usize = 16;
    pub const O_GROUPS: usize = 1;
    pub const MOE_INTERMEDIATE_SIZE: usize = 32;
    pub const NUM_ROUTED_EXPERTS: usize = 4;
    pub const NUM_SHARED_EXPERTS: usize = 1;
    pub const NUM_EXPERTS_PER_TOK: usize = 2;
    pub const VOCAB_SIZE: usize = 128;
    pub const MAX_POSITION_EMBEDDINGS: usize = 64;
    pub const RMS_NORM_EPS: f64 = 1e-6;
    pub const ROPE_THETA: f32 = 10000.0;
    pub const COMPRESS_ROPE_THETA: f32 = 40000.0;

    /// Build the full HF-format V4 tensor set. Naming follows
    /// `mistralrs-core/src/models/deepseek4.rs` exactly:
    /// - `model.embed_tokens.weight`, `model.norm.weight`, `lm_head.weight`
    /// - `model.layers.{i}.input_layernorm.weight`
    /// - `model.layers.{i}.post_attention_layernorm.weight`
    /// - `model.layers.{i}.self_attn.q_a_proj.weight` / `q_a_layernorm.weight` / `q_b_proj.weight`
    /// - `model.layers.{i}.self_attn.kv_a_proj_with_mqa.weight` (V3-style fallback for fused wkv)
    /// - `model.layers.{i}.self_attn.kv_a_layernorm.weight` (V3-style fallback for kv_norm)
    /// - `model.layers.{i}.self_attn.o_a_proj.weight` / `o_b_proj.weight`
    /// - `model.layers.{i}.mlp.gate.weight`
    /// - `model.layers.{i}.mlp.experts.gate_up_proj` (stacked, [E, H, 2*I])
    /// - `model.layers.{i}.mlp.experts.down_proj`     (stacked, [E, I, H])
    /// - `model.layers.{i}.mlp.shared_experts.{gate,up,down}_proj.weight`
    ///
    /// V4-specific note: we publish the V3-style legacy `kv_a_proj_with_mqa`
    /// (hidden→head_dim) rather than V4-native `wkv`. The V4 loader's
    /// auto-detection (`deepseek4.rs:762-770`) accepts this fallback path.
    pub fn synthetic_v4_weights(device: &Device) -> CandleResult<HashMap<String, Tensor>> {
        let mut t = HashMap::new();
        let z = |s: &[usize]| zeros(s, device);
        let o = |s: &[usize]| ones(s, device);

        // ---------- globals (HF naming) ----------
        t.insert(
            "model.embed_tokens.weight".to_string(),
            o(&[VOCAB_SIZE, HIDDEN_SIZE])?,
        );
        t.insert("model.norm.weight".to_string(), o(&[HIDDEN_SIZE])?);
        t.insert("lm_head.weight".to_string(), z(&[VOCAB_SIZE, HIDDEN_SIZE])?);

        for i in 0..NUM_LAYERS {
            insert_v4_decoder_layer_tensors(&mut t, &format!("model.layers.{i}"), device)?;
        }
        Ok(t)
    }

    /// Synthesise one full V4 decoder layer (attention + MoE) at `prefix`.
    /// Shared between the main `model.layers.{i}` stack and the MTP block at
    /// `mtp.layers.0` — the MTP block IS a V4 decoder layer (audit §2 "MTP
    /// module"), so the tensor contract is identical by construction.
    pub fn insert_v4_decoder_layer_tensors(
        t: &mut HashMap<String, Tensor>,
        prefix: &str,
        device: &Device,
    ) -> CandleResult<()> {
        let l = prefix;
        let z = |s: &[usize]| zeros(s, device);
        let o = |s: &[usize]| ones(s, device);

        // layer norms (ones — keeps the residual stream non-zero)
        t.insert(format!("{l}.input_layernorm.weight"), o(&[HIDDEN_SIZE])?);
        t.insert(
            format!("{l}.post_attention_layernorm.weight"),
            o(&[HIDDEN_SIZE])?,
        );

        // Q LoRA: q_a_proj + q_a_layernorm + q_b_proj
        // (loader probes V4-native `wq_a`/`q_norm` first; with no such
        // tensors it falls back to HF `q_a_proj`/`q_a_layernorm`.)
        t.insert(
            format!("{l}.self_attn.q_a_proj.weight"),
            z(&[Q_LORA_RANK, HIDDEN_SIZE])?,
        );
        t.insert(
            format!("{l}.self_attn.q_a_layernorm.weight"),
            o(&[Q_LORA_RANK])?,
        );
        t.insert(
            format!("{l}.self_attn.q_b_proj.weight"),
            z(&[NUM_ATTN_HEADS * HEAD_DIM, Q_LORA_RANK])?,
        );

        // V4 fused wkv (HF-legacy naming `kv_a_proj_with_mqa`): output dim
        // = head_dim (a SINGLE MQA head — V4 absorbs V into K). The V4
        // loader's auto-detection (`deepseek4.rs:762-770`) accepts this
        // V3-style fallback for synthetic / older checkpoints.
        t.insert(
            format!("{l}.self_attn.kv_a_proj_with_mqa.weight"),
            z(&[HEAD_DIM, HIDDEN_SIZE])?,
        );
        // kv_norm over head_dim (HF-legacy naming `kv_a_layernorm`).
        t.insert(
            format!("{l}.self_attn.kv_a_layernorm.weight"),
            o(&[HEAD_DIM])?,
        );

        // Grouped o_proj LoRA (HF naming `o_a_proj`/`o_b_proj`).
        // wo_a in = num_heads*head_dim/o_groups; wo_a out = o_groups*o_lora_rank.
        let wo_a_in = NUM_ATTN_HEADS * HEAD_DIM / O_GROUPS.max(1);
        let o_inner = O_GROUPS * O_LORA_RANK;
        t.insert(
            format!("{l}.self_attn.o_a_proj.weight"),
            z(&[o_inner, wo_a_in])?,
        );
        t.insert(
            format!("{l}.self_attn.o_b_proj.weight"),
            z(&[HIDDEN_SIZE, o_inner])?,
        );

        // ---- MoE block: every layer is MoE (first_k_dense_replace=0) ----
        // gate.weight: [n_routed_experts, hidden]
        t.insert(
            format!("{l}.mlp.gate.weight"),
            z(&[NUM_ROUTED_EXPERTS, HIDDEN_SIZE])?,
        );
        // TD-MoE hash-routing table (RUN-161): layers < num_hash_layers
        // (default 3 — with NUM_LAYERS=2 every layer hash-routes, and so
        // does the MTP block at virtual layer index 2) require
        // `gate.tid2eid`, a fixed token-id -> expert-id map of shape
        // [vocab_size, top_k], dtype I64. All-zeros routes every token to
        // expert 0, which is fine for a load/forward smoke.
        t.insert(
            format!("{l}.mlp.gate.tid2eid"),
            Tensor::zeros(&[VOCAB_SIZE, NUM_EXPERTS_PER_TOK], DType::I64, device)?,
        );
        // PackedExperts (Slow backend on CPU, unquantized) reads STACKED
        // `gate_up_proj` (one tensor for both gate+up, dim 2 doubled) and
        // `down_proj`. Shapes per `mistralrs-quant::PackedExperts::new` in
        // `mistralrs-quant/src/distributed/layers.rs:1377`.
        t.insert(
            format!("{l}.mlp.experts.gate_up_proj"),
            z(&[NUM_ROUTED_EXPERTS, HIDDEN_SIZE, 2 * MOE_INTERMEDIATE_SIZE])?,
        );
        t.insert(
            format!("{l}.mlp.experts.down_proj"),
            z(&[NUM_ROUTED_EXPERTS, MOE_INTERMEDIATE_SIZE, HIDDEN_SIZE])?,
        );
        // Shared experts (Mlp::new — `gate_proj`/`up_proj`/`down_proj`).
        let shared_inter = MOE_INTERMEDIATE_SIZE * NUM_SHARED_EXPERTS;
        t.insert(
            format!("{l}.mlp.shared_experts.gate_proj.weight"),
            z(&[shared_inter, HIDDEN_SIZE])?,
        );
        t.insert(
            format!("{l}.mlp.shared_experts.up_proj.weight"),
            z(&[shared_inter, HIDDEN_SIZE])?,
        );
        t.insert(
            format!("{l}.mlp.shared_experts.down_proj.weight"),
            z(&[HIDDEN_SIZE, shared_inter])?,
        );
        Ok(())
    }

    /// The V4 fixture plus the FULL `mtp.layers.0.*` module: the light
    /// `h_proj`/`e_proj` heads, the `hnorm`/`enorm`/`norm` wrapper norms, and
    /// a complete decoder layer (attention + MoE) — the tensor set the real
    /// V4 Flash checkpoint ships at `mtp.0.*` per `research/v4_audit.md` §2
    /// ("MTP module"), in HF naming.
    pub fn synthetic_v4_weights_with_mtp(device: &Device) -> CandleResult<HashMap<String, Tensor>> {
        let mut t = synthetic_v4_weights(device)?;
        let l = "mtp.layers.0";
        let o = |s: &[usize]| ones(s, device);

        // The MTP fixture is deliberately NON-degenerate, unlike the base V4
        // fixture. Audit §5 ("Existing Arc MTP tests are structurally
        // vacuous") called this out: with `h_proj`/`e_proj` both zero, `fused`
        // is identically 0, every draft token is the same constant, and no
        // assertion about the MTP math can fail. Here:
        //   * `embed_tokens` varies with the token id, so `e_proj`'s input is
        //     token-dependent,
        //   * `lm_head` varies with the hidden state, so argmax is not
        //     trivially 0, and
        //   * `h_proj` and `e_proj` are DIFFERENT non-zero maps, so a test can
        //     tell the two branches apart.
        // Confined to the MTP fixture on purpose — the base fixture's
        // degenerate weights are load-path assertions and stay as they are.
        t.insert(
            "model.embed_tokens.weight".to_string(),
            ramp(&[VOCAB_SIZE, HIDDEN_SIZE], 0.03, 0.011, device)?,
        );
        t.insert(
            "lm_head.weight".to_string(),
            ramp(&[VOCAB_SIZE, HIDDEN_SIZE], 0.02, 0.007, device)?,
        );

        // The base fixture's decoder weights are all zero, so attention and
        // MoE contribute nothing and the residual stream leaves the embedding
        // untouched — the hidden state and the embedding would be the SAME
        // vector, and a test asserting `h != embed(t)` could not distinguish a
        // correct capture from the finding-1 bug. Give each main layer a
        // non-zero shared expert so the residual actually moves.
        let shared_inter = MOE_INTERMEDIATE_SIZE * NUM_SHARED_EXPERTS;
        for i in 0..NUM_LAYERS {
            let l = format!("model.layers.{i}");
            t.insert(
                format!("{l}.mlp.shared_experts.gate_proj.weight"),
                ramp(&[shared_inter, HIDDEN_SIZE], 0.01, 0.004, device)?,
            );
            t.insert(
                format!("{l}.mlp.shared_experts.up_proj.weight"),
                ramp(&[shared_inter, HIDDEN_SIZE], 0.015, 0.003, device)?,
            );
            t.insert(
                format!("{l}.mlp.shared_experts.down_proj.weight"),
                ramp(&[HIDDEN_SIZE, shared_inter], 0.012, 0.002, device)?,
            );
        }

        // Light heads (always loaded when present). Distinct maps: a scaled
        // identity vs. a scaled cyclic shift.
        let mut h_data = vec![0f32; HIDDEN_SIZE * HIDDEN_SIZE];
        let mut e_data = vec![0f32; HIDDEN_SIZE * HIDDEN_SIZE];
        for i in 0..HIDDEN_SIZE {
            h_data[i * HIDDEN_SIZE + i] = 0.5;
            e_data[i * HIDDEN_SIZE + (i + 1) % HIDDEN_SIZE] = 0.25;
        }
        t.insert(
            format!("{l}.h_proj.weight"),
            Tensor::from_vec(h_data, (HIDDEN_SIZE, HIDDEN_SIZE), device)?,
        );
        t.insert(
            format!("{l}.e_proj.weight"),
            Tensor::from_vec(e_data, (HIDDEN_SIZE, HIDDEN_SIZE), device)?,
        );
        // Wrapper norms for the full block.
        t.insert(format!("{l}.hnorm.weight"), o(&[HIDDEN_SIZE])?);
        t.insert(format!("{l}.enorm.weight"), o(&[HIDDEN_SIZE])?);
        t.insert(format!("{l}.norm.weight"), o(&[HIDDEN_SIZE])?);
        // The full decoder layer (attention + MoE) at the MTP root.
        insert_v4_decoder_layer_tensors(&mut t, l, device)?;
        Ok(t)
    }

    pub fn config_json() -> String {
        // All compress_ratios = 0 keeps every layer on the standard
        // MLA-fallback path (no compressor / indexer / MTP / mHC tensors
        // needed). topk_method=greedy + scoring_func=softmax avoids the
        // noaux_tc bias-tensor requirement.
        serde_json::json!({
            "architectures": ["DeepseekV4ForCausalLM"],
            "vocab_size": VOCAB_SIZE,
            "hidden_size": HIDDEN_SIZE,
            "head_dim": HEAD_DIM,
            "moe_intermediate_size": MOE_INTERMEDIATE_SIZE,
            "num_hidden_layers": NUM_LAYERS,
            "num_attention_heads": NUM_ATTN_HEADS,
            "num_key_value_heads": NUM_KV_HEADS,
            "n_shared_experts": NUM_SHARED_EXPERTS,
            "n_routed_experts": NUM_ROUTED_EXPERTS,
            "routed_scaling_factor": 1.0,
            "topk_method": "greedy",
            "scoring_func": "softmax",
            "num_experts_per_tok": NUM_EXPERTS_PER_TOK,
            "moe_layer_freq": 1,
            "first_k_dense_replace": 0,
            "hidden_act": "silu",
            "max_position_embeddings": MAX_POSITION_EMBEDDINGS,
            "rms_norm_eps": RMS_NORM_EPS,
            "tie_word_embeddings": false,
            "rope_theta": ROPE_THETA,
            "attention_bias": false,
            "q_lora_rank": Q_LORA_RANK,
            "qk_rope_head_dim": QK_ROPE_HEAD_DIM,
            "n_group": 1,
            "topk_group": 1,
            "compress_ratios": [0, 0],
            "sliding_window": 16,
            "compress_rope_theta": COMPRESS_ROPE_THETA,
            "o_lora_rank": O_LORA_RANK,
            "o_groups": O_GROUPS,
            "hc_mult": 4,
        })
        .to_string()
    }
}

#[test]
fn v4_flash_synthetic_load_smoke() {
    let device = Device::Cpu;

    // Build synthetic tensors + config.
    let tensors =
        v4::synthetic_v4_weights(&device).expect("V4 synthetic tensor construction must not fail");
    let vb = wrap_as_vb(tensors, &device);
    let config = v4::config_json();

    // Sanity: the auto-dispatcher must recognise V4.
    assert!(matches!(
        NormalLoaderType::from_causal_lm_name("DeepseekV4ForCausalLM").unwrap(),
        NormalLoaderType::DeepSeekV4
    ));

    // Drive the actual V4 loader.
    let loader = DeepSeekV4Loader;
    let mut model = loader
        .load(
            &config,
            vb,
            make_metadata(&device),
            AttentionImplementation::Eager,
        )
        .expect(
            "DeepSeekV4Loader::load must succeed on synthetic HF-shaped \
             weights. If this panics, the missing tensor name in the error \
             is the contract drift: add it to v4::synthetic_v4_weights or \
             fix deepseek4.rs.",
        );

    // Single-token forward.
    let input_ids = Tensor::from_vec(vec![0u32, 1], &[1usize, 2], &device).unwrap();
    let logits = run_forward_smoke(model.as_ref(), &input_ids)
        .expect("V4 forward on synthetic weights must not error");

    // Shape: context_lens=[(0, seq)] keeps all positions.
    assert_eq!(
        logits.dims(),
        &[1usize, 2, v4::VOCAB_SIZE],
        "V4 logits shape mismatch — extract_logits / lm_head contract drift?",
    );
    assert_finite(&logits, "V4 Flash");

    // Reset cache and verify determinism on a second pass.
    {
        let mut cache = model.cache_mut().normal();
        cache.0.iter_mut().for_each(|c| c.reset());
    }
    let logits2 =
        run_forward_smoke(model.as_ref(), &input_ids).expect("V4 second forward must succeed");
    assert_eq!(logits.dims(), logits2.dims(), "V4 shape drifted on rerun");
}

/// Full MTP block load + draft-chain smoke (feat/mtp-full-block).
///
/// The real V4 Flash checkpoint ships a FULL decoder layer (attention +
/// 256-expert MoE) at `mtp.0.*` alongside `h_proj`/`e_proj`/`hnorm`/`enorm`/
/// `norm` (audit §2 "MTP module"). This test synthesises that tensor set (HF
/// naming, `mtp.layers.0.*`) and asserts:
///   1. With `mtp_load_depth == 0` (the default) the block is NOT loaded —
///      the ~3GB module must never cost memory when MTP is off — while the
///      light heads still are (kit exists, `has_full_block() == false`).
///   2. With `mtp_load_depth > 0` the block loads, and a depth-2 draft chain
///      flows through the real transformer step (embedding combine → decoder
///      layer with per-chain KV cache → norm → lm_head) producing in-vocab
///      tokens.
#[test]
fn v4_flash_mtp_full_block_load_smoke() {
    let _gate = mtp_gate_guard();
    let device = Device::Cpu;
    let config = v4::config_json();

    // ---- Gate check: depth 0 → heads-only, no block. ----
    mistralrs_core::set_mtp_load_depth(0);
    let tensors = v4::synthetic_v4_weights_with_mtp(&device)
        .expect("V4+MTP synthetic tensor construction must not fail");
    let vb = wrap_as_vb(tensors, &device);
    let model = DeepSeekV4Loader
        .load(
            &config,
            vb,
            make_metadata(&device),
            AttentionImplementation::Eager,
        )
        .expect("V4 load with MTP tensors present + depth=0 must succeed");
    let kit = model
        .mtp_decode_kit()
        .expect("mtp.layers.0.h_proj present -> the light MTP kit must exist");
    assert!(
        !kit.has_full_block(),
        "mtp_load_depth == 0 must NOT load the full MTP block (memory gate)"
    );

    // ---- Full-block path: depth > 0 loads the decoder block. ----
    mistralrs_core::set_mtp_load_depth(2);
    let tensors = v4::synthetic_v4_weights_with_mtp(&device)
        .expect("V4+MTP synthetic tensor construction must not fail");
    let vb = wrap_as_vb(tensors, &device);
    let model = DeepSeekV4Loader
        .load(
            &config,
            vb,
            make_metadata(&device),
            AttentionImplementation::Eager,
        )
        .expect(
            "V4 load with the full mtp.layers.0.* module must succeed. A \
             missing-tensor error here means the MtpBlock loader drifted from \
             the audit §2 MTP schema (or the fixture is stale).",
        );
    // Reset the process-wide gate before any assertion can panic, so a
    // failure here can't leak depth>0 into concurrently-running tests.
    mistralrs_core::set_mtp_load_depth(0);

    let kit = model
        .mtp_decode_kit()
        .expect("MTP kit must exist when mtp tensors are present");
    assert!(
        kit.has_full_block(),
        "mtp_load_depth > 0 + full mtp.layers.0.* tensors must load the block"
    );

    // Draft a 2-token chain through the real block, starting from an empty
    // context (start_pos 0 with an empty draft KV is the degenerate but
    // self-consistent case: slot k really is absolute position k).
    let seed_hidden = Tensor::from_vec(
        (0..v4::HIDDEN_SIZE).map(|i| 0.01 * i as f32).collect(),
        (1, v4::HIDDEN_SIZE),
        &device,
    )
    .unwrap();
    let mut draft_cache = kit
        .new_draft_cache()
        .expect("a full-block kit must offer a draft KV cache");
    let toks = kit
        .propose_chain(
            &seed_hidden,
            1,
            /*depth=*/ 2,
            /*max_tokens=*/ 8,
            /*start_pos=*/ 0,
            Some(&mut draft_cache),
        )
        .expect("MTP full-block draft chain must not error");
    assert_eq!(toks.len(), 2, "depth-2 chain must yield exactly 2 tokens");
    for t in &toks {
        assert!(
            (*t as usize) < v4::VOCAB_SIZE,
            "draft token {t} out of vocab range {}",
            v4::VOCAB_SIZE
        );
    }
}

/// Load a depth-2-capable V4 fixture and hand back the model.
fn load_v4_with_mtp_block(device: &Device) -> Box<dyn NormalModel + Send + Sync> {
    let config = v4::config_json();
    mistralrs_core::set_mtp_load_depth(2);
    let tensors = v4::synthetic_v4_weights_with_mtp(device)
        .expect("V4+MTP synthetic tensor construction must not fail");
    let vb = wrap_as_vb(tensors, device);
    let model = DeepSeekV4Loader.load(
        &config,
        vb,
        make_metadata(device),
        AttentionImplementation::Eager,
    );
    // Reset the process-wide gate before any assertion can panic.
    mistralrs_core::set_mtp_load_depth(0);
    model.expect("V4 load with the full mtp.layers.0.* module must succeed")
}

/// **Audit finding 1** — the MTP head must be fed TWO DIFFERENT signals: the
/// target model's hidden state into `h_proj`, and the token embedding into
/// `e_proj`.
///
/// Reference: `deepseek_v4_nextn.py:155-161` —
/// `h_proj(hnorm(spec_info.hidden_states))` + `e_proj(enorm(embed(input_ids)))`.
/// The pre-fix pipeline passed `embed(T0)` to *both*, collapsing the head to a
/// single-input function of the token id: acceptance ≈ noise regardless of
/// quantization.
///
/// The test pins the observable consequence: with the two inputs held apart, a
/// draft step must respond to a change in the hidden state alone. Under the
/// old seeding it could not — the hidden state was not an input at all.
#[test]
fn v4_mtp_draft_depends_on_the_target_hidden_state_not_only_the_token() {
    let _gate = mtp_gate_guard();
    let device = Device::Cpu;
    let model = load_v4_with_mtp_block(&device);
    let kit = model.mtp_decode_kit().expect("MTP kit must exist");
    assert!(kit.has_full_block());

    let hidden_a = Tensor::from_vec(
        (0..v4::HIDDEN_SIZE).map(|i| 0.05 * i as f32).collect(),
        (1, v4::HIDDEN_SIZE),
        &device,
    )
    .unwrap();
    let hidden_b = Tensor::from_vec(
        (0..v4::HIDDEN_SIZE)
            .map(|i| -0.05 * (i as f32) - 1.0)
            .collect(),
        (1, v4::HIDDEN_SIZE),
        &device,
    )
    .unwrap();

    // Teeth check: the two candidate `h` inputs are genuinely different, and
    // neither is the embedding that `e_proj` receives. If a future refactor
    // re-seeds `h` from `embed(tok)`, this is the assertion that goes stale
    // first — so assert it explicitly rather than relying on the value test.
    let e_input = kit
        .embed_tokens
        .forward(&Tensor::from_vec(vec![3u32], (1,), &device).unwrap())
        .unwrap();
    let as_vec = |t: &Tensor| -> Vec<f32> { t.flatten_all().unwrap().to_vec1().unwrap() };
    assert_ne!(as_vec(&hidden_a), as_vec(&hidden_b));
    assert_ne!(
        as_vec(&hidden_a),
        as_vec(&e_input),
        "the h_proj input and the e_proj input must not be the same tensor — \
         feeding embed(T0) to both is audit finding 1"
    );

    // Same token, same positions, same (empty) context: the ONLY difference is
    // the hidden state fed to `h_proj`. `propose_chain` at depth 2 is the
    // smallest public surface that runs real steps.
    let mut cache_a = kit.new_draft_cache().unwrap();
    let mut cache_b = kit.new_draft_cache().unwrap();
    let chain_a = kit
        .propose_chain(&hidden_a, 3, 2, 8, 0, Some(&mut cache_a))
        .expect("draft step must not error");
    let chain_b = kit
        .propose_chain(&hidden_b, 3, 2, 8, 0, Some(&mut cache_b))
        .expect("draft step must not error");
    assert_ne!(
        chain_a, chain_b,
        "the MTP draft step ignored the target hidden state — h_proj is \
         receiving something that does not vary with `h` (audit finding 1)"
    );

    // ---- The part that actually pins the pipeline fix. -------------------
    // The old pipeline had no way to obtain `h` at all, so it seeded BOTH
    // branches with `embed(T0)`. The fix is the capture channel: the target's
    // own forward must hand out its pre-`lm_head` hidden states, and they must
    // not be the token embedding.
    let input_ids = Tensor::from_vec(vec![4u32, 9, 16], &[1usize, 3], &device).unwrap();
    let _ = run_forward_smoke(model.as_ref(), &input_ids).expect("V4 forward must succeed");
    let (start_pos, captured) = kit
        .hidden_capture
        .take()
        .expect("an armed MTP kit must capture the target's hidden states");
    assert_eq!(
        start_pos, 0,
        "the capture must be tagged with the absolute position of its first row"
    );
    assert_eq!(
        captured.dims(),
        &[1usize, 3, v4::HIDDEN_SIZE],
        "the capture must cover EVERY input position (the draft-KV prefill \
         needs h_i at each one), not just the sampled row"
    );
    let h_last = captured.i((0, 2)).unwrap();
    let embed_last = kit
        .embed_tokens
        .forward(&Tensor::from_vec(vec![16u32], (1,), &device).unwrap())
        .unwrap()
        .squeeze(0)
        .unwrap();
    assert_ne!(
        as_vec(&h_last),
        as_vec(&embed_last),
        "the captured hidden state is the token embedding — the pipeline would \
         be feeding the SAME vector to h_proj and e_proj (audit finding 1)"
    );

    // Taking clears the slot: a stale block must never be readable by a later
    // step (that is how one sequence's hidden state would leak into another's).
    assert!(
        kit.hidden_capture.take().is_none(),
        "the capture slot must be empty after a take"
    );
}

/// **Audit finding 2** — the MTP block's KV cache must hold the accepted
/// context before the first draft step, because the block applies ABSOLUTE
/// RoPE positions.
///
/// Reference: the draft model owns its own KV pool and prefills it over the
/// whole context (`eagle_worker.py:134-138`, `:1094-1128`), so a draft query
/// at position `P` attends over real keys at `0..P`. The pre-fix code
/// allocated a fresh empty cache per chain and then indexed positions
/// `P, P+1, …` into it — a query the model believes is at position 200 000
/// attending one key.
///
/// Pins three things:
///   (i)   a full-block kit REFUSES to draft without a draft KV,
///   (ii)  the draft KV is non-empty at the first draft step after a prefill,
///         and the chain appends to it rather than starting from zero, and
///   (iii) slot index and absolute position cannot silently drift apart.
#[test]
fn v4_mtp_draft_kv_holds_the_accepted_context() {
    let _gate = mtp_gate_guard();
    let device = Device::Cpu;
    let model = load_v4_with_mtp_block(&device);
    let kit = model.mtp_decode_kit().expect("MTP kit must exist");

    // (i) No draft KV -> hard error, never a silent empty-cache draft.
    let seed = Tensor::zeros((1, v4::HIDDEN_SIZE), DType::F32, &device).unwrap();
    let err = kit
        .propose_chain(&seed, 1, 2, 8, 0, None)
        .expect_err("full-block drafting without a draft KV must be refused");
    assert!(
        format!("{err}").contains("draft KV"),
        "unexpected error: {err}"
    );

    // (ii) Prefill the draft KV over a 5-token "accepted context", exactly as
    // the pipeline does from the prompt forward's captured hidden states:
    // slot i is the MTP state of (h_i, tok_{i+1}).
    const CTX: usize = 5;
    let mut cache = kit.new_draft_cache().unwrap();
    assert_eq!(cache.current_seq_len(), 0, "a fresh draft KV starts empty");

    let ctx_hidden = ramp(&[1, CTX, v4::HIDDEN_SIZE], 0.02, 0.005, &device).unwrap();
    let next_tokens = Tensor::from_vec(vec![2u32, 3, 5, 7, 11], (1, CTX), &device).unwrap();
    let filled = kit
        .extend_draft_cache(&mut cache, 0, &ctx_hidden, &next_tokens)
        .expect("draft-KV prefill over the accepted context must succeed");
    assert_eq!(filled, CTX);
    assert_eq!(
        cache.current_seq_len(),
        CTX,
        "the draft KV must hold one entry per accepted context position — an \
         empty cache here is audit finding 2"
    );

    // The first draft step now attends over a NON-EMPTY cache, and lands at
    // the next absolute position.
    let toks = kit
        .propose_chain(&seed, 13, 2, 8, CTX, Some(&mut cache))
        .expect("chain against a primed draft KV must not error");
    assert_eq!(toks.len(), 2);
    assert_eq!(
        cache.current_seq_len(),
        CTX + 2,
        "each draft step must append exactly one entry at its absolute position"
    );

    // (iii) A cache whose length disagrees with the step's absolute position
    // is a desync, not something to paper over.
    let mut stale = kit.new_draft_cache().unwrap();
    let err = kit
        .propose_chain(&seed, 13, 1, 8, 137, Some(&mut stale))
        .expect_err("start_pos far past an empty draft KV must be refused");
    assert!(
        format!("{err}").contains("desync"),
        "unexpected error: {err}"
    );
}

/// **Acceptance path, end to end on the synthetic model.**
///
/// Uses the real V4 fixture as its own verifier: run a forward, take the
/// target's greedy argmax per position, and drive the accept/reject logic with
/// (a) a deliberately CORRECT draft — the target's own tokens, which must be
/// accepted with no correction — and (b) a deliberately WRONG draft, which
/// must be rejected at exactly the corrupted slot and corrected to the
/// target's token.
///
/// The load-bearing assertion is the losslessness contract the whole module
/// rests on: whatever the draft proposes, the committed stream equals the
/// target's own greedy decode.
#[test]
fn v4_mtp_accept_path_accepts_correct_drafts_and_rejects_wrong_ones() {
    let _gate = mtp_gate_guard();
    let device = Device::Cpu;
    let model = load_v4_with_mtp_block(&device);
    // Fetch the kit BEFORE the forward: that is what arms the hidden capture.
    let kit = model.mtp_decode_kit().expect("MTP kit must exist");

    // Target's greedy tokens over a short window — the verifier's answer.
    let input_ids = Tensor::from_vec(vec![4u32, 9, 16, 25], &[1usize, 4], &device).unwrap();
    let logits = run_forward_smoke(model.as_ref(), &input_ids).expect("V4 forward must succeed");
    assert_finite(&logits, "V4 MTP verify");
    let rows = logits.squeeze(0).unwrap();
    let verifier: Vec<u32> = (0..rows.dim(0).unwrap())
        .map(|i| {
            rows.get(i)
                .unwrap()
                .argmax(0)
                .unwrap()
                .to_dtype(DType::U32)
                .unwrap()
                .to_scalar::<u32>()
                .unwrap()
        })
        .collect();
    assert_eq!(verifier.len(), 4);

    // (a) Deliberately CORRECT draft: every proposal matches the target.
    let correct = verifier.clone();
    let res = mistralrs_core::verify_proposed(&correct, &verifier);
    assert_eq!(
        res.accepted, verifier,
        "a draft that matches the target's greedy tokens must be fully accepted"
    );
    assert!(
        res.rejection.is_none(),
        "no correction is due when nothing was rejected"
    );
    assert_eq!(res.commit_len(), verifier.len());
    let committed: Vec<u32> = res.accepted.clone();
    assert_eq!(
        committed, verifier,
        "losslessness: the committed stream must equal the target's own decode"
    );

    // (b) Deliberately WRONG draft: corrupt slot 2 only.
    let mut wrong = verifier.clone();
    wrong[2] = (verifier[2] + 1) % v4::VOCAB_SIZE as u32;
    assert_ne!(wrong[2], verifier[2], "the corruption must actually differ");
    let res = mistralrs_core::verify_proposed(&wrong, &verifier);
    assert_eq!(
        res.accepted,
        verifier[..2].to_vec(),
        "the matching prefix must still be accepted"
    );
    assert_eq!(
        res.rejection,
        Some((2, verifier[2])),
        "rejection must land on the corrupted slot and carry the TARGET's token"
    );
    let mut committed = res.accepted.clone();
    committed.push(res.rejection.unwrap().1);
    assert_eq!(
        committed,
        verifier[..3].to_vec(),
        "losslessness: accepted ++ correction must equal the target's own decode"
    );

    // (c) A REAL draft from the MTP block, verified against the same target.
    // Whatever it proposes, the committed stream must still be a prefix of the
    // target's own greedy decode — the invariant the whole module rests on.
    let (_, captured) = kit
        .hidden_capture
        .take()
        .expect("the forward above must have captured hidden states");
    let seed = captured.i((0, captured.dim(1).unwrap() - 1)).unwrap();
    let mut cache = kit.new_draft_cache().unwrap();
    let drafted = kit
        .propose_chain(&seed, verifier[0], verifier.len(), 8, 0, Some(&mut cache))
        .expect("a real MTP draft chain must not error");
    let res = mistralrs_core::verify_proposed(&drafted, &verifier);
    let mut committed = res.accepted.clone();
    if let Some((_, correction)) = res.rejection {
        committed.push(correction);
    }
    assert_eq!(
        committed,
        verifier[..committed.len()].to_vec(),
        "losslessness violated: the MTP-committed stream diverged from the \
         target's own greedy decode (drafted={drafted:?} verifier={verifier:?})"
    );
    assert!(
        res.accepted.len() <= drafted.len(),
        "cannot accept more tokens than were proposed"
    );
}

// ===========================================================================
// V4 UQFF ↔ MTP ARTIFACT COVERAGE
// ===========================================================================
//
// GPU session 2 crash: a UQFF baked WITHOUT `--mtp-depth` and then served
// WITH `--mtp-depth 2` died with `DummyLayer not replaced at index 522` —
// the MTP decoder block registers trailing ISQ layers that the artifact
// never covered. Two-sided fix under test here:
//   1. WRITE: a bake force-loads the MTP block when the checkpoint ships it
//      (`set_mtp_uqff_bake`, set by the loader whenever `--write-uqff` is
//      given) and serializes its layers under `mtp.<j>` names, so the same
//      artifact also stays loadable at `--mtp-depth 0` (entries skipped).
//   2. READ: serving an OLD artifact (no `mtp.*` entries) with the block
//      requested reloads the block UNQUANTIZED from the source checkpoint
//      instead of dying.

mod uqff {
    use super::*;
    use std::path::{Path, PathBuf};

    /// Fresh per-test scratch dir under the system temp dir.
    pub fn scratch_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("arc-uqff-smoke-{tag}-{}", std::process::id()));
        // Stale leftovers from a previous run would confuse shard reads.
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("scratch dir must be creatable");
        dir
    }

    /// Serialize the model's ISQ tensors (no additional quantization — the
    /// synthetic weights stay unquantized) to `<dir>/model-0.uqff` via the
    /// REAL `IsqModel::quantize` write path, and return the shard path.
    pub fn bake(model: &mut (dyn NormalModel + Send + Sync), dir: &Path) -> PathBuf {
        let tokenizer = tokenizers::Tokenizer::new(tokenizers::models::bpe::BPE::default());
        let uqff_target = dir.join("model.uqff");
        model
            .quantize(
                None,
                Device::Cpu,
                None,
                /*silent=*/ true,
                None,
                IsqOrganization::Default,
                /*apply_quantization=*/ false,
                Some(&uqff_target),
                UqffFullSer {
                    tokenizer: &tokenizer,
                    template_filename: &None,
                    modules: None,
                    module_paths: None,
                    generation_config: None,
                    config: v4::config_json(),
                    processor_filename: &None,
                    preprocessor_filename: &None,
                },
                Arc::new(MultiProgress::new()),
            )
            .expect("UQFF serialization of the synthetic V4 model must succeed");
        let shard = dir.join("model-0.uqff");
        assert!(shard.exists(), "expected a single UQFF shard at {shard:?}");
        shard
    }

    /// Tensor names inside a UQFF shard.
    pub fn artifact_names(shard: &Path) -> Vec<String> {
        let st = unsafe {
            candle_core::safetensors::MmapedSafetensors::new(shard).expect("UQFF shard must mmap")
        };
        st.tensors().into_iter().map(|(name, _)| name).collect()
    }

    /// Drive the actual V4 loader over an in-memory synthetic tensor set.
    pub fn load_v4(tensors: HashMap<String, Tensor>) -> Box<dyn NormalModel + Send + Sync> {
        let device = Device::Cpu;
        let vb = wrap_as_vb(tensors, &device);
        DeepSeekV4Loader
            .load(
                &v4::config_json(),
                vb,
                make_metadata(&device),
                AttentionImplementation::Eager,
            )
            .expect("V4 load on synthetic weights must succeed")
    }
}

/// WRITE side: `set_mtp_uqff_bake(true)` force-loads the MTP block at depth
/// 0, its layers serialize under `mtp.<j>` names, the main-model names are
/// UNCHANGED vs a block-less bake, and the artifact roundtrips both with
/// (`--mtp-depth 2`) and without (`--mtp-depth 0`) the block.
#[test]
fn v4_uqff_bake_includes_mtp_and_roundtrips() {
    let _gate = mtp_gate_guard();
    let device = Device::Cpu;

    // ---- Bake WITHOUT the MTP block (depth 0, bake flag off). ----
    mistralrs_core::set_mtp_load_depth(0);
    mistralrs_core::set_mtp_uqff_bake(false);
    let dir_plain = uqff::scratch_dir("bake-plain");
    let mut model = uqff::load_v4(v4::synthetic_v4_weights_with_mtp(&device).unwrap());
    let shard_plain = uqff::bake(model.as_mut(), &dir_plain);
    drop(model);
    let plain_names = uqff::artifact_names(&shard_plain);
    assert!(
        plain_names.iter().all(|n| n.parse::<usize>().is_ok()),
        "depth-0 bake must contain only positional-index names, got {plain_names:?}"
    );
    let n_main = plain_names.len();

    // ---- Bake with the production `--write-uqff` gate: the block force-
    // loads even at depth 0 and serializes under `mtp.<j>` names. ----
    mistralrs_core::set_mtp_uqff_bake(true);
    let dir_mtp = uqff::scratch_dir("bake-mtp");
    let mut model = uqff::load_v4(v4::synthetic_v4_weights_with_mtp(&device).unwrap());
    // Reset the process-wide gate before any assertion can panic.
    mistralrs_core::set_mtp_uqff_bake(false);
    assert!(
        model
            .mtp_decode_kit()
            .expect("MTP kit must exist")
            .has_full_block(),
        "set_mtp_uqff_bake(true) must force-load the MTP decoder block at depth 0"
    );
    let shard_mtp = uqff::bake(model.as_mut(), &dir_mtp);
    drop(model);
    let mtp_names = uqff::artifact_names(&shard_mtp);
    let numeric = mtp_names
        .iter()
        .filter(|n| n.parse::<usize>().is_ok())
        .count();
    let mtp_tail = mtp_names
        .iter()
        .filter(|n| n.starts_with(UQFF_MTP_TENSOR_PREFIX))
        .count();
    assert_eq!(
        numeric + mtp_tail,
        mtp_names.len(),
        "unexpected artifact names: {mtp_names:?}"
    );
    assert_eq!(
        numeric, n_main,
        "main-model artifact names must be identical with and without the MTP block"
    );
    assert!(
        mtp_tail > 0,
        "an MTP bake must include mtp.<j> tensors, got {mtp_names:?}"
    );

    // ---- Serve the MTP-baked artifact WITH the block (`--mtp-depth 2`). ----
    mistralrs_core::set_mtp_load_depth(2);
    let mut model = uqff::load_v4(v4::synthetic_v4_weights_with_mtp(&device).unwrap());
    mistralrs_core::set_mtp_load_depth(0);
    model
        .load_from_artifacts(Device::Cpu, None, true, &[shard_mtp.clone()], None)
        .expect("depth>0 serve of an MTP-baked artifact must load cleanly");
    assert!(model.mtp_decode_kit().unwrap().has_full_block());
    let input_ids = Tensor::from_vec(vec![0u32, 1], &[1usize, 2], &device).unwrap();
    let logits = run_forward_smoke(model.as_ref(), &input_ids)
        .expect("V4 forward after UQFF roundtrip must not error");
    assert_finite(&logits, "V4 UQFF roundtrip (depth 2)");
    drop(model);

    // ---- Serve the SAME artifact WITHOUT the block (`--mtp-depth 0`): the
    // mtp.* entries must be skipped, not fatal. This is the index-collision
    // hazard that the `mtp.<j>` naming exists to prevent. ----
    let mut model = uqff::load_v4(v4::synthetic_v4_weights_with_mtp(&device).unwrap());
    model
        .load_from_artifacts(Device::Cpu, None, true, &[shard_mtp.clone()], None)
        .expect("depth-0 serve of an MTP-baked artifact must skip mtp.* entries");
    assert!(!model.mtp_decode_kit().unwrap().has_full_block());

    let _ = std::fs::remove_dir_all(&dir_plain);
    let _ = std::fs::remove_dir_all(&dir_mtp);
}

/// READ side: an artifact with NO `mtp.*` coverage (baked without
/// `--mtp-depth` — the session-2 shape) served with the block requested must
/// reload the block unquantized from the source checkpoint; without a source
/// it must surface the DummyLayer error rather than panic.
#[test]
fn v4_uqff_missing_mtp_falls_back_to_source_checkpoint() {
    let _gate = mtp_gate_guard();
    let device = Device::Cpu;
    let dir = uqff::scratch_dir("mtp-fallback");

    // Source checkpoint on disk: the FULL fixture, every mtp tensor present.
    let full = v4::synthetic_v4_weights_with_mtp(&device).unwrap();
    let source_path = dir.join("source.safetensors");
    candle_core::safetensors::save(&full, &source_path).expect("source fixture must serialize");

    // ---- Bake WITHOUT the MTP block: the session-2 artifact shape. ----
    mistralrs_core::set_mtp_load_depth(0);
    mistralrs_core::set_mtp_uqff_bake(false);
    let mut bake_model = uqff::load_v4(full.clone());
    let shard = uqff::bake(bake_model.as_mut(), &dir);
    drop(bake_model);
    assert!(
        uqff::artifact_names(&shard)
            .iter()
            .all(|n| !n.starts_with(UQFF_MTP_TENSOR_PREFIX)),
        "depth-0 bake must not contain mtp.* tensors"
    );

    // Serve-side fixture: the block's Q-LoRA projection tensors are ABSENT,
    // so the quant constructors emit DummyLayers — exactly how the
    // production from-UQFF load builds ISQ-matched layers (make_dummy
    // regexes drop the tensors). The norms stay present so the block's
    // structure probes still pass.
    let dummy_fixture = || {
        let mut t = full.clone();
        t.remove("mtp.layers.0.self_attn.q_a_proj.weight")
            .expect("fixture must carry the MTP q_a_proj");
        t.remove("mtp.layers.0.self_attn.q_b_proj.weight")
            .expect("fixture must carry the MTP q_b_proj");
        t
    };

    // ---- Negative control: no source fallback -> the session-2 failure,
    // surfaced as an error (not a panic). ----
    mistralrs_core::set_mtp_load_depth(2);
    let mut model = uqff::load_v4(dummy_fixture());
    mistralrs_core::set_mtp_load_depth(0);
    let err = model
        .load_from_artifacts(Device::Cpu, None, true, &[shard.clone()], None)
        .expect_err("missing MTP artifacts without a source fallback must error");
    assert!(
        err.to_string().contains("DummyLayer not replaced"),
        "unexpected error: {err}"
    );
    drop(model);

    // ---- The fix: with source weights the block reloads unquantized. ----
    mistralrs_core::set_mtp_load_depth(2);
    let mut model = uqff::load_v4(dummy_fixture());
    mistralrs_core::set_mtp_load_depth(0);
    let source_files = vec![source_path.clone()];
    model
        .load_from_artifacts(
            Device::Cpu,
            None,
            true,
            &[shard.clone()],
            Some(UqffSourceWeights {
                weight_files: &source_files,
                dtype: DType::F32,
            }),
        )
        .expect("missing MTP artifacts + source fallback must load cleanly");
    let kit = model.mtp_decode_kit().expect("MTP kit must exist");
    assert!(
        kit.has_full_block(),
        "the source fallback must produce a REAL MTP block"
    );

    // The reloaded block must actually draft.
    let seed_hidden = ramp(&[1, v4::HIDDEN_SIZE], 0.01, 0.003, &device).unwrap();
    let mut draft_cache = kit
        .new_draft_cache()
        .expect("a full-block kit must offer a draft KV cache");
    let toks = kit
        .propose_chain(
            &seed_hidden,
            1,
            /*depth=*/ 2,
            /*max_tokens=*/ 8,
            0,
            Some(&mut draft_cache),
        )
        .expect("draft chain through the source-reloaded block must not error");
    assert_eq!(toks.len(), 2, "depth-2 chain must yield exactly 2 tokens");

    let _ = std::fs::remove_dir_all(&dir);
}

// ===========================================================================
// V4 XS-HISTORY MULTI-SEQUENCE (VOTING) TESTS
// ===========================================================================
//
// The V4 compressor input history (`xs_history`) used to be a single
// per-model buffer, which crashed / cross-contaminated when a batch held
// more than one sequence (Arc Boost `n_votes` voting chains — the hardware
// repro: `n_votes: 2` → `narrow invalid args ... [1, 2, 18, 512]`). The fix
// stores the history as extra per-layer entries in the model's NormalCache
// so the engine's per-sequence clone_in/clone_out machinery batches it along
// dim 0 exactly like the KV cache.
//
// These tests pin the correctness contract for voting: a 2-sequence batch
// through the compressor path (CSA ratio-4 + HCA ratio-128 layers), crossing
// the 128-token sliding window, must produce per-sequence outputs equal to
// the same sequences run separately (batch-of-1), within f32 tolerance —
// both when the two chains run in lockstep from prefill, and when they are
// prefilled separately and merged into one decode batch (the engine's
// clone_in path).

mod v4_compress {
    use super::*;

    pub const NUM_LAYERS: usize = 3;
    pub const HIDDEN_SIZE: usize = 64;
    pub const HEAD_DIM: usize = 32;
    pub const QK_ROPE_HEAD_DIM: usize = 8;
    pub const NUM_ATTN_HEADS: usize = 2;
    pub const NUM_KV_HEADS: usize = 1;
    pub const Q_LORA_RANK: usize = 16;
    pub const O_LORA_RANK: usize = 16;
    pub const O_GROUPS: usize = 1;
    pub const MOE_INTERMEDIATE_SIZE: usize = 32;
    pub const NUM_ROUTED_EXPERTS: usize = 4;
    pub const NUM_SHARED_EXPERTS: usize = 1;
    pub const NUM_EXPERTS_PER_TOK: usize = 2;
    pub const VOCAB_SIZE: usize = 128;
    pub const MAX_POSITION_EMBEDDINGS: usize = 256;
    pub const RMS_NORM_EPS: f64 = 1e-6;
    pub const ROPE_THETA: f32 = 10000.0;
    pub const COMPRESS_ROPE_THETA: f32 = 40000.0;
    /// Real V4 window. The test prefills PAST this so the compressed
    /// (distant-context) branch is live with a non-trivial history — the
    /// exact shape class of the voting crash.
    pub const SLIDING_WINDOW: usize = 128;
    /// Per-layer compress dispatch: Standard, CSA (ratio 4, overlap
    /// compressor), HCA (ratio 128).
    pub const COMPRESS_RATIOS: [i32; NUM_LAYERS] = [0, 4, 128];

    /// Deterministic non-degenerate weights: unlike the zero-weight smoke
    /// fixture, the equality assertions here are only meaningful if two
    /// different token sequences produce two different logit streams.
    pub fn patterned(
        shape: &[usize],
        scale: f32,
        phase: f32,
        device: &Device,
    ) -> CandleResult<Tensor> {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n)
            .map(|i| ((i as f32) * 0.37 + phase).sin() * scale)
            .collect();
        Tensor::from_vec(data, shape, device)
    }

    pub fn weights(device: &Device) -> CandleResult<HashMap<String, Tensor>> {
        let mut t = HashMap::new();
        let o = |s: &[usize]| ones(s, device);

        t.insert(
            "model.embed_tokens.weight".to_string(),
            patterned(&[VOCAB_SIZE, HIDDEN_SIZE], 0.35, 0.0, device)?,
        );
        t.insert("model.norm.weight".to_string(), o(&[HIDDEN_SIZE])?);
        t.insert(
            "lm_head.weight".to_string(),
            patterned(&[VOCAB_SIZE, HIDDEN_SIZE], 0.2, 1.0, device)?,
        );

        for i in 0..NUM_LAYERS {
            let l = format!("model.layers.{i}");
            let ph = i as f32 + 0.5;

            t.insert(format!("{l}.input_layernorm.weight"), o(&[HIDDEN_SIZE])?);
            t.insert(
                format!("{l}.post_attention_layernorm.weight"),
                o(&[HIDDEN_SIZE])?,
            );

            // Q LoRA (HF fallback naming).
            t.insert(
                format!("{l}.self_attn.q_a_proj.weight"),
                patterned(&[Q_LORA_RANK, HIDDEN_SIZE], 0.15, ph, device)?,
            );
            t.insert(
                format!("{l}.self_attn.q_a_layernorm.weight"),
                o(&[Q_LORA_RANK])?,
            );
            t.insert(
                format!("{l}.self_attn.q_b_proj.weight"),
                patterned(
                    &[NUM_ATTN_HEADS * HEAD_DIM, Q_LORA_RANK],
                    0.15,
                    ph + 0.1,
                    device,
                )?,
            );

            // Fused wkv (V3-style fallback naming) + kv_norm.
            t.insert(
                format!("{l}.self_attn.kv_a_proj_with_mqa.weight"),
                patterned(&[HEAD_DIM, HIDDEN_SIZE], 0.15, ph + 0.2, device)?,
            );
            t.insert(
                format!("{l}.self_attn.kv_a_layernorm.weight"),
                o(&[HEAD_DIM])?,
            );

            // Grouped o_proj LoRA.
            let wo_a_in = NUM_ATTN_HEADS * HEAD_DIM / O_GROUPS.max(1);
            let o_inner = O_GROUPS * O_LORA_RANK;
            t.insert(
                format!("{l}.self_attn.o_a_proj.weight"),
                patterned(&[o_inner, wo_a_in], 0.15, ph + 0.3, device)?,
            );
            t.insert(
                format!("{l}.self_attn.o_b_proj.weight"),
                patterned(&[HIDDEN_SIZE, o_inner], 0.15, ph + 0.4, device)?,
            );

            // Per-head attn_sink — REQUIRED for finite decode on CPU: the
            // no-sink CPU attention path NaNs on rows with a masked window
            // prefix (see the `standard_decode_window_boundary_exact` note in
            // dsv4_attention.rs); deployment always ships attn_sink.
            t.insert(
                format!("{l}.self_attn.attn_sink"),
                patterned(&[NUM_ATTN_HEADS], 0.4, ph + 0.5, device)?,
            );

            // Real compressor weights on CSA/HCA layers, so the compressed
            // branch runs the actual `forward_from_xs` (softmax-pooled,
            // overlap-windowed for ratio 4) over the xs history.
            let ratio = COMPRESS_RATIOS[i];
            if ratio != 0 {
                let coff = if ratio == 4 { 2 } else { 1 };
                t.insert(
                    format!("{l}.self_attn.compressor.wkv_gate.weight"),
                    patterned(&[2 * coff * HEAD_DIM, HIDDEN_SIZE], 0.12, ph + 0.6, device)?,
                );
                t.insert(
                    format!("{l}.self_attn.compressor.norm.weight"),
                    o(&[HEAD_DIM])?,
                );
                t.insert(
                    format!("{l}.self_attn.compressor.ape"),
                    patterned(&[ratio as usize, coff * HEAD_DIM], 0.05, ph + 0.7, device)?,
                );
            }

            // MoE (hash-routed: tid2eid is required for layers below the
            // default num_hash_layers).
            t.insert(
                format!("{l}.mlp.gate.weight"),
                patterned(&[NUM_ROUTED_EXPERTS, HIDDEN_SIZE], 0.1, ph + 0.8, device)?,
            );
            let tid2eid: Vec<i64> = (0..VOCAB_SIZE * NUM_EXPERTS_PER_TOK)
                .map(|j| (j % NUM_ROUTED_EXPERTS) as i64)
                .collect();
            t.insert(
                format!("{l}.mlp.gate.tid2eid"),
                Tensor::from_vec(tid2eid, &[VOCAB_SIZE, NUM_EXPERTS_PER_TOK], device)?,
            );
            t.insert(
                format!("{l}.mlp.experts.gate_up_proj"),
                patterned(
                    &[NUM_ROUTED_EXPERTS, HIDDEN_SIZE, 2 * MOE_INTERMEDIATE_SIZE],
                    0.08,
                    ph + 0.9,
                    device,
                )?,
            );
            t.insert(
                format!("{l}.mlp.experts.down_proj"),
                patterned(
                    &[NUM_ROUTED_EXPERTS, MOE_INTERMEDIATE_SIZE, HIDDEN_SIZE],
                    0.08,
                    ph + 1.0,
                    device,
                )?,
            );
            let shared_inter = MOE_INTERMEDIATE_SIZE * NUM_SHARED_EXPERTS;
            t.insert(
                format!("{l}.mlp.shared_experts.gate_proj.weight"),
                patterned(&[shared_inter, HIDDEN_SIZE], 0.08, ph + 1.1, device)?,
            );
            t.insert(
                format!("{l}.mlp.shared_experts.up_proj.weight"),
                patterned(&[shared_inter, HIDDEN_SIZE], 0.08, ph + 1.2, device)?,
            );
            t.insert(
                format!("{l}.mlp.shared_experts.down_proj.weight"),
                patterned(&[HIDDEN_SIZE, shared_inter], 0.08, ph + 1.3, device)?,
            );
        }
        Ok(t)
    }

    pub fn config_json() -> String {
        serde_json::json!({
            "architectures": ["DeepseekV4ForCausalLM"],
            "vocab_size": VOCAB_SIZE,
            "hidden_size": HIDDEN_SIZE,
            "head_dim": HEAD_DIM,
            "moe_intermediate_size": MOE_INTERMEDIATE_SIZE,
            "num_hidden_layers": NUM_LAYERS,
            "num_attention_heads": NUM_ATTN_HEADS,
            "num_key_value_heads": NUM_KV_HEADS,
            "n_shared_experts": NUM_SHARED_EXPERTS,
            "n_routed_experts": NUM_ROUTED_EXPERTS,
            "routed_scaling_factor": 1.0,
            "topk_method": "greedy",
            "scoring_func": "softmax",
            "num_experts_per_tok": NUM_EXPERTS_PER_TOK,
            "moe_layer_freq": 1,
            "first_k_dense_replace": 0,
            "hidden_act": "silu",
            "max_position_embeddings": MAX_POSITION_EMBEDDINGS,
            "rms_norm_eps": RMS_NORM_EPS,
            "tie_word_embeddings": false,
            "rope_theta": ROPE_THETA,
            "attention_bias": false,
            "q_lora_rank": Q_LORA_RANK,
            "qk_rope_head_dim": QK_ROPE_HEAD_DIM,
            "n_group": 1,
            "topk_group": 1,
            "compress_ratios": COMPRESS_RATIOS,
            "sliding_window": SLIDING_WINDOW,
            "compress_rope_theta": COMPRESS_ROPE_THETA,
            "o_lora_rank": O_LORA_RANK,
            "o_groups": O_GROUPS,
        })
        .to_string()
    }
}

/// Prompt length: past the 128-token sliding window, and NOT a multiple of
/// the CSA ratio (4) so the ragged `< ratio` tail is exercised too.
const V4C_PREFILL_T: usize = 130;
/// Decode steps after prefill (crosses further ratio-4 block boundaries).
const V4C_DECODE_STEPS: usize = 6;

fn v4c_load() -> Box<dyn NormalModel + Send + Sync> {
    let device = Device::Cpu;
    let tensors =
        v4_compress::weights(&device).expect("V4 compress fixture construction must not fail");
    let vb = wrap_as_vb(tensors, &device);
    let loader = DeepSeekV4Loader;
    loader
        .load(
            &v4_compress::config_json(),
            vb,
            make_metadata(&device),
            AttentionImplementation::Eager,
        )
        .expect(
            "DeepSeekV4Loader::load must succeed on the compress fixture. A \
             missing-tensor error here means the compressor probing contract \
             drifted (see v4_compress::weights).",
        )
}

/// One forward step, returning the LAST position's logits `[B, 1, vocab]`.
fn v4c_step(
    model: &(dyn NormalModel + Send + Sync),
    ids: &Tensor,
    offset: usize,
) -> CandleResult<Tensor> {
    let (b, t) = ids.dims2()?;
    let context_lens = vec![(t - 1, 1); b];
    let position_ids = (offset..offset + t).collect::<Vec<_>>();
    let seqlen_offsets = vec![offset; b];
    model.forward(
        ids,
        &seqlen_offsets,
        context_lens,
        position_ids,
        None,
        &empty_flash_params(),
    )
}

fn v4c_reset(model: &mut Box<dyn NormalModel + Send + Sync>) {
    let mut cache = model.cache_mut().normal();
    cache.0.iter_mut().for_each(|c| c.reset());
}

/// Extract batch row `row` of `[B, 1, vocab]` logits as a flat Vec<f32>.
fn v4c_row(logits: &Tensor, row: usize) -> Vec<f32> {
    logits
        .narrow(0, row, 1)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .to_vec1()
        .unwrap()
}

/// Relative tolerance for comparing a batched run against a batch-of-1 run.
///
/// These two runs are real-number-identical but NOT bit-identical: `MatMul`'s
/// CPU path rounds through F16, and a batch-2 GEMM tiles (and therefore
/// rounds) differently from a batch-1 GEMM — the same documented CPU-MatMul
/// F16 noise floor that `dsv4_attention::standard_decode_window_boundary_exact`
/// budgets for. The fixture's logits land exactly on the F16 grid (e.g.
/// 6.8867188 == 1763 * 2^-8, ulp 2^-8 in [4,8)), and CI (x86 / Windows /
/// macOS-CI) observed a 2-ulp divergence — 0.0078125 abs at |logit| ~4.6,
/// i.e. ~1.7e-3 relative — where this dev host (ARM) shows exactly 0.
///
/// F16's relative ulp is 2^-10 ~ 9.8e-4, so this budget is ~10 ulps: enough
/// headroom for a differently-tiled accumulation on any arch, and still ~2
/// orders of magnitude below a genuine cross-sequence leak (pinned by the
/// negative control in `v4c_assert_chains_differ`, which measures what real
/// contamination looks like — O(1) absolute, ~100x this bound).
///
/// Scaled by the compared magnitude because logits here reach ~7; a fixed
/// absolute bound would be either vacuous for small logits or too tight for
/// large ones.
const V4C_F16_REL_TOL: f32 = 1e-2;

fn v4c_max_abs(v: &[f32]) -> f32 {
    v.iter().fold(0f32, |m, x| m.max(x.abs()))
}

fn v4c_max_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max)
}

/// Tolerance for "same sequence, different batch composition": scaled to the
/// magnitude being compared, floored at 1.0 so near-zero logit rows still get
/// the full F16 budget.
fn v4c_tol(expected: &[f32]) -> f32 {
    V4C_F16_REL_TOL * v4c_max_abs(expected).max(1.0)
}

/// Assert a batched row matches the same sequence's batch-of-1 run within the
/// F16 GEMM-tiling budget. Any real cross-sequence state leak is O(1) and
/// blows through this by ~100x.
fn v4c_assert_close(actual: &[f32], expected: &[f32], what: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{what}: logit length mismatch"
    );
    for (x, y) in actual.iter().zip(expected.iter()) {
        assert!(
            x.is_finite() && y.is_finite(),
            "{what}: non-finite logit ({x} vs {y})"
        );
    }
    let max_diff = v4c_max_diff(actual, expected);
    let tol = v4c_tol(expected);
    assert!(
        max_diff < tol,
        "{what}: batched vs single-sequence logits diverged (max abs diff \
         {max_diff}, tol {tol} = {V4C_F16_REL_TOL} * max(1, |expected|max = \
         {})). That is far above the CPU-MatMul F16 tiling floor, so the \
         xs-history / KV state of one sequence leaked into the other — the \
         voting correctness contract is broken.",
        v4c_max_abs(expected)
    );
}

/// Bit-exact equality, for comparisons where the batch SHAPE is identical and
/// only a sibling sequence's CONTENT differs. GEMM tiling depends on shapes,
/// not values, so a correct implementation is bit-identical here and any
/// difference at all is cross-sequence contamination. This is the
/// arch-independent anti-leak assertion — no rounding budget to hide behind.
fn v4c_assert_bit_identical(actual: &[f32], expected: &[f32], what: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{what}: logit length mismatch"
    );
    let max_diff = v4c_max_diff(actual, expected);
    assert!(
        max_diff == 0.0,
        "{what}: a sequence's logits changed (max abs diff {max_diff}) purely \
         because its BATCH-MATE changed. Batch shape is identical in both \
         runs, so this is not GEMM-tiling rounding — it is one sequence's \
         xs-history / KV state bleeding into another."
    );
}

/// Negative control / teeth: two distinct chains must produce grossly
/// different logits. Establishes the magnitude of a genuine cross-sequence
/// leak so the F16 budget above is provably not wide enough to hide one.
fn v4c_assert_chains_differ(a: &[f32], b: &[f32], what: &str) {
    let signal = v4c_max_diff(a, b);
    let tol = v4c_tol(a);
    assert!(
        signal > 20.0 * tol,
        "{what}: the two chains' logits differ by only {signal}, which is not \
         comfortably above the equality tolerance {tol} — the equality \
         assertions would have no teeth (a leak could pass unnoticed). Make \
         the fixture chains more distinguishable."
    );
}

/// Run one chain solo (batch=1): prefill + decodes. Returns the per-step
/// last-token logits (index 0 = prefill) and the post-prefill cache snapshot
/// `(k, v)` per cache entry (KV layers AND xs-history entries), which the
/// merged-decode test batches the way `NormalCacheManager::clone_in_cache`
/// does.
#[allow(clippy::type_complexity)]
fn v4c_solo_run(
    model: &mut Box<dyn NormalModel + Send + Sync>,
    prompt: &[u32],
    next_toks: &[u32],
) -> (Vec<Vec<f32>>, Vec<(Tensor, Tensor)>) {
    let device = Device::Cpu;
    v4c_reset(model);
    let mut outs = Vec::new();

    let ids = Tensor::from_vec(prompt.to_vec(), (1, prompt.len()), &device).unwrap();
    let logits = v4c_step(model.as_ref(), &ids, 0).expect("solo prefill must not error");
    outs.push(v4c_row(&logits, 0));

    let snapshot: Vec<(Tensor, Tensor)> = model
        .cache()
        .normal()
        .0
        .iter()
        .map(|c| {
            let k = c
                .k()
                .unwrap()
                .expect("cache entry populated after prefill")
                .copy()
                .unwrap();
            let v = c
                .v()
                .unwrap()
                .expect("cache entry populated after prefill")
                .copy()
                .unwrap();
            (k, v)
        })
        .collect();

    for (s, tok) in next_toks.iter().enumerate() {
        let ids = Tensor::from_vec(vec![*tok], (1, 1), &device).unwrap();
        let logits =
            v4c_step(model.as_ref(), &ids, prompt.len() + s).expect("solo decode must not error");
        outs.push(v4c_row(&logits, 0));
    }
    (outs, snapshot)
}

/// A 2-sequence batch through the V4 compressor path (CSA + HCA), crossing
/// the 128-token sliding window, must not panic AND must produce, for every
/// step, per-sequence logits equal to the same sequences run separately.
/// This is the correctness contract for Arc Boost voting (`n_votes`), whose
/// sibling chains run as one batch. Covers both engine batch shapes:
///   1. lockstep — both chains prefill AND decode in one batch;
///   2. merged  — chains prefilled separately (batch=1), their per-sequence
///      caches batched along dim 0 (the `clone_in_cache` dance), then decoded
///      together.
#[test]
fn v4_xs_history_two_seq_batch_matches_single_sequence() {
    let device = Device::Cpu;
    let mut model = v4c_load();

    let t = V4C_PREFILL_T;
    let vocab = v4_compress::VOCAB_SIZE;
    let a_prompt: Vec<u32> = (0..t).map(|i| ((i * 7 + 1) % vocab) as u32).collect();
    let b_prompt: Vec<u32> = (0..t).map(|i| ((i * 13 + 5) % vocab) as u32).collect();
    let a_next: Vec<u32> = (0..V4C_DECODE_STEPS)
        .map(|s| ((s * 11 + 3) % vocab) as u32)
        .collect();
    let b_next: Vec<u32> = (0..V4C_DECODE_STEPS)
        .map(|s| ((s * 17 + 9) % vocab) as u32)
        .collect();

    // ---- Reference: each chain alone (batch=1). Also the bit-identity
    // baseline for the sacred single-sequence path: any regression here
    // fails the equality asserts below symmetrically. ----
    let (a_solo, a_snap) = v4c_solo_run(&mut model, &a_prompt, &a_next);
    let (b_solo, b_snap) = v4c_solo_run(&mut model, &b_prompt, &b_next);

    // Teeth: the chains must be grossly distinguishable at every step, so a
    // genuine leak cannot hide inside the F16 tolerance.
    for s in 0..=V4C_DECODE_STEPS {
        v4c_assert_chains_differ(
            &a_solo[s],
            &b_solo[s],
            &format!("fixture chains at step {s}"),
        );
    }

    // ---- Scenario 1: separate prefills, merged decode batch. This is the
    // engine-real flow (`clone_out_cache` after each solo prefill, then
    // `clone_in_cache` batching per-sequence caches along dim 0 for the
    // joint decode) — the exact composition change the old per-model
    // xs_history could not survive: at this point the shared history buffer
    // held ONLY chain B's batch-1 state, so the batch-2 decode either
    // crashed on the shape mismatch or fed chain A chain B's history. ----
    {
        let mut cache = model.cache_mut().normal();
        assert_eq!(
            a_snap.len(),
            cache.0.len(),
            "snapshot length must match the model cache (KV entries + xs-history entries)"
        );
        for (entry, ((ka, va), (kb, vb))) in
            cache.0.iter_mut().zip(a_snap.iter().zip(b_snap.iter()))
        {
            let k = Tensor::cat(&[ka, kb], 0).unwrap();
            let v = Tensor::cat(&[va, vb], 0).unwrap();
            entry.reset();
            entry
                .append(&k, &v)
                .expect("merging per-sequence caches into a batch must not fail");
        }
    }
    // First half of the decode: both chains in one batch.
    let shrink_at = V4C_DECODE_STEPS / 2;
    for s in 0..shrink_at {
        let ids = Tensor::from_vec(vec![a_next[s], b_next[s]], (2, 1), &device).unwrap();
        let logits = v4c_step(model.as_ref(), &ids, t + s)
            .expect("merged 2-chain decode through the compressor path must not panic");
        v4c_assert_close(
            &v4c_row(&logits, 0),
            &a_solo[s + 1],
            &format!("merged decode step {s} chain A"),
        );
        v4c_assert_close(
            &v4c_row(&logits, 1),
            &b_solo[s + 1],
            &format!("merged decode step {s} chain B"),
        );
    }
    // Batch shrink: chain A finishes (EOS in the voting run); chain B
    // continues alone. Engine-wise: clone_out chunks the batch back per
    // sequence, then clone_in rebuilds a batch of 1 from chain B's slots.
    // The old shared xs_history kept its batch-2 buffer here and crashed on
    // the next batch-1 append.
    {
        let mut cache = model.cache_mut().normal();
        for entry in cache.0.iter_mut() {
            let k = entry
                .k()
                .unwrap()
                .expect("cache entry populated during decode")
                .narrow(0, 1, 1)
                .unwrap()
                .copy()
                .unwrap();
            let v = entry
                .v()
                .unwrap()
                .expect("cache entry populated during decode")
                .narrow(0, 1, 1)
                .unwrap()
                .copy()
                .unwrap();
            entry.reset();
            entry
                .append(&k, &v)
                .expect("splitting chain B back out of the batch must not fail");
        }
    }
    for s in shrink_at..V4C_DECODE_STEPS {
        let ids = Tensor::from_vec(vec![b_next[s]], (1, 1), &device).unwrap();
        let logits = v4c_step(model.as_ref(), &ids, t + s)
            .expect("chain B's post-shrink decode must not panic");
        v4c_assert_close(
            &v4c_row(&logits, 0),
            &b_solo[s + 1],
            &format!("post-shrink decode step {s} chain B"),
        );
    }

    // ---- Scenario 2: lockstep 2-chain batch (prefill + decode), the
    // voting fast path where sibling chains enter one bucket together.
    // Chain A's rows are recorded so Scenario 3 can re-run it against a
    // different batch-mate. ----
    let mut a_with_b: Vec<Vec<f32>> = Vec::new();
    v4c_reset(&mut model);
    let mut both = a_prompt.clone();
    both.extend_from_slice(&b_prompt);
    let ids = Tensor::from_vec(both, (2, t), &device).unwrap();
    let logits = v4c_step(model.as_ref(), &ids, 0)
        .expect("2-chain batched prefill through the compressor path must not panic");
    v4c_assert_close(&v4c_row(&logits, 0), &a_solo[0], "lockstep prefill chain A");
    v4c_assert_close(&v4c_row(&logits, 1), &b_solo[0], "lockstep prefill chain B");
    a_with_b.push(v4c_row(&logits, 0));

    for s in 0..V4C_DECODE_STEPS {
        let ids = Tensor::from_vec(vec![a_next[s], b_next[s]], (2, 1), &device).unwrap();
        let logits = v4c_step(model.as_ref(), &ids, t + s)
            .expect("2-chain batched decode through the compressor path must not panic");
        v4c_assert_close(
            &v4c_row(&logits, 0),
            &a_solo[s + 1],
            &format!("lockstep decode step {s} chain A"),
        );
        v4c_assert_close(
            &v4c_row(&logits, 1),
            &b_solo[s + 1],
            &format!("lockstep decode step {s} chain B"),
        );
        a_with_b.push(v4c_row(&logits, 0));
    }

    // ---- Scenario 3: batch-mate invariance — the strongest, and the only
    // arch-independent, anti-leak assertion here.
    //
    // Re-run the identical lockstep schedule for chain A, but pair it with a
    // DIFFERENT sibling (chain C instead of chain B). Chain A's own tokens,
    // the batch shape, and every GEMM shape are unchanged, so F16 rounding is
    // unchanged and a correct per-sequence implementation is BIT-IDENTICAL for
    // chain A — no rounding budget for a leak to hide in. Under the old shared
    // xs_history this is exactly what broke: chain A's compressed branch read
    // whatever history the sibling had last written.
    //
    // Keeping GEMM shapes fixed takes one deliberate construction. The MoE
    // slow backend (`moe/experts.rs::forward_slow`) flattens the batch and
    // gathers per expert, so the per-expert GEMM row count depends on how ALL
    // sequences in the batch route. Every layer here hash-routes (all 3 layers
    // are < the default `num_hash_layers` = 3), so routing is a pure function
    // of token id: with `tid2eid[j] = j % 4` over `[vocab, top_k=2]`, token
    // `x` selects experts `[(2x)%4, (2x+1)%4]` — i.e. PARITY of the token id
    // picks the expert pair. Deriving C from B by `+2 (mod even vocab)`
    // preserves every token's parity, so C routes identically to B, the
    // per-expert row counts are unchanged, and only the sibling's VALUES
    // differ. (Per-row dot products are independent, so differing sibling
    // values cannot perturb chain A's rows.)
    let c_prompt: Vec<u32> = b_prompt.iter().map(|x| (x + 2) % vocab as u32).collect();
    let c_next: Vec<u32> = b_next.iter().map(|x| (x + 2) % vocab as u32).collect();
    assert_eq!(vocab % 2, 0, "parity-preserving +2 requires an even vocab");
    for (bt, ct) in b_prompt.iter().zip(c_prompt.iter()) {
        assert_eq!(
            bt % 2,
            ct % 2,
            "chain C must preserve chain B's token parity so hash routing — \
             and therefore every MoE GEMM shape — is identical"
        );
    }
    // Chain C must still be a genuinely different neighbour, else the
    // invariance check is vacuous.
    let (c_solo, _c_snap) = v4c_solo_run(&mut model, &c_prompt, &c_next);
    v4c_assert_chains_differ(&b_solo[0], &c_solo[0], "sibling chains B vs C");

    v4c_reset(&mut model);
    let mut both = a_prompt.clone();
    both.extend_from_slice(&c_prompt);
    let ids = Tensor::from_vec(both, (2, t), &device).unwrap();
    let logits = v4c_step(model.as_ref(), &ids, 0).expect("A+C batched prefill must not panic");
    v4c_assert_bit_identical(
        &v4c_row(&logits, 0),
        &a_with_b[0],
        "batch-mate invariance, prefill chain A (sibling B -> C)",
    );

    for s in 0..V4C_DECODE_STEPS {
        let ids = Tensor::from_vec(vec![a_next[s], c_next[s]], (2, 1), &device).unwrap();
        let logits =
            v4c_step(model.as_ref(), &ids, t + s).expect("A+C batched decode must not panic");
        v4c_assert_bit_identical(
            &v4c_row(&logits, 0),
            &a_with_b[s + 1],
            &format!("batch-mate invariance, decode step {s} chain A (sibling B -> C)"),
        );
    }
}

// ===========================================================================
// V4 RAGGED-BATCH PREFILL (multi-sequence serving correctness)
// ===========================================================================
//
// The fleet thesis is many sequences per GPU, so "one batch == N solo runs"
// is the load-bearing contract. Two defects broke it on every live V4 path
// (reference audit §1(e)):
//
//   1. `b_sz > 1` prefill routed to `sinks_attn_varlen`, a backend with no
//      mask parameter whose CPU fallback attended with `mask = None` — i.e.
//      no causality at all. `cumulative_seqlens_k` is populated on every
//      flash-attn build (the documented production build), so this fired for
//      every real batched prompt.
//   2. `dsv4_attention` read the caller's `attention_mask` on exactly one
//      env-gated branch and discarded it everywhere else, so padded columns
//      voted in their neighbours' softmax.
//
// This test is the engine-level counterpart of the unit tests in
// `dsv4_attention.rs`: three prompts of DIFFERENT lengths, right-padded into
// one batch exactly the way `make_prompt_chunk` does, with the same
// padded-length `cu_seqlens` the input processor emits.

/// Cumulative-seqlen `FlashParams` for the CPU device, mirroring
/// `inputs_processor.rs`: lengths are the PADDED per-sequence lengths, and the
/// maps are keyed by device location. Their mere presence is what used to
/// divert a batched prefill onto the mask-free varlen backend.
fn v4c_varlen_flash_params(batch: usize, padded_len: usize, device: &Device) -> FlashParams {
    let cu: Vec<u32> = (0..=batch).map(|i| (i * padded_len) as u32).collect();
    let cu = Tensor::from_vec(cu, batch + 1, device).unwrap();
    let mut q_map = HashMap::new();
    let mut k_map = HashMap::new();
    q_map.insert(device.location(), cu.clone());
    k_map.insert(device.location(), cu);
    FlashParams {
        max_q: padded_len as u32,
        max_k: padded_len as u32,
        cumulative_seqlens_q: q_map,
        cumulative_seqlens_k: k_map,
        causal: true,
    }
}

/// Three prompts of DIFFERENT lengths, prefilled as one right-padded batch,
/// must each yield the logits of their own last REAL token — the same logits
/// the sequence produces when prefilled alone.
///
/// Pre-fix this does not merely drift: the batched prefill fails inside
/// `sinks_attn_varlen`, which reads the `[B, 1, T, D]` K/V as if it were a
/// packed `[total_kv, kv_H, D]` buffer.
#[test]
fn v4_ragged_batch_prefill_matches_solo() {
    let device = Device::Cpu;
    let mut model = v4c_load();
    let vocab = v4_compress::VOCAB_SIZE;

    // Lengths straddle the 128 sliding window and are deliberately not
    // multiples of the CSA ratio (4), so the ragged compressor tail is live
    // and the batch's compressed axis is wider than the short sequences'.
    let lens = [V4C_PREFILL_T, 97, 61];
    let t_max = lens[0];
    let prompts: Vec<Vec<u32>> = lens
        .iter()
        .enumerate()
        .map(|(s, &len)| {
            (0..len)
                .map(|i| ((i * (7 + 6 * s) + 1 + 4 * s) % vocab) as u32)
                .collect()
        })
        .collect();

    // ---- Reference: each prompt alone. ----
    let solo: Vec<Vec<f32>> = prompts
        .iter()
        .map(|p| {
            v4c_reset(&mut model);
            let ids = Tensor::from_vec(p.clone(), (1, p.len()), &device).unwrap();
            let logits = v4c_step(model.as_ref(), &ids, 0).expect("solo prefill must not error");
            v4c_row(&logits, 0)
        })
        .collect();

    // Teeth: the prompts must be grossly distinguishable, so per-sequence
    // agreement below cannot be satisfied by an implementation that mixes
    // them.
    v4c_assert_chains_differ(&solo[0], &solo[1], "ragged fixture prompts 0 vs 1");
    v4c_assert_chains_differ(&solo[1], &solo[2], "ragged fixture prompts 1 vs 2");

    // ---- Batched: right-pad to the batch max, exactly like the engine. ----
    let mut flat: Vec<u32> = Vec::with_capacity(lens.len() * t_max);
    for p in &prompts {
        flat.extend_from_slice(p);
        flat.extend(std::iter::repeat_n(0u32, t_max - p.len()));
    }
    let ids = Tensor::from_vec(flat, (lens.len(), t_max), &device).unwrap();
    // Each row's logits come from its own last REAL token, not the padding.
    let context_lens: Vec<(usize, usize)> = lens.iter().map(|&len| (len - 1, 1)).collect();
    let flash = v4c_varlen_flash_params(lens.len(), t_max, &device);

    v4c_reset(&mut model);
    let logits = model
        .forward(
            &ids,
            &vec![0usize; lens.len()],
            context_lens,
            (0..t_max).collect(),
            None,
            &flash,
        )
        .expect(
            "ragged batched prefill must not error. A shape error from the varlen sinks \
             backend here means `sinks_attn` is still routing a masked V4 call onto the \
             mask-free packed-KV path.",
        );

    for (i, &len) in lens.iter().enumerate() {
        v4c_assert_close(
            &v4c_row(&logits, i),
            &solo[i],
            &format!("ragged batch prefill, sequence {i} (len {len})"),
        );
    }
}

// ===========================================================================
// KIMI K2.5 / K2.6 SYNTHETIC SMOKE
// ===========================================================================
//
// Per SGLang's `kimi_k25.py` (line 39: `from sglang.srt.models.deepseek_v2
// import DeepseekV3ForCausalLM`), the K2.x text architecture IS DeepSeek V3.
// The K2 dispatcher routes to `DeepSeekV3Loader`. This test exercises that
// exact path with synthetic V3-MLA-shaped tensors using K2-style config
// scale (small vocab + n_routed_experts so the test stays under a second).

mod k2 {
    use super::*;

    pub const NUM_LAYERS: usize = 2;
    pub const HIDDEN_SIZE: usize = 64;
    pub const INTERMEDIATE_SIZE: usize = 64;
    pub const MOE_INTERMEDIATE_SIZE: usize = 32;
    pub const NUM_ATTN_HEADS: usize = 2;
    pub const NUM_ROUTED_EXPERTS: usize = 4;
    pub const NUM_SHARED_EXPERTS: usize = 1;
    pub const NUM_EXPERTS_PER_TOK: usize = 2;
    pub const QK_NOPE_HEAD_DIM: usize = 32;
    pub const QK_ROPE_HEAD_DIM: usize = 8;
    // CPU flash-attn backend (`attention/backends/cpu.rs:603`) uses Q's
    // last dim as the output dim, so on CPU we need `v_head_dim ==
    // q_head_dim` to avoid an out-of-bounds V read. Real Kimi K2.5/K2.6
    // configs have `v_head_dim != q_head_dim` (128 vs 192); production
    // CUDA paths handle that via `mla_decode_forward` / `mla_cache_forward`
    // (`mla/forward.rs`). For the CPU smoke test, we align them.
    pub const V_HEAD_DIM: usize = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM;
    pub const KV_LORA_RANK: usize = 16;
    pub const Q_LORA_RANK: usize = 16;
    pub const VOCAB_SIZE: usize = 128;
    pub const MAX_POSITION_EMBEDDINGS: usize = 64;
    pub const RMS_NORM_EPS: f64 = 1e-6;
    pub const ROPE_THETA: f32 = 10000.0;

    pub fn q_head_dim() -> usize {
        QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM
    }

    /// HF-format V3 tensor layout (which is also Kimi K2.5/K2.6 layout per
    /// SGLang). Shapes derived from
    /// `mistralrs-core/src/models/deepseek3.rs::Attention::new` and `Moe::new`.
    pub fn synthetic_k2_weights(device: &Device) -> CandleResult<HashMap<String, Tensor>> {
        let mut t = HashMap::new();
        let z = |s: &[usize]| zeros(s, device);
        let o = |s: &[usize]| ones(s, device);

        // globals
        t.insert(
            "model.embed_tokens.weight".to_string(),
            o(&[VOCAB_SIZE, HIDDEN_SIZE])?,
        );
        t.insert("model.norm.weight".to_string(), o(&[HIDDEN_SIZE])?);
        t.insert("lm_head.weight".to_string(), z(&[VOCAB_SIZE, HIDDEN_SIZE])?);

        // First layer is dense (first_k_dense_replace=1); remaining are MoE.
        // This mirrors Kimi K2.5 / K2.6 published configs.
        const FIRST_K_DENSE_REPLACE: usize = 1;

        for i in 0..NUM_LAYERS {
            let l = format!("model.layers.{i}");
            t.insert(format!("{l}.input_layernorm.weight"), o(&[HIDDEN_SIZE])?);
            t.insert(
                format!("{l}.post_attention_layernorm.weight"),
                o(&[HIDDEN_SIZE])?,
            );

            // Q LoRA
            t.insert(
                format!("{l}.self_attn.q_a_proj.weight"),
                z(&[Q_LORA_RANK, HIDDEN_SIZE])?,
            );
            t.insert(
                format!("{l}.self_attn.q_a_layernorm.weight"),
                o(&[Q_LORA_RANK])?,
            );
            t.insert(
                format!("{l}.self_attn.q_b_proj.weight"),
                z(&[NUM_ATTN_HEADS * q_head_dim(), Q_LORA_RANK])?,
            );

            // V3-style MLA KV: kv_a_proj_with_mqa (hidden → kv_lora + qk_rope),
            // kv_a_layernorm (kv_lora), kv_b_proj (kv_lora → heads * (qk_nope + v_head)).
            t.insert(
                format!("{l}.self_attn.kv_a_proj_with_mqa.weight"),
                z(&[KV_LORA_RANK + QK_ROPE_HEAD_DIM, HIDDEN_SIZE])?,
            );
            t.insert(
                format!("{l}.self_attn.kv_a_layernorm.weight"),
                o(&[KV_LORA_RANK])?,
            );
            t.insert(
                format!("{l}.self_attn.kv_b_proj.weight"),
                z(&[
                    NUM_ATTN_HEADS * (QK_NOPE_HEAD_DIM + V_HEAD_DIM),
                    KV_LORA_RANK,
                ])?,
            );
            // V3 o_proj: standard (heads*v_head → hidden).
            t.insert(
                format!("{l}.self_attn.o_proj.weight"),
                z(&[HIDDEN_SIZE, NUM_ATTN_HEADS * V_HEAD_DIM])?,
            );

            if i < FIRST_K_DENSE_REPLACE {
                // dense MLP (layer 0)
                t.insert(
                    format!("{l}.mlp.gate_proj.weight"),
                    z(&[INTERMEDIATE_SIZE, HIDDEN_SIZE])?,
                );
                t.insert(
                    format!("{l}.mlp.up_proj.weight"),
                    z(&[INTERMEDIATE_SIZE, HIDDEN_SIZE])?,
                );
                t.insert(
                    format!("{l}.mlp.down_proj.weight"),
                    z(&[HIDDEN_SIZE, INTERMEDIATE_SIZE])?,
                );
            } else {
                t.insert(
                    format!("{l}.mlp.gate.weight"),
                    z(&[NUM_ROUTED_EXPERTS, HIDDEN_SIZE])?,
                );
                // sigmoid scoring + noaux_tc routing (K2's published config) →
                // need `gate.e_score_correction_bias`.
                t.insert(
                    format!("{l}.mlp.gate.e_score_correction_bias"),
                    zeros(&[NUM_ROUTED_EXPERTS], device)?,
                );
                t.insert(
                    format!("{l}.mlp.experts.gate_up_proj"),
                    z(&[NUM_ROUTED_EXPERTS, HIDDEN_SIZE, 2 * MOE_INTERMEDIATE_SIZE])?,
                );
                t.insert(
                    format!("{l}.mlp.experts.down_proj"),
                    z(&[NUM_ROUTED_EXPERTS, MOE_INTERMEDIATE_SIZE, HIDDEN_SIZE])?,
                );
                let shared_inter = MOE_INTERMEDIATE_SIZE * NUM_SHARED_EXPERTS;
                t.insert(
                    format!("{l}.mlp.shared_experts.gate_proj.weight"),
                    z(&[shared_inter, HIDDEN_SIZE])?,
                );
                t.insert(
                    format!("{l}.mlp.shared_experts.up_proj.weight"),
                    z(&[shared_inter, HIDDEN_SIZE])?,
                );
                t.insert(
                    format!("{l}.mlp.shared_experts.down_proj.weight"),
                    z(&[HIDDEN_SIZE, shared_inter])?,
                );
            }
        }
        Ok(t)
    }

    /// K2-style config with `topk_method=noaux_tc` and `scoring_func=sigmoid`,
    /// matching the real Kimi K2.5/K2.6 `config.json` published at HF.
    pub fn config_json() -> String {
        serde_json::json!({
            "architectures": ["KimiK25ForConditionalGeneration"],
            "vocab_size": VOCAB_SIZE,
            "hidden_size": HIDDEN_SIZE,
            "intermediate_size": INTERMEDIATE_SIZE,
            "moe_intermediate_size": MOE_INTERMEDIATE_SIZE,
            "num_hidden_layers": NUM_LAYERS,
            "num_attention_heads": NUM_ATTN_HEADS,
            "num_key_value_heads": 1,
            "n_shared_experts": NUM_SHARED_EXPERTS,
            "n_routed_experts": NUM_ROUTED_EXPERTS,
            "num_experts_per_tok": NUM_EXPERTS_PER_TOK,
            "moe_layer_freq": 1,
            "first_k_dense_replace": 1,
            "max_position_embeddings": MAX_POSITION_EMBEDDINGS,
            "rms_norm_eps": RMS_NORM_EPS,
            "rope_theta": ROPE_THETA,
            "attention_bias": false,
            "q_lora_rank": Q_LORA_RANK,
            "qk_nope_head_dim": QK_NOPE_HEAD_DIM,
            "qk_rope_head_dim": QK_ROPE_HEAD_DIM,
            "kv_lora_rank": KV_LORA_RANK,
            "v_head_dim": V_HEAD_DIM,
            "n_group": 1,
            "topk_group": 1,
            "topk_method": "noaux_tc",
            "scoring_func": "sigmoid",
            "routed_scaling_factor": 1.0,
            "hidden_act": "silu",
            "tie_word_embeddings": false,
        })
        .to_string()
    }
}

#[test]
fn kimi_k2_synthetic_load_smoke() {
    let device = Device::Cpu;

    // Dispatcher recognises every K2 family variant.
    for arch in &[
        "KimiK25ForConditionalGeneration",
        "KimiK2ForCausalLM",
        "KimiK26ForCausalLM",
        "KimiK26VLForConditionalGeneration",
    ] {
        assert!(
            matches!(
                NormalLoaderType::from_causal_lm_name(arch).unwrap(),
                NormalLoaderType::KimiK2
            ),
            "K2 arch {arch} must dispatch to NormalLoaderType::KimiK2"
        );
    }

    let tensors = k2::synthetic_k2_weights(&device).expect("K2 synthetic tensors must build");
    let vb = wrap_as_vb(tensors, &device);
    let config = k2::config_json();

    // Per `pipeline/normal.rs:244`, `NormalLoaderType::KimiK2` → DeepSeekV3Loader.
    // Going through the V3 loader directly exercises the exact path the
    // production dispatcher would take.
    let loader = DeepSeekV3Loader;
    let model = loader
        .load(
            &config,
            vb,
            make_metadata(&device),
            AttentionImplementation::Eager,
        )
        .expect(
            "DeepSeekV3Loader::load must succeed on K2-shaped synthetic \
             weights. The K2 architecture IS V3 per SGLang's kimi_k25.py — \
             if this panics, V3 has drifted from its real-K2 expectation.",
        );

    let input_ids = Tensor::from_vec(vec![0u32, 1], &[1usize, 2], &device).unwrap();
    let logits = run_forward_smoke(model.as_ref(), &input_ids)
        .expect("K2 (V3-backed) forward on synthetic weights must not error");

    assert_eq!(
        logits.dims(),
        &[1usize, 2, k2::VOCAB_SIZE],
        "K2 logits shape mismatch — V3 lm_head contract drift?",
    );
    assert_finite(&logits, "Kimi K2 (V3-backed)");
}

// ===========================================================================
// GLM-4.5 / GLM-5 (DSA) SYNTHETIC SMOKE
// ===========================================================================
//
// GLM-5 DSA routes to `GLM4MoeLoader` per `pipeline/normal.rs:245` — the
// production Tier-A path. The DSA-specific attention isn't ported yet (would
// add a `Glm5Loader`); this test exercises the GLM4Moe layer that GLM-5
// checkpoints actually load through. When/if GLM-5 grows its own loader,
// the test will catch the divergence.

mod glm5 {
    use super::*;

    pub const NUM_LAYERS: usize = 2;
    pub const HIDDEN_SIZE: usize = 64;
    pub const INTERMEDIATE_SIZE: usize = 64;
    pub const MOE_INTERMEDIATE_SIZE: usize = 32;
    pub const NUM_ATTN_HEADS: usize = 2;
    pub const NUM_KV_HEADS: usize = 2;
    pub const HEAD_DIM: usize = 32;
    pub const NUM_ROUTED_EXPERTS: usize = 4;
    pub const NUM_SHARED_EXPERTS: usize = 1;
    pub const NUM_EXPERTS_PER_TOK: usize = 2;
    pub const VOCAB_SIZE: usize = 128;
    pub const MAX_POSITION_EMBEDDINGS: usize = 64;
    pub const RMS_NORM_EPS: f64 = 1e-6;
    pub const ROPE_THETA: f64 = 10000.0;
    pub const PARTIAL_ROTARY_FACTOR: f32 = 0.5;

    /// HF-format Glm4Moe tensor layout, derived from
    /// `mistralrs-core/src/models/glm4_moe.rs`:
    /// - standard `q_proj/k_proj/v_proj/o_proj` attention
    /// - sigmoid-gated MoE with `e_score_correction_bias`
    /// - dense MLP for layers < first_k_dense_replace
    pub fn synthetic_glm5_weights(device: &Device) -> CandleResult<HashMap<String, Tensor>> {
        let mut t = HashMap::new();
        let z = |s: &[usize]| zeros(s, device);
        let o = |s: &[usize]| ones(s, device);

        t.insert(
            "model.embed_tokens.weight".to_string(),
            o(&[VOCAB_SIZE, HIDDEN_SIZE])?,
        );
        t.insert("model.norm.weight".to_string(), o(&[HIDDEN_SIZE])?);
        t.insert("lm_head.weight".to_string(), z(&[VOCAB_SIZE, HIDDEN_SIZE])?);

        const FIRST_K_DENSE_REPLACE: usize = 1;

        for i in 0..NUM_LAYERS {
            let l = format!("model.layers.{i}");
            t.insert(format!("{l}.input_layernorm.weight"), o(&[HIDDEN_SIZE])?);
            t.insert(
                format!("{l}.post_attention_layernorm.weight"),
                o(&[HIDDEN_SIZE])?,
            );

            // Standard attention (Glm4Moe uses straight q/k/v/o, no MLA, no LoRA).
            t.insert(
                format!("{l}.self_attn.q_proj.weight"),
                z(&[NUM_ATTN_HEADS * HEAD_DIM, HIDDEN_SIZE])?,
            );
            t.insert(
                format!("{l}.self_attn.k_proj.weight"),
                z(&[NUM_KV_HEADS * HEAD_DIM, HIDDEN_SIZE])?,
            );
            t.insert(
                format!("{l}.self_attn.v_proj.weight"),
                z(&[NUM_KV_HEADS * HEAD_DIM, HIDDEN_SIZE])?,
            );
            t.insert(
                format!("{l}.self_attn.o_proj.weight"),
                z(&[HIDDEN_SIZE, NUM_ATTN_HEADS * HEAD_DIM])?,
            );

            if i < FIRST_K_DENSE_REPLACE {
                // dense MLP layer 0
                t.insert(
                    format!("{l}.mlp.gate_proj.weight"),
                    z(&[INTERMEDIATE_SIZE, HIDDEN_SIZE])?,
                );
                t.insert(
                    format!("{l}.mlp.up_proj.weight"),
                    z(&[INTERMEDIATE_SIZE, HIDDEN_SIZE])?,
                );
                t.insert(
                    format!("{l}.mlp.down_proj.weight"),
                    z(&[HIDDEN_SIZE, INTERMEDIATE_SIZE])?,
                );
            } else {
                // MoE layer. `glm4_moe.rs:MoeGate::new` always loads
                // `e_score_correction_bias` (no topk_method gate, unlike V3).
                t.insert(
                    format!("{l}.mlp.gate.weight"),
                    z(&[NUM_ROUTED_EXPERTS, HIDDEN_SIZE])?,
                );
                t.insert(
                    format!("{l}.mlp.gate.e_score_correction_bias"),
                    zeros(&[NUM_ROUTED_EXPERTS], device)?,
                );
                t.insert(
                    format!("{l}.mlp.experts.gate_up_proj"),
                    z(&[NUM_ROUTED_EXPERTS, HIDDEN_SIZE, 2 * MOE_INTERMEDIATE_SIZE])?,
                );
                t.insert(
                    format!("{l}.mlp.experts.down_proj"),
                    z(&[NUM_ROUTED_EXPERTS, MOE_INTERMEDIATE_SIZE, HIDDEN_SIZE])?,
                );
                // Shared experts use moe_intermediate_size (NOT shared_inter:
                // Glm4Moe's shared_experts passes `Some(cfg.moe_intermediate_size)`
                // for intermediate, not multiplied by n_shared_experts).
                t.insert(
                    format!("{l}.mlp.shared_experts.gate_proj.weight"),
                    z(&[MOE_INTERMEDIATE_SIZE, HIDDEN_SIZE])?,
                );
                t.insert(
                    format!("{l}.mlp.shared_experts.up_proj.weight"),
                    z(&[MOE_INTERMEDIATE_SIZE, HIDDEN_SIZE])?,
                );
                t.insert(
                    format!("{l}.mlp.shared_experts.down_proj.weight"),
                    z(&[HIDDEN_SIZE, MOE_INTERMEDIATE_SIZE])?,
                );
            }
        }
        Ok(t)
    }

    pub fn config_json() -> String {
        // Real GLM-5 DSA config has `topk_method`/`scoring_func`/`n_group`
        // (V3-shaped) BUT it dispatches to GLM4MoeLoader. We use a GLM4Moe-
        // shaped config (no topk_method/n_group/etc — Glm4MoeConfig is
        // simpler) to drive the Tier-A path the production dispatcher takes.
        serde_json::json!({
            "architectures": ["GlmMoeDsaForCausalLM"],
            "vocab_size": VOCAB_SIZE,
            "hidden_size": HIDDEN_SIZE,
            "intermediate_size": INTERMEDIATE_SIZE,
            "moe_intermediate_size": MOE_INTERMEDIATE_SIZE,
            "num_hidden_layers": NUM_LAYERS,
            "num_attention_heads": NUM_ATTN_HEADS,
            "num_key_value_heads": NUM_KV_HEADS,
            "partial_rotary_factor": PARTIAL_ROTARY_FACTOR,
            "n_routed_experts": NUM_ROUTED_EXPERTS,
            "n_shared_experts": NUM_SHARED_EXPERTS,
            "num_experts_per_tok": NUM_EXPERTS_PER_TOK,
            "first_k_dense_replace": 1,
            "routed_scaling_factor": 1.0,
            "n_group": 1,
            "topk_group": 1,
            "norm_topk_prob": true,
            "rms_norm_eps": RMS_NORM_EPS,
            "rope_theta": ROPE_THETA,
            "max_position_embeddings": MAX_POSITION_EMBEDDINGS,
            "hidden_act": "silu",
            "tie_word_embeddings": false,
            "head_dim": HEAD_DIM,
        })
        .to_string()
    }
}

#[test]
fn glm5_synthetic_load_smoke() {
    let device = Device::Cpu;

    // Dispatcher routes GLM-5 DSA to GLM4MoeLoader (Tier-A path).
    assert!(matches!(
        NormalLoaderType::from_causal_lm_name("GlmMoeDsaForCausalLM").unwrap(),
        NormalLoaderType::GLM5MoeDsa
    ));
    // GLM-4.5 (the non-DSA variant) is also covered: it dispatches via
    // `Glm4MoeForCausalLM` to the same GLM4MoeLoader.
    assert!(matches!(
        NormalLoaderType::from_causal_lm_name("Glm4MoeForCausalLM").unwrap(),
        NormalLoaderType::GLM4Moe
    ));

    let tensors =
        glm5::synthetic_glm5_weights(&device).expect("GLM-5 synthetic tensors must build");
    let vb = wrap_as_vb(tensors, &device);
    let config = glm5::config_json();

    let loader = GLM4MoeLoader;
    let model = loader
        .load(
            &config,
            vb,
            make_metadata(&device),
            AttentionImplementation::Eager,
        )
        .expect(
            "GLM4MoeLoader::load must succeed on GLM-5 / GLM-4.5 synthetic \
             weights. If this panics, the missing tensor in the error tells \
             you where the GLM4Moe loader has drifted vs glm4_moe.rs.",
        );

    let input_ids = Tensor::from_vec(vec![0u32, 1], &[1usize, 2], &device).unwrap();
    let logits = run_forward_smoke(model.as_ref(), &input_ids)
        .expect("GLM-5 forward on synthetic weights must not error");

    assert_eq!(
        logits.dims(),
        &[1usize, 2, glm5::VOCAB_SIZE],
        "GLM-5 logits shape mismatch — Glm4Moe lm_head contract drift?",
    );
    assert_finite(&logits, "GLM-5 / GLM-4.5 (GLM4Moe-backed)");
}
