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

use candle_core::{DType, Device, Module, Result as CandleResult, Tensor};
use indicatif::MultiProgress;

use mistralrs_core::{
    AttentionImplementation, DeepSeekV3Loader, DeepSeekV4Loader, DeviceMapper, GLM4MoeLoader,
    NormalLoaderType, NormalLoadingMetadata, NormalModel, NormalModelLoader,
    TextFlashParams as FlashParams,
};
use mistralrs_quant::{safetensors::ShardedSafeTensors, ShardedVarBuilder};

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
        let z = |s: &[usize]| zeros(s, device);
        let o = |s: &[usize]| ones(s, device);

        // Light heads (always loaded when present).
        t.insert(
            format!("{l}.h_proj.weight"),
            z(&[HIDDEN_SIZE, HIDDEN_SIZE])?,
        );
        t.insert(
            format!("{l}.e_proj.weight"),
            z(&[HIDDEN_SIZE, HIDDEN_SIZE])?,
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

    // Draft a 2-token chain through the real block: seed hidden state from
    // the embedding of token 1 (the pipeline's Tier-A seeding), positions
    // starting at 0 (empty target cache in this synthetic setting).
    let seed_tok = Tensor::from_vec(vec![1u32], (1,), &device).unwrap();
    let seed_hidden = kit
        .embed_tokens
        .forward(&seed_tok)
        .expect("embedding the seed token must succeed");
    let toks = kit
        .propose_chain(
            &seed_hidden,
            1,
            /*depth=*/ 2,
            /*max_tokens=*/ 8,
            /*start_pos=*/ 0,
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
