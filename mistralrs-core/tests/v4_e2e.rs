//! V4 end-to-end smoke test.
//!
//! Builds a tiny synthetic DeepSeek V4 model (4 layers, 4 routed experts + 1
//! shared, MLA-LoRA Q+KV, V4 grouped-LoRA o_proj, alternating
//! compress_ratios=[0,4,128,0]) entirely in CPU memory, then runs `forward`
//! end-to-end and asserts:
//!   - output is finite (no NaN / no Inf)
//!   - re-running with the same input is deterministic
//!   - output shape matches `[batch, seq, vocab]` for the prefill context_lens
//!
//! Rationale: this is the contract the Tier-A V4 loader must honor before any
//! B200 rental cycle. It does NOT require real DeepSeek V4 checkpoints — it
//! exercises the *load path* (HF-format tensor name probing in
//! `mistralrs-core/src/models/deepseek4.rs`) and the *forward path* (MLA, MoE
//! gating, compressed-KV blending) against synthetic zeros.
//!
//! If this test fails to compile, the V4 Config / DeepSeekV4::new signature
//! has drifted. If it compiles but panics, the load path can't find a tensor
//! it expects — note the missing key from the error and fix it in deepseek4.rs.
//!
//! Per the task spec the test runs CPU-only and only uses default features.

use std::collections::HashMap;
use std::sync::Arc;

use candle_core::{DType, Device, Result as CandleResult, Tensor};
use indicatif::MultiProgress;

use mistralrs_core::{
    device_map::DummyDeviceMapper,
    paged_attention::AttentionImplementation,
    pipeline::{
        loaders::DeepSeekV4Loader, text_models_inputs_processor::FlashParams, NormalLoadingMetadata,
        NormalModelLoader,
    },
};
use mistralrs_quant::{safetensors::ShardedSafeTensors, ShardedVarBuilder};

// ---------------------------------------------------------------------------
// Model dimensions for the synthetic test.
//
// IMPORTANT: the chosen dims (qk_nope_head_dim=32, qk_rope_head_dim=8) are
// the *minimum* that exercises the V3 RoPE machinery without tripping the
// fixed-divisibility checks inside `DeepSeekV2RotaryEmbedding`. If you bump
// these, also update the synthesized weight shapes below.
// ---------------------------------------------------------------------------

const NUM_LAYERS: usize = 4;
const HIDDEN_SIZE: usize = 256;
const INTERMEDIATE_SIZE: usize = 256; // dense MLP intermediate (layer 0)
const MOE_INTERMEDIATE_SIZE: usize = 64;
const NUM_ATTN_HEADS: usize = 4;
const NUM_ROUTED_EXPERTS: usize = 4;
const NUM_SHARED_EXPERTS: usize = 1;
const NUM_EXPERTS_PER_TOK: usize = 2;
const QK_NOPE_HEAD_DIM: usize = 32;
const QK_ROPE_HEAD_DIM: usize = 8;
const V_HEAD_DIM: usize = 64;
const KV_LORA_RANK: usize = 32;
const Q_LORA_RANK: usize = 32; // V4 uses LoRA Q
const O_LORA_RANK: usize = 32; // V4 grouped-LoRA o_proj
const O_GROUPS: usize = 1;
const SLIDING_WINDOW: usize = 16;
const FIRST_K_DENSE_REPLACE: usize = 1;
const VOCAB_SIZE: usize = 128;
const RMS_NORM_EPS: f64 = 1e-6;
const ROPE_THETA: f32 = 10000.0;
const MAX_POSITION_EMBEDDINGS: usize = 256;

fn q_head_dim() -> usize {
    QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM // 40
}

// ---------------------------------------------------------------------------
// Synthetic weight builder.
//
// All tensors are stored as F32 zeros. That's sufficient for "did the forward
// path emit finite outputs and stay deterministic" — which is the contract
// this test enforces. Real V4 inference quality is a *separate* property and
// requires real checkpoints + a B200.
// ---------------------------------------------------------------------------

fn zeros(shape: &[usize], device: &Device) -> CandleResult<Tensor> {
    Tensor::zeros(shape, DType::F32, device)
}

fn ones(shape: &[usize], device: &Device) -> CandleResult<Tensor> {
    Tensor::ones(shape, DType::F32, device)
}

fn build_synthetic_v4_weights(device: &Device) -> CandleResult<HashMap<String, Tensor>> {
    let mut tensors = HashMap::new();

    // ---------- Globals (HF format) ----------
    // model.embed_tokens.weight: (vocab, hidden) — zeros are degenerate but the
    // norm + non-zero RmsNorm gain after layer 0 will keep activations non-NaN.
    tensors.insert(
        "model.embed_tokens.weight".to_string(),
        ones(&[VOCAB_SIZE, HIDDEN_SIZE], device)?,
    );
    // model.norm.weight: (hidden,) — RmsNorm gain. Use ones so the norm
    // multiplies by 1 (otherwise we get all-zero hidden states).
    tensors.insert(
        "model.norm.weight".to_string(),
        ones(&[HIDDEN_SIZE], device)?,
    );
    // lm_head.weight: (vocab, hidden) — top of LMHead.
    tensors.insert(
        "lm_head.weight".to_string(),
        zeros(&[VOCAB_SIZE, HIDDEN_SIZE], device)?,
    );

    // ---------- Per-layer tensors ----------
    for i in 0..NUM_LAYERS {
        let l = format!("model.layers.{i}");

        // RmsNorm weights (use ones so they don't zero out the residual stream).
        tensors.insert(
            format!("{l}.input_layernorm.weight"),
            ones(&[HIDDEN_SIZE], device)?,
        );
        tensors.insert(
            format!("{l}.post_attention_layernorm.weight"),
            ones(&[HIDDEN_SIZE], device)?,
        );

        // ---- Attention block ----
        // V4 with q_lora_rank: q_a_proj + q_a_layernorm + q_b_proj.
        // (deepseek4.rs probes `q_norm` first, falls back to `q_a_layernorm`;
        // we publish the V3-style name so the loader takes the fallback path.)
        tensors.insert(
            format!("{l}.self_attn.q_a_proj.weight"),
            zeros(&[Q_LORA_RANK, HIDDEN_SIZE], device)?,
        );
        tensors.insert(
            format!("{l}.self_attn.q_a_layernorm.weight"),
            ones(&[Q_LORA_RANK], device)?,
        );
        tensors.insert(
            format!("{l}.self_attn.q_b_proj.weight"),
            zeros(
                &[NUM_ATTN_HEADS * q_head_dim(), Q_LORA_RANK],
                device,
            )?,
        );

        // V3-style fallback for the LoRA-A KV projection. deepseek4.rs probes
        // `kv_proj` first (V4 HF native), falls back to `kv_a_proj_with_mqa`.
        tensors.insert(
            format!("{l}.self_attn.kv_a_proj_with_mqa.weight"),
            zeros(
                &[KV_LORA_RANK + QK_ROPE_HEAD_DIM, HIDDEN_SIZE],
                device,
            )?,
        );
        tensors.insert(
            format!("{l}.self_attn.kv_a_layernorm.weight"),
            ones(&[KV_LORA_RANK], device)?,
        );
        // kv_b_proj: (kv_lora_rank → num_heads * (qk_nope + v_head_dim))
        tensors.insert(
            format!("{l}.self_attn.kv_b_proj.weight"),
            zeros(
                &[
                    NUM_ATTN_HEADS * (QK_NOPE_HEAD_DIM + V_HEAD_DIM),
                    KV_LORA_RANK,
                ],
                device,
            )?,
        );

        // V4 grouped-LoRA o_proj: o_a_proj + o_b_proj. deepseek4.rs probes
        // `o_a_proj.weight` + `o_b_proj.weight` first, falls back to single
        // `o_proj.weight`. We publish the V4-style LoRA decomposition.
        let o_inner = O_GROUPS * O_LORA_RANK;
        tensors.insert(
            format!("{l}.self_attn.o_a_proj.weight"),
            zeros(&[o_inner, NUM_ATTN_HEADS * V_HEAD_DIM], device)?,
        );
        tensors.insert(
            format!("{l}.self_attn.o_b_proj.weight"),
            zeros(&[HIDDEN_SIZE, o_inner], device)?,
        );

        // ---- MLP / MoE block ----
        if i < FIRST_K_DENSE_REPLACE {
            // Dense MLP (layer 0).
            tensors.insert(
                format!("{l}.mlp.gate_proj.weight"),
                zeros(&[INTERMEDIATE_SIZE, HIDDEN_SIZE], device)?,
            );
            tensors.insert(
                format!("{l}.mlp.up_proj.weight"),
                zeros(&[INTERMEDIATE_SIZE, HIDDEN_SIZE], device)?,
            );
            tensors.insert(
                format!("{l}.mlp.down_proj.weight"),
                zeros(&[HIDDEN_SIZE, INTERMEDIATE_SIZE], device)?,
            );
        } else {
            // MoE: gate + stacked routed experts + shared experts.
            // Gate: (n_routed_experts, hidden)
            tensors.insert(
                format!("{l}.mlp.gate.weight"),
                zeros(&[NUM_ROUTED_EXPERTS, HIDDEN_SIZE], device)?,
            );

            // PackedExperts unquantized format (used by MoEExperts::Slow
            // backend on CPU): stacked `gate_up_proj` + `down_proj`.
            // Shapes per `mistralrs-quant/src/distributed/layers.rs:1377` —
            // gate_up_proj: (num_experts, hidden, 2 * intermediate)
            // down_proj:   (num_experts, intermediate, hidden)
            tensors.insert(
                format!("{l}.mlp.experts.gate_up_proj"),
                zeros(
                    &[NUM_ROUTED_EXPERTS, HIDDEN_SIZE, 2 * MOE_INTERMEDIATE_SIZE],
                    device,
                )?,
            );
            tensors.insert(
                format!("{l}.mlp.experts.down_proj"),
                zeros(
                    &[NUM_ROUTED_EXPERTS, MOE_INTERMEDIATE_SIZE, HIDDEN_SIZE],
                    device,
                )?,
            );

            // Shared experts: per Mlp::new, names are gate_proj/up_proj/down_proj.
            // Intermediate = moe_intermediate_size * n_shared_experts.
            let shared_inter = MOE_INTERMEDIATE_SIZE * NUM_SHARED_EXPERTS;
            tensors.insert(
                format!("{l}.mlp.shared_experts.gate_proj.weight"),
                zeros(&[shared_inter, HIDDEN_SIZE], device)?,
            );
            tensors.insert(
                format!("{l}.mlp.shared_experts.up_proj.weight"),
                zeros(&[shared_inter, HIDDEN_SIZE], device)?,
            );
            tensors.insert(
                format!("{l}.mlp.shared_experts.down_proj.weight"),
                zeros(&[HIDDEN_SIZE, shared_inter], device)?,
            );
        }
    }

    Ok(tensors)
}

// ---------------------------------------------------------------------------
// Config builder.
//
// All V4 fields are populated explicitly so this test breaks loudly if the
// `DeepSeekV4Config` schema changes.
// ---------------------------------------------------------------------------

fn build_v4_config_json() -> String {
    // n_routed_experts must be divisible by n_group, and topk_group <= n_group.
    // With Greedy topk_method we don't use these fields, but they must parse.
    serde_json::json!({
        "vocab_size": VOCAB_SIZE,
        "hidden_size": HIDDEN_SIZE,
        "intermediate_size": INTERMEDIATE_SIZE,
        "moe_intermediate_size": MOE_INTERMEDIATE_SIZE,
        "num_hidden_layers": NUM_LAYERS,
        "num_attention_heads": NUM_ATTN_HEADS,
        "n_shared_experts": NUM_SHARED_EXPERTS,
        "n_routed_experts": NUM_ROUTED_EXPERTS,
        "routed_scaling_factor": 1.0,
        "topk_method": "greedy",
        "num_experts_per_tok": NUM_EXPERTS_PER_TOK,
        "moe_layer_freq": 1,
        "first_k_dense_replace": FIRST_K_DENSE_REPLACE,
        "scoring_func": "softmax",
        "hidden_act": "silu",
        "max_position_embeddings": MAX_POSITION_EMBEDDINGS,
        "rms_norm_eps": RMS_NORM_EPS,
        "tie_word_embeddings": false,
        "rope_theta": ROPE_THETA,
        "rope_scaling": null,
        "attention_bias": false,
        "q_lora_rank": Q_LORA_RANK,
        "qk_rope_head_dim": QK_ROPE_HEAD_DIM,
        "kv_lora_rank": KV_LORA_RANK,
        "v_head_dim": V_HEAD_DIM,
        "qk_nope_head_dim": QK_NOPE_HEAD_DIM,
        "n_group": 1,
        "topk_group": 1,
        // V4 fields
        "compress_ratios": [0, 4, 128, 0],
        "sliding_window": SLIDING_WINDOW,
        "compress_rope_theta": 40000.0,
        "o_lora_rank": O_LORA_RANK,
        "o_groups": O_GROUPS,
        "index_n_heads": 1,
        "index_head_dim": 8,
        "index_topk": 4,
    })
    .to_string()
}

// ---------------------------------------------------------------------------
// Load + forward helpers.
// ---------------------------------------------------------------------------

fn load_synthetic_v4() -> anyhow::Result<Box<dyn mistralrs_core::pipeline::NormalModel + Send + Sync>>
{
    let device = Device::Cpu;
    let tensors = build_synthetic_v4_weights(&device)?;
    let backend: Box<dyn candle_nn::var_builder::SimpleBackend + 'static> = Box::new(tensors);
    let vb: ShardedVarBuilder = ShardedSafeTensors::wrap(backend, DType::F32, device.clone());

    let cfg = build_v4_config_json();
    let mapper = Box::new(DummyDeviceMapper {
        nm_device: device.clone(),
    });

    let metadata = NormalLoadingMetadata {
        mapper,
        loading_isq: false,
        real_device: device,
        multi_progress: Arc::new(MultiProgress::new()),
        matformer_slicing_config: None,
    };

    let loader = DeepSeekV4Loader;
    let model = loader.load(&cfg, vb, metadata, AttentionImplementation::Eager)?;
    Ok(model)
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

fn run_forward(
    model: &(dyn mistralrs_core::pipeline::NormalModel + Send + Sync),
    input_ids: &Tensor,
) -> CandleResult<Tensor> {
    let seq_len = input_ids.dim(1)?;
    let context_lens = vec![(0_usize, seq_len)];
    let position_ids = (0..seq_len).collect::<Vec<_>>();
    let seqlen_offsets = vec![0_usize];
    let flash_params = empty_flash_params();

    model.forward(
        input_ids,
        &seqlen_offsets,
        context_lens,
        position_ids,
        None, // no PagedAttention metadata
        &flash_params,
    )
}

// ---------------------------------------------------------------------------
// Tests.
// ---------------------------------------------------------------------------

/// The whole load → forward → finiteness → determinism contract in one test.
#[test]
fn v4_synthetic_load_and_forward_is_finite_and_deterministic() {
    let device = Device::Cpu;

    // Build + load.
    let mut model = match load_synthetic_v4() {
        Ok(m) => m,
        Err(e) => panic!(
            "DeepSeekV4Loader::load failed on synthetic weights.\n\
             This usually means deepseek4.rs is probing for a tensor that\n\
             build_synthetic_v4_weights() does not emit, or the weight shape\n\
             does not match what the model expects.\n\n\
             Error chain:\n{e:?}"
        ),
    };

    // Build input: [batch=1, seq=4] with token ids 0..4 (within vocab).
    let input_ids = Tensor::from_vec(vec![0_u32, 1, 2, 3], &[1usize, 4], &device).unwrap();

    // First forward.
    let out1 = run_forward(model.as_ref(), &input_ids)
        .expect("V4 forward #1 failed on synthetic CPU weights");

    // Shape check: context_lens=[(0, seq_len)] selects all logits, so the
    // shape must be [batch=1, seq=4, vocab=VOCAB_SIZE].
    assert_eq!(
        out1.dims(),
        &[1usize, 4, VOCAB_SIZE],
        "Unexpected output shape; deepseek4.rs::DeepSeekV4::forward may have \
         changed extract_logits or LM-head semantics."
    );

    // Finiteness check — fail loudly if any NaN or Inf shows up.
    let flat: Vec<f32> = out1
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    let nan_count = flat.iter().filter(|v| v.is_nan()).count();
    let inf_count = flat.iter().filter(|v| v.is_infinite()).count();
    assert_eq!(
        nan_count, 0,
        "V4 output contained {nan_count} NaN(s). \
         Likely a divide-by-zero in softmax over zero logits, or an \
         uninitialized cache path. Investigate MLA + MoE gating numerics."
    );
    assert_eq!(
        inf_count, 0,
        "V4 output contained {inf_count} Inf(s). \
         Likely overflow in attention scaling — check sdpa_params.softmax_scale \
         and the compress-branch blending in deepseek4.rs::Attention::forward."
    );

    // Determinism: reset cache and re-run.
    {
        let cache = model.cache_mut().normal();
        cache.0.iter_mut().for_each(|c| {
            *c = mistralrs_core::pipeline::KvCache::Normal(
                mistralrs_core::pipeline::NormalCache::new(1, MAX_POSITION_EMBEDDINGS)
                    .normal()
                    .0
                    .remove(0),
            )
        });
    }

    let out2 = run_forward(model.as_ref(), &input_ids)
        .expect("V4 forward #2 failed on synthetic CPU weights");

    // Same shape.
    assert_eq!(out1.dims(), out2.dims(), "Determinism failure: shape diverged");

    // Same values (allow tight FP tolerance for any nondeterministic CPU ops).
    let flat2: Vec<f32> = out2
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    let mut max_abs_diff = 0_f32;
    for (a, b) in flat.iter().zip(flat2.iter()) {
        max_abs_diff = max_abs_diff.max((a - b).abs());
    }
    assert!(
        max_abs_diff < 1e-4,
        "Determinism failure: max abs diff between two forward passes = {max_abs_diff}. \
         This points at a non-deterministic op (likely in MoE expert dispatch \
         or attention softmax) — check that cache reset is complete."
    );
}
