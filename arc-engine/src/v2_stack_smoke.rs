//! v2 stack-composition smoke test.
//!
//! Exercises the full Arc v2 path on a tiny synthetic model:
//!   1. Generate Gaussian weights for a 2-layer MLP-style FFN.
//!   2. Quantize the weights using `mistralrs_quant::NVFP4Layer` (Phase 0 #1).
//!   3. Apply dReLU activation (Phase 1 #5 Turbo Sparse).
//!   4. Run the forward pass and verify the output is finite + bounded.
//!
//! This isn't a perplexity test — it's a "the wires connect" check that runs
//! on any host without hardware. It catches integration regressions across
//! multiple v2 components in one test.

#[cfg(test)]
mod tests {
    use candle_core::{DType, Device, Result, Tensor};

    /// Sanity: every targeted frontier model family parses through Arc's
    /// architecture-detection layer without panicking, picking the right
    /// model-family loader. This is the "model loads cleanly" smoke test.
    #[test]
    fn all_frontier_model_configs_dispatch_correctly() {
        use crate::deepseek_v4::{is_v4_config, DeepSeekV4Config};
        use crate::glm_moe::{is_glm_moe_config, GlmMoeConfig, GlmVariant};
        use crate::kimi_k2::{is_kimi_k2_config, KimiK2Config};

        // DeepSeek V4 Flash (small variant)
        let v4_flash = r#"{
            "architectures": ["DeepseekV4ForCausalLM"],
            "model_type": "deepseek_v4",
            "num_hidden_layers": 27,
            "hidden_size": 2048,
            "n_routed_experts": 128,
            "compress_ratios": [0, 4, 4, 128, 4, 4, 128, 4, 4, 128, 4, 4, 128, 4, 4, 128, 4, 4, 128, 4, 4, 128, 4, 4, 128, 4, 4]
        }"#;
        assert!(is_v4_config(v4_flash));
        let cfg = DeepSeekV4Config::from_json(v4_flash).unwrap();
        let (std, csa, hca) = cfg.compress_ratio_layer_counts();
        assert_eq!(std + csa + hca, 27);

        // DeepSeek V4 Pro (full scale)
        let v4_pro = r#"{
            "architectures": ["DeepseekV4ForCausalLM"],
            "model_type": "deepseek_v4",
            "num_hidden_layers": 43,
            "hidden_size": 4096,
            "n_routed_experts": 256,
            "kv_lora_rank": 512,
            "qk_nope_head_dim": 448,
            "qk_rope_head_dim": 64,
            "max_position_embeddings": 1048576
        }"#;
        assert!(is_v4_config(v4_pro));
        let cfg = DeepSeekV4Config::from_json(v4_pro).unwrap();
        assert_eq!(cfg.num_hidden_layers, 43);
        assert_eq!(cfg.q_head_dim(), 512);
        assert_eq!(cfg.max_position_embeddings, 1_048_576);

        // Kimi K2.5
        let k25 = r#"{
            "architectures": ["KimiK25ForConditionalGeneration"],
            "model_type": "kimi_k25",
            "text_config": {
                "vocab_size": 160000,
                "num_hidden_layers": 61,
                "hidden_size": 7168,
                "n_routed_experts": 384,
                "num_experts_per_tok": 8,
                "max_position_embeddings": 262144
            }
        }"#;
        assert!(is_kimi_k2_config(k25));
        let cfg = KimiK2Config::from_json(k25).unwrap();
        assert_eq!(cfg.text_config.vocab_size, 160_000);
        assert_eq!(cfg.text_config.n_routed_experts, 384);
        assert!(cfg.is_kimi_architecture());
        assert!(!cfg.has_vision());

        // Kimi K2.6 with vision tower
        let k26_vl = r#"{
            "architectures": ["KimiK26VLForConditionalGeneration"],
            "model_type": "kimi_k26",
            "text_config": {
                "vocab_size": 160000,
                "num_hidden_layers": 61,
                "hidden_size": 7168,
                "n_routed_experts": 384,
                "num_experts_per_tok": 8
            },
            "vision_config": {
                "patch_size": 14,
                "num_attention_heads": 16,
                "num_hidden_layers": 27,
                "hidden_size": 1152,
                "intermediate_size": 4304
            }
        }"#;
        assert!(is_kimi_k2_config(k26_vl));
        let cfg = KimiK2Config::from_json(k26_vl).unwrap();
        assert!(cfg.has_vision());
        let vis = cfg.vision_config.as_ref().unwrap();
        assert_eq!(vis.num_hidden_layers, 27);

        // GLM-4.5 (plain GLM-4 MoE)
        let glm45 = r#"{
            "architectures": ["Glm4MoeForCausalLM"],
            "model_type": "glm4_moe",
            "num_hidden_layers": 46,
            "hidden_size": 4096,
            "n_routed_experts": 128,
            "num_experts_per_tok": 8,
            "max_position_embeddings": 131072
        }"#;
        assert!(is_glm_moe_config(glm45));
        let cfg = GlmMoeConfig::from_json(glm45).unwrap();
        assert_eq!(cfg.variant(), GlmVariant::Glm4Moe);

        // GLM-5.1 (DSA variant)
        let glm51 = r#"{
            "architectures": ["GlmMoeDsaForCausalLM"],
            "model_type": "glm5",
            "vocab_size": 151552,
            "num_hidden_layers": 80,
            "hidden_size": 5120,
            "num_attention_heads": 64,
            "num_key_value_heads": 8,
            "n_routed_experts": 256,
            "num_experts_per_tok": 8,
            "max_position_embeddings": 200000,
            "max_output_length": 131072,
            "kv_lora_rank": 512,
            "qk_nope_head_dim": 128,
            "qk_rope_head_dim": 64,
            "v_head_dim": 128
        }"#;
        assert!(is_glm_moe_config(glm51));
        let cfg = GlmMoeConfig::from_json(glm51).unwrap();
        assert_eq!(cfg.variant(), GlmVariant::Glm5MoeDsa);
        assert_eq!(cfg.num_attention_heads, 64);
        assert_eq!(cfg.kv_lora_rank, Some(512));
    }

    /// Mutual exclusion check: each frontier config dispatches to exactly one
    /// model family, never multiple. Prevents ambiguous routing at load time.
    #[test]
    fn architecture_dispatch_is_mutually_exclusive() {
        use crate::deepseek_v4::is_v4_config;
        use crate::glm_moe::is_glm_moe_config;
        use crate::kimi_k2::is_kimi_k2_config;

        let configs = vec![
            ("V4", r#"{"architectures": ["DeepseekV4ForCausalLM"]}"#),
            ("Kimi", r#"{"architectures": ["KimiK25ForConditionalGeneration"]}"#),
            ("GLM", r#"{"architectures": ["Glm4MoeForCausalLM"]}"#),
            ("GLM5", r#"{"architectures": ["GlmMoeDsaForCausalLM"]}"#),
            ("Llama", r#"{"architectures": ["LlamaForCausalLM"]}"#),
            ("Qwen", r#"{"architectures": ["Qwen3ForCausalLM"]}"#),
            ("V3", r#"{"architectures": ["DeepseekV3ForCausalLM"]}"#),
        ];
        for (name, json) in &configs {
            let v4 = is_v4_config(json);
            let kimi = is_kimi_k2_config(json);
            let glm = is_glm_moe_config(json);
            let count = v4 as u8 + kimi as u8 + glm as u8;
            assert!(
                count <= 1,
                "Config '{name}' matched multiple architectures: v4={v4}, kimi={kimi}, glm={glm}"
            );
        }
    }

    /// Malformed / empty configs must error cleanly, not panic.
    /// This is the "no crash" guarantee at the loader boundary.
    #[test]
    fn malformed_configs_error_cleanly() {
        use crate::deepseek_v4::DeepSeekV4Config;
        use crate::glm_moe::GlmMoeConfig;
        use crate::kimi_k2::KimiK2Config;

        // Completely empty — should still parse via defaults (no crash)
        assert!(DeepSeekV4Config::from_json("{}").is_ok());
        assert!(KimiK2Config::from_json("{}").is_ok());
        assert!(GlmMoeConfig::from_json("{}").is_ok());

        // Malformed JSON → error (not crash)
        assert!(DeepSeekV4Config::from_json("not json").is_err());
        assert!(KimiK2Config::from_json("not json").is_err());
        assert!(GlmMoeConfig::from_json("not json").is_err());

        // Wrong field types → error (not crash)
        let bad_types = r#"{"num_hidden_layers": "forty-three"}"#;
        assert!(DeepSeekV4Config::from_json(bad_types).is_err());
    }


    /// Tiny FFN with NVFP4 weights and dReLU activation. Verifies the full
    /// pipeline composes without panic and produces finite values.
    #[test]
    fn nvfp4_drelu_ffn_forward_is_finite() -> Result<()> {
        let device = Device::Cpu;
        let hidden = 32;
        let intermediate = 128; // 2 * 64 for dReLU split

        // Build "gate_up_proj" weights of shape [intermediate, hidden]
        // such that the output [batch, intermediate] later splits into
        // [gate || up] for dReLU.
        let mut w_data = vec![0f32; intermediate * hidden];
        for (i, v) in w_data.iter_mut().enumerate() {
            *v = ((i as f32) * 0.0137).sin() * 0.7;
        }
        let w = Tensor::from_vec(w_data, (intermediate, hidden), &device)?;

        // Quantize via NVFP4
        let nvfp4 = mistralrs_quant::NVFP4Layer::quantize(&w, None, &device)?;

        // Build a small batch of inputs
        let batch = 2;
        let x_data: Vec<f32> = (0..(batch * hidden))
            .map(|i| ((i as f32) * 0.07).cos() * 0.5)
            .collect();
        let x = Tensor::from_vec(x_data, (batch, hidden), &device)?.to_dtype(DType::F32)?;

        // Forward through NVFP4 — yields [batch, intermediate]
        let gate_up = nvfp4.forward(&x)?;
        assert_eq!(gate_up.dims(), &[batch, intermediate]);

        // Apply dReLU (split-and-relu on intermediate dim → [batch, intermediate/2])
        use candle_core::Module;
        use mistralrs_core::layers::Activation;
        let out = Activation::DRelu.forward(&gate_up.to_dtype(DType::F32)?)?;
        assert_eq!(out.dims(), &[batch, intermediate / 2]);

        // All outputs must be finite and non-negative (dReLU output is product of two ReLUs).
        let v: Vec<f32> = out.flatten_all()?.to_vec1()?;
        for (i, &val) in v.iter().enumerate() {
            assert!(val.is_finite(), "Output[{i}] = {val} is not finite");
            assert!(val >= 0.0, "Output[{i}] = {val} should be >= 0 (dReLU)");
        }

        Ok(())
    }

    /// QTIP weight + dense matmul produces non-trivial output.
    #[test]
    fn qtip_weight_forward_is_finite() -> Result<()> {
        let device = Device::Cpu;
        let out_dim = 8;
        let in_dim = 64;

        let w_data: Vec<f32> = (0..(out_dim * in_dim))
            .map(|i| ((i as f32) * 0.21).sin())
            .collect();
        let w = Tensor::from_vec(w_data, (out_dim, in_dim), &device)?;

        let qtip = mistralrs_quant::QtipLayer::quantize(&w, None, &device)?;

        let batch = 4;
        let x_data: Vec<f32> = (0..(batch * in_dim))
            .map(|i| ((i as f32) * 0.05).cos())
            .collect();
        let x = Tensor::from_vec(x_data, (batch, in_dim), &device)?.to_dtype(DType::F32)?;

        let out = qtip.forward(&x)?;
        assert_eq!(out.dims(), &[batch, out_dim]);

        // Output values should be bounded and finite (not all zero).
        let v: Vec<f32> = out.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
        let max_abs = v.iter().fold(0f32, |m, &x| m.max(x.abs()));
        assert!(max_abs > 0.0, "QTIP output is all zero");
        for (i, &val) in v.iter().enumerate() {
            assert!(val.is_finite(), "Output[{i}] = {val} is not finite");
        }
        Ok(())
    }

    /// MTP + verify composes correctly with the speculative protocol.
    #[test]
    fn mtp_verify_composes_correctly() {
        use crate::mtp::{verify_proposed, MtpConfig, MtpStack};

        let cfg = MtpConfig {
            num_heads: 2,
            hidden_size: 32,
            vocab_size: 256,
        };
        let stack = MtpStack::new(cfg);
        assert_eq!(stack.depth(), 2);

        // Simulate: heads propose [42, 17], target says [42, 99]. Accept 1, reject 17 → 99.
        let proposed = vec![42, 17];
        let target = vec![42, 99];
        let result = verify_proposed(&proposed, &target);
        assert_eq!(result.accepted, vec![42]);
        assert_eq!(result.rejection, Some((1, 99)));
        // Commit length = 1 accepted + 1 correction = 2 tokens.
        assert_eq!(result.commit_len(), 2);
    }

    /// Sarathi + expert affinity compose: prefill chunks while decoders batch
    /// by expert affinity.
    #[test]
    fn sarathi_and_expert_affinity_compose() {
        use crate::expert_affinity::{group_by_expert, AffinityConfig, PendingRequest};
        use crate::sarathi::{ChunkConfig, ChunkedPrefillScheduler};

        // 3 decoders + 1 prefill of 1000 tokens
        let mut sched = ChunkedPrefillScheduler::new(ChunkConfig {
            chunk_size: 256,
            token_budget: 1024,
        });
        sched.register_decode_user(1);
        sched.register_decode_user(2);
        sched.register_decode_user(3);
        sched.enqueue_prefill(99, 1000);

        let (decoders, chunks) = sched.next_batch();
        assert_eq!(decoders, vec![1, 2, 3]);
        // 3 decoders use 3 tokens of budget, leaving 1021 for prefill chunks of 256 each
        // = 4 chunks (256+256+256+253) → all 4 chunks fit, covering tokens 0..1000
        assert!(!chunks.is_empty());

        // Now group the decoders by expert affinity (simulate they all hit expert 5)
        let reqs = vec![
            PendingRequest { user_id: 1, expert_ids: vec![5, 0] },
            PendingRequest { user_id: 2, expert_ids: vec![5, 7] },
            PendingRequest { user_id: 3, expert_ids: vec![5, 0] },
        ];
        let buckets = group_by_expert(&reqs, AffinityConfig { num_experts: 8 });
        // All 3 hit expert 5: bucket[5] = [1, 2, 3]
        assert_eq!(buckets[&5], vec![1, 2, 3]);
        // 2 hit expert 0: bucket[0] = [1, 3]
        assert_eq!(buckets[&0], vec![1, 3]);
        // 1 hits expert 7
        assert_eq!(buckets[&7], vec![2]);
    }

    /// YOCO + MagicDec compose: cache layout is consistent with long-context spec.
    #[test]
    fn yoco_and_magicdec_compose() {
        use crate::magicdec::MagicDecConfig;
        use crate::yoco::YocoLayout;

        let layout = YocoLayout::default_split(32);
        let cfg = MagicDecConfig::default();

        // At 1M context, MagicDec activates and uses a 4K window for the draft.
        assert!(cfg.should_activate(1_000_000));
        let (s, e) = cfg.compute_draft_kv_window(1_000_000, 500_000);
        assert_eq!(e - s, 4096);

        // YOCO halves the per-user KV memory.
        assert!((layout.savings_ratio() - 2.0).abs() < 1e-9);

        // Together: per-user memory at 1M context = (1M positions * KV size) / 2 / window_compression
        // Conceptual sanity check, not a numerical check — the structures compose.
    }

    /// SageAttention quantization composes with the dispatcher pattern.
    #[test]
    fn sage_quantize_dequantize_close_to_identity() -> Result<()> {
        use crate::sage::{per_block_dequantize_i8, per_block_quantize_i8, Sm100QuantConfig};

        let device = Device::Cpu;
        let n = 1 * 2 * 64 * 8;
        let data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.011).sin() * 3.0).collect();
        let t = Tensor::from_vec(data.clone(), (1, 2, 64, 8), &device)?;

        let cfg = Sm100QuantConfig::default();
        let (q, s) = per_block_quantize_i8(&t, cfg)?;
        let recon = per_block_dequantize_i8(&q, &s, cfg)?;

        let v: Vec<f32> = recon.flatten_all()?.to_vec1()?;
        let mut max_abs_err = 0f32;
        for (o, r) in data.iter().zip(v.iter()) {
            max_abs_err = max_abs_err.max((o - r).abs());
        }
        // Per-block INT8: max error ≤ max_abs_per_block / 127. With data in [-3, 3],
        // expect ≤ 0.025 per element.
        assert!(max_abs_err < 0.03, "max abs err {max_abs_err} > 0.03");
        Ok(())
    }
}
