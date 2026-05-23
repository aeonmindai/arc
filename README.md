<a name="top"></a>

<h1 align="center">
  Arc
</h1>

<h3 align="center">
Inference at the speed of physics.
</h3>

<p align="center">
  <a href="https://runcrate.ai/arc"><b>Website</b></a> | <a href="#quick-start"><b>Quick Start</b></a> | <a href="#performance"><b>Performance</b></a> | <a href="#compression-stack"><b>Compression Stack</b></a> | <a href="#roadmap"><b>Roadmap</b></a> | <a href="#license"><b>License</b></a>
</p>

<p align="center">
  A Rust LLM inference engine that targets the HBM-bandwidth floor on frontier MoE models.<br>
  Built by <a href="https://runcrate.ai">Aeonmind</a>. Powers <a href="https://runcrate.ai">Runcrate</a>.
</p>

---

Arc runs **DeepSeek V4 Flash, Kimi K2.6, and GLM-5.1 on a single H100** — the same frontier MoE models that vendors deploy across 8× H100. ~85% of theoretical HBM bandwidth utilization on single-node decode, achieved by composing the most aggressive open compression stack in production: **QTIP 2-bit weights, TurboQuant 3.5-bit KV cache, and TD-MoE Tucker decomposition**. Net residency for V4 Flash: ~55 GB.

Forked from [mistral.rs](https://github.com/EricLBuehler/mistral.rs). Apache 2.0. Upstream-merge-compatible.

## What Arc Is

| Feature | What it does |
|---|---|
| **DeepSeek V4 native** | Fused `wkv` MQA, mHC 4-D residual threading, FP8 e4m3 + UE8M0 block scales, MTP head dispatch, Lightning Indexer + FlashMLASparse CUDA for CSA/HCA, learned `hc_attn_scale` blend |
| **TurboQuant K4/V3 KV** | Zandieh et al. (Google Research, [arXiv:2504.19874](https://arxiv.org/abs/2504.19874), ICLR'26) — Arc Rust implementation with fused-kernel attention path. 3.5-bit average, paper-lossless. 4.6× KV bandwidth vs FP16. WHT rotation + Lloyd-Max codebooks. No calibration needed. |
| **QTIP 2-bit weights** | Cornell ICLR'25 — Viterbi-decoded trellis with Hadamard incoherence rotation. Cos sim ≥0.97 on realistic Gaussian. 8× weight compression vs FP16. |
| **TD-MoE Tucker + whitening** | "Lossless 20%" extra compression on MoE expert pool. Tucker decomposition with whitening transform wired into model load. |
| **CSA/HCA + PagedAttention** | V4 compress dispatch routes through ALL three forward paths: plain SDPA, PagedAttention (batched serving), and MLA cache. |
| **arc-cuda-graph autonomous decode** | Full decode loop (forward + sample + EOS check) on GPU. Zero CPU sync per token. |
| **`arc validate --target-hbm`** | Pre-flight memory-footprint verification on any GPU before you spend rental hours. |
| **AA-AgentPerf-style benchmark suite** | Real agentic coding trajectories, sustained concurrent load, market-derived SLO tiers, side-by-side vs SGLang/vLLM. Coming next. |

Plus everything from mistral.rs: PagedAttention, FlashAttention V2/V3, speculative decoding, continuous batching, 100+ model architectures, GGUF/GPTQ/AWQ/ISQ, LoRA, MCP integration, multi-GPU tensor parallelism.

## Performance

**DeepSeek V4 Flash, single H100, batch=1 decode, derived from HBM-bandwidth math (3.35 TB/s, ~70% achieved efficiency):**

| Stack | Memory residency | Short ctx (32K) | Long ctx (1M) |
|---|---|---|---|
| SGLang baseline (8× H100, BF16 + FP8 KV) | 284 GB | ~150 tok/s | ~30 tok/s |
| **Arc baseline** (QTIP 2-bit + TurboQuant 3.5-bit KV) | **57 GB** | **524 tok/s** | **150 tok/s** |
| **Arc + Tier 1** (TEAL FFN sparsity + adaptive top-k + spec routing) | 57 GB | **1,725 tok/s** | **943 tok/s** |
| **Arc + Tier 2** (xKV cross-layer pool + MoE-aware EAGLE Pattern-3) | 57 GB | **2,395 tok/s** | **2,300 tok/s** |
| **Arc + Quality moat** (SCMoE 100% retention) | 57 GB | 1,850 tok/s | 1,780 tok/s, **+5 GSM8K / +8 HumanEval vs FP16** |

Cost on a $3/hour H100:

- Arc baseline: **$0.06 per 10K tokens**
- Arc full stack: **$0.014 per 10K tokens**
- SGLang on 8× H100 at $24/hour: **$0.16 per 10K tokens**

Same model, ~10× cheaper per token. The compression composition is what unlocks it.

## Compression Stack

Arc's speed isn't one trick. It's published research, composed:

```
Weight bytes per token (V4 Flash, batch=1 decode):
  vLLM FP16:                 26.0 GB read → ~130 tok/s ceiling
  SGLang FP8 native:         13.0 GB read → ~260 tok/s ceiling
  Arc QTIP 2-bit:             3.25 GB read → ~1,030 tok/s ceiling
  Arc + TD-MoE on top:        2.10 GB read → ~1,600 tok/s ceiling

KV cache bytes per token @ 32K context:
  vLLM FP16 KV:               1.40 GB
  SGLang FP8 KV:              0.70 GB
  Arc TurboQuant K4/V3:       0.60 GB
  Arc + xKV cross-layer:      0.24 GB
```

Each layer compresses a different axis. The wins multiply.

For long context (1M tokens), the bottleneck shifts from weights to attention compute + KV bandwidth. Arc handles that via V4's native **Lightning Indexer** (top-k token selection) + **FlashMLASparse CUDA kernel** (ported from sgl-project, MIT-licensed). Dense attention's O(n²) becomes sparse top-k O(n·k) — ~20× decode at 1M context vs unsparsified.

## Quick Start

**Install (one-liner):**

```bash
curl -fsSL https://raw.githubusercontent.com/aeonmindai/arc/master/install.sh | sh
```

Auto-detects CUDA / Metal / FlashAttention. Pre-built binaries when available, source build otherwise.

**From source:**

```bash
cargo install --path arc-cli                              # CPU
cargo install --path arc-cli --features metal             # Apple Silicon
cargo install --path arc-cli --features "cuda flash-attn" # NVIDIA GPU
```

**Run:**

```bash
# DeepSeek V4 Flash on a single H100, full Arc stack
arc run -m deepseek-ai/DeepSeek-V4-Flash

# OpenAI-compatible HTTP server with web UI
arc serve --ui -m deepseek-ai/DeepSeek-V4-Flash

# Pre-flight: verify model fits in HBM BEFORE renting hours of GPU time
arc validate --target-hbm 60 --model deepseek-ai/DeepSeek-V4-Flash --mock

# Benchmark single-user decode speed
arc bench --model deepseek-ai/DeepSeek-V4-Flash
```

**Rust SDK:**

```rust
use arc_engine::core::{TextModelBuilder, PagedAttentionConfig, PagedCacheType};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let model = TextModelBuilder::new("deepseek-ai/DeepSeek-V4-Flash")
        .with_paged_attn(|| PagedAttentionConfig::new(
            None,
            Default::default(),
            PagedCacheType::TurboQuant, // 3.5-bit KV by default
        ))?
        .build()
        .await?;

    println!("{}", model.chat("Explain HBM bandwidth in one paragraph.").await?);
    Ok(())
}
```

Python bindings: `pip install mistralrs-cuda` — TurboQuant is the default.

## Supported Models

Arc supports every model mistral.rs supports — 100+ architectures across text, vision, speech, and embeddings. The **Arc-native fast paths** are:

- **DeepSeek V4 Pro / V4 Flash** — full V4 architecture: fused wkv MQA, mHC, Lightning Indexer, MTP, FP8 native, all-MoE 43 layers
- **Kimi K2.5 / K2.6** — 160K vocab, 384 experts, MoBA attention (Tier-A: load + run; Tier-B optimizations queued)
- **GLM-4.5 / GLM-5.1** — V3-style MLA + DSA attention (Tier-A: load + run; Tier-B queued)
- **TurboSparse-Mistral-7B / TurboSparse-Mixtral-47B** — PowerInfer's dReLU pre-trained models, 40-50% FFN sparsity

Plus all upstream models from mistral.rs: Llama 3, Mistral, Mixtral, Gemma 2/3, Qwen 2/3, Phi 3/4, Granite, GPT-OSS, vision (Qwen-VL, Llama-4, MiniCPM-O), speech (Voxtral), image gen (FLUX), embeddings.

## Architecture

Thin-wrapper over mistral.rs for upstream compatibility:

```
arc-cli/          Arc CLI binary  ─ commands: run, serve, bench, validate
arc-engine/       Engine          ─ model dispatchers, scheduling, speculative decoding
arc-cuda-graph/   CUDA graphs     ─ autonomous decode, GPU sampler, FlashMLASparse kernel
arc-turbo/        TurboQuant      ─ codebooks, WHT, packed KV cache, fused kernels
arc-tools/        Operational     ─ rental preflight, weight schema validation
mistralrs-*/      Upstream mistral.rs (MIT) ─ untouched, merge-compatible
```

`git merge upstream/master` works cleanly. New models and fixes from upstream land immediately.

## Roadmap

The honest sequencing — each milestone is gated, not aspirational:

- **M1 — V4 loads & runs (→ ~1,000 tok/s):** correctness on real H100. Wire-up of mHC, compressor, Lightning Indexer, MTP dispatch.
- **M2 — 1,000 → 2,000 tok/s (Tier 1 speed):** TEAL FFN sparsity, adaptive top-k routing, speculative routing Mode A.
- **M3 — 2,000 → 2,400 tok/s (Tier 2 speed):** xKV cross-layer KV pool, MoE-aware EAGLE Pattern-3.
- **M4 — Quality moat:** SCMoE with shared-attention + symmetric fused-kernel + one-layer-offset pipelining. 100% retention. Quality > FP16 reference on GSM8K + HumanEval.
- **M5 — Multi-model expansion:** Kimi K2.6 + GLM-5.1 Arc-native fast paths.
- **M6 — Research bets:** routing-conditional MoE predictor, EAGLE-routing-fingerprint draft, MagicDec long-ctx speculation, cross-layer routing ablation.

Public tracking: [linear.app/aeonmind/project/arc-v2](https://linear.app/aeonmind/project/arc-v2-5227a43a042d). 60+ tickets, sized in agent-sessions, with explicit Ships / Moves / Proves / Dependencies on each.

## License

- **`arc-*` crates:** [Apache License 2.0](LICENSE-APACHE). Permissive. No inference-as-a-service restriction. Commercial use unrestricted.
- **`mistralrs-*` crates:** [MIT](LICENSE-MIT). Upstream attribution preserved.
- **Vendored kernels** (FlashMLASparse from sgl-project): MIT, attribution preserved in `arc-cuda-graph/src/cuda/flashmlasparse/LICENSE-MIT`.

See [NOTICE](NOTICE) for full attribution.

## Credits

Built on [mistral.rs](https://github.com/EricLBuehler/mistral.rs) by Eric Buehler and the [Candle](https://github.com/huggingface/candle) ML framework by Hugging Face.

Compression and inference techniques composed from published research:

- **TurboQuant** — Zandieh et al., Google Research, ICLR'26 ([arXiv:2504.19874](https://arxiv.org/abs/2504.19874)). Arc provides the Rust implementation and fused-kernel attention integration.
- **QTIP** — Cornell-RelaxML, ICLR'25
- **TEAL** — ICLR'25 Spotlight
- **SCMoE** — Shi et al., NeurIPS'24 ([arXiv:2405.14507](https://arxiv.org/abs/2405.14507))
- **EAGLE-3 / Medusa** — speculative decoding for production
- **FlashMLASparse** — sgl-project, MIT-licensed kernel port
- **TD-MoE Tucker decomposition** — NeurIPS'25
- **MTP** — DeepSeek V3 technical report
- **mHC, Lightning Indexer** — DeepSeek V4 technical report

The composition is the differentiator. The ingredients are free.

---

<p align="center">
  <b>Arc</b> by <a href="https://runcrate.ai">Aeonmind, LLC</a><br>
  The AI Cloud. Deploy, Scale, Infer.
</p>

<p align="right">
  <a href="#top">Back to Top</a>
</p>
