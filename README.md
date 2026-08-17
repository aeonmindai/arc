<a name="top"></a>

<h1 align="center">
  Arc
</h1>

<h3 align="center">
Inference at the speed of physics.
</h3>

<p align="center">
  <a href="https://runcrate.ai/arc"><b>Website</b></a> | <a href="#quick-start"><b>Quick Start</b></a> | <a href="#results"><b>Results</b></a> | <a href="#compression-stack"><b>Compression</b></a> | <a href="#supported-models"><b>Models</b></a> | <a href="#license"><b>License</b></a>
</p>

<p align="center">
  A Rust LLM inference engine that targets the HBM-bandwidth floor on any model.<br>
  Built by <a href="https://runcrate.ai">Aeonmind</a>. Powers <a href="https://runcrate.ai">Runcrate</a>.
</p>

---

Arc is an inference engine built to serve frontier MoE models on radically less hardware, by composing published compression research end-to-end: **QTIP 2-bit weights, TurboQuant 3.5-bit KV cache, TD-MoE Tucker decomposition**, and model-native sparse attention kernels. Measured today: DeepSeek V4 Flash (284B / 13B active) serving from a **74.18 GB** artifact on a **single H200**, at GSM8K 87.0% (provisional — see [Results](#results)).

Forked from [mistral.rs](https://github.com/EricLBuehler/mistral.rs). Apache 2.0. Upstream-merge-compatible.

## What Arc Adds

| Feature | What it does |
|---|---|
| **TurboQuant K4/V3 KV** | Zandieh et al. (Google Research, [arXiv:2504.19874](https://arxiv.org/abs/2504.19874), ICLR'26) — Arc Rust impl with fused-kernel attention path. 3.5-bit average, paper-lossless. Measured: **4.27× KV compression** end-to-end (Qwen3-32B, single H100: 39K→169K context). GQA-attention models; MLA models currently fall back to the standard KV path. |
| **QTIP 2-bit weights** | Cornell ICLR'25 — Viterbi-decoded trellis with Hadamard incoherence rotation. 8× weight compression vs FP16. Works on any model. |
| **TD-MoE Tucker + whitening** | Whitened Tucker decomposition of the MoE expert pool. The **"lossless 20%"** figure is the **paper's** (NeurIPS'25), not ours — *published, not reproduced by us*. Works on any MoE. |
| **Sparse attention kernels** | FlashMLASparse CUDA kernel (MIT, ported from sgl-project) for top-k attention. Dense O(n²) → sparse O(n·k). |
| **arc-cuda-graph autonomous decode** | Full decode loop (forward + sample + EOS check) on GPU. Zero CPU sync per token. Works on any model. |
| **`arc validate --target-hbm`** | Pre-flight memory-footprint verification on any GPU before you spend rental hours. |
| **AA-AgentPerf-style benchmark suite** | Real agentic coding trajectories, sustained concurrent load, market-derived SLO tiers. The harness is vendor-abstracted and drives any OpenAI-compatible server, so side-by-side runs against SGLang/vLLM are *supported* — but **we have never run one**, on any engine, ever. No third-party engine has been benchmarked here, so any competitor performance figure anywhere in this tree is an unsourced leftover being struck on sight; please report one if you find it. |

Plus everything from mistral.rs: PagedAttention, FlashAttention V2/V3, speculative decoding (EAGLE-3, Medusa, MTP), continuous batching, 100+ model architectures, GGUF/GPTQ/AWQ/ISQ, LoRA, MCP integration, multi-GPU tensor parallelism.

## Results

Measured on rented hardware; full protocols, raw-artifact provenance, and limitations in **[docs/BENCHMARKS.md](docs/BENCHMARKS.md)**. Fleet-economics analysis — with every claim tagged measured vs projected — in **[docs/FLEET.md](docs/FLEET.md)**.

Validated as of Aug 2026 (DeepSeek V4 Flash, 284B / 13B-active MoE, single H200 141 GB):

| Claim | Status | Number |
|---|---|---|
| Frontier MoE on one GPU | **Measured** | 284B/13B V4 Flash serves from a **74.18 GB** artifact (2-bit trellis experts + FP8 attention, 8 shards + residual) on a single H200 — size HF-API-verified on the published `aeonmind/DeepSeek-V4-Flash-UQFF-qtip2` |
| Quality at 2-bit experts | **Measured — provisional** | GSM8K **87.0%** (n=100, 0-shot chat, greedy, seed 161, 2048-token cap) vs 90.8 published for the base model (**8-shot** — a different and easier protocol); facts 22/22, arithmetic 8/8, coherence 6/6. **Provisional:** PR #35 changed the decode math after this was measured (SwiGLU clamp missing on 4 of 5 expert paths incl. the shared expert; YaRN on ratio-0 layers). Direction expected neutral-to-better, **unmeasured** — [details](docs/BENCHMARKS.md) |
| Long-context correctness | **Measured** | 5/5 coherence + 4/4 needle recall (ablation matrix in BENCHMARKS.md) |
| TurboQuant KV | **Measured** | **4.27×** KV compression, Qwen3-32B on one H100 (39K→169K-token context) |
| qtip2b bitshift-trellis format | **Measured** | CUDA↔CPU parity, 20/20 tests on H200 |
| Single-user decode speed | **Measured** | **14.58 tok/s** (batch=1, no-`cudnn` build; progression 5.4 → 13.99 → 14.58 across the kernel-fix PRs). The tuned gather-GEMV variants reach 450–467 GB/s ≈ 9.5% of peak HBM — **measured-kernel**, end-to-end effect pending (profile in BENCHMARKS.md) |

Throughput beyond these numbers — saturated-batch floors, per-node replica math — is arithmetic, not measurement, and lives in [docs/FLEET.md](docs/FLEET.md) explicitly marked as projected. Total spend to produce every measured number above: ≈ $123 of rented H200 time across four sessions.

## Compression Stack

Arc's speed isn't one trick. It's published research, composed.

```
Weight bytes per token (single-user decode, any model):
  FP16 baseline:               (params × 2 bytes)
  QTIP 2-bit:                  (params × 0.25 bytes)     → 8× less HBM read
  + TD-MoE Tucker (MoE only):  (params × ~0.16 bytes)    → additional 1.5×

KV cache bytes per token @ 32K context:
  FP16 KV:                    (head_dim × n_kv × 2 bytes)
  TurboQuant K4/V3:           (head_dim × n_kv × 0.44 bytes)  → 4.6× less
  + xKV cross-layer pool:     (head_dim × n_kv × 0.18 bytes)  → additional 2.5×
```

Each layer compresses a different axis. The wins multiply.

Measured end-to-end so far: **4.27×** KV (TurboQuant, Qwen3-32B on one H100) and 284B → **74.18 GB** weights (V4 Flash qtip2 bake, serving on one H200 — see [Results](#results)). The remaining multipliers above are format arithmetic until measured.

For long context (1M+ tokens), the bottleneck shifts from weights to attention compute + KV bandwidth. Arc handles that via FlashMLASparse (CUDA kernel, MIT-licensed, ported from sgl-project), turning dense attention's O(n²) into sparse top-k O(n·k) on models with native sparse-attention training (DeepSeek V3.2+ family) and via top-k attention + sink preservation on the rest.

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
# Any HuggingFace model, full Arc stack
arc run -m meta-llama/Llama-3.1-8B-Instruct
arc run -m Qwen/Qwen3-32B-Instruct
arc run -m deepseek-ai/DeepSeek-V4-Flash

# OpenAI-compatible HTTP server with web UI
arc serve --ui -m <model>

# Pre-flight: verify model fits HBM before renting GPU time
arc validate --target-hbm 60 --model <model> --mock

# Benchmark
arc bench --model <model>
```

**Rust SDK:**

```rust
use arc_engine::core::{TextModelBuilder, PagedAttentionConfig, PagedCacheType};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let model = TextModelBuilder::new("meta-llama/Llama-3.1-8B-Instruct")
        .with_paged_attn(|| PagedAttentionConfig::new(
            None,
            Default::default(),
            PagedCacheType::TurboQuant, // 3.5-bit KV by default
        ))?
        .build()
        .await?;

    println!("{}", model.chat("Explain HBM bandwidth.").await?);
    Ok(())
}
```

Python bindings: `pip install mistralrs-cuda` — TurboQuant is the default.

## Supported Models

Arc supports every model mistral.rs supports — 100+ architectures across text, vision, speech, image generation, and embeddings.

> **"Supported" here means an architecture loads — not that Arc has served it.** The only model Arc has ever run end-to-end on hardware is **DeepSeek V4 Flash**; every number in [Results](#results) comes from it. DeepSeek V4 Pro, Kimi K2.5 / K2.6 and GLM-5.1 have **never been loaded here** — they are roadmap targets ([Roadmap](#roadmap), Phase 1), listed below for architecture coverage only.

<details>
<summary><b>Text</b> — Llama, Mistral, Qwen, Gemma, Phi, DeepSeek, Kimi, GLM, Granite, GPT-OSS, and more</summary>

DeepSeek V4 (Pro + Flash), DeepSeek V3 family, Kimi K2.5 / K2.6, GLM-4.5 / 5.1, Mixtral 8x7B / 8x22B, Llama 3 / 3.1 / 3.2 / 3.3, Qwen 2 / 2.5 / 3 / 3-Next / 3 MoE, Mistral, Gemma 2 / 3 / 3n, Phi 3 / 3.5 / 4, Granite 4.0, GPT-OSS, SmolLM 3, Phi 3.5 MoE, Starcoder 2, and all upstream architectures
</details>

<details>
<summary><b>Vision</b> — Qwen-VL, Llama-4, Gemma 3, Phi 4, MiniCPM-O, LLaVA, and more</summary>

Qwen 3.5 / 3-VL / 3-VL MoE, Llama 4, Gemma 3 / 3n, Mistral 3, Phi 4 multimodal, Qwen 2.5-VL, MiniCPM-O, Llama 3.2 Vision, Qwen 2-VL, Idefics 2 / 3, LLaVA Next, Phi 3V
</details>

<details>
<summary><b>Speech</b> — Voxtral, Dia</summary>

Voxtral (ASR/speech-to-text), Dia
</details>

<details>
<summary><b>Image Generation</b> — FLUX</summary>

FLUX
</details>

<details>
<summary><b>Embeddings</b> — Embedding Gemma, Qwen 3 Embedding</summary>

Embedding Gemma, Qwen 3 Embedding
</details>

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

In rough order:

- **Phase 1 — Correctness across frontier models:** Llama, Qwen, Mixtral, DeepSeek V4, Kimi K2.6, GLM-5.1 all loading and serving correctly with Arc-native fast paths.
- **Phase 2 — Compression composition:** TEAL FFN sparsity, adaptive top-k routing, speculative routing, xKV cross-layer KV pool, MoE-aware EAGLE-3 speculative drafting.
- **Phase 3 — Quality moat:** SCMoE (Self-Contrastive MoE decoding) with 100% retention via shared-attention + symmetric fused-kernel + one-layer-offset pipelining. The open question this phase exists to answer is whether it can match — or beat — our own FP16 reference on GSM8K + HumanEval. That is a hypothesis, not a scheduled result.
- **Phase 4 — Hardware tier expansion:** B200 / NVFP4 path for trillion-parameter models.

## License

- **`arc-*` crates:** [Apache License 2.0](LICENSE-APACHE). Permissive. Commercial use unrestricted.
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
- **Lightning Indexer, mHC** — DeepSeek V4 technical report

The composition is the differentiator. The ingredients are public.

---

<p align="center">
  <b>Arc</b> by <a href="https://runcrate.ai">Aeonmind, LLC</a><br>
  The AI Cloud. Deploy, Scale, Infer.
</p>

<p align="right">
  <a href="#top">Back to Top</a>
</p>
