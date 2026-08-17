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

Arc is an inference engine built to serve frontier MoE models on radically less hardware, by composing published compression research end-to-end: **QTIP 2-bit weights, TD-MoE Tucker decomposition**, and model-native sparse attention. Measured today: DeepSeek V4 Flash (284B / 13B active) serving from a **~74 GB** artifact on a **single H200**, at **GSM8K 96.3%** on the full test set and **111.69 tok/s aggregate at B=256** — see [Results](#results).

Forked from [mistral.rs](https://github.com/EricLBuehler/mistral.rs). Apache 2.0. Upstream-merge-compatible.

## What Arc Adds

| Feature | What it does |
|---|---|
| **QTIP 2-bit weights** | Cornell ICLR'25 — trellis quantization with Hadamard incoherence rotation. 8× weight compression vs FP16. Two rungs: `qtip2` (LUT) and `qtip2b` (computed codebook), the latter being what Arc serves. Works on any model. |
| **TurboQuant KV** *(experimental, never measured)* | Zandieh et al. (Google Research, [arXiv:2504.19874](https://arxiv.org/abs/2504.19874), ICLR'26) — Arc Rust implementation of WHT + Lloyd-Max KV coding. **No TurboQuant forward pass has ever been benchmarked.** The paged kernels are instantiated at head_dim 64/128/256/512, but only 128 has ever executed on hardware, and only 128 is reachable without asking — the paged default routes onto TurboQuant at head_dim 128 only, and 64/256/512 require an explicit `--pa-cache-type turboquant`. Eager KV is opt-in via `ARC_TURBOQUANT_KV=1`. |
| **TD-MoE Tucker + whitening** | Whitened Tucker decomposition of the MoE expert pool. The **"lossless 20%"** figure is the **paper's** (NeurIPS'25), not ours — *published, not reproduced by us*. Works on any MoE. |
| **ArcAttention/Indexer** — sparse attention | DeepSeek V4's Lightning Indexer with FlashMLASparse CUDA kernels (MIT, ported from sgl-project) for top-k attention. Dense O(n²) → sparse O(n·k). |
| **ArcGraph** — GPU-autonomous decode | Full decode loop (forward + sample + EOS check) on GPU. Zero CPU sync per token. Works on any model. |
| **ArcLab/Validate** — `arc validate --target-hbm` | Pre-flight memory-footprint verification on any GPU before you spend rental hours. |
| **ArcLab/Bench** — AA-AgentPerf suite | Real agentic coding trajectories, sustained concurrent load, market-derived SLO tiers. The harness is vendor-abstracted and drives any OpenAI-compatible server, so side-by-side runs against SGLang/vLLM are *supported* — but **we have never run one**, on any engine, ever. No third-party engine has been benchmarked here, so any competitor performance figure anywhere in this tree is an unsourced leftover being struck on sight; please report one if you find it. |

Plus everything from mistral.rs: PagedAttention, FlashAttention V2/V3, speculative decoding (EAGLE-3, Medusa, MTP), continuous batching, 100+ model architectures, GGUF/GPTQ/AWQ/ISQ, LoRA, MCP integration, multi-GPU tensor parallelism.

> **DeepSeek V4 does not use FlashAttention, at any generation.** Attention sinks are set on all 43 layers, so dispatch takes the sinks path before flash is considered — and the fused flash-with-sinks kernels only cover `head_dim ∈ {64,80,96,112,128,192,256}`, while V4 uses **512**. V4 runs an unfused matmul + `softmax_with_sinks` path on GPU. A fused 512-wide kernel is a *planned* rung with no implementation in the tree. Building with `--features flash-attn` is still correct and still helps every other architecture.

## Results

Measured on rented hardware; full protocols, raw-artifact provenance, and limitations in **[docs/BENCHMARKS.md](docs/BENCHMARKS.md)**. Fleet-economics analysis — with every claim tagged measured vs projected — in **[docs/FLEET.md](docs/FLEET.md)**.

Validated as of Aug 2026 (DeepSeek V4 Flash, 284B / 13B-active MoE, single H200 141 GB):

| Claim | Status | Number |
|---|---|---|
| Frontier MoE on one GPU | **Measured** | 284B/13B V4 Flash serves from a ~**74 GB** artifact (2-bit trellis experts, 8 shards + residual, **2.09 bits/param**) on a single H200. The `qtip2` rung is **74.18 GB**, independently verified against the HF API and again on disk; the `qtip2b` rung Arc serves is **74.12 GB** (recorded in BENCHMARKS.md, not separately HF-API-verified) |
| Quality at 2-bit experts | **Measured** | GSM8K **1270/1319 = 96.3% ±1.0pp** — the **full** test set, 0-shot, **0 degenerate / 0 truncated / 0 errors**, mean 157.8 tokens. The base model's published 90.8 is **8-shot**, a different and easier protocol, so this is *not* a like-for-like win over it |
| End-to-end serving throughput | **Measured** | **111.69 tok/s aggregate at B=256, $12.06/Mtok**, rising monotonically from 18.27 at B=1; `effective_B == B` on all seven batch rows; **0 errors across 505 requests** |
| Long-context correctness | **Measured** | 5/5 coherence + 4/4 needle recall (ablation matrix in BENCHMARKS.md) |
| qtip2b bitshift-trellis format | **Measured** | CUDA↔CPU bit-for-bit parity, 20/20 tests on H200 |
| Batched MoE kernel crossover | **Measured-kernel** | Grouped GEMM overtakes GEMV at **B=64** and keeps climbing (527 tok/s at B=128) while GEMV is flat (315→317). MoE-GEMM path only |
| TurboQuant KV | **Never measured** | A "4.27× KV compression" figure appeared here previously. It was **format arithmetic — bytes per token at 3.5 bits vs BF16 — and was never produced by a forward pass.** Retracted 2026-08-17 |

Per-user speed is the honest weak spot: at B=128 each user sees 1.09 tok/s, and Arc is **overhead-bound rather than bandwidth-bound** today — 111.69 tok/s is low single-digit % of the H200's 4.8 TB/s. One named contributor is a GPU→CPU sampler fallback firing on every token. Saturated-batch floors and per-node replica math are arithmetic, not measurement, and live in [docs/FLEET.md](docs/FLEET.md) explicitly marked projected. **No side-by-side run against SGLang, vLLM or any other engine has ever been performed**, so every $/Mtok figure is Arc-versus-Arc.

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

Each layer compresses a different axis, and in principle the wins multiply.

**Read the block above as arithmetic, because that is all it is.** Exactly one line of it has been measured end-to-end: 284B → **~74 GB** of weights at 2.09 bits/param (V4 Flash trellis bake, serving on one H200 — see [Results](#results)). The KV lines are bytes-per-token ratios; **no TurboQuant forward pass has ever been benchmarked**, and the paged kernels now cover V4's head_dim of 512 without anyone having run one. The xKV pool is not implemented.

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
            PagedCacheType::Auto, // opts *out* of TurboQuant; see note below
        ))?
        .build()
        .await?;

    println!("{}", model.chat("Explain HBM bandwidth.").await?);
    Ok(())
}
```

Python bindings: `pip install mistralrs-cuda`.

> **TurboQuant *is* the paged default at head_dim 128 — correcting an earlier claim here that it "is not the default anywhere," which was wrong when written.** `defaults::PAGED_CACHE_TYPE` is `PagedCacheType::TurboQuant`, so leaving `--pa-cache-type` unset on CUDA gives a standard-layout head_dim-128 model TurboQuant KV with no flag, and **silently drops prefix caching**, which no TurboQuant variant supports. Pass `--pa-cache-type auto` (or `PagedCacheType::Auto`) to opt out.
>
> Every other geometry falls back to `Auto` with a warning: MLA layouts, uninstantiated head dims, and the instantiated-but-unmeasured widths 64/256/512. Those three are reachable only by asking for them — `--pa-cache-type turboquant` is accepted at any instantiated width and is a hard error off that set. **None of this has a measured serving run behind it at any width, 128 included.** The eager KV path is separate and opt-in via `ARC_TURBOQUANT_KV=1`.

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

Arc is organised as **named systems**. Every subsystem resolves upward to one of
these, so there is always an answer to "what am I working on?" The full tree —
every subsystem, what it does, where it lives, and what is shipped versus
planned — is in **[memory/mission/TAXONOMY.md](memory/mission/TAXONOMY.md)**.

```
Arc
├── ArcServe      the front door: HTTP/OpenAI, CLI, SDKs, MCP
├── ArcInfer      the runtime: request → tokens
│   ├── ArcSched      serving loop, admission, batching policy
│   ├── ArcKV         key/value memory: sharing tree, paged, dense, FP8
│   ├── ArcAttention  attention math + kernel dispatch
│   ├── ArcSpec       speculative decoding (MTP, EAGLE-3)
│   ├── ArcMoE        mixture-of-experts serving, TD-MoE
│   ├── ArcGraph      GPU-autonomous decode (CUDA graphs)
│   ├── ArcSample     token sampling
│   └── ArcBoost      training-free serving-side quality
├── ArcModels     architecture support — where new models land
├── ArcQuant      compression: QTIP (weights), TurboQuant (KV), ArcBake
├── ArcKernels    the GPU substrate — ArcTarget is where new GPUs land
├── ArcFormat     the artifact: UQFF + the ArcOverlay serving convention
├── ArcLab        measurement: profiler, benchmarks, ops tooling
└── ArcGate       correctness gates and release discipline
```

Mapped onto the workspace:

```
arc-cli/          ArcServe/CLI    ─ the `arc` binary: run, serve, bench, validate
arc-engine/       ArcServe/SDK    ─ façade + Tier-A research modules
arc-cuda-graph/   ArcGraph        ─ autonomous decode, GPU sampler, FlashMLASparse
arc-turbo/        TurboQuant      ─ packed KV cache type over the quant crate
arc-profiler/     ArcLab/Profiler ─ wall/device/sync span tree
arc-bench/        ArcLab/Bench    ─ AA-AgentPerf trajectory replay harness
arc-tools/        ArcLab/Ops      ─ shell + Python only; NOT a Cargo crate
mistralrs-*/      upstream mistral.rs (MIT), plus Arc's engine work in-place
```

> `mistralrs-*` is **not** untouched. Most of Arc's runtime — the V4
> architecture, the KV sharing tree, attention dispatch, MTP, the trellis
> quantizers and their kernels — lives inside `mistralrs-core/` and
> `mistralrs-quant/` rather than in the `arc-*` crates. `git merge
> upstream/master` still works cleanly, which is the property worth protecting;
> "thin wrapper" was never an accurate description.

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

- **TurboQuant** — Zandieh et al., Google Research, ICLR'26 ([arXiv:2504.19874](https://arxiv.org/abs/2504.19874)). Arc provides the Rust implementation, runtime Lloyd-Max codebook generation for arbitrary block dimensions, and the non-power-of-two head_dim layout decomposition. Experimental and never measured; on by default on the paged path at head_dim 128 (see [Quick Start](#quick-start)).
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
