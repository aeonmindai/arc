<a name="top"></a>

<h1 align="center">
  Arc
</h1>

<h3 align="center">
Inference at the speed of physics, not software.
</h3>

<p align="center">
  <a href="https://runcrate.ai/arc"><b>Website</b></a> | <a href="#rust-sdk"><b>Rust SDK</b></a> | <a href="#python-sdk"><b>Python SDK</b></a> | <a href="#turboquant"><b>TurboQuant</b></a> | <a href="https://github.com/runcrate/arc"><b>GitHub</b></a>
</p>

<p align="center">
  A Rust-native LLM inference engine with near-optimal KV cache compression.<br>
  Built by <a href="https://runcrate.ai">Aeonmind</a>. Powers <a href="https://runcrate.ai">Runcrate</a>.
</p>

---

Arc is a high-performance LLM inference engine that extends [mistral.rs](https://github.com/EricLBuehler/mistral.rs) with **TurboQuant** — near-optimal KV cache compression from Google Research (ICLR 2026). The default 3.5-bit configuration (K4/V3) is **lossless**: identical LongBench scores to FP16, with 2.2x memory reduction.

## What Arc Adds

| Feature | What it does | Status |
|---------|-------------|--------|
| **TurboQuant KV cache** | 3.5-bit lossless compression via Walsh-Hadamard rotation + Lloyd-Max codebooks | Default, shipping |
| **Fused attention kernels** | CUDA + Metal kernels that read compressed KV directly — no dequantization | Shipping |
| **3 compression presets** | Default (3.5-bit, lossless), Balanced (3.0-bit), Aggressive (2.5-bit) | Shipping |
| Elastic tensor parallelism | Per-request GPU allocation, TP=1 to TP=8 dynamically | Planned |
| Disaggregated serving | Prefill-decode separation with KV-aware routing | Planned |
| CUDA graph capture | Zero-overhead decode for common batch sizes | Planned |

Everything from mistral.rs is included: PagedAttention, FlashAttention V2/V3, speculative decoding, continuous batching, 100+ model architectures, GGUF/GPTQ/AWQ/ISQ, LoRA, MCP, multi-GPU tensor parallelism, and more.

## Quick Start

### Install

```bash
# Install the Arc CLI
cargo install --path arc-cli

# Or install the upstream mistralrs CLI (also defaults to TurboQuant)
cargo install --path mistralrs-cli
```

### Run

```bash
# Interactive chat — TurboQuant enabled by default
arc run -m Qwen/Qwen3-4B

# Start a server with web UI
arc serve --ui -m google/gemma-3-4b-it

# Benchmark
arc bench -m meta-llama/Llama-3.1-8B-Instruct
```

### TurboQuant Presets

```bash
# Default: 3.5-bit (K4/V3) — lossless, 2.2x compression
arc serve -m <model>

# Balanced: 3.0-bit (K3/V3) — 2.56x compression
arc serve -m <model> --pa-cache-type turboquant-3

# Aggressive: 2.5-bit (K3/V2) — 4.1x compression, ~1.2% quality loss
arc serve -m <model> --pa-cache-type turboquant-aggressive

# Disable TurboQuant (upstream behavior)
arc serve -m <model> --pa-cache-type auto
```

## TurboQuant

TurboQuant ([arXiv:2504.19874](https://arxiv.org/abs/2504.19874), ICLR 2026) compresses KV cache vectors to 2-4 bits using:

1. **Walsh-Hadamard rotation** — O(d log d) random orthogonal transform that makes every coordinate follow a known Beta distribution, regardless of input data
2. **Lloyd-Max codebooks** — pre-computed optimal scalar quantizers for the Beta distribution (no calibration, no training data needed)
3. **Sub-byte packing** — 4-bit nibble and 3-bit 10-in-32 formats, with fused GPU kernels

The key insight: after rotation, optimal quantization is data-oblivious. The codebooks are compile-time constants.

### Performance

| Preset | Bits | Compression | Decode speedup (32K ctx) | Quality |
|--------|------|-------------|--------------------------|---------|
| **Default** | 3.5 (K4/V3) | **2.2x** | ~1.5-1.8x | **Lossless** |
| Balanced | 3.0 (K3/V3) | 2.56x | ~1.8-2.2x | ~0.1% loss |
| Aggressive | 2.5 (K3/V2) | 4.1x | ~2.5-3x | ~1.2% loss |

At 3.5 bits: LongBench 50.06 = identical to FP16 50.06 on Llama-3.1-8B-Instruct. Zero quality loss on needle-in-a-haystack at all context lengths.

### How It Works

```
Write path (per token):
  K/V vector → L2 normalize → WHT rotate (D·H·D) → Lloyd-Max quantize → pack

Read path (attention):
  Rotate Q once → for each cached K: unpack → codebook lookup → dot product
  No full dequantization. Codebook in shared memory.
```

## Python SDK

The Python SDK wraps the upstream mistralrs package with TurboQuant defaults.

```bash
pip install mistralrs              # CPU
pip install mistralrs-cuda         # NVIDIA GPU
pip install mistralrs-metal        # Apple Silicon
```

```python
from mistralrs import Runner, Which, ChatCompletionRequest, PagedCacheType

runner = Runner(
    which=Which.Plain(model_id="Qwen/Qwen3-4B"),
    in_situ_quant="4",
    pa_cache_type=PagedCacheType.TurboQuant,  # default
)

res = runner.send_chat_completion_request(
    ChatCompletionRequest(
        model="default",
        messages=[{"role": "user", "content": "Hello!"}],
        max_tokens=256,
    )
)
print(res.choices[0].message.content)
```

TurboQuant presets available: `PagedCacheType.TurboQuant` (default), `PagedCacheType.TurboQuant3`, `PagedCacheType.TurboQuantAggressive`.

[Python examples](examples/python) | [Cookbook](examples/python/cookbook.ipynb)

## Rust SDK

```bash
cargo add arc-engine
```

```rust
use arc_engine::core::{TextModelBuilder, PagedAttentionConfig, MemoryGpuConfig, PagedCacheType};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    arc_engine::print_banner();

    let model = TextModelBuilder::new("Qwen/Qwen3-4B")
        .with_paged_attn(|| {
            // TurboQuant is the default — this is automatic
            PagedAttentionConfig::new(
                None,
                MemoryGpuConfig::default(),
                PagedCacheType::TurboQuant,
            )
        })?
        .build()
        .await?;

    let response = model.chat("What is Rust's ownership model?").await?;
    println!("{response}");
    Ok(())
}
```

The upstream `mistralrs` Rust crate also works — TurboQuant is the default there too.

[Rust API docs](https://docs.rs/arc-engine) | [Examples](mistralrs/examples)

## HTTP API

Arc serves an OpenAI-compatible API:

```bash
arc serve -p 8080 -m meta-llama/Llama-3.1-8B-Instruct

curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

[Full API documentation](docs/HTTP.md)

## Supported Models

Arc supports every model that mistral.rs supports — 100+ architectures across text, vision, speech, image generation, and embeddings.

<details>
<summary><b>Text</b> — Granite 4.0, SmolLM 3, DeepSeek V3, GPT-OSS, Qwen 3, GLM 4, Gemma 2, Phi 3, Llama, Mistral, Mixtral, Starcoder 2, and more</summary>

Granite 4.0, SmolLM 3, DeepSeek V3, GPT-OSS, DeepSeek V2, Qwen 3 Next, Qwen 3 MoE, Phi 3.5 MoE, Qwen 3, GLM 4, GLM-4.7-Flash, GLM-4.7 MoE, Gemma 2, Qwen 2, Starcoder 2, Phi 3, Mixtral, Phi 2, Gemma, Llama, Mistral
</details>

<details>
<summary><b>Vision</b> — Qwen 3.5, Gemma 3n, Llama 4, Mistral 3, Phi 4, MiniCPM-O, LLaVA, and more</summary>

Qwen 3.5, Qwen 3.5 MoE, Qwen 3-VL, Qwen 3-VL MoE, Gemma 3n, Llama 4, Gemma 3, Mistral 3, Phi 4 multimodal, Qwen 2.5-VL, MiniCPM-O, Llama 3.2 Vision, Qwen 2-VL, Idefics 3, Idefics 2, LLaVA Next, LLaVA, Phi 3V
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

Arc uses a thin-wrapper architecture over mistral.rs for upstream compatibility:

```
arc-cli/          Arc binary (BSL-1.1)
arc-engine/       Wrapper crate (BSL-1.1)
arc-turbo/        TurboQuant: codebooks, WHT, cache, kernels (BSL-1.1)
mistralrs-*/      Upstream mistral.rs (MIT) — untouched, merge-compatible
```

`git merge upstream/master` works cleanly. New models and fixes from upstream are available immediately.

## Building from Source

```bash
# CPU only
cargo build --release -p arc-cli

# NVIDIA GPU (CUDA + FlashAttention)
cargo build --release -p arc-cli --features "cuda flash-attn"

# Apple Silicon (Metal)
cargo build --release -p arc-cli --features metal

# Install globally
cargo install --path arc-cli --features <your-features>
```

## Documentation

- [CLI Reference](docs/CLI.md)
- [HTTP API](docs/HTTP.md)
- [Quantization](docs/QUANTS.md) — ISQ, GGUF, GPTQ, AWQ, TurboQuant
- [PagedAttention](docs/PAGED_ATTENTION.md)
- [Device Mapping](docs/DEVICE_MAPPING.md)
- [MCP Integration](docs/MCP/README.md)
- [Full documentation](https://runcrate.ai/arc/docs)

## License

- **`arc-*` crates**: [Business Source License 1.1](LICENSE-BSL) — free for non-commercial and sub-$1M revenue use. Commercial inference-as-a-service requires a license from Aeonmind.
- **`mistralrs-*` crates**: [MIT](LICENSE-MIT) — upstream open source.

See [NOTICE](NOTICE) for details. For commercial licensing: support@runcrate.ai

## Credits

Arc is built on [mistral.rs](https://github.com/EricLBuehler/mistral.rs) by Eric Buehler and contributors, and the [Candle](https://github.com/huggingface/candle) ML framework by Hugging Face. TurboQuant is based on research by Zandieh et al. at Google Research ([arXiv:2504.19874](https://arxiv.org/abs/2504.19874)).

---

<p align="center">
  <b>Arc</b> by <a href="https://runcrate.ai">Aeonmind, LLC</a><br>
  The AI Cloud. Deploy, Scale, Infer.
</p>

<p align="right">
  <a href="#top">Back to Top</a>
</p>
