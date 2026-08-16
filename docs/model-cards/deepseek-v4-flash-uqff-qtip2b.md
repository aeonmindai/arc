---
tags:
  - uqff
  - mistral.rs
  - arc
  - qtip2b
base_model: deepseek-ai/DeepSeek-V4-Flash
base_model_relation: quantized
---

# DeepSeek-V4-Flash — Arc UQFF (qtip2b)

Repository: **`aeonmind/DeepSeek-V4-Flash-UQFF-qtip2b`** (**public**)

A ~2.09 bits/param **qtip2b** quantization of DeepSeek-V4-Flash (284 B total /
13 B active), produced by [Arc](https://github.com/aeonmindai/arc) and
distributed in Arc/mistral.rs's **UQFF** format.

**`qtip2b` is the computed-codebook rung.** Where `qtip2` ships a 65,536 × 2
Gaussian lookup table inside the artifact, `qtip2b` derives its codebook from a
multiplicative congruential generator at decode time and stores no LUT tensor at
all. The two land within 0.002 bits/param of each other, so **the case for
`qtip2b` was never size** — it is that a computed codebook is what the grouped
trellis GEMM kernel needs.

> **This is not a standalone model, despite appearances.** The repository ships
> a `config.json` and a tokenizer, which makes it *look* self-contained. It is
> not: its only non-quantized weight file, `residual.safetensors`, is ~1.29 GB
> — embeddings and norms, nothing else. Everything else is either in the qtip2b
> shards or **not in this repository at all**. You must also have the **source
> DeepSeek-V4-Flash checkpoint** on disk; Arc builds the model from it and
> overlays the quantized layers from these shards. See
> [How to run it](#how-to-run-it).

---

## How to run it

### The binary

You need [Arc](https://github.com/aeonmindai/arc) built with CUDA. `qtip2b` is
an Arc quantization; an upstream mistral.rs build will not read these shards.

```bash
cargo build --release -p mistralrs-cli --features "cuda flash-attn"
```

> **Do not add the `cudnn` feature.** A same-box A/B on V4 measured it as a
> large decode regression, not a speedup.

🔴 **You need a build that carries the KV preallocation fix**
(`mistralrs-core/src/kv_cache/single_cache.rs`, merged 2026-08-16). Without it
V4 cannot complete a single prompt step — see
[Known limitations §1](#1-you-need-a-recent-arc-build-or-v4-will-not-generate-at-all).

### The command

```bash
# 1. Have the SOURCE checkpoint locally (config, tokenizer, weights).
#    <SOURCE_DIR> = the DeepSeek-V4-Flash model directory.
#
# 2. Have the FULL artifact locally: all 8 `qtip2b-N.uqff` shards
#    AND `residual.safetensors`, in one directory <UQFF_DIR>.
#
# 3. Run. Point -m at the SOURCE, --from-uqff at the FIRST shard.

mistralrs run \
  -m <SOURCE_DIR> \
  -a deepseekv4 \
  --from-uqff <UQFF_DIR>/qtip2b-0.uqff
```

Serving uses the same two flags:

```bash
mistralrs serve -p 1234 \
  -m <SOURCE_DIR> \
  -a deepseekv4 \
  --from-uqff <UQFF_DIR>/qtip2b-0.uqff \
  --chat-template chat_templates/deepseek_v4.json \
  --max-seqs <N>
```

**`--chat-template` is required for serving.** Without it
`/v1/chat/completions` returns 422.

**Shards auto-discover.** Naming `qtip2b-0.uqff` is enough; Arc finds
`qtip2b-1.uqff` … `qtip2b-7.uqff` next to it.

### The one error you are most likely to hit

```
Error: DummyLayer not replaced at index 1, layer Some(0) after load_from_artifacts
```

A quantizable layer never received its weights, so it is still the placeholder
Arc installs before deserialization
(`mistralrs-core/src/pipeline/isq.rs:1659`). The message names an index, not a
file, so it never tells you what is actually missing. Two causes, in order of
likelihood:

1. **`-m` points at the UQFF repo instead of the source checkpoint.** This
   repository is an overlay. `-m` must be the DeepSeek-V4-Flash **source**
   directory.
2. **The artifact set is incomplete.** All **8** `qtip2b-N.uqff` shards **and**
   `residual.safetensors` must be present.

---

## Quantization

| setting | value |
|---|---|
| Method | **qtip2b** (trellis-coded, **computed codebook**) |
| Trellis | K = 2 / V = 1, MCG-derived — **no codebook tensor in the artifact** |
| Search | **Viterbi beam, W = 256** |
| Objective | **MSE** (unweighted) |
| Rotation | **Hadamard-128** |

**`qtip2b` emits no bake-header log line.** The search cannot be verified from
the bake log, so it is verified from the **artifact** instead:
`Qtip2bLayer::serialize` appends `[stamp:u8][flags:u8]` (plus a `u16` beam width
when `FLAG_BEAM` is set) after the last tensor of each payload.
`arc-tools/quality/read_qtip_stamp.py` decodes it out of the UQFF container over
the safetensors header and per-tensor `data_offsets` — it never reads a whole
shard.

> ⚠️ A reader that takes "the last two bytes of each payload" is **wrong for a
> beam bake**: a beam writes two extra bytes, so the last two are the *width's*.
> At W = 256 that decodes as stamp = 0, which is reserved and invalid. Decode
> the tail **from the flags byte**.

**Greedy trellis search is banned in Arc (doctrine D4)** and the stamp scan is
the artifact-side confirmation that none was used.

---

## Hardware requirements

Same envelope as the `qtip2` artifact — the two are within 0.002 bits/param.

| | |
|---|---|
| Measured resident footprint, load only | **~75.9 GB of an 80 GB A100** |
| ⇒ Practical minimum | **≥ 96 GB** of VRAM |
| Comfortable | **141 GB H200** |

**An 80 GB A100 loads it and then has ~4 GB left.** That is enough to generate
at small batch and not enough for useful context or batching. Size for 96 GB
or more.

---

## Known limitations

### 1. You need a recent Arc build, or V4 will not generate at all

Between 2026-08-15 and 2026-08-16, **no** V4-Flash artifact of any rung could
complete a prompt step on Arc `master`. The engine preallocates a BF16
`[1, num_kv_heads, cap, head_dim]` KV buffer and installs it as
`SingleCache::all_data` *before the first append*, which collided with both of
V4's cache layouts:

* dense K + the 1-wide V marker → `shape mismatch on dim 3, 512 <> 1`
* the opt-in FP8 K code cache → `dtype mismatch in slice-set, lhs: BF16, rhs: U8`

Both are fixed (`single_cache.rs` now rebuilds a mismatched buffer while the
cache is still empty, and refuses a layout change once tokens exist). **If you
see either error, your Arc build predates the fix.**

FP8 K storage is **opt-in** and off unless `ARC_V4_FP8_KV=1`.

### 2. No throughput figures are published here

Arc's serving throughput on this rung has not been measured under a stated
protocol. Nothing about tokens/s, latency, or cost-per-token belongs on this
card until it has been. Do not infer performance from size or load time.

### 3. The V4 sparse indexer

On CSA layers this artifact may log an indexer shape mismatch and fall back to
dense-over-compressed attention. **The artifact is correct; Arc's loader was
wrong**, and the fix is entirely on the read side — no re-bake is required.
Generation is unaffected either way, because the loaded indexer is not read on
the current dispatch path.

---

## Provenance

* Base model: DeepSeek-V4-Flash (284 B total / 13 B active).
* Quantized by: [Arc](https://github.com/aeonmindai/arc) (a fork of
  [mistral.rs](https://github.com/EricLBuehler/mistral.rs)).
