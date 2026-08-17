---
tags:
  - uqff
  - mistral.rs
  - arc
  - qtip2
base_model: deepseek-ai/DeepSeek-V4-Flash
base_model_relation: quantized
---

# DeepSeek-V4-Flash — Arc UQFF (qtip2)

Repository: **`aeonmind/DeepSeek-V4-Flash-UQFF-qtip2`** (**public**)

A **2.09 bits/param** **qtip2** quantization of DeepSeek-V4-Flash (284 B total /
13 B active), produced by [Arc](https://github.com/aeonmindai/arc) and
distributed in Arc/mistral.rs's **UQFF** format.

> **Retraction.** Earlier revisions of this card said "~1.9 bits/param". That
> was arithmetic over a **68 GB estimate** of the artifact, not over the artifact.
> The measured size is 74.19 GB ⇒ **2.09 bits/param**. Do not quote 1.9.

> **This is not a standalone model, despite appearances.** The repository ships
> a `config.json` and a tokenizer, which makes it *look* self-contained. It is
> not: its only non-quantized weight file, `residual.safetensors`, is **1.29 GB**
> — embeddings and norms, nothing else. Everything else is either in the qtip2
> shards or **not in this repository at all**. You must also have the **source
> DeepSeek-V4-Flash checkpoint** on disk; Arc builds the model from it and
> overlays the quantized layers from these shards. See
> [How to run it](#how-to-run-it).

---

## How to run it

### The binary

You need [Arc](https://github.com/aeonmindai/arc) built with CUDA. qtip2 is an
Arc quantization; an upstream mistral.rs build will not read these shards.

```bash
cargo install --path mistralrs-cli --features "cuda flash-attn"
```

> **Do not add the `cudnn` feature.** A same-box A/B on V4 measured it as a
> large decode regression, not a speedup. The measurement is kept in Arc's
> internal record; no throughput numbers are quoted on this card (see
> [Known limitations §2](#2-no-throughput-figures-are-published-on-this-artifact)).

### The command

```bash
# 1. Have the SOURCE checkpoint locally (config, tokenizer, weights).
#    <SOURCE_DIR> = the DeepSeek-V4-Flash model directory.
#
# 2. Have the FULL artifact locally: all 8 `qtip2-N.uqff` shards
#    AND `residual.safetensors`, in one directory <UQFF_DIR>.
#
# 3. Run. Point -m at the SOURCE, --from-uqff at the FIRST shard.

mistralrs run \
  -m <SOURCE_DIR> \
  -a deepseekv4 \
  --from-uqff <UQFF_DIR>/qtip2-0.uqff
```

Serving uses the same two flags:

```bash
mistralrs serve -p 1234 \
  -m <SOURCE_DIR> \
  -a deepseekv4 \
  --from-uqff <UQFF_DIR>/qtip2-0.uqff \
  --max-seqs <N>          # defaults to 32; set it to your real max batch
```

**Shards auto-discover.** Naming `qtip2-0.uqff` is enough; Arc finds
`qtip2-1.uqff` … `qtip2-7.uqff` next to it and logs
`Auto-discovered 8 UQFF shard files (from 1 specified)`.

### What "success" looks like

```
Auto-discovered 8 UQFF shard files (from 1 specified)
... 517 tensors ...
Loaded in 12.94s
```

### The one error you are most likely to hit

```
Error: DummyLayer not replaced at index 1, layer Some(0) after load_from_artifacts
```

This means a quantizable layer never received its weights, so it is still the
placeholder Arc installs before deserialization
(`mistralrs-core/src/pipeline/isq.rs:1659`). The message names an index, not a
file, so it never tells you what is actually missing. Two causes, in order of
likelihood:

1. **`-m` points at the UQFF repo instead of the source checkpoint.** This
   repository is an overlay — see the file table below. `-m` must be the
   DeepSeek-V4-Flash **source** directory.
2. **The artifact set is incomplete.** All **8** `qtip2-N.uqff` shards **and**
   `residual.safetensors` must be present.

Check both before debugging anything else. This error cost the first two attempts
to load this artifact "the way a customer would" — including one attempt where
all 8 shards were correctly auto-discovered, so shard discovery was *not* the
cause.

---

## What is in the repository

Full listing, read from the HF API (not from the uploader's own report):

| file | bytes |
|---|---:|
| `qtip2-0.uqff` | 10,291,490,269 |
| `qtip2-1.uqff` | 10,357,567,511 |
| `qtip2-2.uqff` | 10,338,701,245 |
| `qtip2-3.uqff` | 10,338,701,245 |
| `qtip2-4.uqff` | 10,349,178,919 |
| `qtip2-5.uqff` | 10,338,701,245 |
| `qtip2-6.uqff` | 10,330,312,539 |
| `qtip2-7.uqff` | 541,597,883 |
| `residual.safetensors` | 1,293,806,700 |
| `config.json` | 1,749 |
| `generation_config.json` | 170 |
| `tokenizer.json` | 10,134,206 |
| `tokenizer_config.json` | 801 |
| `README.md` | 875 |
| `.gitattributes` | 1,911 |
| **15 files** | **74,190,197,268 (74.19 GB)** |

**All 9 weight files — the 8 shards and `residual.safetensors` — must be
present.** A partial download does not fail with a "missing file" message; it
fails with the `DummyLayer` error above.

Note the shape of that table: `residual.safetensors` is **1.7% of the bytes**.
It carries the tensors that were never quantized (embeddings, norms). It is not
a base model, and the presence of `config.json` + tokenizer does not make this
repository runnable on its own.

---

## Quantization

| setting | value |
|---|---|
| Method | **qtip2** (trellis-coded quantization) |
| Search | **Viterbi beam, W = 256** |
| Objective | **MSE** |
| Rotation | **Hadamard-128** |
| Effective rate | **2.09 bits/param** [derived from the measured 74.19 GB over 284 B params] |

The bake header emitted for this artifact was
`mode=viterbi search=viterbi-beam(W=256) objective=mse rotation=hadamard-128`.

Beam W = 256 is Arc's default and is what this artifact used. **Exhaustive
Viterbi is the better search on quality** — it wins **8 of 9 fixture cells**
(+0.0013…+0.0021 cos on `fp4_dequant`) — and it is roughly **2×** the bake time
(~510 s/layer against the beam's 241 s/layer on an H200, both [measured] pre-PR-#40).
**Beam is shipped knowingly for speed, at a declared quality cost**, not because
the two are equal. Do not restate this as "no quality cost".

*(An earlier revision added "beam width has almost no effect on bake time
(W = 256 vs W = 32 differ ~1%)". That is **unverified** and it conflicts with the
in-tree width sweep in
[QUANTIZATION_PERFORMANCE.md](../engineering/QUANTIZATION_PERFORMANCE.md#the-beams-architectural-ceiling--measured-not-projected),
where kernel time rises 410.9 → 1011.1 ms from W = 32 to W = 256. Treat the
width/bake-time relation as unsettled.)*

---

## Hardware requirements

| | |
|---|---|
| Resident footprint, load only | **75.7 GB of an 80 GB A100** [measured] |
| ⇒ Practical minimum | **≥ 96 GB** of VRAM [derived from the above] |
| Comfortable | **141 GB H200** (~59 GB left for KV after weights + reserve) [derived] |
| Load time | **12.94 s** for 517 tensors [measured] |

**An 80 GB A100 technically loads it and then has ~4 GB left.** That is not
enough KV cache for useful context or batching. Treat 80 GB as "it fits, you
cannot use it"; size for 96 GB or more.

---

## Known limitations

Read this section before relying on the artifact.

### 1. The V4 sparse indexer does not load from *this* artifact

On every CSA layer (2, 4, 6 … 42) this artifact logs:

```
V4 CSA layer N: indexer load failed (shape mismatch for
layers.N.attn.indexer.compressor.wgate.weight, expected: [256, 512], got: [256, 4096])
```

and the layer silently falls back to dense-over-compressed attention.

* **The artifact is correct; Arc's loader was wrong.** `[256, 4096]` is
  `[coff * index_head_dim, hidden_size]`, which is exactly what the reference
  publishes. Arc's indexer asked for `[256, 512]`
  (`[coff * index_head_dim, ratio * index_head_dim]`) because it fed the inner
  compressor grouped **K** instead of the layer's hidden states.
* **Generation is unaffected today**, because the loaded indexer is never read
  on the current dispatch path — CSA layers run dense-over-compressed either way.
* **The loader is fixed in Arc** (`mistralrs-core/src/models/dsv4_indexer.rs`;
  the indexer now shares the corrected `V4Compressor`). **No re-bake is
  required** — the fix is entirely on the read side.
* Until you are on an Arc build carrying that fix, the sparse indexer path is
  unavailable with this artifact and the warning above is expected.

### 2. No throughput figures are published on *this* artifact

Arc's end-to-end batched serving **has since been measured** (1×H200, 0 errors
across 505 requests) — but it was measured on the **`qtip2b`** rung, a *different*
artifact from the `qtip2` one this card describes. Those numbers live in
[BENCHMARKS.md](../BENCHMARKS.md) and belong to `qtip2b`; quoting them here would
attribute another artifact's measurement to this one.

So: nothing about tokens/s, latency, or cost-per-token is stated on this card.
Do not infer performance from the size or the load time, and do not import the
`qtip2b` figures.

*(Superseded: an earlier revision of this section said Arc's serving throughput at
batch was "currently poor and under active repair" and that nothing had been
measured. The second half is no longer true — see above.)*

### 3. One quality measurement exists on this artifact, on a small sample

**GSM8K = 96.0%** (96/100, ±3.8 pp), **0 degenerate, 0 truncated**, mean
completion 148.5 tokens.

* Protocol: **n = 100, 0-shot chat, t = 0, 2048-token cap, seed 161**, measured
  **on this artifact** (2026-08-15, 1×H200).
* **n = 100 is a small sample.** The ±3.8 pp is the binomial interval at that
  n; treat it as such. The full 1,319-problem set has not been run **on this
  artifact**.
* An earlier **87.0%** figure is **retired, not beaten**: it came from a
  different bake on superseded decode math (a missing SwiGLU clamp on the
  shared-expert path and a YaRN layer-set fix both landed after it). It is not
  a comparable baseline and no delta should be quoted against it.
* The published DeepSeek V4-Flash-Base reference figure of **90.8** is
  **8-shot** — a different and easier protocol. The two are not comparable.

**The full-set number belongs to the other rung.** GSM8K **1270/1319 = 96.3%
± 1.0 pp** (full test set, 0-shot, 0 degenerate / 0 truncated / 0 errors, mean
157.8 tokens) was measured 2026-08-17 on the **`qtip2b`** artifact — see
[BENCHMARKS.md](../BENCHMARKS.md) and
[the `qtip2b` card](deepseek-v4-flash-uqff-qtip2b.md). **It is not a
measurement of this card's bake**, and the two should not be conflated: this
card's 96.0% is n=100 on `qtip2`; that 96.3% is n=1319 on `qtip2b`.

### 4. This card supersedes an earlier auto-generated one

Until 2026-08-16 this repository carried the default UQFF card, whose example
was:

```
mistralrs run -m aeonmind/DeepSeek-V4-Flash-UQFF-qtip2 --from-uqff qtip2-0.uqff
```

**That command does not work** — it points `-m` at the overlay instead of the
source checkpoint and produces the `DummyLayer not replaced` error documented
above. Use the two-flag form in [How to run it](#how-to-run-it).

### 5. Bake-side caveats

* The post-bake smoke test ("Dummy run") **fails on every bake** with
  `device mismatch in matmul, lhs: Cuda, rhs: Cpu`. UQFF generation completes
  after the error, so the artifact is intact — but it means **this artifact was
  never validated by that check**. Generation was verified separately (3/3).
* The bake is buffered and written at the **end**; there is no partial resume.

---

## Evidence table

Every number on this card, with how it was obtained.

| claim | value | evidence |
|---|---|---|
| Repo file count / total bytes | 15 files, 74,190,197,268 B (74.19 GB) | Per-file sizes read from the HF API `?blobs=true` listing, 2026-08-15 (independently corroborates the earlier `missing vs local: NONE` check) |
| Shards | 8 × `qtip2-N.uqff` + `residual.safetensors` | same |
| `residual.safetensors` size | 1,293,806,700 B (1.7% of total) | same — this is the evidence the repo is not self-contained |
| Tensors restored | 517 | Load log, A100, 2026-08-15 |
| Load time | 12.94 s | Measured on the same A100 that baked it, 2026-08-15 |
| Resident on load | 75.7 GB of 80 GB | Measured, A100, 2026-08-15 |
| Bits/param | **2.09** [derived] | 74.19 GB × 8 over 284 B params. ⚠️ The former "≈ 1.9" is **retracted** — it divided a **68 GB estimate**, not the artifact |
| Bake config | beam W=256 / hadamard-128 / mse | Bake header string, read off the box |
| Bake cost | 43 layers @ 370–376 s/layer on a \$1.49/hr A100, completed 04:44:51Z 2026-08-15 [measured] | Differenced consecutive layer markers (never a running average) |
| Beam vs exhaustive | exhaustive wins **8 of 9** fixture cells; beam ships for speed [measured] | Fixture sweep in [QUANTIZATION_PERFORMANCE.md](../engineering/QUANTIZATION_PERFORMANCE.md); the quality cost is declared, not absent |
| Indexer shape mismatch | expected `[256,512]`, got `[256,4096]`, every CSA layer | Load log, this artifact |
| GSM8K 96.0% | 96/100, ±3.8 pp, 0 degenerate, 0 truncated [measured] | n=100, 0-shot chat, t=0, 2048-cap, seed 161, **on this artifact**, 1×H200, 2026-08-15 |
| GSM8K 87.0% | **SUPERSEDED** — do not quote as current | n=100, 0-shot chat, greedy, 2048-cap, seed 161; superseded decode math, different bake |
| GSM8K 96.3% (1270/1319, ±1.0 pp) | **[measured]** 2026-08-17, full test set — **belongs to `qtip2b`, not this artifact** | Reference model's 90.8 is **8-shot**, so it is **not** like-for-like. Do not conflate with this card's n=100 96.0% |
| Throughput (any form) | **not published on this artifact** | Batched serving is measured on **`qtip2b`** ([BENCHMARKS.md](../BENCHMARKS.md)), a different artifact. The `cudnn` warning above is a build-flag direction, deliberately stated without numbers |

---

## Provenance

* Base model: DeepSeek-V4-Flash (284 B total / 13 B active).
* Quantized by: [Arc](https://github.com/aeonmindai/arc) (a fork of
  [mistral.rs](https://github.com/EricLBuehler/mistral.rs)).
* Bake completed 2026-08-15 04:44:51Z on a single A100-80GB, 43 layers.

## License

Inherits the license of the base DeepSeek-V4-Flash checkpoint. The quantized
weights are a derivative of it.
