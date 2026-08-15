# DeepSeek-V4-Flash — Arc UQFF (qtip2)

Repository: **`aeonmind/DeepSeek-V4-Flash-UQFF-qtip2`** (currently **private**)

A ~1.9 bits/param **qtip2** quantization of DeepSeek-V4-Flash (284 B total /
13 B active), produced by [Arc](https://github.com/aeonmindai/arc) and
distributed in Arc/mistral.rs's **UQFF** format.

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
| Effective rate | **≈ 1.9 bits/param** |

The bake header emitted for this artifact was
`mode=viterbi search=viterbi-beam(W=256) objective=mse rotation=hadamard-128`.

Beam W = 256 is Arc's default and is what this artifact used. Exhaustive Viterbi
measures very slightly *better* on fixture quality (wins 8/9 fixture cells,
+0.0013…+0.0021 cos on `fp4_dequant`); beam is shipped knowingly because it is
faster at equal-or-near quality. Beam width has almost no effect on bake time
(W = 256 vs W = 32 differ ~1%), so there is no reason to bake narrower.

---

## Hardware requirements

| | |
|---|---|
| Measured resident footprint, load only | **75.7 GB of an 80 GB A100** |
| ⇒ Practical minimum | **≥ 96 GB** of VRAM |
| Comfortable | **141 GB H200** (~59 GB left for KV after weights + reserve) |
| Load time | **12.94 s** for 517 tensors |

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

### 2. No throughput figures are published here

Arc's serving throughput at batch is **currently poor and under active repair**.
Nothing about tokens/s, latency, or cost-per-token belongs on this card until it
has been measured on the published artifact under a stated protocol. Do not infer
performance from the size or the load time.

### 3. Quality numbers are provisional

The only quality measurement Arc has for a qtip2 V4-Flash bake is **GSM8K 87.0%**
— and it is **provisional**, because the decode math changed after it was taken
(a missing SwiGLU clamp on the shared expert path and a YaRN layer-set fix both
landed afterwards and both alter decode output). It has not been re-measured.

* Protocol for that 87.0%: **n = 100, 0-shot chat, greedy, 2048-token cap,
  seed 161**; 2 degenerate, 9 truncated; ±6.6 pp.
* It was measured on a *different* bake (session-3 GPU-Viterbi), **not on this
  artifact**.
* The published DeepSeek V4-Flash-Base reference figure of **90.8** is
  **8-shot** — a different and easier protocol. The two are not comparable.

### 4. The repository is private

It cannot be downloaded without access. Nothing on this card works until that
changes, and making it public is a publication decision, not a technical one.

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
| Bits/param | ≈ 1.9 | 74.19 GB over 284 B params |
| Bake config | beam W=256 / hadamard-128 / mse | Bake header string, read off the box |
| Bake cost | 43 layers @ 370–376 s/layer on a \$1.49/hr A100, completed 04:44:51Z 2026-08-15 | Differenced consecutive layer markers (never a running average) |
| Indexer shape mismatch | expected `[256,512]`, got `[256,4096]`, every CSA layer | Load log, this artifact |
| GSM8K 87.0% | **PROVISIONAL**, superseded math, different bake | n=100, 0-shot chat, greedy, 2048-cap, seed 161 |
| Throughput | **not published** | Not measured on this artifact |

---

## Provenance

* Base model: DeepSeek-V4-Flash (284 B total / 13 B active).
* Quantized by: [Arc](https://github.com/aeonmindai/arc) (a fork of
  [mistral.rs](https://github.com/EricLBuehler/mistral.rs)).
* Bake completed 2026-08-15 04:44:51Z on a single A100-80GB, 43 layers.

## License

Inherits the license of the base DeepSeek-V4-Flash checkpoint. The quantized
weights are a derivative of it.
