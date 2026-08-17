# wave48-BY — the `qtip2b` artifact was baked. It is not published, and it does not serve.

**Branch** `perf/qtip2b-artifact` · **PR** #75 (draft, base `master`)
**Date** 2026-08-16 · **Box** `arc-s13-qtip2b`, 1×A100-80GB PCIe (sm_80), Montreal,
$1.49/hr · created **05:44:04Z**, self-destruct **ARMED 05:48:08Z**, deleted
**08:55:44Z** (3.13 h) · **SPEND $4.66** ($54.46 → $49.80), `runcrate ps` **empty**

**The bake worked and beat its projection. Two things after it did not.**

| stage | result |
|---|---|
| bake, `qtip2b` beam W=256, 43 layers | ✅ **211.3 s/layer, 2 h 40 m, rc=0** |
| search provenance read back from the artifact | ✅ **132/132 layers `trellis/beam(W=256)/mse`** |
| publish to `aeonmind/…-qtip2b` | ❌ **HF private-storage quota — 0 bytes of weights landed** |
| sanity check (3 prompts) | ❌ **every forward dies: `dtype mismatch in slice-set, lhs: BF16, rhs: U8`** |

**Nothing was published and nothing is claimed about quality.** The artifact
died with the box.

---

## 1. 🔴 The brief was wrong about `ARC_QTIP_BEAM`, and following it would have burned the session

The dispatch said, inheriting the claim from wave41-BS:

> **No codebook env var** — `ARC_QTIP_CODEBOOK`/`ARC_QTIP_BEAM` are read only by
> the `qtip2` rung's `QtipBakeConfig::get()` and would set variables this path
> never reads.

**Half true.** `ARC_QTIP_CODEBOOK` is genuinely unreachable here — `qtip2b` is
computed-codebook by construction, no LUT tensor exists. But **PR #74
(wave46-BX) gave `qtip2b` its beam and wired it to the same env var.** On
`f76b6af0a`, `mistralrs-quant/src/qtip/bitshift.rs:569`:

```rust
/// The **search axis**, read from `ARC_QTIP_BEAM` exactly like the LUT
/// rung's [`super::QtipBakeConfig`].
fn env_search() -> TrellisSearch {
    TrellisSearch::from_env()          // viterbi.rs:147 -> reads ARC_QTIP_BEAM
}
```

reached from `quantize_with_options` / `quantize_with_options_concrete`, which
is exactly where ISQ dispatch (`unquantized/mod.rs:438` →
`Qtip2bLayer::quantize_with_mode`) lands. **Unset, it returns `Exhaustive`**
(`viterbi.rs:150`) — which wave46-BX measured on this box class at
**8,257 s/layer ⇒ 98.6 h ⇒ ~$147.** The 6 h fuse would have fired around layer 2.

The bake therefore ran with `ARC_QTIP_BEAM=256`, and the value was verified
**from the running process**, not from the launch script:

```
# tr '\0' '\n' < /proc/35006/environ | grep -E '^ARC_|^MISTRALRS_'
ARC_QTIP_BEAM=256
ARC_QTIP_EXPERT_BATCH=8
MISTRALRS_ISQ_SINGLETHREAD=1
```

**Correction owed to the record:** wave41-BS §4 ("Beam W=256 is also a no-op on
this rung") and its FACTS descendant were true at `372976933` and are **false
from `f76b6af0a`**. FACTS already carries wave46-BX's retraction of
"un-bakeable"; this is the operational half of the same retraction — *the env
var is live on both rungs now, and it is now load-bearing.*

---

## 2. Bake pace — 43 differenced consecutive intervals

Marker: `Detected INT4-packed MoE expert weights`
(`distributed/layers.rs:1267`), one per MoE layer, emitted as the expert stack
is dequantized — so a marker-to-marker interval is one layer's whole quantize.

**Never a running average.** `(t_n − t_start)/n` folds layer 0's one-time init
into every layer; that is the error that produced the bogus 135 s/layer figure,
and it is why wave41-BS's ≥984 s was only a loose bound.

| interval | s | | interval | s |
|---|---|---|---|---|
| 0→1 | **218.0** *(carries init)* | | 20→21 | 213.0 |
| 1→2 | 218.0 | | 25→26 | 211.3 |
| 2→3 | 216.8 | | 30→31 | 210.1 |
| 3→4 | 209.8 | | 35→36 | 211.7 |
| 4→5 | 209.8 | | 40→41 | 210.7 |
| 5→6 | 212.7 | | 42→43 | 210.6 |

Full 43-row table in `pace.txt`. Range after the first gap: **208.4 – 218.0 s**.

**Measured steady-state median: 211.3 s/layer** against wave46-BX's
**213.2 s/layer** projection — **0.9% FASTER than projected.**

**Total bake: 2 h 40 m 03 s** (06:01:46Z → 08:41:49Z, `BAKE_RC=0`), 43 layers,
≈ **$3.98** of A100 time.

### The projection held, and that is the finding

wave46-BX measured **kernel throughput only** — rotation, LS refine, packing and
host↔device I/O excluded — and argued from its `qtip2` control (372.0 s/layer
measured vs 370–376 on the published end-to-end bake) that non-kernel overhead
is ≲1% at this scale. **This is the end-to-end test of that argument on the
`qtip2b` rung, and it came in 0.9% under.** The steady per-layer search time the
engine logged itself (`Quantized fused experts (Qtip2b) in 196.1–196.2s`) plus
the ~3.5 s expert unpack and per-layer overhead accounts for the 211.3 s.

⇒ **wave41-BS's 11.75 h / $57 → 2.67 h / $3.98. The rung we want to SERVE is
confirmed the cheapest rung to BAKE**, on hardware, end to end.

### Box health

100% GPU util, **148–212 W of 300 W** (49–71%, far above the 28.6% starvation
floor that caught s5a), SM clock **1410 MHz = full A100 PCIe boost**, 34–40 °C,
15.7 GB VRAM. **Zero** `fallback` / `panicked` / `CUDA error` / `out of memory`
lines across the whole bake — not wave6-Q's silent CPU reroute, and no
wave18-AO fragmentation OOM (it cleared layer 25, where the H200 bake once
died). `ISQ thread policy: 1 thread(s)`.

**Bootstrap was 11 minutes** — `BOOTSTRAP_COMPLETE 06:01:15Z` from a
05:44:04Z box creation, with the **159.63 GB source pull finishing in ~6 min**
(hf_xet high-performance) in parallel with the CUDA build.

> **Noticed:** `ARC_QTIP_EXPERT_BATCH=8` left **64 of the A100's 80 GB idle**
> (15.7 GB used). It is an OOM guard sized for a fragmentation failure seen on a
> *fuller* box. wave26-AX measured batch width at ~1% of bake time so it costs
> little, but the guard may be unnecessary post-#41. Worth a separate look.

---

## 3. Verifying the search WITHOUT a bake header

**`qtip2b` emits no `log_bake_header` line at all.** Confirmed on this bake:
`grep -c "QTIP bake" bake.log` → **0**. The LUT rung's gates ("ABORT-IF the
header is absent / says greedy / says the wrong width") have nothing to read.

**The artifact is the stronger instrument, and it is what was read.**
`Qtip2bLayer::serialize` (`bitshift.rs:2071-2088`) appends, after the last
tensor:

```
[stamp:u8][flags:u8]  (+ [beam_width:u16 LE] when flags & FLAG_BEAM)
```

`stamp` 1 = Trellis, 2 = Greedy (`mod.rs:1134`; 0 reserved and invalid so a
zero-filled buffer cannot read as valid); `flags` bit0 `FLAG_BEAM`, bit1
`FLAG_HESSIAN`, rest reserved-must-be-zero (`mod.rs:1263`). The payload head
carries `[version:u32][quant_type:u8]`, so one read proves the **rung** as well
as the **search**.

`arc-tools/quality/read_qtip_stamp.py` (new) decodes it straight out of the UQFF
safetensors container — header JSON, per-tensor `data_offsets`, then 14 bytes at
the head and 8 at the tail of each payload. It never reads a whole shard.

### 🔴 The wave41-BS reader would have mis-read a beam artifact

wave41-BS built a reader that takes **"the last two bytes of each layer
payload"**. Correct for an *exhaustive* bake, **wrong for a beam bake**: a beam
writes two more bytes, so the last two bytes are the *width's*. At W=256 that is
`0x00 0x01`, which decodes as stamp=0 — reserved and invalid — and at other
widths could decode as a plausible wrong answer. This reader decodes the tail
**from the flags byte**, the only thing that decides whether a width follows,
and refuses to guess.

### Result

```
STAMP_SCAN files=8 payloads=174 qtip2b=132 unquant=42
   132  trellis/beam(W=256)/mse
    42  (skipped) unquant
STAMP_OK all 132 qtip2b layers == trellis/beam(W=256)/mse (42 unquant skipped)
```

**Every quantized layer reads `trellis / beam(W=256) / mse`**: rung `Qtip2b`
(quant type 10), the search asked for, the width asked for, unweighted
objective. **D4 satisfied from the artifact, not from a log line.**

### The 42 `Unquant` payloads are structural — proven against the good artifact

The first run of the reader **failed**, calling 42 payloads an error
(`quant type 1 != Qtip2b`). Type 1 is `UnquantLinear` — a layer ISQ declined to
quantize, which carries no trellis and therefore no stamp. That was the
reader's bug, not the artifact's.

Rather than assume, the same read was run against the **published, verified-good
`qtip2` artifact** (GSM8K 96.0%, 0 degenerate, 0 truncated) over **HTTP range
requests** — headers plus 6 bytes per payload, no 74 GB download:

```
CONTROL repo=aeonmind/DeepSeek-V4-Flash-UQFF-qtip2 shards=8
CONTROL total=174 qtip_lut(8)=132 unquant(1)=42 other=[]
CONTROL unquant ordinals: 0,28,40,52,64,...,496,508
```

**Identical structure, identical ordinals** — 174 payloads, 132 quantized,
42 `Unquant` at exactly 0, 28, 40, … 508. The set is the model's shape, not a
qtip2b defect. Cost: ~1 min of box time.

### The reader is proven on both arms (D12)

`test_read_qtip_stamp.py` plants nine payloads with known answers:

```
PASS rc=0  beam W=256 (the artifact we want)
PASS rc=1  GREEDY stamp — D4 banned
PASS rc=1  exhaustive (no beam)
PASS rc=1  one greedy layer hidden in 3
PASS rc=1  beam at the WRONG width (64)
PASS rc=1  wrong rung (qtip LUT, type 8)
PASS rc=0  unquant passthroughs are skipped, not failed
PASS rc=1  greedy layer hidden among unquants
PASS rc=1  ALL unquant — nothing verified, must not pass
ALL ARMS PASS
```

The last three were added *after* the Unquant discovery: skipping a payload type
must not become a way to hide a bad layer, and an artifact with nothing to
verify must not pass by vacuity.

**D4 greedy ban intact** — untouched by this session; the stamp scan is the
artifact-side confirmation.

---

## 4. ❌ Publish blocked: HF private-storage quota

```
UPLOAD_PLAN files=14 shards=8 bytes=74120986972
UPLOAD_AUTH ok as heydryft
UPLOAD_FAIL upload_folder failed: BadRequestError:
  Private repository storage limit reached, please upgrade your plan to
  increase your private storage limit
```

**State on the Hub now:** `aeonmind/DeepSeek-V4-Flash-UQFF-qtip2b` exists,
private, containing **only `.gitattributes` (1,519 B)**. **No weights landed.**
The org holds exactly two model repos, and the existing 74.19 GB `qtip2`
artifact alone saturates the private quota; a second 74.12 GB artifact needs
~148 GB.

**This needs a decision that is not an agent's to make.** The three ways out:

1. **Raise the HF private-storage quota** (paid plan change).
2. **Publish `qtip2b` public** instead of private — public repos are not charged
   against private storage. This is a disclosure decision about model weights.
3. ~~Delete or replace the `qtip2` artifact~~ — **forbidden**, and correctly so.

No fallback was improvised, per the standing rule. The upload was not retried
against a different repo, a different owner, or a public visibility.

**The good artifact was not touched.** `aeonmind/DeepSeek-V4-Flash-UQFF-qtip2`
read from the HF API **before** the session and **after** teardown:

| | sha | lastModified | files | bytes |
|---|---|---|---|---|
| before | `108fbc2e3d…` | 2026-08-15T10:40:02Z | 15 | 74,190,197,268 |
| after | `108fbc2e3d…` | 2026-08-15T10:40:02Z | 15 | 74,190,197,268 |

Byte-identical.

> `s6_upload_uqff.py`'s allow-patterns are `*.uqff,*.json,*.txt,*.safetensors`,
> so **`README.md` — the model card — is silently excluded**, which is why its
> plan read 14 files against 15 on disk. The generated card also documents the
> **failing standalone form** (`-m aeonmind/…-qtip2b`), the same packaging
> defect FACTS already flags on the `qtip2` card. Both need fixing before any
> publish; neither was reached this session.

---

## 5. ❌ The artifact does not serve — and the evidence points away from the artifact

The UQFF **loads cleanly**: `Loaded in-situ quantization artifacts into 517
total tensors. Took 7.73s`, 75,863 MiB resident (matching FACTS' 75.7 GB for
`qtip2` on an A100). Then **every forward fails, including the engine's own
dummy run, 23 ms in**:

```
INFO  mistralrs_core: Beginning dummy run.
ERROR mistralrs_core::engine: prompt step - Model failed with error:
      dtype mismatch in slice-set, lhs: BF16, rhs: U8
```

All three sanity prompts returned `finish_reason: error`, 0 completion tokens.
**This is not the `device mismatch in matmul, lhs: Cuda, rhs: Cpu` the brief
predicted as benign.** It is a different, fatal error, and it is 100%
reproducible.

### Why this is probably NOT the qtip2b artifact

- **`slice_set` does not appear anywhere in the qtip2b weight path.** Every call
  site is in `mistralrs-core/src/kv_cache/`. The failing op is a **KV-cache
  append**, whose dtype logic is a function of attention geometry and has
  nothing to do with which QTIP rung the *weights* use.
- **U8 is the FP8 KV code cache.** wave43-BU (PR #72, merged `592eaf6f6`, before
  `f76b6af0a`) changed the K cache to store **448 E4M3 codes as U8** beside BF16
  `amax`/rope. `SingleCache::append` allocates `all_data` from the *first*
  `src.dtype()`, so a BF16 buffer meeting a U8 source is exactly a
  packed/dense disagreement inside that change.
- 🔑 **wave43-BU was done with no GPU at all** — its own words: *"**No GPU.**
  Every byte count below is read off source or driven through the real code on
  CPU; nothing here is measured on hardware."* The published V4 serving numbers
  (14.58 tok/s, GSM8K 96.0%) all predate it. **This session is plausibly the
  first V4 forward pass ever attempted on a commit containing the U8 KV cache.**

### The decisive test was NOT run, and I will not claim otherwise

The control that settles artifact-vs-serving-path is: **serve the known-good
`qtip2` artifact on this same binary and box.** If it fails identically, the
qtip2b artifact is exonerated. It was not run — the failure surfaced after the
upload had already failed, and holding a $1.49/hr box to download another 74 GB
while the publish was blocked on a human decision was not a defensible spend.
**So: the artifact is unproven, not proven bad.**

### The next session's first move is 15 minutes, not a re-bake

The code names its own triage switch. `deepseek4.rs:2377`:

> *Off when `ARC_V4_FP8_KV=0` (**on-GPU A/B triage**)*

⇒ **Serve with `ARC_V4_FP8_KV=0`.** If it generates, the artifact is good and
PR #72 is the bug. If it still dies, the artifact is implicated. Either way the
answer costs one short serve, and it should be run **before** anything is
re-baked.

---

## 6. Artifact facts (for whoever re-bakes or resumes)

| | `qtip2` (published) | **`qtip2b` (this bake)** |
|---|---|---|
| total bytes | 74,190,197,268 (74.19 GB) | **74,120,991,328 (74.12 GB)** |
| bits/param (284B) | 2.090 | **2.088** |
| shards | 8 + residual | 8 + residual |
| payloads | 174 (132 quantized + 42 unquant) | 174 (132 + 42) |
| trellis | K=4/V=2, 65,536×2 Gaussian LUT **in the artifact** | K=2/V=1, **computed MCG, no codebook tensor** |
| search | beam W=256 | beam W=256 |
| rotation | hadamard-128 | hadamard-128 |
| s/layer (A100) | 370–376 | **211.3** |

**The two rungs land within 0.002 bits/param of each other** (2.088 vs 2.090) —
`qtip2b` is marginally smaller because it carries no codebook tensor, and that
saving is almost exactly cancelled by its different scale/packing layout. **The
serving case for `qtip2b` was never about size; it is the grouped kernel.**

Recipe, for exact reproduction:

```bash
ARC_QTIP_BEAM=256 MISTRALRS_ISQ_SINGLETHREAD=1 ARC_QTIP_EXPERT_BATCH=8 \
mistralrs quantize text -m "$V4_DIR" -a deepseekv4 --isq qtip2b -o "$V4_DIR/uqff/" \
  --uqff-base-model deepseek-ai/DeepSeek-V4-Flash \
  --uqff-repo-id aeonmind/DeepSeek-V4-Flash-UQFF-qtip2b
```

---

## 7. Honest limits

- **No quality number exists for this rung.** GSM8K was deliberately not run
  (next session's job), and the three sanity prompts produced **no tokens at
  all** — so nothing whatsoever is known about `qtip2b` output quality on V4.
  wave46-BX's bit-exact GPU parity proves the *search* is correct; it says
  nothing about whether 43 layers of it compose into a good model, and this
  session did not find out.
- **The artifact is gone.** It lived on the box's ephemeral disk and was never
  uploaded. Re-deriving it costs ~2.7 h + ~$4.
- **The serving failure is diagnosed by static reading, not by experiment.** The
  attribution to PR #72 is circumstantial — strong (the op is in the KV cache,
  U8 is that PR's code cache, that PR never ran on a GPU) but **not measured**.
  §5's `ARC_V4_FP8_KV=0` A/B is the experiment that would settle it.
- **One box, one run, one silicon.** No repeat.
- **The 211.3 s/layer is a median of 42 steady intervals on one A100-80GB PCIe**;
  it is not an H200 number and must not be scaled to one without the
  kernel-specific ratio wave46-BX measured (beam 1.54×, exhaustive 3.03× — they
  differ, and quoting a single ratio is the trap that document flags).

---

## 8. Files

- `arc-tools/quality/read_qtip_stamp.py` — **new**, artifact-side search
  verifier (§3), with the `Unquant` handling the real artifact forced.
- `arc-tools/quality/test_read_qtip_stamp.py` — **new**, nine-arm D12 test.
- `arc-tools/quality/hf_repo_diff.py` — **new**, independent upload diff
  (written, but never got to run: the upload failed first).
- `memory/mission/wave48-BY-qtip2b-artifact.md` — this record.

No engine code changed.
