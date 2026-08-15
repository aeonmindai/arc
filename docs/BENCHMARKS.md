# Arc Benchmarks — DeepSeek V4 Flash on one H200

Measured results from six rented GPU sessions (2026-08-12/13/14/15). Every
number on this page comes from a saved eval artifact produced by the
`arc-tools/quality` harness; the one exception — the grouped-GEMM batch
curve — is a kernel-level microbenchmark and is labeled as such where it
appears. Where our protocol differs from a published reference, the
difference is stated inline.

## Setup

| | |
|---|---|
| Model | DeepSeek V4 Flash — 284B logical / 13B active MoE (HF-verified: model card + release announcement + config geometry, 43 layers / 256+1 experts) |
| Artifact | **74.19 GB** UQFF bake, **8 shards + residual** (15 files): 2-bit trellis (qtip2) experts; `lm_head` and the context compressor excluded from 2-bit. Published as `aeonmind/DeepSeek-V4-Flash-UQFF-qtip2`; size and file count **verified against the HF API** and again on disk. ⚠️ **Not standalone** — it is an overlay that requires the source checkpoint at `-m`; see the banner below |
| Hardware | 1× NVIDIA H200. Sessions 1–4: Runcrate **New York**, $4.92/hr. Sessions 5–6: Runcrate **Helsinki**, **$4.85/hr**. `nvidia-smi` reports **143,771 MiB = 150.75 GB decimal = 140.40 GiB**; NVIDIA's "141 GB" spec figure is really ~141 **GiB** (see the units note under session 5) |
| Engine | Arc (this repo), CUDA + flash-attn build; serve via OpenAI-compatible HTTP API |
| Sessions | Session 1: 2026-08-12T23:31Z, 9.2 h. Session 2: 2026-08-13T12:47Z (re-bake with the Viterbi encoder fix, PR #9). Session 3: 2026-08-13T20:31Z, ≈6.2 h (throughput session: all kernel-fix PRs #8/#10/#14/#15 in the build; token budget raised to 2048). Session 4: 2026-08-14T03:35Z, ≈3 h (payoff session: GEMV autotune sweep, grouped-GEMM batch curve, no-cudnn rebuild). Session 5: 2026-08-15T11:37Z (first end-to-end **batched serving** measurement; found the published artifact non-loadable and re-baked in situ). Session 6: 2026-08-15T21:08Z, ≈1.9 h (post-fix re-sweep on the **published** artifact: B up to 256, component profile at batch, GSM8K re-measure) |

## Headline results (latest measured session per row)

| Eval | Result | Protocol | Reference point |
|---|---|---|---|
| **GSM8K (session 6)** — on the **published** artifact | **96.0%** (96/100, ±3.8 pp CI95) | n=100, **0-shot chat**, greedy (t=0), **seed 161**, max_tokens **2048**; 1 degenerate, **0 truncated**, mean completion **148.5** tokens; arc `3460656d3` | Base model card: **90.8** — but that is **8-shot EM**, a different and **easier** protocol; **not directly comparable**. Meets DOCTRINE D6's ≥90 commitment on our own protocol for the first time |
| GSM8K (session 3) — **VOID, superseded** | 87.0% (87/100, ±6.6 pp CI95) | same protocol but a **different bake** (session-3 GPU-Viterbi, not the published artifact) and **superseded decode math** — PR #35 later fixed a missing SwiGLU clamp on 4 of 5 expert paths and a wrongly-applied YaRN layer set; 2 degenerate, 9 truncated, mean completion 528.4 tokens | Retired by session 6, not beaten by it: the two numbers are not measurements of the same thing |
| Factual recall (session 6) | **21/22** | greedy; on the published artifact, arc `3460656d3` | was 22/22 in session 2 — a one-item move on a small battery, reported as measured |
| Arithmetic (session 6) | **8/8** | greedy; on the published artifact | unchanged |
| Coherence battery (session 6) | **5/6** | t=1.0, p=0.95; on the published artifact | June anchor 6/6 (commit `6102b4d84`) — a one-item move, not distinguishable from noise at this n, but not an improvement either |
| Perplexity (session 2) | **12.50 ± 3.46** | 70 × 1024-token chunks, wiki mini-corpus (`wiki.test_small`) | See corpus caveat below |
| Long context (session 2) | **5/5 coherence + 4/4 needle** | greedy; ablation matrix below | — |
| Decode speed (b=1, session 4) | **14.58 tok/s** | single stream, 256 decode tokens, 525-token prompt; prefill ~57 tok/s, TTFT ~9.2 s; build **without** the `cudnn` feature (see pitfall below) | Progression 5.4 → 13.99 → 14.58 across the kernel-fix PRs; see kernel profile below |
| Grouped-GEMM batch curve (session 4) | **~63.5 ms/step flat from B=16 to B=64 ⇒ ~1,006 aggregate tok/s** on one H200 | **kernel-level microbench** of the batched 2-bit MoE expert path (40 MoE layers extrapolation), not end-to-end serving | **Still not reproduced end-to-end.** Session 6 measured **30.41 tok/s aggregate at B=64** through the server (up from session 5's 8.14) — closer, but still ~33× below this projection. See the batched-serving section |
| **Batched serving (session 6)** — the fleet number | **peak aggregate 30.65 tok/s, at B=16**, $43.96/Mtok; B=1 16.27, B=8 26.96, B=32 30.59, B=64 30.41, B=128 28.86, B=256 19.02 | end-to-end through the OpenAI-compatible server on the **published** artifact, `--max-seqs 256`, 64 decode tokens, `effective_B == B` on every row, 0 errors in 505 requests, $4.85/hr | **2.00× session 5's peak at half the $/Mtok, and the peak moved off b=1.** Aggregate stops collapsing but does not scale: flat ~30.5 across B=16–64. B=256 is host-CPU-bound (GPU 0–4%, 121 W) |
| Batched serving (session 5) — **superseded** | peak aggregate 15.35 tok/s, at B=1; B=8 14.83, B=16 10.31, B=32 5.07, B=64 8.14 | same probe, `--max-seqs 128`, in-situ W=32 bake, 0 errors in 121 requests | Aggregate **fell** with batch size. Retained as the pre-fix baseline |

> ### ✅ Aggregate throughput no longer falls with batch — session 6 [measured]
>
> Session 5 measured aggregate throughput **falling** as batch rose, peaking at
> b=1. Four fixes later (PRs #54/#56/#59/#46), measured on the **published**
> artifact through the same probe and the same 64-token protocol:
>
> ```
> B=1 16.27   B=8 26.96   B=16 30.65   B=32 30.59
> B=64 30.41  B=128 28.86  B=256 19.02      (tok/s aggregate)
> ```
>
> **Peak aggregate is 30.65 tok/s at B=16 — 2.00× session 5's peak, at half the
> cost per token ($43.96 vs $87.77 /Mtok @ $4.85/hr) — and the peak has moved
> off b=1 onto a real batch row.** The worst pre-fix row, B=32, improved
> **6.03×** (5.07 → 30.59). `effective_B == B` on all seven rows, every verdict
> `pass`, 0 errors in 505 requests. B=128 and B=256 ran for the first time.
>
> **What it does not do is scale.** Per-token cost falls 61.5 → 32.6 ms from
> B=1 to B=16 and is then flat within 1% across B=16/32/64 — the predicted
> consequence of PR #56 making the fused MoE path flat per token rather than
> amortizing. **B=256 regresses to 19.02**, and the cause was measured: the
> server pegged **one CPU core of 44 at 100% while the GPU idled at 0–4% and
> 121 W of 700 W**. That is host overhead, not bandwidth.
>
> Against the physics ceiling (`memory/mission/CEILINGS.json`, uniform-routing
> **lower bounds on the ceiling**): B=64 is 30.41 vs 5,289 (174×), B=256 is
> 19.02 vs 16,602 (873×). **Every one of those gaps is implementation, not
> physics** — and 3–6× of it closed in a single day of fixes.
>
> ### ⚠️ The published UQFF artifact is not standalone — session 5 [measured]
>
> `aeonmind/DeepSeek-V4-Flash-UQFF-qtip2` (74.19 GB, 15 files) **loads and
> serves — but only when `-m` points at the SOURCE checkpoint**, not at the
> artifact:
>
> | invocation | result |
> |---|---|
> | `-m <ARTIFACT dir> --from-uqff <artifact>/qtip2-0.uqff` | **fails** — `DummyLayer not replaced at index 1, layer Some(0)` |
> | `-m <SOURCE checkpoint> --from-uqff <artifact>/qtip2-0.uqff` | **works** — 3 m 10 s cold load, 75,859 MiB, generation 3/3 |
>
> The artifact is an **overlay of quantized experts**, not a self-contained
> model: `residual.safetensors` carries 575 tensors (embeddings, norms, router
> gates, compressor) and **zero** attention projections or shared-expert
> weights; the `.uqff` shards carry 171 quantized tensors against a ~512-slot
> graph. `paths.rs:365-386` decides which weight files are used from `-m`, so
> pointing it at the source supplies the rest.
>
> **The model card documents the failing form** (`-m <the artifact repo>`), so a
> user following it cannot load the model, and `DummyLayer not replaced at index
> 1` gives them nothing to act on. Either the packaging or the card needs to
> change, and the bake should assert at write time that the quantized set and
> the residual cover the model between them.
>
> ### GSM8K 87.0% is PROVISIONAL — the decode math changed after it was measured
>
> It was honestly measured under the protocol stated above (**n=100, 0-shot chat,
> greedy, seed 161, 2048-token cap**). It is provisional because **PR #35**
> (`830a41ed9`, merged after the session-3 run) changed the numerics the model
> decodes with:
>
> - A **SwiGLU clamp** (`swiglu_limit: 10.0`, published in the model's own
>   `config.json`) was missing on **4 of 5 expert paths** — *including the shared
>   expert, which every token traverses in every layer*. On a fixture where it
>   bites, clamped 0.7311 vs unclamped 14.8996 = **20.4×**.
> - **YaRN** rope scaling was being applied to ratio-0 layers that must not
>   receive it. The correct affected set is exactly **{0, 1, 43}**.
>
> Expected direction is neutral-to-better, but **that is unmeasured on the real
> model**. Until the eval is re-run on post-#35 math, 87.0% must be quoted with
> its vintage and never as the current quality of the shipped engine. The
> perplexity (12.50) and long-context rows below carry the **same vintage and are
> provisional for the same reason** — YaRN is a long-context effect, so the
> long-context results are the most exposed. See
> [engineering/OPEN_QUESTIONS.md §4](engineering/OPEN_QUESTIONS.md).

**Perplexity caveat:** 12.50 is measured on a 70-chunk wiki mini-corpus, **not**
the full wikitext-2 test set. It is NOT comparable to published full-wikitext
perplexities and we do not present it as such. Its role here is internal: the
same rung measured 58.85 (3 chunks, same pipeline) before the Viterbi encoder
fix — a 4.7× improvement into healthy-model range. A same-corpus GGUF `q2k`
comparison rung (session-1 mini-corpus value: 22.50 on 3 chunks of the same
text as the 58.85) has not yet been re-run on the 70-chunk corpus, so we make
no "beats q2k" claim yet.

### Session 1 → 2 → 3 → 4 ladder (why the numbers moved)

| Metric | Session 1 | Session 2 | Session 3 | Session 4 | What changed |
|---|---|---|---|---|---|
| GSM8K | 64.0% (32/50, ±13.3 pp; 4 degenerate, 33/50 truncated at a 640-token cap) | 84.0% (n=100; 0 degenerate, 17/100 truncated at 1024) | **87.0%** — **provisional** (n=100; 2 degenerate, 9/100 truncated at 2048) | not re-run | Session 2: PR #9 — the bake had run greedy trellis encoding (not Viterbi) on 3-D expert stacks and disabled the Hadamard rotation — fixed; token budget 640 → 1024. Session 3: token budget 1024 → 2048 (truncation was still costing points) |
| Decode (b=1) | 5.24 tok/s | 5.4 tok/s | 13.99 tok/s | 14.58 tok/s | Sessions 2→3: kernel-fix PRs #8/#10/#14/#15 (Sinkhorn fused default-on, absorbed-MLA decode, FP8 GEMV, dequant-materialize guard) compounding in one build. Session 4: rebuild **without** the `cudnn` feature (see pitfall below) |
| Perplexity (same rung) | 58.85 ± 24.76 (3 chunks) | 12.50 ± 3.46 (70 chunks) | not re-run | not re-run | Viterbi fix; wider corpus |
| Facts | 21/22 | 22/22 | not re-run | not re-run | — |

**The cudnn pitfall (session 4 root-cause):** building with
`--features "cuda flash-attn cudnn"` — the runbook's documented build line —
costs **−62% decode on V4** (5.45 vs 14.58 tok/s, measured back-to-back on the
same box and bake in session 4). Session 4's first speed probes reproduced only
5.4-class decode and the regression was root-caused to the `cudnn` flag;
rebuilding without it recovered and slightly bettered the session-3 number.
The working hypothesis (unverified) is that candle's `cudnn` feature swaps the
SDPA backend and bypasses flash-attn + the absorbed-decode fast path.
**Build without `cudnn`.**

## Long-context ablation matrix

Three serve configurations, same build, same eval (greedy). This validates the
PR #7 window fix: V4 Flash's "standard" layers (0/1/42 + the MTP block) must
run sliding-window-128 + attention-sink, not dense-causal, matching the
reference implementations.

| Config | Coherence | Needle recall | What it shows |
|---|---|---|---|
| `fixed` (windowed standard layers + compressed-KV branch) | **5/5** | **4/4** | Both mechanisms together: correct long-context behavior |
| `standard_dense` (`ARC_V4_STANDARD_DENSE=1`, pre-fix repro) | 2/5 | 4/4 | Dense-causal standard layers see relative distances >128 they were never trained on → coherence collapses; retrieval still works because the compressed branch is intact |
| `window_only` (`ARC_V4_WINDOW_ONLY=1`, compressed branch off) | 5/5 | 0/4 | Window + sink alone keeps generation fluent, but long-range retrieval is gone — the compressed-KV branch is what finds the needle |

Mechanism: the 128-token sliding window (+ sink) carries local fluency; the
compressed-KV branch carries long-range retrieval. Session 1 (pre-fix) scored
2/5 coherence on the normal config; the fix takes it to 5/5 with retrieval
intact.

## Format correctness gates (hardware-validated)

| Gate | Result |
|---|---|
| qtip2b bitshift-trellis CUDA ↔ CPU parity | **20/20 tests passed on H200** (quantize, dequantize, gather-GEMV, fused GEMV, 2-D and 3-D expert layouts, UQFF roundtrip) |
| Trellis grouped-GEMM (batched 2-bit MoE, Stage 4) | **5/5 parity tests passed on H200** (session 3 — first hardware run of the keystone kernel: deterministic output, CPU-reference match on ragged expert-group shapes, tile-map partitioning, descriptor table, window-state recurrence). Batched throughput curve measured at kernel level in session 4 — see below |
| Sinkhorn fused kernel vs reference path | **Token-identical** on 6/6 greedy 128-token prompts, and perplexity **bit-identical** — after the PR #8 identity fix. (Session 1, pre-fix: rejected with 4/6 token divergence — recorded, fixed, re-validated.) The fused path is now on by default (PR #15) |
| MLA absorbed decode vs non-absorbed | **Token-identical** on 6/6 prompts |

## Batched serving — the fleet-capacity measurement (session 6) [measured]

**Session 6 supersedes session 5 on every row below.** Four throughput fixes
landed between them (PRs #54 scheduler coalescing, #56 derived MoE gather
boundary, #59 rolling `xs` state, #46 computed codebook) and session 6 is the
first sweep taken on the **published** artifact rather than an in-situ bake.
**Aggregate throughput no longer falls with batch size**, and the peak is no
longer at b=1. Session 5's table is retained below it as the pre-fix baseline.

**Protocol.** 1× H200 @ **$4.85/hr** (Helsinki). arc **`3460656d3`**,
`cuda flash-attn`, no `cudnn`. Model: the **published**
`aeonmind/DeepSeek-V4-Flash-UQFF-qtip2` (Viterbi **beam W=256**, hadamard-128,
mse) overlaid on the source checkpoint. Server
`--max-seqs 256 --prefix-cache-n 0` with the V4 chat template. Probe
`arc-tools/quality/batch_load_probe.py`, `/v1/chat/completions` streaming,
distinct ~68-token prompts, **64 decode tokens, 1 rep** (session 5's protocol,
kept so the rows compare line by line), `--max-ctx 545000`, temperature 0.
Cold load **2 m 11 s**; post-load footprint **78,865 MiB of 143,771 MiB**;
peak resident at B=256 **114,665 MiB**.

| B | prefill agg tok/s (TTFT-derived, **lower bound**) | prefill agg tok/s (server-instrumented, **compute only**) | decode tok/s per user (p50) | **decode AGGREGATE tok/s** | TTFT p50/p95 (s) | $/Mtok @ $4.85/hr | effective_B |
|---|---|---|---|---|---|---|---|
| 1 *(diagnostic)* | 77.81 | 78.14 | 16.02 | **16.27** | 0.822 / 0.822 | 82.80 | 1 |
| 8 | 73.17 | 79.50 | 4.11 | **26.96** | 4.258 / 7.187 | 49.97 | 8 |
| 16 | 75.18 | 81.76 | 2.37 | **30.65** | 7.367 / 12.989 | **43.96** | 16 |
| 32 | 78.30 | 84.30 | 1.25 | **30.59** | 16.611 / 26.711 | 44.04 | 32 |
| 64 | 79.40 | 85.66 | 0.59 | **30.41** | 27.590 / 52.689 | 44.30 | 64 |
| 128 | 103.66 | 114.34 | 0.27 | **28.86** | 48.467 / 80.698 | 46.68 | 128 |
| 256 | 128.32 | 156.78 | 0.08 | **19.02** | 70.659 / 130.385 | 70.83 | 256 |

**Peak aggregate 30.65 tok/s at B=16, $43.96/Mtok @ $4.85/hr** — 2.00× session
5's peak at half the cost per token, and the peak has moved off b=1 onto a real
batch row. `effective_B == B` on all seven rows, every verdict `pass`,
**0 errors in 505 requests**, no `WARNING[KV]`, no `WARN[CLIENT]`.
**B=128 and B=256 ran for the first time** — they were 3.91× out of memory
before PR #59.

| B | session 5 (pre-fix, in-situ W=32 bake) | session 6 (post-fix, published W=256) | ratio |
|---|---|---|---|
| 1 | 15.35 | 16.27 | 1.06× |
| 8 | 14.83 | 26.96 | 1.82× |
| 16 | 10.31 | 30.65 | 2.97× |
| 32 | 5.07 | 30.59 | **6.03×** |
| 64 | 8.14 | 30.41 | 3.74× |
| 128 / 256 | *unreachable* | 28.86 / 19.02 | — |

Two uncontrolled differences ride along with those ratios: a different bake
(W=32 in situ vs published W=256) and a different rental. b=1 moved only 1.06×,
which is the evidence that the swap is not what produced the 3–6× at batch.

**Aggregate stops collapsing; it does not start scaling.** Per-token cost falls
61.5 → 32.6 ms from B=1 to B=16 (1.89× amortization) and is then flat within 1%
across B=16/32/64. That is the predicted consequence of PR #56: the fused MoE
path is **flat per token, not amortizing** (one GEMV per `(token, expert)` pair,
no dedup). The pre-fix doubling at the gather cap (97.0 → 197.2 ms) is gone.

**B=256 regresses, and the cause was measured, not guessed:** with
`256 running, 0 waiting`, the server sat at **100% of one CPU core (of 44)**
while the **GPU idled at 0–4% and 121–123 W** of a 700 W limit — host-bound,
GPU-starved. B=64 and B=128 ran the same box at 420–476 W and 97–100% GPU. The
client was idle, so this is not a harness limit.

**Refuted:** the CPU sampler fallback, which session 5 named as the prime
suspect for the collapse. The whole session logged exactly **2**
`GPU radix top-k sampling failed` lines, both from the startup dummy run and
none during any measured row — the probe sends `temperature=0`, whose greedy
path returns before the radix branch.

**Component profile at batch** (separate `ARC_TIME_DECODE=1` server; read shares,
not the sync-inflated ms). `moe` **35% at B=1 → 49–64% at B=64**, while
`mhc_attn_pre`/`mhc_ffn_pre` fall 10% → 4%. This supersedes the only prior
profile, which was b=1-only on an abandoned cudnn build and reported
`mla_attn 49% · moe 16%`. At fleet batch sizes MoE is the dominant term *and*
the one that does not get cheaper with batch.

**Against the physics ceiling** (`memory/mission/CEILINGS.json`; uniform-routing
assumptions make these **lower bounds on the ceiling**, not ceilings): B=16
30.65 vs 3,144 (103×), B=64 30.41 vs 5,289 (174×), B=256 19.02 vs 16,602 (873×).
Every one of those gaps is implementation, not physics.

---

### Session 5 — the pre-fix baseline, retained [measured, superseded]

The first end-to-end batched measurement through the real server. **b=1 is a
diagnostic row and is never the headline** — but on this build it is also,
unfortunately, the peak.

**Protocol.** 1× H200 @ **$4.85/hr** (Helsinki). arc `092b0967b`,
`cuda flash-attn`, no `cudnn`. Model quantized **in situ** from the source
checkpoint to qtip2 (`ARC_QTIP_BEAM=32`, `ARC_QTIP_EXPERT_BATCH=8`) — not the
published artifact (see banner). Server `--max-seqs 128 --prefix-cache-n 0`
with the V4 chat template. Probe `arc-tools/quality/batch_load_probe.py`,
`/v1/chat/completions` streaming, distinct ~68-token prompts, **64 decode
tokens, 1 rep**, `--max-ctx 118000`, temperature 0. Post-load footprint
**89,543 MiB of 143,771 MiB**.

| B | prefill agg tok/s (TTFT-derived, **lower bound**) | prefill agg tok/s (server-instrumented, **compute only**) | decode tok/s per user (p50) | **decode AGGREGATE tok/s** | TTFT p50/p95 (s) | $/Mtok @ $4.85/hr | effective_B |
|---|---|---|---|---|---|---|---|
| 1 *(diagnostic)* | 11.08 | 11.09 | 15.11 | **15.35** | 5.78 / 5.78 | **87.77** | 1 |
| 8 | 18.67 | 19.12 | 2.705 | **14.83** | 17.31 / 28.17 | 90.84 | 8 |
| 16 | 17.85 | 18.79 | 1.18 | **10.31** | 28.11 / 47.83 | 130.67 | 16 |
| 32 | 18.90 | 32.07 | 0.21 | **5.07** | 32.82 / 110.66 | 265.72 | 32 |
| 64 | 36.37 | 52.71 | 0.175 | **8.14** | 37.53 / 115.03 | 165.51 | 64 |

**Prefill and decode are never blended** — they have different bottlenecks and
both prefill columns carry their method label. Prefill is the good news:
server-instrumented prefill rises **4.8×** across the sweep, so the
compute-bound grouped-GEMM leg amortizes across a batch as designed. Decode
aggregate falls.

**Controls, because a probe that measures nothing still prints a number:**

- `effective_B == B` on all five rows; every concurrency verdict `pass`;
  **0 errors in 121 requests**. Concurrency is measured from per-request decode
  window overlap, not from how the client launched them.
- The probe's failure detector was mutation-tested **on this box**:
  `mutation(serial): probe exited 1, effective_B 1/4 — the assertion CAN fail`,
  and `mutation(cap 2 of 8): FAIL at default gate`.
- Not KV-bound: the guard put B=64 at ~8,448 tokens against a 118,000-token
  budget; no `WARNING[KV]` on any row.
- Not client-bound: no `WARN[CLIENT]`; 44 cores.
- Not a bad bake: b=1 measured 15.35 tok/s against the session-4 historical
  14.58 tok/s.

The engine's own scheduler corroborates independently — at B=8 it logs
`5 running, 3 waiting`; at B=64, `32 running, 32 waiting`. The requests do
overlap; they just slow down faster than concurrency grows.

**Against the roofline** (74.19 GB ÷ 4.8 TB/s = 15.46 ms/step ⇒ ~4,141 tok/s at
B=64): the measured B=64 aggregate is **~0.2%** of the bandwidth bound. Note the
roofline ignores KV traffic and assumes 100% of theoretical peak; at a realistic
70–85% it would be ~3,300–3,500 tok/s. Either way the served path is nowhere
near it.

> **This is a negative result and it is reported as one.** The footprint claim —
> V4-Flash on **one** H200 at 69.10 GiB where the smallest published artifact of
> any kind is 142.00 GiB — is measured and stands. The *throughput* half of the
> fleet-capacity thesis is not yet earned.

## Decode speed and kernel profile

Single-user (batch=1) decode is **14.58 tok/s** (session 4, best of 2 runs,
no-cudnn build; session-3 13.99; session-2 baseline 5.4; session-1 5.24 at
92% GPU utilization with the model fully resident in 82 GB — i.e.
kernel-bound, not memory-bound on capacity). The ×2.6 step from 5.4 to 13.99
came from the four kernel-fix PRs (#8 Sinkhorn bit-identity → fused
default-on via #15, #10 absorbed-MLA decode + GPU top-k, #14 FP8 blockwise
GEMV + dequant-materialize decode guard, #15 Sinkhorn default + MTP UQFF
coverage) compounding in one build; the step to 14.58 is the no-cudnn
rebuild. This is still an unoptimized state and we publish it as such.

### Gather-GEMV: pre-tune profile and the session-4 autotune sweep

Pre-tune gather-GEMV microbenchmark on the H200 (sessions 3/4, 2-bit trellis
expert weights, 2.11 MB/expert):

| Shape | b=1 | b=8 | Marginal kernel BW | Fixed overhead |
|---|---|---|---|---|
| gate/up N=2048 K=4096 | 153 GB/s (3.2% of peak) | 186 GB/s (3.9%) | 192 GB/s (4.0%) | ~18.7 µs/call |
| down N=4096 K=2048 | 159 GB/s (3.3%) | 184 GB/s (3.8%) | 189 GB/s (3.9%) | ~13.1 µs/call |

Session 4 ran the launch-parameter autotune sweep (43+ kernel variants over
warp layout, rows-per-warp, unroll, vector width, blocking, streaming
options). Best variants: **~36 µs at b=1 with 450–467 GB/s marginal
bandwidth (~9.5% of peak; the down-shape winner reached 484 GB/s / 10.1%) —
2.3× the pre-tune kernel.** The winning variants (v21 `w4_r2_i1_v2` for
gate/up, v6 `w8_r4_i1_v2` for down) are now baked in as the dispatch
defaults (PR #20), with a serve-path test pinning the winner selection.

Honesty note: session 4's "tuned" serve probe accidentally ran without the
winner table applied (an export-glob bug, fixed in PR #20), so the 14.58
tok/s above does **not** yet include the tuned dispatch. The end-to-end
decode number with tuned winners in the serve path is pending session 5.

### Grouped-GEMM batch curve (kernel-level microbench)

Session 4 measured the trellis grouped-GEMM batched expert path on the H200
(topk=6, 2-bit experts; extrapolated to a 40-MoE-layer step floor). **This is
an expert-path microbenchmark, not an end-to-end serving measurement** — it
excludes attention, routing, sampling, and engine overhead:

| B (batch) | Step floor (ms) | Aggregate tok/s |
|---|---|---|
| 1 | 14.2 | 70 |
| 8 | 52.5 | 152 |
| 16 | 63.5 | 252 |
| 32 | 63.6 | 504 |
| 64 | 63.6 | **1,006** |

The step floor goes **flat at ~63.5 ms from B=16 to B=64**: once essentially
every expert is activated each step, adding sequences costs nothing on the
weight-read side — the expert-read amortization the format was designed for,
demonstrated on silicon. At B=64 that is ~1,006 aggregate tok/s on one H200,
i.e. **≈$1.36/Mtok at the $4.92/hr rental rate** — as a kernel-level floor
figure. The full end-to-end batched-serving number (with attention + engine
overhead on top) is the headline gate for session 5; until it exists, treat
per-token economics from this curve as kernel-level, not serving-level (see
[FLEET.md](FLEET.md) for the tagged fleet-economics table).

## Reproduction

The full harness (eval clients, data fetcher, PPL runner, speed probe) lives in
`arc-tools/quality/` on branch `run161-quality-harness`; the step-by-step
session scripts are `arc-tools/quality/GPU_SESSION_RUNBOOK.md` (session 1),
`GPU_SESSION_RUNBOOK_2.md` (branch `session2-runbook`, PR #11), and
`GPU_SESSION_RUNBOOK_4.md` (sessions 3 and 4; session 3 ran a condensed
script derived from runbook 2).

Outline:

1. Rent 1× H200 (≥720 GB disk). **First check `nvidia-smi`'s max CUDA version
   against `nvcc --version`** — a driver/toolkit mismatch cost us the first
   bake attempt (`CUDA_ERROR_UNSUPPORTED_PTX_VERSION`).
2. `cargo build --release --features "cuda flash-attn"` — **without** `cudnn`
   (−62% decode on V4: 5.45 vs 14.58 tok/s, session 4).
3. `arc-tools/quality/fetch_data.sh` (model + eval data; ~149 GB download).
4. Bake the UQFF artifact (qtip2 experts + FP8 attention; Viterbi is the
   default expert encode mode since PR #9). Expect **~74 GB across 8 shards +
   residual** (the published artifact measures 74.18 GB / 15 files).
5. Serve on port 1234; run `run_coherence.py`, `run_gsm8k.py`, `run_longctx.py`
   (with the `ARC_V4_STANDARD_DENSE` / `ARC_V4_WINDOW_ONLY` gates for the
   ablation), `speed_probe.py`, `run_sinkhorn_ab.py`; run `run_ppl.sh` with the
   server stopped (PPL and serve are mutually exclusive on one GPU).
6. All evals must run against the **same bake** — never mix artifacts.

## Hardware and cost transparency

Total spend for every number on this page: **≈ $123** across four sessions on
a single rented H200 at $4.92/hr (session 1: 9.2 h ≈ $45; session 2: ≈ $32;
session 3: ≈6.2 h ≈ $30.5; session 4: ≈3 h ≈ $15). No lab cluster, no
reserved capacity.

## Limitations

- **Single-user speed is mid-optimization.** 14.58 tok/s at batch=1
  (no-cudnn build). The autotune sweep found 2.3×-faster GEMV variants and
  they are baked in as dispatch defaults (PR #20), but the 14.58 was measured
  **without** the tuned dispatch in the serve path (export-glob bug, fixed);
  the tuned end-to-end number is pending session 5. We publish the current
  number rather than a projection.
- **The grouped-GEMM batch curve is a kernel-level microbench.** The ~1,006
  aggregate tok/s at B=64 (and the ≈$1.36/Mtok arithmetic on it) is the
  batched expert-path floor, excluding attention, routing, sampling, and
  engine overhead. The end-to-end batched-serving measurement is planned for
  session 5 and is the number the fleet-economics claim will stand on.
- **MTP (multi-token prediction) acceptance is unmeasured.** The session-1/2
  UQFF artifacts did not carry the MTP decoder block. The bake/load fix has
  since merged (PR #15 — MTP now loads from UQFF or falls back to source
  weights) and session 3 validated the fallback load path (clean INFO, no
  crash). Session 4 found and fixed an off-by-one KV-cache desync in the MTP
  wrapper (the cache was over-truncated by one position on every rejected
  draft chain — PR #20), and also hit a new serve-path hang on the first MTP
  request (idle GPU; triage note in `docs/notes/mtp-hang-triage.md`, open
  hypothesis: verify-forward `xs_history` rollback). Acceptance measurement
  is re-queued for session 5 behind that diagnostic.
- **Server-side multi-sequence voting (Arc Boost) is blocked pending a fix's
  hardware validation.** Multi-chain vote requests crash in the model: the
  context compressor's `xs_history` was shared across sibling sequences in a
  batch (session-4 repro captured: `narrow` error on `[1, 2, 18, 512]`). The
  per-sequence history fix is PR #21; it has not yet been validated on
  hardware, so no voting-based quality numbers (the 90+ GSM8K target) are
  claimed anywhere on this page. Also note the chat endpoint requires
  `--chat-template chat_templates/deepseek_v4.json` — without it, requests
  422 (session-4 root cause; now in the runbook serve lines).
- **Perplexity corpus.** 12.50 is a 70-chunk wiki mini-corpus number — not
  comparable to full wikitext-2 results, and the same-corpus q2k comparison
  rung is still pending (see caveat above).
- **One model family validated so far.** Everything on this page is DeepSeek
  V4 Flash. Claims about other architectures are untested until they get the
  same treatment.
- **GSM8K is an n=100 subset** (±6.6 pp CI95) under a 0-shot chat protocol,
  seed 161, with a 2048-token budget (9/100 answers still truncated even at that
  cap, and 2 degenerate outputs — both counted as wrong); the published 90.8
  base-model figure is **8-shot EM**, a different and easier protocol. We state
  both because hiding either would flatter us.
- **…and it is provisional.** PR #35 (`830a41ed9`) changed the decode math
  after it was measured — a missing SwiGLU clamp on 4 of 5 expert paths
  (including the shared expert) and YaRN applied to ratio-0 layers. Direction is
  expected neutral-to-better and is **unmeasured**. The same applies to the
  perplexity and long-context rows. See the banner under the headline table.
- **TurboQuant KV was validated on a different model** (4.27× KV compression,
  Qwen3-32B on one H100, 39K→169K context, Apr 2026). MLA-attention models —
  including V4 Flash — currently fall back to the standard KV path, so the KV
  compression gain does not yet apply to the V4 numbers above.
