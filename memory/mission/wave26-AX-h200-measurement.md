# wave26-AX — the first end-to-end batched serving measurement, and it is bad news

**Session:** 2026-08-15, Runcrate H200 (Helsinki), instance `arc-s8-h200`
**Rate:** **$4.85/hr** — NOT the $4.92 of the New York boxes sessions 1–4 used.
Every `$/Mtok` below carries this rate.
**Spend:** **$15.80** (balance $100.73 → $84.93). Instance **DELETED**,
confirmed by `list_instances` returning `[]`.
**Binary:** arc `092b0967b` (master), `cuda flash-attn`, **no `cudnn`**.
**Card:** NVIDIA H200, `nvidia-smi` reports **143,771 MiB**, driver 580.126.09,
compute capability 9.0, 700 W limit; 44 vCPU, 178 GB RAM, 1.3 TB disk.

---

## 0. Headline: aggregate throughput FALLS as batch size rises

The product thesis is fleet capacity — more concurrent users per node. Measured
end-to-end through the real OpenAI-compatible server for the first time, **that
does not happen on this build.**

```
B=1    15.35 tok/s aggregate   (baseline, diagnostic row)
B=8    14.83 tok/s   0.97x
B=16   10.31 tok/s   0.67x
B=32    5.07 tok/s   0.33x
B=64    8.14 tok/s   0.53x
```

**Peak aggregate across the entire sweep is at B=1.** A single H200 serving 32
concurrent users delivers **one third** the total throughput of the same card
serving one user, at **3.0× the cost per token**. This is the opposite of the
capacity claim, and it is the number the thesis has to answer.

**The one thing that does work: prefill.** Server-instrumented prefill
throughput rises **4.8×** from B=1 to B=64 (11.09 → 52.71 tok/s). The
compute-bound leg amortizes across a batch exactly as designed. The
bandwidth-bound decode leg does not.

---

## 1. THE SPEED TABLE [measured]

**Protocol.** One H200 @ **$4.85/hr**. Model: DeepSeek-V4-Flash quantized
**in situ** from the source checkpoint to qtip2 (2-bit trellis),
`ARC_QTIP_BEAM=32`, `ARC_QTIP_EXPERT_BATCH=8` — see §2 for why this is not the
published artifact. Server: `--max-seqs 128 --prefix-cache-n 0`, chat template
supplied. Probe: `arc-tools/quality/batch_load_probe.py`, `/v1/chat/completions`
streaming, distinct ~68-token prompts, **64 decode tokens, 1 rep**,
`--max-ctx 118000`, temperature 0. Post-load footprint **89,543 MiB of 143,771**.

| B | prefill agg tok/s (TTFT-derived, **lower bound**) | prefill agg tok/s (server-instrumented, **compute only**) | decode tok/s per user (p50) | **decode AGGREGATE tok/s** | TTFT p50/p95 (s) | $/Mtok @ $4.85/hr | effective_B | verdict |
|---|---|---|---|---|---|---|---|---|
| 1 *(diagnostic only — never a headline)* | 11.08 | 11.09 | 15.11 | **15.35** | 5.78 / 5.78 | **87.77** | 1 | exempt |
| 8 | 18.67 | 19.12 | 2.705 | **14.83** | 17.31 / 28.17 | 90.84 | 8 | pass |
| 16 | 17.85 | 18.79 | 1.18 | **10.31** | 28.11 / 47.83 | 130.67 | 16 | pass |
| 32 | 18.90 | 32.07 | 0.21 | **5.07** | 32.82 / 110.66 | 265.72 | 32 | pass |
| 64 | 36.37 | 52.71 | 0.175 | **8.14** | 37.53 / 115.03 | 165.51 | 64 | pass |

`BATCHSWEEP[s8_sweep]: peak decode 15.35 tok/s @B=1 (effective_B=1) | errors 0`

**Every row ran at full requested concurrency.** `effective_B == B` on all five
rows, every concurrency verdict is `pass`, and there were **0 errors out of 121
requests**. No row is a silently-capped measurement.

### Why this is not an artifact of the harness

- **Not `--max-seqs`.** Set to 128 ≥ every B; `effective_B == B` everywhere.
- **Not KV pressure.** The probe's guard put B=64 at ~8,448 tokens against a
  118,000-token budget. No `WARNING[KV]` fired on any row.
- **Not the client.** No `WARN[CLIENT]`; the box has 44 cores.
- **Not a bad bake.** b=1 measured **15.35 tok/s** against the session-4
  historical **14.58 tok/s** — single-stream decode is at full expected speed.
- **Not the reduced beam width.** qtip2 is fixed-rate; W=32 and W=256 move
  identical bytes through identical kernels. The b=1 agreement with the
  W=256-era 14.58 tok/s is the evidence.
- **The probe's failure detector works.** Its mutation tests were run on this
  box: `mutation(serial): probe exited 1, effective_B 1/4 — the assertion CAN
  fail`, and `mutation(cap 2 of 8): FAIL at default gate`. A check that cannot
  fail proves nothing; this one can, and did not.

### Corroborated independently by the engine's own scheduler

```
B=8   -> "5 running, 3 waiting"     (probe mean in-flight 5.49)
B=16  -> "8 running, 8 waiting"     (probe mean in-flight 8.78)
B=32  -> "13 running, 19 waiting"   (probe mean in-flight 23.87)
B=64  -> "32 running, 32 waiting"   (probe mean in-flight 45.99)
```

The requests genuinely overlap — that is why `effective_B` passes. They simply
get slower faster than concurrency grows.

### Against the roofline

§5's bound is 15.46 ms/step ⇒ **~4,141 tok/s at B=64**. Measured B=64 aggregate
is **8.14 tok/s — about 0.2% of the bandwidth bound.** The grouped-GEMM
microbenchmark that projected **~1,006 aggregate tok/s at B=64** is **not
reproduced end-to-end, by two orders of magnitude.** Reconciling that gap is the
most important question this session raises: either the microbenchmark measures
something the served path never reaches, or the serving path has a defect
between the kernel and the scheduler.

> **Noticed:** the server logs `GPU radix top-k sampling failed; falling back to
> CPU: unexpected dtype, expected: U8, got: F32` at startup. A per-step CPU
> sampling fallback would serialize every decode step across the whole batch —
> exactly the "aggregate falls as B rises" signature measured here. Untested
> this session, and the single highest-value thing to check next.
> Worth a separate change?

---

## 2. The published UQFF artifact: it loads — but not the way its README says

**I got this wrong mid-session and am correcting it in full.**

`aeonmind/DeepSeek-V4-Flash-UQFF-qtip2` (15 files, **74.19 GB**, verified
against the HF API and again on disk) **does load and serve.** What it is not is
**standalone**: `-m` must point at the **source checkpoint**, not at the
artifact.

| invocation | result |
|---|---|
| `-m <ARTIFACT dir> --from-uqff <artifact>/qtip2-0.uqff` | **FAILS** |
| `-m <SOURCE checkpoint dir> --from-uqff <artifact>/qtip2-0.uqff` | **WORKS** |

### The failures, verbatim, with timestamps

```
11:55:50Z  Error: cannot find tensor model.layers.0.mlp.gate.tid2eid
12:00:27Z  Error: DummyLayer not replaced at index 1, layer Some(0) after load_from_artifacts
12:05:32Z  Error: DummyLayer not replaced at index 1, layer Some(0) after load_from_artifacts
```

The 12:05 attempt added `--isq-organization moqe`; identical failure. (For V4
that flag is inert anyway — `isq_layer_regexes_moqe` delegates verbatim to
`isq_layer_regexes`, `normal_loaders.rs:3320-3325`.) All 8 shards +
`residual.safetensors` were present on disk and
`Auto-discovered 8 UQFF shard files (from 1 specified)` fired every time, so
shard discovery was never the problem.

### Why `-m` decides it

`paths.rs:365-386` picks weight files as `if !safetensors.is_empty() {
safetensors } else if uqff_residual ... `. With `-m` at the artifact the only
match is `residual.safetensors`; with `-m` at the source it logs
`Found model weight filenames ["model-00021-of-00046.safetensors", ...]` and the
46 source shards supply everything the artifact does not carry.

And the artifact genuinely does not carry it. Parsing its headers directly:
`residual.safetensors` holds 575 tensors — embeddings, norms, router gates,
compressor, `hc_*` — and **zero** attention projections
(`q_a_proj`/`q_b_proj`/`kv_a_proj`/`kv_b_proj`/`o_proj`: 0), **zero**
shared-expert weights, and no `gate.tid2eid`. The `.uqff` shards hold 171
quantized tensors (~4/layer) against a ~512-slot model graph. **The artifact is
an overlay of quantized experts, not a self-contained model.**

### The working invocation, measured

```
mistralrs serve -p 1234 --max-seqs 128 --prefix-cache-n 0 \
  -m /mnt/models/v4src \
  --from-uqff /mnt/models/uqff/qtip2-0.uqff \
  --chat-template chat_templates/deepseek_v4.json
```

- load 14:52:47 → serving 14:55:57 (**3 m 10 s**, cold page cache), dummy run
  **0.22 s**
- footprint **75,859 MiB** — **13.7 GB less than the in-situ ISQ bake's 89,543**
- generation **3/3 sanity pass** (below)

### The finding that survives

**The artifact's own README documents the failing form.** It publishes:

```
mistralrs run -m aeonmind/DeepSeek-V4-Flash-UQFF-qtip2 --from-uqff qtip2-0.uqff
```

— `-m` at the artifact repo, which is exactly the shape that fails locally. A
user following the model card cannot load it. *(I tested the local-directory
form; I did not test the HF-repo form directly, and `paths.rs` treats them
through the same listing path.)* Two things are worth fixing regardless of which
form is "intended":

1. **The README is wrong or the packaging is.** A 74 GB artifact that silently
   requires a 159 GB source checkpoint alongside it is not a deployable
   artifact, and nothing says so.
2. **The error is unactionable.** `DummyLayer not replaced at index 1` gives a
   user no path to `-m must point at the base checkpoint`.

> **Noticed:** nothing asserts that the quantized set and the residual set cover
> the model between them at serialization time. A bake can emit a pair that only
> loads against a specific external base, and the failure surfaces as an opaque
> index. One assertion at write time would turn this into a clear message.
> Worth a separate change?

---

## 3. Generation sanity — this path had never emitted a token

Both configurations, 3/3, greedy:

| check | output |
|---|---|
| factual | `Paris` |
| arithmetic | `17 * 23 = 17 * (20 + 3) = (17*20)+(17*3) = 340 + 51 = 391` → **391** |
| fluency | *"The sky appears blue because sunlight is scattered in all directions by the gases and particles in Earth's atmosphere, with blue light being scattered more than other colors due to its shorter wavelength."* |

The bake's old post-generation smoke failure (`device mismatch in matmul, lhs:
Cuda, rhs: Cpu`) **did not reproduce**; the dummy run completed cleanly in both
paths (3.49 s in-situ, 0.22 s from UQFF).

---

## 4. Bake rate on H200, post-#40 [measured] — and a second mistake of mine

Differenced between **consecutive** `Quantized fused experts (QtipBitshift2) in
Xs` markers, not averaged:

| config | kernel s/layer | wall s/layer (steady state) |
|---|---|---|
| `ARC_QTIP_BEAM=256`, `ARC_QTIP_EXPERT_BATCH=16` (default) | **82.7 s** | **94–96 s** |
| `ARC_QTIP_BEAM=32`, `ARC_QTIP_EXPERT_BATCH=8` | **83.6 s** | **~95 s** |

Conditions: 1× H200; ISQ submission width **1 thread** (the GPU cap from PR #25,
`isq_thread_policy`); host-side expert unpack pool **44 threads**; dtype BF16;
`mode=viterbi search=viterbi-beam objective=mse (unweighted)
rotation=hadamard-128`; quantizing from the FP4/FP8 source checkpoint; 43 layers
≈ 68 min end to end.

**No speedup ratio is computed against the pre-#40 record (225.2 s kernel /
241 s per layer).** The configurations do not match on beam width, expert batch,
card, or source dtype. Both are stated; the differences are flagged; nothing is
divided.

**My mistake.** I first measured 212 s/layer at W=256 and concluded the beam
dominated bake time. That was the gap between the *first two* layers, which
carries one-time initialization. Differenced properly, **W=256 and W=32 are
82.7 s vs 83.6 s — about 1% apart. Beam width is very nearly free at bake time.**
I restarted at W=32 expecting ~2.2×; it bought nothing. The restart was still
necessary (the W=256 run OOM'd), but **the served model is a W=32 bake rather
than the quality-neutral W=256 of PR #29.** Throughput is unaffected; **no
quality number from this session should be read as the shipping bake's quality.**

`ARC_QTIP_BEAM` unset = the **exhaustive 2^L** search (`qtip/mod.rs:643-647`),
the ~6.2 h path. It must be set.

## 5. The bake OOMs at layer 25 with 72 GiB free

```
Error: DriverError(CUDA_ERROR_OUT_OF_MEMORY, "out of memory")
```

**Peak memory ever observed: 70,067 MiB (68.4 GiB) of 143,771 MiB** — it ran out
with roughly **72 GiB free**. That is allocator fragmentation after ~24 cycles of
allocating and freeing the ~12.9 GB BF16 expert staging buffer, not exhaustion.
Fixed with the documented mitigation (`bake_budget.rs:296`)
**`ARC_QTIP_EXPERT_BATCH=8`** (default 16); the rerun cleared layer 25 at
53.9 GiB and later exceeded the old peak without incident.

> **Noticed:** a bake that OOMs with half the card free will resurface on any
> larger model. Worth a separate change?

## 6. Box health, and the gate that could not run

S0's gate passed on structure (PCIe gen5 ×16, 44 cores, 1301 GB free) but its
**POWER check did not run** — a bare image has no torch/nvcc/cargo to build a
synthetic load, so the one check that catches the s5a failure was skipped.
Verified under **real** load instead:

| metric | reading | verdict |
|---|---|---|
| power under load | mean **237.4 W**, max **304.9 W** of 700 W (3,935 samples) | above the 200 W floor |
| SM clock | **1980 / 1980 MHz** | pinned at max |
| throttle reasons | all `Not Active` | none |
| temperature | 45 °C | cool |
| PCIe rx/tx | **1–2 MB/s** | not transfer-starved |
| memory util | 1% at 100% SM util | compute-bound, low-occupancy |

**The box was healthy.** This mattered: 3.5 min/layer initially looked exactly
like the s5a signature (~3 min/layer against a proven ~30 s/layer) until PCIe and
clocks were checked. Diagnosing the rental would have been wrong.

## 7. Units: "141 GB" is really ~141 **GiB**

`nvidia-smi` reports 143,771 MiB = **150.75 GB decimal / 140.40 GiB**. NVIDIA's
"141 GB" matches 141 **GiB** (151.40 GB decimal) to 99.6%; read as decimal GB it
would be 131.32 GiB and miss by 7%. **Never compare a byte-counted checkpoint to
the marketing "141 GB"** — the error runs in the direction that flatters the fit.
The task brief's "~59 GB usable" understated headroom for this reason; the real
figure after the 74.19 GB artifact is **~68.6 GB**.

Per-token cache cost, confirmed from the **served config** rather than memory —
the server logged `num_key_value_heads: 1`, `head_dim: 512`,
`num_hidden_layers: 43` (MQA):

| cache | formula | B/token |
|---|---|---|
| attention KV, 43 layers | `43 × 2 × 1 × 512 × 2 B` | 88,064 |
| compressor `xs` history, 41 layers | `41 × (4096+1) × 2 B` | 335,954 |
| **total** | | **424,018** |

| context | per-seq | max B in 68.6 GB |
|---|---|---|
| 512 (this sweep) | 0.217 GB | ~316 |
| 2048 | 0.868 GB | **~79** |
| 4096 | 1.737 GB | ~39 |

So B=128 fits at the sweep's context and B=64 fits at 2048; **B=128 at 2048-token
context does not** (~111 GB). None of this bound the measurement — the sweep
never came close to the KV budget.

## 8. Corrections to `GPU_SESSION_RUNBOOK_8.md`

Its serve command (lines 115–119) **cannot run**. Four independent defects:

| # | Runbook | Reality |
|---|---|---|
| 1 | no `-m` | clap rejects: `--model-id (-m) is required` — and `-m` is what decides whether the artifact loads at all (§2) |
| 2 | `--from-uqff /workspace/uqff` | must be the **first shard file**, not a directory (`isq.rs:301-328`) |
| 3 | no `--chat-template` | `/v1/chat/completions` 422s without it — exactly the endpoint the probe uses |
| 4 | `--label s8` on `run_gsm8k.py` / `run_coherence.py` | neither has `--label`; they take `--out`. Would have died **after** the sweep |

Also: `--max-ctx 153000` assumed a 68 GB artifact and a decimal-GB card (→ ~160k,
and 118k against the in-situ footprint actually used); `$4.92/hr` is New York's
rate, not this box's `$4.85`.

## 9. NOT DONE, and why

- **GSM8K was not run.** It is ~40 min ≈ $3.2, which would have taken the session
  past $19. It was also the *right* thing to cut: the served model is a **W=32**
  bake (§4), so a GSM8K number from it would not be a valid re-measure of the
  provisional 87.0% and would risk becoming a quoted figure. **87.0% remains
  PROVISIONAL and un-re-measured.**
- **Sustained/closed-loop (S4) was not run** — same budget reason. All rows are
  one-shot batches.
- **B=128 was not run.** It fits at this context, but the sweep already showed
  aggregate falling; the marginal value did not justify the spend.
- The sweep used **1 rep × 64 decode tokens**, trimmed from the runbook's
  3 × 256, because the measured throughput made the original protocol cost ~45
  min. Repeatability at the settings that were run twice was tight (B=1: 15.0 /
  14.95 / 15.35 across three independent runs).

## 10. Baseline — footprint, not a fabricated head-to-head

**No in-class single-H200 baseline exists, because no published V4-Flash artifact
fits one H200.** Verified against primary sources this session; anything
unverifiable was dropped.

| Artifact | bytes | GB (dec) | GiB | fits 140.40 GiB? |
|---|---|---|---|---|
| `deepseek-ai/DeepSeek-V4-Flash` (native FP4+FP8, 46 shards) | 159,617,149,040 | 159.62 | 148.66 | No |
| `canada-quant/…-W4A16-FP8` | 152,468,985,986 | 152.47 | 142.00 | No — smallest that exists, still 1.6 GiB over |
| `Intel/…-W4A16-AutoRound` | 155,590,474,496 | 155.59 | 144.90 | No |
| `RedHatAI/…-NVFP4-FP8` | 164,212,367,866 | 164.21 | 152.93 | No |
| `nvidia/…-NVFP4` | 168,281,985,176 | 168.28 | 156.72 | No (also Blackwell-only) |
| `EnsueAI/…-Base-INT4` | 186,443,831,288 | 186.44 | 173.64 | No |
| **Arc qtip2** | **74,190,000,000** | **74.19** | **69.10** | **Yes — 49% of the card** |

| Source | Smallest published H200 config |
|---|---|
| NVIDIA Dynamo `recipes/…/vllm/agg-h200-agentic/deploy.yaml` | **4× H200** (`--data-parallel-size 4 --tensor-parallel-size 1 --enable-expert-parallel`, `nvidia.com/gpu: "4"`) |
| LMSYS/SGLang day-0, 2026-04-25 | verbatim: *"B200 Pro (1.6T) at TP=8; H200 Flash (285B) at TP=4."* |
| `canada-quant` model card | verbatim: *"TP=2 is the only validated configuration. **TP=1 OOMs on a single 141 GB H200**."* |
| `nvidia/…-NVFP4` card | *"Supported Hardware Microarchitecture Compatibility: NVIDIA Blackwell"* |

Two notes kept deliberately: LMSYS says **285B** in one line and 284B in its own
TL;DR — **284B is authoritative** (DeepSeek's card + `config.json`), so the TP=4
line is quoted as written but 284B used in prose; and the canada-quant card
self-reports as no longer loading on current upstream vLLM as of 2026-05-25.

**What the footprint row proves — and what it does not.** It proves **node
capacity in memory terms**: the same model needs 4 H200s by published recipe, or
2 at absolute minimum, and Arc fits it on 1 at 69.10 GiB. A 4×H200 node is
**$19.40/hr** at this box's rate. It does **not** prove Arc serves that node's
traffic — §1 shows it does not, today. **The footprint claim is real; the
throughput claim is not yet earned.**

Rejected as arithmetically impossible: a "158 GB checkpoint fits on a single
141 GB H200" guide, and an unsourceable "INT4 V4-Flash on a single H200 at
~34 tok/s" (284B at 4 bits is a ~142 GB floor before KV). That family of
aggregator sources is unusable.

### Roofline — needs no third party, cannot be gamed

```
74.19 GB / 4.8 TB/s = 15.46 ms/step -> 64.7 steps/s
   B=64  ceiling ~ 4,141 tok/s aggregate
   B=128 ceiling ~ 8,281 tok/s aggregate
```

Caveats stated up front: per-stream stays 64.7 tok/s by construction (an artifact
of "touches all weights", not a finding); KV traffic is ignored, so this is a
genuine **upper** bound; and 4.8 TB/s is theoretical peak — real kernels land at
~70–85%, so a realistic target is **~3,300–3,500 tok/s at B=64**. Measured
today: **8.14**.

## 11. Raw markers

```
CONC[B=1]  effective_B=1  mean_inflight=1.0   server_prefill_batch=1  verdict=exempt
BATCH[B=1] decode agg 15.35 tok/s | per-user p50 15.11 (p95 15.11) | TTFT p50 5.776s (p95 5.776) | $/Mtok 87.77 @ $4.85/hr | errors 0/1
CONC[B=8]  effective_B=8  mean_inflight=5.49  server_prefill_batch=2  verdict=pass
BATCH[B=8] decode agg 14.83 tok/s | per-user p50 2.705 (p95 4.78) | TTFT p50 17.305s (p95 28.166) | $/Mtok 90.84 @ $4.85/hr | errors 0/8
CONC[B=16] effective_B=16 mean_inflight=8.78  server_prefill_batch=6  verdict=pass
BATCH[B=16] decode agg 10.31 tok/s | per-user p50 1.18 (p95 1.38) | TTFT p50 28.108s (p95 47.83) | $/Mtok 130.67 @ $4.85/hr | errors 0/16
CONC[B=32] effective_B=32 mean_inflight=23.87 server_prefill_batch=12 verdict=pass
BATCH[B=32] decode agg 5.07 tok/s | per-user p50 0.21 (p95 0.24) | TTFT p50 32.824s (p95 110.664) | $/Mtok 265.72 @ $4.85/hr | errors 0/32
KV[B=64]: est ~8448 tokens (B x (prompt~68 + decode 64))
CONC[B=64] effective_B=64 mean_inflight=45.99 server_prefill_batch=24 verdict=pass
BATCH[B=64] decode agg 8.14 tok/s | per-user p50 0.175 (p95 0.21) | TTFT p50 37.534s (p95 115.026) | $/Mtok 165.51 @ $4.85/hr | errors 0/64
BATCHSWEEP[s8_sweep]: peak decode 15.35 tok/s @B=1 (effective_B=1) | errors 0
```

Artifact: `arc-tools/quality/results/batch_load_s8_sweep.json`.
