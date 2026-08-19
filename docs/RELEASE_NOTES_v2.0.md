# Arc v2.0 — Release Notes (DRAFT)

> **DRAFT — not released.** No tag exists, no artifact is public. Everything
> here is staged for review; the go/no-go steps and every item requiring
> founder approval are in
> [docs/notes/release-checklist.md](notes/release-checklist.md).

## What Arc v2 is

Arc v2.0 is the first release of Arc as a product rather than a fork-in-
progress: a Rust inference engine that serves a frontier MoE — DeepSeek V4
Flash, 284B logical / 13B active parameters — **on a single H200** from a
**74.18 GB artifact**, with the quality, correctness, and speed numbers measured
on rented hardware and published with their protocols
([docs/BENCHMARKS.md](BENCHMARKS.md)) and the fleet-economics case tagged
measured vs projected ([docs/FLEET.md](FLEET.md)).

The release is built on five pieces:

1. **Single-GPU 284B serving.** A 2-bit trellis expert bake (qtip2 family) +
   FP8 attention produces a 74.18 GB, 8-shard-plus-residual UQFF artifact
   (15 files, 2.09 bits/param, HF-API-verified) that loads and
   serves on one 141 GB H200 with the model fully resident. Quality on that
   artifact: **GSM8K 96.3% (1270/1319) ± 1.0 pp on the FULL test set** — 0-shot
   chat, greedy, seed 161, 2048-token cap, **0 degenerate / 0 truncated / 0
   errors**, mean completion 157.8 tokens — plus (session-6 latest) facts
   21/22, arithmetic 8/8, coherence 5/6, and long-context 5/5 coherence + 4/4
   needle with a published ablation matrix. (The earlier 87.0% on an n=100
   subset was measured under pre-PR-#35 decode math and is **superseded**; see
   the caveats section.)

2. **The qtip2b bitshift-trellis format.** A QTIP 2-bit rung whose codebook
   is *computed* (3 ALU ops from the trellis state, in registers) instead of
   gathered from a 512 KB LUT — same 2 bits/weight, zero codebook memory,
   quality tied with the LUT rung (cos 0.9654 vs 0.9657). CUDA ↔ CPU parity:
   20/20 tests on H200. The benchmarked V4 bakes use the trellis rung via
   `--isq qtip2`; qtip2b is the rung the batched-serving path (grouped-GEMM,
   autotuned GEMV) is built around.

3. **TurboQuant KV cache (K4/V3, 3.5-bit average) — shipped as the paged
   default; served once on hardware; compression ratio and quality still
   unmeasured.** Arc's Rust implementation of Zandieh et al. (ICLR'26) with a
   fused-kernel attention path.

   🔴 **Retraction, unchanged.** Earlier drafts described **"4.27× context
   capacity, measured end-to-end (Qwen3-32B on one H100, 39K → 169K tokens)"**.
   **4.27× is format arithmetic** — bytes per token at 3.5 bits versus BF16 —
   and **was never produced by a forward pass on any GPU**. Retained below only
   as **[projected]**.

   🔵 **Counter-correction (2026-08-17).** The retraction above used to carry a
   further sentence — *"No TurboQuant measurement exists anywhere in this
   repo's record, on any model"* — and **that was false**. Commit
   `4eba13905` (2026-04-06): **55 tok/s with TurboQuant on a B200**, correct
   output, serving Qwen3-32B through `deploy/modal_b200.py`. Eight CUDA
   correctness defects were found on that hardware on 2026-04-02. **What it
   covers:** b=1, one card, one model, head_dim 128, `Default` preset. **What
   it does not:** the "46% over Candle baseline" in the same commit compares
   Arc's whole dedicated decode path against Candle's and isolates nothing
   about TurboQuant; there is no A/B against an unquantized cache, and **no
   quality evaluation at any preset**. The ratio stays retracted; the run does
   not.

   Ship state: TurboQuant **is the paged default** at head_dim 128 on a
   standard KV layout (`defaults::PAGED_CACHE_TYPE`), and disables prefix
   caching while active; `--pa-cache-type auto` opts out. The *eager* path is
   the opt-in one (`ARC_TURBOQUANT_KV=1`). There is **no head_dim-512 kernel**,
   so **DeepSeek V4 Flash does not use TurboQuant**, independently of the fact
   that MLA models fall back to the standard KV path.

4. **Arc Boost (tier 1).** Serve-time quality orchestration: top-nsigma
   sampling, confidence-weighted voting, and reasoning-budget policy. The
   sampling and budget dials are active in v2.0; **server-side multi-chain
   voting is not enabled in this release** — the per-sequence compressor
   history fix it requires (PR #21) is awaiting hardware validation.

5. **Trellis grouped-GEMM (the batched-serving keystone).** The W2A16
   grouped GEMM that decodes 2-bit trellis experts for whole batches of
   token→expert pairs. Hardware parity 5/5 on H200, and the session-4 batch
   curve shows the step floor going **flat at ~63.5 ms from B=16 to B=64**
   **[measured-kernel]** — the expert-read amortization the format was designed
   around. Extrapolating that floor gives ~1,006 aggregate tok/s ⇒ ≈$1.36/Mtok
   at $4.92/hr **[projected]**. **The end-to-end measurement has since been
   taken and is far below the extrapolation: 91.46 tok/s aggregate at B=64,
   $14.73/Mtok, and a peak of 111.69 tok/s at B=256, $12.06/Mtok** — quote
   those for economics, never the kernel floor.

   The kernel choice itself is measured: on H200 the gemv path is **flat**
   (315 tok/s at B=64, 317 at B=128) while grouped GEMM **climbs** (322 at
   B=64, 527 at B=128) — crossover at **B=64** **[measured-kernel]**.

Plus the supporting cast: `arc validate --target-hbm` rental preflight, the
GEMV autotune sweep harness with the H200-tuned winners shipped as dispatch
defaults, the `arc-tools/quality` eval harness and GPU-session runbooks that
produced every number, and everything inherited from mistral.rs
(PagedAttention, FlashAttention, speculative decoding, 100+ architectures).

> **What FlashAttention does *not* mean here.** It is inherited and it works for
> inherited architectures — but **DeepSeek V4 Flash does not use it**. V4 sets
> attention sinks on all 43 layers, and the fused flash-with-sinks kernels
> accept `head_dim ∈ {64, 80, 96, 112, 128, 192, 256}`; V4's head_dim is
> **512**, so every V4 attention call runs the **unfused matmul +
> `softmax_with_sinks`** path. **There is no fused head_dim-512 attention kernel
> in this tree** — only feasibility probes under `arc-tools/fa4/`. Nothing in
> v2.0 ships one.
>
> **Expert parallelism does not exist either.** The communicator is
> `Comm::Dummy` with `world_size 1`. EP is a design; every number in this
> release is single-process, single-GPU.

## The numbers, each tagged

Taken on rented 1×H200 across eight GPU sessions, 2026-08-12 → 08-17
(sessions 1–4 at $4.92/hr; sessions 5–8 at $4.85/hr); protocols and artifacts in
[docs/BENCHMARKS.md](BENCHMARKS.md). Spend: **≈ $123** for sessions 1–4 and
**$16.11** for session 7; sessions 5, 6 and 8 are not itemized, so ≈$123 is
**not** the total behind this table.

**Type column:** *Measured* = ran end-to-end on hardware. *Measured (kernel)* =
a microbenchmark of one kernel or path, not the served engine. *Projected* =
arithmetic — never a measurement.

| Number | Value | Type |
|---|---|---|
| Model on one GPU | 284B/13B V4 Flash from a 74.18 GB artifact, 2.09 bits/param (2-bit trellis experts + FP8 attention) | Measured |
| **GSM8K** | **96.3%** — 1270/1319, the **FULL** test set, ±1.0 pp, 0-shot chat greedy, seed 161, 2048-token cap, **0 degenerate / 0 truncated / 0 errors**, mean 157.8 tokens. The base model card's 90.8 is **8-shot** — a different and **easier** protocol, and must be labelled as such wherever the two are shown together | Measured |
| Quality ladder | GSM8K 64 → 84 → 87.0 (n=100, pre-PR-#35 math) → 96.0 (n=100) → **96.3 (n=1319)**. The 87.0 rung is **superseded**, not merely re-run: PR #35 changed the decode math underneath it | Measured |
| Factual recall / arithmetic / coherence | 21/22 · 8/8 · 5/6 (session-6 latest; session 2 saw 22/22 and 6/6 — single-item moves on small batteries) | Measured |
| Perplexity (internal rung) | 12.50 ± 3.46 (70-chunk wiki mini-corpus; not comparable to full-wikitext numbers) | Measured — **pre-PR-#35 vintage, not re-run** |
| Long context | 5/5 coherence + 4/4 needle, with 3-config ablation isolating window vs compressed-KV mechanisms | Measured — **pre-PR-#35 vintage, not re-run** |
| **Decode speed (b=1)** | **18.27 tok/s aggregate / 17.99 per-user p50** (session 7, `qtip2b`, no-cudnn build; ladder 5.24 → 5.4 → 13.99 → 14.58 → **18.27**). The 14.58 figure is **superseded** | Measured |
| **Batched serving (peak)** | **111.69 tok/s aggregate at B=256, $12.06/Mtok** @ $4.85/hr; B=64 91.46 / $14.73. Aggregate rises **monotonically** to B=256; `effective_B == B` on all seven rows, **0 errors in 505 requests** | Measured |
| **Per-user decode under batch** | 17.99 (b=1) → 1.82 (B=64) → 1.09 (B=128) → **0.53 (B=256)** tok/s p50; TTFT p50/p95 27.8 s / 58.5 s at B=256. Aggregate throughput is bought with per-user latency and the table says so | Measured |
| **MTP yield** | `tok_per_step` 1.9326 (b=1) → 1.6680 (b=8) → 1.5492 (b=32) → **1.0558 (b=128)**: the speculative gain decays to ~nothing at fleet batch. **Never quote `accept_rate`** — it holds ~0.43 at b=128 and flatters | Measured |
| qtip2b format correctness | CUDA ↔ CPU parity 20/20 on H200 | Measured |
| Grouped-GEMM correctness | Parity 5/5 on H200 (first hardware run) | Measured |
| GEMV autotune | Best variants ~36 µs / 450–467 GB/s (~9.5% of peak, 2.3× pre-tune); shipped as dispatch defaults | Measured (kernel) |
| MoE kernel crossover | gemv **flat** 315 → 317 tok/s (B=64 → B=128); grouped GEMM **climbs** 322 → 527. Crossover at **B=64** | Measured (kernel) |
| Grouped-GEMM batch curve | Flat ~63.5 ms/step B=16→64 | Measured (kernel; expert path only) |
| …its tok/s and $/Mtok | ~1,006 aggregate tok/s ⇒ ≈$1.36/Mtok at $4.92/hr | **Projected** — arithmetic on the step floor; the served figures are 91.46 tok/s / $14.73/Mtok at B=64 |
| TurboQuant KV — compression | 4.27× context (Qwen3-32B, 1×H100, 39K → 169K) | 🔴 **Projected — retracted as measured.** Format arithmetic (3.5 bits vs BF16 bytes/token); **no run has ever produced a KV compression figure** |
| TurboQuant KV — serving | 55 tok/s, correct output (Qwen3-32B, 1×B200, b=1) | 🔵 **Measured** — `4eba13905`, 2026-04-06, harness `deploy/modal_b200.py`. Narrow: one card, one model, head_dim 128, `Default` preset, no A/B isolating TurboQuant |
| TurboQuant KV — quality | — | 🔴 **Never measured** at any preset or width. The paper's "lossless" LongBench result is Zandieh et al.'s, on their model |

## Known limitations

Stated in full in [docs/BENCHMARKS.md](BENCHMARKS.md) § Limitations; the
short version:

- **b=1 speed is mid-optimization.** 18.27 tok/s aggregate / 17.99 per-user p50.
  No isolated end-to-end A/B of the autotuned GEMV dispatch has been run, so
  none of that number is attributed to it. Building with the `cudnn` feature
  costs −62% decode on V4 — release binaries must be built without it.
- **The served path is overhead-bound, not bandwidth-bound.** Peak aggregate
  111.69 tok/s is a low single-digit percentage of the H200's 4.8 TB/s. One
  measured contributor: GPU radix top-k sampling falls back to CPU on **every
  token** (`tensor_device_ptr: unsupported dtype I32`, ~10 lines/sec) — a
  device→host round trip per step. The headroom here is implementation, not
  physics.
- **The $1.36/Mtok figure is kernel-level arithmetic, and reality is ~11×
  worse.** It extrapolates the expert-path microbenchmark. The end-to-end
  measurement exists now: **$14.73/Mtok at B=64, $12.06/Mtok at B=256**. Fleet
  economics must be quoted from those, not from $1.36.
- **Per-user decode collapses as batch rises.** 17.99 tok/s at b=1 falls to
  **0.53** at B=256, with TTFT p95 of 58.5 s. The aggregate peak is a
  capacity number, not a user-experience number, and must not be presented as
  one.
- **Server-side voting (Arc Boost multi-chain) is off.** Multi-chain
  requests crash on a shared compressor history (`xs_history`); the
  per-sequence fix (PR #21) awaits hardware validation. No voting-based
  quality numbers are claimed.
- **MTP yield is measured, and it does not survive batch.** `tok_per_step`
  1.9326 at b=1 decays to **1.0558 at b=128** — essentially no speculative gain
  at fleet batch sizes. Report `tok_per_step`, never `accept_rate` (which holds
  ~0.43 at b=128 and flatters, because saturated sequences draft nothing and
  contribute `proposed=0`). Reliability is the blocker, not the metric: a
  KV-cache desync was found and fixed, a serve-path hang on the first MTP
  request remains under triage (`docs/notes/mtp-hang-triage.md`), and session 7
  hit a hard panic at B=8. **Do not enable the MTP speculative path in
  production in v2.0.**
- **Two engine panics on the ordinary decode path** (no MTP) were observed in
  ~1,300 long-generation requests at `--concurrency 16`. The engine recovered
  both times, but this is a serving-reliability defect on the default path.
- **One model family validated.** Every V4 number is DeepSeek V4 Flash;
  other architectures inherit mistral.rs support but have not had the same
  measurement treatment. **TurboQuant KV's 4.27× is not a measurement on any
  model** (see the retraction above) — and V4 does not use TurboQuant anyway:
  no head_dim-512 kernel exists. TurboQuant's one hardware result is on a
  *different* model family entirely (Qwen3-32B on a B200), which is exactly why
  it must not be read as a V4 number.
- **GSM8K is the full 1,319-problem test set**, 0-shot chat, seed 161,
  2048-token cap, 0 degenerate / 0 truncated / 0 errors. The published
  base-model 90.8 is **8-shot EM**, a different and easier protocol — state
  that every time the two appear together. Both stated because hiding either
  would flatter us.
- **GSM8K is no longer provisional; perplexity and long context still are.**
  PR #35 (`830a41ed9`) changed the decode math after the session-3 evals: a
  SwiGLU clamp was missing on 4 of 5 expert paths — including the shared
  expert, which every token traverses in every layer — and YaRN was applied to
  ratio-0 layers (the correct set is exactly {0, 1, 43}). **GSM8K has since
  been re-measured twice on post-#35 math (96.0% n=100, then 96.3% n=1319), so
  its provisional banner is retired.** Perplexity (12.50) and the long-context
  rows have **not** been re-run and still carry the pre-#35 vintage — YaRN is a
  long-context effect, so long context is the most exposed of the two.

## Upgrade notes

- **UQFF 0.3.0 — QTIP artifacts now record which trellis search baked them,
  and old ones may be refused.** Every QTIP payload (both the `qtip2` LUT rung
  and the `qtip2b` bitshift rung) carries a search-provenance stamp, and the
  loader checks it. This closes a real hole: nothing in the 0.2.x format said
  whether a bake used the full trellis search or the greedy walk, so a
  low-quality artifact was indistinguishable from a good one at load time —
  and greedy costs matmul cosine 0.675 vs 0.963 on the FP4-lattice experts V4
  uses, while also disabling the Hadamard incoherence rotation.

  What this means for existing artifacts:

  | artifact | behaviour under v2.0 |
  |---|---|
  | baked by v2.0 (stamped `trellis`) | loads normally |
  | pre-0.3.0, **with** incoherence rotation | loads, with a one-time warning that its search cannot be verified from the file |
  | pre-0.3.0, **without** incoherence rotation | **refused.** Under every bake policy shipped so far, rotation was on if and only if the trellis search ran, so such a file is a greedy bake. Re-bake with `mistralrs quantize`, or set `ARC_ALLOW_UNSTAMPED_QTIP=1` to load it anyway for diagnostics. |
  | stamped `greedy` | **refused, no override** |

  The stamp is accompanied by a one-byte **search-detail** field recording
  *which* trellis search ran: the beam width (`ARC_QTIP_BEAM`, 0 meaning the
  exhaustive `2^L` dynamic program) and whether the activation-Hessian
  objective was used. A beam is a real quality trade — `W = 64` measures matmul
  cosine 0.951 against the exhaustive DP's 0.965 — so an artifact records which
  one produced it rather than leaving the two indistinguishable. Reserved bits
  in that field, a width outside `1..65536`, and a `greedy` stamp carrying a
  beam width are all **refused rather than normalised**: a self-contradictory
  provenance claim never gets read as a plausible one. The field is mandatory
  wherever the stamp is present, so a truncated 0.3.0 payload fails closed
  instead of reading as an exhaustive unweighted bake.

  If you baked a `--isq qtip2b` MoE artifact with a build before this release,
  **re-bake it**: that path selected the greedy walk for 3-D expert stacks, so
  the artifact is at the degraded quality even though nothing in the log or the
  file said so. Greedy is no longer selectable at all — the
  `ARC_QTIP_EXPERT_GREEDY` environment variable has been removed, and a bake
  handed the greedy mode hard-errors rather than warning.

  Version mechanics: this is a **minor** bump, so pre-0.3.0 builds now refuse
  v2.0 artifacts cleanly ("newer than this build supports") instead of reading
  them and ignoring the stamp; 0.3.0 reads every 0.1.x/0.2.x artifact subject
  to the table above. Rank-3 stacked-expert (MoE) QTIP payloads arrived in
  0.2.1 and **readers older than 0.2.1 mis-decode them rather than failing**,
  so pre-v2.0 builds must not be pointed at v2.0 artifacts either way.
- **Breaking: the `qtip2` trellis parser/naming moved to the `bitshift`
  family.** The QTIP trellis implementation now lives under
  `qtip::bitshift`; in the Rust API the 2-bit rung is
  `IsqType::QtipBitshift2` and the computed-codebook rung is
  `IsqType::Qtip2b` (UQFF payload tag 10) — code that matched the old
  `Qtip2`-style names or pre-bitshift module/kernel paths must be updated.
  On the CLI, `--isq qtip2` (alias `qtip`) still parses and selects the
  2-bit trellis rung; `--isq qtip2b` selects the computed-codebook bitshift
  rung used by the v2.0 production bakes.
- **Build flags.** CUDA builds: `--features "cuda flash-attn"` — do **not**
  add `cudnn` (−62% decode on V4, measured). The baked GEMV dispatch
  defaults are the H200-measured sweep winners; on other GPUs, re-run the
  `qtip_gemv_tune` sweep and point `ARC_QTIP_TUNE_TABLE` at the produced
  winner table (it overrides the baked defaults), or force
  `ARC_QTIP_GEMV_VARIANT=legacy`.
- **Serving V4 Flash.** Pass `--chat-template chat_templates/deepseek_v4.json`
  — the chat endpoint rejects requests (HTTP 422) without it.
