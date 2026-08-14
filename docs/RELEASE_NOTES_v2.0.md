# Arc v2.0 — Release Notes (DRAFT)

> **DRAFT — not released.** No tag exists, no artifact is public. Everything
> here is staged for review; the go/no-go steps and every item requiring
> founder approval are in
> [docs/notes/release-checklist.md](notes/release-checklist.md).

## What Arc v2 is

Arc v2.0 is the first release of Arc as a product rather than a fork-in-
progress: a Rust inference engine that serves a frontier MoE — DeepSeek V4
Flash, 284B logical / 13B active parameters — **on a single H200** from a
**68 GB artifact**, with the quality, correctness, and speed numbers measured
on rented hardware and published with their protocols
([docs/BENCHMARKS.md](BENCHMARKS.md)) and the fleet-economics case tagged
measured vs projected ([docs/FLEET.md](FLEET.md)).

The release is built on five pieces:

1. **Single-GPU 284B serving.** A 2-bit trellis expert bake (qtip2 family) +
   FP8 attention produces a 68 GB, 7-shard UQFF artifact that loads and
   serves on one 141 GB H200 with the model fully resident. Quality on that
   artifact: GSM8K 87.0% (n=100, 0-shot chat greedy, ±6.6 pp), facts 22/22,
   arithmetic 8/8, coherence 6/6, long-context 5/5 coherence + 4/4 needle
   with a published ablation matrix.

2. **The qtip2b bitshift-trellis format.** A QTIP 2-bit rung whose codebook
   is *computed* (3 ALU ops from the trellis state, in registers) instead of
   gathered from a 512 KB LUT — same 2 bits/weight, zero codebook memory,
   quality tied with the LUT rung (cos 0.9654 vs 0.9657). CUDA ↔ CPU parity:
   20/20 tests on H200. The benchmarked V4 bakes use the trellis rung via
   `--isq qtip2`; qtip2b is the rung the batched-serving path (grouped-GEMM,
   autotuned GEMV) is built around.

3. **TurboQuant KV cache (K4/V3, 3.5-bit average).** Arc's Rust
   implementation of Zandieh et al. (ICLR'26) with a fused-kernel attention
   path. Measured end-to-end: **4.27× context capacity** (Qwen3-32B on one
   H100, 39K → 169K tokens). GQA-attention models; MLA models (including V4
   Flash) currently fall back to the standard KV path.

4. **Arc Boost (tier 1).** Serve-time quality orchestration: top-nsigma
   sampling, confidence-weighted voting, and reasoning-budget policy. The
   sampling and budget dials are active in v2.0; **server-side multi-chain
   voting is not enabled in this release** — the per-sequence compressor
   history fix it requires (PR #21) is awaiting hardware validation.

5. **Trellis grouped-GEMM (the batched-serving keystone).** The W2A16
   grouped GEMM that decodes 2-bit trellis experts for whole batches of
   token→expert pairs. Hardware parity 5/5 on H200, and the session-4 batch
   curve shows the step floor going **flat at ~63.5 ms from B=16 to B=64**
   — the expert-read amortization the format was designed around,
   demonstrated at kernel level: ~1,006 aggregate tok/s ⇒ ≈$1.36/Mtok at
   $4.92/hr rental. That is an expert-path microbenchmark; the end-to-end
   batched-serving number is the next measurement.

Plus the supporting cast: `arc validate --target-hbm` rental preflight, the
GEMV autotune sweep harness with the H200-tuned winners shipped as dispatch
defaults, the `arc-tools/quality` eval harness and GPU-session runbooks that
produced every number, and everything inherited from mistral.rs
(PagedAttention, FlashAttention, speculative decoding, 100+ architectures).

## Measured numbers

All measured on rented 1×H200 (141 GB, $4.92/hr) across four GPU sessions,
2026-08-12 → 08-14; protocols and artifacts in
[docs/BENCHMARKS.md](BENCHMARKS.md). Total spend for every number in this
table: **≈ $123**.

| Number | Value | Type |
|---|---|---|
| Model on one GPU | 284B/13B V4 Flash from a 68 GB artifact (2-bit trellis experts + FP8 attention) | Measured |
| GSM8K | **87.0%** (n=100, 0-shot chat greedy, 2048-token cap, ±6.6 pp; base model card 90.8 is 8-shot — different protocol) | Measured |
| Quality ladder | GSM8K 64 → 84 → 87.0 across sessions (Viterbi encoder fix, then token budget) | Measured |
| Factual recall / arithmetic / coherence | 22/22 · 8/8 · 6/6 | Measured |
| Perplexity (internal rung) | 12.50 ± 3.46 (70-chunk wiki mini-corpus; not comparable to full-wikitext numbers) | Measured |
| Long context | 5/5 coherence + 4/4 needle, with 3-config ablation isolating window vs compressed-KV mechanisms | Measured |
| Decode speed (b=1) | **14.58 tok/s** (no-cudnn build; progression 5.4 → 13.99 → 14.58 across kernel-fix PRs) | Measured |
| qtip2b format correctness | CUDA ↔ CPU parity 20/20 on H200 | Measured |
| Grouped-GEMM correctness | Parity 5/5 on H200 (first hardware run) | Measured |
| GEMV autotune | Best variants ~36 µs / 450–467 GB/s (~9.5% of peak, 2.3× pre-tune); shipped as dispatch defaults | Measured (kernel-level) |
| Grouped-GEMM batch curve | Flat ~63.5 ms/step B=16→64 ⇒ ~1,006 aggregate tok/s ⇒ ≈$1.36/Mtok at $4.92/hr | Measured (kernel-level; expert path only) |
| TurboQuant KV | 4.27× context (Qwen3-32B, 1×H100, 39K → 169K) | Measured (different model/GPU) |

## Known limitations

Stated in full in [docs/BENCHMARKS.md](BENCHMARKS.md) § Limitations; the
short version:

- **b=1 speed is mid-optimization.** 14.58 tok/s does not yet include the
  autotuned GEMV dispatch in the serve path (the winners are baked in as
  defaults but the end-to-end effect is unvalidated). Building with the
  `cudnn` feature costs −62% decode on V4 — release binaries must be built
  without it.
- **The $1.36/Mtok figure is kernel-level.** It comes from the expert-path
  microbenchmark, not an end-to-end batched-serving run; the serving-level
  $/Mtok is the next measurement and the number the fleet claim will
  ultimately stand on.
- **Server-side voting (Arc Boost multi-chain) is off.** Multi-chain
  requests crash on a shared compressor history (`xs_history`); the
  per-sequence fix (PR #21) awaits hardware validation. No voting-based
  quality numbers are claimed.
- **MTP acceptance is unmeasured.** A KV-cache desync in the MTP wrapper was
  found and fixed, but a serve-path hang on the first MTP request remains
  under triage (`docs/notes/mtp-hang-triage.md`). Do not enable the MTP
  speculative path in production in v2.0.
- **One model family validated.** Every V4 number is DeepSeek V4 Flash;
  other architectures inherit mistral.rs support but have not had the same
  measurement treatment. TurboQuant KV's 4.27× is measured on Qwen3-32B and
  does not yet apply to MLA models (V4 included).
- **GSM8K is an n=100 subset** under a 0-shot chat protocol; the published
  base-model 90.8 is 8-shot EM. Both stated because hiding either would
  flatter us.

## Upgrade notes

- **UQFF 0.2.1.** v2.0 writes UQFF 0.2.1, which adds rank-3 stacked-expert
  (MoE) QTIP payloads — the format the 68 GB V4 artifact uses. **Readers
  older than 0.2.1 do not fail cleanly on rank-3 payloads: they mis-decode
  them.** Do not load v2.0 artifacts with pre-v2.0 builds. (0.2.1 readers
  load older 0.2.x artifacts fine.)
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
