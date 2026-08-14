# Arc Benchmarks — DeepSeek V4 Flash on one H200

Measured results from three rented GPU sessions (2026-08-12/13/14). Every
number on this page comes from a saved eval artifact produced by the
`arc-tools/quality` harness; nothing here is derived from bandwidth math.
Where our protocol differs from a published reference, the difference is
stated inline.

## Setup

| | |
|---|---|
| Model | DeepSeek V4 Flash — 284B logical / 13B active MoE (HF-verified: model card + release announcement + config geometry, 43 layers / 256+1 experts) |
| Artifact | 68 GB UQFF bake, 7 shards: 2-bit trellis (qtip2) experts + FP8 attention; `lm_head` and the context compressor excluded from 2-bit |
| Hardware | 1× NVIDIA H200 (141 GB HBM3e), rented via Runcrate (NY), $4.92/hr |
| Engine | Arc (this repo), CUDA + flash-attn build; serve via OpenAI-compatible HTTP API |
| Sessions | Session 1: 2026-08-12T23:31Z, 9.2 h. Session 2: 2026-08-13T12:47Z (re-bake with the Viterbi encoder fix, PR #9). Session 3: 2026-08-13T20:31Z, ≈6.2 h (throughput session: all kernel-fix PRs #8/#10/#14/#15 in the build; token budget raised to 2048) |

## Headline results (latest measured session per row)

| Eval | Result | Protocol | Reference point |
|---|---|---|---|
| GSM8K (session 3) | **87.0%** (87/100, ±6.6 pp CI95) | n=100, 0-shot chat, greedy (t=0), seed 161, max_tokens 2048; 2 degenerate outputs, 9 still truncated even at the 2048 cap; mean completion 528.4 tokens | Base model card: **90.8** — but that is **8-shot EM**, a different protocol; not directly comparable |
| Factual recall (session 2) | **22/22** | greedy | — |
| Arithmetic (session 2) | **8/8** | greedy | — |
| Coherence battery (session 3) | **6/6** | t=1.0, p=0.95; re-validated on the session-3 bake | — |
| Perplexity (session 2) | **12.50 ± 3.46** | 70 × 1024-token chunks, wiki mini-corpus (`wiki.test_small`) | See corpus caveat below |
| Long context (session 2) | **5/5 coherence + 4/4 needle** | greedy; ablation matrix below | — |
| Decode speed (b=1, session 3) | **13.99 tok/s** | single stream, 256 decode tokens, 525-token prompt; prefill ~57 tok/s, TTFT ~9.3 s | ×2.6 over the 5.4 baseline from the four kernel-fix PRs (#8/#10/#14/#15) compounding; GEMV tuning still in progress — see kernel profile below |

**Perplexity caveat:** 12.50 is measured on a 70-chunk wiki mini-corpus, **not**
the full wikitext-2 test set. It is NOT comparable to published full-wikitext
perplexities and we do not present it as such. Its role here is internal: the
same rung measured 58.85 (3 chunks, same pipeline) before the Viterbi encoder
fix — a 4.7× improvement into healthy-model range. A same-corpus GGUF `q2k`
comparison rung (session-1 mini-corpus value: 22.50 on 3 chunks of the same
text as the 58.85) has not yet been re-run on the 70-chunk corpus, so we make
no "beats q2k" claim yet.

### Session 1 → 2 → 3 ladder (why the numbers moved)

| Metric | Session 1 | Session 2 | Session 3 | What changed |
|---|---|---|---|---|
| GSM8K | 64.0% (32/50, ±13.3 pp; 4 degenerate, 33/50 truncated at a 640-token cap) | 84.0% (n=100; 0 degenerate, 17/100 truncated at 1024) | 87.0% (n=100; 2 degenerate, 9/100 truncated at 2048) | Session 2: PR #9 — the bake had run greedy trellis encoding (not Viterbi) on 3-D expert stacks and disabled the Hadamard rotation — fixed; token budget 640 → 1024. Session 3: token budget 1024 → 2048 (truncation was still costing points) |
| Decode (b=1) | 5.24 tok/s | 5.4 tok/s | 13.99 tok/s | Kernel-fix PRs #8/#10/#14/#15 (Sinkhorn fused default-on, absorbed-MLA decode, FP8 GEMV, dequant-materialize guard) compounding in one build |
| Perplexity (same rung) | 58.85 ± 24.76 (3 chunks) | 12.50 ± 3.46 (70 chunks) | not re-run | Viterbi fix; wider corpus |
| Facts | 21/22 | 22/22 | not re-run | — |

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
| Trellis grouped-GEMM (batched 2-bit MoE, Stage 4) | **5/5 parity tests passed on H200** (session 3 — first hardware run of the keystone kernel: deterministic output, CPU-reference match on ragged expert-group shapes, tile-map partitioning, descriptor table, window-state recurrence). Batched throughput curve not yet measured |
| Sinkhorn fused kernel vs reference path | **Token-identical** on 6/6 greedy 128-token prompts, and perplexity **bit-identical** — after the PR #8 identity fix. (Session 1, pre-fix: rejected with 4/6 token divergence — recorded, fixed, re-validated.) The fused path is now on by default (PR #15) |
| MLA absorbed decode vs non-absorbed | **Token-identical** on 6/6 prompts |

## Decode speed and kernel profile

Single-user (batch=1) decode is **13.99 tok/s** (session 3, best of 2 runs;
session-2 baseline 5.4; session-1 5.24 at 92% GPU utilization with the model
fully resident in 82 GB — i.e. kernel-bound, not memory-bound on capacity).
The ×2.6 came from the four kernel-fix PRs (#8 Sinkhorn bit-identity → fused
default-on via #15, #10 absorbed-MLA decode + GPU top-k, #14 FP8 blockwise
GEMV + dequant-materialize decode guard, #15 Sinkhorn default + MTP UQFF
coverage) compounding in one build. This is still an unoptimized state and we
publish it as such.

Where the remaining time goes — gather-GEMV microbenchmark on the H200
(session 3, 2-bit trellis expert weights, 2.11 MB/expert):

| Shape | b=1 | b=8 | Marginal kernel BW | Fixed overhead |
|---|---|---|---|---|
| gate/up N=2048 K=4096 | 153 GB/s (3.2% of peak) | 186 GB/s (3.9%) | 192 GB/s (4.0%) | ~18.7 µs/call |
| down N=4096 K=2048 | 159 GB/s (3.3%) | 184 GB/s (3.8%) | 189 GB/s (3.9%) | ~13.1 µs/call |

The expert-weight read path still runs at **3–4% of the H200's HBM bandwidth
with ~13–19 µs of fixed overhead per call — this is the active tuning
target** (a launch-parameter autotune sweep is queued for the next session).
The trellis grouped-GEMM kernel passed its first hardware parity run in
session 3 (5/5); its batched throughput curve is not yet measured. Until
those numbers exist, throughput beyond 13.99 tok/s is a target, not a claim.

## Reproduction

The full harness (eval clients, data fetcher, PPL runner, speed probe) lives in
`arc-tools/quality/` on branch `run161-quality-harness`; the step-by-step
session scripts are `arc-tools/quality/GPU_SESSION_RUNBOOK.md` (session 1),
`GPU_SESSION_RUNBOOK_2.md` (branch `session2-runbook`, PR #11), and
`GPU_SESSION_RUNBOOK_4.md` (branch `session4-runbook`; session 3 ran a
condensed script derived from runbook 2).

Outline:

1. Rent 1× H200 (≥720 GB disk). **First check `nvidia-smi`'s max CUDA version
   against `nvcc --version`** — a driver/toolkit mismatch cost us the first
   bake attempt (`CUDA_ERROR_UNSUPPORTED_PTX_VERSION`).
2. `cargo build --release --features "cuda flash-attn cudnn"`.
3. `arc-tools/quality/fetch_data.sh` (model + eval data; ~149 GB download).
4. Bake the UQFF artifact (qtip2 experts + FP8 attention; Viterbi is the
   default expert encode mode since PR #9). Expect ~68 GB across 7 shards.
5. Serve on port 1234; run `run_coherence.py`, `run_gsm8k.py`, `run_longctx.py`
   (with the `ARC_V4_STANDARD_DENSE` / `ARC_V4_WINDOW_ONLY` gates for the
   ablation), `speed_probe.py`, `run_sinkhorn_ab.py`; run `run_ppl.sh` with the
   server stopped (PPL and serve are mutually exclusive on one GPU).
6. All evals must run against the **same bake** — never mix artifacts.

## Hardware and cost transparency

Total spend for every number on this page: **≈ $108** across three sessions on
a single rented H200 at $4.92/hr (session 1: 9.2 h ≈ $45; session 2: ≈ $32;
session 3: ≈6.2 h ≈ $30.5). No lab cluster, no reserved capacity.

## Limitations

- **Single-user speed is mid-optimization.** 13.99 tok/s at batch=1 — ×2.6
  from the kernel fixes, GEMV tuning in progress. The kernel profile above
  shows the expert read path still at 3–4% of peak HBM bandwidth with
  ~13–19 µs fixed overhead per call; the launch-parameter autotune sweep and
  the grouped-GEMM batched throughput curve are the queued next steps. We
  publish the current number rather than a projection.
- **MTP (multi-token prediction) acceptance is unmeasured.** The session-1/2
  UQFF artifacts did not carry the MTP decoder block. The bake/load fix has
  since merged (PR #15 — MTP now loads from UQFF or falls back to source
  weights) and session 3 validated the fallback load path (clean INFO, no
  crash), but the acceptance-rate probe itself failed to produce an artifact
  (the acceptance telemetry the probe greps for is not on master — root
  cause identified); the measurement is re-queued for the next GPU session.
- **Perplexity corpus.** 12.50 is a 70-chunk wiki mini-corpus number — not
  comparable to full wikitext-2 results, and the same-corpus q2k comparison
  rung is still pending (see caveat above).
- **One model family validated so far.** Everything on this page is DeepSeek
  V4 Flash. Claims about other architectures are untested until they get the
  same treatment.
- **GSM8K is an n=100 subset** (±6.6 pp CI95) under a 0-shot chat protocol
  with a 2048-token budget (9/100 answers still truncated even at that cap,
  and 2 degenerate outputs — both counted as wrong); the published 90.8
  base-model figure is 8-shot EM. We state both because hiding either would
  flatter us.
- **TurboQuant KV was validated on a different model** (4.27× KV compression,
  Qwen3-32B on one H100, 39K→169K context, Apr 2026). MLA-attention models —
  including V4 Flash — currently fall back to the standard KV path, so the KV
  compression gain does not yet apply to the V4 numbers above.
