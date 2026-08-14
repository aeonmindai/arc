# Arc Benchmarks — DeepSeek V4 Flash on one H200

Measured results from two rented GPU sessions (2026-08-12/13). Every number on
this page comes from a saved eval artifact produced by the `arc-tools/quality`
harness; nothing here is derived from bandwidth math. Where our protocol
differs from a published reference, the difference is stated inline.

## Setup

| | |
|---|---|
| Model | DeepSeek V4 Flash — 284B logical / 13B active MoE (HF-verified: model card + release announcement + config geometry, 43 layers / 256+1 experts) |
| Artifact | 68 GB UQFF bake, 7 shards: 2-bit trellis (qtip2) experts + FP8 attention; `lm_head` and the context compressor excluded from 2-bit |
| Hardware | 1× NVIDIA H200 (141 GB HBM3e), rented via Runcrate (NY), $4.92/hr |
| Engine | Arc (this repo), CUDA + flash-attn build; serve via OpenAI-compatible HTTP API |
| Sessions | Session 1: 2026-08-12T23:31Z, 9.2 h. Session 2: 2026-08-13T12:47Z (re-bake with the Viterbi encoder fix, PR #9) |

## Headline results (session 2)

| Eval | Result | Protocol | Reference point |
|---|---|---|---|
| GSM8K | **84.0%** (84/100, ±7.2 pp CI95) | n=100, 0-shot chat, greedy (t=0), seed 161, max_tokens 1024; 0 degenerate outputs, 17 truncated at the 1024 cap; mean completion 412.6 tokens | Base model card: **90.8** — but that is **8-shot EM**, a different protocol; not directly comparable |
| Factual recall | **22/22** | greedy | — |
| Arithmetic | **8/8** | greedy | — |
| Coherence battery | **6/6** | t=1.0, p=0.95 | — |
| Perplexity | **12.50 ± 3.46** | 70 × 1024-token chunks, wiki mini-corpus (`wiki.test_small`) | See corpus caveat below |
| Long context | **5/5 coherence + 4/4 needle** | greedy; ablation matrix below | — |
| Decode speed (b=1) | **5.4 tok/s** | single stream, 256 decode tokens, 525-token prompt; prefill ~57 tok/s, TTFT ~9.3 s | Unoptimized — see kernel profile below |

**Perplexity caveat:** 12.50 is measured on a 70-chunk wiki mini-corpus, **not**
the full wikitext-2 test set. It is NOT comparable to published full-wikitext
perplexities and we do not present it as such. Its role here is internal: the
same rung measured 58.85 (3 chunks, same pipeline) before the Viterbi encoder
fix — a 4.7× improvement into healthy-model range. A same-corpus GGUF `q2k`
comparison rung (session-1 mini-corpus value: 22.50 on 3 chunks of the same
text as the 58.85) has not yet been re-run on the 70-chunk corpus, so we make
no "beats q2k" claim yet.

### Session 1 → session 2 delta (why the numbers moved)

| Metric | Session 1 | Session 2 | What changed |
|---|---|---|---|
| GSM8K | 64.0% (32/50, ±13.3 pp; 4 degenerate, 33/50 truncated at a 640-token cap) | 84.0% (n=100; 0 degenerate, 17/100 truncated at 1024) | PR #9: the bake had run greedy trellis encoding (not Viterbi) on 3-D expert stacks and disabled the Hadamard rotation — fixed; token budget raised 640 → 1024 |
| Perplexity (same rung) | 58.85 ± 24.76 (3 chunks) | 12.50 ± 3.46 (70 chunks) | Same Viterbi fix; wider corpus |
| Facts | 21/22 | 22/22 | — |

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
| Sinkhorn fused kernel vs reference path | **Token-identical** on 6/6 greedy 128-token prompts, and perplexity **bit-identical** — after the PR #8 identity fix. (Session 1, pre-fix: rejected with 4/6 token divergence — recorded, fixed, re-validated.) The fused path is now on by default (PR #15) |
| MLA absorbed decode vs non-absorbed | **Token-identical** on 6/6 prompts |

## Decode speed and kernel profile

Single-user (batch=1) decode is **5.4 tok/s** (best 5.43; session-1 baseline
5.24 at 92% GPU utilization with the model fully resident in 82 GB — i.e.
kernel-bound, not memory-bound on capacity). This is the unoptimized state and
we publish it as such.

Where the time goes — gather-GEMV microbenchmark on the H200 (2-bit trellis
expert weights, 2.11 MB/expert):

| Shape | b=1 | b=8 | Marginal kernel BW | Fixed overhead |
|---|---|---|---|---|
| gate/up N=2048 K=4096 | 153 GB/s (3.2% of peak) | 186 GB/s (3.9%) | 192 GB/s (4.0%) | ~18.6 µs/call |
| down N=4096 K=2048 | 159 GB/s (3.3%) | 184 GB/s (3.8%) | 189 GB/s (3.9%) | ~13.2 µs/call |

The expert-weight read path runs at 3–4% of the H200's HBM bandwidth. That gap
is the current engineering front: a trellis grouped-GEMM kernel is merged
(compile-gated) with hardware bring-up pending. Until those numbers exist,
throughput beyond 5.4 tok/s is a target, not a claim.

## Reproduction

The full harness (eval clients, data fetcher, PPL runner, speed probe) lives in
`arc-tools/quality/` on branch `run161-quality-harness`; the step-by-step
session scripts are `arc-tools/quality/GPU_SESSION_RUNBOOK.md` (session 1) and
`GPU_SESSION_RUNBOOK_2.md` (branch `session2-runbook`, PR #11).

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

Total spend for every number on this page: **≈ $77** across two sessions on a
single rented H200 at $4.92/hr (session 1: 9.2 h ≈ $45; session 2: ≈ $32).
No lab cluster, no reserved capacity.

## Limitations

- **Single-user speed is unoptimized.** 5.4 tok/s at batch=1. The kernel
  profile above shows the expert read path at 3–4% of peak HBM bandwidth;
  grouped-GEMM bring-up is in progress. We publish the current number rather
  than a projection.
- **MTP (multi-token prediction) acceptance is unmeasured.** The UQFF artifact
  used in these sessions did not carry the MTP decoder block, so
  speculative-decode acceptance rates could not be measured. The bake/load fix
  has since merged (PR #15 — MTP now loads from UQFF or falls back to source
  weights); the acceptance measurement is queued for the next GPU session.
- **Perplexity corpus.** 12.50 is a 70-chunk wiki mini-corpus number — not
  comparable to full wikitext-2 results, and the same-corpus q2k comparison
  rung is still pending (see caveat above).
- **One model family validated so far.** Everything on this page is DeepSeek
  V4 Flash. Claims about other architectures are untested until they get the
  same treatment.
- **GSM8K is an n=100 subset** (±7.2 pp CI95) under a 0-shot chat protocol; the
  published 90.8 base-model figure is 8-shot EM. We state both because hiding
  either would flatter us.
- **TurboQuant KV was validated on a different model** (4.27× KV compression,
  Qwen3-32B on one H100, 39K→169K context, Apr 2026). MLA-attention models —
  including V4 Flash — currently fall back to the standard KV path, so the KV
  compression gain does not yet apply to the V4 numbers above.
