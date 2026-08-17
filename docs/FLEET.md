# Arc Fleet Economics — measured vs projected

Arc's wedge is datacenter capacity: the same GPU fleet serving a multiple of
its original throughput, cheaper per token. This page states that case with
every claim explicitly tagged **[measured]** (a saved artifact exists — see
[BENCHMARKS.md](BENCHMARKS.md)), **[measured-kernel]** (measured on hardware
at the kernel level — a microbenchmark of the bottleneck path, not an
end-to-end serving run), or **[projected]** (arithmetic from measured
quantities plus vendor specs, not yet run end-to-end).

## 1. Replica granularity: 8 independent replicas per node, not 1×TP8

- DeepSeek V4 Flash is 284B logical parameters **[measured — HF-verified]**.
  At BF16 that is ~568 GB of weights (284B × 2 bytes) **[arithmetic]** — an
  8×H200 node (8 × 141 GB) runs **one** tensor-parallel replica, with
  all-reduce traffic on every layer.
- Arc's artifact for the same model is **74.18 GB** (2-bit trellis experts + FP8
  attention) **[measured — HF API, `aeonmind/DeepSeek-V4-Flash-UQFF-qtip2`, 15
  files, 8 shards + residual]**, and one H200 loads and serves it with the
  quality numbers in BENCHMARKS.md (**GSM8K 1270/1319 = 96.3% ±1.0pp**, full
  test set, 0-shot, 0 degenerate / 0 truncated / 0 errors; long-context 5/5+4/4)
  **[measured — 2026-08-17, on the `qtip2b` artifact]**. The reference model's
  published 90.8 is **8-shot**, a different and easier protocol, so 96.3 is not
  a like-for-like win over it.
- Therefore the same node can instead host **8 independent replicas** — no
  inter-GPU traffic, per-replica failure isolation, per-replica batch
  scheduling **[projected — the 8-up deployment itself has not been run; the
  single-replica-per-GPU building block is what's measured]**.

## 2. Saturated-batch expert-read floor

Decode throughput on a MoE is bounded by how fast expert weights stream from
HBM. At batch ≈ 32+ concurrent sequences, essentially all 257 experts are
activated every step, so per-step weight traffic approaches the entire expert
pool — and the pool size is what compression shrinks.

- Arc pool: the full artifact is **74.18 GB** **[measured]**. One step's weight
  read at H200 spec bandwidth (4.8 TB/s, vendor figure) ≈ 74.18 GB / 4.8 TB/s ≈
  **15.5 ms** → ~65 steps/s; at batch 64 that is **~4.1K tok/s per GPU**, ~33K
  tok/s per 8-replica node **[projected]**. (Was quoted as ~4.5K/~36K off a
  68 GB estimate; the artifact is measured at 74.18 GB, so the floor is
  correspondingly lower.)
- BF16 comparison: the per-GPU read floor under TP8 is similar (~568 GB / 8 ≈
  71 GB per GPU per step), but the whole node serves **one** replica, so the
  node's aggregate is ~4.5K tok/s **[projected]**. The ~8× node-level gain
  comes from replica granularity — same silicon, same bandwidth, eight
  independent batches — which compression is what makes possible.
- **First hardware evidence [measured-kernel]** (session 4): the trellis
  grouped-GEMM batch curve on one H200 is **flat at ~63.5 ms/step from B=16
  to B=64** — the expert-read amortization mechanism above, demonstrated on
  silicon. At B=64 that is **~1,006 aggregate tok/s** on the expert path,
  which at the $4.92/hr rental rate is **≈$1.36/Mtok** — kernel-level floor
  economics. This is a microbenchmark of the batched 2-bit MoE expert path
  (40-layer extrapolation); it excludes attention, routing, sampling, and
  engine overhead, so it is an upper bound on serving throughput at today's
  kernel efficiency, not a serving number. The end-to-end batched-serving
  measurement is planned for session 5.
- **The end-to-end serving run now exists [measured]** (2026-08-17, 1×H200 @
  $4.85/hr, published `qtip2b` artifact, $16.11, 0 errors across 505 requests,
  `effective_B == B` on all seven batch rows):

  | B | 1 | 8 | 16 | 32 | 64 | 128 | **256** |
  |---|---|---|---|---|---|---|---|
  | **decode aggregate tok/s** | 18.27 | 41.43 | 54.75 | 74.52 | 91.46 | 106.36 | **111.69** |
  | per-user p50 tok/s | 17.99 | 5.67 | 3.97 | 2.87 | 1.82 | 1.09 | 0.53 |
  | **$/Mtok** | 73.74 | 32.52 | 24.61 | 18.08 | 14.73 | 12.67 | **12.06** |

  **Aggregate rises monotonically all the way to B=256** — the batching argument
  survives the full serving path, not just the kernel microbenchmark. The b=1
  row (1.12× over the `qtip2` rung) is the **control**: it shows the gain comes
  from the rung plus batching, not from a faster box.
- Reality check **[measured]**: the gap to the floor is large and is the work.
  Serving aggregate at B=64 is **91.46 tok/s** against the ~4.1K tok/s
  spec-bandwidth floor — **~2% of it**. Per-user at B=128 is 1.09 tok/s. A
  named contributor: GPU top-k sampling falls back to CPU on **every token**
  (`tensor_device_ptr: unsupported dtype I32`), a device→host round trip per
  token in the decode loop **[measured, session 8]**. Arc is overhead-bound
  today, not bandwidth-bound — 111.69 tok/s is low single-digit % of the H200's
  4.8 TB/s.

## 3. KV tenancy at 3.5-bit

Once weights fit, concurrent capacity is bounded by KV cache per sequence.

- 🔴 **RETRACTION (2026-08-17).** This section previously claimed TurboQuant
  K4/V3 measured **4.27× context capacity end-to-end on Qwen3-32B on a single
  H100 (39K → 169K tokens)**. **That number is format arithmetic — bytes per
  token at 3.5 bits versus BF16 — and was never produced by a forward pass on
  any GPU.** No TurboQuant measurement exists anywhere in this repo's record.
  It must not be cited as measured, and the "Apr 2026, quality confirmed"
  provenance attached to it was not real.
- **TurboQuant is not on any default serving path today [code-verified]:**
  - The eager KV path is **opt-in via `ARC_TURBOQUANT_KV=1`**, default off
    (`mistralrs-core/src/kv_cache/mod.rs`).
  - The paged path has a kernel at **head_dim 128 only**
    (`TURBOQUANT_HEAD_DIM`); the ambient default auto-falls back to `Auto`
    with a warning, and an explicit `--pa-cache-type turboquant` hard-errors
    off-envelope.
  - **No kernel exists at head_dim 512**, so DeepSeek V4 cannot use it at all.
  - The prefix cache auto-disables under TurboQuant (packed U8 blocks cannot
    be gathered).
- Fleet meaning **[projected — arithmetic only]**: *if* TurboQuant were on a
  serving path, 3.5-bit KV would buy roughly 4× the concurrent tenants per
  replica at fixed HBM on GQA-attention models. Nothing here is measured, and
  the honest gap is larger than "MLA is pending": the feature is a compression
  format with no measured serving run behind it.

## Summary table

| Claim | Status |
|---|---|
| 284B/13B MoE serves on one H200 from a **74.18 GB** artifact (`qtip2`) | **Measured** (size HF-API-verified; the `qtip2b` rung Arc serves is 74.12 GB per BENCHMARKS.md, not separately HF-API-verified) |
| **GSM8K 96.3%** (1270/1319, full set, 0-shot, 0 degenerate / 0 truncated / 0 errors) at 2-bit experts | **Measured** (2026-08-17, `qtip2b`) |
| **End-to-end serving: 111.69 tok/s aggregate @ B=256, $12.06/Mtok**, monotonic from B=1, 0 errors / 505 requests | **Measured** (1×H200, $16.11) |
| qtip2b bitshift-trellis CUDA parity (20/20 on H200) | **Measured** |
| Trellis grouped-GEMM hardware parity (5/5 on H200) | **Measured** |
| b=1 decode 18.27 tok/s aggregate (17.99 per-user p50) | **Measured** |
| GEMV autotune: ~36 µs / 450–467 GB/s best-variant (~9.5% peak, 2.3×), shipped as dispatch defaults | **Measured-kernel** |
| Grouped-vs-GEMV crossover at **B=64**; grouped keeps climbing (527 tok/s @ B=128) while GEMV is flat (315→317) | **Measured-kernel** (MoE-GEMM floor only) |
| TurboQuant KV compression | **Never measured** — the former "4.27×" was format arithmetic, now retracted; not on any default path |
| 8 replicas per 8×H200 node | Projected (building block measured) |
| ~4.1K tok/s/GPU at saturated batch; ~8× node aggregate vs 1×TP8 | Projected (floor arithmetic off the measured 74.18 GB artifact; serving is at ~2% of the floor) |
| Expert parallelism across cards | Projected — **not implemented** (`Comm::Dummy`, world_size 1); design only |
| ~4× KV tenancy on V4-class MLA models | Projected (no TurboQuant kernel at head_dim 512) |

No side-by-side run against SGLang, vLLM, or any other engine has ever been
performed, so every $/Mtok figure here is Arc-versus-Arc. Protocols, artifacts,
and limitations: [BENCHMARKS.md](BENCHMARKS.md).
