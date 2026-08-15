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
  quality numbers in
  BENCHMARKS.md (GSM8K 87.0%, long-context 5/5+4/4) **[measured — provisional,
  pre-PR-#35 decode math; see BENCHMARKS.md]**.
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
- Reality check **[measured]**: single-stream decode is 14.58 tok/s
  (no-cudnn build; progression 5.4 → 13.99 → 14.58 across the kernel-fix
  PRs). The session-4 autotune sweep lifted the gather-GEMV kernels from
  153–192 GB/s (3–4% of H200 peak) to ~36 µs / 450–467 GB/s best-variant
  (~9.5% of peak, 2.3× — now the shipped dispatch defaults, PR #20), but
  the tuned dispatch is not yet reflected in the 14.58 end-to-end number
  (see BENCHMARKS.md). The floor math above is the destination, not the
  present: measured kernel-level throughput at B=64 is ~1,006 tok/s against
  the ~4.1K tok/s spec-bandwidth floor (~24% of it), and serving-level
  throughput will be lower still until the engine-level batched run exists.

## 3. KV tenancy at 3.5-bit

Once weights fit, concurrent capacity is bounded by KV cache per sequence.

- TurboQuant K4/V3 (3.5-bit average KV) measured **4.27× context capacity**
  end-to-end on Qwen3-32B on a single H100: 39K → 169K tokens, with quality
  confirmed **[measured — different model/GPU than the V4 runs; Apr 2026]**.
- Fleet meaning: at fixed HBM headroom, ~4× the concurrent tenants per replica
  (or ~4× the context per tenant) on GQA-attention models **[projected beyond
  the measured Qwen3-32B configuration]**.
- Honest gap: MLA-attention models — including DeepSeek V4 — currently fall
  back to the standard KV path, so the 4.27× does **not** yet stack onto the
  V4 replica math above. TurboQuant×MLA is in progress **[not measured]**.

## Summary table

| Claim | Status |
|---|---|
| 284B/13B MoE serves on one H200 from a **74.18 GB** artifact, with quality numbers | **Measured** (artifact size HF-API-verified) |
| GSM8K 87.0% (n=100, 0-shot greedy, seed 161, 2048-cap) at 2-bit experts | **Measured — provisional** (PR #35 changed the decode math after the run; re-measure pending) |
| TurboQuant KV 4.27× context (Qwen3-32B, 1×H100) | **Measured** |
| qtip2b bitshift-trellis CUDA parity (20/20 on H200) | **Measured** |
| Trellis grouped-GEMM hardware parity (5/5 on H200) | **Measured** |
| b=1 decode 14.58 tok/s (no-cudnn build; 5.4 → 13.99 → 14.58) | **Measured** |
| GEMV autotune: ~36 µs / 450–467 GB/s best-variant (~9.5% peak, 2.3×), shipped as dispatch defaults | **Measured-kernel** (end-to-end effect pending) |
| Grouped-GEMM batch curve flat ~63.5 ms/step B=16→64 ⇒ ~1,006 aggregate tok/s ⇒ ≈$1.36/Mtok at $4.92/hr | **Measured-kernel** (expert path only; serving run pending) |
| End-to-end batched serving tok/s and $/Mtok on one H200 | Pending (planned session 5) |
| 8 replicas per 8×H200 node | Projected (building block measured) |
| ~4.1K tok/s/GPU at saturated batch; ~8× node aggregate vs 1×TP8 | Projected (floor arithmetic off the measured 74.18 GB artifact; kernel-level evidence at ~24% of the floor) |
| ~4× KV tenancy on V4-class MLA models | Projected (MLA path not done) |

Cost of every measured number above: ≈ $123 of rented H200 time across four
sessions. Protocols, artifacts, and limitations:
[BENCHMARKS.md](BENCHMARKS.md).
