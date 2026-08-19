# THE BATCH-SCALING BUDGET — why width does not convert, measured

**Parent system: ArcInfer / ArcSched + ArcQuant**
**Session 8, 2026-08-19. H200, exclusive card, 0 Xid. $16.72 of a $30 budget.**
**Binary `SHA256=0c4eb523…3455`, git `0fb8e588` = `origin/master` `9af390ed` + ragged instrumentation
`3bd58d59` + `fix/ragged-gate-coherence` (merged clean). Same binary every arm.**

**Read this before proposing ANY throughput work. It reverses the previous plan.**

---

## 0. The one sentence

**`launches/token` FALLS 12.5× with batch size — the host already amortizes.
At width the machine is 81% GPU-busy with 82.8% of GPU time in 1.9% of the launches.
⇒ fusion + graph capture is the *b=1* fix. The *width* fix is the trellis grouped GEMM.**

---

## 1. The sweep (decode-only steady state)

Method: fire B ragged rows, wait until **every** row is past prefill and resident, then open an 18 s
nsys window over pure steady-state decode. Prefill is ~40% of a whole-request leg at B=32, so a
launches/token taken over a full request blends two different diseases. Tokens counted at the socket;
launches by nsys; randomised leg order; one server per leg. Config `floor+ragged`, verified from
`/proc/PID/environ`; beacon and floor both fired.

**`launches/token` is a pure count ratio** (launches/step ÷ tokens/step, every window running lockstep
at exactly 1 token/row/step), so **nsys's own slowdown cancels out of it entirely** (measured
1.40/1.18/1.12/1.15/1.04× at B=1/4/8/16/32).

| B | resident | agg tok/s | per-user | ms/step | **launch/tok** | launch/step | D2H/step | D2H %wall | alloc/step | GPU busy | memCtrl |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1.0 | 15.82 | 15.82 | 88.3 | **9,063.7** | 9,064 | 45.2 | 2.7% | 22,658 | 40.7% | 4.6% |
| 4 | 4.0 | 33.97 | 8.49 | 138.6 | **2,983.3** | 11,933 | 44.1 | 21.1% | 31,847 | 55.4% | 4.0% |
| 8 | 8.0 | 36.85 | 4.61 | 243.5 | **1,736.1** | 13,889 | 44.5 | 45.0% | 35,733 | 70.5% | 3.0% |
| 16 | 16.0 | 49.77 | 3.11 | 367.7 | **1,055.1** | 16,860 | 44.0 | 51.2% | 41,019 | 71.7% | 4.1% |
| 32 | 31.3 | **59.39** | 1.86 | 563.1 | **725.7** | 23,221 | 44.2 | **63.6%** | 52,417 | **80.9%** | **4.6%** |

⚠️ **GPU busy is the union of kernel intervals.** `nvidia-smi utilization.gpu` reads 87–95% here and is
**not usable** — do not quote it.

---

## 2. What the numbers say

- **`launches/token` falls 12.5×** (9,064 → 726). **Not flat.** The host does *not* issue a
  per-sequence stream. **This reverses the prior hypothesis.**
- **`launches/decode-step` grows only 2.56× for 32× the width.** Issue cost is already shared.
- **Blocking D2H is 44–45 per decode step at EVERY width** — dead flat, ~1 per layer, a structural
  constant of the layer loop. Per *token* it falls 32.6× (near-perfect 1/B). **But the wall-clock
  fraction spent inside those same 44 calls goes 2.7% → 63.6%.** The sync *count* is not the disease
  at width; **what it waits for is.**
- **Memory controller is 3.0–4.6% at every width.** 32× the users does not move it. **We are nowhere
  near a physical limit — the headroom is real.**
- **At B=32, 82.8% of GPU time sits in 1.9% of the launches**: 61.5% QTIP MoE expert-gather GEMV,
  21.2% FP8 matmul.

### 2.1 🔑 THE TELL — one path amortizes, the other does not

`qtip_gather_gemv_warp_kernel` is called **~129.5 times per decode step at every width** — flat — but
its cost goes **10.2 ms → 281 ms/step (27.5×)** and its share of GPU time **28.3% → 61.5%**.

Meanwhile the **dense FP8 path switches from `fp8_gemv_warp` to `fp8_matmul_tiled` at B≥8**, and its
share then **falls 37.1% → 21.2%**. It amortizes, exactly as batching is supposed to.

**The QTIP MoE path never switches. It stays a GEMV at all widths and its share grows.**
**That is the entire difference, and it is the wedge.** Every extra user re-reads the same expert
weights instead of sharing one pass over them.

⇒ **The width fix is the trellis grouped GEMM** — already identified as the keystone, and already
written for the shipped LUT rung on `perf/qtip-lut-grouped-gemm`. The shipped `qtip2` artifact
**cannot reach it** (above `DECODE_REGIME_MAX_TOKENS = 8` it falls to `gather_forward_cuda`,
`qtip/mod.rs:3649`: a host D2H of the router, then dequantize **every distinct expert** to BF16 in HBM).

---

## 3. Ragged decode — what it is actually worth

3 arms × 3 repeats, **block-randomised** (C1 B1 A1 | A2 C2 B2 | A3 B3 C3). Instrument spread ≤1.7%
within every arm; b=1 held 16.12–16.38 across all nine runs (no thermal/cache drift).

| arm | per-user decode, med [spread] | aggregate, med [spread] | mean resident [spread] | peak |
|---|---|---|---|---|
| **A** no floor | **1.94** [1.93–1.94] | 36.71 [36.51–36.75] | 17.53 [17.45–17.68] | 28 |
| **B** floor only | 1.62 [1.61–1.62] | 37.15 [37.12–37.30] | 16.14 [16.02–16.14] | 29 |
| **C** floor + ragged | 1.49 [1.49–1.50] | **41.56** [41.52–41.70] | **25.85** [25.79–26.24] | 32 |

- **Ragged is worth ×1.10 aggregate** (×1.119 over floor-only, ×1.097 over repaired no-floor),
  **0.4% spread across 3 repeats. It replicates. It is the only real effect measured.**
- **⚠️ Quote it as CAPACITY, not speed.** C wins aggregate and **loses per-user decode** (1.49 vs 1.94,
  −23%): higher residency admits rows earlier, and they then share the GPU longer. Opposite directions.
- Residency gain replicates: C/B = ×1.601, C/A = ×1.475.

---

## 4. RETRACTIONS from this session's own GPU work

| claim | status |
|---|---|
| **"`ARC_PREFILL_FLOOR_STEPS=1` lifts residency 8→32, worth ×4.5"** | **RETRACTED — CONFOUNDED.** The `8.23 → 37.18` compared a **TOKENS=64** arm against a **TOKENS=384** arm — a 6× lifetime difference, and residency ≈ admission_rate × lifetime. The floor is worth **×1.012, or ×0.980** once arm A is credited for the row it drops. It does **not** lift residency: mean resident is *lower* with it (16.14 vs 17.53). It fires exactly as documented and buys ~0%. |
| **"Ragged buys ×4.84 per-user decode / ×1.61 aggregate"** | **RETRACTED — ARTIFACT.** Measured at width 8 in **both** arms with prefill starvation active in both. Remove starvation and the baseline reaches 28–29 wide unaided. |
| **"The ×3.25 reproduces on V4"** | **REFUTED at genuine width.** |
| **"launches/token is flat per sequence; no scheduling change can help"** | **REVERSED.** It falls 12.5×. |
| **"There is a residency cap of 8"** | **WRONG — no constant exists.** The 8 was a TOKENS=64 artifact. |

**The instrument was never the problem.** ≤1.7% spread across 9 runs. **The confound was.**
⇒ **New rule: before repeating a surprising result to reduce noise, first check the two arms differ in
exactly one variable.** Three agents and main all read a 6× lifetime difference as an effect.

---

## 5. Prediction check — the profiler is CONFIRMED

Measured b=1 **9,063.7**, repeat **9,054.0** — **0.11% spread**, and **0.7–0.85% below** the 9,131
baseline in `BUDGET_V4_B1.md`. **Two independent instruments (arc-profiler and nsys) agree.**
The profiler's attribution is **confirmed, not impeached**, and the three unshipped launch-reduction
branches (predicted 9,131 → 8,650) still have a clean baseline to be judged against.

---

## 6. New defects found while measuring

| defect | evidence |
|---|---|
| 🔴 **Intermittent admission starvation — a request silently returns nothing.** In the B=32 repeat, **row 15 of 32 produced ZERO tokens across 600 s** while the other 31 hit their 500-token cap. No error; reported as a normal completion. ~1 run in 3. **Under floor+ragged a user's tail latency is unbounded.** | `/root/logs/s10_legC.out` |
| 🔴 **Deterministic dropped row without the floor.** Arm A drops exactly one sequence per run — 0 tokens, `finish_reason: stop` — in **3/3 A runs, 0/6 B/C runs** (p ≈ 0.012). Rows 12, 30, 30 all satisfy `i mod 9 == 3`: **always the same prompt bucket.** | s9 arm-A logs |
| **1.68 M `cuMemAllocAsync`/`FreeAsync` per 18 s at B=32** — 52,417/step, **61% of all API calls** | nsys |
| **242,610 `cuMemcpyHtoDAsync` averaging 158 bytes** — 7,582/step | nsys |
| The engine issues **zero** `cuStreamSynchronize`/`cuCtxSynchronize`; its host/device rendezvous is `cuMemcpyDtoHAsync_v2` — **an "Async" call that returns in 8.1 ms mean is blocking.** Classify syncs by measured duration, never by name. | nsys |

---

## 7. Where the effort goes now

1. **Trellis grouped GEMM for the shipped rung** — the width fix. 61.5% of GPU time at B=32, and the
   only major path that does not switch algorithm with batch size.
2. **Fusion + graph capture** — the **b=1** fix, where we are 41% busy and launch-bound. Not a width fix.
3. **The per-step allocator** — 52,417 allocations/step at B=32 should not exist.
4. **The two admission defects** — a silently empty response is production-blocking.

**Do NOT** spend further effort on batch policy or the prefill floor at width: measured ~0%.
**D1 note:** that is a scoping result about the *floor*, not about admission policy in general — the
two starvation defects above are admission bugs worth fixing on correctness grounds alone.
