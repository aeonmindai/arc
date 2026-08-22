# Arc, complete — the single gate for selling tokens

**Merging this PR means Arc is the engine we set out to build, and it can be sold on OpenRouter.**
Not a defect list. The vision, shipped. Nothing reaches `master` for this effort except through here.

Base `d7742670a` (the 15.0 → **34.48 tok/s** b=1 ladder, on master).

---

## 🎯 THE GATE IS ONE NUMBER: **14,000 tok/s AGGREGATE ON ONE H200**

| aggregate | tokens/GPU-hr | revenue @ $1/M | vs $4.92 card | verdict |
|---|---|---|---|---|
| ~50 (today, slow build) | 0.18 M | $0.18 | **27× underwater** | — |
| 1,400 | 5.0 M | $5 | break-even | **pointless** |
| **14,000** | **50.4 M** | **$50** | **10× margin** | **the business** |

**Why 14,000 is an engineering target and not a fantasy:**

| batch | memory-bound roofline | 14,000 is |
|---|---|---|
| 256 | 16,600 tok/s | **84% of peak** — nobody runs there |
| **512** | **~33,000 tok/s** | **42% of peak — normal for a good engine** |

Memory traffic per step is **flat** as batch grows — 256 or 512 users read the same 74 GB once.
Every extra concurrent user is nearly free until compute-bound, somewhere around B≈512–1000.
**So 14K is a batching problem, not a kernel-speed problem.**

🔴 **And batching is exactly what is broken.** Amortisation only pays when memory is the
constraint. **Our memory controller sits at 4%** — we are instruction-bound, so adding users
adds work instead of sharing it. Measured: b=1 is 85× off its bound, **aggregate at B=8 is
227× off** — we are *further* off at batch, which is the signature.

**b=1 speed is a waypoint, not the goal.** Single-user is what one customer feels; aggregate
is what you sell.

---

## What has to be true

### 1 · Batching actually amortises  ← **the gate**
- [ ] Sweep **B = 1/8/32/64/128/256/512**, per-request **and** aggregate. Current sweep stops at 128 — **too early; the answer lives past 256.**
- [ ] Find the knee. 256 experts top-6 means tokens barely collide until batch is large; the knee is where sharing starts and it is the single most important measurement in the program.
- [ ] Memory-controller utilisation at each point. **4% → 60%+ is the whole job.**
- [ ] KV bytes resident per user, and the OOM ceiling.

### 2 · The engine runs near the silicon
- [ ] **Kernels at their own bounds.** Real profile on `qtip2b`: tail (sinkhorn + fast_sum + bmul + ucopy) **27.2%**, `fp8_gemv_warp` **16.0% → 13.0%** (1.31× landed, bit-exactness **not yet verified**), `qtip2b_gemv_tuned_kernel` **15.4%**. 🚩 A **4×4** Sinkhorn costing 9.3% of an H200 is structural.
- [ ] **4-stage async pipelining** — measured 54.8% → **92.9%** of peak across depths 1→4, saturating at 4. Issue rate first, then this.
- [ ] **TCFRAG** — 26.31 → 3.56 inst/weight, zero shuffles, cosine 1.0, **B=8 costs what b=1 costs** (the capacity property). Attaches to `qtip2b_gemv_tuned_kernel`. **Permute at LOAD (D22)** — artifacts stay architecture-neutral, one file for every GPU generation.
- [ ] **ArcGraph is INERT by default** — candle sits on the legacy NULL stream, which CUDA forbids capturing, so decode runs fully eagerly. Capture never engages unless env vars are set.
- [ ] **GPU WHILE-node loop** — syncs once per *generation* instead of once per token. Written, **zero callers**. Blocked by an unconditional `return Ok(None)` (`pipeline/normal.rs:2324`) and a U8-dtype bug failing all 28 call sites. **Neither competitor has this.**
- [ ] **`CudaSampler`** — complete, tested bit-exact, **zero callers, no gate**. Just unwired.

### 3 · Prefill is a product, not an afterthought
- [ ] **TTFT ~24 s ⇒ usable.** ~12 ms per *input* token, ~100× off bound. **Prefill is the opposite disease from decode — the GPU is already 99.5% busy, so graphs and launch reduction buy it nothing.** Its own lane, its own kernels.
- [ ] **Prefill crashes**: b=1 on `qtip2b` exits 139 (SIGSEGV) and prints `pp = 0.000±0.000` into a results table instead of failing.
- [ ] **Chunked prefill** — inert until the MoE gather lands.

### 4 · The memory system Arc was designed around
- [ ] **Prefix caching is silently disabled** by our own default (TurboQuant is the default cache type). Most real traffic is repeated prefixes.
- [ ] **56× KV over-retention** — root-caused; the fix reportedly never reached V4.
- [ ] **Long context measured**: at 1M ctx the answer is **~3 users or ~55** depending on whether the 128-token window + compress ratios are live. Unmeasured, and it is the capacity claim.
- [ ] **Ragged batching / ArcSched** — admit mixed-length cohorts; the per-seq advance dies on ragged batches today.

### 5 · The multipliers we designed and haven't switched on
- [ ] **Speculative decoding (MTP)** — measured 1.93 tok/step at b=1, 1.06 at b=128. A near-2× at low batch, off.
- [ ] **Expert parallelism** — does not exist. The natural MoE scaling axis.
- [ ] **ArcTarget multi-arch** — B200/B300 as first-class, one artifact, N load-time tables (D22).

### 6 · Trust — because our instruments have been the problem
- [ ] **Twelve instruments lied in one session.** bench counted *requested* tokens (**1280 tok/s for a zero-token run**) · `cargo clean -p` silently no-ops on git deps · `nsys` hides graph-internal kernels by default · a stale `.o` satisfied a link · `replay()` returned a clone of the graded buffer · `append_graph` compared a buffer against itself · an XOR control cancelled over an even tile count · an atomic read 0 while the work ran · a sampler slept past the run · bf16 parity vacuous at 8 mantissa bits · a guard died on Windows CRLF · prefill printed 0.000.
- [ ] **The live V4 fixture is all zeros** — identity tests against it prove nothing.
- [ ] **Quality re-confirmed on the exact artifact served.** GSM8K 96.0% exists — on a different rung.
- [ ] **One measurement harness**, and nothing measures any other way.

### 7 · Commercial and operational
- [ ] **Token accounting exact** — OpenRouter bills on it, and our own bench was miscounting.
- [ ] **Sustained-load soak** — 400 requests passed once; nobody has run hours.
- [ ] **Deploy, monitor, restart-on-crash.**
- [ ] **OpenRouter onboarding + pricing** — requirements unknown; needs a doc read, not a guess.

---

### Standing rules for anything landing here
**Fast path is the DEFAULT** — flags are kill switches, never opt-ins. **Artifacts stay
architecture-neutral; permute at load (D22).** Every arm proves it engaged. Compare **F32 bit
patterns before narrowing**, with a control that must fire — a single-element 1-ULP
perturbation, never an XOR that can cancel. **A scoping result is never a verdict (D21).**
