**Merging this PR means Arc can serve and sell tokens on OpenRouter.** Nothing lands on `master` for this effort except through here.

Base `d7742670a` (the 15.0 → **34.48 tok/s** b=1 ladder, already on master).

---

## 🔴 Blocking — cannot sell tokens at all

- [ ] **1. Time-to-first-token is ~24 s.** ~12 ms per *input* token, ~100× off bound. A user on OpenRouter feels this immediately. **Prefill is a different disease from decode — the GPU is already 99.5% busy, so graphs/launch-reduction buy it nothing.** Needs its own lane and its own kernel work.
- [ ] **2. Prefill crashes.** b=1 on `qtip2b` exits 139 (SIGSEGV) and the harness prints `pp = 0.000±0.000` into a results table instead of failing. A 0.000 is not a measurement.
- [x] **3. The fast build reaches `master`.** Landed as #175/#177/#185–#189 → `d7742670a`. ⚠️ Still owed: re-measure **master itself** and confirm it reads ~34.48.
- [ ] **4. Aggregate throughput on the fast build.** Never measured at any batch size. Sweep B = 1/8/32/64/128, per-request **and** aggregate. Expect flat-then-steep: 256 experts top-6 means tokens barely collide until batch is large. **Finding the knee is the deliverable** — the GPU-hour economics depend on it.
- [ ] **5. Reliability under sustained load.** 400 requests passed once. Nobody has run this for hours.

## 🟡 Product quality

- [ ] **6. Generation speed → 250 tok/s.** Real profile on `qtip2b`: tail (sinkhorn + fast_sum + bmul + ucopy) **27.2%**, `fp8_gemv_warp` **16.0%**, `qtip2b_gemv_tuned_kernel` **15.4%**. 🚩 A **4×4** Sinkhorn costing 9.3% of an H200 is structural, not tuning.
- [ ] **7. Long-context memory, measured.** At 1M ctx the answer is **~3 users or ~55** depending on whether the 128-token window + compress ratios are live. Unmeasured. This is the capacity claim.
- [ ] **8. Prefix caching is silently disabled** by our own default (TurboQuant is the default cache type). Most real traffic is repeated prefixes.
- [ ] **9. Switched-off finished work.** `CudaSampler` — complete, tested bit-exact, **zero callers, no gate**. The **GPU WHILE-node loop** — the one design that removes the per-token sync; neither competitor has it. Blocked by an unconditional `return Ok(None)` at `pipeline/normal.rs:2324` and a U8-dtype bug failing all 28 call sites.
- [ ] **10. 56× KV over-retention** — root-caused previously; the fix reportedly never reached V4.

## 🟢 Trust and correctness

- [ ] **11. The live V4 test fixture is all zeros** — identity tests against it prove nothing.
- [ ] **12. Verification code is less reliable than production code.** Twelve instruments lied in one session: bench counting *requested* tokens (printed **1280 tok/s** for a zero-token run), `cargo clean -p` silently no-opping on git deps, `nsys` hiding graph-internal kernels by default, a stale `.o` satisfying a link, `replay()` returning a clone of the graded buffer, `append_graph` comparing a buffer against itself, an XOR control cancelling over an even tile count, an atomic counter reading 0 while the work ran, a sampler sleeping past the run, bf16 parity vacuous at 8 mantissa bits, a guard dying on Windows CRLF, prefill printing 0.000.
- [ ] **13. Re-confirm quality on the exact artifact we serve.** GSM8K 96.0% exists — on a different rung than the one now in play.

## 💰 Commercial and operational

- [ ] **14. Token accounting must be exact** — OpenRouter bills on it, and we just found our own bench miscounting. Audit `usage` end-to-end.
- [ ] **15. Deployment, monitoring, restart-on-crash.** Not built, not designed.
- [ ] **16. OpenRouter onboarding + pricing.** Requirements unknown to me — needs Jish or a doc read, not a guess.

---

### Standing rules for anything landing here
**Fast path is the DEFAULT** (D22-adjacent policy) — flags are kill switches, never opt-ins. **Artifacts stay architecture-neutral; permute at load** (D22) — one file serves every GPU generation. Every arm proves it engaged; a green result must prove work happened. Compare **F32 bit patterns before narrowing**, with a negative control that must fire — a single-element 1-ULP perturbation, never an XOR that can cancel.
