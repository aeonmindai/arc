# THE V4 PREFILL BUDGET — a DIFFERENT disease from decode

**Read this before proposing any prefill or TTFT work. MEASURED, not estimated.**
Companion to `BUDGET_V4_B1.md` (decode). **The two have opposite causes and opposite fixes.**

Provenance: `arc-v4-stack` (H200), SHA `05af600e`, binary md5 `41d0bc20…`, exclusive card, both locks
held, exclusivity asserted per leg with a selftest proving the assertion can fail.
`arc_profiler` device selftest **PASS 43.0×**, overhead **0.034%**, reconciliation **0 violations /
0 misnested / 0 unresolved**, self-times sum to **100.0%**. Drift control: N=128 re-measured 14 slots
/ 17 min later at **1578 vs 1574 ms = 0.26%**. Session 8, 2026-08-18.

---

## 🔴 THE HEADLINE: PREFILL IS 99.5% GPU-BUSY. DECODE IS 49% GPU-IDLE.

| | decode (B=1) | **prefill** |
|---|---|---|
| GPU idle waiting on host | **49%** | **~1%** |
| kernel launches | 9,131/token | **15,744/step** (N=128) — only **1.78×** |
| but wall clock | 66.68 ms | **1,531.8 ms — 21×** |
| kernel mean duration | 3.81 µs | **92.9 µs — 24× longer** |
| `cuLaunchKernel` host cost | the bottleneck | **50.7 ms of 1,545 = 3.3%** |

> **PREFILL IS NOT LAUNCH-BOUND. CUDA graphs, launch reduction and the arena buy prefill NOTHING.**
> Everything session 8 built for decode is irrelevant here. This needs a different fix.

## Per-step attribution, N=128, B=1 (self-times sum to 100.0%)

| component | calls/step | ms/step | % |
|---|---|---|---|
| **`experts.down/gate/up_proj` — QTIP MoE gather** | **129** | **1092.4** | **71.3%** |
| `q_proj` + `o_proj` (MLA) | 86 | 184.4 | 12.0% |
| `moe.shared_expert` | 43 | 67.4 | 4.4% |
| `sdpa` / `mhc_attn_pre` / `kv_fp8_quant` / `mhc_ffn_pre` | 43 ea | 100.8 | 6.6% |
| everything else | — | 86.8 | 5.7% |
| **step total** | | **1531.8** | **100%** |

**Two independent instruments agree to 0.2%:** profiler says the three expert-gather spans total
**1092.4 ms**; nsys says `qtip_gather_gemv_warp_kernel` is **1090.5 ms across 129 launches**.
That is **8.45 ms per launch** reading **1.61 GB** of packed weights = **190 GB/s ≈ 4% of peak HBM**,
independently corroborated by `nvidia-smi` at **2% memory-controller, 98% SM, 450 W**.
**The SMs are pinned serially decoding the trellis, not moving weights.**

## Counts per prefill step — with a decode control that reproduces the recorded budget first

| | N=128 (fused GEMV) | N=1024 (dequant fallback) | decode control |
|---|---|---|---|
| kernel launches | 15,744 | **76,626** | 8,830 *(recorded: 9,131)* |
| kernel mean / median | 92.9 µs / 1.18 µs | — | 3.81 µs / 1.18 µs |
| `cuMemAllocAsync` | 14,008 | **137,976** | 10,992 *(recorded: 11,436)* |
| blocking `cuMemcpyDtoHAsync` | 44 | **176** | 43.8 *(recorded: 44)* |
| `cuMemcpyHtoDAsync` | 4,169 | **45,274** | 2,577 |
| **ALL `*Synchronize*`** | **0.00** | **0.00** | 0.00 |
| dominant kernel | `qtip_gather_gemv_warp` **74.6%** | `qtip_dequantize_v2_k4_l16` **68.1%**, ~20,307 launches | gather_gemv 29.7% |

*(The control reproducing the recorded decode numbers before any prefill number was trusted is what
makes this table credible — the instrument was validated on a known answer.)*

---

## 🎯 NAMED CAUSE: ArcQuant / QTIP **LUT rung** (`QtipLayer`) — what `qtip2-*.uqff` actually ships

**`gather_forward` has exactly two prefill paths and NEITHER IS A BATCHED GEMM:**

1. **< 683 tokens — fused GEMV**, one (token,slot) pair per `grid.y`, **no dedup**:
   768 pairs against ~244 distinct experts at N=128 ⇒ **3.15× redundant weight reads.**
2. **≥ 683 tokens — a host-synced, per-distinct-expert dequantize-materialize loop**
   (`mistralrs-quant/src/qtip/mod.rs:2887`), with `Tensor::from_vec` in the hot loop and `out_flat`
   reallocated per expert. **Measured launch signature matches the loop body exactly:**
   `qtip_dequantize` **81,228** · `is_u32_bf16` (index_select) **81,641** · `ia_u32_bf16`
   (index_add) **81,073** — all equal.

**⛔ `ARC_QTIP_ONDEVICE_MOE_MAX_TOKENS` CANNOT FIX THIS.** Throughput is the same either side of the
crossover — **86.4 vs 84.7 tok/s** — so the policy is choosing between **two equally bad paths**.

## ✅✅✅✅ CUMULATIVE TTFT — **5.93 → 1.80 s at N=512 (3.3×)**, **16.87 → 5.90 s at N=2048 (2.9×)**

Two shipped PRs, both measured on an exclusive card with fitted estimators.

### PR #138 — the FP8 scalar GEMM (`perf/fp8-prefill-cublas`), N=2048 MEASURED
| slot | arm | TTFT | fitted |
|---|---|---|---|
| 1 / 3 | A | 9590.66 / 9627.47 ms | **c = −3720.69 ms** |
| 2 / 4 | B | **5897.88 / 5896.69 ms** | **1.632× — 9.61 → 5.90 s** |

**4.694 → 2.880 ms/prompt-token.** ⇒ **It beat its own projection (1.61×) by 1.7% — the projection
was PESSIMISTIC, not optimistic.**
🔑 **Slot drift was +8.9 ms/slot here versus −0.27 at N=512 — exactly the bias the fit exists to
remove. A raw A−B would have misread it.** Arm A independently reproduces the recorded post-#133
baseline (9590.66 / 9627.47 vs 9655.92) **within 0.7%.**

**🔑 SHIPPED DEFAULT IS *ON*, NOT OPT-IN — and the reasoning is the session's whole lesson:**
> *"Leaving it behind an unset env var would have made this the TENTH wired-but-unreached — the
> precise disease being fixed."*

`DEFAULT_MIN_M = 512` is **the measured point, not an interpolation**: at N=512 `m_rows == 512`, so
**the shipped default is code-identical to measured arm B.** Decode untouched (own `fp8_gemv_warp`
at M ≤ 4). **The crossover below 512 is unmeasured and flagged as a follow-up.**

### THE ROOT CAUSE — a NINTH wired-but-unreached
`fp8_matmul_tiled` is a **scalar CUDA-core GEMM** (`acc += s_input[ty][k]*s_weight[tx][k]`, **no
tensor-core instruction anywhere**) at **4.4–4.8 TFLOPS**, serving **both** MLA q/o and the shared
expert — **39.4% of the step at N=2048, exceeding the gather** while the recorded attribution still
said the gather was 71%. **A dequantize + cuBLASLt path already existed in-tree, unreachable:
`forward()` consulted device and kernels-compiled but NEVER token count.** Span diff proves
mechanism: q+o **706.1 → 26.2 ms (27×)**, shared **253.2 → 11.9 (21×)**, **gather and sdpa unchanged
within 1%.**

### 🔑 `sum2` REPRICED ON MEASURED NUMBERS — its value is CONTINGENT ON ORDERING
| | gather share | `sum2` worth |
|---|---|---|
| before #138 | 36–44% | **1.32× / 1.26×** |
| **after #138** | **66.6% (N=512) / 59.9% (N=2048)** | **1.59× / 1.50×** |

**The 2.27× headline is worth far less than it reads, and how much depends on what lands first.**
Supporting: the gather's mma floor is **27 ms of 3494 (0.8%)**, so it is **~99% trellis decode** and
the inst/weight transfer function applies nearly undiluted.

### MEASURED MAP TO JISH'S 1-SECOND BAR
| N | now | with `sum2` |
|---|---|---|
| 512 | **1.8× short** | **1.1× — reachable** |
| 2048 | **5.9× short** | **3.9× short** |
**N=2048 needs a third lever, and there is no single hot spot left** — every remaining term sits at
4.4–7.6 TFLOPS, 130–226× below peak; the whole step is 5.6 TFLOPS = **71.7× the 400 TFLOPS
reference, independently reproducing the 72×.** It becomes a broad efficiency problem.

**Filed (issues disabled ⇒ BACKLOG):** `weighted_sum` at 262 ms/N=2048, ~145× above its own
bandwidth floor — **with the honest caveat that no kernel was named for that span (the nsys leg
SIGSEGV'd), so 145× is a ratio against a DERIVED floor, not a measured achievable rate.**

## ✅ (superseded) END-TO-END — TTFT ROUGHLY HALVES (commit `52aec71`, PR #133)

| N | ms/token before | **after** | **gain** | **TTFT** |
|---|---|---|---|---|
| 128 | 12.356 | **8.091** | **1.53×** | **1.58 → 1.04 s** |
| 512 | 11.585 / 11.581 | **5.329 / 5.312** | **2.18×** | **5.93 → 2.73 s** |
| 1024 | 11.811 | **4.901** | **2.41×** | **12.09 → 5.02 s** |
| 2048 | 8.236 | **4.694** | **1.75×** | **16.87 → 9.61 s** |

**Baselines reproduce the independently-established table** (12.36 / 11.58 / 11.81 / 8.24 vs
11.98 / 11.48 / 11.81 / 8.23) — **the harness validating itself against a run it did not produce.**

### 🔴 THE FIRST RE-RUN FALSIFIED THE PLAN: wiring the kernel in was NOT enough
| N | before | after wiring | gain |
|---|---|---|---|
| 128 | 12.352 | 12.385 | **1.00×** |
| 512 | 11.582 | 11.588 | **1.00×** |

**Not noise-masked: five interleaved replicates of IDENTICAL code at N=512 gave 11.578–11.589 — a
0.011 ms/token floor (±0.05%).** ⇒ **the replicate design turned "looks like nothing" into "provably
nothing".** Without it this reads as an inconclusive run and someone re-runs it hoping.

**Cause — an EIGHTH wired-but-unreached, and a general lesson.** `lut_fused_gather_preferred`'s 16×
traffic ratio answers *"GEMV or dequantize-materialize?"* — **two paths that both scale with
(token × top_k) PAIRS.** The grouped GEMM is a **third path scaling with DISTINCT EXPERTS**, so that
boundary **no longer describes the decision at all.** Below ~683 tokens (683·6 ≈ 4096 = 16·256) the
call returned at the on-device GEMV and **never reached the new kernel.**
> 🔑 **A dispatch policy that chooses between two options cannot accommodate a third without being
> rewritten. Adding a code path is not the same as making it reachable** — and the component win
> was real *and* unreachable exactly where most prefill lives.

Decode regime stays fused unconditionally (**RUN-161 floor, not a performance choice**);
`ARC_QTIP_ONDEVICE_MOE_MAX_TOKENS` still overrides so a harness can pin the GEMV arm.

### ⚠️ TWO CORRECTIONS THE CHAIN MADE AGAINST ITSELF
- **The ~3.5 ms/token projection was TOO OPTIMISTIC** — measured **4.69–8.09**. Arithmetic
  projections from a component win overstate; the dispatch and the other 29% of the step both bite.
- **The PR title advertised the 6.6× COMPONENT number** — retitled to the **end-to-end range**.
- **MEMCTRL still FALLS, not rises** (0.5→0.2 at N=128, 6.1→1.8 at N=2048), consistent throughout:
  **decode-ALU-bound, not HBM-bound.** The original criterion still says this is not bandwidth work.

### 🐛 The runner's own failure, and the rule it produced
The e2e runner **did not inherit `LD_LIBRARY_PATH=/usr/local/cuda/compat`** — *the exact trap the
same chain had filed in BACKLOG one script earlier.* Every process died with
`CUDA_ERROR_UNSUPPORTED_PTX_VERSION` **at line 83 of an 18 KB log, well past where it was tailing**,
producing a complete 14-row table of `RAW`.
⇒ **"Don't gate on rc" MUST be paired with a positive engagement assertion.** The runner now requires
each slot to yield a parseable `Prefill (N tokens) … ms (TTFT)` line and **exits 2 on the first slot
that doesn't.** Recorded in BACKLOG.

## ✅ COMPONENT-LEVEL (superseded by the e2e above, kept for provenance)

One MoE expert matmul, E=256, top-6, **exclusive idle card** (`apps=[]`, 143 GB free, builders=0).
Pairs are V4's two shapes 4096→2048 / 2048→4096. Engagement counters written to files.

| N | shipped path | shipped ms | **grouped ms** | **gain** | SM | MEMCTRL |
|---|---|---|---|---|---|---|
| 128 | gemv | 8.38 / 8.62 | 7.46 / 7.14 | **1.15×** | 100→97% | 5.0→**2.0%** |
| 512 | gemv | 33.47 / 34.26 | 8.53 / 8.51 | **3.98×** | — | — |
| 1024 | dequant | 127.87 / 75.02 | 15.41 / 15.37 | **6.59×** | — | — |
| 2048 | dequant | 133.68 / 84.62 | 26.69 / 26.78 | **4.08×** | 100→100% | 6.8→**1.2%** |

### 🔑 THE MEMCTRL ANSWER IS *NOT* THE HOPED-FOR ONE — and that is the finding
**MEMCTRL FELL (5.0→2.0% and 5.0→1.2%); SM stayed 97–100%.** By the stated criterion this is
**NOT** a move to bandwidth-bound. **What did happen: traffic per unit work collapsed ~21× at
N=2048** (1.2%×26.7 ms vs 5.0%×133.8 ms) — so the **3.15× redundant-read defect is genuinely
removed**, but the kernel is now **trellis-decode-ALU-bound, not HBM-bound.**
⇒ **The ~14 ms weight-traffic floor is NOT the operative limit.** At N=128 the grouped kernel costs
**7.45 ms/layer/shape — ~100× above that floor.** **Scope, not verdict: the next lever is decode ALU**
(the V=2 pairing already halves state reconstructions vs the qtip2b rung), **not weight traffic.**

### ✅ NO RE-BAKE NEEDED, ZERO COST
The kernel **templates on the codebook** exactly as every other K=4/V=2 launcher does (`cb_mult==0` →
stored-LUT gather, nonzero → computed MCG), so **it runs on the shipped `qtip2-*.uqff` unchanged**.
*(ISQ-to-`qtip2b` was never the route — it would need Viterbi over the full 284 B source.)*

### Quality: bit-identity was never available; cosine measured instead
mma **f32** accumulation vs the fallback's **cuBLAS BF16** matmul is a different summation order.
**Measured: cosine 1.00000 against the dequantize-materialize reference at all 8 (shape, N) points**,
same decoded weights.

### 🔑 Why this was always reachable — the load-bearing insight
The LUT rung's `state = ((state<<4)|sym) & 0xFFFF` **looks sequential but is just a 16-bit nibble
window**: `state(t) = nibble_reverse_16(bits[4t-12 .. 4t+3])` — **random-access**, exactly like the
qtip2b rung's pair-reversal. **That is why a grouped GEMM was reachable here all along.**

### ⚠️ NOT ESTABLISHED
**No end-to-end V4 ms/token** — the 138 GB model was never loaded; the card was contended by three
chains all session and the chain chose the component measurement over queueing. **So the
11.98 → ~3.5 ms/token projection remains ARITHMETIC.** Also unestablished: the 71.3% component share
at N≠128, and there is no 6-arm slot-fitted design (one exclusive block instead).

**Code:** `kernels/qtip/qtip_grouped_gemm_lut.cu` (new, ~330 lines) · `grouped_gemm_lut_cuda` in
`src/qtip/cuda_ops.rs` · FFI in `ffi.rs` · wired into `QtipLayer::gather_forward` · harness
`examples/qtip_lut_grouped_bench.rs`. Worktree `/root/arc-wt/lutgemm`.

### 🚨 SEVENTH "GREEN THAT COULD NOT CONTAIN THE ANSWER" — and it is live for anyone adding a kernel
**`mistralrs-quant/build.rs` emits `rerun-if-changed` only for the sage/metal paths, so a NEW `.cu`
in the `kernels/*/*.cu` glob does not trigger a build.rs rerun.** A build reported **`BUILD_RC=0`
while the kernel was never compiled.** **Caught only because the "Compiling 33 of 33 kernels" count
did not change** (fix: touch `build.rs` → "1 of 34").
**Operational:** driver 580.173.02 is **CUDA 13.0**, toolkit is **13.1** ⇒ first GPU run died
`CUDA_ERROR_UNSUPPORTED_PTX_VERSION`; needs **`LD_LIBRARY_PATH=/usr/local/cuda/compat`**.
**Also: `qtip_grouped_curve` Viterbi-bakes E=256 single-threaded for HOURS while holding the
exclusive lock** — use a random-packed-bytes fixture instead; **decode cost is content-independent.**

## 🎯 THE ORIGINAL TARGET (superseded above): ~14 ms WEIGHT-TRAFFIC FLOOR vs 1,092 ms TODAY

**A grouped GEMM that reads each distinct expert exactly once has a ~14 ms weight-traffic floor at
N=128.** Against the measured **1,092.4 ms**, that is **roughly two orders of magnitude of headroom
on this one component** — and it is **the same component in both regimes**, so one fix covers the
fused-GEMV side and the dequantize-fallback side together.

**Arithmetic projection, NOT a measurement** (state it as such): step 1531.8 − 1092.4 + 14 ≈
**453 ms**, i.e. **11.98 → ~3.5 ms per prompt token ≈ 3.4× on prefill**. Still above the compute
bound afterwards — the remaining 12% MLA projections and 4.4% shared expert become the next terms.

## 🔑 THE KERNEL ALREADY EXISTS — WE SHIP THE RUNG WITHOUT IT

**The sibling bitshift rung (`Qtip2bLayer`, `--isq qtip2b`) HAS the amortizing
`grouped_gemm_2b_cuda`**, and **V4's shapes satisfy all three of its gates**: 4096 and 2048 both
divisible by `GROUPED_TILE_K` = 64, BF16, uniform 2-bit.
⇒ **The sixth instance this session of correct code the shipped path cannot reach.**

## Scaling, and where "24 s TTFT" comes from

| N | ms/prefill | **ms/token** | MoE path | SM | MEMCTRL | W | vs 400 TFLOPS |
|---|---|---|---|---|---|---|---|
| 128 | 1534 | 11.98 | fused GEMV | 98% | 2% | 440 | 112× |
| 512 | 5876 | 11.48 | fused GEMV | 98% | 2% | 469 | 177× |
| 1024 | 12092 | 11.81 | dequant fallback | 100% | 5% | 243 | **182×** |
| 2048 | 16855 | 8.23 | dequant fallback | 100% | 7% | 259 | 127× |

> **TTFT 24 s = ~2,000 prompt tokens × 11.8 ms on an IDLE EXCLUSIVE H200.** The recorded 43.2 s step
> = ~3,700 tokens at the same rate. **Compute, not starvation, not queueing.**

**Four forward predictions, recorded BEFORE the run, all landed:** crossover at 683 ⇒ fallback
warning absent at 512, present at 1024 ✓ · memory-controller 2%→5%→7% ✓ · fallback D2H ≈129 router
syncs (measured 176 = 129 + 43 KV + logits) ✓ · fallback H2D ≈33k `Tensor::from_vec` uploads
(measured 45,274) ✓.

## Qwen vs V4 — the named difference
**Qwen3-32B is DENSE.** `qwen3.rs` has **zero** `gather_forward`/`MoEExperts`, so every prefill
projection is one batched cuBLAS GEMM. **V4 routes every prompt token through QTIP expert gather.**

## 🔴🔴 PREFILL EXCLUDES DECODE — the likely aggregate ceiling, and it COMPOSES with the above

**`mistralrs-core/src/paged_attention/scheduler.rs:216` — the paged scheduler returns EITHER a
prefill batch OR a decode batch, never both.** The prompt-admission loop **returns early at
`if !scheduled.is_empty()`, before the decode leg ever runs.**

⇒ **Every arriving request stalls ALL running decodes for a full prefill step.**

**Combine with the two facts already on this page and the magnitude is severe:**
- **prefill is NOT chunked** (`get_prompt_input` → one `make_prompt_chunk`; `prompt_chunksize` is a
  stale comment with no wired field), so a prefill step is **the whole prompt, indivisibly**
- a prefill step at N=512 measures **5,876 ms**; a decode step is **66.68 ms**

⇒ **One arriving 512-token request freezes every running user for ~5.9 seconds — ~88 decode steps.**
At B=64 that is a throughput catastrophe, and it is invisible in any single-user measurement.

### 🔑 THIS MULTIPLIES THE VALUE OF THE GROUPED-GEMM FIX
The kernel is worth **4–6.6× on the prefill component** for the *arriving* user. But because that
prefill step **blocks every other user**, the same fix **shortens the stall the whole batch eats.**
**Its aggregate value is larger than its single-user value** — and no single-user benchmark can show
that. *(Chunked prefill would decouple them entirely — see the "not chunked" finding above.)*

**Being tested, not assumed:** two diagnostic legs at B=64 — `decode_heavy` (128-token prompts) vs
`prefill_heavy` (spread prompts, 32 gen tokens). **If aggregate jumps on `decode_heavy`, prefill
interference is named rather than guessed.**

## 🐛 `logit_bias` IS A DEAD WIRE — the API accepts it and silently ignores it
`mistralrs-server-core/src/{completions,chat_completion}.rs` populate `SamplingParams::logits_bias`;
**nothing in `mistralrs-core` ever reads it** — the only references are the field declaration and its
defaults. **A customer can send `logit_bias` on an OpenAI-compatible endpoint and it does nothing,
with no error.** Sixth wired-but-dead of the session, and the first that is **user-facing**.

## Ruled out
- **Prefill is NOT chunked** — `get_prompt_input` → one `make_prompt_chunk`; `prompt_chunksize` is a
  **stale comment with no wired field.**
- **The compressor is fully vectorised** — not a contributor.
- **Batching cannot rescue it:** cost is linear in (token × top_k) pairs with **no cross-token
  amortisation**, and `cache_bucket_len()` falls back to `len()-1` for fresh sequences, so differing
  prompt lengths **serialise one bucket per prefill step**.

## ⚠️ NOT ESTABLISHED — do not quote as fact
- **No qtip2b A/B. The fix is code-identified and UNMEASURED — treat the headroom as scope, not a
  promise.**
- **The Qwen 565–792 tok/s figures are not certifiable on this box** — FACTS records Qwen3-32B on a
  **B200**. The "30× worse" ratio is withdrawn; **nothing above rests on it.**
- **N=1024 per-step counts are ±10%** — derived by subtracting measured decode cost from trace
  totals because segmentation failed there (**196 fragments — itself a symptom of the per-call
  syncs**). The N=128 numbers come from two byte-identical segments and carry no such caveat.

## 🐛 Separate defect, needs its own ticket
**The bench binary corrupts its heap at teardown** — SIGABRT *"corrupted double-linked list"* ×3,
SIGSEGV ×2 — **after printing results.** ⇒ **Never gate a prefill result on the process exit code;**
two runs were nearly discarded that way, and the traces were recoverable from disk both times.
