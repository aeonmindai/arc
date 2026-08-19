# 🔴 MATH CHANGED ON MASTER 2026-08-14 (PR #35 merged, `830a41ed9`).
# **GSM8K 87.0% AND PPL 12.50 WERE MEASURED ON SUPERSEDED MATH.** They are now
# PROVISIONAL and MUST BE RE-MEASURED before any republication. Do not quote
# them as current; quote them as "session-3, pre-clamp/pre-YaRN-fix".
# What changed (both confirmed to alter decode output, agent AI, wave13-AI):
#  - **SwiGLU clamp** was missing on 4 of 5 expert paths — including the SHARED
#    expert, which every token traverses in every layer, on every prior run.
#    Config publishes `swiglu_limit: 10.0`; reference clamps unconditionally
#    (`silu_and_mul_masked_post_quant.cuh:55-73`). Fixture magnitude when it
#    bites: clamped 0.7311 vs unclamped 14.8996 = **20.4x**. Expected direction
#    neutral-to-better, but UNMEASURED on the real model — could land either
#    side of 87.0 and that is not a regression, it is the first number measured
#    on math that matches the reference.
#  - **YaRN** was applied to Standard (ratio-0) layers. Correct ratio-0 set is
#    exactly **{0, 1, 43}** — the audit's "0, 1, 42" was WRONG (layer 42 is
#    ratio-4). Little effect at GSM8K's 2048 cap; matters at long context, so
#    **longctx 5/5 + needle 4/4 are also provisional**. MTP block still carries
#    the wrong RoPE table until #30's owner applies the deferred one-liner
#    (`deepseek4.rs:2707-2716`).
# Speed numbers are effectively unaffected (elementwise clamp) but were also
# taken on the older tree — re-baseline opportunistically, don't retract.
#
# ⚠️ AUDIT INVALIDATION (2026-08-14, PR #27 reference audit) — READ BEFORE
# CITING ANY MULTI-SEQUENCE OR MTP NUMBER:
# - **b_sz>1 prefill had NO CAUSALITY** (sinks_attn_varlen passes None mask) and
#   caller attention_mask was discarded on every live path ⇒ any batched-quality
#   result (incl. voting/n_votes>1) taken before the fix is INVALID. Batch
#   THROUGHPUT numbers remain valid (same kernel work); batched OUTPUTS do not.
# - **MTP acceptance numbers are meaningless pre-fix**: draft seeded with
#   embed(T0) for BOTH h_proj and e_proj (identical vector) + draft KV never
#   prefilled while using absolute RoPE. Fixes in flight (wave12-AB/AC).
# - Bake-pace 6× variance is NOT our code: same-box A/B showed the session-3
#   binary equally slow (1 layer/3min @523W healthy box). Provider/rental
#   variance. FIX THE COST, NOT THE MYSTERY: bake ONCE and reuse the UQFF
#   (needs HF_TOKEN to push to hub) instead of re-baking every session.

# FACTS.md — hardware-measured numbers ONLY. Never publish or reason from
# anything not on this list. Every row: value + how/when measured.
# (Sessions: s1=08-12 H200, s2=08-13 H200, s3=08-13/14 H200, s4=08-14 H200.)

## 🟢 2026-08-19 — SESSION-8 CLOSE. **FOUR MEASUREMENTS, ZERO GPU HOURS.**

**Provenance for the whole block:** compiler and source instruments only — no
card was touched. Companions: `memory/mission/CENSUS_SESSION8.md` and
`memory/mission/LADDER_POST_CENSUS.md` (branch `docs/census-session8`).

### 1. TRELLIS inst/weight — **COMPILED, not estimated**

`nvcc -cubin -std=c++17 -lineinfo -Xptxas -v`, **CUDA 12.4.131** (the version the
in-tree anchors were produced with), block = 256 threads. Method: **difference
two unroll depths** so prologue/epilogue cancel; validated to **1.3% vs ncu**.
`bpw = K/V` ⇒ **all three geometries are 2 bpw and share one budget** — the
error behind the retracted K4/V4 claim cannot recur here.

| geometry | **sm_90** | **sm_80** | static smem | >48 KB? | occupancy |
|---|---|---|---|---|---|
| K4/V2/L16 computed — **SHIPPED** | **15.125**‡ | 14.812‡ | 0 B | no | 100% |
| K4/V2/L13 bf16 LUT | **11.250**‡ | **10.250** | 32,768 B | **no** | 62.5% |
| K8/V4/L12 bf16 LUT | **5.375** | **4.625** | 32,768 B | **no** | 62.5% |
| **K8/V4/L12 + row-scale hoist** | **4.375** | **3.625** | 32,768 B | **no** | 62.5% |

‡ two-point differential; the third point saturated a hard **4096-instruction
unroller ceiling**, so linearity is unverified **for those two rows only**. The
**K8/V4/L12 rows are clean 3-point, 0.00% linearity on both arches** — the
headline does not depend on the pending re-run.

- **HEADLINE: 15.125 → 4.375 = 3.46× fewer instructions per weight, COMPILED.**
- **Still 3.1–3.9× short of the 1.13–1.41 budget on sm_90** (2.6–3.2× on sm_80),
  **and that is the inner loop alone** — it excludes serving scaffolding (expert
  gather, shared staging, tail guards). ⚠️ **Geometry-to-geometry ratios are
  apples-to-apples; absolute values are NOT comparable to the shipped 35.52
  whole-kernel figure.**
- **Both LUT geometries land at exactly 32,768 B — under the 48 KB static limit,
  so NO `cudaFuncSetAttribute` opt-in is needed.**
- **Occupancy is 62.5%, not the predicted 31%, and the limiter is REGISTERS, not
  shared memory.** The "5 blocks/SM" prediction was right; the "20/64 warps"
  half was wrong.
- **Lever measured:** row-scale hoist = **−1.000 inst/wt exactly**, as the source
  predicted. **The random-access window is a REGRESSION at K=8 (+0.375)** — at
  K=8 warm-up is only `L/K = 2` byte-aligned symbols, cheaper than reconstructing
  the window. **Its value is confined to K=4.**

### 2. V4 KV OVER-RETENTION — **30–57×**, source-derived

**361.4 MB/seq → 6.35–11.96 MB/seq at 8k context = 30–57×.**
At **B=32 @ 8k: 11.57 GB → 0.38 GB.** Compaction copies **1.12 rows per row
stored**. *(Supersedes the earlier ~64× / 5.65 MB / 11.6→0.18 GB estimate in
`LADDER_POST_CENSUS.md` §2.1 — quote this row, not that one.)*

### 3. LAUNCH REDUCTIONS SHIPPED — **9,131 → 8,650 per token**

−430 dead mHC casts, −43 router-weight casts, −8 per sequence per token from the
sampler. **Casts: 1,571 → 1,141.** Counts, not timings — immune to host drift and
neighbours, which is why they were run first.

### 4. FROZEN GUMBEL — **1 distinct token before, 64 after**

Flat 64-token distribution, **512 draws**. A frozen Gumbel key collapsed the
entire sampler to a single token; the fix restores full support. **A sampler bug
that produces valid-looking tokens is invisible without a distinct-token count.**

### 🔴 RETRACTED 2026-08-19 — do not quote any of these

| claim | status |
|---|---|
| "K4/V4/L12 → 1.38 inst/wt is at budget, full 16,602 recoverable" | **RETRACTED** — 1 bpw scored against a 2-bpw budget (`bpw = K/V`, `qtip/mod.rs:376-381`) |
| "MTP acceptance p ≈ 0.485" | **RETRACTED** — it appears **nowhere in the tree**. The measured value is **0.4194** (`wave51-CB-the-measurement.md:166-186`, and the row further down this file), B=1 only |
| "The 2-bit draft head explains the acceptance gap" | **RETRACTED** — the `floor_mtp_isq` fix `07766cfa1` is an **ancestor** of the measured binary `46ea6948d`. The gap was measured *with* the fix in |
| "Fixing acceptance is worth ×5.53" | **RETRACTED — it is ×1.55** against the measured base. A multiplier on a 6× problem, not a fix for it |
| "PD disaggregation is behind every ≥1.5k tok/s/GPU figure" | **RETRACTED** — it is the **metric definition**: `(input+output)/GPU` at isl 8192 / osl 1024 = a **9× multiplier**, plus a TP≥4 divisor and a 2.97× spec multiplier |
| "The re-bake gets cheaper at K8/V4/L12" | **RETRACTED, INVERTED** — the production baker is **beam, not exhaustive** (213 vs 8,257 s/layer, A100) and is **issue-bound, not bandwidth-bound**. Cost is `(n/V)×W×2^K` ⇒ **~8× MORE**: ~213 → **~1,700 s/layer ≈ 20 h ≈ $30** |
| "The default config silently disables prefix caching" | **NARROWED** — CUDA **and** PagedAttention **and** standard layout **and** head_dim 128. **V4 is MLA, so it does not apply to the flagship** |
| "Expert parallelism is env-only" (and its mirror, "`ep_size` is a config field, **not** an env var") | **BOTH WRONG — there are TWO doors.** `ep_size` is a serde `config.json` field (`deepseek4.rs:317-318`, default `1` at `:140`) **and** `ARC_EP_SIZE` overrides it at run time (`effective_ep_size()`, `:371-375`). `build_expert_parallel_plan` is production-reachable (`:2227` → `:2300`). The real gap is "no CLI/server flag" |
| "`CrossPrefixMeter` is dead" | **WRONG** — it records in production; only its readout is test-only |
| "There is no runtime LUT on the GPU" | **NARROWED** — true only of `qtip_gather_gemv.cu`. `qtip_gemv.cu:266` still selects a stored Gaussian LUT, and that is the **shipped default** (`mod.rs:570`) |
| "`ragged_decode_supported`" | **DOES NOT EXIST** — the predicate is `batch_can_be_ragged` (`kv_cache/mod.rs:1227`) |
| "head_dim 512 is the blocker" | **INVERTED** — 512 compiles; **448** fails (`vec_size 14 % 8 ≠ 0`) |
| "SGLang uses page_size=1" | **WRONG FOR V4** — overridden to **256** |

## 🔴🔴 2026-08-17 — **AGGREGATE THROUGHPUT IS SATURATED AT ~37 tok/s, AND ON
## LENGTH-DIVERSE TRAFFIC IT COLLAPSES TO ~13 — WHAT ONE USER GETS.**
##
## **Read the saturation first, not the ratio.** Uniform arm: B=32 → B=128 is
## **4× the users for +6.4% throughput** (35.40 → 37.65). B=1 → B=128 is **128×
## the users for 3.3×**. So the ceiling is low *even with perfect bucketing* —
## fixing the scheduler alone does NOT buy the fleet multiplier. The 2.84×
## bucketing penalty sits on top of a machine that has already stopped scaling.
## (The ratio alone invites "fix the bucketing and you're fine". It is not fine.)
##
## The scheduler's one-bucket-per-step rule is the ceiling on diverse traffic;
## something else — not the kernels either, since arithmetic intensity should
## still be rising at B=128 — is the ceiling on uniform traffic. **The second
## ceiling is unidentified and is the more important open question.**
##
## ⚠️ **DO NOT nominate a cause for it without measuring.** The obvious suspect
## is the per-step KV re-materialisation (`clone_in_cache`, two `Tensor::zeros`
## per layer per step) — work that grows with batch and produces no tokens. That
## is a **candidate on a mechanism argument, not a measurement**, and tonight
## produced a clean lesson in what that is worth: the graph chain's
## drain-ordering theory explained the shape of the evidence perfectly and its
## trigger (`cuMemPoolDestroy`) fired **zero** times on the failing path.
## **The falsifying test is cheap:** measure allocation time as a fraction of
## the decode step at B=32 vs B=128 on the *uniform* arm. If it is not growing
## with batch, it is not the second ceiling however good the story sounds.
## Nobody has run it. Until someone does, the second ceiling has no owner and
## no cause — and five chains tonight worked downstream of it.
## (An earlier revision of this header said "8 users produce HALF of 1 user".
##  That rested on an unmatched B=1 control and is RETRACTED — see the
##  correction note below. The B=128 cell above supersedes it and is stronger.)
## (batchinv chain, arc-graph-probe H200, V4 qtip2b, **be29f397** provenance-
##  asserted, `--paged-attn off --prefix-cache-n 0 --max-seqs 256`, graphs OFF,
##  every leg bracketed by an exclusivity assertion. Controlled A/B: same B,
##  same total prompt tokens, **only prompt-length diversity differs**.)

↩️ **CORRECTION (same session, second pass).** The B=1 reference below was taken
with a **50-token** prompt while every B≥8 arm used ~275-token prompts, so it
was not a matched control. Re-measured at 275 tokens: **B=1 = 10.99 / 11.94
tok/s** (two arms, same workload). The honest restatement is therefore **"8
concurrent users deliver ~two-thirds of a single user's aggregate throughput"
(7.91 vs ~11.9), not "half"**. The direction is unchanged and the A/B between
the two B=8 arms is unaffected — both used the same prompt budget as each other
— but the single-user comparison was inflated by my own workload mismatch and
the "half" figure must not be quoted. Kept visible rather than silently edited
because it was already quoted upward.

| B | arm | prompt lengths | aggregate tok/s | decode-only tok/s | engine's own log |
|---|---|---|---|---|---|
| 1 | — | one, 50 tok ⚠️ *not matched* | 15.36 | 16.92 | 1 running |
| 1 | — | one, 275 tok ✅ matched | **10.99 / 11.94** | — | 1 running |
| 8 | uniform | all identical | **24.54** | 26.0 | **8 running**, 0 waiting |
| 8 | spread | 8 distinct (50–500) | **7.91** | 7.84 | **1 running, 7 waiting**, sustained at 15.0–15.4 T/s |
| 32 | uniform | all identical | **35.20 / 35.40** (two passes) | 125.76 | **32 running** |
| 32 | spread | 8 distinct | **12.79** | 12.69 | **4 running, 28 waiting** throughout |
| 128 | uniform | all identical | **37.65** (clean restart) | 37.32 | **127–128 running** |
| 128 | spread | 8 distinct | **13.26** | 13.08 | **16 running, 80 waiting** |

**⇒ B=128 IS THE DECISIVE CELL: 2.84× wall-clock, 2.85× decode-only.** The two
metrics agree to 0.01×, so no part of this rests on prefill accounting. Both
arms ran on a freshly restarted server with exclusivity asserted in and out.

**⇒ EVERY batch size with realistic length spread is BELOW single-user
throughput.** B=1 ≈ 11.9 · B=8 spread 7.91 · B=32 spread 12.79 — flat-to-negative
across the whole ladder, against a uniform arm that reaches 35–37. Not
diminishing returns: **the concurrency is being thrown away.**

**⇒ Even the BEST case saturates.** Uniform arm: B=1 11.0 → B=32 35.4 (3.2×) →
B=128 37.1 (3.4×). A 128× increase in users buys **3.4×** aggregate. Against
`CEILINGS.json`'s 16,600 tok/s aggregate target, this build is ~450× short, and
the gap is scheduler shape, not kernel math.

⚠️ The B=128 uniform 37.07 was taken while orphaned sequences from a previous
run were still resident — my drain guard printed `WARNING: engine did not go
idle before this run`, so it is **flagged, not clean** (see BACKLOG: sequences
survive their clients and are never reaped). A clean B=128 pair was re-run on a
restarted server.

**⇒ 3.10× wall-clock / 3.32× decode-only, at the same batch size.** This is the
load-bearing number: both arms pushed the same prompt budget, so it is immune to
the control error corrected above.
**⇒ B=8 spread (7.91) is ~⅔ of a single user at matched context (~11.9).
Batching is not merely failing to help on diverse traffic — it is worse than
serving one user at a time.**

**The law, `running bucket = B ÷ distinct lengths`, confirmed at four points and
by two independent instruments** — the engine's own `N running` line, and a
client-side measure that bins streamed token-arrival times into one-engine-step
windows and counts distinct rows making progress (they agree everywhere):

| B / distinct | predicted | measured (client-side) |
|---|---|---|
| 1 / 1 | 1 | median 1, mode 1 |
| 8 / 1 (uniform) | 8 | median 8, mode 8 |
| 8 / 8 (spread) | 1 | median 1, mode 1 |
| 32 / 1 (uniform) | 32 | median 32, mode 32 |
| 32 / 8 (spread) | **4** | **median 4, mode 4, p10–p90 = 4–4** |
| 128 / 1 (uniform) | 128 | median 128, mode 128 |
| 128 / 8 (spread) | **16** | **median 16, mode 16** — engine logged `16 running, 80 waiting` |

🎯 **The B=128 spread point is a FORWARD PREDICTION, not a fit.** 16 was written
down from the law before the run and both instruments returned exactly 16.

⚠️ **Instrument caveat, recorded because it nearly produced a wrong answer:** a
fixed 0.5 s window read the B=8 spread arm as "8 concurrent" while the engine
logged 1 — it was counting rows the scheduler served in *different* steps as if
they shared a forward. The window must be ~one engine step and anchored to the
**global** b=1 step, never to the run's own inter-token gaps, which widen in
exact proportion to the effect being measured and hide it.
Aggregate throughput tracks the *running bucket size*, not B. With 8 distinct
lengths the engine ran ONE sequence at a time at exactly the single-stream rate
while 7 sat waiting — visible in its own `N running, M waiting` line.

**Mechanism, and it is two costs, not one.**
1. `scheduler/default_scheduler.rs` buckets by EXACT cache length and runs one
   bucket per step, because the batched cache is one dense `[B,H,L,D]` built
   from `seqs[0]` with a single shared write offset. Its own comment records the
   steady state as `32 running, 32 waiting` at B=64.
2. A bucket *switch* is not free: `clone_in_cache` (`kv_cache/mod.rs:1004-1005`,
   verified at this ref) allocates **two fresh `Tensor::zeros` per layer, per
   step**, sized by the current context length — 43 layers × 2 on V4. With
   diverse lengths the scheduler switches constantly and pays this every switch.
   My B=32 spread arm was slower than serialisation alone predicts, which is
   what points at the switch cost on top.

🔗 **Same root as the graph chain's finding — but read the status line before
citing it.** The KV cache is re-materialised per step instead of written in
place. ArcGraph's capture work trips over it as unstable allocation sizes
(`4096 × {18,19,20,21}`, one per decode step, un-warmable); this chain trips
over it as a per-switch re-stack. Plausibly **one fix, two subsystems**.
⚠️ **The two halves are at DIFFERENT evidence levels and must not be quoted as
one.** The throughput numbers above are **measured**. The graph chain's
drain-ordering fix for the heap corruption was **TESTED AND FAILED** — A/B on a
fresh binary with all three fix markers verified present: graphs-off served 24
tokens alive with 0 glibc diagnostics, graphs-on died with `corrupted
double-linked list` and 0 tokens, identical to pre-fix.
**Why it failed, recorded so nobody re-derives it:** the crash is at
`cuGraphInstantiate`, and `cuMemPoolDestroy` appears **zero** times before it, so
none of the fix's drains ever ran on the failing path. The mechanism explained
the *shape* of the evidence (capture-only, varying diagnostic, moving abort
site) without its trigger ever firing. A causal story that fits is not a cause.
⇒ "One fix, two subsystems" remains a **hypothesis about a shared cause**, and
the graph half is now *less* demonstrated than when this entry was written, not
more. What survives on that side is a single capture-time event: a `172032`-byte
allocation (`4096 × 21 × 2` = hidden_size × tokens, BF16) missing the warm pool,
which is the `xs` history buffer.
🔴 **UNOWNED WORK:** making the `xs` allocation size constant per decode step is
**not assigned and not started** as of this entry. It is not this chain's work
(this chain wrote zero lines of code); the `xs` commits `6e408eb2e`,
`73a0c72e3`, `4cad4976e`, `8bc6af45c` belong to another author. Needs an owner.

### The production cost, in one line
**On traffic with realistic prompt-length diversity, an H200 serving V4 delivers
~13 tok/s aggregate no matter how many users are on it — the same as ~1 user —
while the same hardware on length-uniform traffic delivers ~37.**

⚠️ **Measured vs not, stated exactly.** The 2.77×/2.84× ratios at B=32 and B=128
and the 3.10× at B=8 are **measured**. The uniform-arm saturation curve
(11.0 → 35.4 → 37.65) is **measured**. Not measured: any B above 128; any prompt
mix other than the two synthetic arms; and real production traffic, whose length
distribution is unknown to me — the 8-way spread is a plausible stand-in, not a
trace. A B=1 point taken with a 275-token prompt is **excluded** — it ran while
a killed sweep's requests were still draining (`1 running, 2 waiting`) and read
3× low, which is why the harness now probes the engine to idle before every
timed run. Instantaneous `tps=` values in the engine's 5 s logger (e.g. 474, 2201)
are **not** throughput results — they are momentary samples over a window that
often contains prefill, and must never be quoted as aggregate numbers.

## 🔴 2026-08-17 — **TEXT-LEVEL BATCH A/Bs ARE UNSOUND ON THIS ENGINE.**
## Greedy output at `temperature 0` depends on the batch the request happens to
## land in. It is REPRODUCIBLE only when the realised batch shape is held fixed,
## and the caller does not control the realised batch shape.
## (batchinv chain, arc-graph-probe H200 SXM5 143 GB, V4 qtip2b UQFF,
##  **be29f397** — the server's own logged `git revision:` asserted equal to the
##  ref built — `--paged-attn off --max-seqs 256 --seed 1234`, temperature 0,
##  greedy, zero `ARC_*` env vars set. 3 server processes (pids 59055 / 66711 /
##  69466), ~350 generations, batch sizes 1..32.
##  EXCLUSIVITY: the later legs are bracketed by an assertion that
##  `nvidia-smi --query-compute-apps` was EXACTLY my server's pid, in and out;
##  the first sweep predates that gate but was spot-checked the same way and
##  `dmesg` attributes **no** Xid to any of my three pids. No neighbour was
##  resident for any leg.)

### 🟢 Cross-process reproducibility check — the numbers below are not a one-off
Two independent server processes ~20 min apart, same prompts, same sweep:
| B | pc16 process | pc0 process |
|---|---|---|
| 2 | 7/8 diverge, first-diff char 24 / 51 / 157 | 7/8, **24 / 51 / 157** |
| 4 | 6/8, 24 / 51 / 156 | 6/8, **24 / 51 / 156** |
| 8 | 5/8, 24 / 51 / 156 | 5/8, **24 / 51 / 156** |

Identical to the character at B=2/4/8 across a full server restart. **The batch
effect is a deterministic function of batch shape** — it is not noise. (B=16/32
drift by one or two rows: 13/16 vs 12/16, 24/32 vs 26/32. That residue is the
admission jitter described below, not the arithmetic.)

### The effect, bounded
| question | answer |
|---|---|
| At what batch size does it start? | **B=2.** 7 of 8 equal-length prompts differ from their own B=1 output over 48 greedy tokens |
| Is it only "B=1 vs B>1"? | **No.** Every batch size disagrees with every other: B2↔B4 5/8, B4↔B8 5/8, B8↔B16 5/8, B16↔B32 4/8 differ |
| Earliest divergent token | **#5 of 48** at B=2; **#2** at B=32 |
| Does it grow with B? | **No.** The perturbation saturates immediately; only the divergence *rate* creeps up |
| Composition-dependent beyond size? | **No.** One probe × 7 different companions × both orders at fixed B=2 → 0/14 changed its output. (Taken in the warm-cache window; contamination inflates divergence, so a **negative** result there is conservative, not an artifact. Worth one clean re-run.) |
| Position-dependent within a batch? | **Essentially no.** 8 identical prompts in one batch agree slot-for-slot in every clean run |

### Why — and it is ORDINARY ARITHMETIC, with the numbers to say so
| quantity | measured |
|---|---|
| Logit dtype on the wire | **BF16** — 19,200/19,200 recovered logits (100.00%) lie exactly on the bf16 grid |
| Logit magnitude / 1 ulp | ~25.6 ⇒ **1 ulp = 0.125** |
| Max step-0 batch-induced \|Δ logit\|, identical context | **0.500 = 4 ulp** (relative 2.07e-2), same at B=2 and at B=32 |
| Fraction of decode steps that are an **EXACT** top-2 tie | **3.52%** (135 of 3,840) |
| ⤷ within 1 ulp | 9.67% |
| ⤷ within 4 ulp | 26.57% |
| Per-step flip rate implied by 7/8 diverging over 48 steps | **4.24%** |

**4.24% observed vs 3.52% exact ties.** The measured divergence rate is what an
approximately-1-ulp perturbation predicts against the measured tie density —
i.e. the batch effect is *tie resolution under sub-ulp noise*, nothing larger.
The divergent text is synonym-level and fluent ("a reddish-brown **color**" vs
"a reddish-brown **hue**"; "calm" vs "calmness"). **This half is not a defect.**

### 🔴 The half that IS a defect: reproducibility, not batch-invariance
| condition | same request, re-run |
|---|---|
| B=1, prefix cache never hit (0.00% hitrate) | **IDENTICAL 5/5**, and 5/5 again after heavy traffic |
| B=8 of 8 *distinct* prompts, cache cold | **IDENTICAL 5/5 across all 8 slots** |
| B=8 of 8 *identical* prompts, cache cold | **2–3 distinct outputs of 5** |
| B=1, after prefix-cache hitrate reached 7–22% | **2 distinct of 3** — but see the strength caveat below |

**1. ESTABLISHED — the realised batch shape is not stable.** Eight identical
requests fired through one barrier were logged as `7 running, 1 waiting` on one
rep: one row was admitted a step late, took a different cache length, and
thereafter sat in a different bucket. The scheduler buckets by EXACT cache
length (`scheduler/default_scheduler.rs`) and runs one bucket per step, so a
one-step admission jitter changes the batch shape for every row. The math is
deterministic **given** a batch shape; the batch shape is not, and the caller
cannot pin it. This alone is sufficient to explain why 8 *identical* prompts
diverge while 8 *distinct* ones (which partition stably) do not.

**2. SUSPECTED, NOT ESTABLISHED — prefix-cache state may break B=1 too.**
⚠️ **n=3 in a single window. Do not quote this as measured.** What is solid:
20/20 B=1 repeats were byte-identical at 0.00% hitrate, across two server
processes; the only 3 B=1 repeats ever taken while the hitrate was 7–22% gave 2
distinct outputs, some degenerating into echoing the prompt. Association, one
window, tiny n — the mechanism is plausible (a hit changes how many tokens are
actually prefilled, hence the GEMM shapes) but **I did not reproduce it.**
I tried and failed: a deliberate forced-hit leg (26-token and 55-token prompts
behind a shared 20-token prelude, fired together after warming) never moved the
hitrate off 0.00%, and both rows matched their solo output exactly, 3/3.
**Hits are erratic**: of three fresh server processes running the same sweep,
only ONE ever left 0.00%, and only after ~6.5 min / ~200 sequences. Closing this
needs a leg that can force a hit on demand. Until then, treat "restart the
server and stay cold" as the safe recipe and don't build on the mechanism.

### The consequence every future measurement inherits
- **Never A/B generated TEXT across batch sizes on this engine.** The control
  diverges from itself. A comparator that does this returns noise; one sibling
  chain already had a real finding blocked because its control diverged.
- If you must compare text, hold the batch shape fixed, run
  `--prefix-cache-n 0`, and **prove the control is stable first** (same request
  ≥5×, byte-identical) before reading anything into the treatment. That check is
  cheap — 5 × 3 s — and it is the step whose absence made a sibling chain's
  comparator return `VERDICT[uncontrolled]`.
- Prefer **logits over text**, and quote deltas in **bf16 ulp**, not characters.
- **Timing/throughput numbers are unaffected** — same kernel work.
- Recorded, not fixed: `sample_argmax` reports `log10(raw_logit)` as `logprob`
  (`sampler.rs`) — positive values, NaN for negative logits, base 10 not e. The
  top-k *ranking* is exact; the numbers are not logprobs. (Recovering the raw
  logit as `10**logprob` is what made the ulp analysis above possible.)

## 🟢 THE b=1 FORWARD/HOST SPLIT — V4's first, and it CORRECTS a claim this
## file has been used to support (arcgraph chain, 2026-08-17, arc-prefill-curve,
## H200 SXM5 143 GB, qtip2b UQFF, `--max-seqs 1 --paged-attn off --prefix-cache-n 0`)
| quantity | measured |
|---|---|
| b=1 decode, 24 tokens, temp 0 | **15.39 tok/s** ⇒ **65.0 ms/step** |
| ⤷ re-measured on a second box, provenance-asserted | **15.51 tok/s** ⇒ **64.5 ms/step** |
| eager V4 forward, **sync'd**, at b=1 | **57.27 ms** (and 49.9 / 54.4 / 59.4 / 59.7 ms across later runs) |
| ⇒ share of the step INSIDE the forward | **88%** |
| b=1 bandwidth floor (3.396 GB active ÷ 4.8 TB/s) | 0.71 ms ⇒ 1,413 tok/s |
| ⇒ the forward alone is off its own roofline by | **~80×** |

**THREE INDEPENDENT MEASUREMENTS NOW AGREE at b=1: 15.11, 15.39, 15.51 tok/s.**
The 15.51 came from `arc-graph-probe` (H200 SXM5, Atlanta) on a binary whose
commit was **asserted at runtime** — `arc_assert_running_revision` matched the
server's own `git revision:` line against the built SHA — so it is the first of
the three whose provenance is proven rather than assumed. Eager forwards of
49.9–59.7 ms against a 64.5 ms step are consistent with the 88% split below.

**PROVENANCE, stated exactly.** Binary was `git revision: ab42c4508` (#104's
branch), NOT the chain's own branch — the run hit the stale-binary bug recorded
below, discovered afterwards from the server's own log line. That makes these
numbers **master-equivalent V4 decode at ab42c4508**, which is a real commit and
a legitimate baseline; it does NOT make them a measurement of anything the
arcgraph branch changed. The 15.39 tok/s independently corroborates the 15.11
recorded elsewhere in this file, which is the main reason to trust the pair.
`ARC_CAPTURE_STREAM/ARC_V4_CAPTURE_PROBE/ARC_CANDLE_ALLOC_CACHE` were all UNSET
for the 15.39 leg; the 57.27 ms came from the capture leg's deferred-free pass,
which is the same eager forward with the caching allocator on.

### 🔑 WHY THIS CORRECTS SOMETHING — b=1 AND B=256 ARE NOT THE SAME REGIME
This file records eight host costs ("EIGHT MORE HOST COSTS, ALL FIXABLE-NOW")
and, next to them, the B=256 evidence: **one core pegged, GPU at 0–4% and 121 W
of 700 W**. That is real, and at B=256 the engine loop genuinely dominates.

**It does not transfer to b=1, and it was being read as if it did.** At b=1,
57 of 65 ms is *inside the forward*; the entire engine loop — detokenize ×5/token,
`get_mut_group!` busy-wait, the O(B) cache clone, the serial responder `.await` —
shares the remaining ~8 ms. Both statements hold at once:

* **B=256: host/engine-loop-bound.** Attack the eight host costs.
* **b=1: forward-bound.** The eight host costs are ~12% of the step; fixing all
  of them cannot move b=1 much. The b=1 gap is the forward being ~80× off its
  own bandwidth floor — kernels and/or in-forward dispatch.

Anyone quoting "we are overhead-bound" must now say *at which batch size*.
Pointing b=1 work at the eight host costs would have been aimed at ~12% of the
step. This distinction is the reason the split was worth measuring at all.

### 🔑 WHERE CUDA GRAPHS CAN AND CANNOT HELP — the ceiling, both regimes
Proposed 2026-08-17: *"the CPU is pegged at B=256 because nothing is replaying;
graphs would quiet the host loop."* **Tested and it does not hold.** Three ways,
all from numbers already in this file:

1. **Arithmetic.** One sweep, nothing mixed (wave51-CB): B=1 = 18.27 tok/s ⇒
   54.7 ms/step; B=256 = 111.69 tok/s ⇒ 0.436 tok/s/user ⇒ **2,292 ms/step**.
   Apply the measured b=1 split (88% forward) ⇒ host ≈ 6.6 ms at B=1. Items
   ②③④⑦⑧ are O(B) by construction, so 6.6 × 256 = **1,690 ms = 74% of the
   B=256 step**. Residual 602 ms = the forward, up 12.5× for 256× the tokens —
   sub-linear, as batched GEMMs should be. Self-consistent.
   *(ESTIMATE, not measured: assumes host cost strictly linear in B and carries
   the b=1 split across sweeps.)*
2. **Mechanism.** ⑥ `get_mut_group!` is a `try_lock`+`yield_now` BUSY-WAIT taken
   5× per token per sequence — this file already calls it *"the mechanism that
   converts contention into 100% of one core with the GPU idle"* — and ⑤ is a
   serial `responder.send()` await holding the pipeline mutex ⇒ 43 of 44 cores
   idle. **"One core pegged, 43 idle" is a serialization signature, not a
   launch-dispatch signature.** A spinlock pegs a core by construction; a CUDA
   graph does nothing to it.
3. **Scaling direction.** Graphs remove per-step launch dispatch, which is
   **O(1) in B** — the same kernel count at B=256 as at B=1. Its share therefore
   SHRINKS as B grows.

⇒ **THE CEILING, AND IT IS THE OPPOSITE OF THE INTUITION:**
| regime | max share of the step graphs could remove |
|---|---|
| **b=1** | **~88%** (the forward's share) — *iff* the forward is dispatch-bound, still unmeasured |
| **B=256** | **~26%**, and only the dispatch fraction of that |

**Graphs matter MOST at b=1 and LEAST at batch.** The reflex — "GPU idle at
batch ⇒ turn on graphs" — points the fix at the regime where its ceiling is
lowest. The cheapest attack on the B=256 collapse is ⑤ and ⑥ (a serial await
and a busy-wait), which need no graph at all. This file already said it:
*"graphs attack launch overhead INSIDE the forward, and every item below is
host work that exists regardless of dispatch."*

### ⚠️ b=1 CORROBORATION EXTENDED, and one caveat for anyone comparing TEXT
b=1 decode has now been measured **five** times across two boxes: 15.11, 15.39,
15.45, 15.51, **15.90** tok/s. The last three are provenance-asserted (the
server's own `git revision:` line matched the built SHA before any number was
banked).

⚠️ **Greedy output is NOT reproducible run-to-run once the prefix cache warms**
(measured by the batch-invariance chain on `arc-graph-probe`, 2026-08-17): a
fresh server is genuinely cold for ~6.5 min / ~200 sequences because the cache
is only populated when a sequence COMPLETES; after hits begin, the same request
at temperature 0 gives 2–3 distinct outputs and some degenerate into echoing the
system prelude mid-answer. Timing legs are unaffected. **Anything comparing
generated TEXT across legs must run `--prefix-cache-n 0` or stay inside the cold
window** — and "I restarted the server so it's clean" is true for about six
minutes and false after, with nothing in the logs announcing the flip.
(Attribution held as correlation until that chain's `--prefix-cache-n 0` control
lands.)

### WHAT IS STILL OPEN
**Whether the b=1 forward is dispatch-bound or kernel-bound is NOT answered.**
The CUDA-graph probe that would answer it (`T_graph/T_eager` from a captured
graph replay vs the eager forward) did not produce a number: capture RECORDS on
V4 but the process dies at instantiate/launch. Do not infer the answer from the
88% figure — 88% locates the cost inside the forward, it does not split dispatch
from compute. That verdict decides whether tiers 2–3 of
`project_cuda_graph_plan.md` are worth building, and it remains unmeasured.

## 🔵 TURBOQUANT KV ON A B200 — a measurement this file had ERASED
## (recorded 2026-08-17 after Jish: *"And turboquant kv was fucking measured too"*)

**MEASURED — `4eba13905`, 2026-04-06, 1×B200 (sm_100, CUDA 12.8), via Modal:**
- **55 tok/s** decode with TurboQuant K4/V3 paged KV, **correct output**.
  Commit message: *"55 tok/s with TurboQuant = 46% over Candle baseline."*
- Corroborating rung, same day, `b00da214c`: *"Eager dedicated path: 50+ tok/s
  with TurboQuant (34% over Candle baseline)."* The two imply a **Candle
  baseline of ~37.5 tok/s** (55/1.46 = 37.7; 50/1.34 = 37.3) — consistent.
- **Protocol / harness:** `deploy/modal_b200.py` (in the tree, added
  `404ee1aad`): `gpu="B200"`, `MODEL="Qwen/Qwen3-32B"`, launched as
  `mistralrs serve --port 8000 --model-id Qwen/Qwen3-32B --pa-cache-type
  turboquant`. Deploy counter ran v73→v83 across 2026-04-07/08, i.e. many real
  redeploys, not one speculative script.
- **Corroborating hardware-only findings from the same window** (unobtainable
  without running): `cuBLASLt: NOT_SUPPORTED (status=15) for BF16 on B200`;
  `cublasGemmEx not capture-compatible on CUDA 12.8 + B200 (status=14)`;
  `cudaDevAttrMaxSharedMemoryPerBlockOptin returns 228KB on B200`;
  `cublasSetWorkspace_v2 breaks cublasGemmEx on B200 — garbage output`.
- **CUDA correctness fixes found against that hardware, 2026-04-02:**
  `143b5ab20` V-cache stride mismatch · `fd0074792` Q·K warp-reduction deadlock
  · `f9f9ff738` warp-level attention rewrite · `562225ade` sub-byte packing
  (K4/V3) · `65440f244` block-count fix for sub-byte packing · `b665facab` /
  `1ba9f11db` dtype handling · `ab1228864` MLA KVCache 4-tuple.

**NOT MEASURED — do not let this row inflate:**
- The **+46% is NOT attributable to TurboQuant.** It compares Arc's whole
  dedicated decode path (custom GEMV, zero-cuBLAS, TurboQuant KV) against
  Candle's forward path. **No A/B against an unquantized KV cache has ever
  been run**, so no speed delta belongs to compression alone.
- **Scope is one point:** b=1, one card, one model, head_dim 128, `Default`
  (K4/V3) preset. `TurboQuant3` / `TurboQuantAggressive` have never executed.
  No Hopper (H100/H200) TurboQuant serving run exists.
- **Context length and quality were never recorded.** There is **no perplexity,
  GSM8K, LongBench or any other eval under any TurboQuant preset, at any
  width.** The paper's "lossless LongBench" is Zandieh et al.'s, on their
  model — published, not reproduced.
- **4.27× KV reduction stays RETRACTED.** It is format arithmetic: 8 KV heads ×
  head_dim 128 → dense `8×128×2×2 = 4096` B/layer vs packed `8×(64+52+4) = 960`
  B/layer. Derivable from the block layout with no forward pass. PR #101
  retracted it deliberately; that retraction is correct and stands.
- **1,026 → 260 B/token for the V4 path stays a DESIGN CLAIM.** V4's head_dim
  is 512; no kernel executes there. Arithmetic over unrun code.

**WHY THIS WAS MISSED, so it is not missed again:** the run was on **Modal**,
not a rental. `docs/PEAK_INFERENCE.md` truthfully said *"no B200 has ever been
rented"*; successive audits read that as *"no B200 evidence exists"* and wrote
"never measured" into the public docs. **"Never rented" ≠ "never ran."**
Provenance searches must cover `deploy/`, not just the rental ledger.

## 🔴 KB FAILURE, recorded next to D18 — a MEASURED result was repeatedly
## relabelled "UNMEASURED" by successive agents until the owner corrected it.
`4eba13905` sat in `git log` the whole time. wave61-CL even **quoted it**
("This is the only throughput number ever tied to TurboQuant") and still
concluded "never measured"; it called the model "unstated" when
`deploy/modal_b200.py` names it one file away. From there the phrase
propagated verbatim into README, ARC_V2, FLEET, BENCHMARKS, PEAK_INFERENCE,
RELEASE_NOTES, TAXONOMY, CODE_INDEX and two crate-level `//!` docs, and main
relayed it to Jish without catching it. Corrected 2026-08-17.
**The rule this establishes: calling a measured result unmeasured is the same
failure as calling an unmeasured one measured — both misstate the evidence.**
The honesty sweep that (correctly) killed four false claims that day also
killed a true one, because "retract it" became cheaper than "check it." Before
writing "never measured" about anything, `git log --all --grep` the feature
name and read `deploy/` — absence of a FACTS row is **not** evidence of
absence of a run; it may just mean nobody wrote the row. This entry is that
missing row.

## ⚠️ BAKE-TIME CORRECTION (2026-08-14) — supersedes every earlier bake claim
**There is NO 6× bake regression. Greedy ≈ 25 min; VITERBI ≈ hours.**
Every "~25 min bake" figure came from a PRE-#9 binary (Greedy default). Every
genuine Viterbi bake has cost hours: session-3's single-thread Viterbi ran
21:49→00:32 (~2h40m ≈ 3.6 min/layer); session-5b measures ~8.5 min/layer on a
healthy box (523W, single process, CPU idle, max clocks) ⇒ ~6h for 44 layers.
**My session-4 "restored the fast bake via the s2/s3 binary" was silently a
GREEDY bake** (pre-#9) — i.e. the quality downgrade Jish explicitly banned
(D4). No quality claim was published from it (coherence only; the 87% GSM8K
came from session-3's real Viterbi bake), but the note was wrong and is fixed.
CONSEQUENCES: (1) box-to-box "slowness" investigations were chasing a
non-existent regression — the model repo is unchanged since 2026-06-22 (sha
60d8d707), same weights all along; (2) **Viterbi bake cost is a REAL PRODUCT
ISSUE — customers bake too**: needs kernel optimization, and until then
bake-once-and-publish is mandatory; (3) always log which QtipMode a bake used.

## 🆕 BEAM BAKE ON HARDWARE — first real measurement (s6, 2026-08-14, H200 NY)
## 🔴 **EVERY NUMBER IN THIS SECTION IS PRE-#40.** Established 2026-08-15 from
## timestamps: PR #40 merged **21:41:26Z**; this bake ran **~20:25Z → ETA
## 21:39Z**, i.e. it started and substantially ran BEFORE the kernel stack
## landed. The instrumentation that split it (PR #39) merged 22:12:46Z, so the
## decomposition was read off the live PRE-#40 run. It has been reading as
## "current H200 performance" all week and is not.
## ⇒ Post-#40 H200 ≈ **170 s kernel / ~186 s per layer — [PROJECTED, NOT
## MEASURED]**: 225.2 s [measured, H200] ÷ 1.33× [measured on an **A100**].
## **That division crosses cards and is UNVERIFIED.** We have direct evidence
## speedups do not transfer cleanly — wave23-AT measured the same GEMV at
## **15.4% of peak on an A30 vs 9.4-9.7% on an H200**. Nobody has ever run a
## post-#40 bake on an H200.
**GPU beam W=256: 241 ± 1 s/layer [PRE-#40].** MEASURED TWICE, marginal deltas between
consecutive layer markers, not a running average:
run A 240 s, 242 s · run B 241 s, 242 s. Rock solid.
vs **exhaustive Viterbi 510 s/layer** (8.5 min, s5b) ⇒ **2.1× faster** — close
to the 2.6× measured on CPU. Full 43-layer bake ≈ **2.9 h ≈ $14**, vs ~6.2 h.
⚠️ **Fable published 135 s/layer and "3.8×" earlier — BOTH WRONG.** They came
from a pace script dividing total elapsed by layers done (which includes the
pre-first-layer load) instead of differencing consecutive markers. Corrected
here. **Always difference consecutive markers; never trust a running average
for a rate.**
⚠️ **The UQFF is buffered and written at the END** — the output dir was still
1 MB after 3 layers. A killed bake loses EVERYTHING; there is no partial
resume. Price that into any decision to interrupt.
Header verified on the box:
`mode=viterbi search=viterbi-beam(W=256) objective=mse rotation=hadamard-128`.
**PARITY PROVEN ON SILICON** (the gate, 21/21 qtip CUDA tests):
`cuda_beam_matches_cpu_beam_bit_for_bit` ✔ ·
`cuda_beam_unpruned_matches_cuda_exhaustive` ✔ ·
`cuda_exhaustive_matches_cpu_exhaustive_bit_for_bit` ✔ ·
`cuda_search_plan_never_substitutes_a_width` ✔ (D4b enforced in code).

### ⚠️ THE PROJECTION WAS WRONG — AND WHY (the useful part)
wave13-AF projected **42-85 s/layer (6-12×)** from a traffic argument: beam-256
cuts HBM traffic 528,392 B → 2,056 B per symbol position (257×). **The traffic
arithmetic was right; it was not the binding constraint.** Measured **2.1×**
(241 s/layer beam vs 510 s/layer exhaustive) — BELOW even the CPU-measured 2.6×,
i.e. the GPU-specific bonus did not materialize at all.
⚠️ This line read **"Measured 3.8×"** until 2026-08-15 — 21 lines below the
correction that retracts it. 3.8× came from the same running-average pace script
that produced the bogus 135 s/layer. **Always difference consecutive markers.**
### ⏱️ PER-LAYER DECOMPOSITION — MEASURED, 4 consecutive layers, ±0.2 s
From the instrumentation agent AM added (wave15), read off the live bake:
| component | time | share |
|---|---|---|
| **GPU beam kernel** (`Quantized fused experts`) | **225.2 s** | **93.4%** |
| host INT4→BF16 unpack, 24 threads | 2.5 s | 1.0% |
| other host (rotation, scales, serialize) | ~13.5 s | 5.6% |
⇒ **HOST FLOOR ≈ 16 s/layer.** This RESOLVES the 10× disagreement wave17-AN
flagged (3.3 s assumed vs a 34 s/layer greedy bake implying a ~30 s host floor)
— **resolved in the optimistic direction**. Consequences:
- The unpack was NEVER the bottleneck (1.0%); AM's thread-pool fix is a correct
  no-op, kept because it is right by construction.
- **30 s/layer is arithmetically OUT OF REACH** for this algorithm: it needs the
  kernel at ~14 s (16×), and wave17-AN's floor for the current instruction
  stream is ~43 s TOTAL even at 100% issue efficiency.
- Realistic target = **42-60 s/layer** (4-5× kernel), bake 30-45 min.

## 🟢 THE KERNEL-STACK A/B IS SETTLED — **1.33× MEASURED ON PRODUCTION SILICON**
(2026-08-15, the same A100 that baked the published artifact, immediately after
it finished. Same GPU, CUDA 12.8, driver, model, data, `ARC_QTIP_BEAM=256`,
`MISTRALRS_ISQ_SINGLETHREAD=1`. ~$0.35 of box time.)
| build | per-layer (differenced consecutive `Detected INT4` markers) | mean |
|---|---|---|
| **pre-#40** (`809643552`, PR #37) | 512, 515, 527, 528, 516, 524 s | **520.3 s** |
| **post-#40** (master) | 390, 392, 390, 392, 393, 394, 392, 392 s | **391.9 s** |
⇒ **520.3 / 391.9 = 1.33×.** Corroborated by power draw: **150 W vs 208 W** on
the same card at 100% util (less work retired per second on the old build).
Kernel-only check: (520.3−16)/(391.9−16) = 1.34×, consistent with the 93.4%
kernel share.
🔴 **THIS OVERTURNS THE "≤1.21× UPPER BOUND" FRAMING — IN THE GOOD DIRECTION,
AND THE REASONING BEHIND IT WAS BACKWARDS.** The claim was that nvcc 11.5's
more register-starved baseline *flattered* the optimisation, making 1.21× a
ceiling. Production measures **higher**, so the old compiler UNDER-stated the
gain. **Lesson: we caveated a real win into near-nothing on an argument nobody
had tested — over-caution is also a way to be wrong,** and D9 requires
correcting it in that direction too.
⚠️ **Scope it honestly: this is the #37→master delta**, not #40 in isolation
(#38/#39/#40/#41 all landed between). #39's unpack pool measured a speed no-op
and #41 is a memory fix, so **#40 is the dominant term** — but the isolated
figure for #40 alone has still never been measured.

**WHAT IS PROVEN vs WHAT IS NOT — do not overstate this (Fable did, twice):**
PROVEN: **the parallel expert unpack is NOT running in parallel.** Of **52
threads** in the bake process, exactly ONE (the main thread) is at 99.9%; every
rayon worker is idle. loadavg 1.14 on 24 cores. utime:stime = 165 s : 19 s.
NOT PROVEN: that this is the *pacing* item. `nvidia-smi dmon` shows the GPU
**continuously occupied** (`sm=100%`, only 3 dips in 20 samples) — but at
`mem=1%` and **261 W of 700**, i.e. resident-but-inefficient. A main thread at
99.9% user time looks identical whether it is unpacking or spinning in a CUDA
sync. TWO live explanations, not one:
 (1) serialized unpack is the bottleneck ⇒ the thread fix wins big;
 (2) the beam kernel is under-parallelized (occupying SMs without using them)
     ⇒ the thread fix barely moves it and the launch geometry is the real work.
Settle it EMPIRICALLY with the fix, not by argument.
`MISTRALRS_ISQ_SINGLETHREAD=1` (PR #25's fix for the 24-threads-thrash-one-GPU
trap, which cost 4-9 min/layer) was calibrated for EXHAUSTIVE search, where GPU
work per layer was ~8.5 min and host threads only contended. Beam made the GPU
part ~64× cheaper, so the *other* per-layer work now dominates: reading and
dequantizing **6.4 B INT4 params/layer → BF16** on one core.
**Amdahl, exactly where we put it.** The safeguard that was correct this morning
is the limiter this afternoon.
⚠️ **THE "PARALLELIZE THE UNPACK" LEVER IS DEAD — RETRACTED.** The instrumented
bake measured unpack at **2.5 s of 241 s (1.0%)**, already running on all 24
cores via rayon's global pool. The "3-5× headroom" written here earlier was
wrong; the ISQ pool never constrained it (`pool.spawn`, never `install`).
PR #39's dedicated unpack pool is correct-by-construction but a **no-op for
speed**. 93.4% is the GPU kernel.
⇒ **THE REAL LEVER IS THE CODEBOOK, not the search and not the unpack.**
Replacing the 512 KB Gaussian LUT with computed arithmetic ≈ **1.68×**
(⇒ ~126 s/layer, bake ~1.5 h), quality-NEUTRAL (`sum2` variant: −0.0008 cos
worst family, +0.0002 mean — **5× smaller than the W=128 delta Jish rejected**).
Last unmeasured number: its real speedup, ~10 min of A30 time. The decode side
pays a SECOND time — the LUT gather is the named inference bottleneck
(388 GB/s ≈ 8% of HBM). ⚠️ The **1.81×** figure was measured on the `split`
variant, which is quality-NEGATIVE (−0.0017 cos; masked-fp16 exponent pinning
puts min|v| at 0.142, a hole where Gaussian weight mass sits). Do not cite it.

⚠️ **"BEAM AT NO QUALITY COST" — RETRACTED.** That was a single-fixture
artifact. Measured across 9 fixture cells (wave19-AP part 1): **exhaustive is
BETTER — wins 8/9, +0.0013..+0.0021 cos on fp4_dequant, strictly lower weight
NMSE everywhere.** Beam is slightly worse AND faster; we ship beam knowingly.
Exhaustive was rejected on SPEED (4% slower byte-identical, 13% slower once both
have the computed codebook), not on quality.

## Fit / density  [SETTLED — and now PUBLISHED, 2026-08-15]
- 🟢 **PUBLISHED ARTIFACT: `aeonmind/DeepSeek-V4-Flash-UQFF-qtip2`** (private).
  **15 files, 74.18 GB of weights, 8 shards + residual.** Verified independently
  from the HF API (`missing vs local: NONE`), NOT from the uploader's own claim.
  284B total / 13B active at **2.09 bits/param**.
  ⚠️ **"≈1.9 bits/param" was WRONG and Fable repeated it several times** — it
  divided a 68 GB *estimate* by the parameter count instead of the real
  74.19 GB. **2.09 is the measured figure.** `residual.safetensors` is
  1,293,806,700 B = **1.7%** of the artifact (wave28-BB). Baked on a **$1.49/hr A100**
  at **370–376 s/layer**, 43 layers, completed 04:44:51Z.
  Header: `beam W=256 / hadamard-128 / mse`.
- **Load from the published UQFF: 12.94 s for 517 tensors** (measured on the
  A100, 2026-08-15). Earlier "~11 s" figures were the same order.
- ⚠️ **It occupies 75.7 of the A100's 80 GB just to load** ⇒ ~4 GB left for KV.
  **This is why measurement MUST happen on a 141 GB H200**, not to save $5.
- ⚠️ Earlier "**68 GB / 7 shards**" was an estimate; the real artifact is
  **74.18 GB / 8 shards + residual**. Update KV-budget arithmetic accordingly
  (H200: 141 − 74.2 − ~8 reserve ⇒ ~59 GB usable, NOT ~65).
- 🟢 **The V4 indexer shape mismatch was a LOADER bug — the artifact is CORRECT.
  NO RE-BAKE NEEDED.** (wave28-BB, PR #55.) `[256,4096]` =
  `[coff*index_head_dim, hidden_size]`, exactly what the reference publishes
  (`sglang .../dsv4/compressor.py:305,318-325`; halves concatenated at
  `.../models/deepseek_v4.py:1633-1652`). Arc asked for `[256,512]` because it
  fed the inner compressor grouped **K** instead of hidden states. Arc had
  ALREADY fixed the identical bug on the attention path (RUN-161/162); the
  indexer kept a stale private copy — **a fifth "N copies of one bug" case**.
  Fixed by deleting the copy (indexer now delegates to `V4Compressor`), with a
  round-trip shape test proven to fail when reverted.
  ⚠️ STILL MISSING before the sparse path can go live: Hadamard
  `rotate_activation` + compress-θ RoPE on compressed indexer K.
- 🆕 **FILE LISTING, read from the HF API 2026-08-15:** 15 files,
  **74,190,197,268 B (74.19 GB)**; 8 shards = 72.85 GB; **`residual.safetensors`
  = 1,293,806,700 B, only 1.7%** ⇒ the repo is an OVERLAY, not standalone —
  `-m` MUST point at the source checkpoint. It also ships `config.json` +
  tokenizer, which makes it *look* standalone: that is the trap that cost two
  failed load attempts.
- ⚠️ **BAKE TIME — this line was WRONG until 2026-08-15.** It read "Bake (GPU
  Viterbi, 44 layers): ~25 min" for days, 106 lines below the correction at the
  top of this file that explicitly retracts every "~25 min bake" figure as a
  **pre-#9 GREEDY binary** (the banned method). Corrected, MEASURED values:
  exhaustive Viterbi **510 s/layer**; GPU beam W=256 **241 s/layer** (H200);
  **369 s/layer** on an A100 (2026-08-15 bake, differenced markers).
  ⇒ a real 43-layer bake is **hours, not minutes**. There is no 25-minute bake.
- CPU-fallback bake (regression, avoid): ~11 min/layer ≈ 8 h.
- **GATE FALSE-ALARM (s5b, Atlanta box): my inline health gate sampled
  power 35s after LAUNCHING the burn but the burn hadn't ramped ⇒ read
  80W/0% util and cried HEALTH_FAIL, while the same box measured **437W @
  100% util @1980MHz** seconds later = HEALTHY. Lesson: sample power in a
  LOOP while confirming util>50%, never a single early sample. PR #25's
  box_health_gate.sh does it properly (auto-picks a burn, samples under
  confirmed load).**
- **BAD-BOX LOTTERY (s5a, NY box 192.241.248.189): bake ran ~3 min/layer even
  with the SESSION-3 BINARY (cca7a9c2e, proven 30s/layer) — bisect proved the
  slowdown is ENVIRONMENTAL, not our merged kernels. Signature: GPU 99% util but
  only ~132W/700W draw = starved on host↔device transfers (Viterbi is
  transfer-heavy), clocks maxed, temps fine. ACTION: kill + delete + re-rent a
  different box. GATE FOR EVERY FUTURE SESSION: after boot, bake 3 min and check
  layers/min AND `nvidia-smi power.draw` — <200W under load ⇒ bad box, re-rent.**
- **BAKE THREADING TRAP (s5, 2026-08-14): "Applying immediate ISQ in parallel on
  24 threads" + GPU 99% = 24 CPU threads thrashing ONE GPU ⇒ ~4-9 min/layer with
  NO fallback warning. Session-3's fast bake printed "1 threads".** Always bake
  with `MISTRALRS_ISQ_SINGLETHREAD=1` until the thread policy is fixed in code
  (PR #20's Viterbi-default flipped get_max_isq_cpu_threads to None=all-cores).
  Check the "Applying immediate ISQ … on N threads" line in the first 30s.

## Quality  [STRONG]
- **GSM8K 87.0%** n=100, 0-shot chat greedy, 2048-cap, seed 161; 2 degenerate,
  9 truncated; ±6.6pp. (s3, on GPU-Viterbi bake.) Ladder: 64.0 (s1, 640-cap,
  33 truncated) → 84.0 (s2) → 87.0 (s3). Reference: published V4-Flash-Base
  **90.8 with 8-shot** (different, easier protocol — always state this).
- Coherence6 **6/6** every session. Facts **22/22** (s2), math **8/8** (s2).
- Perplexity (wiki mini, 70 chunks): **12.50 ±3.46** (s2, post-Viterbi-fix).
  Pre-fix (greedy-bake bug): 58.85. q2k reference on the small mini corpus: 22.50
  (s1; NOT same corpus as the 12.50 run — do not compare directly).
- Long-context: **coherence 5/5 + needle 4/4** post-fix (s2). Ablation proved
  causality: STANDARD_DENSE 2/5 (pre-fix repro), WINDOW_ONLY 5/5+0/4.

## 🟢🟢 2026-08-17 — **THE TRUCK, MEASURED END TO END** (wave51-CB, 1×H200
## $4.85/hr, published `qtip2b` artifact, $16.11, box deleted)
**First time `qtip2b` was ever served** — 517 tensors, no `Applying ISQ`, 2m05s.
PR #76's KV fix is now HARDWARE-PROVEN.
| B | 1 | 8 | 16 | 32 | 64 | 128 | **256** |
|---|---|---|---|---|---|---|---|
| **decode agg tok/s** | 18.27 | 41.43 | 54.75 | 74.52 | 91.46 | 106.36 | **111.69** |
| per-user p50 | 17.99 | 5.67 | 3.97 | 2.87 | 1.82 | 1.09 | 0.53 |
| prefill agg (server) | 171.58 | 216.11 | 235.59 | 266.67 | 293.96 | 308.49 | 314.72 |
| TTFT p50/p95 (s) | .375/.375 | 1.6/2.9 | 3.1/5.3 | 5.5/9.0 | 9.9/16.0 | 18.7/30.1 | 27.8/58.5 |
| **$/Mtok** | 73.74 | 32.52 | 24.61 | 18.08 | 14.73 | 12.67 | **12.06** |
**`effective_B == B` on ALL SEVEN ROWS. 0 errors / 505 requests.**
🔑 **AGGREGATE RISES MONOTONICALLY — the crossover SURVIVED the full serving
path.** vs `qtip2`: **3.01× / 3.69× / 5.87×** at B=64/128/256. **b=1 only 1.12×
— that is the CONTROL** proving the gain is the rung + batching, not the box.
Peak **3.64× and 3.65× cheaper** than qtip2's 30.65 @B=16 / $43.96.
🟢 **FULL GSM8K: 1270/1319 = 96.3% ±1.0pp**, **0 degenerate / 0 truncated / 0
errors**, mean 157.8 tok, `GATE[OK]`. 0-shot; reference's 90.8 is **8-shot**.
⇒ **The faster rung is also the higher-scoring one.** (First pass read 94.0%
±1.3pp with 34 requests lost to the panics below; those 34 were re-run.)
🟢 **FP8 KV: `FP8_VERDICT=TOKEN_IDENTICAL` 5/5**, env verified in
`/proc/<pid>/environ`. Bit-exactness now proven ON HARDWARE, not just by
argument.
🟢 **MTP acceptance, first ever measured:** `accept_rate=0.4194`, 26/62
accepted, `tok_per_step=1.8387` — **B=1 ONLY**.
> 🔴 **THIS IS THE ONLY ACCEPTANCE NUMBER THAT EXISTS. `p ≈ 0.485` is RETRACTED**
> (2026-08-19) — it appears nowhere in the tree; it was fitted, then quoted as
> measured. And **the "2-bit draft head explains it" theory is dead**: the
> `floor_mtp_isq` fix `07766cfa1` is an **ancestor** of the binary that produced
> `0.4194` (`46ea6948d`), so the gap was measured *with* the fix already in.
> Against this base, fixing acceptance is worth **×1.55, not ×5.53**.
> ⚠️ **Per-position telemetry does not exist** — `MtpAcceptance.accepted` is a
> scalar (`mtp_pipeline.rs:977`), so `p₁≈p₂≈0.54` and `p₁≈0.75, p₂≈0.35` are
> **indistinguishable**. Any acceptance work before that lands is a guess bought
> with rental money.

### ✅ THREE PRODUCTION CRASHES — ROOT-CAUSED AND FIXED (wave56-CG, PR #84)
**(A) MTP B=8** — `kv_cache/mod.rs:499` is a `slice_set` in **`clone_in_cache`**,
batching V4's `XsRollingCache::tail` — **the one tensor whose dim-1 is EXACT,
not `CACHE_GROW_SIZE`-quantised.** On ratio-128 HCA layers width =
`T − 128·floor((T−16)/128)` ⇒ 18@T=274 / 22@T=278. **The 4 is not a miscount:**
`mtp_pipeline.rs:2503` commits `1+accepted_i` per sequence while rolling the
shared cache to the batch MINIMUM, and `Sequence::cache_bucket_len` buckets on
**slot 0 only**. **Both choices are individually correct and jointly unsound.**
**(B) Ordinary path `576 <> 64`** — both are `comp.capacity_seq_len`.
`SingleCache::reset` clears `current_seq_len`/`all_data` but **NOT
`capacity_seq_len`** [PROVEN, reproduced on CPU]; `clone_out_cache` stamps the
batch capacity onto every member [PROVEN]. ⚠️ **The third link — that a
waitlisted sequence frozen at 64 meets a runner grown to 576 when
`select_running_bucket` coalesces — is INFERRED from documented scheduler
behaviour, NOT observed in the wave51-CB log.** It is the most plausible
mechanism and explains why the sweep (one uniform bucket, ~132 tokens, below the
260-token growth line) never crashed while GSM8K did — **but keep the qualifier
attached.** Fable stated this as fact once; it is not.
**(C) SendError — symptom AND its own defect.** A client that had already timed
out closed its channel, so **the act of REPORTING one error killed the engine
and orphaned 16 requests.** Three sibling macros already handled it correctly.
🔑 **THE FINDING THAT OUTLIVES ALL THREE:
`ensure_uniform_batch_cache_lens` was a `debug_assert!` — present in CI, ABSENT
FROM THE SERVING BINARY.** Same family as the KV suite that proved a tautology,
the status feed that froze while looking healthy, and three monitors Fable built
this week that alarmed on the wrong signal: **a guard whose passing condition
does not require the thing you care about to be true.** Now runs in release.
**FIXED:** `clone_in_cache` returns `Result` (12 implementors, 5 call sites) so
refusals reach the error handler; `BatchSrc` declares per slot which tensor is
*slack* vs *content*; `step_catching_panics` wraps all three `pipeline.step`
sites; `send_response_or_log` replaces the unwrapped sends. **332 tests pass.**
⚠️ **MTP AT BATCH IS NOT FIXED — it now FAILS SAFELY instead of fatally.**
🔑 **THE TWO FIXES ARE NOT EQUAL COST:** lockstep `xs` advance **preserves** the
batched-MTP throughput property `cache_bucket_len` exists to protect; a wider
bucket key **sacrifices** it. And there is a **red test waiting** —
`ragged_xs_tail_is_refused_by_name_not_panicked` fails the moment either lands
correctly, so the next agent has a failing test rather than a blank page.
**Still open (wave56-CG §7):** `search_request.rs:555`'s unwrapped send survives
but sits inside a `tokio::spawn` — blast radius is one leaked task, deliberately
left. The engine's lazy reboot (`lib.rs:938`) is untouched, and wave51-CB §3.2's
"rebooted, then served nothing at 0% GPU" is a SEPARATE failure this does not
address.

### 🔴 (superseded — original crash entry)
1. **MTP panics at B=8 and never recovers** — `kv_cache/mod.rs:499`, `18 <> 22`,
   reproduced as `19 <> 23`. Deltas are both **4**, run used `--mtp-depth 3` ⇒
   gap looks like `depth+1`. **PR #73's "works at every batch size" does NOT
   hold on the serving path.**
2. **Two panics on the ORDINARY decode path** in the GSM8K run, no MTP —
   `kv_cache/mod.rs:498` `576 <> 64`, and `engine/mod.rs:428` `SendError`.
   **34 requests lost; 2 deaths in ~1,300 requests.** Zero in the sweep ⇒
   load/length dependent.
3. ⇒ **Per-user at B=128 is 1.09 tok/s against a 68 ceiling. Jish's 100
   tok/s-per-user target is NOT met on one card**, and MTP-at-batch — the lever
   for it — is unmeasurable until (1) is fixed.

### ❌ PAGED ATTENTION: RULED OUT BY JISH after a hardware probe (wave53-CD)
Enabling it produced **ZERO tokens** — engine thread died on request 1.
**Root cause is NEITHER objection on record**: `context_lens` is **one absolute
slot index per token** flattened to `[batch*max_context_len]`
(`inputs_processor.rs:300-302`, `:399-408`), while
`build_cu_seqlens_kv_from_context_lens` (`paged_attention/mod.rs:59-78`) reads
`dim(0)` as batch and **cumsums it as lengths**. RUN-167 wired that arm and it
**had never once executed**. Even fixed it is a net loss at b=1.
⚠️ **RECORD CORRECTION — Fable said the flag was "turned off in a drive-by
profiler commit". WRONG.** `ba026e9d1` is the fork's **ROOT commit** (no
parents, 1,430 files). **Nobody ever turned it off; it was never on.**
🔴 Also found: `try_dedicated_decode` (`pipeline/mod.rs:659-664`) needs paged
metadata ⇒ **the DedicatedDecodePath has never run on V4**, and `DecodeConfig`
describes a **dense Llama** (fused QKV, one MLP, no sliding window/sinks) with
**no architecture check anywhere**.

## 🟢 2026-08-16 CROSSOVER MEASURED — **BAKE `qtip2b`. THE DECISION IS MADE.**
(wave38, H200, `qtip_grouped_curve` repaired by PR #64, E=256 top-6, real config)
| B | E(B) | **gemv tok/s** | **grouped tok/s** | grouped speedup |
|---|---|---|---|---|
| 1 | 6.0 | 203 | 63 | 0.31× |
| 32 | 135.3 | 312 | 234 | 0.75× |
| 52 | 180.1 | 314 | 290 | 0.92× |
| **64** | 199.9 | 315 | **322** | **1.02× ← CROSSOVER** |
| 128 | 244.6 | 317 | **527** | **1.66×** |
🔑 **THE GEMV PATH IS FLAT AND DOES NOT SCALE — 315 at B=64, 317 at B=128.**
**The grouped kernel is the ONLY path that keeps climbing.** That is the whole
fleet argument in one table.
✅ **wave29-BD's BLIND PREDICTION HELD**: crossover **B=64**, inside its stated
**52 ± 15**; B=64 predicted 1.10× / measured 1.02×; B=128 predicted 1.80× /
measured 1.66× — both within 25%. **It also pre-committed to refutation ("dead
if grouped never overtakes below B=68") and did not need it.** First prediction
in this program to survive contact with hardware twice.
⚠️ Per-unit handicap is real and stable: grouped 8.2 µs/unit vs gemv 4.4 —
**1.88×** — which is exactly why the crossover sits at B=64 and not lower.
⚠️ **LABELED EXTRAPOLATION** by the harness itself: MoE-GEMM floor only, 40 MoE
layers × 3 calls, **no attention, KV, sampling or launch overhead.** Treat the
tok/s columns as the MoE term, not end-to-end serving.
⇒ ~~**ACTION: the next bake is `--isq qtip2b` + computed codebook + W=256.**~~
🔴 **RETRACTED 2026-08-16 (wave41-BS) — this recipe is INCOHERENT. The crossover
table above is MEASURED and STANDS; only this action line is wrong.**
`qtip2b` (`IsqType::Qtip2b`, `qtip/bitshift.rs`, **K=2/V=1**) has **neither a
selectable codebook nor a beam**:
- `ARC_QTIP_CODEBOOK` + `ARC_QTIP_BEAM` are read ONLY by `QtipBakeConfig::get()`
  (`qtip/mod.rs:932`), whose only callers are `QtipLayer::quantize_*`
  (`mod.rs:1613, 2092`) — the **`qtip2`** rung. `bitshift.rs` never calls it;
  it hardcodes `QTIP2B_MCG_MULT = 0xCAF6_A435` at `bitshift.rs:461,617,634,693`.
- `qtip2b` **has no beam**: `viterbi_quantize_row_2b` is exhaustive DP and always
  stamps `QtipSearchDetail::EXHAUSTIVE_MSE` (`bitshift.rs:344-349, 568,637,760`).
- PR #46's `sum2` is a **V=2** (pair) construction on the **K=4/V=2** `qtip2`
  rung. `qtip2b` is V=1 — a geometry mismatch no flag can bridge. wave24-AU said
  so: *"`qtip_grouped_gemm.cu` is the qtip2b rung and never read this table."*
✅ `qtip2b` is **already** computed-codebook by construction (no LUT tensor
exists), so there is no Gaussian variant of it to bake by accident.
⇒ **CORRECTED ACTION: `--isq qtip2b`, no codebook/beam env vars** (they are
no-ops). Activating PR #46 is a **separate** bake:
`ARC_QTIP_CODEBOOK=mcg ARC_QTIP_BEAM=256 --isq qtip2`.

## 🔴 2026-08-16 `qtip2b` s/layer MEASURED — **≥984 s/layer ⇒ ≈11.75 h. UN-BAKEABLE TODAY.**
(wave41-BS, 1×H200 Helsinki $4.85/hr, **$2.47, box DELETED, ps empty**)
The gap flagged above was measured and it is **an order of magnitude worse than
the plan assumed**:
- **Layer 0 ran ≥984 s (16.4 min) and had NOT finished** when killed. **1 of 43
  markers.** ⇒ **≥42,312 s ≈ 11.75 h ≈ $57** for 43 layers. Aborted at the 2 h gate.
- Box was HEALTHY throughout — 99–100% util, **240–274 W**, SM **1980 MHz** (full
  boost), 40 °C, `0` fallback/panic/OOM/CUDA-error, **`cpu_reroute=0`** (so NOT
  wave6-Q's silent CPU reroute, which looks similar and costs ~20×).
- 🔑 **CAUSE: `qtip2b` HAS NO BEAM.** `viterbi_quantize_row_2b` is the exhaustive
  DP over all 2^16 states (`bitshift.rs:344-349`: *"has no beam"*). **The
  83 s/layer at FACTS:582 is a W=256 BEAM number on the `qtip2` rung** — beam
  prunes 65,536 states → 256. Compare the sibling rung's own lever: exhaustive
  **510** vs beam **241** s/layer.
- ⇒ **This is why no `qtip2b` s/layer was ever on record: no qtip2b bake at V4
  scale has ever COMPLETED.**
- ⇒ **wave29-BD's crossover STANDS — `qtip2b` is still the right SERVING rung —
  but shipping it needs a BEAM KERNEL for `qtip2b` first.** That is a port of the
  working `qtip_beam.cu`, not new research; `search_detail` was already written
  to survive it (*"the moment qtip2b grows a beam kernel"*).
- ⚠️ **LIMIT: this is a lower bound on ONE layer, not a differenced interval.**
  Only 1 marker appeared, so nothing was differenced and layer-0 init cannot be
  separated from steady state. Established: it is **not ~83 s** and the total is
  far past 2 h. `ARC_QTIP_EXPERT_BATCH=8` (default 16) was set; same total rows,
  finer launches — small effect expected, **not measured**.
- 🟢 **SUPERSEDED 2026-08-16 by wave46-BX-VAL (below): `qtip2b` GREW ITS BEAM and
  is now the CHEAPEST rung to bake. "Un-bakeable" is RETRACTED.**

## 🟢🟢 2026-08-16 `qtip2b` BEAM **MEASURED ON HARDWARE** — **un-bakeable → 2.55 h / $3.80**
(wave46-BX-VAL, PR #74 merged `f76b6af0a`. 1×**A100-80G PCIe, sm_80**, Montreal
$1.49/hr, 17.4 min, **$0.33**, box DELETED, `ps` empty. No A30 existed in fleet.)

**All three GPU parity tests PASS, non-vacuity PROVEN** (`HAVE_QTIP_KERNELS=true`,
`Device::new_cuda(0)` ok, `beam_2b_max_width()=256`, `qtip2b_beam-*.o` present —
these tests silently `return Ok(())` otherwise, so this check is mandatory):
`cuda_beam_2b_unpruned_matches_exhaustive` (**an unpruned W=256 beam IS the
exhaustive DP, byte for byte, 4 timesteps deep**), `cuda_beam_2b_matches_cpu_beam_
bit_for_bit`, `cuda_exhaustive_2b_matches_cpu_exhaustive_bit_for_bit` (**0
mismatched bytes ⇒ the `--use_fast_math` fix holds; a GPU bake and a CPU bake of
the same weights are now the SAME checkpoint**). D4 greedy ban intact on hardware.

**s/layer, V4 expert shapes (k=7168 gate/up, k=2048 down), 6.4545e9 elem/layer,
2:1 weighted, kernel-only, differenced/independent runs — never a running avg:**

| config | **A100 s/layer** | 43-layer |
|---|---|---|
| **`qtip2b` beam W=256** | **213.2** | **2.55 h ≈ $3.80** |
| `qtip2b` exhaustive | 8,257 | 98.6 h ≈ $147 |
| `qtip2` beam W=256 (CONTROL) | 372.0 | 4.44 h |
| `qtip2` exhaustive (CONTROL) | 1,544 / 1,621 | ~19 h |

- 🎯 **CONTROL VALIDATES THE HARNESS: `qtip2` beam measured 372.0 s/layer vs the
  published V4 bake's 370–376 s/layer on this exact $1.49/hr A100 class — 0.3%.**
- 🔴 **THE BEAM AND EXHAUSTIVE KERNELS HAVE DIFFERENT A100:H200 RATIOS. NEVER
  QUOTE A SPEEDUP ACROSS GPUs.** beam 372/241 = **1.54×**; exhaustive
  1,544/510 = **3.03×**. Mechanism: exhaustive streams a 16,384-prefix backtrace
  to HBM every timestep ⇒ **bandwidth-bound** (collects H200's ~2.5× HBM × 1.22× SM);
  beam lives in shared memory, issue/latency-bound (*sm=100%, mem=1%*) ⇒ collects
  SMs×clock only. ⇒ the 38.7× measured here is **A100-only**; on an H200 it is ~20×.
- **H200-EQUIVALENTS (computed, NOT measured — nothing here ran on an H200):**
  `qtip2b` beam ≈ **138 s/layer ≈ 1.65 h**; `qtip2b` exhaustive ≈ 2,400–2,700
  s/layer ≈ **28–33 h** ⇒ **wave41-BS's "≥984 s/layer ⇒ ≥11.75 h" was a LOOSE
  lower bound, off by ~2.5–3×.**
- ⇒ 🔑 **THE RUNG WE SERVE IS NOW THE CHEAPEST RUNG TO BAKE**: `qtip2b` beam
  213 s/layer **beats `qtip2` beam's 372** on the same box (K=2 beam is **1.74×
  cheaper per element** than K=4 — ¼ the candidates/atomics more than pay for
  V=1's 2× timesteps). **PR #74's own 340–480 s/layer projection was PESSIMISTIC
  by 2.5–3.5×; its declined "~1–2 h target" was in fact HIT (1.65 h).**
- ⇒ 💰 **BAKE `qtip2b` ON THE A100, NOT THE H200**: 2.55 h×$1.49 = **$3.80** vs
  1.65 h×$4.85 = **$8.00**. H200 wins wall-clock, loses dollars 2.1× — the beam
  kernel is exactly the one that does NOT cash in H200 bandwidth.
- ⚠️ LIMITS: kernel-only (rotate/LS-refine/pack/host I/O excluded — the 0.3%
  control bounds that at ≲1%); synthetic weights at real shapes (timing is
  data-independent: fixed width, fixed state count); **linearity CHECKED** —
  per-element cost moved 17.7% across 512→4,096 rows and *improved*
  monotonically, so the layer extrapolation is **conservative**.
- 🐛 BACKLOG (pre-existing, ATTRIBUTED by re-running on `master` on the same box):
  `cuda_fused_gather_matches_dequantize_fallback_across_the_old_cap` fails on
  **sm_80** — *"64 tokens must dispatch to the fused gather"*. Plus 4 FP8/cuBLASLt
  failures (sm_80 has no FP8 hardware) and `afq3_gpu_bit_identical_to_cpu`.

## 🟢🟢 2026-08-15 POST-FIX SWEEP (wave34-BL, 1×H200 @$4.85, published W=256
## artifact, $7.60, box DELETED) — **THE COLLAPSE IS FIXED AND GSM8K IS 96.0%**
| B | prefill agg | **decode AGG** | per-user | $/Mtok | eff_B |
|---|---|---|---|---|---|
| 1 | 78.14 | 16.27 | 16.02 | 82.80 | 1 |
| 8 | 79.50 | 26.96 | 4.11 | 49.97 | 8 |
| **16** | 81.76 | **30.65 ← peak** | 2.37 | **43.96** | 16 |
| 32 | 84.30 | 30.59 | 1.25 | 44.04 | 32 |
| 64 | 85.66 | 30.41 | 0.59 | 44.30 | 64 |
| 128 | 114.34 | 28.86 | 0.27 | 46.68 | 128 |
| 256 | 156.78 | 19.02 | 0.08 | 70.83 | 256 |
**0 errors / 505 requests, `effective_B == B` on all seven rows.**
**vs pre-fix: 1.06× / 1.82× / 2.97× / 6.03× / 3.74× at B=1/8/16/32/64.** Peak
aggregate **2.00×** AND moved off b=1 onto a real batch row, at half the $/Mtok.
🔑 **b=1 moved only 1.06× — that is the CONTROL.** The gains are batching, not
the bake/box swap.
🟢 **GSM8K = 96.0%** (96/100, ±3.8pp, **0 degenerate, 0 truncated** — see the
correction below; it was reported as "1 degenerate" and that was OUR DETECTOR
being wrong, not the model), mean 148.5 tok) — n=100, 0-shot chat, t=0, seed 161, 2048-cap, published artifact.
**RETIRES the void 87.0 (different bake + pre-#35 math); it does not "beat" it.**
Truncations 9→0, completion length 528→148. **Meets D6's ≥90 on OUR protocol for
the first time.** Reference 90.8 is **8-shot** — not comparable. Also coherence6
**5/6**, facts **21/22** — both one below anchor, reported as measured.
### ✅ "1 DEGENERATE" WAS A HARNESS FALSE POSITIVE — there were ZERO
(wave39-BQ, PR #68. Raw output was never lost: `arc-tools/quality/results/
gsm8k_s9.json`, gitignored hence invisible.)
**idx=211 is CORRECT** — gold 4, pred 4, `finish_reason: "stop"`, 185 tok,
ends `#### 4`. `looks_degenerate` fired because a **7-word phrase repeated
ONCE** and its bar was `run + period >= 14`. ⇒ **96.0%, 0 degenerate, 0
truncated.** New detector: union of word-cycle / char-cycle / 4-gram
saturation, threshold 0.45 against a **measured** separation (worst clean 0.574
vs best real loop 0.418), **0 false positives on 350 archived completions**, and
it catches two real loops the old one MISSED (periods 13 and 22). 51 assertions,
each D12-paired — including asserting the OLD detector false-positives on
idx=211 and is blind to `"User"×200`.
**It declined to inflate the score**: idx=1017 could be "fixed" by summing
numbers the model never summed — called fabrication, not extraction. **96.0
stays 96.0.** The 4 real misses are genuine model errors.

### 🔴 THE TOKTRIE WARNINGS ARE NOT COSMETIC — TOKENS ARE BEING DELETED
`WARN toktrie_hf_tokenizers: missing char: ｜ for "｜DSML｜"` (and `<｜/table>｜`).
**EOS is SAFE** — it is a plain `Vec<u32>::contains` on `metadata.eos_tok`
(`sequence.rs:982-987`); the trie is never consulted.
**BUT** `toktrie_hf_tokenizers-1.7.0/src/lib.rs:205-212` leaves those ids'
**bytes empty**, `decode_ext` turns empty into **nothing** (`toktree.rs:451-458`),
and `sampling.rs:38-48` detokenizes **every** token through it on the normal
path ⇒ **a broken token id is silently DELETED from the output text**, and a
stop-STRING can never match across the gap (`sequence.rs:1003-1013`).
**The splice signature is all over session-3's outputs**: `...####276000The
total cost...`. Fixed in PR #68 (`llg.rs` restores the dropped bytes after the
existing `added_special` pass).
⚠️ **UNVERIFIED AND POTENTIALLY IMPORTANT: `｜DSML｜` is the delimiter EVERY V4
tool call is built from**, and `parse_tool_calls` raises on exactly the shape an
empty entry produces. **V4 tool calling may be broken today.** Mechanism traced,
NOT measured — settle it with one served request with `tools` set.

### ❌ THREE OF FABLE'S CLAIMS, REFUTED BY THIS RUN
1. **"aggregate = 1/per-seq-step-time" is UNFALSIFIABLE, not a finding.**
   wave27 defines `step_ms = running/agg`, so `ms-per-seq ≡ 1000/agg`
   **algebraically**. Fable presented an identity as a measured signature.
   *The falsifiable SHAPE was real and is now broken*: per-seq cost was flat
   65.1→67.4 then **doubled to 197.2**; it now **falls 61.5→32.6 ms** (1.89×
   amortization), flat within 1% across B=16/32/64. **The MoE-cap discontinuity
   is gone — that half of the diagnosis was correct.**
2. **The CPU-sampler theory is REFUTED on hardware**: 2 radix-fallback lines all
   session, both at startup, **none in any measured row**.
3. **"Attention is the frontier" — WRONG AT BATCH.** Profile captured at both:
   B=1 `moe 35% / mla_attn 36% / mhc_attn_pre 10% / mhc_ffn_pre 10%`;
   **B=64 `moe 49–64% / mla_attn 39% / 4% / 4%`.** **MoE share GROWS with batch
   while everything else collapses.** The stale b=1 cudnn profile
   (`mla_attn 49%, moe 16%`) that made Fable declare attention the frontier is
   superseded. **MoE is the keystone, confirmed end-to-end at batch.**
### 🔑 WHY B=256 REGRESSES — **AN O(B²) HOST COPY, FOUND AND FIXED** (wave36-BN, PR #67)
**`Tensor::i()` returns a VIEW** (candle `tensor.rs:906`: `storage: self.storage
.clone()`) **and `to_device` copies the WHOLE storage** (`tensor.rs:2379` →
`cuda_backend/mod.rs:1788`). So the per-sequence D2H of logits
(`pipeline/mod.rs:924`/`:1161`) dragged **all B×vocab elements, B times**.
⇒ V4 vocab 129,280, BF16 ⇒ **B²·129,280·2 B = 16.94 GB PER STEP at B=256**
(1.06 GB at B=64), with all B host `Vec`s alive at once.
**THE SHAPE IS THE PROOF, not just the code.** `agg = 1/(k/B + a + cB)` ⇒ **a
purely linear step cost can only PLATEAU; only a non-zero `c` can make aggregate
FALL** — which is exactly what 128→256 does. Fit to the measured table:
`c=0.109 ms, a=24.26 ms, k=105.9 ms` (residuals +3%/−5%/+13% at B=8/32/128) ⇒
quadratic = **7.14 s of the 13.46 s step (53%)** at B=256, 40% at B=128, 21% at
B=64. And `c = vocab·2/BW` ⇒ **BW = 2.37 GB/s**, an ordinary pageable
`cudaMemcpyDtoH` rate — the single free parameter lands on a physical number.
**FIXED**: one batched D2H hoisted above the split (`host_copy_batched_result`,
`pipeline/mod.rs:492`) ⇒ **66.2 MB vs 16.94 GB at B=256, a factor-of-B cut.**
**[PROJECTED, ±30%]** B=64 30.41→**34–40**; B=128 28.86→**40–52**; B=256
19.02→**32–46**; **b=1 unchanged**. Confident part is the SHAPE: aggregate
stops declining and resumes rising with B.

### ❌ TWO MORE FABLE ERRORS, CORRECTED BY THE SAME AGENT
1. 🔴 **"The 87× b=1 gap is per-step host overhead" — WRONG. It is GPU-SIDE
   KERNEL EFFICIENCY.** This term contributes **exactly zero at b=1**
   (`logits_on_cpu = len>1` is false there), and wave34-BL's own profile shows
   `forward_total 67.24 ms` against a 61.5 ms step — the step IS the forward.
   ⇒ **The computed codebook (3.86–4.03× on the decode GEMV) attacks the b=1
   gap DIRECTLY**, and so does anything that lifts GEMV off ~15% of peak. This
   is good news mis-stated as bad: b=1 is a kernel problem we already have a
   measured fix for, not a host-loop problem.
2. ⚠️ **The quadratic does NOT explain the bulk of B=64.** The linear term
   **`a = 24.26 ms/sequence` survives untouched and is UNATTRIBUTED** — the B=64
   profile puts forward at 210 ms of a 2,105 ms step (10%), quadratic adds 21
   points, **~69 points unexplained.** PR #67 does not close this. **Next hunt.**
   ⚠️ Blocked on instrumentation: the `STEP_us TOTAL/fwd/sample/other` line
   exists **only on the PagedAttention branch** (`pipeline/mod.rs:1300-1324`),
   so **V4's actual path logs no fwd/host split at all.** Add it before renting.

### 🔴 EIGHT MORE HOST COSTS, ALL "FIXABLE-NOW", NONE NEEDING CUDA GRAPHS
Ranked, with `file:line`, from wave36-BN — **the megakernel stays correctly
deferred; graphs attack launch overhead INSIDE the forward, and every item below
is host work that exists regardless of dispatch:**
② O(B) device allocs + O(B) H2D — `Tensor::new(ctxt, device)` **inside the
decode loop** (`inputs_processor.rs:538`, B-way `cat` at `:815`) = CLAUDE.md
pitfall #5, still live. ③ O(B×layers) — `post_op` is `CacheInstruction::Out`
**unconditionally** (`engine/mod.rs:397-404`) ⇒ `clone_out_cache` rebuilds
B×~86 `KvCache`s **including a full `XsRollingCache` clone per (layer,seq) every
step**. ④ O(B×ctx) — `seq.get_toks().to_vec()` per seq per step
(`sampling.rs:499`). ⑤ O(B) **SERIAL** — the post-sample loop `.await`s
`responder.send()` per sequence **on the engine task holding the pipeline
mutex** while sampling itself is rayon-parallel ⇒ **43 of 44 cores idle.**
⑥ **`get_mut_group!` is a `try_lock` + `yield_now` BUSY-WAIT**
(`utils/mod.rs:227-237`), taken **5× per token per sequence** — **this is the
mechanism that converts contention into "100% of one core with the GPU idle."**
⑦ O(B·L²)/completion — stop-string search rescans all `completion_bytes` per
token (`sequence.rs:1005`); **zero cost in our sweep (no stop strings), a
customer cliff.** ⑧ O(B) `Sequence` moves through a HashMap every step.
⑨ **Batched decode has NEVER used the GPU sampler** — gated on
`!logits.device().is_cpu()` (`sampler.rs:1220`). Real project.

### WHY B=256 REGRESSES (19.02) — HOST-BOUND, NOT GPU-BOUND (superseded above)
Server pegged **100% of ONE core of 44**, GPU at **0–4% / 121 W of 700 W**.
B=64/128 ran the same box at **420–476 W**. ⇒ per-step host overhead, the same
term that dominates the **87× b=1 gap** to the 1,413 tok/s ceiling.
### 🔑 MTP: THREE SESSIONS OF EMPTY FILES EXPLAINED — NOT A LOGGER BUG
`MTP block is not loaded (--mtp-depth 0); skipping them`. **The artifact ships
the tensors; the default serve does not load them.** PR #36 did fix the callers.
⇒ **Cheap unclaimed experiment: serve with `--mtp-depth 1` and re-run.**
⚠️ PR #46's computed codebook is merged but **needs a BAKE to be active at serve
time** — none of its 3.86–4.03× is in any number above.

## 🔴 SUPERSEDED — the PRE-FIX sweep (kept for provenance)
## 🔴🔴 FULL-SERVING THROUGHPUT — **MEASURED AT LAST, AND IT IS BAD**
## (wave26-AX, 2026-08-15, 1×H200 @ $4.85/hr, `--max-seqs 128`, 64 decode
## tokens, temp 0, in-situ qtip2 W=32 bake — NOT the published UQFF, see below)
| B | decode/user | **AGGREGATE** | TTFT p50/p95 | $/Mtok | eff_B |
|---|---|---|---|---|---|
| 1 | 15.11 | **15.35** | 5.78/5.78 | $87.77 | 1 |
| 8 | 2.705 | 14.83 | 17.31/28.17 | $90.84 | 8 |
| 16 | 1.18 | 10.31 | 28.11/47.83 | $130.67 | 16 |
| 32 | 0.21 | 5.07 | 32.82/110.66 | $265.72 | 32 |
| 64 | 0.175 | 8.14 | 37.53/115.03 | $165.51 | 64 |
🔴 **AGGREGATE THROUGHPUT FALLS WITH BATCH SIZE — the exact inverse of D1's
fleet thesis.** Peak aggregate is at **B=1**. B=32 delivers ⅓ of single-user
throughput at 3× the cost.
**THE RESULT IS REAL BUT ALMOST CERTAINLY NOT ARC'S CEILING:**
- It is **~0.2% of the roofline** (74.2 GB ÷ 4.8 TB/s ⇒ ~4,141 tok/s at B=64).
- The grouped-GEMM microbench predicted ~1,006 aggregate tok/s; end-to-end
  reproduces **2 orders of magnitude** below it.
- ❌ **THE PRIME SUSPECT WAS WRONG — `GPU radix top-k … falling back to CPU`
  explains 0% of the gap.** The probe sent `temperature=0.0`
  (`batch_load_probe.py:629`); `sampler.rs:324` maps `<1e-7` to greedy, which
  returns at `:1224-1240`, **above** the radix branch at `:1276`. Unreachable
  for every measured request. (The radix bug is REAL and now fixed — see below —
  it just was not on the measured path.) **Fable named this suspect confidently
  in the user-facing report. Name a suspect, then PROVE the path executes.**

### ✅ THE REAL CAUSE — all CONFIRMED in source (wave27-AY, no GPU needed)
1. 🔑 **The fused MoE gather is HARD-CAPPED AT 8 TOKENS/STEP** on both QTIP
   rungs (`qtip/mod.rs:3189`, `bitshift.rs:1495`). Above the cap the LUT rung
   falls to what its own comment calls *"per-expert dequantize … materializes
   weights to HBM."* AX's engine log crosses that cap **between B=16 (8
   running) and B=32 (13 running) — exactly where aggregate halves.**
   Per-sequence step cost: **65.1 / 67.4 / 97.0 ms below the cap → 197.2 ms
   above it.**
2. **No dedup below the cap either**: one GEMV per (token, expert) pair
   (`bitshift.rs:1311`) ⇒ zero amortization, which is why aggregate is flat
   rather than rising even at B≤16.
3. **The scheduler runs ONE length-bucket at a time.** V4 reports
   `supports_paged_attention=false` (`normal_loaders.rs:3231`) ⇒ the bucketing
   `DefaultScheduler` (`default_scheduler.rs:128-168`) waitlists the rest.
   **Measured `32 running, 32 waiting` at B=64** — a flat 2–2.5× loss, and the
   reason B=64 scored *better* than B=32.
4. The same `supports_paged_attention=false` also kills the CUDA-graph
   autonomous decode path (`normal.rs:1841`).

### 🔴 ROOFLINE CORRECTION — Fable's "0.2% of roofline" WAS WRONG
**74.2 GB/step is only valid ASYMPTOTICALLY** for 256-expert top-8. Distinct
experts activated go **8 → 57 → 163** for 1 / 8 / 32 tokens ⇒ **batching an MoE
reads MORE weights, not the same weights amortized.** A *correct* grouped kernel
gains only **1.57× from 1→32 tokens.**
🔴🔴 **"MoE BATCH ECONOMICS ARE STRUCTURALLY MODEST" — FABLE WROTE THAT HERE AND
IT IS WRONG. RETRACTED 2026-08-15 after Jish pushed back ("who gave you the
permission to accept defeat").** The 1.57× is real ONLY for the 1→32 range. It
was generalized into a structural ceiling from the ONLY region we had sampled.
**Expert coverage SATURATES, and past saturation an MoE amortizes exactly like
a dense model.**
🔴 **CORRECTED 2026-08-15 (wave29-BD): V4-Flash is TOP-6, NOT TOP-8.** Fable's
first table used top-8 and overstated every row. Correct curve —
`E(B) = 256 × (1 − (1 − 6/256)^B)`:
| B | experts woken | weight-amortization vs B=1 = `6B / E(B)` |
|---|---|---|
| 1 | 6 | 1.0× |
| 8 | 43.9 | 1.09× |
| 32 | 135.3 | **1.42×**  ← the only region we measured |
| 64 | 199.1 | 1.93× |
| 128 | 243.4 | **3.15×** |
| 256 | 255.4 | **6.01×** |
🔴 **AND THE GROUPED KERNEL IS NOT FREE.** It carries a **MEASURED 1.76×
per-byte deficit** vs the GEMV it replaces (**403.8 vs 228.9 µs at identical
traffic**), so the amortization above is an upper bound the kernel does not
reach until it has paid that off ⇒ **crossover at B≈52 ± 15; ≈1.1× at B=64,
≈1.8× at B=128.** The rung switch pays at B=128+, NOT at the batch sizes we can
currently schedule.
⇒ **THE ×4–8/NODE TARGET IS REAL AND LIVES AT B=128–256.** We measured B≤64 —
the worst stretch of the curve, before overlap pays off — and I declared the
thesis structurally capped. **Never extrapolate a ceiling from the edge of the
sampled range.**
⚠️ ASSUMPTIONS, state them when quoting: uniform random routing (real routing
has locality/imbalance — could go either way, MEASURE it); this is
**weight-traffic amortization only** — attention and KV reads scale differently
and are not in this model.
### ❌ PAGED ATTENTION FOR V4 — **NOT FEASIBLE. Fable called it "the keystone";
### it is not, and both reasons Fable gave were wrong.** (wave29-BC, PR #57)
1. **Geometry:** `flashinfer_mla_decode.cu:12-13` fixes `HEAD_DIM_CKV=512`/
   `KPE=64` as **template** args. A 448 instantiation gives
   `vec_size_ckv = max(16/2, 448/32) = 14` (`decode.cuh:1107`) and trips
   `static_assert(vec_size % 8 == 0)` (`vec_dtypes.cuh:1362,1566`) — **it does
   not compile.** `concat_and_cache_mla_kernel.cu:53-64` WOULD take 448, so the
   cache write was never the problem.
2. **Algorithm (the real blocker):** the kernel is
   `DefaultAttention<false,false,false,false>` = dense causal, no sliding
   window, no sinks, ONE key set. **Every V4 layer is sliding-window +
   `attn_sink` with CSA/HCA folding a compressed key set into the same
   softmax** (`dsv4_attention.rs:9-46`).
3. ⚠️ **The `head_dim=512` rationale was INAPPLICABLE.** That switch
   (`pagedattention.cuh:714`) is only reached from `PagedAttention::forward`
   (`paged_attention.rs:341`); V4's ONLY `paged_attn.` call site is
   `deepseek4.rs:1424` `cache_write_and_gather` — **storage, no head-size
   dispatch.** Rationale corrected in-code at `normal_loaders.rs:3231`.
4. ⚠️ **Fable claimed the flag was "the single prerequisite for THREE dead
   capabilities". FALSE.** Flipping it would NOT unlock graph decode —
   `normal.rs:1771-1779`: the `prime_for_step` bridge does not exist, so the
   runner never finishes a batch. Necessary, not sufficient. The fused sampler
   rides the same runner; ragged batching is blocked separately by the varlen
   pack.
⇒ **The flag stays `false`, correctly.** BA's §1e 4-step port plan is
**STRUCK — NOT IMPLEMENTABLE.**

🔑 **WHAT ACTUALLY BLOCKS B=256 IS MEMORY, NOT THE ARCHITECTURE.** 424,018
B/token of context ⇒ at 2048 ctx one sequence costs ~868 MB ⇒ ~59 GB usable
caps us near **B≈68**. **336 KB of that 424 KB — 79% — is the `xs` compressor
cache** (BACKLOG). **Cut `xs` ~4× and B=256 fits, which is where the 8× lives.**
⇒ That makes the `xs` cache THE KEYSTONE of the fleet claim, not a nice-to-have.
⇒ Corrected **b=1 bound ≈ 527 tok/s**; we measure 15.11 ⇒ **34× off at
single-stream — the largest single term, previously MASKED by the inflated
roofline.** Fix b=1 before chasing batch.

### FIXED (PR #52): `cuda_tensor_ptr` called `as_cuda_slice::<u8>()`
unconditionally (`arc-cuda-graph/src/flashmlasparse.rs:188`,
`sampling_cuda.rs:218`); Candle type-checks that against storage dtype
(`candle-core/src/cuda_backend/mod.rs:1295-1304`) ⇒ **every non-U8 tensor
failed. Unconditional, not an edge case.** Un-deads THREE paths: radix sampler,
**V4 Lightning Indexer kernel**, fused GPU sampler. Test
`radix_topk_rows_f32_accepts_f32_scores` fails without it with the exact H200
error. **Sixth "wired but never invoked" instance.**

### NEXT GPU RUN IS ONE EXPERIMENT, NOT AN EXPLORATION
Re-run the identical sweep with **`ARC_QTIP_ONDEVICE_MOE_MAX_TOKENS=64` +
`ARC_WARN_DEQUANT_MATERIALIZE=1`** (env only, no rebuild). If the 8→13 cliff
vanishes, cause (1) is confirmed. Log which QTIP rung the artifact uses — it
decides "raise the cap" vs "finish the grouped kernel". Run the new test under
`--features cuda`.
- **PREFILL SCALES FINE: 4.8×** from B=1→64 (11.09→52.71 server-instrumented).
  Compute leg amortizes; decode leg does not. Use this as the discriminator.
**VALIDITY — do not overstate or dismiss:** `effective_B == B` on ALL five rows
(probe asserts genuine concurrency, mutation-tested on-box), **0 errors / 121
requests**, not KV-bound (B=64 used ~8,448 of ~118,000 tokens). So it is NOT
under-batching and NOT memory pressure. **It is the honest end-to-end number
today and must be quoted as such** — with the CPU-sampling caveat attached.
⚠️ Numbers came from an **in-memory ISQ bake, NOT the published UQFF** (the
agent's first two load attempts failed, see below) — same quantizer, so speed is
representative; it does NOT validate the artifact.
⚠️ **GSM8K NOT re-measured — 87.0% stays PROVISIONAL.** The served model was a
**W=32** bake; a quality number from it would be invalid against our
0-shot/n=100/seed-161/2048-cap W=256 protocol. Correctly declined.

### Corollaries from the same session (2026-08-15, H200)
- 🟢 **GENERATION PROVEN, 3/3** — the first tokens this artifact has produced:
  "Paris"; `17*23 = 340+51 = 391` with correct working; a coherent
  Rayleigh-scattering sentence. The `device mismatch in matmul` smoke failure
  did NOT reproduce (dummy run 3.49 s in-situ, **0.22 s from UQFF**).
- 🟢 **THE PUBLISHED ARTIFACT IS NOT BROKEN — Fable's "unloadable" was WRONG.**
  It is an **OVERLAY on the source checkpoint, not standalone**: `-m` must point
  at the source model. **The published model card documents the FAILING form.**
  Packaging/doc defect, not corruption. Fix the card before any public release.
- 🔑 **BEAM WIDTH BARELY AFFECTS BAKE TIME: W=256 = 82.7 s/layer vs W=32 =
  83.6 s/layer (~1%) on H200.** ⇒ **ALWAYS BAKE W=256** — better quality at no
  cost. (The agent restarted at W=32 believing W drove bake time, extrapolating
  from a first layer that carries one-time init. It cost quality headroom and
  bought nothing. Same running-average trap as the 135 s/layer error.)
- ⚠️ **POST-#40 H200 BAKE ≈ 83 s/layer** vs the pre-#40 241 s/layer on record.
  That implies ~2.9×, against the **1.33× measured on A100**. **DO NOT quote the
  2.9× — the configs differ** (44 unpack threads here vs
  `MISTRALRS_ISQ_SINGLETHREAD=1` there, different `ARC_QTIP_EXPERT_BATCH`).
  Unexplained discrepancy; needs a matched-config run.
- ⚠️ Bake **OOM'd at layer 25 with 72 GiB FREE** — allocator fragmentation, not
  exhaustion. Worked around with `ARC_QTIP_EXPERT_BATCH=8`.
- ✅ Indexer shape mismatch **RESOLVED — loader side, no re-bake** (wave28-BB,
  PR #55). Was: expected [256,512], got [256,4096] on every CSA layer. See the
  Fit/density section above for the full diagnosis.

## Speed  [WEAKEST AXIS — batch numbers are the real metric]
- b=1 decode (DIAGNOSTIC ONLY): 5.4 (s2 baseline) → 13.99 (s3, four kernel
  fixes) → **14.58** (s4, no-cudnn build). cudnn build = 5.45 (−62%!).
- Prefill ~57-61 tok/s (all sessions).
- 🔴 **RETRACTED 2026-08-15 (wave35-BM): the "~1,006 aggregate tok/s"
  grouped-GEMM microbench is VOID and must not be quoted again.** Two
  independent defects, both confirmed when the harness was repaired:
  1. **Wrong fixture — E=64 experts, when V4-Flash has 256 and routes top-6.**
     The repaired harness now **rejects that fixture outright**.
  2. **The grouped-vs-GEMV mode switch was a SILENT NO-OP** (a `LazyLock` cap
     trap), so **every prior "gemv" row re-measured the GROUPED kernel.** Past
     grouped-vs-GEMV comparisons compared grouped to grouped.
  ⇒ Also retracts the framing Fable used repeatedly — *"the microbench predicted
  ~1,006 and end-to-end reproduces two orders of magnitude below"*. **There was
  no valid prediction to miss.** The real end-to-end numbers stand on their own;
  the gap to quote is against the PHYSICS ceiling in `CEILINGS.json`, not
  against this.
  Old text, for provenance: *"step time FLAT ~63.5 ms from B=16→64 ⇒ ~1,006
  aggregate tok/s ⇒ ~15.7 tok/s per user at B=64 ⇒ ≈$1.36/Mtok at $4.92/hr."*
- GEMV bandwidth (kernel bench): pre-tune 153-192 GB/s (3-4% of 4.8TB/s peak);
  gen-1 sweep winner 36μs / **450-467 GB/s (9.4-9.7%)** = 2.3× (s4).
- 🆕 **GEN-2 GEMV IS DEAD — 98/98 VARIANTS MEASURED, IT LOSES** (wave23-AT,
  2026-08-15, A30 @ $0.39/hr, $0.42 total, parity run BEFORE timing).
  Winner on BOTH V4 decode shapes = **v6 (`w8_r4_i1_v2`) — a GEN-1 variant**.
  **144/140 GB/s = 15.4%/15.0% of A30's 933 GB/s peak** (quote the % — the
  GB/s is A30-specific and does NOT transfer). Legacy baseline 60/59 GB/s.
  **Gen-2 loses −8.3% (gate) / −10.7% (down); gate top-10 is entirely gen-1.**
  Every structural axis measured NEGATIVE: split-K 132/115/110 (KS=1/2/4),
  staged width 132/119/73 (W=2/4/8), warp specialization 132→82 (**−38%**).
  Only STAGES=3 helped, and it is already the deepest compiled.
  ⇒ **THE cp.async PREMISE IS DISPROVEN, NOT UNREALIZED.** At ~15% of peak this
  GEMV is **NOT latency-bound**, so hiding memory latency cannot pay. The
  ceiling is **per-symbol trellis decode serialization** — decode FEWER SYMBOLS
  (computed codebook / V=4), do not hide latency. Next attempt belongs there.
  ⚠️ Note A30 gen-1 reaches a HIGHER fraction of peak (15.4%) than H200 gen-1
  did (9.4-9.7%) ⇒ the H200 figure is **not a format limit**; the cheap box was
  not flattering gen-1.
  **WHY THIS WENT UNMEASURED FOR WEEKS:** the sweep was **unrunnable on master**
  — its fixture called `QtipMode::Greedy`, banned by D4 outside `cfg(test)`.
  **The quality ban silently bricked the speed benchmark.** `qtip_gemv_bw.rs`
  and `qtip_grouped_curve.rs` carry the same dead call and are equally
  unrunnable (BACKLOG). Dispatch verified at the ID level, not by timing:
  `serve_dispatch_picks_tuned_winners` returns 21 without the table and fails
  `left: Some(6)` with it — proving the production path consults it.
  Shipped table deliberately NOT retuned from A30 data (it is H200-tuned, where
  v6/v21 tie within 0.23%; retuning on the wrong silicon would regress).
- Component profile (s4, ARC_TIME_DECODE, cudnn build — RE-MEASURE on no-cudnn):
  forward_total ~231 ms: mla_attn 49% (q_proj 22ms, kv_proj_rope 7.7ms,
  invrope_oproj 16.6ms, rest = SDPA), mhc_attn_pre 16%, mhc_ffn_pre 16%,
  moe 16%, mixes 3%.

## 🆕 MEASUREMENT HARNESS — verified before spending H200 money (wave25-AV,
## 2026-08-15, A6000 + Qwen2.5-0.5B, **$0.17**, box deleted). PR #48.
🔴 **THE HEADLINE NUMBER WAS ABOUT TO BE FABRICATED.** mistral.rs `--max-seqs`
**defaults to 32** ⇒ a B=64/128 sweep against a default server is **silently a
B=32 sweep**; the server queues the remainder and the probe reports a plausible
number. **Proof: a run capped to B=2-of-8 reported 190.01 tok/s vs a genuine
B=16's 198.34 — 4% apart, INDISTINGUISHABLE without `effective_B`.** The probe
(PR #23) also had **zero prefill measurement**, the thing Jish explicitly asked
for. Both fixed; `effective_B` is now asserted and the run exits non-zero on
under-batching.
**Concurrency signal that actually works:** overlap of per-request **DECODE**
windows. ⚠️ Submit-window overlap is a **check that can never fail** — the
launch barrier makes submissions overlap even under full serialization. 4
mutation tests, 2 on hardware (`--max-seqs 1` ⇒ exit 1 `effective_B=2`;
`--max-seqs 4` at B=16 ⇒ exit 1 `effective_B=6`).
**KV BUDGET — V4-Flash is MQA, NOT MLA:** 88,064 B/token KV (43 layers) **PLUS
335,954 B/token of compressor `xs` history (41 layers)** = **424,018 B/token**.
🔑 **The `xs` history is 3.8× the KV cache and is what ACTUALLY caps batch
size** — halving it ≈ **4× the feasible batch at long context** (D1 multiplier;
see BACKLOG). Against ~65 GB usable (141 − 68 − ~8): **B=128 fits at 512-token
context (27.8 GB); at 2048 tokens the max is ~74 ⇒ the sweep tops out at B=64.**
**NO IN-CLASS SINGLE-H200 BASELINE FOUND — every published config we checked
needs >1 GPU.** Native ckpt ≈160 GB; smallest published config **4×H200**; the
one W4A16 quant is 143 GB and its own card states *"TP=1 OOMs on a single 141 GB
H200"*; NVFP4 is Blackwell-only. **Arc's ~68 GB (≈1.9 bits/param) is what makes
1×H200 possible at all** — state as footprint (1 GPU vs published 4s) + $/Mtok
per node, NOT a head-to-head we cannot run.
⚠️ **Scope this claim to what was surveyed — it is "none found", NOT "none
exists on any engine."** WHAT WOULD REFUTE IT: any published single-GPU
V4-Flash config, or any engine loading it in ≤141 GB. **We want to know if one
exists** — a claim we would rather not have tested is a claim D9 forbids.
**Third-party-free ceiling (roofline):** 68 GB ÷ 4.8 TB/s = **14.2 ms/step ⇒
~4,500 tok/s at B=64**.
🚫 **RETRACTED 2026-08-17 — "the grouped-GEMM sits at ~22% of roofline" is VOID.**
It derived from the 63.5 ms microbench **which this same file already retracted at
§800-812** (bad fixture + a silent mode-switch no-op). A retracted measurement was
still being quoted three sections later, and main repeated it to an agent and to Jish
as a live baseline. **Do not cite 22%.**
⚠️ **The roofline is also the WRONG CEILING for this kernel.** The gen-2 sweep
measured **98/98 variants** and concluded the binding limit is **per-symbol trellis
decode**, not memory bandwidth — trellis decoding is a sequential state machine, so
the kernel is ALU/dependency-bound, not HBM-bound. Optimise decode work per weight,
not bytes moved. Any future grouped-GEMM target must be stated against decode cost.
Also found: PagedAttention **crashes on sm_86**
(`pagedattention_v1_bf16.cu:30`) — irrelevant for V4-Flash (PA off there).

## Correctness proofs on hardware
- **gen-2 GEMV kernels (54 variants: cp.async staging, smem-fed ILP, split-K):
  28/28 parity tests PASS on H200 (s5, 2026-08-14) + 2/2 GPU-quantize fallback
  guards.** All bit-exact vs dequant reference. Measurement pending same session.
- qtip2b (bitshift trellis) CUDA parity **20/20** (s3). Grouped-GEMM parity
  **5/5** first HW run (s3).
- Sinkhorn fused kernel: bit-identical ppl + token-identical 6/6 (s2) → now
  DEFAULT ON. ✅ **STANDS — checked 2026-08-17, and NOT retracted.** The
  harness that produced it (`arc-tools/quality/run_ppl.sh --sinkhorn-ab`)
  toggles `ARC_FUSED_SINKHORN`, and the engine now reads
  `ARC_NO_FUSED_SINKHORN` — so the script looks like a no-op A/B today. It
  was not one at s2: the gate was still opt-in `ARC_FUSED_SINKHORN` until
  commit `9387e2bc5` (2026-08-13) flipped both the polarity and the default,
  **after** this measurement and *because* of it. Two independent
  confirmations that the toggle was live at s2: the same harness had already
  returned a **negative** result in s1 ("Sinkhorn fused: REJECTED — ppl drift
  + 4/6 token divergence"), which a tautological A/B cannot produce; and
  wave3-H then fixed bit-identity, which is what s2 re-verified.
  ⚠️ **The landmine is forward-looking, not backward.** Since `9387e2bc5`
  the script has been comparing fused against fused, so **any sinkhorn A/B
  run between 2026-08-13 and 2026-08-17 proves nothing by construction.**
  None is recorded — s2 is the only sinkhorn entry in this file — so no
  published claim is affected. Script fixed to drive `ARC_NO_FUSED_SINKHORN`
  in arc PR #110; re-check this entry only if a post-08-13 sinkhorn run
  surfaces from a session log.
- Absorbed MLA decode: token-identical A/B (s2).

## Costs
s1 ~$45 · s2 ~$31 · s3 ~$30.5 · s4 ~$15 ≈ **$123 cumulative** (H200 @$4.92/hr).

## Known-unmeasured (do not claim)
Full-serving batched aggregate/per-user/TTFT · MTP acceptance rate ·
voting-boosted GSM8K · twin-seed ensemble ppl · gen-2 kernel numbers ·
$/Mtok vs an in-class SGLang-on-H200 baseline (must measure to compare).

🚨 **Do not measure twin-seed ensemble ppl until `ARC_QTIP_ROTATION_SEED` is
wired.** (Found 2026-08-17 during the CLI-surface env-var audit.)
`arc-tools/quality/ensemble_ppl.py` documents its own premise as two bakes
"that differ only in the Hadamard rotation seed (bake A: default seed; bake B:
`ARC_QTIP_ROTATION_SEED=<other>`)". **Nothing in Rust reads that variable** —
it appears only in that script and in GPU_SESSION_RUNBOOK_2/4. So bake B would
come out bit-identical to bake A, and the "ensemble" would average a
distribution with itself: guaranteed to return exactly the single-bake ppl and
zero gain. That is a null result produced by plumbing, and it would look
exactly like the scientific conclusion "error patterns are correlated,
ensembling doesn't help." Wire the seed through the Hadamard rotation and
assert the two bakes' weights differ **before** spending GPU time on this.

Audit result for the other suspected orphan names, so they are not re-chased:
`ARC_QTIP_EXPERT_GREEDY` / `ARC_QTIP_EXPERT_VITERBI` **do** have Rust readers
(`mistralrs-quant/src/qtip/mod.rs`) — the wave29-BD rung decision rests on live
knobs. `ARC_FORCE_GPU_QTIP_QUANTIZE` is absent from Rust **deliberately** —
removed in `12527af2d`, with a "do not re-add" note in
`arc-tools/CUDA_VALIDATION.md`. `ARC_DISABLE_YARN_STD` appears only inside a
proposed snippet in `docs/notes/v4-reference-audit.md`; the var that actually
shipped is `ARC_YARN_ON_STANDARD_LAYERS`, with **inverted** polarity — the doc
shows code that was never merged. No recorded result depends on any of these.

---

## Session 8 (2026-08-17) — H200 `arc-mtp-radix`, driver 580.173.02, CUDA 13.1 toolkit + `cuda-compat-13-1`

All numbers below gated by a **coherence canary** (coherence6 **6/6**, facts 21/22,
math 8/8) before timing was recorded. V4 = `deepseek-ai/DeepSeek-V4-Flash` (149 GB
source) + `aeonmind/DeepSeek-V4-Flash-UQFF-qtip2b` overlay, `--features "cuda flash-attn"`.

### MTP acceptance at batch — the accept_rate/tok_per_step divergence, MEASURED
`--mtp-depth 3 --prefix-cache-n 0`, bucketed per batch fence by `mtp_bucket.py`:

| batch | **tok_per_step** | accept_rate | requests |
|---|---|---|---|
| 1   | **1.9326** | 0.3109 | 5 |
| 8   | **1.6680** | 0.4271 | 24 |
| 32  | **1.5492** | 0.4765 | 96 |
| 128 | **1.0558** | 0.4333 | 385 |

**This is the divergence, on hardware.** At b=128 `accept_rate` holds ~0.43 (looks
healthy) while `tok_per_step` collapses to **1.0558** — MTP delivers ≈nothing at
batch. **Never quote `accept_rate`**: a saturated sequence drafts 0, contributing
`proposed=0`, so the ratio flatters while the yield dies.
**Validates PR #92**: its arithmetic predicted cohort **1.1297** → per-sequence
**1.4783** at B=128. Measured cohort **1.0558** — model correct, slightly optimistic.
⇒ the ~+31% from per-sequence KV advance is real headroom.

### Aggregate throughput, MTP-on — NOT monotonic
B=1 **17.85** · B=8 **32.01** · B=32 **47.45** (peak) · B=128 **43.73** tok/s.
⚠️ **NOT comparable to the 111.69 @ B=256 baseline**: MTP depth-3 adds verify work,
prefix cache off, and PHASE1 ran `--max-seqs 128` so B=128 was pinned at the
scheduler cap and B=256 could not run. An MTP-off control at identical settings is
required before calling the B=32 peak a regression. **Do not cite as a regression.**

### 🔴 V4 serving is broken for long prompts (blocks radix, #90, real workloads)
| prompt | result |
|---|---|
| 9 words | OK — **100.15 s** for 8 tokens |
| 198 words | OK — **386.15 s** for 8 tokens |
| ~1,055 words | **CONNECTION REFUSED — server dead** |

All radix cells failed 100% (`896/768/896 × "zero tokens streamed"`). **Radix itself
is un-run, not unfavourable.** Prefix-cache `declining=4` at `match_len=8..10` ⇒ the
tree **never matched** (workload never presented a shared prefix); PR #82's KV
contract is **NOT implicated**.
⚠️ **Retraction:** the "1.43% prefix hit rate" is a **cumulative lifetime ratio** —
`engine/logger.rs:55-85` uses `load` not `swap`, so only `tokens_processed` resets.
It belongs to no single cell. Never quote it.

### 🔴 GPU sampling falls back to CPU on EVERY token
`WARN mistralrs_core::sampler: GPU radix top-k sampling failed; falling back to CPU:
tensor_device_ptr: unsupported dtype I32` — ~**10/sec, continuous**. Device→host
round trip per token in the decode loop. **5th instance of the `as_cuda_slice::<u8>()`
dtype family** (#52 fixed 3, #53 fixed 2). Consistent with being overhead-bound:
111.69 tok/s ⇒ ~0.44 decode steps/s ⇒ **low single-digit % of the H200's 4.8 TB/s**.

### FA4 substrate gate — PASS
`GATE_PASSES__aot_object_is_linkable_from_rust_without_python`, all 5 checks, symbol
`arc_fa4_probe__mlir_ciface_cutlass_arc_probe_noop`, cutlass DSL 4.7.0.
⚠️ Gate **v1 was a false negative** — it looked for the docs-promised `__tvm_ffi_*`
prefix; real objects use MLIR C-interface names. **The wheel is the authority, not
the docs.** Now enumerates + ranks symbols instead of predicting a name.

### Box environment (image trap — bake into every preflight)
Driver **580.173.02 caps at CUDA 13.0**; only toolkit present is **13.1**.
`candle-kernels/build.rs:20` calls `build_ptx()` which **always** emits PTX
(`CUDA_COMPUTE_CAP` cannot stop it), so PTX-JIT fails.
**Fix: `apt-get install cuda-compat-13-1` + `LD_LIBRARY_PATH=/usr/local/cuda/compat`.**
Differential probe: SASS `val=42` ✅ / PTX `val=0` ❌ → after compat, PTX `val=42` ✅.
⚠️ **Retraction of an overclaim:** this was reported as "silent corruption". A kernel
that never launched has nothing to sync on, so `cudaDeviceSynchronize()` returns
success and the fault sits in `cudaGetLastError()` — a **probe-discipline hazard**.
mistral.rs uses the driver API and threw `CUDA_ERROR_UNSUPPORTED_PTX_VERSION` 6×; it
refused to start rather than serve garbage.

### Costs
s8 ≈ **$14** so far (H200 @ $4.92/hr). Balance $127.16 at session start.
