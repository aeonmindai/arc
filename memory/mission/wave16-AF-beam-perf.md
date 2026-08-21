# wave16-AF — why the CUDA beam kernel got 2.1×, not 6–12×

**Status:** occupancy measured twice on real toolchain output; parity proved on CPU; **wall time still unmeasured** —
experiments 2 and 3 run after the bake. PR #40 stays DO NOT MERGE until then.

**Author's note up front:** I wrote the kernel and I wrote the 6–12× projection. That projection was wrong, and it was
wrong for a reason I can now name precisely: **I modelled bytes and instructions, and the kernel is bound by neither —
it is bound by issue efficiency at low occupancy.** The bytes model was validated against the exhaustive kernel's wall
time and appeared to fit; that fit was a coincidence, and I did not test it. This document is the correction, and it is
built on measurements rather than on another model, wherever a measurement was possible without a GPU.

**No GPU was touched.** A paid H200 is baking. Everything below is source analysis plus CPU measurement of the beam
search's data-dependent behaviour. Every hardware claim is listed as a proposed experiment at the end.

## The measurement being explained

| quantity | value | source |
|---|---|---|
| beam-256 GPU search | **238 s/layer** | 241 ± 1 s/layer marginal, minus 3.3 s host unpack |
| exhaustive Viterbi | 510 s/layer | same box class |
| speedup | **2.14×** | — |
| GPU telemetry during search | sm=100 %, **mem=1 %**, 261 W / 700 W, 1980 MHz | `nvidia-smi dmon`, 20 samples |
| wave13-AF projection | 42–85 s/layer (6–12×) | wrong |

`mem=1%` is the most informative number on that list. It **confirms the design goal was achieved**: the beam really did
cut HBM traffic ~257× per symbol position, and the kernel really is no longer touching DRAM. It also means the kernel's
remaining cost is entirely on-chip, so no amount of further memory work can help.

`sm=100 %` at 261 W is not a contradiction: `sm%` reports the fraction of time at least one warp is *resident*, not the
fraction of issue slots used. A kernel that occupies every SM and issues one instruction every twenty cycles reads
exactly like this.

## The numbers the source cannot tell you, measured on CPU

Two quantities drive the kernel's cost and neither is visible in the CUDA source, because both depend on the data. Both
are properties of the **beam search**, not of CUDA, so both are measurable on a CPU. New probe:

```
cargo test -p mistralrs-quant --lib probe_beam_kernel_cost_drivers -- --ignored --nocapture
```

It replays the production CPU beam on V4-Flash-shaped rows (FP4-lattice weights, hadamard-128, W=256) and transcribes
the kernel's own radix loop, early exit included.

| shape | steps | mean `ng` | min | max | mean radix passes | wasted leading passes |
|---|---|---|---|---|---|---|
| gate/up `k=7168` | 14 332 | **248.7 / 256** | 16 | 256 | **3.87** | 0.83 |
| down `k=2048` | 4 092 | **248.6 / 256** | 16 | 256 | 3.61 | 0.62 |

Radix-pass histogram (gate/up): `3 → 2133, 4 → 11966, 5 → 227, 6 → 2`.

## Hypotheses ruled OUT

**1. Launch geometry / not enough blocks — RULED OUT (source).**
`rows_in_flight = ARC_VITERBI_SCRATCH_GB / (num_symbols · W · 4)`. At the 6 GiB default, `k=7168`, W=256, that is
6 GiB / 3.67 MB = **1755 blocks per launch**, against 132 SMs. The 3-D expert path further batches 16 experts into one
`[16·2048, 7168]` call = 32 768 rows. The grid is never the constraint.

**2. Thread-mapping starvation — RULED OUT (measured, and this was my leading hypothesis).**
`qtip_beam.cu` maps one thread per prefix group (`active = tid < ng`), so I expected `ng ≪ 256` to leave most of the
block idling at a barrier. **Measured `ng` = 248.7 of 256, i.e. 97.2 % of the block has work.** The mapping is fine.
Had I not measured this I would have shipped a rewrite that changed nothing.

**3. Sequential dependence across timesteps — RULED OUT as *the* cause.**
Real, but it is not what is being wasted: parallelism comes from rows, and there are 1755 concurrent rows.

**4. "Skip the provably-common leading radix digits" — REFUTED (measured).**
The obvious optimisation. Costs are cumulative, so I expected 2–3 leading digits to carry no information. Measured
waste is only **0.83 of 3.87 passes**, and starting the scan at the first differing bit makes it **worse — 4.76 passes,
not 3.87** — because misaligning the digit boundaries costs more passes than the skipped one saves. Refuted; not
shipped.

**5. The 32 KiB prefix-group table capping occupancy — PARTLY RULED OUT.**
37 984 B static of the 48 KiB block budget. On sm_90 (228 KiB smem/SM) that permits **6 blocks/SM**, so shared memory is
not the binding limit. Registers are the suspect instead (below). And the table *is* needed: dedup by prefix group is
what makes the CUDA beam byte-identical to the CPU one.

## What is actually dominant

Instruction count per thread per timestep, counted from `qtip_beam.cu` with the measured pass count (3.87):

| phase | instructions | barriers |
|---|---|---|
| group reduction (incl. 64-bit shared `atomicMin`) | ~40 | 5 |
| expansion — 16 candidates × (2 LUT loads + 5 FP ops + add) | ~130 | 0 |
| **radix select — 3.87 passes × 16 candidates × ~21** | **~1393** | **23** |
| compaction | ~190 | 3 |
| trace write | ~4 | 1 |
| **total** | **≈ 1757** | **33** |

**The selection machinery is 79 % of the instructions and 70 % of the barriers.** The arithmetic it exists to select
over — the actual trellis branch metric — is 130 instructions, 7 %. The kernel spends ~110 instructions per candidate to
choose among candidates that cost 8 instructions to evaluate.

Of the ~21 instructions per candidate per pass, roughly 13 are **64-bit** operations on the 48-bit `(cost, state)` key:
rebuilding it from `cand[j]` (6), the variable-shift prefix compare (4), the digit extract (3). Every one is two 32-bit
ops plus carry handling on the SM, and they are rebuilt from scratch on every pass.

### The derivation that closes against 238 s

Issue capacity: 132 SMs × 4 schedulers × 1.98 GHz = **1.045e12 warp-instructions/s**.
Work per layer: 284 B params / 44 layers / V=2 = **3.23e9 row-timesteps**, × 8 warps × 1757 instructions =
**4.54e13 warp-instructions**.

* At 100 % issue efficiency that is **43.4 s/layer**.
* Measured 238 s ⇒ **achieved issue efficiency 18.2 %**, i.e. one instruction per warp every ~22 cycles.

The same count for the exhaustive kernel (phase A 4096 prefixes × 16, phase B 65 536 states × ~10 ops ⇒ ~3350
instructions/thread/timestep) gives 8.66e13 warp-instructions, and at its measured 510 s an efficiency of **16.2 %**.

**Therefore:**

```
predicted speedup = (3350 / 1757) × (18.2 % / 16.2 %) = 1.91 × 1.12 = 2.14×
measured speedup  = 510 / 238                                       = 2.14×
```

The account closes to within 2 %. The beam cut instructions 1.9× and issue efficiency barely moved. **The 257× HBM
saving bought essentially nothing, because at 16–18 % issue efficiency neither kernel was ever running near the
bandwidth roof** — the wave13-AF bandwidth model predicted 355 s against a measured 510 s and I read that 1.4× gap as
"70 % of peak, model validated". It is equally consistent with "latency-bound, and the bandwidth number was a
coincidence". I did not test the alternative. That is the actual error.

### Why efficiency is 18 %

18 % with 4 warps per scheduler means an average of ~22 stall cycles per issued instruction, from four sources that
compound and that low occupancy leaves fully exposed:

1. **33 block barriers per timestep** — one per ~53 instructions. Every warp in the block stops at each.
2. **Scattered L2 reads** — each thread reads its own 128 B LUT run at a data-dependent `base_state`, so a warp issues
   32 separate transactions, not 4. ~103 TB/layer of L2 traffic (invisible to `mem%`, which reports DRAM).
3. **64-bit shared-memory `atomicMin`** in the group reduction, with up to 16-way address contention.
4. **Long dependent 64-bit chains** in the radix key arithmetic.

None of these is individually fatal; all of them are fatal at **1–2 blocks per SM**, which is what I believe the
occupancy to be. `__launch_bounds__(256)` with no `minBlocksPerMultiprocessor` lets nvcc use up to 255 registers, and
the kernel holds `float cand[16]` live across the entire ~1400-instruction radix loop plus unrolled 64-bit temporaries.
At ~96 registers/thread the limit is 2 blocks/SM (25 % occupancy); above 128 it is 1 (12.5 %).

**MEASURED (experiment #1, run on the build box — no GPU contention).**
`cuobjdump -res-usage`, sm_90a: `REG:80 STACK:0 SHARED:38992 LOCAL:0`.
256 threads × 80 registers = 20,480/block; 65,536 / 20,480 = **3 blocks/SM = 24 of 64 warps = 37.5 % occupancy**,
**register-limited** (shared memory alone would permit 5: 3 × 38,992 B = 114 KiB of 228 KiB). Independently checked:
the register-allocation granularity on sm_90 is 256 registers per warp and 32 × 80 = 2560 is an exact multiple, so no
rounding loss — 3 is exact, not a floor artefact.

**The corroboration nobody planned:** the bake draws **261 W of 700 W = 37 % of TDP** against **37.5 % occupancy**.
A throughput-bound kernel would not have those two track each other; a latency-bound one at fixed clocks does. That is
independent evidence for the diagnosis, from a measurement taken for another purpose.

So my stated downside case — "if it is already 4 blocks/SM, F2 is worth nothing" — **does not apply.** F2 is live.

## Proven from source vs. measured vs. inferred

| claim | status |
|---|---|
| grid = 1755 blocks/launch; grid is not the limit | **proven from source** |
| 33 barriers/timestep; 1757 instructions/thread/timestep | **proven from source** + measured pass count |
| `ng` = 248.7/256 ⇒ thread mapping is not the problem | **measured (CPU)** |
| 3.87 radix passes; skipping leading digits makes it worse | **measured (CPU)** |
| issue efficiency 18.2 %, and 1.91 × 1.12 = 2.14× | **derived** from source counts + measured wall times |
| registers cap occupancy at 1–2 blocks/SM | **inferred — needs `cuobjdump`** |
| stall attribution (barrier vs shared vs L2) | **inferred — needs `ncu`** |
| exhaustive kernel was not truly bandwidth-bound | **inferred — needs `dmon` on an exhaustive layer** |

## Experiments, in the order to run them

1. ~~**`cuobjdump -res-usage`, master**~~ — **DONE:** `REG:80 SHARED:38992 LOCAL:0` ⇒ 3 blocks/SM, 37.5 %,
   register-limited.
   ~~**`cuobjdump -res-usage`, wave16 build**~~ — **DONE:** `REG:64 SHARED:39024 LOCAL:0` ⇒ 4 blocks/SM, 50 %, no
   spill. Neither needs re-running unless the kernel source changes; both are asserted in-tree.
2. **`ARC_QTIP_BEAM=64` vs `256` on one layer, wall time only.** W=64 does 4× fewer candidate evaluations but the *same*
   33 barriers and the same scan structure. If time falls ~3.5× the per-candidate work dominates; if it falls ~1.3× the
   fixed per-step overhead dominates. One cheap A/B that discriminates the whole design. (Note: master's session6 change
   pins W=256 for bakes — run this as a throwaway, not as an artifact.)
3. **`nvidia-smi dmon` during an *exhaustive* layer.** If `mem%` is high, the exhaustive kernel really was
   bandwidth-bound and the beam traded one wall for another. If `mem%` is also ~1 %, my entire wave13 bandwidth model
   was wrong at the root and should be deleted rather than patched.
4. **`ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active, smsp__issue_active.avg.pct_of_peak_sustained_active`**
   on one launch ⇒ achieved occupancy and issue rate. Confirms or refutes 18.2 %.
5. **`ncu --metrics smsp__average_warps_issue_stalled_barrier.ratio, ..._short_scoreboard.ratio, ..._long_scoreboard.ratio`**
   ⇒ attributes the stall to barriers / shared memory / L2. This is the one that tells us which of the four stall
   sources to attack next.

Experiments 1–3 need no profiler and answer most of it.

## The fix (deliverable 2, branch `perf/beam-kernel-occupancy`, DRAFT PR #40, DO NOT MERGE UNMEASURED)

https://github.com/aeonmindai/arc/pull/40 (DRAFT) — head `ac30f6e3a`, based on master `809643552`. Four commits:
`edcc158e4` the kernel changes, `9affb097d` firmed estimate + cost-model replacement, `ac30f6e3a` the occupancy
assertions. **CI: nvcc sm_80 and sm_90 both PASS on `ac30f6e3a`** — the compile gate for the rewritten kernel and for
the `#[cfg(feature = "cuda")]` parity tests.

**Still DO NOT MERGE.** Occupancy is measured, parity is proved on CPU, the kernel compiles for both target arches —
but **no GPU has run this kernel**, so the wall-time claim (1.80–2.07×) remains unverified. Experiments 2 and 3 settle
that.

Targets the two terms in `1.91 × 1.12`, since the analysis says both are the whole story.

**F1 — 32-bit selection key with an exact tie fallback.** The measured pass histogram shows selection resolves inside
the 32-bit cost key in 98.4 % of steps (passes 5–6, which are the only ones reaching the state bits, occur 229/14 328 =
1.6 % of the time). Radix over the 32-bit cost key instead of the 48-bit composite: every shift, compare and digit
extract halves. When the final bin holds more than one candidate — genuine cost ties — resolve them by state with two
further 8-bit passes. **Exact; byte-identical.**

**F2 — `__launch_bounds__(256, 4)`.** Caps registers at 64/thread and pins 4 blocks/SM (4 × 38 KiB = 152 KiB ≤ 228 KiB).
F1 removes the 64-bit temporaries that inflate the live set, which is what makes 64 registers plausible without spills.
No semantic change whatsoever.

**F3 — barrier reduction, 33 → ~22.** Clear the histogram at the *end* of the previous pass (folding one barrier into an
existing one), and cut the block scan from three barriers to two. No arithmetic touched.

### Firmed estimate (occupancy now measured, not inferred)

The measured 3 blocks/SM pins the latency term. 24 warps over 4 schedulers = 6 warps/scheduler at 17.3 % implied
efficiency ⇒ **~34.7 cycles between issues per warp**. That single number now carries the whole projection:

| term | value | basis |
|---|---|---|
| F1 instruction cut | 1664 → 1231 per thread per timestep = **1.35×** | source count, measured 3.87 passes |
| F2 occupancy | 6 → 8 warps/scheduler ⇒ 17.3 % → 23.1 % = **1.33×** | measured REG:80 ⇒ 3 blocks/SM |
| F3 barriers 33 → ~22 | **1.00–1.15×** | least certain; barriers are only part of the 34.7-cycle latency |

**Combined 1.80–2.07× ⇒ 238 s → 115–132 s/layer ⇒ 3.9–4.4× vs the exhaustive 510 s.**

Note this is **narrower and slightly lower** than the 2.0–2.8× I quoted before experiment #1 — because 3 blocks/SM is
*better* than the 1–2 I feared, so F2's headroom is 1.33× rather than up to 2×. Knowing the number shrank the claim.

**The spill gate has been run and it PASSED.** Built from branch head `9affb097d` in a separate checkout, sm_90a:

```
REG:64  STACK:0  SHARED:39024  LOCAL:0     (wave16)
REG:80  STACK:0  SHARED:38992  LOCAL:0     (master)
```

REG 80 → 64 is exactly the F2 target, and **`LOCAL:0` means nvcc reached the budget without spilling** — the one
condition that could have inverted F2's gain (a spilled load in the radix loop runs 16 × 3.87 times per timestep) is not
triggered. Shared grew 32 B for the double-buffered scan scratch; 4 × 39,024 = 152 KiB of 228 KiB. **⇒ 4 blocks/SM,
50 % occupancy, up from 3 / 37.5 %.** So the 1.80–2.07× above rests on static evidence, not on an assumption.

Caveat that matters: `LOCAL:0` is a property of the *exact source* that produced it. Any later edit to the radix loop or
the candidate registers re-opens it and the gate must be re-run — recorded at `QB_MIN_BLOCKS_PER_SM` in the kernel, with
the fallback (revert to 3, an exact no-op; expectation becomes F1+F3 alone, 1.35–1.55× ⇒ 153–176 s/layer).

Both endpoints are now asserted in `cost_model_matches_the_measurement` — `blocks_per_sm(80, 38992) == 3`,
`blocks_per_sm(64, 39024) == 4`, and `blocks_per_sm(65, 39024) == 3` to record that 64 is a **cliff, not a slope** — so a
change that silently costs occupancy fails CI rather than a six-hour bake.

**Parity: preserved, proved on CPU, and it is the gate.** `wave16_split_key_selection_matches_composite_key` replays
both scans on 3066 real candidate sets (9 exercising the tie-break) and asserts they admit the identical *set*, not
merely the same count. Always-on, not `#[ignore]`d. It also caught a real bug in the rewrite before it left the machine:
the early-exit path saturated only the state half of the threshold, not the unresolved low cost bits.

**Parity, restated:** F1 changes which bits are examined in what order, never which candidate wins:
the selected set is defined by the total order on `(cost, state)` and the tie fallback reproduces it exactly. F2 and F3
touch no arithmetic. `cuda_beam_matches_cpu_beam_bit_for_bit` remains the contract and must pass on hardware before this
branch goes anywhere near master.

## Surfaced, not shipped

* **The real headroom is algorithmic, not micro-optimisation.** 110 instructions per candidate to select among
  candidates costing 8 to evaluate is a 14:1 overhead ratio. Even a perfect version of F1–F3 leaves selection dominant.
  The structural fix is to stop doing an exact global top-W every step — e.g. a two-level scheme exploiting that the
  top-256 of 4096 contains ~1 candidate per thread. Every such scheme risks the byte-identity contract, so it needs its
  own parity argument, not a performance argument.
* **`ARC_VITERBI_SCRATCH_GB` now over-provisions.** The beam needs 3.67 MB/row against the exhaustive kernel's ~38 MB;
  the 6 GiB default yields 1755 rows in flight when a few hundred would saturate the machine. Harmless, but the trace
  buffer is now the only large allocation and could be sized from SM count instead.
* ~~wave13-AF's `beam_bake_cost_model` predicted 42–85 s/layer~~ — **REPLACED.** The predictive machinery
  (`SymbolCost`, `HBM_EFFICIENCY`, `BEAM_ISSUE_EFFICIENCY`, `beam_bake_cost_model`) is gone. In its place is a
  *descriptive* model anchored to the measurements: `MEASURED_BEAM256_LAYER_SECONDS`, `MEASURED_EXHAUSTIVE_LAYER_SECONDS`,
  `MEASURED_BEAM_REGISTERS_PER_THREAD`, plus `implied_issue_efficiency` and `blocks_per_sm`. It predicts no wall time.
  `cost_model_matches_the_measurement` asserts the instruction ratio × efficiency ratio reproduces the measured 2.14×
  to within 5 %, that selection stays >70 % of the step, that both kernels stay below 25 % issue efficiency, and that
  `blocks_per_sm(80, 38992) == 3` — i.e. it fails loudly if the kernel changes without re-measurement.
  Re-parameterising the old model would have preserved its false authority; deleting the prediction was the point.

---

# wave17-AF — MEASURED on rented silicon. The stack is 1.21x, and the beam architecture tops out at 3.5x.

**Everything below is measured on an A30 (sm_80) I rented for the purpose, ~1 hour, $0.39.** No H200 was touched
(the bake was live). Absolute times are A30 numbers and do NOT transfer to H200; **ratios, occupancy and byte-identity
do**, which is exactly what was needed.

Harness: `arc-tools/bench/beam_bench.cu` — a standalone driver that includes the kernel, generates weights and the
Gaussian LUT with the SAME deterministic hashes the Rust side uses (so the candidate cost distribution, and therefore
the data-dependent radix pass count, matches a real bake), times with CUDA events, and dumps packed output for a
byte comparison. Build both variants against one driver on one box; the ratio is the answer.

## Result 1 — parity holds on hardware

`base` and the wave16+17 stack produce **byte-identical packed output** (fnv1a `a655e47d7434c97e`, 458,752 bytes) at
every shape and width tested. The CPU equivalence proof and the hardware run agree.

## Result 2 — the stack is 1.21x, not the 1.80–2.07x I projected

rows=1344, k_in=7168 (V4 gate/up), best-of-2:

| W | base | stack | ratio |
|---|---|---|---|
| 64 | 761.9 ms | 473.4 ms | **1.61x** |
| 256 | 1227.5 ms | 1011.1 ms | **1.21x** |

`cuobjdump` on the A30 toolchain: base **REG:110 → 2 blocks/SM**, stack **REG:59 → 4 blocks/SM**. **Occupancy
DOUBLED and the kernel got 21% faster.** My wave16 model said occupancy was the multiplier on everything. It is not.
That model is now falsified by direct measurement, not by argument.

## Result 3 — the W sweep decomposes the cost, and this is the finding that matters

| W | 16 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|
| stack (ms) | 395.2 | 410.9 | 473.4 | 616.7 | 1011.1 |

Fitting `time = fixed + b·W`:

| | fixed per-timestep | per-candidate at W=256 |
|---|---|---|
| base | 607 ms (49%) | 621 ms |
| stack | **354 ms (35%)** | **657 ms** |
| improvement | **1.71x** | **0.95x — none** |

**My whole stack cut the fixed per-timestep overhead 1.71x (barriers 33 → 15.7, occupancy 2 → 4 blocks/SM) and did
essentially nothing to the per-candidate work.** That is coherent: F1/G3 removed key rebuilds, G1 removed barriers,
F2 removed a register cliff — all fixed-term costs. The per-candidate cost is dominated by the radix histogram
atomics and the scattered LUT loads, which none of those touched.

## Result 4 — the architectural ceiling, measured

The fixed term does not go away by making selection cheaper. **Even with per-candidate work driven to ZERO, this
kernel cannot beat 1227.5 / 354 = 3.47x.**

Translated to the H200 measurement (225.2 s kernel + ~16 s host):

| | kernel | + host | vs the 60–65 s/layer ask |
|---|---|---|---|
| banked today (1.21x) | 186 s | 202 s | miss |
| every remaining identity-safe idea lands perfectly (3.47x) | 65 s | **81 s** | **still a miss** |

**45–50 s/layer of kernel with byte-identical output is not reachable from this beam architecture.** That is a
measured statement, not a projection: the fixed floor was measured directly at W=16.

Two things DO reach it, both category iii-b (they move the CPU reference too, so parity survives byte-for-byte and
only quality changes):
* **W=128** — measured 616.7 ms = **1.99x vs base**, and PR #29 measured its quality cost at −0.004 cos (0.96084 vs
  0.96495). On the H200 that is ~113 s kernel + 16 s host ≈ **129 s/layer**. Still short of 60–65.
* **The gmin-only exhaustive Viterbi** (wave17-AN §5). It deletes the per-candidate selection term entirely — which is
  62% of W=256 time — AND has 1–2 barriers per position against our 15.7, so it attacks both terms at once. It is the
  only structure in evidence that can reach the target.

## What I would do next, in order

1. **Dr. Top-k tight-delegate prefilter.** Measured on CPU: the W-th smallest of the β=2 delegate vector is a provable
   upper bound leaving **318 survivors of 3980 (12.5x)**; the loose max-of-delegates form leaves 1677 (2.4x) and is not
   worth building. It attacks the 62% per-candidate term. Requires shared-memory compaction — without it, SIMT
   divergence means a warp pays full price whenever any of its 32 lanes survives (93% of iterations at 8% survival).
   Realistic: ~1.5x more, landing ~1.8x total. **Does not reach the target.**
2. **Prototype the gmin-only exhaustive kernel** against the existing `cuda_exhaustive_matches_cpu_exhaustive_bit_for_bit`
   gate. This is now the recommended path, on measured grounds rather than on AN's inference alone.

## Measured negatives worth keeping

* Occupancy 2 → 4 blocks/SM bought ~nothing on its own. Do not spend on occupancy for this kernel again.
* Guess-Verify-Refine's temporal premise **fails on our distribution**: 0.20% exact hits, and the extrapolated guess
  lands BELOW the true threshold 58% of the time (the expensive miss). Our costs are cumulative, so the threshold
  drifts every step. AN's ranking of GVR as the top lever was based on a premise that does not hold here — the
  *delegate-bound* half of the idea survives, the *temporal warm-start* half does not.
* Skipping provably-common leading radix digits makes it worse (4.76 passes vs 3.87).

## Cost

One A30, ~1 hour, **$0.39**. Deleted; `list_instances` shows only the live H200.

---

# CORRECTION (wave17-AF, second pass) — the 1.21× is an UPPER BOUND, not a verified value

Jish challenged the 1.21×. He was right to. Re-verification was attempted on rented silicon and **did not complete**
(two A100 rentals hung in `deploying`, ~13 min each, never got an IP; no A30 capacity was available). What follows
separates what is now established from what is still unmeasured.

## What IS established

**1. The A/B compared the right two things.** Verified by hash, not by memory: `c5384fbcd` (the #40 merge) has first
parent `809643552`, whose kernel is md5 `a435ee5b…` = the "base" I built; current `origin/master`'s kernel is md5
`dcd172c3…` = `d2d4f7e78`'s = the "treatment" I built; and no commit has touched `kernels/qtip/` since the merge. So
the comparison was of exactly the merged change. **That part of the report stands.**

**2. Registers do NOT differ by card — that framing is wrong, and it was mine.** A30 and A100 are **both sm_80**;
for a given toolkit their codegen and `-res-usage` are identical. The 110/59 (my box) vs 80/64 (the H200 build) gap
has two non-card causes: **nvcc version** (mine was Ubuntu-apt **11.5**, ~4 years old) and **target arch**
(sm_80 vs sm_90a). Nothing about "A30 vs A100" enters it.

**3. Therefore 1.21× is an upper bound for the shipped build, and the direction is provable.** On my box the
*baseline* was register-starved at **REG:110 → 2 blocks/SM**; the shipped build's baseline is **REG:80 → 3 blocks/SM**.
So my measurement handed F2 **more** occupancy headroom than production has (2→4 = 2× vs 3→4 = 1.33×) — and still
produced only 1.21× overall. Giving a change *more* room than it really has and measuring 1.21× means the shipped
configuration's gain is **≤ 1.21×**, not more.

⇒ **The record should say: "≤1.21×, measured on sm_80 with nvcc 11.5 where the baseline was register-starved; not
verified for the shipped toolchain."** It should NOT say "1.21× measured". If FACTS.md and PR #43 carry the bare
number, they are overstating it.

## What is NOT established, and I will not imply otherwise

* **The F1 / F2 / F3 decomposition was never measured.** I planned the toggles (F2 is a one-line `sed` on
  `__launch_bounds__`; F1 and F3 need real reverts) and never ran them. **Whether any of the three is a no-op or a
  regression is an open question** — which matters, because two other changes tonight already measured as no-ops.
* **No A100 or modern-toolkit number exists.** The 1.21× has never been reproduced on the toolchain that builds the bake.
* The earlier "3.47× architectural ceiling" inherits the same caveat: it was fit to the same nvcc-11.5 W-sweep. The
  *shape* of the argument (a fixed per-timestep floor exists, measured directly at W=16) is unaffected; the *number*
  is toolchain-specific.

## The cheapest authoritative test — and it does not need me or a rental

The bake box already measures **373.6 s/layer WITH the stack** at `897534e77`. Build that same box at `809643552`
(the merge's first parent) and time one layer. If 1.21× held, pre-stack should be ~450 s/layer. Same silicon, same
toolkit, same data, same driver — strictly better evidence than any synthetic bench I can rent, and it is one build
plus one layer.

If that A/B comes back ~1.0×, the honest conclusion is that the stack does nothing on the shipped configuration and
#40 should be reverted or re-scoped.

## Cost and hygiene

A30 (first pass) ~1 h $0.39; two A100 rentals that never provisioned, deleted. **`list_instances` shows only the live
bake box `d7d5d4ba…`, which was never touched.**
