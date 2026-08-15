# Quantization performance: the trellis bake

This is the engineering record for the cost of Arc's QTIP trellis quantization
("the bake") — where the time goes, which optimizations were tried, which ones
worked, and which predictions were wrong and why. It is written for someone who
has never seen this codebase.

Everything here is about **bake time** (a one-time cost per model, paid by us and
by anyone who quantizes their own weights), not inference time. Where the two
interact, it is called out explicitly.

## Evidence grades

Every number below carries one of these labels. Nothing is stated without one.

| grade | meaning |
|---|---|
| **[measured]** | someone ran it on hardware; the box and the shape are stated |
| **[derived]** | arithmetic over measured quantities or over constants read in shipped source |
| **[source-verified]** | read directly in shipped code (ours or a third party's) |
| **[projected]** | a forward estimate. Not a measurement. Several of ours were wrong — see "Predictions that failed" |
| **[published]** | a third party's number, measured against *their* baseline, not ours |

## What is being quantized

- Format: trellis-coded quantization (TCQ), `L = 16` state bits, `K = 4` bits per
  step, `V = 2` weights per step ⇒ **2 bits per weight**. Hadamard-128 rotation
  on by default. Search is Viterbi — either the exhaustive dynamic program or a
  width-256 beam over the same trellis.
- Reference model: DeepSeek-V4-Flash, 284 B total / 13 B active, 43 quantized
  MoE layers, **6.4 B parameters per layer** ⇒ 3.2e9 symbol positions per layer
  at `V = 2`. [source-verified, `mistralrs-quant/kernels/qtip/qtip_beam.cu:11-25`]
- Artifact: ~68 GB UQFF in 7 shards. [measured, H200, reproduced 4×]

## Headline: where a layer's 241 seconds go

**Beam W=256 on an H200: 241 ± 1 s/layer.** [measured] Measured twice, as
marginal deltas between consecutive layer markers: run A 240 s, 242 s; run B
241 s, 242 s. The exhaustive DP over the same trellis is **510 s/layer**
[measured, same box class] ⇒ the beam is **2.1×** faster.

Per-layer decomposition, from in-tree instrumentation read off a live bake over
**4 consecutive layers, ±0.2 s** [measured]:

| component | time | share |
|---|---|---|
| GPU beam kernel (`Quantized fused experts`) | **225.2 s** | **93.4 %** |
| host INT4→BF16 expert unpack, 24 threads | 2.5 s | 1.0 % |
| other host (rotation, scales, serialize) | ~13.5 s | 5.6 % |

Two things follow, and both changed what we worked on:

1. **The host floor is ~16 s/layer.** Any kernel-side target below ~16 s/layer of
   total wall time is arithmetically unreachable without also attacking host
   work, and the ~13.5 s "other host" bucket has never been decomposed.
2. **The host unpack was never the bottleneck** — 1.0 %. A prior estimate put the
   remaining headroom at "3–5× from parallelizing the unpack" [projected]; the
   measurement retired it. The unpack work that shipped
   (a dedicated rayon pool, so unpack width is independent of the GPU-submission
   width) is correct by construction and a **no-op on wall time**. Keeping it was
   a deliberate choice, not a claim.

The 43-layer bake is therefore **≈2.9 h ≈ $14** at $4.92/hr [derived], which
works out to **~27 M parameters/s** [derived].

## Why the kernel is slow: selection, not arithmetic

Instruction count per thread per timestep, counted from `qtip_beam.cu` using a
**measured** radix-pass count of 3.87 (measured on CPU by replaying the
production beam on V4-shaped rows) [derived from source + measured]:

| phase | instructions | barriers |
|---|---|---|
| group reduction (incl. 64-bit shared `atomicMin`) | ~40 | 5 |
| expansion — 16 candidates × (2 LUT loads + 5 FP ops + add) | ~130 | 0 |
| **radix select — 3.87 passes × 16 candidates × ~21** | **~1393** | **23** |
| compaction | ~190 | 3 |
| trace write | ~4 | 1 |
| **total** | **≈1757** | **33** |

**Selection is 79 % of the instructions and 70 % of the barriers. The trellis
branch metric — the arithmetic the search exists to perform — is 7 %.** The
kernel spends roughly 110 instructions per candidate to choose among candidates
that cost 8 instructions to evaluate: a 14:1 overhead ratio.

The wall time closes against that count [derived]:

- Issue capacity: 132 SMs × 4 schedulers × 1.98 GHz = 1.045e12 warp-instructions/s.
- Work: 3.23e9 row-timesteps/layer × 8 warps × 1757 instructions = 4.54e13
  warp-instructions ⇒ **43.4 s/layer at 100 % issue efficiency**.
- Against a then-current 238 s, achieved issue efficiency is **18.2 %** — one
  instruction per warp every ~22 cycles.

The same count for the exhaustive kernel (~3350 instructions/thread/timestep)
gives 16.2 % at its measured 510 s, and

```
predicted speedup = (3350 / 1757) × (18.2 % / 16.2 %) = 1.91 × 1.12 = 2.14×
measured speedup  = 510 / 238                                       = 2.14×
```

The account closes to within 2 %.

> **Precision note.** That derivation used 238 s for the beam kernel, obtained by
> subtracting an *assumed* 3.3 s of host unpack from the 241 s layer time. The
> later direct instrumentation puts the kernel at 225.2 s, which moves implied
> issue efficiency to ~19.3 %. The conclusion is unchanged; the input is now
> measured rather than assumed.

Occupancy was measured statically rather than guessed, and the answer was better
than feared: `cuobjdump -res-usage` on sm_90a reports `REG:80 SHARED:38992
LOCAL:0` ⇒ **3 blocks/SM = 37.5 % occupancy, register-limited** [measured].
A corroboration nobody planned: the bake draws **261 W of 700 W = 37 % of TDP**
against 37.5 % occupancy. A throughput-bound kernel would not have those two
track each other; a latency-bound one at fixed clocks does.

## The beam's architectural ceiling — measured, not projected

A width sweep on a rented A30 decomposes the kernel's cost into a fixed
per-timestep term and a per-candidate term. 1344 rows, `k_in = 7168` (the V4
gate/up shape), best-of-2 [measured, A30 (sm_80), ~1 h, $0.39]:

| W | 16 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|
| optimized kernel (ms) | 395.2 | 410.9 | 473.4 | 616.7 | 1011.1 |

Fitting `time = fixed + b·W`:

| | fixed per-timestep | per-candidate at W=256 |
|---|---|---|
| baseline kernel | 607 ms (49 %) | 621 ms |
| optimized | **354 ms (35 %)** | **657 ms** |
| improvement | **1.71×** | **0.95× — none** |

**The fixed term does not go away by making selection cheaper.** Even with
per-candidate work driven to *zero*, this kernel cannot beat
`1227.5 / 354 = 3.47×` [measured — the floor was measured directly at W=16, not
extrapolated]. Converted to the H200 measurement (225.2 s kernel + ~16 s host):

| | kernel | + host |
|---|---|---|
| banked today (**≤1.21×** stack — upper bound, see caveat) | 186 s | **202 s/layer** [projected from the A30 ratio] |
| every remaining identity-safe idea lands perfectly (3.47×) | 65 s | **81 s/layer** [projected] |

So **45–50 s/layer of kernel with byte-identical output is not reachable from
this beam architecture.** That is a measured statement about the architecture,
not a forecast about our effort.

## The exhaustive prototype: a clean negative

The strongest remaining structural idea was to delete selection entirely — run an
exhaustive Viterbi over the deduplicated 4096-group frontier (the shape the
closest comparable project, EXL3, actually ships [source-verified]). It has
**1 barrier per symbol position against the beam's 15.7**, zero atomics, zero
divergence, at identical occupancy.

It was built, proven byte-identical against the CPU exhaustive reference, and
measured against the beam **in one binary, one process, one weight matrix, one
LUT, best-of-4** [measured, A30, ~20 min, $0.07]:

| kernel | codebook | ms | H200 s/layer (+16 s host) |
|---|---|---|---|
| **beam W=256 — what ships today** | LUT | **998.0** | 201.5 |
| beam W=256 | computed | **551.7** | 118.5 |
| **gmin exhaustive — byte-identical** | LUT | **1035.0** | 208.4 |
| gmin exhaustive | computed | **625.8** | 132.3 |
| gmin exhaustive, `__launch_bounds__(256,4)` — spills | LUT | 1281.0 | 254.1 |

Second shape, 1344 rows × `k_in = 2048` (down_proj): beam 282.5 ms (LUT) /
159.1 ms (computed); exhaustive 290.8 / 176.8. Same ordering, slightly wider gap.

**Result: the byte-identical exhaustive kernel is 4 % SLOWER than the beam it
would replace, and 13 % slower once both run on a computed codebook.** Removing
15.7 barriers per position down to 1, deleting every atomic and all divergence,
at identical occupancy, produced a regression.

The arithmetic that explains it was available before the box was rented, and is
the general lesson: exhaustive must relax 65,536 (state, predecessor) pairs per
position — the information-theoretic floor. At ~8 arithmetic operations per
relaxation that is ~524k thread-ops per row-position against the optimized
beam's ~315k. **Exhaustive starts 1.66× behind on instruction count and has to
win it all back on issue efficiency.** It won back most of it — enough to land
within 4 % — and there was nothing left over.

**Exhaustive is nonetheless the better search on quality**, which is what theory
requires and what we had previously got backwards from a single fixture.
Re-measured across 3 weight fixtures × 3 activation dispersions inside the
realistic 1e2–1e4 channel-energy band [measured, CPU]:

| fixture | exhaustive | beam W=256 | Δ |
|---|---|---|---|
| gaussian | 0.96335–0.96349 | 0.96066–0.96128 | −0.0021 to −0.0028 |
| student_t4 | 0.96372–0.96387 | 0.96332–0.96425 | −0.0004 to **+0.0004** |
| **fp4_dequant** (V4's real source distribution) | 0.96375–0.96412 | 0.96162–0.96282 | −0.0013 to −0.0021 |

**Exhaustive wins 8 of 9 cells**, and weight NMSE — the quantity the search
actually minimizes — is strictly lower for exhaustive on every fixture (fp4:
0.072858 vs 0.074319). An earlier claim that "beam W=256 is 2.6× faster at no
quality cost" should be restated as **"at a −0.002 cos cost on realistic
fixtures, which we accept."** The quality edge is real and it is bought with wall
time, not saved by it.

## The real lever is the codebook, not the search

The same experiment priced replacing the 512 KiB Gaussian lookup table with a
codebook computed arithmetically per state (a multiplicative congruential
generator, the construction QTIP's own paper proposes and EXL3 ships):

- **beam: 998.0 → 551.7 ms = 1.81×** [measured]
- **exhaustive: 1035.0 → 625.8 ms = 1.65×** [measured]

This was not derivable from a traffic model, and a traffic model gets the **sign**
of the comparison backwards. The beam reads roughly **16× less** codebook per
position than the exhaustive DP (`ng` × 128 B against the full 512 KiB), so a
bytes argument says the beam should gain *less*. It gains **more**, because the
loads the beam removes are **scattered and dependent** — one of the four stall
sources already identified in the beam kernel — while the exhaustive kernel's are
coalesced streaming. **Access shape is the predictor; traffic volume is not.**

> The source log summarizes this as "1.81× on the beam and 1.60× on exhaustive".
> The 1.60× is normalized against the *beam* baseline (998.0 / 625.8). The
> same-kernel ratio for exhaustive is 1.65×. Either way the beam gains more.

### The correction that matters: the fast codebook is not the shippable one

The 1.81× was measured on a construction (`split`) that takes the two fp16 halves
of one 32-bit product as the two `V=2` values. That construction has a **hole**:
the mask keeps the sign bit and the low 12 bits of each half and the XOR pins the
exponent into 12..15, so **a single masked fp16 half can never have magnitude
below 0.142** — a gap exactly where a Gaussian weight distribution has most of
its mass. Quality on the realistic fixture family, beam W=256, rotation 128,
n=48, k=2048, 3 weight draws × 2 activation draws per family [measured, CPU]:

| codebook | mean cos | Δcos | mean w_nmse | Δ_rel nmse |
|---|---|---|---|---|
| gaussian LUT (incumbent) | 0.96096 | — | 0.076945 | — |
| **mcg-sum2** (sums two chained products) | **0.96113** | **+0.00017** | 0.077226 | +0.37 % |
| mcg-pair | 0.95936 | −0.00160 | 0.078706 | +2.29 % |
| **mcg-split** (what the 1.81× was measured on) | 0.95921 | **−0.00174** | 0.079811 | **+3.73 %** |

Per family for the shippable `sum2`: gaussian **+0.00080**, student_t4
**+0.00051**, fp4_dequant **−0.00081**. The effect is 3–8× smaller than the
within-family fixture noise it sits in.

`sum2` costs ~10 instructions per 2 weights against `split`'s 4. Over 16
candidates per thread per timestep that is +96 instructions on ~1263 = **+7.6 %**,
so the honest prize is:

| | speedup | H200 kernel | + 16 s host | 43-layer bake |
|---|---|---|---|---|
| today (Gaussian LUT) | 1.00× | 185.5 s | 201.5 s/layer | **2.9 h** |
| computed **split** — **[measured]**, quality-negative | 1.81× | 102.5 s | 118.5 s/layer | 1.4 h |
| computed **sum2** — **[projected]**, quality-neutral | **~1.68×** | ~110 s | **~126 s/layer** | **~1.5 h** |

**Do not quote 1.81× as the codebook's value.** It belongs to a variant that
costs 10× more quality than the extra 0.13× buys. The shippable figure is a
projection until one GPU run settles it — one binary, three codebook settings,
~10 minutes of A30 time.

Calibrated against every other quality delta this program has measured:

| change | Δcos | verdict |
|---|---|---|
| greedy vs Viterbi search | −0.29 | catastrophic; permanently banned |
| beam W=128 vs W=256 | −0.004 | rejected |
| beam W=256 vs exhaustive | −0.002 | accepted, ships today |
| **computed `sum2` vs Gaussian LUT** | **−0.0008 worst family, +0.0002 mean** | **inside fixture noise** |

### The codebook pays a second time, on the inference side

The 512 KiB LUT is read by six kernels, three of them on the decode path
(`qtip_dequantize`, `qtip_gemv`, `qtip_gather_gemv`, plus `qtip_grouped_gemm`),
and it is serialized into the UQFF artifact. The LUT gather is already named as
*the* decode bottleneck — ~388 GB/s ≈ 8 % of HBM peak [measured, profiler-
attributed]. For context, the tuned GEMV path runs at 450–467 GB/s = 9.4–9.7 % of
a 4.8 TB/s peak, up 2.3× from a pre-tuning 153–192 GB/s [measured].

So the same change that makes the bake ~1.68× faster also attacks the named
inference bottleneck. **Nobody has priced that half.** It is not a one-line swap:
it needs the CPU quantizer, four decode kernels, a UQFF codebook discriminator so
old artifacts stay readable, and parity across all of it. The `K=2 / V=1` rung
already does exactly this and has 20/20 CUDA parity on hardware, so it is a port
of a proven pattern to a second geometry, not new research.

## Predictions that failed, and why

This program has now had **four** architectural predictions die on measurement.
They are recorded because the failure mode repeats: *the mechanism was real, the
magnitude was assumed.*

**1. "Beam will be 6–12× faster than exhaustive." Measured 2.1×.**
The projection came from a traffic argument: beam-256 cuts HBM traffic from
528,392 B to 2,056 B per symbol position (257×). **The traffic arithmetic was
right; traffic was not the binding constraint.** Neither kernel was ever running
near the bandwidth roof — both sit at 16–18 % issue efficiency. The bandwidth
model had predicted 355 s against a measured 510 s, and that 1.4× gap was read as
"70 % of peak, model validated". It is equally consistent with "latency-bound,
and the bandwidth number was a coincidence." The alternative was never tested.
That is the actual error, and it is why the predictive cost model was **deleted**
rather than re-parameterized — re-parameterizing would have preserved its false
authority. What replaced it is a *descriptive* model anchored to measurements
that predicts no wall time and fails CI if the kernel changes without
re-measurement.

**2. "The optimization stack will be 1.80–2.07×." Measured ≤1.21×.**

> **Caveat on this number, added after review.** The A/B compared the correct
> commits (verified by hash), but it was built with **nvcc 11.5 targeting sm_80** —
> not the toolchain that builds the bake. Registers do **not** differ by card:
> A30 and A100 are both sm_80 and share codegen for a given toolkit, so the
> `REG:110/59` seen there versus `REG:80/64` on the H200 build is explained by
> nvcc version and sm_80-vs-sm_90a alone. This matters directionally: on that box
> the *baseline* was register-starved at **REG:110 → 2 blocks/SM** where the
> shipped baseline is **REG:80 → 3 blocks/SM**, so the measurement handed the
> register squeeze **more** headroom than production has (2→4 rather than 3→4)
> and still yielded only 1.21×. Giving a change more room than it really has and
> measuring 1.21× bounds the shipped gain at **≤1.21×**. Treat it as an upper
> bound on an unrepresentative toolchain, not a verified value for the shipped
> build. The contributions of the three parts (short key / launch bounds /
> barriers) were **never separated**, so it is still open whether any one of them
> is a no-op. The cheap authoritative test is to build the bake box itself at the
> merge's first parent and time one layer against the 373.6 s/layer it already
> measures with the stack in.
The stack (32-bit selection key with exact tie fallback, a register squeeze
pinning 4 blocks/SM, and a barrier reduction 33 → 15.7) delivered exactly what it
was designed to deliver: it cut the **fixed** per-timestep term 1.71× and did
essentially nothing to the per-candidate term (0.95×). The projection had modeled
occupancy as a multiplier on everything. **Occupancy 2 → 4 blocks/SM bought
~nothing on its own** — that model is falsified by direct measurement. Do not
spend on occupancy for this kernel again.

**3. "Deleting selection will win big — GPUs are built for that shape."
Measured 4 % slower.** See above. Barriers were not the constraint.

**4. "The published 1.81× codebook figure is the prize." It was measured on a
quality-negative variant.** The neutral construction is ~7.6 % more instructions.

Two further measured negatives worth keeping:

- **Guess-Verify-Refine's temporal premise fails on our distribution.** The idea
  (warm-start the selection threshold from the previous timestep) is exact and
  reportedly 1.88–2.42× elsewhere [published]. On our data: 0.20 % exact hits,
  and the extrapolated guess lands *below* the true threshold 58 % of the time —
  the expensive miss. Our costs are cumulative so the threshold drifts every
  step. The *delegate-bound* half of the idea survives; the *temporal warm-start*
  half does not.
- **Skipping "provably common" leading radix digits makes it worse.** Measured
  waste is only 0.83 of 3.87 passes, and starting the scan at the first differing
  bit costs **4.76 passes, not 3.87**, because misaligning the digit boundaries
  costs more than the skipped pass saves.

## Where Arc sits against other trellis quantizers

There is no incumbent to catch. Per-parameter bake rates:

| method | model | wall time | rate | grade |
|---|---|---|---|---|
| QTIP | 11 B, 350 sublayers | ~12 h on A100 (~14 min/layer) | — | [published, third-party] |
| AQLM | 11 B | ~38 h A100 (~46 min/layer) | 0.03 M param/s | [published] |
| QuIP# | 70 B | "a few hours" on an 8-GPU node | ~1.9 M param/s | [published] |
| EXL3 (closest relative) | 70 B+ | "a few hours" on one RTX 4090 | ~6.5 M param/s | [published] |
| **Arc** | 284 B MoE, 43 layers | 2.9 h on H200 | **~27 M param/s** | [measured + derived] |

**EXL3 publishes no per-layer number**; the one clean third-party trellis figure
is QTIP at ~14 min/layer on an A100, against our ~4 min/layer on a 25× larger
model. Whatever Arc lands becomes the reference number, so bake-speed work here
should be framed as advancing the state of the art, not as catching up.

## What is settled, and what is next

**Settled by measurement — do not re-litigate:**

- Beam W=256 stays. Exhaustive is 4 % slower byte-identical and 13 % slower on a
  computed codebook, and its −0.002 cos quality edge is bought with wall time.
- Barrier reduction and occupancy are exhausted for this kernel family.
- 30 s/layer is arithmetically impossible for this algorithm: the instruction
  floor at 100 % issue efficiency is ~43 s/layer, and the host floor is ~16 s.
  A realistic identity-safe target is **42–60 s/layer**.

**Next, in order of measured value:**

1. **The computed `sum2` codebook.** Largest measured improvement available,
   orthogonal to the search, quality-neutral, and it pays a second time on the
   decode path. One ~10-minute GPU run closes the last unmeasured number.
2. **Parallel layer baking** — layers are independent; see
   [OPEN_QUESTIONS.md](OPEN_QUESTIONS.md).
3. Everything else on the search is worth ±13 % and the sign favours the
   incumbent.

## Provenance

Full agent logs (internal, cited so numbers can be traced to their run):
`wave16-AF-beam-perf.md` (occupancy, instruction accounting, the 6–12×
correction, the W sweep), `wave17-AN-speed-research.md` (literature survey,
issue-efficiency floor, EXL3 source analysis), `wave19-AP-gmin-exhaustive.md`
(the exhaustive prototype, codebook speed, codebook quality),
`wave20-AQ-exhaustive-research.md` (traffic accounting for the exhaustive
kernel, pipe-level floor), `wave15-AM-unpack.md` (host unpack call-graph trace).

Pull requests: **#29** (CPU beam + W=128 quality cost), **#33** (CUDA beam
kernel, bake header), **#34** (search stamped into the artifact), **#39**
(unpack pool + per-layer timing instrumentation), **#40** (the ≤1.21×
kernel stack), **#42** (the exhaustive prototype — correct, byte-identical,
slower; **not merged**, its value is the measurement).

In-tree source: `mistralrs-quant/kernels/qtip/qtip_beam.cu`,
`qtip_quantize.cu:356-378` (the prefix-group reduction — already present, and
already inside the 510 s), `mistralrs-quant/src/qtip/viterbi.rs`,
`mistralrs-quant/src/qtip/search_bench.rs`,
`mistralrs-quant/src/qtip/bake_quality_tests.rs`.

External: QTIP (arXiv 2406.11235) for the format and the "compute-based codes"
argument; [exllamav3](https://github.com/turboderp-org/exllamav3) for the
exhaustive-frontier kernel and the MCG codebook. The full bibliography for the
GPU top-k and Viterbi-decoder literature, including which results do *not*
transfer to this regime and why, is in `wave17-AN` and `wave20-AQ`.
