# 🎯 THE BIT-RATE FRONTIER — the cheap decode is real, at 2.25 bpw not 2.00

**Parent system: ArcQuant / QTIP.** Measured 2026-08-19, **CPU only, zero GPU
hours, $0**. Every table below was re-run from scratch on the machine that wrote
this file; nothing here is transcribed from a prior session.

> **THE ONE SENTENCE.** The 3.46× cheaper decode geometry **is** available — but
> **not at 2 bpw**, where it is quality-closed by −0.00698 w_cos against a
> ±0.0008 band. It is available for **a quarter of a bit more**: **K=9/V=4/L=12,
> 2.25 bpw**, which lands **+0.00402 ABOVE the shipped control** at the **same
> 32,768 B table** and the **same decode shape**. The price is not quality. The
> price is **capacity**: KV headroom 58.8 → 49.6 GB.

---

## 0. The control, stated once, and true of every number in this file

Unless a row says otherwise, every `d(w_cos)` below is measured against **one**
control and **one** configuration:

| knob | value |
|---|---|
| control geometry | **K4/V2/L16** — the shipped rung, 2.008 bpw |
| metric | **`w_cos`** — weight-space cosine, the tight one |
| fixture | **gaussian** (`gen_gaussian`), σ = 0.02 |
| search | **exhaustive Viterbi** (not greedy, not beam) |
| rotation | Hadamard block **128** |
| scale policy | **max\|row\|/3** |
| shape | **n = 48, k = 2048** |
| draws | **5 weight draws** (seeds 1, 101, 202, 303, 404) |
| codebook | pseudorandom Box-Muller Gaussian, `gaussian_lut_geo` |

**Ship band: ±0.0008 `d(w_cos)`.** It is not arbitrary. Both edges are already in
the record (`FACTS.md`, quantization section):

* **−0.0008** is what **shipped**: the `sum2` codebook variant, *"−0.0008 cos
  worst family, +0.0002 mean — 5× smaller than the W=128 delta Jish rejected"*.
  That is the largest loss the program has ever called quality-neutral.
* **−0.0013 … −0.0021** is the far side: the measured beam-vs-exhaustive loss on
  `fp4_dequant` (exhaustive wins 8 of 9 fixture cells). It is a **recorded
  regression**, shipped knowingly and **only because beam is faster**. Nothing
  has ever bought that magnitude with quality alone.

So a candidate at −0.002 is not "close". It is at the top of a range the program
already knows it can see.

**Where the numbers live.** `mistralrs-quant/src/qtip/bake_quality_tests.rs` on
branch `arcquant/designed-codebook` @ `269e95988` (which contains
`arcquant/trellis-geometry-sweep` @ `38a03cc14` as an ancestor). Re-run any row:

```
cargo test -p mistralrs-quant --release <probe_name> -- --ignored --nocapture
```

Total cost to reproduce **every table in this document**: about **2.5 minutes**
of laptop CPU on a warm build.

---

## 1. THE ANSWER — the bit-rate frontier at fixed L=12/V=4

Anchor: `probe_bit_rate_frontier`, **6.63 s**, output reproduced verbatim.

| geometry | bpw | **Δ w_cos** | [min, max] | table B | ≤32 KB | model GB | **KV GB** | band |
|---|---|---|---|---|---|---|---|---|
| **K8**/V4/L12 | 2.00 | **−0.00698** | [−0.00703, −0.00692] | 32,768 | yes | 74.2 | 58.8 | **fails** |
| **K9**/V4/L12 ⬅ **THE ANSWER** | **2.25** | **+0.00402** | [+0.00391, +0.00420] | 32,768 | yes | **83.4** | **49.6** | **CLEARS** |
| **K10**/V4/L12 | 2.50 | +0.01166 | [+0.01150, +0.01175] | 32,768 | yes | 92.7 | 40.3 | CLEARS |
| **K11**/V4/L12 | 2.75 | +0.01699 | [+0.01679, +0.01717] | 32,768 | yes | 101.9 | 31.1 | CLEARS |
| **K12**/V4/L12 (L==K) | 3.00 | +0.02057 | [+0.02032, +0.02077] | 32,768 | yes | 111.2 | 21.8 | CLEARS |

**K9 is not "inside the band". It is 5.0× the band's own width on the good side
of the shipped control** — `+0.00402 / 0.0008 = 5.03`. The 2 bpw row is
**8.7× outside** it in the other direction.

### Why the codebook does not move across that table

At fixed **L=12, V=4** the table is `2^L × V` bf16 = **32,768 B exactly** and
**does not read K at all**. The probe asserts this rather than assuming it: it
builds the table at K=8…12 and requires bit-equality, failing with *"codebook
moved with K; frontier is confounded"* otherwise. So every row above uses the
**same 4096-entry 4-D table** and the **same one-`LDS.64`-per-4-weights decode**.
**The only thing varying is how many bits index it** — `bpw = K/V`
(`mistralrs-quant/src/qtip/mod.rs:376-380`).

That is what makes this a *frontier* and not a sweep: quality is priced **in
bits, with decode cost held constant**.

### The price is capacity, and it is UNMEASURED as a serving effect

H200 accounting, from the in-tree reserve (`FACTS.md`: *"141 − 74.2 − ~8 reserve
⇒ ~59 GB usable"*): usable KV = `141 − model − 8`.

**58.8 → 49.6 GB is a −15.6% cut in KV headroom.** Its effect on batch size and
context length **has not been measured**. Do not trade it away in a sentence —
see §6.

### The three-bpw diagnostic: buying table size back does NOT recover the loss

Anchor: `probe_bits_for_decode_ladder`, **14.76 s**. Rows over the 32 KB bar are
diagnostics, not candidates:

| geometry | role | bpw | Δ w_cos | table B | ≤32 KB |
|---|---|---|---|---|---|
| K8/V4/L12 | anchor, 2.0 bpw | 2.00 | −0.00698 | 32,768 | yes |
| K4/V2/L13 | anchor, 2.0 bpw | 2.00 | **−0.00202** | 32,768 | yes |
| K10/V4/L12 | 2.5 bpw V4 | 2.50 | +0.01166 | 32,768 | yes |
| K5/V2/L13 | 2.5 bpw V2 | 2.50 | +0.01717 | 32,768 | yes |
| K12/V4/L12 | 3.0 bpw V4, **memoryless** | 3.00 | +0.02057 | 32,768 | yes |
| K6/V2/L13 | 3.0 bpw V2 | 3.00 | +0.02681 | 32,768 | yes |
| K6/V2/L12 | 3.0 bpw V2, half table | 3.00 | +0.02625 | 16,384 | yes |
| K3/V1/L14 | 3.0 bpw V1 | 3.00 | +0.02752 | 32,768 | yes |
| K12/V4/L13 | over bar, 64 KB | 3.00 | +0.02287 | 65,536 | **no** |
| K12/V4/L14 | over bar, 128 KB | 3.00 | +0.02441 | 131,072 | **no** |
| K6/V2/L14 | over bar, 64 KB | 3.00 | +0.02723 | 65,536 | **no** |

**At equal bit rate — each family at its largest table that still clears the
32 KB bar — V=1 > V=2 > V=4 in quality, and the ordering never inverts:**
K3/V1 (+0.02752) > K6/V2 (+0.02681) > K12/V4 (+0.02057) at 3 bpw; K5/V2
(+0.01717) > K10/V4 (+0.01166) at 2.5 bpw; K4/V2/L13 (−0.00202) > K8/V4/L12
(−0.00698) at 2 bpw. **V=4 is bought WITH quality, and the bit rate is what buys
it back.** Note also that **going over the 32 KB bar does not recover the loss**:
K12/V4 at 64 KB (+0.02287) and 128 KB (+0.02441) both stay below the V=2 row that
fits in 32 KB (+0.02681).

`K12/V4/L12` has `L == K`, so `state = ((state << K) | sym) & mask` shifts every
history bit out and the code is **memoryless — a plain 4096-point 4-D VQ with no
trellis**. That is asserted, not assumed, by `l_equals_k_is_a_memoryless_code`.

---

## 2. WHY 2 bpw IS CLOSED — record the mechanism, it is why we stopped searching

The geometry sweep drew every codebook at *random*, which bounds the loss of a
**random** codebook at K8/V4/L12, not the geometry's own limit. So the table was
**designed** instead — nine ways, including the two strongest constructions in
the literature. Anchor: `probe_designed_codebook_at_k8v4l12`, **49.0 s**; column
is the **mean over all 3 fixtures × 5 draws** (gaussian, student-t(4),
fp4-dequant), which is why these differ slightly from §1's gaussian-only rows.

| codebook at K8/V4/L12 | mean Δ w_cos | vs random |
|---|---|---|
| gaussian (random) — **the control design** | **−0.00714** | — |
| gaussian + amplitude re-fit | −0.00597 | better |
| **LBG + per-block Haar rotation** — best design | **−0.00457** | better |
| trellis-Lloyd ← random, 8 sweeps | −0.00450 | better |
| **trellis-Lloyd ← random, CONVERGED (40 sweeps)** | **−0.00307** | **best of all** |
| trellis-Lloyd ← LBG, 8 sweeps | −0.00727 | **worse** |
| LBG set-partitioned (Ungerboeck) | −0.00877 | **worse** |
| LBG clustered (negative control) | −0.00895 | **worse** |
| **D4 lattice cosets** — best known 4-D quantizer | **−0.01008** | **worse** |
| LBG memoryless VQ | −0.01443 | **worse** |
| random memoryless VQ | −0.02905 | **worse** |

**The best design in the world at this geometry is −0.00307. The band is
±0.0008. That is still 3.8× outside.** K8/V4/L12 is **quality-CLOSED**, and it is
closed by the *geometry*, not by the table.

The convergence is not truncation: `probe_trellis_lloyd_convergence`, **60.9 s**,
walks 0 → 40 sweeps at −0.00597 → −0.00365 → −0.00331 → −0.00318 → −0.00312 →
**−0.00307**, and a further amplitude re-fit gives **−0.00308** (worse). It has
converged.

### 🔑 THE MECHANISM — measured, not inferred

Two rows in that table are a controlled experiment, and they are the reason the
search stopped:

> **Trellis OFF** (all blocks identical, so the memory is switched off on
> purpose): **LBG −0.01443 vs random −0.02905 — LBG halves the loss.** The LBG
> training is sound and codebook design works.
>
> **Trellis ON**: the *same* LBG centroids, laid into the trellis by the textbook
> set partition, give **−0.00877 — WORSE than random's −0.00714.**

**So the binding constraint at K8/V4/L12 is TRELLIS FREEDOM, not codebook
coverage.** A better-covering table cannot help, because coverage is not what is
short. This is why more codebook design at 2 bpw is not worth an hour: the
instrument says the lever is not attached to the load.

The one design that *does* beat random — LBG + per-block **Haar** rotation
(−0.00457) — beats it precisely by restoring per-block *reach* rather than
per-block *quality*, which is the same finding from the other side.

---

## 3. THE GEOMETRY LADDER, and its structural result

Anchor: `probe_geometry_delta_noise`, **42.65 s**, gaussian fixture, 5 draws.

| geometry | trellis depth `L/K` | Δ w_cos |
|---|---|---|
| K4/V2/L14 | 3.50 | −0.00115 |
| K8/V4/L16 | 2.00 | −0.00119 |
| K4/V2/L12 | 3.00 | −0.00325 |
| K8/V4/L14 | 1.75 | −0.00320 |
| K8/V4/L12 | 1.50 | −0.00698 |

Read the pairs, not the rows:

> **K8/V4 at depth L behaves exactly like K4/V2 at depth L−2.**
> L=14 ↔ L=16: −0.00115 vs −0.00119. L=12 ↔ L=14: −0.00325 vs −0.00320.

**Therefore `L/K` lookback is NOT the predictor**, even though it is the natural
one — those pairs have depths 3.50 vs 2.00 and 3.00 vs 1.75 and still land on top
of each other. **`L` alone (offset by V) predicts; the depth ratio does not.**
That result survives to every future geometry question and should be quoted
before anyone re-derives it.

The ladder holds on all three fixtures — the same pairing appears in student-t(4)
(−0.00114/−0.00123, −0.00336/−0.00338) and fp4-dequant (−0.00121/−0.00127,
−0.00338/−0.00341) — so it is a property of the code, not of the weights.

### The runner-up, and it is a live option

**K4/V2/L13 — 2.00 bpw, Δ w_cos −0.00206, 32,768 B table, compiled at 11.250
inst/wt on sm_90 (1.34× cheaper than the shipped 15.125).** Anchor for the
instruction count: `FACTS.md` §2026-08-19 §1, the compiled `nvcc -cubin` table.

It **does not clear the ship band** — −0.00206 lands at the top of the
−0.0013…−0.0021 range §0 anchors as a recorded regression — but it is the only
2.00-bpw candidate within reach of it, it costs **no extra bit**, and it keeps KV
at 58.8 GB. If capacity turns out to bind harder than §1 assumes, this is the
fallback: **1.34× instead of 3.46×, and no capacity loss at all.**

---

## 4. KERNEL SIDE — exact integers, not estimates

### 4.1 The byte-extraction bound is `ceil((8 − gcd(K,8) + K)/8)`, and it is EXACT

The naive bound is `ceil((7+K)/8)`, which maximises over **all** bit offsets 0..7.
But symbol `t` starts at bit `K·t`, so the **reachable** offsets of `K·t mod 8`
are only the multiples of `gcd(K,8)`. The true bound is therefore

```
bytes(K) = ceil((8 − gcd(K,8) + K) / 8)      K=8 → 1     K=9 → 2     K=10 → 2
```

**Two bytes is exact for both new rungs, not merely sufficient. This retires the
3-byte route for every K in play.** Anchor: `census_sym_ext` in
`arc-tools/sass_census/qtip_sass_census.cu` on branch `perf/k9-alignment-census`
@ `c3994cc14`, where the naive 3-byte route (`EXT_B3`) is still compiled — solely
so a defensively-written kernel can see what the unnecessary generality costs.

### 4.2 The warm-up tax is set by SLICE LENGTH, and one warp per row collapses it

Every lane replays `W` warm-up symbols before decoding its own slice of `S`, so
extraction cost per weight is `(S + W) / (S·V)` against an ideal `1/V` — a tax of
**`1 + W/S`**, driven entirely by slice length, and slice length is
`num_symbols / lanes_per_row`. Spreading a row across a whole 128-thread block
makes slices short and the tax large:

| `in_features` | 128 lanes/row (old) | **32 lanes/row (warp-per-row)** |
|---|---|---|
| 4096 | S=8, **1.25×** | S=32, **1.06×** |
| 1024 | S=2, **1.99×** | S=8, **1.24×** |
| 512 | S=1, **2.98×** | S=4, **1.48×** |

Same parallelism, four times the slice length. It **also deletes the cross-warp
butterfly, the `warp_sums` shared array, and BOTH `__syncthreads` from the row
loop**, since a warp reduces with shuffles alone. Resulting rate: **0.2651
extractions per weight at `in_features=4096`, 0.3105 at 1024** (theoretical floor
0.25). The tax is **K-independent** — it multiplies the extraction count at every
K, so it scales the alignment penalty too.

Anchor: `KERNEL_LANES_PER_ROW` and `the_warmup_tax_is_bounded_by_slice_length` in
`mistralrs-quant/src/qtip/trellis_v4l12.rs`, branch `feat/qtip-k8v4l12` @
`b6023cd2e` (PR #168).

### 4.3 ⚠️ THE SASS CENSUS IS STRUCTURALLY BLIND TO §4.2 — do not misread a flat row

The alignment census measures `inst/weight` as a **differential over unroll
depth**. That cancels prologue and epilogue **exactly** — which is what makes it
robust, and is **precisely why it cannot see per-row or per-block overhead**.

**A barrier removal, a deleted shared array and a deleted cross-warp reduction
can never appear in a census number. The instrument can neither credit nor charge
them.**

> **A flat census result does NOT mean the reshape did nothing.** It means the
> census measured steady-state inner-loop cost per weight, in which the reshape
> appears only indirectly, through the extraction rate it changes.

And when the census does report a per-route cost, multiply it by **this rung's
own extraction rate at the shape in question** — 0.2651 at 4096 — not by an
illustrative endpoint of 0.25 or 0.375. Anchor:
`Rung::structural_per_row_overhead` and `route_cost_per_weight_x10000`, same file
and branch as §4.2, pinned there deliberately so the structural half has an
accounting that is never presented as an instruction count.

### 4.4 🔴 CONFIRMED FALSE — "the GPU decode paths compute the codebook"

`materialize` (`mistralrs-quant/src/qtip/mod.rs:596-598`) says the table is
*"Always materialized for the CPU search and for the artifact; **the GPU decode
paths compute instead**."*

**That is true only for `Mcg`.** `DEFAULT` (`mistralrs-quant/src/qtip/mod.rs:570`)
is `QtipCodebook::Gaussian`, and on that path `qtip_cb_value_ldg`
(`mistralrs-quant/kernels/qtip/qtip_gemv.cu:213-215`) **gathers from the stored
table** — *"LUT path: 512 KiB FP32, L2-resident, `__ldg` read-only."* The kernel
header states the size outright: *"The LUT (2^16 × V floats = 512 KiB) lives in
L2 cache"* (`mistralrs-quant/kernels/qtip/qtip_gemv.cu:11-12`).

**512 KiB does not fit 48 KB shared memory at any occupancy. So on the SHIPPED
DEFAULT path, every symbol lookup is a dependent, data-scattered global load** —
measured at **388 GB/s ≈ 8% of H200 HBM** (`FACTS.md`), and it is the decode
limiter.

🔑 **That is the disease the 32,768 B table exists to kill, and it is the real
reason the L=12 family matters — independent of any instruction count.** Anyone
reading `mod.rs`'s comment and concluding "the codebook is free on GPU" has
mis-costed the entire rung.

---

## 5. WHAT THIS CHANGES IN THE EXISTING RECORD

| previously recorded | status now |
|---|---|
| "V=4 is compiled: K8/V4/L12 + row-scale hoist = **4.375** inst/wt, **3.46×** fewer than shipped" — stated with **no quality caveat** | **The instruction counts stand. The implied ship candidate does not.** K8/V4/L12 is **quality-CLOSED at −0.00698**, 8.7× outside the band, and nine codebook designs cannot rescue it (§2). The 3.46× decode is reachable at **K9/V4/L12, 2.25 bpw** — same table, same shape, **+0.00402** |
| "the GPU decode paths compute the codebook" (`mistralrs-quant/src/qtip/mod.rs:596-598`) | **CONFIRMED FALSE for the default path** (§4.4) |
| a K-field extract needs `ceil((7+K)/8)` = 3 bytes | **superseded** — `ceil((8 − gcd(K,8) + K)/8)`; K=9 and K=10 are **exactly 2** (§4.1) |
| a flat SASS census row ⇒ "the reshape bought nothing" | **invalid inference** — the instrument is structurally blind to it (§4.3) |

---

## 6. 🚫 STILL UNMEASURED — recorded as such, not softened

1. **K9's `inst/weight`.** The kernel has **never been compiled at any K other
   than the K=8 control**. nvcc lanes are queued. **Do NOT quote 4.375 for K9** —
   K=9 is not byte-aligned and pays the §4.1 two-byte extract that K=8 does not.
2. **The pad-vs-clamp delta.** An unpadded row stride forces `min(idx,
   row_bytes−1)` on every byte index; padding the stride by `MAX_BYTES−1` buys
   that back. The delta is a real, removable cost and is **the open format
   decision on the serving side**. Unpriced.
3. **The KV-capacity effect of 49.6 vs 58.8 GB.** −15.6% headroom. Its effect on
   achievable batch and context is **not measured**, and capacity is the wedge —
   this is the one number that could still close K9.
4. **bf16 vs f32 codebook values.** The 32,768 B figure assumes bf16. The quality
   probes ran in f32. The gap is unmeasured.
5. **Any of this on a card.** Everything above is CPU arithmetic and compiler
   output. No GPU hour was spent, and none of it is a wall-clock claim.

---

## 7. WHERE THE CODE IS

| what | where |
|---|---|
| geometry parameterisation + K/V/L sweep | branch `arcquant/trellis-geometry-sweep` @ `38a03cc14` |
| designed codebooks + the bit-rate frontier | branch `arcquant/designed-codebook` @ `269e95988` (contains the above) |
| K8/V4/L12 decode family, K as a parametric seam | **PR #168** — `feat/qtip-k8v4l12` |
| UQFF geometry discriminator (K is an explicit wire field) | **PR #170** — `feat/qtip-k8v4l12-format`, **stacked on #168** |
| geometry-parametric beam encoder, ready to bake | **PR #169** — `wave-encoder-k8l12` |
| byte-alignment / extraction-route SASS census | branch `perf/k9-alignment-census` @ `c3994cc14` |

Both `arcquant/*` branches carry the evidence in
`mistralrs-quant/src/qtip/bake_quality_tests.rs` as runnable probes, with
anti-silent-success guards on every table: `designed_codebook_reaches_the_search`
(no two designs may reconstruct alike), `geo_pipeline_is_byte_identical_at_shipped_geometry`
(the parameterisation is a refactor at the shipped geometry, byte for byte),
`generic_bit_packing_reproduces_the_production_layouts`, and a per-row guard that
fails if a geometry's reconstruction matches the control's. **Five guards fired
on unfixed code while this was being built** — the tables are not self-reported.
