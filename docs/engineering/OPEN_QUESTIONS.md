# Open questions and deferred work

This is the register of things that are **deliberately not done**, with the
reasoning that led to each decision, so nobody spends a day re-deriving a
conclusion that was already reached — or, worse, reopens a decision without the
argument that closed it.

It is also the honest list of **what has not been measured**. Nothing on that
list may be claimed, quoted, or reasoned from.

Evidence grades are as in [QUANTIZATION_PERFORMANCE.md](QUANTIZATION_PERFORMANCE.md):
**[measured]**, **[derived]**, **[source-verified]**, **[projected]**, **[published]**.

---

## 1. Deferred with a decision

### 1.1 The `V=4` trellis geometry — the only single-GPU route to a sub-hour bake

**Status: deferred, documented, revisit after the ordering conditions below are met.**

**What it is.** The trellis today is `L = 16` state bits, `K = 4` bits per step,
`V = 2` weights per step ⇒ 2 bits per weight. Moving to **`V = 4`, `K = 8`** keeps
2 bits per weight — **same file size, same compression ratio** — but halves the
number of steps. Work per weight is `2^L / V`, *independent of `K`*, because the
prefix-group reduction collapses the successors [derived, and the grouping is
already implemented in `mistralrs-quant/kernels/qtip/qtip_quantize.cu:356-378`].
So `V=4` is a clean **2× on the kernel**.

**Numbers** [projected]: ~100 s/layer (after the currently-scoped single-GPU work)
→ **~58 s/layer**, i.e. a 43-layer bake from 72 min to **42 min**. Note it is 2× on
the *kernel* but only **1.7× on the total**, because the ~16 s/layer host floor
does not shrink and becomes 28 % of the time. (The ~13.5 s "other host" component
of that floor has never been decomposed — that is its own open item.)

**It might improve quality rather than trade it.** Vector quantization beats
scalar for two independent reasons: correlation capture, and pure geometry —
a hexagonal packing covers space better than a square grid ("granular gain"), and
the advantage grows with dimension. One code over 4 weights sits closer to the
rate-distortion bound than one over 2. **Theory only; must be measured.** The
counter-consideration is that trellis memory `L / kV` drops from 4 symbols to 2,
and shallower memory hurts. **Our literature search found nothing covering this
trade at `L = 16`** [survey by us, `wave20-AQ` — an internal agent log, *not* a
published survey; absence of evidence, not evidence of absence] — the classical
trellis-coded quantization we found tops out at 256 states, and the only prior
art we located above ~1,000 states is QTIP itself.

**What it touches — this is why it is a project, not a flag:**

1. **The codebook generator.** Today one 32-bit multiply splits into two fp16
   halves — elegant *because* 32 = 2 × 16. `V=4` needs 64 bits of product or two
   mixing rounds. **This runs on every token at inference**, so a costlier
   generator is an inference-speed regression.
2. **Quantize kernels.** Successors per state go 4 → 256. The grouping flips from
   4096 groups of 16 to **256 groups of 256** — a different shared-memory layout
   entirely.
3. **Inference kernels — the bulk of the work.** GEMV, the grouped-GEMM path, and
   dequantize all decode 4 weights per state. **There are 98 tuned GEMV variants
   and a measured winner table, all built for `V=2`.** They need rewriting *and*
   re-tuning: its own GPU session.
4. UQFF version bump plus clean rejection of old artifacts.
5. A new CPU reference for byte-identity parity, and new parity tests.
6. Full quality re-measurement against `V=2`.

**Effort** [projected]: ~10–20 agent-hours across parallel agents (~3–5 h wall),
plus **two** GPU sessions — one to validate, one to re-tune the inference
variants.

**Why it was deferred — read this before reopening.** This is a **bake-time**
optimization that puts **inference** speed at risk. Baking happens once per model;
inference happens on every token forever, and it is what the product's economics
rest on. A 4-value codebook even slightly more expensive to decode loses on the
axis that matters — and reopening it would discard the tuning that took GEMV from
~3 % to 9.4–9.7 % of memory-bandwidth peak [measured].

**Correct order:** parallel layer baking (§1.2 — free, no format risk) → publish
an artifact → measure full-serving throughput → *then* reopen `V=4`.

### 1.2 Parallel layer baking — the cheaper alternative that needs no format change

**Status: not built. Should be done before anything in §1.1.**

Layer 7's quantization needs nothing from layer 6. The bake is **embarrassingly
parallel across layers**, and this was missed for hours because the problem was
framed as a kernel problem from the first question asked about it.

| configuration | wall clock | cost |
|---|---|---|
| 1 GPU (today) | ~2.9 h | ~$14 |
| 4 GPUs | ~44 min | ~$13.50 |
| 8 GPUs | ~24 min | ~$14.80 |

[projected, from the measured 241 s/layer and current rental prices]

**Same cost, 4–8× less wall clock**, with zero risk to the artifact format, the
inference path, or quality. The only obstacle is **naming**: UQFF shard names are
positional indices into the model's layer list, so splitting the bake across
processes needs a naming-and-merge story. That is solvable engineering, not
physics.

A related, already-noted landmine: the promotion path for a CPU-target expert
stack hardcodes device 0. It is not reached today because the mapped device is
passed through, but it will bite the first multi-GPU bake.

### 1.3 Search strategy — settled, do not reopen

Beam-vs-exhaustive was measured to convergence and is closed: the byte-identical
exhaustive kernel is **4 % slower**, and 13 % slower once both run on a computed
codebook, while being **better on quality** (+0.002 cosine, winning 8 of 9 fixture
cells). Barrier count was not the constraint; instruction count was. Full record
and numbers in
[QUANTIZATION_PERFORMANCE.md](QUANTIZATION_PERFORMANCE.md#the-exhaustive-prototype-a-clean-negative).

Likewise, **occupancy work on this kernel family is exhausted** — 2 → 4 blocks/SM
bought essentially nothing [measured]. The 2 → 4 is the measurement box's range
under **nvcc 11.5**; the shipped toolchain's baseline is already at 3 blocks/SM.
Register counts move with the **compiler version and target arch**, not with the
card.

⚠️ **What we then got wrong:** we inferred from "less occupancy headroom in
production" that the run's **speed** figure had to be an upper bound on the
shipped gain. Production subsequently measured **1.33×, higher than the 1.21×
bound we published**. Knowing a toolchain is unrepresentative tells you the number
is unreliable, **not which direction it errs.** See
[QUANTIZATION_PERFORMANCE.md](QUANTIZATION_PERFORMANCE.md#predictions-that-failed-and-why),
item 5.

---

## 2. Ready to run, not yet run

These are cheap, decided, and blocked only on someone doing them.

| # | item | why it matters | cost to settle |
|---|---|---|---|
| 1 | **Speed of the quality-neutral `sum2` codebook** | The last unmeasured number on the bake path. The measured 1.81× belongs to a *quality-negative* variant; the shippable one is projected at ~1.68×. | One binary, three settings, **~10 min of A30 time** |
| 2 | **Port the computed codebook to the decode side** | The 512 KiB LUT gather is the *named* inference bottleneck (~388 GB/s ≈ 8 % of HBM). The same change pays twice, and **we have not priced the inference half.** | A port of a pattern already proven at `K=2 / V=1` with 20/20 CUDA parity |
| 3 | **Decompose the ~13.5 s/layer "other host" bucket** | It is 5.6 % today and becomes 28 % after any 2× kernel win. Never decomposed. | Instrumentation only; zero GPU time |
| 4 | **Profiler stall attribution (`ncu`)** | Which of barriers / shared-memory atomics / scattered L2 dominates the beam kernel's stalls is still **inferred**, never measured. It decides what to attack next if anyone reopens the kernel. | One profiled launch |
| 5 | ~~**What PR #40's kernel stack is actually worth on the shipped build**~~ **— RUN 2026-08-15. Answer: 1.33×** | Measured on the production toolchain (same A100, CUDA 12.8, same driver/model/data, immediately post-bake): pre-#40 520.3 s/layer vs `master` 391.9 s/layer. The previously published **≤1.21× "upper bound" was wrong, and wrong in the cautious direction** — nvcc 11.5 *under*-stated the gain, it did not flatter it. Full protocol and the retraction in [QUANTIZATION_PERFORMANCE.md](QUANTIZATION_PERFORMANCE.md#predictions-that-failed-and-why), items 2 and 5. **Still open:** it is the #37→`master` delta (#38–#41 all landed in between), and #40's three parts were never separated. | ~~Zero rental~~ **Done, ~$0.35** |

---

## 3. Not measured — do not claim

These are the honest gaps. Each has been asserted at some point in conversation
and none of them is backed by a measurement.

- **Full-serving batched throughput.** Aggregate tokens/s at batch, per-user
  tokens/s *at that batch*, and TTFT under load have **never been measured**. This
  is the number the whole fleet-economics argument rests on. What exists is a
  kernel-level microbenchmark showing step time flat at ~63.5 ms from B=16→64
  [measured] ⇒ ~1,006 aggregate tok/s ⇒ ~15.7 tok/s per user at B=64 [derived] —
  a *kernel* result, not a serving result.
- **An in-class baseline.** Without a same-box, same-model comparison against an
  established serving stack, any $/Mtok figure floats free. Comparisons must be
  in-class (same silicon class, same model, same sharding); a comparison against a
  different generation of hardware is not evidence.
- **MTP (speculative) acceptance rate.** Three sessions attempted it and collected
  empty files because the logger had no call site; that is fixed, but the number
  has still never been produced. Any acceptance figure predating the draft-seeding
  and draft-KV fixes is meaningless regardless.
- **Voting-boosted accuracy**, and **twin-seed ensemble perplexity.** Unfired.
- **Generation-2 GEMV kernel performance.** 54 variants (async-copy staging,
  shared-memory-fed ILP, split-K) are written and **28/28 parity-proven bit-exact
  on hardware** [measured] — and completely unmeasured for speed.
- **Any absolute time on a card other than the one it was measured on.** The A30
  kernel experiments transfer as *ratios* and as byte-identity. The per-layer H200
  columns derived from them use a conversion anchor (0.1859 s/ms) and inherit that
  anchor's assumptions. The two kernels compared could also scale differently
  between architectures — one is latency- and shared-memory-bound, the other
  arithmetic- and L2-streaming-bound.
- **Any ratio measured on a different toolchain, when the change under test is a
  register or occupancy change.** Ratios transfer between *cards*; they do not
  transfer between *compilers*. Register pressure — and therefore blocks/SM, and
  therefore the entire value of a register squeeze — is set by the nvcc version
  and the `-arch` target. Such a ratio is **unrepresentative in an unknown
  direction** — do not assume it bounds the production ratio from either side
  until the direction is measured. PR #40's 1.21× was published as an upper
  bound on exactly that reasoning; production then measured **1.33×**.

---

## 4. Provisional — measured, then invalidated by a fix

These numbers were honestly measured. They are **stale**, because the code they
measured has since changed in ways confirmed to alter output. They must be
re-measured before republication, and quoted meanwhile only with their vintage.

- **Task accuracy (GSM8K 87.0 %, n=100, 0-shot chat, greedy *decoding*, 2048-token
  cap, ±6.6 pp) and perplexity (12.50 ± 3.46 on a 70-chunk corpus)** were both
  measured before two math fixes:
  - a **SwiGLU clamp** was missing on 4 of 5 expert paths — *including the shared
    expert, which every token traverses in every layer*. Fixture magnitude when it
    bites: clamped 0.7311 vs unclamped 14.8996 = **20.4×** [measured on a fixture].
    Expected direction is neutral-to-better, but that is **unmeasured on the real
    model**; whatever it lands at is the first number measured on math that matches
    the reference implementation.
  - **YaRN** rope scaling was being applied to layers that should not receive it.
    Little effect at a 2048-token cap; it matters at long context, so the
    **long-context results are provisional too**.
- **Any batched-quality result predating the causality fix is invalid.** Batched
  prefill was passing no causal mask and the caller's attention mask was discarded
  on every live path. Batch **throughput** numbers remain valid — the same kernel
  work was done — but batched **outputs** do not.

Speed numbers are effectively unaffected by the above (an elementwise clamp), but
they were taken on the older tree; re-baseline opportunistically rather than
retracting.

---

## 5. Open research questions, with the position already reached

Recorded so the next person starts from the argument rather than from zero.

**Reduced-state sequence estimation (RSSE / DDFSE).** Keep `2^J` states with
`J < L` and reconstruct the missing state bits from each survivor's own history.
This has the beam's work reduction with **zero selection cost** — a fixed,
structured survivor set instead of a top-k — which is exactly the right shape,
given that selection is 79 % of the beam kernel's instructions. `J = 8` would be
16× less work per position. **The caveat that must be tested, not assumed:**
RSSE's quality argument in the communications literature rests on the truncated
channel taps being *weak*. Our codebook is a hash of the full 16-bit state, so
every bit matters equally, and state merging may be as lossy as beam pruning —
just chosen by suffix rather than by score. This is the cheapest decisive
experiment left in the search space.

**Tail-biting tiles (T = 256).** Both QTIP and EXL3 quantize in short tiles rather
than full rows [source-verified]. It does **not** reduce work — closing a
tail-biting tile costs a second forward pass — but it caps traceback memory and
removes the fp16 range problem below. It costs bitrate at tile boundaries.
Unevaluated.

**fp16 path metrics.** Attractive on paper (2× the arithmetic throughput of fp32
on recent hardware) and **it breaks here**: at T = 9,472 positions the accumulated
metric reaches ~1e3, where the fp16 ULP is 1.0 against per-step increments of ~0.2
⇒ **the dynamic program freezes** [derived]. EXL3 escapes only because it tiles at
T = 256. Any fp16 path needs per-position metric normalization (subtract the
running frontier minimum) or tiling — **and the CPU reference must do the
identical thing, or byte-identity parity dies.**

**Block error feedback (LDLQ / GPTQ-style) stacked on rotation.** A normalized
Hadamard rotation whitens the *diagonal* of the transformed Hessian, which is why
diagonal-Hessian weighting is redundant under rotation (held in all 15 cells of
the dispersion sweep, |Δ| ≤ 0.006) [measured]. **Off-diagonal correlations
survive**, so block error feedback stacks with rotation rather than competing with
it. That — not diagonal-Hessian search — is where remaining quality lives, and it
belongs as an *opt-in* calibration pass, preserving the data-free default.
**Currently blocked:** the calibration command loads the model unquantized
(~149 GB against 141 GB of device memory), so the measurement it needs will likely
OOM before it starts.

**Whether `V=4` at `kV=8` preserves quality.** See §1.1. No literature at this
state count; it must be measured.

---

## 6. Known risks not yet addressed

Recorded because each is a live way for a claim to be wrong, not because any is
scheduled.

- **Model-agnosticism.** The published quantization config block does **not**
  describe routed experts — they ship packed under a separate key the loader does
  not parse. Anything trusting that block uniformly mis-reads **11,008 of 69,187
  tensors**. The current code is right by hardcoded assumption, not by reading
  configuration. That is a correctness risk the moment a second model arrives.
- **Sparse-attention indexer memory.** Arc's indexer key tensor is per-head where
  SGLang's V4 implementation uses a single shared head, costing roughly **64×**
  the indexer key memory and compute [source-verified against SGLang's source,
  not benchmarked — see `docs/notes/v4-reference-audit.md:1354`]. It bites exactly when the sparse path goes live —
  which is the feature that motivates this model class.
- **Silent, math-changing environment variables.** Several env-gated overrides can
  alter numerical behaviour and **nothing records whether they were set during a
  benchmark**. The standing proposal is to audit them and stamp the active set into
  every run record. Until then, a benchmark's environment is part of its result and
  is not being captured.
- **Fast-math in kernel builds has now caused three bugs**, two of them CPU/GPU
  parity divergences at the last ULP. Grep for it before trusting any new parity
  claim.

---

## Provenance

Internal agent logs: `BACKLOG.md` (the standing surfaced-not-shipped register,
including the `V=4` deferral in full), `wave20-AQ-exhaustive-research.md` (the
geometry arithmetic, the fp16 range failure, the state-count survey),
`wave19-AP-gmin-exhaustive.md` (codebook speed and quality, the RSSE framing),
`wave17-AN-speed-research.md` (the identity-safe ceiling and the literature
survey), `wave13-AD-beamhessian.md` (the rotation/Hessian dispersion sweep),
`wave13-AH-config.md` and `wave11-Z-indexer.md` (the model-agnosticism risks),
and `FACTS.md` for the "known-unmeasured" ledger this section mirrors.

Related: [QUANTIZATION_PERFORMANCE.md](QUANTIZATION_PERFORMANCE.md) for what *is*
measured on the bake, [HARDWARE_LESSONS.md](HARDWARE_LESSONS.md) for the rental
economics behind "cost to settle", and [TESTING_DISCIPLINE.md](TESTING_DISCIPLINE.md)
for why several entries above are listed as unmeasured despite having once had a
number attached.
