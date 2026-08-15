# Testing discipline

**A test that passes while its assertion is unreachable is worse than no test.**
No test is an absence of information. A vacuous test is *false* information, and
it is trusted in proportion to how green the suite looks.

In a single day of auditing this codebase, **seven tests were found passing while
verifying nothing**. They were not sloppy tests — several were carefully written,
one of them by the same person who later found it. This document records each
mechanism, then the practices adopted in response.

Assume more exist. Audit before trusting any suite, including this one.

## The seven

| # | test | mechanism of vacuity |
|---|---|---|
| 1 | `mistralrs-quant/src/fp8/quantize.rs::test_roundtrip_f8e4m3` (`:73`) | **No assertion at all.** Computes a difference, `println!`s it, returns `Ok(())`. Passes for any numbers, including all-NaN. |
| 2 | `mistralrs-quant/src/hqq/quantize.rs::test_quantize_hqq` (`:92`) | Same: compute, print, `Ok(())`. |
| 3 | every CUDA parity test | **Skips instead of failing below compute capability 8.0.** A box with the wrong capability reports the whole parity suite green — and that suite is the gate that decides whether a quantized artifact is trustworthy. |
| 4 | the MTP (multi-token-prediction) fixture | **Degenerate fixture.** Both projection inputs were all-zero, so the fused projection was identically 0 and every draft token was the same constant. The test could not have detected *any* MTP behaviour, correct or broken. |
| 5 | the generation-2 GEMV kernel suite (54 variants) | **Precondition silently unmet.** The variants have an applicability condition on the `k` dimension; the fixture's `k` was too small, so every variant was rejected and dispatch fell back to the legacy kernel. The suite passed — it had tested the legacy kernel 54 times. |
| 6 | `mistralrs-quant/src/qtip/mod.rs::qtip_matmul_cosine_similarity` (`:3734`) | **Fixture that inverts the result.** It used a pure sinusoid — the pathological worst case for Hadamard rotation. On it, the **banned** greedy trellis search scored **0.900** against Viterbi's **0.836**. On a Gaussian fixture the ordering is correct (0.869 vs 0.945). We had a passing test that actively rewarded the method the project forbids.<sup>1</sup> |
| 7 | the search-provenance / beam-width stamp assertions | **Assertion satisfiable by the bug.** They check that a baked artifact carries the search stamp the caller requested and that the stamp round-trips through the serialized format. But the stamp is derived from the *request*, so they pass identically on a bake where the requested search never reached the trellis at all. |

A related eighth: a unit test for the MTP acceptance counters constructed two bare
`AtomicUsize` values and asserted on those. It never touched product code.

> <sup>1</sup> Greedy trellis search is permanently banned in this project: it costs
> **−0.29 matmul cosine** against Viterbi on realistic fixtures, and it silently
> disables the Hadamard rotation as well, so it degrades on two axes at once while
> making the forward path *cheaper* — meaning a greedy artifact also reports
> inflated speed. There is no metric it is honest on. The search that produced an
> artifact is now stamped into the artifact and checked at load, so a mislabelled
> bake cannot pass itself off as Viterbi.

### The four mechanisms, generalized

Every one of the above is an instance of one of four failures. They are worth
naming because they are what a reviewer should look for:

1. **No assertion** — the test computes and prints.
2. **A degenerate fixture** — the assertion is reachable but trivially satisfied,
   because the input makes the interesting path collapse (zeros, too-small
   dimensions, a value that never crosses the threshold).
3. **A skip that looks like a pass** — capability guards, `cfg` gates, missing
   hardware. Green means "did not run".
4. **An assertion the bug also satisfies** — the test asserts something derived
   from the *request* rather than from the *result*, or uses a fixture on which
   the broken behaviour scores better than the correct one.

Case 4 is the dangerous one, because such a test is indistinguishable from a good
test by reading it. The only reliable way to detect it is to break the code on
purpose.

## Antidote 1: mutation verification

**Break the implementation deliberately, run the test, and confirm the assertion
fails — and that *only* the intended assertion fails.**

This is now standard practice for any test guarding a correctness contract, and
it has repeatedly changed what shipped:

- On the provenance-stamp test (#7 above), a mutation that drops the requested
  search at the very bottom of the call chain leaves assertions 1–3 **passing**.
  Only the added non-degeneracy assertion fails:

  > *"a W=1 beam must pack different symbols than the exhaustive DP over 192
  > expert-stack bytes — identical output means the requested width never reached
  > the trellis search and the beam stamp is decorative"*

  Without that one assertion the test was decorative. Two independent agents hit
  this on two independent code paths (2-D and 3-D expert stacks) on the same day.

- A second mutation on the same test (reverting the plumbing itself) turned the
  new 3-D test red while leaving the pre-existing 2-D test green — evidence that
  the new test is **specific to the gap it closes** rather than a duplicate.

### The worked example: five mutations with teeth, two rejected for having none

A CPU replay test asserting byte identity between a new GPU kernel's recursion and
the existing exhaustive reference was mutation-verified in two rounds. Five
transcription errors each break the assertion, and each is a real bug someone
could write: wrong predecessor group; branch metric dropping the accumulated
cost; local error evaluated at the group instead of the full state; backtrace read
one position late; symbol taken from the wrong nibble.

**Two further mutations were tried and rejected for having no teeth.** Both were
pure tie-break rules (last-predecessor-wins on equal cost; keying the final state
by `(cost, group)` rather than `(cost, state)`). An exact f32 tie between two
predecessors essentially never occurs on finite costs, and the ties that *do*
occur are all in the `+inf` region of the early timesteps, in groups the optimal
path never enters. Both rules are still implemented exactly as the reference has
them — that is what makes this an identity rather than an approximation — and the
**negative was recorded in the code's documentation rather than dropped**, so
nobody re-derives it.

The first version of that test failed on exactly that point. **That is the whole
reason to run the mutation arm.**

## Antidote 2: assert non-degeneracy explicitly

Make "the fixture actually exercises the thing" a *checked assertion*, not an
assumption. Examples now in-tree:

- **The variant is applicable:** a W=1 beam must pack *different* symbols than the
  exhaustive DP. If it does not, the requested search never ran.
- **The value actually crosses the interesting range:** an FP8 round-trip asserts
  `max_error > 1e-3` as well as an upper bound. FP8 E4M3 keeps 3 mantissa bits, so
  a real round trip must lose something; without the floor an all-zero fixture
  sails past the upper bound while testing nothing.
- **The fixture covers the alphabet:** a trellis replay asserts its fixture family
  exercises at least 12 of the 16 symbols.
- **The output is not trivially empty:** a hardware byte-identity check asserts
  both `0 of 1024 mismatched` *and* `1022 nonzero`, so two all-zero buffers cannot
  compare equal and pass.

The rule of thumb: for every assertion of the form "X equals Y", ask what makes
X and Y *interesting*, and assert that too.

## Antidote 3: fixture realism — fixtures lie

Three separate results in this project were nearly decided by unrealistic
fixtures, two of them on the same day.

**Strike 1 — the original incident.** A greedy search with no rotation scores
**0.888** matmul cosine on Gaussian weights and **0.675** on `fp4_dequant`
weights (Student-t snapped to the FP4 e2m1 magnitude lattice with per-32-column
block scales) [measured]. Gaussian fixtures hid a defect that only appears on the
heavy-tailed, lattice-quantized distribution that is the **actual source chain**
of this model's expert weights. The production consequence was a perplexity of
**58.85** against a 22.50 reference.

**Strike 2 — nearly overturned the compression recipe.** A probe reported
"no-rotation + Hessian weighting 0.995 beats rotation 0.965". Its fixture had a
`diag(H)` channel spread of **1.2e7 : 1**; real LLM channel energy runs
**1e2–1e4** — over-dispersed by three orders of magnitude. Re-run as a sweep of
3 weight fixtures × 5 activation dispersions [measured]:

| `diag(H)` channel spread | rotation | no-rotation + Hessian |
|---|---|---|
| 1,181 : 1 (realistic) | **0.957** | 0.874 |
| 12,757 : 1 (realistic ceiling) | **0.957** | 0.931 |
| 1.2e7 : 1 (the original fixture) | 0.963 | 0.992 |

The arms were genuinely comparable — same seeds, weights, activations, reference.
**The fixture was not realistic.** That sweep is now the reference pattern for any
result that would change a default.

**Strike 3 — the sinusoid**, #6 above.

### The rules that came out of it

- A quantization or quality fixture **must match the real source distribution**.
  Weights: FP4-lattice / heavy-tailed, not Gaussian-only. Activations: channel
  spread in the 1e2–1e4 band.
- **State the fixture's distribution and its dispersion in the test, in code.**
  A reader must be able to judge realism without running anything.
- Any result that would **change a default** must be re-run across the fixture
  *family* (gaussian / student_t4 / fp4_dequant × realistic dispersions), never a
  single fixture.
- Report the *within-family spread* alongside the effect. One later result
  (a codebook change) measured +0.0002 mean cosine with a within-family spread of
  0.0025–0.0067 — i.e. **the effect was 3–8× smaller than the noise it sits in**,
  which is the honest way to say "neutral".

## Antidote 4: measure rates by differencing consecutive markers

A pace script computed bake speed as *total elapsed ÷ layers completed*. That
includes the one-time pre-first-layer load, so it under-reports the per-layer cost
and keeps drifting as the run proceeds. Two numbers were published from it:
**135 s/layer** and a **3.8×** speedup.

Measured as **marginal deltas between consecutive layer markers**, twice, on two
runs: run A 240 s, 242 s; run B 241 s, 242 s ⇒ **241 ± 1 s/layer**, and the
speedup is **2.1×**, not 3.8×.

**A running average is not a rate.** It is a rate only in the limit, and only if
nothing one-time is included in the numerator. Difference consecutive markers.
The same error, in the opposite direction, is what made three health gates fire on
healthy hardware (see [HARDWARE_LESSONS.md](HARDWARE_LESSONS.md)): a window that
includes setup does not measure steady state.

## Antidote 5: seed the fixture, then set the bound from the measurement

Unseeded random fixtures against marginal bounds produce flakes that are invisible
until they are expensive. Measured failure rates, **20,000 independent draws
through the real round trip** for each [measured]:

| test | gate | old bar | **measured flake rate** | new bar (seeded) | measured value on the pinned input |
|---|---|---|---|---|---|
| `blockwise_fp8` round-trip | CUDA | `< 0.16` | **22.6 %** (4517/20000) — 1 in 4.4 | **`< 0.096`** | 0.09135818, seed `0xB10C0003` |
| `vector_fp8` round-trip | CUDA | `< 0.24` | **4.26 %** (851/20000) — 1 in 23 | **`< 0.185`** | 0.17539787, seed `0xF8E40002` |
| `vector_fp8` quant (CPU) | CPU | `< 0.27` | 0.54 % (108/20000) — 1 in 185 | **`< 0.17`** | 0.16177654, seed `0xF8E40001` |

Three things about that table:

1. **Every new bound is TIGHTER than the one it replaced.** Seeding is not a way
   to loosen a bar to make a flake go away; it converts a wide bar guarding a
   random input into a narrow bar guarding a fixed one. The tests gained teeth.
2. **The two worst were CUDA-gated**, so they never ran on a developer machine or
   in CI. Their first opportunity to fire was on a paid GPU session, where a
   spurious red costs money and attention.
3. **The rate was measured, not estimated.** The initial estimate for the third
   row was "roughly 1 run in 8"; the measurement was 1 in 185. The estimate had
   been left behind in a code comment and was replaced with the measured table.
   Estimates in comments become facts to the next reader.

Seeding uses a shared deterministic generator
(`mistralrs-quant/src/test_rng.rs`, SplitMix64 + Box–Muller, bit-identical per
seed across platforms) which carries the flake-class table so the next instance
has somewhere to go.

### When you cannot run the test, say so — and derive the bound in the open

Two of those bounds could not be executed where they were written: no GPU was
available, and the blockwise path has no CPU implementation at all. Rather than
guess, the bound was derived along two independent legs, **both written into the
code beside the assertion**:

- **Kernel replication.** Both CUDA kernels round *twice* (`__float2half`, then
  `__nv_cvt_halfraw_to_fp8` with saturating round-to-nearest-even) where the CPU
  path rounds once. Emulating the double rounding on the pinned input: exactly
  **1 code of 256 flips**, and `max_error` is **bit-identical** either way —
  because a code only flips within half an f16 ULP of an fp8 midpoint, where both
  choices sit half a step from the true value.
- **Drift bound.** Perturbing every input element by **1e-6 relative** (orders of
  magnitude beyond last-bit `ln`/`sin_cos` drift across libm implementations) over
  64 patterns moves `max_error` by ±1.5e-5 and ±1.1e-5. The ~5 % margins are
  300–400× that.

A derived bound is acceptable. A derived bound presented as a measurement is not.

## Antidote 6: before trusting instrumentation, grep for its call site

Three things were found **present, plumbed, documented, and dead**:

- `log_acceptance_rate()` had **zero callers**, while the function that
  accumulates its counters was called correctly on the hot path. The counters
  accumulated into a void. **Three consecutive measurement sessions collected an
  MTP acceptance rate and got empty files.**
- The bake header logger was unreachable on the GPU path, so **no GPU bake ever
  recorded which search produced it** — the audit trail for the project's most
  important quality rule did not exist where it mattered.
- A whitening step was fed an identity matrix, which made a "lossless" claim plain
  lossy truncation.

The gate that catches this class is a test that asserts the line **actually
reaches the logging framework**: install a capturing `tracing` subscriber, run the
production entry point, and assert the emitted text appears exactly once — and
that with the gate off nothing is emitted while the counters still accumulate.

A cadence bug in the same area shows why the assertion has to be on the *emitted
output*: a naive `total % 64 == 0` reporting gate emits **zero** reports when the
speculation depth is 3, because 64/3 is not an integer, so the boundary is crossed
63 → 66 and never landed on. The test pins that: 900 proposed tokens at depth 3
must produce exactly 14 reports.

## Antidote 7: a test must fail loudly, never skip

Every test added under this discipline is a plain CPU test with **no `cfg` gate,
no capability guard, and no skip path**. Where a test genuinely requires a device,
it must **error** when the device is absent rather than return success — verified
by running the ungated body and confirming it fails with
`"…requires CUDA feature"` rather than passing.

This is the antidote to failure #3, and it is not fully applied: the existing CUDA
parity tests still skip silently below compute capability 8.0. The session
procedure now asserts the capability separately, which is a workaround, not a fix.

## Checklist for a new test

- [ ] Does it **assert**, rather than print?
- [ ] **Mutate the implementation** — does the test go red? Does *only* the
      intended assertion go red?
- [ ] Is there a **non-degeneracy assertion** (the fixture crosses the threshold /
      the variant is applicable / the output is not trivially empty)?
- [ ] Is the fixture's **distribution and dispersion stated in code**, and does it
      match the real source distribution?
- [ ] If it would change a default, has it been run across the **fixture family**?
- [ ] Is any randomness **seeded**, with the bound set from a **measured** value on
      that seed — and is the new bound tighter, not looser?
- [ ] Can it **skip**? If yes, does green genuinely mean "verified"?
- [ ] If it tests instrumentation, does it assert the **emitted output**, not the
      counters?
- [ ] If a number in it could not be executed, is it labelled as **derived**, with
      the derivation beside the assertion?

## Provenance

Internal agent logs: `wave14-AK-blockers.md` and `wave14-AL-landmines.md` (the
flake measurements, the mutation-verification pattern, the two assertion-free
tests), `wave13-AG-qtip2b-default.md` (the sinusoid fixture),
`wave13-AD-beamhessian.md` (the dispersion sweep), `wave12-AB-mtp.md` (the
all-zero MTP fixture), `wave9-X-gen2grid.md` (the too-small `k` fixture),
`wave19-AP-gmin-exhaustive.md` (the seven-mutation replay test),
`wave3-G-bakequality.md` (the original Gaussian-vs-FP4 finding).

Pull requests: **#36** (MTP telemetry call site + capturing-subscriber test),
**#38** (3-D bake testability, seeded FP8 flakes), **#33** (bake header reachable
on the GPU path), **#34** (search stamped into the artifact and checked at load).

In-tree: `mistralrs-quant/src/test_rng.rs` (the shared deterministic generator and
the flake-class table), `mistralrs-quant/src/qtip/greedy_ban_tests.rs`,
`mistralrs-quant/src/qtip/bake_quality_tests.rs` (the realistic fixture
generators), `mistralrs-quant/src/qtip/bake_memory_tests.rs`,
`mistralrs-core/src/pipeline/mtp_pipeline.rs`.
