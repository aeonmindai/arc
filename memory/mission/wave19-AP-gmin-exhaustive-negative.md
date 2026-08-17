# wave19-AP — exhaustive Viterbi LOSES to the beam. The codebook was the lever all along.

**Branch** `perf/gmin-exhaustive-prototype` · **PR** #42 (closed 2026-08-17, not merged — by
the author's own instruction: *"a measured negative. Not for merge; the kernel here is correct
and slower."*)
**Date** 2026-08-14 · **Box** 1×A30 (sm_80), ~20 minutes, **SPEND $0.07**, box deleted.

**Carried onto `master` during the 2026-08-17 PR triage** so the measurement survives the
closed PR. Nothing here is new work; it is the record of an experiment that was run, paid
for, and answered.

---

## The question

Should Arc's trellis bake abandon beam search for a gmin-only exhaustive Viterbi kernel?

## The answer: no, and it is not close.

One binary compiled the beam, the existing exhaustive kernel and the new gmin kernel into a
**single translation unit** — one process, one weight matrix, one LUT, no cross-build
confound. A30 sm_80, 1344 rows × k_in=7168 (the V4 gate/up shape), best-of-4.

| kernel | codebook | ms | H200 kernel s | +16 s host ⇒ s/layer |
|---|---|---|---|---|
| **beam W=256 — shipping today** | LUT | **998.0** | 185.5 | **201.5** |
| **beam W=256** | **computed** | **551.7** | 102.5 | **118.5** |
| **gmin exhaustive — byte-identical** | LUT | **1035.0** | 192.4 | **208.4** |
| **gmin exhaustive** | **computed** | **625.8** | 116.3 | **132.3** |
| gmin, backtrace deleted (not shippable) | computed | 561.7 | 104.4 | 120.4 |
| beam architectural ceiling (wave17-AF) | — | 354 | 64.9 | **80.9** |

Same ordering at the down_proj shape (k_in=2048): beam 159.1 ms vs gmin 176.8 ms with the
computed codebook. The box reproduced wave17-AF's beam number to **1.3%** (998.0 vs
1011.1 ms), which is what licenses the extrapolated H200 column.

Three findings, in increasing order of how much they close the question:

1. **The byte-identical exhaustive kernel is 4% SLOWER** than the beam it would replace —
   same box, same process, no extrapolation.
2. **The computed codebook is worth 1.81× on the beam and 1.60× on exhaustive.** The
   codebook is the lever, and it is **orthogonal to the search**.
3. **With the codebook on both, the beam still wins by 1.13×.** Exhaustive never overtakes
   anywhere on this grid, and nothing measured comes within 1.5× of the 81 s/layer ceiling.

## Why the thesis failed — worth reading before proposing a barrier-reduction kernel again

The kernel delivered **exactly** its promised shape: one `__syncthreads()` per symbol
position against the beam's measured 15.7, no atomics, no divergence, the same 4 blocks/SM.
That bought a **4% regression**. **Barriers were not the constraint.**

Exhaustive must relax 65,536 (state, predecessor) pairs per position — the
information-theoretic floor, which prefix-group reduction already sits on
(`qtip_quantize.cu:356-378`). So there was never a "gmin grouping" speedup to get. What the
kernel removes is the 65,536-entry *cost array* the existing exhaustive kernel ping-pongs
through HBM at 512 KiB per position. At ~8 ops per relaxation that is ~524k thread-ops per
row-position against the post-stack beam's ~315k: exhaustive **starts 1.66× behind on
instruction count** and must win all of it back on issue efficiency. It won back all but 4%.

## 🔑 The transferable finding

The computed codebook is worth **more to the beam** (1.81×) than to exhaustive (1.60×) —
even though the beam reads ~16× **less** codebook per position. A traffic model gets the
*sign* of that comparison wrong: the loads it removes from the beam are **scattered and
dependent** (one of the four stall sources wave16-AF named), while the exhaustive kernel's
were coalesced streaming.

> **Access shape, not traffic volume, is the predictor.**

On H200 that is **201.5 → 118.5 s/layer from a change that has nothing to do with which
search runs.**

## What became of it

The computed-codebook half **shipped**: `mcg_codebook_v2` is on `master`, and `qtip2b` — the
rung measured end-to-end in session 7 — *is* the computed-codebook rung. The exhaustive
kernel did not ship and should not be revived without a new argument, because the one on
record has been measured and lost.

**Do not re-rent a box to re-confirm this.** The prototype kernel (`qtip_gmin.cu`, 513
lines), its standalone benchmark (`arc-tools/bench/gmin_bench.cu`, `gmin_bench_run.sh`) and
the bit-for-bit CPU replay test (`gmin_replay_matches_exhaustive_bit_for_bit`) all remain on
branch `perf/gmin-exhaustive-prototype`, which was not deleted. They were deliberately not
merged: `mistralrs-quant/build.rs` globs `kernels/*/*.cu`, so landing `qtip_gmin.cu` would
compile a kernel with **zero Rust callers** into every CUDA build forever, for a result that
says not to use it.
