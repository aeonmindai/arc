# Wave 25 — AW (part 2): the kernel stack is 1.33×, measured

Follow-on to PR #49 (merged `092b0967b`). Documentation only, branch
`docs/kernel-stack-measured`.

## The measurement

Run 2026-08-15 on the **same A100 that baked the published artifact**,
immediately after the bake finished — same GPU, CUDA 12.8, same driver, same
model, same data, `ARC_QTIP_BEAM=256`, `MISTRALRS_ISQ_SINGLETHREAD=1`, **~$0.35**
of box time. Per-layer times differenced between consecutive
`Detected INT4-packed MoE expert weights` markers (the only event **both** builds
emit — the pre-#40 binary predates the `Quantized fused experts` instrumentation).

| build | samples (s) | mean |
|---|---|---|
| pre-#40 (`809643552`, PR #37) | 512, 515, 527, 528, 516, 524 | **520.3** |
| post-#40 (`master`) | 390, 392, 390, 392, 393, 394, 392, 392 | **391.9** |

**520.3 / 391.9 = 1.33×.** Corroborated by power draw (**150 W vs 208 W**, same
card, 100 % util) and by a kernel-only cross-check against the ~16 s host floor:
`(520.3−16)/(391.9−16) =` **1.34×**, consistent with the 93.4 % kernel share.

**Scope:** this is the **#37 → `master`** delta. #38, #39, #40, #41 all landed in
between; #39's unpack pool measured a speed no-op (1.0 % of layer time) and #41
is a memory fix, so **#40 is the dominant term — but #40 in isolation has still
never been measured**, and its three parts were never separated.

## 🔴 We were wrong, in the cautious direction

The previous revision published **"≤1.21×, possibly much less, never measured on
the production toolchain."** Production is **1.33× — higher than the bound.**

The bound rested on: nvcc 11.5's more register-starved baseline (`REG:110` ⇒ 2
blocks/SM vs production's `REG:80` ⇒ 3) gave the register squeeze *more* headroom,
so 1.21× must be flattered. **That reasoning was backwards.** The old compiler
**under**-stated the gain.

The mechanism was plausible; the **sign** was assumed and never tested — the
identical error as failed predictions 1–4, applied to a *caveat* instead of a
kernel. Cost: a real 1.33× win read as near-worthless for the window between the
two revisions, and the derived per-layer rows degraded from figures to
inequalities.

**Doctrine this yields:** *over-caution is also a way to be wrong.* D9 does not
say "when uncertain, publish the pessimistic bound" — an untested bound is itself
an unmeasured claim. The honest form was three facts and no bound: "measured
1.21× on a non-production toolchain; direction of the toolchain effect untested;
production unmeasured."

Written up as **item 5** in "Predictions that failed" (the section header now
says *five*, and flags that the fifth is different in kind — a prediction about
our own measurement). The `[upper bound]` grade is **retained** but now requires
the direction of the effect to be measured before it may be applied.

## Recomputed derived rows

`225.2 / 1.33 = 169.6` ⇒ kernel **≈170 s**, +16 s host ⇒ **≈186 s/layer**
(was `≥186 s` / `≥202 s/layer`).

### `225.2 s` ambiguity — RESOLVED: it is **PRE-#40**

Settled from the git record (coordinator): PR #40 merged **2026-08-14T21:41:26Z**
(`c5384fbcd`); the s6 H200 bake the decomposition was read off ran from ~20:25Z
with ETA 21:39Z; the instrumentation that produced the split (PR #39) merged
22:12:46Z. So the split was read off a live **pre-#40** run.

Consequences applied:
- `225.2 / 1.33 = 169.6` stands. The original `225.2 / 1.21 = 186` implied the
  same pre-stack reading, so that lineage was always internally consistent.
- **The headline's use of 225.2 s as the *shipped* kernel was wrong** and is
  fixed: the whole "where a layer's 241 seconds go" section now opens with a red
  banner stating every number in it is pre-#40 and was read as current for a
  week, with the timestamps. The kernel row is labelled
  `[measured, PRE-#40, H200, s6 bake]`. Host figures explicitly carry forward
  (#40 touched only the kernel); the kernel figure explicitly does not.
- The `65 s` / `81 s/layer` ceiling row now states its divisor's side of #40:
  the 3.47× is computed from the **pre-#40 baseline** (`1227.5 / 354`), so
  divisor and numerator sit on the same side. Consistent lineage — and it is
  *also* cross-card (A30).

### 🔴 Cross-card grading error — caught by the coordinator, not by me

**1.33× was measured on an A100. 225.2 s was measured on an H200.** Dividing one
by the other is a **cross-card projection, not a measurement.**

`≈170 s` and `≈186 s/layer` are therefore now graded **`[projected]`**, with the
basis stated inline: *pre-#40 H200 kernel 225.2 s [measured] ÷ 1.33× [measured on
A100] — assumes the speedup transfers across cards, which is unverified.*

Direct counter-evidence is cited in place: the same gen-1 GEMV kernel measured
**15.4 % of peak on an A30** vs **9.4–9.7 % on an H200** (agent AT) — a ratio
that does not transfer at all. Same trap, one day apart.

The `65 s` / `81 s/layer` row carries the identical caveat (A30 divisor) and was
already `[projected]`, which was right for the wrong reason — now stated.

**Lineage audit for everything else I recomputed:** the artifact-derived chain
(15.5 ms/step, ~4.1K tok/s, ~33K node, ~24 % of roofline, ~59 GB usable, the
7.7×) runs through *artifact size ÷ vendor bandwidth* — it does **not** pass
through the cross-card division, so its existing `[derived]`/`[projected]` grades
stand unchanged. Only the two `225.2 ÷ ratio` cells moved.

### Pre-#40 labelling swept beyond the one section

The pre-#40 rate was being read as current in five more places, all now labelled:
the bake-rate leaderboard row (`2.9 h on H200` → `pre-#40`, with the ~2.2 h
post-#40 projection beside it), the `~4 min/layer` comparison, the parallel-bake
cost table in `OPEN_QUESTIONS.md` (a projection someone would act on),
`HARDWARE_LESSONS.md`'s unpack-share example, and `TESTING_DISCIPLINE.md`'s
worked measurement example.

Also newly recorded: the gap between the A30 A/B's 1.21× and production's 1.33×
is measured but **not attributed** — no fixed-vs-per-candidate decomposition has
been run on the production toolchain.

## Artifact size corrected: 68 GB / 7 shards → 74.18 GB / 8 shards + residual

Published artifact is real: `aeonmind/DeepSeek-V4-Flash-UQFF-qtip2`, **15 files,
74.18 GB**, verified against the **HF API** (not the uploader's claim), loads in
**12.94 s / 517 tensors**. The 68 GB / 7-shard figure was a pre-publication
estimate. Fixed at 13 sites across 8 files, and every downstream figure
recomputed rather than left stale:

| derived figure | was (68 GB) | now (74.18 GB) |
|---|---|---|
| H200 step read | 14.2 ms | **15.5 ms** |
| per-GPU floor at B=64 | ~4.5K tok/s | **~4.1K tok/s** |
| 8-replica node aggregate | ~36K tok/s | **~33K tok/s** |
| measured 63.5 ms microbench vs roofline | ~22 % | **~24 %** |
| H200 usable after weights + ~8 GB runtime | ~65 GB | **~59 GB** (141 − 74.18 − 8) |

Note the roofline correction moves the *measured-vs-floor* ratio in our favour
(22 % → 24 %) while moving the floor itself down — both directions recorded.

The `~8×` node-level gain survives: 8 × 4.1K = 33K against the BF16 node's
~4.3K ⇒ 7.7×.

## Gates

`typos . --config .typos.toml` → **exit 0** before push. Markdown only; no cargo
gates apply. No `cargo fmt` / `rustfmt` run.
