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

⚠️ **Surfaced, not resolved:** `225.2 s` does two jobs in that document — the
*shipped* post-stack kernel at the headline, and the *divisor* both ceiling rows
scale down from (which only works if it is the pre-stack baseline). This predates
the correction; the `65 s`/`81 s` ceiling row is computed the same way
(`225.2/3.47`). Flagged in-place rather than silently re-based, because guessing
would swap a known ambiguity for an invented number. **Needs one decision from
the coordinator: which build does 225.2 s belong to?** Both rows move with it.

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
