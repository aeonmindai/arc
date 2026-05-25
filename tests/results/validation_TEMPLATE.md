<!--
  TEMPLATE for the M1 structured findings report (deliverable RUN-136).
  On the box: cp tests/results/validation_TEMPLATE.md tests/results/validation_$(date +%Y%m%d).md
  Fill one row per technique. P-tier rule (M1_GATE.md go/no-go):
    P0/P1 = blocks M1, roll back to the owning wire-up issue and fix offline.
    P2    = may be deferred WITH a written "post-M1" rationale.
  Do NOT commit this template as the report; the report is validation_<date>.md.
-->

# Arc M1 — V4 Flash Validation Findings (`<YYYY-MM-DD>`)

- **Arc commit:** `<sha>` · **GPU:** `<sm_XX, HBM GB>` · **Operator:** `<name>`
- **Companion:** `arc-tools/RENTAL_DAY1.md` (run log) · this file = per-technique pass/fail matrix.

## Findings matrix

One row per technique. "Offline expectation" = what the CPU/mock test or prior
small-model run gave; "regression" = real-GPU measured minus that expectation
(flag anything materially worse, not just sign).

| Technique | Path / how exercised | Pass/Fail | Measured | Offline expectation | Regression | P-tier | Notes |
|---|---|---|---|---|---|---|---|
| V4 forward parity (compress dispatch) | C2 run A `--paged-attn off` + run B `--paged-attn auto`; both coherent | `<P/F>` | `<cos-sim or coherent y/n>` | RUN-151 proxy: layer 0.94/0.92, logits 0.90 (synthetic) | `<…>` | `<P?>` | branch A/B/C all hit? |
| MTP acceptance (speculative depth=4) | `--mtp-depth 4` on the V4 run | `<P/F>` | `<accept rate %, eff tok/fwd>` | `<offline MTP test>` | `<…>` | `<P?>` | |
| TurboQuant KV residency (K4/V3) | `--pa-cache-type turboquant`; HBM report | `<P/F>` | `<KV GB measured>` | mock estimator `v4_flash_h100_footprint.json` | `<…>` | `<P?>` | ≤ target? |
| SageAttention / attn-kernel cos-sim | `<test or run>` | `<P/F>` | `<cos-sim>` | `<offline>` | `<…>` | `<P?>` | bidirectional-attn pitfall #6 |
| arc-cuda-graph end-to-end decode | DedicatedDecodePath (capture-once+replay); V4 decode | `<P/F>` | `<decode T/s>` | `<offline graph test>` | `<…>` | `<P?>` | autonomous loop NOT engaged (capture deferred) — see bench JSON `moats_plumbed_not_engaged` |
| QTIP 2-bit GPU kernel parity | rental step 4b smoke (`/tmp/qtip_gpu_smoke.log`) | `<P/F>` | `<cos-sim ≥ 0.999>` | CPU ref ≥ 0.999 | `<…>` | `<P?>` | no CPU fallback on CUDA |

## Numerical stack-composition (criterion 3, real V4)

- Per-layer cos-sim vs unquantized baseline (need **≥ 0.95** every layer): `<min / which layer>`
- First-100-token greedy match vs unquantized baseline: `<N/100 matched>` → `<PASS if 100/100>`
- Offline proxy (synthetic, weaker bar) reference: layer 0.94/0.92, logits 0.90, decode min 0.77 — does **not** pre-clear this; this row is the real measurement.

## P0/P1 blockers (must fix before M1 closes)

1. `<technique>` — `<what failed, measured vs expected>` → owning issue `<RUN-…>`, fix-loop tracker RUN-137.

## P2 deferrals (allowed, with written rationale)

1. `<technique>` — `<why safe to defer post-M1>`

## Verdict

- **`<GO — all P0/P1 clear, 4 criteria ✓ | NO-GO — P0/P1 open>`**
