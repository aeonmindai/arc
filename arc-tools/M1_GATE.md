# Arc M1 gate — "V4 loads & runs (today → ~1,000 tok/s)"

The single operator-facing definition of milestone **M1**. If anything in the
rental tooling is ambiguous about "is M1 done?", **this file wins.** The live
per-issue status lives in Linear (project *Arc v2*, milestone
`38de95aa-…d936f`); this file is the stable acceptance bar that does not change
as issues open and close.

M1 is the **rental gate**: the bar V4 Flash must clear on a real GPU before any
speed-ladder work (M2+) makes sense. It is deliberately a *correctness +
baseline* gate, **not** a performance, quality, or scale gate — those are later
milestones (see "Done / not-done boundary" below).

---

## Acceptance criteria (the gate)

Verbatim from the Linear milestone, made operator-checkable. **All four must
hold on one real H100 (or H200) run:**

1. **Coherent end-to-end decode.** V4 Flash loads through Arc's `deepseekv4`
   dispatcher and generates a coherent paragraph (not garbage tokens) from a
   real prompt.
2. **All three forward paths route through the V4 compress dispatch.** Plain
   SDPA, PagedAttention, and the MLA-cache path each exercise the V4 compress
   branch — not just the default SDPA path. (A run that only ever hits one path
   does **not** satisfy this; the validation matrix must cover all three.)
3. **Numerical stack-composition test passes.** QTIP-2bit weights + TurboQuant
   KV + TD-MoE whitening + speculative MTP compose without numerical drift:
   per-layer cosine similarity vs the reference ≥ 0.95, and greedy decode
   reproduces the unquantized baseline's first 100 tokens.

   > **What the offline test (RUN-151) actually clears — read this before
   > claiming criterion 3 met.** The offline proxy is
   > `arc-engine/tests/numerical_stack_composition.rs::arc_compression_stack_composes_within_drift_budget`.
   > It composes the same stack (QTIP 2-bit Viterbi + TD-MoE Tucker +
   > TurboQuant K4/V3) on a **synthetic 128-dim / 2-layer** model and asserts a
   > deliberately **weaker** bar: per-layer cos-sim ≥ **0.85**, final-logits ≥
   > **0.80**, and **20 teacher-forced** (not greedy) decode steps ≥ **0.75**.
   > It uses teacher-forced tokens on purpose — argmax is near-uniform-noisy on
   > random init, so a greedy-match bar is meaningless at that scale. So the
   > offline test proves the stack **composes without NaN/drift blow-up**; it
   > does **not** pre-clear the ≥ 0.95 / first-100-greedy bar above. **That bar
   > is measured fresh on the real V4 weights on the rental — no offline test
   > clears it for you.** Run the proxy pre-rental (it must stay green), then
   > measure the real bar on the box.
4. **Baseline decode number captured.** The day-1 playbook records a baseline
   `tok_per_s_decode` (target order-of-magnitude **~1,000 tok/s** on V4 Flash)
   plus TTFT. The *number itself is not a pass/fail threshold* — capturing a
   trustworthy baseline is. ~1,000 tok/s is the expectation, not the gate;
   100 tok/s captured cleanly still closes M1 (it just sets where M2 starts).

---

## Required deliverables (committed artifacts)

M1 is not "done" until these exist in the tree, produced from the rental run:

| Deliverable | Path | Produced by | Linear |
|---|---|---|---|
| Day-1 rental report | `arc-tools/RENTAL_DAY1.md` | `cp arc-tools/RENTAL_DAY1.template.md arc-tools/RENTAL_DAY1.md` then fill from the run | RUN-161 |
| One-shot bench JSON | `/ephemeral/arc-v4flash-bench.json` | `arc-tools/rental_h100_v4_flash.sh` | RUN-161 |
| Structured findings report | `tests/results/validation_<date>.md` | `cp tests/results/validation_TEMPLATE.md tests/results/validation_$(date +%Y%m%d).md` then fill one row per technique + P0/P1/P2 | RUN-136 |
| HBM footprint report | `tests/results/v4_flash_h100_footprint.json` | `arc validate --target-hbm` | RUN-191 (Done) |

> **Templates exist so the operator fills blanks, not structure, on a paid box.**
> `arc-tools/RENTAL_DAY1.template.md` and `tests/results/validation_TEMPLATE.md`
> carry every required field + the exact commands. Copy (don't edit-in-place)
> them to the real deliverable paths above.

`RENTAL_DAY1.md` must contain: the `nvidia-smi` topology, the
`preflight.sh --cuda` log tail, weight-download timing, and the `arc bench`
output. `validation_<date>.md` has one row per technique (V4 forward parity,
MTP acceptance, TurboQuant residency, SageAttention cos-sim, arc-cuda-graph
end-to-end decode) with: pass/fail, measured number, regression vs the offline
expectation, and a P-tier.

---

## Wire-up debt that must land first

M1 is "wire-up of code that exists but isn't dispatched." Before launching the
rental (RUN-161), **each of these must be `Done` in Linear, or explicitly
deferred behind a runtime flag** — never silently skipped:

- RUN-155 — wire `arc_engine::dsv4` into `mistralrs-core` forward
- RUN-156 — wire `arc_engine::mtp` into the speculative pipeline
- RUN-162 — load real V4 compressor weights
- RUN-163 — Lightning Indexer for V4 CSA top-k (merged FlashMLASparse scope)
- RUN-164 — V4 mHC residual (sinkhorn-normalized mixture math)
- RUN-167 — V4 compress dispatch in the PagedAttention + MLA-cache branches
- RUN-168 — TD-MoE whitening wired into model load
- RUN-170 — `arc-cuda-graph` `MAX_KEEP=256` fix

Already closed and feeding M1: RUN-151 (numerical test), RUN-158 (Viterbi
regression), RUN-191 (HBM footprint). Check the live milestone for the current
state of the rest — do not trust this list's ordering as a status.

---

## Operator go/no-go checklist

Run top to bottom on the rental box. Every command here is verified against the
current `arc-cli`/scripts. Stop at the first ✗ and fix before paying for more
hours.

**Pre-rental (free — before you pay for a box):**

- [ ] `./arc-tools/preflight.sh` → `✓ ALL CHECKS PASSED` (CPU-only; compiles no CUDA)
- [ ] `gh workflow run "CUDA compile check (no GPU)" -R aeonmindai/arc && gh run watch -R aeonmindai/arc` → green (sm_80 + sm_90 nvcc). `-R` required: dual-remote clones resolve `gh` to upstream otherwise.
- [ ] closes the flash-attn gap for free (one-click): `gh workflow run "flash-attn compile check (no GPU)" -R aeonmindai/arc && gh run watch -R aeonmindai/arc` → green (sm_90 nvcc). On a borrowed nvcc box instead: `CUDA_COMPUTE_CAP=90 FEATURES="cuda flash-attn" bash arc-tools/cuda_compile_check.sh`.

**On the box — gate steps:**

- [ ] `./arc-tools/preflight.sh --cuda` → `0 failed`, `nvcc 12.4+`, GPU detected, `mistralrs-core` builds with `--features cuda`
- [ ] `cargo build --release -p arc-cli -p mistralrs-cli --features "cuda flash-attn"` succeeds; `export PATH="$PWD/target/release:$PATH"`
- [ ] `arc validate --index <V4 index.json> --arch deepseekv4` → schema OK (0 required tensors missing)
- [ ] `arc validate --target-hbm 60 -m deepseek-ai/DeepSeek-V4-Flash --compression-stack qtip2+td-moe` → JSON `pass: true` (≤60 GB on one H100)
- [ ] `bash arc-tools/rental_h100_v4_flash.sh` → `/ephemeral/arc-v4flash-bench.json` written; probe-mid emits coherent text ("Paris" check passes)

**Gate evaluation (the four acceptance criteria):**

- [ ] **Criterion 1** — V4 Flash decodes a coherent paragraph (`arc run -m deepseek-ai/DeepSeek-V4-Flash -a deepseekv4`)
- [ ] **Criterion 2** — all three forward paths covered in `validation_<date>.md`. The dispatch (`mistralrs-core/src/models/deepseek4.rs:1068-1150`) has exactly three outer branches, selected by `--paged-attn` × per-layer `compress_ratio`. **Two runs cover all three:**
  - `mistralrs run … -a deepseekv4 --isq qtip2 --paged-attn off` → branch C (`None` → `dsv4_attention`): every layer goes through the V4 compress dispatch (its Standard/CSA/HCA sub-paths). Covers the plain-SDPA-via-compress and MLA-cache paths.
  - `mistralrs run … -a deepseekv4 --isq qtip2 --paged-attn auto --pa-cache-type turboquant` → standard layers hit branch A (plain paged kernel), compressed (CSA/HCA) layers hit branch B (`cache_write_and_gather` → `dsv4_attention`, RUN-167). Covers the PagedAttention-routes-through-compress path.
  - Branch B requires the model to actually have CSA/HCA layers — confirm from `config.json` `compress_ratios` (values 4/128 present, not all 0). Real V4 Flash has them by design; if a run shows only `compress_ratio=0` layers, branch B is never reached and criterion 2 is **not** met.
- [ ] **Criterion 3** — numerical stack-composition passes on the real stack (cos-sim ≥ 0.95 per layer; first-100-token greedy match). Offline proxy (weaker bar, synthetic — see criterion 3 above) is already run by `preflight.sh`; to see the numbers: `cargo test -p arc-engine --test numerical_stack_composition arc_compression_stack_composes_within_drift_budget -- --nocapture`
- [ ] **Criterion 4** — `tok_per_s_decode` + TTFT captured in the bench JSON and `RENTAL_DAY1.md`

**Deliverables committed:**

- [ ] `arc-tools/RENTAL_DAY1.md`
- [ ] `tests/results/validation_<date>.md`
- [ ] `tests/results/v4_flash_h100_footprint.json`

> **GO (M1 closed):** all four acceptance criteria ✓ **and** all deliverables
> committed. Mark RUN-161, RUN-136, RUN-137 Done; M1 closes; M2 speed work
> unblocks against the captured baseline.
>
> **NO-GO:** any acceptance criterion ✗. Log the failing row in
> `validation_<date>.md` at P0/P1, roll back to the owning wire-up issue
> (RUN-137 is the fix-loop tracker), fix offline, and re-run. P2 issues may be
> deferred *with a written "post-M1" rationale* — P0/P1 may not.

---

## Done / not-done boundary

M1 is **only** correctness + a captured baseline. These are explicitly **out of
M1** and belong to later milestones — do not block M1 on them, and do not let
the playbook's broader "v2 launch" bar leak into the M1 go/no-go:

| Out of M1 | Belongs to | That milestone's gate |
|---|---|---|
| Any decode-speed *threshold* (2× lift, etc.) | M2 | ~2× over M1 baseline, ≤1% drift on GSM8K/HumanEval |
| Long-context (1M) decode not collapsing | M3 | 1M decode holds; EAGLE Pattern-3 ≥40% over MTP-only |
| Beating FP16 quality / ArcAttention | M4 | SCMoE exceeds FP16 on GSM8K/HumanEval; ArcAttention ≥5% throughput at cos-sim ≥0.9999 |
| Vendor parity (≥SGLang/vLLM), multi-node, sustained N-user runs | M5 | ≥10× SGLang on 8×H100, multi-node stable, multi-tenant SLOs hold |

Vendor-parity (`arc bench` vendor comparison) is **not yet wired** and is
post-rental by design. A V4 Flash run that decodes coherently at any speed,
covers all three forward paths, passes the numerical test, and captures a
baseline **closes M1** — even if it is slower than SGLang. That is the whole
point of the gate: get to a trustworthy baseline, then climb the ladder.

---

## Related docs

- `arc-tools/RENTAL_PLAYBOOK.md` — the hour-by-hour day-1 runbook (this gate is its target)
- `arc-tools/RENTAL_VALIDATE.md` — `arc validate --target-hbm` HBM-footprint mode (deliverable 4)
- `arc-tools/CUDA_VALIDATION.md` — the three free/paid CUDA gates feeding the pre-rental checks
- `ARC_V2.md` — the full v2 technique stack and roadmap context
