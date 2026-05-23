# Arc Runway — Linear Cleanup Pass 2 Summary

Date: 2026-05-23
Project: **Arc v2** (slug `arc-v2-5227a43a042d`, team Runcrate)
Prior pass: `research/linear_restructure_summary.md` (cleanup-1, M1–M5 created)

## TL;DR

Created M6 milestone (multi-model expansion). Closed 21 orphan/duplicate/out-of-scope/merged issues with explanatory comments. Moved 15 still-valid items into M1/M5/M6 with the standard Ships/Moves/Proves/Dependencies/Effort template + correct labels. One inline runway duplicate merged: RUN-163 absorbed RUN-172's kernel scope (Lightning Indexer + FlashMLASparse are the same kernel family).

Result: every active backlog item (48 of them) now lives in a milestone with labels and a templated description. No untouched orphans remain.

## Before → After

| Metric | Before pass 2 | After pass 2 |
| --- | --- | --- |
| Milestones | 5 (M1–M5) | **6 (M1–M6)** |
| Issues in milestones | 34 | **51** |
| Orphan issues (no milestone) | 37 | **20** (all in closed state) |
| Active Backlog items | 34 | **48** |
| Closed (Done + Canceled) | 37 | **23** |
| Total issues | 71 | 71 |

## Per-milestone counts

| Milestone | Backlog | Canceled | Total |
| --- | --- | --- | --- |
| M1: V4 loads & runs | 17 | 2 | 19 |
| M2: Tier 1 speed | 4 | 1 | 5 |
| M3: Tier 2 speed | 3 | 0 | 3 |
| M4: Quality moat | 3 | 0 | 3 |
| M5: Research bets | 14 | 0 | 14 |
| **M6: Multi-model expansion (NEW)** | **7** | **0** | **7** |
| Orphan (closed) | 0 | 2 | 20 (18 Done + 2 Canceled) |
| **Total** | **48** | **5** | **71** |

Note: the brief's pre-execution math estimated M1 at "14 + 3 = 17" and total alive at 50ish. Actual final M1 Backlog = 17 (matches). M5 has 14 (brief's pre-exec said 14, matches). M2 dropped to 4 Backlog (5 if you count the canceled FlashMLASparse), M6 has 7 (matches the brief's count of 7). The brief's "total alive after cleanup: 71 - 21 = 50" was close; actual is 48 Backlog + a few Canceled-but-in-milestone items.

## M6 creation

Created via `save_milestone`.

- ID: `48a088cf-289e-4bf9-acb5-b95250b35cd2`
- Name: **M6: Multi-model expansion (K2.6 + GLM-5.1)**
- Description: ships K2/GLM-family loaders + model code + native sparse-attention routing once V4 stack lands.

## Closed-as-shipped (Step 2 — 6 items)

All were already in **Done** state when the cleanup-2 pass started (they were closed en masse during the prior 12-agent orchestration sprint). The pass added an explanatory closure comment to each so the "why this is closed" is preserved on the thread.

| ID | Title | Verified via |
| --- | --- | --- |
| RUN-145 | Wire DeepSeek V4 into mistralrs-core Pipeline | commit `8f0a71644` + `4a54b0164` |
| RUN-152 | Real deepseek4.rs with V4-specific tensor shapes | commit `8f0a71644` |
| RUN-139 | Add DeepSeek V4 model loader (Pro + Flash) | commit `8f0a71644` (duplicate of RUN-145) |
| RUN-135 | Build-complete gate: 13 v2 components compile + test | commit `4a54b0164`, 383/383 tests |
| RUN-128 | Complete arc-cuda-graph autonomous decode | commit `42a6ed691`, `arc-cuda-graph/src/` |
| RUN-119 | Port QTIP trellis quantization | `mistralrs-quant/src/qtip/mod.rs` (Greedy shipped; Viterbi tracked at RUN-158) |

All six commit hashes verified via `git log --oneline -100`.

## Closed-as-duplicate (Step 3 — 11 items)

All already in closed state (Done or Canceled). Closure comments added pointing to canonical issue.

| ID | Canonical target | Title |
| --- | --- | --- |
| RUN-149 | RUN-170 | arc-cuda-graph GPU sampling kernel matching CPU reference |
| RUN-150 | RUN-159 (M5) | Wire EAGLE-3 head loading into mistralrs-core |
| RUN-124 | RUN-159 (M5) | Add EAGLE-3 pre-trained head loader |
| RUN-157 | RUN-168 (M1) | Wire td_moe whitening into tucker_decompose |
| RUN-144 | RUN-168 (M1) | TD-MoE multi-linear whitening |
| RUN-120 | RUN-168 (M1) | Implement TD-MoE Tucker decomposition pipeline at model load |
| RUN-142 | RUN-162 (M1) | Replace uniform compressor with learned 2D matrix |
| RUN-138 | RUN-163 (M1, merged) | Port DeepSeek V4 CSA + HCA hybrid attention from SGLang |
| RUN-121 | RUN-156 (M1) | Wire DeepSeek V3+ native MTP heads into Arc's speculative path |
| RUN-123 | RUN-148 (M2) | Port SageAttention 3 Blackwell INT8 kernels |
| RUN-130 | RUN-175 (M3) | Implement YOCO cross-layer KV sharing |

Plus the bonus duplicate from Step 6:

| ID | Canonical target | Title |
| --- | --- | --- |
| RUN-143 | RUN-158 (M1) | QTIP Viterbi quantizer (replace greedy nearest-state search) |

## Closed-as-out-of-scope (Step 4 — 2 items)

| ID | Title | Reason |
| --- | --- | --- |
| RUN-122 | Build Turbo Sparse dReLU fine-tune pipeline for MoE models | Requires 150B-token retraining; v2 is no-retraining only |
| RUN-133 | Differential Transformer compatibility layer | Requires the underlying model to be trained with Differential Transformer; no target model uses it |

Both already in closed state.

## Merged (Step 5 — 1 merge)

**RUN-163** absorbed **RUN-172**.

- **RUN-163 (renamed)**: "Port Lightning Indexer + FlashMLASparse CUDA kernel for V4 CSA/HCA top-k attention"
  - Milestone: M1 (was M1)
  - Description rewritten to absorb both the Rust scaffold dispatch wiring + the CUDA kernel port.
  - Labels: `applies-to:V4-only`, `category:sparsity`.
- **RUN-172 (cancelled)**: "Port FlashMLASparse CUDA kernel for V4 DSA/CSA top-k attention"
  - State set to `Canceled` (note: Linear's state name is `Canceled` with one L, not `Cancelled` — first attempt with `Cancelled` was a no-op).
  - Closure comment points to RUN-163.

Rationale: V4's CSA Lightning Indexer is the V4 generation of V3.2's DSA Indexer. Both use the same FlashMLASparse CUDA kernel family from SGLang. Splitting into "indexer issue" + "kernel issue" was an artifact of the prior agent's templating pass — they are one deliverable.

## Moved into existing milestones (Step 6 — 9 items)

All re-opened from Done/Canceled to **Backlog** because they're forward-looking work (not actually shipped). Standard Ships/Moves/Proves/Dependencies/Effort template applied. Labels applied.

| ID | New milestone | Labels | Title |
| --- | --- | --- | --- |
| RUN-158 | M1 | category:bug, applies-to:Universal | Debug QTIP Viterbi matmul cosine-similarity regression |
| RUN-118 | M1 | category:quant, applies-to:Universal | Verify NVFP4 weight matmul on B200 tensor cores |
| RUN-160 | M1 | category:plumbing, applies-to:Universal | Synthetic-weight load smoke tests for V4 / K2.5 / GLM-5.1 |
| RUN-127 | M5 | category:research, applies-to:Universal | Port Sarathi-Serve chunked prefill + two-batch overlap |
| RUN-125 | M5 | category:research, applies-to:Universal | Layer MagicDec long-context speculation on top of EAGLE-3 / MTP |
| RUN-132 | M5 | category:research, applies-to:Universal | Research spike: holographic prefix cache |
| RUN-131 | M5 | category:research, applies-to:Universal | Evaluate DuoAttention per-head KV differentiation |
| RUN-126 | M5 | category:research, applies-to:MoE-class | Cross-request expert affinity batching for MoE serving |
| RUN-129 | M6 | category:sparsity, applies-to:MoE-class | Route to native MoBA / NSA sparse attention for K2.6 + V3+ |

## Moved into new M6 milestone (Step 7 — 6 items)

All re-opened from Done/Canceled to **Backlog**. Template applied. `applies-to:MoE-class` + `category:plumbing` labels.

| ID | Title |
| --- | --- |
| RUN-154 | Real glm_moe_dsa.rs — GLM-5 with V3-style MLA + DSA attention |
| RUN-153 | Real kimi_k2.rs (160K vocab, 384 experts, K2-specific embedding) |
| RUN-147 | Wire GLM MoE into mistralrs-core Pipeline |
| RUN-146 | Wire Kimi K2 into mistralrs-core Pipeline |
| RUN-141 | Add GLM MoE family model loader (single file) |
| RUN-140 | Add Kimi K2 family model loader (text + vision) |

Plus RUN-129 (MoBA/NSA routing) from Step 6.9 brings M6 to 7 issues total.

Note in RUN-141 description: it's largely overlapping with RUN-147. Flagged in the description for the implementer to collapse if scope merges at code time.

## Anomalies / human-decision items

1. **Linear state name is `Canceled` (single L), not `Cancelled`.** First `save_issue` with `state="Cancelled"` was a silent no-op. Retried with `Canceled` and it succeeded. The brief used both spellings (mostly Cancelled). All state changes in this pass used `Canceled`.

2. **Most "orphan" issues were already in closed states (Done or Canceled) when this pass started.** The brief's framing ("37 LEGACY ORPHAN ISSUES untouched") was slightly misleading — they were untouched by cleanup-1's milestone/label work, but already in closed states from the prior 12-agent orchestration sprint. For shipped/duplicate/out-of-scope items the pass added closure comments without regressing the state (Done is already a closed terminal state; cancelling it would be confusing).

3. **Step 6/7 items were in Done state but the brief asks to move them into milestones with the standard forward-looking Ships/Moves/Proves template.** Interpreted this as: these items were marked Done prematurely (likely by the orchestration sprint commits) and the work isn't actually shipped. Reopened them to Backlog and applied the template. If any of these items ARE actually shipped, the user should re-close them after spot-checking.

4. **Outstanding overlap: RUN-141 vs RUN-147** (and RUN-140 vs RUN-146). The brief intentionally kept both pairs as separate tickets but flagged them as "framed differently" (outer family detection vs inner per-version dispatcher). At implementation time these may collapse into a single file each.

5. **No `category:speculative` or `category:plumbing` labels were applied to RUN-127 (Sarathi-Serve scheduler).** The brief specified `category:research` only. Surface in case the user wants speculative/plumbing tags too — Sarathi-Serve is more accurately a "serving / scheduler" item than pure research.

6. **MoE-aware research items in M5 (RUN-126, RUN-185, RUN-186, etc.)** all use `applies-to:MoE-class`. RUN-126 specifically depends on RUN-186 (cross-layer routing correlation) per the brief — that dependency was noted in the description but not added as a structured `blocks`/`blockedBy` relation. Could be surfaced if cross-issue blockers matter for ordering.

7. **Time of execution:** ~13 minutes wall clock for all 21+ closures + 15 moves + 1 milestone + 1 merge + 1 summary. The Linear MCP rate-limited zero times.

## Recommended next manual review

1. **Spot-check the 15 re-opened items** (Step 6 + Step 7). If any are actually shipped (the engineer has a feature working end-to-end), re-mark them Done. The cleanup pass assumed they're forward-looking based on the brief, but the prior 12-agent orchestration sprint may have actually shipped some of them.

2. **Review RUN-141 / RUN-147 overlap** (and RUN-140 / RUN-146). Decide whether to consolidate to one issue per model family or keep the inner/outer split.

3. **Confirm M5 size (14 items) is the desired scope** for the "research bets" lane. The brief moved a lot of items here (Sarathi, MagicDec, holographic prefix, DuoAttention, cross-request expert batching). If M5 should stay focused on the publishable-novel-direction definition from cleanup-1, some of these (Sarathi-Serve in particular — that's a port, not a research bet) might belong in a "tier 3 speed" or "serving systems" milestone.

4. **Verify the M1 gate still makes sense at 17 Backlog items.** The brief added Viterbi debug, NVFP4 verify, and smoke tests on top of the prior 14 — that's a wide M1. If the user wants a sharper M1 cutoff (e.g. "just the V4 wire-ups + day-1 playbook"), these three could move to a sibling milestone like M1.5.
