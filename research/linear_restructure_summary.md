# Arc Runway — Linear Restructure Summary

Date: 2026-05-23
Project: **Arc v2** (slug `arc-v2-5227a43a042d`, team Runcrate)

## TL;DR

Restructured the Arc v2 runway into a 5-milestone speed ladder anchored on the M1 rental gate. Created 9 taxonomy labels, 5 milestones, 18 new issues for runway items surfaced by research. Audited all 19 existing backlog items: kept 16, cancelled 3 as already-shipped (verified against `mistralrs-core/src/models/deepseek4.rs` and commit log).

## Before → After

| Metric | Before | After |
| --- | --- | --- |
| Milestones in project | 0 | **5** |
| Workspace/team labels on `applies-to:` / `category:` | 0 | **9** |
| Open backlog issues | 19 | **34** |
| Issues cancelled this pass | 0 | **3** |
| Done issues | 28 (untouched) | 28 |

## New milestones

| ID | Name | Purpose |
| --- | --- | --- |
| `38de95aa-…d936f` | M1: V4 loads & runs (today → ~1,000 tok/s) | Wire-up debt + rental gate |
| `bf549093-…4970` | M2: 1,000 → 2,000 tok/s (Tier 1 speed ship) | Validated port-only speedups |
| `4ef28a29-…b82a1` | M3: 2,000 → 2,400 tok/s (Tier 2 speed ship) | xKV + MoE-aware EAGLE Pattern-3 |
| `2e6b303c-…e719` | M4: Quality moat (SCMoE family) | Better-than-FP16 quality |
| `c59cc216-…5530` | M5: Research bets | Publishable novel directions |

## New labels (workspace-scoped on team Runcrate)

| Label | Color | Purpose |
| --- | --- | --- |
| `applies-to:V4-only` | `#1F4E96` | DeepSeek V4 / Flash / Pro |
| `applies-to:MoE-class` | `#8E4EC6` | Any MoE target |
| `applies-to:Universal` | `#4CB782` | Dense + MoE |
| `category:quant` | `#F2994A` | Weight/KV quantization |
| `category:sparsity` | `#F2C94C` | Activation / expert / KV sparsity |
| `category:speculative` | `#26B5CE` | MTP / EAGLE / routing-prefetch |
| `category:plumbing` | `#95A2B3` | Loader / dispatcher / wire-up |
| `category:bug` | `#EB5757` | Correctness regression |
| `category:research` | `#D33682` | Novel direction, publishable |

## Existing issues — milestone + label assignments

All 19 prior backlog items audited. Descriptions rewritten to the Ships/Moves/Proves/Dependencies/Effort/Labels template.

| ID | Title | Milestone | Labels |
| --- | --- | --- | --- |
| RUN-134 | Build Arc v2 benchmark suite vs SGLang | M1 | Universal, plumbing |
| RUN-136 | Single-shot hardware validation pass | M1 | Universal, plumbing |
| RUN-137 | Iterate on hardware-validation findings | M1 | Universal, plumbing |
| RUN-148 | SageAttention SM89/SM90 FFI binding | M2 | Universal, quant |
| RUN-151 | End-to-end numerical stack-composition test | M1 | Universal, plumbing |
| RUN-155 | Wire arc_engine::dsv4 into mistralrs-core forward | M1 | V4-only, plumbing |
| RUN-156 | Wire arc_engine::mtp into speculative pipeline | M1 | V4-only, speculative |
| RUN-159 | EAGLE-3 draft model loader (dense Llama) | M5 | MoE-class, speculative |
| RUN-161 | Rental day 1 — execute playbook | M1 | V4-only, plumbing |
| RUN-162 | Load real V4 compressor weights | M1 | V4-only, plumbing |
| RUN-163 | Port Lightning Indexer for V4 CSA top-k | M1 | V4-only, sparsity |
| RUN-164 | Implement V4 mHC residual | M1 | V4-only, plumbing |
| RUN-167 | V4 compress dispatch in PA + MLA cache branches | M1 | V4-only, plumbing |
| RUN-168 | Wire TD-MoE whitening into model load | M1 | MoE-class, quant |
| RUN-169 | Load V4 learned hybrid attention blend | M1 | V4-only, plumbing |
| RUN-170 | arc-cuda-graph MAX_KEEP=256 fix | M1 | Universal, bug |

## Issues cancelled (verified already shipped)

| ID | Reason |
| --- | --- |
| RUN-149 | Superseded by commit `42a6ed691` — CPU simulator parity fixed. Remaining kernel-side gap split into RUN-170. |
| RUN-165 | Already shipped: `rope_standard` + `rope_compress` HashMaps in `deepseek4.rs:1560-1636`, `compress_rope_theta` default 160000.0, per-layer dispatch lines 1631-1636, test `v4_compress_rope_theta_default_is_160000` passes. Original ticket cited 40000.0; verified actual V4 Flash value is 160000.0. |
| RUN-166 | Already loaded: `attn_sink: Option<Tensor>` field at `deepseek4.rs:615`, loader at 809-819, wired into SdpaParams::sinks at 854. Remaining (PA / MLA-cache branches) absorbed by RUN-167. |

## New issues created (18 — runway items surfaced by recent research)

### M2 — Tier 1 speed ship

| ID | Title | Labels | Effort |
| --- | --- | --- | --- |
| RUN-171 | Port TEAL FFN sparsity into Arc QTIP forward path | Universal, sparsity | M |
| RUN-172 | Port FlashMLASparse CUDA kernel for V4 DSA/CSA top-k | V4-only, sparsity | M |
| RUN-173 | Implement Speculative Routing Mode A | MoE-class, speculative | S |
| RUN-174 | Port Adaptive Top-K LExI | MoE-class, sparsity | S |

### M3 — Tier 2 speed ship

| ID | Title | Labels | Effort |
| --- | --- | --- | --- |
| RUN-175 | Port xKV cross-layer KV pooling | Universal, sparsity | M |
| RUN-176 | Implement MoE-aware EAGLE Pattern-3 | MoE-class, speculative | M + offline training |
| RUN-177 | Port GeluAndMulSparse fused kernel | Universal, sparsity | S |

### M4 — Quality moat

| ID | Title | Labels | Effort |
| --- | --- | --- | --- |
| RUN-178 | Port SCMoE (Shi et al. NeurIPS 2024) | MoE-class, research | M |
| RUN-179 | Calibrate SCMoE α + rank-K on V4 Flash | MoE-class, research | S |
| RUN-180 | Implement Gated SCMoE composite gate | MoE-class, research | S |

### M5 — Research bets

| ID | Title | Labels | Effort |
| --- | --- | --- | --- |
| RUN-181 | Distilled SCMoE corrector head | MoE-class, research | L |
| RUN-182 | Fused-kernel parallel SCMoE | MoE-class, research | M |
| RUN-183 | Implicit-contrast SCMoE (DoLa-on-MoE family) | MoE-class, research | M |
| RUN-184 | Amortized cross-token SCMoE | MoE-class, research | S |
| RUN-185 | Routing-conditional MoE neuron predictor | MoE-class, research | L |
| RUN-186 | Cross-layer routing correlation V3/V4 ablation | MoE-class, research | S |
| RUN-187 | DoLa-on-MoE quality experiment | MoE-class, research | S |
| RUN-188 | EAGLE with routing-fingerprint as draft input | MoE-class, research | L |

## Anomalies found

1. **User's "51 backlog items" count was overstated.** Actual count at the start of restructure was 19 open backlog issues. The 51 likely included the 28 Done items visible in dashboard views.
2. **RUN-165 had wrong `compress_rope_theta` value in description (40000.0).** Verified actual V4 Flash value is 160000.0 via `default_compress_rope_theta()` in `deepseek4.rs:290` and test at line 2222. Cancelled with note.
3. **Missing research doc: `research/scmoe_fused.md`.** Other four SCMoE variants (`scmoe_amortized.md`, `scmoe_distilled.md`, `scmoe_gated.md`, `scmoe_implicit.md`) are present. RUN-182 (Fused-kernel parallel SCMoE) flags this in its body — issue still tracked so it doesn't fall out of memory.
4. **Naming overlap: "Lightning Indexer" (RUN-163) and a possible future "DSA Indexer" issue refer to the same mechanism.** RUN-163's description was updated to flag this explicitly.
5. **`V4Indexer` struct already exists in Rust.** RUN-163 was originally framed as "port from scratch"; verified `mistralrs-core/src/models/dsv4_indexer.rs` already has the Rust port. RUN-163's "Ships" section was reframed to "actually USE it inside CSA forward to drop FLOPs" instead.
6. **mHC infrastructure is partly shipped.** Config fields + per-layer params loaded + 4-D residual stack threading scaffold exists in `deepseek4.rs:1303-1422`. RUN-164's "Ships" section was reframed to focus on the actual sinkhorn-normalized mixture math that remains.
7. **RUN-159 (vanilla EAGLE-3 on Llama)** was moved from M1 to M5 with a reframe note — it's potentially superseded by the new MoE-aware EAGLE Pattern-3 (RUN-176). Kept open with explicit "close if M3 ships and dense fallback isn't needed for the moat" guidance.
8. **RUN-135 (build-complete gate)** — already Done. Not touched.
9. **B200 framing in RUN-136/137** — descoped to single H100/H200 per the rental cost-realism note in M1's gate. V4 Pro / B200 deferred to a future milestone (not created yet — surface to user if needed).

## Rental playbook — M1 as the gate

Everything in the runway is gated by **M1 closing**. Specifically: V4 Flash must decode coherent tokens end-to-end on a real H100/H200 with all three forward paths (plain SDPA, PagedAttention, MLA-cache) routing through the V4 compress dispatch (RUN-155 + RUN-167), real compressor weights loaded (RUN-162), Lightning Indexer wired (RUN-163), mHC residual implemented (RUN-164), MTP head wired into the speculative pipeline (RUN-156), TD-MoE whitening wired into load (RUN-168), and the `arc-cuda-graph` `MAX_KEEP=256` bug fixed (RUN-170). RUN-161 then executes the rental playbook to capture the baseline ~1,000 tok/s decode number; RUN-134/136/137 produce the structured findings report and fix loop. Until M1 holds (numerical stack-composition test RUN-151 green), no M2/M3 speed work makes sense — there's nothing to compare against. After M1 holds, the speed ladder (M2 → M3) and quality moat (M4) graduate to ship-gates of their own, while M5 stays open as the publishable research surface area.
