# Wave 29 — BD: should V4-Flash be re-baked on the `qtip2b` rung?

**Scope.** Decision document only. No kernel touched, no GPU rented, nothing
baked. Base `origin/master` @ `57bd1ba70`. Every claim below is **CONFIRMED**
(source read, `file:line` given), **MEASURED** (an existing hardware log, cited),
**DERIVED** (arithmetic shown from confirmed inputs), or **UNKNOWN**.

---

## 0. VERDICT

> ## 🟡 DO NOT RE-BAKE YET — BLOCKED ON THREE MEASUREMENTS, TWO OF WHICH COST ~$1 EACH.
>
> **The rung is right. The kernel is live. D4 is clean. The *evidence* is not
> there, and one existing hardware log argues the payoff at the batch we can
> actually schedule is ~1.1×, not 4–8×.**

Broken out:

| question | answer |
|---|---|
| Is `qtip_grouped_gemm.cu` reachable from serving decode on a `qtip2b` artifact? | ✅ **YES — CONFIRMED, live by default, no flag needed.** Not a tenth dead-code case. |
| Does it amortize the way the fleet thesis needs? | ✅ **YES structurally, CONFIRMED + MEASURED.** ⚠️ But it starts from a **measured 1.76× per-byte deficit** vs the GEMV it replaces, which eats most of the win below B≈52. |
| D4 / Greedy on the `qtip2b` rung? | ✅ **CLEAN.** Viterbi-**exhaustive** + hadamard-128, stamped and enforced at load. Not beam-256 — this rung has no beam, and exhaustive is the *better* search (wave19-AP). 🟡 One gap: qtip2b bakes log no header (D4 clause 4). |
| Quality parity qtip2 vs qtip2b? | ❌ **UNMEASURED on the fixture that matters.** One Gaussian head-to-head inside its own noise. Zero fp4_dequant comparison, zero PPL, zero GSM8K, zero served qtip2b artifact — ever. **This is the blocker.** |
| Artifact / format consequences? | ✅ **Zero technical risk.** 🟡 High *surface* cost (repo name, every shard path, PR #55's whole model card). |
| Bake cost? | ❌ **UNKNOWN**, different kernel, no per-layer datapoint anywhere. Code-derived parallelism deficit says "slower", magnitude undetermined across an order of magnitude. |

**Do this instead of a bake:** three gates below, ≈$2 and one CPU test session.
Two of them close questions that a bake would otherwise answer *during* a
multi-hour paid run — which is precisely the D4b / D10 failure mode.

---

## 1. Is the grouped kernel LIVE on the serving decode path? — **CONFIRMED, traced**

The repo has nine "wired but never invoked" cases in BACKLOG, so this was traced
call-by-call rather than inferred from the file existing.

**The chain, for a `--from-uqff` serve of a `qtip2b` artifact on CUDA:**

| # | site | what it establishes |
|---|---|---|
| 1 | `mistralrs-core/src/pipeline/normal.rs:615` — `loading_isq \|= self.config.from_uqff.is_some();` | serving from UQFF sets `loading_isq = true`; the `from_uqff` arm at `:672-686` deliberately does **not** clear it |
| 2 | `mistralrs-core/src/moe/experts.rs:82-112` — `MoEExpertsBackend::select` | `is_cuda && loading_isq` ⇒ `Fast`. Not `Fused` (that needs no ISQ), not `Slow` |
| 3 | `moe/experts.rs:457-466` | `Fast` ⇒ `forward_fast` |
| 4 | `moe/experts.rs:583-588`, `:610-612` | three `gather_forward_autocast` calls per MoE layer (gate, up, down) |
| 5 | `mistralrs-quant/src/lib.rs:1402-1410` | `gather_forward_autocast` ⇒ `QuantMethod::gather_forward` |
| 6 | `mistralrs-core/src/pipeline/isq.rs:1485`, `:1596` | `QuantizedSerdeType::Qtip2b ⇒ Qtip2bLayer::deserialize` — a qtip2b artifact **is** a `Qtip2bLayer` |
| 7 | `mistralrs-quant/src/qtip/bitshift.rs:1461` | `Qtip2bLayer::gather_forward` (the trait impl) |
| 8 | `bitshift.rs:1487-1497` | decode cap: `n_tokens <= 8` ⇒ per-pair on-device GEMV. **Above 8 it falls through** |
| 9 | `bitshift.rs:1505-1526` | grouped branch: `!ARC_NO_QTIP_GROUPED_MOE` (unset by default) ∧ BF16/F16 ∧ `in_features % 64 == 0` ∧ uniform-2-bit table ⇒ `gather_forward_batched` |
| 10 | `bitshift.rs:1236`, CUDA gate `:1265-1291`, call `:1292` | ⇒ `cuda_ops::grouped_gemm_2b_cuda` |
| 11 | `mistralrs-quant/src/qtip/cuda_ops.rs:1689`, `:1823-1841`, `:1876-1877` | ⇒ `launch_qtip2b_moe_route` + `launch_qtip2b_grouped_gemm_bf16` |
| 12 | `mistralrs-quant/build.rs:85` `source_glob("kernels/*/*.cu")`, `:100-113` `has_qtip_kernels` for compute cap ≥ 80 | the `.cu` **is compiled in** on any H100/H200 build |

**Every gate is satisfied by V4-Flash as configured** (CONFIRMED):
`hidden_size = 4096`, `moe_intermediate_size = 2048`, 43 MoE layers,
`n_routed_experts = 256`, **top-6** — `models/deepseek4.rs:30-31`,
`research/v4_audit.md:21-22, 263-264, 446-447`. So gate/up are
`[256, 2048, 4096]` (`in_features = 4096`, `4096 % 64 == 0` ✅) and down is
`[256, 4096, 2048]` (`2048 % 64 == 0` ✅).
`expert_bpw` is reconstructed as `uniform_2bit` at deserialize
(`bitshift.rs:1442`, `:1712`) ✅. `num_experts` is recovered from
`blocks.rank() == 3` ✅. `rotation_signs` round-trip through UQFF ✅.

**No flag, env var, or feature is required.** The three env vars in the area
(`ARC_NO_QTIP_GROUPED_MOE`, `ARC_NO_QTIP_ONDEVICE_MOE`,
`ARC_QTIP_ONDEVICE_MOE_MAX_TOKENS`) all *disable or narrow* the path; unset is
the fast configuration.

**Two independent corroborations that this is the live decode path, not a
plausible-looking one:**

1. **The 8-token cap has already been observed as a step-time cliff on
   hardware.** wave27-AY / FACTS §"THE REAL CAUSE": per-sequence step cost
   65.1 / 67.4 / 97.0 ms below the cap → **197.2 ms above it**, crossing between
   B=16 (8 running) and B=32 (13 running). A cap inside
   `QuantMethod::gather_forward` cannot produce a measured cliff unless that
   function is on the hot path.
2. **The grouped kernel itself has run on hardware through this exact public
   dispatch** — `mistralrs-quant/examples/qtip_grouped_curve.rs:87` drives
   `layer.gather_forward(...)`, and its s4 output is in
   `mission/gpu-run4-results/results/grouped_curve.txt`. Grouped-GEMM parity
   5/5 (s3), `mission/gpu-run3-results/results/grouped_parity.txt`.

**Not intercepted upstream.** `arc-cuda-graph`'s dedicated decode path is a
**dense** layout — `projs_per_layer = 7` (q,k,v,o,gate,up,down) at
`arc-cuda-graph/src/weights.rs:230-273` — with no expert dimension and no
router. It cannot represent a 256-expert MoE, so V4 decode runs Candle's
`forward_inputs` (`pipeline/mod.rs:575-592`) and reaches the layer as traced.
FACTS separately records that `supports_paged_attention=false` on V4 already
kills the autonomous-decode path (`normal.rs:1841`).

---

## 2. Does it amortize the way we need? — **YES structurally. ⚠️ measured efficiency is the catch.**

### 2a. Grouping — CONFIRMED

`kernels/qtip/qtip_grouped_gemm.cu` is a real group-by-expert kernel:
histogram → exclusive scans + ragged tile map → grouped scatter
(`:185-260`), then a persistent CTA loop over the flattened (m-tile × n-tile)
domain (`:338-353`). Zero host syncs anywhere.

Granularity: `QG_TILE_M = 16` pairs per m-tile, `QG_TILE_N = 64` weight rows per
n-tile (`:96-98`). Each expert's *full* weight matrix is staged once per m-tile
(across all n-tiles), so:

```
weight bytes moved  =  Σ_e ceil(count_e / 16) × (one expert's packed bytes)
```

versus the per-pair gather GEMV's `n_pairs` expert reads. **When the average
pairs-per-active-expert is ≤ 16, that sum equals E(B) exactly** — the
theoretical optimum, i.e. each woken expert read once per step. For V4
(top-6, E=256) that holds up to **B ≈ 683**; the asymptotic ceiling beyond that
is 16× (TILE_M), above the 8× the fleet thesis asks for. **DERIVED.**

### 2b. Token ceiling — **NONE. CONFIRMED.**

- Grid is **1-D**: `min(max_m_tiles × n_tiles_n, QG_MAX_GRID = 1024)` persistent
  CTAs that grid-stride the live tile count (`:111`, `:498-503`, `:344`).
- `n_pairs` is `int`; routing grid is `ceil(n_pairs / QG_ROUTE_THREADS)` on
  `grid.x` (`:472-479`).
- ⇒ **the `grid.y ≤ 65535` limit that bounds the gather-GEMV path (wave28-AZ §2)
  does not exist here.** No degradation, no fallback, no silent-zeros mode
  above any batch size.

### 2c. ⚠️ The measured catch — the grouped kernel is **1.76× less bandwidth-efficient than the GEMV it replaces**

From `mission/gpu-run4-results/results/grouped_curve.txt` (s4, H200, real V4
shapes N=2048/K=4096, **E=64**, top-6) — **MEASURED**:

| B | pairs | distinct experts | grouped µs/call | gemv µs/call |
|---|---|---|---|---|
| 8 | 48 | 48 | **403.8** (251 GB/s, 5.2 % peak) | **228.9** (443 GB/s, 9.2 % peak) |
| 16 | 96 | 64 | 532.0 (254 GB/s) | — (capped) |
| 32 | 192 | 64 | 530.0 (255 GB/s) | — |
| 64 | 384 | 64 | 532.9 (254 GB/s) | — |

The **flat 532 µs across a 4× increase in pairs is the amortization working,
measured** — with E=64, expert coverage saturates by B≈11, so weight traffic is
constant from B=16 up and so is the time. That is the mechanism, demonstrated on
hardware. This is the honest content of the "1,006 aggregate tok/s" headline
(`FACTS.md:397-399`, `STATUS.md:462`).

**But the B=8 row is apples-to-apples — same batch, same 48 distinct experts,
same bytes — and the grouped kernel is 1.76× SLOWER.** 8.41 µs per distinct
expert vs 4.77 µs per pair. The grouped kernel runs at 5.3 % of H200 peak
bandwidth; the GEMV at 9.2 %. Amortization has to pay off that handicap first.

⚠️ **The published `E(B)` amortization table (FACTS, and the brief for this
task) is for top-8. V4-Flash is top-6** (`v4_audit.md:263-264`). Corrected:

| B | E(B), top-6 of 256 | pairs/E(B) = ideal amortization |
|---|---|---|
| 16 | 80.7 | 1.19× |
| 32 | 136.1 | 1.41× |
| 64 | 199.9 | 1.92× |
| 128 | 243.7 | **3.15×** (not 4.07×) |
| 256 | 255.4 | **6.01×** (not 8.00×) |

**DERIVED projection to V4's real routing** (E=256, top-6, 43 MoE layers × 3
calls, using the two measured per-unit costs above; MoE-GEMM floor only — no
attention, KV, sampling, or launch overhead):

| B | grouped step | GEMV step | grouped agg tok/s | GEMV agg tok/s | winner |
|---|---|---|---|---|---|
| 16 | 86.5 ms | 59.1 ms | 185 | 271 | GEMV 1.46× |
| 32 | 145.9 ms | 118.1 ms | 219 | 271 | GEMV 1.24× |
| **52** | ~176 ms | ~192 ms | ~295 | 271 | **crossover** |
| 64 | 214.3 ms | 236.3 ms | 299 | 271 | grouped 1.10× |
| 128 | 261.3 ms | 472.5 ms | 490 | 271 | grouped 1.81× |
| 256 | 273.8 ms | 945 ms | 935 | 271 | grouped 3.45× |

Two readings, both true and both load-bearing:

- 🔴 **Against.** An *uncapped* per-pair GEMV gives a **flat** aggregate (~271
  tok/s at the MoE floor) — it never falls. The grouped kernel only overtakes it
  around **B≈52**, and FACTS records that memory caps us near **B≈68** at
  2048-token context until the `xs` compressor cache shrinks (the BACKLOG
  keystone). **At the batch we can actually schedule today, the grouped kernel
  is worth ~1.1×, not 4–8×.** PR #56 raises the LUT rung's cap to ~682 tokens
  and would capture most of that flat 271 with no re-bake at all.
- 🟢 **For.** The GEMV curve is *flat forever* by construction (cost linear in
  pairs). Only the grouped kernel *rises*. Every multiple above 1× that the
  fleet thesis promises lives on this rung and nowhere else, and the 1.76×
  handicap is a tuning number, not a structural one — `qtip_grouped_gemm.cu:63-79`
  lists an entire unswept axis set (TILE_M 16→32/48, TILE_N 64→128, cp.async
  depth 2→4, k-fragments in flight). **UNMEASURED** — but the LUT rung's GEMV
  went 153→467 GB/s (3.2×) on exactly this kind of sweep (FACTS §Speed).

---

## 3. 🔴 D4 COMPLIANCE — **PASS, with one clause-4 gap**

**Greedy is unreachable on the `qtip2b` rung. CONFIRMED.**

- `mistralrs-core/src/pipeline/isq.rs:232` `"qtip2b" ⇒ IsqType::Qtip2b`
- `mistralrs-quant/src/unquantized/mod.rs:438-460` — the `Qtip2b` arm.
  `:452` `let mode = crate::QtipMode::default_expert_mode();`
- `mistralrs-quant/src/qtip/mod.rs:594-596` — `default_expert_mode()` is a
  **`const fn` returning `Viterbi`**. A `const fn` structurally cannot read an
  env var. This is the fix for the exact defect D4 records; the comment at
  `unquantized/mod.rs:444-451` names it.
- Two independent doors: `bitshift.rs:364` `mode.deny_greedy(...)` and
  `bitshift.rs:422` `deny_greedy_outside_tests` — both **hard-error**, not warn.
- `ARC_QTIP_EXPERT_GREEDY` is read **nowhere** in the repo. `ARC_QTIP_EXPERT_VITERBI`
  likewise. CONFIRMED by grep over the tree excluding `target/`.
- **Rotation is on:** `QtipRotation::for_mode(Viterbi).enabled()` ⇒ `On`
  (`mod.rs:805`), `rotation_block_size(4096) == 128` (`mod.rs:292-301`, pinned
  by a test at `mod.rs:4758`) ⇒ **hadamard-128**, shared across the expert
  stack (`bitshift.rs:730-736`).
- **Stamped and enforced at load, both rungs:** write `bitshift.rs:1765-1779` /
  `mod.rs:3638-3656`; check `bitshift.rs:1582` `enforce_at_load("qtip2b-layer")`
  / `mod.rs:3446`; policy `mod.rs:905-933` — a Greedy stamp is a hard bail with
  **no override**.
- `greedy_ban_tests.rs` (998 lines) is **not vacuous**: every test is
  `Device::Cpu`, no CUDA gate, no early-return-on-no-GPU anywhere in the file.
  `:131-201` `isq_dispatch_never_bakes_greedy_on_either_rung_or_rank` loops
  `[QtipBitshift2, Qtip2b] × [2-D, 3-D]` through the real `apply_isq` and
  asserts on the **serialized bytes** (stamp == Trellis, flags == 0x00,
  reload passes the gate, `rotation_block >= 2`).

### 🔵 It is not beam W=256 — it is EXHAUSTIVE, and that is *better*

The brief asks whether a qtip2b bake takes "Viterbi with beam W=256". It does
not, because **the `qtip2b` rung has no beam kernel at all** — `wave13-AF-cuda-beam.md:230`
("its prefix count is 2^14, so the same design needs a 128 KiB table"), and
`cuda_ops::quantize_rows_2b_cuda` has **no `search` parameter**
(`bitshift.rs:617`, `cuda_ops.rs:1499-1503`). `ARC_QTIP_BEAM` is not referenced
in `bitshift.rs`. It runs the full 2^16-state DP (`Q2B_NSTATES = 1<<16`,
`cuda_ops.rs:1100`, `:1588-1642`) and honestly stamps `EXHAUSTIVE_MSE`
(`bitshift.rs:568`, `:637`, `:760`).

Per wave19-AP §4b, **exhaustive BEATS beam-256 in 8/9 fixture cells**, including
`+0.0013…+0.0021` cos on `fp4_dequant`. And the *published* artifact was baked
at **beam W=32** (FACTS §"FULL-SERVING THROUGHPUT": "in-situ qtip2 W=32 bake";
`mod.rs:942` records W=64 at 0.95054 vs exhaustive 0.96495 on FP4-lattice
fixtures). **On the search axis alone, a qtip2b bake is an upgrade, not a
downgrade.** The codebook axis (§4) is the unknown, not the search axis.

### 🟡 GAP — D4 clause 4 is not satisfied on this rung

"Every bake logs mode/search/objective/rotation" — `log_bake_header` /
`bake_header_line` are called **only** from `mod.rs:1486` and `mod.rs:1702`,
both LUT rung. **A `--isq qtip2b` bake prints nothing about its own search.**
The artifact *is* stamped and the stamp *is* enforced, so this is not the
invisible-Greedy failure mode — but D4 exists because "nothing recorded which
search produced an artifact" three times. **Land the header before any qtip2b
bake.** Small, mechanical, ~1 agent-hour.

### 🟡 Three stale records that actively mislead an operator

- `bitshift.rs:275` still says *"3-D expert stacks default to greedy"*.
- `docs/notes/v4-reference-audit.md:1296, 1317` and
  `arc-tools/quality/GPU_SESSION_RUNBOOK_2.md:48, 201, 546` still tell an
  operator `ARC_QTIP_EXPERT_GREEDY=1` reverts to greedy. It does nothing.
- Four `mistralrs-quant/examples/*.rs` construct `QtipMode::Greedy` and now
  **hard-error at runtime** — including `qtip_grouped_curve.rs:129`, the very
  harness Gate 2 below needs. They compile (CI clippy runs `--examples`) and
  fail only when run, which is why the rot is invisible.

---

## 4. 🔴 Quality parity — **THE BLOCKER. It has never been measured where it counts.**

### What exists

**One designed head-to-head**, `bitshift.rs:2232-2296`
`matmul_cosine_meets_lut_rung_bar_on_gaussian_fixture` — both rungs from the
identical weight tensor, both Viterbi + rotation, matmul cos vs dense:

| seed | qtip2b | qtip2 (LUT) |
|---|---|---|
| 0 | 0.9581 | 0.9676 |
| 555000 | 0.9645 | 0.9593 |
| 42000000 | 0.9737 | 0.9703 |
| **mean** | **0.9654** | **0.9657** |

(raw: `mission/gpu-run2-results/results/qtip2b_parity.txt`; also wave2-F §.)

**Fixture: pure Gaussian, `n=32, k_in=256`, three seeds** (`bitshift.rs:1848-1861`,
`:2234-2244`). The test's own docstring puts per-fixture noise at ±0.008 — the
0.0003 mean gap is **~25× inside its own noise bar**. "Indistinguishable on
Gaussian" is what this supports. Nothing more.

`docs/RELEASE_NOTES_v2.0.md:31` states this publicly as *"quality tied with the
LUT rung (cos 0.9654 vs 0.9657)"* **with no fixture disclosure** — itself a D12
problem in the public record, independent of this decision.

### What does not exist

- ❌ **No qtip2 vs qtip2b comparison on `fp4_dequant` has ever been run.**
  `gen_fp4_dequant` never appears in `bitshift.rs`. D12 says in as many words
  that this is the fixture family that decides quantization questions for V4's
  experts, and it is the exact fixture on which the two rungs' *shared* greedy
  failure mode diverged by 0.29 cos.
- 🟡 The nearest thing is a pairing **I assembled, that nobody ran as one
  experiment**: `bake_quality_tests.rs:497-539` (LUT) and `:654-707` (qtip2b)
  use a byte-identical fixture line `gen_fp4_dequant(e*n, k, 0.02, 42)` at
  `(E,N,K)=(2,16,512)` — wave3-G reports **0.963** for qtip2, wave13-AG §2
  reports **0.9623** for qtip2b. Different agents, different commits, different
  entry points, gap again inside noise. **INFERRED, not a measurement.**
- ❌ **No model-level quality number for qtip2b exists at all.** No PPL, no
  GSM8K, no served artifact, ever. Every published quality number
  — PPL 12.50 ± 3.46, GSM8K 87.0 % — is `--isq qtip2`
  (`gpu-run2-results/results/ppl_qtip2_c1024.json`; FACTS §Quality).
  `wave9-V-docs.md:66-67` records the explicit correction that the claim
  *"v2.0 artifacts use qtip2b"* was **wrong**.
- 🟡 The **"20/20 CUDA parity"** headline overstates by ~5×: the log is a
  filtered run of the whole `qtip::bitshift::tests` module; **4** of the 20 are
  CUDA↔CPU parity tests. "Grouped-GEMM 5/5" is 3 CPU-side + 2 CUDA tests.
  **All of it is kernel-vs-CPU-reference numerical agreement — none of it
  measures compression fidelity vs dense BF16.** Do not let it stand in for
  quality; `docs/FLEET.md:88` and `RELEASE_NOTES_v2.0.md:79` currently do.

### Why the two rungs could genuinely differ

They are different codebooks at the same 2.00 bits/weight (both pack 4 weights
per byte): qtip2 is **K=4 / L=16 / V=2** with a 512 KiB computed-Gaussian LUT
(`mod.rs:356-358`); qtip2b is **MCG K=2 / L=16 / V=1** (`bitshift.rs:99-101`).
Different lattice geometry, different scale policy (`max|row|/3.0` vs
`/3.62`). There is no theoretical reason to expect them equal on a heavy-tailed
FP4-lattice source, and the one thing measured on such a source (wave3-G's
ladder) shows this weight family is exactly where rung differences show up.

**An unmeasured quality claim is not a claim.** Publishing a customer-facing
artifact on a rung with zero model-level evidence is a D9 violation regardless
of how good the throughput looks.

---

## 5. Artifact / format consequences — **technically free, editorially expensive**

**CONFIRMED — no format risk:**

- Rung tag is `QuantizedSerdeType::Qtip = 8` vs `Qtip2b = 10`
  (`mistralrs-quant/src/lib.rs:1255,1258`), read at a fixed offset
  (`utils/uqff.rs:36`) and dispatched at `isq.rs:1479-1489`, `:1590-1600`,
  `distributed/layers.rs:338-341, 690-693, 1009-1012`. Each deserializer
  **re-checks its own tag and bails** (`mod.rs:3467-3472`,
  `bitshift.rs:1601-1606`). **Cross-rung misload is impossible in either
  direction.**
- **UQFF version stays 0.3.0** — tag 10 already exists. No format bump.
- **Overlay semantics unchanged.** The residual (`isq.rs:1126-1149`) is
  embeddings + norms only, 1,293,806,700 B = 1.7 % of the repo, and is written
  above the layer serde — entirely rung-independent. A qtip2b artifact is
  equally an overlay on the source checkpoint at `-m`.
- **Shard count unchanged: 8.** Sharding is size-driven
  (`MAX_UQFF_SIZE_BYTES = 10 GiB`, `isq.rs:158-159`), not rung-driven.
- **Bits/param essentially identical.** Both are exactly 2.00 bits/weight in
  the block payload with identical F32 row scales and rotation signs. The only
  delta is qtip2's 512 KiB LUT written per serialized layer entry
  (`mod.rs:3616` — qtip2b stores a 4-byte MCG multiplier instead): ≤ 517 entries
  × 524,306 B ≈ **271 MB, 0.37 %**. Not user-visible.

**⚠️ Premise correction carried forward:** the published artifact is
**74,190,197,268 B (74.19 GB) in 8 shards + residual**, not 68 GB — the 68 GB /
7-shard figure is explicitly retracted at `FACTS.md:195-198`. Consequently
bits/param is **2.09, not 1.9** (74.19e9 × 8 / 284e9). The "≈1.9 bits/param"
line survives in `FACTS.md:188` and in PR #55's model card (lines 5, 246) and is
stale **today, on the current artifact** — worth fixing regardless of this
decision.

**🟡 The real cost is editorial, and it is not small:**

- Directory-output bakes auto-name shards from `Display for IsqType`
  (`mistralrs-cli/src/commands/quantize.rs:140`, `lib.rs:1028-1029`) ⇒
  **`qtip2b-0.uqff … qtip2b-7.uqff`**. Every `--from-uqff` path in every doc
  changes.
- The HF repo name `DeepSeek-V4-Flash-UQFF-qtip2` becomes wrong.
- **PR #55's 265-line model card is hardcoded to `qtip2` throughout** — title,
  repo, invocation lines 50/59, shard auto-discovery text 63-65, the byte-exact
  per-file table at 106-113, the `| Method |` row at 138, provenance 240-249.
  A qtip2b bake invalidates all of it including every per-file byte count.
- Plus `docs/BENCHMARKS.md:15,58,64,65,165,292`, `docs/FLEET.md:18`,
  `docs/PEAK_INFERENCE.md:71`, `docs/RELEASE_NOTES_v2.0.md:20,125,167,173`,
  `docs/notes/release-checklist.md:110`.
- **CLI aliases (CONFIRMED, `isq.rs:231-232`):** `qtip2` rung accepts exactly
  `qtip2` and `qtip`; the bitshift rung accepts exactly `qtip2b`. Case-insensitive.
  Note `qtip2b` **rejects imatrix outright** (`unquantized/mod.rs:440-442`)
  where qtip2 accepts it as a Hessian diagonal under `ARC_QTIP_HESSIAN=1` — a
  capability the artifact loses, currently unused.

---

## 6. Bake cost — **UNKNOWN, and the code says "probably worse"**

**There is no recorded per-layer qtip2b bake time anywhere** — not in the repo,
not in `mission/`, not in any GPU-run results directory. `STATUS.md:674` still
lists "qtip2b bake" as a *planned* session TODO that never ran. **UNKNOWN.**

**The rungs do not share a quantize kernel. CONFIRMED:**

| | qtip2 (LUT) | qtip2b (bitshift) |
|---|---|---|
| CUDA quantize | `qtip_beam.cu` + `qtip_quantize.cu`, FFI `ffi.rs:96,107,139` | `qtip_bitshift.cu:288,368`, FFI `ffi.rs:438,449` |
| Rust entry | `quantize_rows_cuda(w, lut, mode, search)` | `quantize_rows_2b_cuda(w, mcg_mult, mode)` — **no `search` arg** |
| Beam available | yes, W ≤ 256 | **no** |
| Shipped V4 bake | **beam W=32** | would be **exhaustive** |

So the anchors in the brief do not transfer: **W=256 82.7 s/layer vs W=32
83.6 s/layer is a LUT-rung number and is irrelevant here** — that rung's bake is
LUT-gather-bound, not search-bound, which is exactly why beam width barely moves
it. qtip2b has no LUT to gather (computed codebook, the thing PR #46 is trying
to bring *to* the LUT rung) but pays full exhaustive DP.

**The one thing derivable without a GPU is a parallelism deficit. DERIVED from
`cuda_ops.rs:782, 790-797, 986-987, 1594-1595`, 6 GiB scratch budget:**

| bake | per-row scratch (gate/up, K=4096) | rows in flight | n_rows |
|---|---|---|---|
| qtip2 beam W=32 (**what shipped**) | 2048 × 32 × 4 = **262 KB** | 2048 (capped by n_rows) — **one launch** | 2048 |
| qtip2 exhaustive | 2048×4096 + 512 KB = **8.91 MB** | 673 | 2048 |
| **qtip2b exhaustive** | 4096×16384 + 576 KB = **64.6 MB** | **95** | 2048 |

**≈21× fewer rows in flight than the bake that produced the published
artifact**, on top of ~2× the timesteps (V=1 ⇒ `num_symbols = k_in`). It will
not OOM — `rows_in_flight` is derived *from* the budget, so the cost is serialized
launches, i.e. **time**. Direction: slower. **Magnitude: genuinely UNKNOWN, and
my own two arithmetic models for it disagree by an order of magnitude** (L2
residency of the 512 KB ping-pong cost tables decides it, and that is not
derivable from source). Reporting the spread rather than picking a number.

**Mitigations already in-code:** `ARC_VITERBI_SCRATCH_GB` (`cuda_ops.rs:790-797`)
raises the budget — documented bit-identical, and on a 141 GB H200 pre-bake
there is room for 40–60 GB ⇒ 600–900 rows in flight. Multi-GPU bake is merged
and **rung-agnostic** (`resolve_bake_devices` in `isq.rs`; wave22-AS:150
measured **1.97× on 2 GPUs, byte-identical**) — halves wall clock at the same
total $.

**Estimate, stated as a range because that is what the evidence supports:**

- Reference point: the published bake was **43 layers @ 370–376 s/layer on a
  $1.49/hr A100 ≈ 4.5 h ≈ $6.7** (`FACTS.md:186-189`). On H200 the LUT rung
  runs ~83 s/layer ⇒ ~1 h of bake.
- qtip2b at **1×–10×** the LUT per-layer time ⇒ **1–10 h of bake**.
- All-in on one H200 @ $4.85/hr, adding ~70 min of boot + 149 GB pull + build
  (D4b's own figure) and ~30 min upload: **$12 at the low end, $60+ at the
  high end.** The budget is $49.97 ≈ 10 H200-hr. **The upper end consumes the
  entire remaining budget on a bake whose per-layer cost was never measured.**
- With 2 GPUs: same dollars, ~half the wall clock, and the D10b "silence is not
  success" risk halves with it.

**⇒ This alone justifies Gate 3 below. A ~$1 single-layer timing converts a
$12–$60 unknown into a budgeted decision.**

---

## 7. THE PLAN — three gates, ≈$2, before any bake is authorised

### Gate 1 — quality parity on the fixture that decides it. **No GPU. Blocking.**
Run qtip2 (beam-256 **and** exhaustive) vs qtip2b (exhaustive), both
Viterbi + hadamard-128, across the **fixture family**: `gaussian` /
`student_t4` / **`fp4_dequant`** × realistic dispersions, at V4's real expert
shape `(N=2048, K=4096)` — the wave13-AD `probe_rotation_vs_hessian_sensitivity`
pattern (3 fixtures × 5 dispersions), not one fixture.
**Abort-if:** qtip2b loses more than the fixture's own noise bar on
`fp4_dequant`. Then the answer to this whole document is **NO**, and the LUT
rung needs its own grouped kernel (wave28-AZ §4 has the spec).
*This is a CPU test in `mistralrs-quant`. It has no dependency on anything else
here and should be written first.*

### Gate 2 — 🔑 does the amortization survive V4's real routing? **≈$1 on an A30. Blocking.**
Repair `mistralrs-quant/examples/qtip_grouped_curve.rs` — it currently
hard-errors at line 129 on the banned `QtipMode::Greedy`, so **the harness that
produced the 1,006 tok/s headline cannot be run today.** Switch it to Viterbi
(required by D4 anyway), then change **`E = 64 → 256`** and sweep
**B ∈ {1, 8, 16, 32, 52, 64, 128}** in both modes.

**This is the one measurement that confirms or refutes the amortization.**
Predictions from §2c, stated in advance so the run can falsify them:

- grouped µs/call rises with `E(B)` and **flattens only near B≈256**, not B≈16;
- grouped **crosses** the per-pair GEMV at **B ≈ 52 ± 15**;
- at B=64 grouped is **≈1.1×** the GEMV; at B=128, **≈1.8×**.

**REFUTED if** grouped never overtakes the GEMV below B=68 (the memory-feasible
ceiling until the `xs` cache shrinks). In that case the honest recommendation is
**do not re-bake — take PR #56's raised LUT-rung cap instead**, and spend the
GPU hour on the `qtip_grouped_gemm.cu` tuning axes (`:63-79`) that would close
the 1.76 × handicap, since the whole thesis rests on it.

### Gate 3 — first qtip2b per-layer bake datapoint. **≈$1, same session. Blocking.**
Time **one** `[256, 2048, 4096]` expert-stack quantize on the qtip2b rung, with
and without `ARC_VITERBI_SCRATCH_GB` raised. Multiply by 43 layers × 3 matrices
for the bake budget. Do this **before** renting a box for the bake, not during.

### Then, and only then
Land the D4-clause-4 bake header for qtip2b (§3), fix the three stale Greedy
records, and bake — 2 GPUs, `ARC_VITERBI_SCRATCH_GB` tuned, `ARC_QTIP_EXPERT_BATCH`
set from Gate 3's memory reading, with PR #55's model card forked to the new
rung and repo name.

---

## 8. What would have to be true for this recommendation to be wrong

Listed so a later reader can check them rather than re-derive them.

1. **If the s4 `grouped_curve.txt` B=8 comparison is not apples-to-apples**, the
   1.76× handicap evaporates and the crossover moves down toward B≈16, making
   qtip2b a clear immediate win. *I believe it is fair — same B, same 48 distinct
   experts, same bytes, both driven through the same public `gather_forward` —
   but both rows came from a Greedy-quantized layer (the example's fixture),
   which skips the rotation kernel on the GEMV path and could bias it. Gate 2
   re-runs both under Viterbi and settles it.*
2. **If the grouped kernel's untuned axes are worth ≥1.8×**, it beats the GEMV
   at every batch and Gate 2's crossover prediction is far too pessimistic. The
   LUT GEMV's own 3.2× tuning gain says this is plausible. UNMEASURED.
3. **If `xs`-cache compression lands and B=128–256 becomes schedulable**, §2c's
   right-hand columns dominate everything else here and the answer flips to an
   unqualified yes. FACTS already names that cache as the keystone.
4. **If Gate 1 shows qtip2b materially better on `fp4_dequant`** (plausible —
   it takes exhaustive where the shipped artifact took beam W=32, and
   `mod.rs:942` puts that gap at 0.951 → 0.965), the re-bake becomes a quality
   upgrade as well as a throughput one, and the editorial cost of §5 is easily
   justified.
5. **If real V4 routing is much less uniform than `E(B)` assumes** — V4 has
   3 hash-routed layers (`num_hash_layers=3`, fixed token-id → expert lookup)
   and learned routing has locality — then `E(B)` overstates the woken-expert
   count, which *helps* the grouped kernel (fewer distinct experts per step)
   and hurts nothing. FACTS already flags uniform routing as an unstated
   assumption. Worth measuring from a real trace; nobody has.
6. **If the dedicated-decode or autonomous path is ever extended to MoE**, §1's
   trace stops being the live path and this analysis needs redoing.

---

## 9. Surfaced, not shipped

> **Noticed:** `mistralrs-quant/examples/{qtip_grouped_curve, qtip_gemv_bw,
> qtip_gather_check, qtip_predict}.rs` all construct `QtipMode::Greedy` and
> hard-error at runtime post-D4. They compile (CI runs `--examples`), so the rot
> is invisible — and one of them is the harness behind a headline number in
> FACTS. Worth a separate change?

> **Noticed:** `docs/RELEASE_NOTES_v2.0.md:31` publishes "quality tied with the
> LUT rung (cos 0.9654 vs 0.9657)" with no fixture disclosure, on a pure-Gaussian
> `n=32/k=256` three-seed test whose own gap is 25× inside its noise bar. D12
> says state the fixture and its dispersion. Worth a separate change?

> **Noticed:** "qtip2b CUDA parity 20/20" (FACTS.md:472, `docs/FLEET.md:88`,
> `RELEASE_NOTES_v2.0.md:79`) is a filtered run of the whole test module —
> **4** of the 20 are CUDA↔CPU parity tests, the rest run on macOS CI. Worth a
> separate change?

> **Noticed:** the published artifact is 74.19 GB ⇒ **2.09 bits/param**, but
> "≈1.9 bits/param" (derived from the retracted 68 GB) still stands in
> `FACTS.md:188` and PR #55's model card lines 5 and 246. This is a live public
> number and is wrong today, independent of any rung decision. Worth a separate
> change?

> **Noticed:** a `--isq qtip2b` bake emits no `log_bake_header` line at all
> (`mod.rs:1486`, `:1702` are LUT-only) — D4 clause 4 unsatisfied on that rung.
> Worth a separate change?
