---
name: 00-resume-here
description: "THE compaction-proof entry point. Read top-to-bottom on any resume, new session, or compaction. Current state, the map to x4-8, owners, retractions, live chains."
metadata: 
  node_type: memory
  type: project
  originSessionId: 4d267202-c569-48de-99ab-c497142fac75
  modified: 2026-08-19T02:25:03.811Z
---

# 00_RESUME_HERE — read this first, always

**Compaction-proof entry point.** Fresh session, compacted, or lost: read THIS
top-to-bottom. Everything needed to continue is here or named here.
**Last rewritten: 2026-08-18, session 8. Session-8 CLOSE 2026-08-19.
🔴 SESSION-9 BLOCKER prepended 2026-08-21 — it supersedes the throughput framing
of everything below it.**
If STATUS.md's top entry is newer than this file, that entry wins — reconcile and
update this file.

---

# 🔴 SESSION 9 (2026-08-21): **V4 COULD NOT SERVE PAST ~22 TOKENS. CAUSE: TCFRAG HELD 63 GB. FIXED BY #209.**

**At `cc5487ad3` on an H200, batch=1 decode died with
`CUDA_ERROR_OUT_OF_MEMORY` after 22–30 generated tokens, 3/3 runs, two
independent arms** — card genuinely full at **143,151 of 143,771 MiB**.

**The cause was not the leak. It was the headroom.**

| probe | frees |
|---|---|
| **`ARC_QTIP_TCFRAG=0`** | **64,262 MiB** |
| TCFRAG on (the shipped default at `cc5487ad3`) | **262 MiB** |

`tcfrag_words` (`bitshift.rs:1334`) is a per-weight `OnceLock` caching a second,
differently-packed copy of every weight for the process lifetime — **~63 GB
against a ~79.5 GB model**. That left decode **844 MiB free**, so a ~12 MiB/token
leak that PR #182's leg survived for **2,600 tokens** killed this one in ~20.

**Two independent single-variable rescues, each sufficient alone:**
`ARC_QTIP_TCFRAG=0` → **256/256 tokens, five runs, `finish=length`** (cache still
on); `ARC_CANDLE_ALLOC_CACHE=0` → **1,000 tokens** (TCFRAG still on).

> 🔑 **PR #209 (merged) is the fix that restored serving. PR #213 is defence in
> depth on the leak RATE — do not record it as "the OOM fix".**

> ## ⛔ THE tok/s CONSEQUENCE — **two claims, and only one is universal**
>
> **A (universal):** every V4 single-user tok/s figure is a **≤30-token
> short-window measurement** — the b=1 trio **15.11 / 15.39 / 15.51** says
> *"24 tokens"* in its own row, as do **9.3 / 13.31 / 14.84 / 10.99 / 11.94 /
> 17.85 / 18.27**. **Warm-up dominated; no steady state shown. None is a
> sustained rate.**
>
> **B (NOT universal):** *"measured inside a crashing run"* holds **only at or
> after `cc5487ad3` (#203, 2026-08-21)**. The b=1 trio is from **2026-08-17**,
> before TCFRAG existed — **its 24 was a harness choice, not a truncation.**
> Saying otherwise is a fabrication, and an earlier revision of this block did.
>
> ⚠️ The capture lane's **33.4 → 34.2** is a third case: its source
> `CAPTURE_LANE.md` **does not exist in this repo, on any branch** — it is
> **unsourced**, like the 27.2% TAIL and #210's 16.0%.
>
> Per-step quantities (ms/step, the 57.27 ms sync'd forward, the 88% forward
> share) are unaffected. Model scope: V4 eager KV only — not Qwen2.5-0.5B, not
> the 55 tok/s TurboQuant B200 result.

✅ **A real steady-state single-user number is now obtainable for the first
time — 256/256 tokens runs clean. Nobody has taken it yet.** That is the
cheapest high-value measurement on the board.

**Full record, with all seven session-9 findings and every citation verified at
master `f709872ab`:
`memory/mission/wave66-CS-session9-the-22-token-wall.md`.** Measured numbers and
seven new retractions → `FACTS.md` §2026-08-21.

### The other six, in one line each

1. ✅ **TCFRAG was default-ON and its own header says "UNVERIFIED ON HARDWARE —
   NEVER RUN"** (`qtip2b_tcfrag.cu:7`). **Fixed, PR #209, merged.** It does not
   panic — `OnceLock::get_or_init` caches `None` permanently
   (`tcfrag_words`, `bitshift.rs:1334`). **#203 costed instructions and never
   costed the 63 GB the repack retains.**
2. 🔴 **Two host round-trips per decode step, not one** —
   `cudaStreamSynchronize` (`graph.rs:362`) **and** a greedy `argmax`
   (`sampler.rs:1479`). **This RETRACTS "op count is retired as a lever"**: the
   1,137× launch cut that bought ~8% **kept both round-trips in the captured
   arm**, so the CPU never left the loop and the test could not price removing
   it. *(It does not show the syncs ARE the limiter — only that the experiment
   could not see them.)* Zero-sync arm exists, unmeasured:
   `arcgraph/device-decode-loop` @ `1b6949244`.
3. 🔴 **The dedicated/autonomous decode tier cannot accept V4 architecturally.**
   `LayerWeights` (`weights.rs:430`) has seven non-optional projection slots
   (`DENSE_PROJS_PER_LAYER`, `weights.rs:44`) — **no MLA slot, no expert slot**.
   Accepting V4 trades a named refusal for a wrong-tensor read.
4. 🔴 **The b=1 FP8 GEMV is instruction-bound, not latency-bound** — a runtime
   signed idiv by a kernel argument (`blockwise_fp8_gemm.cu:199`), once per four
   weight bytes. **This retracts the "4% memory-controller ⇒ latency-bound"
   story.** Ceiling **1.5–3.0 TB/s** vs a 4.8 TB/s roof — quote the range, not
   the 2.97 (that used the FP32 warp rate; the INT32 pipe gives 1.49).
   ⚠️ **PR #210 merged (`f709872ab`) with a projected win multiplied out of an
   unsourced 16.0% share** — default OFF and `UNVERIFIED ON HARDWARE`, so the
   kernel is safe, but **do not quote "34.2 → ~35.8 tok/s" as expected.**
5. 🔴 **The consequence of a session-8 finding, never drawn.** That
   `mark_unreachable` is itself dark was already recorded below (§TAXONOMY
   corrections). What was not: it is inert unless `ARC_PROFILE` is exactly
   `"1"` (`arc-profiler/src/lib.rs:467`), so **its six sites in `normal.rs` —
   `:1589`, `:1668`, `:1954`, `:2690`, `:2760`, `:2888` — have been dark in
   every ordinary run, and we have been reading their silence as evidence.**
   Separately and newly: `ci_cuda.yaml` is `workflow_dispatch`-only
   (`ci_cuda.yaml:3`) on a self-hosted ARM64 GPU runner — **it gates nothing**;
   the real lane is `cuda-typecheck` (`cuda_compile_check.yaml:337`).
6. 🔴 **21 env flags are presence-tested, so `=0` turns them ON** — **any past
   A/B that used `=0` as its "off" leg was never a comparison.** Sweep is
   PR #212, open.

---

# 🟢 SESSION-8 CLOSE (2026-08-19) — THE HEADLINE. READ BEFORE ANYTHING ELSE.

## WHEN INSTRUCTION-BOUND, BITS-PER-WEIGHT **CANCELS OUT** OF THE THROUGHPUT EQUATION

`achieved = ceiling(bpw) × budget(bpw)/X`, `ceiling ∝ 1/bpw`, `budget ∝ bpw`
⇒ **`achieved = 23,409/X` at B=256 — bpw-invariant.**

**The 2-bit format does NOT buy speed. It buys the model EXISTING ON ONE CARD.**

| format | bpw | V4-284B weights | fits one H200 (141 GB)? |
|---|---|---|---|
| **Arc QTIP** | **2.09** | **74 GB** | **YES** |
| MXFP4 / Marlin — best competitor weight-only | 4.25 | **151 GB** | no |
| FP8 — what SGLang actually serves | 8.0 | **284 GB** | no |

**SGLang's V4 numbers run on TWO GH200s. No competitor format puts this model on
one card at all.**

> 🔴 **THE CAPACITY WEDGE IS ALREADY WON BY THE FORMAT. IT IS A DIFFERENT CLAIM
> FROM SPEED, WHERE WE ARE 15–25× BEHIND. NEVER PUT THEM IN THE SAME TABLE.**
> Conflating them is what produced the **"16,600 → 1,477"** framing Jish
> rejected. Capacity-per-node is measured in *models that fit and users served*;
> speed is measured in tok/s. **Two claims, two tables, always.**

## The full record

**`memory/mission/CENSUS_SESSION8.md`** — the complete Arc/SGLang/vLLM census,
zero GPU hours. **`memory/mission/LADDER_POST_CENSUS.md`** — the GPU ladder
reordered by it, with Rung 0 (~3.75 sessions, **zero GPU hours**) gating
everything above. **Both are on branch `docs/census-session8` until it merges.**
Measured numbers → `FACTS.md` §2026-08-19. New rules → `KERNEL_RULES.md` top block.

🎯 **`memory/mission/FRONTIER_BITS_FOR_DECODE.md`** — **THE BIT-RATE FRONTIER.
Read it before quoting the 3.46× decode win, before proposing any codebook work
at 2 bpw, and before choosing a geometry.** The cheap decode is real but is
**NOT at 2 bpw**; the shipping geometry is **K9/V4/L12 at 2.25 bpw**. Carries the
measured reason 2 bpw is closed (trellis freedom, not codebook coverage), the
exact byte-extraction bound, the census's structural blind spot, and a
`CONFIRMED FALSE` on the "GPU computes the codebook" comment. **CPU only, $0,
every table re-run from scratch.**

## MEASURED this session — zero GPU hours, four results

- **Trellis inst/weight, COMPILED** (`nvcc -cubin`, CUDA 12.4.131, inner loop by
  unroll differencing): shipped K4/V2/L16 **15.125** sm_90 / 14.812 sm_80 ·
  K4/V2/L13 LUT **11.250 / 10.250** · K8/V4/L12 LUT **5.375 / 4.625** ·
  **+ row-scale hoist 4.375 / 3.625**. K8/V4/L12 rows are clean 3-point,
  **0.00% linearity**, both arches. ⇒ **3.46× fewer instructions than shipped —
  and still 3.1–3.9× short of the 1.13–1.41 budget.** Both LUT geometries are
  exactly **32,768 B**, under the 48 KB static limit, **no `cudaFuncSetAttribute`
  needed**; occupancy **62.5%** (not the predicted 31%) and
  **register-limited, not shared-limited**.
  🔴 **AND THEN A FIFTH RESULT SUPERSEDED THE GEOMETRY, SAME DAY, SAME $0:**
  **K8/V4/L12 is quality-CLOSED** — **−0.00698** `w_cos` against a **±0.0008**
  band, and **nine codebook designs top out at −0.00307**, still 3.8× outside.
  The 3.46×-shaped decode is available **for a quarter of a bit more**:
  **K9/V4/L12, 2.25 bpw, +0.00402 — 5.0× the band on the GOOD side of the shipped
  control** — at the **same 32,768 B table** and the **same decode shape**, since
  at fixed L=12/V=4 **the codebook does not read K**. **The price is capacity: KV
  58.8 → 49.6 GB, and that effect is UNMEASURED.** ⚠️ **Do not quote 4.375 for
  K9 — the kernel has never been compiled at any K but 8.**
  ✅ **The one-card wedge SURVIVES the extra quarter-bit:** 2.25 bpw ⇒ **83.4 GB
  of 141**, still **1.8× under MXFP4's 151 GB**, which does not fit at all.
  Full record + anchors: **`memory/mission/FRONTIER_BITS_FOR_DECODE.md`**.
- **V4 KV over-retention:** 361.4 MB/seq → **6.35–11.96 MB/seq at 8k = 30–57×**;
  **B=32 @ 8k: 11.57 GB → 0.38 GB.** Compaction copies 1.12 rows per row stored.
- **Launch reductions shipped:** 9,131 → **8,650/token** (−430 dead mHC casts,
  −43 router weight, −8/seq/token sampler). Casts **1,571 → 1,141**.
  ⛔ **2026-08-21: the 1,571 BASELINE IS RETRACTED AS STALE** — it was profiled
  at `05af600e7`, and the three commits that retired it (`9f110905b`,
  `1f6ef9da9`, `179e405ac`) are ancestors of HEAD but not of that ref. MHC's
  whole remaining b=1 cast budget is **86 launches/token**
  (`dsv4_mhc.rs:279`). **The 1,571 → 1,141 delta is scored against a number
  that no longer existed; do not quote it as a saving.** (PR #206, FACTS
  §2026-08-21.)
- **Frozen Gumbel:** flat 64-token distribution, 512 draws ⇒
  **1 distinct token before, 64 after.**

## 🔴 RETRACTIONS — FOUR OF THEM ARE MAIN'S. Do not quote any row's left column.

| claim | status |
|---|---|
| "K4/V4/L12 → 1.38 inst/wt is at budget, full 16,602 recoverable" | **RETRACTED** — 1 bpw scored against a 2-bpw budget (`bpw = K/V`, `qtip/mod.rs:376-381`) |
| "MTP acceptance p ≈ 0.485" | **RETRACTED** — appears nowhere in the tree; the real number is **0.4194** (`wave51-CB-the-measurement.md:166-186`) |
| "The 2-bit draft head explains the acceptance gap" | **RETRACTED** — `floor_mtp_isq` fix `07766cfa1` is an **ancestor** of the measured binary `46ea6948d` |
| "Fixing acceptance is worth ×5.53" | **RETRACTED — ×1.55** against the measured base; a multiplier on a 6× problem |
| "PD disaggregation is behind every ≥1.5k tok/s/GPU figure" | **RETRACTED** — it is the **metric definition**: `(input+output)/GPU` at isl 8192/osl 1024 = a **9× multiplier**, plus a TP≥4 divisor and a 2.97× spec multiplier |
| "The re-bake gets cheaper at K8/V4/L12" | **RETRACTED, INVERTED** — production baker is **beam, not exhaustive** (213 vs 8,257 s/layer A100), **issue-bound not bandwidth-bound**; cost `(n/V)×W×2^K` ⇒ **~8× MORE**: ~213 → ~1,700 s/layer ≈ 20 h ≈ **$30** |
| "The default config silently disables prefix caching" | **NARROWED** — only CUDA + PagedAttention + standard layout + head_dim 128. **V4 is MLA, so it does NOT apply to the flagship** |
| "Expert parallelism is env-only" (and its mirror, "`ep_size` is a config field, **not** an env var") | **BOTH WRONG — there are TWO doors.** `ep_size` is a serde `config.json` field (`deepseek4.rs:317-318`, default `1` at `:140`) **and** `ARC_EP_SIZE` overrides it at run time (`effective_ep_size()`, `:371-375`). `build_expert_parallel_plan` is production-reachable (`:2227` → `:2300`). Real gap: **no CLI/server flag** |
| "`CrossPrefixMeter` is dead" | **WRONG** — it records in production; only its readout is test-only |
| "There is no runtime LUT on the GPU" | **NARROWED** — true only of `qtip_gather_gemv.cu`. `qtip_gemv.cu:266` still selects a stored Gaussian LUT — the **shipped default** (`mod.rs:570`) |
| "`ragged_decode_supported`" | **DOES NOT EXIST** — the predicate is `batch_can_be_ragged` (`kv_cache/mod.rs:1227`) |
| "head_dim 512 is the blocker" | **INVERTED** — 512 compiles; **448** fails (`vec_size 14 % 8 ≠ 0`) |
| "SGLang uses page_size=1" | **WRONG for V4** — overridden to **256** |

## 🔒 SEQUENCING — each is a GATE, not a preference

1. **MoE expert gather blocks chunked prefill.** Chunking multiplies steps and
   the gather is billed **per step**, so at 71.3% of an N=128 step chunked
   prefill is **~3× NEGATIVE today**. It is also 71.3% of prefill outright ⇒
   **the single highest-value item in the program. Do not reorder these two.**
2. **Per-position MTP telemetry blocks acceptance work.** `MtpAcceptance.accepted`
   is a scalar; distribution mismatch and chain compounding are indistinguishable
   without it. Renting a card first buys a guess.
3. **Token-level cache hit rate blocks evaluating any prefill change.** The
   request-level metric reads **100%** when one 32-token block of 2048 is reused
   ⇒ **no Arc run can currently report whether the cache hit.**
4. **Graph-mode mask blocks any quality claim on the CUDA-graph arm.**
   `set_graph_mode_mask` is called from nowhere; the unwritten tail is attended as
   zero-padding and **takes softmax weight**. A captured graph replays a subtly
   wrong attention rather than failing.
5. **The shipped `qtip2` artifact cannot reach the grouped GEMM at all** — above 8
   tokens it dequantizes every distinct expert to BF16 in HBM (**16 traffic units
   vs 1**). **Every throughput number assumes a kernel we do not ship.**
6. **The KV window and ragged batching cannot both be on yet.** A windowed slot
   masks dead space **by position**, which equals physical width only when the
   whole sequence is resident.
7. **Ragged decode must NOT be defaulted on.** Live correctness defect: shorter
   rows in a mixed batch attend compressed blocks **they have not reached**, and
   do it silently. **`fix/ragged-gate-coherence` lands first.**

## TAXONOMY corrections (details in `TAXONOMY.md`)

- **Expert parallelism IS on master, with TWO doors** — PR #89 merged
  (`610c4506b`), feature `fce33ae22`: `deepseek4.rs:2227` → `:2300` → `:2308`,
  plus `moe/expert_parallel.rs` (847 lines). Door 1 is the serde `config.json`
  field `ep_size` (`:317-318`); door 2 is `ARC_EP_SIZE`, which overrides it in
  `effective_ep_size()` (`:371-375`). **The gap is the missing CLI/server
  flag** — neither "env-only" nor "not an env var" is true.
- **`mark_unreachable` — the registry for dark features — is ITSELF dark**:
  inert unless `ARC_PROFILE=1`, and **4 of its 6 `site` strings have drifted**.
- 🔴 **Arc has never run a forward pass on more than one GPU.** Every multi-GPU
  claim is "code-complete, never run".

---

## 🔴 SESSION-8 LATE BLOCK (2026-08-18, ~11:00 UTC) — READ THIS BEFORE §0

**The 100× is ATTRIBUTED. Read `BUDGET_V4_B1.md` before proposing ANY decode optimisation.**
66.68 ms/token = **98× the bandwidth bound**, = **51% kernels / 49% GPU idle waiting on the host**.
The disease is **OP COUNT**: 9,131 launches + 11,436 allocations per token, median kernel 1.18 µs,
memory controller **4%** utilised, 204 W of 700.

### The four things that were BUILT, WIRED, AND SWITCHED OFF (the pattern of the night)
1. **The non-default CUDA stream existed** — gated behind `ARC_CAPTURE_STREAM`, which
   `ARC_V4_CAPTURE_PROBE` did **not** imply ⇒ asking for capture bound the **NULL stream** and the
   runner disabled itself behind one `warn`.
2. **`graph_wrapped_forward`, the dedicated decode path, and the autonomous runner all exist** —
   constructed **only on the PagedAttention arm**. V4 takes `DefaultInstructions`
   ⇒ **never constructed** (`normal.rs:1907`, pinned by `normal_loaders.rs:5687`).
   **ONE CAUSE, THREE SYMPTOMS.** ← the critical path; a chain owns it.
3. **candle's `AllocCache` was fully wired and `enabled: false`** — the 11,406 allocs/token were a
   *disabled feature*, not a missing one.
4. **`ARC_CANDLE_ALLOC_CACHE`** — one call site, gated behind an unrelated probe; does nothing
   alone. Recommendation: **delete it**, have capture require `ARC_ARENA`.
5. **`CudaSampler::sample` was dead on EVERY call** — `tensor_device_ptr` had no I32 arm, so it
   returned `unsupported dtype I32` every time. Its tests are a CPU simulator that **structurally
   cannot observe** that the GPU version never ran.
6. ⚠️ **RETRACTED — the capture-safety mechanism is WIRED, not dead.** An earlier claim that
   `GRAPH_MODE_POSITIONS` "appears only inside `layers.rs`, nothing calls it" came from grepping a
   **stale local master checkout instead of the branch the work lives on.** On `wt/pagedgate` it is
   wired in three places: `normal.rs:1663` sets it, `deepseek4.rs:1623` consumes it in a dedicated
   MLA arm, `normal_loaders.rs:3314` documents it. **Do not count this as a dark room.**
   ⇒ **Grepping one tree and reporting it as the state of the world is its own recurring fault** —
   third instance this session, alongside two source-reads reported as observations.

   **What IS true, and it is better:** V4's MLA **does** honour graph-mode positions —
   `None if has_graph_mode_positions() && seq_len == 1 => { let position = graph_mode_positions()?…;
   append_graph_kv_mqa(kv_cache, &k, &position, cap) }` — so the **KV write uses the device slot,
   not a host `seqlen_offsets`**, and the window is fixed at `sliding_window` so every step is
   shape-identical. That is the replay-safe contract, already satisfied.

   🔴 **But the mask half is unwired, and it is a live CORRECTNESS hole:** `set_graph_mode_mask` is
   never called, so `graph_mode_mask()` returns `None` and — in the tree's own words — *"the
   unwritten tail slots are attended as zero-padding (**finite, not yet correct**)."*
   **A captured graph would replay a subtly wrong attention rather than fail.** Fix is small:
   compute the additive `[*, sliding_window]` mask on device from the same positions tensor.
7. ⚠️ **RETRACTED — the MHC fused-cast path DOES run.** `V4MHCLayerParams::hc_pre` **is** the
   `hc_pre` the V4 forward executes; there is no second one. **The counter printed from a spawned
   thread via `println!` and that output never reached the bench log**, so *"no output"* was
   indistinguishable from *"no execution"*. A file-based probe settled it in one run.
   ⇒ 🔑 **"NO OUTPUT" IS NOT "NO EXECUTION."** A reachability claim resting on a missing print is
   not a measurement. Write engagement evidence to a **file**, never to stdout that a harness,
   a thread, or buffering can swallow.

## 📊 THE DARK-ROOM COUNT IS **FIVE**, NOT SEVEN — two retracted, and BOTH were reporting artifacts
**Confirmed (5):** the gated non-default CUDA stream · the three runners on the PagedAttention arm ·
candle's `AllocCache` at `enabled: false` · `ARC_CANDLE_ALLOC_CACHE` (one call site, unrelated gate) ·
`CudaSampler::sample` dead on every call (missing I32 arm).
**Retracted (2):** `set_graph_mode_positions` (claim came from grepping a **stale tree**) and the
MHC path above (**lost stdout**).
> ⚠️ **Main's own pattern: I relayed all seven as fact. Neither retraction was a code fact — both
> were an agent's instrument failing to report.** *Before amplifying "X is dead", require the
> claim to name its tree/SHA and to rest on a file-written counter, not on absence of output.*

> **Arc's problem is repeatedly not "unbuilt" — it is "built, wired, and switched off by something
> adjacent." Check reachability before writing a replacement.**

### MEASURED and standing
- **GPU-autonomous decode WORKS on this H200**: WHILE conditional node, **2.0 → 0.0156 host calls
  per decode step** (2 per *generation*, not per step), device-loop overhead **constant in body
  size** (~2 µs = 0.004% of the step). Control: device-side `target=1` then `target=N` returned
  counter 1 then N ⇒ **the device chose the trip count**. Blocked only by (2) above and by
  `cuda.h:1971` barring alloc nodes in a conditional body.
- **Trellis GEMV: 77.31 → 66.19 µs/call, −14.4% at 0.16% drift**, 35.52 → 28.91 inst/weight,
  bit-identical with a mutation control at 21,494 differing bytes. **The one measured kernel win.**
  It is **29× above its OWN bandwidth bound (NOT 15× — that divided by the whole model's floor).**
- **`sum2` codebook = 2.27× AVAILABLE**, contingent on a re-bake and **Jish's call**. Timing-valid,
  **NOT a quality claim** (it decodes Gaussian-baked bytes against the MCG codebook).
- Sampler: replay-safety and distribution-equivalence **proven**, including the order-scrambled
  case (TV 0.0061, per-draw agreement 15.6%). **Speed unmeasured ⇒ wiring stays on legacy.**

### ✅ RAGGED BATCHING TURNED ON FOR V4 — **143× RECOVERY at B=8**, but capacity still FLAT
| | baseline | **ragged ON** |
|---|---|---|
| B=1 aggregate | 9.3 | **13.31** (per-user 14.84) |
| **B=8 aggregate** | **0.1** | **14.34 — ~143× recovery** |
| B=8 achieved width | 1 of 8 | **3 of 8** |
| B=8 per-user | — | **8.38** (fell from 14.84) |

**Engagement PROVEN, not assumed: `ragged_mask_engaged: true`** — per-row masking actually ran, so
this is a real ragged decode and **not a uniform batch in disguise**. All 8 rows exactly 128 tokens;
`ensure_uniform_batch_cache_lens` never fired. **At B=1 the beacon correctly did NOT fire** (B=1 is
uniform by construction) — the marker discriminates rather than always lighting up.

> **HONEST READING: the collapse is fixed; the multiplication is not there yet.**
> **B=8 aggregate is only 1.08× B=1.** Concurrency went from **catastrophic to roughly neutral.**
> Per-user *fell* 14.84 → 8.38, i.e. 8 users share about 1.7 users' worth of throughput.

**The remaining limiter is the PROMPT side, exactly where predicted:** prefill still buckets by
**exact** length (`default_scheduler.rs:281`), so rows **trickle into the decode cohort instead of
joining it together.** That is what capped the earlier Qwen result at 47/128 too.
⇒ **NEXT LEVER: prompt-side bucketing.** The decode side is now solved.

**The sweep will exit 1 — correctly** — because width 3 misses the floor of 4 the chain **declared
in advance.** A declared floor honoured against your own result is the practice, not a failure.

### 🔴🔴 (superseded above at B=8) THE WEDGE MEASURED — CONCURRENCY MADE V4 WORSE (08-18)

**master + PR #134, DeepSeek-V4-Flash, H200, spread prompts 128–4096, client-observed:**
| B | **aggregate tok/s** | per-user | TTFT p50 | **requested → ACHIEVED width** | waiting p50 | SM% |
|---|---|---|---|---|---|---|
| 1 | **9.3** | 14.34 | 3.05 s | 1 → **1** | 0 | 81 |
| 8 | **0.1** | — | — | 8 → **1** | 7 | **100** |
| 32 | **0.3** | — | — | 32 → **4** (max 6) | 31 | **100** |

**Aggregate does not scale — it COLLAPSES: ×0.01 at B=8, ×0.04 at B=32.** Per-user and TTFT are
blank at B≥8 because **no request completed, and none even received a first token, inside 100 s.**

**FORWARD PREDICTION MATCHED THE CARD:** computed *before the run* as achieved ≈ `B/9` —
**predicted 1 at B=8 (measured 1), ~3 at B=32 (measured 4).** Mechanism established, not guessed.

### ⛔ TWO NUMBERS THAT MUST NEVER BE QUOTED AS CAPACITY
1. **The engine's own `Throughput (T/s)` read 358–667 while users received 0.1 tok/s** — it counts
   **prefill** tokens. **Overstates serving capacity by ~3 orders of magnitude.**
2. **SM utilisation was 100% at B=8/32 while delivering ~0.1 tok/s.** The card is saturated doing
   prefill and cache materialisation. **"The GPU is busy" is not evidence of throughput.**

### ⏳ THE ONE MEASUREMENT THAT CLOSES THE ARGUMENT — not yet run
**The uniform-length control.** By the same arithmetic uniform prompts coalesce (`0 ≤ 16384`) and
should reach **full width**. **If aggregate jumps there, the cache-uniformity constraint is proven
sole cause** and the fix target is unambiguous. **Cheap and fast, precisely because uniform traffic
is the case that works.** Also unestablished: B=64/128 and the drift controls (killed mid-flight).

### 🔴🔴🔴 **ROOT CAUSE — ONE CAUSE, FOUR SYMPTOMS** (08-18)

> **No PagedAttention for V4 → dense `NormalCache` → a forward requires EXACTLY EQUAL cache lengths
> → mixed-length traffic fragments into length buckets → aggregate is bounded by the LARGEST BUCKET,
> not by B.**

**That is why capacity per node has not multiplied.** It is not scheduling policy, not launch
overhead, not the kernels. **The dense batch cache cannot hold sequences of differing length in one
forward**, so only one length-bucket decodes per step.

**Measured refusal, before any run:** the coalescence rule `(total − n_min)·gap ≤ n_min·256` is
**refused at every B** on a 9-distinct-length pool — at B=128, `14592 ≤ 3584` is false.
⇒ **Predicted achieved width ≈ B/9 against B requested.** *(Prediction recorded BEFORE the sweep;
if it lands, the mechanism is proven rather than inferred.)*

⚠️ **`32 running, 32 waiting` is documented as ALREADY FIXED** by the coalescence override, with a
regression test — **but that case had `gap = 1`.** The spread-traffic refusal above is a **different
and much larger effect.** Report achieved width as *verification of that fix*, not as an expected
failure.

⇒ **THE ROADMAP ITEM: to get 4–8× on V4 we need either PagedAttention for V4, or a dense cache that
tolerates ragged lengths.** Everything else is downstream of that choice.

✅ **CORRECTED (twice): the dense scheduler DOES interleave prefill and decode** — two forwards in
the same iteration, early returns firing only when one side is empty. **The "~5.9 s stall per
arriving request" claim is paged-path only and does NOT transfer to V4.** Main relayed it as a V4
finding; that was wrong both times it was stated.

### 🔴🔴 PAGEDATTENTION IS **NOT ACTIVE FOR V4 AT ALL** — three subsystems are dark on the flagship
`DeepSeekV4Loader::supports_paged_attention()` returns **`false`** (explicit Wave-29 audit comment:
*"Still `false`"*), so `normal.rs:345` sets `paged_attn_config = None`. **That is why no paged
allocation line appears at startup.** Three consequences, and they reprice tonight's work:

1. **V4 runs the dense `DefaultScheduler`, NOT the paged one.** ⇒ **tonight's headline "paged decode
   was bucketing by sequence length ⇒ 8.7× on spread traffic" lives in `paged_attention/scheduler.rs`,
   WHICH V4 NEVER EXECUTES.** The V4-relevant fix is the **dense decode-side length-limiter removal**,
   and the documented **`32 running, 32 waiting` at B=64** failure mode is in **`default_scheduler.rs`**
   — that is the one to watch.
   ⚠️ **CORRECTION: the "prefill excludes decode" finding below is at `paged_attention/scheduler.rs:216`
   and therefore ALSO does not apply to V4.** It must be re-checked against `default_scheduler.rs`
   before being quoted as a V4 ceiling. Main relayed it to Jish as a V4 finding — that was wrong.
2. `max_kv_tokens = max_seq_len × max_batch_size` (4096×1) feeds `calculate_cache_config`, **never
   reached for V4** — that number is moot here, **though the auto device mapper still placed layers
   assuming `max_batch_size: 1`.**
3. **ArcGraph autonomous decode AND the dedicated decode path are ALSO inert for V4** —
   `mark_unreachable("cuda_graph.autonomous_decode", "cache_config is None…")` plus the startup line
   *"Dedicated decode path declined… not the dense shape"*.

⇒ **Any aggregate number for V4 characterises the DENSE scheduler and must be reported that way.**

### 🔴🔴 (PAGED PATH ONLY — see correction above) PREFILL EXCLUDES DECODE
`paged_attention/scheduler.rs:216` — **the scheduler returns EITHER a prefill batch OR a decode
batch, never both.** The prompt-admission loop **returns early at `if !scheduled.is_empty()`,
before the decode leg.** ⇒ **every arriving request stalls ALL running decodes for a full prefill
step.** Compounded by **prefill not being chunked** (`prompt_chunksize` is a stale comment, no wired
field), so a prefill step is the **whole prompt, indivisibly**:
> **prefill @ N=512 = 5,876 ms · decode step = 66.68 ms ⇒ one arriving 512-token request freezes
> every running user for ~5.9 s ≈ 88 decode steps.** Invisible in every single-user measurement.

🔑 **This multiplies the value of the prefill grouped-GEMM fix**: it is 4–6.6× for the *arriving*
user, but because that step **blocks everyone else**, it shortens the stall **the whole batch eats**.
**Its aggregate value exceeds its single-user value, and no B=1 benchmark can show that.**
**Being tested, not assumed:** `decode_heavy` vs `prefill_heavy` legs at B=64.

⚠️ **THE WEDGE IS STILL UNMEASURED.** Batching was fixed four times on 08-18 (paged bucketing 8.7×,
decode-side length limiter, prefill floor, and a **dedicated decode path that gave every user but the
first silently wrong output**) — and **aggregate throughput was never measured afterwards.** Every
number on this page is B=1 or component-level. **Jish's bar: a couple hundred tok/s single-user, a
couple thousand aggregate.**

### 🐛 `logit_bias` IS A DEAD WIRE — user-facing
`mistralrs-server-core/src/{completions,chat_completion}.rs` populate `SamplingParams::logits_bias`;
**nothing in `mistralrs-core` reads it.** The OpenAI-compatible endpoint **accepts it and silently
ignores it, with no error.**

### 🚨 MEASUREMENT DISCIPLINE — see KERNEL_RULES **D33, D35, D35b, D35c, D36**
- **D35c**: the bench lock must cover **HOST CPU**, not just GPU. A 12-way compile made an
  *unmodified* control arm read **169.49 ms/T against its 66.99 reference**. Two locks:
  builds `flock -s host.lock`, benchmarks `flock -x bench.lock` **then** `-x host.lock`.
- **D36**: naive A-B-A-B deltas are **BIASED** (drift **+0.91 ms/T per slot**, derived
  independently by two chains). Fit `y = a + b·slot + c·arm`, report `c`. **Six arms, not four.**
  ⇒ **The `GpuApprox` delta is RETRO-INVALIDATED** (c = +0.572 ± 0.855, 1 dof, not distinguishable
  from zero). Its *conclusion* stands on the engagement evidence (`kv_fp8_quant` +83%), not the delta.
- **A2 = 68.58 vs B1 = 69.78 would have shown the optimised arm SLOWER** — the mirror of the
  "2.4× win". **A contaminated window manufactures wins and regressions equally.**
- **Instrument the environment, not just the result**: log builders/load/GPU-util **per arm**, so
  exclusion is a rule stated in advance, never post-hoc selection on the outcome.
- **D35b**: never share `CARGO_TARGET_DIR` — one slot measured a binary containing none of its
  chain's code. Gate every run on a symbol check.
- **Box access**: the platform **wipes `/root/.ssh/authorized_keys`**; put the key in
  **`authorized_keys2`**, which it does not touch. Paths are under **`/root/budget-chain/`**,
  NOT `/root/arc` or `/root/models`. See `GPU_ACCESS_RULE.md` (D34).

---

## 0. FIRST 60 SECONDS

1. **`~/.config/arc/bin/arcgpu instances list`** — never bare `runcrate` (single-use
   OAuth token; concurrent callers burn Jish's login and only he can restore it,
   browser-only). If a box is up with nobody driving it, that is $4.92/hr burning.
2. **Occupancy = `nvidia-smi --query-compute-apps=pid,used_memory --format=csv`.
   NOT the lock file.** `/root/locks/gpu.lock` is opt-in and worthless in BOTH
   directions — held-and-idle and absent-and-busy have both been observed.
   **Claim the lock FIRST, then read compute-apps, then launch.** Never check-then-claim.
2b. **`~/.config/arc/bin/arc-verify-master`** — run it **after every merge batch, before merging
   anything further**. Builds `origin/master` in a clean detached worktree; exit codes captured
   into variables, **never through a pipe**. **Three states: `0` PASS · `1` FAIL (stop merging) ·
   `2` ENVFAIL.** Refuses to call a run green if the tests printed **no result line** or
   **`0 passed`**. `--self-test` proves it can say no, using `03dd1d7f3` (known red) and
   `d63ad2c5a` (known green) as fixtures.
   **Why it exists (2026-08-18):** two PRs, **textually disjoint, no merge conflict, both a genuine
   17/17 CI complete**, broke master's *test target* — `#118` added a fixture building
   `tokio::sync::Mutex<SequenceGroup>`; `#115` then moved `Sequence::new_waiting` to
   `std::sync::Mutex`. **Neither CI run could have caught it: GitHub tests each PR against the base
   as it stood.** Shipping lib was fine throughout; only the gate was red.
   **Branch protection stays `strict: false` — deliberate.** Requiring up-to-date branches would
   catch this class but serialise ~16 open PRs behind a ~20-min cycle each, and that pressure is
   what makes people merge without re-checking. **One check per batch beats one CI cycle per PR.**
   ⇒ Corollary (D29): **a verification pinned to a SHA stops being evidence when the SHA moves.**
   "477 passed at `006657e05`" was quoted as standing after seven further merges. Re-run, don't recall.
3. Read **§3 (state)** below, then `STATUS.md` top entry. Only then act.
4. `gh pr list -R aeonmindai/arc` — **always `-R aeonmindai/arc`**; bare `gh` hits
   upstream mistral.rs.

---

## 1. THE MISSION

Runcrate rents GPUs. The wedge is **capacity per node** — one node serving 4–8×
more users multiplies a fleet without buying a card. Score by aggregate tok/s per
node and $/Mtok. **×4–8 is credible; ~×1 is shipped.**

**The moat is the byte formats** — trellis weights (QTIP) and compressed KV
(TurboQuant) — *not* any one kernel. That is why the GEMM and the attention kernel
must be ours: anyone can adopt a better kernel; nobody can adopt our weight format
without writing the decoder.

Owner: Jish (Nirupam), Aeonmind, building Arc for Runcrate.

---

## 2. THE MAP TO ×4–8 — two multipliers, and they stack

**A — fill the machine (~2.8×).** On length-diverse traffic the scheduler runs
**16 of 128** users. Measured, four confirmations, one of them a forward prediction.

**B — cut the per-user cost (~1.7–2.3×).** Each extra sequence costs **19.4 ms**;
attention is **11.0 ms (57%)**, moe 4.2, host 3.2, sync 0.2. Step time is *linear*
in B, so throughput converges to `1000 / marginal`. **Saturation IS the marginal cost.**

```
2.8 x 1.7  ~ 4.8x
2.8 x 2.3  ~ 6.4x     <- lands in the band, on measured or registered-prediction numbers
```

⚠️ **B's mechanism is UNRESOLVED.** At 320 tokens, reading all the KV costs ~9 µs
against 11.0 ms of attention — **~1250×**. So at short context attention is NOT
byte-bound and compression cannot be the lever there. Payoff scales with context
length. **Decomposing the 11.0 ms by existing child spans is free and is the single
most decision-changing measurement outstanding.**

---

## 3. STATE — 2026-08-18

### ✅ MEASURED AND HOLDING
| What | Number | Model |
|---|---|---|
| Length-bucketing fix, paged path | **8.72× / 9.15×** @B=8 spread; **8.10× / 8.35×** @B=32 | Qwen2.5-0.5B |
| Prefill throttle | **4 → 101** of 128 users got a first token; TTFT −45%; aggregate +33% | V4 |
| Per-sequence advance | **4.46×** @B=8 spread, **3.25×** @B=32 | V4 |
| `xs` window pin | **+23.2%** @uniform B=32 (ms/step 2371→1287); neutral elsewhere | V4 |
| Dense ragged decode | **+17.4%** spread; **+28.4%** uniform-in-isolation | Qwen2.5-0.5B |
| The bucketing law | `running = B ÷ distinct lengths`; 1.06/4.07/16.06 vs 1/4/16 | V4 |
| Ragged-prefill token bug | every row but the longest sampled its first token from PADDING — FIXED | — |
| **KEYSTONE GEMM (PR #124)** | **8.16→4.95 and 8.10→4.91 µs/m-tile = +39.4%, bit-identical, margins 444× / 218×.** Handicap **1.76×→1.12×**; crossover **B=64→B=12**; B=256 projection 3.24×→**5.35×** (derived, checked instrument). **The two shapes agree to 0.03pp — transposes with identical decode work but different access patterns everywhere else, so a bandwidth/tiling fix could not land twice. Mechanism confirmed by invariance.** ⚠️ D16 NOT satisfied: SM90 measured, SM100 not compile-verified | H200 |
| Prompt-starvation floor — **Qwen2.5-0.5B** | baseline **65 of 128** served; floor=4 → 128 served, TTFT −67%, **throughput −34%**; floor=16 → 126 served for **−7.6%** | Qwen2.5-0.5B |
| Prompt-starvation floor — **V4, and it INVERTS** | baseline TTFT p50 **87.28 s**; **floor=4 → 24.17 s (−72%) for −4.7% throughput**; **floor=16 does NOTHING** (+0.3%, TTFT −0.06%). ⚠️ `run_max` tops out at **52 on every arm** ⇒ V4 admission is **prefill-throughput-bound (~281–405 tok/s)** and NO scheduling change can lift it | V4 |

> 🔴 **DO NOT SHIP A CONSTANT. Main called floor=16 "the setting to ship" on Qwen data — V4
> refutes it.** The floor's price is set by **how decode-productive the machine is**, because
> forcing a prompt bucket costs whatever the decode steps it displaces were producing. Qwen's
> decode is cheap and productive ⇒ one step in five costs 34%. V4 is already sitting behind
> tens of seconds of prefill ⇒ the same forcing costs ~nothing and buys 63 s of median
> latency. **Qwen wants 16 and finds 4 overpriced; V4 finds 16 inert and wants 4.** The knee
> moves *left*, not right. The knob is right; a single default this data supports does not exist.

### 🔴 RETRACTED / UNRESOLVED — do not quote these
| Claim | Status |
|---|---|
| "fused 512 attention kernel is neutral" | **RETRACTED.** Entered **0 times** in 320 decode steps × 43 layers; positive control fired 12,000+. The flag sits downstream of an early return. Cost is UNKNOWN. |
| "the 47-ceiling is the prefill limiter" | **WRONG (main's).** Opposite failures: throttle vs **starvation** (47 seqs outscore one fresh prompt bucket ~47× on summed priority). Applying the cap to a starved cohort makes it worse. |
| "attention is byte-bound" | **UNRESOLVED.** 760× above its byte bound at B=1. Whether it inverts at B=32 is the open question. |
| "`clone_in_cache` costs 28.3 ms/step" | **WRONG BY 20×** — **+572 ms/step** at constant batch width (uniform B=32, both arms `1.000×` offered). |
| "+6.1% B=32 spread, +5.1% B=128 uniform" | **UNRESOLVED** — under the ~10% order-effect threshold. |
| "TurboQuant was never measured" | **RETRACTED** — `4eba13905`, 55 tok/s = **+46%** over Candle, B200, Qwen3-32B, on Modal not a rental. The separate "4.27× end-to-end" IS arithmetic and stays retracted. |

> 🔴 **THIRTEEN FURTHER RETRACTIONS WERE ISSUED 2026-08-19 — four of them main's.**
> They are in the **SESSION-8 CLOSE block at the top of this file** and in
> `FACTS.md` §2026-08-19. **Read that table before quoting any ceiling, any MTP
> acceptance number, any re-bake cost, or any competitor per-GPU figure.**

### 🐛 OPEN DEFECTS
| Defect | Severity | Owner |
|---|---|---|
| ✅ **ROOT-CAUSED + FIXED (PR #126): the dedicated decode path is STRUCTURALLY BATCH-1 ONLY, and default-on.** `arc_launch_gemv_bf16_f32out(weight, input, output, M, K, stream)` has **no batch dimension** — `grid` is over output rows (vocab), not sequences. **Every projection in `decode_forward` — qkv, o_proj, gate, up, down, lm_head — is a matrix-VECTOR product.** Only the elementwise and attention kernels ever consumed `bs`. MEASURED, same binary, one env var: B=8 **71/512** tokens ON vs 512/512 OFF; B=32 **95/2048** vs 2048/2048; **guarded → 512/512 and 2048/2048.** 🔑 `71 = 64 + 7×1`, `95 = 64 + 31×1` — **two batch sizes agreeing on `64 + (B−1)` is a structural signature; bandwidth or scheduling cannot produce that formula, only "one sequence computed, the rest garbage" can.** Zero errors, zero CUDA errors, zero glibc diagnostics — it fails **silently**. Earlier theories (block-table stride, capacity pin) are real and necessary but were **not** this bug. ⚠️ SCOPE not verdict: the guard makes it *correct* (falls back); making it *useful* above B=1 needs a genuinely batched GEMM. ⚠️ **Residual 511/512 at B=1 — the batch-1 path is not bit-identical to Candle either.** Open, flagged, not rounded away. | 🔴 was stop-everything | **CLOSED** |
| V4 host heap corruption during capture (`corrupted double-linked list`) | 🔴 | **UNNAMED.** Pool-lifetime and the 172032 `xs` alloc both eliminated |
| `clone_in_cache` allocates+zeros a capacity-sized block per layer per step, then overwrites it. ~84 slots on V4. **+572 ms/step** | 🟠 perf | ladder chain |

### ⚠️ "CLOSED" session 8 — 4 defects, 10 files, +1264/−11, 478/0 core tests. THREE were WORSE than filed.
> 🔴 **CORRECTION 2026-08-18: THIS WORK IS NOT ON MASTER AND NOT ON ANY PUSHED BRANCH.**
> `mask_must_be_applied_as_bias`, `supersedes_the_pr122_predicate` and
> `default_responder_is_closed_from_birth` exist **nowhere in origin** — verified by ArcGate.
> It is **unpushed, in the local worktree `.claude/worktrees/agent-ade618abed0214978`
> (branch `worktree-agent-ade618abed0214978`), with no PR**, one `rm -rf` from gone.
> **Main recorded it as closed without checking it was pushed** — told other chains to open PRs
> for durability and never asked this one. Rescue is in flight. **Until a PR exists, treat every
> row below as WRITTEN BUT NOT LANDED.** The "deliberate name collision" with #122 is therefore
> a *future* event, not a live conflict.
>
> ✅ **RESCUED 2026-08-18 → PR #127** (`fix/sliding-window-encoders-phantoms`). It was **worse
> than unpushed: UNCOMMITTED** — 10 modified files in the worktree, HEAD on old master, nothing
> staged. Not one `rm -rf` from gone — **one `git checkout .`**. Verified +1264/−11 across 10
> files with all three markers present before committing.
>
> ⚠️ **AND MAIN'S "8 PRs HAVE ZERO CI LANES / the gating hole is still live" WAS ALSO WRONG** —
> corrected by ArcGate against the artefact. The workflow fix **IS on master**:
> `ci.yml` carries `pull_request: branches: ['**']` plus `ci-complete` (`if: always()`,
> `needs:` all 8 lanes) and a concurrency group; commits `91befb0cf` and `71485376d` proven
> ancestors of master via `git merge-base --is-ancestor`. The 8 one-check PRs are carrying
> **pre-fix STALE STATUS**, and get real lanes the moment anything is pushed — confirmed
> empirically by pushing to three and watching 16 lanes start each time.
> ⇒ **A stale-status problem, not a gating hole.** (ArcGate found the workflow commits in
> *another* leftover worktree and briefly assumed unpushed meant unlanded — the same inference
> that produced main's error. **Unpushed ≠ unlanded: check `merge-base --is-ancestor`, not the
> worktree.**)
>
> Still true: main's claim that **"#104's CI was green" was wrong** — it had 1 check and never
> ran. The hold is right for a better reason: **ungated**, not questionable.
| Was filed as | Actually was |
|---|---|
| "`qwen2.rs` ignores `use_sliding_window`, mostly inert at 32768" | **`Qwen2.5-7B-Instruct-1M` ships 32768 + `use_sliding_window:false` + 1,010,000 max_position ⇒ master clamps a 1M-context model to 32k.** Blast radius wider: the declared window also reached `PagedAttentionMeta`, **capping the KV budget**. The rename to `declared_sliding_window` is what surfaced the second site. |
| "extraction silently mis-indexed" | **Phi-2 emits 6 projections/layer, not 7** ⇒ master mis-assigns from layer 1 **then reads index 224 into a 193-entry vector.** The guard converts a **panic** into a named refusal. |
| "siglip + idefics3 attend over padding" | **FOUR models**, and the worst is not a padding mask: **Phi-4-multimodal's audio encoder loses a LEARNED T5 relative bias on EVERY forward.** A padding mask is wrong only with padding; a learned weight is wrong unconditionally ⇒ **every Phi-4 audio result from a `cuda flash-attn` build is suspect.** Also mllama. idefics2 confirmed unaffected. |
| "sequences never reaped" | **Never marked finished** — the engine only learns a client left by *failing to send*, and non-streaming sends nothing until done. Streaming works; non-streaming holds a slot to `max_tokens`. |

**The best find was a FIXTURE bug, not a defect:** `seq_of_len` bound its receiver to `_rx`
— a real binding that drops on return — so **every sequence the scheduler suite ever built had
an already-closed responder.** 100% of fixtures modelled a disconnected client; the suite could
not distinguish live from dead. A correct reap therefore looked like it *broke* a passing test.
**Tree-wide sweep found 1 more, in PRODUCTION** (`request.rs::default_responder`, using `_` so
it drops *immediately*) — safe only because both daemon replicators replace it, which was
**nowhere written down**. Now documented and pinned by `default_responder_is_closed_from_birth`,
so anyone "tidying" it trips a test instead of shipping a phantom.

**And "ZERO disconnect warnings" was never evidence** — the only such warning exists on an
*error* path. Two chains greped for a string that could not have been there. Compounding it:
`<N> running, <M> waiting` is **suppressed unless `total_new_seqs != 0 && tokens_processed != 0`**,
so an idle server holding phantoms prints nothing at all.

**Still open from that chain (neither is theirs to close):** the CUDA-gated edits ride the
`cuda-typecheck` lane — **count the checks, don't read the colour** — and the end-to-end decay
on a live `serve` is filed as an **open verification**, not folded into "done".
| `clone_in_cache` allocates+zeros a capacity-sized block per layer per step, then overwrites it. ~84 slots on V4. +572 ms/step | 🟠 perf | ladder chain |

---

## 4. THE SYSTEM NAMES — no subsystem may lack an absolute parent

```
Arc
├── ArcServe      HTTP/OpenAI, CLI, SDKs, MCP
├── ArcInfer      request -> tokens
│   ├── ArcSched      serving loop, admission, batching   <- both fixes above live here
│   ├── ArcKV         KV memory (Share/Paged/Dense/Xs/Fp8)
│   ├── ArcAttention  attention math + dispatch; ArcFlash = our fused d=512 kernel
│   ├── ArcSpec       speculative decoding (MTP, EAGLE-3)
│   ├── ArcMoE        MoE serving/routing/TD-MoE           <- 21% of marginal cost, UNOWNED
│   ├── ArcGraph      GPU-autonomous decode
│   ├── ArcSample     sampling
│   └── ArcBoost      training-free serving-side quality
├── ArcModels     NEW MODELS LAND HERE
├── ArcQuant      QTIP (weights) · TurboQuant (KV) · ArcBake (offline)   <- THE MOAT
├── ArcKernels    GPU substrate; ArcKernels/Trellis = the keystone GEMM; ArcTarget = new GPUs
├── ArcFormat     UQFF + ArcOverlay
├── ArcLab        profiler, benchmarks, ops tooling
└── ArcGate       correctness gates + release discipline
```
Full tree, what is shipped vs wired-but-dead vs nonexistent: **`TAXONOMY.md`**.

---

## 5. THE MOAT'S TWO HALVES — both idle until session 8, both now owned

**Attention (ArcFlash).** head_dim 512 now compiles (the cap was an **inherited
instantiation bound**: smem held at 32 KB by halving BC as HEAD_DIM doubles;
512/BC=8 continues the series). ⚠️ **CORRECTED 2026-08-19: "head_dim 512 is the
blocker" is INVERTED — 512 compiles; the width that FAILS is 448**
(`vec_size 14 % 8 ≠ 0`). Do not plan work around removing a cap that is not
there. **But V4 decode never reaches it** —
`dsv4_attention.rs:682` returns into `absorbed_mqa_decode` first. So the real
ArcFlash target is **two `MatMul`s + `softmax_with_sinks` with no shared-memory
stage to insert decompression into.** Rung 1 (read TurboQuant bytes) is a different
problem than "teach the fused kernel our format".

**Weights (ArcKernels/Trellis).** `qtip_grouped_gemm.cu` — "the keystone".
✅ **PHASE 1+2 DONE, PR #124: +41.6%/+41.3% cumulative, bit-identical, margins 2422×/850×.**
Handicap **1.76× → 1.12×**; crossover **B=64 → B=12**.

🔴 **CEILING STATEMENT — the kernel is now near its INSTRUCTION-COUNT FLOOR *for this
format*.** 32 decodes cost ~160 essential instructions of 672 per thread-iteration; the rest
is window extraction (`LOP3` 86, `SHF` 49), the scale `FMUL` 32, the pack `F2FP` 24, ~35
fragment-shuffle MOVs, 22 smem accesses. **The only two levers left are (a) the epilogue scale
hoist — one op/weight, but it breaks the bit-identical gate that makes every variant
checkable — and (b) a V=4 trellis, which is a FORMAT change and a re-bake.**

⇒ **FURTHER LARGE GROUPED-GEMM GAINS MUST COME FROM THE BYTE FORMAT, NOT THE KERNEL.** The
1.76× handicap was *implementation* and is now 1.12×; what remains is **the format's own
decode cost**, and no amount of kernel work reaches it. **This promotes V=4 from "deferred
optimisation" (BACKLOG) to the only large remaining lever on the keystone.**

✅ **AND V=4 IS NOW COMPILED, 2026-08-19 — the ladder is a measurement, not an estimate.**
K4/V2/L16 shipped **15.125** inst/wt (sm_90) → **K8/V4/L12 + row-scale hoist 4.375** =
**3.46× fewer instructions per weight**, LUT exactly 32,768 B (fits static smem, no
`cudaFuncSetAttribute`), occupancy 62.5% and **register-limited**. **Still 3.1–3.9× short of
the 1.13–1.41 budget** — that is scope, not a sentence (D21). Full table: `FACTS.md` §2026-08-19.
🔴 **THAT GEOMETRY IS QUALITY-CLOSED AT 2 bpw AND MUST NOT BE BAKED — the winning
rung is K9/V4/L12 at 2.25 bpw.** K8/V4/L12 costs **−0.00698** `w_cos` (band
±0.0008) and **nine codebook designs stop at −0.00307**; the measured reason is
**trellis freedom, not codebook coverage** (trellis OFF, LBG halves the loss vs
random; trellis ON, the same LBG loses to random). K9 buys **+0.00402** at the
**same 32,768 B table and same decode shape** for **0.25 bpw**, costing **KV 58.8
→ 49.6 GB (UNMEASURED as a batch/context effect)**. ⚠️ **K9's inst/weight is
UNMEASURED — the kernel has never been compiled at any K but 8. Do not quote
4.375 for K9.** Runner-up if capacity binds: **K4/V2/L13, 2.00 bpw, −0.00206,
11.250 inst/wt (1.34×)** — misses the band, costs no bit and no KV.
Full record: **`memory/mission/FRONTIER_BITS_FOR_DECODE.md`**.
🔴 **PRICE IT HONESTLY: the re-bake gets ~8× MORE expensive, not cheaper.** The production
baker is **beam, not exhaustive** (213 vs 8,257 s/layer on A100) and is **issue-bound**, so cost
`(n/V)×W×2^K` ⇒ **~213 → ~1,700 s/layer ≈ 20 h ≈ $30**. Affordable — but it **inverts** the
earlier recommendation, and *"the re-bake gets cheaper at K8/V4/L12"* is **RETRACTED**.
⚠️ **And none of it moves b=1**, which is 49% GPU-idle and launch-bound with the trellis GEMV at
only 29% of kernel time. **Format work pays at high concurrency; getting the CPU out of the loop
pays at b=1. Complementary, not substitutes.**

**Cost model VALIDATED IN BOTH DIRECTIONS** — the strongest evidence in the PR:
`v1→v2` instructions −3.4% / time −3.3% (near-exact 1:1 ⇒ instruction-issue-bound once
conflicts are gone); `v0→v1` instructions −22.3% / time −39.4% (**the excess IS the bank
conflicts — cycles burned without instructions**). `BREV 32→5` is the third leg. The *shape*
of the cost matches the claimed mechanism, measured independently.

⚠️ **TWO MISSES, SAME ERROR, BOTH SELF-REPORTED — reading an instruction mnemonic as though
it meant what its name suggests.** Phase 2 registered +12–20%, delivered **+3.3%**: `F2FP` was
**already 24 in the baseline** — nvcc always fused the pack, so that lever bought zero. Then
"~83 IMADs are address arithmetic, 12% of remaining" was written into the kernel comments and
the PR as the next lever — **it does not exist**: on Turing+ `IMAD.MOV` *is* the register-move
idiom, so ~35 of 115 are MOVs, only 13 are 64-bit `WIDE`, ~32 of ~59 true multiplies are the
codeword. Caught by **building the prototype and disassembling before predicting** — identical
SASS, reverted not shipped. Cost: a 12-second compile instead of a card.
🔑 **The rule that came out of it: prototype → disassemble → confirm the instructions actually
move → THEN predict.** Never cost C++ source as though it maps 1:1 to SASS.
🔑 **wgmma is NOT the lever** — the mma pipe is idle **93%** (33.5 TFLOP/s = 6.8% of
peak), so perfect wgmma recovers **<10%** against a 76% deficit. The kernel is
**ALU-issue-bound on decode** (~1.5e13 op/s vs H200's ~1.67e13 INT32-lane/s).
🔑 **And wgmma/tcgen05 CANNOT be used at all** until decode stops living in
registers — both require the B operand in SMEM/TMEM, and a register-synthesised
trellis B is unfeedable. Restructuring decode into SMEM is a **prerequisite**, not
an optimisation.

---

## 6. ORCHESTRATION — how to run this

- **MAIN ORCHESTRATES.** Dispatch agents; don't do their work. Jish has corrected
  this repeatedly.
- **Every agent gets its own worktree** — use the harness `isolation: "worktree"`
  flag, NOT a request in the prompt. Two agents in one file cost a card once.
  (One agent's worktree was never created; it silently used the shared checkout and
  HEAD moved under it. **Verify `git worktree list`.**)
- **An endorsement is a claim.** When main repeats an agent's finding it stops being
  tentative and becomes the project's position. Three theories reached builds that
  way in one session. **Check the arithmetic before amplifying.**
- Agents cannot always reach each other — **main relays**.
- **Never leave a card idle.** Release the moment you know you have code to write,
  not when you finish writing it.
- **Price a run before claiming a card**, against numbers already in FACTS.
- **Budget is the constraint, not engineering time.** Two H200s = $9.84/hr.
  A keystone sweep cost 15 min of card for <1 min of measurement until the bake
  cache landed — the instrument was **20× the measurement**.

---

## 7. THE DOCTRINE — D1–D32

`DOCTRINE.md` D1–D13 · `GPU_ACCESS_RULE.md` D14–D15 · `KERNEL_RULES.md` D16–D32.

The ones that keep costing money:
- **D18 — silent success is the house fault** (15+ instances). A green must prove
  work happened. Env failure exits **2**, never 1.
- **D21 — a scoping result is NEVER a verdict.** Doesn't-work-yet ⇒ build it and fix it.
- **D23 — "explains" ≠ "verified."** Name the observation that would differ if the
  explanation were false, then go look.
- **D29 — the thing you read is not the thing that is current.** Pin baselines as
  **literal SHAs** and prove the pair; `FIX~3` rots the moment you add a commit.
- **D30 — a null control gives the floor for THAT RUN, not the floor** (sampled
  0.67/2.45/2.54/1.30% in one session). Interleave `U F U N`, compare paired.
- **D32 — "NEUTRAL" IS THE SIGNATURE OF AN UNREACHED PATH.** Twice in one session.
  Prove execution before believing a null: launch counter, marker *inside the arm*,
  `reachable && calls > 0`.
- **A negative control proves the comparison is sensitive to its INPUTS. It says
  nothing about which code path produced them.**
- **Knowing about a footgun ≠ having removed it.** Delete the tool from the script.
  `pkill -f`/`pgrep -f` matched its own command line in **three** chains in one session.

---

## 8. INVARIANTS THAT COST MONEY

- Build `--features "cuda flash-attn"`. **NEVER `cudnn`** (−62% decode on V4).
- `nvcc` is at `/usr/local/cuda-13.1/bin/nvcc`, not on PATH. Driver caps at CUDA 13.0
  ⇒ `cuda-compat-13-1` + `LD_LIBRARY_PATH=/usr/local/cuda/compat`.
- **`-gencode arch=compute_90a,code=sm_90a`, then read the emitted `.target` back.**
  `nvcc` accepts `-arch=sm_90a` with no diagnostic and emits `compute_90`.
- **macOS `cargo check` does NOT type-check CUDA-gated Rust** (one PR shipped 15 such
  errors), and **`cargo check -p <crate>` does not build tests** — use
  `--workspace --tests`.
- `cargo fmt -p <crate>`, **never `--all`** (mass-reformats upstream files; fork policy).
- `setsid nohup … < /dev/null > LOG 2>&1 &`; a plain `nohup &` through ssh does not survive.
- **HF org is `aeonmind`; GitHub org is `aeonmindai`. THEY DIFFER.** Token at
  `~/.config/arc/env` (0600). Every bake harvest shreds the on-box token.
- Never `Tensor::{from_vec,arange}` in a hot loop — CPU→GPU sync.

---

## 9. READ ORDER (bounded — do not read everything)

1. **This file** — the SESSION-9 BLOCKER block and the SESSION-8 CLOSE block at
   the top are not optional.
2. 🔴 **`wave66-CS-session9-the-22-token-wall.md`** — **before quoting any
   single-user throughput number, and before proposing any decode
   optimisation.** V4 does not serve past ~22 tokens; the seven session-9
   findings; six retractions; and eight corrections to how they were first
   reported.
3. **`CENSUS_SESSION8.md`** — the complete Arc/SGLang/vLLM census (zero GPU
   hours) — and **`LADDER_POST_CENSUS.md`**, the GPU ladder reordered by it.
   **Read both before proposing any GPU spend.** *(Branch `docs/census-session8`
   until it merges.)*
4. `STATUS.md` — **top entry only** (reverse-chron; the file is 70 KB).
5. `FACTS.md` — only to look up a specific number. **Never reason from a number not in it.**
   Start at **§2026-08-21** (the blocker + six retractions), then **§2026-08-19**
   for the compiled trellis ladder.
6. `KERNEL_RULES.md` — when about to measure, build a kernel, or write a guard.
   **The top block is the session-8 close: seven rules, each paid for.**
7. `TAXONOMY.md` — when naming anything or asking "what am I working on".
8. `CEILINGS.json` — **before quoting any speed number or saying "not achievable"**.
   Separates PHYSICS bounds from IMPLEMENTATION gaps + the anti-pessimism protocol.
   ⚠️ **Its 16,600 at B=256 is a BANDWIDTH ceiling and assumes the format is free
   to decode.** It is a physics bound, not a target this build approaches — and it
   is **not** the capacity claim. See the headline block at the top of this file.
9. `BACKLOG.md` — surfaced-not-shipped debt.
10. `wave*-*.md` — per-agent deep logs, on demand only.
