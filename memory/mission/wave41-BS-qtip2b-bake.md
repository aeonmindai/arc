# wave41-BS — three of the four bake knobs are no-ops on `qtip2b`; the bake then ran without them

**Part 1** (§1–8) is the pre-rental finding that stopped the original recipe for $0.00.
**Part 2** (§9–) is the corrected bake that followed it, on the rung wave29-BD chose,
with no codebook and no beam env var — because neither reaches this rung.

---

## Part 1 — why the original recipe could not be run

The session was stopped at the research gate, before renting, because the bake recipe it
was handed cannot do the thing it exists to do.

**The one-line finding:** `ARC_QTIP_CODEBOOK=mcg` and `ARC_QTIP_BEAM=256` are **read by
the `qtip2` rung and by nothing else**. A `--isq qtip2b` bake never calls the code that
reads them. The recipe `--isq qtip2b + computed codebook + W=256` (FACTS.md:283) names
two knobs that rung does not have.

This is not a re-litigation of the rung decision. wave29-BD's crossover measurement
stands untouched and `qtip2b` is still the right rung. What does not stand is the claim
that this session could *activate PR #46* on it.

---

## 1. The two rungs, and which one PR #46 landed on

They are different trellis geometries, not two spellings of one thing.

| | `qtip2` (`IsqType::QtipBitshift2`) | `qtip2b` (`IsqType::Qtip2b`) |
|---|---|---|
| source | `qtip/mod.rs` (`QtipLayer`) | `qtip/bitshift.rs` (`Qtip2bLayer`) |
| trellis | **K=4 / V=2** (16 predecessors, pairs) | **K=2 / V=1** (2 predecessors, scalars) |
| codebook | 65,536×2 Gaussian LUT, **stored in the artifact** | computed MCG, **no table exists** |
| codebook selectable? | **yes** — `ARC_QTIP_CODEBOOK` | **no** — `QTIP2B_MCG_MULT` is hardcoded |
| beam search? | **yes** — `ARC_QTIP_BEAM=W` | **no** — exhaustive DP only |
| PR #46 (`sum2`) | **this rung** | untouched, by wave24-AU's own words |

wave24-AU said it plainly and it was correct: *"`qtip_grouped_gemm.cu` is the **qtip2b**
rung and never read this table; it is untouched."*

`sum2` is a **V=2** construction — it emits a *pair* `(fold(x·m), fold(x·m²))` per state.
`qtip2b` is **V=1**. The codebooks are not interchangeable, and no flag could make them
so. This is a geometry mismatch, not a missing wire.

## 2. Verification — the call graph, not the comments

`ARC_QTIP_CODEBOOK` has exactly one reader, and it is reachable from exactly one rung:

```
ARC_QTIP_CODEBOOK
  └─ QtipCodebook::from_env()                       qtip/mod.rs:570
       └─ QtipBakeConfig::get()                     qtip/mod.rs:932   ← sole caller
            ├─ QtipLayer::quantize_*                qtip/mod.rs:1613, 2092   [qtip2]
            └─ UnquantLinear::apply_isq             unquantized/mod.rs:395   [reads .hessian only]
```

`grep -n "QtipCodebook\|from_env\|QtipBakeConfig\|ARC_QTIP" qtip/bitshift.rs` returns
**two hits, both `ARC_QTIP_EXPERT_BATCH`**. `Qtip2bLayer` never reaches
`QtipBakeConfig::get()`. Instead every one of its quantize sites passes the constant:

```rust
pub const QTIP2B_MCG_MULT: u32 = 0xCAF6_A435;   // bitshift.rs:112
```

used at `bitshift.rs:461, 617, 634, 693` — the 2-D door, the CUDA door, the 3-D expert
door, and the deserializer.

The ISQ dispatch confirms the split at the top: `unquantized/mod.rs` has two separate
arms, `Some(IsqType::QtipBitshift2) => … QtipLayer::quantize_with_calibration(…)` and
`Some(IsqType::Qtip2b) => …`. A `qtip2b` run constructs **no** `QtipLayer`, so the env
var is not even partially live.

**Consequence, stated precisely:** setting `ARC_QTIP_CODEBOOK=mcg` on a `--isq qtip2b`
bake changes nothing about the artifact. The bake would have succeeded, the artifact
would have been valid, and the session report would have claimed "computed codebook
activated" — true of the rung by construction, but with **zero** of PR #46's measured
3.86–4.03× fused-GEMV win engaged, because that win lives on the K=4/V=2 kernels a
`qtip2b` artifact never executes. That is the failure mode the brief was trying to
prevent, arriving through a door it did not check.

## 3. `qtip2b` already ships a computed codebook — there is no Gaussian to bake by accident

The good news, and it is genuinely good: the risk the brief was guarding against does not
exist on this rung. `bitshift.rs:1` — *"QTIP 'bitshift trellis' with a **computed
codebook** — the `qtip2b` rung"*; `bitshift.rs:54` — *"Unlike the LUT rung there is **no
codebook tensor**: only the 4-byte MCG"*.

There is no Gaussian variant of `qtip2b`. A `qtip2b` artifact is computed-codebook or it
does not exist. So "a bake that silently used Gaussian is a wasted rental" is
unfalsifiable-by-construction here, which is exactly why it needed checking rather than
asserting.

## 4. Beam W=256 is also a no-op on this rung — and it moves the schedule

`bitshift.rs:344-349`, on the `search_detail` field:

> *"This rung's `viterbi_quantize_row_2b` is the exhaustive DP with an unweighted branch
> metric and **has no beam**, so a bake here always records
> `QtipSearchDetail::EXHAUSTIVE_MSE`; the field exists so both rungs share ONE wire
> format rather than diverging the moment qtip2b grows a beam kernel."*

Stamped unconditionally at `bitshift.rs:568, 637, 760`. `ARC_QTIP_BEAM` is parsed by
`TrellisSearch::from_env()` (`viterbi.rs:147`), reached only via `QtipBakeConfig::get()`
— the same door `bitshift.rs` never opens.

**This is the schedule risk, and it is why stopping was the cheap move.** The brief's
"~83 s/layer ⇒ 43 layers ≈ 60 min" derives from FACTS.md:582, which is explicitly a
**W=256 vs W=32 beam** comparison — a `qtip2` number. `qtip2b` bakes exhaustive. Its
exhaustive DP is cheaper per step than `qtip2`'s (2 predecessors vs 16), so it is *not*
the 510 s/layer exhaustive figure either, but **no measured `qtip2b` per-layer bake rate
exists in FACTS.md** — grep returns only the crossover table, the action line, and a
parity count. The 3-hour self-destruct and the ~$15 budget were sized from a rate
measured on the other rung. Renting against an unmeasured bake rate is how a box runs
past its deadline with a half-finished artifact.

## 5. The D4 gap the brief flagged is already closed — better than by a log header

The brief asked for an independent verification of search mode because "`qtip2b` bakes
emit NO bake header." The log header is the weaker instrument anyway; the artifact
carries the stamp:

```rust
/// Which trellis search produced these blocks. Serialized into UQFF from
/// 0.3.0 and checked at load (DOCTRINE D4 §3).
search: QtipSearchStamp,          // bitshift.rs:341-342
```

Three independent teeth, none of which is a log line:
1. **The production door hard-errors on greedy** — `mode.deny_greedy("Qtip2bLayer::quantize_with_mode")` (`bitshift.rs:365`), in every build, not behind `cfg(test)`.
2. **The mode is not selectable** — ISQ dispatch hard-wires `QtipMode::default_expert_mode()`, a `const fn` (`mod.rs:853`). `mod.rs:811` documents the whole table: *"ISQ dispatch (`--isq qtip2` / `qtip2b`) | never — hard-wired."*
3. **The search is stamped into the UQFF and checked at load**, so the verification is `read the artifact back`, not `trust the log`.

So the answer to "how do you verify search mode without a bake header" is: **read
`search`/`search_detail` off the deserialized layer** (`Qtip2bLayer::search_detail()`,
`bitshift.rs:1402`). That is a stronger check than the header the brief was worried about
missing, and it is checkable by anyone holding the artifact.

## 6. What each goal actually requires

The two goals in the brief are **mutually exclusive**, and neither is wrong — they just
cannot be the same bake.

| goal | command | what you get | what you lose |
|---|---|---|---|
| **A. Ship wave29-BD's rung decision** | `--isq qtip2b` (no codebook/beam env) | the grouped-GEMM kernel that scales past B=64; computed codebook by construction; exhaustive search stamped | PR #46 not engaged (wrong geometry) |
| **B. Activate PR #46 (`sum2`, 3.86–4.03× GEMV)** | `ARC_QTIP_CODEBOOK=mcg ARC_QTIP_BEAM=256 --isq qtip2` | the first real-model bake on the computed codebook | the grouped kernel; contradicts wave29-BD |

**A is the one to run**, on the brief's own reasoning: the rung decision is measured and
the fleet argument (D1/D2) rests on the grouped kernel scaling, which is `qtip2b`. Goal B
is a separate, still-unpaid debt — wave24-AU's *"the flip should follow one real bake"* is
about the **`qtip2`** rung and remains true and unaddressed.

**Before A is rented, one number is missing: a measured `qtip2b` s/layer.** Everything
else in the runbook (source dir, overlay form, `ARC_QTIP_EXPERT_BATCH=8` for the
fragmentation OOM, the known-benign `device mismatch in matmul` smoke failure, the upload
diff) transfers unchanged.

> **Part 2 measured that number and it changes the recommendation.** `qtip2b` costs
> **≥ 984 s/layer** ⇒ **≈ 11.75 h** for 43 layers, because this rung has no beam kernel
> and the 83 s/layer figure was a `qtip2` **beam** number. A is therefore not runnable
> today either. See §9–12; the unlock is a beam kernel for `qtip2b`.

## 7. Corrections owed to the record

* **FACTS.md:283** — *"the next bake is `--isq qtip2b` + computed codebook + W=256"* is
  incoherent as written: `qtip2b` has neither a selectable codebook nor a beam. The
  crossover table above it is measured and stands; only the action line is wrong. A
  retraction has been added there.
* **`qtip/mod.rs:574`** — the `ARC_QTIP_CODEBOOK` error text said *"Use `mcg` (computed,
  **default**)"*, but `QtipCodebook::DEFAULT` is `Gaussian` (`mod.rs:564`). The message
  told an operator the opposite of what the code does — in the one place an operator
  reads it, having just mis-set the variable. **Fixed in this PR**: the message now marks
  `gaussian` as the default.

## 8. Honest limits of this finding

* **Static analysis, no GPU.** Every claim is a call-graph or constant read from the tree
  at `372976933`, cited by file and line. Nothing here was executed. The claim "the env
  var is a no-op on `qtip2b`" is a statement about which functions read it, and that is
  the kind of claim static reading settles — but it was not confirmed by running a bake
  under both settings and diffing the artifact, which is the experiment that would close
  it beyond argument.
* **It does not say `qtip2b` is slow.** No `qtip2b` per-layer bake rate is asserted here,
  only that none is on record and that the 83 s/layer figure belongs to the other rung.
* **It does not weaken wave29-BD.** The crossover stands; the rung choice stands.

---

# Part 2 — the corrected bake ran, and `qtip2b` turned out to be un-bakeable at V4 scale

**ABORTED at the measurement gate. Spend $2.47. Box deleted, `runcrate ps` empty.**
Nothing was published; `aeonmind/DeepSeek-V4-Flash-UQFF-qtip2b` was never created.

The corrected recipe from Part 1 — `--isq qtip2b`, no `ARC_QTIP_CODEBOOK`, no
`ARC_QTIP_BEAM` — ran on a clean H200 and produced the number nobody had:
**`qtip2b` costs ≥ 984 s/layer**, against the ~83 s/layer the plan was sized from.

## 9. The missing number, and why it was missing

| | |
|---|---|
| box | `arc-s11-qtip2b`, 1×H200 141 GB, Helsinki, $4.85/hr |
| created / running / **self-destruct armed** | 03:39:07Z / 03:45:05Z / **03:45:09Z** |
| build (`cuda flash-attn`, **no cudnn**) | 8 m 45 s |
| source pull (159.63 GB, 73 files) | done 03:58:25Z |
| bake launched | 03:58:36Z |
| layer-0 marker | 03:58:49Z |
| killed (by PID, never `pkill -f`) | 04:15:13Z |
| **layer 0 elapsed** | **≥ 984 s (16.4 min), had NOT finished** |
| markers observed | **1 of 43** |

**Projection: ≥ 984 s × 43 = 42,312 s ≈ 11.75 h ≈ $57 of H200 at $4.85/hr** — about
6× the 2 h abort threshold and ~4× the entire remaining balance. Aborted.

The box was healthy the whole time, which is what makes the number trustworthy rather
than a box-fault story: **99–100% GPU util, 240–274 W, SM clock pinned at 1980 MHz
(full boost), 40 °C**, `0` occurrences of fallback / panic / OOM / CUDA error, and
`cpu_reroute=0` — so this is not wave6-Q's silent CPU reroute, which would have cost
~20× and looked similar from a distance.

## 10. Why: `qtip2b` has no beam, and the 83 s/layer figure was a beam number

This is the same root cause as Part 1, arriving on the cost axis instead of the
correctness axis.

* `qtip2b`'s search is `viterbi_quantize_row_2b` — the **exhaustive DP over all 2^16
  trellis states**. `bitshift.rs:344-349` says it outright: this rung *"has no beam."*
* `qtip2`'s **83 s/layer is a beam W=256 measurement** (FACTS:582, a W=256-vs-W=32
  comparison). Beam prunes 65,536 states to 256 — a ~256× cut in the DP's inner loop.
* FACTS already recorded the size of that lever on the sibling rung: **exhaustive
  510 s/layer vs beam 241 s/layer** (pre-#40).

So sizing a `qtip2b` bake from a `qtip2` beam number understates it by more than an
order of magnitude. **And this explains the gap Part 1 could only name:** no measured
`qtip2b` s/layer existed in FACTS because no `qtip2b` bake at V4 scale has ever
completed — not because nobody wrote the number down.

⇒ **The rung wave29-BD selected for its serving-side scaling is, today, gated behind a
~12 GPU-hour bake.** The crossover measurement is untouched and still says `qtip2b` is
the right serving rung. What this session adds is that **shipping it requires a beam
kernel for `qtip2b` first.** The codebase already anticipates exactly that — the
`search_detail` field exists, in its author's words, *"so the two rungs cannot drift
apart the moment qtip2b grows a beam kernel."* That kernel is the unlock, and it is a
port of an already-working one (`qtip_beam.cu`), not new research.

## 11. Verifications that did land

* **Search mode, without a bake header.** Part 1's method was built and proven, though
  the artifact it was to read never existed. `.scratch_stamp.py` reads the last two
  bytes of each layer payload straight out of the UQFF safetensors container —
  `[stamp][flags]`, where `0x01 0x00` = trellis / no-beam / unweighted-MSE
  (`bitshift.rs:1775-1791`, `mod.rs:1134-1139`). It was **smoke-tested on both arms**
  offline: a clean fixture reads `STAMP_OK`, and a planted `0x02` greedy stamp reads
  `STAMP_FAIL … GREEDY-BANNED`. The negative control is the half that matters.
* **The env was clean, read from the process itself.** `/proc/<pid>/environ` of the
  live bake contained exactly one Arc variable: `ARC_QTIP_EXPERT_BATCH=8`. No
  `ARC_QTIP_CODEBOOK`, no `ARC_QTIP_BEAM` — Part 1's finding applied, and verified from
  the running process rather than from the launch script.
* **Thread policy came from the code, not from an env override.** The log reads
  `ISQ thread policy: 1 thread(s) — QTIP quantize runs in GPU kernels on one device`.
  `MISTRALRS_ISQ_SINGLETHREAD` was deliberately **not** set; the policy table
  (`lib.rs:1133-1145`) already picks 1 for `Qtip2b` on a GPU backend.
* **The good artifact was not touched.** `aeonmind/DeepSeek-V4-Flash-UQFF-qtip2`
  measured `sha=108fbc2e3dbae0103010c81d7418b99255dfbf99`,
  `lastModified=2026-08-15T10:40:02Z`, 15 files **both before and after** this session.

## 12. Honest limits of the ≥984 s number

* **It is a lower bound on ONE layer, not a completed interval.** Only 1 of 43 markers
  appeared, so nothing was differenced — the discipline the brief asked for could not be
  applied for want of a second marker. Layer 0's 984 s also *includes* whatever one-time
  init the first layer carries, and this run cannot separate init from steady state.
  A steady-state `qtip2b` s/layer is still unmeasured; what is established is that it is
  **not ~83 s**, and that the total is far past 2 h.
* **`ARC_QTIP_EXPERT_BATCH=8` vs the default 16** halves the batch, so 32 launches per
  layer instead of 16. The total row count quantized is identical and the kernel is
  row-parallel over 8×2048 = 16,384 rows per launch (already enough to fill an H200), so
  this should be a small effect — but it was **not measured**, and it is stated rather
  than dismissed.
* **One box, one run.** No repeat, no second silicon.
