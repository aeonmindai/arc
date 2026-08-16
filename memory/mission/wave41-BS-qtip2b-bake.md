# wave41-BS — the qtip2b bake was NOT run, because three of its four knobs are no-ops on that rung

**Spend: $0.00. No box was ever created; `runcrate ps` was empty at dispatch and is
empty now.** The session was stopped at the research gate, before renting, because the
bake recipe it was handed cannot do the thing it exists to do.

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

## 7. Corrections owed to the record

* **FACTS.md:283** — *"the next bake is `--isq qtip2b` + computed codebook + W=256"* is
  incoherent as written: `qtip2b` has neither a selectable codebook nor a beam. The
  crossover table above it is measured and stands; only the action line is wrong. A
  retraction has been added there.
* **`qtip/mod.rs:574`** — the `ARC_QTIP_CODEBOOK` error text says *"Use `mcg` (computed,
  **default**)"*, but `QtipCodebook::DEFAULT` is `Gaussian` (`mod.rs:564`). The message
  tells an operator the opposite of what the code does. One-word docfix, not touched here
  to keep this PR evidence-only.

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
