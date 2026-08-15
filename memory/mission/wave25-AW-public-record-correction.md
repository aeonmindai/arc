# Wave 25 — AW: public-record correction (1.21× upper bound, GSM8K provisional)

**Scope:** documentation only. No Rust, no CUDA, no Python, no tests. No GPUs
rented, no ssh. Branch `docs/correct-kernel-stack-bound`, draft PR.

## 1. The defect

PR #43 (`docs/engineering-record`, 1,304 lines under `docs/engineering/`) merged
as `a71a9c046`. PR #44 was opened to correct the 1.21× claim inside it. Its
metadata:

```
number: 44
state: CLOSED, mergedAt: null
headRefName: fix/qual-1.21x-upper-bound
baseRefName: docs/engineering-record     <-- the defect
createdAt: 2026-08-15T00:09:20Z
closedAt:  2026-08-15T00:17:36Z
```

`baseRefName` is `docs/engineering-record` — PR #43's own branch, not `master`.
When #43 merged, GitHub auto-closed #44 because its base ref ceased to exist.
It was **not** a deliberate rejection. **Its correction never landed.**

## 2. Was PR #44's diff recoverable?

**Yes, fully** — `gh -R aeonmindai/arc pr diff 44` still returns it, and the PR
body survives with the reasoning. The diff touched exactly one file,
`docs/engineering/QUANTIZATION_PERFORMANCE.md`, in three hunks:

| hunk | line | change |
|---|---|---|
| 1 | 142 | table row `banked today (1.21× stack)` → `(**≤1.21×** stack — upper bound, see caveat)` |
| 2 | 305 | heading `Measured 1.21×` → `Measured ≤1.21×`, plus an 18-line blockquote caveat |
| 3 | 384 | PR-list entry `#40 (the measured 1.21 × kernel stack)` → `(the ≤1.21×…)` |

The recovered diff was applied as written, then **extended** on three points the
original did not make forcefully enough:

- PR #44 called the decisive experiment "the cheap authoritative test" — implying
  pending-and-fine. It is now stated flatly: **it has not been run, and until it
  runs the shipped value is unknown.**
- PR #44 left the derived `186 s / 202 s/layer` row untouched. Those numbers are
  `225.2 / 1.21`, so if the ratio is an upper bound the times are **lower**
  bounds. Changed to `≥186 s` / `≥202 s/layer` with an explicit best-case note.
- Added an `[upper bound]` row to the document's evidence-grade table. That
  document opens with "Every number below carries one of these labels. Nothing is
  stated without one," and had no grade that could express a bound.

## 3. The truth as now recorded

- The 1.21× came from PR #40's stack: F1 short selection key + F2
  `__launch_bounds__(256,4)` + F3 barriers 33 → 15.7.
- It was measured under **nvcc 11.5 / sm_80**, against a production build on
  **CUDA 12.8**. The old compiler produced a more register-starved *baseline*
  (`REG:110` ⇒ 2 blocks/SM) than the shipped one (`REG:80` ⇒ 3 blocks/SM), so the
  register squeeze was handed **more** headroom than production has (2→4 rather
  than 3→4) — and still only reached 1.21×.
- ⇒ **1.21× is an upper bound on the shipped gain, not a measurement of it.**
  ≤1.21×, possibly much less, never measured on the production toolchain.
- The A30-vs-A100 register difference was written up as a **card** difference. It
  is not: both are **sm_80** and share codegen for a given toolkit. It is a
  **compiler version + target arch** difference.
- The three parts of the stack were never separated; any one may be a no-op.
- Decisive experiment (build the existing bake box at the merge's first parent,
  time one layer against the 373.6 s/layer it already measures with the stack in
  — same silicon/toolkit/driver/data, zero rental): **staged, not run.**

## 4. Files changed (8, all markdown)

| file | what |
|---|---|
| `docs/engineering/QUANTIZATION_PERFORMANCE.md` | PR #44's 3 hunks + `[upper bound]` evidence grade + register-counts-are-a-compiler-property note at the occupancy measurement + `2 → 4` restated as the measurement box's range |
| `docs/engineering/OPEN_QUESTIONS.md` | occupancy-exhausted claim caveated; new §2 item 5 = the unrun decisive experiment; new §3 entry "ratios do not transfer between compilers" |
| `docs/engineering/HARDWARE_LESSONS.md` | removed "occupancy" from the list of things that transfer between cards; added the exception rule with the PR #40 story and a "record `nvcc --version` and `-arch`" gate |
| `docs/BENCHMARKS.md` | GSM8K row marked PROVISIONAL; new banner with the PR #35 reason + full protocol; ladder row and limitations bullet flagged |
| `README.md` | 84.0% → 87.0% provisional; 5.4 → 14.58 tok/s; ≈$77 → ≈$123; 8-shot noted on 90.8 |
| `docs/FLEET.md` | two GSM8K 87.0% "Measured" tags → "Measured — provisional" |
| `docs/RELEASE_NOTES_v2.0.md` | three GSM8K sites flagged; new caveats bullet with the PR #35 mechanism |
| `docs/notes/release-checklist.md` | stale README item closed; new gate: do not tag with provisional labels stripped and no re-measure |

**1.21× claim: 3 distinct instances found, all in
`docs/engineering/QUANTIZATION_PERFORMANCE.md`, all fixed.** The grep
`1\.21\|1\.2x\|launch_bounds\|A30\|register` over `docs/` plus a repo-wide
`--include='*.md'` sweep found no fourth instance (the only other `1.21` hit was
`research/moe_speculative_decoding.md:102`, an unrelated arithmetic
intermediate `3.25 / 1.217`).

**Card-vs-compiler error beyond the known spot: yes, one more, and it was the
load-bearing one.** `docs/engineering/HARDWARE_LESSONS.md:23` listed
**occupancy** among the things that "transfer between cards" — the general rule
that licensed the specific error. Fixed. `OPEN_QUESTIONS.md:112`'s bare
"2 → 4 blocks/SM" also inherited the unrepresentative baseline and is now
caveated.

## 5. GSM8K 87.0 / README 84.0

`BENCHMARKS.md` carried 87.0% unflagged. It is provisional: **PR #35**
(`830a41ed9`, merged 2026-08-14T13:55Z — after the session-3 run) changed the
decode math. Verified against the PR #35 body:

- `swiglu_limit: 10.0` is published in the model's own `config.json`, so the
  reference clamps unconditionally. Arc clamped on **1 of 5** expert paths. The
  four unclamped ones included the **shared expert**, which every token traverses
  in every layer, unweighted. Fixture magnitude: clamped 0.7311 vs unclamped
  14.8996 = **20.4×**.
- YaRN was applied to ratio-0 layers that must not receive it; correct affected
  set is exactly **{0, 1, 43}** (the audit's "0, 1, 42" was wrong;
  `compress_ratios[42] == 4`).

PR #35's own body says it plainly: *"the published GSM8K 87.0% / PPL 12.50 were
measured on different math… Both need a re-measure before any number is
republished."* Direction expected neutral-to-better, **unmeasured on the real
model**.

Protocol now stated everywhere the number appears: **n=100, 0-shot chat, greedy
(t=0), seed 161, 2048-token cap**. The reference **90.8** is now annotated
**8-shot EM — a different and easier protocol** at every site.

README's **84.0%** was a superseded session-2 number. Reconciled to 87.0%
(provisional). The same README rows were frozen at session 2 on two further
numbers — **5.4 tok/s** (BENCHMARKS: 14.58) and **≈$77** (BENCHMARKS: ≈$123) —
both stating a superseded value as current-measured. `release-checklist.md:38-40`
already tracked all three as one pre-tag item, so all three were reconciled
together and the checklist item closed.

Perplexity 12.50 and the long-context rows carry the identical PR #35 vintage
(`OPEN_QUESTIONS.md` §4 already said so internally). Flagging only GSM8K would
have implied the others were clean, so the banner names all three. No number was
changed — only labeled.

## 6. Surfaced, not fixed — out of scope

- **`docs/PEAK_INFERENCE.md:7`** states *"Current state: 33 tok/s (Arc), 60 tok/s
  (SGLang), 43 tok/s (vLLM). All on same hardware, same model, same benchmark."*
  as bare fact. `OPEN_QUESTIONS.md` §3 says the opposite in writing: *"**An
  in-class baseline.** Without a same-box, same-model comparison against an
  established serving stack, any $/Mtok figure floats free."* The whole file
  carries **zero** evidence labels (`grep -c 'measured\|projected'` → 0), and its
  section headings (`33 → 90+`, `90 → 110+`, `110 → 120 tok/s`) read as states
  rather than targets. Highest-value remaining D9 defect in `docs/`.
- **`docs/CUDA_GRAPH_PLAN.md:5,277`** — *"No existing system does this"* and
  *"Better than every existing system — vLLM, SGLang, TRT-LLM all have >0μs
  per-step CPU cost."* Competitive claims about third-party internals with no
  source citation and no label.

## 7. Gates

Markdown only, so no cargo gates apply. `git status` shows 8 modified `.md`
files and nothing else. No `cargo fmt` / `rustfmt` was run anywhere.
