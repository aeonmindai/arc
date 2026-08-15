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

---

# EXTENSION (same PR #49) — the fabricated head-to-head

Coordinator extended scope after item 6: `docs/PEAK_INFERENCE.md:7` is a worse
defect than the 1.21×. Governed by **DOCTRINE D3 (revised 2026-08-15 by
wave25-AV)** and **D9**.

## The defect was larger than reported

`PEAK_INFERENCE.md:7` published:

> *"Current state: 33 tok/s (Arc), 60 tok/s (SGLang), 43 tok/s (vLLM). All on
> same hardware, same model, same benchmark."*

I reported this as a fabricated competitor comparison. On investigation it is
**two** fabrications, and the second was not in the brief:

1. **The competitor half.** We have never benchmarked SGLang or vLLM.
2. 🔴 **Arc's own 33 tok/s is equally unmeasured.** Verified: `grep -rni
   'B200\|Qwen3-32B'` across all of `memory/mission/` returns **zero** hits in
   `FACTS.md` and zero in every wave log. **No B200 was ever rented.**
   `memory/project_cuda_graph_findings.md` says "Ready for single deploy to
   B200" — i.e. never deployed. So the "Arc" column was as invented as the other
   two, and the sentence "All on same hardware, same model, same benchmark"
   describes a run that never occurred in any column.
3. Line 264 cited the harness: *"Measurement: `deploy/benchmark.py` … on Modal
   B200."* **`deploy/` contains only `modal_b200.py`.** The named measurement
   script does not exist in the tree.

Arc's only measured decode figure is **14.58 tok/s**, on a different model and a
different card (V4 Flash, 1×H200, b=1, no-`cudnn`). It does not transfer to
Qwen3-32B/B200 and is explicitly marked as non-transferable in the rewrite.

## What replaced it (D3-compliant — deleted, not re-estimated)

No substitute competitor row was invented. The number is **gone**, and in its
place:

- A **document-level banner**: this is a PLAN, not a results page; nothing here
  has been run; the cited harness does not exist.
- An **evidence-grade table** matching the rest of `docs/`, with a new
  **`[target]`** grade — *"a goal we are designing toward. Not a measurement and
  not a forecast."* The file previously had **zero** labels (`grep -c
  'measured\|projected'` → 0) while every sibling doc claims universal grading.
- An explicit **"What this document does NOT claim"** section naming the deleted
  sentence verbatim, so the retraction is discoverable by anyone who saw the
  original.
- The **footprint argument** (D3): no single-GPU baseline exists on any engine
  for V4 Flash — native ckpt ≈160 GB, smallest published config **4×H200**, sole
  W4A16 quant 143 GB whose own card says *"TP=1 OOMs on a single 141 GB H200"*,
  NVFP4 Blackwell-only; Arc's ~68 GB (≈1.9 bits/param) is what makes 1×H200
  possible at all. One GPU vs a published four, plus $/Mtok per node.
- The **third-party-free roofline**: 68 GB ÷ 4.8 TB/s = 14.2 ms/step ⇒ ~4,500
  tok/s at B=64 ⇒ the measured 63.5 ms grouped-GEMM microbench is at **~22% of
  roofline**. Refutable, ours, needs nobody else.
- Headings changed from achieved-state to target form: `33 → 90+ tok/s` becomes
  `Phase 1: CUDA Graph Capture — target 90+ tok/s [target]`, same for phases 2
  and 3.
- The **Benchmark Targets** table's `Current (Arc) | 33 | ~22ms | 26%` row is
  replaced by an explicit "**Never run.** No B200 rental exists" row, with the
  closing note that *the first honest step for this document is to produce its
  own baseline row*, since without one every speedup ratio in the table is
  unfounded.

## CUDA_GRAPH_PLAN.md

`:5` *"No existing system does this. vLLM, SGLang, and TensorRT-LLM all launch
graphs from the CPU each step (~10μs)"* and `:277` *"Better than every existing
system"* — uncited claims about third-party internals never inspected. Reduced
to what we know about our own implementation: Arc's loop uses CUDA 12.4
**conditional graph nodes**, so no per-step host round-trip is required **by
construction**. The retracted sentences are quoted in place so the correction is
visible. `FACTS.md` has **no** CUDA-graph decode numbers at all, so the "What
this achieves" list — which read as achieved outcomes — is retitled "What this
is designed to achieve" with every bullet marked `[target]`, and closes by
stating that the number which would settle any comparison is our own per-step
CPU cost on hardware, unproduced.

## Not changed (checked, correctly framed)

- `README.md` Roadmap phases 3–4 ("Benchmark wins on GSM8K + HumanEval vs FP16
  reference", "B200 / NVFP4 path") sit under an explicit **Roadmap / "In rough
  order"** heading — aspirational by construction, not presented as measured.
- `QUANTIZATION_PERFORMANCE.md`'s "Where Arc sits against other trellis
  quantizers" table is third-party bake rates already graded **[published]**,
  with "no incumbent to catch" stated plainly. Correct as-is.

## Repo-wide sweep for other competitor / superiority claims

Nine more found beyond the two in the brief. Fixed:

| # | site | defect | fix |
|---|---|---|---|
| A1 | `README.md:36` | "AA-AgentPerf-style benchmark suite … **side-by-side vs SGLang/vLLM**" advertised as a shipped feature. Harness *can* target them (`arc-bench/src/lib.rs:10`) but **no run ever happened** — direct contradiction with the PEAK_INFERENCE retraction | reworded to "supported, but **we have not run one**" |
| B1 | `README.md:22` | "the **most aggressive open compression stack in production**" — superlative over every other open engine, in the repo's headline sentence, with nothing anywhere inspecting another engine's stack | → "composing published compression research end-to-end" |
| A7 | `README.md:32` | TD-MoE **"Lossless 20%"** — a third-party paper's headline number presented as an Arc feature, unattributed | attributed as the paper's (NeurIPS'25), *published, not reproduced by us* |
| B9 | `README.md:193` | Roadmap "Benchmark **wins** on GSM8K + HumanEval" — a future result stated as a deliverable | → an open question / hypothesis, not a scheduled result |
| A2/B3 | `QUANTIZATION_PERFORMANCE.md:401-404` | QTIP's **~14 min/layer on an A100** set against "our ~4 min/layer on a 25× larger model" — cross-card, cross-model, no shared benchmark — then "advancing the **state of the art**" | states plainly the two **cannot be divided**; quoting a ratio would be the same fabricated head-to-head banned elsewhere. Consequence narrowed to: no external number to tune against, so targets come from our own roofline |
| B2/A3 | `QUANTIZATION_PERFORMANCE.md:389-391` | "There is **no incumbent to catch**", under heading "Where Arc sits **against** other trellis quantizers", with Arc's bolded row in the same column as EXL3's — reads as a leaderboard even though each cell is graded `[published]` | heading → "What other trellis quantizers **report about themselves**"; explicit "**not a leaderboard, rows are not comparable**" preamble |
| B5 | `OPEN_QUESTIONS.md:42-44` | "**No literature covers this trade**" + "the **only** prior art" labeled `[published survey]` — but `wave20-AQ` is *our own agent log*, not a published survey. Two universal claims about a field, self-sourced | relabeled "survey **by us** … *not* a published survey; **absence of evidence, not evidence of absence**", claims narrowed to "we found" |
| A4 | `OPEN_QUESTIONS.md:261` | "**64×** the indexer key memory" vs "a comparable implementation" — unnamed third party, unlabeled | named (SGLang's V4 impl) and graded `[source-verified …, not benchmarked]` with the audit line cite that already existed |
| B6 | `QUANTIZATION_PERFORMANCE.md:295`, `OPEN_QUESTIONS.md:132` | "**Nobody** has priced that half" — means *we* have not, but written as a claim about the field | → "**we** have not priced" |

### 🔴 The sweep also caught my own new PEAK_INFERENCE text — and was right

Two defects in the replacement I had just written for the fabricated table:

1. It landed on a **universal negative** — "no single-GPU baseline on **any**
   engine … for **anyone else**" — derived from a handful of unnamed model
   cards. Rewritten as a scoped table of *what we actually checked and what it
   said*, graded `[published … survey by us, not exhaustive]`, with the claim
   restated as "we searched and found no published single-GPU configuration",
   explicitly **absence of evidence, not evidence of absence**.
2. It argued for the footprint claim partly *because* "unlike a number it
   cannot be refuted by rerunning it" — i.e. recommending a claim on grounds of
   **unfalsifiability**, which inverts D9. Deleted. Replaced with what *would*
   refute it (anyone producing a single-GPU config for this model) and the
   principle that a claim worth publishing states what would falsify it.

This is the failure mode the whole PR is about, reproduced inside the fix for
it, and it is why the sweep was run by a second pair of eyes rather than by the
agent that wrote the text.

### Found, deliberately NOT fixed — upstream-inherited, out of fork scope

- `docs/UQFF.md:4` — *"The uniquely powerful quantized file format."*
- `docs/QUANTS.md:7` — *"Automatic selection to use the **fastest and most
  accurate** method."*

Both are upstream mistral.rs marketing prose describing upstream features, not
Arc claims or competitor comparisons. Per fork policy (do not churn upstream
files) they are reported, not edited. Worth a decision if Arc is going to own
these pages at publish time.

### Checked and clean

`BENCHMARKS.md` (the 90.8 handling and the "no 'beats q2k' claim yet" are model
honesty), `FLEET.md` (all rows tagged; "~8× node aggregate vs 1×TP8" compares
against *our own* BF16 arithmetic and is marked Projected), `RELEASE_NOTES_v2.0.md`,
`HARDWARE_LESSONS.md`, `TESTING_DISCIPLINE.md`, `v4-reference-audit.md` (SGLang
refs are source-path citations for a correctness audit; `:1067` actively
*refuses* comparison to published V4 acceptance numbers), `PAGED_ATTENTION.md:106`
(attribution, no number), `IMATRIX.md` (llama.cpp interop), and the model/CLI/MCP
reference docs.

## PR-base trap → BACKLOG

Landed in `memory/mission/BACKLOG.md` under a new **🔴 CI/PR TRAPS** heading at
the top of the file. Content: a PR whose `baseRefName` is another PR's branch is
auto-closed silently when the parent merges; #44 died ~2 s after #43 with
`mergedAt=null` and the correction was reported up as fixed when it was not.
Rules recorded: (1) always open against `master` unless deliberately stacking,
and re-target the child *before* the parent merges; (2) verify by
`state=MERGED` **and** non-null `mergedAt`, never by a watcher's exit code —
`CLOSED` also vanishes from the open list; (3) a correction PR is not done when
opened — re-verify file content on `master` after the merge. Plus a detector:
`gh pr list --state closed --json number,title,mergedAt --jq
'.[]|select(.mergedAt==null)'`.

## 7. Gates

Markdown only, so no cargo gates apply. `git status` shows modified `.md`
files and nothing else. No `cargo fmt` / `rustfmt` was run anywhere.
