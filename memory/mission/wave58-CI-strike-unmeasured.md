# wave58-CI — Strike unmeasured third-party performance claims

**Branch:** `docs/strike-unmeasured-claims` · **Base:** `c763123be` (post-#83)
**Scope:** markdown only, no GPU, no code changes.

## Why

`ARC_V2.md:11` — the opening sentence of the architecture document — claimed Arc
serves frontier LLMs "at **10–70x the throughput of SGLang/vLLM on the same
hardware**". We have never run SGLang or vLLM, and the only model this program
has ever served is DeepSeek-V4-Flash. It directly contradicted `README.md:36`
in the same public repo.

Third instance of one pattern:

1. `docs/PEAK_INFERENCE.md` fabricated `33 / 60 / 43 tok/s` table — no column
   ever run, including ours (deleted, PR #49).
2. `ARC_V2.md` `SGLang today` comparison column (struck, PR #83).
3. This one — survived both passes **because it is prose, not a table**.

## Complete inventory

Full sweep of all non-`memory/`, non-`research/` markdown plus crate rustdoc.

### Fixed here (4)

| Location | Claim | Fix |
|---|---|---|
| `ARC_V2.md:11` | "10-70x the throughput of SGLang/vLLM on the same hardware"; "serves ... DeepSeek V4 Pro, Kimi K2.6, GLM 5.1" | Replaced with the measured, checkable record: 74.19 GB at 2.09 bits/param on one GPU vs every published V4-Flash config needing four; GSM8K 1270/1319 = 96.3% +/-1.0pp 0-shot (reference 90.8 is 8-shot); 111.69 tok/s aggregate at B=256, $12.06/Mtok, `effective_B == B` on all rows, 0 errors / 505 requests. Plus an explicit "only V4 Flash has ever been served" and "no side-by-side has ever been run; `arc-bench` is vendor-abstracted so a baseline is one rental away". |
| `ARC_V2.md:165` (pre-edit) | "attention ... caps single-user tok/s at **~30**" | The `~30` was the SGLang-today column figure leaked into prose. Replaced with the mechanism plus "the ceiling is unmeasured, on Arc and every other engine". |
| `ARC_V2.md:274` (pre-edit) | "Cache hit rate goes from **~30% (RadixAttention)** to **~60%**" | Both unsourced; the doc itself admits "no application paper". Replaced with the mechanism plus "neither hit rate has been measured". |
| `README.md:36` | "no third-party comparison appears anywhere in this repo" — **false** while `autonomous.rs:10` stands | Rewritten to assert only what is true ("never run one, on any engine, ever") and to invite reports of leftovers. |

### Also corrected — never-served models (1)

- `README.md:141` — "Supported Models" listed DeepSeek V4 Pro / Kimi K2.5 + K2.6 /
  GLM-5.1 without qualification. Added a banner: "supported" means the
  architecture loads, not that Arc has served it; only V4 Flash has ever run
  end-to-end. (`README.md:191` Phase 1 was already correctly under "Roadmap".)

### Also corrected — unsourced projection (1)

- `ARC_V2.md:260` — "2-3x aggregate throughput" for expert-affinity batching had
  no paper and no run behind it. Relabelled as an open question until measured.

## Deliberately left alone

- `arc-tools/M1_GATE.md:160` — ">=10x SGLang on 8xH100" is an explicit **M5
  milestone target** in a gate document. Legitimate. `:165` ("even if it is
  slower than SGLang") is exemplary honesty — keep.
- `arc-tools/RENTAL_PLAYBOOK.md:266,336,362,371` — cost estimates and gating for
  a **future** comparison run. Legitimate.
- `docs/PEAK_INFERENCE.md:40-48` — quotes the fabricated table **as a retraction
  record**. Must survive.
- `docs/CUDA_GRAPH_PLAN.md:14,296` — explicitly *disclaim* comparison. Model
  behaviour.
- `docs/engineering/QUANTIZATION_PERFORMANCE.md:491,494` — EXL3's **own
  published** figure, tagged `[published]`, with "EXL3 publishes no per-layer
  number" stated plainly. Citing a project's published number with attribution
  is allowed under D9.
- `docs/engineering/OPEN_QUESTIONS.md:279` — "SGLang's V4 impl uses a single
  shared head, ~64x the indexer key memory" is **source-verified architecture**,
  not a benchmark. Allowed: naming what a competitor *implements*.
- All `research/code/...` citations, `docs/notes/v4-reference-audit.md`, and the
  `flashmlasparse*.rs` rustdoc — source and architecture references.
- `ARC_V2.md:360` — "End-to-end benchmark vs SGLang" is a **plan to run one**,
  which is the honest path forward, not a result.
- Repo-wide `74.18` vs `74.19` rounding split (README / FLEET / PEAK_INFERENCE /
  RELEASE_NOTES say 74.18; BENCHMARKS and the model card say 74.19 with the exact
  byte count 74,190,197,268). Same measurement, cosmetic; not this PR's mandate.

## ONE VIOLATION LEFT UNFIXED — needs a follow-up (not markdown)

`arc-cuda-graph/src/autonomous.rs:10`:

```rust
//!   Still better than vLLM's ~10μs (we skip re-capture).
```

An unmeasured third-party latency figure, in a **public** repo, in crate
rustdoc. We have never profiled vLLM. This wave was fenced to markdown, so it is
**not** fixed here. It is the reason `README.md:36` had to be reworded rather
than left asserting a clean sweep. **Recommend a one-line follow-up PR** deleting
the comparison clause and keeping "~2.5μs per step (we skip re-capture)".

## Doc-sync debt surfaced (not fixed)

The wave51-CB results — full GSM8K **96.3%**, aggregate **111.69 tok/s at
B=256**, **$12.06/Mtok** — appear in **no** `docs/` page. `README.md:22,48-53`,
`docs/BENCHMARKS.md`, `docs/FLEET.md` and `docs/RELEASE_NOTES_v2.0.md` still
carry the superseded provisional **87.0%** (n=100) and **14.58 tok/s** b=1.
`ARC_V2.md` now notes the supersession inline so it does not read as a
contradiction, but **publishing the wave51 numbers across `docs/` is a separate
job and should be scheduled.**
