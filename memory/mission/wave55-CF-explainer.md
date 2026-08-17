# wave55-CF — `docs/engine-explainer.html`

**Ask (Jish, verbatim):** *"there's too many things to keep in fucking mind, Cuda
graph, flash attention, paged attention, flash attention 3, baking, trellis,
qtip, so many fucking terms and I don't wanna miss any of them while working,
can you make a beautiful layman html which shows how an inference engine
completely functions, which things affects what (throughput(agg, single), ttft,
size, quality) I don't wanna miss any term at all"*

**Deliverable:** one self-contained file, `docs/engine-explainer.html` (~119 KB).
No CDNs, no external fonts, no network, no build step. Opens by double-click.
**No engine code was touched.**

---

## What it is

A thinking tool for prioritisation calls, not documentation. Two halves.

**Half 1 — The journey of a token.** A hand-authored inline SVG (viewBox
`0 0 1160 800`, 12 clickable stage boxes, arrowhead markers, a dashed
`DECODE LOOP` container, and a **thick accent-coloured return path** from
`Sample` back to `Attention` so the loop is visibly a loop rather than a bullet
list). Order: request → tokenize → schedule → prefix cache → **prefill** → KV
write → **decode loop** (attention → route → experts → sample → back) → exit on
EOS → detokenize → stream. Four cross-cutting "layer" chips (memory, execution,
compression, parallelism) sit under it.

**Half 2 — The term matrix.** **97 terms**, each with: a one-line plain-English
definition, an analogy, colour-coded chips for **which of the five metrics it
moves** (aggregate throughput · per-user throughput · TTFT · model size ·
quality), a status badge, and an "In Arc today" panel grounded in a measured
number or a `file:line`.

Status vocabulary (4, because 3 was not honest enough):
`shipped` 56 · `off` (built but not on) 11 · `none` (not built) 8 ·
`concept` (physics or an idea; nothing to ship) 22.

**Interactive:** live search (token-AND, so *"flashattention 3"* finds
*"FlashAttention (1 → 2 → 3)"*, and the index is doubled with `_ - / ( )`
flattened so *"paged attention"* finds `supports_paged_attention`); 5 metric
filters; 4 status filters; click any diagram stage or layer chip to filter the
matrix to it (with a clearable pill and the stage highlighted in the SVG);
`Browse` ⇄ `What should I care about` view toggle; `/` focuses search, `Esc`
clears everything; every stage is `tabindex`/`Enter`-activatable.

**"What should I care about" view.** Ranks by `gap × leverage`, each 0–5. Top of
the list today: computed codebook, decode, expert dedup, host-vs-device time,
the qtip2b rung (all 25), then expert parallelism / GEMV / grouped GEMM (20).
31 of 97 terms carry scores; concepts and finished work score 0 on gap and drop
out. **The two scores are labelled in the UI as judgement, not measurement** —
they are anchored to the measured gaps quoted on each card, but the weighting is
an opinion and is meant to be argued with.

---

## Grounding

Read in full before writing: `memory/mission/CEILINGS.json`,
`memory/mission/FACTS.md` (both pages), `memory/mission/00_RESUME_HERE.md`,
`memory/mission/STATUS.md` (top ~200 lines), `docs/BENCHMARKS.md`.
Every number is tagged `[measured]` / `[physics]` / `[projected]` /
`[measured-by-design]`, and the footer defines all four.

A parallel code sweep confirmed each status claim against `file:line` rather
than against the brief. Load-bearing anchors used: 74.19 GB @ 2.09 bits/param ·
GSM8K 96.0% (n=100, our 0-shot protocol) · aggregate 111.7 tok/s @ B=256 /
$12.06/Mtok · b=1 ceiling 1,413 tok/s · B=256 aggregate ceiling ~16,600 ·
`E(B)=256×(1−(1−6/256)^B)` · MTP k≈1.84 · B=256 host-bound at 0–4% GPU /
121 W of 700.

**No competitor benchmark we have not run ourselves appears anywhere.** The
three reference figures that do appear — the 90.8 8-shot GSM8K, the ~80%-of-peak
GEMV, the 2.30 MTP acceptance floor — are each labelled as *theirs* and as
unverified by us, in the card text and again in the footer.

### Four places the code disagreed with the brief — page follows the code

1. **Prefix cache is ON by default**, not off: `no_prefix_cache` defaults
   `false` and `--prefix-cache-n` defaults to 16
   (`mistralrs-cli/src/args/mod.rs:423-426`). What is true is that **every sweep
   we have ever run passed `--prefix-cache-n 0`**, so all our TTFT and
   throughput numbers are the no-cache worst case. The card says both.
2. **CUDA-graph bails.** The three refusals in `normal.rs::autonomous_decode`
   are *not* the paged flag: `:1797` dedicated decode path unavailable, `:1815`
   custom logits processors, `:1822` `top_nsigma`. The path *also* needs a
   `cache_config` (`:1845`) and **that** is the one downstream of
   `supports_paged_attention=false`. Stated precisely rather than as
   "three bails downstream of the flag".
3. **EAGLE is not "not built"** — `arc-engine/src/eagle3.rs` exists, alongside
   `magicdec.rs`. Marked `built but off`, with the caveat that neither is the V4
   path nor appears in any measured number.
4. **Lightning Indexer** is stronger evidence than "loaded, never called": the
   model field at `deepseek4.rs:918` carries an explicit `#[allow(dead_code)]`
   and there is no `self.indexer` use anywhere in `deepseek4.rs` or
   `dsv4_attention.rs`. Its CUDA twin is likewise unwired.

Also confirmed in source and used: `supports_paged_attention` → `Ok(false)` at
`normal_loaders.rs:3275` (fn at `:3231`); `radix_trie` is a declared dependency
in `Cargo.toml:126` **with no use site** (RadixAttention genuinely not built);
`ArcAttention` appears only as a roadmap row at `arc-tools/M1_GATE.md:159`, zero
`.rs`/`.cu` matches; FP8 KV gate `var == Some("1") && !capture_probe`
(`deepseek4.rs:2421-2423`); `--mtp-depth` defaults to 0; top-6 default at
`deepseek4.rs:1807`, 256 experts at `:215`; FA2 and FA3 are separate mutually
exclusive features and **every measured session was FA2**; scheduler coalescing
`select_running_bucket` at `default_scheduler.rs:115`.

### Not grounded in a file — flagged, not hidden

- **111.7 tok/s @ B=256 and $12.06/Mtok.** Supplied as a measured anchor in the
  task brief; **not in `FACTS.md`, `CEILINGS.json`, `STATUS.md` or
  `docs/BENCHMARKS.md`**, whose latest recorded B=256 row is still session 6's
  **19.02 tok/s / $70.83**. The internal arithmetic checks out exactly
  ($4.85/hr ÷ (111.7 × 3600 ÷ 1e6) = $12.06), so it is consistent with an H200
  at the Helsinki rate. ⇒ **`FACTS.md` and `docs/BENCHMARKS.md` are now stale on
  the headline throughput row and need the wave entry that produced it.**
- **~94% GSM8K on the qtip2b rung (1,319-problem run).** Brief says in
  progress; no file records it. Tagged `[measured, in flight]` on the page.
- **PagedAttention "probe running"** and **"a RadixAttention port with
  improvements is in flight"** come from the brief. The *code* state behind both
  (flag false; no radix code) is grounded.

---

## Verification

- **97 terms**, 97 cards, 10 group headers, 0 malformed entries; status and
  metric vocabularies validated programmatically against the allowed sets.
- **Rendered from the real `file://` URL in headless Chrome** (not just over
  HTTP): DOM contains `Showing 97 of 97 terms`, i.e. the script executed with no
  server and no network.
- **Zero console errors and zero warnings** (Playwright, all levels).
- **Zero external references.** The only `http` string in the file is the
  `www.w3.org/2000/svg` XML namespace inside the favicon data-URI — an
  identifier, never fetched. Favicon is an inline data-URI, so even the
  automatic `/favicon.ico` request is gone.
- **No horizontal page scroll**; the one overflowing element (a 74-char `<code>`
  span) was fixed with `overflow-wrap:anywhere`. The wide diagram scrolls inside
  its own container.
- **Interactions exercised programmatically:** stage click (`experts` → 17 of
  97, pill set, SVG box highlighted), pill clear, layer chip (`memory` → 8),
  metric filter (`agg` → 56), stacked metric+status (`agg` + `not built` → 4:
  RadixAttention, ArcAttention, expert dedup, expert parallelism), reset, view
  toggle, and back.
- **58 search probes covering every term Jish named — 0 misses.**
- **0 leaked `*` and 0 leaked backticks** in rendered text (37 `<em>`, 287
  `<strong>`, 231 `<code>` render correctly).

Two bugs were found by testing and fixed: the substring search could not find
"flashattention 3" (now token-AND), and the formatter handled `**bold**` but not
`*italic*`, leaking 37 literal asterisks.

---

## Surfaced, not shipped

- `FACTS.md` / `docs/BENCHMARKS.md` do not contain the 111.7 tok/s row that is
  now the headline number. Worth a separate change.
- Writing these cards surfaced that **every TTFT number on record was taken with
  the prefix cache explicitly disabled**, and nobody has measured what turning
  it on is worth — cheap, and unclaimed.
