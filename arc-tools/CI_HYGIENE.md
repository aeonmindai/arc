# CI hygiene (Rustfmt / Clippy / Typos) — status & follow-up

**TL;DR: the "Continuous integration" workflow is red only on the three hygiene
jobs, and that is NOT a blocker for M1 or the H100 rental.** All functional jobs
are green: Test Suite (ubuntu/windows/macOS), Check (all platforms + metal),
MSRV 1.90, Docs. The rental binaries build
(`cargo build --release -p arc-cli -p mistralrs-cli --features "cuda flash-attn"`)
and the rental never runs fmt/clippy/typos.

## Why this is not a *bounded* fix — don't reflexively `cargo fmt --all`

Arc is a fork of `EricLBuehler/mistral.rs`. The hygiene gates were inherited from
upstream and the red is dominated by **upstream-derived files**, not Arc code:

- **Rustfmt:** 137 files need formatting; **91 are `mistralrs-*`** upstream
  model/core files (`qwen2.rs`, `llama.rs`, `mistral.rs`, `llava/*`, `phi4`,
  `smollm3`, …). `cargo fmt --all` reformats all 91 → a large merge-conflict
  surface on every future upstream sync. This is rustfmt-version drift, not Arc
  sloppiness. **Verified 2026-05-25** (`cargo fmt --all -- --check`, unique
  files): 137 total = 46 Arc-authored (`arc-bench` 16, `arc-engine` 14,
  `arc-cuda-graph` 11, `arc-cli` 5) + 91 upstream (`mistralrs-quant` 2, other
  `mistralrs-*` 89). **Reformatting only the 46 Arc files does NOT turn the
  `fmt --all -- --check` job green** — the 91 upstream files keep it red — so a
  pre-rental Arc-only reformat buys churn (and merge-conflict risk against the
  in-flight worktree agents + the numerically-sensitive `arc-engine`
  `dsv4.rs`/`td_moe.rs`/`sage.rs`) without flipping the gate. Greening the gate
  requires the CI-scope change (2(b)) below, not a code reformat. Hence:
  deferred, by decision, not by oversight.
- **Clippy `--workspace --tests --examples -D warnings`:** exact lint inventory
  from CI run `26373440253` (Clippy job `77629534163`, 2026-05-24), split by what
  is safe to touch:
  - **Safe to auto-fix (non-numerical) — `arc-bench`:** `manual_div_ceil` (×3),
    `should_implement_trait` (a `from_str` at `replay.rs:120` — rename or `#[allow]`),
    `doc_lazy_continuation`, `doc_overindented_list_items`, `derivable_impls`,
    `unwrap_or_default`, `useless_conversion`, `manual_checked_ops`. These are
    harness code; `clippy --fix -p arc-bench` + a `cargo test -p arc-bench` guard
    is low-risk.
  - **DO NOT `--fix` — `mistralrs-quant/src/qtip/*` (Viterbi/scales hot paths):**
    `needless_range_loop` (loop var `s` indexing `prev_cost`; loop var `row`
    indexing `scales_data`), `unnecessary_cast` (`usize`→`usize`),
    `manual_is_multiple_of`. Their parity is **sm_80+-GPU-only-validated** — rewriting
    the indexing is exactly the risk the rental can't catch offline. **Suppress with
    a targeted `#[allow(clippy::needless_range_loop)]` etc. on the specific fns
    (an attribute is behavior-preserving — NOT a logic rewrite), never `--fix`.**
  - Net: fixing only `arc-bench` will **not** flip the `--workspace -D warnings`
    gate (the qtip errors remain); greening needs the qtip `#[allow]`s too, plus
    any upstream lints. Hence the gate is post-rental, by decision.
- **Typos:** pervasive **false positives** in upstream/binary code (`"BA"`,
  `"UE"`, `"nd"`, `"writeable"`, `"mis"`, `"fied"`). Green = `.typos.toml`
  ignore-list bloat. (The one real Arc typo — a misspelled `triggered` — is
  already fixed.)

## Do NOT auto-fix the QTIP hot loops before the rental

Several Clippy lints (`needless_range_loop`, `unnecessary_cast`) live in
`mistralrs-quant/src/qtip/{viterbi,mod}.rs`. `clippy --fix` would rewrite indexing
loops in the Viterbi quantize path, whose **numerical parity is only validated on
an sm_80+ GPU** (rental step 4b). Refactoring it right before a rental adds risk
the free CPU tests cannot fully catch. Clean QTIP clippy **after** the rental,
with the GPU parity test as the guard.

## Exact follow-up (when someone wants green CI, post-rental)

```bash
# 1a. SAFE auto-fix — non-numerical harness/CLI crates only:
cargo clippy -p arc-bench -p arc-cli --tests --fix
# 1b. Numerical crates — do NOT --fix. Add targeted #[allow(...)] on the flagged
#     fns instead (attributes are behavior-preserving; the rewrites are not):
#       mistralrs-quant/src/qtip/{viterbi,mod}.rs : needless_range_loop,
#         unnecessary_cast, manual_is_multiple_of  (Viterbi/scales hot loops)
#       arc-engine, arc-cuda-graph : inspect each lint; #[allow] any in a
#         numerical loop, --fix only the trivially-safe ones.
cargo test -p mistralrs-quant -p arc-engine -p arc-cuda-graph  # guard, every time

# 2. fmt — decide policy first (fork tradeoff). Either:
#    (a) accept upstream divergence:  cargo fmt --all
#    (b) or narrow the Rustfmt CI check to Arc crates so the fork stops
#        inheriting upstream rustfmt drift.

# 3. typos — add the upstream false positives to .typos.toml
#    [default.extend-ignore-re]; review the rest, fix any real ones.
typos
```

**Recommended:** option 2(b) — scope the Rustfmt/Clippy/Typos CI jobs to
Arc-authored crates so the fork stops inheriting upstream hygiene drift, instead
of mass-reformatting upstream files.
