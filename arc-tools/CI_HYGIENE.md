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
  sloppiness.
- **Clippy `--workspace`:** lints span `arc-bench` (Arc) and
  `mistralrs-quant/src/qtip/*` + `distributed/layers.rs` (Arc-authored QTIP work),
  possibly also upstream. Fixing only the Arc subset will not flip the
  `--workspace -D warnings` gate green.
- **Typos:** pervasive **false positives** in upstream/binary code (`"BA"`,
  `"UE"`, `"nd"`, `"writeable"`, `"mis"`, `"fied"`). Green = `.typos.toml`
  ignore-list bloat. (The one real Arc typo, `trigggered`, is already fixed.)

## Do NOT auto-fix the QTIP hot loops before the rental

Several Clippy lints (`needless_range_loop`, `unnecessary_cast`) live in
`mistralrs-quant/src/qtip/{viterbi,mod}.rs`. `clippy --fix` would rewrite indexing
loops in the Viterbi quantize path, whose **numerical parity is only validated on
an sm_80+ GPU** (rental step 4b). Refactoring it right before a rental adds risk
the free CPU tests cannot fully catch. Clean QTIP clippy **after** the rental,
with the GPU parity test as the guard.

## Exact follow-up (when someone wants green CI, post-rental)

```bash
# 1. Arc-only clippy (does NOT fight upstream; verify with CPU tests after):
cargo clippy -p arc-bench -p arc-engine -p arc-cuda-graph -p arc-cli --tests --fix
cargo test -p mistralrs-quant -p arc-engine   # guard the qtip/* rewrites

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
