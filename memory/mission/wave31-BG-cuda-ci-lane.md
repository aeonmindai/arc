# wave31-BG — CI type-checks CUDA-gated Rust

**Date:** 2026-08-15
**Branch:** `ci/cuda-typecheck-lane` → PR #61 (draft, based on `master`)
**Scope fence:** CI workflow configuration, plus any compile errors the new lane
exposes. No kernel restructuring, no runtime behaviour changes.

---

## 1. The claim, and what was actually true

The backlog entry ("🔴 CI NEVER TYPE-CHECKS CUDA-GATED RUST", 2026-08-15) says
the `nvcc compile (sm_80)` / `(sm_90)` lanes "build `.cu` files" and that every
`#[cfg(feature = "cuda")]` **Rust** block is invisible to CI.

**That is the right alarm with the wrong mechanism.** Checking the workflow
before writing anything:

```yaml
- name: Compile QTIP CUDA kernels (mistralrs-quant)
  run: cargo build -p mistralrs-quant --lib --features cuda
- name: Compile arc-cuda-graph CUDA kernels
  run: cargo build -p arc-cuda-graph --lib --features cuda
- name: Compile arc-engine cuda (transitively quant + cuda-graph + core + paged-attn)
  run: cargo build -p arc-engine --lib --features cuda
```

`cargo build -p X --features cuda` compiles X's *Rust* as well as running its
build script. So the CUDA-gated Rust of five crates **was** covered:
mistralrs-quant, arc-cuda-graph, arc-engine, and — via
`mistralrs-core/cuda = [... "mistralrs-paged-attn/cuda", "arc-cuda-graph/cuda"]`
(mistralrs-core/Cargo.toml:119) — mistralrs-core and mistralrs-paged-attn.

Two holes were real, and both were live today.

### Hole 1 — the `paths:` filter, which is worse than the scope problem

```yaml
paths:
  - "**/*.cu"
  - "**/*.cuh"
  - "**/build.rs"
  - "mistralrs-quant/**"
  - "arc-cuda-graph/**"
  - "arc-engine/**"
```

`mistralrs-core/**` is not there. A PR that touches only mistralrs-core did not
run this workflow **at all** — not a reduced version of it, none of it.

Verified, not assumed. `gh -R aeonmindai/arc pr checks 59` on the merged
XsRollingCache PR returns exactly:

```
Check (macOS/ubuntu/windows), Check (metal), Clippy, Docs,
MSRV Check (1.90), Rustfmt, Test Suite (macOS/ubuntu/windows), Typos, comment
```

No `nvcc compile`, no CUDA anything. **PR #59 — `XsRollingCache` +
`KvCache::XsRolling`, ~1,480 lines touching `kv_cache/mod.rs`,
`models/deepseek4.rs`, `prefix_cacher.rs` — merged to master having never been
compiled with `--features cuda` anywhere.**

The other four PRs (#46, #52, #53, #56) all touched `mistralrs-quant/**` or
`arc-cuda-graph/**` and did run the lane.

### Hole 2 — nine crates whose CUDA-gated Rust nothing has ever compiled

Fourteen workspace crates define a `cuda` feature:

```
mistralrs-cli  arc-cli  arc-turbo  mistralrs-pyo3  mistralrs-paged-attn
mistralrs-server-core  arc-engine  arc-cuda-graph  mistralrs-quant
mistralrs-core  mistralrs-bench  mistralrs-web-chat  mistralrs  mistralrs-server
```

Five were covered (above). The remaining **nine** — arc-cli, arc-turbo,
mistralrs-cli, mistralrs-server-core, mistralrs, mistralrs-pyo3,
mistralrs-bench, mistralrs-web-chat, mistralrs-server — have `cuda` enabled by
no CI lane, ever.

Sizing that honestly (`grep -rlE 'feature *= *"cuda"' --include='*.rs'` over the
nine): only **two** of them contain gated blocks of their own —

| file | gated surface |
|---|---|
| `arc-cli/src/validate.rs:374` | `run_real_gpu` — the whole real-GPU HBM path |
| `arc-cli/src/bench/vendor.rs:214` | `pub mod in_process` — the in-process vendor |
| `arc-cli/src/bench/mod.rs:117` | the real-vendor branch of `run` |
| `mistralrs-cli/src/commands/doctor.rs` | gated doctor block |

The other seven only *forward* the feature to their dependencies, so their own
code compiles identically either way. The honest claim is therefore "two crates
of never-compiled gated Rust plus every never-compiled test target", not "nine
crates of dead code". Still worth closing: `run_real_gpu` and `in_process` are
exactly the code a rental session runs first.

Also uncovered: **test targets** outside mistralrs-quant. The old lane compiled
`cargo test --no-run -p mistralrs-quant --features cuda` and nothing else, so
the CUDA-gated tests in arc-cuda-graph (`tensor_ptr_accepts_every_decode_buffer_dtype`,
the FlashMLASparse parity tests) and mistralrs-core were never compiled either.

---

## 2. What shipped

`.github/workflows/cuda_compile_check.yaml`:

1. **New job `cuda-typecheck`** (`cargo check (cuda, workspace)`), reusing the
   existing GPU-less toolkit setup verbatim (Jimver/cuda-toolkit@v0.2.35,
   CUDA 12.4.1, `--toolkit`, driver-stub link path):
   - `cargo check --workspace --features cuda`
   - `cargo check --workspace --features cuda --tests`

   One compute cap (`sm_90`, the rental target) — no Arc Rust is `cfg`-gated on
   the arch, and sm_80/sm_90 `.cu` coverage is unchanged in the nvcc jobs.
   `cargo check` never links, so the absent driver (`-lcuda`) is a non-issue;
   `cargo check` *does* still run every build.rs, so nvcc still compiles the
   `.cu` files in this job.

2. **`paths:` re-keyed on `**/*.rs`** (plus `**/*.cu`, `**/*.cuh`,
   `**/Cargo.toml`, `Cargo.lock`, the workflow itself). Any Rust change
   anywhere now runs the lane; a docs/wave-log-only PR still skips it. This is
   the fix for hole 1 and it matters more than the new job does — a lane that
   does not run on the PR that needs it is worth nothing.

3. **`concurrency` group with `cancel-in-progress`**, because widening the
   trigger means this workflow now runs on most PRs and superseded runs should
   not be paid for.

Existing jobs are untouched.

---

## 3. What this buys — and what it does not

**Buys:** compile errors, type errors, borrow errors, FFI signature drift,
renames, trait-impl breakage, missing struct fields — in CUDA-gated Rust. The
class that used to surface at minute one of a *paid* GPU session, on a box the
Mac cannot reproduce.

**Does not buy:** runtime correctness, and specifically not the bug that
motivated the entry. `as_cuda_slice::<u8>()` on a BF16 buffer (five copies,
wave27-AY / AY2) **compiles perfectly**. Candle checks the dtype at runtime,
inside `slice_as_ptr`. A type-checker cannot see it; neither can nvcc. For that
class the answer is the GPU-gated regression tests wave27 added — and actually
running them on a GPU. This lane is a floor, not a ceiling.

Nor does it cover: the `flash-attn` feature combo (still
`arc-tools/cuda_compile_check.sh` on the build box), the final `-lcuda` binary
link, or nvcc-version differences if the rental ships something other than 12.4.

---

## 4. Findings from the first run — **clean**

**Zero compile errors.** Run
[31906350042](https://github.com/aeonmindai/arc/actions/runs/31906350042),
commit `64c9439`:

```
cargo check (cuda, workspace)   success   15m00s
nvcc compile (sm_80)            success   14m07s
nvcc compile (sm_90)            success   14m08s
```

Not a single rustc warning either. All **19 workspace members** were checked with
`cuda` on — confirmed from the log (`Checking arc-cli`, `Checking mistralrs-cli`,
`Checking mistralrs-pyo3`, `Checking mistralrs`, … ) and the feature really was
enabled, since build scripts ran nvcc:

```
warning: mistralrs-paged-attn: Compiling 15 of 15 kernels
warning: arc-cuda-graph:       Compiling 8 of 8 kernels
warning: mistralrs-quant:      Compiling ... kernels
```

So the five PRs merged today (#46, #52, #53, #56, #59) introduced **no** type,
borrow or signature errors in CUDA-gated Rust. Including #53, where the agent
explicitly flagged "both fixes are CUDA-gated, so only syntax is checked
locally; a CUDA build is needed to confirm compilation." It compiles. That is a
good outcome and worth stating plainly rather than manufacturing a finding.

### Timing — `--tests` is nearly free

| step | cold (first run) |
|---|---|
| `cargo check --workspace --features cuda` | **9m27s** |
| `cargo check --workspace --features cuda --tests` | **+45.7s** |
| whole job (incl. ~4m toolkit install) | 15m00s |

`check --tests` does no codegen, so it costs a fraction of the existing
`cargo test --no-run -p mistralrs-quant` step. Leaving it out would have been
the wrong call. The new job runs **in parallel** with the two nvcc jobs
(14m07s), so workflow wall-clock is essentially unchanged; the cost is one extra
runner slot.

---

## 5. Is the lane vacuous? Falsified, twice.

A green lane proves nothing unless it can go red. Per the backlog's own
recurring lesson ("the check could not fail for the real risk"), two throwaway
commits planted deliberate errors in CUDA-gated Rust that **no lane had ever
compiled**.

### Probe 1 — `715dcfe`: type error in `arc-cli`'s `#[cfg(feature = "cuda")]` code

`arc-cli/src/validate.rs:347`, `.unwrap_or(0)` → `.unwrap_or(0u32)` inside
`#[cfg(feature = "cuda")] fn cuda_device_index() -> usize`.

Run [31907602869](https://github.com/aeonmindai/arc/actions/runs/31907602869):

```
cargo check (cuda, workspace)   FAILURE  5m32s
nvcc compile (sm_80)            success  13m26s
nvcc compile (sm_90)            success  13m49s
```

```
error[E0308]: mismatched types
   --> arc-cli/src/validate.rs:347:20
347 |         .unwrap_or(0u32)
    |          --------- ^^^^ expected `usize`, found `u32`
error: could not compile `arc-cli` (bin "arc") due to 1 previous error
```

**Both nvcc lanes stayed green on the same commit.** That is the hole, measured:
a type error in CUDA-gated Rust, invisible to the pre-existing CI, caught by the
new lane. (Warm cache also shows the steady-state cost — 5m32s to failure, most
of it toolkit install, not compilation.)

### Probe 2 — `b2df2b7`: error in a `#[cfg(all(test, feature = "cuda"))]` test

Probe 1 exposed a design detail worth recording: **step 1 failing skips step 2**,
so probe 1 said nothing about whether `--tests` has teeth. Probe 2 reverts the
arc-cli break and keeps only a planted error inside
`tensor_ptr_accepts_every_decode_buffer_dtype`
(`arc-cuda-graph/src/autonomous.rs`) — a test added today by PR #53 and compiled
by nothing before this lane.

Run [31908315527](https://github.com/aeonmindai/arc/actions/runs/31908315527):

```
cargo check (cuda, workspace)   FAILURE  6m29s
  ✓ Type-check CUDA-gated Rust (workspace)      <- green, as expected
  X Type-check CUDA-gated tests (no codegen)    <- red, as expected
nvcc compile (sm_80)            success  13m50s
nvcc compile (sm_90)            success  14m43s
```

```
error[E0308]: mismatched types
   --> arc-cuda-graph/src/autonomous.rs:889:41
889 |         let _falsification_probe: u32 = "not a u32";
    |                                   ---   ^^^^^^^^^^^ expected `u32`, found `&str`
error: could not compile `arc-cuda-graph` (lib test) due to 1 previous error
```

Both steps have independent teeth, and the nvcc lanes were blind a second time.
Both probes reverted in `120ae16`; the branch carries only the workflow change.

**Design note worth keeping:** because a failing step aborts the job, the
`--tests` step never runs when the workspace check fails. That is the right
ordering (the library error is the more fundamental one) but it means a single
red run reports only the first failure class. Do not read "step 2 skipped" as
"tests are fine."

---

## 6. Surfaced, not shipped

1. **`master` has no branch protection.** `GET
   /repos/aeonmindai/arc/branches/master/protection` → `404 Branch not
   protected`. **No status check blocks any merge** — not this lane, not Clippy,
   not the test suite. Every gate in this repo is advisory. Coverage was the
   thing being fixed today; *enforcement* is now the binding constraint, and it
   is a settings change, not an engineering one.
2. **Dead `cfg` pair in `deepseek4.rs:3489-3492`.** The `#[cfg(all(feature =
   "cuda", target_family = "unix"))]` and `#[cfg(not(...))]` arms assign the
   *identical* value (`KvCacheLayout::Standard`). Harmless, but it reads as if
   the layout differs by platform when it does not.
3. **The lane still does not cover `cuda flash-attn` together**, which is the
   actual rental feature combo. `arc-tools/cuda_compile_check.sh` covers it on
   the build box; nothing covers it in GitHub CI.
4. **Cache pressure.** This adds a third `target/` cache entry
   (`cuda-check-cc90-*`) alongside `cuda-cc80-*` and `cuda-cc90-*`. GitHub's
   10 GB per-repo cache limit evicts LRU; if the nvcc lanes start missing cache
   and slowing down, this is why.
