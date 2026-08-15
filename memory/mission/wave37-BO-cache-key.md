# wave37-BO — the CUDA lane's cache poisons itself (`target-cpu=native` + an ISA-blind key)

**Scope:** `.github/workflows/cuda_compile_check.yaml` cache configuration only.
No Rust, no `.cargo/config.toml`, no other workflow touched.
**Branch:** `ci/fix-cuda-cache-key`. **Base:** `master` (`10297fb8b`).
**Prior art:** wave31-BG (the lane itself, PR #61), wave35-BM §11 (the diagnosis).

---

## 1. The defect, restated precisely

Two pre-existing pieces, each defensible alone, lethal together:

1. `.cargo/config.toml` — `[build] rustflags = ["-C", "target-cpu=native"]`,
   repo-wide. Deliberate; stays.
2. `cuda_compile_check.yaml` — caches `target/` under a key with **no
   ISA/CPU component**, plus a bare restore-key prefix.

Runner A compiles `build-script-build` with its own ISA and saves `target/`.
Runner B restores it and *executes* that binary → `signal: 4, SIGILL`.

The subtlety that makes this a trap rather than a bug you notice: **`native` is
fingerprint-opaque.** Cargo hashes the flag *string*, not the ISA it resolves
to. Two runners produce byte-incompatible artifacts under an *identical*
fingerprint, so Cargo has no reason to rebuild. Every other rustflag value is
self-invalidating; `native` is the one that isn't.

## 2. Mechanism — verified locally, not recalled

All four claims below were probed with a throwaway crate + `cargo build -v`
(rustc 1.95.0), because getting any of them wrong would have shipped a no-op.

| # | Claim | Probe result |
|---|---|---|
| 1 | `[build] rustflags` reaches build scripts | `--crate-name build_script_build` **and** `--crate-name probe` both carry `-C target-cpu=native`. This is the SIGILL vector. |
| 2 | `CARGO_BUILD_RUSTFLAGS` overrides `[build] rustflags` | Yes — both invocations flipped to the env value. |
| 3 | `CARGO_BUILD_RUSTFLAGS` overrides `[target.<triple>] rustflags` | **NO.** With a matching `[target.<triple>]` section present, the env var was *silently ignored* and `native` won. |
| 4 | plain `RUSTFLAGS` overrides `[target.<triple>]` | **Yes.** Highest precedence. |

⇒ Precedence is `RUSTFLAGS` > `target.<triple>.rustflags` > `build.rustflags`,
and only **one** level applies — they do not merge.

**This changed the fix.** The obvious spelling, `CARGO_BUILD_RUSTFLAGS` (what
wave35-BM suggested), works *today* only because `.cargo/config.toml` happens
to have no `[target.x86_64-unknown-linux-gnu]` section — it has
`aarch64-apple-darwin`, `x86_64-apple-darwin`, `wasm32-unknown-unknown`. The
day anyone adds a Linux target section, `CARGO_BUILD_RUSTFLAGS` is defeated
without a word and the trap silently re-arms. `RUSTFLAGS` cannot be defeated
that way. Shipped `RUSTFLAGS`.

Fifth probe, on the ISA choice itself:

```
rustc --print cfg --target x86_64-unknown-linux-gnu -C target-cpu=x86-64-v3
  -> avx, avx2, bmi1, bmi2, f16c, fma, lzcnt, movbe, popcnt, sse..sse4.2, ...
rustc --print cfg --target x86_64-unknown-linux-gnu           (plain x86-64)
  -> fxsr, sse, sse2   ONLY
```

`mistralrs-quant/src/f8q8/mod.rs` has 7 `#[cfg(target_feature = ...)]` gates,
including `avx`. Pinning plain `x86-64` would have silently stopped compiling
the AVX branch — a coverage regression disguised as a fix. `x86-64-v3` keeps it
enabled, and is the Haswell/Zen1 baseline every GitHub-hosted x64 Linux runner
satisfies.

## 3. Option chosen: (c) pin the ISA — and why not (a) or (b)

**(a) hash the runner CPU into the key — rejected, and it is the dangerous one.**
It would work mechanically (a step reading `/proc/cpuinfo` into `$GITHUB_OUTPUT`),
but: GitHub exposes no first-class runner-CPU variable; the honest thing to hash
is the `flags` line, not `model name`, since `native` resolves via CPUID
features; it fragments one cache into N across GitHub's mixed Xeon/EPYC fleet,
so the warm-hit rate collapses and the 10 GB repo quota churns via LRU; and
**it cannot be proved** — you cannot force a run onto a chosen CPU model, so
"the key varies" is unfalsifiable in CI. That is precisely the
test-that-cannot-fail pattern this repo has now hit 7+ times. A fix that looks
like a fix and cannot be demonstrated is worse than the bug.

**(b) drop `target/` from the cache paths — correct, and strictly the most
robust, but it costs the wrong lane.** It assumes nothing about the fleet. For
the `cargo check` lane it is genuinely free (see §5). But the same defect is
latent in the **nvcc** jobs, and those cache 1.37 GiB of nvcc output each and
*are* the workflow's critical path. Making them cold to fix a bug they have not
yet hit trades a real, recurring CI cost for robustness already obtainable
another way. Kept as the documented fallback in the file header.

**(c) pin `RUSTFLAGS: -C target-cpu=x86-64-v3` — shipped.** It removes the ISA
*variance* at the source rather than working around it, so `target/` becomes
safe to cache on **all three** jobs, at zero CI-time cost, and the lane stays
structurally safe against someone later re-adding a cache path. Residual
assumption, stated honestly: every hosted x64 Linux runner supports v3 (true —
v3 is 2013 Haswell; GitHub runs Skylake-SP/Ice Lake Xeons and Zen 3 EPYCs). If
`ubuntu-latest` ever became ARM64, `-C target-cpu=x86-64-v3` makes rustc fail
loudly rather than SIGILL, and `${{ runner.arch }}` in the key buckets it
separately anyway.

Both invariants are documented in the workflow header so the next editor cannot
remove one without seeing the other.

## 4. Old key vs new key (verbatim)

`cuda-typecheck` — the job that actually failed:

```yaml
# OLD
key: cuda-check-cc90-${{ hashFiles('Cargo.lock') }}
restore-keys: |
  cuda-check-cc90-
# NEW
key: cuda-check-cc90-${{ runner.arch }}-isa-x86-64-v3-${{ hashFiles('Cargo.lock') }}
restore-keys: |
  cuda-check-cc90-${{ runner.arch }}-isa-x86-64-v3-
```

`cuda-compile` (nvcc, both matrix legs) — same latent defect, not yet bitten:

```yaml
# OLD
key: cuda-cc${{ matrix.compute_cap }}-${{ hashFiles('Cargo.lock') }}
restore-keys: |
  cuda-cc${{ matrix.compute_cap }}-
# NEW
key: cuda-cc${{ matrix.compute_cap }}-${{ runner.arch }}-isa-x86-64-v3-${{ hashFiles('Cargo.lock') }}
restore-keys: |
  cuda-cc${{ matrix.compute_cap }}-${{ runner.arch }}-isa-x86-64-v3-
```

**Changing the restore-keys is not cosmetic.** A restore-key is a *prefix*
match. Bumping only the `key` would have left `cuda-check-cc90-` matching — and
therefore restoring — the exact poisoned entry. Checked against the three live
entries from `gh cache list`:

| restore-key | matches live poisoned entry? |
|---|---|
| `cuda-cc80-` | ✅ yes — `cuda-cc80-10d492c78c97…` |
| `cuda-cc90-` | ✅ yes — `cuda-cc90-10d492c78c97…` |
| `cuda-check-cc90-` | ✅ yes — `cuda-check-cc90-10d492…` |
| `cuda-cc80-X64-isa-x86-64-v3-` | ❌ orphaned |
| `cuda-cc90-X64-isa-x86-64-v3-` | ❌ orphaned |
| `cuda-check-cc90-X64-isa-x86-64-v3-` | ❌ orphaned |

No manual `gh cache delete` is needed; the old entries are unreachable and age
out by LRU.

## 5. Timing — measured from the runs, not estimated

Per-job wall clock via `gh run view <id> --json jobs`:

| run | `cargo check (cuda, workspace)` | `nvcc sm_80` | `nvcc sm_90` |
|---|---|---|---|
| `31911343776` master, check=**cache MISS** | **12m18s** (cold) | 14m04s | 13m52s |
| `31912205278` #64, post-eviction | **14m39s** (cold) | 13m27s | 17m42s |
| `31911841156` #64, cache **restored** | **3m38s → SIGILL** | (cancelled) | (cancelled) |

Two things fall out:

1. **There has never been a warm *successful* run of this lane.** Every warm
   restore SIGILL'd. The "warm" figure is a time-to-crash (3m38s), not a
   speedup. Any "6m warm vs 15m cold" framing is comparing a green cold run to
   a red crash.
2. **The nvcc jobs are the critical path, not `cargo check`.** In both complete
   runs an nvcc leg finished last. A fully cold `cargo check` (12–15m) hides
   entirely behind them — which is why option (b) is free *for that lane* and
   expensive for the nvcc legs.

Cost of the shipped fix: **one cold run per job** (the key changed, so the first
run misses), then warm as before. Steady-state CI time is unchanged; only the
ISA the artifacts are built for changes.

## 6. Every other `target/`-caching site in the repo

Audited all 9 workflows. Four cache `target/`; a fifth caches Docker layers.

| workflow | key | verdict |
|---|---|---|
| `cuda_compile_check.yaml` `cuda-typecheck` | `cuda-check-cc90-<lock>` | 🔴 **bitten — FIXED here** |
| `cuda_compile_check.yaml` `cuda-compile` (×2) | `cuda-cc<cap>-<lock>` | 🟠 **same defect, latent — FIXED here.** Live entries since 08-14, 1.37 GiB each; they simply have not been restored across a CPU boundary yet. |
| `flash_attn_compile_check.yaml` | `flash-attn-cc<cap>-<lock>` | 🟠 **same defect, NOT fixed — out of scope.** Caches `target` identically. Lower blast radius: `workflow_dispatch`-only, so it runs rarely and its cache is usually cold. |
| `release.yml` | `<variant>-cargo-<lock>` | 🟠 **same defect, NOT fixed — out of scope.** `matrix.variant` encodes OS+features but not CPU. Worse, this lane *ships binaries*: `target-cpu=native` means release artifacts are built for whatever CPU the runner had, which is a separate and arguably larger problem than the cache. Flagged, not touched. |
| `build_cuda_all.yaml` | `${{ runner.os }}-buildx-${{ github.sha }}` | ✅ safe — Docker layer cache, not `target/`, and SHA-keyed. |
| `ci.yml`, `ci_cuda.yaml`, `build_cpu.yaml`, `analysis.yaml`, `docs.yml` | — | ✅ no caches at all. |

## 7. What this cost, and the generalisable lesson

The lane's **first verdict on its first customer was a false positive**, and it
was reported up to Jish as a genuine compile error the new lane had caught. It
was not. The tell was available immediately and was not read: **zero** Rust
diagnostics in the log, and a failure in `mistralrs-paged-attn`, a crate the
diff never touched.

> A CI failure that names crates outside the diff and prints no compiler
> diagnostic is an *environment* failure, not a code failure. Read the first
> error verbatim before attributing it to the change.

Corollary, now in BACKLOG: **master being green is not evidence the lane is
sound.** Master was green *because* it was the cache miss that populated the
poison. The producer never pays; the next PR does.
