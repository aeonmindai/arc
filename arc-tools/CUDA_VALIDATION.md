# CUDA validation — catch GPU problems before the rental

Arc is developed on Apple Silicon (Metal/CPU). The CUDA kernels are **never
compiled by a Mac build**, so historically the first time `nvcc` ever saw them
was `cargo build --features cuda` on a **paid GPU rental**. A syntax error, an
FFI signature drift, or a feature-gated Rust compile error then surfaced only
after the box was already running — which sent work back to the Mac, where the
CUDA error can't be reproduced. That loop is what this tooling exists to break.

**Key fact:** compiling CUDA needs only the toolkit (`nvcc`), **not a GPU**.
`cudaforge` reads `CUDA_COMPUTE_CAP` before probing `nvidia-smi`
(`compute_cap.rs` detect order: env → nvidia-smi), so a GPU-less machine can
cross-compile the kernels for the rental's real arch (sm_80 = A100,
sm_90 = H100/Hopper).

## The three gates (cheapest first)

| Gate | Where | GPU? | Cost | Catches |
|------|-------|------|------|---------|
| 1. `cuda_compile_check.yaml` | GitHub Actions | No | Free, automatic | nvcc errors, FFI drift, `cuda`-feature Rust compile errors in the **library** crates — for sm_80 **and** sm_90. Does **not** compile `flash-attn` or link the CLI binaries (see "flash-attn coverage" below). |
| 2. `colab_cuda_build_check.ipynb` | Google Colab | nvcc only | Free, manual | Same as gate 1, plus it *links the CLI binaries* (`cuda_compile_check.sh` step 4, `FEATURES=cuda`); also runtime tests **iff** Colab gives sm_80+ |
| 3. `cuda_compile_check.sh` (GPU mode) + rental step 4b | Rental / sm_80+ box | Yes | Paid box | Kernel **runtime**: parity, the prefix-grouped Viterbi quantize kernel actually running, no hang |

### What CANNOT be done for free, and why
Free Colab/Kaggle GPUs are **T4 (sm_75)**. The QTIP kernels are gated to
**sm_80+** (`has_qtip_kernels` in `mistralrs-quant/build.rs`), so **no free GPU
can run them**. Runtime validation of the QTIP path genuinely requires an
sm_80+ device — the paid rental, or a Colab Pro A100. We do not pretend
otherwise. Everything *compilable* is validated for free; only *execution* of
the sm_80 kernels needs the paid box.

### flash-attn coverage — not free by default
The rental's step-4 build is `cargo build -p arc-cli -p mistralrs-cli --features
"cuda flash-attn"`. Both free gates above compile the **`cuda` feature only** —
gate 1 builds the library crates, gate 2 also links the CLI binaries, but neither
turns on `flash-attn`. So a compile error gated specifically behind `flash-attn`
(not `cuda`) is first seen on the paid box at step 4 (or in `preflight.sh --cuda`,
which also runs on the rented host). To close that gap for free, run the full
feature set on any box with `nvcc` — a free Colab works, flash-attn just makes
the build slower (force sm_90 to avoid the sub-sm_80 `qtip_*.cu` footgun noted at
the bottom of this doc):
```bash
CUDA_COMPUTE_CAP=90 FEATURES="cuda flash-attn" bash arc-tools/cuda_compile_check.sh
```

## Pre-rental checklist (do this, then the rental only runs one command)

1. **Watch GitHub Actions go green.** `cuda_compile_check.yaml` runs on every
   push to `master` / PR that touches `*.cu`, `build.rs`, or the quant /
   cuda-graph / arc-engine crates. Trigger manually any time:
   ```bash
   # -R is required in a clone with both origin (aeonmindai/arc) + upstream
   # remotes — gh otherwise resolves to upstream and reports no such workflow.
   gh workflow run "CUDA compile check (no GPU)" -R aeonmindai/arc
   gh run watch -R aeonmindai/arc
   ```
   Green = the rental's step-4 build will not fail on an Arc `cuda`-feature
   *compile* error. It does **not** cover the `flash-attn` feature or the final
   CLI-binary link — see "flash-attn coverage" above to close that gap for free.

2. **(Optional) Colab second opinion.** Open
   `arc-tools/colab_cuda_build_check.ipynb` in Colab, set the commit, Run all.
   Uses a real `nvcc`; runs kernels too if it lands an sm_80+ GPU.

3. **Rent and run.** The rental box only needs:
   ```bash
   bash arc-tools/rental_h100_v4_flash.sh
   ```
   It builds (step 4), then **step 4b runs the QTIP GPU parity tests on the real
   GPU before the 148 GB download** — a kernel hang/crash fails in ~1 min instead
   of after a huge download + 30 min of ISQ. If you have a borrowed sm_80+ box
   and want only the compile+kernel gate:
   ```bash
   bash arc-tools/cuda_compile_check.sh        # auto-detects arch, runs GPU tests if present
   ```

## Hard rule: no CPU fallback for QTIP GPU quantize

On a CUDA device the QTIP quantize path has **no CPU fallback and no env
bypass**. `mistralrs-quant/src/qtip/mod.rs` (`quantize_with_options_concrete`):

- If the tensor is on CUDA but the kernels weren't compiled in → `bail!`
  (rebuild with sm_80+), it does **not** quantize on CPU.
- If the GPU path returns `None` → `bail!` telling the caller to fix
  preconditions (F32 / contiguous / supported rotation block), it does **not**
  detour to CPU.

A broken kernel must therefore surface as an **error**, never as a silent
slow CPU quantize. The rental step-4b smoke test and the `cuda_compile_check.sh`
GPU mode enforce this in practice: they run the real kernels and assert
parity ≥ 0.999 against the CPU reference.

> ⚠️ Do **not** reintroduce a `Device::Cpu` detour inside the `#[cfg(feature =
> "cuda")]` block in `quantize_with_options_concrete`, and do **not** re-add an
> `ARC_FORCE_GPU_QTIP_QUANTIZE`-style env gate. Both were removed deliberately
> (commit `12527af2d`).

## Known latent footgun (not blocking — surfaced for a decision)

`mistralrs-quant/build.rs` excludes `marlin_*.cu / *_fp8*.cu / *_wmma.cu` for
compute capability < 80, but **not** `qtip_*.cu`. A sub-sm_80 build therefore
asks `nvcc` to compile QTIP kernels that need sm_80 intrinsics and fails
confusingly, even though `has_qtip_kernels` is already false for that arch. It
does not affect the sm_80/sm_90 gates above (we always compile-check at sm_90),
which is why it's not fixed here. If you want sub-sm_80 builds to compile
cleanly (skipping QTIP), add `qtip_*.cu` to the `cc < 80` exclude list.
