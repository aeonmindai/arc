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
| 1. `cuda_compile_check.yaml` | GitHub Actions | No | Free, automatic | nvcc errors, FFI drift, `cuda`-feature Rust compile errors in the **library** crates — for sm_80 **and** sm_90. Does **not** compile `flash-attn` or link the CLI binaries (see gate 1b). |
| 1b. `flash_attn_compile_check.yaml` | GitHub Actions | No | Free, manual | The `flash-attn` feature (candle-flash-attn nvcc compile) the rental uses but gate 1 omits — for the chosen arch (default sm_90). See "flash-attn coverage" below. |
| 1c. `qtip_beam_res_usage_check.sh` | Step *inside* gate 1 | No | Free, automatic | **Local-memory spills in the QTIP beam kernels** — `cuobjdump -res-usage` must report `LOCAL:0 STACK:0`. Both beam kernels demand this check by name in their own source and nothing ran it before. See "the spill gate" below. |
| 2. `colab_cuda_build_check.ipynb` | Google Colab | nvcc only | Free, manual | Same as gate 1, plus it *links the CLI binaries* (`cuda_compile_check.sh` step 4, `FEATURES=cuda`); also runtime tests **iff** Colab gives sm_80+ |
| 3. `cuda_compile_check.sh` (GPU mode) + rental step 4b | Rental / sm_80+ box | Yes | Paid box | Kernel **runtime**: parity, the prefix-grouped Viterbi quantize kernel actually running, no hang |

### What CANNOT be done for free, and why
Free Colab/Kaggle GPUs are **T4 (sm_75)**. The QTIP kernels are gated to
**sm_80+** (`has_qtip_kernels` in `mistralrs-quant/build.rs`), so **no free GPU
can run them**. Runtime validation of the QTIP path genuinely requires an
sm_80+ device — the paid rental, or a Colab Pro A100. We do not pretend
otherwise. Everything *compilable* is validated for free; only *execution* of
the sm_80 kernels needs the paid box.

### The spill gate — a resource check, not a compile check (gate 1c)
Compiling is not the only thing that can go silently wrong for free. Both beam
kernels (`qtip_beam.cu`, `qtip2b_beam.cu`) keep a per-thread `cand[]` array in
registers and raise `__launch_bounds__(256, 4)` to cap registers at 64.
**`__launch_bounds__` does not refuse to compile when it cannot reach that
budget — it SPILLS to local memory**, and a spilled load inside the radix loop
runs ~`CAND x 3.87` times per timestep, costing more than the occupancy it
bought. Both files name the check in their source ("`LOCAL:` must remain 0").
Nothing ran it.

`arc-tools/qtip_beam_res_usage_check.sh` runs it, with `build.rs`'s exact nvcc
flags (a different flag set is a different register allocation, i.e. a
measurement of a kernel we do not ship), for the matrix arch inside gate 1:

```bash
arc-tools/qtip_beam_res_usage_check.sh sm_90
```

It is built so it cannot pass vacuously — the failure mode of every resource
gate that greps for a pattern:
* a **negative control** kernel (a dynamically-indexed 256-float local array,
  written to a temp dir, not part of the build) goes through the *same* parser,
  and the script fails if that is not reported as a spill;
* a **kernel-count assertion** per source, so a template instantiation that
  quietly stopped being emitted cannot make "no spill" a statement about a
  kernel that is no longer there.

This is also the check that arbitrates encode-cost claims. Hand-counting C++
undercounts SASS by ~2.05x on this kernel family; a spilled `cand[]` is exactly
what produced the retracted "~1,700 s/layer" K=8/V=4/L=12 encode estimate. A
number you did not compile is an estimate.

### flash-attn coverage — now free, one-click (gate 1b)
The rental's step-4 build is `cargo build -p arc-cli -p mistralrs-cli --features
"cuda flash-attn"`. Gates 1 and 2 above compile the **`cuda` feature only** —
neither turns on `flash-attn`. A compile error gated specifically behind
`flash-attn` (not `cuda`) — candle-flash-attn `build.rs` / cutlass / arch drift —
would first surface on the paid box at step 4.

**Closed for free by `flash_attn_compile_check.yaml` (GitHub Actions, no GPU).**
`flash-attn` implies `cuda`, so the job runs `nvcc` on the FA kernels (and Arc's
own kernels) by building `mistralrs-core --lib --features flash-attn` — a pure
rlib build, so no final `-lcuda` link is attempted (GPU-less-safe). It is
`workflow_dispatch`-only because the FA kernel compile is heavy; run it before a
rental:
```bash
# sm_90 (H100 rental target) is the default input; -R required (dual-remote clone)
gh workflow run "flash-attn compile check (no GPU)" -R aeonmindai/arc
gh run watch -R aeonmindai/arc
```
Green = the rental's step-4 `cuda flash-attn` build will not fail on a flash-attn
**compile** error for that arch. It does **not** do the final CLI-binary `-lcuda`
link or run any FA kernel — those stay on the box (`preflight.sh --cuda`, the
rental run, or `cuda_compile_check.sh` on a borrowed nvcc box):
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
   CLI-binary link — see step 1b to close the flash-attn gap for free.

1b. **Close the flash-attn compile gap (free, one-click).** Trigger gate 1b and
    watch it go green before paying for a box:
   ```bash
   gh workflow run "flash-attn compile check (no GPU)" -R aeonmindai/arc
   gh run watch -R aeonmindai/arc
   ```
   Green = the rental's `--features "cuda flash-attn"` build will not fail on a
   flash-attn *compile* error for sm_90. (Final `-lcuda` link + kernel runtime
   still happen on the box.)

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
