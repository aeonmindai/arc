# `arc validate --target-hbm` — HBM Footprint Verification (RUN-191)

Verify that a model + compression-stack pair fits the target HBM ceiling on
the rental host *before* committing to a long benchmark run.

## TL;DR

```bash
# On a CUDA-enabled rental host (H100/B200):
arc validate \
  --model deepseek-ai/DeepSeek-V4-Flash \
  --compression-stack qtip2+td-moe \
  --target-hbm 60 \
  --output tests/results/v4_flash_h100_footprint.json

# Exit code 0 = pass, 1 = fail (residency exceeds target), 2 = bad invocation.

# On a CPU-only / CI host:
arc validate \
  --model deepseek-ai/DeepSeek-V4-Flash \
  --compression-stack qtip2+td-moe \
  --target-hbm 60 \
  --mock
```

## Why this command exists

The Arc launch story is: V4 Flash fits in 60 GB on an H100 thanks to
TurboQuant + TD-MoE. That claim has to be backed by a measurement and a
report that downstream CI / runbooks can read. This command produces that
report — JSON, deterministic schema, exit codes you can branch on.

## Modes

### Real-GPU mode (default)

When invoked without `--mock` on a CUDA host:

1. Bind to CUDA device 0 (override with `ARC_VALIDATE_DEVICE_INDEX=N`).
2. Snapshot free HBM via `cudaMemGetInfo`.
3. Hand off to `MistralRsForServerBuilder::build()` with the user's model id
   and the requested compression stack (see "ISQ mapping" below). The
   builder is driven on a one-off Tokio current-thread runtime so the
   command stays sync. PagedAttention is disabled and `max_seqs=1` so the
   load grabs roughly *just* the model weights, not a 90 %-of-HBM KV
   reservation.
4. Snapshot free HBM again — while still holding a strong reference to the
   loaded `Arc<MistralRs>`, otherwise the allocations would have already
   been freed.
5. Compute breakdown: `used = before − after`, split into weight / workspace
   / KV by subtracting the workspace estimate (KV is treated as zero on the
   gpu path because paged-attn is off).
6. Drop the runtime + builder (releases HBM) and write the JSON report.
   Return 0 (pass) or 1 (fail).

Requires building with `--features cuda`. Without the `cuda` feature, the
command refuses to run (returns exit 2 with a clear message); pass
`--mock` instead.

#### ISQ mapping

| Compression stack | ISQ string passed to mistralrs | Loader behavior |
|---|---|---|
| `bf16` | (none) | model loads at native dtype (BF16) |
| `fp8-only` | `"fp8"` | in-situ quantization to F8E4M3 |
| `nvfp4` | (none) | model is expected to be pre-quantized at rest |
| `qtip2-only` | (none) | model's `config.json` carries `quant_method: "qtip"` |
| `qtip2+td-moe` | (none) | same as `qtip2-only` + TD-MoE expert path |

The `none` cases trust the on-disk quantization. Only `fp8-only` actively
asks the loader to convert weights at load time.

### `--mock` mode

For CI and off-GPU development. Computes the *expected* footprint
analytically from:
- A known-models table (`known_model_params()` in `arc-cli/src/validate.rs`),
  keyed on the HF model id, returning `(total_params, active_params, has_moe)`.
- A bytes-per-parameter table calibrated against published TurboQuant
  numbers (3.5-bit average), FP8, NVFP4, and BF16.

Mock numbers are conservative. The pass/fail logic is identical — a
60-GB target with `qtip2+td-moe` passes, BF16 fails, etc.

## Compression stacks

| Stack | Attention/dense | Experts | Approx. bytes/param (MoE-weighted) |
|---|---|---|---|
| `bf16` | BF16 | BF16 | 2.000 |
| `fp8-only` | FP8 | FP8 | 1.000 |
| `nvfp4` | NVFP4 (block-wise) | NVFP4 | 0.563 |
| `qtip2-only` | QTIP2 3.5-bit | QTIP2 3.5-bit | 0.438 |
| `qtip2+td-moe` | QTIP2 3.5-bit | TD-MoE (sparser) | 0.408 |

Add a new stack by extending `CompressionStack` in `arc-cli/src/validate.rs`
plus the corresponding bytes-per-param constant.

## Output JSON schema

```json
{
  "model": "deepseek-ai/DeepSeek-V4-Flash",
  "compression_stack": "qtip2+td-moe",
  "target_gb": 60.0,
  "measured": {
    "total_gb":      57.98,
    "weight_gb":     53.13,
    "workspace_gb":   4.25,
    "kv_estimate_gb": 0.60
  },
  "pass": true,
  "mode": "mock",
  "gpu": null,
  "notes": [
    "mock mode — numbers come from known_model_params() + bytes-per-param table, not real GPU"
  ]
}
```

On a real-GPU run, `mode` is `"gpu"` and `gpu` is populated with the GPU
name (from `nvidia-smi --query-gpu=name`) and total HBM in GB.

## Exit codes

| Code | Meaning |
|---|---|
| 0 | Pass — `measured.total_gb` ≤ `target_gb` |
| 1 | Fail — residency exceeds target. Report written. |
| 2 | Invocation error (unknown stack, unknown model in mock mode, missing CUDA feature, etc.) |

## Rental host invocation

```bash
# Inside the cloned Arc tree on the rental host, after `./arc-tools/preflight.sh --cuda`:
cargo build --release --features cuda -p arc-cli
./target/release/arc validate \
  --model deepseek-ai/DeepSeek-V4-Flash \
  --compression-stack qtip2+td-moe \
  --target-hbm 60

# Check the JSON the next build step / CI gate will read:
cat tests/results/v4_flash_h100_footprint.json
```

If the JSON `pass` is `false`, the rental verification gate should fail
loudly — don't proceed to long benchmark runs until the residency story is
clean.

## Status

The script itself is **complete** — both the off-GPU (mock) and on-GPU
(real-loader) paths are wired:

- Mock path: shipped + unit-tested in `arc-cli/src/validate.rs`.
- Real-GPU path: shipped — calls `MistralRsForServerBuilder::build()`
  between HBM snapshots; gated behind the `cuda` feature so the no-CUDA
  build still works.

What remains is **hardware verification**, not implementation:

1. **Calibration**: run the command on real hardware against an
   actually-deployed V4 Flash + `qtip2+td-moe` build. The known-good
   numbers from that run become the new floor for the mock estimator.

This is a 1–3 hour H100 rental, performed as part of the benchmark step,
not as part of the script's development.

## Environment variables

| Variable | Default | Effect |
|---|---|---|
| `ARC_VALIDATE_DEVICE_INDEX` | `0` | CUDA device index to bind to (real-GPU mode only). |
| `HUGGING_FACE_HUB_TOKEN` | — | Standard HF token used by the cache loader; required for gated models. |
