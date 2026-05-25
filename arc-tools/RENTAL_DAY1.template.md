<!--
  TEMPLATE for the M1 day-1 rental report (deliverable RUN-161).
  Copy this to arc-tools/RENTAL_DAY1.md ON THE BOX and fill every <…> blank
  from the live run. Do NOT commit this template as the report — the report is
  RENTAL_DAY1.md (no `.template`). M1 is not closed until RENTAL_DAY1.md exists
  with real numbers. Authoritative bar: arc-tools/M1_GATE.md.

    cp arc-tools/RENTAL_DAY1.template.md arc-tools/RENTAL_DAY1.md
-->

# Arc M1 — V4 Flash Day-1 Rental Report

- **Date (UTC):** `<YYYY-MM-DD>`
- **Operator:** `<name>`
- **Arc commit:** `<git log -1 --format=%H>`
- **Provider / instance:** `<e.g. Lambda 1×H100-80GB-SXM>` · **$/hr:** `<…>`
- **Outcome:** `<GO — M1 closed | NO-GO — see blockers>`

---

## 1. Hardware + topology (`nvidia-smi`)

```
<paste: nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv>
<paste: nvidia-smi topo -m   (if multi-GPU)>
```

- GPU: `<name>` · HBM: `<GB>` · driver: `<ver>` · compute_cap: `<sm_XX>`
- nvcc: `<nvcc --version | grep release>`

## 2. Pre-flight (`./arc-tools/preflight.sh --cuda`)

```
<paste the last ~20 lines — must end with "✓ ALL CHECKS PASSED", "0 failed",
 nvcc 12.4+, GPU detected, mistralrs-core builds with --features cuda>
```

- Build line: `cargo build --release -p arc-cli -p mistralrs-cli --features "cuda flash-attn"` → `<ok / fail + time>`
- `export PATH="$PWD/target/release:$PATH"` done? `<y/n>`

## 3. QTIP GPU kernel smoke (rental script step 4b — BEFORE the download)

```
<paste tail of /tmp/qtip_gpu_smoke.log — must show "cos sim" lines ≥ 0.999,
 and must NOT show "CUDA not available; skipping">
```

- Result: `<PASS — parity ≥ 0.999 | FAIL — STOP, code defect, do not download>`

## 4. HBM footprint (`arc validate --target-hbm`, deliverable RUN-191)

```
arc validate --target-hbm 60 -m deepseek-ai/DeepSeek-V4-Flash \
  --compression-stack qtip2+td-moe \
  --output tests/results/v4_flash_h100_footprint.json
<paste JSON>
```

- `pass`: `<true/false>` · measured `total_gb`: `<…>` (weight `<…>` / workspace `<…>` / kv `<…>`)

## 5. Weight download timing (rental script step 5)

- Model: `deepseek-ai/DeepSeek-V4-Flash` · on-disk: `<du -sh output>` · shards: `<n>`
- Wall time: `<start → end, minutes>` · preemptions/resumes: `<n>`

## 6. The four M1 acceptance criteria

| # | Criterion | Result | Evidence |
|---|---|---|---|
| C1 | Coherent end-to-end decode | `<PASS/FAIL>` | `<first ~40 words of the V4 paragraph>` |
| C2 | All 3 forward paths route through V4 compress dispatch | `<PASS/FAIL>` | two runs below + `compress_ratios` histogram |
| C3 | Numerical stack-composition (≥0.95/layer, first-100 greedy) | `<PASS/FAIL>` | `<per-layer cos-sim; greedy-match count>` |
| C4 | Baseline decode tok/s + TTFT captured | `<PASS/FAIL>` | bench JSON below |

### C2 — the two runs (see M1_GATE.md criterion 2)

- `compress_ratios` from `config.json`: Standard(0)=`<n>` · CSA(4)=`<n>` · HCA(128)=`<n>`  ← CSA/HCA must be > 0
- Run A `--paged-attn off` (branch C, all layers → dsv4_attention): `<coherent? decode T/s>`
- Run B `--paged-attn auto --pa-cache-type turboquant` (branch A standard + branch B compressed): `<coherent? decode T/s>`

## 7. Baseline bench (criterion 4)

```
<paste /ephemeral/arc-v4flash-bench.json produced by rental_h100_v4_flash.sh>
```

- **`tok_per_s_decode`: `<…>`** · TTFT: `<…>s` · prompt T/s: `<…>`
- (Expectation is ~1,000 tok/s; the *number is not the gate* — a trustworthy capture is. This sets where M2 starts.)

## 8. Issues / deviations / cost

- Blockers hit + `FAIL:` markers + fix applied: `<…>`
- Total billable hours: `<…>` · total $: `<…>`

## 9. Sign-off

- M1 verdict: `<GO / NO-GO>` — all four criteria ✓ **and** deliverables committed?
- Committed deliverables: `RENTAL_DAY1.md` `<y>` · `tests/results/validation_<date>.md` `<y>` · `tests/results/v4_flash_h100_footprint.json` `<y>`
- Linear updated (RUN-161, RUN-136, RUN-137): `<y/n>`
