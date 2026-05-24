# Arc B200 rental playbook

Day-1 checklist for spinning up an Arc inference engine on rented Blackwell hardware.

## Pre-rental check (free — do BOTH before you pay for a box)

Arc is developed on Apple Silicon, so the CUDA path is never compiled by a Mac
build. Two independent **free** gates catch problems before the rental meter
starts; run both.

**1. Offline CPU suite (local, ~5 min)** — verifies the Rust logic + offline tests:

```bash
./arc-tools/preflight.sh
```

Expected: `✓ ALL CHECKS PASSED` with ~250 tests across arc-engine, mistralrs-quant,
arc-cuda-graph. **This compiles no CUDA code** — it is CPU-only, so a green
preflight says nothing about whether the CUDA build will succeed.

**2. Free CUDA compile gate (GitHub Actions, no GPU)** — cross-compiles the
QTIP + arc-cuda-graph kernels for sm_80 **and** sm_90 with only `nvcc`:

```bash
gh workflow run "CUDA compile check (no GPU)" && gh run watch
```

Green means the rental's step-4 `cargo build --features cuda` will not fail on an
Arc `cuda`-feature *compile* error. See **[CUDA_VALIDATION.md](CUDA_VALIDATION.md)**
for the full three-gate story (CI → Colab → paid-box kernel runtime).

> ⚠️ **flash-attn is not covered by the free gates.** Both free gates compile the
> `cuda` feature only. The rental build — and `preflight.sh --cuda` in Hour 0
> below — use `cuda flash-attn`, and the `flash-attn`-gated code is otherwise
> first compiled on the rented box. To close that gap for free, run it on any box
> with `nvcc` (a free Colab works; flash-attn just makes the build slower):
>
> ```bash
> CUDA_COMPUTE_CAP=90 FEATURES="cuda flash-attn" bash arc-tools/cuda_compile_check.sh
> ```

## Hour 0: rental host setup (60 minutes)

1. **Provision the host.** Recommended: 4× B200 + 8× H100 fallback. Linux distro with CUDA 12.4+ pre-installed.

2. **Clone Arc:**
   ```bash
   git clone https://github.com/aeonmindai/arc.git
   cd arc
   ```

3. **Run rental preflight with CUDA:**
   ```bash
   ./arc-tools/preflight.sh --cuda
   ```
   Expected output:
   - `✓ cargo X.Y.Z`
   - `✓ rustc X.Y.Z`
   - `✓ nvcc 12.X` (CUDA 12.4+ required)
   - `✓ nvidia driver 575.X`
   - `✓ GPU: NVIDIA B200`
   - `✓ core crates build clean`
   - `✓ arc-engine: ~136 tests`
   - `✓ mistralrs-quant: ~87 tests`
   - `✓ arc-cuda-graph: ~28 tests`
   - `✓ mistralrs-core builds with --features cuda`

   (Counts grow as tests are added — the gate is `0 failed`, not an exact number.)

   **If CUDA build fails:** check `cargo build -p mistralrs-core --features cuda 2>&1` output. Common issues:
   - PTX target mismatch — set `CUDA_COMPUTE_CAP=100` for B200 (SM100)
   - Candle version conflict — check `Cargo.lock`
   - cudnn version — verify `apt list cudnn` shows 9.x+

## Hour 1: weight schema validation (15 minutes per model)

For each target model, download the safetensors index (NOT the weights yet — index is ~1 MB):

```bash
# Install HF CLI if needed (provides the `hf` entrypoint; the rental script
# uses `hf download`. On older huggingface_hub the verb is `huggingface-cli`.)
pip install -U huggingface_hub

# Authenticate
hf auth login   # legacy: huggingface-cli login

# Validate V4 Flash
./arc-tools/preflight.sh --cuda --model deepseek-ai/DeepSeek-V4-Flash

# Validate V4 Pro
./arc-tools/preflight.sh --cuda --model deepseek-ai/DeepSeek-V4-Pro

# Validate Kimi K2.6
./arc-tools/preflight.sh --cuda --model moonshotai/Kimi-K2.6-Instruct

# Validate GLM-5.1
./arc-tools/preflight.sh --cuda --model zai-org/GLM-5.1
```

**Expected behavior — three outcomes:**

### Outcome A: All-green ✓
```
Schema validation: OK
  required tensors found: <N>
  required tensors missing: 0
Informational — extra tensors present:
  compressor (V4 CSA/HCA): 24 tensors    # only V4 should have these
  MTP heads: 6 tensors                    # V4 ships with MTP
✓ PRE-FLIGHT OK — model should load through Arc's dispatcher
```
**Next:** proceed to full weight download and `arc run`.

### Outcome B: Missing o_proj.weight
```
Schema validation: FAIL
  Missing:
    - model.layers.0.self_attn.o_proj.weight (OR o_a_proj + o_b_proj)
    - ... (one per layer)
```

This means V4 uses neither single nor LoRA o_proj — some other naming convention. **Action:**
1. Inspect the actual safetensors index keys:
   ```bash
   python -c "import json; d = json.load(open('model.safetensors.index.json')); print([k for k in d['weight_map'] if 'o_' in k][:10])"
   ```
2. Whatever the naming convention turns out to be, update `arc-engine/src/weight_schema.rs::validate_v4_against_keyset` and `mistralrs-core/src/models/deepseek4.rs::Attention::new` to match.
3. Rebuild + re-validate.

### Outcome C: Missing q_a/q_b_proj or kv_a/kv_b_proj
This would mean V4 doesn't use MLA-LoRA decomposition, which would be surprising. **Action:** consult the published V4 paper to confirm attention layout.

## Hour 2: actual weight download + first load (30 minutes)

Once preflight is green for the target model:

```bash
# Build the release binaries with CUDA (same features the rental script uses).
cargo build --release -p arc-cli -p mistralrs-cli --features "cuda flash-attn"

# Put them on PATH for this shell. `arc run`/`arc serve` exec the `mistralrs`
# binary by name, so it MUST be resolvable — otherwise they fail with
# "Failed to execute mistralrs". (`arc validate`/`arc bench` run in-process and
# work from ./target/release/arc without this.)
export PATH="$PWD/target/release:$PATH"

# Download full weights (V4 Flash ≈ 100GB, V4 Pro ≈ 2TB)
hf download deepseek-ai/DeepSeek-V4-Flash   # legacy: huggingface-cli download ...

# First run — interactive (interactive is the default mode of `run`; there is
# no --interactive flag).
arc run -m deepseek-ai/DeepSeek-V4-Flash -a deepseekv4

# Engage MTP speculative decode (V4-only; ~1.8× lossless speedup at depth=4)
arc run -m deepseek-ai/DeepSeek-V4-Flash -a deepseekv4 --mtp-depth 4

# OR serve OpenAI-compatible API
arc serve -p 1234 -m deepseek-ai/DeepSeek-V4-Flash -a deepseekv4

# OR run the AA-AgentPerf benchmark (in-process; needs the cuda build above)
arc bench -m deepseek-ai/DeepSeek-V4-Flash
```

**What to expect on first load:**

1. **Tensor mismatch errors:** V4 has its own loader (`-a deepseekv4` → `DeepSeekV4Loader`). `arc validate --index <index.json> --arch deepseekv4` (step above) checks the schema offline first; if a tensor is missing/mismatched at load it fails with a clear "expected shape X, got Y" message.
2. **CSA/HCA, mHC, indexer/FlashMLASparse are implemented** (RUN-138 / RUN-164 / RUN-163, all Done) and engaged by the `deepseekv4` path — they are **not** absent. This rental is their first end-to-end *hardware* validation: confirm coherent text + per-layer parity first, then chase tok/s. Do not assume correctness from a green compile.
3. **MTP heads dispatched via `--mtp-depth N` (RUN-156 / RUN-121):** V4 Flash ships with one MTP head. Pass `--mtp-depth 4` to engage `MtpSpeculativePipeline` — the engine wraps the target pipeline at construction time and drafts up to N tokens per target forward (acceptance ≈ 50% on greedy decode → ~1.5-1.8× wallclock speedup, lossless by construction). For non-V4 models the flag logs a warning and falls back to bare-target decode automatically.

## Hour 3: baseline numbers (30 minutes)

**Authoritative one-shot:** for the full composed-stack number (QTIP 2-bit +
TurboQuant KV + TD-MoE + MTP + mHC + CSA/HCA, no opt-outs) on a fresh H100,
run the canonical script — it builds, runs the QTIP GPU kernel smoke tests
*before* the weight download, then benches V4 Flash end to end:

```bash
bash arc-tools/rental_h100_v4_flash.sh   # writes /ephemeral/arc-v4flash-bench.json
```

For ad-hoc shape sweeps, use the real bench flags. `mistralrs bench` controls
prompt/gen shape and concurrency (TurboQuant KV + `--isq qtip2` apply). Note the
flag names: concurrency is `--max-seqs` (there is **no** `--batch-size`; device
mapping uses `--max-batch-size`), and `--prompt-len`/`--gen-len`/`--max-seq-len`
set the workload shape:

```bash
# Single-user latency (fixed prompt/gen shape)
mistralrs bench -m deepseek-ai/DeepSeek-V4-Flash -a deepseekv4 --isq qtip2 \
  --prompt-len 4096 --gen-len 512 --max-seqs 1

# Aggregate throughput (concurrency sweep)
mistralrs bench -m deepseek-ai/DeepSeek-V4-Flash -a deepseekv4 --isq qtip2 \
  --prompt-len 4096 --gen-len 512 --max-seqs 64

# Long-context (large prompt)
mistralrs bench -m deepseek-ai/DeepSeek-V4-Flash -a deepseekv4 --isq qtip2 \
  --prompt-len 131072 --gen-len 256 --max-seqs 1
```

For the SLO-tiered AA-AgentPerf ramp (binary-search to the max concurrent users
that pass a tier), use `arc bench` (its real flags are `--slo-tier`,
`--max-users`, `--headless`, `--mock`):

```bash
arc bench -m deepseek-ai/DeepSeek-V4-Flash --slo-tier 2 --max-users 64 --headless
```

**Baselining note:** M1's gate is *coherent end-to-end decode at ~1,000 tok/s
on V4 Flash*, not headline long-context numbers. Capture the script's
`tok_per_s_decode` and TTFT as the day-1 baseline; treat vendor-parity
(SGLang/vLLM) comparison as post-rental / M2 (`arc bench` vendor-parity is not
yet wired). This is the moment "rent and watch numbers" becomes "rent and
identify the next pieces to ship."

## Common failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| `cargo build --features cuda` fails with PTX error | SM target mismatch | `export CUDA_COMPUTE_CAP=100` (B200) or `90` (H100) |
| "Tensor X not found in safetensors" | V4 layout mismatch | Update weight_schema + deepseek4.rs Attention struct |
| "Shape mismatch: expected [N, 128], got [N, 448]" | qk_nope_head_dim hardcoded somewhere | Find + fix the hardcode (should use config) |
| `arc validate` says OK but `arc run` crashes | Issue is downstream of name validation (compute kernel, dtype, etc.) | Run with `RUST_LOG=debug` to find the layer where it crashes |
| Loads but produces garbage tokens | RoPE or MLA dispatch wrong | Compare layer-0 hidden states to PyTorch reference |
| Loads but very slow | No CUDA dispatch — falling back to CPU | Verify `--features cuda` was set; check `nvidia-smi` for utilization |

## Decision points

After hour 3 baseline:

- **If V4 loads and produces correct output at V3-quality speed:** ship Arc v2 with the dispatcher path; queue CSA/HCA TileLang for v2.1.
- **If V4 loads but output is garbled:** the V3 forward path has subtle V4 incompatibilities. Debug layer-by-layer.
- **If V4 doesn't load:** address the specific missing tensor before any other work.

## Models worth testing in priority order

1. **DeepSeek V4 Flash** (smaller, faster iteration)
2. **Llama-3.1-8B** (sanity check that Arc still works on a baseline model)
3. **Kimi K2.6** (verify the K2 prefix remap is needed and document the fix)
4. **GLM-5.1** (verify DSA dispatch)
5. **DeepSeek V4 Pro** (the headline; large, slow to download)

## Cost estimate

- 4× B200 spot: ~$25-40/hr
- Day 1 (load + validate + baseline): ~$200-300
- Day 2 (CSA/HCA wiring + re-test): ~$300-500
- Day 3 (perf vs SGLang side-by-side): ~$200-300

**Total: $700-1100 for "Arc v2 ships with V4 support."**

## When to call it done

Arc v2 launch is shipped when:
- V4 Flash + V4 Pro load correctly through Arc
- Single-user @ 1M context ≥ 30 tok/s (current V3-quality estimate ≈ 5-10, after CSA/HCA ≈ 50-100)
- Aggregate throughput ≥ 50% of SGLang's number on identical hardware
- Quality drift from BF16 < 2% on standard benchmarks
- No crashes for 1-hour sustained run with 50 concurrent users

After that, customer pilot starts.
