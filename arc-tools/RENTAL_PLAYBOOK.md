# Arc B200 rental playbook

Day-1 checklist for spinning up an Arc inference engine on rented Blackwell hardware.

## Pre-rental check (5 minutes, free)

On your local box, verify the offline test suite is green:

```bash
./arc-tools/preflight.sh
```

Expected: `✓ ALL CHECKS PASSED` with ~200 tests across arc-engine, mistralrs-quant, arc-cuda-graph.

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
   - `✓ arc-engine: 123 tests`
   - `✓ mistralrs-quant: 58 tests`
   - `✓ arc-cuda-graph: 11 tests`
   - `✓ mistralrs-core builds with --features cuda`

   **If CUDA build fails:** check `cargo build -p mistralrs-core --features cuda 2>&1` output. Common issues:
   - PTX target mismatch — set `CUDA_COMPUTE_CAP=100` for B200 (SM100)
   - Candle version conflict — check `Cargo.lock`
   - cudnn version — verify `apt list cudnn` shows 9.x+

## Hour 1: weight schema validation (15 minutes per model)

For each target model, download the safetensors index (NOT the weights yet — index is ~1 MB):

```bash
# Install HF CLI if needed
pip install huggingface_hub

# Authenticate
huggingface-cli login

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
2. Whatever the naming convention turns out to be, update `arc-engine/src/weight_schema.rs::validate_v4_against_keyset` and `mistralrs-core/src/models/deepseek3.rs::Attention::new` to match.
3. Rebuild + re-validate.

### Outcome C: Missing q_a/q_b_proj or kv_a/kv_b_proj
This would mean V4 doesn't use MLA-LoRA decomposition, which would be surprising. **Action:** consult the published V4 paper to confirm attention layout.

## Hour 2: actual weight download + first load (30 minutes)

Once preflight is green for the target model:

```bash
# Download full weights (V4 Flash ≈ 100GB, V4 Pro ≈ 2TB)
huggingface-cli download deepseek-ai/DeepSeek-V4-Flash

# First run — interactive
arc run -m deepseek-ai/DeepSeek-V4-Flash --interactive

# OR serve OpenAI-compatible API
arc serve -p 1234 -m deepseek-ai/DeepSeek-V4-Flash

# OR run benchmarks
arc bench -m deepseek-ai/DeepSeek-V4-Flash
```

**What to expect on first load:**

1. **Tensor mismatch errors:** the V4 loader currently routes through V3's loader. If V4 has any tensor that V3 doesn't expect (or vice versa), load will fail with a clear "expected shape X, got Y" message.
2. **CSA/HCA dispatch absent:** V4 will run as dense MLA (V3-quality). Performance will be V3-class, not the headline V4 numbers. CUDA TileLang for CSA/HCA is the next-day rental work.
3. **MTP heads ignored:** V4's `mtp.layers.*` tensors are loaded but not dispatched. The 1.8× MTP speedup is gated on speculative pipeline integration (RUN-156).

## Hour 3: baseline numbers (30 minutes)

Run benchmarks at multiple settings:

```bash
# Single-user latency
arc bench -m deepseek-ai/DeepSeek-V4-Flash --batch-size 1 --max-seq-len 32768

# Aggregate throughput
arc bench -m deepseek-ai/DeepSeek-V4-Flash --batch-size 64 --max-seq-len 4096

# Long-context (V4's strength)
arc bench -m deepseek-ai/DeepSeek-V4-Flash --batch-size 1 --max-seq-len 524288
```

**Expected baseline (Arc-as-V3, no CSA/HCA):**
- Single-user @ 32K: ~50-70 tok/s on 4× B200
- Aggregate @ 4K × 64 batch: ~3000-5000 tok/s
- Long-context @ 512K: ~5-10 tok/s (dense MLA is slow at this scale)

**After CSA/HCA wiring (rental cycle 2):**
- Single-user @ 32K: ~150-200 tok/s
- Long-context @ 512K: ~50-100 tok/s (CSA's 73% FLOP reduction kicks in)

**vs SGLang baseline on same hardware:**
- Single-user: SGLang is 2-5× faster today (mature CUDA kernels)
- Aggregate: SGLang is 1.5-2× faster
- Long-context: SGLang has CSA/HCA already; ours is dense — they win.

This is the moment when "rent and watch numbers" becomes "rent and identify next pieces to ship."

## Common failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| `cargo build --features cuda` fails with PTX error | SM target mismatch | `export CUDA_COMPUTE_CAP=100` (B200) or `90` (H100) |
| "Tensor X not found in safetensors" | V4 layout mismatch | Update weight_schema + deepseek3.rs Attention struct |
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
