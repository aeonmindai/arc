# wave61-CL — TurboQuant on every head dim

Branch `feat/turboquant-every-model`, draft PR **#87**, base `master` @ `c763123be`.

---

## 1. Did it ever work, and via which path?

Jish: *"turboquant works in arc tried and tested when we built it months ago."*
Both his testimony and the tree's `#[allow(dead_code)]` are true, because there
are two paths and only one of them ever ran.

### NormalCache (eager) path — never ran, and never on a GPU

| SHA | date | what |
|---|---|---|
| `7dc99d394` | 2026-03-31 | feature born; `#[default]` on `PagedCacheType::TurboQuant` from the very first commit |
| `152e77b3f` | 2026-03-31 | adds `set_turboquant_head_dim` **and its only call site ever**, in `pipeline/normal.rs`. Commit message: *"1. Disables paged attention (falls back to eager/normal cache)"* |
| `01a6ef743` | 2026-03-31 (same day) | **removes that call site.** Message: *"Disabling paged attention also disables GPU acceleration (Metal/CUDA), making the engine CPU-only."* |
| `fb181275e` | 2026-05-26 | clippy-greening annotates the now-orphaned setter: `+#[allow(dead_code)] // TurboQuant config setter; not yet wired into the active path` |

Zero callers for 16 months ⇒ `TURBOQUANT_HEAD_DIM` was permanently `0` ⇒
`NormalCache::new` always took the plain branch. `KvCache::new_turboquant` had
exactly one call site, inside that unreachable branch. The whole `arc-turbo`
crate (~300 lines) was unreachable dead code with **zero tests**, and its `cuda`
feature is enabled by no CI lane (`wave31-BG`).

`git grep 'KvCache::TurboQuant\|new_turboquant' -- '*test*'` → **zero hits**.
`git log --diff-filter=D --name-only --all | grep -i turbo` → **zero**. No
TurboQuant cache test was ever deleted; none ever existed.

### Paged path — almost certainly ran once, April 2026, head_dim 128 only

- `849477ce6` (2026-03-31) wired the paged path end to end.
- Eight consecutive CUDA correctness fixes on 2026-04-02 — `fix: correct V cache
  stride mismatch`, `fix: eliminate deadlock and fix Q·K warp reduction`
  (*"All lanes collaborate on ONE token (not one token per lane)"*). These bug
  classes are not findable by inspection; someone ran the kernels.
- `4eba13905` (2026-04-06): **`55 tok/s with TurboQuant = 46% over Candle
  baseline`**, B200 / CUDA 12.8. Model, context length, and quality unstated.
  This is the only throughput number ever tied to TurboQuant.
- `329b74380` (2026-04-04), docs-only, 2 files: *"Qwen3-32B single H100:
  39K→169K context, 4.27x KV cache compression. Lossless quality confirmed
  end-to-end."*

**The 4.27× is format arithmetic, not measurement.** Qwen3-32B has 8 KV heads ×
head_dim 128: FP16 `8×128×2×2 = 4096`; K4 `8×64 = 512`, V3 `8×⌈128/10⌉×4 = 416`,
f16 norms `8×2×2 = 32` → **exactly 960**. Derivable from the block layout with no
forward pass. `Lossless quality confirmed end-to-end` has no eval score, no
perplexity, no LongBench number, no RUN-id, no log — in a repo that records GSM8K
to one decimal with n, shots, seed and token cap. **No RUN-### is attached to any
TurboQuant measurement anywhere in `memory/`.**

Two further facts undercut it. `daf525f00` (2026-08-13) establishes that from
2026-03-31 to 2026-08-13 — the entire period TurboQuant was `#[default]` and
shipping — any model with head_dim ≠ 128 *"decoded pure garbage"* silently,
because the kernels `return` early leaving an uninitialised F32 output buffer.
Nobody noticed, which is only possible if the default path was essentially never
exercised in serving. And `docs/BENCHMARKS.md:447` already concedes *"MLA-attention
models — including V4 Flash — currently fall back to the standard KV path, so the
KV compression gain does not yet apply."*

### Verdict

The brief's hypothesis is **confirmed**. It worked through the **paged** path on a
**head_dim = 128** model (Qwen3-32B, April 2026). V4 — head_dim 512, MLA layout,
paged disabled — can reach it through neither. `arc-engine/tests/numerical_stack_composition.rs`
does exercise `quantize_vector`/`dequantize_vector` directly at HEAD_DIM 64, but
never touches `TurboQuantCache` or any kernel, and its own header disclaims the
quality claim.

---

## 2. Why the head-dim limit existed

| Layer | Mechanism | Real constraint? |
|---|---|---|
| Codebooks | `get_codebook` was a `match` over three hand-pasted `(dim, bits)` tables (64/128/256 × 2/3/4 bits) and `panic!`ed otherwise | **No.** The Lloyd–Max codebook is a closed-form function of the dimension; nobody had written the function. |
| WHT | `fwht_inplace` needs a power-of-two order | **Yes** — but it does not explain excluding 512, which is a power of two. |
| CUDA | `static_assert(HEAD_SIZE == 128)`, `__constant__ SIGNS_128[128]`, `tq_rotate_128`, `if (hs != 128) return;`, `__shared__ float s[128]` | **No** — fixed template instantiation and fixed constant tables. Not shared-memory sizing: a 512-float scratch is 2 KB. |
| Rust gate | `head_dim == 64 \|\| head_dim == 128 \|\| head_dim == 256` in `NormalCache::new` | **No** — mirrored the table. |
| Paged resolver | `TURBOQUANT_HEAD_DIM: usize = 128`, exact k/v match | **Yes, today** — it correctly reflects the kernel. |

The mathematics is on 512's side: `total_distortion_is_dimension_free` measures the
Lloyd–Max distortion at 2/3/4 bits for `dim ∈ [8, 1024]` and it is flat at
`0.117 / 0.0345 / 0.0095` — the Gaussian limit, approached from below. **512 is
the easiest case, not the hardest.**

---

## 3. What works now

`mistralrs-quant/src/turboquant/generate.rs` — Lloyd–Max for the `S^{d-1}`
coordinate marginal `f_d(x) ∝ (1-x²)^((d-3)/2)`. The centroid-rule numerator is
analytic (`∫x(1-x²)^m dx = -(1-x²)^(m+1)/(2(m+1))`), so only the cell mass needs
quadrature; ~10 ms per `(dim, bits)`, memoised.

The module doc on `codebook.rs` said `Beta(d/2, d/2)`; that is **off by a half in
both shape parameters**. `(d-3)/2` is the exponent that reproduces the shipped
tables — `(d-2)/2` and `(d-1)/2` were both tried and both miss:

| exponent | d=128 b=4 centroid max-err | mse_per_coord vs shipped |
|---|---|---|
| `(d-1)/2` | 1.83e-3 | 0.0000716741 vs 0.0000727717 |
| `(d-2)/2` | 9.36e-4 | 0.0000722188 |
| **`(d-3)/2`** | **6.8e-5 → 1.4e-4 (analytic)** | **0.0000727721, Δ = 4e-10** |

`layout.rs` — block decomposition for non-powers-of-two:
`80 → 64+16`, `96 → 64+32`, `112 → 64+32+16`, `192 → 128+64`, residue `< 8` →
one zero-padded 8-block (`100 → 64+32+8`). One norm per block. Chosen over
pad-to-next-power-of-two because it is strictly cheaper (head_dim 80 at 4 bits:
44 bytes blocked vs 66 padded) **at the same error bound** — splitting a unit
vector into blocks with norms `n_j` gives `Σ n_j² · D_b = D_b`, since each block
uses a codebook generated for its own dimension.

A power-of-two head dim yields exactly one block, and
`single_block_layout_is_bit_identical_to_the_original_path` asserts bit-for-bit
agreement with the pre-change `quantize_vector`/`dequantize_vector` at
64/128/256 × 2/3/4 bits. **Nothing that was validated moved.**

**Working head dims: 64, 80, 96, 112, 128, 192, 256, 512** (and anything else
`≥ 8`), K and V independently. Measured round-trip error at every one, against
`sqrt(D_b)`.

---

## 4. Default-on status

| Path | Default | Head dims | Proof |
|---|---|---|---|
| Paged (`PagedCacheType::TurboQuant`) | **On** — unchanged, `#[default]` since `7dc99d394` | **128 only.** Kernels still `return` early otherwise; ambient default auto-falls back to `Auto` with a warning, an explicit `--pa-cache-type turboquant` hard-errors (`daf525f00`, 7 tests) | the April 2026 B200 run, quality unrecorded |
| Eager (`NormalCache`) | **Opt-in**, `ARC_TURBOQUANT_KV=1` | every width `≥ 8`, K and V independent | 27 + 6 + 3 tests, round-trip bounded at 8 head dims |

**The eager path is deliberately not defaulted on.**
`TurboQuantSingleCache::current_data` reconstructs *every compressed token on the
host* and ships it back to the device once per layer per decode step — `O(T·d)`
host work plus a full H2D transfer per step. `V4CachedK::span`
(`deepseek4.rs:2306`) already documents the fix shape for exactly this problem
(*"reconstructing the whole context every decode step would write T × head_dim
activations per layer, which at large batch costs more bandwidth than the storage
saves"*), and it needs a device-side dequant kernel. Defaulting a KV change on
before measuring it is precisely what happened with FP8 KV. The env gate mirrors
`ARC_V4_FP8_KV` (`deepseek4.rs:2405`).

`NormalCache::new_plain` added and used by V4's two cache constructors: V4's V
half is a 1-wide marker (`V4_V_MARKER_WIDTH`) and `require_normal_kv_slot`
rejects anything else. A 1-wide vector is *also* refused by name at the layout
level, so both belts hold.

---

## 5. TurboQuant vs FP8 K — **alternatives, not composable**

Both compress the same tensor. For **V4 specifically, FP8 K wins today**:

- V4 is FP8-QAT — the reference round-trips the non-rope K dims through block-wise
  E4M3 on every forward — so storing the 8-bit code is **bit-exact** with what the
  model computes (`dsv4_kv_fp8.rs`, pinned by
  `kv_fp8_roundtrip_is_bit_exact_vs_reference`). TurboQuant would be genuinely
  lossy on the same tensor.
- Proven TOKEN_IDENTICAL 5/5 on hardware (PR #72). TurboQuant has no such record.
- V4's V half is a 1-wide marker; there is nothing for `D·H·D` to rotate.
- No kernel exists at head_dim 512.

On paper TurboQuant K4 is denser (≈354 B/token/layer against FP8's measured 590,
vs 1026 dense) but that is format arithmetic on an unbuilt kernel. **Jish is
getting one thing for V4, not two.**

---

## 6. Tests

`mistralrs-quant::turboquant` 12 → **27**; `arc-turbo` **0 → 6**; `kv_cache` gate
**+3**.

Round-trip: `roundtrip_error_is_bounded_for_every_head_dim` (8 dims × 3 bit
widths × 24 probes, asserted `≤ 1.5·sqrt(D_b)` **and `> 1e-4`** so a no-op codec
fails). Generation-equivalence: `prefill_then_decode_roundtrips_at_every_head_dim`
runs prefill + 6 decode steps per head dim and asserts the FP16 window comes back
bit-exact while the compressed tail does not.

Mutation proofs:
- **break the WHT butterfly** → `wht_is_involutory_at_every_block_order` at orders
  8…512, which *also* asserts the transform changed something, so a stubbed-out
  FWHT fails rather than passing the round trip.
- **break the codebook lookup** → `codebook_lookup_is_an_identity_on_centroids`
  (catches the `Ok(i) => i+1` off-by-one class) at every generated dim, not just
  the three with hand-checked tables.
- **unsupported head dim** → `each_refusal_names_its_own_mechanism`; six distinct
  refusals each asserted to contain their own reason string.
- **generator drift** → `generator_reproduces_shipped_tables` pins `mse_per_coord`
  to the nine shipped constants at 1e-9.

**D12.** Every cache fixture asserts `compressed_seq_len() > 0`. With the shipped
`fp16_window = 128` a short test compresses **nothing** and exercises a plain
FP16 buffer — exactly the vacuous shape that let V4 ship unable to serve a
prompt. The fixtures use a small window and outlier-bearing, asymmetric
activations (a constant or symmetric probe cannot disagree with a broken sign
vector).

---

## 7. Needs a GPU

Two things this branch cannot claim without hardware:

1. **The eager path's decode cost.** Predicted large; the mechanism is stated
   above. If it is tolerable, the default flips.
2. **Generalising the CUDA kernels from `HEAD_SIZE == 128` to
   `{64, 128, 256, 512}`.** This is the change that widens the *paged* default,
   which is the path with a fused kernel and therefore the only one where
   compression is a straight win. Deliberately not in this commit: it cannot be
   compiled, let alone verified, without nvcc.

### Box

1 × H100 or H200, 200 GB disk, self-destruct armed at creation, one on-box
poller, deleted when done.

### Script

```bash
set -euo pipefail
cd /root && git clone -b feat/turboquant-every-model https://github.com/aeonmindai/arc.git && cd arc
cargo build --release --features "cuda flash-attn" -p mistralrs-cli 2>&1 | tail -20
BIN=./target/release/mistralrs
$BIN serve --help | sed -n '1,60p'   # pin the real flag surface before trusting any of the below

# Head dims under test. All ungated on HF.
#   Qwen/Qwen2.5-0.5B-Instruct   hidden  896 / 14 heads =  64
#   microsoft/phi-2              hidden 2560 / 32 heads =  80   <-- non-power-of-two
#   Qwen/Qwen2.5-1.5B-Instruct   hidden 1536 / 12 heads = 128
#   unsloth/gemma-2-2b-it        head_dim              = 256   <-- also mixes sliding-window layers
run_case () {  # $1 model  $2 label  $3 extra flags  $4 env
  ( export $4; $BIN serve -p 8765 --paged-attn off -m "$1" > /root/log_$2.txt 2>&1 & echo $! > /root/pid )
  for i in $(seq 1 240); do grep -q "Server listening" /root/log_$2.txt && break; sleep 5; done
  S=$(date +%s.%N)
  curl -s localhost:8765/v1/chat/completions -H 'content-type: application/json' -d '{
    "model":"default","messages":[{"role":"user","content":"List the first 20 prime numbers, comma separated."}],
    "temperature":0,"top_p":1,"seed":0,"max_tokens":128}' > /root/out_$2.json
  E=$(date +%s.%N)
  kill $(cat /root/pid); wait || true
  python3 -c "
import json,sys; d=json.load(open('/root/out_$2.json'))
print('$2', 'tokens=', d['usage']['completion_tokens'], 'wall=', round($E-$S,2))
print('$2 TEXT:', json.dumps(d['choices'][0]['message']['content']))"
  grep -iE "turboquant|TurboQuant KV cache" /root/log_$2.txt | head -5
}

for M in "Qwen/Qwen2.5-0.5B-Instruct:d64" "microsoft/phi-2:d80" \
         "Qwen/Qwen2.5-1.5B-Instruct:d128" "unsloth/gemma-2-2b-it:d256"; do
  MODEL=${M%%:*}; TAG=${M##*:}
  run_case "$MODEL" "${TAG}_off" "" "ARC_TURBOQUANT_KV=0"
  run_case "$MODEL" "${TAG}_on"  "" "ARC_TURBOQUANT_KV=1"
done

# Paged path: does the ambient TurboQuant default engage, and where does it fall back?
for M in "Qwen/Qwen2.5-1.5B-Instruct:d128" "Qwen/Qwen2.5-0.5B-Instruct:d64"; do
  MODEL=${M%%:*}; TAG=${M##*:}
  $BIN serve -p 8765 --paged-attn on -m "$MODEL" > /root/paged_$TAG.txt 2>&1 &
  sleep 120; kill %1 || true
  grep -iE "turboquant|falling back|cache type" /root/paged_$TAG.txt | head -10
done
```

### Numbers to extract

| # | Claim under test | Extraction |
|---|---|---|
| a | default-on engages | `TurboQuant KV cache ON for the eager path: … k_head_dim=N` in `log_*_on.txt`; and the paged logs must say TurboQuant for d128 and *fall back* for d64 |
| b | head_dim ∉ {64,128,256} works | `d80_on` (Phi-2, head_dim 80, block plan `64+16`) must produce coherent text at all |
| c | quality | `d*_on TEXT` vs `d*_off TEXT` at temperature 0 — token-identical is the pass; any divergence gets reported verbatim, not summarised |
| d | speed / memory | `wall` and `completion_tokens` per case → tok/s ratio on/off; `nvidia-smi --query-gpu=memory.used` sampled during each run |

If (d) shows the eager on/off tok/s ratio is materially below 1, that is the
measured justification for the CUDA-kernel follow-up and the default stays off.
If it is at or near 1, the default flips in this PR before it leaves draft.
