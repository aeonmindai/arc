# wave49-BZ — FP8 K storage: opt-in, root-caused, and the bug behind the bug

**No GPU.** Nothing here is measured on hardware. Everything below is either
read off source, or driven through the real code on CPU and shown failing before
the fix and passing after it. The one claim that still needs silicon is named in
§6 and is **not** claimed.

| | |
|---|---|
| branch | `fix/fp8-kv-opt-in` (off `master` @ `309fbcbb8`) |
| scope | FP8 K cache path + the `SingleCache` contract it broke |
| verdict | **FP8 K is now OFF unless `ARC_V4_FP8_KV=1`**, and the crash is fixed for **both** layouts |

---

## 1. What shipped broken, and why it got through

PR #72 (wave43-BU) landed FP8 K storage **defaulted ON**:

```rust
// deepseek4.rs:2386, as merged
let off = std::env::var("ARC_V4_FP8_KV").map(|v| v == "0").unwrap_or(false);
!off && std::env::var_os("ARC_V4_CAPTURE_PROBE").is_none()
```

`off` is false unless someone explicitly opts out, so unset meant **on**. The
same document that shipped it wrote *"Needs a GPU: first CUDA exercise of a U8
`SingleCache`, `index_select` over the LUT, and the U8 H2D leg."* That run never
happened before the merge.

On hardware (wave48-BY) every forward died, including the engine's own dummy
run, 23 ms in:

```
INFO  mistralrs_core: Beginning dummy run.
ERROR mistralrs_core::engine: prompt step - Model failed with error:
      dtype mismatch in slice-set, lhs: BF16, rhs: U8
```

3/3 sanity prompts returned `finish_reason: error`, 0 completion tokens.

**Why CI was green anyway.** Every FP8 test built its cache with
`KvCache::new_normal(...)`, whose `all_data` is `None`. `SingleCache::append`
then allocates the buffer *from the first `src`*, so it can never disagree with
it. The live engine does the exact opposite — see §3. The feature was
CUDA-gated in the author's mind and CPU-only in the test suite, and the two
never met.

---

## 2. 🔴 Step 1 — the gate is now opt-in

`deepseek4.rs:2405-2422`:

```rust
fn v4_fp8_kv_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        fp8_kv_enabled_from(
            std::env::var("ARC_V4_FP8_KV").ok().as_deref(),
            std::env::var_os("ARC_V4_CAPTURE_PROBE").is_some(),
        )
    })
}

fn fp8_kv_enabled_from(var: Option<&str>, capture_probe: bool) -> bool {
    var == Some("1") && !capture_probe
}
```

Unset → **off**. `0`, ``, `true`, `on`, `2` → **off**. Only the literal `1`
turns it on, and `ARC_V4_CAPTURE_PROBE` still vetoes it.

Two structural changes go with it, both so the thing can be tested at all:

* **The decision is a pure function.** `v4_fp8_kv_enabled` is a `OnceLock`; it
  resolves once per process, so a test that sets the variable proves nothing
  about the default. `fp8_kv_enabled_from` is what `v4_fp8_kv_is_opt_in` pins,
  across the whole input table.
* **The gate moved to the caller.** `append_kv_mqa` used to apply
  `.filter(|_| v4_fp8_kv_enabled())` itself, which meant that after the flip its
  packed arm would have been unreachable from any test in a normal environment —
  and `v4_kv_bytes_per_token_per_layer` would have silently started measuring
  the *dense* layout and still passed. The filter now sits at the single call
  site (`deepseek4.rs:1626`); `append_kv_mqa` takes an already-filtered
  `Option<&V4PackedK>` and both of its layouts stay reachable.

---

## 3. Step 2 — the root cause, with file:line

Not the FP8 layout. **The K half's `SingleCache` is allocated BF16 before the
first append, by the engine, for every model.**

1. `engine/add_request.rs:464-482` — each sequence gets a preallocated KV buffer
   `[1, num_kv_heads, cap, k_head_dim]` in `activation_dtype`. For V4 Flash that
   is **BF16, 512 wide, for both halves** (`ModelConfigMetadata` in
   `DeepSeekV4::new` sets `k_head_dim = v_head_dim = cfg.head_dim`,
   `deepseek4.rs:3756`).
2. `engine/mod.rs:467-473` — a prompt step with `token_offset() == 0` issues
   `CacheInstruction::Reset { load_preallocated_cache: true }`. (The dummy run
   is exactly this.)
3. `kv_cache/mod.rs:800-815` — `NormalCacheManager::set_none_cache` installs
   that tensor as `all_data` for **both** halves, before anything is appended.
4. `kv_cache/single_cache.rs:161` (pre-fix) — `ad.slice_set(src, dim, ...)` with
   `ad` BF16 512-wide and `src` the U8 448-wide codes.
5. `candle-core/src/tensor_cat.rs:254` — `slice_set` checks dtype first:
   **`dtype mismatch in slice-set, lhs: BF16, rhs: U8`.** Verbatim.

`V4::do_preallocated_cache()` is the trait default
(`pipeline/mod.rs:263`, `matches!(self.cache(), EitherCache::Normal(_))` → true)
and nothing overrides it, so V4 always takes this path.

### 🔑 The bug behind the bug: `ARC_V4_FP8_KV=0` would NOT have rescued the box

wave48-BY's proposed 15-minute triage was to re-serve with the flag off. That
would have moved the error, not removed it.

`candle`'s `slice_set` checks **dtype at `:254` and shape at `:278`** — dtype
first. With FP8 off, the K half (dense BF16 512-wide) fits the preallocated
buffer and passes. Then `KvCache::append` appends the V half, which since
wave33/PR #63 is the **1-wide zero marker** (`V4_V_MARKER_WIDTH`, because V4's
fused MQA makes V bit-identical to K). 1 into 512:

```
shape mismatch on dim 3, 512 <> 1
```

Reproduced on CPU (§5). So **V4 has been unable to complete a prompt step on the
preallocated path since PR #63 (2026-08-15), one day before PR #72 (2026-08-16)
added a second, earlier failure on top of it.** Both published V4 serving
numbers (14.58 tok/s, GSM8K 96.0%) predate both.

### The fix

`kv_cache/single_cache.rs:137-193`. `SingleCache::append` now asks whether the
buffer it holds can actually receive `src` — same dtype, same width on every dim
but the sequence dim, which is precisely what `slice_set` demands:

* **fits** → reuse it, unchanged. This is the overwhelmingly common case and the
  whole point of `load_preallocated_cache`.
* **does not fit, cache empty** → the buffer holds nothing, so discard it and
  allocate from `src` at the **same `capacity_seq_len`**. The pre-grow is kept;
  only dtype and width are corrected.
* **does not fit, cache non-empty** → the tokens already stored are in a
  different layout, so this is real corruption. Bail loudly rather than paper
  over it.

This is generic, not V4-specific: it fixes both V4 layouts and makes the
`SingleCache` contract honest for any future model whose slot halves are not
dense activation-precision `head_dim` tensors.

---

## 4. Step 3 — the tests, and the mutation proof

Five new tests. The two that matter build the cache **the way the engine builds
it**, which no existing test did.

| test | file | pins |
|---|---|---|
| `packed_append_into_the_engines_preallocated_slot_round_trips` | `deepseek4.rs` | FP8 codes into a preallocated BF16 slot: appends, and reads back **bit-identical** to the reference round trip |
| `dense_append_into_the_engines_preallocated_slot_round_trips` | `deepseek4.rs` | the default layout: dense K + 1-wide marker into the same slot |
| `v4_fp8_kv_is_opt_in` | `deepseek4.rs` | the full input table for the gate; `None` → off |
| `a_fitting_preallocation_is_reused` | `single_cache.rs` | a buffer that fits is **not** thrown away (sentinel past the write offset) |
| `a_mismatched_preallocation_is_rebuilt_while_empty` / `a_layout_change_mid_sequence_is_refused` | `single_cache.rs` | rebuild-while-empty, and the loud bail once tokens exist |

Round-trip equality is asserted, not just absence of error: a cache that
silently dropped the write, or reallocated and lost the token, would not error.

### Mutation proof (DOCTRINE D12)

**(a) Fix disabled** — `if false && !self.preallocation_fits(src)`:

```
test kv_cache::single_cache::tests::a_mismatched_preallocation_is_rebuilt_while_empty ... FAILED
test models::deepseek4::kv_footprint_tests::dense_append_into_the_engines_preallocated_slot_round_trips ... FAILED
test models::deepseek4::kv_footprint_tests::packed_append_into_the_engines_preallocated_slot_round_trips ... FAILED
test kv_cache::single_cache::tests::a_layout_change_mid_sequence_is_refused ... FAILED

---- ...packed_append_into_the_engines_preallocated_slot_round_trips stdout ----
Error: dtype mismatch in slice-set, lhs: BF16, rhs: U8      <-- the hardware error, verbatim
---- ...dense_append_into_the_engines_preallocated_slot_round_trips stdout ----
Error: shape mismatch on dim 3, 512 <> 1                    <-- the one hiding behind it
```

`a_fitting_preallocation_is_reused` **still passes** under this mutation — it is
the control, and it proves the guard does not simply reallocate everything.

**(b) Gate reverted to PR #72's form** — `var != Some("0")`:

```
test models::deepseek4::kv_footprint_tests::v4_fp8_kv_is_opt_in ... FAILED
panicked at deepseek4.rs:4894: unset must be OFF
```

Both mutations restored; suites re-run green.

### Standing block

* `cargo test -p mistralrs-core -p mistralrs-quant` — **310 + 13 + 241 + 2 passed, 0 failed**
* `cargo check --workspace` — clean (one pre-existing unused-import warning)
* scoped clippy lane (`arc-bench arc-engine arc-cuda-graph arc-cli mistralrs-quant --tests --examples -D warnings`) — **exit 0**
* fmt: both touched files were **already** non-rustfmt-clean on `master`, so
  `rustfmt` was not run on them (it would mass-reformat unrelated lines, and
  neither is a `mod.rs`/`lib.rs`). Instead the added hunks were hand-formatted
  and verified by diffing `rustfmt --check` output against the `master`
  baseline: **zero new deviations, one pre-existing deviation removed**
  (`deepseek4.rs` 578 → 567 lines of `--check` output).

---

## 5. What is fixed, and what is merely *no longer masked*

**Fixed and proven on CPU:** the dtype/width contract between the engine's
preallocated buffer and both V4 cache layouts, in the empty-cache case that the
prompt step always takes. Bit-exact readback through the real `KvCache`.

**Fixed by consequence:** the wave33 V-marker crash. Untouched by the gate flip;
only the `single_cache` change addresses it.

**Not touched, deliberately:** `SingleCache::append_graph` has the same latent
disagreement (it reuses `all_data` and hands it to `write_kv_inplace`, which
would bail `kv-write: src dtype ... != cache dtype ...` or on the `[B,H,1,D]`
shape check). It is CUDA-only and unreachable without a GPU, so no CPU test can
hold a fix there honest. `write_kv_inplace` already fails loudly with a precise
message. → BACKLOG.

**Surfaced, not shipped:** on the FP8-on path the engine still preallocates a
BF16 `head_dim`-wide buffer that is then immediately discarded — one wasted
allocation per prompt per layer. The real fix is a per-model cache-layout
descriptor so `add_request.rs` preallocates the right dtype and width, rather
than assuming every model stores dense activation-precision K and V. Worth a
separate change; not done here.

---

## 6. 🔴 What still needs a GPU, and what it costs

Two of wave43-BU's three "needs a GPU" items are **still unexercised**, and they
are exactly why this is opt-in rather than on:

1. `index_select` over the 256-entry E4M3 LUT with U32 indices on a
   `[B*1*window*448]` index tensor, 43× per step.
2. The U8 H2D leg of the code extraction (the CPU-exact quantizer now carries
   back U8 instead of F32).

The third — a U8 `SingleCache` — is now driven on CPU, but CPU `slice_set` and
`copy2d_u8` on CUDA are different code.

**Acceptance test:** greedy generation, same prompts, `ARC_V4_FP8_KV=1` vs
unset. The token streams must be **byte-identical** — the layout is bit-exact by
construction (`kv_fp8_roundtrip_is_bit_exact_vs_reference`), so any divergence
is a CUDA-path bug, not a quality question. Then the batch sweep: the claim to
falsify is 493 → ~696 concurrent sequences at 2048 ctx
(64,342 → 45,594 B/token).

**Cost: ~15 minutes inside any V4 rental that is already up** (≈$0.40 at
$1.49/hr). It does **not** justify a rental of its own, and it does not need a
re-bake, a quality re-measurement, or any PagedAttention work. **Not run here.**

Until it runs, the honest statement of the win is: *FP8 K storage takes V4 from
64,342 to 45,594 B/token and max batch from 493 to ~696, on CPU-proven
bit-exact arithmetic, behind a flag that has never executed on CUDA.*

---

## 7. Docs corrected

* `memory/mission/wave43-BU-fp8-kv-bytes.md` — the "Gates" section (was
  "`ARC_V4_FP8_KV=0` turns packing off") and the GPU-check paragraph (was
  "`ARC_V4_FP8_KV=1` (default)"), each with a correction banner naming this
  wave, plus an addendum recording which of its three GPU items is now covered.
* `CEILINGS.json` → `with_FP8_K_storage` — new `🔴_status_2026-08-16_wave49-BZ`
  recording that the 45,594 B/token figure is *behind a flag*, not what `master`
  runs; `needs_gpu` rewritten to say what is still unrun and what it costs;
  "ZERO changes to cache infrastructure" flagged as wrong.
* `deepseek4.rs` — the doc comment on `v4_fp8_kv_enabled` (was "on by default")
  and the `append_graph_kv_mqa` bail (was "Set `ARC_V4_FP8_KV=0`", now "Unset
  `ARC_V4_FP8_KV`").
