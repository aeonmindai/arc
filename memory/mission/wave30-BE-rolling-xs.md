# Wave 30 — BE: the `xs` compressor history is a recompute buffer, not state. Rolled.

**Scope.** V4's compressor-input (`xs`) history cache and its per-step
recomputation: `models/deepseek4.rs` (`V4Compressor`, `Attention::forward`,
the extra `NormalCache` slots) and the KV-cache plumbing that carries it.
No GPU rented — CPU only. Base: `origin/master` @ `57bd1ba70`.
Starting point: `wave29-BC-v4-paged-attention.md` §8 ("Noticed: the `xs`
compressor history is recomputed from a raw `[B, T, hidden]` buffer every
decode step … a rolling compressed state would cut the dominant per-token
memory term 4–128×"). **BC's claim is confirmed and it was an
under-estimate.** Every number below is CONFIRMED (code read, `file:line`, or
a test that runs).

---

## 0. The one-paragraph answer

The compressor consumes the raw history through **exactly one call site**
(`deepseek4.rs:1414` → `compressed_kv` → `forward_from_xs`), and that function
is a **strided local reduction**: compressed row `j` is a softmax-weighted
pool over the `ratio` tokens of group `j` — plus group `j-1` on the ratio-4
(overlap) compressor — then an RMSNorm. Row `j` is therefore final the instant
token `(j+1)·ratio − 1` arrives and **can never change again**, and no future
row can ever look further back than `span_groups · ratio` tokens. So the raw
`[B, T, hidden]` buffer is not state; it is a recompute buffer for a value we
can just keep. Keeping the result instead — `[B, T/ratio, head_dim]` plus a
bounded raw tail — is **`8 × ratio` smaller per token** (BC said `ratio`: the
row count drops `ratio×` *and* the width drops `4096 → 512`, another 8×), i.e.
**32× on the 21 CSA layers and 1024× on the 20 HCA layers**. Landed, behind a
cache type that owns the token↔row mapping so the engine's token-unit
truncations stay correct and the unrecoverable ones are refused loudly.
**At 2048-token context the per-sequence footprint goes 424,018 → 108,288
B/token, and the H200 batch ceiling goes B≈68 → B≈266. B=256 fits, with
~2.2 GB spare.**

---

## 1. Does every `xs` consumer accept the strided reduction? — CONFIRMED, one consumer

`grep -n "xs_for_compressor\|compressed_kv\|xs_hist" deepseek4.rs` — the raw
history is produced in exactly one place and consumed in exactly one:

| site | what it does |
|---|---|
| `deepseek4.rs:1292-1303` (pre-change) | **producer** — appends this step's `xs` into the per-sequence slot, returns the whole `[B, T, hidden]` history |
| `deepseek4.rs:1414` | **the only consumer** — `self.compressed_kv(&xs_for_compressor)` |
| `deepseek4.rs:1211-1245` | `compressed_kv` → truncates to the largest `ratio`-multiple prefix and calls `forward_from_xs`, then applies compress-θ RoPE at positions `j·ratio` |
| `deepseek4.rs:703-756` | `forward_from_xs` — the reduction itself |

Nothing else reads it. Specifically checked and cleared:

- **`dsv4_attention`** (`deepseek4.rs:1432,1451,1484,1503`) takes only the
  *compressed* tensor plus the KV; it never sees the raw history.
- **The Lightning Indexer** — the plausible second consumer, and the answer
  bounds the risk twice over. `V4Indexer::forward(&q_a, &k_full, &xs)` wants
  `xs` as `[B, T_q, hidden]` — **the current step's queries, not the history**
  (`dsv4_indexer.rs:320-325`). And it is **not invoked at all**:
  `grep -n "self\.indexer" deepseek4.rs` returns nothing; the field is loaded
  at `:1095-1105` and never used. (That is BACKLOG "wired but dead" — this
  wave does not clear it, but records that the indexer's `xs` is per-step
  and would remain correct.)
- **The MTP block** passes `None` for the slot (ratio-0 layer).
- **The RoPE** applied to the rows depends only on the row index
  (`positions = j·ratio`), so rows can be cached **pre-RoPE** — which is what
  the new cache stores.

**The locality claim is not asserted, it is executed.**
`models::deepseek4::tests::rolling_xs_matches_full_recompute_{csa_ratio4,
hca_ratio128}` feed a real `V4Compressor` one token at a time through the
rolling state and require the result to equal the whole-history recompute at
**every** step, for both compressor shapes (`coff == 2` overlap and
`coff == 1`).

---

## 2. The real reduction factor — arithmetic

**Config** (verified against the verbatim `config.json` fixture,
`deepseek4.rs:4090-4160`): `hidden_size 4096`, `head_dim 512`,
`num_hidden_layers 43`, bf16. `compress_ratios`: layers 0/1 standard, **even
2..=42 → ratio 4 (CSA, 21 layers)**, **odd 3..=41 → ratio 128 (HCA, 20
layers)**, slot 43 = the MTP block. 41 compressed layers, matching FACTS.

### Per token, per layer

| | raw `xs` (today) | rolling rows | factor |
|---|---|---|---|
| CSA, ratio 4 | `4096 × 2 = 8192 B` | `512 × 2 / 4 = 256 B` | **32×** |
| HCA, ratio 128 | `8192 B` | `512 × 2 / 128 = 8 B` | **1024×** |

`8 × ratio`, because the row count falls `ratio×` **and** the row is
`hidden → head_dim` narrower (8×). BC's 4–128× counted only the first.

### Per token, per sequence (the growing term)

- today: `41 × 8192` = **335,872 B/token** (FACTS says 335,954; the 82 B
  delta is a derivation rounding, immaterial)
- rolling: `21 × 256 + 20 × 8` = **5,536 B/token** → **60.7×** on the term
  that scales with context

### The constant term (what the rolling state adds back)

Bounded raw tail, `≤ span_groups · ratio + margin` rows of `hidden` (margin =
16 tokens, see §4), plus the compressed-row buffer's block allocation
(`SingleCache` grows in 512-row blocks from a 64-row initial):

| | tail rows | tail bytes/layer | × layers | comp buffer @2048 ctx |
|---|---|---|---|---|
| CSA ×21 | 24 | 196,608 | 4,128,768 | 576 rows → 589,824 ×21 = 12,386,304 |
| HCA ×20 | 144 | 1,179,648 | 23,592,960 | 64 rows → 65,536 ×20 = 1,310,720 |
| | | | **27,721,728** | **13,697,024** |

### Totals at 2048-token context, per sequence

```
today   xs = 41 × 2048 × 8192                      = 687,865,856 B  (687.9 MB)
rolling xs = 27,721,728 + 13,697,024               =  41,418,752 B   (41.4 MB)   16.6x
        ⇒ per-token equivalent 41,418,752 / 2048   =      20,224 B/token
KV (unchanged)                                     =      88,064 B/token
        today   88,064 + 335,872                   =     423,936 B/token
        rolling 88,064 +  20,224                   =     108,288 B/token   3.91x
```

### New maximum batch at 2048 ctx (H200, ~59 GB usable after the 74.18 GB artifact + ~8 GB reserve)

```
today:   59e9 / (423,936 × 2048) = 59e9 / 868,181,  ≈  67.9  ⇒ B = 68   (matches FACTS)
rolling: 59e9 / (108,288 × 2048) = 59e9 / 221,773,824 ≈ 266.0 ⇒ B = 266
B=256 costs 256 × 221,773,824 = 56.8 GB — fits, ~2.2 GB spare.
```

**That is the whole point of the change**: `E(B) = 256·(1−(1−8/256)^B)` puts
the 8× expert-amortisation at B=256, and B=256 was 3.9× out of reach on
memory. It now fits at 2048 context. (At 512-token context the ceiling is
B≈682, so context, not batch, becomes the binding term again.)

**Not measured, and not claimed:** aggregate tok/s at B=256. Memory is a
necessary condition, not a sufficient one — the B-sweep in FACTS *fell* with
batch for reasons (decode serialisation) that this change does not touch.
This wave moves the ceiling; it does not by itself move the throughput.

### Decode compute, as a side effect

Today every decode step recompresses the **entire** history on every
compressed layer: `T/ratio` rows of GEMM + softmax, growing without bound.
The rolling state computes **at most one row** per step (and none at all on
`ratio−1` of every `ratio` steps). At 2048 context that is ~512× less
compressor work per step on a CSA layer. Unmeasured on GPU — CPU-only wave.

---

## 3. What landed

1. **`mistralrs-core/src/kv_cache/xs_rolling.rs` (new)** — `XsRollingCache`:
   completed compressed rows (`SingleCache`, one row per group) + the bounded
   raw tail (`Option<Tensor>`, tokens `[base, tokens)`), and `advance()`,
   which appends new tokens, compresses whatever groups completed, and drops
   the raw rows no future row can consume. `compress` is passed in as a
   closure, so the cache module stays model-agnostic.
2. **`kv_cache/mod.rs`** — `KvCache::XsRolling(Box<XsRollingCache>)` plus its
   arms in `k`/`v`/`append`(refuses)/`current_seq_len`/`reset`/`set_len`/
   `try_set_len`, `NormalCacheManager::{clone_in_cache, clone_out_cache,
   set_none_cache}`. The compressed rows batch where `k` does and the tail
   where `v` does, so the existing dim-0 batching machinery is reused
   verbatim; only the metadata rebuild needed a new arm.
3. **`prefix_cacher.rs`** — two device-probe arms.
4. **`deepseek4.rs`** — the slots are built as `XsRolling` with the layer's
   `ratio` and `span_groups` (`:3337-3365`); `Attention::forward` advances the
   state instead of appending raw history (`:1292+`); `compressed_kv` split
   into `compress_prefix` (whole-history, still the prefill path and the
   test reference) + `compressed_kv_from_rows` (RoPE).
5. **`lib.rs`** — `KvCache`/`SingleCache`/`XsRollingCache` re-exported, for
   the integration test that drives the per-sequence cache dance by hand
   (same rationale as the existing `TextFlashParams` export).

Deliberately **not** done: `mistralrs-quant/qtip/**`, `scheduler/**`, the
paged-attention capability flag and `normal_loaders.rs:3231` are untouched
(agent BC's PR #57 owns those).

---

## 4. The contract that makes it safe — and what it costs

Every cache entry is truncated in **token** units by three callers:
`prefix_cacher.rs:316-321`, `mtp_pipeline.rs:1115-1127` (MTP verify rollback)
and `speculative.rs:810,892`. A cache whose rows are not tokens has to map
that itself, or corrupt silently. `XsRollingCache` therefore reports
`current_seq_len()` in **tokens** and:

- `set_len(n)` → `comp` to `n / ratio` rows, tail to `n − base` rows;
- `try_set_len(n)` **refuses** any `n` whose resume point
  (`(n/ratio + 1 − span_groups)·ratio`) is behind the retained tail. The
  prefix cacher already treats a refusal as "no match, prefill from scratch";
  MTP/speculative cannot decline, which is what the margin is for.

**The subtlety that a naive implementation gets wrong** (and that this one got
wrong first — caught by a test, see §5 mutation 3): "retain the last
`span·ratio + margin` tokens" is NOT sufficient. A rollback that crosses a
group boundary invalidates that group's compressed row, and rebuilding it
needs the group's raw tokens **from its start**, not 16 of them. The retention
point must be the one the *worst allowed* rollback would need — computed at
`tokens − margin`, not at `tokens`. It costs nothing: the tail only reaches
back further during the first `margin` tokens of a group, when the current
group is that much shorter, so the bound is unchanged.

### Honest costs

- **Prefix-cache partial matches are declined for V4.** An exact prefix
  extension (`match_len == cached length` — the multi-turn case, the valuable
  one) is still accepted. A *partial* match (shared prefix that then diverges)
  now returns "no match" and re-prefills. Correct, slower, and exactly how
  every rotating/sliding-window cache in this engine already behaves.
- **Rollbacks are bounded at `XS_TAIL_MARGIN_TOKENS = 16`.** V4's MTP head is
  a single NextN block, so a verify rollback is 1–4 positions. A speculative
  configuration with `gamma > 16` would hit a **loud** error, not corruption.
  Raising the margin is a one-constant change at 8 KB/layer/seq/token.
- **One small allocation per compressed layer per step** (the tail is rebuilt
  by `cat` + `narrow` rather than written in place). Against a full-history
  GEMM per layer per step, this is strictly cheaper — but it is churn a
  fixed-capacity ring would avoid. Noted, not done.

---

## 5. The tests, and the proof each can fail (D12)

Seven tests. Three mutations, each applied, run, observed to fail, reverted.

**Unit (`models::deepseek4::tests`, real `V4Compressor`, patterned
non-degenerate weights, asserted non-degenerate):**
- `rolling_xs_matches_full_recompute_csa_ratio4` — 10 prefill + 14 single-token
  decode steps, crossing five ratio-4 boundaries; every step's rows must equal
  `forward_from_xs` over the whole history.
- `rolling_xs_matches_full_recompute_hca_ratio128` — same, two 128-strides.
- `rolling_xs_tail_is_bounded_by_the_compressor_span` — the memory claim
  itself: 200 decode steps, tail never exceeds `span·ratio + margin`.
- `rolling_xs_accepts_any_rollback_within_the_margin` — a `margin`-token
  rollback is accepted **and resumable to the whole-history answer** at every
  length, both ratios, including boundary-crossing ones.
- `rolling_xs_set_len_truncates_both_time_bases` — truncation lands on both
  time bases; a rollback past the span is refused by both entry points and
  leaves the state unmutated.

**End-to-end (`tests/synthetic_load_smoke.rs`, the 3-layer V4 fixture with
ratios `[0, 4, 128]`):**
- `v4_rolling_xs_decode_matches_whole_history_prefill` — 120-token prompt then
  20 single-token decode steps, crossing five ratio-4 strides **and** the
  ratio-128 stride during decode; ground truth is a **fresh full prefill of
  the same tokens** (the whole-history recompute, not "what the old code
  printed"). It asserts on the **compressor state** — the compressed rows in
  the model's own `XsRolling` slots — and on the logits second. See §5a: the
  first version of this test asserted on the sampled token, and that was
  wrong.
- `v4_xs_history_two_seq_batch_matches_single_sequence` (pre-existing, R3's
  voting contract) — updated to snapshot/merge/split whole cache **entries**
  through helpers that mirror `clone_in_cache`/`clone_out_cache` field for
  field, instead of the old `reset()+append(k,v)` stand-in. Still asserts
  batched == solo across merged / shrink / lockstep, still bit-exact on the
  batch-mate invariance.

### 5a. The first version of this test was arch-dependent and toothless — CI caught it, and the numbers say why

Worth recording, because it is exactly the D12 failure mode and it nearly
shipped.

**v1 asserted `argmax(rolling) == argmax(reference)` at every step.** Green on
this dev host (ARM: max logit diff **exactly 0.0** at all 21 steps), red on all
three CI platforms (x86):

```
rolling decode sampled a different token than the whole-history recompute at
122 tokens … left: 111, right: 85
```

Probing the fixture rather than loosening the bound:

```
PROBE len=120 top1=3.302734 top2=3.296875 gap=0.005859 tol=0.033027 maxdiff=0.000000
PROBE len=121 top1=3.150391 top2=3.150391 gap=0.000000 tol=0.031504 maxdiff=0.000000
PROBE len=122 top1=3.003906 top2=3.001953 gap=0.001953 tol=0.030039 maxdiff=0.000000
…all 21 steps: gap ∈ [0.0000, 0.0098], tol ∈ [0.027, 0.067]
```

The patterned `lm_head` has near-duplicate rows, so **the top-2 gap is below
the CPU-MatMul F16 noise floor at every step**: the sampled token is decided by
GEMM tiling, not by the model. No implementation — including the unmodified one
— has an arch-independent argmax on this fixture. The assertion was testing the
host.

**v2 asserted logit equality within the documented F16 budget.** Also wrong,
and worse — it could not fail. Measured under mutation 1 (a fully corrupted
compressor):

```
             logit max-diff   tolerance
window=128:  0.008 – 0.043    0.030 – 0.056     (passes!)
window=8:    0.018 – 0.027    0.026 – 0.044     (passes!)
```

A 3-layer fixture with patterned weights is close to a constant function: even
a one-token history change moves the logits by only 0.006–0.05, i.e. ~1x the
tolerance. **A logit-level equality test on this fixture cannot have teeth.**

**v3 asserts on the compressor state itself** — the compressed rows in the
model's `XsRolling` slots after 20 decode steps, against the rows a single
whole-history prefill of the same tokens produces — with the logits kept as a
secondary plumbing check. The signal is undiluted:

```
mutation 1 vs v3:  diverged by 2.1469789 (magnitude 2.614, budget 0.026)  = 82x margin
```

The fixture also gained `config_json_with_window` so this test can run at
`sliding_window = 8`: at the real 128 the window covers the whole 140-token
fixture and the compressed branch barely reaches the output, which is not a
sensitivity a compressor test should be run at.

### Mutations

> **Mutation 1** — `need_start = g_done * ratio` (hand the compressor only the
> current group, dropping the `coff == 2` predecessor):
> ```
> rolling_xs_matches_full_recompute_csa_ratio4 ... FAILED
>   rolling compressed rows diverged from the whole-history recompute at 12
>   tokens (ratio 4): max abs diff 0.74833465
> v4_rolling_xs_decode_matches_whole_history_prefill ... FAILED
>   compressed layer 0: 20 decode steps of rolling state diverged from the
>   whole-history recompute by 2.1469789 (magnitude 2.6140704, budget 0.0261)
> ```
> `hca_ratio128` correctly still passes — `span_groups == 1` there.

> **Mutation 2** — retain no tail (`keep_from = g_target * ratio`):
> ```
> 3 of 5 unit tests FAILED; end-to-end FAILED
>   xs rolling cache: compressor history gap — row 30 needs tokens from 116
>   but the retained window starts at 120.
> ```
> The guard fires *loudly*; nothing is silently skipped.

> **Mutation 3** — the naive retention rule (§4), i.e. compute the retention
> point at `tokens` instead of `tokens − margin`:
> ```
> rolling_xs_accepts_any_rollback_within_the_margin ... FAILED
>   ratio 128: rolling back 16 tokens from 128 (the MTP rollback bound) must
>   be accepted, got: … rollback to 112 tokens is behind the retained raw
>   window (resuming needs the compressor input from token 0, retained from
>   112 …)
> ```
> This is the one that found a real bug rather than confirming a known one:
> without it, ~3% of MTP verify steps would have hard-errored in production.

All reverted; `cargo test -p mistralrs-core` = 261 lib + 12 smoke + 2 e2e
green (and green on all three CI platforms, which v1 was not). `cargo check --workspace --tests` green. Scoped clippy lane green.
Formatting: `kv_cache/mod.rs`, `prefix_cacher.rs`, `synthetic_load_smoke.rs`
returned to their **zero-hunk** baseline; `deepseek4.rs` left at its
pre-existing 27 hunks (fork policy — no mass reformat).

---

## 6. What needs a GPU

Nothing landed here needs one. Two claims do, and neither is made in this
branch:

1. **That B=256 actually runs.** The arithmetic says it fits in 59 GB at 2048
   context. Confirming it needs one H200 and the B-sweep from
   `wave25-AV-measurement-runbook.md`, re-run at B ∈ {64, 128, 256}.
   ⚠️ Memory is necessary, not sufficient: the existing sweep shows aggregate
   throughput *falling* with batch (15.35 → 5.07 tok/s, B=1 → 32) for reasons
   this change does not address. Expect the ceiling to move and the curve to
   need separate work.
2. **The decode-speed side effect** (~512× less compressor GEMM per step at
   2048 ctx on CSA layers). Measurable with `ARC_TIME_DECODE=1` on the same
   rental — it should show up as the `mla` bucket shrinking.

The one thing a GPU run *must* watch: the per-step tail `cat` allocation
(§4). If it shows in a profile, the fix is a fixed-capacity ring.

---

## 7. STRUCK — `wave28-BA-scheduler-batch.md` §1e is NOT IMPLEMENTABLE

BA §1e proposes a four-step plan to port V4's attention onto
`mla_decode_forward`. **Do not budget a session against it.** BC's wave29 §1b
and §2 disprove it from the tree, no GPU needed:

- `flashinfer_mla_decode.cu:12-13` fixes `HEAD_DIM_CKV = 512` as a **template
  argument**; a 448 instantiation does not compile —
  `vec_size_ckv = max(8, 448/32) = 14` trips
  `static_assert(vec_size % 8 == 0)` at `flashinfer/vec_dtypes.cuh:1566`
  (bf16) / `:1362` (half). The next legal width below 512 is 256.
- `flashinfer_mla_decode.cu:33` compiles the variant
  `DefaultAttention<false, false, false, false>` — dense causal, **no sliding
  window, no custom mask, no sinks, one key set**
  (`flashinfer/attention/variants.cuh:31-32`). **Every** V4 layer needs
  sliding-window + attention-sink, and CSA/HCA fold a second compressed key
  set into the same softmax (`dsv4_attention.rs:9-46, 281-322`). No V4 layer
  computes the function that kernel computes.

The replacement item, if anyone wants paged V4 decode, is *"a V4 decode kernel
with sliding window + sinks + a second key set"* — a kernel project sized like
the trellis grouped-GEMM, not a wiring project.

The BA doc lives on PR #54's branch (`fix/scheduler-runs-full-batch`), not on
master, so this wave records the strike here and comments it on that PR rather
than editing another agent's open branch.

---

## 8. Surfaced, not shipped

> **Noticed:** the V4 Lightning Indexer is loaded on every CSA layer
> (`deepseek4.rs:1095-1105`, `dsv4_indexer.rs`, its own tests, and PR #52
> un-deaded its CUDA kernel) but `self.indexer` is **never referenced in the
> forward path** — `grep -n "self\.indexer" deepseek4.rs` is empty. CSA is
> supposed to be top-k token selection *via the indexer*; today it is the
> compressed branch alone. That is either dead weight in the artifact or a
> missing piece of CSA. Worth a separate change?

> **Noticed:** the compressed rows are now a small, self-contained,
> quantisation-friendly tensor (`[T/ratio, 512]`, feeding a compressor
> branch rather than attention logits — BACKLOG already guessed it is more
> error-tolerant than KV). At 2048 ctx they are 13.7 MB/seq of the 41.4;
> 8-bit would take the ceiling from B≈266 to ≈300. Quality unmeasured.
> Worth a separate change?

> **Noticed:** the HCA raw tail (144 rows × 4096 × 2 B × 20 layers = 23.6 MB)
> is now **57% of the remaining `xs` footprint**, and it exists only to hold a
> partial ratio-128 group. Storing the tail as the compressor's *fused
> projection* (`2·coff·head_dim` wide instead of `hidden`) would cut it 4×,
> taking the ceiling to ≈297. Costs an extra dtype decision inside
> `forward_from_xs`. Worth a separate change?
