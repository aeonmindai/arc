# Wave 28 — BA: why half the admitted batch never runs

**Scope.** `mistralrs-core/src/scheduler/**` and the V4 paged-attention
capability flag. Source investigation + one contained scheduler fix. No GPU
rented. Base: `origin/master` @ `57bd1ba70`. Every claim is CONFIRMED (code
read, `file:line` cited) or labelled SUSPECTED.

---

## 0. The one-paragraph answer

Two separate things were conflated. **(a)** V4 declaring
`supports_paged_attention=false` is *correct for the kernel it actually wires*
— `PagedAttention::new(cfg.head_dim=512)` targets the generic vLLM-derived
kernel whose `switch (head_size)` has no case for 512. But the reason is
incomplete: V3 is also MLA and *does* get paged attention, via a **separate
MLA-specific paged path** (`mistralrs-core/src/mla/forward.rs`) that V4 never
uses. So the honest verdict is **unimplemented, not impossible**. **(b)** The
scheduler waitlisting is a genuinely separate defect and it is *not* that the
buckets exist — bucketing is a correctness invariant of the dense KV cache. The
defect is that **the split never heals**: the bucket selector used on the mixed
running+waiting path is perfectly *fair*, and perfect fairness makes two
equal-sized buckets alternate forever at a constant length gap. That is the
measured `32 running, 32 waiting`. Fixed.

---

## 1. Why V4 declares `supports_paged_attention=false` — **unimplemented**

### 1a. The stated reason is true for the kernel V4 wires

`mistralrs-core/src/pipeline/loaders/normal_loaders.rs:3231-3238`:

```rust
fn supports_paged_attention(&self, _config: &str) -> Result<bool> {
    // V4 MLA uses head_dim=512, which exceeds the PagedAttention kernel's
    // supported head sizes (64/80/96/112/128/192/256). ... (RUN-161)
    Ok(false)
}
```

Both halves check out:

- `mistralrs-core/src/models/deepseek4.rs:133` — `serde_default_fn!(usize,
  default_head_dim, 512)`; `:198-199` makes it the config field. `:3220` wires
  `PagedAttention::new(cfg.head_dim, device, None)`.
- `mistralrs-paged-attn/src/cuda/pagedattention.cuh:714-742` (V1) and
  `:817-…` (V2) — `switch (head_size)` has cases **64, 80, 96, 112, 128, 192,
  256** and a `default:` error arm. 512 is not among them.

So enabling the flag as-is would crash on a paid box. **Do not flip it.**

### 1b. But it is not an incompatibility — V3 solves the same problem

`supports_paged_attention` defaults to `Ok(true)`
(`normal_loaders.rs:143-144`). **V4 is the only loader in the file that
overrides it to `false`.** DeepSeek V2/V3 — same MLA family, same oversized
latent — inherit `true`, because they route decode through a *different*
kernel entirely:

- `mistralrs-core/src/mla/forward.rs:180` — `concat_and_cache_mla(...)`
- `mistralrs-core/src/mla/forward.rs:204` — `flashinfer_mla_decode(...)`
- imported and dispatched by `models/deepseek2.rs:23,299-308`,
  `models/deepseek3.rs:23,300-309`, `models/glm4_moe_lite.rs:23,261-270`
- kernels: `mistralrs-paged-attn/src/cuda/concat_and_cache_mla_kernel.cu`,
  `mistralrs-paged-attn/src/cuda/flashinfer_mla_decode.cu`

`deepseek4.rs` contains **no** `mla_decode_forward` / `should_use_mla_decode`
import. V4 is the only member of the family that does not use the MLA paged
path.

### 1c. Why V4 is structurally a *good* fit for that path

`deepseek4.rs:14-20` (module docs): V4 K/V is a **single fused `wkv`
projection, `hidden=4096 → head_dim=512`**, with RoPE applied in place to the
last `qk_rope_head_dim=64` dims. That is one shared 512-dim latent per token
per layer — i.e. **MQA over an MLA-shaped latent**, which is exactly the tensor
`concat_and_cache_mla` stores (`kv_lora_rank` + `qk_rope_head_dim` parts) and
`flashinfer_mla_decode` reads. V3's split is 512 + 64 = 576; V4's is 448 + 64 =
512 (`deepseek4.rs:251` — `qk_nope_head_dim` derived as `head_dim -
qk_rope_head_dim`).

Also note: `deepseek4.rs` already carries a **complete PagedAttention
integration** — `:855`, `:891`, `:2365` fields, `:1386-1418` the RUN-167
CSA/HCA + `cache_write_and_gather` handling, and dispatch tests at `:4942`
("PagedAttention + MLA-cache compress dispatch tests"). All of it is dead code
today, because the loader flag means `AttentionImplementation::PagedAttention`
is never selected (`:3217-3221`).

### 1d. Provenance

`git log --reverse -S "V4 MLA uses head_dim=512"` →
**`ba026e9d1`, 2026-06-18, "perf(v4): ARC_TIME_DECODE per-component decode
profiler"** — the flag arrived as a drive-by inside an unrelated profiler
commit (that commit shows `normal_loaders.rs` as a new file, so it is a
squash/import boundary; the flag is not the product of a design review).

### 1e. Verdict and scope — NOT attempted here

**Unimplemented.** Enabling paged attention for V4 means porting V4's attention
onto `mla_decode_forward`, i.e.:

1. Reshape V4's fused `wkv` output into the `(ckv, k_pe)` pair
   `concat_and_cache_mla` expects (448 + 64 rather than 512 + 64).
2. Confirm `flashinfer_mla_decode` accepts `kv_lora_rank=448`, or pad to 512.
   **Needs a GPU** — the kernel's supported latent widths are a compile-time
   property not readable from the Rust side.
3. V4 has no `kv_b_proj`, so the `w_uk`/`w_uv_t` absorption at
   `mla/forward.rs:190-202` has no analogue and must be replaced by V4's
   grouped-LoRA `wo_a`/`wo_b` (`deepseek4.rs:26`).
4. Only then flip `normal_loaders.rs:3231`.

That is a real project. Per the brief it is **scoped and stopped**, not
half-enabled.

---

## 2. Why the second bucket is not co-scheduled

### 2a. The bucketing itself is a correctness invariant — CONFIRMED

Sequences may share a forward pass only when their cache lengths are **exactly**
equal:

- `mistralrs-core/src/kv_cache/mod.rs:307-470`
  (`NormalCacheManager::clone_in_cache`) builds **one** dense batched cache,
  taking `seqs[0]` as the template for shape *and* for `current_seq_len` /
  `capacity_seq_len` (`:418-424`), then `slice_set`s every other sequence's
  `all_data` into it at `offset = i * first_k.dims()[0]` (`:377-379`).
- `mistralrs-core/src/kv_cache/single_cache.rs:160-162` —
  `SingleCache::append` writes the step's new K/V at that **single shared**
  `current_seq_len` offset for the whole batch.
- The decode mask cannot rescue it: `layers_masker.rs:172-181`
  (`make_causal_mask_matrix`) returns `Ok(None)` when `tgt_len == 1`, and the
  mask it builds otherwise is `(tgt_len, past_kv_len)` — **no batch
  dimension at all**.

And it fails *silently*, not loudly: `NormalCache::CACHE_GROW_SIZE = 512`
(`kv_cache/mod.rs:241`), so two sequences at length 100 and 200 have identical
`all_data` shapes and `slice_set` succeeds. Only `current_seq_len` differs — so
the shorter sequence would write its token at the wrong slot and attend over a
window of zeros. **That is why the bucket key contains `seq.len()`.** It is not
a memory limit and not a scheduling preference. The B=64 run having ~8,448 of a
~118,000-token budget free is irrelevant to it.

(The engine does issue one forward for the whole running set —
`engine/mod.rs:413-425` — so running-set size *is* tokens per decode step.)

### 2b. The actual defect: the split never heals — CONFIRMED

`FixedBucketingManager` (`default_scheduler.rs`) runs exactly one bucket per
step. *Which* one decides whether the batch ever becomes whole again, because
running a bucket advances it by exactly one token:

- **`discrete = true`** (used only from `bucket_and_waitlist_seqs`, i.e. the
  `(waiting=0, running>0)` and `(waiting>0, running=0)` arms at
  `default_scheduler.rs:225` / `:240`) picks the **minimum length** bucket.
  This *converges*: the gap to the next bucket shrinks by one each step until
  the two keys coincide and the buckets merge. This is what the manager's own
  doc comment describes — "run the ones with the shortest lengths… Allow the
  min seqs to catch up."
- **`discrete = false`** (the general mixed path, `:279`) instead picks
  `argmax` of summed `compute_priority()` = `scheduling_urgency + log2(len)`
  (`sequence.rs:625-628`). This *does not* converge.

The moment any bucket is waitlisted, `waiting` is non-empty for every
subsequent step, so the mixed path is the steady state and the converging
branch is never reached again.

**The alternation, traced.** Two equal buckets of 32, lengths 100 and 102:

| step | bucket A | bucket B | priorities (A vs B) | runs |
|---|---|---|---|---|
| 1 | 100 (u=0) | 102 (u=0) | discrete arm → min | A → 101 |
| 2 | 101 (u=0) | 102 (u=1) | 213.1 vs 245.5 | B → 103 |
| 3 | 101 (u=1) | 103 (u=0) | 245.1 vs 213.9 | A → 102 |
| 4 | 102 (u=0) | 103 (u=1) | 213.5 vs 245.9 | B → 104 |
| 5 | 102 (u=1) | 104 (u=0) | 245.5 vs 214.2 | A → 103 |

`32·log2(len)` differences between adjacent lengths are ~0.5; a single step of
waiting adds `32·1 = 32`. So urgency dominates, the loser always wins next, and
each bucket advances one token every **two** steps. The gap oscillates
1, 2, 1, 2, … and **never reaches 0**. Half the admitted batch idles forever —
exactly the H200 steady state `32 running, 32 waiting` at B=64,
`8 running, 8 waiting` at B=16 (`wave26-AX-h200-measurement.md:83-88`), with
`--max-seqs 128` so the admission cap (`sequence_fits`, `:305`) is provably not
the binder.

The irony is exact: **the urgency mechanism is anti-starvation machinery, and
it is what perpetuates the split.** It makes the two buckets perfectly fair,
and perfect fairness is precisely the state in which neither ever catches up.

---

## 3. What was changed

One function, `mistralrs-core/src/scheduler/default_scheduler.rs`:
`select_running_bucket`, extracted from the inline selection at the old
`:141-154` and given a **coalescence override**.

Take the greedy highest-priority bucket as before — *except* when running the
shortest bucket would merge it into the next-shortest soon enough to pay for
itself. Coalescing idles `total - n_min` sequences for `gap` steps and then
adds `n_min` sequences to every later forward, so the override is taken iff

```text
(total - n_min) * gap  <=  n_min * COALESCE_PAYBACK_STEPS      // 256
```

- Measured case (32 + 32, gap 1): `32 <= 8192` → **taken**; the batch is whole
  on the next step and stays whole.
- Fresh 21-token arrival vs a 63-sequence cohort at length 500 (gap 479):
  `30177 <= 256` is false → **refused**; the cohort keeps running, byte-identical
  to today. Anti-starvation behaviour of the greedy rule is preserved.
- The merge target must match the **whole** bucket key, not just length — two
  buckets differing in `token_offset` or the image flag can never merge, so
  coalescing toward them is refused.
- `discrete = true` is untouched.

Deliberately **not** done: no correctness invariant was relaxed. Every
co-scheduled sequence still has an exactly equal cache length. This does not
make ragged batching possible; it makes the batch stop being ragged.

**Bound on the win.** This recovers the flat **2×** term that AY's
decomposition attributes to `#3` (`wave27-AY-decode-serialization.md:282`). It
does **not** touch the 5.6× MoE-cap term (`ARC_QTIP_ONDEVICE_MOE_MAX_TOKENS`,
default 8) or the 34× single-stream term. Aggregate throughput will still fall
with batch until the MoE cap is addressed — this removes one of the three
multipliers, and it is the one that also caps us below the B=128–256 regime
where expert-coverage amortisation lives.

---

## 4. The test, and the proof it can fail

`scheduler_runs_the_whole_admitted_batch` — 64 admitted sequences, `Fixed(128)`
so KV budget is ample, split 32/32 across two cache lengths two tokens apart.
Steps the real `DefaultScheduler::schedule()` loop, decodes one token into
every scheduled sequence each step as the engine would, and asserts all 64 land
in one running set with 0 waiting **and stay there**.

Plus three unit tests on `select_running_bucket`: override taken for adjacent
equal buckets, override refused for the expensive case, `discrete` unchanged.

**Mutation-proved (D12).** Reverting `select_running_bucket` to pre-fix
behaviour (`return greedy;` before the override) and re-running:

```
test scheduler::default_scheduler::tests::scheduler_runs_the_whole_admitted_batch ... FAILED
test scheduler::default_scheduler::tests::select_running_bucket_coalesces_adjacent_equal_buckets ... FAILED

scheduler never ran all 64 admitted sequences in one step:
running=32, waiting=32 — half the fleet batch is idling
```

The failure message reproduces the H200 log line verbatim. The mutation was
reverted; all 4 pass on the branch.

Note the gap of **two** tokens is load-bearing in the fixture: at gap 1 the
first `discrete` step alone happens to merge them, and the test would pass
against the unfixed scheduler — a vacuous test. Checked explicitly.

---

## 5. Is the CUDA-graph autonomous decode path now reachable? — **No**

Still blocked, and for a reason this change cannot touch.
`mistralrs-core/src/pipeline/normal.rs:1841-1847` bails when
`metadata.cache_config` is `None`, and `cache_config` is populated only from
the PagedAttention config, which `normal.rs:345-347` nulls for V4 precisely
because of §1. The gate is the paged-attention flag, not the scheduler. The
whole 3-tier GPU-autonomous decode plan (`memory/project_cuda_graph_plan.md`)
stays unreachable on the only model Arc serves until §1e is done — and it is a
`debug!`, so nothing in a session log says so.

Corollary worth stating plainly: **§1e is not just a batching improvement. It
is the single prerequisite for CUDA-graph decode, ragged/continuous batching,
and the fused GPU sampler (`arc-cuda-graph/src/sampling_cuda.rs:361-368`, only
reachable through the autonomous-decode runner) — all three at once.**

---

## 6. What needs a GPU

1. **Re-run the B-sweep on this branch** and read the engine's
   `running`/`waiting` counters. Prediction: `waiting` goes to 0 within a few
   steps of each batch release, and `running == B` thereafter. If it does not,
   the arrival pattern produces buckets further apart than the payback rule
   accepts, and `COALESCE_PAYBACK_STEPS` needs raising.
2. **Expect aggregate throughput to roughly double at B=32/B=64, not to scale.**
   The MoE 8-token cap still bites. Run with
   `ARC_QTIP_ONDEVICE_MOE_MAX_TOKENS=64` to separate the two terms — that is
   AY's item 1 and it remains the whole ballgame.
3. **§1e feasibility probe** (cheap, ~15 min): does
   `flashinfer_mla_decode` accept `kv_lora_rank=448`? Not answerable from the
   Rust side.

Nothing here needs a re-bake or a re-quantize.

---

## 7. Surfaced, not shipped

> **Noticed:** `NormalCacheManager::clone_in_cache`
> (`kv_cache/mod.rs:307-470`) silently trusts that every sequence in the batch
> has `seqs[0]`'s `current_seq_len`, and `CACHE_GROW_SIZE = 512` guarantees the
> shape check that would catch a violation passes anyway. A single
> `debug_assert!` on equal `current_seq_len` would convert any future
> scheduler bug from wrong tokens into a loud failure.
> Worth a separate change?

> **Noticed:** `deepseek4.rs` carries a complete, RUN-167-tested PagedAttention
> integration (`:855`, `:1386-1418`, `:3217-3221`, tests at `:4942`) that is
> unreachable because of one `Ok(false)` in the loader. The dispatch tests pass
> against a path production never executes — a ninth entry for the BACKLOG's
> "wired but dead" list, and structurally identical to the four `cuda_tensor_ptr`
> cases.
> Worth a separate change?

> **Noticed:** `COALESCE_PAYBACK_STEPS` is a hard-coded 256 with real
> throughput consequences and no logging — the same silent-tuning-knob pattern
> AY flagged for the four `ARC_QTIP_*` env vars. If the GPU re-run needs it
> moved, it should become an observable.
> Worth a separate change?
