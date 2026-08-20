# wave59-CJ — MTP at batch: lockstep landed, and the ceiling that is left

**Branch:** `feat/mtp-batch-lockstep` off `c763123be` (origin/master). **Draft PR #86.**
**Hardware used: none.** Everything below is code, arithmetic, or a CPU unit test.
**Nothing here is an acceptance measurement.** The only acceptance number on record is
still wave51-CB's `accept_rate=0.4194`, `tok_per_step=1.8387`, **B=1**.

---

## 0. The correction to the brief's framing

The brief (from wave56-CG §2) says the `18 <> 22` gap is produced by
`mtp_pipeline.rs:2503` committing `1 + accepted_i` while rolling the shared cache to the
batch minimum. **That mechanism cannot produce it, and I could not re-derive the gap from
master.** The reason is short and worth keeping:

* the fused step operates on **one** dense cache. `truncate_cache_by` drops `n_drop` from
  every slot of that one object, and `clone_out_cache` copies the result onto every
  sequence in the batch. So a cohort that enters a step uniform **leaves it uniform**, in
  `tokens` and in `base`, on the K/V slots and the `XsRolling` slots alike.
* the divergence is in `get_toks().len()`, not in the cache. That is by design and is
  exactly what `u ∈ [1, w]` absorbs.

So the `274 vs 278` pair came from sequences whose per-sequence cache state was stamped in
**different** batches (bucket coalescing, a waitlisted sequence rejoining, a prefix-cache
restore), which `ensure_uniform_batch_cache_lens` — a `debug_assert!` in the shipped
binary, absent from it — never saw. wave56-CG's own §7.1 called this "the structural hole"
and was right; §2's attribution to the commit/rollback asymmetry was one inference too far.
**Recorded as a correction, not as a defect in #84** — #84's fix is correct and this builds
on it.

## 1. The invariant that was actually missing

`clone_in_cache` needs **two** things from a batch and only checked one:

1. every sequence agrees on `current_seq_len` per slot — `ensure_uniform_batch_cache_lens`;
2. every sequence agrees on **`XsRollingCache::base`**.

(2) does not follow from (1). `tail` is `tokens - base` wide, and `base` is
`canonical(high-water tokens)`, **not** a function of the current length: `set_len` narrows
the tail without moving `base`, and `advance` never lowers it. Two sequences at the
identical token count can therefore hold tails **4 and 132** wide.

🔑 **This needs no MTP.** `prefix_cacher.rs:434` calls `set_len(match_len)` on every stored
layer, so a sequence restored from an entry stored at a greater length holds a *narrower*
tail than one that reached the same point directly. It is a second, independent route into
`kv_cache/mod.rs:499` on the **ordinary decode path**, and it was not on anyone's list.
Pinned by `xs_base_divergence_at_equal_lengths_is_reconciled_not_refused`, which asserts
the legacy batcher still panics on the fixture (`shape mismatch on dim 1, 4 <> 132`) so the
reproduction is re-checked every CI run.

## 2. What was implemented — decision (A), lockstep

| piece | file | what it does |
|---|---|---|
| `XsRollingCache::trim_tail_to` | `kv_cache/xs_rolling.rs` | raises `base`, refusing if it would cut into rows the next compressed row is built from |
| `XsRollingCache::compressor_needs_from` | same | the retention rule and the trim rule, named once, `debug_assert`-tied to `advance`'s own `need_start` |
| `reconcile_xs_bases` | `kv_cache/mod.rs` | per layer, trims every member to the batch's **largest** `base` |
| `align_batch_cache_lens` | `kv_cache/mod.rs` | rolls a ragged cohort back to its shortest slot, bounded by `max_uncached` |
| `enforce_shared_cache_lockstep` | `pipeline/mtp_pipeline.rs` | release postcondition after the verify rollback |

**Why trimming to the batch maximum is lossless, not a compromise:** the member already
sitting at `base_max` proves, by its own successful history, that no future compressed row
needs a token below it. The rows dropped only ever shortened how far a *rollback* could
reach, and after the trim every member shares the reach the shared batched cache would have
had anyway.

**Why the alignment is MTP-only:** dropping cache positions is lossless iff the caller
re-feeds them next forward. Plain decode feeds `toks[len-1..]` — one token — so a blanket
alignment in `clone_in_cache` would serve requests from a hole, silently. Batched MTP feeds
`toks[cache_len..]` as *real* tokens before any draft, for any `u ∈ 1..=w`. So the call
sits in `MtpSpeculativePipeline::clone_in_cache`, skipped entirely when the batch already
agrees (metadata-only check), and refuses by name past `window()`.

`Sequence::cache_bucket_len` is **unchanged.** Option (B) was not taken and was not needed.

## 3. 🔴 THE CEILING THAT IS LEFT — arithmetic, not measurement

`plan_batch_step` advances the one dense cache by `keep = min_i(u_i + a_i)`, so
`next_u_i = u_i + a_i + 1 - keep`. **The moment one sequence in the batch rejects its first
draft** (`u=1, a=0`), `keep = 1` and that collapses to:

```
next_u_i = u_i + a_i          — monotonically non-decreasing
```

A sequence's uncached tail ratchets upward until it reaches `w = depth + 1`, where it
drafts `w - u = 0`, accepts 0, and **is a fixed point**. ⇒ **the sequences that accept best
are exactly the ones that stop drafting.**

* At **B=1** this cannot happen: `keep = u + a`, so `next_u ≡ 1` and the sequence always
  drafts the full depth. **That is why the only number ever measured was measured at B=1.**
* At **B=128**, `P(no sequence rejects immediately) = p^B`. With the implied per-position
  `p ≈ 0.485` (fitted below), that is `0.485^128 ≈ 10^-40`. `keep = 1` is certain.

Pinned by `one_laggard_ratchets_every_other_sequences_tail_to_the_window`: saturation
inside **2 steps** at B=5, with a B=1 control in the same test.

**This is a property of one dense cache carrying one length, not of the draft head.** It
bounds what any amount of draft-quality work can buy at batch, and it is why the honest
projection below is a *range with the ceiling attached*, not a single number.

### Fitting `p` from the one real measurement
`tok_per_step = 1.8387` at depth 3 ⇒ `E[accepted] = 0.8387` per step. Solving
`p + p² + p³ = 0.8387` gives **`p ≈ 0.485`**. Then:

| depth | `E[acc]` | `k` | Δ vs depth 3 |
|---|---|---|---|
| 1 | 0.485 | 1.485 | −0.35 |
| 2 | 0.720 | 1.720 | −0.12 |
| **3** | **0.839** | **1.84** | — (measured) |
| 4 | 0.893 | 1.89 | **+0.05** |
| 5 | 0.919 | 1.92 | +0.08 |

⇒ **Depth tuning is NOT the lever.** Depth 3 is already in the saturating regime; depth 4
buys +0.05 `k` and costs a wider verify window (`w = depth+1` rows per step, so +25%
verify compute) *and* raises the saturation bound in §3. Scoped and declined, with numbers.

### Ranked, for whoever takes this next
1. **Lift the min-rollback ceiling** — the only item that changes the batch answer. Needs
   the KV cache to advance per sequence: paged attention (**ruled out for V4**) or splitting
   the cohort by accept length, which is option (B) under another name and pays
   `select_running_bucket`'s one-bucket-per-step tax. **Expected gain: everything, at
   B ≥ 8. Everything else is worth ~nothing until this moves.**
2. **Accept rule / typical acceptance** — the standard lever for raising `p`. At the
   temperature the benchmarks run (`t=0`) typical acceptance is *identical* to the exact
   argmax match `verify_proposed` already does, so **zero gain on our own protocol**. Only
   pays at `t > 0`, and it is not lossless. Declined, with the reason.
3. **Depth** — +0.05 `k`, see table. Declined.
4. Tree/multi-candidate drafting and EAGLE-3 were already declined in wave45-BW and were
   not re-litigated.

## 4. Tests — all mutation-proven, all D12-shaped

Fixtures are built the way production builds them: preallocated K/V `all_data` (never
`KvCache::new_normal`'s `None`), and `XsRolling` state produced by running the real
`XsRollingCache::advance` / `V4Compressor::forward_from_xs`.

| test | asserts |
|---|---|
| `batched_ragged_accept_keeps_every_v4_cache_slot_in_lockstep` | 7 steps of ragged accept through the real V4 cache at **both** ratios (4 and 128); every slot at one length after each step, **and** the compressor rows bit-comparable to a from-scratch recompute |
| `a_cache_slot_that_drifts_out_of_lockstep_is_repaired_or_named` | the 274/278 pair; a slot ahead is rolled back 22→18 wide, a slot behind is refused naming slot and both lengths |
| `ragged_xs_tail_is_aligned_for_a_speculative_caller` | the exact crash fixture batches after alignment; committed tokens shown untouched |
| `align_batch_cache_lens_refuses_past_the_speculative_window` | 9 uncached vs a 4-wide window refused, batch left unmutated (two-loop discipline) |
| `xs_base_divergence_at_equal_lengths_is_reconciled_not_refused` | equal lengths, tails 4 vs 132, reconciled |
| `trimming_the_retained_window_past_what_the_compressor_needs_is_refused` | the lossless boundary is exactly `compressor_needs_from()` |
| `one_laggard_ratchets_every_other_sequences_tail_to_the_window` | §3, with a B=1 control |
| `ragged_xs_tail_is_refused_by_name_not_panicked` (kept, extended) | the **plain** path still refuses — it has no way to re-feed |

**Mutation runs (fix stubbed, tests re-run):**
```
enforce_shared_cache_lockstep -> Ok(())
  a_cache_slot_that_drifts_out_of_lockstep_is_repaired_or_named ... FAILED
  left: [274, 274, 278]   right: [274, 274, 274]
reconcile_xs_bases call removed
  xs_base_divergence_at_equal_lengths_is_reconciled_not_refused ... FAILED
```
⚠️ `batched_ragged_accept_keeps_every_v4_cache_slot_in_lockstep` **passes** with
`enforce_shared_cache_lockstep` stubbed — `truncate_cache_by` alone already holds lockstep
on that fixture. That is not a vacuous test, it is the correct pairing: it proves the
*arithmetic*, and the drift test proves the *enforcement*. Recorded so nobody later reads
it as the enforcement's own proof.

## 5. Gates

`cargo check` green · `cargo test -p mistralrs-core` **369 passed, 0 failed** (was 332) ·
scoped clippy lane green · `mistralrs-core` clippy clean in every file touched ·
**zero new rustfmt drift** — `deepseek4.rs` checked like-for-like on a copy, 567 → 567.

⚠️ **`rustfmt` on `mistralrs-core/src/models/deepseek4.rs` reformatted 165 pre-existing
lines** the first time I ran it. Reverted and the new tests re-applied by hand. The
`mod.rs` hazard in the standing block is real and **it is not limited to `mod.rs`** — any
large upstream-derived file will do it. Also: `rustfmt --check` run *in place* on a
`mod.rs` recurses into submodules and reports their drift as yours (72 lines for
`kv_cache/mod.rs`, 779 for `pipeline/mod.rs`); **copy the file to `/tmp` and check it
there** to get a like-for-like number. Both were 0 → 0 that way.

## 6. Surfaced, not shipped

1. **The prefix cacher can hand back an `XsRolling` slot whose `base` is arbitrarily far
   from canonical** (§1). Reconciled at batch time now, but the underlying asymmetry —
   `set_len` moving `tokens` and not `base` — is still there and is worth its own look.
2. **wave56-CG §7.3 is closed by reading, not by a fix**: `engine/mod.rs:567` sets
   `last_completion_ids = vec![]` after any prompt step, so the next completion step gets
   `pre_op = In`. The "prompt step overwrites a pipeline cache a `Nothing` completion step
   then reuses" scenario cannot fire through that path. Leaving the backlog entry open for
   the *other* half (prompt and completion in one selected bucket) which I did not verify.
3. **`MtpAcceptance::from_fused_verify(d, n_acc)` records `proposed = d`**, so a sequence
   that drafted 0 because it saturated (§3) contributes a step with 0 proposed and 0
   accepted. `accept_rate = accepted/proposed` therefore stays *flattering* while
   `tok_per_step` collapses. **At batch, read `tok_per_step`, not `accept_rate`** — they
   diverge exactly where it matters. Worth an explicit `drafting_sequences / batch` counter.
