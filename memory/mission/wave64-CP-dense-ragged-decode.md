# wave64-CP — ragged dense decode: correct, limiter removed, does NOT pay yet

**Branch:** `feat/dense-ragged-decode`, 4 commits on `120f34875` (PR #95 merge).
**Hardware: H200 box B, Qwen2.5-0.5B-Instruct, exclusive card.** §2 is a measurement.
**Headline: this does NOT reproduce the 2.84×. It regresses 15.7% at B=128 spread.**
Reported as measured, not shopped for a configuration where it looks better.

---

## 0. What was built

The dense scheduler ran exactly one cache-length bucket per step, so a
length-diverse cohort decoded one sequence at a time. Three of the four pieces
already existed and were unreachable — `front_align_batch`,
`make_left_padded_causal_mask` + `RaggedKvLens`, `supports_per_sequence_len`.
Only MTP called any of it, and it discarded the result.

🔑 **The code named the wrong blocker.**
`MtpSpeculativePipeline::target_masks_ragged_batches` is a hardcoded `Err`
reading *"no model in this tree threads a ragged-batch mask into its forward
yet"*. Threading it through all forty-odd models would **not** have worked:
`Sdpa::run_attention`'s flash branch calls `flash_attn(&q,&k,&v,..)`, which takes
**no mask argument**, so the mask was discarded at the dispatcher. Fixed first
(`mask_must_be_applied_as_bias`, scoped to `seq_len == 1`).

Then: a thread-local carrying each row's dead prefix, set by `clone_in_cache`
and read by `CausalMasker`, which gets the mask to every model **without a new
argument anywhere**; `clone_in_cache` front-aligns instead of refusing;
`clone_out_cache` hands each row back its own length; the scheduler pins the
length key to 0 for decode only.

The channel carries the **prefix**, not the live lengths: `clone_in_cache` runs
only when membership changes while the cache grows every step, so a stored
`live` is stale on the next token, whereas `lead_pad` stays true and
`live = past_kv_len - lead_pad` is recovered from the cache.

## 1. Guard narrowed, and the hazard that came with it

`clone_in_cache` no longer refuses every length-mismatched batch — only those it
cannot align. wave51-CB's guarantee (refuse in release, return rather than
panic, never reach `slice_set`) is preserved, now covered by a `Rotating`-slot
fixture, with `ensure_uniform_batch_cache_lens` kept as a hard postcondition.

⚠️ Review found a real hazard in that narrowing: `front_align_batch` pads **in
place**, so it could return `Err` having already rewritten the rows ahead of the
failure, where the blanket refusal it replaced mutated **nothing**.
`front_align_would_succeed` restores the property — alignment is attempted only
when known to complete. Pinned by `a_refusal_does_not_leave_a_half_aligned_batch`.

Mutation results, both run:
* remove the postcondition ⇒ **three** refusal tests go red. It discriminates.
* replace `batch_can_be_ragged` with `true` ⇒ **suite stayed GREEN.** The
  alignment decision is double-covered by `front_align_would_succeed`; the
  *capability* — what the scheduler reads — was covered by nothing. Now pinned.

Release-mode verified: 439/439, refusals return rather than panic.

## 2. 🔴 MEASURED

|   B | arm     | baseline | fix     | delta |
|----:|---------|---------:|--------:|------:|
|   1 | —       |   254.47 |  257.29 | +1.1% |
|   8 | uniform |   837.26 |  866.67 | +3.5% |
|   8 | spread  |   225.18 |  251.05 | **+11.5%** |
|  32 | uniform |  1190.24 | 1129.98 | −5.1% |
|  32 | spread  |   503.35 |  493.73 | −1.9% |
| 128 | uniform |   963.45 | 1007.14 | +4.5% |
| 128 | spread  |   801.31 |  675.24 | **−15.7%** |

Baseline B=128 spread has two samples (801.31, 771.22 ⇒ ~786±15); the fix's
single 675.24 is outside that spread.

**Correct, provably.** Uniform-batch control, teeth verified at 8 distinct
completions: baseline vs fix at B=8-uniform is **8/8 identical**; each arm
diverges from its own B=1 by exactly **4/8** — the fix adds none of its own.
Every cell returned exactly `B × 64` tokens.

### Why it does not win

* **MEASURED — the decode limiter really is gone.** Baseline sits at
  `16 running, 112 waiting` throughout B=128 spread; the fix climbs
  `33 running, 95 waiting` → `47 running, 48 waiting`.
* **MEASURED — it never reaches 128 because PROMPT admission still gates.**
  Prompts keep the exact-length key by design here, so 128 spread prompts form 8
  buckets that prefill one per step. That limiter sits in front of this one and
  is fixed on another branch. This change cannot show its ceiling behind it.
* **INFERRED, NOT MEASURED — the regression is most likely a cost I added.**
  `clone_out_cache` narrows + `contiguous()` each row carrying a dead prefix:
  `O(B × layers × 2)` extra device copies **every** decode step (~2,200/token at
  B=47). Consistent with uniform cells (where `lead == 0`, no strip) being
  flat-to-better while only spread cells regress — but not profiled.

⇒ **Named fix, not built.** The strip need not run per step: `clone_out_cache`
runs every token, but its per-sequence output is only READ when membership
changes. Carry the prefix per sequence and strip lazily at `clone_in`, moving an
`O(B × layers)` per-token cost onto membership-change events only.

**D21:** a scope result, not a verdict. Correct, limiter provably removed, named
reason it does not yet pay. Not ranked down.

## 3. 🔴 The defect the completeness guard caught

The first dense fix run returned **433/512 tokens at B=8 spread, 1518/2048 at
B=32, 7956/8192 at B=128 — ZERO reported errors, every UNIFORM cell exact**, and
the harness exited 0. That is a fake result that would otherwise have shipped.

Cause: `CausalMasker` has **two** entry points; the channel was wired into one.
Qwen2 reaches the other (see §4), so `make_sliding_window_causal_mask_matrix`
returned `None` at `tgt_len == 1` and every ragged decode batch attended its
zero-filled dead prefix — logit 0, real softmax weight, silently wrong.

Fixed: the check is factored into `ragged_mask_from_channel` and called from
**both** entry points before **every** early return; the window is applied on top
of the dead prefix (absolute-position edge shifted by `lead`, so `j > last - w`);
and `set_none_cache` clears the channel, since a prompt step takes
`CacheInstruction::Reset` and would otherwise inherit a torn-down cohort's
geometry.

🔑 **The family, hit twice in one session.** A new channel wired into one of two
dispatch paths, where which one you hit depends on model config — the vision-mask
finding (wave63-CO §5.2) is the same shape. **When adding a channel, enumerate
every entry point and every early return, and make sure a fixture reaches each.**
The suite was green throughout; it could not reach the second path because no
fixture had that config. That is the gap, not the bug.

## 4. 🟠 PRE-EXISTING, NOT MINE — `models/qwen2.rs` ignores `use_sliding_window`

`Config` (`models/qwen2.rs:30-46`) deserialises `sliding_window: Option<usize>`
and never reads `use_sliding_window`. `qwen3.rs:33`, `qwen3_moe.rs:33` and
`embedding_models/qwen3_embedding.rs:28` all gate on it.

⇒ **Every Qwen2-architecture model runs with sliding-window attention ON when its
config says OFF.** Verified on Qwen2.5-0.5B-Instruct: `sliding_window = 32768`,
`use_sliding_window = false`, engine takes the SWA mask path.

Inert while the window ≥ served context, which is why it has not surfaced — but
it changes attention on any Qwen2 model whose window is smaller than its context.

## 5. Method notes worth keeping

**A relative named ref is still a moving ref.** The baseline was pinned as
`FIX~3`, correct at three commits and silently wrong at four — it resolved to
*this branch's own first commit*, which would have put half the change under
test inside the "baseline". Both SHAs are now literals, and the build script
**proves** the pair with `git merge-base --is-ancestor` and prints the commit
count under test.

**A guard that refuses itself is working.** The identity control's first run
reported `CANNOT-ANSWER: fewer than 4 distinct B=1 completions` — a 0.5B model
summarises eight numbered copies of the same filler identically, so "8/8
identical" would have been vacuous. Fixture changed to ask for a per-prompt
password; teeth then verified at 8 distinct completions.

**Idle-card cost, recorded:** ~35 minutes of box B at $4.92/hr (~$2.90) was burnt
holding the card while doing CPU-side diagnosis after a failed arm. The rule:
release the moment you know you have code to write, not when you finish writing
it.
