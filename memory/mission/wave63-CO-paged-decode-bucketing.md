# wave63-CO — the paged scheduler serialised decode for a reason that does not apply to it

**Branch:** `worktree-agent-a22bfedb9b05a584f` off `52b9a934f`, stacked on
`fix/ragged-prefill-logits` (**PR #117**).
**Hardware used: H200 box B, Qwen2.5-0.5B-Instruct, exclusive card.** §3 is a measurement.
Everything in §1–§2 is code read; §4–§5 are defects routed elsewhere, not fixed here.

---

## 0. The question, and the answer

> Is the paged path's `(cache length, has_images && is_prompt, token_offset)` bucketing a
> genuine requirement, or a workaround for a padding bug that should be fixed at its source?

**Neither, exactly. On decode it is a requirement that was never derived — it was inherited.
On prefill it is a workaround, and only one of the two prefill cases needs it.**

`paged_attention/scheduler.rs` stated the reason as *"required for correct flash attention
varlen operation (avoiding soundness issues with padding)"* (upstream #1746, Dec 2025).
Traced end to end, that sentence is true of exactly one of four cases.

## 1. 🔑 No flash-varlen call happens on either paged decode path at all

A decode step is `tgt_len == 1`. Therefore:

1. `CausalMasker::make_causal_mask_matrix` returns `None` (`layers_masker.rs:285`).
2. `PagedAttention::forward`'s `att` is `None`, so `Sdpa` is never called
   (`paged_attention/layers/paged_attention.rs:249`).
3. Control goes straight to the paged kernel, which takes **per-sequence** lengths by
   contract:

| path | evidence |
|---|---|
| non-MLA | `pagedattention.cuh:138` `context_len = context_lens[seq_idx]`; `:144` per-seq block count; `:288` per-seq key mask |
| MLA | `flashinfer/page.cuh:478` `get_length(i)`; `mla_decode_forward` does not even take `flash_params` |

The decode input is `[B, 1]` — **there is no padding to be unsound about** — and RoPE already
takes a per-sequence branch when the offsets differ (`layers.rs:1624`).

⇒ Length bucketing on decode bought nothing. What it cost is in §3.

## 2. Prefill: one case needs it, one does not

| case | verdict |
|---|---|
| **no prefix hit** | ✅ safe ragged. Q/K are right-padded to the batch max and `cu_seqlens_q/k` are built from those **same padded widths** (`inputs_processor.rs:270`), so the varlen call is self-consistent. End padding is unreachable under causal masking, and `reshape_and_cache` skips `slot < 0` (`reshape_and_cache_kernel.cu:45`) so pad K/V never enters the cache. |
| **prefix hit** | ❌ genuinely broken ragged. `PagedAttention::forward:229-235` swaps in the packed `cu_seqlens_kv` (real gathered lengths) while leaving `cumulative_seqlens_q` **padded**. Each row's bottom-right causal offset `seqlen_k - seqlen_q` is then short by that row's pad width. |

So the rule is kept for the prefix-hit case only (`LengthRule::MustMatch`) and dropped
everywhere else (`LengthRule::Free`).

**The ragged-Q machinery to fix even that case already exists and is unwired:**
`PagedAttentionInputMetadata::cu_seqlens_q` — the real per-sequence query lengths — is
built (`inputs_processor.rs:451-466`) and **read by nothing in the workspace**.

## 3. 🔴 MEASURED — H200, Qwen2.5-0.5B-Instruct, `--paged-attn on`, `--prefix-cache-n 0`

Harness `/root/batchinv_tput.py` copied **verbatim** from the batchinv chain
(md5 `e6232773a2cc7c51ddb48bb9ff43a1e4`) so the numbers are comparable to that A/B.
Card exclusive: `compute-apps` empty at start, only the server pid at end.

|   B | arm     | baseline | fix     | delta     |
|----:|---------|---------:|--------:|----------:|
|   1 | —       |   278.14 |  274.70 |     0.99x |
|   8 | uniform |   923.19 |  869.36 |     0.94x |
|   8 | spread  | **99.21**| **865.50** | **8.72x** |
|  32 | uniform |  1230.87 | 1264.72 |     1.03x |
|  32 | spread  |**147.77**|**1196.47** | **8.10x** |

(aggregate tok/s; `max_tokens=64`; spread = prompt lengths 50…500, mean matched to the
uniform arm so both arms push the same total prompt tokens.)

* Baseline B=8 spread is **2.80× slower than baseline B=1** — batching was NEGATIVE.
* After: B=8 spread is 3.15× faster than B=1, and length diversity is nearly free —
  spread/uniform is **0.996** at B=8 and **0.946** at B=32.
* The uniform arms move in opposite directions (−5.8% at B=8, +2.8% at B=32) on **single
  samples**. A neighbour took the card before a repeat, so this is noise-vs-signal
  **unresolved**, not "no regression".

**Mechanism, from the engine's own log rather than inferred** — running/waiting histogram:

```
baseline: run=2 wait=6 | run=16 wait=16 | run=28 wait=4 (x2) | run=4 wait=0
fix:      run=32 wait=0 (x2) | run=8 wait=0 | run=1 wait=0
```

Zero `waiting` anywhere in the fix arm. `running bucket = B / distinct lengths` is gone.

**Predicted from source before it was measured**, on the dense scheduler's own coalescence
rule: for B=8 at 8 distinct lengths, `(total − n_min) · gap ≤ n_min · 256` is
`(8−1)·64 = 448 > 256` ⇒ refused ⇒ the split is permanent ⇒ `1 running, 7 waiting`. That
matches the H200 measurement taken independently by the batchinv chain.

### 3.1 ⚠️ What this result does NOT say

The V4 number that motivated this work — **24.54 → 7.91 tok/s (3.10×)** — was taken on
**V4 with `--paged-attn off`**, i.e. the **dense** scheduler.
`DeepSeekV4Loader::supports_paged_attention` returns `false`
(`normal_loaders.rs:3265`). **This change does not address that number.** Different
scheduler, different model. Do not read the 8.72× as the flagship getting faster.

## 4. Also fixed at source: ragged prefill sampled from padding (PR #117, split out first)

`make_prompt_chunk` narrowed **every** row's logits at `padded.len() − 1`, the batch max —
the right-hand padding for every row but the longest. Every such sequence returned a first
token sampled from a pad column. Silent. One line; identity on uniform batches.

## 5. Defects found here, routed, NOT fixed here

### 5.1 ArcGraph — capture crashes the server at B=8

```
CUDA error at src/cuda/reshape_and_cache_kernel.cu:140: invalid argument
```
during `Capturing CUDA graph for batch_size=8` (`arc_cuda_graph::dedicated`), on
Qwen2.5-0.5B + PagedAttention. **Reproduced on the BASELINE binary — it predates this
change.** Both arms therefore ran with `ARC_NO_DEDICATED_DECODE=1`, identically, to isolate
the scheduler. Repro: `/root/schedfix_run.sh` on box B with that env var removed.

### 5.2 ArcAttention — vision encoder padding masks are discarded on the flash path

`Sdpa::run_attention`'s CUDA branch is `flash_attn(&q, &k, &v, flash_params, sdpa_params)`
(`attention/mod.rs:186`) — **no mask argument**. `siglip.rs:518` and
`idefics3/vision.rs:490` both pass a real `expand_mask` padding mask into it, and both are
dropped on a flash build.

🔑 **The obvious diagnosis is wrong and is corrected here before anyone acts on it.** This is
*not* CLAUDE.md pitfall #6 (encoders silently running causally): both call sites already
pass `FlashParams { causal: false, .. }` with empty `cumulative_seqlens`
(`siglip.rs:297-305`, `idefics3/vision.rs:288-293`), exactly as the pitfall prescribes.
Attention stays bidirectional. The real defect is narrower and still a wrong answer:

> the encoder attends over **padding patches** in variable-resolution image batches.

`idefics2` is **not** affected — it never uses `Sdpa`, applying its mask by hand via
`apply_mask_one_and_zero` (`idefics2/mod.rs:443`, `:800`).

Not fixed here because the one-line widening (`seq_len == 1` → any `seq_len`) in
`mask_must_be_applied_as_bias` on `feat/dense-ragged-decode` would move **every** vision
encoder off flash in the same change. That deserves its own measurement.

## 6. The dense path (V4's path) — scoped, not built

Three of the four pieces already exist and are tested:
`front_align_batch` (`kv_cache/mod.rs:835`), `front_pad_kv_cache` (`:807`), and
`make_left_padded_causal_mask` + `RaggedKvLens` (`layers_masker.rs:221`). Only caller is
MTP, which discards the `lead_pad` it gets back (`mtp_pipeline.rs:2219`).

🔑 **The blocker is not the one the code names.**
`MtpSpeculativePipeline::target_masks_ragged_batches` (`mtp_pipeline.rs:1363`) is a
**hardcoded `Err(...)` that probes nothing**, and its message — *"no model in this tree
threads a ragged-batch mask into its forward yet"* — points at the wrong work. Threading
`RaggedKvLens` through all 40-odd models would have changed **nothing**, because the
dispatcher discards the mask (§5.2). It would have been a silent no-op that passed every
test.

The real cost is a choice:

* **(a)** route ragged dense decode to `run_attention_noflash` with the ragged mask. Gives up
  the flash kernel on decode, where its advantage is smallest (`[B, H, 1, L]` scores). Smallest
  real change; **to be measured, not assumed**. Foundation committed on
  `feat/dense-ragged-decode`.
* **(b)** gather the dense cache into a packed layout each step so `cu_seqlens` can express
  it — that is rebuilding PagedAttention inside the dense path. **Rejected.**
* **(c)** put V4 on the paged path. The long-term answer. Two named blockers, both defects
  rather than policy: the `cache_write_and_gather` `context_lens` contract bug
  (`normal_loaders.rs:3339`, wave53-CD) and V4's `xs` compressor history having no block
  table. That reframes `supports_paged_attention() == false` from *"V4 can never page"* to
  *"V4 needs two fixes"*.

Front-alignment cannot be pushed into `cu_seqlens` instead: those describe **packed** spans
and flash's causal alignment is bottom-right, so a **front**-padded buffer is inexpressible,
and `window_size_left` is one scalar for the whole batch.

## 7. Test discipline

Every test below was **seen red by mutation** before it was trusted:

| test | red output |
|---|---|
| `decode_runs_every_admitted_sequence_at_mixed_lengths` | `decode ran at widths [1, 1, 1]` — the `1 running, 7 waiting` pathology reproduced on CPU |
| `ragged_prefill_reads_each_row_at_its_own_last_real_token` | `left: [(7, 1), (7, 1), (7, 1)]` — three rows all reading column 7, padding for two |
| `a_ragged_decode_mask_must_go_to_the_bias_path` | `assertion failed: mask_must_be_applied_as_bias(1, Some(&[8, 1, 1, 500]))` |

Plus `must_match_still_partitions_and_preempts`, the mutation guard proving the retained
prefix-hit rule still bites, and `uniform_prefill_is_unchanged`, proving the logits fix is
the identity on uniform batches.

**A measurement guard was also seen red.** The first baseline arm printed `ARM baseline OK`
while the server had **died** at B=8 — 72 errored requests, harness exit 0. The completeness
check added in response (`schedfix_check.py`: every run must return `B × max_tokens` with
zero errors) was verified against that saved JSON and correctly flags the subtlest case,
`B=8 uniform: 0 errored, 8/512 tokens`. All §3 numbers are post-guard.

## 8. Provenance hazard to know about

Both binaries log `git revision: d02079b7d…` — the baseline was built by reverting the two
files **without moving HEAD**, so the server log **cannot** distinguish the arms. What
distinguishes them: md5 `9d34749e…` (baseline) vs `7131ddc2…` (fix), `grep -c LengthRule` = 0
in the baseline tree at build time, and the §3 histograms.

`/root/locks/gpu.lock` on box B was found truncated to **0 bytes** twice (21:42:33Z and
22:14:27Z), each time an ownerless empty file, while `nvidia-smi --query-compute-apps` showed
the truth. The lock is not a signal in either direction; compute-apps is the only gate.
