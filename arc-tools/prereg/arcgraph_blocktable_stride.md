# PRE-REGISTERED — ArcGraph block-table stride defect

> ## AMENDMENT, written before any run — the stride defect is MASKED today
>
> After registering the below I checked what `decode_forward` uses for its row
> count and found `let bs = buffers.batch_size`, i.e. the capacity pinned by the
> first `ensure_buffers` call. In ordinary serving the first decode is a single
> request, so `num_seqs` is pinned to 1 and **`seq_idx` never exceeds 0 — which
> makes the stride defect below unobservable in exactly the runs that reproduce.**
>
> The same pin fires a different defect at batch > 1: `wrap_f32_logits`
> (`pipeline/mod.rs`) copies `batch_size * vocab * 4` bytes out of a buffer
> allocated `capacity * vocab * 4`. For qwen05 at B=8 with capacity 1 that is
> 4,861,952 bytes read from a 607,744-byte allocation — **4,254,208 bytes past the
> end** — and the `cudaMemcpy` return was discarded. A D2D copy whose source
> leaves its allocation returns `cudaErrorInvalidValue` = **"invalid argument"**,
> which then gets reported by the next `cudaGetLastError()` in the process: the
> only one in the captured forward is `reshape_and_cache_kernel.cu:140`. That is
> the reported mode-3 string, at the reported file and line, by misattribution.
>
> **So my predicted outcome shifts against my own first fix: S2 ("stride fix
> alone still dies") is now MORE likely than S1.** Recorded here before the box.
>
> Both defects are now fixed together (`ensure_buffers` grows and invalidates the
> graph; `decode_forward` takes the real batch size; the stride copy is pitched).
> The mutation protocol below therefore has TWO arms, and they must be run
> separately or the result cannot attribute:
>   M1. revert ONLY the pitched copy → should still serve at B=8, because the
>       capacity fix leaves `num_seqs` correct but the stride is what breaks once
>       `seq_idx > 0` is actually reached. If B=8 dies with M1, the stride defect
>       was load-bearing after all and S1/S2 must be re-read.
>   M2. revert ONLY the `ensure_buffers` growth → should die at B=8 again with
>       `invalid argument`. This is the arm that proves the capacity fix was the
>       necessary one.
>
> The severity also changes: this is not a graph bug. `PAGED_ATTN_CUDA = true`
> and `ARC_NO_DEDICATED_DECODE` is an opt-OUT, so the path is **default-on in
> production** for dense models on CUDA — and default-OFF in every measurement
> harness on the boxes (`schedfix_run.sh`, `w53_probe.sh` both export it). Nothing
> measured exercised it; nothing that exercised it was measured.


Parent system: ArcInfer / ArcGraph (capture) + ArcKV (paged metadata staging).

Written BEFORE any hardware run, commit-timestamped. The point is that the result
cannot select its own interpretation afterwards.

## The defect, stated as a claim that can be wrong

`DedicatedDecodePath` copies the per-step paged-attention block table into a
fixed-address staging buffer so a captured graph can replay against a stable
pointer. It writes that buffer with one row stride and reads it with another.

- `stage_paged_attn` (`arc-cuda-graph/src/dedicated.rs:571-597`) issues ONE flat
  D2D copy of `batch_size * actual_blocks * 4` bytes, where
  `actual_blocks = paged_attn.max_num_blocks_per_seq` (the source tensor's
  `dims()[1]`). The destination therefore carries the SOURCE row stride,
  `actual_blocks`.
- `staged_paged_attn` (`dedicated.rs:629`) hands the kernel
  `max_num_blocks_per_seq = self.staging_max_blocks_per_seq` — the ALLOCATED
  CAPACITY, `max_possible_blocks = (max_position_embeddings / block_size)
  .max(actual_blocks)` (`dedicated.rs:747-750`).
- The kernel indexes `block_table = block_tables + seq_idx * max_num_blocks_per_seq`
  (`mistralrs-paged-attn/src/cuda/pagedattention.cuh:229`).

So for `seq_idx >= 1` the kernel reads at `seq_idx * capacity` while the data was
written at `seq_idx * actual_blocks`. Whenever `capacity != actual_blocks` — which
is the normal case, capacity is derived from `max_position_embeddings` and
`actual_blocks` from the current context length — every sequence after the first
reads **uninitialised `cudaMalloc` memory and uses it as a physical KV block
index**, then dereferences `key_cache + idx * kv_block_stride`.

### Why this is invisible at batch_size = 1 and fires at batch_size = 8

At `batch_size == 1` only `seq_idx == 0` exists and `0 * capacity == 0 * actual`.
The write and the read agree at offset zero, exactly and always. The defect
cannot be observed. At `batch_size >= 2` it mis-addresses `batch_size - 1` of the
`batch_size` sequences.

The staged repro dies at the **B=8 cell**. That is the first cell in which this
defect is expressible.

### Why capture changes when it becomes visible, without being its cause

Stream capture RECORDS launches; it does not execute them. The first execution of
a captured decode is the first `cuGraphLaunch`. So under capture, an
out-of-bounds read produced by the recorded kernel surfaces at first launch, not
during the capture calls — which matches the reported
`CUDA_ERROR_ILLEGAL_ADDRESS (700) at first cuGraphLaunch`. Eagerly, the same
kernel with the same wrong stride faults during the eager step instead.

This predicts the counter-evidence the brief kept separate and told me not to
merge: **GPU-side out-of-bounds access is NOT capture-exclusive.** Four of the
five `Xid 31` MMU-fault events on one box are present and unattributed to any
capture leg. This defect produces exactly that — eager decode at batch >= 2 does
the same wild read with no graph involved. If this claim is right, those four are
explained; if the four are shown to come from batch-1-only runs, this claim is
weakened.

## What this claim does NOT explain, stated up front

- **It does not explain the host heap corruption** (`malloc_consolidate(): invalid
  chunk size`, `corrupted double-linked list`) on the V4 path. V4 uses
  `CacheBackendMetadata::DefaultInstructions` and never reaches
  `try_dedicated_decode` at all, so it never executes this code. Two signals, not
  merged.
- **It does not obviously explain `invalid argument`** (mode 3). A wild block
  index yields `cudaErrorIllegalAddress` (700, "an illegal memory access was
  encountered"), not `cudaErrorInvalidValue` (1, "invalid argument"). See the
  second candidate below, which is NOT being fixed in this change.

## Second candidate, recorded but NOT fixed here

`staged_paged_attn` sets `max_context_len = max_position_embeddings` on the
non-turbo branch (`dedicated.rs:626`). `paged_attention_v1_launcher`
(`pagedattention.cuh:704-710`) derives `shared_mem_size = padded_max_context_len *
4` from it. `run_step`'s own comment at `dedicated.rs:775-777` says using
`max_position_embeddings` here "would request 160KB+ smem and silently fail", and
the turbo branch (`dedicated.rs:608-627`) computes a device-limit-derived cap for
exactly that reason. The non-turbo branch does not. If the requested dynamic smem
exceeds the device opt-in limit, the launch returns `cudaErrorInvalidValue` =
**"invalid argument"**, which is latched and then reported by the next
`cudaGetLastError()` — and the only such check in the whole captured forward is
`reshape_and_cache_kernel.cu:140`. That is the reported mode-3 string, at the
reported file and line, by misattribution.

Left unfixed deliberately: on sm_90 the opt-in limit is ~227 KB and the computed
size at `max_seq_len = 32768` is 128 KB, so it should PASS on an H200. I do not
want to fix two things at once and be unable to say which mattered. Measure first.

## Outcomes, decided in advance

CONTROL   : baseline binary, dedicated decode ON, B=8 cell.
TREATMENT : same binary + this stride fix only. One variable.

### Gate on the control, checked FIRST
The control MUST reach the B=8 cell and MUST die. If it survives, the run is
UNPROVEN — the harness was never shown capable of reproducing the thing the
treatment is supposed to remove. Exit 2. No treatment conclusion is drawn.

### The three outcomes

**S1. Control dies at B=8; treatment serves tokens at B=8 with 0 CUDA errors and
0 new Xid 31.**
=> CONFIRMED. The stride mismatch was the B=8 death. Merge on its own merits.
   This does NOT thereby explain the V4 host heap corruption; that stays open and
   I will say so in the same breath.

**S2. Control dies at B=8; treatment still dies at B=8.**
=> ELIMINATED, NOT FIXED. Reported with equal weight. The stride bug is real by
   construction (the write stride and the read stride are different integers in
   the source) so it still gets fixed, but it is not the B=8 death, and the next
   suspect is the `max_context_len` smem sizing above.

**S3. Control does not reach or does not die at B=8.**
=> CANNOT ANSWER, exit 2. Says nothing in either direction.

### The mutation, committed now
A fix I cannot show was necessary is not demonstrated. After S1, I revert ONLY
the stride correction in the treatment binary and re-run the B=8 cell. It MUST
die again. If it does not, S1 was luck (or the arms differed in something else)
and the CONFIRMED verdict is withdrawn.

This is also why the staging buffers are deliberately **NOT** zero-filled at
allocation in this change. Zeroing would turn a wild block index into block 0 —
wrong output, but mapped memory and no fault — which would mask the defect and
destroy the mutation test above. Zeroing may be added later as defence in depth,
never as the fix.

### What would make me distrust a green treatment arm
- Control and treatment differing in anything but the stride correction.
- Treatment serving tokens at B=8 but with any CUDA error string present.
- A green B=8 that never actually captured (assert the capture log line AND a
  replay, not merely "no failures").
- Only B=1 cells green: B=1 cannot discriminate, by the argument above.
