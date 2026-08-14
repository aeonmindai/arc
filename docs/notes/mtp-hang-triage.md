# MTP depth-2 serve hang — desk triage (session-5 prep)

**Symptom (session 4, 2026-08-14):** `serve … --mtp-depth 2 --from-uqff <qtip2 bake>`
came up healthy, logged the `MTP speculative decode engaged (depth=2)` line
(the #15 source-checkpoint fallback loaded the block UNQUANTIZED), accepted the
first request — then GPU sat at 0% and the probe hung until its qlib timeout
(3600 s) fired. No error response, no crash line in the serve log.

Desk analysis only — no GPU available. Code read: `mistralrs-core/src/pipeline/
mtp_pipeline.rs` (fast-path `step()`), `models/deepseek4.rs` (`MtpBlock`,
`load_mtp_block_from_source`), `pipeline/isq.rs` (#15 DummyLayer repair),
`pipeline/inputs_processor.rs` (prompt/completion chunk builders),
`engine/mod.rs` (step invocation + error macro).

## Finding 1 — CONFIRMED BUG (fixed in this PR): KV cache over-truncated by one on every rejected chain

`MtpSpeculativePipeline::step()` (step 5, mtp_pipeline.rs) dropped
`n_proposed - n_accepted` cache positions after the verify forward.

Derivation of the correct amount:

- Plain-decode invariant: entering a decode step, the target cache holds the
  KV of every committed token **except the last** — `make_completion_chunk`
  (inputs_processor.rs:527) feeds only `toks[len-1..]`, with token-count-derived
  `seqlen_offset`s. Verified, not assumed.
- One MTP step: the step-1 forward adds 1 position; the verify forward over
  `[T0, P1, …, P_{d-1}]` adds `d = n_proposed`; the step commits
  `1 (T0) + accepted + correction` tokens.
- Restoring the invariant requires keeping `accepted + correction`
  (= `VerifyResult::commit_len()`) of the `d` extras, i.e.
  `n_drop = n_proposed - commit_len()`. On a rejection that is
  `n_proposed - n_accepted - 1` — one FEWER than the old code dropped: the
  correction token is committed but was never a verify input, so it has no
  cache slot to charge.

Effect of the old code: after the **first** rejected chain the cache is one
position short of the committed tokens, and the gap grows by one per
rejection. All downstream position bookkeeping (RoPE offsets, flash-attn
`cumulative_seqlens_k`, V4 absorbed-decode gathers) is token-count-derived, so
attention silently reads a cache that is shorter than every locus says it is —
garbage logits at minimum; on the CUDA flash/varlen paths a K-length larger
than the physical cache is an out-of-bounds read class. With the real block at
depth 2, first-token rejections are expected (30-70%), so the desync begins
within the first few decode steps of the very first request.

Fix: `n_cache_positions_to_drop()` (pure function + CPU unit tests pinning the
per-case counts and the restored invariant).

Is this THE hang? It is the only confirmed defect on the path, and it fires
exactly where the hang was observed (first request, real block, depth 2). A
cache/position desync of this kind on the V4 CUDA decode path can plausibly
present as a stuck request rather than a clean error (see Finding 3). It is
not, however, *proven* to hang — treat "fixed but hang unverified" as the
status until the session-5 re-probe.

## Finding 2 — hazard removed (this PR): pipeline-mutex `block_on` inside the async step

`current_normal_cache_len` / `truncate_normal_cache` ran
`futures::executor::block_on(this.target.lock())` from inside the async
`step()` — parking a tokio worker on the pipeline mutex, adjacent to the
engine's `get_mut_arcmutex!` try-lock spin (utils/mod.rs). Any future path
that holds the target mutex across an await turns this into a
parked-thread/live-spin deadlock: GPU 0%, no error — the exact observed shape.
No concrete deadlocking interleaving was identified in the current code, so
this is a hazard class, not a proven cause. Both helpers now read
`self.target_cache` directly (an `EitherCache::Normal` clone shares its
`Arc<Mutex<NormalCache>>` with the target), so the pipeline lock is not
touched at all.

## Finding 3 — OPEN (most likely remaining hang mechanism): the verify forward exercises V4 shapes nothing else ever runs

The verify forward is a **multi-token, prompt-style forward at a non-zero KV
offset mid-decode** (`set_prefill_toks` + `last_n_context_len`). Every other
V4 code path is either a from-zero prefill or a single-token decode. Two known
V4-specific side-state suspects:

1. **Compressor `xs_history`** is one per-model `Mutex<SingleCache>`
   (deepseek4.rs:791). The verify forward appends `d` entries to it, and
   `truncate_normal_cache` rolls back ONLY the per-layer `KvCache`s — xs_history
   is never truncated, so rejected speculative positions stay in compressor
   history forever. This is the same state-family as the session-4 vote crash
   (`narrow [1,2,18,512]`: shared xs_history cannot hold 2 sibling chains) —
   single shared history vs branched/rolled-back decode. The queued
   per-sequence xs_history fix should treat MTP rollback as a client, not just
   voting.
2. **Absorbed-decode / flash path dispatch for seq_len=2 at offset**: never hit
   outside MTP verify; a mask/varlen mismatch here surfaces as a CUDA-side
   fault mid-stream, which candle can report at an arbitrary later sync point
   — or wedge the stream — rather than as a clean `Err` at the launch site.

Also noted while reading: the "EOS short-circuit" comment in `step()` (T0 ==
EOS commits-and-bails) does not match the code — the chain + verify still run
before the EOS commit. Wasted work only, not a hang; left as-is.

## What was ruled out

- Engine-level error swallowing: `handle_pipeline_forward_error!` sends an
  error response to the client on any `Err` from `step()` — a propagated error
  cannot present as a silent hang.
- `propose_chain` non-termination: the loop is `for i in 0..depth.min(budget)`,
  strictly bounded; depth 2 cannot loop.
- Source-fallback block on wrong device/dtype: `load_mtp_block_from_source`
  pins to the last real layer's device via `PinnedLayerMapper` and loads with
  the serve dtype; `MtpBlock::forward_step` moves inputs `to_device` both ways.
  On a single H200 there is no cross-device edge, and a dtype mismatch would
  error (→ error response), not hang.

## Session-5 on-box diagnostic (10 minutes, run BEFORE any MTP timed probe)

1. Serve `--mtp-depth 2` with `ARC_MTP_LOG_ACCEPTANCE=1`, send the 128-token
   probe with a **short client timeout (120 s)**, never 3600.
2. If it hangs: `gdb -p $SERVE_PID -batch -ex 'thread apply all bt'` (or
   `eu-stack -p`) — one snapshot answers spin vs park vs stuck-in-kernel
   instantly; plus `top -H -p $SERVE_PID` (100% CPU thread = live-spin, 0% =
   parked/deadlock) and `nvidia-smi` (kernel resident = wedged stream).
3. Re-run with `CUDA_LAUNCH_BLOCKING=1` — converts any async CUDA fault from
   Finding 3 into an immediate, correctly-attributed error line.
4. If it now generates but output degrades after the first rejections, the
   Finding-3 xs_history rollback gap is the remaining bug — capture
   acceptance-rate lines + output tail and stop (fix offline).
