# PRE-REGISTERED — pre-warm probe outcomes
# Written 2026-08-17, BEFORE the run, commit-timestamped. The point is that the
# result cannot select its own interpretation afterwards.

CONTROL   : capture as-is
TREATMENT : ARC_GRAPH_PREWARM_SIZES=172032 seeds the alloc cache with a block of
            exactly that size before capture, so the allocation that misses
            during capture becomes a hit.

## Gate on the control (checked FIRST, before any treatment claim)
The control MUST show the capture-time MISS and MUST crash. If it doesn't, the
run is UNPROVEN — the log was never shown capable of reporting the thing whose
absence the treatment would rely on. No treatment conclusion is drawn.

## The three outcomes, decided in advance

A. MISS gone (treatment misses = 0) AND crash gone (glibc = 0, tokens > 0)
   => CONFIRMED BY CONSTRUCTION. The capture-time allocation is the corruption.
   => The fix is NOT pre-warming. Pre-warming hides the symptom by making one
      hard-coded size a cache hit; it does nothing for the next context length,
      because the buffer is sized from the running token count and every step
      asks for a different size. The fix is that the buffer stops being
      allocated per decode step — owned by the xs_rolling chain, routed.
   => I will say "confirmed, and the probe must not be merged" in the same breath.

B. MISS gone (treatment misses = 0) BUT still crashes
   => ELIMINATED, NOT FIXED. The last capture-time allocation is NOT the cause.
   => This is a RESULT, not a setback: it removes the only remaining
      capture-time suspect and forces the search elsewhere (into cuGraph*
      internals, the cuBLAS workspace, or the stream/event handling under
      RELAXED capture). I will report it with the same weight as A.

C. MISS still present (treatment misses > 0)
   => CANNOT ANSWER, exit 2. The warming did not engage — wrong size, wrong
      timing relative to set_capture_mode, or the allocation isn't served from
      the cache at all. The run says NOTHING about the hypothesis in either
      direction. Specifically: this is NOT evidence against the hypothesis.
   => Next step is to fix the probe, not to conclude anything about the buffer.

## What would make me distrust a green treatment arm
- Control and treatment differing in anything but the env var.
- Treatment serving tokens but with a diagnostic present (partial).
- MISS absent in BOTH arms => the control gate above fires; UNPROVEN.

---

# ADDENDUM — pre-registered against the xs chain's PIN (not my pre-warm probe)
# Written 2026-08-17 BEFORE their fix landed. The pin supersedes the probe:
# it is the real change, so a green result cannot be dismissed as an artefact
# of warming one hard-coded size.

## Their derivation, verified from the code rather than accepted

  XS_TAIL_MARGIN_TOKENS = 16      (xs_rolling.rs:64)
  CSA: ratio = 4, span_groups = 2 (ratio == 4 -> overlapping compressor)
  tail bound = span_groups*ratio + margin = 2*4 + 16 = 24
  W_max      = bound - 1 = 23

=> Pinning the tail width at 24 is PROVABLY SUFFICIENT and can never truncate.
   Confirmed independently. The doc comment already states the tail is
   "bounded by span_groups*ratio + margin and independent of context length",
   so the constancy is a property the type already claims — `advance` narrowing
   to the exact width is what breaks it.

## ⚠️ A DISCREPANCY I am flagging BEFORE the result, so it cannot be waved away

  their model predicts widths {20, 21, 22, 23}   (100 decode steps, steady state)
  I measured                 {18, 19, 20, 21}   (4096 x N x 2 bytes, one run)

Both are `ratio`-consecutive and both contain 21, but they are OFFSET BY 2.

Most likely reconciliation: mine is the PRE-SATURATION RAMP. My run generated
~21 tokens total, so `base` had not finished advancing and `W = tokens - base`
was still climbing; theirs is the SATURATED CYCLE after 100 steps. Both are
bounded by 23, so a pin at 24 covers both.

TESTABLE, and I am committing to it now: a LONGER run (>= 64 generated tokens)
should show the miss sizes move to {20,21,22,23} and stop there. If a long run
still shows {18..21}, the steady-state model is wrong even though the bound
holds, and the pin's sufficiency argument needs re-deriving rather than
re-asserting.

## Outcomes against the pin (same three, applied to the real fix)

P1. MISS line gone AND server survives AND tokens > 0
    => CONFIRMED. The per-step-varying allocation was the corruption. This is
       the real fix, not a simulation; merge on its own merits.

P2. MISS line gone BUT still corrupts
    => ELIMINATED, NOT FIXED. Equal weight. My suspect is dead and the
       corrupting write is somewhere I have not looked.

P3. MISS line still present
    => CANNOT ANSWER. The pin did not take (wrong path, or a second allocation
       of the same family). Says nothing in either direction.

## WHERE I LOOK NEXT IF P2 — committed now so the result cannot pick its own follow-up

In this order, cheapest discriminator first:

1. THE cuBLAS WORKSPACE. `new_with_stream` allocates 32 MiB and calls `leak()`
   on it, then hands the raw pointer to `cublasSetWorkspace_v2`. A leaked
   pointer that cuBLAS writes to during a captured graph is exactly the shape
   of an out-of-bounds host-visible write. Test: vary the workspace size; if
   the corruption's arrival changes with it, that is the site.
2. EVENT TRACKING vs `leak()`. The fork disables cudarc event tracking because
   `leak()` does `cudaEventDestroy` + `stream.wait()` mid-capture. `leak()`
   also does `Arc::decrement_strong_count` twice by hand. If tracking is ever
   re-enabled on some path, those manual refcount decrements run against events
   that capture is concurrently touching. Test: assert `is_event_tracking()`
   is false at capture time, not just at device construction.
3. RELAXED CAPTURE MODE. `cuStreamBeginCapture_v2(..., RELAXED)` was chosen to
   tolerate cross-stream dependencies. RELAXED permits operations THREAD_LOCAL
   forbids, including some host-side allocation during capture. Test: try
   THREAD_LOCAL; if it fails with STREAM_CAPTURE_ISOLATION rather than
   corrupting, the corruption is coming from whatever RELAXED is permitting.
4. Only then the CUDA driver itself, via a minimal standalone reproducer
   outside mistralrs. Last because it is the most expensive and the least
   likely.

What I will NOT do in P2: reach for another allocation-shaped explanation
because the first one was allocation-shaped. That is the pattern that produced
three wrong answers tonight.
