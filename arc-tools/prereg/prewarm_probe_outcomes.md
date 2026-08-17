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
