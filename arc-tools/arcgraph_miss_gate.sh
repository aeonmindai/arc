#!/bin/bash
# arcgraph_miss_gate.sh — did a capture leg allocate anything the warm pool had
# never seen?
#
# Runnable without understanding CUDA graphs. Point it at a server log from a
# capture leg; it exits 0 only if the log proves the answer is NO.
#
#   ./arcgraph_miss_gate.sh /root/logs/arcgraph/capture_w8d3.server.log
#
# WHY YOU CARE (if you own a buffer, not the graph):
#   A CUDA graph records the device POINTERS its kernels were launched with. Any
#   buffer whose allocation SIZE changes from one decode step to the next gets a
#   fresh address, the graph keeps the stale one, and the result is an unstable
#   graph memory node — candle's own warning calls it a correctness bug.
#
#   The requirement is therefore stronger than "correct on ragged batches":
#
#       THE BUFFER'S ALLOCATION SIZE MUST BE CONSTANT PER DECODE STEP.
#
#   A buffer can be semantically perfect and still fail this, by being sized
#   from the running token count. That is exactly the observed failure: sizes
#   147456 / 155648 / 163840 / 172032 in one run = 4096 (hidden_size) x
#   {18,19,20,21} in BF16 — consecutive integers, one per decode step.
#
# THE SHAPE NEEDED, NOT A PRESCRIBED FIX:
#   This buffer has already been optimised hard — `8bc6af45c` took the xs
#   compressor state from per-sequence to rolled, 16.6x less memory per
#   sequence, max batch 68 -> 266. So `172032` is not a naive allocation nobody
#   thought about; it is what REMAINS after a large win, and whoever owns it
#   knows constraints this gate cannot see.
#
#   So this states a PROPERTY, not an implementation:
#     the number of BYTES requested from the allocator must not vary with the
#     token count on the decode path.
#   How that is achieved is the owner's call — allocate once at a fixed
#   capacity and narrow for reads; round the request up to a quantum so the
#   distinct-size set is finite and warmable; or keep the tail in a slab that
#   is written in place. Each trades memory or complexity against constancy,
#   and the trade is theirs to make, not this gate's to dictate.
#
#   EXPECT IT TO COST SOMETHING. Constant size on a rolled, ragged buffer very
#   likely means padding to a worst case, i.e. giving back some of that 16.6x.
#   A requirement that pretends to be free is one the owner is right to
#   distrust.
#
# EXIT CODES
#   0  proven clean: capture ran, the forward completed, the counter demonstrably
#      fires, and zero misses occurred after capture began
#   1  misses occurred after capture began (the number is printed)
#   2  UNPROVEN — the log cannot support a verdict either way. NOT success.
set -u

LOG="${1:-}"
[ -n "$LOG" ] || { echo "usage: $0 <server.log>" >&2; exit 2; }

fail_unproven() { echo "UNPROVEN: $*" >&2; echo "  -> this is NOT a pass; the log cannot answer the question." >&2; exit 2; }

# 1. The log must exist and be non-empty. `grep -c` over a missing file prints
#    0 and exits 1, which reads exactly like "no misses". That near-miss is the
#    reason this check is separate and first.
[ -f "$LOG" ] || fail_unproven "no such log: $LOG"
[ -s "$LOG" ] || fail_unproven "log is empty: $LOG"

# 2. Capture must actually have started, or "zero misses after capture" is
#    vacuously true.
grep -q "capture started for batch_size" "$LOG" \
    || fail_unproven "capture never started — zero post-capture misses would be vacuous"

# 3. The forward must have COMPLETED. A precondition confirmed is not the
#    postcondition: a truncated forward never allocates the full set, and a
#    clean-looking zero from one had to be retracted earlier today.
if grep -q "Model failed with error" "$LOG"; then
    echo "UNPROVEN: the forward errored, so any count is over a truncated allocation set." >&2
    grep -m1 -oE "Model failed with error: .{0,160}" "$LOG" | sed 's/^/    /' >&2
    exit 2
fi

# 4. POSITIVE CONTROL. Misses during the deferred-free warmup passes carry the
#    same warning string, so every healthy run exercises this counter. If the
#    string appears nowhere, the counter never fired and a zero proves nothing.
#    A zero that has never been seen non-zero is decoration.
total=$(grep -c "MISS during capture" "$LOG" 2>/dev/null); total=${total:-0}
[ "$total" -ge 1 ] 2>/dev/null \
    || fail_unproven "the miss counter never fired anywhere in this log (total=0), so a zero is unverifiable"

after=$(awk '/capture started for batch_size/{f=1} f && /MISS during capture/' "$LOG" 2>/dev/null | wc -l | tr -d ' ')

echo "counter proven live: $total miss line(s) in the log overall"
if [ "$after" -eq 0 ] 2>/dev/null; then
    echo "PASS: 0 allocations missed the warm pool after capture began."
    exit 0
fi

echo "FAIL: $after allocation(s) missed the warm pool AFTER capture began."
echo "Sizes, and what they decompose to (hidden_size=4096, BF16):"
awk '/capture started for batch_size/{f=1} f && /MISS during capture/' "$LOG" \
  | grep -oE "size [0-9]+" | sed 's/size //' | sort -n | uniq \
  | while read -r b; do
        printf '  %-10s = %s tokens x hidden_size x 2B\n' "$b" "$((b / 2 / 4096))"
    done
echo
echo "Each of these is a buffer reallocated because its size grew with the token"
echo "count. Making it constant per decode step removes it."
exit 1
