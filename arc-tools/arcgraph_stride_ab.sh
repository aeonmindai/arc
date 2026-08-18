#!/usr/bin/env bash
# ArcGraph dedicated-decode A/B: does the path survive a multi-sequence batch?
#
# Derived from /root/schedfix_run.sh, with ONE deliberate inversion: that script
# exports ARC_NO_DEDICATED_DECODE=1 to steer around this defect. This one runs
# with the path LIVE, which is the default configuration in production.
#
# Pre-registered outcomes: arc-tools/prereg/arcgraph_blocktable_stride.md
#
# Exit codes are load-bearing:
#   0 = A RESULT (served, or died -- both are results)
#   1 = harness itself failed
#   2 = ENVIRONMENT could not answer. Never 1 for an environment problem.
#
# A leg killed by the defect under test is a RESULT, not a vacuum. The control
# arm is EXPECTED to die; reporting that as ENVFAIL would discard the only
# observation the run exists to make.
set -uo pipefail

BIN=${1:?binary path}
TAG=${2:?arm tag, e.g. control|M1|M2}
PORT=${3:-1257}
EXPECT_MD5=${4:-}          # optional: prove the binary you measure

LOG=/root/arcgraph-stride-logs
OUT=$LOG/$TAG
rm -rf "$OUT"; mkdir -p "$OUT"
[ -f /root/arcenv.sh ] && source /root/arcenv.sh

# Everything this script prints goes to a file AS IT HAPPENS, not at exit.
#
# A sibling chain piped its run through `| tail -60`, which buffers until EOF,
# and spent twenty minutes unable to distinguish "working" from "hung" except
# via ps and nvidia-smi. That matters more here than there: the control arm is
# EXPECTED to die, and "died the way the defect dies" has to be separable from
# "died on load" while it is happening, not reconstructed from a truncated
# buffer afterwards.
#
# LAUNCH CONVENTION -- do NOT pipe this script through tail/head/less:
#   setsid nohup arc-tools/arcgraph_stride_ab.sh <bin> <tag> <port> \
#     </dev/null >/root/arcgraph-stride-logs/<tag>.run.log 2>&1 &
# then poll  tail -n 40 /root/arcgraph-stride-logs/<tag>/run.log
exec > >(tee -a "$OUT/run.log") 2>&1

say() { echo "[$(date -u +%T)] $*"; }

# ---- 0. Prove the binary. A stale binary yields a clean flat result that is
#         indistinguishable from an honest negative, biased toward "no effect".
[ -x "$BIN" ] || { say "ENVFAIL: $BIN is not executable"; exit 2; }
GOT_MD5=$(md5sum "$BIN" | cut -d' ' -f1)
say "binary=$BIN md5=$GOT_MD5"
echo "$GOT_MD5" > "$OUT/binary.md5"
if [ -n "$EXPECT_MD5" ] && [ "$GOT_MD5" != "$EXPECT_MD5" ]; then
  say "ENVFAIL: binary md5 $GOT_MD5 != expected $EXPECT_MD5 -- measuring the wrong build"
  exit 2
fi

# ---- 1. CLAIM THE LOCK FIRST. Claim-then-read, never read-then-claim.
LOCK=/root/locks/gpu.lock
mkdir -p /root/locks
if [ -e "$LOCK" ] && grep -q "^arcgraph-stride" "$LOCK" 2>/dev/null; then
  say "lock: taking over my own reservation: $(cat "$LOCK")"
  echo "arcgraph-stride/$TAG started=$(date -u +%FT%TZ) pid=$$" > "$LOCK"
elif ! ( set -o noclobber; echo "arcgraph-stride/$TAG started=$(date -u +%FT%TZ) pid=$$" > "$LOCK" ) 2>/dev/null; then
  say "ENVFAIL: gpu.lock held by: $(cat "$LOCK" 2>/dev/null)"
  exit 2
fi
cleanup() {
  [ -n "${SRVPID:-}" ] && kill "$SRVPID" 2>/dev/null
  sleep 3
  echo "arcgraph-stride reserved=$(date -u +%FT%TZ) reason=between-arms" > "$LOCK"
}
trap cleanup EXIT

# ---- 2. The lock is worthless as a signal, in BOTH directions. An 87 GB server
#         has run with no lock file. compute-apps is the only real check.
APPS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ' | sort -u | tr '\n' ',')
if [ -n "$APPS" ]; then
  say "ENVFAIL: GPU already has compute apps [$APPS] -- aborting rather than measuring beside a neighbour"
  exit 2
fi
say "exclusive OK at start"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader > "$OUT/gpu.txt"

# ---- 3. Serve with the dedicated decode path LIVE (production default).
#         ARC_NO_DEDICATED_DECODE is deliberately NOT exported.
if [ -n "${ARC_NO_DEDICATED_DECODE:-}" ]; then
  say "ENVFAIL: ARC_NO_DEDICATED_DECODE is set in the environment -- that is the"
  say "         configuration this leg exists to NOT test. Unset it and re-run."
  exit 2
fi
setsid nohup "$BIN" serve -p "$PORT" -m /root/models/qwen05 \
  --max-seqs 256 --prefix-cache-n 0 \
  --paged-attn on --pa-cache-type auto --pa-memory-mb 8192 \
  < /dev/null > "$OUT/server.log" 2>&1 &
SRVPID=$!
say "server pid=$SRVPID tag=$TAG port=$PORT"

# ---- 4. Readiness, distinguishing "not yet" from "died during startup".
READY=0
for _ in $(seq 1 180); do
  if ! kill -0 "$SRVPID" 2>/dev/null; then
    say "ENVFAIL: server exited during startup (before any decode) -- tail:"
    tail -40 "$OUT/server.log"
    exit 2
  fi
  curl -s -m 3 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && { READY=1; break; }
  sleep 2
done
[ "$READY" = 1 ] || { say "ENVFAIL: server never became ready"; tail -40 "$OUT/server.log"; exit 2; }

# ---- 5. ASSERT ENGAGEMENT. A green run must prove the code under test ran.
#         This is the INVERSION of schedfix_run.sh, which asserts the path is
#         OFF. Here it must be ON, or the arm says nothing about the defect.
if grep -q "Dedicated decode path extraction skipped" "$OUT/server.log"; then
  say "ENVFAIL: extraction was SKIPPED -- the path under test never initialised."
  exit 2
fi
if ! grep -qE "Dedicated decode path initialized|Decode path: [0-9]+ layers extracted" "$OUT/server.log"; then
  say "ENVFAIL: no evidence the dedicated decode path initialised. This arm cannot"
  say "         answer the question; 'no failures' is not 'no defect'."
  grep -iE "dedicated|decode path" "$OUT/server.log" | head -10
  exit 2
fi
say "engagement OK: $(grep -m1 -E 'Dedicated decode path initialized|Decode path: [0-9]+ layers extracted' "$OUT/server.log")"
grep -qi "PagedAttention KV cache" "$OUT/server.log" || {
  say "ENVFAIL: no PagedAttention KV cache -- the paged path is what reaches the defect"; exit 2; }
say "paged OK: $(grep -m1 -i 'PagedAttention KV cache' "$OUT/server.log")"

# ---- 6. The ladder. B=1 CANNOT discriminate (both strides agree at offset 0,
#         and capacity==batch), so 1 is a control cell, not evidence. B=8 is the
#         first cell that can express either defect.
python3 /root/schedfix_tput.py --url "http://127.0.0.1:$PORT/v1/completions" \
  --out "$OUT/tput.json" --max-tokens 64 --ladder 1,8,32
RC=$?
say "harness rc=$RC"

# ---- 7. Verdict. Death here is a RESULT.
DIED=0
kill -0 "$SRVPID" 2>/dev/null || DIED=1
CUDAERR=$(grep -icE "CUDA error|illegal memory access|invalid argument" "$OUT/server.log" || true)
GLIBC=$(grep -icE "malloc_consolidate|corrupted double-linked list|free\(\): |munmap_chunk" "$OUT/server.log" || true)
CAPLINE=$(grep -m1 "Capturing CUDA graph" "$OUT/server.log" || echo "(no capture attempted)")

{
  echo "tag=$TAG"
  echo "binary_md5=$GOT_MD5"
  echo "server_died=$DIED"
  echo "cuda_error_lines=$CUDAERR"
  echo "glibc_diagnostic_lines=$GLIBC"
  echo "capture=$CAPLINE"
  echo "harness_rc=$RC"
} | tee "$OUT/verdict.txt"

if [ "$DIED" = 1 ] || [ "$CUDAERR" -gt 0 ] || [ "$GLIBC" -gt 0 ]; then
  say "VERDICT: DIED / errored -- this is a RESULT, not an environment failure."
  grep -inE "CUDA error|illegal memory access|invalid argument|malloc_consolidate|corrupted double-linked list|panicked" \
    "$OUT/server.log" | tail -10
  exit 0
fi

python3 /root/schedfix_check.py "$OUT/tput.json" || {
  say "ENVFAIL: the throughput harness exits 0 even when every request errored;"
  say "         its own checker rejected this output, so there is no measurement."
  exit 2; }

say "VERDICT: SERVED all cells with the dedicated decode path live, 0 CUDA errors, 0 glibc diagnostics."
exit 0
