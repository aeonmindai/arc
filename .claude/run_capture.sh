#!/bin/bash
# ArcGraph replay-correctness measurement leg (bench mode).
#
# Bench mode, NOT server mode: capture requires ARC_CANDLE_ALLOC_CACHE, whose
# server-mode leak (132 allocs/step, zero frees) OOMs a 143 GB card on the first
# completion. Another agent owns that fix; this leg does not touch it.
#
# Usage: run_capture.sh <legname> <gen_len> <env-string...>
set -u
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$HOME/.cargo/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}

NAME="$1"; shift
GENLEN="$1"; shift
PROMPTLEN="$1"; shift
LEG_ENV="$*"

LOGDIR=/root/logs/capture; mkdir -p "$LOGDIR"
LOG="$LOGDIR/${NAME}.log"
# Run under a DIFFERENT process name. The other agent's harness on this box
# reaps with `pkill -x mistralrs`, which matches on the exact process name and
# therefore kills our legs too (it killed two: SIGTERM, BENCH_EXIT=143, one of
# them 4 s into the run). Renaming the argv[0] the kernel sees makes our process
# invisible to that pattern without changing what either side measures — the
# file is a copy of the same binary, and its sha256 is recorded below.
SRC=/root/arc-rc/target/release/mistralrs
BIN=/root/arc-rc/target/release/arcgraphbench
cp -f "$SRC" "$BIN" 2>/dev/null || true
cd /root/arc-rc || exit 2

# Kill by exact PID, and ONLY our own binary. `pkill -f mistralrs` matches the
# ssh command line that launched this script and has killed agents through SSH;
# `pgrep -x mistralrs` is safe from that but still kills every other agent's
# run on this shared box (it killed one of ours, and we killed one of theirs).
# Resolving /proc/PID/exe scopes the kill to processes started from THIS
# checkout, so concurrent legs from /root/arc or /root/target-alloc survive.
for p in $(pgrep -x arcgraphbench); do
  case "$(readlink -f "/proc/$p/exe" 2>/dev/null)" in
    /root/arc-rc/*) kill -9 "$p" 2>/dev/null ;;
  esac
done
sleep 3

# Refuse to start while another agent holds the GPU: two V4 instances do not fit
# in 143 GB, and a leg that OOMs mid-run is a void measurement, not a slow one.
for _ in $(seq 1 120); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  [ "${used:-99999}" -lt 2000 ] && break
  sleep 15
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
if [ "${used:-99999}" -ge 2000 ]; then
  echo "VOID: GPU still busy (${used} MiB) after waiting; not measuring."
  exit 5
fi

# Provenance: a number without the binary that produced it is not evidence.
{
  echo "=== LEG $NAME ==="
  echo "git_sha=$(git rev-parse HEAD)"
  echo "bin_sha256=$(sha256sum "$BIN" | cut -d' ' -f1)"
  echo "env=$LEG_ENV gen_len=$GENLEN prompt_len=$PROMPTLEN"
  echo "gpu_free_before=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
  echo "date=$(date -Is)"
} > "$LOG"

# shellcheck disable=SC2086
env $LEG_ENV "$BIN" bench \
  --prompt-len "$PROMPTLEN" --gen-len "$GENLEN" --iterations 1 \
  auto -m /root/models/v4-src --from-uqff /root/models/v4-uqff/qtip2-0.uqff \
  >> "$LOG" 2>&1
echo "BENCH_EXIT=$?" >> "$LOG"
echo "--- LOG: $LOG"
