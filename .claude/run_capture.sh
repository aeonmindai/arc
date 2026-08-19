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
BIN=/root/arc-rc/target/release/mistralrs
cd /root/arc-rc || exit 2

# Kill by exact PID only. `pkill -f mistralrs` matches the ssh command line that
# launched this script and has killed agents through SSH.
for p in $(pgrep -x mistralrs); do kill -9 "$p" 2>/dev/null; done
sleep 3

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
