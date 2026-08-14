#!/usr/bin/env bash
#
# wave19-AP — build and run `gmin_bench` on a rented sm_80 box.
#
# Everything the driver needs is in one translation unit, so this is an nvcc
# invocation and nothing else: no Rust toolchain, no model weights, no crate
# build. A compile error therefore costs ~2 minutes of box time to fix rather
# than a re-entry, which is why holding the box through an iteration beats
# tearing it down here (DOCTRINE D4b's 5-minute rule).
#
# Expects the five sources in $DIR:
#   qtip_exact_fp.cuh  qtip_quantize.cu  qtip_beam.cu  qtip_gmin.cu  gmin_bench.cu
#
# Usage:  nohup bash gmin_bench_run.sh > log.txt 2>&1 &   then poll the log.

set -uo pipefail
DIR=${DIR:-/root/gmin}
ARCH=${ARCH:-sm_80}
cd "$DIR" || { echo "FAIL: no $DIR"; exit 1; }
export PATH=/usr/local/cuda/bin:${PATH}

echo "===== box ====="
nvidia-smi --query-gpu=name,compute_cap,memory.total,power.limit,clocks.max.sm \
           --format=csv 2>&1
echo "--- driver CUDA vs toolkit CUDA (must not be a downgrade) ---"
nvidia-smi 2>&1 | head -4
nvcc --version 2>&1 | tail -2

echo
echo "===== build ($ARCH) ====="
time nvcc -O3 -std=c++17 --use_fast_math -arch="$ARCH" -lineinfo -I. \
     gmin_bench.cu -o gmin_bench 2>&1 | tail -60
if [ ! -x ./gmin_bench ]; then
  echo "FAIL: build produced no binary"
  exit 1
fi

echo
echo "===== register / shared usage (occupancy is measured, never assumed) ====="
cuobjdump -res-usage ./gmin_bench 2>&1 \
  | grep -E "Function properties for|REG:|ptxas info" \
  | grep -B0 -A0 -E "gmin|beam|REG:" | head -60

echo
echo "===== A: parity + timing at 448 rows ====="
./gmin_bench 448 7168 3 2>&1

echo
echo "===== B: timing at 1344 rows (wave17-AF's shape; beam base was 1227.5 ms, stack 1011.1 ms) ====="
./gmin_bench 1344 7168 3 2>&1

echo
echo "===== C: down_proj shape, k_in=2048 ====="
./gmin_bench 1344 2048 3 2>&1

echo
echo "===== power/clocks under load (bad-box gate: <200 W under load ⇒ re-rent) ====="
( ./gmin_bench 1344 7168 6 > /dev/null 2>&1 & echo $! > /tmp/burn.pid )
for i in 1 2 3 4 5 6 7 8; do
  nvidia-smi --query-gpu=utilization.gpu,utilization.memory,power.draw,clocks.sm \
             --format=csv,noheader 2>&1
  sleep 4
done
BPID=$(cat /tmp/burn.pid)
wait "$BPID" 2>/dev/null
echo "DONE"
