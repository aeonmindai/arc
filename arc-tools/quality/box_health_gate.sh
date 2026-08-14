#!/usr/bin/env bash
#
# Box health gate — run this IMMEDIATELY after boot, before any long job.
#
# WHY THIS EXISTS (session 5, 2026-08-14, box s5a = NY H200 192.241.248.189):
# the SAME session-3 binary (cca7a9c2e, proven ~30 s/layer) baked at ~3 min/
# layer on that rental. A bisect proved the slowdown was ENVIRONMENTAL, not
# our kernels. The signature was:
#
#     GPU utilization  99%        <- looks perfectly healthy
#     power.draw       ~132 W of a 700 W limit   <- the tell
#     clocks.sm        at max
#     temperature      fine
#
# i.e. the device was starved waiting on host<->device transfers (the QTIP
# Viterbi quantize streams every expert CPU->GPU->CPU, so it is transfer-heavy,
# not compute-heavy). ~1.5 h and ~$7 were burned before anyone noticed.
#
# The lesson: a GPU rental can be individually bad in a way that neither
# nvidia-smi utilization nor a compile check will show you. Detect it in
# 4 minutes, then DELETE the instance and re-rent a different box/region.
# Do not debug a rental. Do not "see how it goes".
#
# Usage:
#   bash box_health_gate.sh                                # ~90 s, static + burn
#   bash box_health_gate.sh --with-bake-probe /path/to/V4  # + 3 min real-bake pace
#   bash box_health_gate.sh --json /srv/arcstatus/box_health.json
#
# Options:
#   --json <path>            Where to write the JSON report.
#                            Default: /srv/arcstatus/box_health.json if that
#                            directory exists, else ./box_health.json
#   --burn <mode>            auto|example|torch|nvcc|cargo|none  (default auto)
#   --burn-secs <n>          Synthetic load duration, seconds (default 60)
#   --min-power <watts>      Absolute sustained-power floor; overrides the
#                            scaled default (see POWER heuristic below)
#   --with-bake-probe <dir>  Run the REAL quantize against <dir> for
#                            --bake-probe-secs, count layer-progress lines,
#                            then kill it. This is the highest-signal check.
#   --bake-probe-secs <n>    Bake probe window, seconds (default 180)
#   --bake-arch <arch>       Arch flag for the bake probe (default deepseekv4)
#   --arc <dir>              Arc repo root (default: $ARC, else this script's
#                            grandparent directory)
#   --gpu <index>            GPU index to interrogate (default 0)
#   --min-disk-gb <n>        Free-disk floor in GB (default 600)
#   --min-cores <n>          nproc floor (default 16)
#
# THRESHOLDS ARE HEURISTICS FROM ONE DATAPOINT (s5a). They are tuned to catch
# a box that is 6x slow, not to grade a box that is 10% slow:
#
#   POWER   FAIL if sustained draw under load < 28.6% of the enforced power
#           limit (= 200 W on a 700 W H200 — the number from the FACTS entry).
#           s5a measured 132 W / 700 W = 18.9%. Override with --min-power.
#   BAKE    FAIL if the bake probe completes < 3 layer-progress lines in
#           3 minutes. Session 3's healthy bake did ~30 s/layer (~6 in 3 min);
#           s5a did ~3 min/layer (~1 in 3 min).
#
# Exit codes:
#   0 — every gate PASSed (warnings allowed): proceed with the session
#   1 — at least one gate FAILed: DELETE THE INSTANCE AND RE-RENT a different
#       box (and preferably a different region). Do not debug it.
#   2 — invocation error / no usable nvidia-smi
#
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

JSON_OUT=""
BURN_MODE="auto"
BURN_SECS=60
MIN_POWER_W=""
BAKE_MODEL=""
BAKE_PROBE_SECS=180
BAKE_ARCH="deepseekv4"
ARC_ROOT="${ARC:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
GPU_INDEX=0
MIN_DISK_GB=600
MIN_CORES=16

# Fraction of the enforced power limit we demand under load.
# 200 W / 700 W on the H200 the heuristic was derived from.
POWER_FLOOR_FRACTION_PCT=286   # per-mille-ish: 28.6% expressed as 286/1000
# Minimum layer-progress lines the bake probe must produce in its window.
MIN_BAKE_LAYERS=3

while [ $# -gt 0 ]; do
  case "$1" in
    --json)             JSON_OUT="${2:?--json needs a path}"; shift 2 ;;
    --burn)             BURN_MODE="${2:?--burn needs a mode}"; shift 2 ;;
    --burn-secs)        BURN_SECS="${2:?--burn-secs needs a number}"; shift 2 ;;
    --min-power)        MIN_POWER_W="${2:?--min-power needs watts}"; shift 2 ;;
    --with-bake-probe)  BAKE_MODEL="${2:?--with-bake-probe needs a model dir}"; shift 2 ;;
    --bake-probe-secs)  BAKE_PROBE_SECS="${2:?--bake-probe-secs needs a number}"; shift 2 ;;
    --bake-arch)        BAKE_ARCH="${2:?--bake-arch needs an arch}"; shift 2 ;;
    --arc)              ARC_ROOT="${2:?--arc needs a directory}"; shift 2 ;;
    --gpu)              GPU_INDEX="${2:?--gpu needs an index}"; shift 2 ;;
    --min-disk-gb)      MIN_DISK_GB="${2:?--min-disk-gb needs a number}"; shift 2 ;;
    --min-cores)        MIN_CORES="${2:?--min-cores needs a number}"; shift 2 ;;
    -h|--help)          sed -n '2,70p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "box_health_gate: unknown argument '$1'" >&2; exit 2 ;;
  esac
done

if [ -z "$JSON_OUT" ]; then
  if [ -d /srv/arcstatus ]; then JSON_OUT=/srv/arcstatus/box_health.json
  else JSON_OUT="$PWD/box_health.json"; fi
fi

WORKDIR=$(mktemp -d "${TMPDIR:-/tmp}/box_health.XXXXXX") || exit 2
cleanup() { rm -rf "$WORKDIR"; }
trap cleanup EXIT

FAILS=0
WARNS=0
JSON_FIELDS=()

json_escape() { printf '%s' "$1" | sed -e 's/\\/\\\\/g' -e 's/"/\\"/g' -e 's/\t/ /g'; }
add_str()  { JSON_FIELDS+=("\"$1\": \"$(json_escape "$2")\""); }
add_raw()  { JSON_FIELDS+=("\"$1\": $2"); }
add_bool() { if [ "$2" = "1" ]; then add_raw "$1" true; else add_raw "$1" false; fi; }

# PID-kill only, never a pkill pattern (session-2 lesson: a pattern kill once
# took out the wrong generation of the chain). TERM, grace, then KILL, then
# reap — stderr is swallowed so bash's "Terminated" job notice stays out of
# the PASS/FAIL stream.
kill_and_reap() {
  local pid=$1 grace=${2:-2}
  [ -z "$pid" ] && return 0
  kill "$pid" 2>/dev/null
  sleep "$grace"
  kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null
  wait "$pid" 2>/dev/null
  return 0
}

pass() { echo "PASS  $1"; }
warn() { echo "WARN  $1"; WARNS=$((WARNS + 1)); }
fail() { echo "FAIL  $1"; FAILS=$((FAILS + 1)); }
note() { echo "      $1"; }

# nvidia-smi CSV field, units stripped and trimmed. Prints "" on error.
# Only leading/trailing whitespace goes — GPU names have interior spaces.
smi() {
  nvidia-smi --query-gpu="$1" --format=csv,noheader,nounits -i "$GPU_INDEX" 2>/dev/null \
    | head -1 | tr -d '\r' | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//'
}

# Integer part of a possibly-fractional number ("132.45" -> "132"); "" stays "".
int_of() { printf '%s' "${1%%.*}"; }

echo "=== Arc box health gate — $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "    reference failure: s5a NY H200, 99% util at 132W/700W, ~3 min/layer"
echo

# ---------------------------------------------------------------------------
# 1. Identity / driver
# ---------------------------------------------------------------------------
if ! command -v nvidia-smi >/dev/null 2>&1; then
  fail "nvidia-smi not found — this box has no usable NVIDIA driver"
  add_bool nvidia_smi 0
  printf '{ "verdict": "FAIL", "reason": "no nvidia-smi" }\n' > "$JSON_OUT"
  exit 2
fi
add_bool nvidia_smi 1

GPU_NAME=$(smi name)
DRIVER=$(smi driver_version)
MEM_TOTAL=$(smi memory.total)
if [ -z "$GPU_NAME" ]; then
  fail "nvidia-smi present but GPU $GPU_INDEX did not answer a query"
  printf '{ "verdict": "FAIL", "reason": "gpu query failed" }\n' > "$JSON_OUT"
  exit 2
fi
pass "GPU $GPU_INDEX: $GPU_NAME, driver $DRIVER, ${MEM_TOTAL} MiB"
add_str gpu_name "$GPU_NAME"
add_str driver_version "$DRIVER"
add_raw memory_total_mib "${MEM_TOTAL:-0}"

# ---------------------------------------------------------------------------
# 2. Link topology — a downshifted PCIe link is the cheapest explanation for a
#    transfer-starved box, and it costs nothing to read.
# ---------------------------------------------------------------------------
LINK_GEN=$(smi pcie.link.gen.current)
LINK_GEN_MAX=$(smi pcie.link.gen.max)
LINK_W=$(smi pcie.link.width.current)
LINK_W_MAX=$(smi pcie.link.width.max)
add_raw pcie_gen "${LINK_GEN:-0}"
add_raw pcie_gen_max "${LINK_GEN_MAX:-0}"
add_raw pcie_width "${LINK_W:-0}"
add_raw pcie_width_max "${LINK_W_MAX:-0}"
if [ -n "$LINK_GEN" ] && [ -n "$LINK_GEN_MAX" ] && [ "$LINK_GEN" -lt "$LINK_GEN_MAX" ] 2>/dev/null; then
  warn "PCIe link downshifted: gen $LINK_GEN of max $LINK_GEN_MAX (idle links do downshift; re-read under load below)"
else
  pass "PCIe gen $LINK_GEN/$LINK_GEN_MAX, width x$LINK_W/x$LINK_W_MAX"
fi

# ---------------------------------------------------------------------------
# 3. Host: cores and disk. A 149 GB download plus a 68 GB bake needs room.
# ---------------------------------------------------------------------------
CORES=$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 0)
add_raw nproc "$CORES"
if [ "$CORES" -lt "$MIN_CORES" ] 2>/dev/null; then
  fail "nproc=$CORES < $MIN_CORES cores"
else
  pass "nproc=$CORES"
fi

DISK_FREE_GB=$(df -BG --output=avail / 2>/dev/null | tail -1 | tr -dc '0-9')
if [ -z "$DISK_FREE_GB" ]; then
  # Fallback for non-GNU df (BSD/macOS): 1K blocks -> GB.
  DISK_FREE_GB=$(df -k / 2>/dev/null | tail -1 | awk '{print int($4/1048576)}')
fi
[ -z "$DISK_FREE_GB" ] && DISK_FREE_GB=0
add_raw disk_free_gb "$DISK_FREE_GB"
if [ "$DISK_FREE_GB" -lt "$MIN_DISK_GB" ] 2>/dev/null; then
  fail "disk free ${DISK_FREE_GB}GB < ${MIN_DISK_GB}GB on /"
else
  pass "disk free ${DISK_FREE_GB}GB on /"
fi

# ---------------------------------------------------------------------------
# 4. Synthetic load + power sampling — THE gate that would have caught s5a.
#    A healthy board pulls a large fraction of its limit under a mixed
#    compute + H2D-transfer load. A transfer-starved board idles at ~19%
#    while still reporting 99% utilization.
# ---------------------------------------------------------------------------
POWER_LIMIT=$(int_of "$(smi power.limit)")
[ -z "$POWER_LIMIT" ] && POWER_LIMIT=0
CLOCK_MAX=$(smi clocks.max.sm)
add_raw power_limit_w "${POWER_LIMIT:-0}"
add_raw clocks_max_sm_mhz "${CLOCK_MAX:-0}"

if [ -n "$MIN_POWER_W" ]; then
  POWER_FLOOR="$MIN_POWER_W"
elif [ "$POWER_LIMIT" -gt 0 ] 2>/dev/null; then
  POWER_FLOOR=$(( POWER_LIMIT * POWER_FLOOR_FRACTION_PCT / 1000 ))
else
  POWER_FLOOR=200
fi
add_raw power_floor_w "$POWER_FLOOR"

# Pick a burn. Order matters: this gate runs BEFORE the session's one build,
# so anything requiring cargo is opt-in only.
pick_burn() {
  if [ -x "$ARC_ROOT/target/release/examples/viterbi_bake_bench" ]; then echo example; return; fi
  if [ -x "$ARC_ROOT/target/release/examples/qtip_gemv_bw" ];      then echo example; return; fi
  if command -v python3 >/dev/null 2>&1 && \
     python3 -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' >/dev/null 2>&1; then
    echo torch; return
  fi
  if command -v nvcc >/dev/null 2>&1; then echo nvcc; return; fi
  echo none
}
[ "$BURN_MODE" = "auto" ] && BURN_MODE=$(pick_burn)
add_str burn_mode "$BURN_MODE"

start_burn() {
  case "$BURN_MODE" in
    example)
      # Preferred when a build already exists: viterbi_bake_bench streams
      # expert stacks CPU->GPU->CPU, which is the exact traffic pattern the
      # real bake pays and the exact thing s5a was starved on.
      local ex="$ARC_ROOT/target/release/examples/viterbi_bake_bench"
      if [ -x "$ex" ]; then
        ARC_QTIP_E=64 ARC_QTIP_N=256 ARC_QTIP_K=7168 \
          "$ex" > "$WORKDIR/burn.log" 2>&1 &
      else
        "$ARC_ROOT/target/release/examples/qtip_gemv_bw" > "$WORKDIR/burn.log" 2>&1 &
      fi
      BURN_PID=$!
      ;;
    torch)
      cat > "$WORKDIR/burn.py" <<'PYEOF'
import sys, time, torch
secs = float(sys.argv[1])
dev = torch.device("cuda")
n = 8192
a = torch.randn(n, n, device=dev, dtype=torch.bfloat16)
b = torch.randn(n, n, device=dev, dtype=torch.bfloat16)
# Pinned host buffer: the H2D leg is the one that starves on a bad box.
host = torch.randn(n, n, dtype=torch.bfloat16).pin_memory()
end = time.time() + secs
while time.time() < end:
    for _ in range(20):
        c = a @ b
    g = host.to(dev, non_blocking=False)
    _ = (c + g).sum().item()
torch.cuda.synchronize()
print("burn done")
PYEOF
      python3 "$WORKDIR/burn.py" "$BURN_SECS" > "$WORKDIR/burn.log" 2>&1 &
      BURN_PID=$!
      ;;
    nvcc)
      cat > "$WORKDIR/burn.cu" <<'CUEOF'
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
__global__ void burn(float* out, const float* in, int n, int iters) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float x = in[i];
  for (int t = 0; t < iters; ++t) x = fmaf(x, 1.000001f, 1e-6f);
  out[i] = x;
}
int main(int argc, char** argv) {
  double secs = (argc > 1) ? atof(argv[1]) : 60.0;
  const int n = 1 << 24;
  size_t bytes = (size_t)n * sizeof(float);
  float *d_in = 0, *d_out = 0, *h = 0;
  if (cudaMallocHost(&h, bytes) != cudaSuccess) return 1;
  if (cudaMalloc(&d_in, bytes) != cudaSuccess) return 1;
  if (cudaMalloc(&d_out, bytes) != cudaSuccess) return 1;
  for (int i = 0; i < n; ++i) h[i] = 1.0f;
  clock_t t0 = clock();
  while ((double)(clock() - t0) / CLOCKS_PER_SEC < secs) {
    /* H2D leg: the transfer path a starved box stalls on. */
    cudaMemcpy(d_in, h, bytes, cudaMemcpyHostToDevice);
    burn<<<(n + 255) / 256, 256>>>(d_out, d_in, n, 4096);
    cudaDeviceSynchronize();
  }
  printf("burn done\n");
  return 0;
}
CUEOF
      if ! nvcc -O2 -o "$WORKDIR/burn" "$WORKDIR/burn.cu" > "$WORKDIR/nvcc.log" 2>&1; then
        return 1
      fi
      "$WORKDIR/burn" "$BURN_SECS" > "$WORKDIR/burn.log" 2>&1 &
      BURN_PID=$!
      ;;
    cargo)
      ( cd "$ARC_ROOT" && cargo run --release -p mistralrs-quant \
          --example qtip_gemv_bw --features cuda ) > "$WORKDIR/burn.log" 2>&1 &
      BURN_PID=$!
      ;;
    none) return 1 ;;
    *) echo "box_health_gate: unknown --burn mode '$BURN_MODE'" >&2; exit 2 ;;
  esac
  return 0
}

POWER_MEDIAN=""
POWER_MAX=""
CLOCK_MEDIAN=""
BURN_PID=""
if start_burn; then
  note "burn: $BURN_MODE (pid $BURN_PID), sampling power for ${BURN_SECS}s"
  : > "$WORKDIR/power.txt"
  : > "$WORKDIR/clock.txt"
  ELAPSED=0
  # Skip the first fifth of the window: ramp-up samples are not "sustained".
  RAMP=$(( BURN_SECS / 5 ))
  while [ "$ELAPSED" -lt "$BURN_SECS" ]; do
    sleep 2
    ELAPSED=$((ELAPSED + 2))
    [ "$ELAPSED" -le "$RAMP" ] && continue
    P=$(int_of "$(smi power.draw)")
    C=$(int_of "$(smi clocks.sm)")
    [ -n "$P" ] && echo "$P" >> "$WORKDIR/power.txt"
    [ -n "$C" ] && echo "$C" >> "$WORKDIR/clock.txt"
    # A burn that dies early (OOM, missing lib) must not be read as "low power".
    if ! kill -0 "$BURN_PID" 2>/dev/null; then break; fi
  done
  kill_and_reap "$BURN_PID" 2 2>/dev/null

  median_of() {
    local f=$1 n
    n=$(wc -l < "$f" | tr -d ' ')
    [ "${n:-0}" -eq 0 ] && return 1
    sort -n "$f" | sed -n "$(( (n + 1) / 2 ))p"
  }
  POWER_MEDIAN=$(median_of "$WORKDIR/power.txt")
  POWER_MAX=$(sort -n "$WORKDIR/power.txt" 2>/dev/null | tail -1)
  CLOCK_MEDIAN=$(median_of "$WORKDIR/clock.txt")
else
  warn "no synthetic load available (--burn $BURN_MODE): the power gate — the one that would have caught s5a — DID NOT RUN"
fi

add_raw power_sustained_w "${POWER_MEDIAN:-null}"
add_raw power_peak_w "${POWER_MAX:-null}"
add_raw clocks_sm_under_load_mhz "${CLOCK_MEDIAN:-null}"

if [ -n "$POWER_MEDIAN" ]; then
  PCT=0
  [ "$POWER_LIMIT" -gt 0 ] 2>/dev/null && PCT=$(( POWER_MEDIAN * 100 / POWER_LIMIT ))
  add_raw power_pct_of_limit "$PCT"
  if [ "$POWER_MEDIAN" -lt "$POWER_FLOOR" ] 2>/dev/null; then
    fail "sustained power ${POWER_MEDIAN}W (${PCT}% of ${POWER_LIMIT}W limit, peak ${POWER_MAX}W) < ${POWER_FLOOR}W floor"
    note "this is the s5a signature: the GPU is starved on host<->device transfers."
    note "DELETE THE INSTANCE AND RE-RENT a different box/region. Do not debug it."
  else
    pass "sustained power ${POWER_MEDIAN}W (${PCT}% of ${POWER_LIMIT}W limit, peak ${POWER_MAX}W) >= ${POWER_FLOOR}W floor"
  fi
else
  add_raw power_pct_of_limit null
fi

if [ -n "$CLOCK_MEDIAN" ] && [ -n "$CLOCK_MAX" ] && [ "$CLOCK_MAX" -gt 0 ] 2>/dev/null; then
  CLOCK_PCT=$(( CLOCK_MEDIAN * 100 / CLOCK_MAX ))
  add_raw clocks_pct_of_max "$CLOCK_PCT"
  if [ "$CLOCK_PCT" -lt 60 ]; then
    warn "SM clocks ${CLOCK_MEDIAN}/${CLOCK_MAX} MHz (${CLOCK_PCT}%) under load — thermal or power capping"
  else
    pass "SM clocks ${CLOCK_MEDIAN}/${CLOCK_MAX} MHz (${CLOCK_PCT}%) under load"
    if [ -n "$POWER_MEDIAN" ] && [ "$POWER_MEDIAN" -lt "$POWER_FLOOR" ] 2>/dev/null; then
      note "clocks maxed AND power low: textbook transfer starvation, not throttling."
    fi
  fi
else
  add_raw clocks_pct_of_max null
fi

# ---------------------------------------------------------------------------
# 5. Bake-pace probe (opt-in) — the ground truth. Everything above is a proxy
#    for "will the real quantize move at 30 s/layer or 3 min/layer".
# ---------------------------------------------------------------------------
add_raw bake_probe_layers null
add_raw bake_probe_secs "$BAKE_PROBE_SECS"
if [ -n "$BAKE_MODEL" ]; then
  BAKE_BIN="$ARC_ROOT/target/release/mistralrs"
  if [ ! -x "$BAKE_BIN" ]; then
    warn "bake probe skipped: $BAKE_BIN not built yet (run this gate again after step 1)"
  elif [ ! -d "$BAKE_MODEL" ]; then
    warn "bake probe skipped: model dir '$BAKE_MODEL' not found"
  else
    BAKE_OUT="$WORKDIR/uqff_probe"
    BAKE_LOG="$WORKDIR/bake_probe.log"
    mkdir -p "$BAKE_OUT"
    note "bake probe: quantizing $BAKE_MODEL for ${BAKE_PROBE_SECS}s, then killing it"
    "$BAKE_BIN" quantize text -m "$BAKE_MODEL" -a "$BAKE_ARCH" --isq qtip2 \
      -o "$BAKE_OUT/" > "$BAKE_LOG" 2>&1 &
    BAKE_PID=$!
    WAITED=0
    while [ "$WAITED" -lt "$BAKE_PROBE_SECS" ]; do
      sleep 5
      WAITED=$((WAITED + 5))
      kill -0 "$BAKE_PID" 2>/dev/null || break
    done
    kill_and_reap "$BAKE_PID" 5 2>/dev/null

    LAYERS=$(grep -c "Detected INT4" "$BAKE_LOG" 2>/dev/null | tr -d ' ')
    [ -z "$LAYERS" ] && LAYERS=0
    add_raw bake_probe_layers "$LAYERS"
    if [ "$LAYERS" -lt "$MIN_BAKE_LAYERS" ] 2>/dev/null; then
      fail "bake pace: $LAYERS layer-progress lines in ${WAITED}s (< $MIN_BAKE_LAYERS)"
      note "s5a did ~1 per 3 min; a healthy session-3 bake did ~30 s/layer (~6 per 3 min)."
      note "DELETE THE INSTANCE AND RE-RENT a different box/region."
    else
      pass "bake pace: $LAYERS layer-progress lines in ${WAITED}s (>= $MIN_BAKE_LAYERS)"
    fi

    # Free intelligence from the same log, both cheap and both session-5 traps.
    if grep -q "QTIP GPU quantize fallback" "$BAKE_LOG" 2>/dev/null; then
      fail "bake probe logged a QTIP GPU quantize fallback — the bake would run on CPU"
      grep -m1 "QTIP GPU quantize fallback" "$BAKE_LOG" | sed 's/^/      /'
      add_bool bake_probe_gpu_fallback 1
    else
      add_bool bake_probe_gpu_fallback 0
    fi
    ISQ_THREADS_LINE=$(grep -m1 -E "Applying (immediate )?ISQ .*threads" "$BAKE_LOG" 2>/dev/null)
    if [ -n "$ISQ_THREADS_LINE" ]; then
      add_str isq_threads_line "$ISQ_THREADS_LINE"
      ISQ_N=$(printf '%s' "$ISQ_THREADS_LINE" | grep -oE '[0-9]+ threads?' | grep -oE '[0-9]+' | head -1)
      note "ISQ threads: ${ISQ_N:-?} — $ISQ_THREADS_LINE"
      if [ -n "$ISQ_N" ] && [ "$ISQ_N" -gt 2 ] 2>/dev/null; then
        warn "QTIP bake asked for $ISQ_N ISQ threads: this binary predates the GPU thread-policy fix. Re-export MISTRALRS_ISQ_SINGLETHREAD=1 for the bake."
      fi
    fi
    rm -rf "$BAKE_OUT"
  fi
fi

# ---------------------------------------------------------------------------
# Verdict + JSON
# ---------------------------------------------------------------------------
if [ "$FAILS" -gt 0 ]; then VERDICT=FAIL; else VERDICT=PASS; fi
add_str verdict "$VERDICT"
add_raw fails "$FAILS"
add_raw warns "$WARNS"
add_str timestamp "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
add_str hostname "$(hostname 2>/dev/null || echo unknown)"

{
  printf '{\n'
  for i in "${!JSON_FIELDS[@]}"; do
    if [ "$i" -eq $(( ${#JSON_FIELDS[@]} - 1 )) ]; then
      printf '  %s\n' "${JSON_FIELDS[$i]}"
    else
      printf '  %s,\n' "${JSON_FIELDS[$i]}"
    fi
  done
  printf '}\n'
} > "$JSON_OUT"

echo
echo "=== VERDICT: $VERDICT ($FAILS fail, $WARNS warn) — report: $JSON_OUT ==="
if [ "$VERDICT" = "FAIL" ]; then
  echo "    ACTION: delete this instance and re-rent a different box/region."
  echo "    Do NOT debug the rental — s5a cost ~1.5h and ~\$7 learning that."
  exit 1
fi
exit 0
