#!/usr/bin/env bash
# Mock dry-run for s6_driver.sh — no GPU, no model, no network, no pip.
#
# Precedent: test_batch_load_probe.py drives the batch probe against a mock
# HTTP server. Same idea one level up — this builds a sandbox of fake
# binaries (nvidia-smi, nvcc, cargo, mistralrs, curl, git, the quality
# scripts) and runs the real driver against them, asserting the CONTROL FLOW:
# which steps run, in what order, which ABORT fires, and what the driver does
# with a partial beam-parity failure.
#
# It is not a substitute for hardware. It is the thing that stops the paid
# session from discovering at minute 90 that an ABORT branch had a typo.
#
# Scenarios:
#   A happy          every step runs; UPLOAD precedes all measurement;
#                    S6_COMPLETE; token deleted; token never appears in a log
#   B health FAIL    ABORT_BOX_HEALTH and nothing after S0 runs
#   C low compute    ABORT_BOX_COMPUTE_CAP (the S2 vacuous-pass guard)
#   D beam fail @64  ABORT_BEAM_PARITY, no bake
#   E beam fail @256 BEAM_GATE=W128 and the bake really receives W=128
#   F no bake header ABORT_BAKE_HEADER
#   G greedy header  ABORT_BAKE_GREEDY (DOCTRINE D4)
#
# Run: bash test_s6_driver.sh      (exit 0 = PASS)
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRIVER="$HERE/s6_driver.sh"
FAILURES=0
SECRET="hf_TESTTOKENvalue0123456789"

pass() { printf 'PASS  %s\n' "$1"; }
fail() { printf 'FAIL  %s\n' "$1"; FAILURES=$((FAILURES + 1)); }

# assert_has <label> <file> <pattern>
assert_has() { grep -qE "$3" "$2" && pass "$1" || { fail "$1 (missing /$3/)"; }; }
assert_not() { grep -qE "$3" "$2" && fail "$1 (unexpected /$3/)" || pass "$1"; }
# assert_before <label> <file> <first> <second>
assert_before() {
  local f=$2 a=$3 b=$4
  local ia ib
  ia=$(grep -nE "$a" "$f" | head -1 | cut -d: -f1)
  ib=$(grep -nE "$b" "$f" | head -1 | cut -d: -f1)
  if [ -n "$ia" ] && [ -n "$ib" ] && [ "$ia" -lt "$ib" ]; then pass "$1"
  else fail "$1 (a=@${ia:-none} b=@${ib:-none})"; fi
}

# --------------------------------------------------------------- sandbox
make_sandbox() {
  SB=$(mktemp -d "${TMPDIR:-/tmp}/s6test.XXXXXX")
  mkdir -p "$SB/bin" "$SB/root/logs" "$SB/status" "$SB/arc/arc-tools/quality/patches" \
           "$SB/arc/chat_templates" "$SB/models/V4" "$SB/arc/arc-tools/quality/results"
  echo "{}" > "$SB/arc/chat_templates/deepseek_v4.json"
  printf '#!/usr/bin/env bash\necho "DATA_READY: 12M in ./data"\n' \
    > "$SB/arc/arc-tools/quality/fetch_data.sh"
  cp "$HERE/patches/s6_mtp_acceptance_telemetry.patch" \
     "$SB/arc/arc-tools/quality/patches/" 2>/dev/null
  cp "$HERE/s6_upload_uqff.py" "$SB/root/" 2>/dev/null
  printf '%s' "$SECRET" > "$SB/root/.hf_token"; chmod 600 "$SB/root/.hf_token"
  # a couple of fake model shards so du/preflight have something to see
  head -c 4096 /dev/zero > "$SB/models/V4/model-00001.safetensors"

  # -------- nvidia-smi
  cat > "$SB/bin/nvidia-smi" <<'EOF'
#!/usr/bin/env bash
for a in "$@"; do
  case "$a" in
    --query-gpu=compute_cap)     echo "${FAKE_CC:-9.0}"; exit 0 ;;
    --query-gpu=memory.total)    echo "143771"; exit 0 ;;
    --query-gpu=name)            echo "NVIDIA H200"; exit 0 ;;
    --query-gpu=utilization.gpu) echo "97"; exit 0 ;;
    --query-gpu=*)               echo "97, 512.0, 68000, 1980"; exit 0 ;;
  esac
done
echo "| NVIDIA-SMI 580.173   Driver Version: 580.173   CUDA Version: 13.0  |"
EOF
  # -------- nvcc
  cat > "$SB/bin/nvcc" <<'EOF'
#!/usr/bin/env bash
echo "Cuda compilation tools, release 13.0, V13.0.88"
EOF
  # -------- curl (health endpoint follows a flag file)
  cat > "$SB/bin/curl" <<EOF
#!/usr/bin/env bash
[ -f "$SB/serve_up" ] && exit 0 || exit 7
EOF
  # -------- timeout (macOS has none; the box has coreutils)
  cat > "$SB/bin/timeout" <<'EOF'
#!/usr/bin/env bash
shift
exec "$@"
EOF
  # -------- du -sb (GNU only; macOS du has no -b)
  cat > "$SB/bin/du" <<'EOF'
#!/usr/bin/env bash
if [ "${1:-}" = "-sb" ]; then echo -e "68000000000\t${2:-.}"; exit 0; fi
exec /usr/bin/du "$@"
EOF
  # -------- git
  cat > "$SB/bin/git" <<'EOF'
#!/usr/bin/env bash
case "$*" in
  *"apply --check"*) [ "${FAKE_PATCH_OK:-1}" = "1" ] && exit 0 || exit 1 ;;
  *"apply "*)        exit 0 ;;
  *"rev-parse HEAD"*) echo "d6ceaf1ad23ed10b2f1434577782076553d98fbc" ;;
  *"log --oneline"*) echo "d6ceaf1ad session6 fake" ;;
  *) exit 0 ;;
esac
EOF
  # -------- box health gate
  cat > "$SB/bin/box_health_gate.sh" <<'EOF'
#!/usr/bin/env bash
echo "PASS  GPU 0: NVIDIA H200"
echo "=== VERDICT: ${FAKE_HEALTH_VERDICT:-PASS} ==="
exit "${FAKE_HEALTH:-0}"
EOF
  # -------- boot
  cat > "$SB/bin/boot.sh" <<'EOF'
#!/usr/bin/env bash
echo ":::::: 0 env ::::::"
echo "OK: V4 already downloaded"
echo "BOOTSTRAP_COMPLETE — ready to serve."
EOF
  # -------- cargo
  cat > "$SB/bin/cargo" <<EOF
#!/usr/bin/env bash
SB="$SB"
EOF
  cat >> "$SB/bin/cargo" <<'EOF'
sub=$1
case "$sub" in
  build) echo "    Finished \`release\` profile"; exit 0 ;;
  test)
    fw=${FAKE_BEAM_FAIL:-0}
    for w in 64 128 256; do
      if [ "$fw" != "0" ] && [ "$w" = "$fw" ]; then
        echo "test qtip::tests::cuda_beam_matches_cpu_beam_bit_for_bit ... FAILED"
        echo "thread 'main' panicked at mistralrs-quant/src/qtip/mod.rs:4437:"
        echo "W=$w: CUDA beam differs from the CPU beam in 12/4096 bytes"
        echo "test result: FAILED. 3 passed; 1 failed; 0 ignored"
        exit 101
      fi
    done
    echo "test qtip::tests::cuda_beam_matches_cpu_beam_bit_for_bit ... ok"
    echo "test qtip::tests::cuda_beam_unpruned_matches_cuda_exhaustive ... ok"
    echo "test qtip::tests::cuda_exhaustive_matches_cpu_exhaustive_bit_for_bit ... ok"
    echo "test qtip::tests::cuda_search_plan_never_substitutes_a_width ... ok"
    echo "test qtip::tests::cuda_3d_expert_quantize_does_not_fall_back_to_cpu ... ok"
    echo "test qtip::bitshift::tests::cuda_3d_expert_quantize_does_not_fall_back_to_cpu ... ok"
    echo "test result: ok. 6 passed; 0 failed; 0 ignored"
    exit 0 ;;
  run)
    case "$*" in
      *qtip_gemv_tune*)
        echo "=== qtip2b GEMV autotune: 98 variants (44 gen1 + 54 gen2) + legacy ==="
        echo "  -> WINNER [gate]: v57 | total 512.0us vs legacy 1665.5us (-69.3%)"
        printf '{"variants":98}\n' > "$SB/tune_results.json"
        echo "WINNER_TABLE_WRITTEN: $SB/tune_results.json"
        exit 0 ;;
      *stats_info*--layers*)
        echo "  isq  name        in_feat     tokens   gram experts  symbolic"
        echo "   12  gate_up         4096       8192      -   64/64  layers.3.mlp.experts"
        exit 0 ;;
      *stats_info*--layer*)
        echo "=== ISQ layer 12 ==="
        echo "in_features     4096"
        echo "diag/tokens     min 1.0e-04  mean 4.0e-03  max 1.2e-01  (dynamic range 1181.0x)"
        exit 0 ;;
    esac
    exit 0 ;;
esac
exit 0
EOF
  # -------- mistralrs
  cat > "$SB/bin/mistralrs" <<EOF
#!/usr/bin/env bash
SB="$SB"
EOF
  cat >> "$SB/bin/mistralrs" <<'EOF'
cmd=$1
case "$cmd" in
  quantize)
    echo "$ARC_QTIP_BEAM" > "$SB/bake_beam_width"
    echo "${MISTRALRS_ISQ_SINGLETHREAD:-unset}" > "$SB/bake_singlethread"
    case "${FAKE_BAKE_HEADER:-beam}" in
      none)   : ;;
      greedy) echo "QTIP bake [qtip2]: mode=greedy search=greedy-walk (no trellis search) objective=mse (unweighted) rotation=off" ;;
      *)      echo "QTIP bake [qtip2]: mode=viterbi search=viterbi-beam(W=${ARC_QTIP_BEAM}) objective=mse (unweighted) rotation=hadamard-128" ;;
    esac
    echo "ISQ thread policy: 1 thread(s) — QTIP quantize runs in GPU kernels on one device"
    for i in 1 2 3 4 5 6 7 8; do echo "Detected INT4 layer $i"; done
    out=""
    for a in "$@"; do case "$prev" in -o) out=$a;; esac; prev=$a; done
    mkdir -p "$out"
    printf 'uqff' > "$out/qtip2-0.uqff"; printf 'uqff' > "$out/qtip2-1.uqff"
    exit 0 ;;
  serve)
    touch "$SB/serve_up"
    echo "Server listening"
    case "$*" in *--mtp-depth*)
      echo "MTP speculative decode engaged (depth=2)"
      echo "MTP acceptance rate: 61.3% (392/640 accepted)"
      echo "MTP acceptance rate: 62.0% (794/1280 accepted)" ;;
    esac
    sleep 900 ;;
  calibrate)
    [ "${FAKE_CALIB_FAIL:-0}" = "1" ] && { echo "CUDA out of memory"; exit 1; }
    out=""; prev=""
    for a in "$@"; do case "$prev" in --out) out=$a;; esac; prev=$a; done
    printf 'ARCCALIB' > "$out"
    echo "Calibration complete"
    exit 0 ;;
esac
exit 0
EOF
  # -------- python shim for the quality scripts
  cat > "$SB/bin/fakepy" <<'EOF'
#!/usr/bin/env bash
script=$(basename "${1:-}")
case "$script" in
  run_coherence.py)   echo "COHERENCE: 6/6"; exit 0 ;;
  speed_probe.py)     echo "SPEED[s6]: decode 14.9 tok/s | prefill ~58 tok/s"; exit 0 ;;
  batch_load_probe.py)
    echo "BATCH[B=1] agg 14.9 tok/s | per-user p50 14.9 tok/s | TTFT p50 0.9s | errors 0/2"
    echo "BATCH[B=64] agg 980.0 tok/s | per-user p50 15.3 tok/s | TTFT p50 3.1s | errors 0/128"
    echo "BATCHSWEEP[s6]: peak 980.0 tok/s @B=64 | per-user p50 @peak 15.3 tok/s | \$/Mtok @peak 1.39"
    exit 0 ;;
  run_gsm8k.py)       echo "GSM8K[chat] greedy n=100: 87.0% (87/100)"; exit 0 ;;
  s6_upload_uqff.py)
    # must never echo the token, even though it is handed a token file
    echo "UPLOAD_PLAN files=3 shards=2 bytes=68000000000"
    echo "UPLOAD_AUTH ok as aeonmind"
    echo "UPLOAD_OK repo=aeonmind/DeepSeek-V4-Flash-UQFF-qtip2 files=3 shards=2 bytes=68000000000"
    exit 0 ;;
esac
exit 0
EOF
  chmod +x "$SB/bin/"*
}

run_driver() {   # run_driver <logfile> [extra env assignments...]
  local out=$1; shift
  env -i PATH="$SB/bin:/usr/bin:/bin:/usr/sbin:/sbin" HOME="$SB" TMPDIR="${TMPDIR:-/tmp}" \
    S6_DRYRUN=1 S6_SKIP_BOOT=0 \
    S6_ROOT="$SB/root" LOG_DIR="$SB/root/logs" STATUS_DIR="$SB/status" \
    WORKROOT="$SB" ARC="$SB/arc" V4_DIR="$SB/models/V4" \
    Q="$SB/arc/arc-tools/quality" RESULTS="$SB/arc/arc-tools/quality/results" \
    UQFF_DIR="$SB/models/V4/uqff" \
    S6_CARGO="$SB/bin/cargo" S6_PY="$SB/bin/fakepy" \
    S6_MISTRALRS="$SB/bin/mistralrs" S6_BOOT="$SB/bin/boot.sh" \
    S6_HEALTH_GATE="$SB/bin/box_health_gate.sh" S6_SENTINEL="/nonexistent" \
    S6_UPLOAD_PY="$SB/root/s6_upload_uqff.py" S6_TOKEN_FILE="$SB/root/.hf_token" \
    CHAT_TEMPLATE="$SB/arc/chat_templates/deepseek_v4.json" \
    "$@" bash "$DRIVER" > "$out" 2>&1
  echo $? > "$SB/driver_rc"
  rm -f "$SB/serve_up"
}

cleanup() { [ -n "${SB:-}" ] && [ "${KEEP:-0}" != "1" ] && rm -rf "$SB"; return 0; }
trap cleanup EXIT

ONLY=${S6TEST_ONLY:-}
scen() { [ -z "$ONLY" ] && return 0; case ",$ONLY," in *",$1,"*) return 0;; *) return 1;; esac; }

echo "=== s6_driver dry run ==="

# ---------------------------------------------------------------- A: happy
if scen A; then
make_sandbox
run_driver "$SB/A.log"
A=$SB/A.log
assert_has   "A: health gate passed"            "$A" '^HEALTH_PASS'
assert_has   "A: build ok"                      "$A" '^BUILD_OK'
assert_has   "A: beam gate at W=256"            "$A" '^BEAM_GATE=W256'
assert_has   "A: bake header printed"           "$A" '^BAKE_HEADER: QTIP bake'
assert_has   "A: bake search verified"          "$A" '^BAKE_SEARCH_OK W=256'
assert_has   "A: rotation verified"             "$A" '^BAKE_ROTATION_OK'
assert_has   "A: bake completed"                "$A" '^BAKE_OK bytes='
assert_has   "A: upload ok"                     "$A" 'UPLOAD_OK repo=aeonmind/'
assert_has   "A: batch sweep ran"               "$A" '^BATCHSWEEP\['
assert_has   "A: gsm8k ran"                     "$A" '^GSM8K\['
assert_has   "A: gemv winners exported"         "$A" '^WINNERS: '
assert_has   "A: calibration spread reported"   "$A" '^CALIB_SPREAD .*dynamic_range=1181'
assert_has   "A: D11 verdict rendered"          "$A" '^CALIB_VERDICT .*D11 stands'
assert_has   "A: mtp acceptance captured"       "$A" '^MTP_ACCEPT .*acceptance rate'
assert_has   "A: results tarball"               "$A" '^RESULTS_TGZ: '
assert_has   "A: token deleted"                 "$A" '^TOKEN_DELETED'
assert_has   "A: completed"                     "$A" '^S6_COMPLETE'
assert_not   "A: no abort"                      "$A" '^ABORT_'
assert_before "A: UPLOAD precedes the batch sweep" "$A" 'UPLOAD_OK' 'BATCHSWEEP\['
assert_before "A: UPLOAD precedes GSM8K"           "$A" 'UPLOAD_OK' 'GSM8K\['
assert_not   "A: token never reaches the log"   "$A" "$SECRET"
[ "$(cat "$SB/bake_singlethread")" = "1" ] \
  && pass "A: bake ran with MISTRALRS_ISQ_SINGLETHREAD=1" \
  || fail "A: bake singlethread env (got '$(cat "$SB/bake_singlethread")')"
cleanup
fi

# ---------------------------------------------------------------- B: bad box
if scen B; then
make_sandbox
run_driver "$SB/B.log" FAKE_HEALTH=1 FAKE_HEALTH_VERDICT=FAIL
B=$SB/B.log
assert_has "B: aborts on health FAIL"     "$B" '^ABORT_BOX_HEALTH'
assert_has "B: tells you to re-rent"      "$B" 'DELETE THIS INSTANCE AND RE-RENT'
assert_not "B: never starts bootstrap"    "$B" ':::::: S1 bootstrap'
assert_not "B: never bakes"               "$B" ':::::: S3 bake'
cleanup
fi

# ------------------------------------------------------- C: vacuous-pass guard
if scen C; then
make_sandbox
run_driver "$SB/C.log" FAKE_CC=7.5
C=$SB/C.log
assert_has "C: aborts on compute_cap < 8" "$C" '^ABORT_BOX_COMPUTE_CAP'
assert_not "C: never runs parity"         "$C" ':::::: S2 beam-parity'
cleanup
fi

# ------------------------------------------------- D: beam broken at every width
if scen D; then
make_sandbox
run_driver "$SB/D.log" FAKE_BEAM_FAIL=64
D=$SB/D.log
assert_has "D: aborts on beam parity"        "$D" '^ABORT_BEAM_PARITY'
assert_has "D: names the failing width"      "$D" '^BEAM_FAILING_WIDTH=64'
assert_has "D: refuses the exhaustive bake"  "$D" 'do NOT fall back to a ~6.2 h exhaustive bake'
assert_not "D: never bakes"                  "$D" '^BAKE_OK'
cleanup
fi

# ------------------------------------------------- E: W=256 bad, W=128 good
if scen E; then
make_sandbox
run_driver "$SB/E.log" FAKE_BEAM_FAIL=256
E=$SB/E.log
assert_has "E: degrades to W=128"          "$E" '^BEAM_GATE=W128 \(DEGRADED'
assert_has "E: bakes anyway"               "$E" '^BAKE_OK'
assert_has "E: flags the quality delta"    "$E" 'FLAG THIS IN THE RESULTS'
[ "$(cat "$SB/bake_beam_width")" = "128" ] \
  && pass "E: the bake really received ARC_QTIP_BEAM=128" \
  || fail "E: bake got ARC_QTIP_BEAM='$(cat "$SB/bake_beam_width")'"
cleanup
fi

# ------------------------------------------------- F: no bake header
if scen F; then
make_sandbox
run_driver "$SB/F.log" FAKE_BAKE_HEADER=none
F=$SB/F.log
assert_has "F: aborts without a bake header" "$F" '^ABORT_BAKE_HEADER'
assert_not "F: never uploads"                "$F" 'UPLOAD_OK'
cleanup
fi

# ------------------------------------------------- G: greedy header (D4)
if scen G; then
make_sandbox
run_driver "$SB/G.log" FAKE_BAKE_HEADER=greedy
G=$SB/G.log
assert_has "G: aborts on a greedy bake"   "$G" '^ABORT_BAKE_GREEDY'
assert_has "G: cites DOCTRINE D4"         "$G" 'D4: greedy is banned forever'
assert_not "G: never uploads"             "$G" 'UPLOAD_OK'
cleanup
fi

echo
if [ "$FAILURES" -eq 0 ]; then
  echo "=== s6_driver dry run: ALL PASS ==="
  exit 0
fi
echo "=== s6_driver dry run: $FAILURES FAILURE(S) ==="
exit 1
