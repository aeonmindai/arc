#!/bin/bash
# Fused head_dim=512 sinks kernel vs the unfused path it replaces.
#
# ONE BINARY, ONE VARIABLE: ARC_FLASH_512. Both arms are the same build; the
# flag only drops 512 from the CUDA allow-list, so the control is exactly what
# V4 runs today (unfused matmul + softmax_with_sinks).
#
# ⚠️ ARM ASSERTION. The mode is logged once per process (target 'arcflash'):
# FUSED / UNFUSED. Both strings are in the binary, so only the RUNTIME line
# proves the arms differ. No line, or the same mode twice -> VOID, exit 2.
#
# ⚠️ MARGINAL, NOT ABSOLUTE. The published baseline — attention 11.0 ms of the
# 19.4 ms marginal cost per additional sequence — is a SLOPE. An absolute
# mla_attn time is not comparable to it. So each arm is measured at TWO
# concurrencies and the per-sequence marginal is the difference; comparing an
# absolute against a slope would be the same category error as comparing arms
# that are not different builds.
set -u
L=/root/logs/arcflash; mkdir -p $L; S=$L/STATUS.txt; : > $S
BIN=${BIN:-/root/arc/target/release/mistralrs}
SRC=/root/models/v4-src; UQFF=/root/models/v4-uqff/qtip2b-0.uqff; PORT=1238
LO=${LO:-1}; HI=${HI:-8}; TOK=${TOK:-16}; REPS=${REPS:-3}
export CUDA_HOME=/usr/local/cuda-13.1
export PATH="$CUDA_HOME/bin:/root/.cargo/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/compat:${LD_LIBRARY_PATH:-}"
say(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a $S; }
PID=""
cleanup(){ [ -n "$PID" ] && kill $PID 2>/dev/null; sleep 3; [ -n "$PID" ] && kill -9 $PID 2>/dev/null; pkill -f "mistralrs serve -p $PORT" 2>/dev/null; return 0; }
trap cleanup EXIT INT TERM

OCC=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' | grep -c .)
[ "${OCC:-0}" = "0" ] || { say "ENV_FAIL: $OCC foreign compute process(es) on the GPU"; say "RESULT: UNANSWERED"; exit 2; }
[ -x "$BIN" ] || { say "ENV_FAIL: no binary at $BIN"; say "RESULT: UNANSWERED"; exit 2; }
grep -qa "ARC_FLASH_512" $BIN || { say "ENV_FAIL: binary predates the ARC_FLASH_512 gate"; say "RESULT: UNANSWERED"; exit 2; }
say "BIN=$BIN mtime=$(date -u -r $BIN +%H:%M:%S)"

# Fire N concurrent completions, return wall seconds for the batch.
burst(){ local n=$1 t0 t1 i
  t0=$(date +%s.%N)
  for i in $(seq 1 "$n"); do
    curl -s --max-time 900 localhost:$PORT/v1/completions -H 'Content-Type: application/json' \
      -d "{\"model\":\"default\",\"prompt\":\"Count from one to twenty in words, then stop. [$i]\",\"max_tokens\":$TOK,\"temperature\":0}" \
      >$L/r_$i.json 2>&1 &
  done
  wait
  t1=$(date +%s.%N)
  python3 -c "print(f'{$t1-$t0:.3f}')"
}

arm(){ # $1 name, $2 env
  local n=$1 extra=$2
  local log=$L/$n.server.log
  say "=== arm $n (${extra:-<default: FUSED>}) ==="
  cd /root/arc || return 1
  # shellcheck disable=SC2086
  env $extra RUST_LOG=info ARC_PROFILE=1 ARC_PROFILE_OUT=$L/prof_$n ARC_PROFILE_LABEL=$n \
      ARC_NO_DEDICATED_DECODE=1 \
      $BIN serve -p $PORT -m $SRC -a deepseekv4 --from-uqff $UQFF \
      --chat-template chat_templates/deepseek_v4.json \
      --prefix-cache-n 0 --paged-attn off --max-seqs $HI --max-seq-len 4096 >$log 2>&1 &
  PID=$!
  local ok=0
  for _ in $(seq 1 90); do curl -fsS --max-time 5 localhost:$PORT/health >/dev/null 2>&1 && { ok=1; break; }; kill -0 $PID 2>/dev/null || break; sleep 5; done
  [ $ok = 1 ] || { say "  server never healthy"; tail -3 $log | sed 's/^/    /' | tee -a $S; cleanup; PID=""; return 1; }

  # ⚠️ MODE IS SAMPLED **AFTER** THE BURSTS, NOT BEFORE.
  # `flash_512_enabled()` is a OnceLock initialised on the FIRST ATTENTION
  # FORWARD, so the line does not exist until a request has run. Sampling it
  # right after the health check — which runs no forward — read NONE/NONE and
  # voided a run whose arms were in fact correctly UNFUSED and FUSED. The
  # assertion failed safe, but for my bug rather than a real one.
  local wlo whi r
  # REPEATS. A single pair cannot distinguish a real effect from run-to-run
  # noise, and the previous run showed an 11% gap between arms before their
  # modes were even confirmed. Best-of-N at each point; best-of is the right
  # statistic for a floor measurement (noise adds time, it does not remove it).
  wlo=""; whi=""
  for r in $(seq 1 "$REPS"); do
    local a b
    a=$(burst $LO); b=$(burst $HI)
    say "    rep$r: B=$LO ${a}s  B=$HI ${b}s"
    wlo=$(python3 -c "print(min([x for x in ['$wlo','$a'] if x]))")
    whi=$(python3 -c "print(min([x for x in ['$whi','$b'] if x]))")
  done

  local mode="NONE"
  grep -q "sinks path is FUSED"   $log && mode="FUSED"
  grep -q "sinks path is UNFUSED" $log && mode="UNFUSED"
  # marginal seconds per additional sequence, over TOK tokens each
  local marg
  marg=$(python3 -c "print(f'{($whi-$wlo)/max($HI-$LO,1):.4f}')")
  say "  mode=$mode  B=$LO wall=${wlo}s  B=$HI wall=${whi}s  marginal=${marg}s/seq"
  echo "$mode" >$L/$n.mode; echo "$wlo" >$L/$n.lo; echo "$whi" >$L/$n.hi; echo "$marg" >$L/$n.marg
  cleanup; PID=""; sleep 4; return 0
}

arm unfused "ARC_FLASH_512=0" || say "unfused arm incomplete"
arm fused   ""                || say "fused arm incomplete"

say "=== VERDICT ==="
um=$(cat $L/unfused.mode 2>/dev/null||echo NA); fm=$(cat $L/fused.mode 2>/dev/null||echo NA)
say "unfused: mode=$um marginal=$(cat $L/unfused.marg 2>/dev/null||echo NA)s/seq (B=$LO $(cat $L/unfused.lo 2>/dev/null||echo NA)s, B=$HI $(cat $L/unfused.hi 2>/dev/null||echo NA)s)"
say "fused  : mode=$fm marginal=$(cat $L/fused.marg 2>/dev/null||echo NA)s/seq (B=$LO $(cat $L/fused.lo 2>/dev/null||echo NA)s, B=$HI $(cat $L/fused.hi 2>/dev/null||echo NA)s)"
if [ "$um" != "UNFUSED" ] || [ "$fm" != "FUSED" ]; then
  say "🔴 ARCFLASH A/B IS VOID — expected unfused=UNFUSED and fused=FUSED, got $um / $fm."
  say "   The arms are not demonstrably different behaviour, so no comparison between"
  say "   them means anything. Most likely a flag-name or log-level problem."
  say "RESULT: UNANSWERED"; exit 2
fi
say "✅ arms differ: unfused=UNFUSED, fused=FUSED"
python3 - "$L" <<'PY' | tee -a $S
import sys
L=sys.argv[1]
g=lambda f: open(f"{L}/{f}").read().strip()
try:
    u=float(g("unfused.marg")); f=float(g("fused.marg"))
except Exception as e:
    print(f"  could not read marginals: {e}"); raise SystemExit
if u<=0: print(f"  unfused marginal is {u:.4f}s — non-positive, so the ratio is meaningless. UNPROVEN."); raise SystemExit
print(f"  marginal per additional sequence: unfused {u:.4f}s -> fused {f:.4f}s  ({u/f if f>0 else float('inf'):.3f}x)")
print("  ⚠️ wall-clock marginal, NOT the profiler's device-time mla_attn cell.")
print("     Comparable in SHAPE to the 11.0ms-of-19.4ms figure (both slopes),")
print("     but not the same instrument. Treat as directional until read from")
print("     the profiler's mla_attn span.")
PY
say "RESULT: COMPLETE"
