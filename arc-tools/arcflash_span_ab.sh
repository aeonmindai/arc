#!/bin/bash
# Fused head_dim=512 sinks kernel vs the unfused path — ON THE PROFILER SPAN,
# in a regime where attention is actually doing its work.
#
# WHY THIS SUPERSEDES THE WALL-CLOCK VERSION. That run answered "below the noise
# floor" for two reasons, and only one was the instrument:
#   * instrument — end-to-end wall time, where attention is a small share;
#   * CONDITIONS — B=8 with 16 tokens. V4's sliding window is 128, so every
#     query attended over at most ~16 keys: an eighth of the window the kernel
#     is built to stream, and a regime where BC=8's 16-sequential-tile cost
#     cannot even arise because there are not 128 positions to tile.
# The published baseline has mla_attn at 58.0 ms of a 113.4 ms step — half the
# step. In that configuration it was a rounding error. Fixing one and not the
# other just buys an unresolvable answer from a sharper instrument.
#
# SO: profiler device time (CUDA events) on `mla_attn`, at >= 256 generated
# tokens so the 128-key window fills and steady-state decode dominates.
#
# PRIMARY METRIC: device_ns / calls for the `mla_attn` node = mean GPU time per
# attention invocation. That is precisely what the kernel swap changes, and it
# is the published baseline's own span.
#
# GUARDS, all carried from failures earlier tonight:
#   * arm assertion on the RUNTIME FUSED/UNFUSED line, sampled AFTER traffic
#     (it is a OnceLock initialised on the first forward);
#   * `reachable` on the mla_attn node must be true — the profiler distinguishes
#     "zero because fast" from "zero because never ran", and so must this;
#   * calls > 0 in both arms;
#   * a null control (unfused vs unfused) to MEASURE the floor rather than
#     assume it.
set -u

L=/root/logs/arcflash-span; mkdir -p $L; S=$L/STATUS.txt; : > $S
BIN=${BIN:-/root/arc/target/release/mistralrs}
SRC=/root/models/v4-src; UQFF=/root/models/v4-uqff/qtip2b-0.uqff; PORT=1239
TOK=${TOK:-320}          # >= 256 so the 128-key window fills and then some
CONC=${CONC:-1}          # per-invocation device time; concurrency not needed
STEPS=${STEPS:-64}       # profiler auto-writes after this many recorded steps
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
say "BIN=$BIN mtime=$(date -u -r $BIN +%H:%M:%S)  TOK=$TOK  window=128"

arm(){ # $1 name, $2 env
  local n=$1 extra=$2
  local log=$L/$n.server.log
  local pout=$L/prof_$n
  rm -rf "$pout"; mkdir -p "$pout"
  say "=== arm $n (${extra:-<default: FUSED>}) ==="
  cd /root/arc || return 1
  # shellcheck disable=SC2086
  env $extra RUST_LOG=info \
      ARC_PROFILE=1 ARC_PROFILE_OUT=$pout ARC_PROFILE_LABEL=$n ARC_PROFILE_STEPS=$STEPS \
      ARC_NO_DEDICATED_DECODE=1 \
      $BIN serve -p $PORT -m $SRC -a deepseekv4 --from-uqff $UQFF \
      --chat-template chat_templates/deepseek_v4.json \
      --prefix-cache-n 0 --paged-attn off --max-seqs $CONC --max-seq-len 4096 >$log 2>&1 &
  PID=$!
  local ok=0
  for _ in $(seq 1 90); do curl -fsS --max-time 5 localhost:$PORT/health >/dev/null 2>&1 && { ok=1; break; }; kill -0 $PID 2>/dev/null || break; sleep 5; done
  [ $ok = 1 ] || { say "  server never healthy"; tail -3 $log | sed 's/^/    /' | tee -a $S; cleanup; PID=""; return 1; }

  say "  generating $TOK tokens (fills the 128-key window ~$((TOK/128))x over)"
  curl -s --max-time 1800 localhost:$PORT/v1/completions -H 'Content-Type: application/json' \
    -d "{\"model\":\"default\",\"prompt\":\"Write a long detailed description of a city, continuing without stopping.\",\"max_tokens\":$TOK,\"temperature\":0}" \
    >$L/$n.resp.json 2>&1
  local tok; tok=$(python3 -c "import json;print(json.load(open('$L/$n.resp.json')).get('usage',{}).get('completion_tokens',0))" 2>/dev/null||echo 0)

  # Mode AFTER traffic: the OnceLock line only exists once a forward has run.
  local mode="NONE"
  grep -q "sinks path is FUSED"   $log && mode="FUSED"
  grep -q "sinks path is UNFUSED" $log && mode="UNFUSED"

  cleanup; PID=""; sleep 3

  # Pull mla_attn out of the profiler JSON.
  python3 - "$pout" "$L/$n" "$mode" "$tok" <<'PY'
import json, sys, glob, os
pout, stem, mode, tok = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
js = sorted(glob.glob(os.path.join(pout, "*.json")))
out = {"mode": mode, "tokens": tok, "device_ns": 0, "calls": 0, "reachable": False, "found": False}
for f in js:
    try:
        d = json.load(open(f))
    except Exception:
        continue
    for n in d.get("nodes", []):
        if n.get("name") == "mla_attn":
            out.update(device_ns=n.get("device_ns", 0), calls=n.get("calls", 0),
                       reachable=bool(n.get("reachable", False)), found=True,
                       note=n.get("note"))
            break
open(stem + ".mla", "w").write(json.dumps(out))
print(json.dumps(out))
PY
  local d; d=$(cat $L/$n.mla 2>/dev/null || echo '{}')
  say "  mode=$mode tokens=$tok  mla_attn=$d"
  return 0
}

arm unfused "ARC_FLASH_512=0" || say "unfused arm incomplete"
arm nullctl "ARC_FLASH_512=0" || say "null-control arm incomplete"
arm fused   ""                || say "fused arm incomplete"

say "=== VERDICT ==="
python3 - "$L" <<'PY' | tee -a $S
import json, sys
L = sys.argv[1]
def load(n):
    try: return json.load(open(f"{L}/{n}.mla"))
    except Exception: return None
u, c, f = load("unfused"), load("nullctl"), load("fused")
if not all([u, c, f]):
    print("  VOID: one or more arms produced no mla_attn record."); print("RESULT: UNANSWERED"); raise SystemExit
# 1. arms must differ
if u["mode"] != "UNFUSED" or c["mode"] != "UNFUSED" or f["mode"] != "FUSED":
    print(f"  VOID: modes are unfused={u['mode']} null={c['mode']} fused={f['mode']}.")
    print("  The arms are not demonstrably different behaviour; no comparison means anything.")
    print("RESULT: UNANSWERED"); raise SystemExit
# 2. the node must have been REACHED and CALLED
for name, d in (("unfused", u), ("nullctl", c), ("fused", f)):
    if not d["found"] or not d["reachable"] or d["calls"] == 0:
        print(f"  VOID: {name} mla_attn found={d['found']} reachable={d['reachable']} calls={d['calls']}.")
        print("  A zero from an unreached node is not a fast node. UNPROVEN.")
        print("RESULT: UNANSWERED"); raise SystemExit
per = lambda d: d["device_ns"] / d["calls"] / 1000.0   # microseconds per invocation
pu, pc, pf = per(u), per(c), per(f)
floor = abs(pu - pc); eff = abs(pu - pf)
fp = 100 * floor / pu if pu else 0.0
ep = 100 * eff / pu if pu else 0.0
print(f"  mla_attn device time per invocation (CUDA events):")
print(f"    unfused {pu:9.2f} us   (calls={u['calls']}, tokens={u['tokens']})")
print(f"    null    {pc:9.2f} us   (same flag as unfused)")
print(f"    fused   {pf:9.2f} us   (calls={f['calls']}, tokens={f['tokens']})")
print(f"  MEASURED NOISE FLOOR = {floor:.2f} us ({fp:.1f}%)")
print(f"  FUSED-VS-UNFUSED EFFECT = {eff:.2f} us ({ep:.1f}%)  {'FASTER' if pf < pu else 'SLOWER'}")
if eff <= floor:
    print(f"  ==> BELOW THE NOISE FLOOR ({ep:.1f}% <= {fp:.1f}%). Not 'no effect' —")
    print("      an effect this instrument cannot resolve even on device time.")
else:
    verdict = "FASTER" if pf < pu else "SLOWER"
    print(f"  ==> RESOLVED: fused is {ep:.1f}% {verdict} than unfused, clearing a {fp:.1f}% floor.")
    if pf > pu:
        print("      A slower fused kernel is a REAL RESULT, not a failure: BC=8 gives only")
        print("      8 KV positions per tile, so the 128-key window costs 16 sequential")
        print("      tiles. The reserve lever is bf16 tiles -> BC=16 at the same 32 KB.")
PY
say "RESULT: COMPLETE"
