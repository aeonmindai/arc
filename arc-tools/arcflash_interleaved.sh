#!/bin/bash
# Fused-512 vs unfused, INTERLEAVED and PAIRED.
#
# WHY. The sequential design (U U U then F F F) puts every bit of drift between
# the two windows straight into the effect. Measured: the null-control floor was
# 0.67% in one run and 2.45% in another — a 3.7x swing — and device and wall
# floors tracked each other WITHIN a run (2.45% vs 2.54%). So the variance is in
# the SYSTEM (clocks, power, placement), not the instrument, and no sharper
# instrument can remove it. A null control tells you the floor for THAT RUN, not
# the floor.
#
# THE FIX: alternate arms and compare each measurement against its temporal
# neighbours, so a slow-clock excursion hits both arms nearly equally and
# cancels. This converts between-run variance (uncontrollable) into within-pair
# variance (small).
#
#   sequence:  U F U N U F U N
#              ^ each F and N is bracketed by U's taken minutes either side
#
# The N (null) arms are unfused-vs-unfused under identical interleaved
# conditions, so the floor is MEASURED in the same design as the effect rather
# than inferred from a separate run.
#
# REPORTED: mean of the paired differences, and the SPREAD of those differences
# — which is the correct floor for a paired design.
set -u

L=/root/logs/arcflash-inter; mkdir -p $L; S=$L/STATUS.txt; : > $S
BIN=${BIN:-/root/arc/target/release/mistralrs}
SRC=/root/models/v4-src; UQFF=/root/models/v4-uqff/qtip2b-0.uqff; PORT=1245
TOK=${TOK:-320}; STEPS=${STEPS:-64}
SEQ=${SEQ:-"U F U N U F U N"}
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
say "BIN=$BIN mtime=$(date -u -r $BIN +%H:%M:%S)  TOK=$TOK  sequence: $SEQ"
: > $L/samples.txt

one(){ # $1 slot index, $2 kind (U|F|N)
  local i=$1 kind=$2 extra="" tag
  case "$kind" in
    U|N) extra="ARC_FLASH_512=0" ;;
    F)   extra="" ;;
  esac
  tag="${i}_${kind}"
  local log=$L/$tag.server.log pout=$L/prof_$tag
  rm -rf "$pout"; mkdir -p "$pout"
  cd /root/arc || return 1
  # shellcheck disable=SC2086
  env $extra RUST_LOG=info ARC_PROFILE=1 ARC_PROFILE_OUT=$pout ARC_PROFILE_LABEL=$tag \
      ARC_PROFILE_STEPS=$STEPS ARC_NO_DEDICATED_DECODE=1 \
      $BIN serve -p $PORT -m $SRC -a deepseekv4 --from-uqff $UQFF \
      --chat-template chat_templates/deepseek_v4.json \
      --prefix-cache-n 0 --paged-attn off --max-seqs 1 --max-seq-len 4096 >$log 2>&1 &
  PID=$!
  local ok=0
  for _ in $(seq 1 90); do curl -fsS --max-time 5 localhost:$PORT/health >/dev/null 2>&1 && { ok=1; break; }; kill -0 $PID 2>/dev/null || break; sleep 5; done
  [ $ok = 1 ] || { say "  slot $tag: server never healthy"; cleanup; PID=""; return 1; }
  curl -s --max-time 900 localhost:$PORT/v1/completions -H 'Content-Type: application/json' \
    -d "{\"model\":\"default\",\"prompt\":\"Write a long detailed description of a city, continuing without stopping.\",\"max_tokens\":$TOK,\"temperature\":0}" >/dev/null 2>&1
  local mode="NONE"
  grep -q "sinks path is FUSED"   $log && mode="FUSED"
  grep -q "sinks path is UNFUSED" $log && mode="UNFUSED"
  cleanup; PID=""; sleep 3
  python3 - "$pout" "$L/samples.txt" "$i" "$kind" "$mode" <<'PY'
import json, sys, glob, os
pout, out, i, kind, mode = sys.argv[1:6]
best = None
for f in sorted(glob.glob(os.path.join(pout, "*.json"))):
    try: d = json.load(open(f))
    except Exception: continue
    nodes = d.get("nodes", [])
    cand = [n for n in nodes if n.get("name") == "mla_attn"
            and "step.decode" in n.get("path","") and n.get("calls",0) > 0]
    if not cand: continue
    n = max(cand, key=lambda x: x.get("calls",0))
    dark = not any(x.get("device_ns",0) > 0 for x in nodes)
    best = (n.get("device_ns",0), n.get("wall_ns",0), n.get("calls",0), dark,
            bool(n.get("reachable",False)))
if best is None:
    open(out,"a").write(f"{i} {kind} {mode} NA NA 0 1 0\n")
else:
    dev, wall, calls, dark, reach = best
    open(out,"a").write(f"{i} {kind} {mode} {dev} {wall} {calls} {int(dark)} {int(reach)}\n")
PY
  local line; line=$(tail -1 $L/samples.txt)
  say "  slot $tag: mode=$mode  $(echo "$line" | awk '{printf "device_ns=%s calls=%s dark=%s reach=%s", $4,$6,$7,$8}')"
  return 0
}

i=0
for kind in $SEQ; do
  i=$((i+1))
  say "=== slot $i: $kind ==="
  one "$i" "$kind" || say "  slot $i incomplete"
done

say "=== PAIRED VERDICT ==="
python3 - "$L/samples.txt" <<'PY' | tee -a $S
import sys
rows=[]
for l in open(sys.argv[1]):
    p=l.split()
    if len(p)<8: continue
    rows.append(dict(i=int(p[0]), kind=p[1], mode=p[2], dev=p[3], wall=p[4],
                     calls=int(p[5]), dark=int(p[6]), reach=int(p[7])))
if not rows: print("  VOID: no samples."); print("RESULT: UNANSWERED"); raise SystemExit
# guards
bad=[r for r in rows if r["calls"]==0 or not r["reach"] or r["dev"]=="NA"]
if bad:
    print(f"  VOID: {len(bad)} slot(s) with calls=0 / unreachable / no record.")
    print("RESULT: UNANSWERED"); raise SystemExit
for r in rows:
    want = "FUSED" if r["kind"]=="F" else "UNFUSED"
    if r["mode"]!=want:
        print(f"  VOID: slot {r['i']} kind={r['kind']} reported mode={r['mode']}, expected {want}.")
        print("RESULT: UNANSWERED"); raise SystemExit
dark = any(r["dark"] for r in rows)
val = lambda r: (float(r["wall"]) if dark else float(r["dev"]))/r["calls"]/1000.0
unit = "SPAN WALL (device timer dark)" if dark else "DEVICE (CUDA events)"
by = {r["i"]: r for r in rows}
print(f"  metric: mla_attn us/invocation — {unit}")
for r in rows: print(f"    slot {r['i']} {r['kind']}  {val(r):8.2f} us")
def paired(kind):
    out=[]
    for r in rows:
        if r["kind"]!=kind: continue
        nb=[by[j] for j in (r["i"]-1, r["i"]+1) if j in by and by[j]["kind"]=="U"]
        if not nb: continue
        base=sum(val(x) for x in nb)/len(nb)
        out.append(val(r)-base)
    return out
eff=paired("F"); nul=paired("N")
def stat(v):
    if not v: return None
    m=sum(v)/len(v); sp=max(v)-min(v)
    return m, sp, len(v)
se, sn = stat(eff), stat(nul)
if not se or not sn:
    print("  VOID: not enough bracketed pairs."); print("RESULT: UNANSWERED"); raise SystemExit
me, spe, ne = se; mn, spn, nn = sn
ubar = sum(val(r) for r in rows if r["kind"]=="U")/len([r for r in rows if r["kind"]=="U"])
print(f"  paired FUSED-minus-neighbouring-UNFUSED : mean {me:+.2f} us ({100*me/ubar:+.2f}%), spread {spe:.2f} us, n={ne}")
print(f"  paired NULL -minus-neighbouring-UNFUSED : mean {mn:+.2f} us ({100*mn/ubar:+.2f}%), spread {spn:.2f} us, n={nn}")
floor=max(abs(mn), spn)
print(f"  PAIRED FLOOR (max of |null mean|, null spread) = {floor:.2f} us ({100*floor/ubar:.2f}%)")
margin = abs(me)/floor if floor>0 else float("inf")
print(f"  EFFECT = {abs(me):.2f} us ({100*abs(me)/ubar:.2f}%) {'SLOWER' if me>0 else 'FASTER'}, margin {margin:.1f}x over the floor")
if margin < 2.0:
    print(f"  ==> MARGIN UNDER 2x. Direction is indicative, SIZE IS NOT ESTABLISHED.")
    print("      Reported as such, not as a win or a loss.")
else:
    print(f"  ==> RESOLVED WITH MARGIN: {'slower' if me>0 else 'faster'} by "
          f"{100*abs(me)/ubar:.2f}%, {margin:.1f}x the paired floor.")
PY
say "RESULT: COMPLETE"
