#!/bin/bash
# ArcGate — deliverable 1: NO SILENT FAILURES.
#
# The bug this exists to catch: ~1 request in 3 returns ZERO tokens and the
# engine reports SUCCESS. No speed number is publishable over that.
#
# PASS is defined as: N consecutive requests each return tokens, proven by TWO
# independent facts that must agree, and every request lands in exactly one of
# the ok/failed lists.
#
# ⚠️ WHY TWO COUNTERS. `usage.completion_tokens` is the server's own
# bookkeeping — the very bookkeeping a silent failure corrupts. A row that
# claims 64 tokens and delivers "" is the bug, not a pass. So each row records
#   (a) usage.completion_tokens   — what the engine says it did
#   (b) len(content)              — what the client actually received
# and DISAGREEMENT (one zero, the other not) is itself a FAIL, not a rounding
# difference. A gate that trusted (a) alone would have graded the bug green.
#
# ⚠️ ROW ACCOUNTING IS ASSERTED, NOT ASSUMED. A recorded house failure had a
# zero-token row appear in NEITHER the ok nor the failed list, so the summary
# read `rows_failed: []` beside `total_tokens: 0`. Here ok+failed must equal
# the number of requests attempted or the gate exits 2 as a broken instrument
# — it will not print a verdict it cannot support.
#
# EXIT CODES (house rule: environment failure is 2, never 1)
#   0  all N requests returned tokens, both counters agreed          -> SHIPPABLE
#   1  at least one request returned zero tokens                     -> THE BUG
#   2  environment failure / broken instrument / cannot answer       -> NO VERDICT
#
# USAGE
#   PORT=1234 N=100 arc-tools/no_silent_failures_gate.sh
#   BASE=http://127.0.0.1:1234 N=100 MODE=both arc-tools/no_silent_failures_gate.sh
#
# ENV
#   BASE      base URL                     (default http://127.0.0.1:$PORT)
#   PORT      port                         (default 1234)
#   N         consecutive requests         (default 100)
#   MAXTOK    max_tokens per request       (default 32)
#   MODE      nostream | stream | both     (default both)
#   MODEL     model name for the payload   (default the first id from /v1/models)
#   LOGDIR    where rows land              (default /root/logs/nosilent)
set -u

PORT=${PORT:-1234}
BASE=${BASE:-http://127.0.0.1:$PORT}
N=${N:-100}
MAXTOK=${MAXTOK:-32}
MODE=${MODE:-both}
LOGDIR=${LOGDIR:-/root/logs/nosilent}

mkdir -p "$LOGDIR" || { echo "ENV_FAIL: cannot create $LOGDIR" >&2; exit 2; }
S=$LOGDIR/STATUS.txt; : > "$S"
say(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$S"; }

command -v curl   >/dev/null 2>&1 || { say "ENV_FAIL: no curl";    say "RESULT: UNANSWERED"; exit 2; }
command -v python3 >/dev/null 2>&1 || { say "ENV_FAIL: no python3"; say "RESULT: UNANSWERED"; exit 2; }

# ---- environment: the server must be up and serving a model -----------------
MODELS=$(curl -sS --max-time 20 "$BASE/v1/models" 2>>"$S") || {
  say "ENV_FAIL: $BASE/v1/models unreachable"; say "RESULT: UNANSWERED"; exit 2; }
MODEL=${MODEL:-$(printf '%s' "$MODELS" | python3 -c \
  'import json,sys
try:
    d=json.load(sys.stdin); print(d["data"][0]["id"])
except Exception:
    print("")' 2>>"$S")}
[ -n "$MODEL" ] || { say "ENV_FAIL: no model id in /v1/models -> $MODELS"; say "RESULT: UNANSWERED"; exit 2; }
say "gate: BASE=$BASE MODEL=$MODEL N=$N MAXTOK=$MAXTOK MODE=$MODE"

# ---- one request = one row, graded on TWO independent facts -----------------
# Row format: seq<TAB>arm<TAB>http<TAB>usage_tok<TAB>content_len<TAB>finish<TAB>secs
ROWS=$LOGDIR/rows.tsv; : > "$ROWS"
ATTEMPTED=0

req_nostream(){ # $1 seq
  local seq=$1 body t0 t1 http
  # Vary the prompt so a prefix/radix cache cannot mask a per-request failure
  # with a replayed answer, and so run-to-run nondeterminism is exercised.
  body=$(python3 -c "
import json,sys
print(json.dumps({'model':sys.argv[1],
 'messages':[{'role':'user','content':'Count from %s to %s, one number per line.'%(sys.argv[2],int(sys.argv[2])+5)}],
 'max_tokens':int(sys.argv[3]),'temperature':0.0,'stream':False}))" "$MODEL" "$seq" "$MAXTOK")
  t0=$(python3 -c 'import time;print(time.time())')
  http=$(curl -sS --max-time 300 -o "$LOGDIR/ns.$seq.json" -w '%{http_code}' \
        -H 'Content-Type: application/json' -d "$body" \
        "$BASE/v1/chat/completions" 2>>"$S") || http=000
  t1=$(python3 -c 'import time;print(time.time())')
  python3 - "$LOGDIR/ns.$seq.json" "$seq" "$http" "$t0" "$t1" >>"$ROWS" <<'PY'
import json,sys
path,seq,http,t0,t1=sys.argv[1:6]
tok,clen,fin=-1,-1,"NONE"
try:
    d=json.load(open(path))
    tok=int((d.get("usage") or {}).get("completion_tokens", -1))
    ch=(d.get("choices") or [])
    if ch:
        clen=len(((ch[0].get("message") or {}).get("content") or ""))
        fin=str(ch[0].get("finish_reason"))
    else:
        clen=0; fin="NO_CHOICES"
except Exception as e:
    fin="UNPARSEABLE:%s"%type(e).__name__
print("%s\tnostream\t%s\t%d\t%d\t%s\t%.3f"%(seq,http,tok,clen,fin,float(t1)-float(t0)))
PY
}

req_stream(){ # $1 seq — a 200 that streams zero deltas is the same bug wearing SSE
  local seq=$1 body t0 t1 http
  body=$(python3 -c "
import json,sys
print(json.dumps({'model':sys.argv[1],
 'messages':[{'role':'user','content':'List %s animals, one per line.'%sys.argv[2]}],
 'max_tokens':int(sys.argv[3]),'temperature':0.0,'stream':True}))" "$MODEL" "$seq" "$MAXTOK")
  t0=$(python3 -c 'import time;print(time.time())')
  http=$(curl -sS --max-time 300 -o "$LOGDIR/st.$seq.sse" -w '%{http_code}' \
        -H 'Content-Type: application/json' -d "$body" \
        "$BASE/v1/chat/completions" 2>>"$S") || http=000
  t1=$(python3 -c 'import time;print(time.time())')
  python3 - "$LOGDIR/st.$seq.sse" "$seq" "$http" "$t0" "$t1" >>"$ROWS" <<'PY'
import json,sys
path,seq,http,t0,t1=sys.argv[1:6]
deltas,clen,fin,saw_done=0,0,"NONE",False
try:
    for line in open(path,errors="replace"):
        if not line.startswith("data:"): continue
        p=line[5:].strip()
        if p=="[DONE]": saw_done=True; continue
        try: d=json.loads(p)
        except Exception: continue
        for c in (d.get("choices") or []):
            t=((c.get("delta") or {}).get("content") or "")
            if t: deltas+=1; clen+=len(t)
            if c.get("finish_reason"): fin=str(c["finish_reason"])
except Exception as e:
    fin="UNPARSEABLE:%s"%type(e).__name__
if not saw_done and fin=="NONE": fin="NO_DONE"
# deltas is the stream's own token counter; clen is what the client received
print("%s\tstream\t%s\t%d\t%d\t%s\t%.3f"%(seq,http,deltas,clen,fin,float(t1)-float(t0)))
PY
}

say "--- firing $N consecutive requests (mode=$MODE) ---"
i=1
while [ "$i" -le "$N" ]; do
  case "$MODE" in
    nostream) req_nostream "$i"; ATTEMPTED=$((ATTEMPTED+1));;
    stream)   req_stream   "$i"; ATTEMPTED=$((ATTEMPTED+1));;
    both)     req_nostream "$i"; req_stream "$i"; ATTEMPTED=$((ATTEMPTED+2));;
    *) say "ENV_FAIL: bad MODE=$MODE"; say "RESULT: UNANSWERED"; exit 2;;
  esac
  [ $((i % 10)) -eq 0 ] && say "  ... $i/$N"
  i=$((i+1))
done

# ---- grade. The verdict is refused unless the rows account for every request -
python3 - "$ROWS" "$ATTEMPTED" <<'PY' | tee -a "$S"
import collections,sys
rows_path,attempted = sys.argv[1], int(sys.argv[2])

rows=[]
for ln in open(rows_path):
    ln=ln.rstrip("\n")
    if not ln: continue
    f=ln.split("\t")
    if len(f)!=7:
        print("BROKEN INSTRUMENT: malformed row %r"%ln); print("RESULT: UNANSWERED"); sys.exit(2)
    rows.append({"seq":f[0],"arm":f[1],"http":f[2],"tok":int(f[3]),
                 "clen":int(f[4]),"fin":f[5],"secs":float(f[6])})

# (1) every request attempted must have produced exactly one row.
if len(rows)!=attempted:
    print("BROKEN INSTRUMENT: %d rows for %d requests attempted -- rows were lost, "
          "so neither a pass nor a fail can be supported."%(len(rows),attempted))
    print("RESULT: UNANSWERED"); sys.exit(2)
if attempted==0:
    print("BROKEN INSTRUMENT: zero requests attempted"); print("RESULT: UNANSWERED"); sys.exit(2)

ok,failed=[],[]
for r in rows:
    why=[]
    if r["http"]!="200":        why.append("http=%s"%r["http"])
    if r["tok"]<=0:             why.append("engine reported %d tokens"%r["tok"])
    if r["clen"]<=0:            why.append("client received %d chars"%r["clen"])
    # the two counters disagreeing is its own fault class: the engine's
    # bookkeeping and the delivered bytes tell different stories.
    if (r["tok"]>0) != (r["clen"]>0):
        why.append("COUNTER DISAGREEMENT tok=%d clen=%d"%(r["tok"],r["clen"]))
    (failed if why else ok).append((r,"; ".join(why)))

# (2) ok and failed must partition the rows -- the recorded house failure was a
#     zero-token row that appeared in neither list.
if len(ok)+len(failed)!=len(rows):
    print("BROKEN INSTRUMENT: ok(%d)+failed(%d) != rows(%d)"%(len(ok),len(failed),len(rows)))
    print("RESULT: UNANSWERED"); sys.exit(2)

total=sum(r["tok"] for r,_ in ok)
print("--- rows=%d  ok=%d  failed=%d  (ok+failed=%d == attempted=%d) ---"
      %(len(rows),len(ok),len(failed),len(ok)+len(failed),attempted))
for arm in sorted({r["arm"] for r in rows}):
    a=[r for r in rows if r["arm"]==arm]
    toks=sorted(r["tok"] for r in a)
    z=sum(1 for r in a if r["tok"]<=0 or r["clen"]<=0)
    print("  arm=%-8s n=%-4d zero=%-4d tok min/med/max=%s/%s/%s  median_s=%.2f"
          %(arm,len(a),z,toks[0],toks[len(toks)//2],toks[-1],
            sorted(r["secs"] for r in a)[len(a)//2]))
print("  finish_reason: %s"%dict(collections.Counter(r["fin"] for r in rows)))
# positive engagement counter: a green that cannot show work is not a green.
print("  TOTAL TOKENS DELIVERED = %d"%total)
if total<=0:
    print("ZERO TOKENS ACROSS EVERY REQUEST -- the engine produced nothing.")
    print("RESULT: FAIL (silent-failure bug at 100%)"); sys.exit(1)

if failed:
    print("--- FAILED ROWS (%d/%d = %.1f%%) ---"%(len(failed),len(rows),100.0*len(failed)/len(rows)))
    for r,why in failed[:25]:
        print("  seq=%s arm=%s %s (finish=%s)"%(r["seq"],r["arm"],why,r["fin"]))
    if len(failed)>25: print("  ... and %d more"%(len(failed)-25))
    print("RESULT: FAIL -- %d of %d requests returned no tokens while the "
          "transport reported success."%(len(failed),len(rows)))
    sys.exit(1)

print("RESULT: PASS -- %d/%d consecutive requests returned tokens, both counters "
      "agreeing, %d tokens delivered."%(len(ok),len(rows),total))
sys.exit(0)
PY
RC=${PIPESTATUS[0]}
say "gate exit=$RC (0=pass 1=silent-failure bug 2=no verdict)"
exit "$RC"
