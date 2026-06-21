#!/bin/bash
# RUN-161 decode-bug tandem differential trace orchestrator.
# Assumes a V4 server is (or will be) listening on $PORT, started with:
#   V4_TRACE=/tmp/tr_work V4_TRACE_LAYER=0 ... serve ... --prefix-cache-n 0 --paged-attn off
# Captures decode (position N) vs from-scratch prefill (position N) per-op dumps and diffs them.
set -u
PORT="${PORT:-1234}"
WORK="${WORK:-/tmp/tr_work}"
DIFF="${DIFF:-/root/work/arc/v4_trace_diff.py}"
PROMPT="${PROMPT:-Counting numbers in order: one, two, three, four, five, six, seven,}"

jpost() { # $1 = json body
  curl -s "localhost:$PORT/v1/completions" -H 'Content-Type: application/json' -d "$1"
}

echo "=== waiting for server on :$PORT ==="
for i in $(seq 1 900); do
  if curl -s "localhost:$PORT/v1/models" >/dev/null 2>&1; then echo "server up"; break; fi
  sleep 2
done

# REQUEST 1: get t1 (the exact first generated token's text) via max_tokens=1.
rm -f "$WORK"/*.npy "$WORK"/*.txt 2>/dev/null
R1=$(jpost "{\"model\":\"default\",\"prompt\":$(python3 -c 'import json,os;print(json.dumps(os.environ["PROMPT"]))' PROMPT="$PROMPT"),\"max_tokens\":1,\"temperature\":0}")
echo "$R1" > /root/logs/r1.json
T1=$(python3 -c "import json;print(json.load(open('/root/logs/r1.json'))['choices'][0]['text'])")
echo "=== t1 (first gen token text) = [$T1] ==="

# REQUEST 2 (DECODE): max_tokens=2 -> last forward is the decode step at position N (token t1).
rm -f "$WORK"/*.npy "$WORK"/*.txt 2>/dev/null
R2=$(jpost "{\"model\":\"default\",\"prompt\":$(python3 -c 'import json,os;print(json.dumps(os.environ["PROMPT"]))' PROMPT="$PROMPT"),\"max_tokens\":2,\"temperature\":0}")
echo "$R2" > /root/logs/r2.json
echo "decode 2-tok text: [$(python3 -c "import json;print(json.load(open('/root/logs/r2.json'))['choices'][0]['text'])")]"
rm -rf /tmp/tr_decode; cp -r "$WORK" /tmp/tr_decode

# REQUEST 3 (REFERENCE prefill): prompt = P + t1, max_tokens=1 -> single prefill forward,
# last row = position N (token t1). prefix-cache disabled => true from-scratch prefill.
rm -f "$WORK"/*.npy "$WORK"/*.txt 2>/dev/null
REFP="$PROMPT$T1"
R3=$(jpost "{\"model\":\"default\",\"prompt\":$(REFP="$REFP" python3 -c 'import json,os;print(json.dumps(os.environ["REFP"]))'),\"max_tokens\":1,\"temperature\":0}")
echo "$R3" > /root/logs/r3.json
rm -rf /tmp/tr_prefill; cp -r "$WORK" /tmp/tr_prefill

echo "=== DIFF (prefill vs decode, both at position N) ==="
python3 "$DIFF" /tmp/tr_prefill /tmp/tr_decode
