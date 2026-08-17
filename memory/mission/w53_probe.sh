#!/usr/bin/env bash
# wave53-CD — V4 PagedAttention A/B. Unattended. One log, one result file.
#
#   OFF run: shipped default (supports_paged_attention == false)
#   ON  run: ARC_V4_PAGED_ATTN=1
#
# Acceptance: byte-identical greedy completions for the same 3 prompts.
# Any difference means the KV path is wrong, not merely slower.
set -uo pipefail

LOG=/root/w53.log
RES=/root/w53_result.txt
exec >>"$LOG" 2>&1

step() { echo; echo "===== $* ====="; date -u; }
fail() { echo "FAIL: $*"; echo "W53_DONE"; exit 1; }
ok()   { echo "OK: $*"; }

# Biggest writable mount wins; these boxes vary.
for c in /ephemeral /workspace /mnt /root; do
  [ -d "$c" ] || continue
  a=$(df -BG --output=avail "$c" 2>/dev/null | tail -1 | tr -dc '0-9')
  [ -n "${a:-}" ] || continue
  if [ "${a:-0}" -gt "${BEST_AVAIL:-0}" ]; then BEST_AVAIL=$a; W=$c; fi
done
W=${W:-/root}
echo "workdir=$W avail=${BEST_AVAIL}G"
mkdir -p "$W/arc" "$W/uqff" "$W/src"

step "0 sanity"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || fail "nvidia-smi"
echo "nproc=$(nproc)"
[ "${BEST_AVAIL:-0}" -ge 300 ] || fail "disk ${BEST_AVAIL}G < 300G; 234 GB of downloads will not fit"

export HF_TOKEN=$(cat /root/.hf_token)
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

step "1 toolchain"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq --no-install-recommends pkg-config libssl-dev build-essential python3-pip curl git || fail "apt"
command -v cargo >/dev/null || curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --profile minimal || fail "rustup"
. "$HOME/.cargo/env"
pip install -q --break-system-packages huggingface_hub 2>/dev/null || pip install -q huggingface_hub || fail "pip hf"
CUDA_GUESS=$(ls -d /usr/local/cuda-* 2>/dev/null | sort -V | tail -1)
export CUDA_HOME=${CUDA_HOME:-${CUDA_GUESS:-/usr/local/cuda}}
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
nvcc --version | tail -2 || fail "nvcc missing at $CUDA_HOME"
ok "toolchain"

step "2 clone the probe branch"
if [ ! -d "$W/arc/.git" ]; then
  git clone --depth=5 --branch feat/v4-paged-attn-probe \
    https://github.com/aeonmindai/arc.git "$W/arc" || fail "clone"
fi
cd "$W/arc" || fail "cd"
git log -1 --oneline
grep -q "ARC_V4_PAGED_ATTN" mistralrs-core/src/pipeline/loaders/normal_loaders.rs \
  || fail "binary would not carry the opt-in — wrong branch"
ok "branch carries the opt-in"

step "3 downloads (background) + build (foreground)"
# huggingface_hub renamed `huggingface-cli` to `hf`; accept either.
if command -v hf >/dev/null; then HFC="hf"; else HFC="huggingface-cli"; fi
command -v "$HFC" >/dev/null || fail "no huggingface downloader on PATH"
echo "downloader=$HFC"
# NOTE: do NOT add `cudnn` — measured -62% decode on V4 (session 4).
( $HFC download aeonmind/DeepSeek-V4-Flash-UQFF-qtip2b --local-dir "$W/uqff" --max-workers 16 \
    >/root/dl_uqff.log 2>&1; echo "UQFF_EXIT=$?" >>/root/dl_uqff.log ) &
DL1=$!
( $HFC download deepseek-ai/DeepSeek-V4-Flash --local-dir "$W/src" --max-workers 16 \
    >/root/dl_src.log 2>&1; echo "SRC_EXIT=$?" >>/root/dl_src.log ) &
DL2=$!
cargo build --release -p mistralrs-cli --features "cuda flash-attn" 2>&1 | tail -20
BIN="$W/arc/target/release/mistralrs"
[ -x "$BIN" ] || fail "cargo build — no binary"
ok "built $(date -u)"
wait $DL1 $DL2
tail -2 /root/dl_uqff.log; tail -2 /root/dl_src.log
NSH=$(ls "$W"/uqff/qtip2b-*.uqff 2>/dev/null | wc -l)
NSRC=$(ls "$W"/src/model-*.safetensors 2>/dev/null | wc -l)
echo "uqff shards=$NSH residual=$(ls -l "$W"/uqff/residual.safetensors 2>/dev/null | awk '{print $5}') src shards=$NSRC"
[ "$NSH" -eq 8 ] || fail "expected 8 uqff shards, got $NSH"
[ -f "$W/uqff/residual.safetensors" ] || fail "residual.safetensors missing"
[ "$NSRC" -ge 40 ] || fail "source checkpoint incomplete ($NSRC shards)"
ok "artifacts complete"

# ---------------------------------------------------------------- the A/B ---
cat >/root/gen.py <<'PY'
import json, sys, urllib.request, urllib.error, time
PROMPTS = [
    "What is the capital of France? Answer in one short sentence.",
    "Write exactly one sentence about why memory bandwidth matters for LLM inference.",
    "List the first five prime numbers, separated by commas.",
]
tag = sys.argv[1]
out = []
for i, p in enumerate(PROMPTS):
    body = json.dumps({
        "model": "default",
        "messages": [{"role": "user", "content": p}],
        "max_tokens": 48,
        "temperature": 0.0,
        "top_p": 1.0,
        "stream": False,
    }).encode()
    req = urllib.request.Request("http://127.0.0.1:1234/v1/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    try:
        r = json.load(urllib.request.urlopen(req, timeout=1800))
        txt = r["choices"][0]["message"]["content"]
        fin = r["choices"][0].get("finish_reason")
        ntok = r.get("usage", {}).get("completion_tokens")
    except urllib.error.HTTPError as e:
        txt, fin, ntok = "HTTPERROR " + e.read().decode()[:2000], "error", -1
    except Exception as e:
        txt, fin, ntok = "EXC " + repr(e)[:2000], "error", -1
    out.append({"i": i, "prompt": p, "text": txt, "finish": fin, "completion_tokens": ntok})
    print(f"[{tag}] p{i} tok={ntok} fin={fin} :: {txt!r}", flush=True)
json.dump(out, open(f"/root/gen_{tag}.json", "w"), indent=1)
PY

serve_and_gen() {  # $1=tag  $2..=extra serve args ; env passed by caller
  local tag=$1; shift
  step "SERVE [$tag]  extra_args: $*"
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
  ARC_NO_DEDICATED_DECODE=1 \
  "$BIN" serve -p 1234 \
    -m "$W/src" \
    -a deepseekv4 \
    --from-uqff "$W/uqff/qtip2b-0.uqff" \
    --chat-template chat_templates/deepseek_v4.json \
    --max-seqs 1 \
    --prefix-cache-n 0 \
    --max-seq-len 4096 \
    "$@" >"/root/serve_$tag.log" 2>&1 &
  local pid=$!
  local up=0
  for _ in $(seq 1 240); do   # 40 min: cold load measured ~3 min, build cache cold
    sleep 10
    if ! kill -0 $pid 2>/dev/null; then
      echo "SERVER [$tag] DIED. Last 60 lines:"; tail -60 "/root/serve_$tag.log"; return 1
    fi
    # /v1/models, not /health: the router answers /health as soon as it binds,
    # which is before the pipeline has finished loading the artifact.
    if curl -s -m 5 http://127.0.0.1:1234/v1/models | grep -q '"data"'; then up=1; break; fi
  done
  [ "$up" = 1 ] || { echo "SERVER [$tag] never became healthy. Last 60:"; tail -60 "/root/serve_$tag.log"; kill $pid 2>/dev/null; return 1; }
  echo "--- [$tag] cache_config / paged / graph evidence ---"
  grep -inE "paged|cache_config|cache engine|blocks|ARC_V4_PAGED_ATTN|autonomous|graph" "/root/serve_$tag.log" | head -40
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
  python3 /root/gen.py "$tag"
  echo "--- [$tag] server log tail after generation ---"
  tail -40 "/root/serve_$tag.log"
  grep -inE "autonomous_decode|no cache_config|falling back|not captured|dsv4|precheck|panic|Error" "/root/serve_$tag.log" | tail -40
  kill $pid 2>/dev/null; wait $pid 2>/dev/null
  sleep 20
  return 0
}

step "4 RUN A — flag OFF (shipped default)"
serve_and_gen off --paged-attn off || echo "RUN A FAILED"

step "5 RUN B — flag ON (ARC_V4_PAGED_ATTN=1)"
export ARC_V4_PAGED_ATTN=1
serve_and_gen on --paged-attn on --pa-cache-type auto --pa-memory-mb 2048 || echo "RUN B FAILED"
unset ARC_V4_PAGED_ATTN

step "6 VERDICT"
{
  echo "=== wave53-CD V4 PagedAttention A/B ==="
  date -u
  echo "commit: $(cd "$W/arc" && git log -1 --oneline)"
  if [ -f /root/gen_off.json ] && [ -f /root/gen_on.json ]; then
    python3 - <<'PY'
import json
a = json.load(open("/root/gen_off.json")); b = json.load(open("/root/gen_on.json"))
same = True
for x, y in zip(a, b):
    m = x["text"] == y["text"]
    same &= m
    print(f"p{x['i']}: identical={m} off_tok={x['completion_tokens']} on_tok={y['completion_tokens']}")
    if not m:
        print("  OFF:", repr(x["text"])[:600])
        print("  ON :", repr(y["text"])[:600])
print("TOKENS_IDENTICAL=" + ("YES" if same else "NO"))
PY
  else
    echo "TOKENS_IDENTICAL=INCONCLUSIVE (one or both runs produced no output)"
    ls -l /root/gen_*.json 2>/dev/null
  fi
  echo "--- serve_on.log paged/cache evidence ---"
  grep -inE "paged|cache_config|blocks|GPU KV|autonomous|captur" /root/serve_on.log 2>/dev/null | head -30
  echo "--- serve_on.log errors ---"
  grep -inE "error|panic|bail|refuse|mismatch" /root/serve_on.log 2>/dev/null | head -30
} | tee "$RES"

echo "W53_DONE"
