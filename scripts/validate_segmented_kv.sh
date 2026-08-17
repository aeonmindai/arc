#!/usr/bin/env bash
# ==============================================================================
# validate_segmented_kv.sh — GPU validation for PR #90 (segmented KV allocator)
#
# THE CLAIM UNDER TEST, in priority order:
#   1. PRIMARY  — output identity. With ARC_SEGMENTED_KV=1 the model must emit
#                 BYTE-IDENTICAL completions to the control. The degenerate
#                 1-segment path is supposed to be the same allocation, the same
#                 block table, the same slot mapping. If text differs, the design
#                 is wrong at the root and the tok/s number is meaningless.
#   2. SECONDARY— decode throughput ratio B/A >= 0.99.
#
# Runs unattended. setsid-safe. Writes append-only progress to $STATUS_FILE
# (tail -f it) and a machine-readable verdict to $RESULT_FILE.
#
#   setsid nohup ./scripts/validate_segmented_kv.sh > /root/segkv.boot.log 2>&1 &
#   tail -f /root/segkv/status.log
#
# Exit codes: 0 = PASS, 1 = FAIL (claim refuted), 2 = ERROR (could not test).
# A refuted claim and a broken harness must never look the same.
# ==============================================================================
set -Eeuo pipefail

# ------------------------------------------------------------------ tunables --
ARC_REPO="${ARC_REPO:-/root/arc}"
ARC_BRANCH="${ARC_BRANCH:-feat/segment-kv-allocator}"
# Small DENSE model. V4 is NOT needed and NOT wanted here: this validates the
# degenerate 1-segment path, which every dense model exercises, and V4's 149 GB
# download is the expensive part. Qwen2ForCausalLM is auto-detected by
# `NormalLoaderType` and does NOT override `supports_paged_attention`, so paged
# attention genuinely engages (the only `false` overrides are DeepSeek-V4,
# GraniteMoeHybrid and Qwen3Next).
MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
PORT="${PORT:-18234}"
OUT_DIR="${OUT_DIR:-/root/segkv}"
FEATURES="${FEATURES:-cuda flash-attn}"   # NEVER add cudnn: -62% decode on V4.
SKIP_BUILD="${SKIP_BUILD:-0}"
IDENTITY_MAX_TOKENS="${IDENTITY_MAX_TOKENS:-192}"
THROUGHPUT_CONCURRENCY="${THROUGHPUT_CONCURRENCY:-32}"
THROUGHPUT_MAX_TOKENS="${THROUGHPUT_MAX_TOKENS:-256}"
SERVER_BOOT_TIMEOUT="${SERVER_BOOT_TIMEOUT:-1800}"   # includes model download

STATUS_FILE="$OUT_DIR/status.log"
RESULT_FILE="$OUT_DIR/result.json"

mkdir -p "$OUT_DIR"
: > "$STATUS_FILE"

say() { printf '[%s] %s\n' "$(date -u +%H:%M:%SZ)" "$*" | tee -a "$STATUS_FILE"; }

SERVER_PID=""
ORIG_REF=""
cleanup() {
  local rc=$?
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    say "cleanup: stopping server pid $SERVER_PID"
    kill "$SERVER_PID" 2>/dev/null || true
    for _ in $(seq 1 30); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 1; done
    kill -9 "$SERVER_PID" 2>/dev/null || true
  fi
  if [[ -n "$ORIG_REF" ]]; then
    say "cleanup: restoring $ARC_REPO to $ORIG_REF"
    git -C "$ARC_REPO" checkout -q "$ORIG_REF" 2>>"$STATUS_FILE" || \
      say "WARNING: could not restore $ORIG_REF — leave the box for a human"
  fi
  say "exit rc=$rc"
}
trap cleanup EXIT

die_err() { say "ERROR: $*"; emit_verdict "ERROR" "$*"; exit 2; }

emit_verdict() {  # $1 verdict, $2 reason  (overwritten by the python driver on success)
  python3 - "$RESULT_FILE" "$1" "$2" <<'PY'
import json, sys
json.dump({"verdict": sys.argv[2], "reason": sys.argv[3]}, open(sys.argv[1], "w"), indent=2)
PY
}

# ------------------------------------------------------------- 0. preflight --
say "=== phase 0: preflight ==="

# BOX TRAP: nvcc is not on PATH on this image; cudarc's build script shells out
# to `nvcc --version` and panics with "`nvcc --version` failed" if it is missing.
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
if ! command -v nvcc >/dev/null 2>&1; then
  for c in /usr/local/cuda-13.1 /usr/local/cuda-13.0 /usr/local/cuda-12.8 /usr/local/cuda; do
    if [[ -x "$c/bin/nvcc" ]]; then
      export CUDA_HOME="$c"; export PATH="$c/bin:$PATH"; break
    fi
  done
fi
command -v nvcc >/dev/null 2>&1 || die_err "nvcc not found; set CUDA_HOME (tried /usr/local/cuda-13.1 etc.)"
say "nvcc: $(command -v nvcc) — $(nvcc --version | tail -1)"
say "CUDA_HOME=$CUDA_HOME"

command -v nvidia-smi >/dev/null 2>&1 || die_err "nvidia-smi not found — this must run on the GPU box"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | tee -a "$STATUS_FILE"

# python3 stdlib only. The image ships pip 22.0.2 with no numpy, so the driver
# below uses urllib + json + concurrent.futures and nothing else.
command -v python3 >/dev/null 2>&1 || die_err "python3 not found"
python3 -c 'import urllib.request, json, concurrent.futures' 2>/dev/null \
  || die_err "python3 stdlib incomplete (urllib/json/concurrent.futures)"

[[ -d "$ARC_REPO/.git" ]] || die_err "no git repo at ARC_REPO=$ARC_REPO"

# Refuse to yank the tree out from under a running benchmark.
if pgrep -f 'mistralrs|arc-bench|cargo (build|test|run)' >/dev/null 2>&1; then
  say "processes currently holding the repo:"; pgrep -af 'mistralrs|arc-bench|cargo' | tee -a "$STATUS_FILE"
  die_err "another build/bench is running in this checkout — wait for it, or set ARC_REPO to a separate clone"
fi

# --------------------------------------------------------------- 1. checkout --
say "=== phase 1: checkout $ARC_BRANCH ==="
ORIG_REF="$(git -C "$ARC_REPO" symbolic-ref --quiet --short HEAD || git -C "$ARC_REPO" rev-parse HEAD)"
say "current ref: $ORIG_REF (will be restored on exit)"
git -C "$ARC_REPO" fetch -q origin "$ARC_BRANCH" || die_err "git fetch failed"
git -C "$ARC_REPO" checkout -q -B "$ARC_BRANCH" "origin/$ARC_BRANCH" || die_err "git checkout failed"
say "HEAD: $(git -C "$ARC_REPO" log --oneline -1)"

# ------------------------------------------------------------------ 2. build --
# NOTE ON REUSING THE EXISTING BOX BUILD: the compiled *binary* from another
# branch CANNOT be reused — it has no ARC_SEGMENTED_KV at all, so both phases
# would run the control and report a perfect fake pass. What IS reused is this
# checkout's target/ dir and cargo registry, which is where nearly all of the
# build time lives (deps + the .cu kernels). Checking out in place, rather than
# into a side worktree, is deliberate: a different path busts cargo's
# fingerprints for every workspace crate including mistralrs-paged-attn's CUDA
# build script.
say "=== phase 2: build (features: $FEATURES; cudnn deliberately excluded) ==="
BIN=""
if [[ "$SKIP_BUILD" == "1" ]]; then
  say "SKIP_BUILD=1 — using existing binary"
else
  # `-p mistralrs-cli` is deliberate. It is the package that produces the
  # `mistralrs` binary (mistralrs-cli/Cargo.toml [[bin]] name = "mistralrs")
  # and it owns the `cuda` / `flash-attn` feature forwarding. A bare
  # `cargo build --release --features ...` at the virtual workspace root also
  # works, but drags in mistralrs-pyo3 / web-chat / bench, which are not needed
  # here and are extra ways for the build to fail.
  ( cd "$ARC_REPO" && cargo build --release -p mistralrs-cli --features "$FEATURES" ) \
    >>"$OUT_DIR/build.log" 2>&1 || { tail -40 "$OUT_DIR/build.log" | tee -a "$STATUS_FILE"; die_err "cargo build failed — see $OUT_DIR/build.log"; }
  say "build ok"
fi

for cand in "$ARC_REPO/target/release/mistralrs" "$ARC_REPO/target/release/arc" "$ARC_REPO/target/release/mistralrs-server"; do
  [[ -x "$cand" ]] && { BIN="$cand"; break; }
done
[[ -n "$BIN" ]] || die_err "no server binary in $ARC_REPO/target/release (looked for mistralrs, arc, mistralrs-server)"
say "binary: $BIN"

# ------------------------------------------------- 3. anti-fake-pass guard 1 --
# If the binary predates this PR it silently ignores ARC_SEGMENTED_KV, the two
# phases become the same run, and the script would "prove" a perfect 1.00x with
# identical output. Refuse before wasting GPU time.
say "=== phase 3: binary provenance guard ==="
if ! strings -a "$BIN" 2>/dev/null | grep -q 'ARC_SEGMENTED_KV'; then
  die_err "binary does not contain the string ARC_SEGMENTED_KV — it was built without PR #90. \
A run with this binary would produce a FAKE PASS. Rebuild on $ARC_BRANCH."
fi
say "guard 1 ok: ARC_SEGMENTED_KV present in binary"

# ------------------------------------------------------------ server helpers --
start_server() {  # $1 = tag ("control"|"segmented"), $2 = 1 to enable the flag
  local tag="$1" seg="$2" log="$OUT_DIR/server.$1.log"
  : > "$log"
  say "starting server [$tag] (ARC_SEGMENTED_KV=${seg:-unset}) -> $log"
  if [[ "$seg" == "1" ]]; then export ARC_SEGMENTED_KV=1; else unset ARC_SEGMENTED_KV; fi
  # Paged attention is explicitly requested: this PR only touches the paged
  # allocator, so a run that silently fell back to the non-paged path would
  # validate nothing.
  # `--paged-attn` is a value-enum (auto|on|off), not a bare flag: `on` forces
  # it and hard-errors if the device cannot support it, which is what we want —
  # a silent fallback to the non-paged path would validate nothing.
  # `--seed` is a GLOBAL option (mistralrs-cli/src/args/mod.rs:396). There is NO
  # per-request `seed` field in ChatCompletionRequest, so it must be set here.
  ( cd "$ARC_REPO" && exec "$BIN" serve -p "$PORT" --paged-attn on --seed 1234 -m "$MODEL" ) >>"$log" 2>&1 &
  SERVER_PID=$!
  unset ARC_SEGMENTED_KV
  local waited=0
  # A bare TCP connect succeeds as soon as the listener binds — before weights
  # are loaded. `/health` (router line 225) only answers once the server is
  # actually serving, so poll that.
  until python3 - "$PORT" <<'PY' 2>/dev/null
import sys, urllib.request
try:
    with urllib.request.urlopen(f"http://127.0.0.1:{sys.argv[1]}/health", timeout=5) as r:
        sys.exit(0 if r.status == 200 else 1)
except Exception:
    sys.exit(1)
PY
  do
    kill -0 "$SERVER_PID" 2>/dev/null || { tail -40 "$log" | tee -a "$STATUS_FILE"; die_err "server [$tag] died during boot — see $log"; }
    waited=$((waited + 5)); sleep 5
    (( waited % 60 == 0 )) && say "  ... waiting for [$tag] (${waited}s; first run includes the model download)"
    (( waited >= SERVER_BOOT_TIMEOUT )) && die_err "server [$tag] did not open :$PORT within ${SERVER_BOOT_TIMEOUT}s"
  done
  say "server [$tag] up after ${waited}s (pid $SERVER_PID)"
}

stop_server() {
  [[ -z "$SERVER_PID" ]] && return 0
  kill "$SERVER_PID" 2>/dev/null || true
  for _ in $(seq 1 60); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 1; done
  kill -9 "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
  SERVER_PID=""
  say "server stopped"
}

run_phase() {  # $1 = tag
  say "  driving workload [$1]"
  MODEL="$MODEL" PORT="$PORT" OUT_DIR="$OUT_DIR" TAG="$1" \
  IDENTITY_MAX_TOKENS="$IDENTITY_MAX_TOKENS" \
  THROUGHPUT_CONCURRENCY="$THROUGHPUT_CONCURRENCY" \
  THROUGHPUT_MAX_TOKENS="$THROUGHPUT_MAX_TOKENS" \
  python3 "$OUT_DIR/driver.py" >>"$STATUS_FILE" 2>&1 \
    || die_err "workload driver failed for [$1] — see $STATUS_FILE"
}

# ------------------------------------------------------------ the driver.py --
cat > "$OUT_DIR/driver.py" <<'PYDRIVER'
"""Workload driver. python3 stdlib only (no numpy, no requests, no jq)."""
import concurrent.futures as cf
import json, os, time, urllib.request, urllib.error

PORT = os.environ["PORT"]; MODEL = os.environ["MODEL"]
TAG  = os.environ["TAG"];  OUT   = os.environ["OUT_DIR"]
IDENT_MAX = int(os.environ["IDENTITY_MAX_TOKENS"])
CONC      = int(os.environ["THROUGHPUT_CONCURRENCY"])
TP_MAX    = int(os.environ["THROUGHPUT_MAX_TOKENS"])
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"

# A long shared prefix, so the block-hash prefix cache is exercised (the
# touch/evict path is where the degenerate allocator is most likely to differ
# from the flat one), followed by distinct suffixes so the completions differ.
PREFIX = ("You are a precise technical assistant. Answer in complete sentences. "
          "Be concrete and avoid hedging. Do not use bullet points. ") * 6
SUFFIXES = [
    "Explain what a page table is in an operating system.",
    "Describe how a hash map resolves collisions.",
    "Summarise why floating point addition is not associative.",
    "Explain the difference between a mutex and a semaphore.",
    "Describe what a write-ahead log does in a database.",
    "Explain how a bloom filter can produce false positives.",
]

def post(body, timeout=1200):
    req = urllib.request.Request(
        URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())

def chat(prompt, max_tokens):
    # Determinism comes from ARGMAX, not from a seed: `Sampler` maps
    # temperature < 1e-7 to `None`, and `sample()` then takes the pure
    # `logits.argmax` path with no RNG at all (sampler.rs:1224-1225).
    # NOTE: ChatCompletionRequest has NO `seed` field — a per-request seed would
    # be silently dropped. The process seed is pinned via the server's global
    # `--seed` instead. `top_k: 1` is belt-and-braces only; the greedy path
    # short-circuits before top-k is consulted.
    return post({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 1,
        "stream": False,
    })

def content(resp):
    return resp["choices"][0]["message"]["content"]

def completion_tokens(resp):
    return (resp.get("usage") or {}).get("completion_tokens", 0)

# ---- warmup: never time a cold cache, and never let warmup pollute identity --
chat("Warm up.", 8)

# ---- PHASE 1: identity. STRICTLY SEQUENTIAL, one request in flight. ----------
# This matters. Run these concurrently and the batch composition differs between
# phases, the reduction order in the attention kernels changes with it, and
# greedy decoding can legitimately diverge — a false FAIL that looks like a real
# one. Batch-of-one removes that variable, so any difference here is the
# allocator.
identity = []
for i, suf in enumerate(SUFFIXES):
    r = chat(PREFIX + suf, IDENT_MAX)
    identity.append({"i": i, "prompt_suffix": suf,
                     "text": content(r), "completion_tokens": completion_tokens(r)})
    print(f"    identity[{i}] {completion_tokens(r)} tok", flush=True)

# ---- PHASE 2: throughput. Concurrent, this is the number we compare. ---------
def one(i):
    t0 = time.perf_counter()
    r = chat(f"Write a detailed paragraph about topic number {i}.", TP_MAX)
    return completion_tokens(r), time.perf_counter() - t0

t0 = time.perf_counter()
with cf.ThreadPoolExecutor(max_workers=CONC) as ex:
    results = list(ex.map(one, range(CONC)))
wall = time.perf_counter() - t0
total_tokens = sum(t for t, _ in results)

out = {
    "tag": TAG,
    "model": MODEL,
    "identity": identity,
    "throughput": {
        "concurrency": CONC,
        "requested_max_tokens": TP_MAX,
        "completion_tokens_total": total_tokens,
        "wall_seconds": wall,
        "tokens_per_second": (total_tokens / wall) if wall > 0 else 0.0,
    },
}
with open(os.path.join(OUT, f"phase.{TAG}.json"), "w") as f:
    json.dump(out, f, indent=2)
print(f"    [{TAG}] {total_tokens} tok in {wall:.2f}s = {total_tokens/wall:.2f} tok/s", flush=True)
PYDRIVER

# ------------------------------------------------------------ 4. phase A/B ----
say "=== phase 4: control run (flag OFF) ==="
start_server control ""
run_phase control
stop_server

say "=== phase 5: segmented run (ARC_SEGMENTED_KV=1) ==="
start_server segmented "1"
run_phase segmented
stop_server

# ------------------------------------------------- 6. anti-fake-pass guard 2 --
# The binary contains the flag (guard 1) — but did it actually take effect?
# KVCacheManager::with_segmentation logs this line exactly once when the
# segmented backing is constructed. Assert it is present in B and ABSENT in A.
# Without this, a flag that silently failed to parse would look like a pass.
say "=== phase 6: flag-took-effect guard ==="
grep -q 'KV cache backing: SEGMENTED allocator' "$OUT_DIR/server.segmented.log" \
  || die_err "segmented run did not log the SEGMENTED backing line — the flag did NOT take effect. \
This would have been a fake pass. Check $OUT_DIR/server.segmented.log."
if grep -q 'KV cache backing: SEGMENTED allocator' "$OUT_DIR/server.control.log"; then
  die_err "control run ALSO logged the SEGMENTED backing — the flag is not defaulting OFF."
fi
say "guard 2 ok: segmented backing engaged in B, absent in A"

# ------------------------------------------------------------ 7. the verdict --
say "=== phase 7: verdict ==="
python3 - "$OUT_DIR" "$RESULT_FILE" <<'PY' | tee -a "$STATUS_FILE"
import json, os, sys
out, result_path = sys.argv[1], sys.argv[2]
A = json.load(open(os.path.join(out, "phase.control.json")))
B = json.load(open(os.path.join(out, "phase.segmented.json")))

# --- PRIMARY: byte-for-byte output identity ---------------------------------
diffs = []
for a, b in zip(A["identity"], B["identity"]):
    if a["text"] != b["text"]:
        # first divergent character, so the report is actionable
        k = next((i for i, (x, y) in enumerate(zip(a["text"], b["text"])) if x != y),
                 min(len(a["text"]), len(b["text"])))
        diffs.append({"i": a["i"], "prompt_suffix": a["prompt_suffix"],
                      "first_divergence_at_char": k,
                      "control": a["text"][max(0, k-60):k+60],
                      "segmented": b["text"][max(0, k-60):k+60]})
identical = not diffs

ta = A["throughput"]["tokens_per_second"]; tb = B["throughput"]["tokens_per_second"]
ratio = (tb / ta) if ta > 0 else 0.0
tokens_match = A["throughput"]["completion_tokens_total"] == B["throughput"]["completion_tokens_total"]

if not identical:
    verdict, reason = "FAIL", "output identity violated: the degenerate 1-segment path is NOT degenerate"
elif ratio < 0.99:
    verdict, reason = "FAIL", f"throughput ratio {ratio:.4f} < 0.99"
else:
    verdict, reason = "PASS", "identical output and throughput within 1%"

json.dump({
    "verdict": verdict, "reason": reason, "model": A["model"],
    "output_identical": identical, "identity_diffs": diffs,
    "throughput_tokens_per_second": {"control": ta, "segmented": tb},
    "throughput_ratio_segmented_over_control": ratio,
    "throughput_completion_tokens_match": tokens_match,
    "identity_prompts": len(A["identity"]),
}, open(result_path, "w"), indent=2)

print(f"SUMMARY verdict={verdict} output_identical={identical} "
      f"tok_s_control={ta:.2f} tok_s_segmented={tb:.2f} ratio={ratio:.4f} :: {reason}")
sys.exit(0 if verdict == "PASS" else 1)
PY
rc="${PIPESTATUS[0]}"

say "result file: $RESULT_FILE"
if [[ "$rc" != "0" ]]; then
  say "VERDICT: FAIL — see $RESULT_FILE (identity_diffs shows the first divergence)"
  exit 1
fi
say "VERDICT: PASS"
exit 0
