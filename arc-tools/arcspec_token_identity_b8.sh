#!/bin/bash
# ArcSpec — TOKEN IDENTITY at B=8, per-sequence KV advance ON vs OFF.
#
# ─────────────────────────────────────────────────────────────────────────────
# THE QUESTION, AND WHY IT OUTRANKS THE PERF NUMBERS
#
# The A/B run measured, at K=8 with identical prompts and `temperature 0`:
#   OFF arm  ->  8 requests started
#   ON  arm  -> 24 requests started
# while at K=1 the two arms were counter-identical (1235 committed / 533 steps /
# 702 accepted / 1599 proposed, byte for byte).
#
# So per-sequence KV advance is an EXACT no-op at B=1 and the two arms produced
# DIFFERENT TOKEN STREAMS at batch. Two possibilities with opposite
# consequences:
#   (a) sequences legitimately hit EOS earlier — fine;
#   (b) the ragged path changes attention — the chain silently changes what the
#       model says, which no throughput number can offset.
#
# ⚠️ THE CPU TEST CANNOT SETTLE THIS.
# `batched_ragged_accept_is_token_identical_to_the_b1_reference`
# (`mtp_pipeline.rs:6104`) is a SIMULATION: `SimSeq` carries fabricated tokens
# (`toks: vec![1, tok]`, `agree_mod`) through `sim_step`, which is scheduling
# arithmetic. It never builds a KV cache, never calls attention, never runs the
# model. It is structurally incapable of observing a divergence caused by real
# attention over a ragged KV layout — so it passes either way, and its passing
# is not evidence. That gap is itself a finding.
#
# ─────────────────────────────────────────────────────────────────────────────
# THE DESIGN — and the control that makes it mean something
#
# Per arm, in ONE server process:
#   Phase A (B=1 reference): the 8 prompts sent STRICTLY SEQUENTIALLY, one in
#                            flight at a time. No batching is possible.
#   Phase B (B=8):           the same 8 prompts sent CONCURRENTLY.
#
# Greedy everywhere (`temperature 0`, fixed `--seed`), so a correct engine must
# return the same text for a prompt regardless of what it was batched with.
#
# 🔑 **The OFF arm is the control and it is not optional.** If OFF's B=8 output
# already differs from OFF's B=1 reference, then BATCHING ITSELF perturbs
# output on this build and any ON divergence is unattributable to the chain
# under test. Only `OFF: B8 == B1` **and** `ON: B8 != B1` implicates
# per-sequence advance. Reporting an ON divergence without that control would
# be blaming the chain for a property of the engine.
#
# Four comparisons are reported, per prompt:
#   1. OFF B=8 vs OFF B=1   <- the control. Must be identical, or nothing else reads.
#   2. ON  B=8 vs ON  B=1   <- the treatment.
#   3. ON  B=1 vs OFF B=1   <- must be identical (the flags are a no-op at B=1).
#   4. ON  B=8 vs OFF B=8   <- the direct arm-to-arm diff.
#
# Divergence is reported as the first differing token INDEX with a window of
# context either side, not as a bare "differs" — where a stream diverges says
# which mechanism did it.
#
# USAGE
#   setsid nohup bash /root/token_identity_b8.sh </dev/null \
#     > /root/logs/token_identity.log 2>&1 &
#   cat /root/logs/tokid/summary.txt
#
set -u

ARCENV="${ARCENV:-/root/arcenv.sh}"
WT="${WT:-/root/arc-arcspec}"          # already built by the A/B run
BIN="${BIN:-$WT/target/release/mistralrs}"
MODEL="${MODEL:-/root/models/v4-src}"
UQFF="${UQFF:-/root/models/v4-uqff/qtip2b-0.uqff}"
ARCH="${ARCH:-deepseekv4}"
PREFLIGHT="${PREFLIGHT:-/root/arc-tools/gpu_box_preflight.sh}"
OUT="${OUT:-/root/logs/tokid}"
STATUS="${STATUS:-/root/logs/token_identity.status}"
PORT="${PORT:-1239}"
MTP_DEPTH="${MTP_DEPTH:-3}"
MAX_SEQS="${MAX_SEQS:-256}"
SEED="${SEED:-1234}"
MAX_TOKENS="${MAX_TOKENS:-48}"
NPROMPTS="${NPROMPTS:-8}"
LOCK="${LOCK:-/root/locks/gpu.lock}"
LOCK_WAIT="${LOCK_WAIT:-2400}"          # queue behind other chains, don't barge
OWNER="tokid"

mkdir -p "$OUT" "$(dirname "$STATUS")" "$(dirname "$LOCK")"
: > "$STATUS"
say() { echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$STATUS"; }
die() { say "ABORT: $*"; echo "ABORT: $*" >> "$OUT/summary.txt"; exit 2; }

say "token-identity B=8 — start"

# ── GPU lock, with OWNER-MATCHED RELEASE.
#
#    Earlier tonight I removed another chain's live lock because I saw the file,
#    assumed it was my own leak, and did not read it. The owner tag was already
#    in the file; nobody was checking it. Release now refuses to remove a lock
#    this process does not own, so the same mistake cannot be made by this
#    script even if it exits abnormally.
release_lock() {
  if [ -f "$LOCK" ] && grep -q "^$OWNER pid=$$ " "$LOCK" 2>/dev/null; then
    rm -f "$LOCK"
  fi
}
acquire_lock() {
  local waited=0
  while :; do
    if ( set -o noclobber; echo "$OWNER pid=$$ started=$(date -u +%FT%TZ)" > "$LOCK" ) 2>/dev/null; then
      trap 'release_lock' EXIT INT TERM
      say "lock: acquired $LOCK"
      return 0
    fi
    if [ "$waited" -ge "$LOCK_WAIT" ]; then
      say "lock: HELD BY -> $(cat "$LOCK" 2>/dev/null)"
      return 1
    fi
    [ "$waited" = 0 ] && say "lock: waiting for -> $(cat "$LOCK" 2>/dev/null)"
    sleep 15; waited=$((waited + 15))
  done
}
acquire_lock || die "GPU lock held by another chain for >${LOCK_WAIT}s"

[ -f "$ARCENV" ] || die "$ARCENV not found"
# shellcheck disable=SC1090
. "$ARCENV"
[ -x "$PREFLIGHT" ] || die "$PREFLIGHT missing"
"$PREFLIGHT" > "$OUT/preflight.log" 2>&1 || die "preflight refused the box"
say "preflight OK"
[ -x "$BIN" ] || die "$BIN not found — run the A/B script first to build it"
SHA=$(git -C "$WT" rev-parse HEAD 2>/dev/null)
say "binary ref: $SHA"

# ── The prober. Sequential phase then concurrent phase, same prompts, greedy.
cat > "$OUT/probe.py" <<'PYEOF'
import argparse, json, sys, threading, urllib.request

# Eight distinct prompts. Distinct so a divergence is attributable to one
# stream; short so prefill is cheap and every request completes fast.
PROMPTS = [
    "List the first eight prime numbers, separated by commas.",
    "Name the seven continents, one per line.",
    "Count from one to twenty in words, separated by spaces.",
    "Write the first eight letters of the alphabet, separated by dashes.",
    "Name eight colours of the visible spectrum, comma separated.",
    "List eight consecutive even numbers starting at two.",
    "Name the days of the week in order, one per line.",
    "List the first eight powers of two, comma separated.",
]

def one(port, prompt, max_tokens, seed):
    body = json.dumps({
        "model": "default", "prompt": prompt, "max_tokens": max_tokens,
        "temperature": 0, "seed": seed, "stream": True,
    }).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    toks = []
    finish = None
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            p = line[5:].strip()
            if p == "[DONE]":
                break
            try:
                ch = json.loads(p)
            except Exception:
                continue
            c = (ch.get("choices") or [{}])[0]
            t = c.get("text", "")
            if t:
                toks.append(t)
            if c.get("finish_reason"):
                finish = c["finish_reason"]
    return {"tokens": toks, "text": "".join(toks), "finish_reason": finish}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--phase", choices=["b1", "b8"], required=True)
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--n", type=int, default=8)
    a = ap.parse_args()
    prompts = PROMPTS[: a.n]

    if a.phase == "b1":
        # STRICTLY sequential: one request in flight at a time, so the engine
        # can never form a batch. This is the reference every batched stream
        # must reproduce.
        out = [one(a.port, p, a.max_tokens, a.seed) for p in prompts]
    else:
        out = [None] * len(prompts)
        def w(i):
            out[i] = one(a.port, prompts[i], a.max_tokens, a.seed)
        ts = [threading.Thread(target=w, args=(i,)) for i in range(len(prompts))]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
    json.dump({"phase": a.phase, "results": out}, sys.stdout)

main()
PYEOF

# ── The comparator. First differing token index, with context.
cat > "$OUT/compare.py" <<'PYEOF'
"""Compare two captured phases token by token and locate the first divergence.

Reports the INDEX of the first differing token plus a window either side. A
bare "differs" does not say which mechanism did it; where a stream splits does.
"""
import json, sys

def load(p):
    with open(p) as f:
        return json.load(f)["results"]

def cmp_streams(a, b, label_a, label_b):
    diffs = []
    for i, (x, y) in enumerate(zip(a, b)):
        tx, ty = x["tokens"], y["tokens"]
        if tx == ty:
            continue
        n = min(len(tx), len(ty))
        first = next((j for j in range(n) if tx[j] != ty[j]), n)
        lo = max(0, first - 4)
        diffs.append({
            "prompt_index": i,
            "first_divergent_token": first,
            "len_a": len(tx), "len_b": len(ty),
            "finish_a": x.get("finish_reason"), "finish_b": y.get("finish_reason"),
            "context_a": "".join(tx[lo:first + 5]),
            "context_b": "".join(ty[lo:first + 5]),
        })
    return diffs

def report(name, a_path, b_path, la, lb):
    a, b = load(a_path), load(b_path)
    d = cmp_streams(a, b, la, lb)
    status = "IDENTICAL" if not d else f"DIVERGES on {len(d)}/{len(a)} prompts"
    print(f"\n### {name}: {status}   ({la} vs {lb})")
    for e in d:
        print(f"    prompt[{e['prompt_index']}] first differing token #"
              f"{e['first_divergent_token']}  len {e['len_a']} vs {e['len_b']}"
              f"  finish {e['finish_a']} vs {e['finish_b']}")
        print(f"      {la}: ...{e['context_a']!r}")
        print(f"      {lb}: ...{e['context_b']!r}")
    return len(d)

o1 = report("1. CONTROL  OFF B=8 vs OFF B=1", sys.argv[1], sys.argv[2], "OFF-B8", "OFF-B1")
t1 = report("2. TREATMENT ON B=8 vs ON  B=1", sys.argv[3], sys.argv[4], "ON-B8", "ON-B1")
n1 = report("3. NULL      ON B=1 vs OFF B=1", sys.argv[4], sys.argv[2], "ON-B1", "OFF-B1")
a1 = report("4. DIRECT    ON B=8 vs OFF B=8", sys.argv[3], sys.argv[1], "ON-B8", "OFF-B8")

# ── COMPLETION, graded INDEPENDENTLY of text.
#
# The first version of this comparator graded only text identity, so when the
# control diverged it returned "uncontrolled" for everything — and that
# MISCLASSIFIED the case it was built to catch. A diverging control invalidates
# comparing *what* was generated; it does NOT invalidate comparing *whether*
# anything was. An arm that returns one token with no `finish_reason` while the
# other returns full-length has failed in a way no amount of control divergence
# explains away, and that has to be gradeable on its own axis.
def completion(path, label):
    rs = load(path)
    lens = [len(r["tokens"]) for r in rs]
    nofin = sum(1 for r in rs if not r.get("finish_reason"))
    return {"label": label, "n": len(rs), "min": min(lens), "max": max(lens),
            "mean": sum(lens) / len(lens), "no_finish": nofin, "lens": lens}

print("\n=== COMPLETION (graded independently of text) ===")
comps = [completion(p, l) for p, l in
         ((sys.argv[2], "OFF-B1"), (sys.argv[1], "OFF-B8"),
          (sys.argv[4], "ON-B1"), (sys.argv[3], "ON-B8"))]
for c in comps:
    print(f"  {c['label']:7} n={c['n']}  token len min={c['min']} max={c['max']} "
          f"mean={c['mean']:.1f}  streams with no finish_reason: {c['no_finish']}/{c['n']}")
by = {c["label"]: c for c in comps}
truncated = [c["label"] for c in comps if c["min"] <= 1 or c["no_finish"] == c["n"]]

print("\n=== VERDICT ===")
if truncated:
    print(f"VERDICT[truncation] {', '.join(truncated)} returned degenerate streams "
          f"(a token length of <=1, or no finish_reason on every stream) while its own "
          f"B=1 reference did not. That is a FAILURE TO GENERATE, not a difference in "
          f"what was generated, and it is attributable even when the text control "
          f"diverges. This outranks every throughput number and the text verdict below.")
if n1:
    print("VERDICT[void] The flags changed output at B=1, where they are supposed to be an "
          "exact no-op. Something other than per-sequence advance moved; the B=8 rows below "
          "cannot be attributed. INVESTIGATE THIS FIRST.")
elif o1:
    print(f"VERDICT[uncontrolled] The CONTROL diverged on {o1} prompt(s): batching alone "
          "changes output on this build, independent of per-sequence advance. The chain "
          "under test CANNOT be blamed for the treatment divergence until this is fixed. "
          "This is an engine-level batching defect and it outranks the ArcSpec question.")
elif t1:
    print(f"VERDICT[implicated] Control identical, treatment diverged on {t1} prompt(s). "
          "Batching alone does NOT change output, but batching WITH per-sequence KV advance "
          "does. That implicates the ragged path in changing what the model says — a "
          "correctness defect, not a performance trade-off.")
else:
    print("VERDICT[clean] Every stream is token-identical to its B=1 reference in both arms. "
          "Per-sequence advance does not change what the model says at B=8. The 24-vs-8 "
          "request-count difference is therefore NOT a divergence in generated text and "
          "needs a different explanation (scheduling/admission, not attention).")
PYEOF

run_arm() {
  local arm="$1"; shift
  local slog="$OUT/server.$arm.log"
  say "arm $arm — starting server (${*:-no flags})"
  # shellcheck disable=SC2086
  env RUST_LOG=info ARC_MTP_LOG_ACCEPTANCE=1 "$@" \
    "$BIN" serve -p "$PORT" -m "$MODEL" -a "$ARCH" --from-uqff "$UQFF" \
    --max-seqs "$MAX_SEQS" --mtp-depth "$MTP_DEPTH" --prefix-cache-n 0 \
    --paged-attn off --seed "$SEED" \
    < /dev/null > "$slog" 2>&1 &
  local spid=$!
  local up=0
  for _ in $(seq 1 450); do
    curl -s -m 5 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && { up=1; break; }
    kill -0 "$spid" 2>/dev/null || break
    sleep 2
  done
  [ "$up" = "1" ] || { say "arm $arm — server never came up"; tail -20 "$slog" | tee -a "$STATUS"; return 1; }

  # Provenance: the running server must be the code we think it is.
  local logged
  logged=$(grep -m1 -o 'git revision: [0-9a-f]*' "$slog" | awk '{print $3}')
  if [ -n "$SHA" ] && [ "$logged" != "$SHA" ]; then
    say "arm $arm — PROVENANCE FAILED: server $logged != worktree $SHA"
    kill -TERM "$spid" 2>/dev/null || true
    return 1
  fi
  say "arm $arm — server up, revision $logged"

  # Engagement, asserted per arm (present in ON, absent in OFF).
  say "arm $arm — phase B=1 (sequential reference)"
  python3 "$OUT/probe.py" --port "$PORT" --phase b1 --max-tokens "$MAX_TOKENS" \
    --seed "$SEED" --n "$NPROMPTS" > "$OUT/$arm.b1.json" || { say "b1 probe failed"; return 1; }
  say "arm $arm — phase B=8 (concurrent)"
  python3 "$OUT/probe.py" --port "$PORT" --phase b8 --max-tokens "$MAX_TOKENS" \
    --seed "$SEED" --n "$NPROMPTS" > "$OUT/$arm.b8.json" || { say "b8 probe failed"; return 1; }

  kill -TERM "$spid" 2>/dev/null || true
  for _ in $(seq 1 30); do kill -0 "$spid" 2>/dev/null || break; sleep 2; done
  kill -KILL "$spid" 2>/dev/null || true
  sleep 5
  say "arm $arm — done"
  return 0
}

run_arm OFF || die "OFF arm failed"
run_arm ON ARC_V4_XS_PER_SEQ=1 ARC_MTP_PER_SEQ_KV=1 || die "ON arm failed"

{
  echo "=== ref ==="; echo "$SHA"
  echo
  echo "=== ENGAGEMENT ==="
  for m in "per-sequence KV advance is ON" "Ragged batch admission is ON"; do
    on=$(grep -c "$m" "$OUT/server.ON.log" 2>/dev/null || echo 0)
    off=$(grep -c "$m" "$OUT/server.OFF.log" 2>/dev/null || echo 0)
    echo "  '$m': ON=$on (want >=1)  OFF=$off (want 0)"
  done
  echo "  'cannot honour it' in ON: $(grep -c 'cannot honour it' "$OUT/server.ON.log" 2>/dev/null || echo 0) (want 0)"
  echo
  echo "=== TOKEN IDENTITY ==="
  python3 "$OUT/compare.py" "$OUT/OFF.b8.json" "$OUT/OFF.b1.json" \
                            "$OUT/ON.b8.json"  "$OUT/ON.b1.json"
} | tee "$OUT/summary.txt"

say "DONE — $OUT/summary.txt"
