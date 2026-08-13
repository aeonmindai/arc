#!/usr/bin/env bash
# Fetch the two public eval corpora for the RUN-161 quality battery.
# Both are PUBLIC — no HF_TOKEN needed (only the V4 model itself might need one).
# Idempotent + resumable; safe to re-run.
#
#   wikitext-2  -> data/wiki.test_small.raw (+ wiki.test_mini.raw, wiki.test.raw)
#                  (EricB/wikitext2, the exact corpus the perplexity example docs use)
#   GSM8K test  -> data/gsm8k_test.jsonl (1319 problems; {"question","answer"} JSONL,
#                  gold answer after '####'). Primary: openai/grade-school-math GitHub
#                  raw. Fallback: HF dataset openai/gsm8k (parquet -> jsonl, needs
#                  pyarrow, installed on demand).
set -uo pipefail
cd "$(dirname "$0")"
mkdir -p data

fail() { echo "FAIL: $*" >&2; exit 1; }
ok()   { echo "OK: $*"; }

# ---------------- wikitext-2 ----------------
WT_BASE="https://huggingface.co/datasets/EricB/wikitext2/resolve/main"
for f in wiki.test_small.raw wiki.test_mini.raw wiki.test.raw; do
  if [ -s "data/$f" ]; then
    ok "data/$f already present ($(wc -c < "data/$f") bytes)"
  else
    curl -fsSL --retry 3 -o "data/$f.part" "$WT_BASE/$f" || fail "download $f"
    mv "data/$f.part" "data/$f"
    ok "data/$f ($(wc -c < "data/$f") bytes)"
  fi
done
# sanity: small file should be ~316KB of text
[ "$(wc -c < data/wiki.test_small.raw)" -gt 100000 ] || fail "wiki.test_small.raw suspiciously small"
# derived: ~100KB slice for the chunk-64 ppl rung (bounds the ~1s/chunk
# CPU-logits overhead: 100KB ~= 22K tokens ~= 350 chunks at chunk_size 64)
head -c 100000 data/wiki.test_small.raw > data/wiki.test_c64slice.raw
ok "data/wiki.test_c64slice.raw (derived)"

# ---------------- GSM8K ----------------
GSM_OUT="data/gsm8k_test.jsonl"
GSM_RAW="https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
if [ -s "$GSM_OUT" ]; then
  ok "$GSM_OUT already present ($(wc -l < "$GSM_OUT") lines)"
else
  if curl -fsSL --retry 3 -o "$GSM_OUT.part" "$GSM_RAW"; then
    mv "$GSM_OUT.part" "$GSM_OUT"
    ok "GSM8K from GitHub raw"
  else
    echo "GitHub raw failed; falling back to HF dataset openai/gsm8k (parquet)"
    python3 -c "import pyarrow" 2>/dev/null || pip install --quiet --break-system-packages pyarrow 2>/dev/null || pip install --quiet pyarrow || fail "pyarrow install"
    hf download openai/gsm8k --repo-type dataset --local-dir data/gsm8k_hf || fail "hf download openai/gsm8k"
    python3 - "$GSM_OUT" <<'EOF' || fail "parquet -> jsonl conversion"
import glob, json, sys
import pyarrow.parquet as pq
out = sys.argv[1]
files = sorted(glob.glob("data/gsm8k_hf/**/test-*.parquet", recursive=True))
assert files, "no test parquet found under data/gsm8k_hf"
n = 0
with open(out + ".part", "w") as f:
    for p in files:
        t = pq.read_table(p)
        for q, a in zip(t.column("question").to_pylist(), t.column("answer").to_pylist()):
            f.write(json.dumps({"question": q, "answer": a}, ensure_ascii=False) + "\n")
            n += 1
import os
os.replace(out + ".part", out)
print(f"wrote {n} problems -> {out}")
EOF
    ok "GSM8K from HF parquet"
  fi
fi
# sanity: must be 1319 problems with the #### marker
LINES=$(wc -l < "$GSM_OUT")
[ "$LINES" -ge 1000 ] || fail "gsm8k_test.jsonl has only $LINES lines"
head -1 "$GSM_OUT" | grep -q '####' || fail "gsm8k_test.jsonl first record lacks '####' marker"
ok "GSM8K test set ready ($LINES problems)"

echo
echo "DATA_READY: $(du -sh data | awk '{print $1}') in $(pwd)/data"
