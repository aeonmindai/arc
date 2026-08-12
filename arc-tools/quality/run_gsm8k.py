#!/usr/bin/env python3
"""GSM8K subset eval for the served 2-bit V4-Flash (RUN-161 quality battery).

Greedy (temperature=0), official V4 chat encoding (encoding_dsv4.encode_messages)
via the raw /v1/completions endpoint — the chat.py machinery. Deterministic,
seeded subset of the public GSM8K test split (data/gsm8k_test.jsonl from
fetch_data.sh).

Published anchor (see EXPECTED.md): DeepSeek-V4-Flash-**Base**, 8-shot EM = 90.8.
Our number is the INSTRUCT model, 0-shot chat-mode CoT, greedy — not
apples-to-apples, but a stable self-anchor for the quant ladder. On 150 problems
the 95% binomial CI is about ±6pp; treat differences smaller than that as noise.

Crash-safe: results JSON is rewritten after every problem; --resume (default)
skips already-answered indices, so a dropped SSH session costs nothing.

Usage (on the box, server already up with --prefix-cache-n 0):
  python3 arc-tools/quality/run_gsm8k.py            # 150 problems, seed 161
  python3 arc-tools/quality/run_gsm8k.py --n 50     # quick pass
  python3 arc-tools/quality/run_gsm8k.py --thinking # thinking-mode variant
"""
import argparse
import json
import math
import os
import random
import re
import sys
import time

import qlib

INSTRUCTION = (
    "\n\nPlease reason step by step, and put your final numeric answer after '####'."
)

NUM_RE = re.compile(r"-?\$?\d[\d,]*\.?\d*")


def norm_num(s):
    """Normalize a numeric string for EM comparison ('$1,234.00' == '1234')."""
    s = s.strip().replace(",", "").replace("$", "").replace("%", "").rstrip(".")
    try:
        v = float(s)
    except ValueError:
        return None
    if v == int(v):
        return str(int(v))
    return repr(v)


def gold_answer(answer_field):
    m = answer_field.rsplit("####", 1)
    if len(m) != 2:
        return None
    return norm_num(m[1])


def extract_pred(text):
    """Prefer the number after the last '####'; fall back to the last number."""
    if "####" in text:
        tail = text.rsplit("####", 1)[1]
        nums = NUM_RE.findall(tail)
        if nums:
            return norm_num(nums[0])
    nums = NUM_RE.findall(text)
    if nums:
        return norm_num(nums[-1])
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150, help="subset size (default 150)")
    ap.add_argument("--seed", type=int, default=161)
    ap.add_argument("--data", default=os.path.join(qlib.DATA_DIR, "gsm8k_test.jsonl"))
    ap.add_argument("--out", default=None, help="results JSON path")
    ap.add_argument("--max-tokens", type=int, default=640)
    ap.add_argument("--thinking", action="store_true", help="use thinking mode")
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    qlib.ensure_dirs()
    mode = "thinking" if args.thinking else "chat"
    out_path = args.out or os.path.join(qlib.RESULTS_DIR, f"gsm8k_{mode}.json")

    if not os.path.exists(args.data):
        print(f"FATAL: {args.data} missing — run fetch_data.sh first", file=sys.stderr)
        sys.exit(2)
    with open(args.data) as f:
        problems = [json.loads(line) for line in f if line.strip()]
    idxs = list(range(len(problems)))
    random.Random(args.seed).shuffle(idxs)
    idxs = idxs[: args.n]

    qlib.health_or_die()

    done = {}
    if not args.no_resume and os.path.exists(out_path):
        try:
            with open(out_path) as f:
                prev = json.load(f)
            done = {r["idx"]: r for r in prev.get("items", [])}
            print(f"resuming: {len(done)} already answered in {out_path}")
        except Exception:
            done = {}

    items = []
    t_start = time.time()
    max_tokens = args.max_tokens * (2 if args.thinking else 1)
    for k, idx in enumerate(idxs):
        if idx in done:
            items.append(done[idx])
            continue
        prob = problems[idx]
        gold = gold_answer(prob["answer"])
        prompt = qlib.encode_chat(prob["question"] + INSTRUCTION, thinking_mode=mode)
        try:
            r = qlib.complete(prompt, max_tokens=max_tokens, temperature=0.0)
        except Exception as e:
            r = {"text": f"[ERROR {e}]", "finish_reason": "error",
                 "prompt_tokens": None, "completion_tokens": 0, "seconds": 0}
        pred = extract_pred(r["text"])
        correct = pred is not None and gold is not None and pred == gold
        rec = {
            "idx": idx,
            "gold": gold,
            "pred": pred,
            "correct": bool(correct),
            "finish_reason": r["finish_reason"],
            "completion_tokens": r["completion_tokens"],
            "seconds": r["seconds"],
            "degenerate": qlib.looks_degenerate(r["text"]),
            "text_tail": r["text"][-300:],
        }
        items.append(rec)

        n_done = len(items)
        n_ok = sum(1 for it in items if it["correct"])
        elapsed = time.time() - t_start
        rate = elapsed / max(n_done - len(done), 1)
        eta_min = rate * (len(idxs) - n_done) / 60
        print(f"[{n_done}/{len(idxs)}] {'PASS' if correct else 'FAIL'} "
              f"gold={gold} pred={pred} ({r['completion_tokens']}tok {r['seconds']}s) "
              f"| running acc {n_ok}/{n_done} | ETA {eta_min:.0f}m", flush=True)

        summary = _summarize(items, args, mode)
        qlib.write_json(out_path, summary)

    summary = _summarize(items, args, mode)
    qlib.write_json(out_path, summary)
    s = summary["summary"]
    print(f"\nGSM8K[{mode}] greedy n={s['n']}: {s['correct']}/{s['n']} = "
          f"{s['accuracy_pct']}% (95% CI ±{s['ci95_pp']}pp) | "
          f"published V4-Flash-Base 8-shot EM: 90.8 (see EXPECTED.md) | "
          f"degenerate loops: {s['degenerate']} | truncated: {s['truncated']}")


def _summarize(items, args, mode):
    n = len(items)
    ok = sum(1 for it in items if it["correct"])
    p = ok / n if n else 0.0
    ci = 1.96 * math.sqrt(p * (1 - p) / n) if n else 0.0
    return {
        "meta": qlib.run_meta({
            "eval": "gsm8k", "mode": mode, "n_target": args.n,
            "seed": args.seed, "max_tokens": args.max_tokens,
            "sampling": "greedy (temperature=0)",
            "published_anchor": {"model": "DeepSeek-V4-Flash-Base", "setting": "8-shot EM", "score": 90.8},
        }),
        "summary": {
            "n": n,
            "correct": ok,
            "accuracy_pct": round(100 * p, 1),
            "ci95_pp": round(100 * ci, 1),
            "degenerate": sum(1 for it in items if it.get("degenerate")),
            "truncated": sum(1 for it in items if it.get("finish_reason") == "length"),
            "errors": sum(1 for it in items if it.get("finish_reason") == "error"),
            "mean_completion_tokens": round(
                sum(it["completion_tokens"] or 0 for it in items) / n, 1) if n else 0,
        },
        "items": items,
    }


if __name__ == "__main__":
    main()
