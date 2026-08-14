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

Session-4 additions:
  --votes K         Arc Boost server-side voting (PR #13): each problem is one
                    /v1/chat/completions request with n_votes=K; the K chains
                    decode in the same forward-pass batch and the server
                    returns the voted winner as choices[0] (+ `vote` tally).
                    Chains are sampled (--vote-temperature/--vote-top-p;
                    greedy chains would be K identical copies). NOTE: this is
                    the SERVER's chat template, not the raw encoding_dsv4
                    path — run the runbook's template-parity probe first.
  --client-votes    Fallback voting that keeps the validated raw-completions
                    encoding path: K independent sampled requests per problem,
                    majority tally client-side. Use when the server-side vote
                    smoke gate fails (e.g. multi-chain compressor state).
  --vote-mode       "confidence_weighted" (default) or "majority" — server-side
                    aggregation (ignored under --client-votes).
  --eight-shot      Prepend 8 exemplars drawn deterministically from the GSM8K
                    TRAIN split (data/gsm8k_train.jsonl — fetch_data.sh), CoT
                    answers with calculator annotations stripped, each ending
                    in '#### <gold>' — the published 8-shot protocol's prompt
                    shape, carried inside our chat-mode user message.
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


CALC_RE = re.compile(r"<<[^>]*>>")


def build_fewshot_block(train_path, k=8, seed=8):
    """Deterministic k-shot exemplar block from the GSM8K TRAIN split.

    Train answers are natural CoT rationales ending in '#### <gold>' — the
    same shape the published 8-shot EM protocol uses. Calculator annotations
    ('<<48/2=24>>') are stripped, as published GSM8K few-shot prompts do.
    """
    if not os.path.exists(train_path):
        print(f"FATAL: {train_path} missing — run fetch_data.sh first "
              f"(session-4 version fetches the train split too)", file=sys.stderr)
        sys.exit(2)
    with open(train_path) as f:
        train = [json.loads(line) for line in f if line.strip()]
    idxs = list(range(len(train)))
    random.Random(seed).shuffle(idxs)
    shots = []
    for i in idxs[:k]:
        q = train[i]["question"].strip()
        a = CALC_RE.sub("", train[i]["answer"]).strip()
        shots.append(f"Question: {q}\nAnswer: {a}")
    return "\n\n".join(shots)


def majority_tally(preds):
    """Client-side majority vote over extracted answers (None excluded).
    Returns (winner_or_None, tally_dict). Ties break by first occurrence."""
    tally = {}
    order = []
    for p in preds:
        if p is None:
            continue
        if p not in tally:
            order.append(p)
        tally[p] = tally.get(p, 0) + 1
    if not tally:
        return None, {}
    best = max(tally.values())
    for p in order:  # first-seen tie-break: earliest chain wins
        if tally[p] == best:
            return p, tally
    return None, tally


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150, help="subset size (default 150)")
    ap.add_argument("--seed", type=int, default=161)
    ap.add_argument("--data", default=os.path.join(qlib.DATA_DIR, "gsm8k_test.jsonl"))
    ap.add_argument("--out", default=None, help="results JSON path")
    ap.add_argument("--max-tokens", type=int, default=640)
    ap.add_argument("--thinking", action="store_true", help="use thinking mode")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--votes", type=int, default=0,
                    help="Arc Boost voting: k chains per problem (0 = off; server needs >= 2)")
    ap.add_argument("--vote-mode", default="confidence_weighted",
                    choices=["confidence_weighted", "majority"],
                    help="server-side vote aggregation (PR #13)")
    ap.add_argument("--client-votes", action="store_true",
                    help="client-side fallback: k sequential sampled raw completions + majority tally")
    ap.add_argument("--vote-temperature", type=float, default=1.0,
                    help="sampling temperature for vote chains (measured-good chat setting)")
    ap.add_argument("--vote-top-p", type=float, default=0.95)
    ap.add_argument("--eight-shot", action="store_true",
                    help="prepend 8 GSM8K-train exemplars (published-protocol prompt shape)")
    ap.add_argument("--fewshot-seed", type=int, default=8)
    ap.add_argument("--train-data",
                    default=os.path.join(qlib.DATA_DIR, "gsm8k_train.jsonl"))
    args = ap.parse_args()

    if args.votes == 1:
        print("FATAL: --votes 1 is not a vote (server requires >= 2, and 1 sampled "
              "chain is strictly worse than greedy). Use --votes 0 or >= 2.",
              file=sys.stderr)
        sys.exit(2)
    if args.client_votes and not args.votes:
        print("FATAL: --client-votes requires --votes K", file=sys.stderr)
        sys.exit(2)
    if args.votes and args.thinking:
        print("FATAL: --votes + --thinking not wired (chat endpoint owns the "
              "template in voting mode)", file=sys.stderr)
        sys.exit(2)

    qlib.ensure_dirs()
    mode = "thinking" if args.thinking else "chat"
    suffix = ""
    if args.eight_shot:
        suffix += "_8shot"
    if args.votes:
        suffix += f"_cvotes{args.votes}" if args.client_votes else f"_votes{args.votes}"
    out_path = args.out or os.path.join(qlib.RESULTS_DIR, f"gsm8k_{mode}{suffix}.json")

    fewshot_block = None
    if args.eight_shot:
        fewshot_block = build_fewshot_block(args.train_data, k=8, seed=args.fewshot_seed)

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
        user_msg = prob["question"] + INSTRUCTION
        if fewshot_block is not None:
            user_msg = (fewshot_block + "\n\nQuestion: " + prob["question"]
                        + INSTRUCTION)
        extra = {}
        if args.votes and not args.client_votes:
            # Server-side Arc Boost voting (one request; k chains, one batch).
            try:
                r = qlib.chat_complete(
                    user_msg, max_tokens=max_tokens,
                    temperature=args.vote_temperature, top_p=args.vote_top_p,
                    n_votes=args.votes, vote_mode=args.vote_mode)
            except Exception as e:
                r = {"text": f"[ERROR {e}]", "finish_reason": "error",
                     "prompt_tokens": None, "completion_tokens": 0,
                     "seconds": 0, "vote": None}
            pred = extract_pred(r["text"])
            if r.get("vote"):
                extra["vote"] = {
                    "winner_answer": r["vote"].get("winner_answer"),
                    "winner_index": r["vote"].get("winner_index"),
                    "candidates": [
                        {"answer": c.get("answer"),
                         "confidence": c.get("confidence"),
                         "weight": c.get("weight")}
                        for c in r["vote"].get("candidates", [])
                    ],
                }
        elif args.votes:
            # Client-side fallback: k sequential sampled chains on the RAW
            # (encoding-validated) endpoint + majority tally.
            prompt = qlib.encode_chat(user_msg, thinking_mode=mode)
            chains = []
            for _ in range(args.votes):
                try:
                    c = qlib.complete(prompt, max_tokens=max_tokens,
                                      temperature=args.vote_temperature,
                                      top_p=args.vote_top_p)
                except Exception as e:
                    c = {"text": f"[ERROR {e}]", "finish_reason": "error",
                         "prompt_tokens": None, "completion_tokens": 0,
                         "seconds": 0}
                chains.append(c)
            preds = [extract_pred(c["text"]) for c in chains]
            pred, tally = majority_tally(preds)
            winner = next((c for c, p in zip(chains, preds) if p == pred),
                          chains[0])
            r = {
                "text": winner["text"],
                "finish_reason": winner["finish_reason"],
                "prompt_tokens": chains[0]["prompt_tokens"],
                "completion_tokens": sum(c["completion_tokens"] or 0 for c in chains),
                "seconds": round(sum(c["seconds"] for c in chains), 3),
            }
            extra["client_vote"] = {"chain_preds": preds, "tally": tally}
        else:
            prompt = qlib.encode_chat(user_msg, thinking_mode=mode)
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
        rec.update(extra)
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
    print(f"\nGSM8K[{mode}{suffix}] {summary['meta']['sampling']} n={s['n']}: "
          f"{s['correct']}/{s['n']} = "
          f"{s['accuracy_pct']}% (95% CI ±{s['ci95_pp']}pp) | "
          f"published V4-Flash-Base 8-shot EM: 90.8 (see EXPECTED.md) | "
          f"degenerate loops: {s['degenerate']} | truncated: {s['truncated']}")


def _summarize(items, args, mode):
    n = len(items)
    ok = sum(1 for it in items if it["correct"])
    p = ok / n if n else 0.0
    ci = 1.96 * math.sqrt(p * (1 - p) / n) if n else 0.0
    if args.votes:
        sampling = (f"{'client' if args.client_votes else 'server'}-voted k={args.votes} "
                    f"(t={args.vote_temperature}/p={args.vote_top_p}"
                    + ("" if args.client_votes else f", vote_mode={args.vote_mode}") + ")")
    else:
        sampling = "greedy (temperature=0)"
    return {
        "meta": qlib.run_meta({
            "eval": "gsm8k", "mode": mode, "n_target": args.n,
            "seed": args.seed, "max_tokens": args.max_tokens,
            "sampling": sampling,
            "votes": args.votes or None,
            "vote_mode": (args.vote_mode if args.votes and not args.client_votes else None),
            "client_votes": bool(args.votes and args.client_votes) or None,
            "eight_shot": args.eight_shot or None,
            "fewshot_seed": args.fewshot_seed if args.eight_shot else None,
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
