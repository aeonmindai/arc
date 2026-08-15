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

Crash-safe: results JSON is rewritten every --checkpoint-every problems
(default 1); --resume (default) skips already-answered indices, so a dropped
SSH session costs nothing.

Usage (on the box, server already up with --prefix-cache-n 0):
  python3 arc-tools/quality/run_gsm8k.py            # 150 problems, seed 161
  python3 arc-tools/quality/run_gsm8k.py --n 50     # quick pass
  python3 arc-tools/quality/run_gsm8k.py --thinking # thinking-mode variant

FULL-SET RUN (wave39-BQ). The public test split is 1,319 problems, not 100.
At n=100 the 95% CI is ±3.8pp — too wide to separate our 96.0 from the
published 90.8. At n=1,319 it is about ±1.1pp.

  python3 arc-tools/quality/run_gsm8k.py --all --concurrency 16 \
      --max-tokens 2048 --checkpoint-every 10 --out results/gsm8k_full.json

  Measured basis (wave34-BL, 1xH200): mean completion 148.5 tok, so the whole
  split is ~196k decode tokens. Aggregate decode peaks at B=16 (30.65 tok/s)
  and is flat to B=64, giving ~1.8 h of decode, call it ~2 h wall with prefill
  and HTTP, ~$10 at $4.85/h. The same run at --concurrency 1 is ~3.8 h / ~$19.
  --concurrency 16 REQUIRES the per-sequence xs_history fix (PR #21) in the
  served build.

QUALITY GATE (wave39-BQ). Degenerate loops, truncations and request errors are
PASS-FAIL outputs, not trailing prose: the run exits 1 with a `FAIL:` marker
when any exceeds its threshold (--max-degenerate/--max-truncated/--max-errors,
all default 0; --min-accuracy is opt-in). Machine-greppable markers, for
orchestrators: RUNPLAN[...] PROGRESS[...] DEGEN[...] TRUNC[...] ERR[...]
GSM8K[...] GATE[OK] WARN[...] FAIL:

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
import concurrent.futures
import json
import math
import os
import random
import re
import sys
import threading
import time

import qlib

INSTRUCTION = (
    "\n\nPlease reason step by step, and put your final numeric answer after '####'."
)

NUM_RE = re.compile(r"-?\$?\d[\d,]*\.?\d*")

# Prose answer markers the model actually emits when it does NOT use '####'.
# Harvested from archived runs, not invented: "Final numeric answer: 8,000."
# (gsm8k s3 idx=88), "Thus final answer: 18.8 hours" (s3 idx=830), "So final
# answer: ####30" (s1 idx=465). Only consulted when '####' is absent, so a
# normal completion's score is untouched — see test_degeneracy.py.
ANSWER_MARKER_RE = re.compile(
    r"final\s+(?:numeric\s+)?answer\s*(?:is)?\s*[:=]?|the\s+answer\s+is",
    re.IGNORECASE,
)


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
    """Three tiers, tried in order:

      1. number after the LAST '####'          (unchanged from the original)
      2. number after the LAST prose answer marker   [ADDED wave39-BQ]
      3. last number anywhere                  (unchanged from the original)

    Tier 2 is purely additive: it only runs when the completion contains no
    '####' at all, so any completion that ends in '#### N' — which is 100/100
    of the wave34-BL run — scores exactly as before.

    MEASURED, not assumed. Re-scoring the 350 archived completions (s1/s2/s3/
    s9 results JSON) changes 3 tails, all 3 from wrong to correct, none the
    other way. Only ONE of those is a real recovered point rather than a
    tail-window artifact: s3 idx=88, where the model answered "Final numeric
    answer: 8,000." (= gold), then failed to stop, restarted the whole
    problem, and ran to the 2048 cap — so the bare last-number fallback
    scored 88,000. On wave34-BL (gsm8k_s9.json) ZERO items change, which is
    the proof that this does not move a healthy run's score.
    """
    if "####" in text:
        tail = text.rsplit("####", 1)[1]
        nums = NUM_RE.findall(tail)
        if nums:
            return norm_num(nums[0])
    markers = list(ANSWER_MARKER_RE.finditer(text))
    if markers:
        nums = NUM_RE.findall(text[markers[-1].end():])
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


def solve_one(idx, prob, args, mode, fewshot_block, max_tokens):
    """Answer one problem and return its result record.

    Pure per-problem work: no shared state, so N of these can run in flight
    concurrently (--concurrency). Every failure path is caught and recorded as
    finish_reason="error" — one dead request must never abort a 1,319-problem
    run that has already cost an hour of H200 time.
    """
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
    degen = qlib.degeneracy_report(r["text"])
    rec = {
        "idx": idx,
        "gold": gold,
        "pred": pred,
        "correct": bool(correct),
        "finish_reason": r["finish_reason"],
        "completion_tokens": r["completion_tokens"],
        "seconds": r["seconds"],
        "degenerate": degen is not None,
        "text_tail": r["text"][-300:],
    }
    if degen is not None:
        # Stored so a flagged run can be triaged offline, without the model.
        rec["degenerate_detail"] = degen
    rec.update(extra)
    return rec


def quality_gate(summary, args):
    """Degenerate/truncated/error counts as PASS-FAIL outputs, not footnotes.

    THE 'FOREVER' PART. Before this, `degenerate` and `truncated` were printed
    at the end of a run and nothing consumed them — session-3 shipped
    "2 degenerate, 9 truncated" and the run still exited 0, so nothing in the
    loop objected. Now every one of them is a threshold with a machine-greppable
    marker and a non-zero exit, so a regression from 0 truncated back to 9
    cannot pass quietly.

    Returns the number of violations (0 = clean).
    """
    s = summary["summary"]
    n = max(s["n"], 1)
    bad = 0

    print(f"DEGEN[{s['degenerate']}/{s['n']}] TRUNC[{s['truncated']}/{s['n']}] "
          f"ERR[{s['errors']}/{s['n']}] MEANTOK[{s['mean_completion_tokens']}]",
          flush=True)

    if args.no_quality_gate:
        print("WARN[GATE] --no-quality-gate: degenerate/truncated thresholds "
              "NOT enforced for this run", flush=True)
        return 0

    if s["degenerate"] > args.max_degenerate:
        bad += 1
        print(f"FAIL: DEGENERATE {s['degenerate']} > --max-degenerate "
              f"{args.max_degenerate}. Repetition loops are back. Inspect "
              f"`degenerate_detail` in the results JSON: `kind` says which "
              f"detector fired and gives the repeating unit.", file=sys.stderr)
    if s["truncated"] > args.max_truncated:
        bad += 1
        print(f"FAIL: TRUNCATED {s['truncated']} > --max-truncated "
              f"{args.max_truncated}. Sequences are running to --max-tokens "
              f"instead of emitting EOS. Either the cap is too low for this "
              f"prompt set or stop handling regressed — check that the served "
              f"model's eos_token_id resolves (chat_template.rs "
              f"calculate_eos_tokens) before blaming the cap.", file=sys.stderr)
    if s["errors"] > args.max_errors:
        bad += 1
        print(f"FAIL: ERRORS {s['errors']} > --max-errors {args.max_errors}.",
              file=sys.stderr)
    if args.min_accuracy > 0 and s["accuracy_pct"] < args.min_accuracy:
        bad += 1
        print(f"FAIL: ACCURACY {s['accuracy_pct']}% < --min-accuracy "
              f"{args.min_accuracy}% (n={n}, CI ±{s['ci95_pp']}pp).",
              file=sys.stderr)
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150,
                    help="subset size (default 150; 0 or --all = whole split, "
                         "1319 problems for the public test set)")
    ap.add_argument("--all", action="store_true",
                    help="run the entire split (equivalent to --n 0)")
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
    ap.add_argument("--concurrency", type=int, default=1,
                    help="requests in flight (default 1 = the historical "
                         "one-at-a-time behaviour). Use 16 for a full-set run: "
                         "aggregate decode peaks at B=16 (30.65 tok/s, "
                         "wave34-BL) and is flat through B=64. REQUIRES the "
                         "per-sequence xs_history fix (PR #21) in the served "
                         "build — concurrency on an older build silently "
                         "corrupts the compressor state.")
    ap.add_argument("--checkpoint-every", type=int, default=1,
                    help="rewrite the results JSON every N completions "
                         "(default 1 = crash-safe after every problem; 10 is "
                         "plenty for a full-set run and does far less IO)")
    ap.add_argument("--max-degenerate", type=int, default=0,
                    help="FAIL above this many degenerate loops (default 0)")
    ap.add_argument("--max-truncated", type=int, default=0,
                    help="FAIL above this many finish_reason=length "
                         "completions (default 0)")
    ap.add_argument("--max-errors", type=int, default=0,
                    help="FAIL above this many failed requests (default 0)")
    ap.add_argument("--min-accuracy", type=float, default=0.0,
                    help="FAIL below this accuracy %% (default 0 = off; "
                         "DOCTRINE D6 commits to 90)")
    ap.add_argument("--no-quality-gate", action="store_true",
                    help="report degenerate/truncated but never fail on them")
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
    if args.concurrency < 1:
        print("FATAL: --concurrency must be >= 1", file=sys.stderr)
        sys.exit(2)
    if args.concurrency > 1 and args.votes and not args.client_votes:
        print("FATAL: --concurrency > 1 with server-side --votes fans out "
              "concurrency x votes sequences; run votes at --concurrency 1 "
              "until that product is measured.", file=sys.stderr)
        sys.exit(2)
    if args.checkpoint_every < 1:
        print("FATAL: --checkpoint-every must be >= 1", file=sys.stderr)
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
    # Shuffle FIRST, then slice. Keeping the shuffle for a full-set run costs
    # nothing and buys a real property: any partial/interrupted run is an
    # unbiased random subset of the split, so its accuracy is still an
    # unbiased estimate rather than "the first K problems".
    random.Random(args.seed).shuffle(idxs)
    want = len(idxs) if (args.all or args.n <= 0) else args.n
    if want > len(idxs):
        print(f"WARN[N] --n {want} exceeds the split ({len(idxs)} problems); "
              f"running all {len(idxs)}.", flush=True)
        want = len(idxs)
    idxs = idxs[:want]

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

    max_tokens = args.max_tokens * (2 if args.thinking else 1)
    todo = [i for i in idxs if i not in done]
    results = {i: done[i] for i in idxs if i in done}
    print(f"RUNPLAN[n={len(idxs)} resumed={len(results)} todo={len(todo)} "
          f"concurrency={args.concurrency} max_tokens={max_tokens}] -> {out_path}",
          flush=True)

    lock = threading.Lock()
    state = {"n_new": 0}
    t_start = time.time()

    def ordered_items():
        return [results[i] for i in idxs if i in results]

    def record(rec):
        """Called once per completed problem, serialized under `lock`."""
        results[rec["idx"]] = rec
        state["n_new"] += 1
        n_done = len(results)
        n_ok = sum(1 for it in results.values() if it["correct"])
        elapsed = time.time() - t_start
        rate = elapsed / max(state["n_new"], 1)
        eta_min = rate * (len(idxs) - n_done) / 60
        print(f"[{n_done}/{len(idxs)}] {'PASS' if rec['correct'] else 'FAIL'} "
              f"idx={rec['idx']} gold={rec['gold']} pred={rec['pred']} "
              f"({rec['completion_tokens']}tok {rec['seconds']}s) "
              f"| running acc {n_ok}/{n_done} | ETA {eta_min:.0f}m", flush=True)
        if state["n_new"] % args.checkpoint_every == 0:
            qlib.write_json(out_path, _summarize(ordered_items(), args, mode))
            print(f"PROGRESS[{n_done}/{len(idxs)} acc={n_ok}/{n_done} "
                  f"elapsed={elapsed / 60:.1f}m eta={eta_min:.0f}m]", flush=True)

    if args.concurrency == 1:
        # Identical to the historical path: one request in flight, results
        # written after every problem. Nothing about a --concurrency 1 run
        # changed, so it stays comparable with every archived run.
        for idx in todo:
            rec = solve_one(idx, problems[idx], args, mode, fewshot_block, max_tokens)
            with lock:
                record(rec)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {
                pool.submit(solve_one, idx, problems[idx], args, mode,
                            fewshot_block, max_tokens): idx
                for idx in todo
            }
            for fut in concurrent.futures.as_completed(futures):
                idx = futures[fut]
                try:
                    rec = fut.result()
                except Exception as e:  # belt and braces; solve_one catches its own
                    rec = {"idx": idx, "gold": gold_answer(problems[idx]["answer"]),
                           "pred": None, "correct": False, "finish_reason": "error",
                           "completion_tokens": 0, "seconds": 0,
                           "degenerate": False, "text_tail": f"[WORKER ERROR {e}]"}
                with lock:
                    record(rec)

    items = ordered_items()
    summary = _summarize(items, args, mode)
    qlib.write_json(out_path, summary)
    s = summary["summary"]
    print(f"\nGSM8K[{mode}{suffix}] {summary['meta']['sampling']} n={s['n']}: "
          f"{s['correct']}/{s['n']} = "
          f"{s['accuracy_pct']}% (95% CI ±{s['ci95_pp']}pp) | "
          f"published V4-Flash-Base 8-shot EM: 90.8 (see EXPECTED.md) | "
          f"degenerate loops: {s['degenerate']} | truncated: {s['truncated']}")

    bad = quality_gate(summary, args)
    if bad:
        print(f"FAIL: {bad} quality gate(s) violated — results still written to "
              f"{out_path}", file=sys.stderr)
        sys.exit(1)
    print("GATE[OK] degenerate/truncated/errors within thresholds", flush=True)


def _kind_counts(items):
    """Which detector fired, per run — a loop and a truncation are different
    bugs and the summary should not blur them."""
    kinds = {}
    for it in items:
        det = it.get("degenerate_detail")
        if det:
            kinds[det["kind"]] = kinds.get(det["kind"], 0) + 1
    return kinds


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
            "concurrency": args.concurrency,
            "thresholds": {
                "max_degenerate": args.max_degenerate,
                "max_truncated": args.max_truncated,
                "max_errors": args.max_errors,
                "min_accuracy": args.min_accuracy,
                "enforced": not args.no_quality_gate,
            },
            "published_anchor": {"model": "DeepSeek-V4-Flash-Base", "setting": "8-shot EM", "score": 90.8},
        }),
        "summary": {
            "n": n,
            "correct": ok,
            "accuracy_pct": round(100 * p, 1),
            "ci95_pp": round(100 * ci, 1),
            "degenerate": sum(1 for it in items if it.get("degenerate")),
            "degenerate_kinds": _kind_counts(items),
            "truncated": sum(1 for it in items if it.get("finish_reason") == "length"),
            "errors": sum(1 for it in items if it.get("finish_reason") == "error"),
            "mean_completion_tokens": round(
                sum(it["completion_tokens"] or 0 for it in items) / n, 1) if n else 0,
        },
        "items": items,
    }


if __name__ == "__main__":
    main()
