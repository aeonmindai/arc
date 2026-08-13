#!/usr/bin/env python3
"""Short-prompt coherence + quick facts/math battery (RUN-161 quality gate).

Part A — coherence6: six short prompts at the measured-good chat.py sampling
settings (temperature=1.0, top_p=0.95, max 60 tok). This reproduces the "6/6
clean, EOS-terminated" result from commit 6102b4d84 (the June H200 session
found 2/6 looping at top_p=1.0, 6/6 clean at 0.95). PASS per prompt requires:
  - finish_reason == "stop"  (EOS emitted before the cap — loops never stop)
  - no repetition loop (4-gram repeated 4x)
  - expected keyword present, where one is defined

Part B — facts/math: the 30-item known-answer set from the branch's eval.py,
run GREEDY through the official encoder WITH a system prompt (eval.py's raw
hand-rolled format predates the encoder; this is the faithful version).

This is the session's smoke gate: if coherence6 < 5/6 or facts+math < 70%,
something is wrong with the build/bake — stop and debug before burning GPU
hours (see runbook "abort if" table).
"""
import argparse
import os
import time

import qlib

COHERENCE6 = [
    ("hello", None),
    ("What is the capital of France? Answer in one short sentence.", "paris"),
    ("What is 7 times 8?", "56"),
    ("Name the three primary colors.", "blue"),
    ("Who wrote Romeo and Juliet?", "shakespeare"),
    ("Write one short sentence about the ocean.", None),
]

FACTS = [
    ("What is the capital of France?", ["paris"]),
    ("What is the capital of Japan?", ["tokyo"]),
    ("What is the capital of Italy?", ["rome"]),
    ("What is the capital of Germany?", ["berlin"]),
    ("What is the capital of Spain?", ["madrid"]),
    ("What is the capital of Russia?", ["moscow"]),
    ("What is the capital of England?", ["london"]),
    ("What is the chemical symbol for gold?", ["au"]),
    ("What is the chemical symbol for oxygen?", ["o2", " o "]),
    ("What is the chemical symbol for iron?", ["fe"]),
    ("Water is made of hydrogen and what?", ["oxygen"]),
    ("What is the opposite of hot?", ["cold"]),
    ("What is the opposite of up?", ["down"]),
    ("What is the opposite of big?", ["small", "little"]),
    ("The sun rises in which direction?", ["east"]),
    ("What is the largest planet in the solar system?", ["jupiter"]),
    ("Who wrote Romeo and Juliet?", ["shakespeare"]),
    ("Who was the first president of the United States?", ["washington"]),
    ("How many sides does a triangle have?", ["three", "3"]),
    ("What color is the sky on a clear day?", ["blue"]),
    ("What is the freezing point of water in Celsius?", ["0", "zero"]),
    ("How many days are in a week?", ["seven", "7"]),
]
MATH = [
    ("What is 2 + 2?", ["4"]),
    ("What is 10 - 3?", ["7"]),
    ("What is 5 times 6?", ["30"]),
    ("What is 7 times 8?", ["56"]),
    ("What is 100 divided by 4?", ["25"]),
    ("What is 12 + 13?", ["25"]),
    ("What is 9 squared?", ["81"]),
    ("What is half of 50?", ["25"]),
]


def run_coherence6():
    items = []
    for prompt_text, keyword in COHERENCE6:
        p = qlib.encode_chat(prompt_text)
        try:
            r = qlib.complete(p, max_tokens=60, temperature=1.0, top_p=0.95)
        except Exception as e:
            r = {"text": f"[ERROR {e}]", "finish_reason": "error",
                 "prompt_tokens": None, "completion_tokens": 0, "seconds": 0}
        stopped = r["finish_reason"] == "stop"
        looped = qlib.looks_degenerate(r["text"])
        kw_ok = keyword is None or keyword in r["text"].lower()
        ok = stopped and not looped and kw_ok
        items.append({
            "prompt": prompt_text, "pass": bool(ok), "stopped": stopped,
            "looped": looped, "keyword_ok": kw_ok,
            "finish_reason": r["finish_reason"],
            "completion_tokens": r["completion_tokens"], "seconds": r["seconds"],
            "text": r["text"][:200],
        })
        print(f"  [{'PASS' if ok else 'FAIL'}] {prompt_text[:44]:44} -> {r['text'][:48]!r}")
    return items


def run_known_answers(name, qa):
    items = []
    for question, answers in qa:
        p = qlib.encode_chat(question)
        try:
            r = qlib.complete(p, max_tokens=30, temperature=0.0)
        except Exception as e:
            r = {"text": f"[ERROR {e}]", "finish_reason": "error",
                 "prompt_tokens": None, "completion_tokens": 0, "seconds": 0}
        low = r["text"].lower()
        hit = any(a in low for a in answers)
        items.append({"q": question, "pass": bool(hit), "want": answers,
                      "text": r["text"][:120], "seconds": r["seconds"]})
        print(f"  [{'PASS' if hit else 'FAIL'}] {question[:48]:48} -> {r['text'].strip()[:32]!r}")
    ok = sum(1 for it in items if it["pass"])
    print(f"  == {name}: {ok}/{len(items)}")
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(qlib.RESULTS_DIR, "coherence.json"))
    ap.add_argument("--skip-facts", action="store_true", help="coherence6 only (fast smoke)")
    args = ap.parse_args()

    qlib.ensure_dirs()
    qlib.health_or_die()
    t0 = time.time()

    print("=== A. coherence6 (sampled t=1.0 p=0.95, the 6102b4d84 gate) ===")
    c6 = run_coherence6()
    c6_ok = sum(1 for it in c6 if it["pass"])

    facts = math_items = []
    if not args.skip_facts:
        print("\n=== B. facts (greedy) ===")
        facts = run_known_answers("facts", FACTS)
        print("\n=== C. math (greedy) ===")
        math_items = run_known_answers("math", MATH)

    f_ok = sum(1 for it in facts if it["pass"])
    m_ok = sum(1 for it in math_items if it["pass"])
    summary = {
        "meta": qlib.run_meta({"eval": "coherence",
                               "sampling": "A: t=1.0/p=0.95; B/C: greedy"}),
        "summary": {
            "coherence6": f"{c6_ok}/{len(c6)}",
            "facts": f"{f_ok}/{len(facts)}" if facts else "skipped",
            "math": f"{m_ok}/{len(math_items)}" if math_items else "skipped",
            "seconds": round(time.time() - t0, 1),
        },
        "coherence6_items": c6,
        "facts_items": facts,
        "math_items": math_items,
    }
    qlib.write_json(args.out, summary)
    print(f"\nCOHERENCE: coherence6 {c6_ok}/{len(c6)}"
          + (f" | facts {f_ok}/{len(facts)} | math {m_ok}/{len(math_items)}"
             if facts else "")
          + f" | June anchor: 6/6 coherence (commit 6102b4d84)")


if __name__ == "__main__":
    main()
