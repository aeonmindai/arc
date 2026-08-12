#!/usr/bin/env python3
"""Needle-recall + long-context coherence ladder (RUN-161 quality battery).

Extends the branch's probe_longctx.py (same neutral filler, same faithful
encoding, same single-fresh-request-per-probe design) into a machine-scored
ladder over context length. This directly validates the June compressor fix
(commit 337fd139a): any context > 128 tokens (V4 sliding window) engages the
compressed/distant KV path that the old broken V4Compressor corrupted.

Probe grid (all GREEDY, deterministic; server must run --prefix-cache-n 0):
  ctx ~= 60 tok    baseline        coherence-Q            (window-only path)
  ctx ~= 300 tok   1x filler       coherence-Q + needle   (compressed engaged)
  ctx ~= 800 tok   3x filler       coherence-Q + needle
  ctx ~= 1900 tok  7x filler       coherence-Q + needle
  ctx ~= 5000 tok  19x filler      coherence-Q + needle   (--deep: 57x, ~15K)

Needle is stated at the very START (max distance); coherence question ignores
the filler ("what is the capital of France?"). PASS criteria are exact keyword
checks. Needle recall across the compressed path is lossy BY DESIGN — treat
coherence as the hard gate and needle as the soft signal (per probe_longctx.py).
"""
import argparse
import os
import time

import qlib

# Same neutral, factually-correct filler as probe_longctx.py (~250 tokens).
FILLER = (
    "The water cycle describes how water moves through the environment. "
    "Water evaporates from oceans, lakes, and rivers into the atmosphere as vapor. "
    "As the vapor rises and cools, it condenses into tiny droplets that form clouds. "
    "When the droplets grow heavy enough, they fall back to the surface as rain or snow, "
    "a process known as precipitation. Some of this water soaks into the ground and "
    "becomes groundwater, while the rest flows over land as runoff into streams and rivers, "
    "eventually returning to the ocean. Plants also release water vapor through their leaves "
    "in a process called transpiration. The sun provides the energy that drives the entire "
    "cycle, and gravity pulls the water downward at every stage. This continuous movement "
    "has been going on for billions of years and keeps fresh water circulating around the planet. "
    "Mountains, forests, and deserts all play a role in shaping local weather patterns and "
    "how much rain a region receives over the course of a year. "
)

NEEDLE = "Important: the secret access code is RIVER-7741. Keep it in mind.\n\n"
COHERENCE_Q = ("Ignoring the passage above, what is the capital of France? "
               "Answer in one short sentence.")
NEEDLE_Q = "What was the secret access code I gave you at the very beginning?"


def probe(label, user_msg, want):
    p = qlib.encode_chat(user_msg)
    t0 = time.time()
    try:
        r = qlib.complete(p, max_tokens=60, temperature=0.0)
    except Exception as e:
        r = {"text": f"[ERROR {e}]", "finish_reason": "error",
             "prompt_tokens": None, "completion_tokens": 0, "seconds": round(time.time() - t0, 1)}
    low = r["text"].lower()
    ok = (want in low) and not qlib.looks_degenerate(r["text"])
    rec = {
        "probe": label,
        "pass": bool(ok),
        "want": want,
        "prompt_tokens": r["prompt_tokens"],
        "finish_reason": r["finish_reason"],
        "completion_tokens": r["completion_tokens"],
        "seconds": r["seconds"],
        "text": r["text"][:250],
    }
    ptok = r["prompt_tokens"]
    path = "compressed" if isinstance(ptok, int) and ptok > 128 else "window-only"
    print(f"  [{'PASS' if ok else 'FAIL'}] {label:34} ctx={ptok} ({path}) "
          f"-> {r['text'][:60]!r}")
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(qlib.RESULTS_DIR, "longctx.json"))
    ap.add_argument("--deep", action="store_true", help="add a ~12K-token rung")
    args = ap.parse_args()

    qlib.ensure_dirs()
    qlib.health_or_die()

    reps = [1, 3, 7, 19] + ([57] if args.deep else [])
    items = []

    print("=== long-context ladder (greedy; sliding_window=128) ===")
    items.append(probe("baseline ctx<128 coherence", COHERENCE_Q, "paris"))
    for n in reps:
        body = FILLER * n
        approx = 300 * n
        items.append(probe(
            f"~{approx}tok coherence", body + "\n\n" + COHERENCE_Q, "paris"))
        items.append(probe(
            f"~{approx}tok needle@start", NEEDLE + body + "\n" + NEEDLE_Q, "river-7741"))

    coh = [it for it in items if "coherence" in it["probe"]]
    ndl = [it for it in items if "needle" in it["probe"]]
    coh_ok = sum(1 for it in coh if it["pass"])
    ndl_ok = sum(1 for it in ndl if it["pass"])

    summary = {
        "meta": qlib.run_meta({
            "eval": "longctx", "sampling": "greedy",
            "validates": "337fd139a faithful compressor (ctx>128 engages compressed KV path)",
        }),
        "summary": {
            "coherence": f"{coh_ok}/{len(coh)}",
            "needle": f"{ndl_ok}/{len(ndl)}",
            "hard_gate": "coherence must be clean at every rung; needle is a soft signal",
        },
        "items": items,
    }
    qlib.write_json(args.out, summary)
    print(f"\nLONGCTX: coherence {coh_ok}/{len(coh)} (HARD gate) | "
          f"needle {ndl_ok}/{len(ndl)} (soft) | "
          f"compressor fix 337fd139a {'VALIDATED' if coh_ok == len(coh) else 'SUSPECT'}")
    if coh_ok < len(coh):
        print("  -> failure triage: restart server with ARC_V4_WINDOW_ONLY=1 and re-run;"
              " if coherence recovers, the compressed branch is the culprit.")


if __name__ == "__main__":
    main()
