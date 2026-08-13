#!/usr/bin/env python3
"""Fused-Sinkhorn A/B: token-identity capture/compare (RUN-161).

The fused single-launch Sinkhorn kernel (commit 8dc4e9933, gate
ARC_FUSED_SINKHORN=1, mistralrs-core/src/models/dsv4_mhc.rs + cuda/sinkhorn.cu)
replaces a ~123-op candle chain that costs ~13k kernel launches per token. The
code comment gates it "until bit-identical-validated" — this script is half of
that validation (greedy TOKEN identity through the served model); the other
half is the perplexity bit-identity check driven by run_ppl.sh --sinkhorn-ab
(identical per-chunk ppl strings => identical f32 logits bits).

Protocol (server restarts are driven by the runbook, not this script):
  1. server WITHOUT the gate  -> run_sinkhorn_ab.py capture --label baseline
  2. server WITH ARC_FUSED_SINKHORN=1 -> ... capture --label fused
  3. run_sinkhorn_ab.py compare
Then speed_probe.py --label fused_sinkhorn gives the tok/s side of the A/B.

Greedy, fixed prompts, 128 tokens each: any single differing token fails.
"""
import argparse
import json
import os
import sys

import qlib

PROMPTS = [
    "Count from one to twenty, writing each number as a word.",
    "Explain in two sentences why the sky is blue.",
    "List the first eight prime numbers.",
    "Write a four-line poem about winter mornings.",
    "What is 17 multiplied by 23? Show the working.",
    # long-ctx prompt so the compressed path's sinkhorn use is covered too
    ("The water cycle moves water between oceans, atmosphere, and land through "
     "evaporation, condensation, precipitation, and runoff, driven by solar "
     "energy and gravity, endlessly circulating fresh water around the planet. "
     ) * 6 + "Summarize the passage above in one sentence.",
]


def path_for(label):
    return os.path.join(qlib.RESULTS_DIR, f"sinkhorn_{label}.json")


def capture(args):
    qlib.ensure_dirs()
    qlib.health_or_die()
    items = []
    for i, ptext in enumerate(PROMPTS):
        r = qlib.complete(qlib.encode_chat(ptext), max_tokens=128, temperature=0.0)
        items.append({"i": i, "prompt_head": ptext[:60], "text": r["text"],
                      "completion_tokens": r["completion_tokens"],
                      "finish_reason": r["finish_reason"], "seconds": r["seconds"]})
        print(f"  [{i + 1}/{len(PROMPTS)}] {r['completion_tokens']}tok in {r['seconds']}s")
    out = {"meta": qlib.run_meta({"eval": "sinkhorn_capture", "label": args.label}),
           "items": items}
    qlib.write_json(path_for(args.label), out)
    print(f"SINKHORN-CAPTURE[{args.label}]: {len(items)} greedy generations "
          f"-> {path_for(args.label)}")


def compare(args):
    pa, pb = path_for(args.a), path_for(args.b)
    for p in (pa, pb):
        if not os.path.exists(p):
            print(f"FATAL: missing capture {p}", file=sys.stderr)
            sys.exit(2)
    with open(pa) as f:
        a = json.load(f)
    with open(pb) as f:
        b = json.load(f)
    mismatches = []
    for ia, ib in zip(a["items"], b["items"]):
        if ia["text"] != ib["text"]:
            # locate first divergence for the report
            k = next((j for j, (x, y) in enumerate(zip(ia["text"], ib["text"])) if x != y),
                     min(len(ia["text"]), len(ib["text"])))
            mismatches.append({"i": ia["i"], "first_divergence_char": k,
                               "a_tail": ia["text"][k:k + 80], "b_tail": ib["text"][k:k + 80]})
    identical = not mismatches
    verdict = {
        "meta": qlib.run_meta({"eval": "sinkhorn_compare", "a": args.a, "b": args.b}),
        "summary": {
            "token_identical": identical,
            "prompts": len(a["items"]),
            "mismatches": len(mismatches),
            "note": ("PASS: greedy 128-tok outputs identical on all prompts. Combine with "
                     "the ppl bit-identity check (run_ppl.sh --sinkhorn-ab) before "
                     "flipping the gate on by default.") if identical else
                    ("FAIL: fused kernel diverges from candle path — keep "
                     "ARC_FUSED_SINKHORN off; file the mismatch details."),
        },
        "mismatches": mismatches,
    }
    qlib.write_json(os.path.join(qlib.RESULTS_DIR, "sinkhorn_verdict.json"), verdict)
    print(f"SINKHORN-AB: token_identical={identical} "
          f"({len(mismatches)} mismatching prompt(s) of {len(a['items'])})")
    sys.exit(0 if identical else 1)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture")
    c.add_argument("--label", required=True, choices=["baseline", "fused"])
    c.set_defaults(fn=capture)
    d = sub.add_parser("compare")
    d.add_argument("--a", default="baseline")
    d.add_argument("--b", default="fused")
    d.set_defaults(fn=compare)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
