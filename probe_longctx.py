#!/usr/bin/env python3
"""Controlled long-context probe for V4-Flash (post-#24 compressor build).

Isolates the long-context path from the two confounds in live multi-turn chat:
  - conditioning on the model's OWN prior wrong answer
  - sampling tail-noise

Each probe is a SINGLE fresh request (server runs --prefix-cache-n 0, so
xs_history resets per request → each is a clean prefill of the whole prompt).

  A. short baseline      — known-answer Q, ctx < 128  → must be coherent+correct
  B. long coherence      — ~250 tok neutral filler + same known-answer Q
                           (ctx > 128 → compressed path engaged, NO needle, NO
                           self-generated garbage). Coherent+correct ⇒ the
                           long-context path itself does not break coherence.
  C. long needle recall  — fact stated at the START, ~250 tok filler, then asked
                           at the END. Tests distant-context retrieval across the
                           sliding window. (Lossy by design — a soft signal.)

Faithful encoding via the official encode_messages → raw /v1/completions.
Run from the box (localhost). ONE client at a time.
"""
import json
import sys
import time
import urllib.request

from encoding_dsv4 import encode_messages, eos_token

BASE = "http://localhost:1234"
URL = f"{BASE}/v1/completions"

# ~250 tokens of neutral, factually-correct filler. No instructions, no
# self-reference, nothing that would derail — purely to push ctx past the
# 128-token sliding window so the compressed/distant path is exercised.
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
    "how much rain a region receives over the course of a year."
)

PROBES = [
    ("A short baseline",
     "What is the capital of France? Answer in one short sentence."),
    ("B long coherence (ctx>128, no needle)",
     FILLER + "\n\nIgnoring the passage above, what is the capital of France? "
              "Answer in one short sentence."),
    ("C long needle recall (fact at start)",
     "Important: the secret access code is RIVER-7741. Keep it in mind.\n\n"
     + FILLER +
     "\n\nWhat was the secret access code I gave you at the very beginning?"),
]

SYSTEM = "You are a helpful assistant."


def run(label, user_msg):
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user_msg},
    ]
    prompt = encode_messages(messages, thinking_mode="chat",
                             drop_thinking=True, add_default_bos_token=True)
    body = json.dumps({
        "model": "default", "prompt": prompt, "max_tokens": 50,
        "temperature": 1.0, "top_p": 0.95, "stop": [eos_token],
    }).encode()
    req = urllib.request.Request(URL, data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    resp = json.load(urllib.request.urlopen(req, timeout=600))
    dt = time.time() - t0
    ch = resp["choices"][0]
    usage = resp.get("usage", {})
    ptok = usage.get("prompt_tokens", "?")
    ctok = usage.get("completion_tokens", 0)
    text = ch["text"].replace(eos_token, "").strip()
    print(f"\n### {label}")
    print(f"    prompt_tokens={ptok}  (sliding_window=128 → {'>128 LONG-CTX' if isinstance(ptok,int) and ptok>128 else 'short'})")
    print(f"    finish={ch.get('finish_reason')}  out={ctok}tok  {dt:.1f}s")
    print(f"    BOT: {text}")
    return ptok, text


if __name__ == "__main__":
    print("=" * 70)
    print("  V4-Flash controlled long-context probe (post-#24)")
    print("=" * 70)
    for label, msg in PROBES:
        try:
            run(label, msg)
        except Exception as e:
            print(f"\n### {label}\n    ERROR: {e}")
    print()
