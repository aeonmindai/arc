#!/usr/bin/env python3
"""Faithful chat client for DeepSeek-V4-Flash served on :1234 (via SSH tunnel).

This client follows the model's OFFICIAL encoding exactly (encoding/encoding_dsv4.py,
mirrored locally as encoding_dsv4.py):

  - prompt is built with `encode_messages(messages, thinking_mode=...)`
  - sent to the RAW /v1/completions endpoint (NOT /v1/chat/completions), so the
    server applies NO template of its own — the bytes we send are the bytes the
    model sees. This is 100% faithful regardless of the server's chat template.
  - the reply is parsed with `parse_message_from_completion_text(...)`.
  - sampling follows the model card / generation_config.json: temperature=1.0,
    top_p=1.0.

Thinking modes (official):
  - "chat"     : generation prompt ends `<｜Assistant｜></think>` → direct answer.
  - "thinking" : generation prompt ends `<｜Assistant｜><think>` → reasoning then answer.

Toggle with /think and /chat. Reasoning is shown dimmed but NOT fed back into
history (drop_thinking=True, matching the official multi-turn behavior).

BOS handling: encode_messages adds the bos token itself (add_default_bos_token).
The raw /v1/completions endpoint must therefore NOT auto-prepend another bos.
mistral.rs completions tokenizes the prompt as-is (add_special_tokens is not
applied to the user prompt for /v1/completions), so a single bos is correct.
If you ever see a doubled `<｜begin▁of▁sentence｜>`, set ADD_BOS=False below.

IMPORTANT: send ONE request at a time — no concurrent clients — until the
compressor xs_history is made sequence-managed; concurrent/overlapping requests
share a global compressor buffer and corrupt each other. Also run the server with
`--prefix-cache-n 0` so xs_history resets each request (see project memory).
"""
import json
import os
import sys
import time
import urllib.request
import urllib.error

from encoding_dsv4 import (
    encode_messages,
    parse_message_from_completion_text,
    eos_token,
    bos_token,
)

# Point at a remote box with: BASE_URL=http://<box-ip>:1234 python3 chat.py
BASE = os.environ.get("BASE_URL", "http://localhost:1234")
URL = f"{BASE}/v1/completions"
MODEL = "default"

# encode_messages already prepends bos; the raw completions endpoint must not add
# another. Flip to False only if you observe a doubled bos in server logs.
ADD_BOS = True

# Sampling: temperature follows the model card / generation_config.json (1.0).
# top_p is tightened to 0.95 (the official default is 1.0 = no truncation).
# RATIONALE (measured on H200, RUN-161 2-bit qtip2, 2026-06-24): 2-bit quant
# slightly flattens the logit tail, so at top_p=1.0 the model occasionally samples
# a derailing tail token (e.g. the <｜User｜> role token) and falls into a
# repetition loop ("UserUserUser…", "Red green blue…" ×N) that never emits EOS.
# top_p=0.95 nucleus truncation removes exactly that tail → 6/6 short-answer
# prompts return clean, correct, EOS-terminated replies (vs 2/6 looping at 1.0).
# This is principled nucleus sampling for quantized weights, NOT a repetition
# penalty. EOS itself is healthy (eos_token_id=1; "hello" stops at 10 tok). The
# official 1.0/1.0 defaults assume full-precision weights.
TEMPERATURE = 1.0
TOP_P = 0.95
MAX_TOKENS = 60

# A system prompt is REQUIRED in practice. A bare `<bos><｜User｜>...<｜Assistant｜>`
# with no system message is out-of-distribution for V4-Flash — trivial inputs like
# "hello" then degenerate into repetition / off-topic text that never emits EOS, so
# generation runs all the way to MAX_TOKENS (minutes at decode speed). With a system
# prompt the same input returns "Hello! How can I assist you today?" and stops
# cleanly in ~10 tokens. The official demos always pass one. Set to None to disable.
SYSTEM_PROMPT = "You are a helpful assistant."


def complete(prompt):
    body = json.dumps({
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        # Stop on eos so the parser receives a clean, terminated turn.
        "stop": [eos_token],
    }).encode()
    req = urllib.request.Request(
        URL, data=body, headers={"Content-Type": "application/json"}
    )
    resp = json.load(urllib.request.urlopen(req, timeout=600))
    choice = resp["choices"][0]
    usage = resp.get("usage", {})
    text = choice["text"]
    # The server strips the stop string; re-append eos so the official parser
    # (which asserts a terminating eos) accepts the turn.
    finish = choice.get("finish_reason")
    if not text.endswith(eos_token):
        text = text + eos_token
    return text, usage, finish


def main():
    thinking_mode = "chat"

    print("=" * 64)
    print("  DeepSeek-V4-Flash chat  (official encoder + raw /v1/completions)")
    print("  Ctrl-C quit | /reset clear | /think | /chat | /mode")
    print("=" * 64)
    try:
        urllib.request.urlopen(f"{BASE}/health", timeout=10)
        print(f"  connected to {BASE}  (mode: {thinking_mode}, "
              f"temp={TEMPERATURE}, top_p={TOP_P})\n")
    except Exception as e:
        print(f"  !! cannot reach server ({e}).")
        print("     tunnel: ssh -L 1234:localhost:1234 -N root@<box-ip>\n")
        sys.exit(1)

    messages = []
    if SYSTEM_PROMPT:
        messages.append({"role": "system", "content": SYSTEM_PROMPT})
    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            break
        if not q:
            continue
        if q == "/reset":
            messages = []
            if SYSTEM_PROMPT:
                messages.append({"role": "system", "content": SYSTEM_PROMPT})
            print("  [history cleared]\n")
            continue
        if q in ("/think", "/chat"):
            thinking_mode = "thinking" if q == "/think" else "chat"
            print(f"  [thinking_mode = {thinking_mode}]\n")
            continue
        if q == "/mode":
            print(f"  [thinking_mode = {thinking_mode}]\n")
            continue

        messages.append({"role": "user", "content": q})
        prompt = encode_messages(
            messages,
            thinking_mode=thinking_mode,
            drop_thinking=True,
            add_default_bos_token=ADD_BOS,
        )

        t0 = time.time()
        try:
            text, usage, finish = complete(prompt)
        except urllib.error.HTTPError as e:
            print(f"Bot: [http error {e.code}] {e.read().decode()[:300]}\n")
            messages.pop()
            continue
        except Exception as e:
            print(f"Bot: [error] {e}\n")
            messages.pop()
            continue
        dt = time.time() - t0

        try:
            parsed = parse_message_from_completion_text(text, thinking_mode)
        except Exception as e:
            # Fall back to raw text if the model produced malformed output
            # (e.g. truncated at max_tokens before eos).
            raw = text[: -len(eos_token)] if text.endswith(eos_token) else text
            print(f"Bot: [unparsed, finish={finish}] {raw.strip()}\n")
            messages.append({"role": "assistant", "content": raw.strip()})
            continue

        ntok = usage.get("completion_tokens", 0)
        tps = ntok / dt if dt else 0

        reasoning = parsed.get("reasoning_content") or ""
        content = parsed.get("content") or ""
        if reasoning.strip():
            print(f"  \033[2m[think] {reasoning.strip()}\033[0m")
        print(f"Bot: {content.strip()}")
        print(f"     ({ntok} tok in {dt:.1f}s, {tps:.1f} tok/s, finish={finish})\n")

        # Persist the assistant turn; drop_thinking=True drops this reasoning on
        # the next encode, matching official multi-turn behavior.
        msg = {"role": "assistant", "content": content}
        if reasoning:
            msg["reasoning_content"] = reasoning
        messages.append(msg)


if __name__ == "__main__":
    main()
