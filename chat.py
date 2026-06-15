#!/usr/bin/env python3
"""Temp chat client for DeepSeek-V4-Flash served via the SSH tunnel on :1234.
Base model -> Q&A completion style. Delete after testing."""
import json
import sys
import time
import urllib.request
import urllib.error

URL = "http://localhost:1234/v1/completions"
MODEL = "default"


def ask(question: str) -> str:
    body = json.dumps({
        "model": MODEL,
        "prompt": f"Question: {question}\nAnswer:",
        "max_tokens": 60,
        "temperature": 0,
        "frequency_penalty": 0.6,
        "stop": ["\n\n", "\nQuestion:", "Question:"],
    }).encode()
    req = urllib.request.Request(
        URL, data=body, headers={"Content-Type": "application/json"}
    )
    resp = json.load(urllib.request.urlopen(req, timeout=300))
    choice = resp["choices"][0]
    usage = resp.get("usage", {})
    return choice["text"].strip(), usage


def main():
    print("=" * 60)
    print("  DeepSeek-V4-Flash chat  (Ctrl-C to quit)")
    print("  Base model: ask factual/short questions for best results")
    print("=" * 60)
    # connection check
    try:
        urllib.request.urlopen("http://localhost:1234/health", timeout=10)
        print("  connected to http://localhost:1234\n")
    except Exception as e:
        print(f"  !! cannot reach server ({e}).")
        print("     Is the SSH tunnel up? ssh -L 1234:localhost:1234 -N root@62.169.159.146\n")
        sys.exit(1)

    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            break
        if not q:
            continue
        t0 = time.time()
        try:
            text, usage = ask(q)
        except urllib.error.HTTPError as e:
            print(f"Bot: [http error {e.code}] {e.read().decode()[:200]}\n")
            continue
        except Exception as e:
            print(f"Bot: [error] {e}\n")
            continue
        dt = time.time() - t0
        ntok = usage.get("completion_tokens", 0)
        tps = ntok / dt if dt else 0
        print(f"Bot: {text}")
        print(f"     ({ntok} tok in {dt:.1f}s, {tps:.1f} tok/s)\n")


if __name__ == "__main__":
    main()
