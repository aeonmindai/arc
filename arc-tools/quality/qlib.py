#!/usr/bin/env python3
"""Shared helpers for the RUN-161 quality harness (arc-tools/quality).

Every script in this directory talks to a locally-served DeepSeek-V4-Flash
(mistralrs serve ... --prefix-cache-n 0) through the RAW /v1/completions
endpoint, using the model's OFFICIAL encoder (repo-root encoding_dsv4.py) —
exactly the chat.py machinery, so the bytes the model sees are faithful.

Design constraints inherited from RUN-161:
  - ONE request at a time (global compressor xs_history is not seq-managed).
  - Server must run with --prefix-cache-n 0.
  - Greedy = temperature 0 (mistral.rs maps it to argmax; June eval.py used it).
  - Sampled = temperature 1.0 / top_p 0.95 (the measured-good chat.py setting).

Stdlib only — no pip deps.
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request

# repo root (= two levels up from arc-tools/quality) holds encoding_dsv4.py
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from encoding_dsv4 import encode_messages, eos_token  # noqa: E402

BASE = os.environ.get("BASE_URL", "http://localhost:1234")
URL = f"{BASE}/v1/completions"
CHAT_URL = f"{BASE}/v1/chat/completions"

SYSTEM_PROMPT = "You are a helpful assistant."  # REQUIRED: bare prompts are OOD for V4-Flash

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
RESULTS_DIR = os.path.join(HERE, "results")


def ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)


def health_or_die(timeout=10):
    try:
        urllib.request.urlopen(f"{BASE}/health", timeout=timeout)
    except Exception as e:
        print(f"FATAL: cannot reach server at {BASE} ({e})", file=sys.stderr)
        print("       start it first (see GPU_SESSION_RUNBOOK.md) or set BASE_URL.", file=sys.stderr)
        sys.exit(2)


def encode_chat(user_msg, system=SYSTEM_PROMPT, thinking_mode="chat"):
    """Official V4 encoding for a single-turn chat."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user_msg})
    return encode_messages(
        messages,
        thinking_mode=thinking_mode,
        drop_thinking=True,
        add_default_bos_token=True,
    )


def complete(prompt, max_tokens=60, temperature=0.0, top_p=1.0, timeout=1800):
    """One raw completion. Returns dict: text, finish_reason, prompt_tokens,
    completion_tokens, seconds."""
    body = json.dumps({
        "model": "default",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stop": [eos_token],
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    resp = json.load(urllib.request.urlopen(req, timeout=timeout))
    dt = time.time() - t0
    ch = resp["choices"][0]
    usage = resp.get("usage", {})
    return {
        "text": ch["text"].replace(eos_token, ""),
        "finish_reason": ch.get("finish_reason"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "seconds": round(dt, 3),
    }


def chat_complete(user_msg, system=SYSTEM_PROMPT, max_tokens=60, temperature=0.0,
                  top_p=1.0, n_votes=None, vote_mode=None, timeout=3600):
    """One /v1/chat/completions request (server-side template — NOT the
    encoding_dsv4 path; run the runbook's template-parity probe before trusting
    it for scored evals).

    Arc Boost voting (PR #13, non-streaming only): pass `n_votes >= 2` to fan
    the request out to k chains decoded in the same forward-pass batch;
    `choices[0]` is the server-voted winner and the response's top-level
    `vote` field carries the tally (`vote_mode`: "majority" or
    "confidence_weighted", server default confidence_weighted).

    Returns dict: text, finish_reason, prompt_tokens, completion_tokens,
    seconds, vote (None unless the server voted).
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user_msg})
    payload = {
        "model": "default",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    if n_votes is not None:
        payload["n_votes"] = n_votes            # verified field: openai.rs ChatCompletionRequest
        if vote_mode is not None:
            payload["vote_mode"] = vote_mode    # "majority" | "confidence_weighted"
    body = json.dumps(payload).encode()
    req = urllib.request.Request(CHAT_URL, data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    resp = json.load(urllib.request.urlopen(req, timeout=timeout))
    dt = time.time() - t0
    ch = resp["choices"][0]
    usage = resp.get("usage", {})
    return {
        "text": (ch.get("message") or {}).get("content") or "",
        "finish_reason": ch.get("finish_reason"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "seconds": round(dt, 3),
        "vote": resp.get("vote"),
    }


def complete_stream(prompt, max_tokens=256, temperature=0.0, top_p=1.0, timeout=1800):
    """Streaming completion. Returns dict with text, ttft (s), decode_tok_s,
    completion_tokens, total seconds. Used by the speed probe."""
    body = json.dumps({
        "model": "default",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stop": [eos_token],
        "stream": True,
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    ttft = None
    n_chunks = 0
    text = []
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[len("data:"):].strip()
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            choices = obj.get("choices") or []
            if not choices:
                continue
            piece = choices[0].get("text", "") or choices[0].get("delta", {}).get("content", "")
            if ttft is None:
                ttft = time.time() - t0
            n_chunks += 1
            text.append(piece)
    total = time.time() - t0
    decode_time = max(total - (ttft or 0.0), 1e-6)
    # chunk count ~= token count for completions streaming (one token per chunk)
    decode_tok_s = (n_chunks - 1) / decode_time if n_chunks > 1 else 0.0
    return {
        "text": "".join(text).replace(eos_token, ""),
        "ttft_s": round(ttft, 3) if ttft is not None else None,
        "stream_chunks": n_chunks,
        "decode_tok_s": round(decode_tok_s, 2),
        "seconds": round(total, 3),
    }


def looks_degenerate(text, max_period=8, min_span=14):
    """Detect the classic 2-bit repetition loop ("UserUserUser...",
    "Red green blue red green blue ..."): >= `min_span` consecutive words
    forming a cycle of period <= `max_period`."""
    words = text.split()
    for p in range(1, max_period + 1):
        run = 0
        for i in range(p, len(words)):
            if words[i] == words[i - p]:
                run += 1
                if run + p >= min_span:
                    return True
            else:
                run = 0
    return False


def write_json(path, obj):
    """Atomic-ish JSON write (tmp + rename) so a crash never truncates results."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def run_meta(extra=None):
    meta = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "base_url": BASE,
        # NOTE: these are the CLIENT process's gates. The gates that matter are
        # on the SERVER process — record those via the runbook's serve log names.
        "client_env_gates": {
            k: os.environ.get(k)
            for k in (
                "ARC_FUSED_SINKHORN", "ARC_QUANT_ATTENTION", "ARC_V4_WINDOW_ONLY",
                "ARC_FORCE_NAIVE_SDPA", "ARC_TIME_DECODE", "ARC_V4_STANDARD_DENSE",
                "ARC_V4_NO_ABSORBED_DECODE", "ARC_QTIP_EXPERT_GREEDY",
                "ARC_MTP_LOG_ACCEPTANCE",
            )
            if os.environ.get(k) is not None
        },
    }
    if extra:
        meta.update(extra)
    return meta
