#!/usr/bin/env python3
"""Quick absolute-quality eval for the served 2-bit V4-Flash."""
import json, urllib.request, time

BASE_URL = "http://localhost:1234/v1"

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


def chat(question, max_tokens=30):
    """Use /v1/chat/completions with the chat template."""
    body = json.dumps({
        "model": "deepseek-ai/DeepSeek-V4-Flash",
        "messages": [{"role": "user", "content": question}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }).encode()
    r = urllib.request.Request(
        f"{BASE_URL}/chat/completions", data=body,
        headers={"Content-Type": "application/json"},
    )
    resp = json.load(urllib.request.urlopen(r, timeout=60))
    return resp["choices"][0]["message"]["content"]


def completion(prompt, max_tokens=10):
    """Fallback: use /v1/completions (no chat template needed)."""
    body = json.dumps({
        "model": "deepseek-ai/DeepSeek-V4-Flash",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "repetition_penalty": 1.1,
        "stop": ["\n\n", "\n"],
    }).encode()
    r = urllib.request.Request(
        f"{BASE_URL}/completions", data=body,
        headers={"Content-Type": "application/json"},
    )
    return json.load(urllib.request.urlopen(r, timeout=60))["choices"][0]["text"]


# Auto-detect: try chat first, fall back to completion
def gen(question):
    try:
        return chat(question)
    except Exception:
        return completion(f"Question: {question}\nAnswer:")


def run(name, items):
    ok = 0
    for question, answers in items:
        try:
            out = gen(question)
        except Exception as e:
            out = f"[err {e}]"
        low = out.lower()
        hit = any(a in low for a in answers)
        ok += hit
        print(f"  [{'PASS' if hit else 'FAIL'}] {question:52} -> {out.strip()[:30]:30} (want {answers})")
    print(f"  == {name}: {ok}/{len(items)} = {100*ok/len(items):.0f}%\n")
    return ok, len(items)


t0 = time.time()
print("=== FACTS ===")
f_ok, f_n = run("facts", FACTS)
print("=== MATH ===")
m_ok, m_n = run("math", MATH)
tot_ok, tot_n = f_ok + m_ok, f_n + m_n
print(f"OVERALL: {tot_ok}/{tot_n} = {100*tot_ok/tot_n:.0f}%   ({time.time()-t0:.0f}s)")
