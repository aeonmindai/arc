#!/usr/bin/env python3
"""Quick absolute-quality eval for the served 2-bit V4-Flash. Temp test, delete after."""
import json, urllib.request, time

URL = "http://localhost:1234/v1/completions"

# (prompt, accepted answers). Phrased as completions (base-model sweet spot).
FACTS = [
    ("The capital of France is", ["paris"]),
    ("The capital of Japan is", ["tokyo"]),
    ("The capital of Italy is", ["rome"]),
    ("The capital of Germany is", ["berlin"]),
    ("The capital of Spain is", ["madrid"]),
    ("The capital of Russia is", ["moscow"]),
    ("The capital of England is", ["london"]),
    ("The chemical symbol for gold is", ["au"]),
    ("The chemical symbol for oxygen is", ["o"]),
    ("The chemical symbol for iron is", ["fe"]),
    ("Water is made of hydrogen and", ["oxygen"]),
    ("The opposite of hot is", ["cold"]),
    ("The opposite of up is", ["down"]),
    ("The opposite of big is", ["small", "little"]),
    ("The sun rises in the", ["east"]),
    ("The largest planet in the solar system is", ["jupiter"]),
    ("Romeo and Juliet was written by", ["shakespeare"]),
    ("The first president of the United States was George", ["washington"]),
    ("A triangle has three", ["sides", "angles"]),
    ("The color of the sky on a clear day is", ["blue"]),
    ("The freezing point of water in Celsius is", ["0", "zero"]),
    ("The number of days in a week is", ["seven", "7"]),
]
MATH = [
    ("2 + 2 =", ["4"]),
    ("10 - 3 =", ["7"]),
    ("5 times 6 is", ["30"]),
    ("7 times 8 is", ["56"]),
    ("100 divided by 4 is", ["25"]),
    ("12 + 13 =", ["25"]),
    ("9 squared is", ["81"]),
    ("Half of 50 is", ["25"]),
]


def gen(prompt, n=6):
    body = json.dumps({"model": "deepseek-ai/DeepSeek-V4-Flash", "prompt": prompt,
                       "max_tokens": n, "temperature": 0}).encode()
    r = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=60))["choices"][0]["text"]


def run(name, items):
    ok = 0
    for prompt, answers in items:
        try:
            out = gen(prompt)
        except Exception as e:
            out = f"[err {e}]"
        low = out.lower()
        hit = any(a in low for a in answers)
        ok += hit
        print(f"  [{'PASS' if hit else 'FAIL'}] {prompt!r:48} -> {out.strip()!r:24} (want {answers})")
    print(f"  == {name}: {ok}/{len(items)} = {100*ok/len(items):.0f}%\n")
    return ok, len(items)


t0 = time.time()
print("=== FACTS ===")
f_ok, f_n = run("facts", FACTS)
print("=== MATH ===")
m_ok, m_n = run("math", MATH)
tot_ok, tot_n = f_ok + m_ok, f_n + m_n
print(f"OVERALL: {tot_ok}/{tot_n} = {100*tot_ok/tot_n:.0f}%   ({time.time()-t0:.0f}s)")
