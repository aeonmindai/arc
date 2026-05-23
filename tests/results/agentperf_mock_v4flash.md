# arc bench — agentperf (mock)

- **Model**: `deepseek-ai/DeepSeek-V4-Flash`
- **Vendor**: `mock`
- **SLO tier**: 2 (P25 output ≥ 60 tok/s, P95 TTFT ≤ 2.00 s)
- **Max users explored / cap**: 64 / 256
- **Saturation point**: **47 concurrent users** (largest K passing tier 2)
- **Total wall time**: 7.9 s

## Phases

| # | K | kind | P25 out tok/s | P95 TTFT (s) | samples | dur (s) | pass |
|---|---|------|---------------|--------------|---------|---------|------|
| 1 | 1 | ramp | 118.8 | 1.17 | 667837 | 0.6 | PASS |
| 2 | 2 | ramp | 117.6 | 1.19 | 482290 | 0.6 | PASS |
| 3 | 4 | ramp | 115.4 | 1.22 | 1080331 | 0.7 | PASS |
| 4 | 8 | ramp | 111.1 | 1.29 | 1303342 | 0.7 | PASS |
| 5 | 16 | ramp | 103.4 | 1.44 | 907241 | 0.7 | PASS |
| 6 | 32 | ramp | 90.9 | 1.73 | 793171 | 0.7 | PASS |
| 7 | 64 | ramp | 73.2 | 2.30 | 665174 | 0.6 | FAIL |
| 8 | 48 | bisect | 81.1 | 2.01 | 723055 | 0.6 | FAIL |
| 9 | 40 | bisect | 85.7 | 1.87 | 759361 | 0.6 | PASS |
| 10 | 44 | bisect | 83.3 | 1.94 | 716901 | 0.6 | PASS |
| 11 | 46 | bisect | 82.2 | 1.98 | 774224 | 0.6 | PASS |
| 12 | 47 | bisect | 81.6 | 1.99 | 752828 | 0.6 | PASS |
