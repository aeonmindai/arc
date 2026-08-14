# EXPECTED — published DeepSeek-V4-Flash numbers (comparison column)

Sourced 2026-08-12. The repo's own `research/` and `docs/` contain **no
V4-Flash benchmark table** (grepped for GSM8K/MMLU/AIME/HumanEval — only
architecture notes and the TAAC paper link). The numbers below come from the
official HF model card mirror, fetched live:
<https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash>
(paper: "DeepSeek-V4: Towards Highly Efficient Million-Token Context
Intelligence", arXiv:2606.19348).

**Model:** 284B total / 13B activated, 1M context. Reasoning modes: Non-think,
Think-High, Think-Max.

## Base model (pre-trained, few-shot)

| Benchmark | Setting | V4-Flash-Base | V4-Pro-Base |
|---|---|---|---|
| MMLU | 5-shot EM | 88.7 | 90.1 |
| MMLU-Pro | 5-shot EM | 68.3 | 73.5 |
| **GSM8K** | **8-shot EM** | **90.8** | 92.6 |
| MATH | 4-shot EM | 57.4 | 64.5 |
| HumanEval | 0-shot pass@1 | 69.5 | 76.8 |
| LongBench-V2 | 1-shot EM | 44.7 | 51.5 |

## Instruct model (Think-Max mode)

| Benchmark | Metric | V4-Flash | V4-Pro |
|---|---|---|---|
| MMLU-Pro | EM | 86.2 | 87.5 |
| GPQA Diamond | pass@1 | 88.1 | 90.1 |
| HumanEval | pass@1 | 91.6 | 93.5 |
| LiveCodeBench | pass@1 | 91.6 | 93.5 |
| SWE-bench Verified | resolved | 79.0 | 80.6 |
| MRCR 1M | MMR | 78.7 | 83.5 |
| CorpusQA 1M | acc | 60.5 | 62.0 |

## How our harness numbers map onto these

| Harness output | Published anchor | Caveat |
|---|---|---|
| `run_gsm8k.py` (instruct, 0-shot chat CoT, greedy, n=150) | GSM8K Base 8-shot EM **90.8** | Different protocol (instruct+0-shot vs base+8-shot); no published instruct non-think GSM8K score — **TODO** if DeepSeek publishes one. n=150 has ±6pp CI. |
| `run_ppl.sh` wikitext-2 ladder | none published | Self-ladder only: qtip2 (2.0 bpw trellis) vs q2k (2.56 bpw) vs q3k (3.44 bpw, opt-in). No FP8 reference ppl possible on one H200 (see below). |
| `run_longctx.py` | MRCR 1M / LongBench-V2 direction only | Our probes are ~0.3-12K tokens — they gate the compressor fix (337fd139a), not 1M-context claims. |
| `run_coherence.py` | none (internal gate) | June anchor: 6/6 coherent at t=1.0/p=0.95 (commit 6102b4d84); facts+math ~>80% expected. |

## Health heuristics for qtip2 (what "good" looks like)

- **PPL ladder ordering:** qtip2 ppl ≤ q2k ppl ×1.05 (the trellis at ~2.0 bpw
  should match or beat GGUF Q2K at 2.56 bpw; if qtip2 is >10% worse than q2k,
  the trellis/codebook path is suspect). q3k (if run) must be strictly better
  than both.
- **chunk64 vs chunk1024:** qtip2_c64 isolates the sliding-window path. A large
  gap (c1024 much worse) points at the compressed long-ctx path, not the quant.
- **GSM8K:** 2-bit experts + FP8 attention typically costs single-digit pp on
  GSM8K-class tasks. Within ~10-15pp of 90.8 = plausible; **<50% = breakage**
  (encoding, sampling, or quant bug — not "quantization loss").
- **Absolute KLD vs FP8 is impossible on this box:** the reference model
  (~300GB+ as served) exceeds 141GB HBM; the local checkpoint itself (~148GB)
  doesn't even fit unquantized. The bpw self-ladder + GSM8K delta is the
  designed substitute (see runbook preamble).
