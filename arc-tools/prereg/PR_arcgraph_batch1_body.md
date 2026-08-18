## 🔴 Default-on production correctness bug. From the first multi-sequence batch onward, only sequence 0 is actually decoded and every other user gets silently wrong output.

**Default-on in production, default-off in every measurement harness on the boxes** — `schedfix_run.sh` and `w53_probe.sh` both export `ARC_NO_DEDICATED_DECODE=1` to steer around it. So **nothing measured tonight exercised this path, and nothing that exercised it was measured.**

Reachability, from source: `PAGED_ATTN_CUDA = true` (`mistralrs_for_server_builder.rs:106`), so PagedAttention is on by default on CUDA — `--paged-attn on` is redundant. That selects the PagedAttention step arm → `graph_wrapped_forward` → `try_dedicated_decode` on **every non-prompt step**. And `ARC_NO_DEDICATED_DECODE` is an **opt-OUT**. Every dense model (Llama/Qwen/Mistral/Gemma family) on CUDA reaches it. V4 is exempt — it routes through `DefaultInstructions` and never executes this code.

## MEASURED — H200, qwen05, paged, same binary, one variable

The only difference between the two rows is the env var.

| | B=8 uniform | B=32 uniform |
|---|---|---|
| dedicated path **ON** | **71**/512 tokens | **95**/2048 tokens |
| dedicated path **OFF** | 512/512 | 2048/2048 |

`71 = 64 + 7×1` and `95 = 64 + 31×1`. Sequence 0 emits its full 64 tokens; **every other sequence emits exactly one and stops.**

Two different batch sizes agreeing on `64 + (B−1)` is a **structural signature**, not a throughput observation: a bandwidth or scheduling problem cannot produce that formula. Only "one sequence computed, the rest garbage" can. Zero errors, zero CUDA errors, zero glibc diagnostics in the ON arm — it fails silently.

## Root cause: the GEMVs have no batch dimension

```c
extern "C" void arc_launch_gemv_bf16_f32out(
    const void* weight, const void* input, void* output,
    int M, int K, cudaStream_t stream
) {
    dim3 grid((M + F32OUT_ROWS - 1) / F32OUT_ROWS);
```

The grid is over **output rows (vocab)**, not over sequences. Every projection in `decode_forward` — qkv, o_proj, gate, up, down, lm_head — is a single matrix-**vector** product. Only the elementwise and attention kernels ever consumed `bs`. The dedicated decode path is **structurally batch-1 only**, so at batch > 1 sequences 1..N−1 read uninitialised rows of every activation buffer and sample from garbage logits.

**The fix is a guard**: refuse the path for `batch_size > 1` and fall through to Candle. Falling back is *proven*-good, not assumed — that is exactly what the OFF arm above measures. Batching the GEMVs is performance work this guard makes safe to attempt.

### Verified after the guard, same harness

| | B=8 uniform | B=32 uniform |
|---|---|---|
| guarded, path live | **512/512** | **2048/2048** |

## I led with the wrong root cause and am saying so

Commits 1–2 fix two other real defects I found first. **They are necessary but were never sufficient**, and I reported the first as the root cause before checking what `decode_forward` used for its row count — one line away.

1. **`ensure_buffers` pinned capacity to the first batch ever decoded** (`decode_forward` then read its row count from the buffers). Consequence: `wrap_f32_logits` copied `batch_size*vocab*4` out of a `capacity*vocab*4` allocation — 4,254,208 bytes past the end at B=8 with capacity 1 — and the `cudaMemcpy` return was discarded, so the latched `cudaErrorInvalidValue` surfaced two steps later as `reshape_and_cache_kernel.cu:140: invalid argument`, an innocent bystander and the only `cudaGetLastError()` in the whole captured forward. **Growth confirmed live on the box: `capacity=1 → 8 → 32`.**
2. **Block-table staging wrote one row stride and the kernel read another** (`stage_paged_attn` flat-copied at the source stride; `staged_paged_attn` reports the capacity, which `pagedattention.cuh:229` uses as the stride). Now a pitched `cudaMemcpy2DAsync`. **Masked today** — with `num_seqs` pinned to 1, `seq_idx` never exceeded 0 and both strides agree at offset 0. It was unreachable, not unneeded, and it fires the moment batching lands.

`DecodeBuffers::batch_size` → **`capacity`**: renaming one of two conflated quantities turned a semantic bug into a compile error at all 13 call sites, forcing each to be reconsidered.

## Open, and deliberately not merged into the above

- **Residual 1-token divergence at batch=1.** B=8 *spread* reads 511/512 with the path live (both pre- and post-guard) but 512/512 with it off, same binary. So the batch-1 path is **not** bit-identical to Candle either. Not diagnosed; flagged rather than rounded away.
- **The V4 host heap corruption is NOT explained by any of this.** V4 never executes this code. Kept separate.
- **`check_dense_layer_inventory` is not on master.** `extract_model_weights` indexes `layers[1 + i*7 + k]` with no inventory check, so V4/Mixtral/Phi-2 extraction is silently mis-indexed rather than refused — which reframes comparisons between guarded and unguarded binaries as differing in whether a large bogus BF16 weight set was extracted and retained.

## Verification

CUDA compile `--features "cuda flash-attn"` **passes** (3m18s) — before that, the gated bodies were hand/grep-verified only, since macOS structurally cannot type-check them. `cargo fmt -p arc-cuda-graph --check` clean, scoped clippy `-D warnings` clean, **37/37** tests pass (7 new, encoding policy as arithmetic: the over-read asserted at exactly 4,254,208 bytes; a ladder dipping to 4 and 2 to prove no shrink-thrash; growth pinned at 4 reallocations across 9 steps).

Pre-registration and the A/B harness are in `arc-tools/`; outcomes were committed before the run, including the amendment predicting **against** my own first fix.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
