# wave65-CQ — V4 decode re-RoPEs the whole compressed KV cache, every step, every layer

Parent system: ArcInfer / ArcAttention, with a dependency on ArcKV (`XsRollingCache`).

Status: **surfaced, not fixed.** Deliberately. See "Why this was not shipped".

Found while censusing pure-data-movement ops on the V4 batch=1 decode path for
PR #206. Every line number below is read from source at `cc5487ad3`. **No
hardware leg was taken; nothing here is measured on a GPU.** The byte counts are
arithmetic from tensor shapes, the launch counts are from reading the candle
call chain. Both are stated as derivations, not measurements.

---

## What happens

`Attention::forward` calls `compressed_kv_from_rows`
(`mistralrs-core/src/models/deepseek4.rs:1409-1422`) on **every decode step**,
for **every compressed layer**. That function hands the entire compressed KV row
set to `DeepSeekV2RotaryEmbedding::forward_at_positions`
(`mistralrs-core/src/layers.rs:1826-1843`), which re-applies compress-theta RoPE
to **all `t_c` rows** from scratch:

```rust
let x_nope = x.narrow(D::Minus1, 0, nope)?;
let x_pe   = x.narrow(D::Minus1, nope, rope_dim)?.contiguous()?;  // real copy
let positions = positions.to_dtype(DType::U32)?;                   // no-op; already U32
let cos = self.cos.index_select(&positions, 0)?;
let sin = self.sin.index_select(&positions, 0)?;
let rotated = candle_nn::rotary_emb::rope_i(&x_pe, &cos, &sin)?;
Tensor::cat(&[&x_nope, &rotated], D::Minus1)?.contiguous()         // 2 copies of the full [1,1,t_c,512]
```

The rotation applied to compressed row `j` is a pure function of `j`.
`compressed_kv_from_rows`'s own doc comment says so, in as many words:

> Row `j` sits at absolute position `j * ratio` and its rotation depends on
> nothing but `j`, which is why the rows can be cached pre-RoPE.

The rows **are** cached pre-RoPE, by the rolling compressor. The rotation on top
of them is not. So it is recomputed for every row on every token, even though
the rolling compressor only ever *appends*, and only appends once every `ratio`
tokens.

The one thing already memoized here is the position vector itself
(`compressed_row_positions`, `deepseek4.rs:3013-3037`) — the thread-local there
is the pattern the rest of this wants.

---

## Cost (derived from shapes, not measured)

V4 Flash geometry from `deepseek4.rs:5988-6047`: 43 layers, `compress_ratios`
giving **2 Standard + 41 compressed**; `head_dim` 512, `qk_rope_head_dim` 64.

At 2 048 ctx:

| | `t_c` | bytes moved / layer / token | layers | per token |
|---|---|---|---|---|
| CSA (ratio 4)   | 512 | ~1.05 MB across the narrow-copy and the two-input cat (read+write) | 21 | **~22 MB** |
| HCA (ratio 128) | 16  | ~33 KB | 20 | ~0.7 MB |

Launches, on top of the bytes:

* **123 / token** in copies alone — `narrow(..).contiguous()` plus the two-input
  `cat`, x41 layers.
* **82 / token** in the two `index_select` gathers, x41 layers.

Redundancy rate: for CSA the row set is unchanged on **3 of every 4** steps; for
HCA on **127 of every 128**. Essentially all of the above recomputes a value that
did not change.

---

## What a fix needs

1. A **post-RoPE cache on `XsRollingCache`**
   (`mistralrs-core/src/kv_cache/xs_rolling.rs:465`) alongside the pre-RoPE rows,
   invalidated only when `advance` (`:1065`) actually appends. A decode step then
   rotates 0 or 1 new rows and concatenates, instead of rotating `t_c`.
2. The `cos`/`sin` gathers are **separately memoizable** — they depend only on
   `(t_c, ratio, device)`, exactly like the already-memoized
   `compressed_row_positions` (`deepseek4.rs:3013-3037`) and `GRAPH_BLOCK_IDS`
   (`dsv4_attention.rs`). Those two are the established in-tree pattern to copy.

Item 2 is independent of item 1 and much cheaper to land; it is worth doing first.

---

## Why this was not shipped with PR #206

Two hazards, neither settleable without a GPU.

1. **Two row-set accessors, with different aliasing.** `Attention::forward`
   (`deepseek4.rs:1522-1534`) takes the result of `state.advance(...)` on the
   eager path, but `state.compressed_rows_fixed(rows)` (`xs_rolling.rs:912`)
   under graph decode — where the buffer has been pinned to a fixed capacity by
   `pin_comp_capacity` (`:867`) and the read is a constant-width narrow of a
   buffer whose tail slots are unwritten. A cache keyed on "the rows did not
   change" has to be right for **both**, across the pin, and in the case where
   `advance` returns `None`.

2. **A stale entry here is a wrong answer, not a slow one.** It silently corrupts
   the distant-context branch: coherent output, degraded long-context recall, and
   nothing downstream that catches it. That is the worst failure shape in this
   repo's history — cf. the RUN-161 long-context collapse, and the
   `ARC_V4_FP8_KV` default-on-without-a-GPU incident recorded on
   `v4_fp8_kv_enabled` (`deepseek4.rs:3149`), which killed every request on the
   first V4 forward that met real hardware.

So it wants the treatment PR #206 got: an identity proof written down, an
engagement counter that **reports** rather than assumes, a poison control that
has been observed going red, and an on-GPU A/B before the default flips. Landing
it **default-off behind a flag with the A/B harness attached** is the right first
PR; flipping the default is the second.

---

## Related

* **PR #206** removes 467 launches/token from the mask block on this same path,
  and names this as the single worst remaining item it did not fix.
* `CAPTURE_LANE`: op count prices at the GPU-side micro-kernel floor, so the ~205
  launches here are worth only ~0.2 ms. **The ~22 MB/token of redundant traffic
  is the bigger half of this item**, and unlike launch count it is not a retired
  lever.
* Filed here rather than as a GitHub issue because `aeonmindai/arc` has issues
  disabled (`gh repo view --json hasIssuesEnabled` -> `false`); `memory/mission/`
  is the tracker this project actually uses.
