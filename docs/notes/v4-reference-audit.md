# V4 Reference Audit — Arc vs SGLang DeepSeek-V4

**Date**: 2026-08-14 · **Scope**: audit only, no production code changed.
**Trigger**: an external engine (colibrì) independently implemented DeepSeek-V4
Flash. Cross-checking it against Arc surfaced a real defect in the Lightning
Indexer (`EXTERNAL_FINDINGS.md` F1). One confirmed drift implies more, so this
document systematically re-derives Arc's V4 math against the authorities.

**Out of scope by assignment**: `mistralrs-core/src/models/dsv4_indexer.rs`,
`arc-cuda-graph/src/cuda/flashmlasparse/indexer_score.cu` and their tests — a
separate agent owned the F1 fix (landed as PR #26 during this audit). Indexer
observations found incidentally are collected in §10 for that agent.

## Authorities used

| Authority | Path | Status |
|---|---|---|
| SGLang V4 model | `research/code/06_foundation/sglang/python/sglang/srt/models/deepseek_v4.py` | primary |
| SGLang V4 NextN (MTP) | `.../srt/models/deepseek_v4_nextn.py` | primary |
| SGLang V2 MoE (V4 delegates to it) | `.../srt/models/deepseek_v2.py`, `.../srt/layers/moe/{topk,hash_topk}.py` | primary |
| SGLang mHC | `.../srt/layers/mhc.py`, `.../srt/layers/mhc_head.py` | primary |
| SGLang DSv4 compressor | `.../srt/layers/attention/dsv4/{compressor,compressor_v2,quant_k_cache,metadata_kernel}.py` | primary |
| SGLang V4 attention backend | `.../srt/layers/attention/deepseek_v4_backend.py` | primary |
| SGLang V4 JIT CUDA kernels | `.../sglang/jit_kernel/csrc/deepseek_v4/*.cuh` | primary (authoritative math) |
| SGLang V4 RoPE / config | `.../srt/layers/deepseek_v4_rope.py`, `.../srt/configs/deepseek_v4.py` | primary |
| SGLang EAGLE/NextN worker | `.../srt/speculative/{eagle_worker,eagle_info}.py` | primary |
| DeepSeek NSA reference | `research/code/03_per_token_speed/deepseek_v3_nsa/inference/model.py` | corroborating (V3.2-era) |
| Prior in-repo audit | `research/v4_audit.md` (2026-05-22) | hint — **demonstrably stale in places, see §4(d)** |
| External oracle | `memory/mission/EXTERNAL_FINDINGS.md` (colibrì v1.6.1 @df2c248) | empirical |

> **Reference availability caveat.** `research/code/**` is untracked in git. It is
> present in the primary clone but **absent from a fresh worktree/checkout**. Any
> future re-audit must run from the primary clone or re-vendor the reference.
> CI can never re-verify these comparisons.

> **The single missing artifact.** Five separate findings below bottom out in the
> same unavailable file: the real `config.json` from `deepseek-ai/DeepSeek-V4-Flash`
> — specifically its `rope_scaling` block, whether it publishes `n_group`, whether
> it spells the hash-layer key `num_hash_layers` or `n_hash_layers`, whether it
> publishes `swiglu_limit`, and its `compress_rope_theta`. Arc's own "real config"
> test fixture (`mistralrs-core/src/models/deepseek4.rs:3811-3845`) has **no
> `rope_scaling` key at all**, which cannot be the true config — the reference
> dereferences `rope_scaling["factor"]` unconditionally
> (`deepseek_v4.py:236, 245-247`) and would `KeyError`. **Fetching that one file is
> the highest-value, lowest-cost action arising from this audit.**

**Confidence legend** — used in every verdict:

| Tag | Meaning |
|---|---|
| `both-lbl` | Read both implementations line by line; formulas quoted from source on both sides. |
| `one-side` | Read one side fully; the other inferred from docs/comments/adjacent code. |
| `inferred` | Neither side read exhaustively for this point; reasoned from structure. |

---

## 0. Verdict summary, ranked by severity

| # | Finding | §  | Severity | Confidence |
|---|---|---|---|---|
| 1 | **MTP draft is seeded with `embed(T0)`, not the target's hidden state** — h and e branches carry the identical vector | 5(a) | **CRITICAL** | `both-lbl` |
| 2 | **MTP block attends over an empty per-chain KV cache** while using absolute RoPE positions | 5(c) | **CRITICAL** | `both-lbl` |
| 3 | **`b_sz > 1` prefill routes to `sinks_attn_varlen`, which drops the mask entirely** — no causality | 1(e) | **HIGH** | `both-lbl` |
| 4 | **Caller's `attention_mask` is discarded on every live V4 attention path** (padding + graph-decode) | 1(e) | **HIGH** | `both-lbl` |
| 5 | **swiglu clamp missing on 4 of 5 expert paths, including the shared expert unconditionally** | 4(e) | **HIGH** | `both-lbl` |
| 6 | **YaRN applied to Standard (ratio-0) layers by default**; reference forces `original_seq_len = 0` | 1(b) | **HIGH** | `both-lbl` |
| 7 | **3-D mHC bridge is live** — always on the MTP path, and a silent whole-model fallback | 3(d) | **HIGH** | `both-lbl` |
| 8 | **Sub-int8 MTP draft head reachable and unguarded** on every ISQ path | 6 | **HIGH** | `both-lbl` |
| 9 | **Draft and verify dispatch through different GEMV/GEMM kernel families**, no pin | 7 | **HIGH** | `both-lbl` |
| 10 | **`Qtip2b` 3-D experts default to Greedy + no rotation** while `QtipBitshift2` defaults to Viterbi | 9 | **HIGH** | `both-lbl` |
| 11 | **Group-limited routing is force-disabled for V4 upstream**; Arc branches on the config value | 4(b2) | **HIGH** (conditional) | `both-lbl` |
| 12 | `num_hash_layers` vs `n_hash_layers` key mismatch — reference may run 0 hash layers where Arc runs 3 | 4(d′) | **HIGH** | UNVERIFIABLE |
| 13 | `mtp.0.hc_head_*` ships in the checkpoint; Arc never loads it | 5(d) | **MED-HIGH** | `both-lbl` |
| 14 | FP8 K-cache scale is continuous, not UE8M0 power-of-two | 1(f) | **MED** | `both-lbl` |
| 15 | mHC `post`/`comb` round-tripped through bf16; reference keeps fp32 end to end | 3(b′) | **MED** | `both-lbl` |
| 16 | Compressor state/`ape`/`norm.weight` in model dtype; reference pins fp32 | 2(d) | **MED** | `both-lbl` |
| 17 | `sdpa_params.sliding_window` set inverted (Some on union layers, None on SWA layers) | 1(e) | **MED** (inert at hd=512) | `both-lbl` |
| 18 | `attn_sink` shaped with global head count under TP > 1 | 1(d) | **MED** (TP>1 only) | `both-lbl` |
| 19 | MTP fast path is temperature-blind — hijacks stochastic sampling, fabricates `logprob: 0.0` | 5(e) | **MED** | `both-lbl` |
| 20 | `hc_mult` / `hc_eps` / `hc_sinkhorn_iters` config fields are dead; runtime hardcodes | 3(g) | **LOW** (latent) | `both-lbl` |
| 21 | `norm_topk_prob` not parsed; renormalization keyed off scoring function instead | 4(b3) | **LOW** | `both-lbl` |
| 22 | Compressed KV not FP8-QAT-simulated | 1(f) | **LOW** | `both-lbl` |
| 23 | Q per-head RMS accumulated in bf16, not fp32 | 1(b) | **LOW** | `both-lbl` |
| 24 | `mscale` multiplied into sin/cos; reference uses unit-magnitude `polar` | 1(b) | **UNVERIFIABLE** | `both-lbl` |
| 25 | reference `attn_sink` kernel contract (denominator-only? scaled?) | 1(d) | **UNVERIFIABLE** | `one-side` |

**Clean** (verified matching, no action): MLA absorption form, nope/pe index split,
interleaved RoPE convention, norm-before-rope order, per-layer θ selection,
compressed-block RoPE positions, softmax scale, sliding-window-on-all-ratios,
block causality, union-in-one-softmax, KV layout ordering, compressor pooling
formula, `overlap_transform`, ape hotfix, raw∪compressed double-counting,
Sinkhorn (all of it), `hc_scale` 3-way blend, 4-D residual threading, `hc_head`,
mHC norm placement, gate-bias-selection-only, sqrtsoftplus, normalize→scale order,
`tid2eid` hash routing, shared-expert ordering, greedy verify/accept.

---

## 1. MLA attention

### (a) Absorption form — **MATCHES** · `both-lbl`

V4 has no `kv_b_proj` / `W_UK` / `W_UV`. The single fused `wkv: hidden → head_dim(512)`
produces one MQA head that *is* the latent, and K aliases V. There is no
absorb-vs-materialize choice to make on either side.

Reference `models/deepseek_v4.py:306-312` (`self.wkv = ReplicatedLinear(hidden_size, head_dim, bias=False)`),
`:214` (`assert config.num_key_value_heads == 1`), and
`layers/attention/deepseek_v4_backend.py:970-971` (`assert k is v, "DeepseekV4 shares k and v"`).

Arc `deepseek4.rs:1238-1245, 1273` — same structure; `let v = k.copy()?;` (`copy` not
`clone` is deliberate, documented at `:1268-1272` as avoiding a PagedAttention
`RwLock` self-deadlock).

`absorbed_mqa_decode` (`dsv4_attention.rs:109-138`) is **not** an MLA absorb — it is a
GEMM-batching rewrite that folds the head axis into the GEMM M-dimension so K/V are
read once instead of `repeat_kv`-expanded 64×. It is a real-number identity with
`sinks_attn_cpu`'s repeat_kv path (`attention/backends/sinks.rs:181-192`), pinned by
`absorbed_decode_matches_repeat_kv_reference` (`dsv4_attention.rs:698-744`) at a
1.5e-3 F16-noise-floor tolerance.

> Minor: the absorbed gate requires `sinks.is_some()` (`dsv4_attention.rs:240-243`), so
> a checkpoint without `attn_sink` silently takes a different code path. Fixture-only today.

### (b) RoPE order, nope/pe split, θ, YaRN — **DIVERGES** · `both-lbl`

**Split and order — MATCHES.** Reference `layers/deepseek_v4_rope.py:217`:
`rope_start = HEAD_DIM - ROPE_DIM` ⇒ nope = `[0,448)`, pe = `[448,512)` — rope occupies
the *tail* of the same 512-dim head vector. Norm applied **before** rope (`:244-245`
then `:267-268`; docstring contract at `:297-300`). Arc `deepseek4.rs:1084-1102` splits
`&[qk_nope, qk_rope]`, ropes the tail, re-cats; `qk_nope_head_dim() = head_dim -
qk_rope_head_dim = 448` (`:340-343`). Norm-then-rope: q per-head RMS `:1223-1231`,
`kv_norm` `:1240`, rope `:1249`.

**Interleaved vs half-split — MATCHES.** Reference loads `2*pair_offs` /
`2*pair_offs+1` (`deepseek_v4_rope.py:222-231`), `is_neox_style=False`
(`deepseek_v4.py:228`); NSA corroboration `inference/model.py:390`
(`view_as_complex(... .view(*shape[:-1], -1, 2))`). Arc uses `rope_i` — the
interleaved variant — at `layers.rs:1621, 1710`.

**θ per layer — MATCHES.** Reference `deepseek_v4.py:220`:
`rope_base = config.compress_rope_theta if self.compress_ratio else rope_theta`.
Arc `deepseek4.rs:2969-2988` builds two tables, dispatched per layer at `:2995-3010`;
MTP uses standard θ (`:2626-2634`), matching `COMPRESS_RATIO_NEXTN_LAYER = 0`.

**Inverse RoPE on the attention output — MATCHES.** Reference `deepseek_v4.py:619-625`:
`fused_rope_inplace(o[..., -qk_rope_head_dim:], ..., inverse=True)`. Arc
`deepseek4.rs:1454-1458` calls `forward_inverse_tail`, which negates sin
(`layers.rs:1670`) — the conjugate rotation, matching `IS_INVERSE`
(`deepseek_v4_rope.py:107-108`). *Cosmetic:* the comment at `deepseek4.rs:1415-1421`
claims the inverse rope "is implicit here" and is contradicted 33 lines later by the
code that performs it. Delete it before someone believes it.

**Compressed-KV rope positions — MATCHES.** Arc `deepseek4.rs:1144-1149` assigns
compressed entry `j` → absolute position `j*ratio`. Reference
`layers/attention/dsv4/metadata_kernel.py:39, 51`: `c4_positions = position & (~3)`,
`c128_positions = position & (~127)`; for block `b` the writing token is at
`position = ratio·b + ratio − 1`, so `position & ~(ratio−1) = ratio·b`. Identical.
Independently corroborated from the kernel side:
`fused_norm_rope_v2.cuh:237-250` (`position = plan.seq_len - params.compress_ratio`)
with `c_plan.cuh:200-212` emitting `seq_len = position + 1` at block completion.

**⚠️ YaRN on Standard layers — DIVERGES (HIGH).** Reference `deepseek_v4.py:234-248`:

```python
if self.compress_ratio:
    original_seq_len = rope_scaling["original_max_position_embeddings"]
else:
    original_seq_len = 0
```

and `deepseek_v4_rope.py:47-53` skips the interpolation branch entirely when
`original_seq_len == 0`. So ratio-0 layers get plain `1/θ^(2i/d)`.

Arc `deepseek4.rs:2969-2982` applies YaRN to Standard layers **by default**; the
reference behavior is behind an opt-in env var:

```rust
rope_scaling: if std::env::var_os("ARC_DISABLE_YARN_STD").is_some() {
    None
} else {
    cfg.rope_scaling.clone()
},
```

Same divergence duplicated for the MTP block at `:2627-2632`. With `factor=16` this
compresses low-frequency rotation ~16× on layers 0, 1, 42 and the MTP block — exactly
the layers Arc's own module docs (`dsv4_attention.rs:12-28`) blame for the RUN-161
long-context repetition collapse. **The default should be flipped**, with the env var
restoring the old path.

When the YaRN body *is* used (compressed layers) the math matches: Arc
`layers.rs:1541-1570` (`freq_inter·(1−mask) + freq_extra·mask`) is algebraically
identical to `freqs/factor*(1−smooth) + freqs*smooth`, and
`yarn_find_correction_range` (`layers.rs:1495-1506`) matches `find_correction_range`
(`deepseek_v4_rope.py:35-38`) including the clamps and the `min == max` guard.

**mscale — UNVERIFIABLE.** Reference returns `torch.polar(torch.ones_like(freqs), freqs)`
(`deepseek_v4_rope.py:57`) — **unit magnitude, no mscale ever**. Arc `layers.rs:1573-1576`
multiplies `sin`/`cos` by `yarn_get_mscale(f, mscale) / yarn_get_mscale(f, mscale_all_dim)`.
Neutral iff the config's `mscale == mscale_all_dim`. Settled by the missing `config.json`.

**Q RMS precision — LOW.** Reference upcasts to fp32 for the reduction
(`deepseek_v4_rope.py:206-209`); Arc reduces in `q`'s dtype (bf16 in deployment,
`deepseek4.rs:1223-1231`), losing ~3 bits over 512 squared values. One-line fix.

**`compress_rope_theta` default — LOW.** Arc defaults to `160000.0`
(`deepseek4.rs:311-313`); the SGLang dataclass defaults to `40000`
(`configs/deepseek_v4.py:104`). Only matters if the key is absent, but the 4×
discrepancy should be reconciled against the real config.

### (c) Softmax scale — **MATCHES** · `both-lbl`

Reference, two independent sites, both pure `head_dim^-0.5` with **no mscale**:
`deepseek_v4.py:194,200` (`self.softmax_scale = self.head_dim**-0.5`) and
`deepseek_v4_backend.py:344-348` (with `assert head_dim == 512`), passed at `:1056`.

Arc `deepseek4.rs:357-366`: `1.0 / (self.head_dim as f32).sqrt()` = `0.0441942`.
Exact match, applied exactly once (`dsv4_attention.rs:127`, `sinks.rs:184`).

This is a *correct deliberate divergence from V2/V3*, which apply `* mscale * mscale`
(`inference/model.py:434-437`). Arc's comment correctly identifies the removal as a
RUN-161 fix.

### (d) `attn_sink` semantics — **MATCHES on Arc's side; reference kernel UNVERIFIABLE** · `one-side`

**Arc, fully verified** (`mistralrs-quant/src/utils/ops.rs:1894-1918`):

```rust
let sink_val = sinks_vals[sinks_offset + h];
let mut max_val = sink_val;
for k in 0..k_len { let v = logits[row_start + k]; if v > max_val { max_val = v; } }
let mut sum = (sink_val - max_val).exp();
for k in 0..k_len { let e = (logits[row_start + k] - max_val).exp(); out_row[k] = e; sum += e; }
```

Three unambiguous properties: **denominator-only** (output has exactly `k_len` columns;
the sink contributes to `sum` and no value vector — the gpt-oss attention-sink
convention); **unscaled** (logits arrive pre-multiplied by `softmax_scale`, but
`sink_val` is exponentiated raw); **per-head** (`h = (row / q_len) % num_heads`, shape
hard-checked at `ops.rs:2178-2184`). The mask is folded in before the sink comparison
(`ops.rs:2158-2162`), so `-inf` columns are excluded and the sink still supplies a
finite denominator — which is what keeps fully-masked decode rows from NaN-ing
(`hca_union_decode_emits_finite`, `dsv4_attention.rs:748-772`).

Applied on **every** layer including Standard (`deepseek4.rs:1028-1034`, no ratio gate);
reference agrees (`assert attn_sink is not None`, `deepseek_v4_backend.py:1035`, on the
shared path for all three ratios). Load shape/dtype match: reference
`deepseek_v4.py:288` `nn.Parameter(torch.empty(n_heads, dtype=torch.float32))`.

**Reference kernel contract — UNVERIFIABLE.** `deepseek_v4_backend.py:1049-1064` passes
`attn_sink` into `flash_mla.flash_mla_with_kvcache`, which is **not vendored anywhere
under `research/`** (no `*flash_mla*` file, no `attn_sink` in any `.cu`/`.cuh`). Whether
the kernel adds `sink` or `sink * softmax_scale`, and whether it emits a zero-value
column, cannot be read from the code present. The denominator-only/unscaled/per-head
reading is the only one consistent with a bare `[n_heads]` fp32 parameter and with
`softmax_scale` passed as a separate argument.

*Artifact that settles it:* `flash_mla/flash_mla_interface.py` + the split-KV MLA kernel
from `deepseek-ai/FlashMLA` at SGLang's pinned commit. Or a 5-line numerical probe:
run with `attn_sink = [0.0]*n_heads` vs `[large]*n_heads` on a 1-key cache and check
whether the output scales by `1/(1+exp(s))` or `1/(1+exp(s·scale))`.

**⚠️ TP sharding — DIVERGES (MED).** `deepseek4.rs:1031` shapes the sink with the
**global** head count while `:1037` computes the per-rank count that `q` carries. At
`world_size > 1` the shape check in `softmax_with_sinks` bails. The reference sidesteps
this by keeping `q_padded` full-width through the kernel and slicing the *output*
(`deepseek_v4.py:586-589`, `:618`).

### (e) Causal masking, prefill vs decode, sliding window — **MATCHES semantically; DIVERGES on caller-mask handling** · `both-lbl`

**Sliding window is a hard local mask on the main attention, all ratios — MATCHES.**
This is the key semantic question and Arc gets it right. The reference realizes
causality and windowing through *index selection*, for every `compress_ratio`
including 0 — `deepseek_v4_backend.py:1175-1188` (`get_swa_page_indices`) with
`SWA_WINDOW = 128` (`:67`) and `swa_topk_lengths = clamp(seq_lens_casual, max=128)`
(`:1146`), fed to `flash_mla` unconditionally at `:1058`. Prefill causality comes from
`expand_extend_with_same_length` (`:1106-1121`) giving each prefill query its own causal
length — **prefill and decode are the identical mechanism**, decode being `qo_len == 1`.

Arc `dsv4_attention.rs:189-204` expresses the same set as an additive mask:
`q0 + r − window < j <= q0 + r`. Window 128 both sides (`deepseek4.rs:295-297` vs
`configs/deepseek_v4.py:102`). The Standard-layer-is-SWA fix is correct and well tested
(`standard_prefill_is_sliding_window_masked` `:331-371` with a negative control;
`standard_decode_window_boundary_exact` `:424-482` sweeping `t_k ∈ {w−1, w, w+1, 2w, 4w+1}`).

**Compressed branch is not windowed, causal over completed blocks — MATCHES.**
Reference `metadata_kernel.py:52, 75-77` (`valid_mask = offsets < c128_seq_lens_raw`
with `seq_len = pos+1`); Arc `dsv4_attention.rs:213-217`
(`threshold = floor((pos+1)/ratio)`). Identical. The union is a single softmax with a
single sink on both sides.

**⚠️ Caller's `attention_mask` is discarded — DIVERGES (HIGH).** `dsv4_attention` takes
`attention_mask: Option<&Tensor>` (`dsv4_attention.rs:158`) and reads it in exactly one
place — the env-gated pre-fix branch at `:167-171`
(`ARC_V4_STANDARD_DENSE`). On the live path the mask handed to SDPA at `:245-252` is the
*locally built* `mask` from `:230-232`, encoding only causal ∧ window ∧ block-causality.
Two consequences:

1. **Batched prefill drops padding masks.** Callers pass a real mask at
   `deepseek4.rs:1342, 1357, 1407`; for `b > 1` prefill, `CausalMasker` marks pad columns
   `-inf` and `dsv4_attention` throws it away, so every sequence attends to its
   neighbours' padding. `deepseek4.rs:1352` even asserts `attention_mask.is_some()`
   immediately before passing a mask that will be ignored.
2. **CUDA-graph decode is unmasked by construction.** `:1383-1392` passes `gmask`;
   the comment at `:1377-1381` acknowledges the gap but attributes it to
   `set_graph_mode_mask` not being wired — wiring it will not help, because the mask is
   discarded regardless. Additionally the graph path builds `k_full` of fixed width
   `cap = sliding_window`, so `q0 = t_k − t_q = cap − 1` treats a rolling window as if it
   began at absolute position 0, mis-basing the compressed-block threshold once the true
   position exceeds `cap`.

Fix is one line: broadcast-add `attention_mask` into `mask` when `Some`. Additive masks compose.

**⚠️ `b_sz > 1` prefill takes a mask-free varlen path — DIVERGES (HIGH).** Independent of
the above. `attention/backends/sinks.rs:32-49`:

```rust
let is_varlen = b_sz > 1
    && flash_params.is_some_and(|fp| fp.cumulative_seqlens_k.contains_key(&q.device().location()));
if is_varlen {
    return sinks_attn_varlen(q, k, v, sinks, flash_params.unwrap(), sdpa_params, window_size);
}
```

`sinks_attn_varlen` (`sinks.rs:101-107`) **has no `mask` parameter at all**. At
`head_dim = 512` the flash-sinks kernels are unavailable (`:68` — head dim must be one of
64/80/96/112/128/192/256), so it falls to `sinks_attn_cpu_varlen`, which calls
`sinks_attn_cpu(&qi, &ki, &vi, sinks, None, sdpa_params)` (`:228`) — **`mask = None`, no
causality whatsoever**. `cumulative_seqlens_k` is populated whenever the `flash-attn`
feature is active (`pipeline/inputs_processor.rs:331, 356-364, 498`), which is the
documented production build. It will likely error rather than silently miscompute
(`sinks_attn_varlen:113-120` calls `k.squeeze(0)` on a `[B,1,T_k,D]` tensor; candle's
`squeeze` is a no-op when the dim size ≠ 1, so the subsequent `narrow(0, kv_start, kv_len)`
on a size-1 axis fails). Either way, **V4 batch-prefill with `b_sz > 1` is broken today.**
The V4 dispatch should never route to the varlen sinks path — it owns its own mask.

**⚠️ `sdpa_params.sliding_window` is set backwards — DIVERGES (MED, inert today).**
`deepseek4.rs:1059-1063`:

```rust
sliding_window: if compress_ratio != CompressRatio::Standard { Some(cfg.sliding_window) } else { None },
```

Inverted. Standard layers *are* pure sliding-window and get `None`; CSA/HCA layers attend
over a union whose compressed half must **not** be windowed, and they get `Some(128)`.
`sinks_attn` forwards this as `window_size` to the flash-sinks kernels (`sinks.rs:29,78,90`),
where it would be applied to the concatenated `[raw ++ compressed]` axis. Inert only
because `head_dim = 512` fails the `flash_sinks_ok` guard. **A landmine for anyone who
later adds a 512-wide flash-sinks kernel.**

**Doc correction.** `dsv4_attention.rs:40-45` claims dense-over-union is "a correct (just
slower) superset" beyond `ctx > 2048`. The first half is right (dense == sparse exactly
for `ctx ≤ index_topk·ratio`); the second half is **wrong** — attending to compressed
entries the reference discards changes the softmax denominator and redistributes mass.
Softmax is not monotone in the key set. Reword to "exact for `ctx ≤ index_topk·ratio`; an
approximation beyond it." HCA has no top-k, so dense is exact at every length there.

### (f) KV latent layout — **layout MATCHES; FP8 scale quantization DIVERGES** · `both-lbl`

**Layout — MATCHES.** Reference `layers/attention/dsv4/quant_k_cache.py:76-93`
(`hidden_dim == 512`, `dim_nope = 448`, `dim_rope = 64`, `tile_size = 64`) and
`mem_cache/deepseek_v4_memory_pool.py:105-111`:

```python
assert bytes_per_token == 448 + 64 * 2 + 8, (
    "DSV4 KV layout: qk_nope_head_dim FP8 (448) + qk_rope_head_dim BF16 "
    "(64*2) + nope FP8 scales + scale_pad = 584 bytes/token")
```

So: one 512-dim MQA latent per token per layer, **nope at `[0,448)` FP8-E4M3, rope at
`[448,512)` BF16** — the rope dims live *inside* the same 512-vector, not appended
outside. `head_dim_v = 512`; V reads the same buffer. Arc stores the same
`[B,1,T,512]` ordering at full BF16/F16 precision. That is a storage-efficiency gap
(584 B/token vs 1024 B/token), not a correctness gap.

**⚠️ FP8 scale is not rounded to a power of two — DIVERGES (MED).** Reference
`quant_k_cache.py:48-69`:

```python
scale = max_abs_clamped / FP8_MAX
ceil_log2 = tl.math.ceil(tl.log2(scale))
scale_pow2_fp32 = tl.exp2(ceil_log2)
...
scale_uint8 = (ceil_log2.to(tl.int32) + 127).to(tl.uint8)
```

The scale is **UE8M0** — a pure power of two, one exponent byte; `ceil` also guarantees
`x_scaled` never clips. Arc `deepseek4.rs:1967-1969`:

```rust
let amax = kb.abs()?.max_keepdim(D::Minus1)?;
let scale = (amax / 448.0)?.affine(1.0, 1e-12)?;
```

A continuous f32 scale, no `2^ceil(log2(·))`. Tile size 64 and `FP8_MAX = 448` both
match; only the scale grid differs — by up to 2×, which shifts every value onto a
different point of the 3-mantissa-bit FP8 grid. **The model is FP8-QAT-trained against
the UE8M0 grid**, so Arc feeds out-of-distribution K on every layer, every token.
Cheap fix: one `log2/ceil/exp2` before the division.

Two riders: the `ARC_GPU_ACT_QUANT=1` path (`:1977-1983`) hand-rolls E4M3 and its
subnormal handling deserves a targeted test; and Arc never applies `act_quant_kv_nope`
to the **compressed** KV (`:1117-1151`), while the reference stores compressed KV through
the same `NopeFp8RopeBf16Pack` path (`deepseek_v4_memory_pool.py:675-685, 792-800`).

---

## 2. V4Compressor / compressed-KV (CSA + HCA)

> **Authority note.** The Python `compressor.py` is a thin dispatcher; the authoritative
> math is in the JIT CUDA it loads — `jit_kernel/csrc/deepseek_v4/{c4_v2,c128_v2,
> fused_norm_rope_v2,c_plan}.cuh` — plus `jit_kernel/dsv4/{compress,gemm}.py`.

### (a) Pooling weights — **MATCHES** · `both-lbl`

There is **no sigmoid, no silu, no mean**. The pooling is a **per-channel softmax over
the slot axis**, with `ape` added to the score as a pre-softmax bias.

Reference `c4_v2.cuh:138-167`:

```c
score_fp32[i][j] = cast<float>(score[j][i]) + cast<float>(bias[j][i]);
...
float max_value = score[0];
for (j=1..8) max_value = fmaxf(max_value, score[j]);
for (j=0..8) {
  const auto exp_score = expf(score[j] - max_value);
  sum_product   += cast<float>(kv[j][i]) * exp_score;
  sum_exp_value += exp_score;
}
result[i] = cast<OutFloat>(sum_product / sum_exp_value);
```

`i` = channel within `head_dim`, `j` = slot ⇒ softmax normalizes **over slots,
independently per channel**. Identical in `c128_v2.cuh:141-173`.

Value/score provenance: one fused linear `kv_score = linear_bf16_fp32(x, wkv_gate.weight)`
(`compressor.py:354`) of width `2·coff·head_dim`, with load-time fusion
`torch.cat([kv, wgate], dim=0)` (`deepseek_v4.py:1651`) ⇒ **first half = value (`wkv`),
second half = score (`wgate`), raw, no activation**. Corroborated by the buffer layout
comment `c4_v2.cuh:4-5` (`| kv overlap | kv | score overlap | score |`) and
`kScoreOffset = kHeadDim * 2` (`:70`).

Arc `deepseek4.rs:631-655` — same split, same concat order (`:481`
`Tensor::cat(&[&wkv_w, &wgate_w], 0)`), `softmax(&score, 2)` over the slot axis, gate
applied after the ape add and before the sum. **MATCHES.**

### (b) `overlap_transform` / `coff` — **MATCHES** · `both-lbl`

`is_overlap_compress(r) → r == 4` (`compressor.py:181-182`) ⇒ **CSA overlaps, HCA does
not**. `coff = 1 + overlap` (`:312`); `ape` is `[ratio, coff·head_dim]` (`:314-316`).
`coff` does **not** change the stride — it doubles the per-token channel budget so each
raw token emits two (value, score) pairs. The compressed entry attends over
`coff·ratio = 8` slots spanning **stride = 4, window = 8 raw tokens `[4n−4, 4n+3]`**
(`c4_v2.cuh:79-127`, whose own comments read `// overlap [4n - 4, 4n - 1]` and
`// normal [4n + 0, 4n + 3]`), with `need_overlap = plan.seq_len > 4` (`:219, 248`).

Arc `deepseek4.rs:673-695` reproduces this slot for slot: Arc slot `j∈[0,4)` = group
`g−1` token `j` channels `0..d` ≡ reference `kv[j]`; Arc slot `4+j` = group `g` token `j`
channels `d..2d` ≡ reference `kv[4+j]`; fill `0.0` / `−inf` ≡ `0.0f` / `−FLT_MAX`.

**`apply_ape_hotfix` is reproduced implicitly and correctly.** Reference
`compressor.py:334-341` permutes the checkpoint `ape [ratio, 2d]` into `[2·ratio, d]`
at load (`deepseek_v4.py:1356-1362`). Arc adds the *un-permuted* `[ratio, 2d]` ape
**before** `overlap_transform` (`deepseek4.rs:635-639`), and the transform then routes
channels `0..d` to overlap slots and `d..2d` to normal slots — algebraically identical.
Arc's scalar-reference test encodes exactly this (`deepseek4.rs:4278-4300`).

### (c) Compressed positions / RoPE — **MATCHES** (see §1(b) for the mscale rider) · `both-lbl`

Position index `j·ratio` verified from both the kernel side and the metadata side —
see §1(b) "Compressed-KV rope positions". `ape` is in addition to, not instead of,
RoPE: `ape` is a pre-softmax *score* bias inside the pooling (`c4_v2.cuh:141`), RoPE is
applied afterwards to the pooled+normed vector (`compressor.py:104-110`). Arc splits it
the same way (ape `deepseek4.rs:639`, RoPE at the caller `:1147`).

### (d) RMSNorm placement & precision — **placement MATCHES; precision DIVERGES (MED)** · `both-lbl`

**Placement.** Chain is `wkv_gate(x)` → split → `score += ape` → (overlap) → softmax-pool
→ **RMSNorm over the full `head_dim`** → RoPE on the last 64 dims → quant/store.
Reference `compressor.py:94-111` then `fused_norm_rope_v2.cuh:255-286` (norm) and
`:296-307` (rope). Arc `deepseek4.rs:658-661`. Match.

**Precision — four separate downgrades:**

| item | reference | Arc |
|---|---|---|
| compressor state dtype | `torch.float32`, a hard DSV4 constant (`model_runner_kv_cache_mixin.py:865-868`; pool at `deepseek_v4_memory_pool.py:576`) | model dtype (bf16) — `forward_autocast` returns in `a.dtype()` (`mistralrs-quant/src/lib.rs:1051-1058`), so the `to_dtype(F32)` at `deepseek4.rs:631` recovers nothing |
| linear accumulation | `linear_bf16_fp32` = bf16×bf16 → **fp32 out** (`jit_kernel/dsv4/gemm.py:16-24`) | bf16 out |
| `ape` param dtype | `nn.Parameter(..., dtype=torch.float32)` (`compressor.py:314-316`) | VarBuilder dtype (bf16) — `deepseek4.rs:512-518` — cast up *after* the rounding |
| compressor `norm` weight | `RMSNorm(..., weight_dtype=torch.float32)` (`compressor.py:326-328`) | VarBuilder dtype (`deepseek4.rs:509-510`) |

`ape` is the most load-bearing: it is a **pre-softmax bias**, so bf16 rounding
(≈3 significant decimal digits) perturbs the pooling weights directly.

### (e) Raw-window ∪ compressed union masking — **the reference DOES double-count; Arc MATCHES it** · `both-lbl`

**This resolves the open question, definitively: the reference does NOT exclude
compressed entries covering positions already in the raw sliding window.**

`clip_down` / `get_raw_loc` (`compressor.py:228-237`) turned out to be **state-pool
ring-buffer addressing, not masking** — `clip_down(seq_lens−1)` picks the last block
boundary to *write*; `get_raw_loc` maps a raw position into a compress-state slot.
They never touch the attention index sets. Red herring.

The attention index sets are built independently:

* compressed — `metadata_kernel.py:38-53, 77-78`: `c128_seq_lens_raw = seq_len // 128`,
  `valid_mask = offsets < c128_seq_lens_raw`.
* raw — `deepseek_v4_backend.py:1180-1186, 1145`: `offsets = pos_causal − arange(SWA_WINDOW)`,
  `swa_topk_lengths = clamp(seq_lens_casual, max=128)`.

Both are fed to **one** `flash_mla_with_kvcache` call — `indices=swa_page_indices` plus
`extra_indices_in_kvcache` / `extra_topk_length` (`deepseek_v4_backend.py:1050-1064`,
assigned `:985-992`) — i.e. one softmax over the concatenation, with **no cross-set
exclusion anywhere**.

Concretely for HCA at `seq_len = s`, `m = s//128`, `r = s%128`: raw window `[s−128, s−1]`,
last visible compressed block `[128(m−1), 128m−1]`. These always intersect (by `128−r`
positions), and at `r == 0` the last compressed entry pools **exactly** the raw window.
Unambiguous double counting — and the query attends a compressed entry that pools the
query's own token.

Arc `dsv4_attention.rs:195-220` produces the identical counts
(`floor((pos+1)/ratio) == seq_len // ratio`; raw window identical, 128 both sides).
**MATCHES.** A future "optimization" that masks window-overlapping compressed entries
would be a *regression* — see test T-C1.

### (f) HCA vs CSA differences — enumerated · `both-lbl`

| # | Axis | Reference | Arc | Verdict |
|---|---|---|---|---|
| 1 | overlap / `coff` | `r==4 → coff=2`; `r==128 → coff=1` (`compressor.py:181,310-312`) | `deepseek4.rs:454-455, 535-536` | MATCHES |
| 2 | slots per entry | 8 (`c4_v2.cuh:92-127`) vs 128 (`c128_v2.cuh:107-127`) | `coff*ratio` (`deepseek4.rs:644-651`) | MATCHES |
| 3 | `ape` shape | `[4,2d]`→view`[8,d]` vs `[128,d]` | `[ratio, coff*head_dim]` | MATCHES |
| 4 | indexer present | CSA only (`deepseek_v4.py:263-273`) | CSA only | MATCHES |
| 5 | indexer's inner compressor | same `Compressor` class, `is_in_indexer=True, rotate=True` (Hadamard), input = hidden `x` (`indexer.py:506-515`; `compressor_v2.py:66` asserts `rotate == is_indexer`) | different pooling shape in `dsv4_indexer.rs` | **DIVERGES — see §10, indexer agent owns it** |
| 6 | RoPE θ | **no CSA/HCA difference**; the split is Standard-vs-compressed | same single compress table for both | MATCHES |
| 7 | top-k | CSA: top-`index_topk` over `seq_len//4`; HCA: none | dense over `seq_len//ratio` for both | HCA MATCHES; CSA is exact for `ctx ≤ topk·4`, approximate beyond |
| 8 | emission gate | CSA `plan.seq_len % 4 == 0`; HCA `write_loc % 128 == 127` | `t_trunc = (t/ratio)*ratio` (`deepseek4.rs:1126`) | MATCHES (equivalent) |
| 9 | cache page size | `page_size//4` vs `page_size//128` | live tensor, not paged (acknowledged TODO) | memory layout, not math |

### Other compressor findings (surfaced, not asked)

1. **Incremental vs recompute.** The reference keeps a per-token fp32
   `CompressStatePool` and emits one compressed entry per completed block. Arc caches the
   full hidden-state history `[B,T,hidden]` (`deepseek4.rs:1198-1209`) and re-runs the
   whole compressor over the entire prefix on **every decode step** (`:1117-1150`).
   O(T·hidden) memory and O(T) FLOPs per layer per token — correct, but this is the
   decode-throughput ceiling for CSA/HCA layers.
2. **`V4Compressor::uniform` is an unusable fallback, not a safe one.** `deepseek4.rs:970-974`
   falls back to it when weights are absent; `uniform` builds `wkv_gate` as
   `zeros((2*coff*head_dim, 1))` with `hidden_size: 1` (`:538-556`), so any real
   `forward_from_xs` errors on the matmul shape. Better to fail loudly at load.
3. **`V4Compressor::forward` (`:561-582`, mean-over-ratio) is dead legacy math** with no
   reference counterpart, kept alive by five unit tests that assert averaging semantics
   the model does not use.

---

## 3. mHC / hyper-connections

### (a) Sinkhorn normalization — **MATCHES exactly, including eps asymmetry and terminal axis** · `both-lbl`

Reference `layers/mhc.py:58-85` (`hc_split_sinkhorn_kernel_`) vs Arc `dsv4_mhc.rs:612-635`.
Point by point:

* Applied to **raw affine scores then exponentiated**: `comb = mixes[2hc..2hc+hc²]·scale[2]
  + base[2hc..]`, row-max-shift, `exp`. Both.
* **Probability space**, not log space. Both.
* **Row then column**; first pass is a *stable row softmax* (max-shift), not a bare
  row division. Both.
* **eps placement is asymmetric and Arc reproduces it exactly** — the easiest thing to
  get wrong:
  * first row pass: `/row_sum[j] + eps` (`mhc.py:73` / `dsv4_mhc.rs:618`) — eps **added
    after** the division;
  * every subsequent normalization: `/(sum + eps)` (`mhc.py:77,82,85` /
    `dsv4_mhc.rs:623,629,632`) — eps **inside** the denominator.
* **Iterations**: 1 initial row+col pass then `sinkhorn_iters − 1` more
  (`mhc.py:79` `T.serial(sinkhorn_iters - 1)` / `dsv4_mhc.rs:626`). Default 20 both
  (`configs/deepseek_v4.py:109`, `dsv4_mhc.rs:69`); eps default `1e-6` both.
* **Terminal axis = COLUMN** on both, so after the last op columns are exactly
  stochastic (minus eps) and rows only approximately 1.
* **Index layout**: reference flat `j*hc + k + hc*2`; Arc
  `narrow(D::Minus1, 2*hc, hc*hc).reshape((n, hc, hc))`. Identical.

The redundant second copy of the kernel in `mhc.py:199-217` is byte-for-byte the same
algorithm — no fast/slow-path divergence upstream.

### (b) `hc_attn_scale` / `hc_ffn_scale` [3] blend — **MATCHES** · `both-lbl`

`mix_hc = (2 + hc_mult)·hc_mult = 24` partitions as `[0..hc) | [hc..2hc) | [2hc..2hc+hc²)`
= `4 | 4 | 16`, and the three scale components map one to one.

Reference `mhc.py:52-62`:
```python
pre[i, j]  = T.sigmoid(mixes_shared[j] * hc_scale[0] + hc_base[j]) + eps
post[i, j] = 2 * T.sigmoid(mixes_shared[j + hc] * hc_scale[1] + hc_base[j + hc])
comb[j, k] = mixes_shared[j*hc + k + hc*2] * hc_scale[2] + hc_base[j*hc + k + hc*2]
```
Arc `dsv4_mhc.rs:292-319` — same. **Plain multiplication on both sides** — no
`(1 + scale)`, no `sigmoid(scale)`, no `exp(scale)`. `+ eps` after sigmoid on `pre` only;
`post` gets `× 2` and no eps (call-site constant `hc_post_mult_value=2.0`,
`deepseek_v4.py:820,837`). `hc_head_scale` [1] likewise plain-multiplied
(`deepseek_v4.py:1101` vs `dsv4_mhc.rs:743-747`).

### (b′) `post` / `comb` quantized to bf16 — **DIVERGES (MED)** · `both-lbl`

Arc's `hc_pre` computes in F32 then casts **all three outputs** to the residual dtype
(`dsv4_mhc.rs:334-336`):

```rust
let y_out    = y.reshape(y_shape)?.to_dtype(in_dtype)?;
let post_out = post.reshape(post_shape)?.to_dtype(in_dtype)?;   // bf16 in production
let comb_out = comb.reshape(comb_shape)?.to_dtype(in_dtype)?;   // bf16 in production
```

`hc_post` then casts them straight back to F32 (`:376-377`). Two rounds of bf16
rounding per mixing point, twice per layer, 43 layers.

The reference never does this. Eager path allocates `pre/post/comb` via
`mixes.new_empty(...)` from an fp32 `mixes` (`mhc.py:103-105`, `deepseek_v4.py:787-791`)
and casts **only `y`** (`deepseek_v4.py:870`). TileLang path allocates `post_mix` and
`comb_mix` as `torch.float32` while only `layer_input` is bfloat16 (`mhc.py:682-690`).
`hc_post_torch_impl` (`deepseek_v4.py:903-906`) consumes fp32 `post`/`comb`, accumulates
in fp32, casts once at the end.

`comb` entries are ~1/hc ≈ 0.25 where the bf16 relative step is ~2⁻⁹ ≈ 0.2%.
One-line fix: return `post`/`comb` as F32 and drop the re-cast.

### (c) 4-D residual threading — **MATCHES** · `both-lbl`

**Embedding lift — replicate, no scaling, no zeroing.** Reference
`deepseek_v4.py:1113-1115`: `hidden_states.unsqueeze(1).repeat(1, self.hc_mult, 1)`.
Arc `dsv4_mhc.rs:115-117`: `xs.unsqueeze(2)?.broadcast_as((b,t,hc_mult,h))?.contiguous()`,
called at `deepseek4.rs:3301`. Identical — **not** `1/hc_mult`-scaled, **not**
stream-0-only.

**Per-layer order.** Reference `deepseek_v4.py:918-935 / 936-999`; Arc
`deepseek4.rs:2467-2509`. hc_pre consumes 4-D and emits the 3-D layer input; hc_post
folds the 3-D branch output back to 4-D. Exact match, both for attn and ffn.

**hc_pre internals.** RMS taken over the **full `hc_mult·hidden`** flattened vector on
both sides (`deepseek_v4.py:787-791` vs `dsv4_mhc.rs:266-279`).

**hc_post orientation — the easy transpose bug, and Arc has it right.** Reference
`deepseek_v4.py:903-906` contracts `out[n,k,d] = Σ_j comb[n,j,k]·residual[n,j,d]` — a
*transposed* contraction. Arc `dsv4_mhc.rs:388`:
`let term2 = comb_n.transpose(1, 2)?.matmul(&residual_n)?;`. Same.

**Collapse at the end.** Reference `deepseek_v4.py:1168-1175` (hc_head, then final norm);
Arc `deepseek4.rs:3355-3381`. Order matches. `collapse_4d_to_3d` (mean over dim 2,
`dsv4_mhc.rs:144`) is **not** on the live path — good, because a mean collapse is not
what the reference does.

Arc does **not** surface the reference's second return value `pre_hc_head`
(`deepseek_v4.py:1175`) — which is exactly the tensor the MTP needs. See §5(a).

### (d) The 3-D bridge — **DIVERGES (HIGH): lossy, and it IS live** · `both-lbl`

**What it does** (`dsv4_mhc.rs:510-540, 545-568`): broadcast the 3-D residual to
`[B,T,hc,h]` (all streams equal), run one `mix_attn`, then `mixed_4d.mean(2)` back to
3-D. It re-symmetrizes the streams at every layer boundary and discards the learned
head-collapse. **Not** equivalent to the 4-D path — Arc's own test
`mhc_4d_path_differs_from_3d_bridge` (`dsv4_mhc.rs:1507-1627`) asserts they diverge for
≥half the elements after two mixes. The reference has no such construct anywhere.

**Where it is live:**

1. **Silent whole-model fallback.** `deepseek4.rs:3294`:
   ```rust
   let use_4d_mhc = self.mhc_head.is_some() && self.layers.iter().all(|l| l.mhc.is_some());
   ```
   Both `V4MHCHead::try_load` (`dsv4_mhc.rs:676-688`) and `V4MHCLayerParams::try_load`
   (`:197-212`) return `None` on **any** missing tensor *or shape mismatch* (`.ok()?`).
   There is a per-layer `tracing::warn!` (`deepseek4.rs:2333-2340`) but **no warning at
   all when `mhc_head` is `None`** — the model silently switches algorithm for the entire
   stack. Combined with the hardcoded `hc_mult = 4` (§3(g)), a checkpoint with
   `hc_mult ≠ 4` shape-mismatches, `try_load` returns `None`, and the model runs the
   bridge with no diagnostic.
2. **The MTP path, unconditionally.** `MtpBlock::forward_step` (`deepseek4.rs:2742-2754`)
   calls `self.layer.forward(...)` — the 3-D bridge — never `forward_4d`. Its own
   doc-comment (`:2727-2730`) admits it.

### (e) `hc_head` — **MATCHES** · `both-lbl`

Reference `deepseek_v4.py:1097-1103` vs Arc `dsv4_mhc.rs:726-757`: identical term for
term — same rsqrt over the full `hc·h` flatten, same `F.linear` (`@ fn^T`), same
`sigmoid(mixes·scale + base) + hc_eps`, same fp32 weighted sum, same final cast.
Shapes match `deepseek_v4.py:1051-1055`. Corroborated against the Triton fast path
(`mhc_head.py`).

### (f) Norm placement — **MATCHES** · `both-lbl`

The layernorm is applied **outside `hc_pre`, to `hc_pre`'s output `y`** — never to the
4-D residual, never to the mixed weights. Reference `deepseek_v4.py:919-927, 937-945`
(with the TileLang variant `mhc.py:643-690` fusing the *same* RMSNorm into
`layer_input`); Arc `deepseek4.rs:2473, 2499`. Same placement, same two norms, same
order relative to `attn`/`mlp`. Neither side norms inside `hc_post`.

### (g) Config fields are dead — **DIVERGES (LOW, latent)** · `both-lbl`

`DeepSeekV4Config` declares `hc_mult`, `hc_eps`, `hc_sinkhorn_iters`
(`deepseek4.rs:295-302`). `V4MHCRuntime::from_cfg` ignores all three
(`dsv4_mhc.rs:86-95`) with a stale comment ("Agent 3 will replace with cfg.hc_mult …
once those fields exist" — they now exist). The reference reads all three from config
(`deepseek_v4.py:696-698, 1046-1047`). For the shipped V4 Flash values the constants
coincide (4 / 20 / 1e-6, `configs/deepseek_v4.py:108-110`), so **no numerical impact
today**; the failure mode is a config that sets anything else silently producing wrong
math or (via the shape check) the silent 3-D-bridge fallback of (d). Three-line fix.

> **UNVERIFIABLE**: the fused CUDA sinkhorn (`mistralrs-core/src/cuda/sinkhorn.cu`) is
> claimed bit-identical to the candle chain (H200 A/B cited in its comments at `:45-53`).
> Not re-verified — no GPU. Settled by re-running `arc-tools/quality/run_sinkhorn_ab.py`
> or the bitwise replica tests in `mistralrs-core/src/cuda/sinkhorn.rs` on any CUDA box.

---

## 4. MoE routing

> **Authority chain**: `deepseek_v4.py:681-689` builds the MoE as
> `deepseek_v2.DeepseekV2MoE(..., is_deepseek_v4=True)` → `deepseek_v2.py:433`. Two
> routing classes: `HashTopK` (`layers/moe/hash_topk.py:23`) for layers `< num_hash_layers`,
> `TopK` (ungrouped `biased_topk_impl`, `layers/moe/topk.py:836`) otherwise. Selector at
> `deepseek_v2.py:483-484`.

### (a) Gate bias — **MATCHES** · `both-lbl`

Bias affects **selection only**; weights come from the *unbiased* scores. Reference
`topk.py:850-861`:

```python
scores = torch.nn.functional.softplus(gating_output).sqrt()
scores_for_choice = scores.view(num_token, -1) + correction_bias.unsqueeze(0)
_, topk_ids = torch.topk(scores_for_choice, k=topk, dim=-1, sorted=False)
topk_weights = scores.gather(1, topk_ids)          # UNBIASED
```

Arc `deepseek4.rs:1673-1678, 1707-1708` — same. Corroborated by
`inference/model.py:581-594`. **The classic porting bug is not present.** Loading
accepts either `.gate.bias` or `.e_score_correction_bias` (`deepseek4.rs:1567-1570`),
matching the reference's remap (`deepseek_v4.py:1416`).

### (b1) sqrtsoftplus scoring — **MATCHES** · `both-lbl`

Reference `topk.py:849-850`: `softplus(gating_output).sqrt()`. Arc
`deepseek4.rs:1633-1638` uses the stable identity
`softplus(x) = max(x,0) + log1p(exp(−|x|))` then `.sqrt()`. Logits in F32 on both sides.

### (b2) Group-limited routing — **DIVERGES (HIGH, conditional)** · `both-lbl`

The reference **disables group limiting for V4, unconditionally**
(`deepseek_v2.py:572-577`):

```python
# DSV4 override: ungrouped sqrtsoftplus + fp4 expert layout flag.
if is_deepseek_v4:
    topk_kwargs.update(use_grouped_topk=False, scoring_func=config.scoring_func, ...)
```

with a belt-and-braces load-time patch (`configs/model_config.py:265-273`) whose comment
is direct evidence about the shipped config:

> "HF config.json inherits topk_group=4 from the V3 template, but DSV4 trains with no
> group limiting (sqrtsoftplus + full-expert top-k). Force topk_group == n_group … The
> grouped impl only supports sigmoid scoring (topk.py:722) and would silently corrupt
> expert weights if hit."

Arc branches on the config value instead (`deepseek4.rs:1679-1709`), on the premise
"V4 Flash config does NOT publish `n_group`" — which the SGLang patch contradicts. If
`n_group` is present, Arc silently enters group-limited top-k with `topk_group` read
straight from the config, restricting selection to 4 of 8 groups. Arc's serde defaults
(`:140-141`, both 1) only save it when the key is genuinely absent.

Two secondary defects inside that branch, live only if taken:

* **Masking by multiply, not `-inf`** (`deepseek4.rs:1703`:
  `scores_for_choice.broadcast_mul(&score_mask)`). Reference masks with `-inf`
  (`inference/model.py:592` `masked_fill_(mask.unsqueeze(-1), float("-inf"))`). With
  `sqrtsoftplus ≥ 0` and a **negative** correction bias, `scores_for_choice` can be
  negative, so a masked-out expert (value 0) outranks an in-group expert with a negative
  biased score ⇒ out-of-group experts get selected. Inherited from `deepseek3.rs:534`.
* **Group score with no bias**: Arc always uses `topk(2).sum`; the reference uses `amax`
  when there is no bias (`inference/model.py:586-589`).

### (b3) `norm_topk_prob` — **DIVERGES (LOW)** · `both-lbl`

Reference gates renormalization on the config flag: `renormalize=config.norm_topk_prob`
(`deepseek_v2.py:553`) → `topk.py:876-882`; `norm_topk_prob: bool = True`
(`configs/deepseek_v4.py:70`). Arc has **no `norm_topk_prob` field** and keys off the
scoring function instead (`deepseek4.rs:1737-1743`). Same behavior at V4 defaults;
a checkpoint with `norm_topk_prob: false` is silently mis-handled.

### (b4) normalize → scale ordering — **MATCHES** · `both-lbl`

Reference renormalizes (`topk.py:876-880`) then scales (`:881-882`), or scales the routed
sum (`moe_runner/deep_gemm.py:649-650`) — algebraically identical. Arc normalizes
(`deepseek4.rs:1741-1742`) then `topk_weight * routed_scaling_factor` (`:1745`).

### (c) Top-k tie-breaking — **MATCHES (CPU) / UNVERIFIABLE (CUDA)** · `both-lbl` CPU, `inferred` CUDA

Reference `torch.topk(..., sorted=False)` (`topk.py:854-858`) — no guaranteed tie order;
CPU returns the lower index in practice. Arc CPU (`ops.rs:557-561` → candle
`ArgSort::asort`, `candle-core/src/sort.rs:35-49`) uses Rust's **stable** sort seeded
with ascending indices ⇒ lowest index wins, deterministically. Same as torch CPU.
Arc CUDA dispatches to `cuda_topk`; neither side is deterministic under exact ties.
Negligible for sqrtsoftplus over F32 logits — **except** in fixtures with zeroed
weights, which is exactly what Arc's synthetic fixtures use.

### (d) `tid2eid` hash routing — **MATCHES; Arc genuinely USES it, not inert** · `both-lbl`

**What it is**: a non-trainable `[vocab_size, topk − num_fused_shared_experts]` **int32**
table (`hash_topk.py:40-43`) implementing **forced routing** — the gate score only
weights the experts the table picked (`hash_topk.py:110-113`):

```python
topk_ids[:, :] = self.tid2eid[input_ids]
topk_weights[:, :] = scores.gather(1, topk_ids[:, :])
if self.score_func != "softmax":
    topk_weights[:, :] /= topk_weights[:, :].sum(dim=-1, keepdim=True)
```

Fused kernel agrees (`jit_kernel/csrc/deepseek_v4/hash_topk.cuh:55-63`).

Arc `deepseek4.rs:1641-1664` loads it at `:1553-1559` and *consumes* it:
`tid2eid.index_select(&ids, 0)` → `scores.gather(&topk_idx, 1)`, then normalize
(`:1737-1743`) and scale (`:1745`). Semantics, shape, and the unbiased-weights rule all
match. The `d4d2609f3` fixture is not decorative — `synthetic_load_smoke.rs:1498-1516`
builds a batch-invariance assertion **on top of** hash routing being a pure function of
token id, which only holds if the table is read. **No divergence.**

**`research/v4_audit.md` is wrong here.** Lines `:88` and `:634` guess the shape as
`[num_hash_layers=3, hidden_size?]` / `[3, 256]`. The authoritative shape is
`[vocab_size, num_experts_per_tok]` = `[129280, 6]` — which is what Arc implements.
Do not trust the prior audit on this point.

### (d′) `num_hash_layers` vs `n_hash_layers` — **UNVERIFIABLE (HIGH if wrong)**

SGLang's own dataclass declares `n_hash_layers: int = 3` (`configs/deepseek_v4.py:107`)
but the runtime reads `getattr(config, "num_hash_layers", 0)` (`deepseek_v2.py:483`) —
a **different key, defaulting to 0**. Arc accepts both (`deepseek4.rs:292`, with
`alias = "n_hash_layers"`, default 3). If the shipped `config.json` spells it
`n_hash_layers`, **the reference runs zero hash layers where Arc runs three** — a total
routing divergence on layers 0–2. Settled by one `grep -o 'hash_layers'` on the real config.

Two smaller riders: (i) the reference excludes the MTP block from hash routing
unconditionally (`is_hash = layer_id < n_hash_layers and not (is_deepseek_v4 and is_nextn)`,
`deepseek_v2.py:484`); Arc has no `is_nextn` term and keys purely on the virtual layer
index, which coincides for the real 43-layer config but diverges for small synthetic
fixtures — and one such fixture depends on it (`synthetic_load_smoke.rs:340-344`).
(ii) Reference stores `int32`; Arc requests `DType::I64` (`deepseek4.rs:1554-1559`) and
no test exercises an I32 checkpoint.

### (e) `swiglu_limit` clamp placement — **DIVERGES (HIGH): 4 of 5 Arc paths drop it** · `both-lbl`

**Reference formula** (`jit_kernel/csrc/deepseek_v4/silu_and_mul_masked_post_quant.cuh:69-84`,
citing DeepGEMM as source of truth — "must clamp in bf16"):

```cpp
if constexpr (kApplySwigluLimit) {
  gate = __hmin2(gate, {limit, limit});          // gate: ONE-SIDED upper clamp
  up   = __hmax2(up, {-limit, -limit});
  up   = __hmin2(up, {limit, limit});            // up:   SYMMETRIC clamp
}
... silu = g / (1 + expf(-g));  val = silu * u;  // clamp is PRE-ACTIVATION
```

Clamp on the **pre-activation** gate and up projections; the product is never clamped;
the gate is **not** clamped from below.

Arc `moe/experts.rs:541-549` (`forward_fast`, CUDA branch) — **MATCHES exactly**:
`(gate.clamp(-1e30, limit)?, up.clamp(-limit, limit)?)` then activation then multiply.

**Every other path drops the clamp:**

| Arc path | line | clamp? |
|---|---|---|
| `forward_fused` (CUDA, unquantized bf16 — the default when no ISQ, `experts.rs:64-69`) | `experts.rs:465` | **NO** |
| `forward_fast`, Metal branch | `experts.rs:570` | **NO** |
| `forward_slow` (CPU + quantized fallback; also `ARC_MOE_SLOW`) | `experts.rs:632-637` | **NO** |
| **shared expert** (all devices, all backends) | `deepseek4.rs:1832-1839` → `Mlp::new`, `layers.rs:3001-3037` has no limit field | **NO** |

The shared-expert omission is the most serious: **unconditional and device-independent.**
The reference passes the limit explicitly (`deepseek_v2.py:613-622`) and applies it at
`:318-323` via `silu_and_mul_clamp`, and SGLang gives up an entire fusion optimization
*specifically* to keep shared-expert clamping correct (`deepseek_v4.py:1266-1271`:
"DeepSeek V4 requires different clamping for shared and routed experts. Shared experts
fusion optimization is disabled."). Arc's own comment at `deepseek4.rs:1811-1813` states
the routed experts explode without the clamp (RUN-161) — the shared expert runs the same
trained weights through the same activation with the clamp missing.

Two notes: (i) the reference reaches the limit through `getattr(config, "swiglu_limit", None)`,
so it clamps **only if the checkpoint publishes the key**, whereas Arc hard-defaults to
`10.0` (`deepseek4.rs:139, 304-305`) — another thing the real `config.json` settles.
(ii) Arc's code comment cites `inference/model.py:596-606` as the authority
(`experts.rs:532-537`); that file is a plain unclamped `w2(silu(w1(x)) * w3(x))` with no
`limit` anywhere. **Stale citation** — the CUDA kernel above is the real authority, and
Arc's CUDA-fast implementation happens to agree with it.

### (f) Shared-expert path — **MATCHES** · `both-lbl`

Reference scales the routed sum first (`moe_runner/deep_gemm.py:649-650`, or
`deepseek_v2.py:793, 900`, or folded into topk weights `topk.py:881-882`), then adds the
shared output **unweighted** (`deepseek_v2.py:800-805, 910-916`). Arc `deepseek4.rs:1745`
then `:1908-1913` — same order, shared weight 1.0. `identity` is captured pre-gate
(`:1878`), so the shared expert sees the router's input.

> **Flagged, not asked.** `deepseek4.rs:1751-1780` contains three env-gated
> **behavior-changing** overrides inside `MoeGate::forward` — `ARC_SOFTMAX_ROUTE` replaces
> routing weights with a softmax over raw logits, `ARC_ROUTE_TOP1` zeroes all but the
> largest. Plus `std::env::var_os` calls per forward at `:1591, 1751, 1762, 1775, 1863,
> 1890` (and `experts.rs:54`). RUN-161 bisection hooks — a per-token syscall and a
> foot-gun in a shipping model file.

---

## 5. MTP (multi-token prediction / NextN)

### (a) Which tensors feed the draft step — **DIVERGES (CRITICAL)** · `both-lbl`

**Reference `h` is the pre-`hc_head`, pre-final-norm, 4-D mHC residual stack, flattened.**
`deepseek_v4.py:1168-1175`:

```python
pre_hc_head = hidden_states.flatten(1)
hidden_states = self.hc_head(hidden_states, self.hc_head_fn, self.hc_head_scale, self.hc_head_base)
hidden_states = self.norm(hidden_states)
return hidden_states, pre_hc_head
```

`:1311-1318` hands `pre_hc_head` to the logits processor as `hidden_states_before_norm`,
and `logits_processor.py:603-606` makes it the captured state unconditionally
("NOTE: when hidden_states_before_norm is provided, we always prefer to return it").
That capture becomes `spec_info.hidden_states`, consumed at `deepseek_v4_nextn.py:152-154`
as `[T·hc_mult, d]`. So: **not** post-final-norm, **not** the collapsed `hc_head` output,
**not** the raw last-decoder-layer 3-D output.

**Arc never produces `h` at all.** `DeepSeekV4::forward` collapses `xs_4d`, norms it, and
returns only logits (`deepseek4.rs:3355-3387`) — the pre-collapse state is dropped. The
pipeline then **substitutes the embedding of the just-sampled token**
(`mtp_pipeline.rs:946-956`):

```rust
let t0_tensor = Tensor::from_vec(vec![t0], (1,), &device)?;
let embedded_t0 = self.kit.embed_tokens.forward(&t0_tensor)?; // [1, hidden]
let proposed = self.kit.propose_chain(&embedded_t0, t0, self.depth, toks_remaining_budget, chain_start_pos)?;
```

The code comment concedes it (`:941-943`: "Seed `prev_hidden` from `embed_tokens(T0)`.
(Plumbing the target's real last hidden state through `forward_inputs` is the remaining
Tier-B follow-up.)").

**Consequence:** `h_proj` receives `embed(t)` where the trained input is the 43-layer mHC
residual — out of distribution by construction. Worse, **the same `embed(t)` also goes
through `e_proj`**, so the h/e distinction the module was trained on collapses entirely.
Expected acceptance ≈ noise, *independent of quantization*. **This is the single
highest-impact finding in this audit, and it gates the measurability of §6.**

### (b) Embedding-combine formula — **DIVERGES (structural)** · `both-lbl`

Reference `deepseek_v4_nextn.py:149-161` — definitively **two projections then add**,
not V3's `h_proj(concat(...))`, norms **before** the projections, and the add is a
**broadcast over the `hc_mult` stream axis** (`e` shared across streams, `h` per-stream),
output rank 3 `[n_tokens, hc_mult, hidden]`.

Arc `mtp_pipeline.rs:237-245` matches on the two-projections-then-add shape and on
norms-before-projections. **Divergence:** Arc's `fused` is `[B, hidden]` and the
`unsqueeze(1)` creates a **sequence** axis of length 1 — *not* the `hc_mult` stream axis.
There is no stream dimension anywhere in Arc's MTP path.

The Tier-A fallback (`mtp_pipeline.rs:194-199`) additionally skips `hnorm`/`enorm`
entirely and feeds `fused` straight to `lm_head` — no reference counterpart at all.

### (c) Position handling and KV slot — **DIVERGES (CRITICAL)** · `both-lbl`

**Positions match in form.** Reference `eagle_worker.py:726` seeds
`spec_info.positions = batch.seq_lens.repeat_interleave(topk)`, incremented per step at
`:910`. Arc `mtp_pipeline.rs:293` passes `start_pos + i` with
`start_pos = current_normal_cache_len(self)` (`:949`); `forward_step` uses
`seqlen_offsets = [pos]` (`deepseek4.rs:2741`).

**KV does not match.** The reference's draft model has its own KV pool but **shares the
allocator and `req_to_token_pool` with the target** (`eagle_worker.py:134-138`) and that
pool is **prefilled over the whole prompt** (`forward_draft_extend`, `:1094-1128`) and
extended over each verify's accepted tokens (`forward_draft_extend_after_decode`, `:1134+`).
So the reference's MTP attention at draft step *i* attends over the **entire committed
context** plus prior draft tokens.

Arc allocates a **fresh, empty** cache per chain (`deepseek4.rs:2691-2696`,
`mtp_pipeline.rs:288`):

```rust
pub fn new_chain_cache(&self) -> KvCache {
    KvCache::new_normal(2, self.max_seq_len, 16)
}
```

So the MTP block attends over the 1–8 draft tokens of this chain **and nothing else**,
while RoPE is applied at absolute positions `P, P+1, …`. A query at position ~200 000
attends keys the model believes are at ~200 000 but which are the only tokens present.
The block never sees the prompt. `deepseek4.rs:2651-2653` accurately describes a design
that does not match the reference.

Combined with (a): the MTP block is asked to predict from a wrong `h`, no context KV, and
a position index that indexes nothing.

### (d) mHC inside MTP — **DIVERGES (partial, MED-HIGH)** · `both-lbl`

The reference NextN has three mHC surfaces: its own `hc_head_{fn,base,scale}`
(`deepseek_v4_nextn.py:75-79`), per-layer `hc_attn_*`/`hc_ffn_*` inside its decoder
(`:98-106`, `COMPRESS_RATIO_NEXTN_LAYER = 0` at `:47`), and the collapse+norm at the end
(`:194-201`). The checkpoint ships them — `deepseek_v4.py:1381-1391` lists `"hc_head"`
among `nextn_spec_prefixes`.

Arc: **per-layer `hc_attn_*`/`hc_ffn_*` are LOADED but degraded** — `MtpBlock::try_new`
builds a real `DecoderLayer` whose `new` calls `V4MHCLayerParams::try_load` against
`mtp_vb` (`deepseek4.rs:2332`), but `forward_step` routes through the **3-D bridge**
(§3(d)), which broadcasts one vector into four identical streams and mean-collapses them
every layer. **`hc_head_*` is NEVER LOADED** — `MtpBlock` has no such field
(`deepseek4.rs:2570-2586`), `V4MHCHead::try_load` is called exactly once against the model
root (`:3162-3166`), and `MtpBlock::norm_out` (`:2708-2711`) applies `shared_head.norm`
directly to the decoder output, skipping the learned collapse. Confirmed from the other
direction: the UQFF residual emitter writes `hc_attn_*`/`hc_ffn_*` for the MTP block but
no `hc_head_*` (`:2899-2906`).

### (e) Verify / accept criterion — **MATCHES (greedy) / DIVERGES (sampled)** · `both-lbl`

The reference selects by `sampling_info.is_all_greedy` (`eagle_info.py:322-344`).
Greedy (`:329-342`): accept while candidate == target argmax, walking the candidate tree;
the emitted token at the stopping node is the target's own argmax. Sampled (`:344-396`):
renormalized `target_probs`, per-candidate `coins`, plus a **separate coin for the
bonus draw** (`coins_for_final_sampling`), with `threshold_single`/`threshold_acc`
defaulting to `1.0`/`1.0` (`server_args.py:564-565`) — **exact rejection sampling by
default**, typical-acceptance relaxation opt-in.

Arc `mtp_pipeline.rs:513-530` implements the greedy chain exactly (a linear-chain special
case of the reference's tree walk; correction = target argmax = the reference's bonus
token). **MATCHES for greedy.**

**⚠️ Sampled — DIVERGES, and it is a correctness bug.** `step()`'s fast-path gate
(`mtp_pipeline.rs:851-860`) checks prompt/batch/xlora/cache-kind and **never inspects the
sampling params**. `T0` is drawn through the real sampler (`:907-916`), but every accepted
proposal and the correction are committed with a fabricated logprob (`:1024-1045`):

```rust
let lp = crate::sampler::Logprobs { token: tok, logprob: 0.0, top_logprobs: None, bytes: None };
```

So with `temperature > 0` the request silently gets **greedy** tokens for the accepted
suffix and `logprob: 0.0` in the response. The module docstring claims "Greedy only"
(`:32-34`) and "no quality loss possible" (`:41-43`) — the first is an unenforced comment,
the second is false under sampling.

### (f) Layer count and speculative depth — **MATCHES on count; DIVERGES on candidate structure** · `both-lbl`

Reference hard-asserts one NextN layer (`deepseek_v4.py:1430-1439`). Arc has **no
`num_nextn_predict_layers` field** and hardcodes `mtp.0` / `mtp.layers.0`
(`deepseek4.rs:3081-3082`) — effectively identical for V4, unvalidated (a checkpoint with
`mtp.1` is silently ignored).

Both chain the same single block autoregressively (`eagle_worker.py:863-873` vs
`mtp_pipeline.rs:289-307`, `--mtp-depth` range `0..=8`, default `0`).

**Divergence:** the reference expands `topk` candidates per step into a **tree**
(`select_top_k_tokens` `:864`, `build_tree_kernel_efficient` `:800`). Arc emits a single
linear chain (greedy argmax `:303`) — the `topk == 1` degenerate case. Not wrong, but it
forfeits the tree's acceptance-length gain and **cannot be compared to published V4
acceptance numbers.**

Also: the reference forwards `logits_output.hidden_states`, which is the NextN's own
`pre_hc_head` (`[T, hc_mult·d]`); Arc forwards `hidden.squeeze(1)`, `[B, hidden]`
(`mtp_pipeline.rs:249`). The doc at `deepseek4.rs:2723-2725` claims this matches the
reference; it matches only in the "pre-norm" sense, not in width or stream structure.

> **`dsa_mtp_verification.py` and `dsa_backend_mtp_precompute.py` are red herrings.** The
> first (`:1-6`) is a debug oracle for a *metadata-copy kernel*, not speculative
> verification. The second (`:1-5`) is a per-draft-step attention-metadata cache, relevant
> only once Arc's MTP attends over real context KV.

### Existing Arc MTP tests are structurally vacuous

`synthetic_load_smoke.rs:390-397` builds the MTP fixture with **zero** projections
(`h_proj`/`e_proj` both `z(&[HIDDEN_SIZE, HIDDEN_SIZE])`), so `fused ≡ 0` and every draft
token is a constant. `v4_flash_mtp_full_block_load_smoke` (`:518-599`) asserts only
"2 tokens, in vocab" — it cannot detect any of (a)–(d). The fixture also omits
`hc_head_*` / `hc_attn_*` / `hc_ffn_*` at `mtp.layers.0` entirely and sets
`compress_ratios: [0, 0]` (`:439`), so the mHC path is never exercised. The unit tests in
`mtp_pipeline.rs:1194-1527` use identity projections and a mock verifier — they pin
`verify_proposed` arithmetic and cache bookkeeping (both correct), never the model math.

---

## 6. MTP draft-head quantization floor

**External input** (`EXTERNAL_FINDINGS.md` F3): colibrì measured an int4 MTP draft head at
**0–4% acceptance**; int8 is mandatory for speculation to pay.

**VERDICT: DIVERGES — Arc has no floor. A 2-bit MTP draft head is reachable today, by
default, on the flagship path.** `both-lbl`

### Path (i) — plain `--isq q4k`/`qtip2b` serve (the common case)

`normal.rs:572-602` sets `use_immediate = true` ⇒ `loading_isq = false` (`:603-609`) ⇒ the
post-load `quantize()` pass is skipped (`:936`). Quantization is **name-regex driven at
load**, from `DeepSeekV4Loader::isq_layer_regexes` (`normal_loaders.rs:3246-3315`):

* **MTP routed experts are quantized to the requested dtype** — `:3300-3313` deliberately
  opts them in:
  ```rust
  Regex::new(&format!(
      r"mtp\.(layers\.)?\d+\.(mlp|ffn)\.experts\.{i}\.gate_proj\.(weight|bias)$"
  ))?,
  ```
  `--isq q2k` / `qtip2b` ⇒ **2-bit MTP experts**.
* MTP attention: only under `ARC_QUANT_ATTENTION` (`:3263-3273`).
* `h_proj` / `e_proj`: not matched ⇒ left at checkpoint precision on this path.

### Path (ii) — anything that runs `quantize()`

`--imatrix`, `--calibration-file`, a topology needing post-quantization, or `--from-uqff`
all force `loading_isq = true` (`normal.rs:610-613`). Then `DeepSeekV4::get_layers`
(`deepseek4.rs:3429-3452`) pushes **every** MTP tensor with `layer_num = None`:

```rust
tensors.push((&mut mtp.h_proj, None));
tensors.push((&mut mtp.e_proj, None));
if let Some(block) = &mut mtp.block { ... tensors.push((t, None)); }
```

and `isq.rs:631-639` — the only dtype selector on that path — resolves `None` to the
**global dtype, unconditionally**:

```rust
let dtype = if let Some(ref layers) = layers {
    if let Some(layer) = layer_num { layers.get(*layer)... } else { dtype }
} else { dtype };
```

So `h_proj`, `e_proj`, the MTP attention, and all 256 MTP experts land at the global
width — 2-bit under qtip. Because `layer_num` is `None`, `--topology` (indexed by layer
number) **cannot reach these tensors at all**; a user cannot exclude or raise the MTP head
even deliberately.

### Path (iii) — regex topology

`Topology` carries `patterns: Vec<(Regex, LayerTopology)>` matched by **tensor name**
(`topology/mod.rs:61-63, 224-231`), converted into `ImmediateIsqOverride`
(`normal.rs:552-561`). A user can write `mtp\..*` and pin the MTP head to `AFQ2`/`Q2K`
with no objection from the engine.

### The project already knows this failure mode — and fixed only `lm_head`

`normal_loaders.rs:3249-3252`:

> "RUN-161: lm_head excluded from ISQ — quantizing the logit projection to 2-bit corrupts
> EOS probabilities and breaks chat/instruction-following (model outputs 1-2 tokens then
> EOS). Keeping lm_head at native precision costs ~1GB but preserves the output distribution."

The MTP head is a *second* logit-producing path and got no such exclusion. `grep` over
`isq.rs`, `mistralrs-cli/src/`, and `mistralrs-quant/src/` finds **no MTP-aware bit-width
check, warning, or floor**. `--mtp-depth N --isq qtip2b` is accepted silently.

> **Sequencing.** Fix §5(a) before measuring this. With the embedding-seeding bug,
> acceptance is near zero at int8 too, so a quantization experiment would measure nothing.
>
> **Adjacent (worth a separate change).** `lm_head` is pushed at `deepseek4.rs:3399, 3462`
> with `layer_num: None`, so the `quantize()` path quantizes it at the global dtype —
> contradicting the RUN-161 rationale, which only holds on the regex path.
>
> **Adjacent (worth a separate change).** The V4 ISQ regexes are unanchored, so
> `layers\.0\.mlp\.experts\.N\.…` also matches `mtp.layers.0.mlp.experts.N.…`, and
> `ARC_QUANT_ATTENTION` matches MTP attention under HF naming but not native `mtp.0.attn.*`.
> Same dtype today, so no visible bug — anchoring would remove a real footgun.

---

## 7. Draft and verify dispatch through different kernel families

**External input** (F3): colibrì ships `SPEC_PIN=1` **by default** because running draft
and verify through different kernels silently destroyed acceptance.

**VERDICT: DIVERGES — Arc splits kernel family on token count, the draft step is always on
one side and the verify step always on the other, and there is no pin.** `both-lbl`

### The split points

**(a) Dense projections — QTIP, default-on, `n_tokens == 1`.**

| Rung | File:line | `n_tokens == 1` | `n_tokens > 1` |
|---|---|---|---|
| `Qtip2b` | `mistralrs-quant/src/qtip/bitshift.rs:988` | `cuda_ops::fused_gemv_2b_cuda` (in-register trellis decode) | `dequantize_2b_cuda` → materialize BF16 W → cuBLAS `matmul` |
| `QtipBitshift2` | `mistralrs-quant/src/qtip/mod.rs:1508` | `cuda_ops::fused_gemv_cuda` | `dequantize_rotated_cuda` → cuBLAS `matmul` |

Different kernels, different accumulation orders, same weights. The module doc is explicit
that this is a deliberate perf split (`qtip/mod.rs:1438-1445`).

**(b) MoE experts — QTIP, default-on, `n_tokens <= 8`.**
`Qtip2bLayer::gather_forward` (`bitshift.rs:1411-1445`) is a *three*-way dispatch:

```rust
let ondevice_max_tokens = std::env::var("ARC_QTIP_ONDEVICE_MOE_MAX_TOKENS")
    .ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(8);
if !ondevice_disabled && n_tokens <= ondevice_max_tokens {
    return self.gather_forward_cuda_ondevice(a, indices);
}
// else: trellis grouped GEMM (Arc Stage 4) ... else CPU reference
```

**(c) Whole-forward engine — CUDA graph, `seq_len == 1`, opt-in today.**
`normal.rs:1455`: `if probe && seq_len == 1 && self.cuda_graph_runner.is_some()`, where
`probe = std::env::var_os("ARC_V4_CAPTURE_PROBE").is_some()` (`:1429`). Probe-gated, so not
a default-on hazard — but a third surface keyed on exactly the draft/verify boundary, and
the roadmap is to make it default.

### Why MTP always straddles the split

`MtpSpeculativePipeline::step` (`mtp_pipeline.rs:815-829`) documents the algorithm:

1. Target forward over the current input — **1 token** ⇒ (a) GEMV, (b) ondevice.
2. `propose_chain` — the draft runs **1 token at a time** (`:283-303`,
   `Tensor::from_vec(vec![tok], (1,), device)`) ⇒ GEMV.
3. Target verify forward over `[T0 … T_{depth−1}]` — **`depth` tokens** ⇒ (a) **dequantize
   + cuBLAS GEMM**, and (b) flips to the grouped GEMM once `depth > 8`.

So the target model's logits at step 1 and step 3 come from **different kernels for the
same weights**. That contradicts the invariant the module asserts twice:

> `mtp_pipeline.rs:317-319`: "Verification is lossless: greedy decode of
> `MtpSpeculativePipeline` equals greedy decode of the target alone."
> `:511-512`: "This is mathematically lossless: greedy output of the target alone equals
> greedy output via MTP+verify."

Lossless in exact arithmetic; **not** in floating point when the two forwards run
different kernels. And `verify_proposed` (`:513`) accepts on exact `u32` argmax equality —
the most tie-sensitive possible criterion. Every near-tie flipped by the kernel change is
a spurious rejection: exactly the "silently destroyed acceptance" colibrì saw.

### No pin exists

`grep -rn "SPEC_PIN\|spec_pin\|pin_kernel\|force_gemm\|ARC_SPEC" --include=*.rs` returns
nothing. The closest knobs are global, not draft/verify-scoped: `ARC_QTIP_GEMV_VARIANT` /
`set_forced_gemv_variant` (`qtip/tune.rs:96, 112`) selects *among GEMV variants* and cannot
force the GEMV kernel for a multi-token batch; `ARC_NO_QTIP_ONDEVICE_MOE` and
`ARC_QTIP_ONDEVICE_MOE_MAX_TOKENS` (`bitshift.rs:1411-1415`) move the MoE boundary but not
the dense-projection one.

---

## 8. Would any Arc quality gate catch EOS starvation?

**External input** (F3): colibrì found quantization-granularity failures surface as **EOS
starvation, not perplexity** — per-row int4 vs group-64 was ~9pp worse, manifesting as
think-mode loops / never-terminating generations.

**VERDICT: PARTIAL — Arc has the right *signals* but no *gate*. Nothing fails, nothing
exits non-zero, nothing runs in CI, and the one termination check has no statistical
power.** `both-lbl`

| Harness | Termination signal | Gate? |
|---|---|---|
| `arc-tools/quality/run_coherence.py` | **Yes** — `stopped = r["finish_reason"] == "stop"` (`:80`), `looped = qlib.looks_degenerate(text)` (`:81`), `ok = stopped and not looped and kw_ok` (`:83`) | **No** — printed and written to JSON (`:153-160`); `main()` returns normally, exit code always 0 |
| `arc-tools/quality/run_gsm8k.py` | **Counted, not judged** — `"truncated": …finish_reason == "length"` and `"degenerate"` (`:349-350`) | **No** — printed in the summary (`:318`), never compared to a threshold |
| `arc-tools/quality/run_ppl.sh` / `parse_ppl.py` | **None** — teacher-forced; EOS is never sampled | n/a — colibrì's exact blind spot |
| `arc-tools/quality/stall_sentinel.sh` | Watches a *log file* for growth, PID-kills on stall | Process liveness only; a model happily emitting non-terminating text keeps the log growing, so this never fires |

`qlib.looks_degenerate` (`qlib.py:194-208`) detects a repeated cycle of period ≤ 8 spanning
≥ 14 words — the classic 2-bit repetition loop, but **not** the failure colibrì describes:
non-repetitive text that simply never emits EOS. Only `finish_reason == "length"` catches
that, and only `run_coherence.py` looks at it per item.

**Why the coherence check has no power.** `run_coherence6` is **6 prompts at
`max_tokens=60`** (`run_coherence.py:76`). A 9pp shift in termination rate on n=6 is a
fraction of a prompt — the 95% binomial CI on 6 trials is roughly ±35pp. `EXPECTED.md:44`
records the bar as a human judgement ("June anchor: 6/6 coherent") and
`GPU_SESSION_RUNBOOK_4.md:278` invokes it as a "6/6 gate" — but the gate is a human reading
a number on a rented box.

**Gap summary.** (1) No assertion — both scripts exit 0 regardless. (2) No CI —
`grep -rn "run_coherence\|run_gsm8k\|run_ppl" .github/workflows/` returns nothing. (3) No
A/B on termination *between quant configs*, which is the measurement that made colibrì's
finding visible. (4) `max_tokens=60` is far too short for think-mode looping to manifest;
GSM8K uses 2048 and *does* count truncations, but never judges them.

---

## 9. `Qtip2b` 3-D experts default to Greedy while `QtipBitshift2` defaults to Viterbi

**VERDICT: CONFIRMED — DIVERGES from the project's own wave3-G decision.** `both-lbl`
**Not fixed, per instruction.**

Two arms, 30 lines apart in the same `match`, with **opposite defaults and inverted env
vars**. `mistralrs-quant/src/unquantized/mod.rs:383-389` — `QtipBitshift2` (`--isq qtip2`):

```rust
let mode = if self.w.dims().len() == 3 {
    crate::QtipMode::default_expert_mode()      // => Viterbi unless ARC_QTIP_EXPERT_GREEDY
} else {
    crate::QtipMode::Viterbi
};
```

`:412-420` — `Qtip2b` (`--isq qtip2b`):

```rust
// Same mode/device policy as the LUT rung above: 3-D MoE
// stacks default to greedy unless ARC_QTIP_EXPERT_VITERBI is set, ...
let expert_viterbi = std::env::var_os("ARC_QTIP_EXPERT_VITERBI").is_some();
let mode = if self.w.dims().len() == 3 && !expert_viterbi {
    crate::QtipMode::Greedy
} else {
    crate::QtipMode::Viterbi
};
```

| `--isq` | 3-D expert default | Env to flip | Rotation |
|---|---|---|---|
| `qtip2` (`QtipBitshift2`) | **Viterbi** | `ARC_QTIP_EXPERT_GREEDY=1` | **on** |
| `qtip2b` (`Qtip2b`) | **Greedy** | `ARC_QTIP_EXPERT_VITERBI=1` | **off** |

Four concrete problems:

1. **The comment is false.** It claims "same mode/device policy as the LUT rung above";
   wave3-G flipped the LUT rung to Viterbi and this arm was never updated.
2. **Greedy also disables the Hadamard incoherence rotation.**
   `Qtip2bLayer::quantize_with_mode` (`qtip/bitshift.rs:343-351`):
   `let use_rotation = matches!(mode, QtipMode::Viterbi);`. So `qtip2b` 3-D experts get
   greedy walk **and** no incoherence processing — the exact combination
   `QtipMode::default_expert_mode`'s own doc-comment (`qtip/mod.rs:539-547`) blames for
   `PPL qtip2=58.85 vs q2k=22.50` on H200 (2026-08-13).
3. **The bake-quality regression test does not cover this rung.**
   `qtip/bake_quality_tests.rs:509-510` calls `QtipLayer::quantize_with_mode` only.
   `Qtip2bLayer` is never exercised through `default_expert_mode`, so nothing fails when
   the two arms disagree.
4. **The ISQ thread policy is keyed off the wrong decision.**
   `IsqType::get_max_isq_cpu_threads` (`mistralrs-quant/src/lib.rs:817-829`) matches
   `QtipBitshift2 | Qtip2b` together and consults `default_expert_mode()`. For `qtip2b`
   that reports Viterbi (⇒ `None` ⇒ all rayon workers) while the bake actually runs Greedy.
   Note wave10-Y's `IsqQuantizeBackend` fix is on PR #25 (DRAFT) and is **not** on
   `master`, so `master` still has the policy described here.

Arc's own measured numbers for the choice, quoted in the sibling arm
(`unquantized/mod.rs:377-382`): "matmul cos vs FP4 is greedy=0.887 vs viterbi=0.962
(3x less error) on real V4 experts."

---

## 10. For the indexer agent (F1 owner) — observations found incidentally

These came from shared files. **No indexer file was opened.**

> **Status at time of writing**: F1 (the per-head-top-k / missing-ReLU scoring defect) was
> fixed in **PR #26** during this audit. That work independently reported two of the items
> below — its finding (i) "Arc's compressor builds indexer K **per-head** `[B,H,T_c,D]`
> while SGLang's indexer cache is **MQA (`num_heads_kv=1`)** ⇒ ~64× the K memory/compute"
> corroborates item 2 here from the opposite direction, and its finding (iii) (Hadamard +
> FP4-QDQ on indexer Q still deferred) is item 3. Items 1, 4 and 5 appear to be new.

1. **The indexer's inner compressor uses `rotate=True` (Hadamard); the core compressor uses
   `rotate=False`.** `compressor.py:111`: `return rotate_activation(kv_compressed) if rotate
   else kv_compressed`; `compressor_v2.py:66` asserts `rotate == is_indexer == (head_dim == 128)`;
   `MQALayer` constructs the attention compressor with `rotate=False` (`deepseek_v4.py:273`).
   If Arc's indexer-side compressed KV skips the Hadamard rotation, it is silent.
2. **The indexer's compressor is the same `Compressor` class** with
   `is_in_indexer=True, compress_ratio=4, head_dim=index_head_dim(128)`, input = **hidden
   `x`**, so `wkv_gate: [2*coff*128, hidden_size]` and it inherits the **overlap** (a c4
   compressed entry pools **8** raw tokens, not 4) — `indexer.py:506-515`. Arc's
   `dsv4_indexer.rs:84-87, 660` appears to describe a different pooling
   (`sigmoid(gate)*kv + sum-over-coff`) — **same wrong-pooling family as F1**, worth
   checking alongside the score fix.
3. **Indexer Q is FP8 + Hadamard-quantized before scoring** —
   `jit_kernel/dsv4/elementwise.py:123-139` (`fused_q_indexer_rope_hadamard_quant`) returns
   `(q_fp8, weights_out)` with a per-token fp32 scale. Scoring in BF16 without the rotation
   drifts top-k ordering. (Matches colibrì's F1 note (c).)
4. **`index_topk = 512` selects compressed c4 entries, not raw tokens**
   (`deepseek_v4_backend.py:985-988`: `extra_indices = c4_sparse_page_indices`). The
   128-wide raw SWA branch is always dense and is **not** part of the top-k budget. If
   Arc's indexer selects over raw tokens or over the union, the budget semantics differ.
5. **Short-sequence edge**: `c4_topk_lengths_clamp1 = max(seq_len // 4, 1)`
   (`metadata_kernel.py:41`) — at `seq_len < 4` the kernel still reads one entry whose page
   index is `-1`. Arc returns `None` for `T_c == 0` (`deepseek4.rs:1127-1129`), which is
   equivalent; confirm the indexer's own short-sequence path agrees.

---

## 11. Proposed failing tests — prioritized

All CPU-only candle/pytest-free tests, zero GPU cost. **P0** = would have caught a
CRITICAL/HIGH finding; **P1** = HIGH/MED; **P2** = lock-in of correct behavior.

### P0 — MTP correctness (findings 1, 2, 13)

**T-M1 `mtp_teacher_forced_acceptance_oracle`** — *the load-bearing one; colibrì's
teacher-forcing method.* Seeded synthetic V4 (`HIDDEN_SIZE=32`, 2 layers, 4 experts,
`hc_mult=4`, `VOCAB=64`, full `hc_head_*`/`hc_attn_*`/`hc_ffn_*`). Teacher-force a 32-token
sequence through `DeepSeekV4::forward`. For each position *n*: oracle = `argmax(logits[n])`;
draft = one MTP step seeded with **the target's pre-`hc_head` state at position n**
(`[1,4,32]`, i.e. `xs_4d` before `mhc_head.forward`, matching `deepseek_v4.py:1168`) and
token `t_n`.
Assert `matches/32 >= 0.5`. **Fails twice over today**: `DeepSeekV4::forward` exposes no
pre-`hc_head` state to capture (`deepseek4.rs:3355-3387` returns logits only), and the
embedding-seeded variant lands near chance. Fixing it requires the capture channel that
finding 1 is missing — which is the point.

**T-M2 `mtp_chain_cache_must_hold_the_committed_prefix`** — *cheapest possible pin, runs in
microseconds.* `let cache = block.new_chain_cache(); assert_eq!(cache.current_seq_len(), start_pos)`
with `start_pos = 137`. **Fails today** (`0 != 137`, `deepseek4.rs:2694-2696`). Reference:
`eagle_worker.py:1094-1128` prefills the draft KV over the whole context.

**T-M3 `mtp_combine_must_preserve_hc_mult_stream_axis`** — extend
`synthetic_v4_weights_with_mtp` with **non-zero** `h_proj`/`e_proj` and MTP `hc_*`; call the
full-block step with `prev_hidden` shaped `[1, 4, 32]`; assert
`next_hidden.dims() == &[1, 4, 32]`. **Fails today** — no stream axis exists.

**T-M4 `mtp_block_must_apply_learned_hc_head_collapse`** — two loads differing only in
`mtp.layers.0.hc_head_scale` (`0.0` vs `4.0`), same prompt; `assert_ne!(chain_a, chain_b)`.
**Fails today** (`a == b`) — `hc_head_*` is never read on the MTP path.

### P0 — attention masking (findings 3, 4)

**T-A1 `dsv4_attention_honors_caller_attention_mask`** — `(b,h,t,d) = (1,2,6,16)`,
`cfg.sliding_window = 6` (so the internal mask is plain causal). Build `pad_mask [1,1,6,6]`
all-zero except column 0 = `−inf` on every row. Reference = `Sdpa.run_attention` with the
*combined* mask. Assert elementwise agreement to 1e-5, **plus a teeth-check** that the
padded and unpadded references differ by > 1e-3 so it cannot pass vacuously.
**Fails today** — the pad column contributes full softmax mass.

**T-A2 `v4_batch_prefill_does_not_route_to_maskless_varlen`** — construct `FlashParams` with
a populated `cumulative_seqlens_k` for the CPU device location, `b_sz = 2`, `head_dim = 512`,
and call `dsv4_attention`. Assert the result equals a per-sequence loop of the masked
reference. **Fails (or errors) today** — `sinks.rs:32-49` diverts to `sinks_attn_varlen`,
which calls `sinks_attn_cpu(..., None, ...)` (`:228`).

### P1 — RoPE / quantization grids (findings 6, 14)

**T-R1 `standard_layer_rope_matches_unscaled_reference`** — build the Standard-layer rope
config exactly as `DeepSeekV4::new` does (`rope_scaling: Some(Yarn{ factor: 16.0,
original_max_position_embeddings: 4096, … })`, `rope_theta: 10000`, `qk_rope_head_dim: 64`)
and a second with `rope_scaling: None`. Apply both at position `8192` (past
`original_max_position_embeddings`, where the interpolated band bites).
Assert `max_diff < 1e-6`. **Fails today** (~1e-1). Add the **mirror assertion** for the
compressed config (YaRN *must* be present there, `max_diff > 1e-3`) so the fix cannot be
over-applied.

**T-R2 `act_quant_kv_nope_uses_ue8m0_power_of_two_scale`** — `k: [1,1,1,512]` F32, zeros
except `k[0] = 300.0`, `k[1] = 1.0` in the first 64-wide tile. Reference math
(`quant_k_cache.py:48-58`): `scale = 300/448 = 0.66964`; `ceil(log2) = 0`; `scale_pow2 = 1.0`;
dequant of `k[1]` = **exactly 1.0**. Arc today: `1.0/0.66964 = 1.4934` → `e4m3 → 1.5` →
`1.5 × 0.66964 = 1.00446`. Assert `|v[1] − 1.0| < 1e-4` **and** `|v[0] − 288.0| < 1.0` (so a
fix that only special-cases small values cannot sneak through). **Fails today** by ~45× the
tolerance. Run with `ARC_GPU_ACT_QUANT` both unset and set.

### P1 — MoE (findings 5, 11)

**T-E1 `v4_shared_expert_swiglu_is_clamped`** — *highest value of the MoE set: device
independent and unconditional.* CPU `Moe::new` with `n_shared_experts = Some(1)`,
`swiglu_limit = 1.0`, routed experts all-zero (so `y_routed == 0`), shared
`gate_proj = up_proj = 5.0·I`, `down_proj = I`, input `ones`. Assert
`out ≈ silu(1.0)*1.0 ≈ 0.7311` per channel, not `silu(5.0)*5.0 ≈ 24.83`. **Fails today** —
`Mlp` (`layers.rs:2992`) has no clamp and `deepseek4.rs:1832` passes none.

**T-E2 `swiglu_limit_applied_on_slow_and_fused_backends`** — same fixture through
`MoEExpertsBackend::Slow` and `::Fused`. **Fails today** on both (`experts.rs:632-637`, `:465`).

**T-E3 `v4_gate_is_ungrouped_even_when_n_group_present`** — parse a config with
`"n_group": 4, "topk_group": 2, "n_routed_experts": 8, "num_experts_per_tok": 3,
"scoring_func": "sqrtsoftplus"`; craft a token whose 3 highest scores live in 3 *different*
groups. Assert plain global top-k (`topk_idx == [0,2,4]`), matching the reference's
`use_grouped_topk=False`. **Fails today.** Second case: add a **negative**
`e_score_correction_bias` and assert no out-of-group expert is ever chosen — pins the
mask-by-multiply bug.

### P1 — mHC (findings 7, 15, 20)

**T-H1 `hc_pre_returns_fp32_post_and_comb_for_bf16_residual`** — build a residual whose
values are exactly bf16-representable, run `hc_pre` from a bf16 and an f32 residual. Assert
`post.dtype() == F32 && comb.dtype() == F32`, and once fixed, that the two runs are
bit-equal. **Fails today** (dtype is BF16; `comb` max error ~2e-3).

**T-H2 `mhc_runtime_honors_config_fields`** — extend `dummy_cfg` with
`"hc_mult": 2, "hc_sinkhorn_iters": 7, "hc_eps": 1e-3`; assert `V4MHCRuntime::from_cfg`
reflects all three. **Fails today** (4 / 20 / 1e-6). Optional second half: with `hc_mult = 2`
and correctly-shaped `hc_head_fn: [2, 2*hidden]`, assert `V4MHCHead::try_load(...).is_some()` —
today it returns `None` (shape probe uses hardcoded 4), which is the exact path that
silently demotes a whole model to the 3-D bridge.

**T-H3 `use_4d_mhc_demotion_is_not_silent`** — construct a model missing one layer's mHC
tensors and assert a `WARN`-level event is emitted at the model level (not just per layer).
**Fails today** — `deepseek4.rs:3294` has no log line.

### P1 — compressor precision (finding 16)

**T-C2 `compressor_state_and_ape_must_be_fp32`** — build a `V4Compressor` twice from the same
checkpoint map, once via a BF16 `ShardedVarBuilder` and once F32, with an `ape` whose values
are non-representable in bf16. Drive both with the same F32 `xs`; assert
`max_abs_diff < 1e-6`. **Fails today** — `ape`, `norm.weight`, and the fused-linear output
are all rounded to the VarBuilder dtype.

### P1 — quantization policy (findings 8, 9, 10)

**T-Q1 `mtp_tail_is_floored_at_int8`** — unit-test the dtype-selection helper: for a tensor
carrying the MTP marker, assert the resolved `IsqType` is never one of
`{Q2K, Q3K, AFQ2, AFQ3, QtipBitshift2, Qtip2b}` regardless of the global request.
**Fails today** — no such helper and no floor exists. Requires introducing a marker
(the simplest is a `layer_num` sentinel or a name-tagged tuple).

**T-Q2 `qtip2b_3d_expert_default_matches_qtip2`** — colocate in `qtip/bake_quality_tests.rs`
next to the existing `QtipLayer` test. Assert
`Qtip2bLayer` 3-D quantize through the production decision path uses
`QtipMode::default_expert_mode()` (i.e. Viterbi) and `use_rotation == true`.
**Fails today** — `unquantized/mod.rs:412-420` hardcodes Greedy.

**T-Q3 `spec_draft_and_verify_use_the_same_kernel_family`** — pure-CPU: assert that the
dispatch predicate is a function of the *layer*, not of `n_tokens`, once a pin is
introduced; until then, assert that `Qtip2bLayer::forward` produces bit-identical rows for
`x[0:1]` and for row 0 of `x[0:4]`. **Fails on GPU today** (GEMV vs dequant+GEMM); on CPU it
documents the intended invariant.

### P1 — quality gate (finding, §8)

**T-G1 `termination_rate_gate`** — make `run_coherence.py` and `run_gsm8k.py` exit non-zero
on threshold breach, and add a termination-rate A/B: run N ≥ 100 prompts at
`max_tokens ≥ 512` under two quant configs and assert
`|stop_rate_a − stop_rate_b| < 5pp`. **Not a Rust test** — a harness change. Today both
scripts exit 0 unconditionally and neither compares configs.

### P2 — lock in correct behavior (regression guards)

**T-C1 `union_mask_double_counts_like_reference`** — `ratio ∈ {4,128}`, `window = 128`,
`t_k ∈ {128, 129, 512, 640}`. Assert (i) `#raw_visible == min(t_k, 128)`;
(ii) `#comp_visible == t_k / ratio` (matching `metadata_kernel.py:41, 52`); (iii) at
`t_k = 128, ratio = 128` the single visible compressed entry and the full raw window cover
the **same** absolute positions — i.e. the intersection is non-empty and no exclusion is
applied. **Passes today.** A future "optimization" that masks window-overlapping compressed
entries — the intuitive-but-wrong fix — trips (iii) immediately.

**T-S1 `sinkhorn_last_normalization_is_over_columns`** — `hc = 3`, strongly asymmetric
`[1,3,3]` input, **`sinkhorn_iters = 1`** (essential: at 20 iters Sinkhorn converges and both
axes look stochastic, hiding the bug). Assert every column sum within 2e-6 of 1.0, max row
deviation > 1e-2, and `max_col_err < max_row_err / 100`. **Passes today.** A row-last
implementation, an off-by-one loop bound, or moving eps into the first row denominator all
break at least one assertion.

**T-S2 `sinkhorn_eps_placement_matches_reference_exactly`** — `hc = 2`, all-zero `[1,2,2]`,
`sinkhorn_iters = 1`, **`eps = 0.25`**. Hand-derived reference (`mhc.py:70-77`): row softmax →
0.5; `+ eps` → 0.75; `col_sum = 1.5`; `/(1.5 + 0.25)` → `0.428571…`. Assert every element
within 1e-6 of `0.75/1.75`. Under the "eps in the first row denominator" variant you get
`0.38095` — fails by 0.048. **Passes today.**

**T-D1 `v4_e2e_synthetic_fixture_is_unignored`** — `mistralrs-core/tests/v4_e2e.rs:455-456`
carries an `#[ignore]` because the synthetic weights use V3-style `q_head_dim` arithmetic.
Re-author the shapes against the current `DeepSeekV4Config` and remove the attribute. **Arc's
only V4 end-to-end test does not run.** The live `synthetic_load_smoke.rs` tests are
structural only (load / forward / finite / deterministic shape) — zero numerical assertions,
which is precisely the gap every test above fills.

**T-T1 `tid2eid_loads_from_int32_checkpoint`** — write `gate.tid2eid` as `DType::I32` (the
reference's real dtype, `hash_topk.py:41`) in the fixture instead of `I64` and re-run the
load+forward smoke. Settles the one load-path unknown in §4(d) without a checkpoint
download; if `get_with_hints_dtype(..., DType::I64)` (`deepseek4.rs:1554-1559`) asserts rather
than converts, the first real V4 load fails hard and this catches it now.

---

## 12. Recommended action order

1. **Fetch `config.json` from `deepseek-ai/DeepSeek-V4-Flash`.** One file settles findings
   11, 12, 24, the `swiglu_limit` presence question, and `compress_rope_theta`. Zero cost.
2. **Fix MTP seeding (finding 1)** — add a `pre_hc_head` capture channel returning
   `[T, hc_mult·hidden]`, mirroring `hidden_states_before_norm`. Nothing else about MTP —
   including the §6 quantization floor — is measurable until this lands.
3. **Fix the two masking bugs (findings 3, 4)** — one broadcast-add for the caller mask;
   route V4 away from `sinks_attn_varlen`. Both are HIGH and both are small.
4. **Add the shared-expert swiglu clamp (finding 5)** — unconditional, device-independent,
   trivially testable via T-E1.
5. **Flip the YaRN-on-Standard default (finding 6)**, keeping the env var as the escape hatch.
6. **Land the P2 regression guards (T-C1, T-S1, T-S2)** before touching the compressor or
   mHC — they lock in the parts that are already correct, and the union-masking one guards
   against an intuitive-but-wrong "optimization".
7. Then findings 7–10, 13–23 in severity order.
