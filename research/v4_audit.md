# V4 Reality Audit — DeepSeek-V4-Flash

**Sources**: HF safetensors index, SGLang reference implementation, Arc current code.
**Date**: 2026-05-22. **Auditor**: Agent 1.

---

## 0. Executive Summary (TL;DR diff)

**Arc's `mistralrs-core/src/models/deepseek4.rs` is wrong about almost every V4 attention detail.** It was written speculatively against the V3 MLA pattern. The actual V4 weight layout, head-dim math, sparse-attention modules, and per-layer dispatch all differ. Concrete deltas:

| Aspect | Arc (current) | V4 reality | Fix priority |
|---|---|---|---|
| Format | Tries V4 native (`attn.*`) AND HF (`self_attn.*`) auto-detect | V4 Flash safetensors are **V4 native only** (`layers.X.attn.*`) | P0 |
| KV projection | LoRA: `kv_a_proj_with_mqa` (4096 → `kv_lora_rank+qk_rope_head_dim`) + `kv_b_proj` (`kv_lora_rank` → `n_heads*(qk_nope+v_head_dim)`) | **FUSED**: single `wkv` (4096 → 512). MQA: one KV head, broadcast to 64 Q heads at the attention kernel. **No** `kv_b_proj`. | P0 |
| `head_dim` | `qk_nope_head_dim + qk_rope_head_dim` (Arc currently expects both in config) | Config has `head_dim=512`, `qk_rope_head_dim=64`. **No `qk_nope_head_dim` in config**; SGLang derives it as `head_dim - qk_rope_head_dim = 448`. | P0 |
| `kv_lora_rank` | Required config field | **Not in V4 config at all** (V4 has no KV LoRA — fused single wkv) | P0 |
| `v_head_dim` | Required config field | **Not in V4 config**. V4 uses unified `head_dim=512` for both K and V (the MLA "absorb V into K" trick at scale). | P0 |
| `qk_nope_head_dim` | Required config field | **Not in V4 config**. Derived: `head_dim - qk_rope_head_dim`. | P0 |
| Number of dense layers | Arc defaults `first_k_dense_replace=0` (correct) | All 43 layers are MoE. `first_k_dense_replace=0` confirmed. | OK |
| `num_experts_per_tok` | Arc reads from config (`Option<usize>`) | `num_experts_per_tok=6` (V3 was top-8) | OK (config-driven) |
| Number of experts | Reads `n_routed_experts` | `n_routed_experts=256`, `n_shared_experts=1` | OK |
| Compressor weight name | Probes `compressor.weight` (single tensor) | Real tensors: `attn.compressor.{wgate.weight, wkv.weight, norm.weight, ape}`. Compressor is a learned **gated linear + RMSNorm + APE** module, not a single linear. | P0 |
| Indexer module | **Missing entirely** in Arc | V4 has a `C4Indexer` on every `compress_ratio=4` layer with own `wq_b`, `weights_proj`, sub-`compressor`, FP8 paged MQA logits + top-k page selection. | P0 |
| mHC residual replacement | Arc only loads `hc_head_*` weights, no forward integration. Doesn't load `hc_attn_*`, `hc_ffn_*`. | mHC **replaces** the standard residual at every layer. Each layer has `hc_attn_{fn,scale,base}` and `hc_ffn_{fn,scale,base}` parameters that gate the residual through a Sinkhorn-normalized soft mixture. The top-level `hc_head_*` is a separate final projection. | P0 |
| MTP head | Arc loads `mtp.layers.0.h_proj` + `e_proj` (2 linears) | V4 MTP has a **full transformer decoder layer** (`mtp.0.attn.*`, `mtp.0.ffn.*`, full 256-expert MoE) + `h_proj`, `e_proj`, `hnorm`, `enorm`, `norm`, and its own `hc_head_*` + `hc_attn_*` + `hc_ffn_*`. Arc's path will silently load 2 small linears and miss ~3 GB of MTP weights. | P0 |
| `attn_sink` parameter | Not loaded | Every layer has `attn.attn_sink` ∈ ℝ^64 (one float per head, fp32). Used as a learned constant absorbing column in the softmax. | P0 |
| o_proj LoRA | `wo_a` + `wo_b` ✓ (right structure) but uses `cfg.v_head_dim` which is absent | Real `wo_a` shape: `[n_heads * head_dim / n_groups, n_groups * o_lora_rank] = [4096, 8192]`. `wo_b`: `[n_groups * o_lora_rank, hidden_size] = [8192, 4096]`. Forward is a `bhr,hdr->bhd` einsum per group, not flat matmul. | P1 |
| RoPE base per layer | Single rope theta for whole model | **Two RoPE bases**: `rope_theta=10000` for `compress_ratio=0` layers, `compress_rope_theta=160000` for all `compress_ratio != 0` (compressed) layers. | P1 |
| Sliding window | `cfg.sliding_window=128` ✓ | `sliding_window=128` ✓ — confirmed; used inside the C4Indexer's sparse-token selection, not as a hard local-window mask. | OK |
| Sparse attention dispatch | Arc has a "blend main + SWA at 0.5/0.5" plan | V4 dispatches K-cache reads through a **top-k page table** built by the C4Indexer (selecting `index_topk=512` pages per query). No 50/50 blend. The compress=128 layers use a different mechanism (HCA: dense MQA over 128× compressed K, no top-k). | P0 |
| Quantization | Arc field `quantization_config: Option<QuantizedConfig>` (loaded but inert at scaffolding tier) | V4 Flash ships in **FP8 e4m3** with UE8M0 scales, 128×128 weight blocks. Every weight has a `.weight` (fp8) and `.scale` (fp32) tensor. Arc's `safetensors` loader will fail unless a fp8 dequant pass is wired in. | P0 |
| Tokenizer | Arc uses `tokenizers` crate via `tokenizer.json` | V4 ships `tokenizer.json` (6.4 MB, PreTrainedTokenizerFast format). **Compatible** with Arc's existing loader. **No `chat_template.jinja`** at the standard HF URL → chat template must be hardcoded or pulled from V3. | P2 |

**Bottom line**: Arc's `deepseek4.rs` will not load even one tensor of a real V4 Flash checkpoint without a major rewrite. It is a V3-shaped stub that pre-dates the V4 weight publication. Plan to rewrite ~80% of the file.

---

## 1. V4 Flash safetensors index — Full breakdown

### Source
- URL: `https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/resolve/main/model.safetensors.index.json`
- Local cache: `/tmp/v4_flash_index.json` (5.1 MB)
- **Total tensors**: 69,187
- **Total model size**: 159.61 GB
- **Shard count**: 46 files (`model-00001-of-00046.safetensors` … `model-00046-of-00046.safetensors`)
- **Quantization**: fp8 e4m3 + UE8M0 scales (weights as `.weight` fp8, scales as `.scale` fp32), 128×128 weight blocks (from `config.json:quantization_config`)

### Top-level tensors (6 tensors)
```
embed.weight              # vocab_size=129280, hidden_size=4096 → [129280, 4096] bf16
head.weight               # lm_head [vocab_size, hidden_size]
norm.weight               # final RMSNorm [4096]
hc_head_base              # [hc_mult=4] fp32
hc_head_fn                # [hc_mult, hc_mult * hidden_size] = [4, 16384] fp32
hc_head_scale             # [1] fp32
```

### Per-layer pattern (43 layers, indexed 0..=42)
There are exactly **4 distinct layer patterns**, matching `config.compress_ratios = [0, 0, 4, 128, 4, 128, ..., 4, 0]`:

#### Pattern A — Layers 0, 1 (compress_ratio = 0; "warmup" full-attention layers)
```
layers.X.attn.attn_sink             # [n_heads=64] fp32
layers.X.attn.kv_norm.weight        # [head_dim=512] fp32
layers.X.attn.q_norm.weight         # [q_lora_rank=1024] fp32
layers.X.attn.wkv.weight            # [head_dim=512, hidden_size=4096] fp8
layers.X.attn.wkv.scale             # [4, 32] fp32 (128×128 blocks)
layers.X.attn.wo_a.weight           # [n_groups*o_lora_rank=8192, n_heads*head_dim/n_groups=4096] fp8
layers.X.attn.wo_a.scale
layers.X.attn.wo_b.weight           # [hidden_size=4096, n_groups*o_lora_rank=8192] fp8
layers.X.attn.wo_b.scale
layers.X.attn.wq_a.weight           # [q_lora_rank=1024, hidden_size=4096] fp8
layers.X.attn.wq_a.scale
layers.X.attn.wq_b.weight           # [n_heads*head_dim=32768, q_lora_rank=1024] fp8
layers.X.attn.wq_b.scale
layers.X.attn_norm.weight           # [hidden_size=4096] fp32
layers.X.hc_attn_base               # [mix_hc=24] fp32 where mix_hc=(2+hc_mult)*hc_mult=(2+4)*4=24
layers.X.hc_attn_fn                 # [mix_hc=24, hc_dim=16384] fp32 where hc_dim=hc_mult*hidden_size
layers.X.hc_attn_scale              # [3] fp32
layers.X.hc_ffn_base                # [24] fp32
layers.X.hc_ffn_fn                  # [24, 16384] fp32
layers.X.hc_ffn_scale               # [3] fp32
layers.X.ffn.experts.{0..255}.{w1,w2,w3}.{weight, scale}    # 256 routed experts
layers.X.ffn.shared_experts.{w1,w2,w3}.{weight, scale}      # 1 shared expert
layers.X.ffn.gate.weight            # [n_routed_experts=256, hidden_size=4096]
layers.X.ffn.gate.bias              # [256] (e_score_correction_bias)
layers.X.ffn.gate.tid2eid           # int32 [num_hash_layers=3, hidden_size?] (hash-routing table)
layers.X.ffn_norm.weight            # [hidden_size]
```

#### Pattern B — Layer 2 only (compress_ratio = 4; first sparse layer, *with* indexer)
Same as Pattern A, PLUS:
```
layers.2.attn.compressor.ape                   # [ratio=4, coff*head_dim] = [4, 1024] fp32 (coff=2 for ratio=4 b/c overlap=True)
layers.2.attn.compressor.norm.weight           # [head_dim=512] fp32
layers.2.attn.compressor.wgate.weight          # [hidden_size=4096, head_dim=512] bf16 (auto-converted by load_weights into wkv_gate)
layers.2.attn.compressor.wkv.weight            # [hidden_size=4096, head_dim=512] bf16
layers.2.attn.indexer.compressor.ape           # [4, 4*head_dim_indexer=4*128*2=1024] fp32
layers.2.attn.indexer.compressor.norm.weight   # [index_head_dim=128] fp32
layers.2.attn.indexer.compressor.wgate.weight  # [hidden_size, 2*coff*index_head_dim] bf16
layers.2.attn.indexer.compressor.wkv.weight    # [hidden_size, 2*coff*index_head_dim] bf16
layers.2.attn.indexer.weights_proj.weight      # [hidden_size, index_n_heads=64] bf16
layers.2.attn.indexer.wq_b.weight              # [q_lora_rank=1024, index_n_heads*index_head_dim=64*128=8192] fp8
layers.2.attn.indexer.wq_b.scale
```

#### Pattern C — Layers 3, 5, 7, ..., 41 (compress_ratio = 128; HCA, NO indexer)
Pattern A + only the `attn.compressor.*` block (4 tensors). **No indexer.** This is dense MQA over 128× compressed K.

#### Pattern D — Layers 4, 6, 8, ..., 42 (compress_ratio = 4; CSA, WITH indexer)
Same as Pattern B (compressor + indexer).

### Layer-to-pattern mapping (from `config.compress_ratios`)
```
Index:  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42
Ratio:  0  0  4 128 4 128 4 128 4 128 4 128 4 128 4 128 4 128 4 128 4 128 4 128 4 128 4 128 4 128 4 128 4 128 4 128 4 128 4 128 4 128 0
Pttn:   A  A  B  C  D  C  D  C  D  C  D  C  D  C  D  C  D  C  D  C  D  C  D  C  D  C  D  C  D  C  D  C  D  C  D  C  D  C  D  C  D  C  A
```
**Layer 42 is special**: ratio=0 (back to standard), no compressor or indexer. So layers 0, 1, 42 are full-attention layers (with mHC); layers 3,5,7,...,41 are HCA; layers 2, 4, 6, ..., 40 are CSA (with indexer).

### MTP module (`mtp.0.*` — single nextn predictor)
The MTP block is a **full transformer decoder layer** (V4-shape attention + 256-expert MoE), wrapped with `h_proj`/`e_proj`/`hnorm`/`enorm`:
```
mtp.0.attn.{attn_sink, kv_norm.weight, q_norm.weight, wkv.{weight,scale}, wo_a.{w,s}, wo_b.{w,s}, wq_a.{w,s}, wq_b.{w,s}}
mtp.0.attn_norm.weight
mtp.0.ffn.experts.{0..255}.{w1,w2,w3}.{weight, scale}      # 256 experts (yes, full set)
mtp.0.ffn.gate.{bias, weight}
mtp.0.ffn.shared_experts.{w1,w2,w3}.{weight, scale}
mtp.0.ffn_norm.weight
mtp.0.h_proj.{weight, scale}     # [hidden, hidden]
mtp.0.e_proj.{weight, scale}     # [hidden, hidden]
mtp.0.hnorm.weight               # [hidden]
mtp.0.enorm.weight               # [hidden]
mtp.0.norm.weight                # [hidden]
mtp.0.hc_attn_{base, fn, scale}  # same shapes as decoder
mtp.0.hc_ffn_{base, fn, scale}
mtp.0.hc_head_{base, fn, scale}  # MTP has its own hc_head (the model has another at top level)
```
**MTP compress_ratio is 0** (per `deepseek_v4_nextn.py:47`: `COMPRESS_RATIO_NEXTN_LAYER = 0`).

### Tensor count breakdown
```
embed:           1
head:            1
norm:            1
hc_head:         3 (top-level)
attn_sink:      43
attn_norm:      43
ffn_norm:       43
attn:          516  (12-14 attn weight tensors × 43 layers, varies)
ffn_expert:  66048  (256 experts × 6 tensors × 43 layers)
ffn_shared:    258  (1 expert × 6 tensors × 43 layers)
ffn_gate:       86  (2 tensors × 43 layers: weight + bias; tid2eid only some layers?)
compressor:    248  (41 layers × 4-5 tensors)
indexer:        63  (21 layers × 3 tensors per indexer; sub-compressor counted under compressor)
hc_attn:       129  (3 × 43)
hc_ffn:        129  (3 × 43)
mtp:          1575  (full MTP layer)
TOTAL:       69187 ✓
```

---

## 2. SGLang MQALayer reference structure (`research/code/06_foundation/sglang/python/sglang/srt/models/deepseek_v4.py:172-654`)

### Class attributes (from `__init__`, lines 172-360)
```python
class MQALayer(nn.Module):
    # Dimensions (lines 191-203)
    self.dim                = config.hidden_size                              # 4096
    self.qk_rope_head_dim   = config.qk_rope_head_dim                         # 64
    self.qk_nope_head_dim   = config.head_dim - config.qk_rope_head_dim       # 512 - 64 = 448
    self.head_dim           = self.qk_rope_head_dim + self.qk_nope_head_dim   # 512 (== config.head_dim)
    self.n_heads            = config.num_attention_heads                      # 64
    self.n_local_heads      = self.n_heads // attn_tp_size                    # 64 // tp_size
    self.n_groups           = config.o_groups                                 # 8
    self.n_local_groups     = self.n_groups // attn_tp_size                   # 8 // tp_size
    self.rope_head_dim      = config.qk_rope_head_dim                         # 64
    self.softmax_scale      = self.head_dim ** -0.5                           # 1/sqrt(512)
    self.q_lora_rank        = config.q_lora_rank                              # 1024
    self.o_lora_rank        = config.o_lora_rank                              # 1024
    self.compress_ratio     = config.compress_ratios[layer_id]                # 0 | 4 | 128

    # Assertions (lines 213-214):
    assert self.head_dim == config.head_dim
    assert config.num_key_value_heads == 1                                    # MQA: 1 KV head

    # RoPE base (line 220): two-mode dispatch
    rope_base = config.compress_rope_theta if self.compress_ratio else rope_theta
    # → 160000 for compressed layers, 10000 for layers 0, 1, 42

    # Sub-modules (lines 263-358):
    self.compressor: Optional[Compressor]           # present iff compress_ratio != 0
    self.indexer:    Optional[C4Indexer]            # present iff compress_ratio == 4
    self.attn_sink:  nn.Parameter[fp32, (n_heads,)] # always present
    self.wq_a:       ReplicatedLinear(hidden_size, q_lora_rank, bias=False)
    self.q_norm:     RMSNorm(q_lora_rank, eps=eps)
    self.wq_b:       ColumnParallelLinear(q_lora_rank, n_heads * head_dim, tp_rank, tp_size)
    self.wkv:        ReplicatedLinear(hidden_size, head_dim, bias=False)     # SINGLE FUSED PROJECTION
    self.kv_norm:    RMSNorm(head_dim, eps=eps)                              # normalises the KV vector
    self.wo_a:       ColumnParallelLinear(n_heads * head_dim // n_groups, n_groups * o_lora_rank)
    self.wo_b:       RowParallelLinear(n_groups * o_lora_rank, hidden_size)
    self.attn_mqa:   RadixAttention(n_local_heads, head_dim, softmax_scale, num_kv_heads=1, ...)
```

### Forward pass (lines 500-654)
```python
def forward(self, x: [T, hidden_size], positions, forward_batch):
    # 1. Q projection (lines 510-515)
    q_lora, _ = self.wq_a(x)                            # [T, q_lora_rank=1024]
    q_lora    = self.q_norm(q_lora)                     # RMSNorm
    q         = self.wq_b(q_lora)                       # [T, n_heads*head_dim=32768]
    q         = q.view(-1, n_local_heads, head_dim)     # [T, H, 512]
    fused_q_norm_rope(q, q_out, eps, freqs_cis, positions)
    # → q has rope applied to the LAST qk_rope_head_dim=64 dims only

    # 2. KV projection — SINGLE FUSED wkv (lines 404-416)
    kv, _ = self.wkv(x)                                 # [T, head_dim=512]
    # The wkv output IS the K AND V for this token, with a single MQA head.
    # The first qk_nope_head_dim=448 dims are the "nope" portion (no rope),
    # the last qk_rope_head_dim=64 dims get rope applied via fused_norm_rope_inplace.
    # kv_norm.weight (length 512) RMSNormalises the full vector before rope.
    # Then it's written directly to the FlashMLA paged cache.

    # 3. Optional compressor / indexer forward (lines 540-553)
    if self.compressor is not None:
        attn_backend.forward_core_compressor(x, ..., self.compressor)
    if self.indexer is not None:
        self.indexer(x, q_lora=q_lora, ...)

    # 4. Attention kernel (lines 608-617)
    o = attn_backend.forward(q=q, k=kv, v=kv, layer=self.attn_mqa,
                              compress_ratio=self.compress_ratio,
                              attn_sink=self.attn_sink, save_kv_cache=False)
    # The backend reads K and V from the paged cache (kv is a sentinel here when kv is None);
    # for compress_ratio=4, it reads ONLY the top-k=512 indexed pages.
    # for compress_ratio=128, it reads the entire 128× compressed cache.
    # for compress_ratio=0, it does standard dense MQA over the full sequence.
    # attn_sink is added as an extra softmax column with key = attn_sink, value = 0.

    # 5. INVERSE rope on the last qk_rope_head_dim dims of o (lines 619-625)
    fused_rope_inplace(o[..., -qk_rope_head_dim:], None, freqs_cis, positions, inverse=True)

    # 6. Grouped o_proj: wo_a (per-group) then wo_b (lines 627-652)
    o = o.view(o.shape[0], n_local_groups, -1)          # [T, G=8, n_heads*head_dim/n_groups=4096]
    o = einsum("tgd,grd->tgr", o, wo_a.view(G, o_lora_rank, -1))   # [T, G, R=1024]
    o, _ = self.wo_b(o.flatten(1))                      # [T, hidden_size=4096]
    return o
```

### MQA broadcast: where does it happen?
Inside `attn_backend.forward` — at the kernel level, not in the Python module. Q has 64 heads, K/V has 1 head. The FlashMLA-style kernel implicitly broadcasts K/V across Q heads. There is **no explicit `.expand()` or `.repeat()`** call. The KV cache stores just one head per token (massive memory savings vs the 64-head expansion).

### compress_ratio dispatch in the backend
- `compress_ratio == 0`: standard dense MQA over the full KV cache for the sequence.
- `compress_ratio == 4` (CSA): the `C4Indexer` produces FP8 paged MQA logits, top-k=512 page selection. The main attention then reads K/V only from those 512 pages.
- `compress_ratio == 128` (HCA): K/V cache is the **128× compressed** version (one stored vector per 128 input tokens). The compressor learns to gate/select. Attention runs dense over this short compressed cache.

### `first_k_dense_replace` confirmation
Config explicitly does NOT set `first_k_dense_replace` (so default = 0). All 43 layers are MoE. Confirmed by tensor inventory: every layer 0..=42 has `ffn.experts.0..255` and `ffn.shared_experts`.

### `num_experts_per_tok` confirmation
Config: `num_experts_per_tok=6` (V3 used 8). V4 routes each token to top-6 routed experts + 1 shared expert.

### DecoderLayer forward — mHC replaces residuals
Lines 910-1001 of `deepseek_v4.py`:
```python
def forward(self, positions, hidden_states, input_ids, forward_batch, input_ids_global):
    # The 3D shape [n_tokens, hc_mult=4, hidden_size] persists through the layer.
    residual = hidden_states                              # 3D
    hidden_states, post, comb, norm_fused = self.hc_pre(  # collapse 4 streams → 1
        hidden_states, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base,
        norm=self.input_layernorm,                        # rmsnorm fused into hc_pre
    )
    hidden_states = self.self_attn(x=hidden_states, ...)  # 2D attn input
    hidden_states = self.hc_post(hidden_states, residual, post, comb)  # back to 3D, mixes residual

    residual = hidden_states
    hidden_states, post, comb, _ = self.hc_pre(
        hidden_states, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base,
        norm=self.post_attention_layernorm,
    )
    hidden_states = self.mlp(hidden_states, ...)          # MoE
    hidden_states = self.hc_post(hidden_states, residual, post, comb)
    return hidden_states                                  # 3D, [T, hc_mult, hidden]
```

mHC math (lines 774-908, simplified):
```
# hc_pre: per-token soft-route across hc_mult=4 parallel streams
x_flat   = x.flatten(1).float()                          # [T, hc_mult*hidden]
mixes    = (linear(x_flat, hc_fn) * rsqrt(...))          # [T, 1, mix_hc=24]
pre, post, comb = sinkhorn_split(mixes, hc_scale, hc_base, ...)    # Sinkhorn-normalised mixing
y        = (pre * x_flat.view(T, hc_mult, hidden)).sum(dim=1)      # [T, hidden] — one stream out
# hc_post: scatter the single output back into hc_mult streams + add residual mix
return (post * x.unsqueeze(1) + (comb * residual.unsqueeze(2)).sum(1))   # [T, hc_mult, hidden]
```

The model carries **4 parallel residual streams** (`hc_mult=4`), and at each layer it learnedly mixes them down to one for attn/MoE, then mixes back up. This replaces the standard `x + sublayer(norm(x))` residual.

### MTP head (`deepseek_v4_nextn.py:50-201`)
```python
class DeepseekV4ModelNextN(nn.Module):
    self.embed_tokens                                       # ParallelLMHead (shared with main)
    self.enorm, self.hnorm = RMSNorm(hidden, eps), RMSNorm(hidden, eps)
    self.hc_head_{fn, base, scale}                          # mHC final head
    self.e_proj  = ReplicatedLinear(hidden, hidden)
    self.h_proj  = ReplicatedLinear(hidden, hidden)
    self.decoder = DeepseekV4DecoderLayer(layer_id=0, is_nextn=True, compress_ratio_override=0)

def forward(self, input_ids, positions, forward_batch):
    # The previous-step hidden_states arrives via forward_batch.spec_info.hidden_states
    # (in shape [T, hc_mult, hidden] — already at mHC width).
    hc_flat            = spec_info.hidden_states.view(T*hc_mult, hidden)
    h_proj_out         = self.h_proj(self.hnorm(hc_flat))
    h_proj_hidden      = h_proj_out.view(T, hc_mult, hidden)
    e_proj_hidden      = self.e_proj(self.enorm(self.embed_tokens(input_ids)))
    hidden_states      = e_proj_hidden[:, None, :] + h_proj_hidden    # [T, hc_mult, hidden]
    hidden_states      = self.decoder(positions, hidden_states, input_ids, fb, input_ids_global)
    pre_hc_head        = hidden_states.flatten(1)
    hidden_states      = self.hc_head(hidden_states, hc_head_fn, hc_head_scale, hc_head_base)
    return self.shared_head.norm(hidden_states), pre_hc_head
```

### Weight renaming (`load_weights` → `remap_weight_name_to_dpsk_hf_format`, lines 1365-1423)
Quoted exactly:
```python
# Top-level (lines 1368-1375)
if name == "embed.weight":   return "model.embed_tokens.weight"
if name == "head.weight":    return "lm_head.weight"
if name == "norm.weight":    return "model.norm.weight"
if name.startswith("hc_head_"): return "model." + name

# MTP rewriting (lines 1377-1403)
if is_nextn and name.startswith("mtp."):
    parts = name.split(".", 2)
    if len(parts) >= 3:
        rest = parts[2]   # e.g. "attn.wq_a.weight" or "h_proj.weight"
        nextn_spec_prefixes = ["e_proj", "h_proj", "emb", "enorm", "hnorm", "norm", "head", "hc_head"]
        is_nextn_spec = any(rest.startswith(p) for p in nextn_spec_prefixes)
        if is_nextn_spec:
            if rest.startswith("emb.tok_emb"):    rest = rest.replace("emb.tok_emb", "embed_tokens")
            elif rest == "norm.weight":           rest = "shared_head.norm.weight"
            elif rest.startswith("head."):        rest = "shared_head.head.weight"
            elif rest == "e_proj.scale":          rest = "e_proj.weight_scale_inv"
            elif rest == "h_proj.scale":          rest = "h_proj.weight_scale_inv"
        name = f"model.layers.{num_hidden_layers}." + rest    # MTP lives at virtual layer index 43

# Layer rewriting (lines 1405-1423)
if name.startswith("layers."):       name = "model." + name
name = name.replace(".attn.", ".self_attn.")           # V4 native → HF
name = name.replace(".ffn.", ".mlp.")
name = name.replace(".attn_norm.", ".input_layernorm.")
name = name.replace(".ffn_norm.", ".post_attention_layernorm.")

if "self_attn" in name:
    name = name.replace(".scale", ".weight_scale_inv")        # fp8 scale tensor rename

name = name.replace(".gate.tid2eid", ".topk.tid2eid")
name = name.replace(".gate.bias", ".gate.e_score_correction_bias")
name = name.replace(".w1.", ".gate_proj.")               # SwiGLU naming
name = name.replace(".w2.", ".down_proj.")
name = name.replace(".w3.", ".up_proj.")
if "mlp" in name:
    name = name.replace(".scale", ".weight_scale_inv")
```

### Weight-loading special cases (lines 1462-1730)
- **wqkv_a fusion** (if `SGLANG_OPT_FUSE_WQA_WKV`): wq_a.weight + wkv.weight stacked into a single `wqkv_a.weight` (`torch.cat([q, kv], dim=0)` along output channel).
- **Compressor wkv_gate fusion** (lines 1632-1663): The compressor's `wkv.weight` and `wgate.weight` are loaded as one combined param `wkv_gate.weight = torch.cat([kv, wgate], dim=0)`. Arc must replicate this fusion.

---

## 3. Q/K shape arithmetic — resolved

The mystery (q is `[H, T, 512]` after split, how does it attend over `k` of shape `[?, T, ?]`):

**Q after projection**: `wq_b` outputs `[T, n_heads * head_dim]` = `[T, 64*512]` = `[T, 32768]`. Reshape to `[T, n_heads=64, head_dim=512]`. RoPE is applied to the last 64 dims of head_dim (the rope portion), not split into separate q_nope/q_rope tensors. The full 512-dim head vector goes into attention.

**K via wkv**: `wkv` outputs `[T, head_dim=512]`. This is a **single MQA head**. So the K cache is `[T, 1, 512]`. The RoPE is applied to the last 64 dims of this single vector (via `fused_norm_rope_inplace`). The kernel broadcasts this single head across all 64 Q heads at attention time.

**Dimensional consistency**: Both q (per head) and k are 512-dim vectors. Softmax dot-product is `q_i · k_j / sqrt(512)`. No nope/rope split is materialised in the kernel API — the rope rotation is just applied in-place to the right 64 dims of both q and k before the kernel reads them.

**No `kv_b_proj`**. V4 absorbed the MLA "split K into nope + rope" structure into a single fused output. The "kv_lora_rank" concept does not exist in V4 weights. The "M" in MQA (multi-query) here means literally one KV head — broadcast across 64 Q heads in the kernel — not the V3-style MLA "compressed KV vector projected back up to per-head".

**The output dim after attention is `[T, n_heads * head_dim]`** (= `[T, 32768]`), exactly like a standard MHA, despite having only 1 KV head — because each Q head produces its own attention-weighted sum over the broadcast K/V.

---

## 4. CUDA kernel sources available for vendoring

### Location: `research/code/06_foundation/sglang/python/sglang/jit_kernel/csrc/deepseek_v4/`
These are header-only TVM-FFI templated CUDA kernels (`.cuh`), JIT-compiled via `load_jit()`. All Blackwell sm100-targeted.

| File | Description | LOC est. |
|---|---|---|
| `common.cuh` | Shared utilities (`plan_compress_prefill` host function, shared types) | small |
| `c4.cuh` | `FlashCompress4Kernel<head_dim, dtype_in, dtype_out, pdl>::run_{decode,prefill}` — CSA 4× compression. | mid |
| `c4_v2.cuh` | v2 (refactored) version of the C4 kernel | mid |
| `c128.cuh` | `FlashCompress128Kernel<...>` — HCA 128× compression | mid |
| `c128_online.cuh` | `FlashCompress128OnlineKernel<head_dim, pdl>` — online (single-token) HCA compression. Lighter than the prefill variant. | mid |
| `c128_online_v2.cuh` | Refactored v2 | mid |
| `c128_v2.cuh` | Refactored prefill variant | mid |
| `c_plan.cuh` | Compress plan / index buffer logic | small |
| `fused_norm_rope.cuh` | `FusedNormRopeKernel<dtype, head_dim, rope_dim, pdl>::forward` — RMSNorm + RoPE fused in one kernel pass | mid |
| `fused_norm_rope_v2.cuh` | Refactored variant | mid |
| `main_norm_rope.cuh` | Q-side normalization + rope (per-head) | small |
| `rope.cuh` | Standalone RoPE inplace + inverse-rope (for o_proj postprocess) | small |
| `store.cuh` | `FusedStoreCache{FlashMLA,Indexer}Kernel<input_dtype, index_dtype, page_size, pdl>::run` — fuses cache-store with norm/rope | mid |
| `paged_mqa_metadata.cuh` | `IndexerMetadataKernel::run` — builds the per-SM page table for the C4 indexer | small |
| `topk_v1.cuh`, `topk_v2.cuh` | `topk_transform_512` — page-level top-512 selection from FP8 paged MQA logits | mid |
| `hash_topk.cuh` | Hash-based MoE expert selection accelerator | small |
| `silu_and_mul_masked_post_quant.cuh` | Fused SiLU(w1) * w3 → fp8 quant (used inside MoE expert MLP) | mid |
| `mega_moe_pre_dispatch.cuh` | All-to-all preparation kernel for the MoE expert dispatch | mid |
| `hisparse_transfer.cuh` | Sparse-attention page transfer between HBM tiers | mid |

### Tilelang alternates (Python DSL)
`research/code/06_foundation/sglang/python/sglang/srt/layers/attention/dsv4/tilelang_kernel.py` provides Tilelang-DSL versions of the same kernels (mha, compress, indexer). These are NOT what we want to vendor — they require the tilelang runtime. Use the `.cuh` headers above.

### Other CUDA sources in sgl-kernel (already-built C++)
`research/code/06_foundation/sglang/sgl-kernel/csrc/attention/`:
- `cutlass_mla_kernel.cu` — Cutlass MLA reference (V3-era, but useful pattern for V4)
- `cutlass_sm100_mla/` — sm100-targeted MLA cutlass templates
- `merge_attn_states.cu` — chunked-prefill state merge
- `vertical_slash_index.cu` — sparse-attention index transforms

### Recommended vendoring approach for Arc
1. Start with **`fused_norm_rope.cuh`** (`FusedNormRopeKernel`) — simplest, no template explosion, directly portable to a `cuda_kernels` crate inside `mistralrs-quant`.
2. Then **`c4.cuh` + `topk_v2.cuh` + `paged_mqa_metadata.cuh`** — the CSA pipeline. These are independent of mHC.
3. Then **`c128_online.cuh`** — HCA (simpler than prefill C128).
4. Defer **`store.cuh`** (cache-write fusion) until after correctness is proved with bf16-intermediate path.

Each `.cuh` is ~200-600 LOC of CUDA. With sm_100 targeting + cutlass dependency, expect each port to take 30-90 min of agent time.

---

## 5. Diff vs Arc `mistralrs-core/src/models/deepseek4.rs` — line-by-line

### Config struct (lines 76-143)

| Line | Arc field | Reality | Action |
|---|---|---|---|
| 80 | `intermediate_size: usize` | Not in V4 config (V4 has only `moe_intermediate_size=2048`). Currently breaks deserialization. | Make optional |
| 81 | `moe_intermediate_size: usize` | `2048` ✓ | OK |
| 84 | `n_routed_experts: Option<usize>` | `256` (always present) | Make required |
| 89 | `num_experts_per_tok: Option<usize>` | `6` (always present) | Make required |
| 92 | `first_k_dense_replace: usize` (default 0) | Field absent in V4 config; default 0 is correct | OK |
| 106 | `q_lora_rank: Option<usize>` | `1024` (always present) | Make required |
| 107 | `qk_rope_head_dim: usize` | `64` ✓ | OK |
| 108 | **`kv_lora_rank: usize`** | **DOES NOT EXIST in V4 config**. Hard-fails to deserialize. | **REMOVE** |
| 109 | **`v_head_dim: usize`** | **DOES NOT EXIST in V4 config**. | **REMOVE** (or derive from head_dim) |
| 110 | **`qk_nope_head_dim: usize`** | **DOES NOT EXIST in V4 config**. Derived: `head_dim - qk_rope_head_dim = 448`. | **REMOVE** (compute from head_dim) |
| 113 | `n_group: usize` | Not in V4 config. NoAuxTc dispatch path uses it; we need to handle absence. | Make optional, default 1 |
| 114 | `topk_group: usize` | Not in V4 config. | Make optional, default 1 |
| MISSING | `head_dim: usize` (V4 has `head_dim=512`) | **Add new field** | ADD |
| MISSING | `num_key_value_heads: usize` (V4 has `=1`) | **Add new field** | ADD |
| MISSING | `num_hash_layers: usize` (V4 has `=3`) | **Add new field** | ADD |
| MISSING | `hc_mult: usize` (V4 has `=4`) | **Add new field** | ADD |
| MISSING | `hc_eps: f32` (V4 has `1e-6`) | **Add new field** | ADD |
| MISSING | `hc_sinkhorn_iters: usize` (V4 has `=20`) | **Add new field** | ADD |
| MISSING | `swiglu_limit: f32` (V4 has `=10.0`) | **Add new field** (clamp inside MoE experts) | ADD |
| MISSING | `routed_scaling_factor: f64` (V4 has `=1.5`, Arc default is 1.0) | Arc has it but default is wrong | Set default to 1.0; honor config value (1.5) |
| MISSING | `scoring_func: ScoringFunc` (V4 has `"sqrtsoftplus"`) | **New variant**, not Softmax/Sigmoid | ADD variant |

### Attention struct (lines 279-307)
- Line 281: `kv_a_proj_with_mqa: Arc<dyn QuantMethod>` — **should be `wkv` (full 4096 → 512 fused projection)**, no LoRA split.
- Line 282: `kv_a_layernorm: RmsNorm` — **rename to `kv_norm`** with normalised_shape = `head_dim=512` (not `kv_lora_rank`).
- Line 283: `kv_b_proj: Arc<dyn QuantMethod>` — **DELETE**. V4 has no kv_b projection.
- MISSING: `q_norm: RmsNorm` over `q_lora_rank=1024` (currently nested inside `QProj::Lora.norm`, which is OK).
- MISSING: `attn_sink: Tensor` (one fp32 vector of length 64).
- Line 299: `compressor_k: Option<V4Compressor>` — **WRONG NAME**. Real: single `attn.compressor` per layer (gated linear, not K/V split).
- Line 301: `compressor_v: Option<V4Compressor>` — DELETE.
- MISSING: `indexer: Option<C4Indexer>` field. CSA layers (compress_ratio=4) need this.
- MISSING: `compress_rope_theta` separate RoPE instance (or pass `rope_base` per-layer).

### Attention::new (lines 308-531)
- Lines 321-359 (Q projection LoRA): Arc probes `q_norm` OR `q_a_layernorm` — **only `q_norm` exists in V4 native**. Probe is harmless but always falls to `q_norm`.
- Lines 365-371 (`kv_a_proj_name`): Probes `kv_proj` OR `kv_a_proj_with_mqa` — **neither exists**. Real name is `wkv`.
- Lines 372-378: Allocates `kv_a_proj` with shape `[hidden, kv_lora_rank + qk_rope_head_dim]` = `[4096, 1024+64=1088]` — **WRONG**. Real shape: `[hidden=4096, head_dim=512]`.
- Lines 382-391: kv layernorm — Arc uses `cfg.kv_lora_rank=1024` for shape. **Wrong**: real shape is `head_dim=512`.
- Lines 392-399: `kv_b_proj` with shape `[kv_lora_rank, n_heads * (qk_nope+v_head_dim)]` — **DELETE entire block, no kv_b in V4**.
- Lines 401-455 (o_proj): wo_a/wo_b allocation. Shape arithmetic is wrong:
  - Arc: `wo_a = [n_heads * v_head_dim, o_groups * o_lora_rank]`. Reality: `wo_a = [n_groups * o_lora_rank, n_heads * head_dim // n_groups] = [8 * 1024, 64*512/8] = [8192, 4096]` (note input/output flipped vs Arc).
  - Arc: `wo_b = [o_groups * o_lora_rank, hidden_size]`. Reality matches: `[8192, 4096]`. ✓
- Lines 477-494: V4Compressor::uniform fallback. The `uniform` initialiser is fine as a placeholder but won't match the real `wkv + wgate + ape + norm` compressor structure.

### Attention::forward (lines 556-812)
- Lines 567-576 (Q split): Arc splits q into `q_nope` (`qk_nope_head_dim`) and `q_pe` (`qk_rope_head_dim`), then RoPEs only `q_pe`. **In V4, RoPE is applied in-place to the last 64 dims of the full 512-dim head vector** — no explicit split tensor. The result is equivalent, but the implementation must NOT split-then-concat (the kernel expects contiguous 512-dim per head).
- Lines 578-587 (KV split): Arc projects `compressed_kv = kv_a_proj(xs)`, splits into `[kv_lora_rank, qk_rope_head_dim]`. **In V4: single `wkv(xs)` produces `[T, 512]` directly**. There's no LoRA-A → norm → LoRA-B pipeline; just one fused linear.
- Line 589: `ckv = self.kv_a_layernorm.forward(&compressed_kv)` — V4 applies RMSNorm to the full 512-dim KV vector before rope.
- Line 591: RoPE call signature — V4 must apply rope to the **last 64 dims of the 512-dim KV vector**, not to a separate `k_pe` tensor.
- Lines 619-625 inverse-rope: **MISSING entirely from Arc**. V4 applies inverse-rope to the last 64 dims of the attention output before passing to wo_a.
- Lines 729-792 (compress-blend logic): The 0.5/0.5 main + SWA blend is **fabricated** and not how V4 sparse attention works. Delete; replace with proper backend dispatch:
  - `compress_ratio=0` → standard MQA
  - `compress_ratio=4` → indexer top-k page selection, then standard MQA on selected pages
  - `compress_ratio=128` → MQA on the (compressed) cache
- Lines 810-812 (o_proj application): Arc does flat `wo_b(wo_a(attn_out))`. Reality requires `attn_out.view(T, G=8, ...)`, per-group `einsum("tgd,grd->tgr")`, then flatten + `wo_b`.

### DecoderLayer (lines 1057-1168)
- Lines 1080-1087 (auto-detect): The probe for `attn.wq_a.weight` is correct **only for V4 native**. V4 HF-converted checkpoints (post-`remap_weight_name_to_dpsk_hf_format`) would use `self_attn.q_a_proj.weight` etc — Arc's detection works but doesn't help if the user supplies an HF-converted V4 checkpoint (which wouldn't exist in the wild but might from a derivative model). OK to leave.
- Line 1109-1132 (Mlp vs Moe dispatch): Arc uses `first_k_dense_replace` for the dispatch. Correct for V4 since all 43 layers should be MoE (`first_k_dense_replace=0`).
- Lines 1142-1167 (forward): **Completely missing mHC pre/post**. Arc does the textbook `residual + sublayer(norm(x))`. V4 does `hc_post(sublayer(norm(hc_pre(x))), residual, post, comb)`. This is a fundamental algorithmic difference.

### DeepSeekV4 model struct + new (lines 1185-1411)
- Lines 1237-1244: Native vs HF probe — V4 Flash is native, so `uses_native=true`. ✓
- Lines 1382-1397: MTP head loading. **Loads only `h_proj` + `e_proj`**, missing:
  - The full transformer decoder layer (`mtp.0.attn.*`, `mtp.0.ffn.experts.*`, `mtp.0.ffn.shared_experts.*`, `mtp.0.ffn.gate.*`, `mtp.0.attn_norm`, `mtp.0.ffn_norm`).
  - `mtp.0.hnorm`, `mtp.0.enorm`, `mtp.0.norm`.
  - `mtp.0.hc_head_*`, `mtp.0.hc_attn_*`, `mtp.0.hc_ffn_*`.
- Lines 1407-1409: mHC fields hardcoded to `None`. **Loads zero mHC weights** even though they're in the checkpoint.

### IsqModel + residual_tensors (lines 1461-1623)
- Line 1480: `kv_a_proj_with_mqa` — DELETE.
- Line 1481: `kv_b_proj` — DELETE.
- Add: `wkv`, `wq_a`, `wq_b` (the actual ISQ-eligible layers).
- Line 1611-1612: `pp("kv_a_proj_with_mqa")` / `pp("kv_b_proj")` save paths — DELETE/replace with `pp("wkv")`.
- The `residual_tensors_moe_experts_only` UnVarBuilder emits HF-style names (e.g. `q_a_proj`). This is fine for ISQ round-trip back to disk but won't match V4 native checkpoint structure.

### ModelConfigMetadata population (lines 1344-1376)
- Line 1353: `k_head_dim: cfg.q_head_dim()` — V4: should be `head_dim=512`, but Arc's `q_head_dim()` returns `qk_rope_head_dim + qk_nope_head_dim` (currently 64+absent=0 since fields don't exist).
- Line 1354-1361: `v_head_dim` ternary — uses `cfg.v_head_dim` which is absent.
- Line 1366-1369: `KvCacheLayout::Mla { kv_lora_rank, kpe_head_dim }` — V4 is NOT MLA in the V3 sense. The cache layout should be MQA-flat: `[T, 1, head_dim=512]` per page.

### Tests (lines 1683-1830)
- The `V4Compressor::uniform` averaging test passes but is irrelevant: V4 compressors are gated learned modules, not averaging-by-ratio.
- Line 1814: `compress_ratios: vec![0, 4, 128, 0]` — test fixture, not a real config. Real V4 Flash starts with `[0, 0, 4, 128, ...]`.

---

## 6. Cargo build feature-flag audit

### `mistralrs-core/Cargo.toml`
- Lines 73, 119-127: `cuda` feature pulls in `candle-core/cuda`, `candle-nn/cuda`, `dep:cudaforge`, `mistralrs-quant/cuda`, `mistralrs-paged-attn/cuda`, `dep:arc-cuda-graph`, `arc-cuda-graph/cuda`.
- The `cuda` feature implies `flash-attn` and `cudnn` only via separate flag composition — `cargo build --features "cuda flash-attn cudnn"` is the canonical "full nvidia" build.

### `mistralrs-quant/Cargo.toml`
- Line 40: `cuda = ["candle-core/cuda", "candle-nn/cuda", "dep:cudaforge"]`
- Lines 58-72: `cuda-1xxxx` features — CUDA version selectors (mostly empty marker features; the build script reads them).
- Lines 75: `cudaforge = { workspace = true, optional = true }`

### `Cargo.lock` confirmation
Workspace pins `candle-core` to `git rev = "c3bb5bf"` (line 42 of root Cargo.toml). This is HuggingFace's main candle repo, not a fork.

### `#[cfg(feature = "cuda")]` blocks in `deepseek4.rs`
Only **one** appears (lines 1362, 1374):
```rust
#[cfg(all(feature = "cuda", target_family = "unix"))]
kv_cache_layout: if matches!(attention_mechanism, AttentionImplementation::PagedAttention) {
    crate::paged_attention::KvCacheLayout::Mla {
        kv_lora_rank: cfg.kv_lora_rank,             // ← REFERENCES MISSING FIELD
        kpe_head_dim: cfg.qk_rope_head_dim,
    }
} else { crate::paged_attention::KvCacheLayout::Standard },
#[cfg(not(all(feature = "cuda", target_family = "unix")))]
kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
```

### `#[cfg(feature = "cuda")]` blocks in `mistralrs-quant/src/{nvfp4,qtip}/mod.rs`
**Zero CUDA-gated blocks** in either file (`grep -rn 'cfg(feature' mistralrs-quant/src/{nvfp4,qtip} → empty`). Both modules are pure CPU implementations at present. NVFP4 (mod.rs:773 LOC) and QTIP (mod.rs:870 LOC) compile on all targets, including macOS.

### Likely break points when `cargo build --features "cuda flash-attn cudnn"` runs
1. **`deepseek4.rs:108-110`**: `kv_lora_rank`, `v_head_dim`, `qk_nope_head_dim` fields are referenced throughout the file. Deserialisation from V4 config.json will fail with `missing field` errors before any CUDA code is hit. Result: cargo compiles fine; runtime crashes at model load.
2. **`deepseek4.rs:1367`**: `KvCacheLayout::Mla { kv_lora_rank: cfg.kv_lora_rank, ... }` — references a field that doesn't exist after the config rewrite. Compile error once fields are removed.
3. **`mla_cache_forward`, `mla_decode_forward`** (lines 602-664) — call sites depend on `kv_lora_rank`, `v_head_dim`, `qk_nope_head_dim` parameters. These functions live in `src/mla/` and are V3-MLA specific. **V4 should not use them at all.**
4. `mistralrs-quant/turboquant` — not gated by `cuda` feature in the file we inspected, but the kernels themselves likely need CUDA primitives. (Outside this audit scope — flag for separate check.)

**No CUDA-only blocks exist in nvfp4 or qtip that would directly break the cuda build.** The bigger compile risk is the missing config fields once we fix the V4 deserialisation.

---

## 7. Multi-GPU + tokenizer status

### Tensor parallelism
Arc's `deepseek4.rs` uses `ColumnParallelLayer` / `RowParallelLayer` / `ReplicatedLayer` in the right places (lines 20, 323, 341, 351, 372, 392, 418, 426, 438, 446, 1253, 1261, 1385). The structure matches SGLang's pattern:
- `wq_a`, `wkv`: ReplicatedLayer (small, no benefit to sharding)
- `wq_b`: ColumnParallelLayer (sharded along output → 64-head dim)
- `wo_a`: ColumnParallelLayer (sharded along output → groups)
- `wo_b`: RowParallelLayer (reduces across TP ranks)

**This part is correct in principle.** The bug is that the layers are wired against the wrong tensor names and dimensions.

### Tokenizer
- V4 ships `tokenizer.json` (6.4 MB, fast tokenizer format) and `tokenizer_config.json`. **Both exist** at `https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/resolve/main/`.
- `tokenizer_config.json` confirms `tokenizer_class: PreTrainedTokenizerFast`, BOS `<｜begin▁of▁sentence｜>`, EOS/PAD `<｜end▁of▁sentence｜>`, vocab implied 129280 (matches `config.vocab_size`).
- Arc's `mistralrs-core/src/pipeline/normal.rs:108,128,152` plumbs `tokenizer_json: Option<String>` and the `tokenizers` crate handles `PreTrainedTokenizerFast`-format JSON. **No tokenizer-side changes needed.**
- **No `chat_template.jinja` at the standard URL** (returns 404). V4 Flash's chat template either lives elsewhere (a follow-up `chat_template` field added to `tokenizer_config.json` in a later commit?), is bundled in tokenizer.json itself (less common for HF fast tokenizers), or must be supplied separately. **Action: copy DeepSeek V3's chat template** as the default; user can override via Arc's `chat_templates/` dir.

---

## 8. Recommended fixes — priority-ordered

### P0 (blockers for any V4 weight to load)
1. **Rewrite `DeepSeekV4Config`** to match real V4 fields. Remove `kv_lora_rank`, `v_head_dim`, `qk_nope_head_dim`, `intermediate_size`. Add `head_dim`, `num_key_value_heads`, `num_hash_layers`, `hc_mult`, `hc_eps`, `hc_sinkhorn_iters`, `swiglu_limit`. Honor `routed_scaling_factor=1.5`. Add `ScoringFunc::SqrtSoftplus`.
2. **Rewrite `Attention` struct + new()**: single `wkv` (Replicated 4096→512), single `kv_norm` (RmsNorm 512), delete `kv_b_proj`, add `attn_sink` parameter, add `q_norm` at `q_lora_rank=1024`.
3. **Add `Compressor` module** (mirror `dsv4/compressor.py:289-393`): gated linear (`wkv_gate` fused from `wkv + wgate`), `RMSNorm`, `ape` parameter, `apply_ape_hotfix` logic.
4. **Add `C4Indexer` module**: `wq_b` (1024 → 64*128=8192), `weights_proj` (4096 → 64), nested `Compressor` (with `head_dim=128`, `rotate=True`).
5. **Add fp8 dequant path** for `.weight` + `.scale` tensors (UE8M0 128×128 blocks). This is required for any V4 weight to be usable; without it, Arc loads zeros (or NaNs) for every fp8 tensor.
6. **Add mHC pre/post** to `DecoderLayer::forward`. Load `hc_attn_*` and `hc_ffn_*` per layer + `hc_head_*` at the top level. Implement Sinkhorn-split + soft mixing (~30 lines of Rust; the math is in `deepseek_v4.py:774-908`).
7. **Add `attn_sink` to the attention kernel call**. The softmax must include `attn_sink` as an extra column (key=attn_sink_value, value=0). For Sdpa fallback: add a "sink_logits" parameter and concat to the attention logits before softmax.
8. **Rewrite forward attention math**: drop the q_nope/q_pe split (apply RoPE in-place to last 64 dims), drop the V3 MLA cache forward (`mla_cache_forward`, `mla_decode_forward`), use a flat MQA pattern (q: `[T, 64, 512]`, kv: `[T, 1, 512]`, broadcast in kernel).
9. **Replace `compress_kv` fake blend** with real backend dispatch. For Tier-A, an Sdpa fallback over the full sequence (compress_ratio=0 behavior) gives a correct-but-slow baseline; the indexer + compressor paths can land in subsequent agents.
10. **Rewrite MTP loader** to load the full transformer layer at `mtp.0.*` + h_proj/e_proj/enorm/hnorm/norm/hc_*. Build a separate `MtpDecoder` type that takes (prev_hidden_3D, cur_input_ids) and returns logits.

### P1 (correctness improvements)
11. **Two-mode RoPE**: instantiate two `DeepSeekV2RotaryEmbedding` (one per `rope_theta`) and route per-layer based on `compress_ratio`.
12. **Grouped o_proj einsum**: implement the per-group `einsum("tgd,grd->tgr")` rather than a flat `wo_a.forward`. The current flat approach gives the wrong output without the group dimension.
13. **Inverse RoPE on attention output's last 64 dims** before wo_a (per `deepseek_v4.py:619-625`). Currently missing.
14. **`SqrtSoftplus` scoring function** for the MoE gate (V4-specific). Implement: `score = sqrt(softplus(logit))`.
15. **`swiglu_limit=10.0` clamp** inside MoE expert SiLU forward (config has it; experts need it).
16. **Compressor wkv_gate fusion at load time** (mirror SGLang's `compressor.py:340` `apply_ape_hotfix` and `deepseek_v4.py:1632-1663` cache_compressor_weight logic).

### P2 (perf / nice-to-haves; can land after first end-to-end V4 generation)
17. **Vendor `fused_norm_rope.cuh`** as a custom kernel in `mistralrs-quant/cuda_kernels/`. Drop-in replacement for the post-projection norm + rope.
18. **Vendor `c4.cuh` + `topk_v2.cuh`** for CSA fast path.
19. **Vendor `c128_online.cuh`** for HCA fast path.
20. **Vendor `store.cuh`** to fuse cache write into the K kernel (eliminates the bf16 KV intermediate).
21. **Add Hash topk** (`hash_topk.cuh`) for MoE expert dispatch — `num_hash_layers=3` per V4 config (the `ffn.gate.tid2eid` tensor is the hash lookup table).
22. **Replicate `_setup_fp8_wo_a_scales`** at post-load (per `deepseek_v4.py:1321-1346`): reshape UE8M0 scales into the layout deep_gemm expects.
23. **Implement `fuse_wqa_wkv`** option (per `SGLANG_OPT_FUSE_WQA_WKV`): concat wq_a + wkv along output channel at load time → single `wqkv_a` matmul at forward time.

### P3 (catch-all)
24. **MTP-driven speculative decoding pipeline wiring**: the SpeculativePipeline must accept the MTP head's `forward(prev_hidden, cur_token_ids)` and verify drafts. Currently no glue at all (Arc loads h_proj/e_proj but never calls them).
25. **Chat template**: bundle V3's chat template under `chat_templates/deepseek-v4.jinja` since V4 ships without one.
26. **Tokenizer hookup**: confirm `tokenizer.json` end-to-end with V4's BOS/EOS markers. Currently expected to "just work" but should be smoke-tested.

---

## 9. Open questions / things to verify on a real GPU rental

- **Exact dtype of `attn_sink`**: index says fp32; SGLang code (`deepseek_v4.py:288`) confirms `dtype=torch.float32`. ✓
- **`fused_q_norm_rope` semantics**: norm-then-rope, or rope-then-norm? SGLang's signature is `fused_q_norm_rope(q, q_out, eps, freqs_cis, positions)` — implies q is RMSNormed with implicit unit weight (no `q_norm.weight` applied here; that happens earlier on q_lora). Verify on rental.
- **`hc_pre` TileLang vs torch reference paths**: the torch fallback (`deepseek_v4.py:786-870`) is the canonical math; the TileLang path is a fused-kernel optimisation. Port the torch path first.
- **`ffn.gate.tid2eid`**: int32 tensor; per `deepseek_v4.py:1415` it gets remapped to `topk.tid2eid`. Used by hash-topk routing for MoE. Verify shape: probably `[num_hash_layers=3, n_routed_experts=256]` or similar.
- **MTP `compress_ratio_override = 0`**: confirmed in `deepseek_v4_nextn.py:47, 105` (the MTP decoder layer is forced to standard attention regardless of any per-layer setting). ✓

---

## 10. Files referenced in this audit (absolute paths)

**Arc**:
- `/Users/jish/Documents/GitHub/arc/Cargo.toml`
- `/Users/jish/Documents/GitHub/arc/CLAUDE.md`
- `/Users/jish/Documents/GitHub/arc/mistralrs-core/Cargo.toml`
- `/Users/jish/Documents/GitHub/arc/mistralrs-core/src/models/deepseek4.rs` (1830 LOC)
- `/Users/jish/Documents/GitHub/arc/mistralrs-core/src/pipeline/loaders/normal_loaders.rs` (V4 registration at lines 170-176, 222, 262, 294, 358, 3130-3300)
- `/Users/jish/Documents/GitHub/arc/mistralrs-core/src/pipeline/chat_template.rs`
- `/Users/jish/Documents/GitHub/arc/mistralrs-core/src/pipeline/normal.rs:108-160` (tokenizer plumbing)
- `/Users/jish/Documents/GitHub/arc/mistralrs-quant/Cargo.toml`
- `/Users/jish/Documents/GitHub/arc/mistralrs-quant/src/nvfp4/mod.rs` (773 LOC, no CUDA gates)
- `/Users/jish/Documents/GitHub/arc/mistralrs-quant/src/qtip/mod.rs` (870 LOC, no CUDA gates)
- `/Users/jish/Documents/GitHub/arc/mistralrs-quant/src/qtip/viterbi.rs`

**SGLang reference**:
- `/Users/jish/Documents/GitHub/arc/research/code/06_foundation/sglang/python/sglang/srt/models/deepseek_v4.py` (1816 LOC)
- `/Users/jish/Documents/GitHub/arc/research/code/06_foundation/sglang/python/sglang/srt/models/deepseek_v4_nextn.py` (280 LOC)
- `/Users/jish/Documents/GitHub/arc/research/code/06_foundation/sglang/python/sglang/srt/layers/attention/dsv4/compressor.py` (399 LOC)
- `/Users/jish/Documents/GitHub/arc/research/code/06_foundation/sglang/python/sglang/srt/layers/attention/dsv4/indexer.py` (584 LOC)
- `/Users/jish/Documents/GitHub/arc/research/code/06_foundation/sglang/python/sglang/srt/layers/attention/dsv4/{compress_hip,compressor_v2,index_buf_accessor,metadata,metadata_kernel,quant_k_cache,tilelang_kernel}.py`
- `/Users/jish/Documents/GitHub/arc/research/code/06_foundation/sglang/python/sglang/jit_kernel/dsv4/{__init__,attn,compress,compress_old,elementwise,gemm,hisparse,moe,topk,utils}.py`
- `/Users/jish/Documents/GitHub/arc/research/code/06_foundation/sglang/python/sglang/jit_kernel/csrc/deepseek_v4/*.cuh` (20 files)
- `/Users/jish/Documents/GitHub/arc/research/code/06_foundation/sglang/sgl-kernel/csrc/attention/{cutlass_mla_kernel,merge_attn_states,vertical_slash_index}.cu`
- `/Users/jish/Documents/GitHub/arc/research/code/06_foundation/sglang/sgl-kernel/csrc/attention/cutlass_sm100_mla/{device,kernel}/`

**HuggingFace**:
- `/tmp/v4_flash_index.json` (5.1 MB safetensors index — keep for downstream agents)
- `/tmp/v4_flash_config.json` (V4 Flash config.json)
- `/tmp/v4_tok.json` (tokenizer_config.json)

---

## 11. Summary for downstream agents

**Agent 2 (model rewrite)**: start by deleting `deepseek4.rs:Attention` body + replacing with the SGLang MQALayer structure. Add `Compressor`, `C4Indexer`, `MhcGate` types. Don't touch `MoE` (V3 dispatch is reusable). The fp8 dequant pass is the second-most urgent item.

**Agent 3 (kernel vendoring)**: vendor `fused_norm_rope.cuh` first (smallest, most isolated, no template cascade). Test on B200 with a single-layer forward pass. Then c4 + topk for CSA. HCA can wait.

**Agent 4 (perf bench)**: do NOT run perf bench until end-to-end correctness is proven on a tiny (~100 token) prompt. Currently zero V4 weights can be loaded — let alone forwarded. Tier-A target: any token at all from V4 Flash. Tier-B: matches HF transformers output within fp8 quantisation noise. Perf is Tier-C.

**Agent 5 (CI / regression)**: add a smoke test that downloads only the V4 safetensors index + config.json, asserts the rename rules survive, and verifies Arc's config deserialiser accepts the real V4 config.json without panic. This catches future regressions without renting a B200.
