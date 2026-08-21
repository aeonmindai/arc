# Wave 13 · Agent AH — Ground Arc's V4 port in the real `config.json`

**Date**: 2026-08-14 · **Branch**: `fix/v4-config-grounded` off `origin/master` @ `bef703326`
**Predecessor**: PR #27 / `docs/v4-reference-audit` (agent AA), log `memory/mission/wave11-AA-audit.md`

## Mission

The V4 reference audit produced 25 findings but stalled on one missing artifact: the
real `config.json` from `deepseek-ai/DeepSeek-V4-Flash`. Five findings bottomed out
there. This task fetched it, settled those five, corrected Arc's demonstrably-fake
"real config" fixture, and fixed what the config proves wrong.

## 1. The fetched artifacts

All three fetches succeeded. The repo is public and MIT-licensed; no token needed.

| Artifact | URL | Result |
|---|---|---|
| `config.json` | `https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/resolve/main/config.json` | HTTP 200, 1749 bytes |
| `model.safetensors.index.json` | same host, `/resolve/main/model.safetensors.index.json` | HTTP 200, 5 371 381 bytes, 69 187 tensors, `total_size` 159 609 485 896 |
| shard-2 safetensors header | ranged GET on `model-00002-of-00046.safetensors` | HTTP 206, 172 232-byte JSON header |

### 1.1 `config.json` — VERBATIM

```json
{
  "architectures": [
    "DeepseekV4ForCausalLM"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 0,
  "eos_token_id": 1,
  "expert_dtype": "fp4",
  "hc_eps": 1e-06,
  "hc_mult": 4,
  "hc_sinkhorn_iters": 20,
  "head_dim": 512,
  "hidden_act": "silu",
  "hidden_size": 4096,
  "index_head_dim": 128,
  "index_n_heads": 64,
  "index_topk": 512,
  "initializer_range": 0.02,
  "max_position_embeddings": 1048576,
  "model_type": "deepseek_v4",
  "moe_intermediate_size": 2048,
  "n_routed_experts": 256,
  "n_shared_experts": 1,
  "norm_topk_prob": true,
  "num_attention_heads": 64,
  "num_experts_per_tok": 6,
  "num_hidden_layers": 43,
  "num_hash_layers": 3,
  "num_key_value_heads": 1,
  "num_nextn_predict_layers": 1,
  "o_groups": 8,
  "o_lora_rank": 1024,
  "q_lora_rank": 1024,
  "qk_rope_head_dim": 64,
  "quantization_config": {
    "activation_scheme": "dynamic",
    "fmt": "e4m3",
    "quant_method": "fp8",
    "scale_fmt": "ue8m0",
    "weight_block_size": [
      128,
      128
    ]
  },
  "rms_norm_eps": 1e-06,
  "rope_scaling": {
    "beta_fast": 32,
    "beta_slow": 1,
    "factor": 16,
    "original_max_position_embeddings": 65536,
    "type": "yarn"
  },
  "rope_theta": 10000,
  "routed_scaling_factor": 1.5,
  "scoring_func": "sqrtsoftplus",
  "sliding_window": 128,
  "swiglu_limit": 10.0,
  "tie_word_embeddings": false,
  "topk_method": "noaux_tc",
  "torch_dtype": "bfloat16",
  "transformers_version": "4.57.1",
  "use_cache": true,
  "vocab_size": 129280,
  "compress_rope_theta": 160000,
  "compress_ratios": [0, 0, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 0]
}
```

### 1.2 Structure derived from `model.safetensors.index.json`

Tensor-name families (`N` = any integer), with counts:

```
1      embed.weight                      1      head.weight        1  norm.weight
1      hc_head_{base,fn,scale}
43     layers.N.attn.attn_sink           43     layers.N.attn.{q_norm,kv_norm}.weight
43     layers.N.attn.{wq_a,wq_b,wkv,wo_a,wo_b}.{weight,scale}
41     layers.N.attn.compressor.{ape,norm.weight,wgate.weight,wkv.weight}
21     layers.N.attn.indexer.compressor.{ape,norm.weight,wgate.weight,wkv.weight}
21     layers.N.attn.indexer.{weights_proj.weight,wq_b.weight,wq_b.scale}
11008  layers.N.ffn.experts.N.{w1,w2,w3}.{weight,scale}      (= 43 × 256)
43     layers.N.ffn.gate.weight     40  layers.N.ffn.gate.bias    3  layers.N.ffn.gate.tid2eid
43     layers.N.ffn.shared_experts.{w1,w2,w3}.{weight,scale}
43     layers.N.{attn_norm,ffn_norm}.weight
43     layers.N.hc_{attn,ffn}_{base,fn,scale}
1      mtp.0.* (attn, ffn w/ 256 experts, e_proj, h_proj, enorm, hnorm, norm, hc_*, hc_head_*)
```

Cross-checks that this settles:

* **`compress_ratios` semantics confirmed.** `Counter(compress_ratios) == {4: 21, 128: 20, 0: 3}`.
  Indices of `0` are exactly `[0, 1, 43]`; of `4` exactly `[2, 4, …, 42]`; of `128` exactly
  `[3, 5, …, 41]`. The 41 layers carrying `attn.compressor.*` are exactly the 41 layers
  with a non-zero ratio (all but 0 and 1). Slot 43 is the MTP block.
* **The Lightning Indexer lives on CSA (ratio-4) layers only.** The 21 layers with
  `attn.indexer.*` are exactly `[2, 4, …, 42]` — the ratio-4 set. Ratio-128 (HCA) layers
  have no indexer.
* **Hash layers are exactly layers 0–2.** `gate.tid2eid` exists on layers `{0, 1, 2}` and
  those are precisely the three layers with **no** `ffn.gate.bias` — hash-routed layers
  carry no noaux_tc correction bias, which is internally consistent with `num_hash_layers: 3`.
* **The MTP block is NOT hash-routed.** `mtp.0.ffn.gate.bias` exists and no
  `mtp.*.tid2eid` exists anywhere. This independently confirms the reference's
  `is_hash = layer_id < n_hash_layers and not (is_deepseek_v4 and is_nextn)` exclusion.
* **Every one of the 43 layers is MoE.** 11008 = 43 × 256 expert tensors, and no dense
  `mlp.*` family exists — consistent with `first_k_dense_replace` being absent (Arc
  defaults it to 0).

### 1.3 Shapes and dtypes (shard-2 header, layer 0)

```
layers.0.attn.attn_sink        F32       [64]              = num_attention_heads
layers.0.attn.q_norm.weight    BF16      [1024]            = q_lora_rank
layers.0.attn.kv_norm.weight   BF16      [512]             = head_dim
layers.0.attn.wq_a.weight      F8_E4M3   [1024, 4096]      scale F8_E8M0 [8, 32]
layers.0.attn.wq_b.weight      F8_E4M3   [32768, 1024]     = 64 heads × 512
layers.0.attn.wkv.weight       F8_E4M3   [512, 4096]       = 1 KV head × 512
layers.0.attn.wo_a.weight      F8_E4M3   [8192, 4096]      = o_groups(8) × o_lora_rank(1024)
layers.0.attn.wo_b.weight      F8_E4M3   [4096, 8192]
layers.0.ffn.gate.weight       BF16      [256, 4096]
layers.0.ffn.gate.tid2eid      I64       [129280, 6]       = [vocab_size, num_experts_per_tok]
layers.0.hc_attn_scale         F32       [3]
layers.0.hc_attn_base          F32       [24]              = m² + 2m for m = hc_mult = 4
layers.0.hc_attn_fn            F32       [24, 16384]       = [24, hc_mult × hidden_size]
layers.0.ffn.experts.0.w1      I8        [2048, 2048]      scale F8_E8M0 [2048, 128]
layers.0.ffn.shared_experts.w1 F8_E4M3   [2048, 4096]      scale F8_E8M0 [16, 32]
```

Every config-derived dimension Arc computes is confirmed: `head_dim = 512`,
`q_lora_rank = 1024`, `o_groups × o_lora_rank = 8192`, `num_key_value_heads = 1`,
`hc_mult = 4` (encoded in the `[24, …]` mHC tensor widths, so it is not a free knob).

## 2. The five findings, settled

### 2.1 Finding 11 — group-limited routing → **SETTLED, not a defect**

`grep -o 'n_group\|topk_group'` on the real config returns **nothing**. Neither key is
published. Arc's serde defaults (`n_group = 1`, `topk_group = 1`) therefore apply, the
`if self.cfg.n_group > 1` guard is false, and `MoeGate::forward` takes the flat top-k
branch — the same end state SGLang reaches by force-disabling group limiting for V4.

Arc's original premise ("V4 Flash config does NOT publish `n_group`") was **correct**.
The audit's counter-inference from SGLang's load-time patch comment ("HF config.json
inherits `topk_group=4` from the V3 template") does not hold for the shipped file — that
patch is defensive, not descriptive. Severity drops from HIGH-conditional to **not
applicable to this checkpoint**.

The two secondary defects the audit found *inside* that branch (masking by multiply
instead of `-inf`; group score always `topk(2).sum` instead of `amax` when unbiased) are
**still real**, but are now confirmed dead code for V4 Flash. They remain live for V3-style
configs via `deepseek3.rs:534`, which is where they should be fixed.

### 2.2 Finding 12 — hash-layer key spelling → **SETTLED, not a defect**

The shipped key is **`num_hash_layers`** (value `3`). `n_hash_layers` does **not** appear.

That is precisely the key SGLang's *runtime* reads — `getattr(config, "num_hash_layers", 0)`
(`deepseek_v2.py:483`). So the reference runs three hash layers, exactly as Arc does. The
feared "reference runs 0 hash layers where Arc runs 3" divergence **does not occur**.
SGLang's `n_hash_layers` dataclass field (`configs/deepseek_v4.py:107`) is simply an unused
declaration. Arc's `alias = "n_hash_layers"` is harmless belt-and-braces.

Independently corroborated by the checkpoint: `gate.tid2eid` exists on exactly layers 0, 1, 2.

**Rider (ii) also settled, and in Arc's favour.** The audit worried that the reference
stores `tid2eid` as `int32` while Arc requests `DType::I64` (`deepseek4.rs:1554-1559`).
The shipped tensor is **`I64 [129280, 6]`**. Arc's request is correct; an I32 read would
have been the bug. Shape matches Arc's `[vocab_size, num_experts_per_tok]` implementation
and confirms the audit's own correction of the stale `research/v4_audit.md` guess.

**Rider (i) stands as written**: the reference excludes the MTP block from hash routing
unconditionally; Arc keys purely on virtual layer index. For the real 43-layer config these
coincide (MTP sits at virtual index 43 ≥ 3), and the checkpoint confirms the MTP block is
biased-routed (`mtp.0.ffn.gate.bias` present, no `mtp.*.tid2eid`). It remains a hazard only
for small synthetic fixtures — `synthetic_load_smoke.rs:340-344` depends on the current
behavior.

### 2.3 Finding 24 — `rope_scaling` / mscale → **SETTLED, not a defect**

`rope_scaling` **is** present — the audit was right that Arc's old fixture (which omitted
it entirely) could not have been the real config:

```json
"rope_scaling": {
  "beta_fast": 32, "beta_slow": 1, "factor": 16,
  "original_max_position_embeddings": 65536, "type": "yarn"
}
```

It carries **no `mscale` and no `mscale_all_dim`**. Arc's `DeepSeekV2RopeScaling::Yarn`
already defaults both to `1.0` (`layers.rs:1440-1445`), so the magnitude factor Arc
multiplies into sin/cos is `yarn_get_mscale(16, 1.0) / yarn_get_mscale(16, 1.0) == 1.0`.
Arc's mscale multiply is a **no-op**, exactly matching the reference's unit-magnitude
`torch.polar(torch.ones_like(freqs), freqs)`. Finding 24 resolves to **no divergence**;
the existing code comment at `layers.rs:1432-1439` was already right.

### 2.4 Finding 5 rider (i) — `swiglu_limit` → **PUBLISHED; finding 5 CONFIRMED and hardened**

`"swiglu_limit": 10.0` is present. The reference's
`getattr(config, "swiglu_limit", None)` therefore resolves to `10.0` and the reference
**does** clamp. Arc's hard default of `10.0` coincides with the shipped value.

This removes the escape hatch from finding 5: the clamp is **mandatory**, not a
config-dependent maybe. Every Arc expert path that drops it is a genuine numerical
divergence against the trained weights — most seriously the **shared expert, on every
device and backend** (`deepseek4.rs` → `Mlp::new`, `layers.rs:3001-3037` has no limit
field). Finding 5 stays **HIGH** and is now unconditional.

### 2.5 `compress_rope_theta` → **SETTLED; Arc's default is right**

`"compress_rope_theta": 160000`. Arc's `default_compress_rope_theta() -> 160000.0`
matches the shipped value exactly. SGLang's dataclass default of `40000`
(`configs/deepseek_v4.py:104`) is never reached for this checkpoint, so the 4×
discrepancy the audit flagged is a dead default, not a divergence. `rope_theta` is
`10000`, and the per-layer θ dispatch Arc implements is correct.

## 3. What changed in the tree

All edits are in `mistralrs-core/src/models/deepseek4.rs`. Scope fences respected: no
touch to `mistralrs-quant/**`, `mtp_pipeline.rs`, the MTP block, `dsv4_attention.rs`,
`sinks.rs`, or `dsv4_indexer.rs`.

1. **Fixture replaced with the verbatim published config.** The old fixture was
   hand-written and omitted `rope_scaling`, `norm_topk_prob`, `quantization_config`,
   `hidden_act`, `attention_bias`, `tie_word_embeddings`, `expert_dtype` and
   `num_nextn_predict_layers`, while *adding* a `first_k_dense_replace` key that the real
   config does not have. It is now a module-level `const V4_FLASH_CONFIG_JSON` reproduced
   byte-for-byte from the fetch, with the source URL and fetch date in its doc comment.

2. **`v4_flash_real_config_parses` extended.** Proves the real file parses through
   `DeepSeekV4Config` including the fp8 `quantization_config` block, and pins the
   absent-key set (`intermediate_size`, `kv_lora_rank`, `v_head_dim`, `qk_nope_head_dim`)
   plus `first_k_dense_replace == 0`. Added `layer_compress_ratio(42) == 4` and an
   assertion that the ratio-0 set is exactly `{0, 1, 43}`.

3. **New test `v4_flash_config_settles_the_audit_questions`.** One assertion block per
   audit finding (11, 12, 24, 5-rider, `compress_rope_theta`, 21, 14-context), each
   commented with the finding it closes, so a future re-fetch that contradicts one fails
   loudly rather than silently.

4. **`norm_topk_prob` now parsed** (audit finding 21), `#[serde(default = "…")] = true`,
   and used to gate renormalization in `MoeGate::forward`:
   `if self.cfg.norm_topk_prob && matches!(scoring_func, Sigmoid | SqrtSoftplus)`.
   Behavior-neutral: the real config publishes `true`, and no fixture in the tree sets
   `false` (`grep '"norm_topk_prob": false'` → no hits). The reference passes the flag
   straight through as `renormalize=config.norm_topk_prob` (`deepseek_v2.py:553`).

5. **Comments upgraded from claim to verified fact.** The `MoeGate` `n_group` comment
   cited `/tmp/v4_flash_config.json` — a path that no longer exists and that CI could
   never check; it now cites the resolve URL and names the test that pins it. The struct
   doc comment's "V4: not in real config" claims are now labelled verified and enumerate
   the absent `mscale`/`mscale_all_dim` too.

### Verification

```
cargo check -p mistralrs-core --tests        → clean (8 pre-existing warnings only)
cargo test  -p mistralrs-core --lib deepseek4 → 22 passed; 0 failed
cargo test  -p mistralrs-core                 → see PR checks
```

No fixture-dependent test needed correcting — none of them were pinning wrong behavior.
The old fixture was *incomplete* rather than *wrong*: every assertion it made was true of
the real config; it simply could not answer the questions the audit needed answered.

## 4. Findings status after this task

**Settled — closed, no code change needed (4)**

| # | Was | Now |
|---|---|---|
| 11 | Group-limited routing — HIGH, conditional | Neither `n_group` nor `topk_group` ships. Branch is dead for V4 Flash. Arc's premise was right. |
| 12 | Hash key mismatch — HIGH, UNVERIFIABLE | Ships as `num_hash_layers: 3`, the key SGLang's runtime reads. No divergence. Rider (ii) also clears: tensor is I64, as Arc requests. |
| 24 | mscale — UNVERIFIABLE | No `mscale`/`mscale_all_dim` in `rope_scaling` ⇒ Arc's multiply is exactly 1.0 ⇒ matches the reference's unit-magnitude polar. |
| — | `compress_rope_theta` default | Ships as 160000 = Arc's default. SGLang's 40000 dataclass default is unreachable. |

**Confirmed real and hardened by the config (3)**

| # | Finding | What the config adds |
|---|---|---|
| 5 | swiglu clamp missing on 4 of 5 expert paths | `swiglu_limit: 10.0` IS published ⇒ the reference clamps ⇒ the omission is unconditional, not config-dependent. Severity HIGH stands. |
| 6 | YaRN applied to ratio-0 layers by default | `rope_scaling` present with `factor: 16`, `original_max_position_embeddings: 65536` ⇒ the reference's `original_seq_len = 0` override for ratio-0 layers is live and Arc's default diverges. **Correction: the affected layers are 0, 1 and the MTP block — not "0, 1, 42". Layer 42 has ratio 4.** |
| 14 | FP8 K-cache scale continuous, not UE8M0 | `quantization_config.scale_fmt: "ue8m0"`, and every shipped `*.scale` tensor has dtype **`F8_E8M0`**. Direct evidence. |

**Fixed here (2)**

| # | Finding | Fix |
|---|---|---|
| 21 | `norm_topk_prob` not parsed | Field added, defaulted `true`, gates renormalization. Behavior-neutral today. |
| — | Fixture is not the real config | Replaced verbatim; two tests now pin the answers. |

**Open, unchanged by the config (16)**: 1, 2, 3, 4, 7, 8, 9, 10, 13, 15, 16, 17, 18, 19,
20, 22, 23, 25. Finding 20 (`hc_mult`/`hc_eps`/`hc_sinkhorn_iters` dead) is confirmed
*latent only*: the shipped values (4 / 1e-6 / 20) equal Arc's hardcoded constants, and
`hc_mult` is additionally pinned by the `[24, 16384]` mHC tensor widths, so a mismatched
config would fail a shape check rather than silently compute wrong math.

## 5. Left deliberately — outside the fence

Written up precisely rather than reached for.

**(a) Finding 5, the shared-expert clamp.** `deepseek4.rs:1832-1839` builds the shared
expert via `Mlp::new` (`layers.rs:3001-3037`), which has no `limit` field, so the clamp is
dropped on every device. Also dropped by `experts.rs:465` (`forward_fused`, the default
un-ISQ'd CUDA path), `experts.rs:570` (Metal), `experts.rs:632-637` (`forward_slow`).
Only `forward_fast`'s CUDA branch (`experts.rs:541-549`) clamps.
*Proposed test*: build a shared-expert `Mlp` with `swiglu_limit = 1.0`, feed a gate
pre-activation of `+50`, and assert the output equals the `limit`-clamped reference rather
than the unclamped one; repeat per expert path with `ARC_MOE_SLOW` set.
Behavior change in the expert MLP, not config — belongs to whoever owns `moe/experts.rs`.

**(b) Finding 6, the YaRN default flip.** `deepseek4.rs` gates the reference behavior
behind `ARC_DISABLE_YARN_STD`. The config now proves the reference path is the correct
default. The flip is a one-line change but is an attention-path behavior change; PR #28
owns `dsv4_attention.rs` and this interacts with it.
*Proposed test*: assert the ratio-0 rotary table's `inv_freq` equals plain
`1/θ^(2i/d)` (no interpolation) while the ratio-4 table shows the YaRN ramp.

**(c) NEW — the single `quantization_config` block does not describe the routed experts.**
The config publishes one fp8 block (`quant_method: "fp8"`, `weight_block_size: [128, 128]`),
which matches the attention and shared-expert tensors exactly (`wkv.weight` F8_E4M3
`[512, 4096]` with scale `[4, 32]` = ceil(512/128) × ceil(4096/128)). It does **not**
describe the routed experts, which ship as packed FP4 — `w1.weight` dtype **`I8`**
`[2048, 2048]` (two nibbles per byte over a logical `[2048, 4096]`) with scale
`F8_E8M0 [2048, 128]`, i.e. **per-row group-32**, not 128×128 blocks. The distinguishing
key is `"expert_dtype": "fp4"`, which Arc does not parse. Anything that loads the native
checkpoint and trusts `quantization_config` uniformly will mis-read 11008 of the 69187
tensors. Owner: native-checkpoint loading / `mistralrs-quant` (fenced from this task).

**(d) NEW — `num_nextn_predict_layers: 1` is published and Arc does not parse it.** Arc
infers MTP presence from tensor availability instead. Not wrong today, but the config
carries the authoritative count. Owner: PR #30 (MTP).

## 6. Proposed audit-doc wording

PR #27 is still **open** at the time of writing (`state: OPEN`, `mergedAt: null`), so the
doc was not edited directly. Exact replacements below.

**Replace the "The single missing artifact" block-quote (`docs/notes/v4-reference-audit.md`,
after the "Reference availability caveat" quote) with:**

> **The missing artifact — now fetched.** Five findings bottomed out in the real
> `config.json` from `deepseek-ai/DeepSeek-V4-Flash`. It was retrieved on 2026-08-14 from
> `https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/resolve/main/config.json` (public,
> MIT, no token) and is now vendored verbatim as `V4_FLASH_CONFIG_JSON` in
> `mistralrs-core/src/models/deepseek4.rs`, with `v4_flash_config_settles_the_audit_questions`
> pinning each answer. **Findings 11, 12 and 24 are closed as non-defects**; the
> `compress_rope_theta` rider is closed in Arc's favour; and **finding 5 is hardened** —
> `swiglu_limit: 10.0` is published, so the reference clamps unconditionally and Arc's four
> unclamped expert paths are a real divergence with no config-dependent escape. See
> `memory/mission/wave13-AH-config.md`.

**Verdict table (§0) — amend these rows:**

* Row 5: severity unchanged, append to the finding text: `— config publishes swiglu_limit: 10.0, so the clamp is unconditional`.
* Row 11: change Severity to `CLOSED` and the finding text to
  `Group-limited routing force-disabled upstream — real config publishes neither n_group nor topk_group, so Arc's branch is dead code for V4 Flash. No divergence.`
* Row 12: change Severity to `CLOSED`, Confidence to `both-lbl`, and the text to
  `Real config spells it num_hash_layers: 3 — the key SGLang's runtime reads. Reference and Arc both run 3 hash layers. tid2eid also ships as I64, matching Arc's request.`
* Row 24: change Severity to `CLOSED`, and the text to
  `rope_scaling publishes no mscale/mscale_all_dim ⇒ Arc's defaults make the multiply exactly 1.0 ⇒ matches the reference's unit-magnitude polar.`
* Row 14: append to Confidence cell: `— corroborated: config sets scale_fmt: "ue8m0" and every shipped scale tensor is dtype F8_E8M0`.
* Row 21: change Severity to `FIXED` and append `— norm_topk_prob (published as true) is now parsed and gates renormalization`.

**§1(b), the YaRN paragraph — correct the affected-layer list.** Replace

> exactly the layers Arc's own module docs (`dsv4_attention.rs:12-28`) blame for the
> RUN-161 long-context repetition collapse

's preceding clause `on layers 0, 1, 42 and the MTP block` with `on layers 0 and 1 and
the MTP block`. `compress_ratios` index 42 is `4` (CSA), not `0`; the ratio-0 indices are
exactly `[0, 1, 43]`, and index 43 *is* the MTP slot.

**§1(b), the mscale paragraph — replace `**mscale — UNVERIFIABLE.**` … `Settled by the
missing config.json.` with:**

> **mscale — SETTLED, MATCHES.** Reference returns
> `torch.polar(torch.ones_like(freqs), freqs)` (`deepseek_v4_rope.py:57`) — unit magnitude.
> The published `rope_scaling` carries no `mscale` and no `mscale_all_dim`, so Arc's serde
> defaults (both `1.0`, `layers.rs:1440-1445`) apply and
> `yarn_get_mscale(16, 1.0) / yarn_get_mscale(16, 1.0) == 1.0`. Arc's multiply is a no-op.
> No divergence.

**§1(b), the `compress_rope_theta` paragraph — replace with:**

> **`compress_rope_theta` — SETTLED.** The published config sets `160000`, matching Arc's
> default (`deepseek4.rs`). SGLang's dataclass default of `40000`
> (`configs/deepseek_v4.py:104`) is never reached for this checkpoint. No action.

**§4(b2) — prepend to the section body:**

> **SETTLED (2026-08-14): this branch is dead for V4 Flash.** The published `config.json`
> contains neither `n_group` nor `topk_group`, so Arc's serde defaults (1/1) apply and the
> flat top-k branch is taken — the same end state SGLang reaches by force-disabling group
> limiting. Arc's premise was correct and SGLang's load-time patch comment is defensive,
> not descriptive. The two secondary defects below remain real but are reachable only via
> V3-style configs (`deepseek3.rs:534`), which is where to fix them.

**§4(d′) — replace the whole section with:**

> ### (d′) `num_hash_layers` vs `n_hash_layers` — **SETTLED: MATCHES** · `both-lbl`
>
> The published config spells it **`num_hash_layers: 3`**; `n_hash_layers` does not
> appear. That is the key SGLang's runtime reads
> (`getattr(config, "num_hash_layers", 0)`, `deepseek_v2.py:483`), so the reference runs
> three hash layers exactly as Arc does. SGLang's `n_hash_layers` dataclass field
> (`configs/deepseek_v4.py:107`) is an unused declaration; Arc's `alias` is harmless.
> Corroborated by `model.safetensors.index.json`: `gate.tid2eid` exists on exactly layers
> 0, 1, 2, which are also the only three layers **without** an `ffn.gate.bias`.
>
> **Rider (ii) settled in Arc's favour**: the shipped tensor is `I64 [129280, 6]`, which is
> what Arc requests (`deepseek4.rs:1554-1559`). An I32 read would have been the bug.
>
> **Rider (i) stands**: the reference excludes the MTP block from hash routing
> unconditionally and Arc keys purely on virtual layer index. The checkpoint confirms the
> exclusion (`mtp.0.ffn.gate.bias` present, no `mtp.*.tid2eid`); for the real 43-layer
> config the two agree, so this is a synthetic-fixture hazard only.

**§4(e), note (i) — replace `— another thing the real config.json settles` with:**

> — **settled: the real config publishes `"swiglu_limit": 10.0`**, so the reference's
> `getattr` resolves and it clamps. Arc's hard default coincides with the shipped value.
> The clamp is therefore mandatory and the four unclamped paths are unconditional
> divergences.

**§12 (Recommended action order) — the "fetch config.json" item is done; replace it with
the follow-ups this fetch created:** flip the finding-6 YaRN default; clamp the shared
expert and the three other expert paths (finding 5, now unconditional); teach the loader
that routed experts are fp4 group-32, not the config's fp8 128×128 (new, §5(c) of the
wave-13 log).
