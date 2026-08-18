#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use candle_core::{DType, Device, Module, Result, Tensor};
use mistralrs_quant::{
    ColumnParallelLayer, QuantMethod, QuantizedConfig, ReplicatedLayer, RowParallelLayer,
    ShardedVarBuilder,
};

use std::{collections::HashMap, sync::Arc};

use crate::{
    amoe::{AnyMoeBaseModelMixin, AnyMoeConfig, AnyMoeExpertType, MlpLayer, MoeMlp},
    attention::SdpaParams,
    device_map::{DeviceMappedMask, DeviceMapper},
    get_delta_from_lora_ab,
    layers::{embedding, Activation, CausalMasker, MatMul, Mlp, RmsNorm, RotaryEmbedding, Sdpa},
    layers_masker::PastKvLenCache,
    paged_attention::{AttentionImplementation, ModelConfigMetadata, PagedAttention},
    pipeline::{
        extract_logits,
        text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata},
        EitherCache, IsqModel, KvCache, NormalCache, NormalLoadingMetadata, NormalModel,
    },
    serde_default_fn,
    utils::{progress::NiceProgressBar, unvarbuilder::UnVarBuilder},
};

serde_default_fn!(bool, word_emb_default, false);
// `Qwen2Config` upstream defaults, verified 2026-08-18 against
// transformers/src/transformers/models/qwen2/configuration_qwen2.py (main):
//   use_sliding_window: bool = False
//   max_window_layers: int = 28
serde_default_fn!(bool, use_sliding_window_default, false);
serde_default_fn!(usize, max_window_layers_default, 28);

#[derive(Debug, Clone, serde::Deserialize, Default, serde::Serialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub max_position_embeddings: usize,
    /// The **declared** window from `config.json`. This is NOT the window in
    /// effect: shipped Qwen2 configs carry a real number here *and*
    /// `use_sliding_window: false` (Qwen2.5-7B-Instruct-1M declares
    /// `sliding_window: 32768` with the flag off and a 1,010,000 context).
    /// Never read this field directly — go through
    /// [`Config::sliding_window_for_layer`] or
    /// [`Config::effective_sliding_window`], which apply the same gate
    /// transformers applies in `Qwen2Config.__post_init__`.
    #[serde(rename = "sliding_window")]
    pub declared_sliding_window: Option<usize>,
    #[serde(default = "use_sliding_window_default")]
    pub use_sliding_window: bool,
    #[serde(default = "max_window_layers_default")]
    pub max_window_layers: usize,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
    pub hidden_act: Activation,
    pub quantization_config: Option<QuantizedConfig>,
    #[serde(default = "word_emb_default")]
    pub tie_word_embeddings: bool,
}

impl Config {
    /// The sliding window actually in effect for `layer_idx`, or `None` for
    /// full attention.
    ///
    /// Mirrors `Qwen2Config.__post_init__`:
    /// `self.sliding_window = self.sliding_window if self.use_sliding_window else None`,
    /// then `"sliding_attention" if self.sliding_window is not None and
    /// i >= self.max_window_layers else "full_attention"`.
    ///
    /// The same shape already ships three times over in this tree
    /// (`qwen3.rs`, `qwen3_moe.rs`, `embedding_models/qwen3_embedding.rs`).
    pub fn sliding_window_for_layer(&self, layer_idx: usize) -> Option<usize> {
        if self.use_sliding_window && layer_idx >= self.max_window_layers {
            self.declared_sliding_window
        } else {
            None
        }
    }

    /// The window in effect for the model as a whole — `Some` only if at least
    /// one layer is actually windowed. Used where a single model-wide value is
    /// required (the causal mask, KV-cache sizing, PagedAttention metadata).
    pub fn effective_sliding_window(&self) -> Option<usize> {
        (0..self.num_hidden_layers).find_map(|i| self.sliding_window_for_layer(i))
    }
}

struct Attention {
    q_proj: Arc<dyn QuantMethod>,
    k_proj: Arc<dyn QuantMethod>,
    v_proj: Arc<dyn QuantMethod>,
    o_proj: Arc<dyn QuantMethod>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_emb: Arc<RotaryEmbedding>,
    paged_attn: Option<PagedAttention>,
    sdpa_params: SdpaParams,
}

impl Attention {
    fn new(
        rotary_emb: Arc<RotaryEmbedding>,
        cfg: &Config,
        vb: ShardedVarBuilder,
        layer_idx: usize,
        paged_attn: Option<PagedAttention>,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = hidden_sz / num_heads;
        let q_proj = ColumnParallelLayer::new(
            hidden_sz,
            num_heads * head_dim,
            &cfg.quantization_config,
            true,
            comm,
            vb.pp("q_proj"),
        )?;
        let kv_shard = mistralrs_quant::compute_kv_shard(
            cfg.num_key_value_heads,
            cfg.hidden_size / cfg.num_attention_heads,
            comm,
        );
        let k_proj = ColumnParallelLayer::new_with_shard(
            hidden_sz,
            num_kv_heads * head_dim,
            &cfg.quantization_config,
            true,
            comm,
            kv_shard,
            vb.pp("k_proj"),
        )?;
        let v_proj = ColumnParallelLayer::new_with_shard(
            hidden_sz,
            num_kv_heads * head_dim,
            &cfg.quantization_config,
            true,
            comm,
            kv_shard,
            vb.pp("v_proj"),
        )?;
        let o_proj = RowParallelLayer::new(
            num_heads * head_dim,
            hidden_sz,
            &cfg.quantization_config,
            false,
            comm,
            vb.pp("o_proj"),
        )?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads: num_heads / comm.world_size(),
            num_kv_heads: (num_kv_heads / comm.world_size()).max(1),
            head_dim,
            rotary_emb,
            paged_attn,
            sdpa_params: SdpaParams {
                n_kv_groups: mistralrs_quant::compute_n_kv_groups(
                    cfg.num_key_value_heads,
                    cfg.num_attention_heads,
                    comm,
                ),
                softcap: None,
                softmax_scale: 1.0 / (head_dim as f32).sqrt(),
                sliding_window: cfg.sliding_window_for_layer(layer_idx),
                sinks: None,
            },
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offsets: &[usize],
        kv_cache: &mut KvCache,
        metadata: Option<((Tensor, Tensor, Option<Tensor>, Option<Tensor>), &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;

        let original_dtype = xs.dtype();
        let mut xs = xs.clone();
        if let Some(t) = self.q_proj.quantized_act_type() {
            xs = xs.to_dtype(t)?;
        }
        let mut q = MatMul.qmethod_matmul(&xs, &*self.q_proj)?;
        let mut k = MatMul.qmethod_matmul(&xs, &*self.k_proj)?;
        let mut v = MatMul.qmethod_matmul(&xs, &*self.v_proj)?;
        if self.q_proj.quantized_act_type().is_some() {
            q = q.to_dtype(original_dtype)?;
            k = k.to_dtype(original_dtype)?;
            v = v.to_dtype(original_dtype)?;
        }

        let (q, k, v) = if q_len != 1 {
            let q = q
                .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
                .transpose(1, 2)?;
            let k = k
                .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
                .transpose(1, 2)?;
            let v = v
                .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
                .transpose(1, 2)?;
            (q, k, v)
        } else {
            let q = q.reshape((b_sz, self.num_heads, q_len, self.head_dim))?;
            let k = k.reshape((b_sz, self.num_kv_heads, q_len, self.head_dim))?;
            let v = v.reshape((b_sz, self.num_kv_heads, q_len, self.head_dim))?;
            (q, k, v)
        };

        let (q, k) = self.rotary_emb.forward(&q, &k, seqlen_offsets)?;

        let mut attn_output = match &self.paged_attn {
            Some(paged_attn) => match metadata {
                Some(((key_cache, value_cache, _, _), input_metadata)) => paged_attn.forward(
                    &q,
                    &k,
                    &v,
                    attention_mask,
                    Some(key_cache),
                    Some(value_cache),
                    input_metadata,
                    &self.sdpa_params,
                    Some(flash_params),
                )?,
                None => {
                    // If we don't have metadata, we are most likely generating an imatrix so we don't want to populate that.
                    // Generating the dummy metadata with the assumption that we are not generating text (only processing prompts).
                    let input_metadata = PagedAttentionInputMetadata::dummy(q.device())?;
                    // Sanity check.
                    assert!(attention_mask.is_some());
                    paged_attn.forward(
                        &q,
                        &k,
                        &v,
                        attention_mask,
                        None,
                        None,
                        &input_metadata,
                        &self.sdpa_params,
                        Some(flash_params),
                    )?
                }
            },
            None => {
                let (k, v) = kv_cache.append(&k, &v)?;

                Sdpa.run_attention(
                    &q,
                    &k,
                    &v,
                    attention_mask,
                    Some(flash_params),
                    &self.sdpa_params,
                )?
            }
        };

        if let Some(t) = self.q_proj.quantized_act_type() {
            attn_output = attn_output.to_dtype(t)?;
        }
        attn_output = if attention_mask.is_some() {
            attn_output.transpose(1, 2)?.reshape((b_sz, q_len, ()))?
        } else {
            attn_output.reshape((b_sz, q_len, ()))?
        };
        let mut res = MatMul.qmethod_matmul(&attn_output, &*self.o_proj)?;
        if self.q_proj.quantized_act_type().is_some() {
            res = res.to_dtype(original_dtype)?;
        }
        Ok(res)
    }
}

struct DecoderLayer {
    self_attn: Attention,
    mlp: Box<dyn MlpLayer>,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl DecoderLayer {
    #[allow(clippy::too_many_arguments)]
    fn new(
        rotary_emb: Arc<RotaryEmbedding>,
        cfg: &Config,
        vb: ShardedVarBuilder,
        mapper: &dyn DeviceMapper,
        layer_idx: usize,
        loading_isq: bool,
        paged_attn: Option<PagedAttention>,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<Self> {
        let self_attn = Attention::new(
            rotary_emb,
            cfg,
            mapper.set_device(layer_idx, vb.pp("self_attn"), loading_isq),
            layer_idx,
            paged_attn,
            comm,
        )?;
        let mlp = Mlp::new(
            mapper.set_device(layer_idx, vb.pp("mlp"), loading_isq),
            cfg.hidden_size,
            cfg.intermediate_size,
            &cfg.quantization_config,
            cfg.hidden_act,
            comm,
        )?;
        let input_layernorm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mapper.set_device(layer_idx, vb.pp("input_layernorm"), false),
        )?;
        let post_attention_layernorm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mapper.set_device(layer_idx, vb.pp("post_attention_layernorm"), false),
        )?;
        Ok(Self {
            self_attn,
            mlp: Box::new(mlp),
            input_layernorm,
            post_attention_layernorm,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offsets: &[usize],
        kv_cache: &mut KvCache,
        metadata: Option<((Tensor, Tensor, Option<Tensor>, Option<Tensor>), &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward(
            &xs,
            attention_mask,
            seqlen_offsets,
            kv_cache,
            metadata,
            flash_params,
        )?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = self
            .mlp
            .forward(&xs.apply(&self.post_attention_layernorm)?)?;
        residual + xs
    }
}

pub struct Model {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: Arc<dyn QuantMethod>,
    sliding_window: Option<usize>,
    device: Device,
    cache: EitherCache,
    max_seq_len: usize,
    mapper: Box<dyn DeviceMapper + Send + Sync>,
    cfg: ModelConfigMetadata,
}

impl Model {
    pub fn new(
        cfg: &Config,
        vb: ShardedVarBuilder,
        is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        if let Some(ref quant_cfg) = &cfg.quantization_config {
            tracing::info!(
                "Using {} quantization: {}.",
                quant_cfg.name(),
                quant_cfg.get_bits_name(&vb)
            );
        }
        let mapper = normal_loading_metadata.mapper;
        let vb_m = vb.pp("model");

        let embed_tokens = embedding(
            cfg.vocab_size,
            cfg.hidden_size,
            mapper.set_nm_device(vb_m.pp("embed_tokens"), false),
            &cfg.quantization_config,
        )?;
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;

        let mut ropes = HashMap::new();
        for layer_idx in 0..cfg.num_hidden_layers {
            let device = mapper
                .device_for(layer_idx, false)
                .unwrap_or(&normal_loading_metadata.real_device);
            ropes.insert(
                device.location(),
                Arc::new(RotaryEmbedding::new(
                    cfg.rope_theta as f32,
                    head_dim,
                    cfg.max_position_embeddings,
                    device,
                    is_gptx,
                    vb_m.dtype(),
                )?),
            );
        }

        let vb_l = vb_m.pp("layers");
        let layers: Vec<DecoderLayer> = NiceProgressBar::<_, 'b'>(
            0..cfg.num_hidden_layers,
            "Loading repeating layers",
            &normal_loading_metadata.multi_progress,
        )
        .par_iter_if_isq(|layer_idx| -> Result<DecoderLayer> {
            let device = mapper
                .device_for(layer_idx, false)
                .unwrap_or(&normal_loading_metadata.real_device);
            let rotary_emb = ropes
                .get(&device.location())
                .expect("No RoPE for device location!")
                .clone();
            let paged_attn = match &attention_mechanism {
                AttentionImplementation::Eager => None,
                AttentionImplementation::PagedAttention => {
                    Some(PagedAttention::new(head_dim, device, None)?)
                }
            };
            let comm = mapper.get_comm_for(layer_idx)?;
            DecoderLayer::new(
                rotary_emb.clone(),
                cfg,
                vb_l.pp(layer_idx),
                &*mapper,
                layer_idx,
                normal_loading_metadata.loading_isq,
                paged_attn,
                &comm,
            )
        })?;
        let norm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mapper.set_nm_device(vb_m.pp("norm"), false),
        )?;
        let lm_head = if !cfg.tie_word_embeddings {
            ReplicatedLayer::new(
                cfg.hidden_size,
                cfg.vocab_size,
                &cfg.quantization_config,
                false,
                mapper.set_nm_device(vb.pp("lm_head"), normal_loading_metadata.loading_isq),
            )?
        } else {
            ReplicatedLayer::from_linear(candle_nn::Linear::new(
                mapper.cast_nm_device(
                    embed_tokens.embeddings(),
                    normal_loading_metadata.loading_isq,
                )?,
                None,
            ))?
        };
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            sliding_window: cfg.effective_sliding_window(),
            device: normal_loading_metadata.real_device,
            cache: EitherCache::Normal(NormalCache::new(
                cfg.num_hidden_layers,
                cfg.max_position_embeddings,
            )),
            max_seq_len: cfg.max_position_embeddings,
            cfg: ModelConfigMetadata {
                max_seq_len: cfg.max_position_embeddings,
                num_layers: cfg.num_hidden_layers,
                hidden_size: cfg.hidden_size,
                num_attn_heads: cfg.num_attention_heads / mapper.get_comm_for(0)?.world_size(),
                num_kv_heads: (cfg.num_key_value_heads / mapper.get_comm_for(0)?.world_size())
                    .max(1),
                sliding_window: cfg.effective_sliding_window(),
                k_head_dim: cfg.hidden_size / cfg.num_attention_heads,
                v_head_dim: cfg.hidden_size / cfg.num_attention_heads,
                kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
            },
            mapper,
        })
    }

    pub fn get_input_embeddings(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.embed_tokens.forward(input_ids)
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offsets: &[usize],
        context_lens: Vec<(usize, usize)>,
        metadata: Option<(Vec<(Tensor, Tensor, Option<Tensor>, Option<Tensor>)>, &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let xs = self.embed_tokens.forward(input_ids)?;
        self.forward_embed(
            input_ids,
            xs,
            seqlen_offsets,
            context_lens,
            metadata,
            flash_params,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward_embed(
        &self,
        input_ids: &Tensor,
        mut xs: Tensor,
        seqlen_offsets: &[usize],
        context_lens: Vec<(usize, usize)>,
        metadata: Option<(Vec<(Tensor, Tensor, Option<Tensor>, Option<Tensor>)>, &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let cache = &mut self.cache.normal().0;
        let attention_mask = CausalMasker.make_sliding_window_causal_mask_matrix(
            input_ids,
            metadata
                .as_ref()
                .map(|(_, _)| &seqlen_offsets as &dyn PastKvLenCache)
                .unwrap_or(cache as &dyn PastKvLenCache),
            self.sliding_window,
            xs.dtype(),
            self.cfg.num_attn_heads,
        )?;
        let attention_mask = attention_mask.filter(|_| {
            metadata
                .as_ref()
                .map(|(_, meta)| meta.is_first_prompt_chunk)
                .unwrap_or(true)
        });
        let attention_mask = DeviceMappedMask::new(attention_mask, &*self.mapper)?;
        for (i, layer) in self.layers.iter().enumerate() {
            xs = self.mapper.map(xs, i)?;
            xs = layer.forward(
                &xs,
                attention_mask.as_ref().map(|m| m.get(xs.device())),
                seqlen_offsets,
                &mut cache[i],
                metadata
                    .as_ref()
                    .map(|(kv_cache, metadata)| (kv_cache[i].clone(), *metadata)),
                flash_params,
            )?
        }
        let xs = xs.to_device(&self.device)?;
        let xs = xs.apply(&self.norm)?;
        let mut xs = extract_logits(&xs, context_lens)?;
        if let Some(t) = self.lm_head.quantized_act_type() {
            xs = xs.to_dtype(t)?;
        }
        MatMul.qmethod_matmul(&xs, &*self.lm_head)
    }

    pub fn embed_dtype(&self) -> DType {
        self.embed_tokens.embeddings().dtype()
    }
}

impl IsqModel for Model {
    fn get_layers(
        &mut self,
    ) -> (
        Vec<(&mut Arc<dyn QuantMethod>, Option<usize>)>,
        &dyn DeviceMapper,
    ) {
        let mut tensors = Vec::new();
        tensors.push((&mut self.lm_head, None));
        for (i, layer) in self.layers.iter_mut().enumerate() {
            tensors.push((&mut layer.self_attn.q_proj, Some(i)));
            tensors.push((&mut layer.self_attn.k_proj, Some(i)));
            tensors.push((&mut layer.self_attn.v_proj, Some(i)));
            tensors.push((&mut layer.self_attn.o_proj, Some(i)));
            tensors.extend(
                layer
                    .mlp
                    .get_isq_layers()
                    .into_iter()
                    .map(|m| (m, Some(i)))
                    .collect::<Vec<_>>(),
            );
        }
        (tensors, &*self.mapper)
    }

    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();

        let uvb_m = uvb.pp("model");
        uvb_m.pp("embed_tokens").add(&self.embed_tokens);
        uvb_m.pp("norm").add(&self.norm);

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let uvb_l = uvb_m.pp("layers").pp(layer_idx);
            uvb_l.pp("input_layernorm").add(&layer.input_layernorm);
            uvb_l
                .pp("post_attention_layernorm")
                .add(&layer.post_attention_layernorm);
        }

        uvb.to_safetensors()
    }

    fn imatrix_names(&self) -> candle_core::Result<Vec<Option<String>>> {
        // NOTE: dependant on the exact implementation in get_layers!
        let mut names = Vec::new();
        // lm_head
        names.push(None);
        for i in 0..self.layers.len() {
            names.push(Some(format!("blk.{i}.attn_q.weight")));
            names.push(Some(format!("blk.{i}.attn_k.weight")));
            names.push(Some(format!("blk.{i}.attn_v.weight")));
            names.push(Some(format!("blk.{i}.attn_output.weight")));
            names.push(Some(format!("blk.{i}.ffn_gate.weight")));
            names.push(Some(format!("blk.{i}.ffn_up.weight")));
            names.push(Some(format!("blk.{i}.ffn_down.weight")));
        }
        Ok(names)
    }
}

impl NormalModel for Model {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offsets: &[usize],
        context_lens: Vec<(usize, usize)>,
        _position_ids: Vec<usize>,
        metadata: Option<(Vec<(Tensor, Tensor, Option<Tensor>, Option<Tensor>)>, &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        self.forward(
            input_ids,
            seqlen_offsets,
            context_lens,
            metadata,
            flash_params,
        )
    }
    fn xlora_forward(
        &self,
        _input_ids: &Tensor,
        _input_ids_full: &Tensor,
        _seqlen_offsets: &[usize],
        _seqlen_offsets_full: &[usize],
        _no_kv_cache: bool,
        _non_granular_state: &Option<crate::xlora_models::NonGranularState>,
        _context_lens: Vec<(usize, usize)>,
        _position_ids: Vec<usize>,
        _flash_params: &FlashParams,
        _flash_params_full: &FlashParams,
    ) -> Result<Tensor> {
        unimplemented!()
    }
    fn cache(&self) -> &EitherCache {
        &self.cache
    }
    fn cache_mut(&mut self) -> &mut EitherCache {
        &mut self.cache
    }
    fn device(&self) -> &Device {
        &self.device
    }
    fn is_xlora(&self) -> bool {
        false
    }
    fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }
    fn config(&self) -> &ModelConfigMetadata {
        &self.cfg
    }
}

impl AnyMoeBaseModelMixin for Model {
    fn get_mlps(&self) -> Vec<&dyn MlpLayer> {
        let mut mlps = Vec::new();
        for layer in &self.layers {
            mlps.push(&*layer.mlp);
        }
        mlps
    }
    fn get_mlps_mut(&mut self) -> Vec<&mut Box<dyn MlpLayer>> {
        let mut mlps = Vec::new();
        for layer in &mut self.layers {
            mlps.push(&mut layer.mlp);
        }
        mlps
    }
    fn create_anymoe_layers(
        &mut self,
        additional_vbs: Vec<ShardedVarBuilder>,
        config: AnyMoeConfig,
        (prefix, mlp): (String, String),
        mut layers: Vec<usize>,
        expert_type: AnyMoeExpertType,
        gate_vb: Option<ShardedVarBuilder>,
    ) -> Result<()> {
        let mut experts: Vec<Vec<Box<dyn MlpLayer>>> = Vec::new();
        if layers.is_empty() {
            layers = (0..self.layers.len()).collect::<Vec<_>>();
        }
        for _ in 0..layers.len() {
            experts.push(Vec::new());
        }
        for vb in additional_vbs {
            let vb = vb.pp(&prefix);
            for (layer, row) in experts.iter_mut().enumerate() {
                if !layers.contains(&layer) {
                    continue;
                }

                let intermediate_size = self.layers[layer].mlp.get_params()[1];
                let hidden_size = self.layers[layer].mlp.get_params()[0];
                match expert_type {
                    AnyMoeExpertType::FineTuned => {
                        let (dtype, device) = self.layers[layer].mlp.dtype_device();
                        row.push(Box::new(Mlp::replicate(
                            self.layers[layer].mlp.get_params(),
                            vb.pp(layer).pp(&mlp).set_dtype(dtype).set_device(device),
                            self.layers[layer].mlp.hidden_act(),
                            &self.mapper.get_comm_for(layer)?,
                        )?));
                    }
                    AnyMoeExpertType::LoraAdapter {
                        rank,
                        alpha,
                        ref target_modules,
                    } => {
                        let vb_mlp = vb.pp(layer).pp(&mlp);

                        let gate_proj_delta = if target_modules.contains(&"gate_proj".to_string()) {
                            Some(get_delta_from_lora_ab!(
                                vb_mlp,
                                rank,
                                alpha,
                                (hidden_size, intermediate_size),
                                "gate_proj"
                            ))
                        } else {
                            None
                        };
                        let up_proj_delta = if target_modules.contains(&"up_proj".to_string()) {
                            Some(get_delta_from_lora_ab!(
                                vb_mlp,
                                rank,
                                alpha,
                                (hidden_size, intermediate_size),
                                "up_proj"
                            ))
                        } else {
                            None
                        };
                        let down_proj_delta = if target_modules.contains(&"down_proj".to_string()) {
                            Some(get_delta_from_lora_ab!(
                                vb_mlp,
                                rank,
                                alpha,
                                (intermediate_size, hidden_size),
                                "down_proj"
                            ))
                        } else {
                            None
                        };

                        row.push(self.layers[layer].mlp.new_added_delta(vec![
                            gate_proj_delta,
                            up_proj_delta,
                            down_proj_delta,
                        ])?);
                    }
                }
            }
        }
        for (layer, expert) in layers.into_iter().zip(experts) {
            let mut experts_all = vec![self.layers[layer].mlp.clone()];
            experts_all.extend(expert);
            let (dtype, device) = self.layers[layer].mlp.dtype_device();
            self.layers[layer].mlp = Box::new(MoeMlp::new(
                experts_all,
                config.clone(),
                dtype,
                &device,
                layer,
                gate_vb.as_ref(),
            )?);
        }
        Ok(())
    }
    fn amoe_supported(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers_masker::NotACache;

    /// Verbatim `config.json` of `Qwen/Qwen2.5-7B-Instruct-1M`, fetched
    /// 2026-08-18 from
    /// `https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-1M/raw/main/config.json`.
    ///
    /// This is the config that makes the bug lethal rather than latent: it
    /// declares `sliding_window: 32768` with `use_sliding_window: false` and a
    /// `max_position_embeddings` of 1,010,000. Reading the declared window
    /// clamps attention — and the PagedAttention KV budget — to 32k on a model
    /// shipped for 1M context, silently and with no error.
    const QWEN25_7B_1M_CONFIG: &str = r#"{
      "architectures": ["Qwen2ForCausalLM"],
      "attention_dropout": 0.0,
      "bos_token_id": 151643,
      "eos_token_id": 151645,
      "hidden_act": "silu",
      "hidden_size": 3584,
      "initializer_range": 0.02,
      "intermediate_size": 18944,
      "max_position_embeddings": 1010000,
      "max_window_layers": 28,
      "model_type": "qwen2",
      "num_attention_heads": 28,
      "num_hidden_layers": 28,
      "num_key_value_heads": 4,
      "rms_norm_eps": 1e-05,
      "rope_scaling": null,
      "rope_theta": 10000000.0,
      "sliding_window": 32768,
      "tie_word_embeddings": false,
      "torch_dtype": "bfloat16",
      "transformers_version": "4.47.1",
      "use_cache": true,
      "use_sliding_window": false,
      "vocab_size": 152064
    }"#;

    fn cfg_with(
        sliding_window: Option<usize>,
        use_sliding_window: bool,
        max_window_layers: usize,
        num_hidden_layers: usize,
    ) -> Config {
        Config {
            vocab_size: 32,
            hidden_size: 8,
            intermediate_size: 16,
            num_hidden_layers,
            num_attention_heads: 2,
            num_key_value_heads: 2,
            max_position_embeddings: 64,
            declared_sliding_window: sliding_window,
            use_sliding_window,
            max_window_layers,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-6,
            hidden_act: Activation::Silu,
            quantization_config: None,
            tie_word_embeddings: false,
        }
    }

    /// Reproduces the mask call `Model::forward_embed` makes:
    /// `CausalMasker.make_sliding_window_causal_mask_matrix(input_ids, cache,
    /// self.sliding_window, dtype, num_attn_heads)`, where `self.sliding_window`
    /// is whatever `Model::new` stored.
    ///
    /// `input_ids` values are irrelevant *by construction*: the masker reads
    /// only `input_ids.dims2()` and `input_ids.device()`, never the token
    /// values, so a zero fill erases nothing this test depends on.
    fn masked_entries_at_swa_entry_point(window: Option<usize>, tgt_len: usize) -> usize {
        let input_ids = Tensor::zeros((1, tgt_len), DType::U32, &Device::Cpu).unwrap();
        let mask = CausalMasker
            .make_sliding_window_causal_mask_matrix(&input_ids, &NotACache, window, DType::F32, 2)
            .unwrap()
            .expect("tgt_len > 1 must produce a real mask on CPU");
        mask.flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .filter(|v| v.is_infinite() && v.is_sign_negative())
            .count()
    }

    /// A shipped Qwen2 config declares a window *and* turns it off. The
    /// declared number must never reach attention.
    ///
    /// Mutation check (run 2026-08-18): make `sliding_window_for_layer` return
    /// `self.declared_sliding_window` unconditionally — master's behaviour —
    /// and this reports `effective must be None, got Some(32768)`.
    #[test]
    fn qwen2_declared_window_is_not_the_effective_window() {
        let cfg: Config = serde_json::from_str(QWEN25_7B_1M_CONFIG)
            .expect("the shipped Qwen2.5-7B-Instruct-1M config must still parse");

        assert_eq!(
            cfg.declared_sliding_window,
            Some(32768),
            "fixture guard: this config must still declare a window, or the test proves nothing"
        );
        assert!(
            !cfg.use_sliding_window,
            "fixture guard: this config must still have the flag OFF"
        );
        assert_eq!(cfg.max_position_embeddings, 1_010_000);

        let effective = cfg.effective_sliding_window();
        assert_eq!(
            effective, None,
            "declared = {:?}, use_sliding_window = {}, max_window_layers = {} => effective must be None, got {:?}",
            cfg.declared_sliding_window,
            cfg.use_sliding_window,
            cfg.max_window_layers,
            effective
        );

        for layer_idx in 0..cfg.num_hidden_layers {
            assert_eq!(
                cfg.sliding_window_for_layer(layer_idx),
                None,
                "layer {layer_idx} of {} must be full attention",
                cfg.num_hidden_layers
            );
        }
    }

    /// `use_sliding_window: false` must not merely be *recorded* — it has to
    /// change what the mask erases at the SWA entry point.
    ///
    /// Numbers are exact and printed: for `tgt_len = 8` an 8x8 causal mask
    /// forbids the 28 strictly-upper-triangular pairs; a window of 4
    /// additionally forbids the 6 pairs with `j + 4 < i`, i.e. 34.
    ///
    /// Mutation check (run 2026-08-18): pass `declared_sliding_window` instead
    /// of `effective_sliding_window()` and the flag-off case reports 34.
    #[test]
    fn qwen2_flag_off_reaches_the_swa_entry_point_and_erases_nothing_extra() {
        const TGT_LEN: usize = 8;
        const CAUSAL_ONLY: usize = 28;
        const CAUSAL_PLUS_WINDOW_4: usize = 34;

        let baseline_causal = masked_entries_at_swa_entry_point(None, TGT_LEN);
        assert_eq!(
            baseline_causal, CAUSAL_ONLY,
            "sanity: a windowless mask over {TGT_LEN} tokens must forbid exactly {CAUSAL_ONLY} pairs, got {baseline_causal}"
        );

        let window_applied = masked_entries_at_swa_entry_point(Some(4), TGT_LEN);
        assert_eq!(
            window_applied, CAUSAL_PLUS_WINDOW_4,
            "sanity: a window of 4 must forbid {CAUSAL_PLUS_WINDOW_4} pairs (> {CAUSAL_ONLY}), got {window_applied} — if these two are equal the assertions below cannot discriminate"
        );

        // Flag OFF, window declared: master fed `Some(4)` in here.
        let off = cfg_with(Some(4), false, 0, 2);
        let off_masked = masked_entries_at_swa_entry_point(off.effective_sliding_window(), TGT_LEN);
        assert_eq!(
            off_masked, CAUSAL_ONLY,
            "use_sliding_window=false with declared window {:?}: mask forbade {off_masked} pairs, must forbid {CAUSAL_ONLY} (windowed = {CAUSAL_PLUS_WINDOW_4})",
            off.declared_sliding_window
        );

        // Flag ON: the window must still be honoured, or the fix over-corrects.
        let on = cfg_with(Some(4), true, 0, 2);
        let on_masked = masked_entries_at_swa_entry_point(on.effective_sliding_window(), TGT_LEN);
        assert_eq!(
            on_masked, CAUSAL_PLUS_WINDOW_4,
            "use_sliding_window=true with declared window {:?}: mask forbade {on_masked} pairs, must forbid {CAUSAL_PLUS_WINDOW_4}",
            on.declared_sliding_window
        );
    }

    /// `max_window_layers` is the second gate transformers applies, and it is
    /// per layer. Every shipped Qwen2 config surveyed on 2026-08-18 has
    /// `max_window_layers == num_hidden_layers`, which makes *every* layer full
    /// attention even if the flag were flipped on.
    #[test]
    fn qwen2_max_window_layers_gates_per_layer() {
        let cfg = cfg_with(Some(4), true, 1, 3);
        assert_eq!(cfg.sliding_window_for_layer(0), None, "layer 0 < 1");
        assert_eq!(cfg.sliding_window_for_layer(1), Some(4), "layer 1 >= 1");
        assert_eq!(cfg.sliding_window_for_layer(2), Some(4), "layer 2 >= 1");
        assert_eq!(cfg.effective_sliding_window(), Some(4));

        // max_window_layers == num_hidden_layers: no layer is ever windowed.
        let none_windowed = cfg_with(Some(4), true, 3, 3);
        for layer_idx in 0..3 {
            assert_eq!(
                none_windowed.sliding_window_for_layer(layer_idx),
                None,
                "max_window_layers=3, num_hidden_layers=3: layer {layer_idx} must be full attention"
            );
        }
        assert_eq!(none_windowed.effective_sliding_window(), None);
    }

    /// Defaults must match `Qwen2Config` upstream (`use_sliding_window=False`,
    /// `max_window_layers=28`) so a config.json that omits them behaves the way
    /// transformers behaves — off.
    #[test]
    fn qwen2_missing_flag_defaults_to_off_like_transformers() {
        let json = r#"{
          "vocab_size": 32, "hidden_size": 8, "intermediate_size": 16,
          "num_hidden_layers": 2, "num_attention_heads": 2, "num_key_value_heads": 2,
          "max_position_embeddings": 64, "sliding_window": 4096,
          "rope_theta": 10000.0, "rms_norm_eps": 1e-6, "hidden_act": "silu"
        }"#;
        let cfg: Config = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.declared_sliding_window, Some(4096));
        assert!(!cfg.use_sliding_window, "upstream default is False");
        assert_eq!(cfg.max_window_layers, 28, "upstream default is 28");
        assert_eq!(cfg.effective_sliding_window(), None);
    }

    /// No call site outside the two gate methods may read the declared window.
    ///
    /// The field was renamed from `sliding_window` precisely so the compiler
    /// would surface every existing reader; this keeps a *new* reader from
    /// being added without the gate.
    ///
    /// Mutation check (run 2026-08-18): restoring the loader's
    /// `cfg.declared_sliding_window` made this fire with
    /// `1 ungated read(s) ... ["sliding_window: cfg.declared_sliding_window"]`.
    #[test]
    fn declared_window_has_no_ungated_readers() {
        let accessor = concat!(".", "declared_", "sliding_window");
        for (name, src) in [
            ("models/qwen2.rs", include_str!("qwen2.rs")),
            (
                "pipeline/loaders/normal_loaders.rs",
                include_str!("../pipeline/loaders/normal_loaders.rs"),
            ),
        ] {
            // Drop each file's own `#[cfg(test)]` tail so fixtures don't self-trip.
            let body = src.split("#[cfg(test)]").next().unwrap();
            // Third state: if an earlier `#[cfg(test)]` ever truncates the body
            // above the code this guards, say "cannot answer" rather than pass
            // vacuously.
            assert!(
                body.contains("effective_sliding_window"),
                "{name}: guard cannot answer — the non-test body it kept ({} of {} bytes) no longer contains the gated call site",
                body.len(),
                src.len()
            );
            let ungated: Vec<String> = body
                .match_indices(accessor)
                .filter(|(pos, _)| !body[..*pos].ends_with("self"))
                .map(|(pos, _)| {
                    let start = body[..pos].rfind('\n').map_or(0, |i| i + 1);
                    body[start..pos + accessor.len()].trim().to_string()
                })
                .collect();
            assert!(
                ungated.is_empty(),
                "{name}: {} ungated read(s) of the declared window (must go through sliding_window_for_layer / effective_sliding_window): {ungated:?}",
                ungated.len()
            );
        }
    }
}
