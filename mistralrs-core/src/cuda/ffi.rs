use std::ffi::c_void;

#[allow(dead_code)]
extern "C" {
    pub(crate) fn asort_asc_f32(
        x: *const c_void,
        dst: *mut c_void,
        nrows: i32,
        ncols: i32,
        inplace: bool,
        stream: i64,
    );
    pub(crate) fn asort_asc_f16(
        x: *const c_void,
        dst: *mut c_void,
        nrows: i32,
        ncols: i32,
        inplace: bool,
        stream: i64,
    );
    pub(crate) fn asort_asc_bf16(
        x: *const c_void,
        dst: *const c_void,
        nrows: i32,
        ncols: i32,
        inplace: bool,
        stream: i64,
    );
    pub(crate) fn asort_asc_f64(
        x: *const c_void,
        dst: *mut c_void,
        nrows: i32,
        ncols: i32,
        inplace: bool,
        stream: i64,
    );
    pub(crate) fn asort_asc_u8(
        x: *const c_void,
        dst: *mut c_void,
        nrows: i32,
        ncols: i32,
        inplace: bool,
        stream: i64,
    );
    pub(crate) fn asort_asc_u32(
        x: *const c_void,
        dst: *mut c_void,
        nrows: i32,
        ncols: i32,
        inplace: bool,
        stream: i64,
    );
    pub(crate) fn asort_asc_i64(
        x: *const c_void,
        dst: *mut c_void,
        nrows: i32,
        ncols: i32,
        inplace: bool,
        stream: i64,
    );
    pub(crate) fn asort_desc_f32(
        x: *const c_void,
        dst: *mut c_void,
        nrows: i32,
        ncols: i32,
        inplace: bool,
        stream: i64,
    );
    pub(crate) fn asort_desc_f16(
        x: *const c_void,
        dst: *mut c_void,
        nrows: i32,
        ncols: i32,
        inplace: bool,
        stream: i64,
    );
    pub(crate) fn asort_desc_bf16(
        x: *const c_void,
        dst: *mut c_void,
        nrows: i32,
        ncols: i32,
        inplace: bool,
        stream: i64,
    );
    pub(crate) fn asort_desc_f64(
        x: *const c_void,
        dst: *mut c_void,
        nrows: i32,
        ncols: i32,
        inplace: bool,
        stream: i64,
    );
    pub(crate) fn asort_desc_u8(
        x: *const c_void,
        dst: *mut c_void,
        nrows: i32,
        ncols: i32,
        inplace: bool,
        stream: i64,
    );
    pub(crate) fn asort_desc_u32(
        x: *const c_void,
        dst: *mut c_void,
        nrows: i32,
        ncols: i32,
        inplace: bool,
        stream: i64,
    );
    pub(crate) fn asort_desc_i64(
        x: *const c_void,
        dst: *mut c_void,
        nrows: i32,
        ncols: i32,
        inplace: bool,
        stream: i64,
    );

    // for unquntized models (decoding)
    pub fn moe_gemm(
        input: *const c_void,   // input [size_m, size_k]
        weights: *const c_void, // weights [num_experts, size_n, size_k]
        sorted_token_ids: *const i32,
        expert_ids: *const i32,
        topk_weights: *const f32, // device ptr or nullptr
        output: *mut c_void,      // output [size_m, size_n]
        num_experts: i32,
        topk: i32,
        size_m: i32,
        size_n: i32,
        size_k: i32,
        dtype: i32, // 0=float16, 1=bf16 (for input)
        stream: i64,
    );

    // for unquntized models (prefill)
    pub fn moe_gemm_wmma(
        input: *const c_void,         // device pointer [size_m, size_k]
        weights: *const c_void,       // device pointer [num_experts, size_n, size_k]
        sorted_token_ids: *const i32, // device pointer [size_m]
        expert_ids: *const i32,       // host array [size_m] (expert id per sorted token)
        topk_weights: *const f32,
        output: *mut c_void, // device pointer [size_m, size_n]
        num_experts: i32,
        topk: i32,
        size_m: i32,
        size_n: i32,
        size_k: i32,
        dtype: i32, // 0=float16, 1=bf16 (for input/output)
        stream: i64,
    );

    // for unquntized models (decoding) with transposed weights [num_experts, size_k, size_n]
    pub fn moe_gemm_transposed(
        input: *const c_void,   // input [size_m, size_k]
        weights: *const c_void, // weights [num_experts, size_k, size_n] - transposed layout
        sorted_token_ids: *const i32,
        expert_ids: *const i32,
        topk_weights: *const f32, // device ptr or nullptr
        output: *mut c_void,      // output [size_m, size_n]
        num_experts: i32,
        topk: i32,
        size_m: i32,
        size_n: i32,
        size_k: i32,
        dtype: i32, // 0=float16, 1=bf16 (for input)
        stream: i64,
    );

    // for unquntized models (prefill) with transposed weights [num_experts, size_k, size_n]
    pub fn moe_gemm_wmma_transposed(
        input: *const c_void,         // device pointer [size_m, size_k]
        weights: *const c_void, // device pointer [num_experts, size_k, size_n] - transposed layout
        sorted_token_ids: *const i32, // device pointer [size_m]
        expert_ids: *const i32, // host array [size_m] (expert id per sorted token)
        topk_weights: *const f32,
        output: *mut c_void, // device pointer [size_m, size_n]
        num_experts: i32,
        topk: i32,
        size_m: i32,
        size_n: i32,
        size_k: i32,
        dtype: i32, // 0=float16, 1=bf16 (for input/output)
        stream: i64,
    );

    // MoE GEMV for decode phase (optimized for small batch sizes M <= 8)
    pub fn moe_gemv(
        input: *const c_void,   // input [size_m or size_m / topk, size_k]
        weights: *const c_void, // weights [num_experts, size_n, size_k]
        sorted_token_ids: *const i32,
        expert_ids: *const i32,
        topk_weights: *const f32, // device ptr or nullptr
        output: *mut c_void,      // output [size_m, size_n]
        num_experts: i32,
        topk: i32,
        size_m: i32,
        size_n: i32,
        size_k: i32,
        dtype: i32, // 0=float16, 1=bf16 (for input)
        stream: i64,
    );

    // MoE GEMV for decode phase with transposed weights [num_experts, size_k, size_n]
    pub fn moe_gemv_transposed(
        input: *const c_void,   // input [size_m or size_m / topk, size_k]
        weights: *const c_void, // weights [num_experts, size_k, size_n] - transposed layout
        sorted_token_ids: *const i32,
        expert_ids: *const i32,
        topk_weights: *const f32, // device ptr or nullptr
        output: *mut c_void,      // output [size_m, size_n]
        num_experts: i32,
        topk: i32,
        size_m: i32,
        size_n: i32,
        size_k: i32,
        dtype: i32, // 0=float16, 1=bf16 (for input)
        stream: i64,
    );

    // Optimized parallel topk for small k (MoE routing)
    // Single kernel call writes to both values and indices buffers
    pub(crate) fn topk_f32(
        input: *const c_void,
        values_out: *mut c_void,  // [nrows, k]
        indices_out: *mut c_void, // [nrows, k] as u32
        nrows: i32,
        ncols: i32,
        k: i32,
        stream: i64,
    );
    pub(crate) fn topk_bf16(
        input: *const c_void,
        values_out: *mut c_void,  // [nrows, k]
        indices_out: *mut c_void, // [nrows, k] as u32
        nrows: i32,
        ncols: i32,
        k: i32,
        stream: i64,
    );
    pub(crate) fn topk_f16(
        input: *const c_void,
        values_out: *mut c_void,  // [nrows, k]
        indices_out: *mut c_void, // [nrows, k] as u32
        nrows: i32,
        ncols: i32,
        k: i32,
        stream: i64,
    );

    // Fused topk + softmax - returns softmax weights directly (not raw logits)
    pub(crate) fn topk_softmax_f32(
        input: *const c_void,
        weights_out: *mut c_void, // [nrows, k] - softmax weights
        indices_out: *mut c_void, // [nrows, k] as u32
        nrows: i32,
        ncols: i32,
        k: i32,
        stream: i64,
    );
    pub(crate) fn topk_softmax_bf16(
        input: *const c_void,
        weights_out: *mut c_void,
        indices_out: *mut c_void,
        nrows: i32,
        ncols: i32,
        k: i32,
        stream: i64,
    );
    pub(crate) fn topk_softmax_f16(
        input: *const c_void,
        weights_out: *mut c_void,
        indices_out: *mut c_void,
        nrows: i32,
        ncols: i32,
        k: i32,
        stream: i64,
    );

    // Mamba SSM selective scan kernel
    pub(crate) fn selective_scan_cuda(
        x: *const f32,       // (batch, seq_len, n_heads * head_dim)
        dt: *const f32,      // (batch, seq_len, n_heads)
        a: *const f32,       // (n_heads,) - negative exp of A_log
        b: *const f32,       // (batch, seq_len, n_heads * d_state)
        c: *const f32,       // (batch, seq_len, n_heads * d_state)
        d: *const f32,       // (n_heads,)
        dt_bias: *const f32, // (n_heads,)
        state: *mut f32,     // (batch, n_heads, head_dim, d_state)
        y: *mut f32,         // (batch, seq_len, n_heads * head_dim)
        batch_size: i32,
        n_heads: i32,
        head_dim: i32,
        d_state: i32,
        seq_len: i32,
        dt_min: f32,
        dt_max: f32,
        stream: i64,
    );

    // GDN (Gated Delta Net) kernels for qwen3_next
    pub(crate) fn gated_delta_rule_recurrence(
        q: *const f32,
        k: *const f32,
        v: *const f32,
        g: *const f32,
        beta: *const f32,
        state: *mut f32,
        output: *mut f32,
        bh: i32,
        seq_len: i32,
        k_dim: i32,
        v_dim: i32,
        stream: i64,
    );
    // Chunked GDN recurrence for prefill (processes tokens in BT=64 chunks)
    pub(crate) fn chunked_gated_delta_rule_recurrence(
        q: *const f32,
        k: *const f32,
        v: *const f32,
        g: *const f32,
        beta: *const f32,
        state: *mut f32,
        output: *mut f32,
        bh: i32,
        seq_len: i32,
        k_dim: i32,
        v_dim: i32,
        stream: i64,
    );
    pub(crate) fn causal_conv1d_update(
        x: *const c_void,
        weight: *const c_void,
        conv_state: *mut c_void,
        output: *mut c_void,
        batch_size: i32,
        conv_dim: i32,
        kernel_size: i32,
        dtype: i32,
        stream: i64,
    );
    pub(crate) fn causal_conv1d_full(
        x: *const c_void,
        weight: *const c_void,
        conv_state_out: *mut c_void,
        output: *mut c_void,
        batch_size: i32,
        conv_dim: i32,
        seq_len: i32,
        kernel_size: i32,
        dtype: i32,
        stream: i64,
    );
    pub(crate) fn fused_gdn_gating(
        b: *const c_void,
        a: *const c_void,
        a_log: *const f32,
        dt_bias: *const f32,
        beta_out: *mut c_void,
        g_out: *mut c_void,
        total_elements: i32,
        num_heads: i32,
        dtype: i32,
        stream: i64,
    );
    pub(crate) fn sinkhorn_normalize_f32(
        in_: *const c_void,
        out: *mut c_void,
        n: i32,
        hc: i32,
        iters: i32,
        eps: f32,
        stream: i64,
    );
    // cuda/hc_fused.cu — fused V4 router-region kernels. Like sinkhorn.cu these
    // are compiled by build.rs's dedicated no-fast-math builder; see the
    // bit-identity contract at the top of that file.
    #[allow(clippy::too_many_arguments)]
    /// Fused DeepSeek-V4 Q/K pre-attention block (`cuda/qk_norm_rope.cu`):
    /// head transpose + per-head Q RMS-norm + adjacent-pair RoPE + NoPE/PE
    /// recombination, in one launch. Returns 0 on launch, non-zero if the
    /// (dtype, head_dim, rope_dim) triple is outside the specialised set — the
    /// caller must then run the eager chain rather than approximate.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn arc_qk_norm_rope_bf16_v2(
        q_in: *const c_void,
        k_in: *const c_void,
        cos_tab: *const c_void,
        sin_tab: *const c_void,
        q_out: *mut c_void,
        k_out: *mut c_void,
        n_heads: i32,
        batch: i32,
        seq_len: i32,
        head_dim: i32,
        rope_dim: i32,
        pos_offset: i32,
        dtype: i32,
        inv_n_bits: u32,
        zero_bits: u32,
        one_bits: u32,
        eps_bits: u32,
        stream: i64,
    ) -> i32;

    pub(crate) fn hc_pre_fused_f32(
        x_flat: *const c_void,
        mixes_raw: *const c_void,
        hc_scale: *const c_void,
        hc_base: *const c_void,
        pre: *mut c_void,
        post: *mut c_void,
        comb_pre: *mut c_void,
        n: i32,
        d: i32,
        m: i32,
        hc: i32,
        block_dim: i32,
        inv_d: f32,
        rms_eps: f32,
        hc_eps: f32,
        stream: i64,
    );
    pub(crate) fn sqrt_softplus_f32(
        inp: *const c_void,
        out: *mut c_void,
        numel: i64,
        stream: i64,
    );
    /// `y = sum_i pre[i] * x_f32[i, :]` narrowed to the model dtype.
    /// Returns 0 on launch, non-zero if `(hc, dtype)` has no specialisation —
    /// the caller must then use the eager chain.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn hc_y_combine(
        x_f32: *const c_void,
        pre: *const c_void,
        y: *mut c_void,
        hc: i32,
        h: i32,
        total: i64,
        dtype: i32,
        stream: i64,
    ) -> i32;
    /// V4 clamped SwiGLU: `narrow(silu(min(gate, limit)) * clamp(up, -limit, limit))`.
    /// Returns 0 on launch, non-zero if the `(in_dtype, out_dtype)` pair has no
    /// specialisation — the caller must then use the eager chain.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn arc_seam_swiglu_clamp(
        gate: *const c_void,
        up: *const c_void,
        out: *mut c_void,
        limit: f32,
        total: i64,
        in_dtype: i32,
        out_dtype: i32,
        stream: i64,
    ) -> i32;
    /// The clamp half only, for the shared-expert site whose activation is a
    /// fast-math `fused_glu` that a non-fast-math kernel cannot reproduce:
    /// `gate_out = min(gate, limit)`, `up_out = clamp(up, -limit, limit)`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn arc_seam_swiglu_clamp_split(
        gate: *const c_void,
        up: *const c_void,
        gate_out: *mut c_void,
        up_out: *mut c_void,
        limit: f32,
        total: i64,
        in_dtype: i32,
        stream: i64,
    ) -> i32;
    /// MoE expert combine: `out[n, h] = narrow(tree_sum_k(ys[n, j, h] * w[n, j]))`.
    /// Same return contract as above; `k` outside the specialised set refuses.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn arc_seam_moe_weighted_sum(
        ys: *const c_void,
        w: *const c_void,
        out: *mut c_void,
        k: i32,
        h: i32,
        total: i64,
        in_dtype: i32,
        out_dtype: i32,
        stream: i64,
    ) -> i32;
    /// MoE gate weight renormalise + scale. Same return contract as above.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn arc_seam_gate_renorm(
        w: *const c_void,
        out: *mut c_void,
        eps: f32,
        scale: f32,
        do_renorm: i32,
        k: i32,
        n: i64,
        stream: i64,
    ) -> i32;
    /// The whole `hc_post` re-expansion. Same return contract as above.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn hc_post_fused(
        x: *const c_void,
        residual: *const c_void,
        post: *const c_void,
        comb: *const c_void,
        out: *mut c_void,
        hc: i32,
        h: i32,
        total: i64,
        dtype: i32,
        stream: i64,
    ) -> i32;
}
