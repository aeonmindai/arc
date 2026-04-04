//! Complete FFI bindings for CUDA graph APIs including 12.4+ conditional nodes.

#[cfg(feature = "cuda")]
use candle_core::cuda::cudarc::driver::sys::CUstream;

#[cfg(feature = "cuda")]
pub type CUgraph = *mut std::ffi::c_void;
#[cfg(feature = "cuda")]
pub type CUgraphExec = *mut std::ffi::c_void;
#[cfg(feature = "cuda")]
pub type CUgraphNode = *mut std::ffi::c_void;
#[cfg(feature = "cuda")]
pub type CUgraphConditionalHandle = u64;

#[cfg(feature = "cuda")]
pub const CUDA_SUCCESS: u32 = 0;
#[cfg(feature = "cuda")]
pub const CUDA_HOST_ALLOC_MAPPED: u32 = 0x02;
#[cfg(feature = "cuda")]
pub const CU_STREAM_NON_BLOCKING: u32 = 0x01;

#[cfg(feature = "cuda")]
#[repr(u32)]
#[allow(non_camel_case_types, dead_code)]
pub enum CUstreamCaptureMode {
    GLOBAL = 0,
    THREAD_LOCAL = 1,
    RELAXED = 2,
}

#[cfg(feature = "cuda")]
#[repr(u32)]
#[allow(non_camel_case_types, dead_code)]
pub enum CUgraphExecUpdateResult {
    SUCCESS = 0,
    ERROR = 1,
    ERROR_TOPOLOGY_CHANGED = 2,
    ERROR_NODE_TYPE_CHANGED = 3,
    ERROR_FUNCTION_CHANGED = 4,
    ERROR_PARAMETERS_CHANGED = 5,
    ERROR_NOT_SUPPORTED = 6,
}

#[cfg(feature = "cuda")]
#[repr(u32)]
#[allow(non_camel_case_types, dead_code)]
pub enum CUgraphConditionalNodeType {
    IF = 0,
    WHILE = 1,
}

// cudaGraphNodeType for cudaGraphAddNode
#[cfg(feature = "cuda")]
#[repr(u32)]
#[allow(non_camel_case_types, dead_code)]
pub enum CudaGraphNodeType {
    Kernel = 0,
    Memcpy = 1,
    Memset = 2,
    Host = 3,
    Graph = 4,
    Empty = 5,
    WaitEvent = 6,
    EventRecord = 7,
    ExtSemSignal = 8,
    ExtSemWait = 9,
    MemAlloc = 10,
    MemFree = 11,
    BatchMemOp = 12,
    Conditional = 13,
}

/// Parameters for cudaGraphAddNode with conditional node type.
/// This is a simplified representation of cudaGraphNodeParams for the conditional case.
/// The full struct is a union — we only need the conditional variant.
#[cfg(feature = "cuda")]
#[repr(C)]
pub struct CudaConditionalNodeParams {
    pub handle: CUgraphConditionalHandle,
    pub cond_type: CUgraphConditionalNodeType,
    pub size: u32,
    pub body_graph_out: *mut CUgraph, // output: body graph to populate via stream capture
}

/// Simplified cudaGraphNodeParams — only the fields we use.
/// The real struct is a 4KB union. We use the conditional path only.
/// Pad to match the ABI size.
#[cfg(feature = "cuda")]
#[repr(C)]
pub struct CudaGraphNodeParams {
    pub node_type: CudaGraphNodeType,
    pub _reserved0: [u32; 3],
    pub conditional: CudaConditionalNodeParams,
    pub _pad: [u8; 3968], // pad to ~4KB to match real struct size
}

#[cfg(feature = "cuda")]
extern "C" {
    // ========================================================================
    // Graph lifecycle
    // ========================================================================
    pub fn cuGraphCreate(graph: *mut CUgraph, flags: u32) -> u32;
    pub fn cuGraphDestroy(graph: CUgraph) -> u32;

    // ========================================================================
    // Stream capture
    // ========================================================================
    pub fn cuStreamBeginCapture_v2(stream: CUstream, mode: CUstreamCaptureMode) -> u32;
    pub fn cuStreamEndCapture(stream: CUstream, graph: *mut CUgraph) -> u32;

    // ========================================================================
    // Graph instantiation and execution
    // ========================================================================
    pub fn cuGraphInstantiate_v2(
        exec: *mut CUgraphExec,
        graph: CUgraph,
        error_node: *mut CUgraphNode,
        log_buffer: *mut u8,
        buffer_size: usize,
    ) -> u32;

    pub fn cuGraphExecUpdate_v2(
        exec: CUgraphExec,
        graph: CUgraph,
        result_info: *mut CUgraphExecUpdateResult,
    ) -> u32;

    pub fn cuGraphLaunch(exec: CUgraphExec, stream: CUstream) -> u32;
    pub fn cuGraphExecDestroy(exec: CUgraphExec) -> u32;

    // ========================================================================
    // Conditional nodes (CUDA 12.4+)
    // ========================================================================
    pub fn cudaGraphConditionalHandleCreate(
        handle: *mut CUgraphConditionalHandle,
        graph: CUgraph,
        default_value: u32,
        flags: u32,
    ) -> u32;

    pub fn cudaGraphSetConditional(handle: CUgraphConditionalHandle, value: u32) -> u32;

    /// Add a node to a graph. For conditional nodes, params.node_type = Conditional
    /// and params.conditional contains the WHILE/IF configuration.
    /// The body graph is returned in params.conditional.body_graph_out.
    pub fn cudaGraphAddNode(
        graph: CUgraph,
        node_out: *mut CUgraphNode,
        dependencies: *const CUgraphNode,
        num_dependencies: usize,
        params: *mut CudaGraphNodeParams,
    ) -> u32;

    // ========================================================================
    // Stream creation and management
    // ========================================================================
    pub fn cuStreamCreate(stream: *mut CUstream, flags: u32) -> u32;
    pub fn cuStreamDestroy_v2(stream: CUstream) -> u32;
    pub fn cudaStreamSynchronize(stream: CUstream) -> u32;

    pub fn cudaHostAlloc(
        ptr: *mut *mut std::ffi::c_void,
        size: usize,
        flags: u32,
    ) -> u32;

    pub fn cudaFreeHost(ptr: *mut std::ffi::c_void) -> u32;

    // ========================================================================
    // Decode kernels
    // ========================================================================
    pub fn launch_gather_rope_decode_bf16(
        q: *mut std::ffi::c_void, k: *mut std::ffi::c_void,
        cos_table: *const std::ffi::c_void, sin_table: *const std::ffi::c_void,
        positions: *const i32,
        num_heads: i32, num_kv_heads: i32, head_dim: i32, rot_dim: i32, cos_stride: i32,
        batch_size: i32, is_neox: i32, stream: CUstream,
    );

    pub fn launch_gather_rope_decode_f16(
        q: *mut std::ffi::c_void, k: *mut std::ffi::c_void,
        cos_table: *const std::ffi::c_void, sin_table: *const std::ffi::c_void,
        positions: *const i32,
        num_heads: i32, num_kv_heads: i32, head_dim: i32, rot_dim: i32, cos_stride: i32,
        batch_size: i32, is_neox: i32, stream: CUstream,
    );

    pub fn launch_fused_argmax_bf16(
        logits: *const std::ffi::c_void, token_ids: *mut i32, log_probs: *mut f32,
        vocab_size: i32, batch_size: i32, stream: CUstream,
    );

    pub fn launch_fused_top_p_bf16(
        logits: *const std::ffi::c_void, token_ids: *mut i32,
        temperature: f32, top_p: f32, vocab_size: i32, batch_size: i32,
        rng_seed: u64, rng_offset: u64, stream: CUstream,
    );

    pub fn launch_apply_penalties(
        logits: *mut std::ffi::c_void,
        generated_tokens: *const i32, n_generated: *const i32,
        frequency_penalty: f32, presence_penalty: f32,
        vocab_size: i32, max_tokens: i32, batch_size: i32, stream: CUstream,
    );

    pub fn launch_decode_step_update(
        sampled_tokens: *const i32,
        input_ids: *mut i32, positions: *mut i32,
        context_lens: *mut i32, slot_mappings: *mut i64,
        block_tables: *const i32,
        n_generated: *mut i32, output_tokens: *mut i32, finished: *mut i32,
        ring_buffer: *mut i32, ring_write_head: *mut i32,
        eos_token_id: i32, max_tokens: i32,
        block_size: i32, max_blocks_per_seq: i32, ring_size: i32,
        loop_condition: *mut i32,
        batch_size: i32, stream: CUstream,
    );

    pub fn launch_check_all_done(
        finished: *const i32, loop_condition: *mut i32,
        batch_size: i32, stream: CUstream,
    );

    /// Only available when compiled with CUDA 12.4+ (ARC_HAS_GRAPH_CONDITIONAL=1).
    /// Calls cudaGraphSetConditional from device code to control the WHILE loop.
    pub fn launch_check_all_done_conditional(
        finished: *const i32, batch_size: i32,
        cond_handle: CUgraphConditionalHandle,
        stream: CUstream,
    );

    /// Returns 1 if CUDA 12.4+ conditional graph API was compiled in, 0 otherwise.
    pub fn arc_has_graph_conditional() -> i32;
}
