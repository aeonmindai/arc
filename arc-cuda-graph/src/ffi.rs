//! Complete FFI bindings for CUDA graph and memory pool APIs.

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
pub type CUmemoryPool = *mut std::ffi::c_void;
#[cfg(feature = "cuda")]
pub type CUcontext = *mut std::ffi::c_void;
/// `cuda.h:1944` (CUDA 13.1) —
/// `#define CU_GRAPH_COND_ASSIGN_DEFAULT 0x1` "Default value is applied when
/// graph is launched."
///
/// `cuda.h:21919` documents `defaultLaunchValue` as "Applied at the beginning
/// of each graph execution **if CU_GRAPH_COND_ASSIGN_DEFAULT is set in
/// flags**". Creating the handle with `flags = 0` therefore leaves the
/// condition at 0 on entry and a WHILE body executes **zero** times — the
/// graph launches, returns success, and generates nothing. Measured on
/// `arc-v4-stack` (H200, CUDA 13.1): `flags=0` → body ran 0 times;
/// `flags=CU_GRAPH_COND_ASSIGN_DEFAULT` → body ran exactly N times.
#[cfg(feature = "cuda")]
pub const CU_GRAPH_COND_ASSIGN_DEFAULT: u32 = 0x1;
#[cfg(feature = "cuda")]
pub type CUdevice = i32;

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
}

// Conditional node types (CUDA 12.4+)
#[cfg(feature = "cuda")]
#[repr(u32)]
#[allow(non_camel_case_types, dead_code)]
pub enum CUgraphConditionalNodeType {
    IF = 0,
    WHILE = 1,
}

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

/// `CUDA_CONDITIONAL_NODE_PARAMS`, `cuda.h:1958` (CUDA 13.1).
///
/// `phGraph_out` is an **OUT** field: "CUDA-owned array populated with
/// conditional node child graphs during creation of the node." Callers leave
/// it null and read it back after `cuGraphAddNode`; assigning a caller-owned
/// pointer to it accomplishes nothing because the driver overwrites it.
///
/// `ctx` (the 5th field) was missing entirely from this struct.
#[cfg(feature = "cuda")]
#[repr(C)]
pub struct CudaConditionalNodeParams {
    pub handle: CUgraphConditionalHandle,
    pub cond_type: CUgraphConditionalNodeType,
    pub size: u32,
    pub body_graph_out: *mut CUgraph,
    pub ctx: CUcontext,
}

#[cfg(feature = "cuda")]
#[repr(C)]
pub struct CudaGraphNodeParams {
    pub node_type: CudaGraphNodeType,
    pub _reserved0: [u32; 3],
    pub conditional: CudaConditionalNodeParams,
    pub _pad: [u8; 3968],
}

// Memory pool types
#[cfg(feature = "cuda")]
#[repr(u32)]
#[allow(non_camel_case_types, dead_code)]
pub enum CUmemAllocationType {
    INVALID = 0,
    PINNED = 1,
}

#[cfg(feature = "cuda")]
#[repr(u32)]
#[allow(non_camel_case_types, dead_code)]
pub enum CUmemHandleType {
    NONE = 0,
}

#[cfg(feature = "cuda")]
#[repr(u32)]
#[allow(non_camel_case_types, dead_code)]
pub enum CUmemLocationType {
    INVALID = 0,
    DEVICE = 1,
}

#[cfg(feature = "cuda")]
#[repr(u32)]
#[allow(non_camel_case_types, dead_code)]
pub enum CUmempoolAttribute {
    REUSE_FOLLOW_EVENT_DEPENDENCIES = 1,
    REUSE_ALLOW_OPPORTUNISTIC = 2,
    REUSE_ALLOW_INTERNAL_DEPENDENCIES = 3,
    RELEASE_THRESHOLD = 4,
}

#[cfg(feature = "cuda")]
#[repr(C)]
pub struct CUmemLocation {
    pub loc_type: CUmemLocationType,
    pub id: i32,
}

#[cfg(feature = "cuda")]
#[repr(C)]
pub struct CUmemPoolProps {
    pub alloc_type: CUmemAllocationType,
    pub handle_type: CUmemHandleType,
    pub location: CUmemLocation,
    pub win32_security_attributes: *mut std::ffi::c_void,
    pub max_size: usize,
    pub usage: u16,
    pub reserved: [u8; 54],
}

#[cfg(feature = "cuda")]
extern "C" {
    // ========== Graph lifecycle ==========
    pub fn cuGraphCreate(graph: *mut CUgraph, flags: u32) -> u32;
    pub fn cuStreamBeginCapture_v2(stream: CUstream, mode: CUstreamCaptureMode) -> u32;
    pub fn cuStreamEndCapture(stream: CUstream, graph: *mut CUgraph) -> u32;
    pub fn cuGraphInstantiate_v2(
        exec: *mut CUgraphExec,
        graph: CUgraph,
        error_node: *mut CUgraphNode,
        log_buffer: *mut u8,
        buffer_size: usize,
    ) -> u32;
    // RUN-161 2b: instantiate with flags. AUTO_FREE_ON_LAUNCH (=1) frees any
    // graph-owned memory allocations at the start of each launch, so a graph
    // that contains memory-alloc nodes can be RE-launched (replayed). Without
    // it, the 2nd cuGraphLaunch returns CUDA_ERROR_INVALID_VALUE (=1).
    pub fn cuGraphInstantiateWithFlags(exec: *mut CUgraphExec, graph: CUgraph, flags: u64) -> u32;
    pub fn cuGraphLaunch(exec: CUgraphExec, stream: CUstream) -> u32;
    pub fn cuGraphExecDestroy(exec: CUgraphExec) -> u32;
    pub fn cuGraphDestroy(graph: CUgraph) -> u32;
    pub fn cuGraphGetNodes(graph: CUgraph, nodes: *mut CUgraphNode, num_nodes: *mut usize) -> u32;

    // ========== Memory pools ==========
    pub fn cuMemPoolCreate(pool: *mut CUmemoryPool, props: *const CUmemPoolProps) -> u32;
    pub fn cuMemPoolDestroy(pool: CUmemoryPool) -> u32;
    pub fn cuMemPoolSetAttribute(
        pool: CUmemoryPool,
        attr: CUmempoolAttribute,
        value: *mut std::ffi::c_void,
    ) -> u32;
    pub fn cuDeviceGetMemPool(pool: *mut CUmemoryPool, dev: CUdevice) -> u32;
    pub fn cuDeviceSetMemPool(dev: CUdevice, pool: CUmemoryPool) -> u32;
    pub fn cuDeviceGetDefaultMemPool(pool: *mut CUmemoryPool, dev: CUdevice) -> u32;

    // ========== Stream / sync ==========
    pub fn cuStreamCreate(stream: *mut CUstream, flags: u32) -> u32;
    pub fn cuStreamDestroy_v2(stream: CUstream) -> u32;
    pub fn cudaStreamSynchronize(stream: CUstream) -> u32;

    // ========== Conditional nodes (CUDA 12.4+) ==========
    /// `cuda.h:21932` (CUDA 13.1) — the driver form takes a `CUcontext`,
    /// which the conditional node params must match ("Context on which to run
    /// the node. Must match context used to create the handle and all body
    /// nodes", `cuda.h:1986`).
    pub fn cuGraphConditionalHandleCreate(
        handle: *mut CUgraphConditionalHandle,
        graph: CUgraph,
        ctx: CUcontext,
        default_launch_value: u32,
        flags: u32,
    ) -> u32;
    pub fn cuCtxGetCurrent(ctx: *mut CUcontext) -> u32;
    /// `cuda.h:21829` (CUDA 13.1):
    /// ```text
    /// CUresult cuGraphAddNode(CUgraphNode *phGraphNode, CUgraph hGraph,
    ///                         const CUgraphNode *dependencies,
    ///                         const CUgraphEdgeData *dependencyData,
    ///                         size_t numDependencies,
    ///                         CUgraphNodeParams *nodeParams);
    /// ```
    /// The OUT node pointer is the **first** argument and there are **six** of
    /// them. This was previously declared as
    /// `cudaGraphAddNode(graph, node_out, deps, num_deps, params)` — five
    /// arguments with the first two transposed — so the driver received the
    /// `CUgraph` handle in the slot where it expected `CUgraphNode *` and
    /// wrote the new node handle through that value, while `numDependencies`
    /// received a pointer and `nodeParams` an uninitialised register.
    pub fn cuGraphAddNode(
        node_out: *mut CUgraphNode,
        graph: CUgraph,
        dependencies: *const CUgraphNode,
        dependency_data: *const std::ffi::c_void,
        num_dependencies: usize,
        params: *mut CudaGraphNodeParams,
    ) -> u32;
    /// `cuda.h:15665` (CUDA 13.1). `cuda.h:1976` names this as a supported way
    /// to populate a conditional node's body graph. Ordinary
    /// `cuStreamBeginCapture_v2` always creates a **new** graph, so a body
    /// captured that way cannot be attached to the conditional node.
    pub fn cuStreamBeginCaptureToGraph(
        stream: CUstream,
        graph: CUgraph,
        dependencies: *const CUgraphNode,
        dependency_data: *const std::ffi::c_void,
        num_dependencies: usize,
        mode: CUstreamCaptureMode,
    ) -> u32;

    // ========== Pinned host memory ==========
    pub fn cudaHostAlloc(ptr: *mut *mut std::ffi::c_void, size: usize, flags: u32) -> u32;
    pub fn cudaFreeHost(ptr: *mut std::ffi::c_void) -> u32;

    // ========== Decode kernels ==========
    pub fn launch_gather_rope_decode_bf16(
        q: *mut std::ffi::c_void,
        k: *mut std::ffi::c_void,
        cos_table: *const std::ffi::c_void,
        sin_table: *const std::ffi::c_void,
        positions: *const i32,
        num_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        rot_dim: i32,
        cos_stride: i32,
        batch_size: i32,
        is_neox: i32,
        stream: CUstream,
    );
    pub fn launch_gather_rope_decode_f16(
        q: *mut std::ffi::c_void,
        k: *mut std::ffi::c_void,
        cos_table: *const std::ffi::c_void,
        sin_table: *const std::ffi::c_void,
        positions: *const i32,
        num_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        rot_dim: i32,
        cos_stride: i32,
        batch_size: i32,
        is_neox: i32,
        stream: CUstream,
    );
    pub fn launch_fused_argmax_bf16(
        logits: *const std::ffi::c_void,
        token_ids: *mut i32,
        log_probs: *mut f32,
        vocab_size: i32,
        batch_size: i32,
        stream: CUstream,
    );
    pub fn launch_fused_top_p_bf16(
        logits: *const std::ffi::c_void,
        token_ids: *mut i32,
        temperature: f32,
        top_p: f32,
        vocab_size: i32,
        batch_size: i32,
        rng_seed: u64,
        rng_offset: u64,
        stream: CUstream,
    );
    pub fn launch_apply_penalties(
        logits: *mut std::ffi::c_void,
        generated_tokens: *const i32,
        n_generated: *const i32,
        frequency_penalty: f32,
        presence_penalty: f32,
        vocab_size: i32,
        max_tokens: i32,
        batch_size: i32,
        stream: CUstream,
    );
    pub fn launch_decode_step_update(
        sampled_tokens: *const i32,
        input_ids: *mut i32,
        positions: *mut i32,
        context_lens: *mut i32,
        slot_mappings: *mut i64,
        block_tables: *const i32,
        n_generated: *mut i32,
        output_tokens: *mut i32,
        finished: *mut i32,
        ring_buffer: *mut i32,
        ring_write_head: *mut i32,
        eos_token_id: i32,
        max_tokens: i32,
        block_size: i32,
        max_blocks_per_seq: i32,
        ring_size: i32,
        loop_condition: *mut i32,
        batch_size: i32,
        stream: CUstream,
    );
    pub fn launch_check_all_done(
        finished: *const i32,
        loop_condition: *mut i32,
        batch_size: i32,
        stream: CUstream,
    );
    pub fn launch_check_all_done_conditional(
        finished: *const i32,
        batch_size: i32,
        cond_handle: CUgraphConditionalHandle,
        stream: CUstream,
    );
    pub fn arc_has_graph_conditional() -> i32;
}
