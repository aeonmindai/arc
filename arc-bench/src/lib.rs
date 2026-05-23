//! Arc benchmark harness.
//!
//! `arc-bench` provides:
//!
//! 1. A reusable JSON schema for **AA-AgentPerf trajectories** —
//!    multi-turn agentic-coding conversations with tool calls, captured
//!    from real public-repo sessions (see
//!    `datasets/agentperf_tuning/README.md` for provenance).
//! 2. A **multi-turn replay engine** that drives any inference vendor
//!    (Arc, vLLM, sglang, ...) through a trajectory and reports per-turn
//!    TTFT / output-speed / token counts. Tool execution is replaced by
//!    injection of the recorded tool responses, so the replay is
//!    deterministic.
//! 3. A **dataset loader** that walks a directory of trajectory JSON
//!    files and yields them as an iterable, sorted-by-id collection.
//!
//! RUN-194 wires this into a TUI + scheduler that benchmarks Arc
//! against rival frameworks.

pub mod dataset;
pub mod dataset_authoring;
pub mod replay;
pub mod trajectory;

pub use dataset::{Dataset, DatasetError, LengthDistribution};
pub use replay::{
    ChatMessage, ReplayDiagnostic, ReplayResult, TrajectoryReplayer, TurnResult, Vendor,
    VendorResponse,
};
pub use trajectory::{
    GeneratedWith, ParseError, Role, Source, ToolCall, Trajectory, Turn,
};

/// Canonical path of the bundled tuning dataset relative to the crate
/// root.
pub const TUNING_DATASET_PATH: &str = "datasets/agentperf_tuning";
