//! The JSON schema. This is the durable artifact; the HTML is a *view* of it.
//!
//! Everything the HTML can show must be reconstructible from this file alone —
//! no back-references into the process that produced it. That is deliberate:
//! a profile taken on a rented H200 has to stay readable months after the box
//! is deleted.

use serde::{Deserialize, Serialize};

/// Bumped whenever a field changes meaning. The HTML refuses to render an
/// unknown major.
pub const SCHEMA: &str = "arc-profile/1";

/// What kind of time a node is allowed to claim.
///
/// This distinction is the whole point of the profiler. A `Host` node's wall
/// clock is a CPU cost; a `Device` node's wall clock is a *launch* cost and its
/// GPU cost lives in `device_ns`; a `Sync` node's wall clock is the host
/// *waiting*, which is neither. Collapsing them is exactly the failure mode
/// that made every previous profile of this engine unactionable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NodeKind {
    /// The synthetic root of one run.
    Root,
    /// Host-side work. `device_ns` is always 0.
    Host,
    /// Brackets GPU work with CUDA events. `wall_ns` is the *launch* cost,
    /// `device_ns` is the measured GPU execution.
    Device,
    /// The host blocked on the device (an explicit `synchronize`, a D2H copy,
    /// a `to_vec`). `wall_ns == sync_ns`.
    Sync,
}

impl NodeKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Root => "root",
            Self::Host => "host",
            Self::Device => "device",
            Self::Sync => "sync",
        }
    }
}

/// Batch geometry observed at a node. Recorded per node because a profile
/// without it cannot be attributed to a batch size after the fact — the mistake
/// `ARC_TIME_DECODE` made for a month (see `deepseek4.rs::emit_decode_profile`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Geometry {
    /// Batch dimension (sequences in the step).
    pub b: u32,
    /// Time dimension (tokens per sequence in this forward).
    pub t: u32,
    /// `b * t`, carried explicitly so a consumer never has to guess.
    pub tokens: u64,
}

/// One aggregated node of the span tree.
///
/// Aggregated, not per-instance: a 43-layer forward produces ONE `layer` node
/// with `calls = 43`, plus min/max so layer-to-layer variance is still visible.
/// `ARC_PROFILE_UNROLL=1` switches to one node per layer index when that is not
/// enough.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    pub id: u32,
    pub parent: Option<u32>,
    /// Leaf name, e.g. `mla_attn`.
    pub name: String,
    /// Full dotted path from the root, e.g. `step.forward.layer.mla_attn`.
    pub path: String,
    pub depth: u32,
    pub kind: NodeKind,
    /// How many times this span opened across the whole run.
    pub calls: u64,

    /// Total wall time inside the span, summed over calls.
    pub wall_ns: u64,
    /// `wall_ns` minus the sum of children's `wall_ns`. Can go slightly
    /// negative on timer noise; clamped at 0 and reported in
    /// [`Reconciliation`] when it does.
    pub wall_self_ns: u64,
    /// GPU time measured with CUDA events. 0 for host/sync nodes.
    pub device_ns: u64,
    /// `device_ns` minus children's `device_ns`.
    pub device_self_ns: u64,
    /// Wall time this node spent blocked on the device.
    pub sync_ns: u64,
    /// `wall_self_ns - sync_ns`: host time genuinely spent computing here.
    pub busy_self_ns: u64,

    pub min_wall_ns: u64,
    pub max_wall_ns: u64,

    pub geom: Geometry,

    /// Name of the ancestor that is a direct child of the root — on this engine
    /// `prompt` or `decode`. `None` for the root itself.
    ///
    /// # Why this is a field and not something you parse out of `path`
    ///
    /// The step tree splits prefill and decode into separate subtrees on
    /// purpose, so **every span name below the split exists twice**. Selecting a
    /// span by NAME therefore has two answers, and the obvious
    /// `nodes.iter().find(|n| n.name == "mla_attn")` returns whichever comes
    /// first — the prefill node, which has `calls == 0` in a decode-only window.
    /// A consumer then reads zero for attention and concludes the kernel is
    /// free.
    ///
    /// That has happened. Use [`Profile::resolve_in`] rather than matching on
    /// `name`; [`Profile::resolve`] refuses outright when a name is ambiguous.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub branch: Option<String>,

    /// `false` when the code path exists but is provably not taken in this
    /// configuration. A zero from an unreached node and a zero from a fast
    /// node are different answers and must not look the same.
    pub reachable: bool,
    /// Why it is unreachable, or any other caveat worth carrying to the reader.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,

    pub children: Vec<u32>,
}

/// Something the profiler was asked to measure but which does not execute in
/// this configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Unreachable {
    pub path: String,
    pub reason: String,
    /// `file.rs:LINE` of the condition that bails, so the claim is checkable.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub site: Option<String>,
}

/// A parent whose children claim more time than it does. Always emitted, even
/// when empty, so a reader can tell "checked and clean" from "never checked".
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Violation {
    pub path: String,
    pub parent_wall_ns: u64,
    pub children_wall_ns: u64,
    pub excess_pct: f64,
    pub channel: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Reconciliation {
    /// Children may exceed a parent by this much before it counts as a
    /// violation. Timer granularity and the guard's own cost make an exact
    /// inequality untestable.
    pub tolerance_pct: f64,
    pub violations: Vec<Violation>,
    /// Spans that closed on a different thread, or out of order. Non-zero here
    /// means the tree shape is not trustworthy and the reader should know.
    pub misnested_spans: u64,
    /// Device spans whose CUDA events never resolved (e.g. the run ended
    /// mid-step). Their `device_ns` is missing, not zero.
    pub unresolved_device_spans: u64,
}

/// Measured cost of the profiler itself, both states. If `enabled_overhead_pct`
/// is large the profile is partly measuring itself and says so.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Overhead {
    /// ns per span open+close with the profiler ON.
    pub enabled_ns_per_span: f64,
    /// ns per gate check with the profiler OFF (the cost every shipped binary
    /// pays).
    pub disabled_ns_per_span: f64,
    /// Spans opened per step, measured.
    pub spans_per_step: f64,
    /// `enabled_ns_per_span * spans_per_step / step_wall_ns`.
    pub enabled_overhead_pct: f64,
}

/// Provenance. Without this a JSON file is an orphan number.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RunHeader {
    pub started_unix_ms: u64,
    pub label: String,
    pub commit: String,
    pub model: String,
    pub artifact: String,
    pub architecture: String,
    pub device: String,
    pub gpu_name: String,
    pub driver: String,
    pub cuda_runtime: String,
    pub host: String,
    pub build_features: Vec<String>,
    /// Requested batch size (`--max-seqs` / concurrency), which is NOT the same
    /// as the batch actually scheduled — see `geom.b` on the step node.
    pub requested_batch: u32,
    pub profile_depth: u32,
    pub warmup_steps: u32,
    pub steps: u64,
    pub tokens: u64,
    pub unroll_layers: bool,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Totals {
    pub wall_ns: u64,
    pub device_ns: u64,
    pub sync_ns: u64,
    pub busy_host_ns: u64,
    pub steps: u64,
    pub tokens: u64,
}

/// The whole artifact.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Profile {
    pub schema: String,
    pub run: RunHeader,
    pub totals: Totals,
    pub overhead: Overhead,
    pub nodes: Vec<Node>,
    pub unreachable: Vec<Unreachable>,
    pub reconciliation: Reconciliation,
}

/// What a node's numbers actually mean — the three ways a zero happens, kept
/// apart.
///
/// Two independent chains hand-rolled this distinction in one night before
/// either found it expressed in the schema, so it is now one call.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Verdict {
    /// `reachable == false`: the code path exists and is provably not taken in
    /// this configuration. Its zero is a structural fact.
    Unreachable,
    /// Reachable, registered, but never entered in the recorded window — e.g.
    /// the `prompt` subtree in a window that contained only decode steps. Its
    /// zero is missing data, **not** a measurement of zero, and it is the shape
    /// that masquerades as "this is free".
    NotEnteredThisWindow,
    /// Entered, but every call landed below the timer's resolution.
    BelowTimerFloor,
    /// Entered and timed.
    Measured,
}

impl Node {
    /// Which of the three zeros — or a real measurement — this node carries.
    ///
    /// Prefer this over testing `calls == 0` or `wall_ns == 0` by hand: those
    /// two tests answer different questions and neither answers "did this run".
    pub fn verdict(&self) -> Verdict {
        if !self.reachable {
            Verdict::Unreachable
        } else if self.calls == 0 {
            Verdict::NotEnteredThisWindow
        } else if self.wall_ns == 0 && self.device_ns == 0 {
            Verdict::BelowTimerFloor
        } else {
            Verdict::Measured
        }
    }
}

/// A span name that does not identify one node.
#[derive(Debug, Clone)]
pub struct Ambiguity {
    pub name: String,
    /// `(branch, path, calls)` for every node carrying the name.
    pub candidates: Vec<(Option<String>, String, u64)>,
}

impl std::fmt::Display for Ambiguity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.candidates.is_empty() {
            return write!(f, "no span named `{}` in this profile", self.name);
        }
        writeln!(
            f,
            "span name `{}` does not identify one node — it appears in {} places:",
            self.name,
            self.candidates.len()
        )?;
        for (branch, path, calls) in &self.candidates {
            writeln!(
                f,
                "  [{}] {}  calls={}",
                branch.as_deref().unwrap_or("-"),
                path,
                calls
            )?;
        }
        write!(
            f,
            "Select by full path (`Profile::node`) or by branch \
             (`Profile::resolve_in(\"decode\", ..)`). Matching on `name` alone returns \
             whichever node was registered first, which is the prefill one — and its \
             `calls` is 0 in a decode-only window."
        )
    }
}

impl std::error::Error for Ambiguity {}

impl Profile {
    /// Look a node up by its dotted path. Used by every test in this crate and
    /// by anyone reading a profile programmatically.
    pub fn node(&self, path: &str) -> Option<&Node> {
        self.nodes.iter().find(|n| n.path == path)
    }

    /// Every node carrying `name`, in registration order.
    ///
    /// Returns a `Vec` rather than an `Option` because the honest answer to
    /// "where is `mla_attn`" is usually *two* places.
    pub fn nodes_named(&self, name: &str) -> Vec<&Node> {
        self.nodes.iter().filter(|n| n.name == name).collect()
    }

    /// The one node named `name` — or an error naming every candidate.
    ///
    /// This is the call that refuses to guess. `resolve("mla_attn")` on a
    /// profile containing both a prefill and a decode step **fails**, and the
    /// message tells the caller which branch to ask for and what each one's
    /// `calls` is.
    pub fn resolve(&self, name: &str) -> Result<&Node, Ambiguity> {
        let found = self.nodes_named(name);
        match found.len() {
            1 => Ok(found[0]),
            _ => Err(self.ambiguity(name, &found)),
        }
    }

    /// The one node named `name` under the top-level branch `branch`
    /// (`"decode"` / `"prompt"` on this engine).
    pub fn resolve_in(&self, branch: &str, name: &str) -> Result<&Node, Ambiguity> {
        let found: Vec<&Node> = self
            .nodes
            .iter()
            .filter(|n| n.name == name && n.branch.as_deref() == Some(branch))
            .collect();
        match found.len() {
            1 => Ok(found[0]),
            _ => Err(self.ambiguity(name, &found)),
        }
    }

    fn ambiguity(&self, name: &str, found: &[&Node]) -> Ambiguity {
        let from = if found.is_empty() {
            self.nodes_named(name)
        } else {
            found.to_vec()
        };
        Ambiguity {
            name: name.to_string(),
            candidates: from
                .iter()
                .map(|n| (n.branch.clone(), n.path.clone(), n.calls))
                .collect(),
        }
    }

    /// Span names that appear on more than one node, with their paths.
    ///
    /// Surfaced in `run.notes` so a reader is told the hazard exists before
    /// they write a name match, rather than after they publish a zero.
    pub fn duplicate_names(&self) -> Vec<(String, Vec<String>)> {
        let mut seen: Vec<(String, Vec<String>)> = Vec::new();
        for n in &self.nodes {
            match seen.iter_mut().find(|(name, _)| *name == n.name) {
                Some((_, paths)) => paths.push(n.path.clone()),
                None => seen.push((n.name.clone(), vec![n.path.clone()])),
            }
        }
        seen.retain(|(_, paths)| paths.len() > 1);
        seen
    }

    pub fn root(&self) -> Option<&Node> {
        self.nodes.iter().find(|n| n.parent.is_none())
    }

    /// Re-derive the reconciliation check from the node table alone.
    ///
    /// Deliberately independent of the accumulator that produced the file: a
    /// consumer can re-run the check on a JSON someone hands them, and the
    /// crate's own tests use this path rather than trusting the writer.
    pub fn recheck(&self, tolerance_pct: f64) -> Vec<Violation> {
        let mut out = Vec::new();
        for n in &self.nodes {
            for (channel, parent, child_sum) in [
                (
                    "wall",
                    n.wall_ns,
                    n.children
                        .iter()
                        .filter_map(|c| self.nodes.iter().find(|x| x.id == *c))
                        .map(|c| c.wall_ns)
                        .sum::<u64>(),
                ),
                (
                    "device",
                    n.device_ns,
                    n.children
                        .iter()
                        .filter_map(|c| self.nodes.iter().find(|x| x.id == *c))
                        .map(|c| c.device_ns)
                        .sum::<u64>(),
                ),
            ] {
                if child_sum == 0 {
                    continue;
                }
                let excess = child_sum as f64 - parent as f64;
                let pct = 100.0 * excess / (parent.max(1) as f64);
                if pct > tolerance_pct {
                    out.push(Violation {
                        path: n.path.clone(),
                        parent_wall_ns: parent,
                        children_wall_ns: child_sum,
                        excess_pct: pct,
                        channel: channel.to_string(),
                    });
                }
            }
        }
        out
    }
}
