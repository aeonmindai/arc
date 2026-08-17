//! The span tree accumulator.
//!
//! Nodes are keyed by `(parent, name)`, so a 43-layer forward collapses into a
//! single `layer` node with `calls = 43` plus min/max — readable by default,
//! with `ARC_PROFILE_UNROLL=1` to split by index when the aggregate hides
//! something.
//!
//! Counters are atomics behind an [`RwLock`] that is only *write*-locked when a
//! node is first created. After the first step every span open/close is a read
//! lock plus a `fetch_add`, which is what keeps the enabled cost in the tens of
//! nanoseconds rather than the microseconds a `Mutex<HashMap>` would cost.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering::Relaxed};
use std::sync::RwLock;

use crate::report::{Geometry, Node, NodeKind};

pub(crate) struct NodeAcc {
    pub id: u32,
    pub parent: Option<u32>,
    pub name: String,
    pub path: String,
    pub depth: u32,
    pub kind: NodeKind,

    pub calls: AtomicU64,
    pub wall_ns: AtomicU64,
    /// The interval this node's own CUDA events measured. Zero for every node
    /// that is not a [`NodeKind::Device`] span. The rolled-up `device_ns` in
    /// the report is derived from these in [`Registry::snapshot`], never
    /// accumulated separately.
    pub device_measured_ns: AtomicU64,
    pub sync_ns: AtomicU64,
    pub min_wall_ns: AtomicU64,
    pub max_wall_ns: AtomicU64,

    pub geom_b: AtomicU32,
    pub geom_t: AtomicU32,
    pub tokens: AtomicU64,

    pub reachable: AtomicBool,
    pub note: RwLock<Option<String>>,

    /// Only mutated while the registry write lock is held (at child creation).
    pub children: Vec<u32>,
}

impl NodeAcc {
    fn new(
        id: u32,
        parent: Option<u32>,
        name: &str,
        path: String,
        depth: u32,
        kind: NodeKind,
    ) -> Self {
        Self {
            id,
            parent,
            name: name.to_string(),
            path,
            depth,
            kind,
            calls: AtomicU64::new(0),
            wall_ns: AtomicU64::new(0),
            device_measured_ns: AtomicU64::new(0),
            sync_ns: AtomicU64::new(0),
            min_wall_ns: AtomicU64::new(u64::MAX),
            max_wall_ns: AtomicU64::new(0),
            geom_b: AtomicU32::new(0),
            geom_t: AtomicU32::new(0),
            tokens: AtomicU64::new(0),
            reachable: AtomicBool::new(true),
            note: RwLock::new(None),
            children: Vec::new(),
        }
    }

    fn reset(&self) {
        self.calls.store(0, Relaxed);
        self.wall_ns.store(0, Relaxed);
        self.device_measured_ns.store(0, Relaxed);
        self.sync_ns.store(0, Relaxed);
        self.min_wall_ns.store(u64::MAX, Relaxed);
        self.max_wall_ns.store(0, Relaxed);
        self.tokens.store(0, Relaxed);
    }
}

#[derive(Default)]
pub(crate) struct Registry {
    pub nodes: Vec<NodeAcc>,
    /// `parent -> name -> id`, nested rather than a `(u32, String)` tuple key
    /// so the hot-path lookup can borrow the name as `&str` and allocate
    /// nothing. A tuple key forces a `String` allocation on **every** span
    /// open, which would put the enabled cost in microseconds.
    index: HashMap<u32, HashMap<String, u32>>,
}

/// Sentinel parent for depth-0 nodes. `u32::MAX` cannot collide with a real id
/// because the registry would have to hold 4 billion nodes first.
pub(crate) const NO_PARENT: u32 = u32::MAX;

impl Registry {
    #[inline]
    pub fn lookup(&self, parent: u32, name: &str) -> Option<u32> {
        self.index.get(&parent)?.get(name).copied()
    }

    pub fn get_or_create(&mut self, parent: u32, name: &str, kind: NodeKind) -> u32 {
        if let Some(id) = self.lookup(parent, name) {
            return id;
        }
        let id = self.nodes.len() as u32;
        let (parent_opt, depth, path) = if parent == NO_PARENT {
            (None, 0, name.to_string())
        } else {
            let p = &self.nodes[parent as usize];
            (Some(parent), p.depth + 1, format!("{}.{}", p.path, name))
        };
        self.nodes
            .push(NodeAcc::new(id, parent_opt, name, path, depth, kind));
        if let Some(p) = parent_opt {
            self.nodes[p as usize].children.push(id);
        }
        self.index
            .entry(parent)
            .or_default()
            .insert(name.to_string(), id);
        id
    }

    pub fn reset_counters(&self) {
        for n in &self.nodes {
            n.reset();
        }
    }

    /// Freeze the accumulator into the serialisable node table.
    ///
    /// Self-times are derived here, in one place, from the parent/child links —
    /// never accumulated separately. A separately-accumulated self-time can
    /// drift from the tree it claims to describe, and then the reconciliation
    /// check is checking the wrong thing.
    ///
    /// Device time is **rolled up**: only `Device` spans measure an interval,
    /// so a host parent's device column is the sum of its children's. Without
    /// this the device column would be zero at every interior node and the root
    /// would report no GPU time at all while its leaves reported milliseconds.
    pub fn snapshot(&self) -> Vec<Node> {
        // Children always have a higher id than their parent (they can only be
        // created from inside the parent's span), so one reverse pass rolls the
        // device column all the way to the root.
        let mut rolled = vec![0u64; self.nodes.len()];
        for n in self.nodes.iter().rev() {
            let child_sum: u64 = n.children.iter().map(|c| rolled[*c as usize]).sum();
            rolled[n.id as usize] = match n.kind {
                NodeKind::Device => {
                    let measured = n.device_measured_ns.load(Relaxed);
                    // A device span whose events never resolved contributes its
                    // children rather than a zero that would read as "the GPU
                    // did nothing here".
                    if measured == 0 {
                        child_sum
                    } else {
                        measured
                    }
                }
                _ => child_sum,
            };
        }

        let mut out: Vec<Node> = Vec::with_capacity(self.nodes.len());
        for n in &self.nodes {
            let wall = n.wall_ns.load(Relaxed);
            let dev = rolled[n.id as usize];
            let sync = n.sync_ns.load(Relaxed);
            let child_wall: u64 = n
                .children
                .iter()
                .map(|c| self.nodes[*c as usize].wall_ns.load(Relaxed))
                .sum();
            let child_dev: u64 = n.children.iter().map(|c| rolled[*c as usize]).sum();
            let wall_self = wall.saturating_sub(child_wall);
            let dev_self = dev.saturating_sub(child_dev);
            let min = n.min_wall_ns.load(Relaxed);
            out.push(Node {
                id: n.id,
                parent: n.parent,
                name: n.name.clone(),
                path: n.path.clone(),
                depth: n.depth,
                kind: n.kind,
                calls: n.calls.load(Relaxed),
                wall_ns: wall,
                wall_self_ns: wall_self,
                device_ns: dev,
                device_self_ns: dev_self,
                sync_ns: sync,
                busy_self_ns: wall_self.saturating_sub(sync),
                min_wall_ns: if min == u64::MAX { 0 } else { min },
                max_wall_ns: n.max_wall_ns.load(Relaxed),
                geom: Geometry {
                    b: n.geom_b.load(Relaxed),
                    t: n.geom_t.load(Relaxed),
                    tokens: n.tokens.load(Relaxed),
                },
                reachable: n.reachable.load(Relaxed),
                note: n.note.read().ok().and_then(|g| g.clone()),
                children: n.children.clone(),
            });
        }
        out
    }
}

/// Record one closed span into its node.
#[inline]
pub(crate) fn accumulate(n: &NodeAcc, wall_ns: u64, sync_ns: u64) {
    n.calls.fetch_add(1, Relaxed);
    n.wall_ns.fetch_add(wall_ns, Relaxed);
    if sync_ns != 0 {
        n.sync_ns.fetch_add(sync_ns, Relaxed);
    }
    n.max_wall_ns.fetch_max(wall_ns, Relaxed);
    n.min_wall_ns.fetch_min(wall_ns, Relaxed);
}
