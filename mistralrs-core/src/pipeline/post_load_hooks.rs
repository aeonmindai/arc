//! Post-load model hooks.
//!
//! Plugins (like Arc's TD-MoE compressor) can register a closure here that
//! runs after a model is fully loaded and quantized, with mutable access to
//! its `IsqModel` interface (which exposes MoE expert layers via
//! `get_layers_moe_experts_only`).
//!
//! This is a runtime indirection so `mistralrs-core` can be agnostic of
//! downstream extensions that live in crates that depend on it (avoiding
//! cyclic dependencies). The hook receives `&mut dyn IsqModel` and may:
//!
//! - inspect / replace quantized layers (e.g. compress MoE experts)
//! - read residual tensors
//! - emit logs / metrics
//!
//! Hooks are invoked in registration order, and an error from any hook
//! short-circuits the load.

use std::sync::{Mutex, OnceLock};

use crate::pipeline::isq::IsqModel;

/// Signature for a post-load hook.
pub type PostLoadHook =
    Box<dyn Fn(&mut dyn IsqModel) -> anyhow::Result<()> + Send + Sync + 'static>;

static HOOKS: OnceLock<Mutex<Vec<PostLoadHook>>> = OnceLock::new();

fn hooks() -> &'static Mutex<Vec<PostLoadHook>> {
    HOOKS.get_or_init(|| Mutex::new(Vec::new()))
}

/// Register a hook that runs after every model load. The hook gets mutable
/// access to the loaded model's `IsqModel` view.
pub fn register_post_load_hook(hook: PostLoadHook) {
    let lock = hooks();
    let mut guard = lock.lock().expect("post-load hook registry poisoned");
    guard.push(hook);
}

/// Whether any post-load hook is registered. Used to defer UQFF serialization
/// until *after* the hooks run, so the written UQFF reflects the post-hook model
/// (e.g. TD-MoE Tucker-factored experts rather than the pre-hook QTIP ones).
pub(crate) fn has_registered_hooks() -> bool {
    HOOKS
        .get()
        .map(|l| !l.lock().expect("post-load hook registry poisoned").is_empty())
        .unwrap_or(false)
}

/// Invoke every registered post-load hook in order. Aborts on the first error.
///
/// Called by the normal-model loader after ISQ quantization completes.
pub(crate) fn run_post_load_hooks(model: &mut dyn IsqModel) -> anyhow::Result<()> {
    let lock = hooks();
    let guard = lock.lock().expect("post-load hook registry poisoned");
    for (idx, hook) in guard.iter().enumerate() {
        if let Err(e) = hook(model) {
            return Err(anyhow::anyhow!(
                "post-load hook #{idx} failed: {e}"
            ));
        }
    }
    Ok(())
}

/// Test helper: clears all registered hooks. Not public API.
#[cfg(test)]
pub fn clear_hooks_for_test() {
    if let Some(lock) = HOOKS.get() {
        if let Ok(mut guard) = lock.lock() {
            guard.clear();
        }
    }
}
