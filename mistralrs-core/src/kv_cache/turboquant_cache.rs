// Re-export from arc-turbo (BSL licensed).
// This file exists only so mistralrs-core can reference TurboQuantCache
// without upstream code changes beyond adding the dependency.
pub use arc_turbo::cache::{TurboQuantCache, TurboQuantSingleCache};
