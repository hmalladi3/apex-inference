//! Model registry — atomically swappable map of (name, version) → ModelEntry.
//!
//! See `docs/llds/config-and-reload.md` and `docs/specs/config-and-reload.md`.
//!
//! V1: fixed-shape models only. Ragged-bucket routing is part of the LLD and
//! will land in v1.1 alongside i64 input dtype and attention-mask generation.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicU8, Ordering};

use arc_swap::ArcSwap;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

use crate::dispatcher::PendingRequest;
use crate::ort_bridge::{ModelBridge, TensorMeta};

pub type SharedRegistry = Arc<ArcSwap<ModelRegistry>>;

pub struct ModelRegistry {
    inner: HashMap<String, HashMap<String, Arc<ModelEntry>>>,
}

impl ModelRegistry {
    pub fn empty() -> Self {
        Self { inner: HashMap::new() }
    }

    pub fn from_entries(entries: Vec<Arc<ModelEntry>>) -> Self {
        let mut inner: HashMap<String, HashMap<String, Arc<ModelEntry>>> = HashMap::new();
        for e in entries {
            inner
                .entry(e.name.clone())
                .or_default()
                .insert(e.version.clone(), e);
        }
        Self { inner }
    }

    /// Look up a model entry. `version = None` or empty selects the version
    /// that sorts last in lexicographic order across registered versions.
    ///
    /// @spec INGRESS-INFER-013
    pub fn get(&self, name: &str, version: Option<&str>) -> Option<Arc<ModelEntry>> {
        let versions = self.inner.get(name)?;
        lookup_version(versions, version).cloned()
    }

    pub fn contains(&self, name: &str, version: &str) -> bool {
        self.inner
            .get(name)
            .map(|v| v.contains_key(version))
            .unwrap_or(false)
    }

    pub fn count(&self) -> usize {
        self.inner.values().map(|v| v.len()).sum()
    }

    pub fn all_entries(&self) -> impl Iterator<Item = &Arc<ModelEntry>> + '_ {
        self.inner.values().flat_map(|v| v.values())
    }
}

pub struct ModelEntry {
    pub name: String,
    pub version: String,
    pub input_meta: TensorMeta,
    pub output_meta: Vec<TensorMeta>,
    pub bridge: Arc<ModelBridge>,
    pub state: AtomicEntryState,
    pub tx: mpsc::Sender<PendingRequest>,
    /// Task handle for the dispatcher. Taken via `take_task_handle` during drain.
    pub task_handle: Mutex<Option<JoinHandle<()>>>,
}

impl ModelEntry {
    pub fn take_task_handle(&self) -> Option<JoinHandle<()>> {
        self.task_handle.lock().ok().and_then(|mut g| g.take())
    }

    /// @spec INGRESS-MODEL-READY-002, CONFIG-STATE-001
    pub fn is_loaded(&self) -> bool {
        matches!(self.state.load(), EntryState::Loaded)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum EntryState {
    Loaded = 0,
    Draining = 1,
}

#[derive(Debug)]
pub struct AtomicEntryState {
    inner: AtomicU8,
}

impl AtomicEntryState {
    pub fn new(state: EntryState) -> Self {
        Self { inner: AtomicU8::new(state as u8) }
    }

    pub fn load(&self) -> EntryState {
        match self.inner.load(Ordering::Acquire) {
            0 => EntryState::Loaded,
            _ => EntryState::Draining,
        }
    }

    pub fn store(&self, state: EntryState) {
        self.inner.store(state as u8, Ordering::Release);
    }
}

/// Free function for version resolution, factored out so it can be unit-tested
/// without constructing a full `ModelEntry` (which requires a real ORT session).
///
/// @spec INGRESS-INFER-013
fn lookup_version<'a, E>(
    versions: &'a HashMap<String, E>,
    requested: Option<&str>,
) -> Option<&'a E> {
    match requested.filter(|v| !v.is_empty()) {
        Some(v) => versions.get(v),
        None => {
            let last_key = versions.keys().max()?;
            versions.get(last_key)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn map<const N: usize>(entries: [(&str, &str); N]) -> HashMap<String, String> {
        entries.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect()
    }

    /// @spec INGRESS-INFER-013
    #[test]
    fn lookup_with_explicit_version_returns_exact_match() {
        let versions = map([("1", "v1-payload"), ("2", "v2-payload")]);
        assert_eq!(lookup_version(&versions, Some("1")), Some(&"v1-payload".to_string()));
        assert_eq!(lookup_version(&versions, Some("2")), Some(&"v2-payload".to_string()));
    }

    #[test]
    fn lookup_with_unknown_version_returns_none() {
        let versions = map([("1", "v1"), ("2", "v2")]);
        assert_eq!(lookup_version(&versions, Some("99")), None);
    }

    /// @spec INGRESS-INFER-013
    #[test]
    fn lookup_with_no_version_returns_lex_last() {
        let versions = map([("1", "v1"), ("2", "v2")]);
        assert_eq!(lookup_version(&versions, None), Some(&"v2".to_string()));
    }

    /// Lex sort, not numeric: "10" < "2" lexicographically, so "2" wins.
    #[test]
    fn lookup_default_uses_lex_order_not_numeric() {
        let versions = map([("1", "v1"), ("2", "v2"), ("10", "v10")]);
        assert_eq!(lookup_version(&versions, None), Some(&"v2".to_string()));
    }

    /// @spec INGRESS-INFER-013
    #[test]
    fn lookup_treats_empty_version_as_no_version() {
        let versions = map([("1", "v1"), ("2", "v2")]);
        assert_eq!(lookup_version(&versions, Some("")), Some(&"v2".to_string()));
    }

    #[test]
    fn lookup_returns_none_for_empty_versions_map() {
        let versions: HashMap<String, String> = HashMap::new();
        assert_eq!(lookup_version(&versions, None), None);
        assert_eq!(lookup_version(&versions, Some("1")), None);
    }

    #[test]
    fn atomic_entry_state_round_trips() {
        let s = AtomicEntryState::new(EntryState::Loaded);
        assert_eq!(s.load(), EntryState::Loaded);
        s.store(EntryState::Draining);
        assert_eq!(s.load(), EntryState::Draining);
    }

    #[test]
    fn empty_registry_has_zero_count() {
        let r = ModelRegistry::empty();
        assert_eq!(r.count(), 0);
        assert!(!r.contains("any", "1"));
        assert!(r.get("any", None).is_none());
        assert_eq!(r.all_entries().count(), 0);
    }
}
