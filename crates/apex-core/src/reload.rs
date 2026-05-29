//! Config reload state machine — SIGHUP handler, diff, swap, drain.
//!
//! See `docs/llds/config-and-reload.md` and `docs/specs/config-and-reload.md`.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use arc_swap::ArcSwap;
use prometheus::{Histogram, HistogramOpts, IntCounterVec, IntGauge, Opts, Registry};
use sha2::{Digest, Sha256};

use crate::config::{Config, ModelConfig};
use crate::dispatcher::{self, BoundMetrics, BucketConfig, DispatcherMetrics};
use crate::error::ConfigError;
use crate::ort_bridge::ModelBridge;
use crate::registry::{AtomicEntryState, EntryState, ModelEntry, ModelRegistry, SharedRegistry};

#[derive(Clone)]
pub struct ReloadMetrics {
    pub version_hash: IntGauge,
    pub reload_total: IntCounterVec,
    pub reload_duration: Histogram,
    pub drain_duration: Histogram,
    pub models_loaded: IntGauge,
}

impl ReloadMetrics {
    pub fn register(registry: &Registry) -> Result<Self, prometheus::Error> {
        let version_hash = IntGauge::new(
            "apex_config_version_hash",
            "first 8 bytes of SHA-256(config) as a u64",
        )?;
        let reload_total = IntCounterVec::new(
            Opts::new(
                "apex_config_reload_total",
                "config reload attempts by outcome",
            ),
            &["outcome"],
        )?;
        let reload_duration = Histogram::with_opts(
            HistogramOpts::new(
                "apex_config_reload_duration_seconds",
                "SIGHUP-to-swap reload duration",
            )
            .buckets(vec![0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 30.0]),
        )?;
        let drain_duration = Histogram::with_opts(
            HistogramOpts::new(
                "apex_model_drain_duration_seconds",
                "per-model drain duration",
            )
            .buckets(vec![0.01, 0.1, 0.5, 1.0, 5.0, 30.0, 120.0]),
        )?;
        let models_loaded = IntGauge::new("apex_models_loaded", "current count of loaded models")?;

        registry.register(Box::new(version_hash.clone()))?;
        registry.register(Box::new(reload_total.clone()))?;
        registry.register(Box::new(reload_duration.clone()))?;
        registry.register(Box::new(drain_duration.clone()))?;
        registry.register(Box::new(models_loaded.clone()))?;

        Ok(Self {
            version_hash,
            reload_total,
            reload_duration,
            drain_duration,
            models_loaded,
        })
    }
}

/// Construct an initial registry from a parsed config. Serial load for v1;
/// parallel load (via JoinSet) is a v1.1 optimization once startup latency
/// shows up in benchmarks.
///
/// @spec CONFIG-LOAD-003
pub async fn build_registry_from_config(
    config: &Config,
    dispatcher_metrics: Arc<DispatcherMetrics>,
) -> Result<ModelRegistry, ConfigError> {
    let mut entries = Vec::with_capacity(config.models.len());
    for model_cfg in &config.models {
        let entry = build_entry(model_cfg, dispatcher_metrics.clone()).await?;
        entries.push(entry);
    }
    Ok(ModelRegistry::from_entries(entries))
}

async fn build_entry(
    cfg: &ModelConfig,
    dispatcher_metrics: Arc<DispatcherMetrics>,
) -> Result<Arc<ModelEntry>, ConfigError> {
    let cfg_clone = cfg.clone();
    let bridge = tokio::task::spawn_blocking(move || ModelBridge::load(&cfg_clone))
        .await
        .map_err(|e| ConfigError::Validation(format!("session load task panicked: {e}")))??;
    let bridge = Arc::new(bridge);

    let bucket_cfg = BucketConfig {
        max_batch_size: cfg.max_batch_size,
        max_queue_delay: Duration::from_micros(cfg.max_queue_delay_us),
        queue_capacity: cfg.resolved_queue_capacity(),
    };
    let bound = BoundMetrics::new(cfg.name.clone(), dispatcher_metrics);
    let (tx, handle) = dispatcher::spawn(bridge.clone(), bucket_cfg, bound);

    Ok(Arc::new(ModelEntry {
        name: cfg.name.clone(),
        version: cfg.version.clone(),
        input_meta: bridge.input_meta().clone(),
        output_meta: bridge.output_meta().to_vec(),
        bridge,
        state: AtomicEntryState::new(EntryState::Loaded),
        tx,
        task_handle: Mutex::new(Some(handle)),
    }))
}

/// SIGHUP watcher. Reloads config on each signal until the receiver shuts down.
///
/// @spec CONFIG-RELOAD-001
#[cfg(unix)]
pub async fn watch_sighup(
    registry: SharedRegistry,
    config_path: PathBuf,
    dispatcher_metrics: Arc<DispatcherMetrics>,
    reload_metrics: Arc<ReloadMetrics>,
    shutdown_grace: Duration,
) {
    use tokio::signal::unix::{SignalKind, signal};
    let mut sighup = match signal(SignalKind::hangup()) {
        Ok(s) => s,
        Err(e) => {
            tracing::error!(error = %e, "failed to install SIGHUP handler");
            return;
        }
    };
    while sighup.recv().await.is_some() {
        tracing::info!("SIGHUP received; reloading config");
        match reload(
            &registry,
            &config_path,
            dispatcher_metrics.clone(),
            &reload_metrics,
            shutdown_grace,
        )
        .await
        {
            Ok(_) => tracing::info!("reload ok"),
            Err(e) => tracing::warn!(error = %e, "reload failed"),
        }
    }
}

#[cfg(not(unix))]
pub async fn watch_sighup(
    _registry: SharedRegistry,
    _config_path: PathBuf,
    _dispatcher_metrics: Arc<DispatcherMetrics>,
    _reload_metrics: Arc<ReloadMetrics>,
    _shutdown_grace: Duration,
) {
    tracing::warn!("SIGHUP config reload is not supported on this platform");
    std::future::pending::<()>().await;
}

/// Reload the config file, diff against the live registry, atomically swap,
/// drain removed dispatchers in the background.
///
/// @spec CONFIG-RELOAD-002, CONFIG-RELOAD-003, CONFIG-RELOAD-004,
///       CONFIG-RELOAD-005, CONFIG-RELOAD-006, CONFIG-RELOAD-007,
///       CONFIG-RELOAD-008, CONFIG-RELOAD-009, CONFIG-RELOAD-010,
///       CONFIG-RELOAD-011
pub async fn reload(
    registry: &SharedRegistry,
    config_path: &Path,
    dispatcher_metrics: Arc<DispatcherMetrics>,
    rm: &Arc<ReloadMetrics>,
    shutdown_grace: Duration,
) -> Result<(), ConfigError> {
    let start = Instant::now();

    let new_config = match Config::from_yaml_file(config_path) {
        Ok(c) => c,
        Err(e) => {
            rm.reload_total.with_label_values(&["invalid"]).inc();
            return Err(e);
        }
    };
    if let Err(e) = new_config.validate() {
        rm.reload_total.with_label_values(&["invalid"]).inc();
        return Err(e);
    }

    let current = registry.load();
    let new_keys: HashSet<(String, String)> = new_config
        .models
        .iter()
        .map(|m| (m.name.clone(), m.version.clone()))
        .collect();
    let current_keys: HashSet<(String, String)> = current
        .all_entries()
        .map(|e| (e.name.clone(), e.version.clone()))
        .collect();

    let to_add: Vec<&ModelConfig> = new_config
        .models
        .iter()
        .filter(|m| !current_keys.contains(&(m.name.clone(), m.version.clone())))
        .collect();
    let to_remove: Vec<Arc<ModelEntry>> = current
        .all_entries()
        .filter(|e| !new_keys.contains(&(e.name.clone(), e.version.clone())))
        .cloned()
        .collect();

    let mut new_entries: Vec<Arc<ModelEntry>> = Vec::with_capacity(to_add.len());
    for m in to_add {
        match build_entry(m, dispatcher_metrics.clone()).await {
            Ok(e) => new_entries.push(e),
            Err(e) => {
                // Drop any partially-loaded entries (RAII via Vec drop).
                rm.reload_total.with_label_values(&["load_failed"]).inc();
                return Err(e);
            }
        }
    }

    let mut all_entries: Vec<Arc<ModelEntry>> = current
        .all_entries()
        .filter(|e| new_keys.contains(&(e.name.clone(), e.version.clone())))
        .cloned()
        .collect();
    all_entries.extend(new_entries);
    let count = all_entries.len();

    let new_registry = ModelRegistry::from_entries(all_entries);
    registry.store(Arc::new(new_registry));

    rm.models_loaded.set(count as i64);
    rm.version_hash.set(config_hash_u64(&new_config) as i64);
    rm.reload_total.with_label_values(&["ok"]).inc();
    rm.reload_duration.observe(start.elapsed().as_secs_f64());

    // Detach drain so the reload handler returns promptly.
    let rm_clone = rm.clone();
    tokio::spawn(async move {
        for entry in to_remove {
            let rm = rm_clone.clone();
            tokio::spawn(async move {
                drain_entry(entry, shutdown_grace, rm).await;
            });
        }
    });

    Ok(())
}

/// @spec CONFIG-RELOAD-010, CONFIG-DRAIN-001, CONFIG-STATE-001
async fn drain_entry(entry: Arc<ModelEntry>, grace: Duration, rm: Arc<ReloadMetrics>) {
    entry.state.store(EntryState::Draining);
    let Some(handle) = entry.take_task_handle() else {
        return;
    };
    let abort = handle.abort_handle();
    let start = Instant::now();

    // Drop our Arc so the dispatcher channel can close when remaining
    // in-flight handler Arcs also drop.
    drop(entry);

    let outcome = tokio::time::timeout(grace, handle).await;
    rm.drain_duration.observe(start.elapsed().as_secs_f64());
    match outcome {
        Ok(Ok(())) => tracing::info!(drain_secs = start.elapsed().as_secs_f64(), "drain complete"),
        Ok(Err(e)) => tracing::warn!(error = ?e, "drain task error"),
        Err(_) => {
            tracing::warn!(grace_secs = grace.as_secs(), "drain timeout; aborting");
            abort.abort();
        }
    }
}

/// Drain every model on shutdown (SIGTERM / SIGINT). Swaps to an empty
/// registry first so new requests get routed nowhere.
///
/// @spec CONFIG-SHUTDOWN-001
pub async fn shutdown_drain(
    registry: &SharedRegistry,
    shutdown_grace: Duration,
    rm: Arc<ReloadMetrics>,
) {
    let old: Arc<ModelRegistry> = registry.load_full();
    registry.store(Arc::new(ModelRegistry::empty()));

    let mut tasks = Vec::new();
    for entry in old.all_entries() {
        let entry = entry.clone();
        let rm = rm.clone();
        tasks.push(tokio::spawn(async move {
            drain_entry(entry, shutdown_grace, rm).await;
        }));
    }
    // Drop our own reference so refcounts can hit zero
    drop(old);

    for t in tasks {
        let _ = t.await;
    }
    rm.models_loaded.set(0);
}

/// @spec CONFIG-METRIC-001
fn config_hash_u64(config: &Config) -> u64 {
    let bytes = format!("{config:?}").into_bytes();
    let digest = Sha256::digest(&bytes);
    let mut out = [0u8; 8];
    out.copy_from_slice(&digest[..8]);
    u64::from_le_bytes(out)
}

/// Helper to construct a SharedRegistry from an initial ModelRegistry.
pub fn shared_registry(initial: ModelRegistry) -> SharedRegistry {
    Arc::new(ArcSwap::from_pointee(initial))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{BridgeRuntimeKind, ModelKind, ServerConfig};
    use std::path::PathBuf;

    fn minimal_config() -> Config {
        Config {
            server: ServerConfig {
                listen: "0.0.0.0:9000".to_string(),
                request_timeout_secs: 30,
                shutdown_grace_secs: 30,
                max_request_bytes: 64 * 1024 * 1024,
            },
            observability: Default::default(),
            admission: Default::default(),
            models: vec![ModelConfig {
                name: "m".to_string(),
                version: "1".to_string(),
                path: PathBuf::from("/nonexistent.onnx"),
                kind: ModelKind::FixedShape,
                max_batch_size: 1,
                max_queue_delay_us: 1000,
                intra_op_threads: 1,
                queue_capacity: None,
                runtime: BridgeRuntimeKind::Blocking,
                seq_len_buckets: None,
                requires_attention_mask: false,
            }],
        }
    }

    #[test]
    fn config_hash_is_stable_for_same_input() {
        let c = minimal_config();
        assert_eq!(config_hash_u64(&c), config_hash_u64(&c));
    }

    #[test]
    fn config_hash_differs_when_config_changes() {
        let a = minimal_config();
        let mut b = minimal_config();
        b.models[0].max_batch_size = 2;
        assert_ne!(config_hash_u64(&a), config_hash_u64(&b));
    }

    #[test]
    fn shared_registry_starts_empty() {
        let r = shared_registry(ModelRegistry::empty());
        assert_eq!(r.load().count(), 0);
    }
}
