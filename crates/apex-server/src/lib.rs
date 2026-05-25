//! apex-server: KServe v2 gRPC entrypoint for the apex inference engine.
//!
//! This crate is consumed both as a binary (`apex-inference`) and as a library
//! by integration tests that need access to the generated proto types and the
//! [`run`] entrypoint.

pub mod grpc;
pub mod proto;

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Context;
use apex_core::admission::{self, AdmissionController};
use apex_core::config::{Config, LogFormat};
use apex_core::dispatcher::DispatcherMetrics;
use apex_core::reload::{self, ReloadMetrics};
use axum::Router;
use axum::routing::get;
use clap::Parser;
use prometheus::{Encoder, Registry, TextEncoder};
use tonic::transport::Server;
use tracing_subscriber::EnvFilter;

use crate::grpc::{GrpcMetrics, InferenceService};
use crate::proto::grpc_inference_service_server::GrpcInferenceServiceServer;

const DEFAULT_RSS_BUDGET_BYTES: u64 = 8 * 1024 * 1024 * 1024;

#[derive(Parser, Debug, Clone)]
#[command(
    name = "apex-inference",
    version,
    about = "KServe v2 ONNX inference server"
)]
pub struct Args {
    /// Path to the YAML config file.
    #[arg(long, env = "APEX_CONFIG")]
    pub config: PathBuf,
}

/// Run the apex-inference server. Returns when SIGTERM/SIGINT is received
/// and all dispatchers have drained (bounded by `shutdown_grace_secs`).
pub async fn run(args: Args) -> anyhow::Result<()> {
    let config = Config::from_yaml_file(&args.config)
        .with_context(|| format!("loading config from {}", args.config.display()))?;
    config.validate().context("validating config")?;

    init_tracing(&config);

    tracing::info!(
        config = %args.config.display(),
        models = config.models.len(),
        listen = %config.server.listen,
        "starting apex-inference"
    );

    let prom_registry = Arc::new(Registry::new());

    let admission_max_rss = config
        .admission
        .max_rss_bytes
        .unwrap_or(DEFAULT_RSS_BUDGET_BYTES);
    let admission_max_queue_depth = config.admission.max_queue_depth.unwrap_or_else(|| {
        config
            .models
            .iter()
            .map(|m| m.max_batch_size)
            .sum::<usize>()
            * 10
    });

    let admission_controller = Arc::new(
        AdmissionController::new(admission_max_rss, admission_max_queue_depth, &prom_registry)
            .context("registering admission metrics")?,
    );
    let dispatcher_metrics = Arc::new(
        DispatcherMetrics::register(&prom_registry).context("registering dispatcher metrics")?,
    );
    let reload_metrics =
        Arc::new(ReloadMetrics::register(&prom_registry).context("registering reload metrics")?);
    let grpc_metrics = GrpcMetrics::register(&prom_registry).context("registering gRPC metrics")?;

    let _rss_sampler = admission::spawn_rss_sampler(
        admission_controller.clone(),
        Duration::from_millis(config.admission.rss_sample_interval_ms),
    );

    let initial_registry = reload::build_registry_from_config(&config, dispatcher_metrics.clone())
        .await
        .context("building initial model registry")?;
    let model_count = initial_registry.count();
    reload_metrics.models_loaded.set(model_count as i64);
    let shared_registry = reload::shared_registry(initial_registry);

    tracing::info!(models = model_count, "models loaded");

    let _reload_watcher = tokio::spawn(reload::watch_sighup(
        shared_registry.clone(),
        args.config.clone(),
        dispatcher_metrics.clone(),
        reload_metrics.clone(),
        Duration::from_secs(config.server.shutdown_grace_secs),
    ));

    if let Some(addr_str) = &config.observability.metrics_listen {
        let addr: SocketAddr = addr_str
            .parse()
            .with_context(|| format!("parsing observability.metrics_listen: {addr_str}"))?;
        let registry = prom_registry.clone();
        tokio::spawn(async move {
            if let Err(e) = serve_metrics(addr, registry).await {
                tracing::error!(error = ?e, "metrics server failed");
            }
        });
        tracing::info!(addr = %addr, "metrics endpoint listening");
    }

    let svc = InferenceService {
        registry: shared_registry.clone(),
        admission: admission_controller.clone(),
        max_request_bytes: config.server.max_request_bytes,
        metrics: grpc_metrics,
    };

    let grpc_addr: SocketAddr = config
        .server
        .listen
        .parse()
        .with_context(|| format!("parsing server.listen: {}", config.server.listen))?;
    tracing::info!(addr = %grpc_addr, "gRPC server listening");

    Server::builder()
        .add_service(GrpcInferenceServiceServer::new(svc))
        .serve_with_shutdown(grpc_addr, shutdown_signal())
        .await
        .context("gRPC serve")?;

    tracing::info!("gRPC server stopped; draining dispatchers");

    reload::shutdown_drain(
        &shared_registry,
        Duration::from_secs(config.server.shutdown_grace_secs),
        reload_metrics,
    )
    .await;

    tracing::info!("shutdown complete");
    Ok(())
}

fn init_tracing(config: &Config) {
    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(&config.observability.log_level));
    let _ = match config.observability.log_format {
        LogFormat::Json => tracing_subscriber::fmt()
            .with_env_filter(env_filter)
            .json()
            .try_init(),
        LogFormat::Pretty => tracing_subscriber::fmt()
            .with_env_filter(env_filter)
            .try_init(),
    };
}

async fn serve_metrics(addr: SocketAddr, registry: Arc<Registry>) -> anyhow::Result<()> {
    let app = Router::new()
        .route("/metrics", get(metrics_handler))
        .with_state(registry);
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

async fn metrics_handler(
    axum::extract::State(registry): axum::extract::State<Arc<Registry>>,
) -> Result<(axum::http::HeaderMap, String), axum::http::StatusCode> {
    let metric_families = registry.gather();
    let encoder = TextEncoder::new();
    let mut buffer = Vec::new();
    encoder
        .encode(&metric_families, &mut buffer)
        .map_err(|_| axum::http::StatusCode::INTERNAL_SERVER_ERROR)?;
    let body =
        String::from_utf8(buffer).map_err(|_| axum::http::StatusCode::INTERNAL_SERVER_ERROR)?;
    let mut headers = axum::http::HeaderMap::new();
    headers.insert(
        axum::http::header::CONTENT_TYPE,
        encoder
            .format_type()
            .parse()
            .map_err(|_| axum::http::StatusCode::INTERNAL_SERVER_ERROR)?,
    );
    Ok((headers, body))
}

#[cfg(unix)]
async fn shutdown_signal() {
    use tokio::signal::unix::{SignalKind, signal};
    let mut sigterm = match signal(SignalKind::terminate()) {
        Ok(s) => s,
        Err(e) => {
            tracing::error!(error = %e, "failed to install SIGTERM handler; using SIGINT only");
            let _ = tokio::signal::ctrl_c().await;
            return;
        }
    };
    tokio::select! {
        _ = tokio::signal::ctrl_c() => tracing::info!("SIGINT received"),
        _ = sigterm.recv() => tracing::info!("SIGTERM received"),
    }
}

#[cfg(not(unix))]
async fn shutdown_signal() {
    if let Err(e) = tokio::signal::ctrl_c().await {
        tracing::error!(error = %e, "Ctrl-C handler failed");
    } else {
        tracing::info!("Ctrl-C received");
    }
}
