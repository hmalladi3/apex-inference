//! Shared helpers for integration tests.
//!
//! `common/mod.rs` is the magic name that tells cargo not to treat this
//! directory as a standalone test binary — it's compiled only into test
//! binaries that `mod common;` it.

#![allow(dead_code)] // each test binary uses a subset

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::{Duration, Instant};

use apex_server::proto::grpc_inference_service_client::GrpcInferenceServiceClient;
use apex_server::proto::model_infer_request::InferInputTensor;
use apex_server::proto::{
    InferParameter, ModelInferRequest, ModelReadyRequest, ServerReadyRequest,
};
use tokio::process::Child;
use tokio::time::sleep;
use tonic::transport::Channel;

pub type ApexClient = GrpcInferenceServiceClient<Channel>;

pub fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .map(|p| p.to_path_buf())
        .expect("repo root")
}

pub fn fixture_path(name: &str) -> PathBuf {
    repo_root().join(format!("tests/fixtures/{name}.onnx"))
}

pub fn find_free_port() -> u16 {
    let l = std::net::TcpListener::bind("127.0.0.1:0").expect("bind ephemeral");
    l.local_addr().expect("local_addr").port()
}

/// `models` is a list of `(model_name, fixture_basename)` pairs.
pub fn write_config(path: &Path, port: u16, models: &[(&str, &str)]) {
    let model_entries: String = models
        .iter()
        .map(|(name, fixture)| {
            format!(
                "  - name: {name}\n    version: \"1\"\n    path: {fp}\n    kind: fixed_shape\n    max_batch_size: 4\n    max_queue_delay_us: 200\n    intra_op_threads: 1\n",
                name = name,
                fp = fixture_path(fixture).display()
            )
        })
        .collect();

    let cfg = format!(
        "server:\n  listen: \"127.0.0.1:{port}\"\n  request_timeout_secs: 30\n  shutdown_grace_secs: 5\n  max_request_bytes: 67108864\nadmission:\n  max_rss_bytes: 8589934592\n  max_queue_depth: 1024\n  rss_sample_interval_ms: 100\nmodels:\n{model_entries}",
        port = port,
        model_entries = model_entries
    );
    std::fs::write(path, cfg).expect("write config");
}

pub fn spawn_apex(config_path: &Path) -> Child {
    tokio::process::Command::new(env!("CARGO_BIN_EXE_apex-inference"))
        .args(["--config", config_path.to_str().unwrap()])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .kill_on_drop(true)
        .spawn()
        .expect("spawn apex-inference")
}

pub async fn wait_for_ready(endpoint: &str, timeout: Duration) -> Result<ApexClient, String> {
    let deadline = Instant::now() + timeout;
    let mut last_err = String::from("(no attempt)");
    while Instant::now() < deadline {
        match ApexClient::connect(endpoint.to_string()).await {
            Ok(mut c) => match c.server_ready(ServerReadyRequest {}).await {
                Ok(r) => {
                    if r.into_inner().ready {
                        return Ok(c);
                    }
                    last_err = "server_ready returned false".into();
                }
                Err(e) => last_err = format!("server_ready: {e}"),
            },
            Err(e) => last_err = format!("connect: {e}"),
        }
        sleep(Duration::from_millis(200)).await;
    }
    Err(last_err)
}

pub async fn wait_for_model_ready(
    client: &mut ApexClient,
    model: &str,
    timeout: Duration,
) -> Result<(), String> {
    let deadline = Instant::now() + timeout;
    let mut last = String::from("(no attempt)");
    while Instant::now() < deadline {
        let resp = client
            .model_ready(ModelReadyRequest {
                name: model.to_string(),
                version: String::new(),
            })
            .await;
        match resp {
            Ok(r) => {
                if r.into_inner().ready {
                    return Ok(());
                }
                last = format!("model {model} not ready");
            }
            Err(e) => last = format!("model_ready RPC: {e}"),
        }
        sleep(Duration::from_millis(100)).await;
    }
    Err(last)
}

pub async fn wait_for_model_gone(
    client: &mut ApexClient,
    model: &str,
    timeout: Duration,
) -> Result<(), String> {
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        let resp = client
            .model_ready(ModelReadyRequest {
                name: model.to_string(),
                version: String::new(),
            })
            .await;
        let ready = resp.map(|r| r.into_inner().ready).unwrap_or(false);
        if !ready {
            return Ok(());
        }
        sleep(Duration::from_millis(100)).await;
    }
    Err(format!("model {model} still ready after {timeout:?}"))
}

pub async fn infer(client: &mut ApexClient, model: &str, payload: &[f32]) -> Vec<f32> {
    let raw_bytes: Vec<u8> = payload.iter().flat_map(|f| f.to_ne_bytes()).collect();
    let req = ModelInferRequest {
        model_name: model.to_string(),
        model_version: String::new(),
        id: format!("{model}-test"),
        parameters: HashMap::<String, InferParameter>::new(),
        inputs: vec![InferInputTensor {
            name: "input".to_string(),
            datatype: "FP32".to_string(),
            shape: vec![1, payload.len() as i64],
            parameters: HashMap::<String, InferParameter>::new(),
            contents: None,
        }],
        outputs: vec![],
        raw_input_contents: vec![raw_bytes],
    };
    let resp = client
        .model_infer(req)
        .await
        .expect("model_infer")
        .into_inner();
    assert_eq!(resp.raw_output_contents.len(), 1, "expected one output");
    resp.raw_output_contents[0]
        .chunks_exact(4)
        .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

pub fn send_sighup(pid: u32) {
    let status = std::process::Command::new("kill")
        .args(["-HUP", &pid.to_string()])
        .status()
        .expect("spawn kill");
    assert!(status.success(), "kill -HUP {pid} failed: {status}");
}
