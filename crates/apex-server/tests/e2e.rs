//! End-to-end integration test.
//!
//! Spawns the `apex-inference` binary as a subprocess pointing at the
//! `doubler.onnx` fixture, waits for the server to become ready, sends a
//! ModelInfer over gRPC, and asserts the round-trip arithmetic.
//!
//! This is the lowest-effort proof that the engine actually serves
//! inference — none of the unit tests touch real ORT.

use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Duration, Instant};

use apex_server::proto::grpc_inference_service_client::GrpcInferenceServiceClient;
use apex_server::proto::{
    InferParameter, ModelInferRequest, ServerLiveRequest, ServerReadyRequest,
    model_infer_request::InferInputTensor,
};
use std::collections::HashMap;
use tokio::time::sleep;
use tonic::transport::Channel;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn end_to_end_inference_via_grpc() {
    let port = find_free_port();
    let fixture = repo_root().join("tests/fixtures/doubler.onnx");
    assert!(
        fixture.exists(),
        "ONNX fixture missing at {} — run `python3 tests/fixtures/build_fixture.py`",
        fixture.display()
    );

    let cfg_file = tempfile::NamedTempFile::new().expect("tempfile");
    std::fs::write(
        cfg_file.path(),
        format!(
            r#"
server:
  listen: "127.0.0.1:{port}"
  request_timeout_secs: 30
  shutdown_grace_secs: 5
  max_request_bytes: 67108864
admission:
  max_rss_bytes: 8589934592
  max_queue_depth: 1024
  rss_sample_interval_ms: 100
models:
  - name: doubler
    version: "1"
    path: {fixture}
    kind: fixed_shape
    max_batch_size: 4
    max_queue_delay_us: 1000
    intra_op_threads: 1
"#,
            port = port,
            fixture = fixture.display()
        ),
    )
    .expect("write config");

    let mut child = tokio::process::Command::new(env!("CARGO_BIN_EXE_apex-inference"))
        .args(["--config", cfg_file.path().to_str().unwrap()])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .kill_on_drop(true)
        .spawn()
        .expect("spawn apex-inference");

    let endpoint = format!("http://127.0.0.1:{port}");
    let mut client = match wait_for_ready(&endpoint, Duration::from_secs(30)).await {
        Ok(c) => c,
        Err(e) => {
            let _ = child.kill().await;
            panic!("server never became ready: {e}");
        }
    };

    // Sanity: ServerLive should also return true.
    let live = client
        .server_live(ServerLiveRequest {})
        .await
        .expect("ServerLive")
        .into_inner();
    assert!(live.live, "expected ServerLive to return live=true");

    // ModelInfer round-trip: doubler should turn [1,2,3,4] into [2,4,6,8].
    let payload: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let raw_bytes: Vec<u8> = payload.iter().flat_map(|f| f.to_ne_bytes()).collect();
    let req = ModelInferRequest {
        model_name: "doubler".to_string(),
        model_version: String::new(),
        id: "test-001".to_string(),
        parameters: HashMap::<String, InferParameter>::new(),
        inputs: vec![InferInputTensor {
            name: "input".to_string(),
            datatype: "FP32".to_string(),
            shape: vec![1, 4],
            parameters: HashMap::<String, InferParameter>::new(),
            contents: None,
        }],
        outputs: vec![],
        raw_input_contents: vec![raw_bytes],
    };

    let resp = client
        .model_infer(req)
        .await
        .expect("ModelInfer")
        .into_inner();

    assert_eq!(resp.model_name, "doubler");
    assert_eq!(resp.id, "test-001");
    assert_eq!(
        resp.raw_output_contents.len(),
        1,
        "expected one output tensor"
    );

    let out: Vec<f32> = resp.raw_output_contents[0]
        .chunks_exact(4)
        .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let expected = vec![2.0_f32, 4.0, 6.0, 8.0];
    assert_eq!(
        out, expected,
        "doubler should multiply input by 2: got {out:?}, expected {expected:?}"
    );

    let _ = child.kill().await;
}

fn repo_root() -> PathBuf {
    // CARGO_MANIFEST_DIR is /<repo>/crates/apex-server; go up two.
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .map(|p| p.to_path_buf())
        .expect("repo root")
}

fn find_free_port() -> u16 {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind ephemeral");
    listener.local_addr().expect("local_addr").port()
}

async fn wait_for_ready(
    endpoint: &str,
    timeout: Duration,
) -> Result<GrpcInferenceServiceClient<Channel>, String> {
    let deadline = Instant::now() + timeout;
    let mut last_err = String::from("(no attempt)");
    while Instant::now() < deadline {
        match GrpcInferenceServiceClient::connect(endpoint.to_string()).await {
            Ok(mut c) => match c.server_ready(ServerReadyRequest {}).await {
                Ok(r) => {
                    if r.into_inner().ready {
                        return Ok(c);
                    }
                    last_err = "ServerReady returned ready=false".to_string();
                }
                Err(e) => last_err = format!("ServerReady call: {e}"),
            },
            Err(e) => last_err = format!("connect: {e}"),
        }
        sleep(Duration::from_millis(200)).await;
    }
    Err(last_err)
}
