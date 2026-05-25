//! Admission rejection wired end-to-end through the gRPC surface.
//!
//! Sets `max_rss_bytes` absurdly low (100 bytes) so the RSS sampler trips
//! immediately. Once the sampler fires (~50 ms cadence in test config),
//! every ModelInfer must return `RESOURCE_EXHAUSTED`.

mod common;

use std::time::Duration;

use tokio::time::sleep;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn rss_pressure_yields_resource_exhausted_on_model_infer() {
    let port = common::find_free_port();
    let cfg = tempfile::NamedTempFile::new().expect("tempfile");

    // 100 bytes is unreachable — process RSS at startup is megabytes.
    common::write_config_with_admission(
        cfg.path(),
        port,
        &[("doubler", "doubler")],
        /* max_rss_bytes = */ 100,
        /* max_queue_depth = */ 1024,
    );

    let mut child = common::spawn_apex(cfg.path());
    let endpoint = format!("http://127.0.0.1:{port}");

    let mut client = match common::wait_for_ready(&endpoint, Duration::from_secs(30)).await {
        Ok(c) => c,
        Err(e) => {
            let _ = child.kill().await;
            panic!("server never became ready: {e}");
        }
    };

    // Wait long enough for the RSS sampler (50 ms in this config) to publish
    // a non-zero value into the admission atomic.
    sleep(Duration::from_millis(300)).await;

    let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let err = common::try_infer(&mut client, "doubler", &input)
        .await
        .expect_err("expected RESOURCE_EXHAUSTED, got success");

    assert_eq!(
        err.code(),
        tonic::Code::ResourceExhausted,
        "expected RESOURCE_EXHAUSTED, got {:?}: {}",
        err.code(),
        err.message()
    );
    assert!(
        err.message().contains("memory") || err.message().contains("limit"),
        "expected memory-limit message, got: {}",
        err.message()
    );

    let _ = child.kill().await;
}
