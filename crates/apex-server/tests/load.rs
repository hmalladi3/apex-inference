//! Burst load: a concurrent request count that the old hardcoded
//! `2 * max_batch_size` channel (16 slots) would have shed now fits in the
//! configurable per-model queue. Guards against regressing the queue-depth
//! fix and proves backpressure is governed by `queue_capacity`, not an
//! accidental constant.

mod common;

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn burst_within_queue_capacity_is_fully_served() {
    let port = common::find_free_port();
    let cfg = tempfile::NamedTempFile::new().expect("tempfile");

    // max_batch_size 8 → old channel was 16 slots. queue_capacity 256 gives
    // plenty of headroom for a 64-request burst.
    let fixture = common::fixture_path("doubler");
    let yaml = format!(
        "server:\n  listen: \"127.0.0.1:{port}\"\n  shutdown_grace_secs: 5\n  max_request_bytes: 67108864\nadmission:\n  max_rss_bytes: 8589934592\n  max_queue_depth: 4096\n  rss_sample_interval_ms: 100\nmodels:\n  - name: doubler\n    version: \"1\"\n    path: {fixture}\n    kind: fixed_shape\n    max_batch_size: 8\n    max_queue_delay_us: 2000\n    intra_op_threads: 1\n    queue_capacity: 256\n",
        port = port,
        fixture = fixture.display()
    );
    std::fs::write(cfg.path(), yaml).expect("write config");

    let mut child = common::spawn_apex(cfg.path());
    let endpoint = format!("http://127.0.0.1:{port}");
    if let Err(e) = common::wait_for_ready(&endpoint, Duration::from_secs(30)).await {
        let _ = child.kill().await;
        panic!("server never became ready: {e}");
    }

    const N: u64 = 64;
    let ok = Arc::new(AtomicU64::new(0));
    let rejected = Arc::new(AtomicU64::new(0));
    let other_err = Arc::new(AtomicU64::new(0));

    let mut handles = Vec::new();
    for _ in 0..N {
        let endpoint = endpoint.clone();
        let ok = ok.clone();
        let rejected = rejected.clone();
        let other_err = other_err.clone();
        handles.push(tokio::spawn(async move {
            let mut client = common::ApexClient::connect(endpoint)
                .await
                .expect("connect");
            match common::try_infer(&mut client, "doubler", &[1.0, 2.0, 3.0, 4.0]).await {
                Ok(out) => {
                    assert_eq!(out, vec![2.0_f32, 4.0, 6.0, 8.0]);
                    ok.fetch_add(1, Ordering::Relaxed);
                }
                Err(s) if s.code() == tonic::Code::ResourceExhausted => {
                    rejected.fetch_add(1, Ordering::Relaxed);
                }
                Err(_) => {
                    other_err.fetch_add(1, Ordering::Relaxed);
                }
            }
        }));
    }
    for h in handles {
        let _ = h.await;
    }

    let ok = ok.load(Ordering::Relaxed);
    let rejected = rejected.load(Ordering::Relaxed);
    let other = other_err.load(Ordering::Relaxed);
    eprintln!("burst of {N}: {ok} ok, {rejected} rejected, {other} other-error");

    assert_eq!(other, 0, "no unexpected errors expected");
    assert_eq!(
        ok,
        N,
        "all {N} requests should be served within queue_capacity=256 \
         (got {ok} ok, {rejected} rejected) — the old 2*max_batch_size=16 \
         channel would have shed ~{}",
        N.saturating_sub(16)
    );

    let _ = child.kill().await;
}
