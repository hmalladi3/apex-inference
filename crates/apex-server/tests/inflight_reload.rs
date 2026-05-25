//! SIGHUP reload while a continuous load is in flight against a kept model.
//!
//! For an "add" reload — where no model is being removed — requests against
//! the kept model should see zero errors throughout the registry swap.
//! This proves the `ArcSwap` semantics are correct and the kept dispatcher
//! is never interrupted.

mod common;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use tokio::time::sleep;

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn add_reload_does_not_drop_inflight_requests_for_kept_model() {
    let port = common::find_free_port();
    let cfg = tempfile::NamedTempFile::new().expect("tempfile");

    common::write_config(cfg.path(), port, &[("doubler", "doubler")]);
    let mut child = common::spawn_apex(cfg.path());
    let pid = child.id().expect("pid");

    let endpoint = format!("http://127.0.0.1:{port}");
    let mut control_client = match common::wait_for_ready(&endpoint, Duration::from_secs(30)).await
    {
        Ok(c) => c,
        Err(e) => {
            let _ = child.kill().await;
            panic!("server never became ready: {e}");
        }
    };

    let stop = Arc::new(AtomicBool::new(false));
    let ok_count = Arc::new(AtomicU64::new(0));
    let err_count = Arc::new(AtomicU64::new(0));

    let stop_l = stop.clone();
    let ok_l = ok_count.clone();
    let err_l = err_count.clone();
    let endpoint_l = endpoint.clone();
    let load = tokio::spawn(async move {
        let mut client = common::ApexClient::connect(endpoint_l)
            .await
            .expect("load client connect");
        let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let expected = vec![2.0_f32, 4.0, 6.0, 8.0];
        let mut first_errors: Vec<String> = Vec::new();
        while !stop_l.load(Ordering::Relaxed) {
            match common::try_infer(&mut client, "doubler", &input).await {
                Ok(out) => {
                    if out == expected {
                        ok_l.fetch_add(1, Ordering::Relaxed);
                    } else {
                        err_l.fetch_add(1, Ordering::Relaxed);
                        if first_errors.len() < 5 {
                            first_errors.push(format!("wrong output: {out:?}"));
                        }
                    }
                }
                Err(e) => {
                    err_l.fetch_add(1, Ordering::Relaxed);
                    if first_errors.len() < 5 {
                        first_errors.push(format!("{} {}", e.code(), e.message()));
                    }
                }
            }
        }
        first_errors
    });

    // Let load warm up.
    sleep(Duration::from_millis(300)).await;

    // Trigger the add-reload mid-stream.
    common::write_config(
        cfg.path(),
        port,
        &[("doubler", "doubler"), ("tripler", "tripler")],
    );
    let reload_at = Instant::now();
    common::send_sighup(pid);

    // Wait for tripler to become ready (server has reloaded).
    if let Err(e) =
        common::wait_for_model_ready(&mut control_client, "tripler", Duration::from_secs(10)).await
    {
        stop.store(true, Ordering::Relaxed);
        let _ = load.await;
        let _ = child.kill().await;
        panic!("tripler should be ready after reload: {e}");
    }
    let reload_completed_at = Instant::now();

    // Continue load briefly after the reload completed so we capture the swap window.
    sleep(Duration::from_millis(300)).await;
    stop.store(true, Ordering::Relaxed);
    let first_errors = load.await.expect("load task panicked");

    let ok = ok_count.load(Ordering::Relaxed);
    let err = err_count.load(Ordering::Relaxed);
    eprintln!(
        "inflight load: {ok} ok, {err} err; reload visible after {:?}",
        reload_completed_at.duration_since(reload_at)
    );

    assert!(ok > 50, "expected sustained load (>50 requests), got {ok}");
    assert_eq!(
        err, 0,
        "kept model should see ZERO errors during add-reload (first errors: {first_errors:?})"
    );

    // Sanity: the newly-added model is also reachable post-reload.
    let t = common::infer(&mut control_client, "tripler", &[1.0, 2.0, 3.0, 4.0]).await;
    assert_eq!(t, vec![3.0_f32, 6.0, 9.0, 12.0]);

    let _ = child.kill().await;
}
