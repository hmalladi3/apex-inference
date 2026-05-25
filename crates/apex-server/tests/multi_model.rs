//! Multi-model routing: two distinct models loaded from one config, each
//! reachable by name with the correct output. Proves the registry router
//! actually dispatches by model_name at runtime.

mod common;

use std::time::Duration;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn routes_to_correct_model_in_multi_model_config() {
    let port = common::find_free_port();
    let cfg = tempfile::NamedTempFile::new().expect("tempfile");
    common::write_config(
        cfg.path(),
        port,
        &[("doubler", "doubler"), ("tripler", "tripler")],
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

    let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];

    let doubler_out = common::infer(&mut client, "doubler", &input).await;
    assert_eq!(
        doubler_out,
        vec![2.0_f32, 4.0, 6.0, 8.0],
        "doubler should multiply by 2"
    );

    let tripler_out = common::infer(&mut client, "tripler", &input).await;
    assert_eq!(
        tripler_out,
        vec![3.0_f32, 6.0, 9.0, 12.0],
        "tripler should multiply by 3"
    );

    // Interleaved to verify routing isn't sticky / cached wrong.
    for _ in 0..4 {
        let d = common::infer(&mut client, "doubler", &input).await;
        let t = common::infer(&mut client, "tripler", &input).await;
        assert_eq!(d, vec![2.0_f32, 4.0, 6.0, 8.0]);
        assert_eq!(t, vec![3.0_f32, 6.0, 9.0, 12.0]);
    }

    let _ = child.kill().await;
}
