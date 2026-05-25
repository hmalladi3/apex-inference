//! SIGHUP reload state machine: add a model, remove a model, both without
//! restarting the process or dropping the model that should stay.

mod common;

use std::time::Duration;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn sighup_reload_adds_then_removes_models() {
    let port = common::find_free_port();
    let cfg = tempfile::NamedTempFile::new().expect("tempfile");

    // Phase 1: start with just doubler.
    common::write_config(cfg.path(), port, &[("doubler", "doubler")]);
    let mut child = common::spawn_apex(cfg.path());
    let pid = child.id().expect("child pid");

    let endpoint = format!("http://127.0.0.1:{port}");
    let mut client = match common::wait_for_ready(&endpoint, Duration::from_secs(30)).await {
        Ok(c) => c,
        Err(e) => {
            let _ = child.kill().await;
            panic!("server never became ready: {e}");
        }
    };

    let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let out = common::infer(&mut client, "doubler", &input).await;
    assert_eq!(
        out,
        vec![2.0_f32, 4.0, 6.0, 8.0],
        "doubler should work before any reload"
    );

    // Phase 2: rewrite config to add tripler, SIGHUP, wait for tripler ready.
    common::write_config(
        cfg.path(),
        port,
        &[("doubler", "doubler"), ("tripler", "tripler")],
    );
    common::send_sighup(pid);
    if let Err(e) =
        common::wait_for_model_ready(&mut client, "tripler", Duration::from_secs(10)).await
    {
        let _ = child.kill().await;
        panic!("tripler should be ready after add-reload: {e}");
    }

    // Both models work after the add.
    let d = common::infer(&mut client, "doubler", &input).await;
    let t = common::infer(&mut client, "tripler", &input).await;
    assert_eq!(d, vec![2.0_f32, 4.0, 6.0, 8.0], "doubler kept after add");
    assert_eq!(t, vec![3.0_f32, 6.0, 9.0, 12.0], "tripler added");

    // Phase 3: rewrite config to remove doubler, SIGHUP, wait for it to go away.
    common::write_config(cfg.path(), port, &[("tripler", "tripler")]);
    common::send_sighup(pid);
    if let Err(e) =
        common::wait_for_model_gone(&mut client, "doubler", Duration::from_secs(10)).await
    {
        let _ = child.kill().await;
        panic!("doubler should be removed after remove-reload: {e}");
    }

    // tripler still works after the removal.
    let t2 = common::infer(&mut client, "tripler", &input).await;
    assert_eq!(
        t2,
        vec![3.0_f32, 6.0, 9.0, 12.0],
        "tripler kept after doubler removed"
    );

    let _ = child.kill().await;
}
