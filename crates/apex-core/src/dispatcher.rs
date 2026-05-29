//! Per-(model, bucket) batching dispatcher.
//!
//! See `docs/llds/per-model-dispatcher.md` and `docs/specs/per-model-dispatcher.md`.

use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, Instant};

use prometheus::{HistogramOpts, HistogramVec, IntCounterVec, IntGaugeVec, Opts, Registry};
use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinHandle;
use tokio::time::{Sleep, sleep};

use crate::error::BridgeError;
use crate::ort_bridge::{BatchOutput, BatchRequest, ModelBridge};

pub struct PendingRequest {
    pub input_bytes: Vec<u8>,
    pub seq_len: Option<u32>,
    pub enqueued_at: Instant,
    pub responder: oneshot::Sender<Result<PerRequestOutput, Arc<BridgeError>>>,
}

#[derive(Debug, Clone)]
pub struct PerRequestOutput {
    pub outputs: Vec<Vec<u8>>,
}

#[derive(Debug, Clone)]
pub struct BucketConfig {
    pub max_batch_size: usize,
    pub max_queue_delay: Duration,
    /// Depth of the request channel. Per-model backpressure: a full channel
    /// makes the gRPC handler return RESOURCE_EXHAUSTED.
    pub queue_capacity: usize,
}

#[derive(Clone)]
pub struct DispatcherMetrics {
    pub batch_size: HistogramVec,
    pub batch_latency_ms: HistogramVec,
    pub wait_time_ms: HistogramVec,
    pub dispatch_reason: IntCounterVec,
    pub batch_errors: IntCounterVec,
    pub overflow: IntCounterVec,
    pub queue_depth: IntGaugeVec,
}

impl DispatcherMetrics {
    pub fn register(registry: &Registry) -> Result<Self, prometheus::Error> {
        let batch_size = HistogramVec::new(
            HistogramOpts::new(
                "apex_dispatcher_batch_size",
                "size of each dispatched batch",
            )
            .buckets(vec![1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0]),
            &["model"],
        )?;
        let batch_latency_ms = HistogramVec::new(
            HistogramOpts::new(
                "apex_dispatcher_batch_latency_ms",
                "time from dispatch start to bridge return, in milliseconds",
            )
            .buckets(vec![0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0]),
            &["model"],
        )?;
        let wait_time_ms = HistogramVec::new(
            HistogramOpts::new(
                "apex_dispatcher_wait_time_ms",
                "time the first-in-batch request waited before dispatch, in milliseconds",
            )
            .buckets(vec![0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]),
            &["model"],
        )?;
        let dispatch_reason = IntCounterVec::new(
            Opts::new(
                "apex_dispatcher_dispatch_reason_total",
                "dispatch trigger reason",
            ),
            &["model", "reason"],
        )?;
        let batch_errors = IntCounterVec::new(
            Opts::new(
                "apex_dispatcher_batch_errors_total",
                "whole-batch failures from the bridge",
            ),
            &["model"],
        )?;
        let overflow = IntCounterVec::new(
            Opts::new(
                "apex_dispatcher_overflow_total",
                "try_send returned Full (client request rejected)",
            ),
            &["model"],
        )?;
        let queue_depth = IntGaugeVec::new(
            Opts::new(
                "apex_dispatcher_queue_depth",
                "current accumulated batch size",
            ),
            &["model"],
        )?;

        for m in [
            Box::new(batch_size.clone()) as Box<dyn prometheus::core::Collector>,
            Box::new(batch_latency_ms.clone()),
            Box::new(wait_time_ms.clone()),
            Box::new(dispatch_reason.clone()),
            Box::new(batch_errors.clone()),
            Box::new(overflow.clone()),
            Box::new(queue_depth.clone()),
        ] {
            registry.register(m)?;
        }

        Ok(Self {
            batch_size,
            batch_latency_ms,
            wait_time_ms,
            dispatch_reason,
            batch_errors,
            overflow,
            queue_depth,
        })
    }
}

/// Per-dispatcher view of metrics, pre-bound to a model label.
#[derive(Clone)]
pub struct BoundMetrics {
    model: String,
    inner: Arc<DispatcherMetrics>,
}

impl BoundMetrics {
    pub fn new(model: String, inner: Arc<DispatcherMetrics>) -> Self {
        Self { model, inner }
    }

    fn observe_batch_size(&self, n: usize) {
        self.inner
            .batch_size
            .with_label_values(&[&self.model])
            .observe(n as f64);
    }
    fn observe_batch_latency(&self, ms: f64) {
        self.inner
            .batch_latency_ms
            .with_label_values(&[&self.model])
            .observe(ms);
    }
    fn observe_wait_time(&self, ms: f64) {
        self.inner
            .wait_time_ms
            .with_label_values(&[&self.model])
            .observe(ms);
    }
    fn incr_dispatch_reason(&self, reason: &str) {
        self.inner
            .dispatch_reason
            .with_label_values(&[&self.model, reason])
            .inc();
    }
    fn incr_batch_errors(&self) {
        self.inner
            .batch_errors
            .with_label_values(&[&self.model])
            .inc();
    }
    fn set_queue_depth(&self, depth: i64) {
        self.inner
            .queue_depth
            .with_label_values(&[&self.model])
            .set(depth);
    }
    pub fn incr_overflow(&self) {
        self.inner.overflow.with_label_values(&[&self.model]).inc();
    }
}

/// Spawn one dispatcher task for the given bridge. Returns the mpsc sender
/// callers use to enqueue requests, and the task's JoinHandle for drain.
///
/// @spec SCHED-TASK-001, SCHED-CHAN-001
pub fn spawn(
    bridge: Arc<ModelBridge>,
    config: BucketConfig,
    metrics: BoundMetrics,
) -> (mpsc::Sender<PendingRequest>, JoinHandle<()>) {
    let (tx, rx) = mpsc::channel(config.queue_capacity.max(config.max_batch_size));
    let handle = tokio::spawn(run_loop(bridge, rx, config, metrics));
    (tx, handle)
}

/// @spec SCHED-TIMER-001, SCHED-TIMER-002, SCHED-SIZE-001, SCHED-SIZE-002,
///       SCHED-PRIORITY-001, SCHED-EXIT-001
async fn run_loop(
    bridge: Arc<ModelBridge>,
    mut rx: mpsc::Receiver<PendingRequest>,
    config: BucketConfig,
    metrics: BoundMetrics,
) {
    let mut batch: Vec<PendingRequest> = Vec::with_capacity(config.max_batch_size);
    let mut timer: Option<Pin<Box<Sleep>>> = None;

    loop {
        tokio::select! {
            biased;

            _ = async {
                timer
                    .as_mut()
                    .expect("timer present when this arm is enabled")
                    .as_mut()
                    .await
            }, if timer.is_some() => {
                metrics.incr_dispatch_reason("time");
                let taken = std::mem::take(&mut batch);
                metrics.set_queue_depth(0);
                dispatch_batch(&bridge, taken, &metrics).await;
                timer = None;
            }

            maybe_req = rx.recv() => {
                let Some(req) = maybe_req else {
                    if !batch.is_empty() {
                        let taken = std::mem::take(&mut batch);
                        metrics.set_queue_depth(0);
                        dispatch_batch(&bridge, taken, &metrics).await;
                    }
                    break;
                };

                if batch.is_empty() {
                    timer = Some(Box::pin(sleep(config.max_queue_delay)));
                }
                batch.push(req);
                metrics.set_queue_depth(batch.len() as i64);

                if batch.len() >= config.max_batch_size {
                    metrics.incr_dispatch_reason("size");
                    let taken = std::mem::take(&mut batch);
                    metrics.set_queue_depth(0);
                    dispatch_batch(&bridge, taken, &metrics).await;
                    timer = None;
                }
            }
        }
    }
}

/// Assemble the batch buffer, call the bridge, scatter outputs.
///
/// @spec SCHED-BUF-001, SCHED-BUF-002, SCHED-OUT-001, SCHED-ERR-001
async fn dispatch_batch(
    bridge: &Arc<ModelBridge>,
    batch: Vec<PendingRequest>,
    metrics: &BoundMetrics,
) {
    let n = batch.len();
    if n == 0 {
        return;
    }
    let bpr = bridge.input_meta().bytes_per_request;
    metrics.observe_batch_size(n);

    let wait_ms = batch
        .first()
        .map(|r| r.enqueued_at.elapsed().as_secs_f64() * 1000.0)
        .unwrap_or(0.0);
    metrics.observe_wait_time(wait_ms);

    let input_bytes = build_batch_buffer(&batch, bpr);

    let start = Instant::now();
    let result = bridge
        .run(BatchRequest {
            input_bytes,
            batch_n: n,
            seq_lens: None,
        })
        .await;
    metrics.observe_batch_latency(start.elapsed().as_secs_f64() * 1000.0);

    match result {
        Ok(out) => scatter_success(batch, &out, n),
        Err(e) => {
            metrics.incr_batch_errors();
            scatter_error(batch, e);
        }
    }
}

/// Pack per-request bytes into a single contiguous batch buffer. Public so
/// benchmarks can target it directly without going through the full
/// scheduler loop.
///
/// @spec SCHED-BUF-001, SCHED-BUF-002
pub fn build_batch_buffer(batch: &[PendingRequest], bytes_per_request: usize) -> Vec<u8> {
    let mut buf = vec![0u8; batch.len() * bytes_per_request];
    for (i, req) in batch.iter().enumerate() {
        let copy_len = req.input_bytes.len().min(bytes_per_request);
        buf[i * bytes_per_request..i * bytes_per_request + copy_len]
            .copy_from_slice(&req.input_bytes[..copy_len]);
    }
    buf
}

/// @spec SCHED-OUT-001, SCHED-OUT-002
fn scatter_success(batch: Vec<PendingRequest>, out: &BatchOutput, batch_n: usize) {
    for (i, req) in batch.into_iter().enumerate() {
        let per_req = slice_per_request(out, i, batch_n);
        let _ = req.responder.send(Ok(per_req));
    }
}

/// @spec SCHED-ERR-001
fn scatter_error(batch: Vec<PendingRequest>, err: BridgeError) {
    let shared = Arc::new(err);
    for req in batch {
        let _ = req.responder.send(Err(shared.clone()));
    }
}

/// Slice one request's portion of a batched output. Public for benchmarks.
pub fn slice_per_request(out: &BatchOutput, i: usize, batch_n: usize) -> PerRequestOutput {
    let outputs = out
        .outputs
        .iter()
        .map(|bytes| {
            // Leading dim of each output is batch_n; per-request slice is
            // bytes.len() / batch_n. If batch_n doesn't divide cleanly, the
            // model's output shape is malformed — yield empty rather than panic.
            if batch_n == 0 || bytes.len() % batch_n != 0 {
                return Vec::new();
            }
            let per = bytes.len() / batch_n;
            bytes[i * per..(i + 1) * per].to_vec()
        })
        .collect();
    PerRequestOutput { outputs }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req_with_bytes(bytes: Vec<u8>) -> PendingRequest {
        let (tx, _rx) = oneshot::channel();
        PendingRequest {
            input_bytes: bytes,
            seq_len: None,
            enqueued_at: Instant::now(),
            responder: tx,
        }
    }

    /// @spec SCHED-BUF-001, SCHED-BUF-002
    #[test]
    fn batch_buffer_packs_requests_in_order() {
        let bpr = 4;
        let batch = vec![
            req_with_bytes(vec![1, 2, 3, 4]),
            req_with_bytes(vec![5, 6, 7, 8]),
            req_with_bytes(vec![9, 10, 11, 12]),
        ];
        let buf = build_batch_buffer(&batch, bpr);
        assert_eq!(buf, vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    }

    #[test]
    fn batch_buffer_pads_short_requests_with_zero() {
        let bpr = 4;
        let batch = vec![req_with_bytes(vec![1, 2]), req_with_bytes(vec![5, 6, 7, 8])];
        let buf = build_batch_buffer(&batch, bpr);
        assert_eq!(buf, vec![1, 2, 0, 0, 5, 6, 7, 8]);
    }

    #[test]
    fn batch_buffer_truncates_overlength_requests() {
        let bpr = 4;
        let batch = vec![req_with_bytes(vec![1, 2, 3, 4, 5, 6])];
        let buf = build_batch_buffer(&batch, bpr);
        assert_eq!(buf, vec![1, 2, 3, 4]);
    }

    /// @spec SCHED-OUT-001
    #[test]
    fn slice_per_request_slices_along_leading_dim() {
        let out = BatchOutput {
            outputs: vec![
                vec![10, 20, 30, 40, 50, 60, 70, 80], // batch_n=4, 2 bytes per req
                vec![1, 2, 3, 4],                     // batch_n=4, 1 byte per req
            ],
            output_shapes: vec![vec![4, 2], vec![4, 1]],
        };
        let r0 = slice_per_request(&out, 0, 4);
        assert_eq!(r0.outputs, vec![vec![10, 20], vec![1]]);
        let r2 = slice_per_request(&out, 2, 4);
        assert_eq!(r2.outputs, vec![vec![50, 60], vec![3]]);
    }

    #[test]
    fn slice_per_request_yields_empty_when_batch_size_mismatches_output() {
        let out = BatchOutput {
            outputs: vec![vec![1, 2, 3, 4, 5]], // not divisible by 2
            output_shapes: vec![vec![2, 2]],
        };
        let r = slice_per_request(&out, 0, 2);
        assert_eq!(r.outputs, vec![Vec::<u8>::new()]);
    }
}
