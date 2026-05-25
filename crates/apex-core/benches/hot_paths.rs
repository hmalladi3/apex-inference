//! Microbenchmarks for the engine's hot paths.
//!
//! Run with:
//!
//!     cargo bench --bench hot_paths
//!
//! Targets:
//! - admission-check: the inline RPC gate (atomic loads + compares only)
//! - batch-buffer build: per-batch packing of request bytes (memcpy-bound)
//! - output slice: per-request slice of the batched output (allocation + copy)

use std::sync::Arc;
use std::time::Instant;

use apex_core::admission::AdmissionController;
use apex_core::dispatcher::{
    PendingRequest, PerRequestOutput, build_batch_buffer, slice_per_request,
};
use apex_core::ort_bridge::BatchOutput;
use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use prometheus::Registry;
use tokio::sync::oneshot;

fn admission_check_admit(c: &mut Criterion) {
    let registry = Registry::new();
    let ctrl = AdmissionController::new(u64::MAX, usize::MAX, &registry).unwrap();
    c.bench_function("admission_check (admit)", |b| {
        b.iter(|| black_box(ctrl.check()));
    });
}

fn admission_full_round_trip(c: &mut Criterion) {
    let registry = Registry::new();
    let ctrl = Arc::new(AdmissionController::new(u64::MAX, usize::MAX, &registry).unwrap());
    c.bench_function("admission round-trip (check + record + incr + decr)", |b| {
        b.iter(|| {
            let start = Instant::now();
            let decision = ctrl.check();
            ctrl.record(&decision, start.elapsed().as_nanos() as u64);
            ctrl.incr_queue();
            ctrl.decr_queue(1);
            black_box(decision);
        });
    });
}

fn batch_buffer(c: &mut Criterion) {
    let mut group = c.benchmark_group("build_batch_buffer");
    // ResNet-sized: 224*224*3*4 = 602112 bytes per request
    let bytes_per_request = 602_112;
    for batch_n in [1usize, 8, 32, 64].iter() {
        let batch: Vec<PendingRequest> = (0..*batch_n)
            .map(|_| dummy_request(bytes_per_request))
            .collect();
        group.throughput(Throughput::Bytes((batch_n * bytes_per_request) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(batch_n), batch_n, |b, _| {
            b.iter(|| black_box(build_batch_buffer(black_box(&batch), bytes_per_request)));
        });
    }
    group.finish();
}

fn output_slice(c: &mut Criterion) {
    let mut group = c.benchmark_group("slice_per_request");
    // Typical classifier output: 1000 logits * 4 bytes = 4000 bytes per request.
    let bytes_per_request = 4_000;
    for batch_n in [1usize, 8, 32, 64].iter() {
        let out = BatchOutput {
            outputs: vec![vec![0u8; batch_n * bytes_per_request]],
            output_shapes: vec![vec![*batch_n as i64, 1000]],
        };
        group.throughput(Throughput::Elements(*batch_n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(batch_n), batch_n, |b, _| {
            b.iter(|| {
                for i in 0..*batch_n {
                    let _r: PerRequestOutput = slice_per_request(black_box(&out), i, *batch_n);
                }
            });
        });
    }
    group.finish();
}

fn dummy_request(bytes_per_request: usize) -> PendingRequest {
    let (tx, _rx) = oneshot::channel();
    PendingRequest {
        input_bytes: vec![0u8; bytes_per_request],
        seq_len: None,
        enqueued_at: Instant::now(),
        responder: tx,
    }
}

criterion_group!(
    benches,
    admission_check_admit,
    admission_full_round_trip,
    batch_buffer,
    output_slice
);
criterion_main!(benches);
