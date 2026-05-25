//! KServe v2 GRPCInferenceService implementation.
//!
//! Spec: docs/specs/grpc-ingress.md

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use apex_core::admission::AdmissionController;
use apex_core::dispatcher::PendingRequest;
use apex_core::error::BridgeError;
use apex_core::ort_bridge::DType;
use apex_core::registry::SharedRegistry;
use prometheus::{
    HistogramOpts, HistogramVec, IntCounterVec, IntGauge, Opts, Registry,
};
use tokio::sync::{mpsc::error::TrySendError, oneshot};
use tonic::{Request, Response, Status};

use crate::proto::grpc_inference_service_server::GrpcInferenceService;
use crate::proto::{
    InferParameter, ModelInferRequest, ModelInferResponse, ModelMetadataRequest,
    ModelMetadataResponse, ModelReadyRequest, ModelReadyResponse, ServerLiveRequest,
    ServerLiveResponse, ServerMetadataRequest, ServerMetadataResponse, ServerReadyRequest,
    ServerReadyResponse,
    model_infer_response::InferOutputTensor,
    model_metadata_response::TensorMetadata,
};

#[derive(Clone)]
pub struct GrpcMetrics {
    pub requests_total: IntCounterVec,
    pub request_duration: HistogramVec,
    pub inflight_requests: IntGauge,
    pub input_bytes: HistogramVec,
}

impl GrpcMetrics {
    pub fn register(registry: &Registry) -> Result<Self, prometheus::Error> {
        let requests_total = IntCounterVec::new(
            Opts::new("apex_grpc_requests_total", "gRPC RPCs by code"),
            &["rpc", "code"],
        )?;
        let request_duration = HistogramVec::new(
            HistogramOpts::new(
                "apex_grpc_request_duration_seconds",
                "RPC wall time from handler entry to response",
            )
            .buckets(vec![0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]),
            &["rpc"],
        )?;
        let inflight_requests = IntGauge::new(
            "apex_grpc_inflight_requests",
            "currently in-flight ModelInfer calls",
        )?;
        let input_bytes = HistogramVec::new(
            HistogramOpts::new(
                "apex_grpc_input_bytes",
                "inline tensor payload size per ModelInfer",
            )
            .buckets(vec![
                1_024.0, 16_384.0, 262_144.0, 1_048_576.0, 16_777_216.0, 67_108_864.0,
            ]),
            &["model"],
        )?;
        registry.register(Box::new(requests_total.clone()))?;
        registry.register(Box::new(request_duration.clone()))?;
        registry.register(Box::new(inflight_requests.clone()))?;
        registry.register(Box::new(input_bytes.clone()))?;
        Ok(Self {
            requests_total,
            request_duration,
            inflight_requests,
            input_bytes,
        })
    }
}

pub struct InferenceService {
    pub registry: SharedRegistry,
    pub admission: Arc<AdmissionController>,
    pub max_request_bytes: usize,
    pub metrics: GrpcMetrics,
}

impl InferenceService {
    fn record(&self, rpc: &'static str, started: Instant, status: &Result<(), Status>) {
        let code = match status {
            Ok(_) => "ok",
            Err(s) => status_code_label(s.code()),
        };
        self.metrics
            .requests_total
            .with_label_values(&[rpc, code])
            .inc();
        self.metrics
            .request_duration
            .with_label_values(&[rpc])
            .observe(started.elapsed().as_secs_f64());
    }
}

#[tonic::async_trait]
impl GrpcInferenceService for InferenceService {
    /// @spec INGRESS-LIVE-001
    async fn server_live(
        &self,
        _req: Request<ServerLiveRequest>,
    ) -> Result<Response<ServerLiveResponse>, Status> {
        let started = Instant::now();
        let resp = Response::new(ServerLiveResponse { live: true });
        self.record("ServerLive", started, &Ok(()));
        Ok(resp)
    }

    /// @spec INGRESS-READY-001
    async fn server_ready(
        &self,
        _req: Request<ServerReadyRequest>,
    ) -> Result<Response<ServerReadyResponse>, Status> {
        let started = Instant::now();
        let r = self.registry.load();
        let ready = r.count() > 0 && r.all_entries().all(|e| e.is_loaded());
        let resp = Response::new(ServerReadyResponse { ready });
        self.record("ServerReady", started, &Ok(()));
        Ok(resp)
    }

    /// @spec INGRESS-META-001
    async fn server_metadata(
        &self,
        _req: Request<ServerMetadataRequest>,
    ) -> Result<Response<ServerMetadataResponse>, Status> {
        let started = Instant::now();
        let resp = Response::new(ServerMetadataResponse {
            name: "apex-inference".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            extensions: vec![],
        });
        self.record("ServerMetadata", started, &Ok(()));
        Ok(resp)
    }

    /// @spec INGRESS-MODEL-READY-001, INGRESS-MODEL-READY-002
    async fn model_ready(
        &self,
        req: Request<ModelReadyRequest>,
    ) -> Result<Response<ModelReadyResponse>, Status> {
        let started = Instant::now();
        let req = req.into_inner();
        let r = self.registry.load();
        let ready = r
            .get(&req.name, version_or_none(&req.version))
            .map(|e| e.is_loaded())
            .unwrap_or(false);
        let resp = Response::new(ModelReadyResponse { ready });
        self.record("ModelReady", started, &Ok(()));
        Ok(resp)
    }

    /// @spec INGRESS-MODEL-META-001
    async fn model_metadata(
        &self,
        req: Request<ModelMetadataRequest>,
    ) -> Result<Response<ModelMetadataResponse>, Status> {
        let started = Instant::now();
        let req = req.into_inner();
        let r = self.registry.load();
        let entry = match r.get(&req.name, version_or_none(&req.version)) {
            Some(e) => e,
            None => {
                let status = Status::not_found(format!(
                    "model not found: {}/{}",
                    req.name, req.version
                ));
                self.record("ModelMetadata", started, &Err(status.clone()));
                return Err(status);
            }
        };

        let resp = Response::new(ModelMetadataResponse {
            name: entry.name.clone(),
            versions: vec![entry.version.clone()],
            platform: "onnxruntime".to_string(),
            inputs: vec![TensorMetadata {
                name: entry.input_meta.name.clone(),
                datatype: dtype_to_kserve(entry.input_meta.dtype),
                shape: entry.input_meta.shape.clone(),
            }],
            outputs: entry
                .output_meta
                .iter()
                .map(|m| TensorMetadata {
                    name: m.name.clone(),
                    datatype: dtype_to_kserve(m.dtype),
                    shape: m.shape.clone(),
                })
                .collect(),
        });
        self.record("ModelMetadata", started, &Ok(()));
        Ok(resp)
    }

    /// @spec INGRESS-INFER-001..014
    async fn model_infer(
        &self,
        req: Request<ModelInferRequest>,
    ) -> Result<Response<ModelInferResponse>, Status> {
        let started = Instant::now();
        self.metrics.inflight_requests.inc();
        let result = self.do_infer(req).await;
        self.metrics.inflight_requests.dec();
        let outcome = match &result {
            Ok(_) => Ok(()),
            Err(s) => Err(s.clone()),
        };
        self.record("ModelInfer", started, &outcome);
        result
    }
}

impl InferenceService {
    async fn do_infer(
        &self,
        req: Request<ModelInferRequest>,
    ) -> Result<Response<ModelInferResponse>, Status> {
        let req = req.into_inner();

        // @spec INGRESS-INFER-014
        let req_bytes: usize = req.raw_input_contents.iter().map(|v| v.len()).sum();
        if req_bytes > self.max_request_bytes {
            return Err(Status::resource_exhausted(format!(
                "request exceeds max_request_bytes ({} > {})",
                req_bytes, self.max_request_bytes
            )));
        }

        // @spec INGRESS-INFER-006
        let admit_start = Instant::now();
        let decision = self.admission.check();
        let admit_ns = admit_start.elapsed().as_nanos() as u64;
        self.admission.record(&decision, admit_ns);
        if !decision.is_admit() {
            self.admission.maybe_log_rejection(&decision);
            return Err(Status::resource_exhausted(decision.reject_message()));
        }

        // @spec INGRESS-INFER-001
        if req.inputs.len() != 1 {
            return Err(Status::invalid_argument(format!(
                "expected exactly one input tensor, got {}",
                req.inputs.len()
            )));
        }
        // @spec INGRESS-INFER-005
        if req.raw_input_contents.len() != 1 {
            return Err(Status::invalid_argument(
                "use raw_input_contents (one entry per input)",
            ));
        }
        let input = &req.inputs[0];
        let input_bytes = req.raw_input_contents.into_iter().next().unwrap();

        // @spec INGRESS-INFER-007
        let r = self.registry.load();
        let entry = r
            .get(&req.model_name, version_or_none(&req.model_version))
            .ok_or_else(|| Status::not_found(format!("model not found: {}", req.model_name)))?;

        if !entry.is_loaded() {
            return Err(Status::resource_exhausted(format!(
                "model {} is draining",
                entry.name
            )));
        }

        // @spec INGRESS-INFER-004
        if input.name != entry.input_meta.name {
            return Err(Status::invalid_argument(format!(
                "input name mismatch: expected '{}', got '{}'",
                entry.input_meta.name, input.name
            )));
        }
        // @spec INGRESS-INFER-003
        let expected_dt = dtype_to_kserve(entry.input_meta.dtype);
        if input.datatype != expected_dt {
            return Err(Status::invalid_argument(format!(
                "datatype mismatch: expected {expected_dt}, got {}",
                input.datatype
            )));
        }
        // @spec INGRESS-INFER-002
        if input.shape.first().copied() != Some(1) {
            return Err(Status::invalid_argument(
                "leading dim must be 1 (engine owns batching)",
            ));
        }
        if input_bytes.len() != entry.input_meta.bytes_per_request {
            return Err(Status::invalid_argument(format!(
                "input bytes mismatch: expected {}, got {}",
                entry.input_meta.bytes_per_request,
                input_bytes.len()
            )));
        }

        self.metrics
            .input_bytes
            .with_label_values(&[&entry.name])
            .observe(input_bytes.len() as f64);

        let (resp_tx, resp_rx) = oneshot::channel();
        let pending = PendingRequest {
            input_bytes,
            seq_len: None,
            enqueued_at: Instant::now(),
            responder: resp_tx,
        };

        self.admission.incr_queue();

        // @spec INGRESS-INFER-009, INGRESS-INFER-010
        if let Err(e) = entry.tx.try_send(pending) {
            self.admission.decr_queue(1);
            return match e {
                TrySendError::Full(_) => {
                    Err(Status::resource_exhausted(format!("queue full for {}", entry.name)))
                }
                TrySendError::Closed(_) => Err(Status::not_found(format!(
                    "model {} no longer accepting requests",
                    entry.name
                ))),
            };
        }

        // @spec INGRESS-INFER-011, INGRESS-INFER-012
        let per_req = match resp_rx.await {
            Ok(Ok(r)) => {
                self.admission.decr_queue(1);
                r
            }
            Ok(Err(bridge_err)) => {
                self.admission.decr_queue(1);
                return Err(bridge_err_to_status(&bridge_err));
            }
            Err(_) => {
                self.admission.decr_queue(1);
                return Err(Status::internal("dispatcher dropped responder"));
            }
        };

        // @spec INGRESS-INFER-008
        let outputs = entry
            .output_meta
            .iter()
            .map(|m| InferOutputTensor {
                name: m.name.clone(),
                datatype: dtype_to_kserve(m.dtype),
                shape: std::iter::once(1)
                    .chain(m.shape.iter().skip(1).copied())
                    .collect(),
                parameters: HashMap::<String, InferParameter>::new(),
                contents: None,
            })
            .collect();

        Ok(Response::new(ModelInferResponse {
            model_name: entry.name.clone(),
            model_version: entry.version.clone(),
            id: req.id,
            parameters: HashMap::<String, InferParameter>::new(),
            outputs,
            raw_output_contents: per_req.outputs,
        }))
    }
}

fn version_or_none(version: &str) -> Option<&str> {
    if version.is_empty() { None } else { Some(version) }
}

fn dtype_to_kserve(dt: DType) -> String {
    match dt {
        DType::F32 => "FP32",
        DType::F16 => "FP16",
        DType::I64 => "INT64",
        DType::I32 => "INT32",
        DType::U8 => "UINT8",
        DType::Bool => "BOOL",
    }
    .to_string()
}

fn bridge_err_to_status(err: &BridgeError) -> Status {
    // All bridge errors at infer-time map to INTERNAL — load-time failures
    // should never surface here, and dispatch-time failures (RunFailed,
    // OutputCopyFailed, DispatchPanic) represent server-side faults.
    Status::internal(format!("{err}"))
}

fn status_code_label(code: tonic::Code) -> &'static str {
    use tonic::Code::*;
    match code {
        Ok => "ok",
        Cancelled => "cancelled",
        Unknown => "unknown",
        InvalidArgument => "invalid_argument",
        DeadlineExceeded => "deadline_exceeded",
        NotFound => "not_found",
        AlreadyExists => "already_exists",
        PermissionDenied => "permission_denied",
        ResourceExhausted => "resource_exhausted",
        FailedPrecondition => "failed_precondition",
        Aborted => "aborted",
        OutOfRange => "out_of_range",
        Unimplemented => "unimplemented",
        Internal => "internal",
        Unavailable => "unavailable",
        DataLoss => "data_loss",
        Unauthenticated => "unauthenticated",
    }
}
