//! ONNX Runtime FFI boundary.
//!
//! All `unsafe` for ORT input construction lives in [`invoke`]. See
//! `docs/llds/ort-bridge.md` and `docs/specs/ort-bridge.md`.

mod invoke;
mod runtime;
mod session;

use std::sync::{Arc, Mutex};

use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;

use crate::config::ModelConfig;
use crate::error::BridgeError;

pub use runtime::BridgeRuntime;
pub use session::{DType, TensorMeta};

pub struct ModelBridge {
    session: Arc<Mutex<Session>>,
    input_meta: TensorMeta,
    output_meta: Vec<TensorMeta>,
    runtime: BridgeRuntime,
}

/// Owned batch input. The bridge takes ownership for the duration of the
/// `Session::run` call (the buffer is moved into the `spawn_blocking` task).
pub struct BatchRequest {
    pub input_bytes: Vec<u8>,
    pub batch_n: usize,
    pub seq_lens: Option<Vec<u32>>,
}

/// Bytes already copied out of ORT; one Vec per output tensor.
pub struct BatchOutput {
    pub outputs: Vec<Vec<u8>>,
    pub output_shapes: Vec<Vec<i64>>,
}

impl ModelBridge {
    /// @spec BRIDGE-LOAD-001, BRIDGE-LOAD-002, BRIDGE-LOAD-003, BRIDGE-LOAD-004,
    ///       BRIDGE-LOAD-005, BRIDGE-LOAD-006, BRIDGE-LOAD-007
    pub fn load(config: &ModelConfig) -> Result<Self, BridgeError> {
        let session = Session::builder()
            .map_err(|e| BridgeError::ModelLoadFailed(e.to_string()))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| BridgeError::ModelLoadFailed(e.to_string()))?
            .with_intra_threads(config.intra_op_threads)
            .map_err(|e| BridgeError::ModelLoadFailed(e.to_string()))?
            .with_inter_threads(1)
            .map_err(|e| BridgeError::ModelLoadFailed(e.to_string()))?
            .commit_from_file(&config.path)
            .map_err(|e| BridgeError::ModelLoadFailed(e.to_string()))?;

        if session.inputs().len() != 1 {
            return Err(BridgeError::ModelLoadFailed(format!(
                "expected exactly one input tensor, found {}",
                session.inputs().len()
            )));
        }
        let input_meta = session::input_to_meta(&session.inputs()[0])?;
        let output_meta = session
            .outputs()
            .iter()
            .map(session::output_to_meta)
            .collect::<Result<Vec<_>, _>>()?;

        let runtime = match config.runtime {
            crate::config::BridgeRuntimeKind::Blocking => BridgeRuntime::Blocking,
            crate::config::BridgeRuntimeKind::DedicatedThread => BridgeRuntime::DedicatedThread,
        };

        Ok(Self {
            session: Arc::new(Mutex::new(session)),
            input_meta,
            output_meta,
            runtime,
        })
    }

    /// @spec BRIDGE-RUN-001, BRIDGE-RUN-002, BRIDGE-RUN-003,
    ///       BRIDGE-RUN-004, BRIDGE-RUN-005, BRIDGE-RUN-006, BRIDGE-CONC-001
    pub async fn run(&self, batch: BatchRequest) -> Result<BatchOutput, BridgeError> {
        let session = self.session.clone();
        let input_meta = self.input_meta.clone();
        let output_meta = self.output_meta.clone();

        match self.runtime {
            BridgeRuntime::Blocking => {
                tokio::task::spawn_blocking(move || {
                    invoke::run_blocking(&session, batch, &input_meta, &output_meta)
                })
                .await
                .map_err(|e| {
                    if e.is_panic() {
                        BridgeError::DispatchPanic(format!("{e}"))
                    } else {
                        BridgeError::RunFailed(format!("spawn_blocking failed: {e}"))
                    }
                })?
            }
            BridgeRuntime::DedicatedThread => Err(BridgeError::RunFailed(
                "DedicatedThread runtime not implemented in v1; see ort-bridge LLD".to_string(),
            )),
        }
    }

    pub fn input_meta(&self) -> &TensorMeta {
        &self.input_meta
    }

    pub fn output_meta(&self) -> &[TensorMeta] {
        &self.output_meta
    }
}
