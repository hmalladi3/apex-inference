//! The only file in apex-core permitted to contain `unsafe` in the ORT path.
//!
//! @spec BRIDGE-UNSAFE-001, BRIDGE-UNSAFE-002

use std::sync::Mutex;

use ndarray::{ArrayView, IxDyn};
use ort::session::Session;
use ort::value::TensorRef;

use crate::error::BridgeError;

use super::session::{DType, TensorMeta, validate_input_size};
use super::{BatchOutput, BatchRequest};

/// Run a batch through the ORT session. Caller-owned input buffer; outputs
/// are copied into freshly-allocated `Vec<u8>` per output tensor.
///
/// V1 supports F32 input tensors only. I64 (token IDs for NLP), F16, and
/// attention masks are planned for v1.1.
///
/// @spec BRIDGE-RUN-003, BRIDGE-RUN-004, BRIDGE-RUN-005
pub(super) fn run_blocking(
    session: &Mutex<Session>,
    batch: BatchRequest,
    input_meta: &TensorMeta,
    output_meta: &[TensorMeta],
) -> Result<BatchOutput, BridgeError> {
    validate_input_size(batch.input_bytes.len(), batch.batch_n, input_meta.bytes_per_request)?;

    match input_meta.dtype {
        DType::F32 => run_f32(session, batch, input_meta, output_meta),
        other => Err(BridgeError::UnsupportedDtype {
            dtype: format!("{other:?}"),
        }),
    }
}

fn run_f32(
    session: &Mutex<Session>,
    batch: BatchRequest,
    input_meta: &TensorMeta,
    output_meta: &[TensorMeta],
) -> Result<BatchOutput, BridgeError> {
    let mut shape: Vec<usize> = Vec::with_capacity(input_meta.shape.len());
    shape.push(batch.batch_n);
    for &d in input_meta.shape.iter().skip(1) {
        if d < 0 {
            return Err(BridgeError::RunFailed(
                "non-leading dimension is dynamic; not supported in v1".to_string(),
            ));
        }
        shape.push(d as usize);
    }

    // SAFETY: bytes was allocated via the global allocator, which provides
    // ≥8-byte alignment on x86_64 and aarch64 (the platforms apex targets).
    // bytes.len() is validated to equal batch_n * bytes_per_request, which for
    // f32 inputs is a multiple of 4. Any 4-byte pattern is a valid f32
    // (including NaN/subnormal), so no UB from invalid bit patterns. The slice
    // borrow is scoped to this function and outlives all downstream use.
    let typed: &[f32] = unsafe {
        debug_assert!(batch.input_bytes.as_ptr() as usize % core::mem::align_of::<f32>() == 0);
        core::slice::from_raw_parts(
            batch.input_bytes.as_ptr() as *const f32,
            batch.input_bytes.len() / 4,
        )
    };

    let view = ArrayView::from_shape(IxDyn(&shape), typed)
        .map_err(|e| BridgeError::RunFailed(format!("shape error: {e}")))?;

    let tensor = TensorRef::from_array_view(view)
        .map_err(|e| BridgeError::RunFailed(format!("tensor construction: {e}")))?;

    let mut s = session
        .lock()
        .map_err(|_| BridgeError::RunFailed("session mutex poisoned".to_string()))?;

    let outputs = s
        .run(ort::inputs![input_meta.name.as_str() => tensor])
        .map_err(|e| BridgeError::RunFailed(e.to_string()))?;

    let mut output_bytes: Vec<Vec<u8>> = Vec::with_capacity(output_meta.len());
    let mut output_shapes: Vec<Vec<i64>> = Vec::with_capacity(output_meta.len());

    for meta in output_meta {
        let value = &outputs[meta.name.as_str()];
        // V1: all outputs assumed f32. When mixed-dtype outputs land in v1.1,
        // dispatch on meta.dtype here.
        let view = value
            .try_extract_array::<f32>()
            .map_err(|e| BridgeError::OutputCopyFailed(e.to_string()))?;
        let shape: Vec<i64> = view.shape().iter().map(|&d| d as i64).collect();
        // Iterate in element order regardless of underlying memory layout.
        let mut bytes: Vec<u8> = Vec::with_capacity(view.len() * 4);
        for &f in view.iter() {
            bytes.extend_from_slice(&f.to_ne_bytes());
        }
        output_bytes.push(bytes);
        output_shapes.push(shape);
    }

    Ok(BatchOutput {
        outputs: output_bytes,
        output_shapes,
    })
}
