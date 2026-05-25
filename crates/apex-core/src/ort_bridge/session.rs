use ort::value::{Outlet, TensorElementType, ValueType};

use crate::error::BridgeError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
    F16,
    I64,
    I32,
    U8,
    Bool,
}

impl DType {
    pub fn bytes(&self) -> usize {
        match self {
            DType::F32 | DType::I32 => 4,
            DType::F16 => 2,
            DType::I64 => 8,
            DType::U8 | DType::Bool => 1,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorMeta {
    pub name: String,
    pub dtype: DType,
    /// Resolved dims; leading entry is `-1` for dynamic batch on inputs.
    pub shape: Vec<i64>,
    /// Bytes per single non-batched item. 0 when any non-leading dim is
    /// dynamic (output with unknown rank, ragged inputs in v1.1+).
    pub bytes_per_request: usize,
}

/// @spec BRIDGE-LOAD-005, BRIDGE-LOAD-006
pub(super) fn input_to_meta(outlet: &Outlet) -> Result<TensorMeta, BridgeError> {
    outlet_to_meta(outlet, /* require_dynamic_batch = */ true)
}

/// @spec BRIDGE-LOAD-007
pub(super) fn output_to_meta(outlet: &Outlet) -> Result<TensorMeta, BridgeError> {
    outlet_to_meta(outlet, /* require_dynamic_batch = */ false)
}

fn outlet_to_meta(outlet: &Outlet, require_dynamic_batch: bool) -> Result<TensorMeta, BridgeError> {
    let value_type: &ValueType = outlet.dtype();
    if !value_type.is_tensor() {
        return Err(BridgeError::UnsupportedInputType {
            description: format!("non-tensor: {value_type:?}"),
        });
    }

    let element_type =
        value_type
            .tensor_type()
            .ok_or_else(|| BridgeError::UnsupportedInputType {
                description: "missing tensor element type".to_string(),
            })?;
    let dtype = element_type_to_dtype(element_type)?;

    let shape = value_type
        .tensor_shape()
        .ok_or_else(|| BridgeError::UnsupportedInputType {
            description: "missing tensor shape".to_string(),
        })?;
    // Shape derefs to [i64].
    let dims: Vec<i64> = shape.iter().copied().collect();

    if require_dynamic_batch && (dims.is_empty() || dims[0] >= 0) {
        return Err(BridgeError::ModelNotBatchable);
    }

    let bytes_per_request = if dims.len() > 1 && dims.iter().skip(1).all(|&d| d > 0) {
        let elements: i64 = dims.iter().skip(1).copied().product();
        (elements as usize) * dtype.bytes()
    } else {
        // Either rank=1 (just batch dim) or some non-leading dim is dynamic;
        // bytes_per_request is resolved per-call instead of at load time.
        0
    };

    Ok(TensorMeta {
        name: outlet.name().to_string(),
        dtype,
        shape: dims,
        bytes_per_request,
    })
}

fn element_type_to_dtype(t: TensorElementType) -> Result<DType, BridgeError> {
    match t {
        TensorElementType::Float32 => Ok(DType::F32),
        TensorElementType::Float16 => Ok(DType::F16),
        TensorElementType::Int64 => Ok(DType::I64),
        TensorElementType::Int32 => Ok(DType::I32),
        TensorElementType::Uint8 => Ok(DType::U8),
        TensorElementType::Bool => Ok(DType::Bool),
        other => Err(BridgeError::UnsupportedDtype {
            dtype: format!("{other:?}"),
        }),
    }
}

/// @spec BRIDGE-RUN-001, BRIDGE-RUN-002
pub(super) fn validate_input_size(
    actual: usize,
    batch_n: usize,
    bytes_per_request: usize,
) -> Result<(), BridgeError> {
    let expected =
        batch_n
            .checked_mul(bytes_per_request)
            .ok_or(BridgeError::InputSizeMismatch {
                expected: usize::MAX,
                actual,
            })?;
    if actual != expected {
        return Err(BridgeError::InputSizeMismatch { expected, actual });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dtype_bytes() {
        assert_eq!(DType::F32.bytes(), 4);
        assert_eq!(DType::F16.bytes(), 2);
        assert_eq!(DType::I64.bytes(), 8);
        assert_eq!(DType::I32.bytes(), 4);
        assert_eq!(DType::U8.bytes(), 1);
        assert_eq!(DType::Bool.bytes(), 1);
    }

    /// @spec BRIDGE-RUN-001
    #[test]
    fn validate_input_size_accepts_exact_match() {
        validate_input_size(32 * 4, 32, 4).expect("exact size should pass");
    }

    /// @spec BRIDGE-RUN-002
    #[test]
    fn validate_input_size_rejects_short() {
        let err = validate_input_size(100, 32, 4).expect_err("should reject");
        match err {
            BridgeError::InputSizeMismatch { expected, actual } => {
                assert_eq!(expected, 128);
                assert_eq!(actual, 100);
            }
            other => panic!("wrong error: {other:?}"),
        }
    }

    /// @spec BRIDGE-RUN-002
    #[test]
    fn validate_input_size_rejects_long() {
        let err = validate_input_size(200, 32, 4).expect_err("should reject");
        assert!(matches!(err, BridgeError::InputSizeMismatch { .. }));
    }
}
