use thiserror::Error;

#[derive(Debug, Error)]
pub enum BridgeError {
    #[error("model load failed: {0}")]
    ModelLoadFailed(String),

    #[error("model has static batch dimension; engine requires dynamic")]
    ModelNotBatchable,

    #[error(
        "unsupported tensor dtype in v1: {dtype} (v1 supports F32 only; I64/F16 planned for v1.1)"
    )]
    UnsupportedDtype { dtype: String },

    #[error("unsupported input value type: {description}")]
    UnsupportedInputType { description: String },

    #[error("input size mismatch: expected {expected} bytes, got {actual}")]
    InputSizeMismatch { expected: usize, actual: usize },

    #[error("ort::Session::run failed: {0}")]
    RunFailed(String),

    #[error("output copy failed: {0}")]
    OutputCopyFailed(String),

    #[error("dispatch task panicked: {0}")]
    DispatchPanic(String),
}

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("parse error: {0}")]
    Parse(String),

    #[error("validation error: {0}")]
    Validation(String),

    #[error("model load failed: {0}")]
    ModelLoad(#[from] BridgeError),
}
