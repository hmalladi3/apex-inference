//! Core types and components for the apex inference engine.
//!
//! Each module corresponds to a low-level design at `docs/llds/`:
//!
//! - [`ort_bridge`] — the ONNX Runtime FFI boundary
//! - [`dispatcher`] — per-(model, bucket) batching scheduler
//! - [`registry`] — atomically swappable model registry
//! - [`config`] — YAML schema and validation
//! - [`admission`] — inline admission controller

pub mod admission;
pub mod config;
pub mod dispatcher;
pub mod error;
pub mod ort_bridge;
pub mod registry;
pub mod reload;
